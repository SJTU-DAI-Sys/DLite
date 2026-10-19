"""High-level DLite inference API."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .modeling import DLiteDraftModel


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: torch.Tensor
    stats: dict


def _resolve_attention(attention_backend: str) -> str:
    if attention_backend != "auto":
        return attention_backend
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"
    return "flash_attention_2"


class DLiteEngine:
    """A target model and DLite draft model ready for speculative decoding."""

    def __init__(self, target, draft: DLiteDraftModel, tokenizer) -> None:
        if not draft.is_student:
            raise ValueError("The inference release requires a student checkpoint.")
        self.target = target
        self.draft = draft
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        target_model: str,
        draft_model: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attention_backend: str = "auto",
        trust_remote_code: bool = False,
    ) -> DLiteEngine:
        attention = _resolve_attention(attention_backend)
        common = {
            "attn_implementation": attention,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        target = AutoModelForCausalLM.from_pretrained(target_model, **common)
        draft = DLiteDraftModel.from_pretrained(draft_model, **common)
        tokenizer = AutoTokenizer.from_pretrained(
            target_model, trust_remote_code=trust_remote_code
        )
        target = target.to(device).eval()
        draft = draft.to(device).eval()
        return cls(target, draft, tokenizer)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        stochastic_verification_mode: str = "match",
        compile_sequential_head: bool = False,
        verify_block_size: int | None = None,
    ) -> GenerationResult:
        if isinstance(prompt, str):
            messages: Iterable[Mapping[str, str]] = (
                {"role": "user", "content": prompt},
            )
        else:
            messages = prompt
        rendered = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        input_ids = self.tokenizer(rendered, return_tensors="pt").input_ids.to(
            self.draft.device
        )
        output_ids = self.draft.spec_generate(
            target=self.target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id],
            temperature=temperature,
            stochastic_verification_mode=stochastic_verification_mode,
            compile_serial_head=compile_sequential_head,
            verify_block_size=verify_block_size,
        )
        generated = output_ids[:, input_ids.shape[1] :]
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return GenerationResult(text, output_ids, self.draft.get_last_decode_stats())


__all__ = ["DLiteEngine", "GenerationResult"]
