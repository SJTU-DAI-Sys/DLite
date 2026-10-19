import logging
from collections.abc import Callable

import torch
from torch import nn
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from ..utils import (
    cuda_sync_time,
    gather_pivot_hidden_states,
    rejection_sample_verify,
    sample_logits,
    validate_verification_mode,
)
from .qwen3 import Qwen3DLiteDecoderLayer
from .sequential_head import SEQUENTIAL_HEAD_TYPES, DLiteSequentialHead

DLITE_ARCHITECTURE_VERSIONS = ("dlite_v1", "dlite_v2")
DLITE_ARCHITECTURE_VERSION = "dlite_v2"
DLITE_MODEL_ROLE = "student"


def _new_target_cache(target: nn.Module):
    try:
        return DynamicCache(config=target.config)
    except TypeError:
        return DynamicCache()


def _begin_recurrent_cache_transaction(cache):
    if not hasattr(cache, "activate_past_recording"):
        return None
    snapshots = []
    for layer in getattr(cache, "layers", ()):
        recurrent_states = getattr(layer, "recurrent_states", None)
        if not isinstance(recurrent_states, dict):
            continue
        initialized = getattr(layer, "is_recurrent_states_initialized", {})
        layer_snapshot = {
            key: value.clone()
            for key, value in recurrent_states.items()
            if value is not None and bool(initialized.get(key, True))
        }
        if layer_snapshot:
            snapshots.append((layer, layer_snapshot))
    if not snapshots:
        return None
    cache.activate_past_recording()
    return snapshots


def _restore_recurrent_cache_transaction(snapshots) -> None:
    for layer, layer_snapshot in snapshots:
        for key, saved in layer_snapshot.items():
            current = layer.recurrent_states.get(key)
            if current is None:
                layer.recurrent_states[key] = saved
            else:
                current.copy_(saved)


def _crop_target_cache(cache, *, current_length: int, tokens_to_remove: int) -> None:
    tokens_to_remove = int(tokens_to_remove)
    if tokens_to_remove < 0:
        raise ValueError("tokens_to_remove must be non-negative")
    if hasattr(cache, "is_croppable"):
        cache.crop(-tokens_to_remove)
    else:
        cache.crop(int(current_length) - tokens_to_remove)


class DLiteDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DLiteDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        dlite_config = getattr(config, "dlite_config", {}) or {}
        architecture_version = dlite_config.get(
            "architecture_version", DLITE_ARCHITECTURE_VERSION
        )
        if architecture_version not in DLITE_ARCHITECTURE_VERSIONS:
            raise ValueError(
                "Incompatible DLite checkpoint/config architecture_version: "
                f"expected one of {DLITE_ARCHITECTURE_VERSIONS}, got "
                f"{architecture_version!r}."
            )
        self.architecture_version = str(architecture_version)
        self.model_role = str(dlite_config.get("model_role", "")).lower()
        if self.model_role != DLITE_MODEL_ROLE:
            raise ValueError(
                f"The inference package requires model_role={DLITE_MODEL_ROLE!r}, got "
                f"{self.model_role!r}."
            )
        required = {
            "model_role",
            "chs_num_layers",
            "target_layer_ids",
            "sequential_head",
            "sequential_rank",
        }
        missing = sorted(required - dlite_config.keys())
        if missing:
            raise ValueError(f"dlite_config is missing required fields: {missing}")
        self.chs_num_layers = int(dlite_config["chs_num_layers"])
        self.target_layer_ids = [
            int(value) for value in dlite_config["target_layer_ids"]
        ]
        if len(self.target_layer_ids) != self.chs_num_layers:
            raise ValueError(
                "target_layer_ids length must equal chs_num_layers: "
                f"{len(self.target_layer_ids)} != {self.chs_num_layers}."
            )
        if self.target_layer_ids != sorted(set(self.target_layer_ids)):
            raise ValueError("target_layer_ids must be unique and strictly increasing.")
        invalid_layer_ids = [
            layer_id
            for layer_id in self.target_layer_ids
            if not 0 <= layer_id < int(config.num_target_layers)
        ]
        if invalid_layer_ids:
            raise ValueError(
                f"target_layer_ids {invalid_layer_ids} are outside [0, "
                f"{int(config.num_target_layers) - 1}]."
            )
        dlite_config["architecture_version"] = self.architecture_version
        dlite_config["model_role"] = self.model_role
        self.sequential_head_type = str(dlite_config["sequential_head"]).lower()
        self.sequential_rank = int(dlite_config["sequential_rank"])
        if self.sequential_head_type not in SEQUENTIAL_HEAD_TYPES:
            raise ValueError(
                f"Unknown sequential_head={self.sequential_head_type!r}; expected "
                f"one of {SEQUENTIAL_HEAD_TYPES}."
            )
        if self.sequential_rank <= 0:
            raise ValueError(
                f"sequential_rank must be positive, got {self.sequential_rank}."
            )
        self.sequential_head = DLiteSequentialHead(
            head_type=self.sequential_head_type,
            vocab_size=config.vocab_size,
            sequential_rank=self.sequential_rank,
            hidden_size=config.hidden_size,
            max_prediction_length=config.block_size - 1,
        )
        self._compiled_serial_sampler_cache: dict[tuple[float, bool], Callable] = {}
        config.dlite_config = dlite_config

        self.layers = nn.ModuleList(
            [
                Qwen3DLiteDecoderLayer(
                    config,
                    layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.block_size = config.block_size
        self.mask_token_id = dlite_config.get("mask_token_id", None)
        self.mask_embedding_mode = "vocab_row"
        self._mask_embedding_cache: tuple[int, torch.Tensor] | None = None
        self._last_decode_stats = {}

        h = config.hidden_size
        self.layer_depth_embedding = nn.Embedding(config.num_target_layers, h)
        self.context_norm = Qwen3RMSNorm(h, eps=config.rms_norm_eps)
        logging.getLogger(__name__).info(
            f"DLite: architecture_version={self.architecture_version}, "
            f"model_role={self.model_role}, "
            f"chs_num_layers={self.chs_num_layers}, "
            f"condition_slots={self.condition_slot_count}, "
            f"target_layer_ids={self.target_layer_ids}, "
            f"sequential_head={self.sequential_head_type}, "
            f"sequential_rank={self.sequential_rank}"
        )

        self.post_init()

    @property
    def chs_len_per_block(self) -> int:
        return self.condition_slot_count

    @property
    def condition_slot_count(self) -> int:
        return self.chs_num_layers

    @property
    def is_student(self) -> bool:
        return True

    @property
    def token_prefix_count(self) -> int:
        return 1

    def get_last_decode_stats(self) -> dict:
        return dict(self._last_decode_stats)

    def build_block_position_ids(
        self,
        anchor_positions: torch.Tensor,
        token_position_ids: torch.Tensor,
        token_keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return local student context and query RoPE ids."""
        if token_position_ids.shape != token_keep_mask.shape:
            raise ValueError("token positions and keep mask must have equal shapes")
        pivot = (anchor_positions - 1).clamp(min=0)
        has_tokens = token_keep_mask.any(dim=-1)
        sentinel = torch.iinfo(token_position_ids.dtype).max
        first_valid = torch.where(
            token_keep_mask,
            token_position_ids,
            torch.full_like(token_position_ids, sentinel),
        ).amin(dim=-1)
        origin = torch.where(has_tokens, first_valid, pivot)
        local_offset = 0 if self.architecture_version == "dlite_v1" else 1
        local_tokens = torch.where(
            token_keep_mask,
            token_position_ids - origin.unsqueeze(-1) + local_offset,
            torch.zeros_like(token_position_ids),
        )
        context_positions = (
            (pivot - origin)
            .clamp(min=0)
            .unsqueeze(-1)
            .expand(-1, -1, self.chs_num_layers)
        )
        mask_offsets = torch.arange(
            1, self.block_size, device=anchor_positions.device
        ).view(1, 1, -1)
        local_masks = (
            anchor_positions.unsqueeze(-1)
            + mask_offsets
            - origin.unsqueeze(-1)
            + local_offset
        )
        draft_positions = torch.cat([local_tokens, local_masks], dim=-1)
        return (
            context_positions.reshape(anchor_positions.shape[0], -1),
            draft_positions.reshape(anchor_positions.shape[0], -1),
        )

    @property
    def draft_block_len(self) -> int:
        """Parallel draft slots per anchor (1 anchor + remaining MASK tokens)."""
        return self.block_size

    @property
    def draft_query_length(self) -> int:
        return self.token_prefix_count + self.proposal_length

    @property
    def proposal_length(self) -> int:
        """Draft tokens proposed after the anchor; total span is block_size."""
        return self.block_size - 1

    @property
    def max_verify_block_size(self) -> int:
        """Anchor-inclusive target verification window (equals config block_size)."""
        return self.proposal_length + 1

    def _prediction_hidden(self, block_hidden: torch.Tensor) -> torch.Tensor:
        """Return one hidden state for each proposed token."""
        return block_hidden[:, -self.proposal_length :, :]

    def _apply_chs_depth_embedding(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Add target-layer identity to current CHS hidden states."""
        _, _, s_len, h = target_hidden.shape
        if s_len != self.chs_num_layers:
            raise ValueError(
                f"Expected {self.chs_num_layers} current CHS slots, got {s_len}."
            )
        depth_ids = torch.tensor(
            self.target_layer_ids, device=target_hidden.device, dtype=torch.long
        )
        depth_emb = self.layer_depth_embedding(depth_ids).view(
            1, 1, self.chs_num_layers, h
        )
        return self.context_norm(target_hidden + depth_emb)

    def build_inference_query_embeddings(
        self,
        embed_tokens: nn.Module,
        draft_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build one anchor query followed by ``B-1`` MASK queries."""
        real_embeddings = embed_tokens(draft_input_ids[:, :1])
        mask_ids = draft_input_ids[:, 1:]
        weight = getattr(embed_tokens, "weight", None)
        if weight is not None and bool((mask_ids >= weight.shape[0]).any()):
            if self.mask_token_id is None or bool(
                (mask_ids != self.mask_token_id).any()
            ):
                raise ValueError(
                    "Draft query contains an out-of-vocabulary non-MASK id."
                )
            if self.mask_embedding_mode == "vocab_row":
                raise ValueError(
                    "Checkpoint requires vocab_row MASK embeddings, but "
                    f"mask_token_id={self.mask_token_id} is outside the provided "
                    f"embedding with {weight.shape[0]} rows."
                )
            weight_ptr = weight.data_ptr()
            cached = self._mask_embedding_cache
            if (
                cached is None
                or cached[0] != weight_ptr
                or cached[1].device != weight.device
                or cached[1].dtype != weight.dtype
            ):
                cached = (weight_ptr, weight.detach().mean(dim=0))
                self._mask_embedding_cache = cached
            mask_embeddings = (
                cached[1]
                .view(1, 1, -1)
                .expand(mask_ids.shape[0], mask_ids.shape[1], -1)
            )
        else:
            mask_embeddings = embed_tokens(mask_ids)
        return torch.cat([real_embeddings, mask_embeddings], dim=1)

    def build_inference_current_chs(
        self,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Build current CHS context for inference."""
        if target_hidden.ndim != 4 or target_hidden.shape[2] != self.chs_num_layers:
            raise ValueError(
                f"target_hidden must have shape (B,N,{self.chs_num_layers},H), got "
                f"{tuple(target_hidden.shape)}."
            )
        if target_hidden.shape[1] != 1:
            raise ValueError("Inference current CHS expects exactly one anchor block.")
        return target_hidden

    def _fuse_target_hidden(
        self,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Flatten the depth-aware CHS context."""
        bsz, n_blk, _, _ = target_hidden.shape
        current_ctx = self._apply_chs_depth_embedding(target_hidden)
        ctx = current_ctx
        return ctx.reshape(bsz, n_blk * ctx.shape[2], current_ctx.shape[-1])

    def build_inference_context(
        self,
        recent_condition_hidden: torch.Tensor,
        current_target_hidden: torch.Tensor,
        anchor_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build student inference context and local RoPE positions."""
        if recent_condition_hidden.ndim != 3:
            raise ValueError(
                "recent_condition_hidden must have shape (B,T,H), got "
                f"{tuple(recent_condition_hidden.shape)}."
            )
        if current_target_hidden.shape[:3] != (
            recent_condition_hidden.shape[0],
            1,
            self.condition_slot_count,
        ):
            raise ValueError(
                "current_target_hidden must have shape (B,1,S,H) matching the "
                f"condition tensor; got {tuple(current_target_hidden.shape)}."
            )
        anchor_position = int(anchor_position)
        token_pos = (
            torch.arange(
                anchor_position - self.token_prefix_count + 1,
                anchor_position + 1,
                device=recent_condition_hidden.device,
                dtype=torch.long,
            )
            .view(1, 1, -1)
            .expand(recent_condition_hidden.shape[0], -1, -1)
        )
        token_keep = torch.ones(
            recent_condition_hidden.shape[0],
            1,
            self.token_prefix_count,
            dtype=torch.bool,
            device=recent_condition_hidden.device,
        )
        anchors = torch.full(
            (recent_condition_hidden.shape[0], 1),
            anchor_position,
            dtype=torch.long,
            device=recent_condition_hidden.device,
        )
        context_positions, draft_positions = self.build_block_position_ids(
            anchor_positions=anchors,
            token_position_ids=token_pos,
            token_keep_mask=token_keep,
        )
        return context_positions, draft_positions

    def initialize_inference_condition(
        self,
        target_hidden_states: tuple | list,
        pivot_index: int = -1,
        token_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Students do not retain target KV history between draft steps."""
        reference = target_hidden_states[-1]
        return reference[:, :0, :]

    def update_inference_condition(
        self,
        recent_condition_hidden: torch.Tensor,
        target_hidden_states: tuple | list,
        pivot_index: int,
        token_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Keep the empty student history buffer unchanged."""
        return recent_condition_hidden

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
        target_hidden: torch.Tensor | None = None,
        rotary_position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        if target_hidden is None or target_hidden.ndim != 4:
            raise ValueError("target_hidden must have shape (B,N,S,H).")
        target_hidden = self._fuse_target_hidden(target_hidden)
        noise_len = hidden_states.shape[1]
        if position_ids.shape[1] != noise_len:
            draft_pos = position_ids[:, -noise_len:]
        else:
            draft_pos = position_ids

        rotary_pos = (
            rotary_position_ids if rotary_position_ids is not None else draft_pos
        )
        # Qwen3RotaryEmbedding only reads x.device and x.dtype, so reuse the
        # activation instead of allocating a dense dummy tensor.
        position_embeddings = self.rotary_emb(hidden_states, rotary_pos)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=draft_pos,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def sample_draft_tokens(
        self,
        *,
        draft_hidden: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        compile_serial_head: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample standard DLite draft positions using configured head semantics."""
        if compile_serial_head:
            cache_key = (float(temperature), False)
            compiled_sampler = self._compiled_serial_sampler_cache.get(cache_key)
            if compiled_sampler is None:
                sequential_head = self.sequential_head
                fixed_temperature = float(temperature)

                def serial_sampler(
                    hidden_states: torch.Tensor,
                    previous_ids: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    return sequential_head.sample_block_tokens(
                        hidden_states=hidden_states,
                        first_prev_token_ids=previous_ids,
                        temperature=fixed_temperature,
                    )

                compiled_sampler = torch.compile(
                    serial_sampler,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                self._compiled_serial_sampler_cache[cache_key] = compiled_sampler
            return compiled_sampler(draft_hidden, first_prev_token_ids)
        return self.sequential_head.sample_block_tokens(
            hidden_states=draft_hidden,
            first_prev_token_ids=first_prev_token_ids,
            temperature=temperature,
        )

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        decode_timing_after_first_token: bool = False,
        verify_block_size: int | None = None,
        stochastic_verification_mode: str = "match",
        compile_serial_head: bool = False,
    ):
        self.eval()
        stochastic_verification_mode = validate_verification_mode(
            stochastic_verification_mode
        )
        self._last_decode_stats = {
            "accept_lengths": [],
            "decode_wall_time": 0.0,
            "target_total_time": 0.0,
            "draft_total_time": 0.0,
            "steps": 0,
            "verification_mode": stochastic_verification_mode,
            "compile_serial_head": bool(compile_serial_head),
        }
        bsz = input_ids.shape[0]
        use_rejection_sampling = (
            temperature >= 1e-5 and stochastic_verification_mode == "rejection"
        )
        if use_rejection_sampling and bsz != 1:
            raise ValueError(
                "stochastic rejection sampling currently requires batch size 1."
            )
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        draft_block_len = self.draft_block_len
        proposal_length = self.proposal_length
        verify_block_size = (
            proposal_length + 1 if verify_block_size is None else int(verify_block_size)
        )
        if not 1 <= verify_block_size <= proposal_length + 1:
            raise ValueError(
                f"verify_block_size must be in [1, {proposal_length + 1}], got "
                f"{verify_block_size}"
            )
        output_ids = torch.full(
            (bsz, max_length + proposal_length + 1),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = (
            torch.arange(output_ids.shape[1], device=target.device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )

        past_key_values_target = _new_target_cache(target)

        # Prefill stage (not included in decode wall time)
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample_logits(
            output.logits, temperature
        )
        target_hidden = gather_pivot_hidden_states(
            output.hidden_states,
            self.target_layer_ids,
            -1,
            self.config.num_target_layers,
        )
        recent_condition_hidden = self.initialize_inference_condition(
            output.hidden_states,
        )

        # Decode stage: single cuda-synced wall clock (draft + target + bookkeeping)
        decode_start: float | None = (
            None if decode_timing_after_first_token else cuda_sync_time(target.device)
        )
        acceptance_lengths = []
        start = input_ids.shape[1]
        while start < max_length:
            draft_input_ids = output_ids[:, start : start + draft_block_len].clone()
            noise_embedding = self.build_inference_query_embeddings(
                target.get_input_embeddings(),
                draft_input_ids,
            )
            current_target_hidden = self.build_inference_current_chs(
                target_hidden,
            )
            ctx_pos_part, draft_block_pos = self.build_inference_context(
                recent_condition_hidden,
                current_target_hidden,
                start,
            )
            full_rotary = torch.cat([ctx_pos_part, draft_block_pos], dim=-1)
            block_hidden = self(
                target_hidden=current_target_hidden,
                noise_embedding=noise_embedding,
                position_ids=draft_block_pos,
                rotary_position_ids=full_rotary,
                is_causal=False,
            )
            draft_hidden = self._prediction_hidden(block_hidden)
            draft_temperature = temperature if use_rejection_sampling else 0.0
            sampled_draft_tokens, draft_logits = self.sample_draft_tokens(
                draft_hidden=draft_hidden,
                first_prev_token_ids=draft_input_ids[:, 0],
                temperature=draft_temperature,
                compile_serial_head=compile_serial_head,
            )
            all_verify_output_ids = torch.cat(
                [draft_input_ids[:, :1], sampled_draft_tokens], dim=1
            )

            # The draft always consumes and predicts the full configured block.  Only
            # the requested prefix is sent to the target; the remaining draft tokens
            # are deliberately discarded before verification.
            verify_output_ids = all_verify_output_ids[:, :verify_block_size]
            verify_position_ids = position_ids[:, start : start + verify_block_size]
            recurrent_snapshot = _begin_recurrent_cache_transaction(
                past_key_values_target
            )
            output = target(
                verify_output_ids,
                position_ids=verify_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            if use_rejection_sampling:
                proposal_count = verify_block_size - 1
                acceptance_length, next_token = rejection_sample_verify(
                    proposed_tokens=verify_output_ids[:, 1:],
                    draft_logits=draft_logits[:, :proposal_count, :],
                    target_logits=output.logits,
                    temperature=temperature,
                )
            else:
                posterior = sample_logits(output.logits, temperature)
                acceptance_lengths_per_row = (
                    (verify_output_ids[:, 1:] == posterior[:, :-1])
                    .cumprod(dim=1)
                    .sum(dim=1)
                )
                acceptance_length = int(acceptance_lengths_per_row.min().item())
                next_token = posterior[:, acceptance_length]
            output_ids[:, start : start + acceptance_length + 1] = verify_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = next_token
            kept_verify_tokens = acceptance_length + 1
            cache_length_after_verify = start + verify_block_size
            if recurrent_snapshot is None:
                _crop_target_cache(
                    past_key_values_target,
                    current_length=cache_length_after_verify,
                    tokens_to_remove=verify_block_size - kept_verify_tokens,
                )
            elif kept_verify_tokens == verify_block_size:
                _crop_target_cache(
                    past_key_values_target,
                    current_length=cache_length_after_verify,
                    tokens_to_remove=0,
                )
            else:
                _crop_target_cache(
                    past_key_values_target,
                    current_length=cache_length_after_verify,
                    tokens_to_remove=verify_block_size,
                )
                _restore_recurrent_cache_transaction(recurrent_snapshot)
                target(
                    verify_output_ids[:, :kept_verify_tokens],
                    position_ids=verify_position_ids[:, :kept_verify_tokens],
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    logits_to_keep=1,
                    output_hidden_states=False,
                )
                _crop_target_cache(
                    past_key_values_target,
                    current_length=start + kept_verify_tokens,
                    tokens_to_remove=0,
                )
            start += kept_verify_tokens
            pivot_index = min(acceptance_length, output.hidden_states[0].shape[1] - 1)
            recent_condition_hidden = self.update_inference_condition(
                recent_condition_hidden,
                output.hidden_states,
                pivot_index,
            )
            target_hidden = gather_pivot_hidden_states(
                output.hidden_states,
                self.target_layer_ids,
                pivot_index,
                self.config.num_target_layers,
            )
            acceptance_lengths.append(acceptance_length + 1)
            self._last_decode_stats["accept_lengths"].append(acceptance_length + 1)
            self._last_decode_stats["steps"] += 1

            if decode_timing_after_first_token and decode_start is None:
                decode_start = cuda_sync_time(target.device)

            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
        if decode_start is None:
            decode_start = cuda_sync_time(target.device)
        decode_wall_time = cuda_sync_time(target.device) - decode_start
        self._last_decode_stats["decode_wall_time"] = decode_wall_time
        # Aggregate timing fields; this path does not split target and draft events.
        self._last_decode_stats["target_total_time"] = decode_wall_time
        self._last_decode_stats["draft_total_time"] = 0.0

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        return output_ids
