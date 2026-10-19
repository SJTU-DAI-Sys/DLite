"""Reproducible target-vs-DLite benchmark for standard evaluation datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import DynamicCache

from .engine import DLiteEngine

SHORT_TEXT_SAMPLE_COUNTS = {
    "gsm8k": 128,
    "math500": 128,
    "aime25": 30,
    "humaneval": 164,
    "mbpp": 128,
    "livecodebench": 128,
    "mt-bench": 80,
    "alpaca": 128,
}


def _cuda_time(device: torch.device) -> float:
    import time

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _new_target_cache(target):
    try:
        return DynamicCache(config=target.config)
    except TypeError:
        return DynamicCache()


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 1e-5:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    shape = probabilities.shape[:-1]
    return torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]), 1
    ).view(*shape)


@torch.inference_mode()
def target_generate(target, input_ids, max_new_tokens, stop_token_ids, temperature):
    batch_size = input_ids.shape[0]
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    positions = torch.arange(max_length, device=input_ids.device).unsqueeze(0)
    positions = positions.expand(batch_size, -1)
    cache = _new_target_cache(target)
    output = target(
        input_ids,
        position_ids=positions[:, :num_input_tokens],
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=False,
    )
    next_token = _sample(output.logits, temperature)
    started = _cuda_time(target.device)
    cursor = num_input_tokens
    stop_tensor = torch.tensor(stop_token_ids, device=input_ids.device)
    while cursor < max_length:
        cursor += 1
        if stop_tensor.numel() and bool(torch.isin(next_token, stop_tensor).all()):
            break
        if cursor >= max_length:
            break
        output = target(
            next_token,
            position_ids=positions[:, cursor - 1 : cursor],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=False,
        )
        next_token = _sample(output.logits, temperature)
    elapsed = _cuda_time(target.device) - started
    return SimpleNamespace(
        num_output_tokens=cursor - num_input_tokens,
        decode_seconds=elapsed,
    )


def _load_livecodebench():
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    files = [
        hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename=f"test{suffix}.jsonl",
            repo_type="dataset",
        )
        for suffix in ("", "2", "3", "4", "5", "6")
    ]
    return load_dataset("json", data_files={"test": files})["test"]


def _format_livecodebench(row):
    system = (
        "You are an expert Python programmer. You will be given a question "
        "(problem specification) and will generate a correct Python program that "
        "matches the specification and passes all tests. You will NOT return "
        "anything except for the program"
    )
    if row.get("starter_code"):
        format_message = "### Format: Use the following code structure:"
        code = f"```python\n{row['starter_code']}\n```"
    else:
        format_message = "### Format: Write your code in the following format:"
        code = "```python\n# YOUR CODE HERE\n```"
    return (
        f"{system}\n\n### Question:\n{row['question_content']}\n\n"
        f"{format_message}\n{code}\n\n"
        "### Answer: (use the provided format with backticks)"
    )


def _load_short_dataset(name: str):
    from datasets import load_dataset

    if name == "gsm8k":
        data = load_dataset("openai/gsm8k", "main", split="test")
        return [
            {
                "turns": [
                    f"{row['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                ]
            }
            for row in data
        ]
    if name == "math500":
        data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [
            {
                "turns": [
                    f"{row['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                ]
            }
            for row in data
        ]
    if name == "aime25":
        data = load_dataset("MathArena/aime_2025", split="train")
        return [
            {
                "turns": [
                    f"{row['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                ]
            }
            for row in data
        ]
    if name == "humaneval":
        data = load_dataset("openai/openai_humaneval", split="test")
        return [
            {
                "turns": [
                    "Write a solution to the following problem and make sure that it passes the tests:\n```python\n"
                    + row["prompt"]
                    + "\n```"
                ]
            }
            for row in data
        ]
    if name == "mbpp":
        data = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        return [{"turns": [row["prompt"]]} for row in data]
    if name == "livecodebench":
        return [
            {"turns": [_format_livecodebench(row)]} for row in _load_livecodebench()
        ]
    if name == "mt-bench":
        data = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        return [{"turns": list(row["prompt"])} for row in data]
    if name == "alpaca":
        data = load_dataset("tatsu-lab/alpaca", split="train")
        return [
            {
                "turns": [
                    f"{row['instruction']}\n\nInput:\n{row['input']}"
                    if row["input"]
                    else row["instruction"]
                ]
            }
            for row in data
        ]
    raise ValueError(f"Unsupported short-text dataset: {name}")


def _resolve_dataset_path(name: str, dataset_path_file: Path | None) -> Path:
    direct = Path(name)
    if direct.exists():
        return direct
    if dataset_path_file is None:
        raise ValueError(f"Dataset {name!r} requires --dataset-path-file")
    paths = json.loads(dataset_path_file.read_text(encoding="utf-8"))
    if name not in paths:
        raise KeyError(f"Dataset {name!r} is missing from {dataset_path_file}")
    return Path(paths[name])


def _load_long_dataset(name: str, dataset_path_file: Path | None):
    path = _resolve_dataset_path(name, dataset_path_file)
    if path.suffix == ".jsonl":
        serialized = path.read_text(encoding="utf-8")
        if serialized.lstrip("\ufeff").lstrip().startswith("["):
            rows = json.loads(serialized)
        else:
            rows = [
                json.loads(line) for line in serialized.splitlines() if line.strip()
            ]
    elif path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported long-text dataset file: {path}")
    return [
        {"turns": [f"{row['context']}\n\nQuestion: {row['question']}"]} for row in rows
    ]


def load_dataset_records(
    *,
    dataset: str | None,
    prompts: Path | None,
    dataset_path_file: Path | None,
    max_samples: int | None,
):
    if prompts is not None:
        with prompts.open(encoding="utf-8") as handle:
            records = []
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    records.append({"turns": row.get("turns") or [row["prompt"]]})
        dataset_name = prompts.stem
    elif dataset in SHORT_TEXT_SAMPLE_COUNTS:
        dataset_name = str(dataset)
        records = _load_short_dataset(dataset_name)
        if max_samples is None:
            max_samples = SHORT_TEXT_SAMPLE_COUNTS[dataset_name]
    elif dataset and dataset.startswith("longbench_v2_"):
        dataset_name = dataset
        records = _load_long_dataset(dataset_name, dataset_path_file)
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    if max_samples is not None and len(records) > max_samples:
        indices = list(range(len(records)))
        random.Random(0).shuffle(indices)
        records = [records[index] for index in indices[:max_samples]]
    return dataset_name, records


def _render_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DLite against target-only decoding"
    )
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompts", type=Path)
    source.add_argument("--dataset")
    parser.add_argument("--dataset-path-file", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--attention-backend",
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--verification-mode", default="match", choices=["match", "rejection"]
    )
    parser.add_argument("--compile-sequential-head", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")

    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    dataset_name, records = load_dataset_records(
        dataset=args.dataset,
        prompts=args.prompts,
        dataset_path_file=args.dataset_path_file,
        max_samples=args.max_samples,
    )
    engine = DLiteEngine.from_pretrained(
        args.target_model,
        args.draft_model,
        attention_backend=args.attention_backend,
    )
    stop_token_ids = [
        token_id for token_id in [engine.tokenizer.eos_token_id] if token_id is not None
    ]

    warmup_ids = _render_messages(
        engine.tokenizer, [{"role": "user", "content": "Warmup."}]
    ).to(engine.target.device)
    target_generate(
        engine.target,
        warmup_ids,
        min(16, args.max_new_tokens),
        stop_token_ids,
        args.temperature,
    )
    engine.draft.spec_generate(
        target=engine.target,
        input_ids=warmup_ids,
        max_new_tokens=min(16, args.max_new_tokens),
        stop_token_ids=stop_token_ids,
        temperature=args.temperature,
        stochastic_verification_mode=args.verification_mode,
        compile_serial_head=args.compile_sequential_head,
    )
    torch.cuda.empty_cache()

    turns = []
    acceptance_lengths = []
    generated_tokens = 0
    baseline_tokens = 0
    baseline_seconds = 0.0
    dlite_seconds = 0.0
    for sample_index, record in enumerate(records):
        messages = []
        for turn_index, question in enumerate(record["turns"]):
            messages.append({"role": "user", "content": question})
            input_ids = _render_messages(engine.tokenizer, messages).to(
                engine.target.device
            )
            baseline = target_generate(
                engine.target,
                input_ids,
                args.max_new_tokens,
                stop_token_ids,
                args.temperature,
            )
            speculative_ids = engine.draft.spec_generate(
                target=engine.target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
                stochastic_verification_mode=args.verification_mode,
                compile_serial_head=args.compile_sequential_head,
            )
            generated = speculative_ids[:, input_ids.shape[1] :]
            stats = engine.draft.get_last_decode_stats()
            speculative_seconds = float(stats["decode_wall_time"])
            num_tokens = int(generated.shape[1])
            generated_tokens += num_tokens
            baseline_tokens += int(baseline.num_output_tokens)
            baseline_seconds += float(baseline.decode_seconds)
            dlite_seconds += speculative_seconds
            acceptance_lengths.extend(stats.get("accept_lengths", []))
            output_text = engine.tokenizer.decode(
                generated[0], skip_special_tokens=True
            )
            messages.append({"role": "assistant", "content": output_text})
            turn = {
                "sample": sample_index,
                "turn": turn_index,
                "input_tokens": int(input_ids.shape[1]),
                "output_tokens": num_tokens,
                "baseline_output_tokens": int(baseline.num_output_tokens),
                "accept_lengths": list(stats.get("accept_lengths", [])),
            }
            turns.append(turn)
            print(json.dumps(turn, ensure_ascii=False), flush=True)

    baseline_s_per_token = baseline_seconds / max(baseline_tokens, 1)
    dlite_s_per_token = dlite_seconds / max(generated_tokens, 1)
    report = {
        "dataset": dataset_name,
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "samples": len(records),
        "turns": len(turns),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "baseline_tokens_per_second": baseline_tokens / max(baseline_seconds, 1e-30),
        "dlite_tokens_per_second": generated_tokens / max(dlite_seconds, 1e-30),
        "token_weighted_speedup": baseline_s_per_token / max(dlite_s_per_token, 1e-30),
        "mean_accept_length": sum(acceptance_lengths) / max(len(acceptance_lengths), 1),
        "turn_results": turns,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
