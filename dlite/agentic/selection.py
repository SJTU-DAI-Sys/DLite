#!/usr/bin/env python3
"""Build a deterministic long-context AgentLongBench acceptance test set.

The output keeps the exact Qwen-rendered ``raw_prompt``.  Benchmark runners must
encode that field directly with ``add_special_tokens=False``; wrapping it in a
second chat template would change both the measured length and the model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import project_root, append_jsonl


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", "--agentlong-root", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--data-root", type=Path,
                        help="Benchmark directory; defaults to source-root/benchmark")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--length-dir",
        action="append",
        default=None,
        help="AgentLongBench nominal length directory; repeatable (default: 64k).",
    )
    parser.add_argument("--min-tokens", "--min-input-tokens", type=int, default=32000)
    parser.add_argument("--max-tokens", "--max-input-tokens", type=int, default=100000)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--reuse-selection", type=Path,
                        help="Preserve an existing ordered source-id selection after revalidating lengths")
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def evenly_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order one source's rows from across its token-length distribution."""
    rows = sorted(rows, key=lambda row: (row["prompt_tokens"], row["source_index"]))
    if len(rows) < 2:
        return rows
    order = []
    left, right = 0, len(rows) - 1
    while left <= right:
        middle = (left + right) // 2
        order.append(rows[middle])
        if middle != left:
            order.append(rows[left])
        if middle != right:
            order.append(rows[right])
        used = {left, middle, right}
        rows = [row for index, row in enumerate(rows) if index not in used]
        left, right = 0, len(rows) - 1
    # The iterative removal can only duplicate when the source has 2 rows.
    deduped = []
    seen = set()
    for row in order:
        key = (row["source_path"], row["source_index"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def select_round_robin(
    candidates: list[dict[str, Any]], max_samples: int
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_source[row["source_path"]].append(row)
    queues = {key: evenly_order(value) for key, value in sorted(by_source.items())}
    selected = []
    while len(selected) < max_samples and queues:
        empty = []
        for source, queue in queues.items():
            if queue:
                selected.append(queue.pop(0))
                if len(selected) == max_samples:
                    break
            if not queue:
                empty.append(source)
        for source in empty:
            queues.pop(source, None)
    return selected


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.min_tokens < 1 or args.max_tokens <= args.min_tokens:
        raise ValueError("invalid token interval")
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    repo_root = (args.source_root or project_root(args.project_root) / "thirdparty/AgentLongBench").resolve()
    data_root = (args.data_root or repo_root / "benchmark").resolve()
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError("Choose new output/summary paths; existing selections are preserved")
    sys.path.insert(0, str(repo_root))
    from eval.common.mapping import infer_context_from_path, require_single_question_type
    from eval.common.question_logic import build_prompt

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    length_dirs = args.length_dir or ["64k"]
    data_files = []
    for length_dir in length_dirs:
        data_files.extend(
            sorted(data_root.glob(f"*/{length_dir}/*/*.jsonl"))
        )
    if not data_files:
        raise FileNotFoundError(
            f"no AgentLongBench JSONL files found for {length_dirs} under {data_root}"
        )

    candidates: list[dict[str, Any]] = []
    scanned = 0
    all_lengths = []
    exclusions = defaultdict(int)
    candidate_audit = summary_path.with_name("candidate_audit.jsonl")
    if candidate_audit.exists():
        raise FileExistsError(candidate_audit)
    for path in sorted(set(data_files)):
        rows = load_jsonl(path)
        question_type = require_single_question_type(rows)
        _, knowledge_label, _, history_label = infer_context_from_path(path)
        # Logical path stays stable when the project or dataset root moves.
        relative_path = str(Path("benchmark") / path.relative_to(data_root))
        for source_index, sample in enumerate(rows):
            scanned += 1
            messages = build_prompt(
                question_type, sample, history_label, knowledge_label
            )
            raw_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_tokens = len(
                tokenizer.encode(raw_prompt, add_special_tokens=False)
            )
            all_lengths.append(prompt_tokens)
            reason = ("at_or_below_min" if prompt_tokens <= args.min_tokens else
                      "at_or_above_max" if prompt_tokens >= args.max_tokens else "eligible")
            exclusions[reason] += 1
            append_jsonl(candidate_audit, {"source_path": relative_path, "source_index": source_index,
                         "id": sample.get("id"), "input_tokens": prompt_tokens, "selection_reason": reason})
            if reason != "eligible":
                continue
            tool_messages = [m for m in messages if m.get("role") == "tool"]
            candidates.append(
                {
                    "turns": ["raw_prompt is authoritative; do not chat-wrap this field"],
                    "raw_prompt": raw_prompt,
                    "prompt_tokens": prompt_tokens,
                    "source_path": relative_path,
                    "source_index": source_index,
                    "id": sample.get("id"),
                    "sample_id": sample.get("sample_id"),
                    "question_type": question_type,
                    "knowledge_type": knowledge_label,
                    "history_type": history_label,
                    "message_count": len(messages),
                    "tool_message_count": len(tool_messages),
                    "tool_result_chars": sum(
                        len(str(message.get("content") or ""))
                        for message in tool_messages
                    ),
                }
            )

    if args.reuse_selection:
        old = load_jsonl(args.reuse_selection)
        lookup = {(row["source_path"], row["source_index"]): row for row in candidates}
        keys = [(row["source_path"], row["source_index"]) for row in old[:args.max_samples]]
        if len(keys) != len(set(keys)) or any(key not in lookup for key in keys):
            raise ValueError("Reused selection has duplicates or no longer satisfies the input interval")
        selected = [lookup[key] for key in keys]
    else:
        selected = select_round_robin(candidates, min(args.max_samples, len(candidates)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for selection_index, row in enumerate(selected):
            output = dict(row)
            output["selection_index"] = selection_index
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    lengths = [int(row["prompt_tokens"]) for row in selected]
    summary = {
        "source_root": str(repo_root),
        "data_root": str(data_root),
        "manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "selection_method": "legacy length-order + source round-robin, no random resampling",
        "tokenizer": str(args.tokenizer),
        "length_dirs": length_dirs,
        "min_tokens_exclusive": args.min_tokens,
        "max_tokens_exclusive": args.max_tokens,
        "scanned_samples": scanned,
        "candidate_audit": str(candidate_audit),
        "candidate_reasons": dict(exclusions),
        "candidate_min_tokens": min(all_lengths) if all_lengths else None,
        "candidate_max_tokens": max(all_lengths) if all_lengths else None,
        "eligible_samples": len(candidates),
        "selected_samples": len(selected),
        "requested_samples": args.max_samples,
        "insufficient_samples": len(selected) < args.max_samples,
        "reused_selection": str(args.reuse_selection) if args.reuse_selection else None,
        "selected_min_tokens": min(lengths) if lengths else None,
        "selected_max_tokens": max(lengths) if lengths else None,
        "selected_mean_tokens": sum(lengths) / len(lengths) if lengths else None,
        "selected_p50_tokens": percentile(lengths, 0.50),
        "selected_p90_tokens": percentile(lengths, 0.90),
        "selected_by_setting": dict(
            sorted(
                {
                    setting: sum(
                        f"{row['knowledge_type']}:{row['history_type']}" == setting
                        for row in selected
                    )
                    for setting in {
                        f"{row['knowledge_type']}:{row['history_type']}"
                        for row in selected
                    }
                }.items()
            )
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
