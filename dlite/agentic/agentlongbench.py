"""Generate on AgentLongBench inputs, recording only request-level acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid

from .common import (AcceptanceLedger, add_selection_args, append_jsonl, fingerprint,
                     fresh_output, jsonl, parse_config, save_json, service_info)
from .prepare import default_data, default_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--profile", choices=["frozen", "legacy50", "legacy-speed3", "official-file", "official-suite"], default="frozen")
    parser.add_argument("--dataset-glob", default="*/*/*/*.jsonl")
    parser.add_argument("--mode", choices=["ar", "sd"], required=True,
                        help="independent run; paired speed comparison is intentionally unsupported")
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--skip-score", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--warmup-tokens", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--rope-policy", help=argparse.SUPPRESS)
    add_selection_args(parser)
    args = parse_config(parser, argv)
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    if args.warmup_tokens:
        parser.error("acceptance-only runs do not perform timed warmups")
    source = (args.source_root or default_source(args.project_root, "agentlongbench")).resolve()
    data = (args.data_root or default_data(source, "agentlongbench")).resolve()
    preparation = None
    if args.prepared:
        preparation = json.loads((args.prepared / "selection_report.json").read_text())
        if preparation.get("benchmark") != "agentlongbench" or preparation.get("status") != "prepared":
            raise ValueError("prepared manifest is not an available AgentLongBench selection")
        if args.manifest is None and args.dataset is None:
            args.manifest = Path(preparation["selection_manifest"])
        for field in ("min_input_tokens", "max_input_tokens"):
            if getattr(args, field) != preparation[field]:
                raise ValueError("run interval differs from prepared selection")
    if args.manifest and args.dataset:
        parser.error("choose --manifest or --dataset")
    rows = []
    if args.manifest:
        rows = jsonl(args.manifest)
        if args.profile.startswith("legacy"):
            from .common import fingerprint as digest
            expected = "3c38b54bac06e247de1dedaa1fb3ac3ff6bdbed2bfbb51a59594ecf21d41bf92"
            if list(digest([args.manifest]).values()) != [expected]:
                raise ValueError("legacy manifest SHA256 mismatch; use frozen for a new selection")
        if args.profile == "legacy-speed3":
            rows = [row for row in rows if row["selection_index"] in {3, 19, 33}]
    elif args.dataset or args.profile == "official-suite":
        sys.path.insert(0, str(source))
        from eval.common.mapping import infer_context_from_path, require_single_question_type
        from eval.common.question_logic import build_prompt
        files = [args.dataset] if args.dataset else sorted(data.glob(args.dataset_glob))
        for path in files:
            source_rows = jsonl(path)
            kind = require_single_question_type(source_rows)
            _, knowledge, _, history = infer_context_from_path(path)
            for index, sample in enumerate(source_rows):
                rows.append({"selection_index": len(rows), "id": sample["id"],
                             "source_path": str(path), "source_index": index,
                             "messages": build_prompt(kind, sample, history, knowledge)})
    else:
        parser.error("set --prepared, --manifest or --dataset")
    if not rows:
        raise ValueError("empty input selection")
    service = service_info(args.base_url, args.mode)
    if service["served_model"] != args.served_model or service["max_output_tokens"] < args.max_new_tokens:
        raise ValueError("service model/output cap mismatch")
    if service["context_limit"] < args.max_input_tokens + args.max_new_tokens + (service.get("verify_block_size") or 0):
        raise ValueError("service context does not cover the selected input window and generation budget")
    if preparation and args.manifest is not None and fingerprint([args.manifest]).get(str(args.manifest)) != preparation["audit"]["manifest_sha256"]:
        raise ValueError("prepared raw prompt manifest changed")
    output = fresh_output(args.output)
    run_id = uuid.uuid4().hex
    ledger = AcceptanceLedger(output, args, run_id)
    report = {"run_id": run_id, "benchmark": "agentlongbench", "mode": args.mode,
              "status": "running", "candidate_requests": len(rows), "service": service,
              "prepared": str(args.prepared) if args.prepared else None,
              "source_sha256": fingerprint([args.manifest] if args.manifest else [args.dataset] if args.dataset else []),
              "scoring_enabled": False, "speed_comparison_enabled": False}
    save_json(output / "run_manifest.json", report)
    import requests
    try:
        with requests.Session() as session:
            for index, row in enumerate(rows):
                if ledger.full:
                    break
                case = {"case_id": f"agentlong:{row.get('selection_index', index)}",
                        "source_id": row.get("id"), "source_path": row.get("source_path"),
                        "source_index": row.get("source_index"), "phase": "generation"}
                payload = {"model": args.served_model, "max_tokens": args.max_new_tokens,
                           "temperature": 0, "stream": False, "return_decode_metrics": True}
                if "raw_prompt" in row:
                    payload["prompt"] = row["raw_prompt"]
                    endpoint = "/completions"
                else:
                    payload.update(messages=row["messages"], chat_template_kwargs={"enable_thinking": False})
                    endpoint = "/chat/completions"
                response_id = None
                try:
                    response = session.post(args.base_url.rstrip("/") + endpoint, json=payload,
                        headers={"x-agent-benchmark": "agentlongbench", "x-agent-case-id": case["case_id"],
                                 "x-agent-run-id": run_id, "x-agent-phase": "generation"}, timeout=args.timeout)
                    response_id = response.headers.get("x-request-id")
                    response.raise_for_status()
                    answer = response.json()
                    response_id = answer["id"]
                    stats = answer["dlite_metrics"]
                    if row.get("prompt_tokens") is not None and stats["input_tokens"] != row["prompt_tokens"]:
                        raise RuntimeError("prepared tokenizer/input length mismatch")
                    if "raw_prompt" in row and stats["prompt_sha256"] != hashlib.sha256(row["raw_prompt"].encode()).hexdigest():
                        raise RuntimeError("authoritative raw prompt changed")
                    chosen = ledger.record({**case, "response_id": response_id}, stats)
                    choice = answer["choices"][0]
                    text = choice["text"] if endpoint == "/completions" else choice["message"]["content"]
                    append_jsonl(output / "responses.jsonl", {**case, "response_id": response_id,
                                 "selected": chosen["selected"], "raw_response": text})
                except Exception as exc:
                    if response_id not in ledger.seen:
                        ledger.record({**case, "response_id": response_id}, error=exc)
                    raise
        report["status"] = "completed" if ledger.full else "insufficient_samples"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["selection"] = ledger.finish()["selection"]
        save_json(output / "run_manifest.json", report)


if __name__ == "__main__":
    main()
