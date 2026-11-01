"""Native BFCL multi_turn_long_context generation, with dynamic length selection."""
from __future__ import annotations

import argparse
from contextvars import ContextVar
import functools
import json
import os
from pathlib import Path
import sys
import time
import uuid

from .common import (AcceptanceLedger, SamplingComplete, add_selection_args, append_jsonl,
                     fingerprint, fresh_output, parse_config, response_stats, save_json, service_info)
from .prepare import default_data, default_source

CASE = ContextVar("bfcl_case", default=None)


def install_hooks(args, output, run_id, ledger):
    from bfcl_eval.model_handler.base_handler import BaseHandler
    from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
    original_inference = OSSHandler.inference
    original_next = BaseHandler._add_next_turn_user_message_prompting
    original_format = QwenFCHandler._format_prompt

    @functools.wraps(original_inference)
    def inference(self, test_entry, *positional, **kwargs):
        if ledger.full:
            raise SamplingComplete()
        case = {"case_id": test_entry["id"], "turn": 0, "step": 0, "phase": "multi_turn"}
        token = CASE.set(case)
        try:
            return original_inference(self, test_entry, *positional, **kwargs)
        finally:
            append_jsonl(output / "completed_cases.jsonl", dict(case))
            CASE.reset(token)

    @functools.wraps(original_next)
    def next_turn(self, *positional, **kwargs):
        case = CASE.get()
        if case is not None:
            case["turn"] += 1
            case["step"] = 0
        return original_next(self, *positional, **kwargs)

    @functools.wraps(original_format)
    def format_prompt(self, *positional, **kwargs):
        prompt = original_format(self, *positional, **kwargs)
        if not args.enable_thinking:
            if not prompt.endswith("<|im_start|>assistant\n"):
                raise RuntimeError("Qwen prompt suffix changed; review the template hook")
            prompt += "<think>\n\n</think>\n\n"
        return prompt

    def query(self, inference_data):
        case = CASE.get()
        if case is None:
            raise RuntimeError("no native case context")
        prompt = self._format_prompt(inference_data["message"], inference_data["function"])
        inference_data["inference_input_log"] = {"formatted_prompt": prompt}
        count = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        start_reason = ledger.selection_reason(count)
        metadata = {**case, "request_start_selection_reason": start_reason}
        append_jsonl(output / "request_candidates.jsonl", {**metadata, "input_tokens": count})
        remaining = args.context_limit - count - args.verify_reserve
        limit = min(args.max_output_tokens, remaining) if remaining > 0 else args.max_output_tokens
        extra = {"return_decode_metrics": True}
        for key in ("stop_token_ids", "skip_special_tokens"):
            if hasattr(self, key):
                extra[key] = getattr(self, key)
        headers = {"x-agent-benchmark": "bfcl_multi_turn_long_context", "x-agent-run-id": run_id,
                   "x-agent-case-id": case["case_id"], "x-agent-turn": str(case["turn"]),
                   "x-agent-step": str(case["step"]), "x-agent-phase": "multi_turn"}
        response_id = None
        started = time.perf_counter()
        try:
            response = self.client.completions.create(
                model=args.served_model, prompt=prompt, max_tokens=limit, temperature=0,
                extra_body=extra, extra_headers=headers, timeout=args.timeout)
            response_id = response.id
            stats = response_stats(response)
            if int(stats["input_tokens"]) != count:
                raise RuntimeError("BFCL tokenizer/service input token count mismatch")
            ledger.record({**metadata, "response_id": response_id}, stats)
            return response, time.perf_counter() - started
        except Exception as exc:
            if response_id not in ledger.seen:
                ledger.record({**metadata, "response_id": response_id}, error=exc)
            raise
        finally:
            case["step"] += 1

    OSSHandler.inference = inference
    BaseHandler._add_next_turn_user_message_prompting = next_turn
    QwenFCHandler._format_prompt = format_prompt
    OSSHandler._query_prompting = query


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--action", choices=["generate", "all", "evaluate"], default="generate",
                        help="legacy all now means generation only; evaluate is disabled")
    parser.add_argument("--mode", choices=["ar", "sd"], required=True)
    parser.add_argument("--model", choices=["Qwen/Qwen3-4B-FC", "Qwen/Qwen3-8B-FC", "Qwen/Qwen3-14B-FC"], required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--target-model", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--categories", default="multi_turn_long_context")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--enable-thinking", action="store_true")
    add_selection_args(parser)
    args = parse_config(parser, argv)
    if args.action == "evaluate":
        parser.error("correctness scoring is disabled in this acceptance-only entry")
    if args.categories != "multi_turn_long_context":
        parser.error("this workload is multi_turn_long_context; Memory is a different benchmark")
    if args.max_output_tokens < 1:
        parser.error("output budget must be positive")
    source = (args.source_root or default_source(args.project_root, "bfcl")).resolve()
    data = (args.data_root or default_data(source, "bfcl")).resolve()
    candidates = None
    if args.prepared:
        preparation = json.loads((args.prepared / "selection_report.json").read_text())
        if preparation.get("benchmark") != "bfcl" or preparation.get("status") != "prepared_dynamic":
            raise ValueError("not a prepared BFCL long-context candidate set")
        source = (args.source_root or Path(preparation["source_root"])).resolve()
        data = (args.data_root or Path(preparation["data_root"])).resolve()
        if args.dataset is None:
            args.dataset = Path(preparation["dataset"])
        expected = preparation["source_sha256"].get(preparation["dataset"])
        if fingerprint([args.dataset]).get(str(args.dataset)) != expected:
            raise ValueError("BFCL prepared dataset changed")
        candidates = json.loads((args.prepared / "candidates.json").read_text())
        for key in ("min_input_tokens", "max_input_tokens"):
            if getattr(args, key) != preparation[key]:
                raise ValueError("run interval differs from preparation")
    dataset = (args.dataset or data / "BFCL_v4_multi_turn_long_context.json").resolve()
    if not (source / "bfcl_eval/__main__.py").is_file() or not (data / "multi_turn_func_doc").is_dir():
        raise ValueError("BFCL native source/tool definitions missing; specify the complete bundled source/data root")
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    service = service_info(args.base_url, args.mode)
    if service["served_model"] != args.served_model or Path(service["target_model"]).resolve() != args.target_model.resolve():
        raise ValueError("service target/served model mismatch")
    if service["max_output_tokens"] < args.max_output_tokens:
        raise ValueError("service output cap below benchmark budget")
    args.context_limit = service["context_limit"]
    args.verify_reserve = service.get("verify_block_size") or 0
    if args.context_limit < args.max_input_tokens + args.max_output_tokens + args.verify_reserve:
        raise ValueError("service context must accommodate the input window plus output/verification")
    output = fresh_output(args.output)
    os.environ.update(BFCL_PROJECT_ROOT=str(output), REMOTE_OPENAI_BASE_URL=args.base_url,
                      REMOTE_OPENAI_TOKENIZER_PATH=str(args.target_model.resolve()))
    os.environ.setdefault("REMOTE_OPENAI_API_KEY", "EMPTY")
    sys.path.insert(0, str(source))
    from bfcl_eval.constants import eval_config
    # Native loader expects its standard filename. A per-run view supports custom
    # dataset paths without copying over or editing raw thirdparty data.
    native_data = output / "native_data"
    native_data.mkdir()
    for path in data.iterdir():
        if path.name != "BFCL_v4_multi_turn_long_context.json":
            (native_data / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    (native_data / "BFCL_v4_multi_turn_long_context.json").symlink_to(dataset)
    eval_config.PROMPT_PATH = native_data
    eval_config.MULTI_TURN_FUNC_DOC_PATH = data / "multi_turn_func_doc"
    eval_config.POSSIBLE_ANSWER_PATH = data / "possible_answer"
    eval_config.MEMORY_PREREQ_CONVERSATION_PATH = data / "memory_prereq_conversation"
    if candidates is not None:
        save_json(output / "test_case_ids_to_generate.json", {
            "multi_turn_long_context": [row["id"] for row in candidates]})
    run_id = uuid.uuid4().hex
    ledger = AcceptanceLedger(output, args, run_id)
    install_hooks(args, output, run_id, ledger)
    from bfcl_eval.__main__ import cli
    command = ["generate", "--model", args.model, "--test-category", "multi_turn_long_context",
               "--skip-server-setup", "--num-threads", "1", "--temperature", "0",
               "--include-input-log", "--local-model-path", str(args.target_model.resolve())]
    if candidates is not None:
        command.append("--run-ids")
    report = {"run_id": run_id, "benchmark": "bfcl", "category": "multi_turn_long_context",
              "mode": args.mode, "status": "running", "service": service,
              "command": command, "source_sha256": fingerprint([Path(__file__),
                  dataset]),
              "scoring_enabled": False, "speed_comparison_enabled": False}
    save_json(output / "run_manifest.json", report)
    try:
        cli(args=command, standalone_mode=False)
        report["status"] = "completed" if ledger.full else "insufficient_samples"
    except SamplingComplete:
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["selection"] = ledger.finish()["selection"]
        save_json(output / "run_manifest.json", report)


if __name__ == "__main__":
    main()
