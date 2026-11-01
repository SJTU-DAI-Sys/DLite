"""Configuration, provenance and request-level acceptance selection. No ML imports."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import threading
import time

BENCHMARKS = ("agentlongbench", "bfcl", "longmemeval")
_lock = threading.Lock()


def project_root(value=None):
    return Path(value or os.getenv("DLITE_ROOT") or Path(__file__).resolve().parents[2]).resolve()


def load_config(path):
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    def expand(value):
        if isinstance(value, str):
            return os.path.expanduser(os.path.expandvars(value))
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value
    return {key.replace("-", "_"): expand(value) for key, value in data.items()}


def parse_config(parser, argv=None):
    parser.add_argument("--config", type=Path, help="Flat JSON defaults; explicit CLI arguments win")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args(argv)
    config = load_config(known.config)
    actions = {action.dest: action for action in parser._actions}
    defaults = {key: value for key, value in config.items() if key in actions}
    for key in defaults:
        actions[key].required = False
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.project_root = project_root(args.project_root)
    return args


def add_selection_args(parser):
    parser.add_argument("--min-input-tokens", type=int, default=32000)
    parser.add_argument("--max-input-tokens", type=int, default=100000)
    parser.add_argument("--max-samples", type=int, default=50)


def validate_selection(args):
    if not 0 <= args.min_input_tokens < args.max_input_tokens or args.max_samples < 1:
        raise ValueError("invalid strict input interval or max-samples")


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": time.time(), **row}, ensure_ascii=False) + "\n")


def save_json(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fingerprint(paths):
    result = {}
    for path in paths:
        if Path(path).is_file():
            with Path(path).open("rb") as handle:
                result[str(path)] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result


def jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fresh_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError("choose a new empty output directory; existing runs are preserved")
    path.mkdir(parents=True, exist_ok=True)
    return path


def service_info(base_url, expected_mode):
    from urllib.request import urlopen
    root = base_url.rstrip("/").removesuffix("/v1")
    with urlopen(root + "/health", timeout=10) as response:
        info = json.load(response)
    if info.get("backend") != expected_mode:
        raise RuntimeError(f"service backend {info.get('backend')!r} != {expected_mode!r}")
    return info


def response_stats(response):
    value = getattr(response, "dlite_metrics", None)
    if value is None:
        value = (getattr(response, "model_extra", None) or {}).get("dlite_metrics")
    if not isinstance(value, dict):
        raise RuntimeError("service did not return dlite_metrics; use the migrated agentic service")
    return value


class SamplingComplete(BaseException):
    """Stop before starting a new native case, never in the middle of a case."""


class AcceptanceLedger:
    def __init__(self, output, args, run_id):
        validate_selection(args)
        self.output, self.args, self.run_id = Path(output), args, run_id
        self.selected = []
        self.reasons = Counter()
        self.seen = set()

    @property
    def full(self):
        return len(self.selected) >= self.args.max_samples

    def selection_reason(self, input_tokens):
        if input_tokens <= self.args.min_input_tokens:
            return "at_or_below_min"
        if input_tokens >= self.args.max_input_tokens:
            return "at_or_above_max"
        return "quota_reached" if self.full else "selected"

    def record(self, metadata, stats=None, error=None):
        row = {**metadata, "run_id": self.run_id, "mode": self.args.mode}
        if error is not None:
            row.update(status="error", error=str(error), selected=False, selection_reason="request_failed")
        else:
            if stats.get("backend") != self.args.mode:
                raise RuntimeError("service backend differs from the requested independent run")
            if self.args.mode == "sd":
                block = stats["verify_block_size"]
                if not isinstance(block, int) or block < 1 or any(
                        not isinstance(length, int) or not 1 <= length <= block
                        for length in stats["accept_lengths"]):
                    raise RuntimeError("invalid core acceptance-length statistics")
            count = int(stats["input_tokens"])
            reason = self.selection_reason(count)
            if metadata.get("request_start_selection_reason", reason) != reason:
                raise RuntimeError("request-start selection differs from returned metrics")
            row.update(status="ok", input_tokens=count, output_tokens=stats["output_tokens"],
                       prompt_sha256=stats["prompt_sha256"], verify_block_size=stats["verify_block_size"],
                       accept_lengths=list(stats["accept_lengths"]) if self.args.mode == "sd" else [],
                       selected=reason == "selected", selection_reason=reason)
        identity = row.get("response_id")
        if identity and identity in self.seen:
            raise RuntimeError("duplicate response ID")
        if identity:
            self.seen.add(identity)
        self.reasons[row["selection_reason"]] += 1
        append_jsonl(self.output / "request_metrics.jsonl", row)
        if row["selected"]:
            self.selected.append(row)
            append_jsonl(self.output / "selected_requests.jsonl", row)
        self.finish()
        return row

    def finish(self, write=True):
        lengths = [length for row in self.selected for length in row["accept_lengths"]]
        phases = {}
        for phase in sorted({str(row.get("phase", "generation")) for row in self.selected}):
            rows = [row for row in self.selected if str(row.get("phase", "generation")) == phase]
            values = [length for row in rows for length in row["accept_lengths"]]
            phases[phase] = {"requests": len(rows), "verification_steps": len(values),
                             "mean_accept_length": sum(values) / len(values) if values else None}
        report = {"run_id": self.run_id, "mode": self.args.mode,
                  "selection": {"min_exclusive": self.args.min_input_tokens,
                                "max_exclusive": self.args.max_input_tokens,
                                "requested_samples": self.args.max_samples,
                                "selected_samples": len(self.selected),
                                "insufficient_samples": not self.full,
                                "request_reasons": dict(self.reasons)},
                  "verification_steps": len(lengths),
                  "mean_accept_length": sum(lengths) / len(lengths) if lengths else None,
                  "acceptance_histogram": dict(sorted(Counter(lengths).items())),
                  "by_phase": phases,
                  "definition": "sum(raw accept_lengths)/verification steps; includes one target anchor; "
                                "AR acceptance is not applicable; no scoring or speed comparison"}
        if write:
            save_json(self.output / "acceptance_summary.json", report)
        return report
