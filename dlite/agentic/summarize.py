"""Recompute acceptance length only. No scores, throughput or AR/SD comparisons."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

from .common import AcceptanceLedger, jsonl, parse_config, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("metrics", nargs="*", type=Path, help="legacy selected request JSONL inputs")
    parser.add_argument("--output", type=Path)
    args = parse_config(parser, argv)
    if args.run_dir is None:
        if len(args.metrics) != 1:
            parser.error("set --run-dir; legacy use accepts one selected_requests.jsonl")
        if args.metrics[0].name != "selected_requests.jsonl":
            parser.error("use the runner's selected_requests.jsonl, not an unfiltered service timing log")
        args.run_dir = args.metrics[0].parent
    run = args.run_dir.resolve()
    selected_path = run / "selected_requests.jsonl"
    if not selected_path.exists():
        existing = run / "acceptance_summary.json"
        if not existing.exists():
            raise ValueError("no selected request log or explicit empty/blocked summary")
        report = json.loads(existing.read_text())
    else:
        previous = json.loads((run / "acceptance_summary.json").read_text())
        selection = previous["selection"]
        settings = SimpleNamespace(mode=previous["mode"], min_input_tokens=selection["min_exclusive"],
                                   max_input_tokens=selection["max_exclusive"], max_samples=selection["requested_samples"])
        ledger = AcceptanceLedger(run, settings, previous["run_id"])
        ledger.selected = jsonl(selected_path)
        if len(ledger.selected) > settings.max_samples:
            raise ValueError("selected quota exceeded")
        ids = [row["response_id"] for row in ledger.selected]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate selected response")
        for row in ledger.selected:
            if row["mode"] != settings.mode or not settings.min_input_tokens < row["input_tokens"] < settings.max_input_tokens:
                raise ValueError("selected request mode/length is inconsistent")
        ledger.reasons = Counter(row["selection_reason"] for row in jsonl(run / "request_metrics.jsonl"))
        report = ledger.finish(write=False)
    save_json(args.output or run / "acceptance_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
