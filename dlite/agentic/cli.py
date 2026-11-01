"""Public entry points. Parsing help and preparing data do not load a model."""
from __future__ import annotations

import argparse
import importlib
import sys

from .common import BENCHMARKS, load_config


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m dlite.agentic {prepare,serve,run,summarize} [options]\n"
              "prepare: fetch/audit raw data and select candidate inputs\n"
              "serve: start an explicitly requested AR or SD model service\n"
              "run: native benchmark generation with request-level selection\n"
              "summarize: recompute acceptance length from selected request logs\n"
              "Use COMMAND --help for options. No command starts a benchmark implicitly.")
        return 0
    command, rest = argv[0], argv[1:]
    if command == "run":
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--benchmark", choices=BENCHMARKS)
        pre.add_argument("--config")
        known, _ = pre.parse_known_args(rest)
        benchmark = known.benchmark or load_config(known.config).get("benchmark")
        if not benchmark:
            print("usage: python -m dlite.agentic run --benchmark {agentlongbench,bfcl,longmemeval} "
                  "[--config FILE] [options]")
            return 0 if "--help" in rest else 2
        if benchmark not in BENCHMARKS:
            raise ValueError("unsupported benchmark")
        module = benchmark
    else:
        module = {"prepare": "prepare", "serve": "server", "summarize": "summarize"}.get(command)
        if module is None:
            raise SystemExit(f"Unknown command: {command}")
    result = importlib.import_module(f".{module}", __package__).main(rest)
    return result or 0
