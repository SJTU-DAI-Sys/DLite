"""Fetch raw dataset metadata, preserve provenance, and prepare benchmark inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
from urllib.request import urlopen

from .common import (BENCHMARKS, add_selection_args, fingerprint, fresh_output,
                     jsonl, parse_config, save_json, validate_selection)


def default_source(project, benchmark):
    return project / "thirdparty" / {
        "agentlongbench": "AgentLongBench", "longmemeval": "LongMemEval-V2",
        "bfcl": "BFCL_v4_memory_source/berkeley-function-call-leaderboard"}[benchmark]


def default_data(source, benchmark):
    return source / {"agentlongbench": "benchmark", "longmemeval": "data/longmemeval-v2",
                     "bfcl": "bfcl_eval/data"}[benchmark]


def fetch_hf(repo, filename, destination, revision):
    from huggingface_hub import hf_hub_download
    cached = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename, revision=revision))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        shutil.copyfile(cached, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    resolved = cached.resolve().parts
    return resolved[resolved.index("snapshots") + 1] if "snapshots" in resolved else revision


def acquire(args, source, data):
    """Downloads are only invoked by an explicit prepare command, never imports/run."""
    downloaded = []
    if args.benchmark == "agentlongbench":
        if any(data.glob("*/*/*/*.jsonl")):
            return {"status": "existing_data", "downloaded": downloaded}
        if args.download == "never":
            return {"status": "missing_data", "downloaded": downloaded}
        archive = data.parent / "benchmark.tar.gz"
        revision = args.revision
        if not archive.exists():
            revision = fetch_hf(args.repo_id or "ign1s/AgentLongBench", "benchmark.tar.gz", archive, revision)
            downloaded.append(str(archive))
        # Extract only regular benchmark files; no tar symlinks or path traversal.
        with tarfile.open(archive, "r:*") as bundle:
            for member in bundle:
                parts = PurePosixPath(member.name)
                if parts.is_absolute() or ".." in parts.parts or member.issym() or member.islnk():
                    raise ValueError("unsafe dataset archive member")
                if not member.isfile() or not parts.parts or parts.parts[0] != "benchmark":
                    continue
                target = data.joinpath(*parts.parts[1:])
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = bundle.extractfile(member)
                with stream, target.open("xb") as handle:
                    shutil.copyfileobj(stream, handle)
        return {"status": "downloaded_or_extracted", "downloaded": downloaded, "revision": revision}
    if args.benchmark == "bfcl":
        target = args.dataset or data / "BFCL_v4_multi_turn_long_context.json"
        if target.exists() or args.download == "never":
            return {"status": "existing_data" if target.exists() else "missing_data", "downloaded": []}
        revision = args.revision or "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
        url = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/" + revision +
               "/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multi_turn_long_context.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return {"status": "downloaded", "url": url, "revision": revision, "downloaded": [str(target)]}
    names = ("questions.jsonl", "trajectories.jsonl", "haystacks/lme_v2_small.json", "haystacks/lme_v2_medium.json")
    revision = args.revision
    missing = []
    for name in names:
        target = data / name
        if target.exists():
            continue
        if args.download == "never":
            missing.append(name)
        else:
            revision = fetch_hf(args.repo_id or "xiaowu0162/longmemeval-v2", name, target, revision)
            downloaded.append(str(target))
    return {"status": "missing_data" if missing else "metadata_available", "missing": missing,
            "downloaded": downloaded, "revision": revision,
            "images_downloaded": False, "note": "strict whole-image-record exclusion needs metadata only"}


def audit_longmemeval(data, output, args):
    questions_path, trajectories_path = data / "questions.jsonl", data / "trajectories.jsonl"
    questions = jsonl(questions_path) if questions_path.exists() else []
    picture_questions = {str(row["id"]) for row in questions if row.get("image")}
    result = {"question_records": len(questions), "picture_questions": len(picture_questions),
              "text_questions": len(questions) - len(picture_questions),
              "image_policy": "drop_whole_record", "selected_samples": 0,
              "native_r_requires_screenshots": True,
              "length_distribution": None,
              "length_note": "no reader prompts were generated; tool/retrieval evidence is already "
                             "part of memory_context, not an additional token budget"}
    if not trajectories_path.exists():
        result.update(status="missing_data", reason="trajectories.jsonl missing; no trajectory or final-prompt audit possible",
                      trajectory_records=None, image_trajectories=None)
    else:
        total, image_ids, invalid_ids = 0, set(), set()
        with trajectories_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                total += 1
                states = row.get("states") or []
                if row.get("image") or any(state.get("screenshot") or state.get("image")
                                            for state in states if isinstance(state, dict)):
                    image_ids.add(str(row["id"]))
                else:
                    # Native R requires nonempty states AND a screenshot for every state.
                    invalid_ids.add(str(row["id"]))
        result.update(status="blocked_no_compatible_trajectories", trajectory_records=total,
                      image_trajectories=len(image_ids), incompatible_image_free_records=len(invalid_ids),
                      reason="whole-record image exclusion removes native R trajectories; "
                             "image-free records do not satisfy its screenshot schema")
        save_json(output / "image_exclusions.json", {"question_ids": sorted(picture_questions),
                  "trajectory_ids": sorted(image_ids), "schema_incompatible_ids": sorted(invalid_ids)})
    save_json(output / "candidates.json", [])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset", type=Path, help="BFCL raw long-context JSONL file; other datasets use --data-root")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", choices=["missing", "never"], default="missing")
    parser.add_argument("--repo-id")
    parser.add_argument("--revision")
    parser.add_argument("--length-dir", action="append", dest="length_dirs")
    parser.add_argument("--reuse-selection", type=Path)
    add_selection_args(parser)
    args = parse_config(parser, argv)
    if args.benchmark not in BENCHMARKS:
        parser.error("set --benchmark or config benchmark")
    validate_selection(args)
    source = (args.source_root or default_source(args.project_root, args.benchmark)).resolve()
    data = (args.data_root or default_data(source, args.benchmark)).resolve()
    marker = {"agentlongbench": "eval/common/question_logic.py",
              "bfcl": "bfcl_eval/__main__.py"}.get(args.benchmark)
    if marker and not (source / marker).is_file():
        raise FileNotFoundError(
            f"Benchmark source missing at {source}. Run "
            "git submodule update --init --recursive, or pass --source-root.")
    output = fresh_output(args.output)
    report = {"benchmark": args.benchmark, "source_root": str(source), "data_root": str(data),
              "tokenizer": str(args.tokenizer) if args.tokenizer else None,
              "min_input_tokens": args.min_input_tokens, "max_input_tokens": args.max_input_tokens,
              "max_samples": args.max_samples, "interval": "strict; request-start full prompt tokens",
              "model_inference_run": False, "runtime_verified": False}
    try:
        report["acquisition"] = acquire(args, source, data)
        if args.benchmark == "agentlongbench":
            if args.tokenizer is None:
                raise ValueError("AgentLongBench preparation requires --tokenizer")
            from .selection import main as select
            command = ["--source-root", str(source), "--data-root", str(data),
                       "--tokenizer", str(args.tokenizer), "--min-tokens", str(args.min_input_tokens),
                       "--max-tokens", str(args.max_input_tokens), "--max-samples", str(args.max_samples),
                       "--output", str(output / "selected.jsonl"), "--summary", str(output / "selection_audit.json")]
            for length in args.length_dirs or ["64k"]:
                command += ["--length-dir", length]
            if args.reuse_selection:
                command += ["--reuse-selection", str(args.reuse_selection)]
            select(command)
            report.update(status="prepared", selection_manifest=str(output / "selected.jsonl"),
                          audit=json.loads((output / "selection_audit.json").read_text()))
        elif args.benchmark == "bfcl":
            dataset = args.dataset or data / "BFCL_v4_multi_turn_long_context.json"
            rows = jsonl(dataset)
            if not rows or any(not str(row.get("id", "")).startswith("multi_turn_long_context_") for row in rows):
                raise ValueError("expected native BFCL multi_turn_long_context records")
            if len({row["id"] for row in rows}) != len(rows):
                raise ValueError("duplicate BFCL source case IDs")
            candidates = [{"id": row["id"], "source_index": index,
                           "user_turns": len(row.get("question", []))} for index, row in enumerate(rows)]
            save_json(output / "candidates.json", candidates)
            report.update(status="prepared_dynamic", category="multi_turn_long_context", dataset=str(dataset.resolve()),
                          candidate_sessions=len(candidates), source_sha256=fingerprint([dataset]),
                          native_source_available=(source / "bfcl_eval/__main__.py").is_file(),
                          tool_definitions_available=(data / "multi_turn_func_doc").is_dir(),
                          selected_samples=None, reason="final lengths depend on native tool feedback; "
                          "select first qualifying requests at runtime, never infer from raw JSON size")
        else:
            report.update(audit_longmemeval(data, output, args))
        report["source_sha256"] = {**report.get("source_sha256", {}),
                                   **(fingerprint(data.rglob("*.jsonl")) if data.exists() else {})}
    except Exception as exc:
        report.update(status="preparation_failed", error=str(exc))
        save_json(output / "selection_report.json", report)
        raise
    save_json(output / "selection_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in ("prepared", "prepared_dynamic") else 2
