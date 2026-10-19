import json
import sys
from types import SimpleNamespace

import torch

from dlite import benchmark
from dlite.benchmark import load_dataset_records


def test_longbench_pretty_json_array_with_jsonl_suffix(tmp_path):
    dataset = tmp_path / "longbench.jsonl"
    dataset.write_text(
        json.dumps(
            [
                {"context": "first context", "question": "first question"},
                {"context": "second context", "question": "second question"},
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    paths = tmp_path / "paths.json"
    alias = "longbench_v2_64000_32000_single_document_qa"
    paths.write_text(json.dumps({alias: str(dataset)}), encoding="utf-8")

    name, records = load_dataset_records(
        dataset=alias,
        prompts=None,
        dataset_path_file=paths,
        max_samples=None,
    )

    assert name == alias
    assert records == [
        {"turns": ["first context\n\nQuestion: first question"]},
        {"turns": ["second context\n\nQuestion: second question"]},
    ]


def test_prompt_jsonl_keeps_multi_turn_records(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"prompt":"single"}\n{"turns":["first","second"]}\n',
        encoding="utf-8",
    )

    name, records = load_dataset_records(
        dataset=None,
        prompts=prompts,
        dataset_path_file=None,
        max_samples=None,
    )

    assert name == "prompts"
    assert records == [
        {"turns": ["single"]},
        {"turns": ["first", "second"]},
    ]


def test_benchmark_reports_only_public_metrics(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "dlite-benchmark", "--target-model", "target", "--draft-model", "draft",
        "--dataset", "test", "--output", str(report_path),
    ])
    monkeypatch.setattr(benchmark, "load_dataset_records", lambda **kwargs: (
        "test", [{"turns": ["question"]}],
    ))
    monkeypatch.setattr(benchmark, "_render_messages", lambda *args: torch.tensor([[1]]))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(benchmark, "target_generate", lambda *args: SimpleNamespace(
        num_output_tokens=2, decode_seconds=4.0,
    ))
    engine = SimpleNamespace(
        target=SimpleNamespace(device=torch.device("cpu")),
        tokenizer=SimpleNamespace(eos_token_id=0, decode=lambda *args, **kwargs: "answer"),
        draft=SimpleNamespace(
            spec_generate=lambda **kwargs: torch.tensor([[1, 4, 5, 6]]),
            get_last_decode_stats=lambda: {"decode_wall_time": 3.0, "accept_lengths": [2]},
        ),
    )
    monkeypatch.setattr(benchmark.DLiteEngine, "from_pretrained", lambda *args, **kwargs: engine)

    benchmark.main()

    report = json.loads(report_path.read_text())
    assert report["baseline_tokens_per_second"] == 0.5
    assert report["dlite_tokens_per_second"] == 1.0
    assert report["token_weighted_speedup"] == 2.0
    removed_fields = {
        "dtype",
        "generated_tokens",
        "baseline_generated_tokens",
        "baseline_decode_seconds",
        "dlite_decode_seconds",
        "baseline_seconds_per_token",
        "dlite_seconds_per_token",
        "all_greedy_tokens_match",
    }
    assert removed_fields.isdisjoint(report)
    assert set(report["turn_results"][0]) == {
        "sample",
        "turn",
        "input_tokens",
        "output_tokens",
        "baseline_output_tokens",
        "accept_lengths",
    }
