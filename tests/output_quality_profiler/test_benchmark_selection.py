from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import yaml

from output_quality_profiler.benchmark_selection import (
    build_benchmark_generation_manifest,
    resolve_workload_group,
    select_missing_benchmark_scores,
)
from output_quality_profiler.manifest import load_quality_manifest


def _write_scorebook(path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(
        [
            "Model",
            "Text Conversation Benchmark",
            "Text Conversation Score",
            "Problem Solving Benchmark",
            "Problem Solving Score",
            "Code Completion Benchmark",
            "Code Completion Score",
            "Long Context Benchmark",
            "Long Context Score",
        ]
    )
    ws.append(
        [
            "org/model-a",
            None,
            None,
            "MMLU-Pro",
            "50",
            "EvalPlus",
            "20",
            "LongBench v2",
            "30",
        ]
    )
    ws.append(
        [
            "org/model-b",
            None,
            None,
            "MMLU-Pro/SuperGPQA",
            "60/40",
            "RepoBench",
            "10",
            "LongBench v1",
            "55",
        ]
    )
    ws.append(
        [
            "openai/gpt-oss-20b",
            None,
            None,
            "SuperGPQA",
            "40",
            "N/A",
            "N/A",
            None,
            None,
        ]
    )
    ws.append(
        [
            "meta-llama/Llama-2-7b-chat-hf",
            None,
            None,
            "MMLU-Pro",
            "20",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
        ]
    )
    ws.append(["Workload Category", None, None, None, None, None, None, None, None])
    wb.save(path)


def _write_workload_group(repo: Path, relative: str, count: int = 1) -> None:
    directory = repo / relative
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"shard_{index:03d}.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": f"shard-{index}",
                    "dataset": {"type": "jsonl", "path": "../shards/shard_000.runner.jsonl"},
                    "sampling": {
                        "seed": 1,
                        "num_requests": 1,
                        "entry_selection": "sequential",
                        "prompt_len": {"mode": "from_dataset"},
                        "output_len": {"mode": "natural_until_eos", "max_tokens": 32},
                    },
                    "request": {"stream": True, "temperature": 0.0, "ignore_eos": False},
                }
            ),
            encoding="utf-8",
        )


def _write_materialization_report(directory: Path, *, materialized: int = 1, repeat_policy=None) -> None:
    (directory / "materialization_report.json").write_text(
        json.dumps(
            {
                "rows": {"materialized": materialized, "total": materialized},
                "sampling": {
                    "expanded_sample_count": materialized,
                    "repeat_policy": repeat_policy,
                    "unique_sample_ids": materialized,
                },
                "selected_tasks": ["task"],
                "prompt_template": "longbench_official",
            }
        ),
        encoding="utf-8",
    )


def _write_registry_dirs(repo: Path) -> None:
    _write_workload_group(repo, "experiments/reasoning_workloads/supergpqa_reasoning/workload_yamls")
    _write_workload_group(repo, "experiments/reasoning_workloads/supergpqa_hard_reasoning/workload_yamls")
    _write_workload_group(
        repo,
        "experiments/code_workloads/repobench_python_java_aggregate_cache_realistic/workload_yamls",
    )
    _write_workload_group(
        repo,
        "experiments/code_workloads/crosscodeeval_rg1_unixcoder_cache_realistic/workload_yamls",
    )
    for profile in (
        "longbench_long_output_summarization_original_official_qwen3_8b",
        "longbench_medium_output_summarization_original_official_qwen3_8b",
        "longbench_medium_answer_rag_qa_original_official_qwen3_8b",
        "longbench_short_answer_document_qa_original_official_qwen3_8b",
    ):
        base = repo / f"experiments/longbench_workloads/benchmark_original/{profile}"
        _write_workload_group(
            repo,
            f"experiments/longbench_workloads/benchmark_original/{profile}/workload_yamls",
        )
        _write_materialization_report(base)


def _write_base_manifest(path: Path, workload: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "run_id": "base-quality",
                    "output_root": str(path.parent / "results"),
                    "default_endpoint": "/v1/chat/completions",
                },
                "launch": {"gpu_count": 1, "tensor_parallel_size": 1, "max_model_len": 32768},
                "generation": {
                    "concurrency_source": "explicit",
                    "max_concurrency": 1,
                    "include_prompt_text": True,
                    "preserve_request_order": True,
                    "decoding": {
                        "temperature": 0.6,
                        "top_p": 0.95,
                        "top_k": 20,
                        "min_p": 0.0,
                        "n": 1,
                        "max_tokens": 32768,
                        "max_tokens_policy": "model_context_minus_prompt_buffer",
                        "prompt_token_buffer": 128,
                        "extra_body": {},
                    },
                },
                "experiments": [
                    {
                        "id": "model-a-base",
                        "model": "org/model-a",
                        "workload": str(workload),
                        "generation": {"concurrency_source": "explicit", "max_concurrency": 3},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_select_missing_benchmark_scores_uses_alias_policy_and_registry(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_registry_dirs(repo)
    scorebook = tmp_path / "scores.xlsx"
    _write_scorebook(scorebook)

    result = select_missing_benchmark_scores(scorebook=scorebook, output_dir=tmp_path / "out", repo_root=repo)

    missing = {(record.model, record.benchmark) for record in result.records}
    assert ("org/model-a", "SuperGPQA") in missing
    assert ("org/model-a", "SuperGPQA-hard") in missing
    assert ("org/model-a", "RepoBench") in missing
    assert ("org/model-a", "CrossCodeEval") in missing
    assert ("org/model-a", "LongBench-v1-covered") in missing
    assert ("org/model-b", "SuperGPQA") not in missing
    assert ("org/model-b", "SuperGPQA-hard") in missing
    assert ("org/model-b", "RepoBench") in missing
    assert ("org/model-b", "CrossCodeEval") in missing
    assert ("org/model-b", "LongBench-v1-covered") in missing
    assert ("openai/gpt-oss-20b", "RepoBench") not in missing
    assert ("openai/gpt-oss-20b", "CrossCodeEval") not in missing
    assert ("meta-llama/Llama-2-7b-chat-hf", "LongBench-v1-covered") not in missing
    assert (tmp_path / "out" / "missing_scores.json").is_file()
    assert any(row["reason"] == "not a model id" for row in result.skipped_rows)
    assert any("gpt-oss" in row["reason"] for row in result.skipped_rows)
    assert any("llama-2" in row["reason"] for row in result.skipped_rows)


def test_resolve_longbench_group_is_marked_subset(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_registry_dirs(repo)

    group = resolve_workload_group("LongBench-v1-covered", repo_root=repo)

    assert group.is_full_benchmark is False
    assert len(group.workload_paths) == 4


def test_resolve_longbench_group_rejects_repeat_materialization(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_registry_dirs(repo)
    report_dir = (
        repo
        / "experiments/longbench_workloads/benchmark_original/"
        / "longbench_long_output_summarization_original_official_qwen3_8b"
    )
    _write_materialization_report(report_dir, materialized=2, repeat_policy="epoch_shuffle")

    import pytest

    with pytest.raises(ValueError, match="no-repeat original materialization"):
        resolve_workload_group("LongBench-v1-covered", repo_root=repo)


def test_build_benchmark_generation_manifest_inherits_model_overrides(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_registry_dirs(repo)
    scorebook = tmp_path / "scores.xlsx"
    _write_scorebook(scorebook)
    selection = select_missing_benchmark_scores(
        scorebook=scorebook,
        output_dir=tmp_path / "selection",
        repo_root=repo,
        targets=("RepoBench",),
    )
    base_workload = selection.records[0].workload_paths[0]
    base_manifest = tmp_path / "base.yaml"
    output_manifest = tmp_path / "benchmark.yaml"
    _write_base_manifest(base_manifest, base_workload)

    result = build_benchmark_generation_manifest(
        missing_plan=tmp_path / "selection" / "missing_scores.json",
        base_manifest=base_manifest,
        output_path=output_manifest,
        run_id="benchmark-run",
    )
    manifest = load_quality_manifest(output_manifest)
    raw = yaml.safe_load(output_manifest.read_text(encoding="utf-8"))

    assert result["experiment_count"] == 3
    assert manifest.run.run_id == "benchmark-run"
    assert manifest.experiments[0].generation.decoding.temperature == 0.0
    assert manifest.experiments[0].generation.decoding.max_tokens == 512
    assert manifest.experiments[0].generation.max_concurrency == 3
    assert raw["experiments"][0]["generation"]["decoding"]["extra_body"] == {"stop": ["\n\n"]}
    assert json.loads((tmp_path / "selection" / "missing_scores.json").read_text(encoding="utf-8"))
