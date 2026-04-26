from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm_mst_finder.reporting import generate_report
from llm_mst_finder.records import (
    BenchmarkMetrics,
    BottleneckResult,
    StabilityResult,
    TrialAnalysisResult,
    TrialSummary,
    WindowSummary,
)


def _benchmark_metrics() -> BenchmarkMetrics:
    return BenchmarkMetrics(
        successful_requests=20,
        failed_requests=0,
        total_input_tokens=2000,
        total_output_tokens=800,
        request_throughput=10.0,
        successful_request_throughput=10.0,
        prompt_token_throughput=2000.0,
        generation_token_throughput=800.0,
        total_token_throughput=2800.0,
        mean_ttft_ms=100.0,
        median_ttft_ms=90.0,
        std_ttft_ms=10.0,
        percentiles_ttft_ms=[(0.9, 120.0), (0.99, 150.0)],
        mean_tpot_ms=30.0,
        median_tpot_ms=28.0,
        std_tpot_ms=4.0,
        percentiles_tpot_ms=[(0.9, 32.0), (0.99, 36.0)],
        mean_itl_ms=25.0,
        median_itl_ms=25.0,
        std_itl_ms=2.0,
        percentiles_itl_ms=[(0.9, 27.0)],
        mean_e2e_ms=650.0,
        median_e2e_ms=640.0,
        std_e2e_ms=30.0,
        percentiles_e2e_ms=[(0.9, 690.0), (0.99, 720.0)],
        prompt_length_summary={"mean": 100.0, "median": 100.0, "p90": 120.0, "p99": 128.0},
        output_length_summary={"mean": 40.0, "median": 40.0, "p90": 48.0, "p99": 64.0},
    )


def _trial_summary(
    trial_id: str,
    *,
    mode: str,
    request_rate: float | None,
    concurrency: int | None,
    completion_rate: float,
) -> TrialSummary:
    return TrialSummary(
        trial_id=trial_id,
        mode=mode,
        status="completed",
        requested_request_rate=request_rate,
        requested_concurrency=concurrency,
        target_duration_s=6.0,
        wall_time_s=6.0,
        started_requests=20,
        successful_requests=20,
        failed_requests=0,
        actual_send_rate=completion_rate,
        successful_completion_rate=completion_rate,
        error_rate=0.0,
        mean_scheduling_delay_s=0.01,
        max_scheduling_delay_s=0.02,
        max_observed_outstanding=4,
        metrics_sample_count=3,
        abort_reason=None,
        benchmark_metrics=_benchmark_metrics(),
        metadata={},
    )


def _window(trial_id: str, idx: int, *, arrival_rate: float, completion_rate: float, outstanding_end: int, ttft_p90_ms: float, tpot_p90_ms: float) -> WindowSummary:
    return WindowSummary(
        trial_id=trial_id,
        window_idx=idx,
        start_s=float(idx),
        end_s=float(idx + 1),
        arrivals=10,
        completions=10,
        failures=0,
        arrival_rate=arrival_rate,
        completion_rate=completion_rate,
        error_rate=0.0,
        outstanding_start=max(0, outstanding_end - 1),
        outstanding_end=outstanding_end,
        outstanding_mean=float(outstanding_end),
        outstanding_slope=0.1,
        ttft_p50_ms=80.0,
        ttft_p90_ms=ttft_p90_ms,
        ttft_p99_ms=ttft_p90_ms + 20.0,
        tpot_p50_ms=25.0,
        tpot_p90_ms=tpot_p90_ms,
        tpot_p99_ms=tpot_p90_ms + 10.0,
        itl_p90_ms=23.0,
        e2e_p90_ms=600.0,
        e2e_p99_ms=650.0,
        prompt_tok_s=1000.0,
        generation_tok_s=400.0,
        total_tok_s=1400.0,
        num_running_mean=3.0,
        num_waiting_mean=0.5,
        num_swapped_mean=0.0,
        kv_cache_usage_mean=0.45,
        kv_cache_usage_max=0.50,
        preemptions_delta=0.0,
    )


def _analysis(trial_id: str, *, status: str, bottleneck_class: str = "scheduler_cap") -> TrialAnalysisResult:
    return TrialAnalysisResult(
        trial_id=trial_id,
        trial_validity="valid",
        validity_reasons=["fixture valid"],
        stability=StabilityResult(
            status=status,
            confidence="high",
            reasons=[f"fixture {status}"],
            key_metrics={"outstanding_end_slope_per_s": 0.1},
        ),
        bottleneck=BottleneckResult(
            bottleneck_class=bottleneck_class,
            confidence="medium",
            evidence=["fixture evidence"],
        ),
    )


def _write_trial(
    result_dir: Path,
    *,
    trial_id: str,
    purpose: str,
    mode: str,
    request_rate: float | None,
    concurrency: int | None,
    analysis: TrialAnalysisResult,
) -> dict[str, object]:
    trial_dir = result_dir / "trials" / trial_id
    trial_dir.mkdir(parents=True)
    summary = _trial_summary(
        trial_id,
        mode=mode,
        request_rate=request_rate,
        concurrency=concurrency,
        completion_rate=8.0 if request_rate is None else request_rate,
    )
    summary_payload = {
        "config": {
            "trial_id": trial_id,
            "mode": mode,
            "duration_s": 6.0,
            "base_url": "http://127.0.0.1:8000",
            "endpoint": "/v1/completions",
            "model": "fake-model",
            "request_rate": request_rate,
            "concurrency": concurrency,
            "metadata": {
                "workload": {
                    "name": "fixture-workload",
                    "source_path": "fixtures/workload.yaml",
                    "dataset_type": "synthetic-fixed",
                    "num_requests": 20,
                    "context_policy": {
                        "max_model_len": 4096,
                        "tokenizer_source": "vllm_model_config",
                        "tokenizer": "fake-model",
                        "over_limit": "fail",
                        "truncation_side": "left",
                        "unsafe_allow_workload_tokenizer_for_real_datasets": False,
                        "total_samples": 20,
                        "kept_samples": 20,
                        "skipped_samples": 0,
                        "truncated_samples": 0,
                        "skipped_source_indexes": [],
                        "truncated_source_indexes": [],
                    },
                },
                "server_config": {
                    "max_num_seqs": 32,
                    "max_num_batched_tokens": 4096,
                    "chunked_prefill": True,
                },
            },
        },
        "summary": summary.to_dict(),
    }
    (trial_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (trial_dir / "analysis.json").write_text(json.dumps(analysis.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    windows = [
        _window(trial_id, idx, arrival_rate=8.0 if request_rate is None else request_rate, completion_rate=8.0 if request_rate is None else request_rate, outstanding_end=idx, ttft_p90_ms=100.0 + idx, tpot_p90_ms=30.0 + idx)
        for idx in range(6)
    ]
    with (trial_dir / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0].to_dict()))
        writer.writeheader()
        for window in windows:
            writer.writerow(window.to_dict())
    return {
        "purpose": purpose,
        "trial_id": trial_id,
        "trial_dir": str(trial_dir),
        "mode": mode,
        "request_rate": request_rate,
        "concurrency": concurrency,
        "summary": {
            "status": summary.status,
            "requested_request_rate": request_rate,
            "requested_concurrency": concurrency,
            "actual_send_rate": summary.actual_send_rate,
            "successful_completion_rate": summary.successful_completion_rate,
            "error_rate": summary.error_rate,
            "generation_token_throughput": summary.benchmark_metrics.generation_token_throughput,
            "total_token_throughput": summary.benchmark_metrics.total_token_throughput,
            "max_observed_outstanding": summary.max_observed_outstanding,
            "abort_reason": summary.abort_reason,
        },
        "analysis": analysis.to_dict(),
    }


def _write_result_dir(result_dir: Path) -> None:
    trials_dir = result_dir / "trials"
    trials_dir.mkdir(parents=True)
    closed_loop_event = _write_trial(
        result_dir,
        trial_id="trial_000_closedloop_N4",
        purpose="closed_loop_scout",
        mode="closed-loop",
        request_rate=None,
        concurrency=4,
        analysis=_analysis("trial_000_closedloop_N4", status="stable"),
    )
    stable_event = _write_trial(
        result_dir,
        trial_id="trial_001_openloop_r8_0",
        purpose="open_loop_bracket",
        mode="open-loop",
        request_rate=8.0,
        concurrency=None,
        analysis=_analysis("trial_001_openloop_r8_0", status="stable"),
    )
    unstable_event = _write_trial(
        result_dir,
        trial_id="trial_002_openloop_r12_0",
        purpose="open_loop_bracket",
        mode="open-loop",
        request_rate=12.0,
        concurrency=None,
        analysis=_analysis("trial_002_openloop_r12_0", status="unstable"),
    )
    trace_payload = {
        "config": {
            "search_id": "fixture-search",
            "search_mode": "hybrid",
            "base_url": "http://127.0.0.1:8000",
            "endpoint": "/v1/completions",
            "model": "fake-model",
            "trial_duration_s": 6.0,
            "final_confirmation_duration_s": 6.0,
            "rate_precision": 0.05,
        },
        "events": [closed_loop_event, stable_event, unstable_event],
        "bounds": {
            "low_rate": 8.0,
            "low_trial_id": "trial_001_openloop_r8_0",
            "high_rate": 12.0,
            "high_trial_id": "trial_002_openloop_r12_0",
        },
        "closed_loop": {
            "peak_request_throughput": 8.0,
            "peak_output_token_throughput": 800.0,
            "plateau_concurrency": 4,
            "stop_reason": "fixture plateau",
        },
        "result": {
            "search_id": "fixture-search",
            "search_mode": "hybrid",
            "max_no_drift_request_rate": 8.0,
            "max_slo_satisfying_request_rate": 8.0,
            "rate_precision": 0.05,
            "confirmation_trial_id": "trial_001_openloop_r8_0",
            "closed_loop": {
                "peak_request_throughput": 8.0,
                "peak_output_token_throughput": 800.0,
                "plateau_concurrency": 4,
                "stop_reason": "fixture plateau",
            },
            "bottleneck_class": "scheduler_cap",
            "confidence": "medium",
            "reasons": ["fixture result"],
        },
    }
    (result_dir / "search_trace.json").write_text(json.dumps(trace_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_generate_report_writes_reports_and_plots(tmp_path: Path) -> None:
    result_dir = tmp_path / "run_001"
    _write_result_dir(result_dir)

    report_result = generate_report(result_dir)

    assert Path(report_result["final_report_json"]).is_file()
    assert Path(report_result["final_report_md"]).is_file()
    payload = json.loads((result_dir / "final_report.json").read_text(encoding="utf-8"))
    assert payload["search_result"]["max_no_drift_request_rate"] == 8.0
    assert payload["bottleneck"]["class"] == "scheduler_cap"
    assert payload["trials"][0]["trial_id"] == "trial_000_closedloop_N4"
    assert (result_dir / "plots" / "search_rate_vs_classification.png").is_file()
    assert (result_dir / "trials" / "trial_001_openloop_r8_0" / "plots" / "ttft_percentiles.png").is_file()


def test_generate_report_requires_existing_comparison_report(tmp_path: Path) -> None:
    result_dir = tmp_path / "run_001"
    _write_result_dir(result_dir)
    comparison_dir = tmp_path / "run_002"
    comparison_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="final_report.json"):
        generate_report(result_dir, compare_result_dirs=[comparison_dir])


def test_generate_report_compares_matching_reports(tmp_path: Path) -> None:
    result_dir = tmp_path / "run_001"
    _write_result_dir(result_dir)
    generate_report(result_dir)

    comparison_dir = tmp_path / "run_002"
    _write_result_dir(comparison_dir)
    generate_report(comparison_dir)

    generate_report(result_dir, compare_result_dirs=[comparison_dir])

    payload = json.loads((result_dir / "final_report.json").read_text(encoding="utf-8"))
    assert payload["comparison"]["comparable"] is True
    assert len(payload["comparison"]["results"]) == 2
    assert (result_dir / "plots" / "comparison_max_sustainable_rate.png").is_file()
