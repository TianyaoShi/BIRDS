from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm_mst_finder.analysis import analyze_trial_dir, write_analysis_artifact
from llm_mst_finder.records import (
    BenchmarkMetrics,
    RequestRecord,
    ServerMetricSample,
    TrialAnalysisResult,
    TrialSummary,
    WindowSummary,
)


def _benchmark_metrics() -> BenchmarkMetrics:
    return BenchmarkMetrics(
        successful_requests=10,
        failed_requests=0,
        total_input_tokens=1000,
        total_output_tokens=300,
        request_throughput=10.0,
        successful_request_throughput=10.0,
        prompt_token_throughput=1000.0,
        generation_token_throughput=300.0,
        total_token_throughput=1300.0,
        mean_ttft_ms=100.0,
        median_ttft_ms=100.0,
        std_ttft_ms=0.0,
        percentiles_ttft_ms=[(0.9, 100.0), (0.99, 120.0)],
        mean_tpot_ms=20.0,
        median_tpot_ms=20.0,
        std_tpot_ms=0.0,
        percentiles_tpot_ms=[(0.9, 20.0), (0.99, 24.0)],
        mean_itl_ms=18.0,
        median_itl_ms=18.0,
        std_itl_ms=0.0,
        percentiles_itl_ms=[(0.9, 18.0)],
        mean_e2e_ms=600.0,
        median_e2e_ms=600.0,
        std_e2e_ms=0.0,
        percentiles_e2e_ms=[(0.9, 600.0), (0.99, 700.0)],
        prompt_length_summary={"mean": 100.0, "median": 100.0, "p90": 100.0, "p99": 100.0},
        output_length_summary={"mean": 30.0, "median": 30.0, "p90": 30.0, "p99": 30.0},
    )


def _summary(
    *,
    status: str = "completed",
    requested_request_rate: float | None = 10.0,
    actual_send_rate: float = 10.0,
    max_scheduling_delay_s: float | None = 0.01,
    abort_reason: str | None = None,
) -> TrialSummary:
    return TrialSummary(
        trial_id="trial-analysis",
        mode="open-loop",
        status=status,
        requested_request_rate=requested_request_rate,
        requested_concurrency=None,
        target_duration_s=6.0,
        wall_time_s=6.0,
        started_requests=10,
        successful_requests=10,
        failed_requests=0,
        actual_send_rate=actual_send_rate,
        successful_completion_rate=10.0,
        error_rate=0.0,
        mean_scheduling_delay_s=0.01,
        max_scheduling_delay_s=max_scheduling_delay_s,
        max_observed_outstanding=2,
        metrics_sample_count=6,
        abort_reason=abort_reason,
        benchmark_metrics=_benchmark_metrics(),
        metadata={},
    )


def _window(idx: int) -> WindowSummary:
    return WindowSummary(
        trial_id="trial-analysis",
        window_idx=idx,
        start_s=float(idx),
        end_s=float(idx + 1),
        arrivals=10,
        completions=10,
        failures=0,
        arrival_rate=10.0,
        completion_rate=10.0,
        error_rate=0.0,
        outstanding_start=0,
        outstanding_end=0,
        outstanding_mean=0.0,
        outstanding_slope=0.0,
        ttft_p50_ms=80.0,
        ttft_p90_ms=100.0,
        ttft_p95_ms=110.0,
        ttft_p99_ms=120.0,
        tpot_p50_ms=15.0,
        tpot_p90_ms=20.0,
        tpot_p95_ms=22.0,
        tpot_p99_ms=24.0,
        itl_p50_ms=14.0,
        itl_p90_ms=18.0,
        itl_p95_ms=19.0,
        itl_p99_ms=20.0,
        e2e_p50_ms=500.0,
        e2e_p90_ms=600.0,
        e2e_p95_ms=650.0,
        e2e_p99_ms=700.0,
        prompt_tok_s=1000.0,
        generation_tok_s=300.0,
        total_tok_s=1300.0,
        num_running_mean=4.0,
        num_waiting_mean=0.0,
        num_swapped_mean=0.0,
        kv_cache_usage_mean=0.40,
        kv_cache_usage_max=0.45,
        preemptions_delta=0.0,
    )


def _request_record(
    request_id: str,
    *,
    success: bool = True,
    error: str | None = None,
) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        trial_id="trial-analysis",
        scheduled_send_ts=0.0,
        actual_send_ts=0.0,
        first_token_ts=0.01 if success else None,
        end_ts=0.02,
        success=success,
        error=error,
        prompt_len=100,
        expected_output_len=30,
        actual_output_len=2 if success else None,
        ttft_s=0.01 if success else None,
        e2e_s=0.02,
        tpot_s=0.01 if success else None,
        itl_s=[0.01] if success else [],
        output_token_timestamps=[0.01, 0.02] if success else [],
        metadata={},
    )


def _request_record_at(request_id: str, *, ts: float) -> RequestRecord:
    actual_send_ts = ts + 0.01
    first_token_ts = actual_send_ts + 0.01
    end_ts = actual_send_ts + 0.02
    return RequestRecord(
        request_id=request_id,
        trial_id="trial-analysis",
        scheduled_send_ts=ts,
        actual_send_ts=actual_send_ts,
        first_token_ts=first_token_ts,
        end_ts=end_ts,
        success=True,
        error=None,
        prompt_len=100,
        expected_output_len=30,
        actual_output_len=2,
        ttft_s=first_token_ts - actual_send_ts,
        e2e_s=end_ts - actual_send_ts,
        tpot_s=end_ts - first_token_ts,
        itl_s=[end_ts - first_token_ts],
        output_token_timestamps=[first_token_ts, end_ts],
        metadata={},
    )


def _write_trial_dir(
    trial_dir: Path,
    *,
    summary: TrialSummary | None = None,
    request_records: list[RequestRecord] | None = None,
    server_metadata: dict[str, object] | None = None,
) -> None:
    trial_dir.mkdir()
    actual_summary = _summary() if summary is None else summary
    summary_payload = {
        "config": {
            "trial_id": actual_summary.trial_id,
            "mode": actual_summary.mode,
            "duration_s": actual_summary.target_duration_s,
            "base_url": "http://127.0.0.1:8000",
            "endpoint": "/v1/completions",
            "model": "fake-model",
            "request_rate": actual_summary.requested_request_rate,
            "concurrency": actual_summary.requested_concurrency,
            "metadata": {},
        },
        "summary": actual_summary.to_dict(),
    }
    (trial_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    windows = [_window(idx) for idx in range(6)]
    with (trial_dir / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0].to_dict()))
        writer.writeheader()
        for window in windows:
            writer.writerow(window.to_dict())

    actual_records = [_request_record("request-1")] if request_records is None else request_records
    with (trial_dir / "request_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in actual_records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    if server_metadata is not None:
        (trial_dir / "server_metadata.json").write_text(
            json.dumps(server_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _write_server_metrics(trial_dir: Path, samples: list[ServerMetricSample]) -> None:
    with (trial_dir / "server_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), sort_keys=True) + "\n")


def _server_metric_sample(idx: int, *, poll_error: str | None = None) -> ServerMetricSample:
    if poll_error is not None:
        return ServerMetricSample(
            ts=float(idx),
            raw={"poll_error": poll_error},
            num_running=None,
            num_waiting=None,
            num_swapped=None,
            kv_cache_usage=None,
            prompt_tokens_total=None,
            generation_tokens_total=None,
            request_success_total=None,
            request_abort_total=None,
        )
    return ServerMetricSample(
        ts=float(idx) + 0.05,
        raw={
            "vllm:num_preemptions_total": [
                {"labels": {}, "value": 0.0, "timestamp_ms": None},
            ],
        },
        num_running=1.0,
        num_waiting=0.0,
        num_swapped=0.0,
        kv_cache_usage=0.55,
        prompt_tokens_total=float(idx * 100),
        generation_tokens_total=float(idx * 30),
        request_success_total=float(idx),
        request_abort_total=0.0,
    )


def test_analyze_trial_dir_classifies_valid_trial(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-valid"
    _write_trial_dir(trial_dir, server_metadata={"server_config": {"max_num_seqs": 8}})

    result = analyze_trial_dir(trial_dir)

    assert result.trial_validity == "valid"
    assert result.stability is not None
    assert result.stability.status == "stable"
    assert result.bottleneck is not None

    output_path = write_analysis_artifact(trial_dir, result)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["trial_validity"] == "valid"
    assert written["stability"]["status"] == "stable"


def test_analyze_trial_dir_rebuilds_windows_from_server_metrics(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-server-metrics"
    request_records = [
        _request_record_at(f"request-{idx}", ts=float(idx) + 0.10) for idx in range(6)
    ]
    _write_trial_dir(trial_dir, request_records=request_records)
    payload = json.loads((trial_dir / "summary.json").read_text(encoding="utf-8"))
    payload["config"]["metrics_url"] = "http://127.0.0.1:8000/metrics"
    payload["config"]["window_s"] = 1.0
    (trial_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    _write_server_metrics(trial_dir, [_server_metric_sample(idx) for idx in range(6)])

    result = analyze_trial_dir(trial_dir)

    assert result.trial_validity == "valid"
    assert result.stability is not None
    assert result.stability.status == "stable"
    assert result.stability.key_metrics["kv_cache_usage_max"] == pytest.approx(0.55)
    assert not any(
        "server-side evidence missing" in reason for reason in result.stability.reasons
    )


def test_analyze_trial_dir_marks_poll_error_metrics_invalid(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-poll-error"
    _write_trial_dir(trial_dir)
    payload = json.loads((trial_dir / "summary.json").read_text(encoding="utf-8"))
    payload["config"]["metrics_url"] = "http://127.0.0.1:8000/metrics"
    (trial_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    _write_server_metrics(
        trial_dir,
        [_server_metric_sample(idx, poll_error="ClientConnectorError: blocked") for idx in range(6)],
    )

    result = analyze_trial_dir(trial_dir)

    assert result.trial_validity == "metrics_invalid"
    assert result.stability is None
    assert result.bottleneck is None
    assert any("Prometheus poll failures" in reason for reason in result.validity_reasons)


def test_analyze_trial_dir_marks_invalid_workload_from_saved_request_failures(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-invalid-workload"
    _write_trial_dir(
        trial_dir,
        request_records=[
            _request_record(
                "request-1",
                success=False,
                error="HTTP 400: prompt is too long for maximum context length",
            )
        ],
    )

    result = analyze_trial_dir(trial_dir)

    assert result == TrialAnalysisResult(
        trial_id="trial-analysis",
        trial_validity="invalid_workload",
        validity_reasons=result.validity_reasons,
        stability=None,
        bottleneck=None,
    )
    assert any("invalidate the workload" in reason for reason in result.validity_reasons)


def test_analyze_trial_dir_marks_aborted_safety_trial_as_client_limited(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-client-limited"
    _write_trial_dir(
        trial_dir,
        summary=_summary(
            status="aborted_safety",
            actual_send_rate=6.0,
            max_scheduling_delay_s=0.4,
            abort_reason="outstanding requests reached safety_max_outstanding=4",
        ),
    )

    result = analyze_trial_dir(trial_dir)

    assert result.trial_validity == "client_limited"
    assert result.stability is None
    assert result.bottleneck is None
    assert any("client safety cap" in reason for reason in result.validity_reasons)


def test_analyze_trial_dir_fails_fast_on_trial_id_mismatch(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-mismatch"
    _write_trial_dir(trial_dir)
    payload = json.loads((trial_dir / "summary.json").read_text(encoding="utf-8"))
    payload["summary"]["trial_id"] = "other-trial"
    (trial_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="trial_id mismatch"):
        analyze_trial_dir(trial_dir)
