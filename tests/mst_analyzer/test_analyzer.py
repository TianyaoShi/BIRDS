from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from local_orchestrator.manifest import load_manifest
from local_orchestrator.matrix import expand_manifest
from mst_analyzer.config import AnalyzerSettings, load_settings
from mst_analyzer.extract import extract_run
from mst_analyzer.models import MSTRow, TraceInstabilityEvidence, TrialArtifactRef
from mst_analyzer.reporting import analyze_orchestrator_run
from mst_analyzer.rules import analyze_rows, analyze_rows_with_diagnostics


def _write_workload(tmp_path: Path, *, name: str = "live_sharegpt_workload_context") -> Path:
    workload = tmp_path / "workloads" / "live_sharegpt_workload_context.yaml"
    workload.parent.mkdir(parents=True, exist_ok=True)
    workload.write_text(
        "\n".join(
            [
                f"name: {name}",
                "dataset:",
                "  type: synthetic-fixed",
                "tokenizer: whitespace",
                "sampling:",
                "  seed: 1",
                "  num_requests: 100",
                "  prompt_len:",
                "    mode: fixed",
                "    value: 128",
                "  output_len:",
                "    mode: fixed",
                "    value: 64",
                "context_policy:",
                "  max_model_len: 32768",
                "  over_limit: skip_sample",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return workload


def _write_manifest(tmp_path: Path, workload: Path, *, models: list[str]) -> Path:
    manifest_path = tmp_path / "experiments" / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "run_id": "fixture-run",
                    "output_root": "results/orchestrator",
                    "python_executable": "python",
                },
                "probe": {
                    "enabled": False,
                },
                "launch": {
                    "gpu_count": 1,
                    "tensor_parallel_size": 1,
                    "dtype": "float16",
                    "max_model_len": 32768,
                },
                "search": {
                    "search_mode": "open-loop",
                    "trial_min_duration_s": 120,
                    "trial_max_duration_s": 240,
                    "final_confirmation_duration_s": 240,
                    "rate_precision": 0.1,
                    "ttft_slo_ms": 250,
                    "tpot_slo_ms": 50,
                    "max_num_seqs": 1024,
                    "max_num_batched_tokens": 8192,
                },
                "experiments": [
                    {
                        "id": "fixture-loop",
                        "models": models,
                        "workloads": [str(workload)],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _summary_payload(
    trial_id: str,
    *,
    workload_path: Path,
    model: str,
    request_rate: float,
    completion_rate: float,
    total_token_throughput: float,
    generation_token_throughput: float,
    prompt_len_mean: float,
    output_len_mean: float,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> dict[str, object]:
    workload_name = yaml.safe_load(workload_path.read_text(encoding="utf-8"))["name"]
    return {
        "config": {
            "trial_id": trial_id,
            "mode": "open-loop",
            "request_rate": request_rate,
            "metadata": {
                "workload": {
                    "name": workload_name,
                    "source_path": str(workload_path),
                },
                "stability_policy": {
                    "ttft_slo_ms": ttft_slo_ms,
                    "tpot_slo_ms": tpot_slo_ms,
                    "ttft_slo_field": "ttft_p90_ms",
                    "tpot_slo_field": "tpot_p90_ms",
                },
                "server_config": {
                    "max_num_seqs": max_num_seqs,
                    "max_num_batched_tokens": max_num_batched_tokens,
                },
            },
        },
        "summary": {
            "trial_id": trial_id,
            "successful_completion_rate": completion_rate,
            "benchmark_metrics": {
                "total_token_throughput": total_token_throughput,
                "generation_token_throughput": generation_token_throughput,
                "prompt_length_summary": {"mean": prompt_len_mean},
                "output_length_summary": {"mean": output_len_mean},
            },
            "metadata": {"model": model},
        },
    }


def _analysis_payload(*, trial_id: str, stability_status: str, confidence: str = "high") -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "trial_validity": "valid",
        "validity_reasons": ["fixture valid"],
        "stability": {
            "status": stability_status,
            "confidence": confidence,
            "reasons": [f"fixture {stability_status}"],
            "key_metrics": {},
        },
        "bottleneck": {
            "bottleneck_class": "unknown",
            "confidence": "low",
            "evidence": ["fixture evidence"],
        },
    }


def _write_result_bundle(
    job,
    *,
    workload_path: Path,
    mst_rps: float,
    high_bound_rate: float,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    confidence: str = "low",
    confirmation_status: str = "stable",
    high_bound_status: str = "unstable",
    include_majority_conflict: bool = False,
) -> None:
    result_dir = Path(job.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    low_trial_id = "trial_000_openloop_low"
    high_trial_id = "trial_001_openloop_high"
    confirmation_trial_id = "trial_002_openloop_confirm"
    majority_trial_id = "trial_003_openloop_confirm_majority"

    trial_specs = [
        ("open_loop_bracket", low_trial_id, mst_rps, mst_rps, confirmation_status, 3200.0, 2200.0, 166.0, 333.0),
        ("open_loop_bracket_high", high_trial_id, high_bound_rate, max(high_bound_rate - 0.4, 0.1), high_bound_status, 3000.0, 2000.0, 166.0, 333.0),
        ("open_loop_confirmation", confirmation_trial_id, mst_rps, mst_rps, "unstable" if include_majority_conflict else confirmation_status, 3200.0, 2200.0, 166.0, 333.0),
    ]
    if include_majority_conflict:
        trial_specs.append(
            ("open_loop_confirmation_majority", majority_trial_id, mst_rps, mst_rps, confirmation_status, 3200.0, 2200.0, 166.0, 333.0)
        )

    events = []
    for purpose, trial_id, request_rate, completion_rate, stability_status, total_tok_s, gen_tok_s, prompt_len, output_len in trial_specs:
        trial_dir = result_dir / "trials" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = _summary_payload(
            trial_id,
            workload_path=workload_path,
            model=job.model,
            request_rate=request_rate,
            completion_rate=completion_rate,
            total_token_throughput=total_tok_s,
            generation_token_throughput=gen_tok_s,
            prompt_len_mean=prompt_len,
            output_len_mean=output_len,
            ttft_slo_ms=ttft_slo_ms,
            tpot_slo_ms=tpot_slo_ms,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        analysis_payload = _analysis_payload(trial_id=trial_id, stability_status=stability_status)
        (trial_dir / "summary.json").write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (trial_dir / "analysis.json").write_text(
            json.dumps(analysis_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        events.append(
            {
                "purpose": purpose,
                "trial_id": trial_id,
                "trial_dir": f"trials/{trial_id}",
                "mode": "open-loop",
                "request_rate": request_rate,
                "analysis": analysis_payload,
            }
        )

    chosen_confirmation_trial_id = majority_trial_id if include_majority_conflict else confirmation_trial_id
    search_trace = {
        "config": {
            "search_id": job.experiment_id,
            "search_mode": "open-loop",
            "endpoint": job.endpoint,
            "model": job.model,
        },
        "bounds": {
            "low_rate": mst_rps,
            "low_trial_id": low_trial_id,
            "high_rate": high_bound_rate,
            "high_trial_id": high_trial_id,
        },
        "result": {
            "search_id": job.experiment_id,
            "search_mode": "open-loop",
            "max_no_drift_request_rate": mst_rps,
            "max_slo_satisfying_request_rate": mst_rps,
            "rate_precision": 0.1,
            "confirmation_trial_id": chosen_confirmation_trial_id,
            "termination_reason": "confirmed_stable",
            "bottleneck_class": "unknown",
            "confidence": confidence,
            "reasons": ["fixture result"],
        },
        "events": events,
    }
    (result_dir / "search_trace.json").write_text(
        json.dumps(search_trace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_report = {
        "workload": {
            "name": yaml.safe_load(workload_path.read_text(encoding="utf-8"))["name"],
            "source_path": str(workload_path),
        },
        "stability_policy": {
            "ttft_slo_ms": ttft_slo_ms,
            "tpot_slo_ms": tpot_slo_ms,
            "ttft_slo_field": "ttft_p90_ms",
            "tpot_slo_field": "tpot_p90_ms",
        },
        "server_config": {
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "model": job.model,
        },
        "search_trace": {
            "model": job.model,
            "search_mode": "open-loop",
            "trial_duration_s": 120,
            "final_confirmation_duration_s": 240,
            "rate_precision": 0.1,
        },
        "search_result": {
            "max_no_drift_request_rate": mst_rps,
            "max_slo_satisfying_request_rate": mst_rps,
            "confidence": confidence,
            "confirmation_trial_id": chosen_confirmation_trial_id,
        },
        "bottleneck": {
            "class": "unknown",
            "confidence": "low",
            "evidence": ["fixture evidence"],
        },
    }
    (result_dir / "final_report.json").write_text(
        json.dumps(final_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_orchestrator_run(tmp_path: Path, *, models: list[str], bundle_specs: dict[str, dict[str, object]]) -> Path:
    workload = _write_workload(tmp_path)
    manifest_path = _write_manifest(tmp_path, workload, models=models)
    manifest = load_manifest(manifest_path)
    jobs = {job.model: job for job in expand_manifest(manifest)}

    run_root = tmp_path / "results" / "orchestrator" / "fixture-run"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "state.json").write_text(
        json.dumps({"manifest_path": str(manifest_path)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary_jobs = []
    for model in models:
        job = jobs[model]
        spec = bundle_specs[model]
        _write_result_bundle(job, workload_path=workload, **spec)
        result_dir = Path(job.result_dir)
        summary_jobs.append(
            {
                "experiment_id": job.experiment_id,
                "status": "succeeded",
                "result_dir": str(result_dir),
                "artifacts": {
                    "search_trace": str(result_dir / "search_trace.json"),
                    "final_report_json": str(result_dir / "final_report.json"),
                },
            }
        )

    (run_root / "summary.json").write_text(
        json.dumps({"jobs": summary_jobs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_root


def _trial_ref(tmp_path: Path, name: str, status: str = "stable") -> TrialArtifactRef:
    trial_dir = tmp_path / name
    trial_dir.mkdir(parents=True, exist_ok=True)
    summary_json = trial_dir / "summary.json"
    analysis_json = trial_dir / "analysis.json"
    summary_json.write_text("{}\n", encoding="utf-8")
    analysis_json.write_text("{}\n", encoding="utf-8")
    return TrialArtifactRef(
        trial_id=name,
        trial_dir=trial_dir,
        summary_json=summary_json,
        analysis_json=analysis_json,
        trial_validity="valid",
        stability_status=status,
        reasons=(f"fixture {status}",),
    )


def _row(
    tmp_path: Path,
    *,
    experiment_id: str,
    model: str,
    family: str,
    variant: str | None,
    size_b: float,
    bucket: str | None,
    mst_rps: float,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    confidence: str = "high",
    trace_instability: TraceInstabilityEvidence | None = None,
) -> MSTRow:
    result_dir = tmp_path / "results" / experiment_id
    result_dir.mkdir(parents=True, exist_ok=True)
    search_trace_path = result_dir / "search_trace.json"
    final_report_path = result_dir / "final_report.json"
    search_trace_path.write_text("{}\n", encoding="utf-8")
    final_report_path.write_text("{}\n", encoding="utf-8")
    return MSTRow(
        experiment_id=experiment_id,
        model=model,
        model_family=family,
        model_variant=variant,
        model_size_b=size_b,
        size_bucket=bucket,
        hardware="l40-48gb",
        workload_name="live_sharegpt_workload_context",
        workload_path=tmp_path / "workload.yaml",
        endpoint="/v1/chat/completions",
        endpoint_type="chat_completions",
        mst_rps=mst_rps,
        termination_reason="confirmed_stable",
        bottleneck_class="unknown",
        confidence=confidence,
        server_signature_key=experiment_id,
        max_num_seqs=float(max_num_seqs),
        max_num_batched_tokens=float(max_num_batched_tokens),
        max_model_len=32768,
        tensor_parallel_size=1,
        gpu_count=1,
        dtype="float16",
        quantization=None,
        is_quantized=False,
        is_moe=False,
        ttft_slo_ms=ttft_slo_ms,
        tpot_slo_ms=tpot_slo_ms,
        ttft_slo_field="ttft_p90_ms",
        tpot_slo_field="tpot_p90_ms",
        confirmation_trial=_trial_ref(tmp_path, f"{experiment_id}-confirm"),
        confirmation_successful_completion_rate=mst_rps,
        confirmation_total_token_throughput=mst_rps * 500,
        confirmation_generation_token_throughput=mst_rps * 350,
        confirmation_prompt_len_mean=166.0,
        confirmation_output_len_mean=333.0,
        high_bound_rate=mst_rps * 1.05,
        high_bound_trial=_trial_ref(tmp_path, f"{experiment_id}-high", status="unstable"),
        result_dir=result_dir,
        search_trace_path=search_trace_path,
        final_report_json_path=final_report_path,
        trace_instability=trace_instability or TraceInstabilityEvidence(),
        has_slo_signal=False,
    )


def test_extract_infers_common_model_sizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    models = [
        "Qwen/Qwen3-0.6B",
        "google/gemma-4-E4B-it",
        "meta-llama/Llama-2-13b-chat-hf",
        "Qwen/Qwen3-14B",
    ]
    run_root = _write_orchestrator_run(
        tmp_path,
        models=models,
        bundle_specs={
            model: {
                "mst_rps": 4.0 + idx,
                "high_bound_rate": 4.5 + idx,
                "ttft_slo_ms": 250,
                "tpot_slo_ms": 50,
                "max_num_seqs": 1024,
                "max_num_batched_tokens": 8192,
            }
            for idx, model in enumerate(models)
        },
    )

    extracted = extract_run(run_root)
    sizes = {row.model: row.model_size_b for row in extracted.rows}
    assert sizes["Qwen/Qwen3-0.6B"] == pytest.approx(0.6)
    assert sizes["google/gemma-4-E4B-it"] == pytest.approx(4.0)
    assert sizes["meta-llama/Llama-2-13b-chat-hf"] == pytest.approx(13.0)
    assert sizes["Qwen/Qwen3-14B"] == pytest.approx(14.0)


def test_rules_flag_expected_anomalies_and_contextual_slo_mismatch(tmp_path: Path) -> None:
    qwen06 = _row(
        tmp_path,
        experiment_id="qwen06",
        model="Qwen/Qwen3-0.6B",
        family="qwen3",
        variant=None,
        size_b=0.6,
        bucket="tiny",
        mst_rps=17.4,
        ttft_slo_ms=250,
        tpot_slo_ms=50,
        max_num_seqs=1024,
        max_num_batched_tokens=8192,
        confidence="low",
        trace_instability=TraceInstabilityEvidence(
            conflicting_rate_labels=("17.4",),
            majority_confirmation_used=True,
            low_confidence=True,
        ),
    )
    llama1 = _row(
        tmp_path,
        experiment_id="llama1",
        model="meta-llama/Llama-3.2-1B-Instruct",
        family="llama",
        variant="instruct",
        size_b=1.0,
        bucket="small",
        mst_rps=25.8,
        ttft_slo_ms=250,
        tpot_slo_ms=50,
        max_num_seqs=1024,
        max_num_batched_tokens=8192,
    )
    gemma4 = _row(
        tmp_path,
        experiment_id="gemma4",
        model="google/gemma-4-E4B-it",
        family="gemma",
        variant="instruct",
        size_b=4.0,
        bucket="mid",
        mst_rps=5.64,
        ttft_slo_ms=500,
        tpot_slo_ms=100,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
        confidence="low",
    )
    qwen4i = _row(
        tmp_path,
        experiment_id="qwen4i",
        model="Qwen/Qwen3-4B-Instruct-2507",
        family="qwen3",
        variant="instruct",
        size_b=4.0,
        bucket="mid",
        mst_rps=7.92,
        ttft_slo_ms=500,
        tpot_slo_ms=100,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
    )
    llama3 = _row(
        tmp_path,
        experiment_id="llama3",
        model="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        variant="instruct",
        size_b=3.0,
        bucket="mid",
        mst_rps=8.6,
        ttft_slo_ms=500,
        tpot_slo_ms=100,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
    )
    qwen4t = _row(
        tmp_path,
        experiment_id="qwen4t",
        model="Qwen/Qwen3-4B-Thinking-2507",
        family="qwen3",
        variant="thinking",
        size_b=4.0,
        bucket="mid",
        mst_rps=4.27,
        ttft_slo_ms=500,
        tpot_slo_ms=100,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
        confidence="low",
    )
    qwen8 = _row(
        tmp_path,
        experiment_id="qwen8",
        model="Qwen/Qwen3-8B",
        family="qwen3",
        variant=None,
        size_b=8.0,
        bucket="large",
        mst_rps=5.42,
        ttft_slo_ms=1000,
        tpot_slo_ms=150,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
    )

    anomalies, _ = analyze_rows([qwen06, llama1, gemma4, qwen4i, llama3, qwen4t, qwen8])
    by_model = {anomaly.model: anomaly for anomaly in anomalies}

    assert "larger_model_inversion" in by_model["Qwen/Qwen3-0.6B"].families
    assert "trace_instability_suspect" in by_model["Qwen/Qwen3-0.6B"].families

    gemma_anomaly = by_model["google/gemma-4-E4B-it"]
    assert "larger_model_inversion" in gemma_anomaly.families
    larger_comparator = next(
        comparator for comparator in gemma_anomaly.comparators if comparator.relation == "larger_model"
    )
    assert larger_comparator.comparison_label == "contextual"

    thinking_anomaly = by_model["Qwen/Qwen3-4B-Thinking-2507"]
    assert "within_size_outlier" in thinking_anomaly.families
    assert "same_family_non_monotonicity" in thinking_anomaly.families


def test_rules_do_not_over_alert_for_sub_one_rps_models(tmp_path: Path) -> None:
    llama13 = _row(
        tmp_path,
        experiment_id="llama13",
        model="meta-llama/Llama-2-13b-chat-hf",
        family="llama",
        variant="chat",
        size_b=13.0,
        bucket="xlarge",
        mst_rps=0.3,
        ttft_slo_ms=1000,
        tpot_slo_ms=150,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
    )
    qwen14 = _row(
        tmp_path,
        experiment_id="qwen14",
        model="Qwen/Qwen3-14B",
        family="qwen3",
        variant=None,
        size_b=14.0,
        bucket="xlarge",
        mst_rps=0.6,
        ttft_slo_ms=1000,
        tpot_slo_ms=150,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
    )

    anomalies, _ = analyze_rows([llama13, qwen14])
    assert anomalies == []


def test_rules_allow_configured_suppressions_and_thresholds(tmp_path: Path) -> None:
    qwen06 = _row(
        tmp_path,
        experiment_id="qwen06",
        model="Qwen/Qwen3-0.6B",
        family="qwen3",
        variant=None,
        size_b=0.6,
        bucket="tiny",
        mst_rps=0.9,
        ttft_slo_ms=250,
        tpot_slo_ms=50,
        max_num_seqs=1024,
        max_num_batched_tokens=8192,
        confidence="low",
        trace_instability=TraceInstabilityEvidence(
            conflicting_rate_labels=("0.9",),
            majority_confirmation_used=True,
            low_confidence=True,
        ),
    )
    llama1 = _row(
        tmp_path,
        experiment_id="llama1",
        model="meta-llama/Llama-3.2-1B-Instruct",
        family="llama",
        variant="instruct",
        size_b=1.0,
        bucket="small",
        mst_rps=1.1,
        ttft_slo_ms=250,
        tpot_slo_ms=50,
        max_num_seqs=1024,
        max_num_batched_tokens=8192,
    )

    default_anomalies, _, default_diagnostics = analyze_rows_with_diagnostics([qwen06, llama1])
    assert default_anomalies == []
    assert [diagnostic.model for diagnostic in default_diagnostics] == ["Qwen/Qwen3-0.6B"]

    trace_as_anomaly_settings = AnalyzerSettings.from_dict({"include_trace_only_findings": True})
    trace_as_anomaly, _ = analyze_rows([qwen06, llama1], settings=trace_as_anomaly_settings)
    assert trace_as_anomaly

    settings = AnalyzerSettings.from_dict(
        {
            "suppressions": {
                "disable_families": ["larger_model_inversion"],
                "suppress_trace_instability_below_rps": 1.0,
            }
        }
    )
    suppressed_anomalies, _ = analyze_rows([qwen06, llama1], settings=settings)
    assert suppressed_anomalies == []


def test_rules_allow_configured_outlier_bands(tmp_path: Path) -> None:
    gemma4 = _row(
        tmp_path,
        experiment_id="gemma4",
        model="google/gemma-4-E4B-it",
        family="gemma",
        variant="instruct",
        size_b=4.0,
        bucket="mid",
        mst_rps=5.64,
        ttft_slo_ms=500,
        tpot_slo_ms=100,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
    )
    qwen4i = _row(
        tmp_path,
        experiment_id="qwen4i",
        model="Qwen/Qwen3-4B-Instruct-2507",
        family="qwen3",
        variant="instruct",
        size_b=4.0,
        bucket="mid",
        mst_rps=7.92,
        ttft_slo_ms=500,
        tpot_slo_ms=100,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
    )
    llama3 = _row(
        tmp_path,
        experiment_id="llama3",
        model="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        variant="instruct",
        size_b=3.0,
        bucket="mid",
        mst_rps=8.6,
        ttft_slo_ms=500,
        tpot_slo_ms=100,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
    )

    default_anomalies, _ = analyze_rows([gemma4, qwen4i, llama3])
    assert default_anomalies == []

    qwen4t = _row(
        tmp_path,
        experiment_id="qwen4t",
        model="Qwen/Qwen3-4B-Thinking-2507",
        family="qwen3",
        variant="thinking",
        size_b=4.0,
        bucket="mid",
        mst_rps=4.27,
        ttft_slo_ms=500,
        tpot_slo_ms=100,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
    )
    anomalies, _ = analyze_rows([gemma4, qwen4i, llama3, qwen4t])
    assert any(anomaly.model == "Qwen/Qwen3-4B-Thinking-2507" for anomaly in anomalies)

    custom_settings = AnalyzerSettings.from_dict(
        {
            "outlier_bands": [
                {
                    "min_rate": 2.0,
                    "max_rate": 10.0,
                    "ratio_threshold": 2.0,
                    "absolute_delta_rps": 3.0,
                },
                {
                    "min_rate": 10.0,
                    "max_rate": None,
                    "ratio_threshold": 1.5,
                    "absolute_delta_rps": 5.0,
                },
                {
                    "min_rate": 0.0,
                    "max_rate": 2.0,
                    "ratio_threshold": 2.5,
                    "absolute_delta_rps": 1.0,
                },
            ],
            "suppressions": {
                "disable_families": ["larger_model_inversion", "same_family_non_monotonicity"],
            },
        }
    )
    custom_anomalies, _ = analyze_rows([gemma4, qwen4i, llama3, qwen4t], settings=custom_settings)
    assert custom_anomalies == []


def test_load_settings_and_report_include_analysis_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    models = [
        "Qwen/Qwen3-0.6B",
        "meta-llama/Llama-3.2-1B-Instruct",
    ]
    run_root = _write_orchestrator_run(
        tmp_path,
        models=models,
        bundle_specs={
            "Qwen/Qwen3-0.6B": {
                "mst_rps": 0.9,
                "high_bound_rate": 1.0,
                "ttft_slo_ms": 250,
                "tpot_slo_ms": 50,
                "max_num_seqs": 1024,
                "max_num_batched_tokens": 8192,
                "confidence": "low",
                "include_majority_conflict": True,
            },
            "meta-llama/Llama-3.2-1B-Instruct": {
                "mst_rps": 1.1,
                "high_bound_rate": 1.2,
                "ttft_slo_ms": 250,
                "tpot_slo_ms": 50,
                "max_num_seqs": 1024,
                "max_num_batched_tokens": 8192,
                "confidence": "high",
            },
        },
    )
    settings_path = tmp_path / "analyzer_settings.yaml"
    settings_path.write_text(
        yaml.safe_dump(
            {
                "suppressions": {
                    "disable_families": ["larger_model_inversion"],
                    "suppress_trace_instability_below_rps": 1.0,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    settings = load_settings(settings_path)

    output_dir = tmp_path / "results" / "analysis" / "fixture-run"
    artifacts = analyze_orchestrator_run(
        orchestrator_run_root=run_root,
        output_dir=output_dir,
        settings=settings,
    )
    payload = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert payload["analysis_config"]["suppressions"]["disable_families"] == ["larger_model_inversion"]
    assert payload["analysis_config"]["suppressions"]["suppress_trace_instability_below_rps"] == pytest.approx(1.0)
    assert payload["summary"]["anomaly_count"] == 0


def test_report_includes_evidence_paths_and_rerun_manifest_uses_distinct_workload_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    models = [
        "Qwen/Qwen3-0.6B",
        "meta-llama/Llama-3.2-1B-Instruct",
    ]
    run_root = _write_orchestrator_run(
        tmp_path,
        models=models,
        bundle_specs={
            "Qwen/Qwen3-0.6B": {
                "mst_rps": 17.4,
                "high_bound_rate": 18.3,
                "ttft_slo_ms": 250,
                "tpot_slo_ms": 50,
                "max_num_seqs": 1024,
                "max_num_batched_tokens": 8192,
                "confidence": "low",
                "include_majority_conflict": True,
            },
            "meta-llama/Llama-3.2-1B-Instruct": {
                "mst_rps": 25.8,
                "high_bound_rate": 27.0,
                "ttft_slo_ms": 250,
                "tpot_slo_ms": 50,
                "max_num_seqs": 1024,
                "max_num_batched_tokens": 8192,
                "confidence": "high",
            },
        },
    )

    output_dir = tmp_path / "results" / "analysis" / "fixture-run"
    artifacts = analyze_orchestrator_run(
        orchestrator_run_root=run_root,
        output_dir=output_dir,
        emit_rerun_manifest=True,
    )

    assert artifacts.report_json_path.is_file()
    assert artifacts.report_md_path.is_file()
    assert artifacts.rerun_manifest_path is not None and artifacts.rerun_manifest_path.is_file()

    report_payload = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report_payload["anomalies"]
    first_anomaly = report_payload["anomalies"][0]
    evidence_paths = first_anomaly["evidence_paths"]
    assert any(path.endswith("search_trace.json") for path in evidence_paths)
    assert any(path.endswith("summary.json") for path in evidence_paths)

    rerun_manifest = yaml.safe_load(Path(artifacts.rerun_manifest_path).read_text(encoding="utf-8"))
    assert rerun_manifest["search"]["trial_min_duration_s"] >= 180
    assert rerun_manifest["search"]["trial_max_duration_s"] >= 300
    assert rerun_manifest["search"]["final_confirmation_duration_s"] >= 300
    rerun_workload_relative = rerun_manifest["experiments"][0]["workloads"][0]
    rerun_workload_path = output_dir / rerun_workload_relative
    assert rerun_workload_path.stem.endswith("_mst_anomaly_rerun")
    rerun_workload_payload = yaml.safe_load(rerun_workload_path.read_text(encoding="utf-8"))
    assert rerun_workload_payload["name"].endswith("_mst_anomaly_rerun")


def test_report_keeps_trace_only_findings_as_diagnostics_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    models = [
        "Qwen/Qwen3-0.6B",
        "meta-llama/Llama-3.2-1B-Instruct",
    ]
    run_root = _write_orchestrator_run(
        tmp_path,
        models=models,
        bundle_specs={
            "Qwen/Qwen3-0.6B": {
                "mst_rps": 0.9,
                "high_bound_rate": 1.0,
                "ttft_slo_ms": 250,
                "tpot_slo_ms": 50,
                "max_num_seqs": 1024,
                "max_num_batched_tokens": 8192,
                "confidence": "low",
                "include_majority_conflict": True,
            },
            "meta-llama/Llama-3.2-1B-Instruct": {
                "mst_rps": 1.1,
                "high_bound_rate": 1.2,
                "ttft_slo_ms": 250,
                "tpot_slo_ms": 50,
                "max_num_seqs": 1024,
                "max_num_batched_tokens": 8192,
                "confidence": "high",
            },
        },
    )

    output_dir = tmp_path / "results" / "analysis" / "fixture-run"
    artifacts = analyze_orchestrator_run(
        orchestrator_run_root=run_root,
        output_dir=output_dir,
    )
    payload = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert payload["summary"]["anomaly_count"] == 0
    assert payload["summary"]["trace_diagnostic_count"] == 1
    assert payload["trace_diagnostics"][0]["model"] == "Qwen/Qwen3-0.6B"
    assert artifacts.rerun_manifest_path is None
