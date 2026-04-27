from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from math import isfinite
from typing import Any, Mapping, Sequence

from .analysis import analyze_trial_dir, write_analysis_artifact
from .loadgen import cycling_request_source
from .metrics_polling import PrometheusMetricsPoller
from .reporting import generate_report
from .records import TrialConfig
from .request_client import RequestClient
from .search import SearchConfig, SearchController
from .stability import StabilityConfig
from .trial_runner import TrialRunner
from .windowing import FixedWindowAggregator
from .workload import prepare_workload_for_trial


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm_mst_finder.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_trial = subparsers.add_parser("run-trial")
    run_trial.add_argument(
        "--trial-id",
        default=_default_trial_id(),
        help="unique trial identifier",
    )
    run_trial.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="artifact directory; defaults to results/<trial-id>",
    )
    run_trial.add_argument(
        "--workload",
        type=Path,
        default=_default_workload_path(),
        help="workload YAML path",
    )
    run_trial.add_argument(
        "--mode",
        choices=("open-loop", "closed-loop"),
        required=True,
        help="load generation mode",
    )
    run_trial.add_argument("--duration-s", type=float, default=60.0)
    run_trial.add_argument("--base-url", default="http://127.0.0.1:8000")
    run_trial.add_argument(
        "--endpoint",
        default="/v1/completions",
        help="OpenAI-compatible completions endpoint",
    )
    run_trial.add_argument("--model", required=True)
    run_trial.add_argument("--request-rate", type=float, default=None)
    run_trial.add_argument("--concurrency", type=int, default=None)
    run_trial.add_argument("--think-time-s", type=float, default=0.0)
    run_trial.add_argument("--burstiness", type=float, default=1.0)
    run_trial.add_argument("--request-timeout-s", type=float, default=6 * 60 * 60)
    run_trial.add_argument("--api-key", default=None)
    run_trial.add_argument(
        "--extra-header",
        action="append",
        default=[],
        help="repeatable KEY=VALUE header",
    )
    run_trial.add_argument(
        "--extra-body-json",
        default=None,
        help="JSON object merged into every request body",
    )
    run_trial.add_argument("--safety-max-outstanding", type=int, default=None)
    run_trial.add_argument("--metrics-url", default=None)
    run_trial.add_argument("--metrics-interval-s", type=float, default=1.0)
    run_trial.add_argument("--window-s", type=float, default=10.0)
    _add_server_metadata_args(run_trial)
    _add_stability_policy_args(run_trial)
    run_trial.set_defaults(handler=_run_trial_command)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--trial-dir", type=Path, required=True)
    _add_stability_policy_args(analyze, defaults_from_policy=False)
    analyze.set_defaults(handler=_analyze_command)

    search = subparsers.add_parser("search")
    search.add_argument("--search-id", default=_default_search_id())
    search.add_argument("--search-mode", choices=("closed-loop", "open-loop", "hybrid"), default="hybrid")
    search.add_argument("--output-dir", type=Path, required=True)
    search.add_argument(
        "--workload",
        "--workload-config",
        dest="workload",
        type=Path,
        default=_default_workload_path(),
        help="workload YAML path",
    )
    search.add_argument("--base-url", default="http://127.0.0.1:8000")
    search.add_argument("--endpoint", default="/v1/completions")
    search.add_argument("--model", required=True)
    search.add_argument("--trial-min-duration-s", type=float, default=60.0)
    search.add_argument("--trial-max-duration-s", type=float, default=None)
    search.add_argument("--uncertain-trial-duration-multiplier", type=float, default=2.0)
    search.add_argument("--final-confirmation-duration-s", type=float, default=None)
    search.add_argument("--rate-precision", type=float, default=0.03)
    search.add_argument("--initial-request-rate", type=float, default=1.0)
    search.add_argument("--max-request-rate", type=float, default=None)
    search.add_argument("--max-binary-steps", type=int, default=24)
    search.add_argument("--max-bracket-trials", type=int, default=16)
    search.add_argument("--closed-loop-initial-concurrency", type=int, default=1)
    search.add_argument("--max-closed-loop-concurrency", type=int, default=128)
    search.add_argument("--closed-loop-plateau-relative-gain", type=float, default=0.05)
    search.add_argument("--think-time-s", type=float, default=0.0)
    search.add_argument("--burstiness", type=float, default=1.0)
    search.add_argument("--request-timeout-s", type=float, default=6 * 60 * 60)
    search.add_argument("--api-key", default=None)
    search.add_argument("--extra-header", action="append", default=[])
    search.add_argument("--extra-body-json", default=None)
    search.add_argument("--safety-max-outstanding", type=int, default=None)
    search.add_argument("--metrics-url", "--server-metrics-url", dest="metrics_url", default=None)
    search.add_argument("--metrics-interval-s", type=float, default=1.0)
    search.add_argument("--window-s", type=float, default=10.0)
    _add_server_metadata_args(search)
    _add_stability_policy_args(search)
    search.set_defaults(handler=_search_command)

    report = subparsers.add_parser("report")
    report.add_argument("--result-dir", type=Path, required=True)
    report.add_argument("--compare-result-dir", action="append", default=[])
    report.add_argument("--disable-plots", action="store_true")
    report.set_defaults(handler=_report_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return asyncio.run(args.handler(args))


async def _run_trial_command(args: argparse.Namespace) -> int:
    prepared_workload = prepare_workload_for_trial(
        args.workload,
        model_name=args.model,
    )
    request_samples = prepared_workload.samples
    request_source = cycling_request_source(request_samples)
    request_client = RequestClient(
        base_url=args.base_url,
        endpoint=args.endpoint,
        model=args.model,
        timeout_s=args.request_timeout_s,
        api_key=args.api_key,
        extra_headers=_parse_extra_headers(args.extra_header),
        extra_body=_parse_json_mapping(args.extra_body_json, field_name="--extra-body-json"),
    )
    metrics_poller = (
        PrometheusMetricsPoller(
            metrics_url=args.metrics_url,
            interval_s=args.metrics_interval_s,
        )
        if args.metrics_url is not None
        else None
    )
    runner = TrialRunner(
        request_client,
        metrics_poller=metrics_poller,
        window_aggregator=FixedWindowAggregator(window_s=args.window_s),
    )
    metadata = _merge_cli_metadata(
        prepared_workload.metadata,
        _parse_server_metadata_args(args),
        _stability_policy_payload_from_args(args),
    )
    config = TrialConfig(
        trial_id=args.trial_id,
        mode=args.mode,
        duration_s=args.duration_s,
        base_url=args.base_url,
        endpoint=args.endpoint,
        model=args.model,
        request_rate=args.request_rate,
        concurrency=args.concurrency,
        think_time_s=args.think_time_s,
        burstiness=args.burstiness,
        request_timeout_s=args.request_timeout_s,
        api_key=args.api_key,
        extra_headers=_parse_extra_headers(args.extra_header),
        extra_body=_parse_json_mapping(args.extra_body_json, field_name="--extra-body-json"),
        safety_max_outstanding=args.safety_max_outstanding,
        metrics_url=args.metrics_url,
        metrics_interval_s=args.metrics_interval_s,
        window_s=args.window_s,
        metadata=metadata,
    )
    output_dir = args.output_dir if args.output_dir is not None else Path("results") / args.trial_id

    try:
        result = await runner.run_trial(
            config,
            request_source=request_source,
            output_dir=output_dir,
        )
    finally:
        await request_client.close()
        if metrics_poller is not None:
            await metrics_poller.close()

    print(
        json.dumps(
            {
                "output_dir": str(result.artifacts.output_dir),
                "status": result.summary.status,
                "trial_id": result.summary.trial_id,
            },
            sort_keys=True,
        )
    )
    return 0


async def _analyze_command(args: argparse.Namespace) -> int:
    result = analyze_trial_dir(
        args.trial_dir,
        stability_config=_stability_config_override_from_args(args),
    )
    write_analysis_artifact(args.trial_dir, result)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


async def _search_command(args: argparse.Namespace) -> int:
    prepared_workload = prepare_workload_for_trial(
        args.workload,
        model_name=args.model,
    )
    request_client = RequestClient(
        base_url=args.base_url,
        endpoint=args.endpoint,
        model=args.model,
        timeout_s=args.request_timeout_s,
        api_key=args.api_key,
        extra_headers=_parse_extra_headers(args.extra_header),
        extra_body=_parse_json_mapping(args.extra_body_json, field_name="--extra-body-json"),
    )
    metrics_poller = (
        PrometheusMetricsPoller(
            metrics_url=args.metrics_url,
            interval_s=args.metrics_interval_s,
        )
        if args.metrics_url is not None
        else None
    )
    runner = TrialRunner(
        request_client,
        metrics_poller=metrics_poller,
        window_aggregator=FixedWindowAggregator(window_s=args.window_s),
    )
    metadata = _merge_cli_metadata(
        prepared_workload.metadata,
        _parse_server_metadata_args(args),
        _stability_policy_payload_from_args(args),
    )
    config = SearchConfig(
        search_id=args.search_id,
        search_mode=args.search_mode,
        base_url=args.base_url,
        endpoint=args.endpoint,
        model=args.model,
        trial_duration_s=args.trial_min_duration_s,
        uncertain_trial_duration_s=args.trial_max_duration_s,
        uncertain_trial_duration_multiplier=args.uncertain_trial_duration_multiplier,
        final_confirmation_duration_s=(
            args.final_confirmation_duration_s
            if args.final_confirmation_duration_s is not None
            else args.trial_max_duration_s
        ),
        rate_precision=args.rate_precision,
        initial_request_rate=args.initial_request_rate,
        max_request_rate=args.max_request_rate,
        max_binary_steps=args.max_binary_steps,
        max_bracket_trials=args.max_bracket_trials,
        closed_loop_initial_concurrency=args.closed_loop_initial_concurrency,
        max_closed_loop_concurrency=args.max_closed_loop_concurrency,
        closed_loop_plateau_relative_gain=args.closed_loop_plateau_relative_gain,
        think_time_s=args.think_time_s,
        burstiness=args.burstiness,
        request_timeout_s=args.request_timeout_s,
        api_key=args.api_key,
        extra_headers=_parse_extra_headers(args.extra_header),
        extra_body=_parse_json_mapping(args.extra_body_json, field_name="--extra-body-json"),
        safety_max_outstanding=args.safety_max_outstanding,
        metrics_url=args.metrics_url,
        metrics_interval_s=args.metrics_interval_s,
        window_s=args.window_s,
        metadata=metadata,
    )
    controller = SearchController(
        runner,
        request_source_factory=lambda: cycling_request_source(prepared_workload.samples),
        output_dir=args.output_dir,
    )
    try:
        result = await controller.search(config)
    finally:
        await request_client.close()
        if metrics_poller is not None:
            await metrics_poller.close()
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


async def _report_command(args: argparse.Namespace) -> int:
    result = generate_report(
        args.result_dir,
        compare_result_dirs=args.compare_result_dir,
        plots_enabled=not args.disable_plots,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _default_trial_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"trial-{ts}"


def _default_search_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"search-{ts}"


def _default_workload_path() -> Path:
    return Path(__file__).resolve().parent / "workloads" / "synthetic_fixed_512_128.yaml"


def _add_server_metadata_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--server-metadata-file",
        type=Path,
        default=None,
        help="JSON object containing explicit serving metadata for analysis/reporting",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=_positive_server_metadata_arg,
        default=None,
        help="explicit vLLM max_num_seqs value for this serving configuration",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=_positive_server_metadata_arg,
        default=None,
        help="explicit vLLM max_num_batched_tokens value for this serving configuration",
    )


def _add_stability_policy_args(
    parser: argparse.ArgumentParser,
    *,
    defaults_from_policy: bool = True,
) -> None:
    defaults = StabilityConfig()
    parser.add_argument(
        "--ttft-slo-ms",
        type=_optional_positive_float_arg,
        default=defaults.ttft_slo_ms if defaults_from_policy else None,
        help="TTFT SLO threshold in milliseconds; use 'none' to disable",
    )
    parser.add_argument(
        "--tpot-slo-ms",
        type=_optional_positive_float_arg,
        default=defaults.tpot_slo_ms if defaults_from_policy else None,
        help="TPOT SLO threshold in milliseconds; use 'none' to disable",
    )
    parser.add_argument(
        "--ttft-slo-field",
        choices=("ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms"),
        default=defaults.ttft_slo_field if defaults_from_policy else None,
    )
    parser.add_argument(
        "--tpot-slo-field",
        choices=("tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms"),
        default=defaults.tpot_slo_field if defaults_from_policy else None,
    )


def _parse_extra_headers(raw_headers: Sequence[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw_header in raw_headers:
        if "=" not in raw_header:
            raise ValueError(f"--extra-header must use KEY=VALUE, got {raw_header!r}")
        key, value = raw_header.split("=", 1)
        if not key:
            raise ValueError(f"--extra-header key must be non-empty, got {raw_header!r}")
        headers[key] = value
    return headers


def _parse_json_mapping(raw_json: str | None, *, field_name: str) -> dict[str, object] | None:
    if raw_json is None:
        return None
    payload = json.loads(raw_json)
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must decode to a JSON object")
    return payload


def _positive_server_metadata_arg(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def _optional_positive_float_arg(raw_value: str) -> float | None:
    if raw_value.lower() in {"none", "null", "off", "disabled"}:
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number or 'none'") from exc
    if not isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number or 'none'")
    return value


def _stability_policy_payload_from_args(args: argparse.Namespace) -> dict[str, object]:
    config = _stability_config_from_args(args)
    return {
        "warmup_windows": config.warmup_windows,
        "min_eval_windows": config.min_eval_windows,
        "completion_arrival_tolerance": config.completion_arrival_tolerance,
        "max_positive_backlog_slope": config.max_positive_backlog_slope,
        "min_backlog_growth_for_hard_pressure": config.min_backlog_growth_for_hard_pressure,
        "token_throughput_plateau_relative_growth": config.token_throughput_plateau_relative_growth,
        "max_error_rate": config.max_error_rate,
        "ttft_slo_ms": config.ttft_slo_ms,
        "tpot_slo_ms": config.tpot_slo_ms,
        "ttft_slo_field": config.ttft_slo_field,
        "tpot_slo_field": config.tpot_slo_field,
    }


def _stability_config_override_from_args(args: argparse.Namespace) -> StabilityConfig | None:
    if all(
        getattr(args, name) is None
        for name in (
            "ttft_slo_ms",
            "tpot_slo_ms",
            "ttft_slo_field",
            "tpot_slo_field",
        )
    ):
        return None
    return _stability_config_from_args(args)


def _stability_config_from_args(args: argparse.Namespace) -> StabilityConfig:
    defaults = StabilityConfig()
    return StabilityConfig(
        ttft_slo_ms=args.ttft_slo_ms,
        tpot_slo_ms=args.tpot_slo_ms,
        ttft_slo_field=args.ttft_slo_field or defaults.ttft_slo_field,
        tpot_slo_field=args.tpot_slo_field or defaults.tpot_slo_field,
    )


def _parse_server_metadata_args(args: argparse.Namespace) -> dict[str, object] | None:
    metadata: dict[str, object] = {}
    if args.server_metadata_file is not None:
        payload = _load_json_mapping(args.server_metadata_file, field_name="--server-metadata-file")
        metadata.update(payload)
        for key, value in _known_server_metadata_values(payload).items():
            _merge_server_metadata_value(metadata, key, value)
    if args.max_num_seqs is not None:
        _merge_server_metadata_value(metadata, "max_num_seqs", args.max_num_seqs)
    if args.max_num_batched_tokens is not None:
        _merge_server_metadata_value(metadata, "max_num_batched_tokens", args.max_num_batched_tokens)
    return metadata or None


def _load_json_mapping(path: Path, *, field_name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{field_name} does not exist: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must decode to a JSON object")
    return payload


def _known_server_metadata_values(payload: Mapping[str, object]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in ("max_num_seqs", "max_num_batched_tokens"):
        found_values: list[float] = []
        for mapping in _server_metadata_candidate_mappings(payload):
            if key not in mapping:
                continue
            found_values.append(_require_positive_metadata_number(mapping[key], key))
        if not found_values:
            continue
        first = found_values[0]
        if any(value != first for value in found_values[1:]):
            raise ValueError(f"conflicting supplied server metadata values for {key!r}: {found_values!r}")
        values[key] = first
    return values


def _server_metadata_candidate_mappings(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    mappings: list[Mapping[str, object]] = [payload]
    for key in ("server_metadata", "server_config", "vllm_config"):
        nested = payload.get(key)
        if nested is None:
            continue
        if not isinstance(nested, Mapping):
            raise ValueError(f"--server-metadata-file field {key!r} must be a JSON object")
        mappings.append(nested)
    return mappings


def _require_positive_metadata_number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"server metadata {key!r} must be a positive finite number")
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"server metadata {key!r} must be a positive finite number")
    return numeric


def _merge_server_metadata_value(metadata: dict[str, object], key: str, value: float) -> None:
    existing = metadata.get(key)
    if existing is not None:
        existing_value = _require_positive_metadata_number(existing, key)
        if existing_value != value:
            raise ValueError(
                f"conflicting supplied server metadata values for {key!r}: "
                f"{existing_value!r} != {value!r}"
            )
    metadata[key] = value


def _merge_cli_metadata(
    workload_metadata: Mapping[str, Any],
    server_metadata: Mapping[str, object] | None,
    stability_policy: Mapping[str, object] | None,
) -> dict[str, Any]:
    metadata = dict(workload_metadata)
    if stability_policy is not None:
        metadata["stability_policy"] = dict(stability_policy)
    if server_metadata is not None:
        existing = metadata.get("server_metadata")
        if existing is None:
            metadata["server_metadata"] = dict(server_metadata)
        else:
            if not isinstance(existing, Mapping):
                raise ValueError("workload metadata field 'server_metadata' must be a mapping when provided")
            metadata["server_metadata"] = _merge_metadata_mappings(existing, server_metadata, path="server_metadata")
    return metadata


def _merge_metadata_mappings(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    path: str,
) -> dict[str, object]:
    merged = dict(left)
    for key, right_value in right.items():
        current_path = f"{path}.{key}"
        if key not in merged:
            merged[key] = right_value
            continue
        left_value = merged[key]
        if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
            merged[key] = _merge_metadata_mappings(left_value, right_value, path=current_path)
            continue
        if left_value != right_value:
            raise ValueError(
                f"conflicting supplied metadata for {current_path}: {left_value!r} != {right_value!r}"
            )
    return merged


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
