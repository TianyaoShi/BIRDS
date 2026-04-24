from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .loadgen import cycling_request_source
from .metrics_polling import PrometheusMetricsPoller
from .records import TrialConfig
from .request_client import RequestClient
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
    run_trial.set_defaults(handler=_run_trial_command)
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
        metadata=prepared_workload.metadata,
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


def _default_trial_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"trial-{ts}"


def _default_workload_path() -> Path:
    return Path(__file__).resolve().parent / "workloads" / "synthetic_fixed_512_128.yaml"


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
