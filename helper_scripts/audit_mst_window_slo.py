#!/usr/bin/env python3
"""Audit MST active-window SLO decisions against aggregate trial percentiles.

This is a one-off helper for estimating how often the active-window max
latency-percentile check would disagree with using aggregate trial percentiles.
By default it scans non-debug/non-validation orchestrator runs under
results/orchestrator and writes a compact audit under results/analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


DEFAULT_EXCLUDES = ("debug", "validation", "validate")
DEFAULT_OUTPUT_DIR = Path("results/analysis/window-slo-audit")
TTFT_PERCENTILES = {
    "ttft_p50_ms": 50.0,
    "ttft_p90_ms": 90.0,
    "ttft_p99_ms": 99.0,
}
TPOT_PERCENTILES = {
    "tpot_p50_ms": 50.0,
    "tpot_p90_ms": 90.0,
    "tpot_p99_ms": 99.0,
}


def main() -> int:
    args = _parse_args()
    run_roots = _selected_run_roots(
        orchestrator_root=args.orchestrator_root,
        explicit_run_roots=args.run_root,
        exclude_substrings=args.exclude_substring,
    )

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run_root in run_roots:
        rows.extend(_scan_run_root(run_root, warnings))

    summary = _summarize(run_roots, rows, warnings)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "window_slo_audit_rows.json", rows)
    _write_csv(output_dir / "window_slo_audit_rows.csv", rows)
    _write_json(output_dir / "window_slo_audit_summary.json", summary)
    _write_markdown(output_dir / "window_slo_audit_summary.md", summary, rows, args.print_top)

    print(
        json.dumps(
            {
                "run_count": summary["run_count"],
                "trial_count": summary["trial_count"],
                "static_policy_trial_count": summary["static_policy_trial_count"],
                "window_only_slo_violation_count": summary["window_only_slo_violation_count"],
                "aggregate_only_slo_violation_count": summary["aggregate_only_slo_violation_count"],
                "status_diff_count": summary["status_diff_count"],
                "warnings_count": summary["warnings_count"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orchestrator-root",
        type=Path,
        default=Path("results/orchestrator"),
        help="Directory containing orchestrator run roots (default: results/orchestrator).",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        default=[],
        help="Specific orchestrator run root to scan. May be repeated.",
    )
    parser.add_argument(
        "--exclude-substring",
        action="append",
        default=list(DEFAULT_EXCLUDES),
        help=(
            "Case-insensitive substring excluded from default run-root discovery. "
            "Defaults: debug, validation, validate. May be repeated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--print-top",
        type=int,
        default=20,
        help="Number of largest disagreement rows to include in Markdown (default: 20).",
    )
    return parser.parse_args()


def _selected_run_roots(
    *,
    orchestrator_root: Path,
    explicit_run_roots: list[Path],
    exclude_substrings: list[str],
) -> list[Path]:
    if explicit_run_roots:
        return sorted(path.resolve() for path in explicit_run_roots)

    excludes = tuple(item.lower() for item in exclude_substrings)
    roots: list[Path] = []
    if not orchestrator_root.exists():
        return roots
    for child in sorted(orchestrator_root.iterdir()):
        lowered = child.name.lower()
        if not child.is_dir() or any(excluded in lowered for excluded in excludes):
            continue
        if (child / "summary.json").is_file():
            roots.append(child.resolve())
    return roots


def _scan_run_root(run_root: Path, warnings: list[str]) -> list[dict[str, Any]]:
    payload = _load_json(run_root / "summary.json", warnings)
    if not isinstance(payload, Mapping):
        warnings.append(f"{run_root}: summary.json is not an object")
        return []

    rows: list[dict[str, Any]] = []
    run_id = run_root.name
    for job in _list(payload.get("jobs")):
        if not isinstance(job, Mapping):
            continue
        if job.get("status") != "succeeded":
            continue
        result_dir_raw = job.get("result_dir")
        if not isinstance(result_dir_raw, str) or not result_dir_raw:
            warnings.append(f"{run_root}: succeeded job {job.get('experiment_id')} has no result_dir")
            continue
        result_dir = Path(result_dir_raw)
        if not result_dir.is_absolute():
            result_dir = (run_root / result_dir).resolve()
        trial_summaries = sorted((result_dir / "trials").glob("*/summary.json"))
        if not trial_summaries:
            warnings.append(f"{run_root}: no trial summaries under {result_dir / 'trials'}")
            continue
        for summary_path in trial_summaries:
            row = _scan_trial(run_id, run_root, job, result_dir, summary_path, warnings)
            if row is not None:
                rows.append(row)
    return rows


def _scan_trial(
    run_id: str,
    run_root: Path,
    job: Mapping[str, Any],
    result_dir: Path,
    summary_path: Path,
    warnings: list[str],
) -> dict[str, Any] | None:
    summary = _load_json(summary_path, warnings)
    analysis_path = summary_path.with_name("analysis.json")
    analysis = _load_json(analysis_path, warnings)
    if not isinstance(summary, Mapping) or not isinstance(analysis, Mapping):
        return None

    config = _mapping(summary.get("config"))
    metadata = _mapping(config.get("metadata"))
    policy = _mapping(metadata.get("stability_policy"))
    if not policy:
        policy = _mapping(job.get("slo_policy"))
    stability = _mapping(analysis.get("stability"))
    key_metrics = _mapping(stability.get("key_metrics"))
    benchmark_metrics = _mapping(_mapping(summary.get("summary")).get("benchmark_metrics"))

    ttft_field = _string(policy.get("ttft_slo_field")) or "ttft_p90_ms"
    tpot_field = _string(policy.get("tpot_slo_field")) or "tpot_p90_ms"
    ttft_slo_ms = _float_or_none(policy.get("ttft_slo_ms"))
    tpot_slo_ms = _float_or_none(policy.get("tpot_slo_ms"))
    ttft_slo_mode = _string(policy.get("ttft_slo_mode")) or "static"
    longbench_preset = policy.get("longbench_ttft_static_preset")
    uses_request_level_ttft = ttft_slo_mode == "length_scaled" or longbench_preset is not None

    ttft_window = _float_or_none(key_metrics.get(f"{ttft_field}_max"))
    tpot_window = _float_or_none(key_metrics.get(f"{tpot_field}_max"))
    ttft_aggregate = _aggregate_percentile(benchmark_metrics, ttft_field)
    tpot_aggregate = _aggregate_percentile(benchmark_metrics, tpot_field)

    window_ttft_violates = _violates(ttft_window, ttft_slo_ms) if not uses_request_level_ttft else None
    aggregate_ttft_violates = _violates(ttft_aggregate, ttft_slo_ms) if not uses_request_level_ttft else None
    window_tpot_violates = _violates(tpot_window, tpot_slo_ms)
    aggregate_tpot_violates = _violates(tpot_aggregate, tpot_slo_ms)

    comparable = not uses_request_level_ttft
    window_slo_violation = _any_true(window_ttft_violates, window_tpot_violates)
    aggregate_slo_violation = _any_true(aggregate_ttft_violates, aggregate_tpot_violates)
    actual_status = _string(stability.get("status"))
    actual_is_slo = actual_status == "slo_violation"
    status_differs = comparable and actual_is_slo != bool(aggregate_slo_violation)

    trial_id = _string(analysis.get("trial_id")) or _string(config.get("trial_id")) or summary_path.parent.name
    request_rate = _float_or_none(config.get("request_rate"))
    workload = _mapping(metadata.get("workload"))

    return {
        "run_id": run_id,
        "run_root": str(run_root),
        "experiment_id": _string(job.get("experiment_id")),
        "model": _string(config.get("model")),
        "workload_name": _string(workload.get("name")),
        "result_dir": str(result_dir),
        "trial_id": trial_id,
        "request_rate": request_rate,
        "actual_status": actual_status,
        "actual_is_slo_violation": actual_is_slo,
        "comparable_static_window_slo": comparable,
        "uses_request_level_ttft_slo": uses_request_level_ttft,
        "ttft_slo_mode": ttft_slo_mode,
        "longbench_ttft_static_preset": longbench_preset,
        "ttft_slo_field": ttft_field,
        "ttft_slo_ms": ttft_slo_ms,
        "ttft_window_max_ms": ttft_window,
        "ttft_aggregate_ms": ttft_aggregate,
        "ttft_window_over_aggregate": _ratio(ttft_window, ttft_aggregate),
        "ttft_window_margin_ms": _delta(ttft_window, ttft_slo_ms),
        "ttft_aggregate_margin_ms": _delta(ttft_aggregate, ttft_slo_ms),
        "window_ttft_violates": window_ttft_violates,
        "aggregate_ttft_violates": aggregate_ttft_violates,
        "tpot_slo_field": tpot_field,
        "tpot_slo_ms": tpot_slo_ms,
        "tpot_window_max_ms": tpot_window,
        "tpot_aggregate_ms": tpot_aggregate,
        "tpot_window_over_aggregate": _ratio(tpot_window, tpot_aggregate),
        "tpot_window_margin_ms": _delta(tpot_window, tpot_slo_ms),
        "tpot_aggregate_margin_ms": _delta(tpot_aggregate, tpot_slo_ms),
        "window_tpot_violates": window_tpot_violates,
        "aggregate_tpot_violates": aggregate_tpot_violates,
        "window_slo_violation": window_slo_violation,
        "aggregate_slo_violation": aggregate_slo_violation,
        "window_only_slo_violation": comparable and window_slo_violation and not aggregate_slo_violation,
        "aggregate_only_slo_violation": comparable and aggregate_slo_violation and not window_slo_violation,
        "status_differs_from_aggregate": status_differs,
        "active_eval_windows": _float_or_none(key_metrics.get("active_eval_windows")),
        "eval_windows": _float_or_none(key_metrics.get("eval_windows")),
        "summary_json": str(summary_path),
        "analysis_json": str(analysis_path),
    }


def _aggregate_percentile(metrics: Mapping[str, Any], field: str) -> float | None:
    if field in TTFT_PERCENTILES:
        return _percentile_from_pairs(metrics.get("percentiles_ttft_ms"), TTFT_PERCENTILES[field])
    if field in TPOT_PERCENTILES:
        return _percentile_from_pairs(metrics.get("percentiles_tpot_ms"), TPOT_PERCENTILES[field])
    return None


def _percentile_from_pairs(raw_pairs: Any, percentile: float) -> float | None:
    for pair in _list(raw_pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        raw_key, raw_value = pair
        key = _float_or_none(raw_key)
        value = _float_or_none(raw_value)
        if key is not None and value is not None and abs(key - percentile) < 1e-9:
            return value
    return None


def _summarize(run_roots: list[Path], rows: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    by_run: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        run_counts = by_run[str(row["run_id"])]
        run_counts["trial_count"] += 1
        if row["comparable_static_window_slo"]:
            run_counts["static_policy_trial_count"] += 1
        if row["actual_is_slo_violation"]:
            run_counts["actual_slo_violation_count"] += 1
        if row["aggregate_slo_violation"]:
            run_counts["aggregate_slo_violation_count"] += 1
        if row["window_only_slo_violation"]:
            run_counts["window_only_slo_violation_count"] += 1
        if row["aggregate_only_slo_violation"]:
            run_counts["aggregate_only_slo_violation_count"] += 1
        if row["status_differs_from_aggregate"]:
            run_counts["status_diff_count"] += 1

    return {
        "run_count": len(run_roots),
        "run_roots": [str(path) for path in run_roots],
        "trial_count": len(rows),
        "static_policy_trial_count": sum(1 for row in rows if row["comparable_static_window_slo"]),
        "request_level_ttft_trial_count": sum(1 for row in rows if row["uses_request_level_ttft_slo"]),
        "actual_slo_violation_count": sum(1 for row in rows if row["actual_is_slo_violation"]),
        "aggregate_slo_violation_count": sum(1 for row in rows if row["aggregate_slo_violation"]),
        "window_only_slo_violation_count": sum(1 for row in rows if row["window_only_slo_violation"]),
        "aggregate_only_slo_violation_count": sum(1 for row in rows if row["aggregate_only_slo_violation"]),
        "status_diff_count": sum(1 for row in rows if row["status_differs_from_aggregate"]),
        "warnings_count": len(warnings),
        "warnings": warnings,
        "by_run": {run_id: dict(counts) for run_id, counts in sorted(by_run.items())},
    }


def _write_markdown(path: Path, summary: Mapping[str, Any], rows: list[dict[str, Any]], print_top: int) -> None:
    ranked = sorted(
        rows,
        key=lambda row: max(
            _abs_or_zero(row.get("ttft_window_margin_ms")),
            _abs_or_zero(row.get("tpot_window_margin_ms")),
            _abs_or_zero(row.get("ttft_window_over_aggregate")),
            _abs_or_zero(row.get("tpot_window_over_aggregate")),
        ),
        reverse=True,
    )
    disagreement_rows = [
        row
        for row in ranked
        if row["window_only_slo_violation"]
        or row["aggregate_only_slo_violation"]
        or row["status_differs_from_aggregate"]
    ]

    lines = [
        "# MST Window SLO Audit",
        "",
        "Compares saved active-window max latency-percentile SLO inputs against aggregate trial percentiles.",
        "",
        "## Summary",
        "",
        f"- Run roots scanned: {summary['run_count']}",
        f"- Trials scanned: {summary['trial_count']}",
        f"- Static-window-comparable trials: {summary['static_policy_trial_count']}",
        f"- Request-level TTFT trials skipped for aggregate TTFT comparison: {summary['request_level_ttft_trial_count']}",
        f"- Actual SLO-violation trials: {summary['actual_slo_violation_count']}",
        f"- Aggregate-percentile SLO-violation trials: {summary['aggregate_slo_violation_count']}",
        f"- Window-only SLO violations: {summary['window_only_slo_violation_count']}",
        f"- Aggregate-only SLO violations: {summary['aggregate_only_slo_violation_count']}",
        f"- Actual status differs from aggregate-percentile SLO: {summary['status_diff_count']}",
        f"- Warnings: {summary['warnings_count']}",
        "",
        "## By Run",
        "",
        "| run | trials | static | actual_slo | aggregate_slo | window_only | aggregate_only | status_diff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_id, counts in sorted(_mapping(summary.get("by_run")).items()):
        lines.append(
            "| {run} | {trials} | {static} | {actual} | {aggregate} | {window_only} | {aggregate_only} | {diff} |".format(
                run=run_id,
                trials=counts.get("trial_count", 0),
                static=counts.get("static_policy_trial_count", 0),
                actual=counts.get("actual_slo_violation_count", 0),
                aggregate=counts.get("aggregate_slo_violation_count", 0),
                window_only=counts.get("window_only_slo_violation_count", 0),
                aggregate_only=counts.get("aggregate_only_slo_violation_count", 0),
                diff=counts.get("status_diff_count", 0),
            )
        )

    lines.extend(
        [
            "",
            f"## Top Disagreements ({min(print_top, len(disagreement_rows))})",
            "",
            "| run | experiment | trial | rps | status | ttft window/agg/slo | tpot window/agg/slo | flags |",
            "| --- | --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in disagreement_rows[:print_top]:
        flags = ",".join(
            name
            for name, enabled in (
                ("window_only", row["window_only_slo_violation"]),
                ("aggregate_only", row["aggregate_only_slo_violation"]),
                ("status_diff", row["status_differs_from_aggregate"]),
            )
            if enabled
        )
        lines.append(
            "| {run} | {experiment} | {trial} | {rps} | {status} | {ttft} | {tpot} | {flags} |".format(
                run=row["run_id"],
                experiment=row["experiment_id"],
                trial=row["trial_id"],
                rps=_fmt(row["request_rate"]),
                status=row["actual_status"],
                ttft=_metric_triplet(row, "ttft"),
                tpot=_metric_triplet(row, "tpot"),
                flags=flags,
            )
        )

    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in list(summary["warnings"])[:50]:
            lines.append(f"- {warning}")
        if len(summary["warnings"]) > 50:
            lines.append(f"- ... {len(summary['warnings']) - 50} more")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_triplet(row: Mapping[str, Any], prefix: str) -> str:
    return "{window}/{aggregate}/{slo}".format(
        window=_fmt(row.get(f"{prefix}_window_max_ms")),
        aggregate=_fmt(row.get(f"{prefix}_aggregate_ms")),
        slo=_fmt(row.get(f"{prefix}_slo_ms")),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_json(path: Path, warnings: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        warnings.append(f"{path}: missing")
    except json.JSONDecodeError as exc:
        warnings.append(f"{path}: JSON decode error at line {exc.lineno}: {exc.msg}")
    except OSError as exc:
        warnings.append(f"{path}: failed to read: {exc}")
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _violates(value: float | None, threshold: float | None) -> bool | None:
    if value is None or threshold is None:
        return None
    return value > threshold


def _any_true(*values: bool | None) -> bool:
    return any(value is True for value in values)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _delta(value: float | None, threshold: float | None) -> float | None:
    if value is None or threshold is None:
        return None
    return value - threshold


def _abs_or_zero(value: Any) -> float:
    numeric = _float_or_none(value)
    return abs(numeric) if numeric is not None else 0.0


def _fmt(value: Any) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return ""
    if abs(numeric) >= 100:
        return f"{numeric:.1f}"
    if abs(numeric) >= 10:
        return f"{numeric:.2f}"
    return f"{numeric:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
