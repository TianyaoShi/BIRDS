from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Sequence

from .models import AnomalyCandidate, BucketSummary, ComparatorEvidence, MSTRow


@dataclass(slots=True)
class _FindingHit:
    family: str
    weight: int
    summary: str
    reasons: list[str]
    comparators: list[ComparatorEvidence]


def analyze_rows(rows: Sequence[MSTRow]) -> tuple[list[AnomalyCandidate], list[BucketSummary]]:
    buckets = build_bucket_summaries(rows)
    anomalies: list[AnomalyCandidate] = []
    for row in sorted(rows, key=lambda item: (item.model, item.experiment_id)):
        hits: list[_FindingHit] = []

        within_size_hit = _within_size_outlier(row=row, rows=rows)
        if within_size_hit is not None:
            hits.append(within_size_hit)

        inversion_hit = _larger_model_inversion(row=row, rows=rows)
        if inversion_hit is not None:
            hits.append(inversion_hit)

        same_family_hit = _same_family_non_monotonicity(row=row, rows=rows)
        if same_family_hit is not None:
            hits.append(same_family_hit)

        trace_hit = _trace_instability_hit(row)
        if trace_hit is not None:
            hits.append(trace_hit)

        slo_hit = _slo_driven_disagreement(row=row, rows=rows, existing_hits=hits)
        if slo_hit is not None:
            hits.append(slo_hit)

        anomaly = _build_anomaly(row=row, hits=hits)
        if anomaly is not None:
            anomalies.append(anomaly)

    anomalies.sort(key=lambda item: (-item.severity_score, item.model, item.experiment_id))
    return anomalies, buckets


def build_bucket_summaries(rows: Sequence[MSTRow]) -> list[BucketSummary]:
    grouped: dict[tuple[str, str], list[MSTRow]] = {}
    for row in rows:
        if row.size_bucket is None or row.mst_rps is None or row.is_quantized or row.is_moe:
            continue
        grouped.setdefault((row.size_bucket, _scope_label(row)), []).append(row)

    summaries: list[BucketSummary] = []
    for (bucket_name, comparable_group), members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        mst_values = [member.mst_rps for member in members if member.mst_rps is not None]
        if len(mst_values) < 2:
            continue
        token_values = [
            member.confirmation_total_token_throughput
            for member in members
            if member.confirmation_total_token_throughput is not None
        ]
        summaries.append(
            BucketSummary(
                bucket_name=bucket_name,
                comparable_group=comparable_group,
                model_count=len(members),
                median_mst_rps=float(median(mst_values)),
                median_total_token_throughput=(None if not token_values else float(median(token_values))),
                models=tuple(member.model for member in sorted(members, key=lambda item: item.model)),
            )
        )
    return summaries


def _within_size_outlier(*, row: MSTRow, rows: Sequence[MSTRow]) -> _FindingHit | None:
    if not _eligible_for_primary_comparisons(row) or row.size_bucket is None or row.mst_rps is None:
        return None

    peer_group, label = _peer_group_for_bucket(row=row, rows=rows)
    if len(peer_group) < 2:
        return None
    bucket_median = float(median(member.mst_rps for member in peer_group if member.mst_rps is not None))
    ratio_threshold, abs_threshold = _outlier_thresholds(row.mst_rps)
    ratio = bucket_median / row.mst_rps if row.mst_rps > 0 else float("inf")
    absolute_delta = bucket_median - row.mst_rps
    if ratio < ratio_threshold or absolute_delta < abs_threshold:
        return None

    peers = [
        member
        for member in sorted(peer_group, key=lambda item: (item.mst_rps or 0.0), reverse=True)
        if member.experiment_id != row.experiment_id and member.mst_rps is not None
    ]
    if not peers:
        return None
    comparators = [
        _comparator_from_rows(
            relation="same_size_peer",
            row=row,
            comparator=peer,
            comparison_label=label,
            reason=(
                f"{row.size_bucket} bucket peer; bucket median is {bucket_median:.2f} rps "
                f"across {len(peer_group)} comparable models"
            ),
        )
        for peer in peers[:3]
    ]
    return _FindingHit(
        family="within_size_outlier",
        weight=30,
        summary=(
            f"{row.model} MST {row.mst_rps:.2f} rps is below the {row.size_bucket} bucket median "
            f"{bucket_median:.2f} rps by {absolute_delta:.2f} rps ({ratio:.2f}x lower)."
        ),
        reasons=[
            f"comparable group: {_scope_label(row)}",
            f"bucket median: {bucket_median:.2f} rps",
            f"rate ratio versus bucket median: {ratio:.2f}x lower",
            f"absolute delta versus bucket median: {absolute_delta:.2f} rps",
        ],
        comparators=comparators,
    )


def _larger_model_inversion(*, row: MSTRow, rows: Sequence[MSTRow]) -> _FindingHit | None:
    if not _eligible_for_primary_comparisons(row) or row.model_size_b is None or row.mst_rps is None:
        return None

    comparator, label = _pick_nearest_larger_comparator(row=row, rows=rows, same_family_only=False)
    if comparator is None or comparator.mst_rps is None or comparator.model_size_b is None:
        return None

    absolute_delta = abs(row.mst_rps - comparator.mst_rps)
    if row.model_size_b * 1.5 > comparator.model_size_b:
        return None
    if row.mst_rps > comparator.mst_rps * 1.15:
        return None
    if not ((row.mst_rps > 2.0 and comparator.mst_rps > 2.0) or absolute_delta > 1.0):
        return None

    comparator_evidence = _comparator_from_rows(
        relation="larger_model",
        row=row,
        comparator=comparator,
        comparison_label=label,
        reason="nearest larger-model reference under comparable serving settings",
    )
    relation_text = "close to" if row.mst_rps >= comparator.mst_rps else "below"
    return _FindingHit(
        family="larger_model_inversion",
        weight=25,
        summary=(
            f"{row.model} MST {row.mst_rps:.2f} rps is {relation_text} larger-model "
            f"{comparator.model} at {comparator.mst_rps:.2f} rps."
        ),
        reasons=[
            f"smaller-to-larger size ratio: {comparator.model_size_b / row.model_size_b:.2f}x",
            f"absolute delta: {absolute_delta:.2f} rps",
            f"comparison label: {label}",
        ],
        comparators=[comparator_evidence],
    )


def _same_family_non_monotonicity(*, row: MSTRow, rows: Sequence[MSTRow]) -> _FindingHit | None:
    if not _eligible_for_primary_comparisons(row) or row.model_size_b is None or row.mst_rps is None:
        return None

    comparator, label = _pick_nearest_larger_comparator(row=row, rows=rows, same_family_only=True)
    if comparator is None or comparator.mst_rps is None:
        return None
    if row.mst_rps > 2.0 and comparator.mst_rps > 2.0 and row.mst_rps <= comparator.mst_rps * 0.8:
        comparator_evidence = _comparator_from_rows(
            relation="same_family_larger",
            row=row,
            comparator=comparator,
            comparison_label=label,
            reason="nearest larger same-family model under comparable serving settings",
        )
        return _FindingHit(
            family="same_family_non_monotonicity",
            weight=20,
            summary=(
                f"{row.model} underperforms larger same-family {comparator.model}: "
                f"{row.mst_rps:.2f} rps vs {comparator.mst_rps:.2f} rps."
            ),
            reasons=[
                f"same-family comparator: {comparator.model}",
                f"smaller model is {row.mst_rps / comparator.mst_rps:.2f}x of the larger model's MST",
                f"comparison label: {label}",
            ],
            comparators=[comparator_evidence],
        )
    return None


def _trace_instability_hit(row: MSTRow) -> _FindingHit | None:
    instability = row.trace_instability
    strong_signal = (
        bool(instability.conflicting_rate_labels)
        or instability.majority_confirmation_used
        or instability.uncertain_retry_count >= 2
        or instability.suspect_termination_reason
    )
    if not strong_signal and not (
        instability.low_confidence and instability.uncertain_retry_count >= 1
    ):
        return None

    reasons = []
    if instability.conflicting_rate_labels:
        reasons.append(
            "same request rate saw conflicting stability outcomes at "
            + ", ".join(instability.conflicting_rate_labels)
            + " rps"
        )
    if instability.majority_confirmation_used:
        reasons.append("confirmation required the second-pass majority check")
    if instability.uncertain_retry_count:
        reasons.append(f"uncertain retries near the final bound: {instability.uncertain_retry_count}")
    if instability.suspect_termination_reason:
        reasons.append(f"termination reason: {row.termination_reason}")
    if instability.low_confidence:
        reasons.append("final search confidence is low")

    return _FindingHit(
        family="trace_instability_suspect",
        weight=15,
        summary=f"{row.model} shows conflicting or low-confidence trace evidence near the selected MST.",
        reasons=reasons,
        comparators=[],
    )


def _slo_driven_disagreement(
    *,
    row: MSTRow,
    rows: Sequence[MSTRow],
    existing_hits: Sequence[_FindingHit],
) -> _FindingHit | None:
    if existing_hits or row.mst_rps is None or row.model_size_b is None:
        return None
    contextual_candidates = [
        candidate
        for candidate in rows
        if candidate.experiment_id != row.experiment_id
        and _same_scope(row, candidate)
        and not _same_slo(row, candidate)
        and candidate.mst_rps is not None
        and candidate.model_size_b is not None
    ]
    if not contextual_candidates or not row.has_slo_signal:
        return None

    same_bucket = [candidate for candidate in contextual_candidates if candidate.size_bucket == row.size_bucket]
    larger = [
        candidate
        for candidate in contextual_candidates
        if candidate.model_size_b >= row.model_size_b * 1.5
        and row.mst_rps <= candidate.mst_rps * 1.15
    ]
    comparator = None
    relation = "contextual_peer"
    if same_bucket:
        comparator = sorted(same_bucket, key=lambda item: abs((item.mst_rps or 0.0) - row.mst_rps))[0]
    elif larger:
        comparator = sorted(larger, key=lambda item: item.model_size_b or 0.0)[0]
        relation = "larger_model"
    if comparator is None or comparator.mst_rps is None:
        return None

    return _FindingHit(
        family="slo_driven_disagreement",
        weight=0,
        summary=(
            f"{row.model} looks unusual against {comparator.model}, but the comparison is SLO-mismatched "
            "and should stay contextual."
        ),
        reasons=[
            "apparent disagreement is only present under different TTFT/TPOT SLO settings",
            f"row has SLO-related evidence and comparator uses a different SLO policy: {comparator.model}",
        ],
        comparators=[
            _comparator_from_rows(
                relation=relation,
                row=row,
                comparator=comparator,
                comparison_label="contextual",
                reason="different SLO policy; contextual reference only",
            )
        ],
    )


def _build_anomaly(*, row: MSTRow, hits: Sequence[_FindingHit]) -> AnomalyCandidate | None:
    if not hits:
        return None

    total_score = sum(hit.weight for hit in hits)
    if row.confidence == "low":
        total_score += 10

    all_rates = [row.mst_rps]
    variant_mismatch = False
    slo_mismatch = False
    comparators: list[ComparatorEvidence] = []
    seen_keys: set[tuple[str, str]] = set()
    reasons: list[str] = []
    evidence_paths = [row.search_trace_path]
    if row.final_report_json_path is not None:
        evidence_paths.append(row.final_report_json_path)

    for hit in hits:
        reasons.extend(hit.reasons)
        for comparator in hit.comparators:
            key = (comparator.experiment_id, "")
            if key not in seen_keys:
                comparators.append(comparator)
                seen_keys.add(key)
                all_rates.append(comparator.mst_rps)
                variant_mismatch = variant_mismatch or comparator.variant_mismatch
                slo_mismatch = slo_mismatch or not comparator.same_slo
                evidence_paths.append(comparator.search_trace_path)
                if comparator.confirmation_summary_json is not None:
                    evidence_paths.append(comparator.confirmation_summary_json)
                if comparator.high_bound_summary_json is not None:
                    evidence_paths.append(comparator.high_bound_summary_json)

    if row.confirmation_trial is not None and row.confirmation_trial.summary_json is not None:
        evidence_paths.append(row.confirmation_trial.summary_json)
    if row.high_bound_trial is not None and row.high_bound_trial.summary_json is not None:
        evidence_paths.append(row.high_bound_trial.summary_json)

    if all(rate is not None and rate < 1.0 for rate in all_rates):
        total_score -= 20
    if slo_mismatch:
        total_score -= 10
    if variant_mismatch:
        total_score -= 10

    total_score = max(0, min(100, int(round(total_score))))
    severity = "high" if total_score >= 60 else "medium" if total_score >= 35 else "low"

    primary_summary = max(hits, key=lambda item: item.weight).summary
    if any(hit.family == "trace_instability_suspect" for hit in hits) and not primary_summary.endswith("."):
        primary_summary += "."
    if any(hit.family == "trace_instability_suspect" for hit in hits) and "trace evidence" not in primary_summary:
        primary_summary += " Trace evidence is conflicted near the selected rate."

    control_models = [comparator.model for comparator in comparators]
    if control_models:
        suggested_action = (
            f"rerun {row.model} with controls {', '.join(control_models[:3])} "
            "using longer trial and confirmation durations"
        )
    else:
        suggested_action = f"rerun {row.model} with longer trial and confirmation durations"

    unique_paths = tuple(dict.fromkeys(evidence_paths))
    return AnomalyCandidate(
        experiment_id=row.experiment_id,
        model=row.model,
        mst_rps=float(row.mst_rps or 0.0),
        confidence=row.confidence,
        severity_score=total_score,
        severity=severity,
        families=tuple(hit.family for hit in hits),
        summary=primary_summary,
        reasons=tuple(reasons),
        comparators=tuple(comparators),
        confirmation_trial_id=(None if row.confirmation_trial is None else row.confirmation_trial.trial_id),
        high_bound_trial_id=(None if row.high_bound_trial is None else row.high_bound_trial.trial_id),
        suggested_action=suggested_action,
        search_trace_path=row.search_trace_path,
        final_report_json_path=row.final_report_json_path,
        evidence_paths=unique_paths,
    )


def _peer_group_for_bucket(*, row: MSTRow, rows: Sequence[MSTRow]) -> tuple[list[MSTRow], str]:
    direct = [
        candidate
        for candidate in rows
        if candidate.size_bucket == row.size_bucket
        and _eligible_for_primary_comparisons(candidate)
        and _same_scope(row, candidate)
        and _same_slo(row, candidate)
    ]
    if len(direct) >= 2:
        return direct, "direct"
    contextual = [
        candidate
        for candidate in rows
        if candidate.size_bucket == row.size_bucket
        and _eligible_for_primary_comparisons(candidate)
        and _same_scope(row, candidate)
    ]
    return contextual, "contextual"


def _pick_nearest_larger_comparator(
    *,
    row: MSTRow,
    rows: Sequence[MSTRow],
    same_family_only: bool,
) -> tuple[MSTRow | None, str]:
    direct = [
        candidate
        for candidate in rows
        if _eligible_for_primary_comparisons(candidate)
        and candidate.experiment_id != row.experiment_id
        and candidate.model_size_b is not None
        and row.model_size_b is not None
        and candidate.model_size_b > row.model_size_b
        and _same_scope(row, candidate)
        and _same_slo(row, candidate)
        and (not same_family_only or candidate.model_family == row.model_family)
    ]
    if direct:
        return sorted(
            direct,
            key=lambda item: (item.model_size_b or 0.0, -(item.mst_rps or 0.0)),
        )[0], "direct"

    contextual = [
        candidate
        for candidate in rows
        if _eligible_for_primary_comparisons(candidate)
        and candidate.experiment_id != row.experiment_id
        and candidate.model_size_b is not None
        and row.model_size_b is not None
        and candidate.model_size_b > row.model_size_b
        and _same_scope(row, candidate)
        and (not same_family_only or candidate.model_family == row.model_family)
    ]
    if contextual:
        return sorted(
            contextual,
            key=lambda item: (item.model_size_b or 0.0, -(item.mst_rps or 0.0)),
        )[0], "contextual"
    return None, "direct"


def _comparator_from_rows(
    *,
    relation: str,
    row: MSTRow,
    comparator: MSTRow,
    comparison_label: str,
    reason: str,
) -> ComparatorEvidence:
    assert row.mst_rps is not None
    assert comparator.mst_rps is not None
    return ComparatorEvidence(
        relation=relation,
        comparison_label=comparison_label,
        model=comparator.model,
        experiment_id=comparator.experiment_id,
        mst_rps=float(comparator.mst_rps),
        model_size_b=comparator.model_size_b,
        rate_ratio_vs_comparator=(row.mst_rps / comparator.mst_rps if comparator.mst_rps > 0 else 0.0),
        absolute_delta_rps=abs(row.mst_rps - comparator.mst_rps),
        same_slo=_same_slo(row, comparator),
        variant_mismatch=_variant_mismatch(row, comparator),
        confirmation_trial_id=(
            None if comparator.confirmation_trial is None else comparator.confirmation_trial.trial_id
        ),
        high_bound_trial_id=(None if comparator.high_bound_trial is None else comparator.high_bound_trial.trial_id),
        reason=reason,
        search_trace_path=comparator.search_trace_path,
        confirmation_summary_json=(
            None if comparator.confirmation_trial is None else comparator.confirmation_trial.summary_json
        ),
        high_bound_summary_json=(
            None if comparator.high_bound_trial is None else comparator.high_bound_trial.summary_json
        ),
    )


def _eligible_for_primary_comparisons(row: MSTRow) -> bool:
    return row.mst_rps is not None and row.model_size_b is not None and not row.is_quantized and not row.is_moe


def _same_scope(left: MSTRow, right: MSTRow) -> bool:
    return _scope_key(left) == _scope_key(right)


def _same_slo(left: MSTRow, right: MSTRow) -> bool:
    return (
        left.ttft_slo_ms,
        left.tpot_slo_ms,
        left.ttft_slo_field,
        left.tpot_slo_field,
    ) == (
        right.ttft_slo_ms,
        right.tpot_slo_ms,
        right.ttft_slo_field,
        right.tpot_slo_field,
    )


def _scope_key(row: MSTRow) -> tuple[object, ...]:
    return (
        row.workload_name,
        row.hardware,
        row.endpoint_type,
        row.max_num_seqs,
        row.max_num_batched_tokens,
        row.max_model_len,
        row.tensor_parallel_size,
        (row.dtype or "").lower(),
        (row.quantization or "").lower(),
    )


def _scope_label(row: MSTRow) -> str:
    return (
        f"{row.workload_name}; {row.hardware}; {row.endpoint_type}; "
        f"seqs={row.max_num_seqs}; batched={row.max_num_batched_tokens}; "
        f"max_model_len={row.max_model_len}; tp={row.tensor_parallel_size}"
    )


def _variant_mismatch(left: MSTRow, right: MSTRow) -> bool:
    variants = {left.model_variant, right.model_variant}
    return "thinking" in variants and len({variant for variant in variants if variant is not None}) > 1


def _outlier_thresholds(rate: float) -> tuple[float, float]:
    if rate >= 10.0:
        return 1.5, 5.0
    if rate >= 2.0:
        return 1.5, 1.0
    return 2.5, 1.0
