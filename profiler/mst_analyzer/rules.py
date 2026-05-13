from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Sequence

from .config import AnalyzerSettings
from .models import AnomalyCandidate, BucketSummary, ComparatorEvidence, MSTRow, TraceDiagnostic


@dataclass(slots=True)
class _FindingHit:
    family: str
    weight: int
    summary: str
    reasons: list[str]
    comparators: list[ComparatorEvidence]


def analyze_rows(
    rows: Sequence[MSTRow],
    *,
    settings: AnalyzerSettings | None = None,
) -> tuple[list[AnomalyCandidate], list[BucketSummary]]:
    anomalies, buckets, _ = analyze_rows_with_diagnostics(rows, settings=settings)
    return anomalies, buckets


def analyze_rows_with_diagnostics(
    rows: Sequence[MSTRow],
    *,
    settings: AnalyzerSettings | None = None,
) -> tuple[list[AnomalyCandidate], list[BucketSummary], list[TraceDiagnostic]]:
    resolved_settings = settings or AnalyzerSettings()
    buckets = build_bucket_summaries(rows, settings=resolved_settings)
    anomalies: list[AnomalyCandidate] = []
    trace_diagnostics: list[TraceDiagnostic] = []
    for row in sorted(rows, key=lambda item: (item.model, item.experiment_id)):
        hits: list[_FindingHit] = []

        within_size_hit = _within_size_outlier(row=row, rows=rows, settings=resolved_settings)
        if within_size_hit is not None:
            hits.append(within_size_hit)

        inversion_hit = _larger_model_inversion(row=row, rows=rows, settings=resolved_settings)
        if inversion_hit is not None:
            hits.append(inversion_hit)

        same_family_hit = _same_family_non_monotonicity(row=row, rows=rows, settings=resolved_settings)
        if same_family_hit is not None:
            hits.append(same_family_hit)

        trace_hit = _trace_instability_hit(row, settings=resolved_settings)
        if trace_hit is not None:
            hits.append(trace_hit)

        slo_hit = _slo_driven_disagreement(
            row=row,
            rows=rows,
            existing_hits=hits,
            settings=resolved_settings,
        )
        if slo_hit is not None:
            hits.append(slo_hit)

        if (
            hits
            and not resolved_settings.include_trace_only_findings
            and all(hit.family == "trace_instability_suspect" for hit in hits)
            and not _promote_trace_only_termination(row.termination_reason)
        ):
            trace_diagnostics.append(_build_trace_diagnostic(row=row, hit=hits[0]))
            continue

        anomaly = _build_anomaly(row=row, hits=hits, settings=resolved_settings)
        if anomaly is not None:
            anomalies.append(anomaly)

    anomalies.sort(key=lambda item: (-item.severity_score, item.model, item.experiment_id))
    trace_diagnostics.sort(key=lambda item: (-(item.mst_rps or 0.0), item.model, item.experiment_id))
    return anomalies, buckets, trace_diagnostics


def build_bucket_summaries(
    rows: Sequence[MSTRow],
    *,
    settings: AnalyzerSettings | None = None,
) -> list[BucketSummary]:
    resolved_settings = settings or AnalyzerSettings()
    grouped: dict[tuple[str, str], list[MSTRow]] = {}
    for row in rows:
        if row.size_bucket is None or row.mst_rps is None:
            continue
        if resolved_settings.suppressions.suppress_quantized_bucket_verdicts and row.is_quantized:
            continue
        if resolved_settings.suppressions.suppress_moe_bucket_verdicts and row.is_moe:
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
                member_labels=tuple(
                    _model_serving_label(member)
                    for member in sorted(members, key=lambda item: (item.model, item.experiment_id))
                ),
            )
        )
    return summaries


def _within_size_outlier(
    *,
    row: MSTRow,
    rows: Sequence[MSTRow],
    settings: AnalyzerSettings,
) -> _FindingHit | None:
    if _family_disabled("within_size_outlier", settings=settings):
        return None
    if not _eligible_for_primary_comparisons(row, settings=settings) or row.size_bucket is None or row.mst_rps is None:
        return None

    peer_group, label = _peer_group_for_bucket(row=row, rows=rows, settings=settings)
    if len(peer_group) < 2:
        return None
    bucket_median = float(median(member.mst_rps for member in peer_group if member.mst_rps is not None))
    ratio_threshold, abs_threshold = _outlier_thresholds(row.mst_rps, settings=settings)
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
        weight=settings.severity_weight_within_size_outlier,
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


def _larger_model_inversion(
    *,
    row: MSTRow,
    rows: Sequence[MSTRow],
    settings: AnalyzerSettings,
) -> _FindingHit | None:
    if _family_disabled("larger_model_inversion", settings=settings):
        return None
    if not _eligible_for_primary_comparisons(row, settings=settings) or row.model_size_b is None or row.mst_rps is None:
        return None

    comparator, label = _pick_nearest_larger_comparator(
        row=row,
        rows=rows,
        same_family_only=False,
        settings=settings,
    )
    if comparator is None or comparator.mst_rps is None or comparator.model_size_b is None:
        return None

    absolute_delta = abs(row.mst_rps - comparator.mst_rps)
    relative_rate = row.mst_rps / comparator.mst_rps if comparator.mst_rps > 0 else 0.0
    if row.model_size_b * settings.larger_model_min_size_ratio > comparator.model_size_b:
        return None
    close_to_larger = (
        settings.larger_model_min_close_relative_rate
        <= relative_rate
        <= settings.larger_model_max_relative_rate
    )
    materially_slower_than_larger = relative_rate <= settings.larger_model_max_underperform_relative_rate
    if not close_to_larger and not materially_slower_than_larger:
        return None
    if not (
        (
            row.mst_rps > settings.larger_model_min_rate_for_relative_compare
            and comparator.mst_rps > settings.larger_model_min_rate_for_relative_compare
        )
        or absolute_delta > settings.larger_model_min_absolute_delta_rps
    ):
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
        weight=settings.severity_weight_larger_model_inversion,
        summary=(
            f"{row.model} MST {row.mst_rps:.2f} rps is {relation_text} larger-model "
            f"{comparator.model} at {comparator.mst_rps:.2f} rps."
        ),
        reasons=[
            f"smaller-to-larger size ratio: {comparator.model_size_b / row.model_size_b:.2f}x",
            f"absolute delta: {absolute_delta:.2f} rps",
            f"comparison label: {label}",
            f"relative rate versus larger model: {relative_rate:.2f}x",
        ],
        comparators=[comparator_evidence],
    )


def _same_family_non_monotonicity(
    *,
    row: MSTRow,
    rows: Sequence[MSTRow],
    settings: AnalyzerSettings,
) -> _FindingHit | None:
    if _family_disabled("same_family_non_monotonicity", settings=settings):
        return None
    if not _eligible_for_primary_comparisons(row, settings=settings) or row.model_size_b is None or row.mst_rps is None:
        return None

    comparator, label = _pick_nearest_larger_comparator(
        row=row,
        rows=rows,
        same_family_only=True,
        settings=settings,
    )
    if comparator is None or comparator.mst_rps is None:
        return None
    if (
        row.mst_rps > settings.same_family_min_rate
        and comparator.mst_rps > settings.same_family_min_rate
        and row.mst_rps <= comparator.mst_rps * settings.same_family_max_relative_rate
    ):
        comparator_evidence = _comparator_from_rows(
            relation="same_family_larger",
            row=row,
            comparator=comparator,
            comparison_label=label,
            reason="nearest larger same-family model under comparable serving settings",
        )
        return _FindingHit(
            family="same_family_non_monotonicity",
            weight=settings.severity_weight_same_family_non_monotonicity,
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


def _trace_instability_hit(row: MSTRow, *, settings: AnalyzerSettings) -> _FindingHit | None:
    if _family_disabled("trace_instability_suspect", settings=settings):
        return None
    if (
        settings.suppressions.suppress_trace_instability_below_rps is not None
        and row.mst_rps is not None
        and row.mst_rps < settings.suppressions.suppress_trace_instability_below_rps
    ):
        return None
    instability = row.trace_instability
    strong_signal = (
        bool(instability.conflicting_rate_labels)
        or instability.majority_confirmation_used
        or instability.uncertain_retry_count >= settings.trace_instability_min_uncertain_retries
        or instability.suspect_termination_reason
    )
    if not strong_signal and not (
        instability.low_confidence
        and instability.uncertain_retry_count
        >= settings.trace_instability_require_low_confidence_uncertain_retries
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
        weight=settings.severity_weight_trace_instability_suspect,
        summary=f"{row.model} shows conflicting or low-confidence trace evidence near the selected MST.",
        reasons=reasons,
        comparators=[],
    )


def _slo_driven_disagreement(
    *,
    row: MSTRow,
    rows: Sequence[MSTRow],
    existing_hits: Sequence[_FindingHit],
    settings: AnalyzerSettings,
) -> _FindingHit | None:
    if _family_disabled("slo_driven_disagreement", settings=settings):
        return None
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


def _build_anomaly(
    *,
    row: MSTRow,
    hits: Sequence[_FindingHit],
    settings: AnalyzerSettings,
) -> AnomalyCandidate | None:
    if not hits:
        return None

    total_score = sum(hit.weight for hit in hits)
    if row.confidence == "low":
        total_score += settings.severity_weight_low_confidence_result

    all_rates = [row.mst_rps]
    variant_mismatch = False
    slo_mismatch = False
    comparators: list[ComparatorEvidence] = []
    seen_keys: set[tuple[str, str]] = set()
    reasons: list[str] = []
    family_reasons: dict[str, tuple[str, ...]] = {}
    evidence_paths = [row.search_trace_path]
    if row.final_report_json_path is not None:
        evidence_paths.append(row.final_report_json_path)

    for hit in hits:
        reasons.extend(hit.reasons)
        family_reasons[hit.family] = tuple(hit.reasons)
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
        total_score -= settings.severity_penalty_all_below_one_rps
    if slo_mismatch:
        total_score -= settings.severity_penalty_slo_mismatch
    if variant_mismatch:
        total_score -= settings.severity_penalty_variant_mismatch

    if (
        settings.suppressions.suppress_contextual_only_findings
        and comparators
        and all(comparator.comparison_label == "contextual" for comparator in comparators)
    ):
        return None

    total_score = max(0, min(100, int(round(total_score))))
    severity = "high" if total_score >= 60 else "medium" if total_score >= 35 else "low"

    primary_summary = max(hits, key=lambda item: item.weight).summary
    if any(hit.family == "trace_instability_suspect" for hit in hits) and not primary_summary.endswith("."):
        primary_summary += "."
    if any(hit.family == "trace_instability_suspect" for hit in hits) and "trace evidence" not in primary_summary:
        primary_summary += " Trace evidence is conflicted near the selected rate."

    control_models = [comparator.model for comparator in comparators]
    if row.termination_reason == "max_request_rate_limited":
        suggested_action = f"rerun {row.model} with a higher max_request_rate cap"
    elif row.termination_reason == "no_confirmed_stable_open_loop_rate":
        suggested_action = f"rerun {row.model}; no stable open-loop MST rate was confirmed"
    elif control_models:
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
        serving_config_label=_serving_config_label(row),
        tensor_parallel_size=row.tensor_parallel_size,
        gpu_count=row.gpu_count,
        mst_rps=float(row.mst_rps or 0.0),
        confidence=row.confidence,
        severity_score=total_score,
        severity=severity,
        families=tuple(hit.family for hit in hits),
        summary=primary_summary,
        reasons=tuple(reasons),
        family_reasons=family_reasons,
        comparators=tuple(comparators),
        confirmation_trial_id=(None if row.confirmation_trial is None else row.confirmation_trial.trial_id),
        high_bound_trial_id=(None if row.high_bound_trial is None else row.high_bound_trial.trial_id),
        suggested_action=suggested_action,
        search_trace_path=row.search_trace_path,
        final_report_json_path=row.final_report_json_path,
        evidence_paths=unique_paths,
    )


def _promote_trace_only_termination(termination_reason: str | None) -> bool:
    return termination_reason in {
        "max_request_rate_limited",
        "no_confirmed_stable_open_loop_rate",
    }


def _build_trace_diagnostic(*, row: MSTRow, hit: _FindingHit) -> TraceDiagnostic:
    evidence_paths = [row.search_trace_path]
    if row.final_report_json_path is not None:
        evidence_paths.append(row.final_report_json_path)
    if row.confirmation_trial is not None and row.confirmation_trial.summary_json is not None:
        evidence_paths.append(row.confirmation_trial.summary_json)
    if row.high_bound_trial is not None and row.high_bound_trial.summary_json is not None:
        evidence_paths.append(row.high_bound_trial.summary_json)
    return TraceDiagnostic(
        experiment_id=row.experiment_id,
        model=row.model,
        serving_config_label=_serving_config_label(row),
        tensor_parallel_size=row.tensor_parallel_size,
        gpu_count=row.gpu_count,
        mst_rps=row.mst_rps,
        confidence=row.confidence,
        reasons=tuple(hit.reasons),
        confirmation_trial_id=(None if row.confirmation_trial is None else row.confirmation_trial.trial_id),
        high_bound_trial_id=(None if row.high_bound_trial is None else row.high_bound_trial.trial_id),
        search_trace_path=row.search_trace_path,
        evidence_paths=tuple(dict.fromkeys(evidence_paths)),
    )


def _peer_group_for_bucket(
    *,
    row: MSTRow,
    rows: Sequence[MSTRow],
    settings: AnalyzerSettings,
) -> tuple[list[MSTRow], str]:
    direct = [
        candidate
        for candidate in rows
        if candidate.size_bucket == row.size_bucket
        and _eligible_for_primary_comparisons(candidate, settings=settings)
        and _same_scope(row, candidate)
        and _same_slo(row, candidate)
    ]
    if len(direct) >= 2:
        return direct, "direct"
    contextual = [
        candidate
        for candidate in rows
        if candidate.size_bucket == row.size_bucket
        and _eligible_for_primary_comparisons(candidate, settings=settings)
        and _same_scope(row, candidate)
    ]
    return contextual, "contextual"


def _pick_nearest_larger_comparator(
    *,
    row: MSTRow,
    rows: Sequence[MSTRow],
    same_family_only: bool,
    settings: AnalyzerSettings,
) -> tuple[MSTRow | None, str]:
    direct = [
        candidate
        for candidate in rows
        if _eligible_for_primary_comparisons(candidate, settings=settings)
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
        if _eligible_for_primary_comparisons(candidate, settings=settings)
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
        serving_config_label=_serving_config_label(comparator),
        tensor_parallel_size=comparator.tensor_parallel_size,
        gpu_count=comparator.gpu_count,
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


def _eligible_for_primary_comparisons(row: MSTRow, *, settings: AnalyzerSettings) -> bool:
    if row.mst_rps is None or row.model_size_b is None:
        return False
    return True


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


def _model_serving_label(row: MSTRow) -> str:
    return f"{row.model} ({_serving_config_label(row)})"


def _serving_config_label(row: MSTRow) -> str:
    parts = [
        f"tp={row.tensor_parallel_size if row.tensor_parallel_size is not None else '-'}",
        f"gpus={row.gpu_count if row.gpu_count is not None else '-'}",
    ]
    if row.max_model_len is not None:
        parts.append(f"max_model_len={row.max_model_len}")
    if row.max_num_seqs is not None:
        parts.append(f"seqs={row.max_num_seqs:g}")
    if row.max_num_batched_tokens is not None:
        parts.append(f"batched={row.max_num_batched_tokens:g}")
    if row.dtype:
        parts.append(f"dtype={row.dtype}")
    if row.quantization:
        parts.append(f"quant={row.quantization}")
    return ", ".join(parts)


def _variant_mismatch(left: MSTRow, right: MSTRow) -> bool:
    variants = {left.model_variant, right.model_variant}
    return "thinking" in variants and len({variant for variant in variants if variant is not None}) > 1


def _outlier_thresholds(rate: float, *, settings: AnalyzerSettings) -> tuple[float, float]:
    for band in settings.outlier_bands:
        if rate < band.min_rate:
            continue
        if band.max_rate is not None and rate >= band.max_rate:
            continue
        return band.ratio_threshold, band.absolute_delta_rps
    raise RuntimeError(f"no outlier threshold band matched rate={rate}")


def _family_disabled(family: str, *, settings: AnalyzerSettings) -> bool:
    return family in settings.suppressions.disable_families
