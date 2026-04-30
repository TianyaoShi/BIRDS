# MST Result Anomaly Analysis

Status: Draft for review on 2026-04-29.

## Goal

Add a post-run analysis layer that reads an orchestrator MST run and flags suspicious model results for human review or targeted rerun. This should automate the kind of reasoning used for the ShareGPT L40 loop:

- Qwen3-0.6B looked unexpectedly low compared with Llama-3.2-1B.
- Gemma E4B and Qwen3-4B-Thinking looked unexpectedly low compared with Qwen3-4B-Instruct and similar to 8B models.
- Llama-2-13B and Qwen3-14B differed by about 2x, but both were below 1 req/s, so the practical significance was lower and should not be treated as the same severity.

The analyzer should not automatically declare a model wrong. It should produce ranked anomaly candidates with evidence, confidence, and suggested rerun plans.

## Proposed Package

Add a small package:

- `profiler/mst_analyzer/__init__.py`
- `profiler/mst_analyzer/models.py`
- `profiler/mst_analyzer/extract.py`
- `profiler/mst_analyzer/rules.py`
- `profiler/mst_analyzer/reporting.py`
- `profiler/mst_analyzer/cli.py`

This package consumes:

- `results/orchestrator/<run_id>/summary.json`
- `results/orchestrator/<run_id>/state.json`
- each job's `search_trace.json`
- each result dir's confirmation/high-bound `summary.json` and `analysis.json`

It should be usable before energy profiling. Energy profiling can later reuse its selected model sets.

## CLI Shape

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m mst_analyzer.cli analyze \
  --orchestrator-run-root results/orchestrator/single-gpu-model-loop-run-sharegpt-000 \
  --output-dir results/analysis/single-gpu-model-loop-run-sharegpt-000
```

Outputs:

```text
results/analysis/<run_id>/
  mst_anomaly_report.json
  mst_anomaly_report.md
  suggested_rerun_manifest.yaml
```

## Extracted Row Schema

For every succeeded job, extract one normalized row:

```json
{
  "experiment_id": "single-gpu-model-loop-...",
  "model": "Qwen/Qwen3-0.6B",
  "model_family": "qwen3",
  "model_size_b": 0.6,
  "hardware": "l40-48gb",
  "workload_name": "live_sharegpt_workload_context",
  "endpoint": "/v1/chat/completions",
  "mst_rps": 17.402,
  "termination_reason": "confirmed_stable",
  "bottleneck_class": "unknown",
  "confidence": "low",
  "server_signature_key": "...",
  "max_num_seqs": 1024,
  "max_num_batched_tokens": 8192,
  "ttft_slo_ms": 250,
  "tpot_slo_ms": 50,
  "confirmation_trial_id": "trial_020_openloop_r17_4021",
  "confirmation_successful_completion_rate": 17.46,
  "confirmation_total_token_throughput": 8726.7,
  "confirmation_generation_token_throughput": 5821.7,
  "confirmation_prompt_len_mean": 166.3,
  "confirmation_output_len_mean": 333.4,
  "high_bound_rate": 18.37,
  "high_bound_status": "unstable",
  "high_bound_reasons": ["..."],
  "result_dir": "results/mst/..."
}
```

Model size inference should reuse or share logic with local orchestrator probing where possible. Add explicit overrides for nonstandard names.

## Comparability Rules

Only compare rows when these fields are compatible:

- same workload
- same hardware
- same endpoint type
- same SLO policy class, unless the report explicitly labels the comparison as SLO-mismatched
- same or comparable serving knobs:
  - `max_num_seqs`
  - `max_num_batched_tokens`
  - `max_model_len`
  - tensor parallel size
  - dtype/quantization where known

If SLOs differ, the analyzer should not suppress the comparison, but it must explain that the comparison is lower confidence.

Example:

- Qwen3-0.6B and Llama-3.2-1B are directly comparable: same small-model SLOs and same `max_num_seqs/max_num_batched_tokens`.
- Gemma E4B and Qwen3-4B-Instruct are directly comparable: same mid-size SLOs and same serving knobs.
- Gemma E4B versus Qwen3-8B is useful as a larger-model reference, but SLOs differ, so it should be labeled as contextual rather than primary.

## Anomaly Families

### 1. Within-Size Outlier

Within a size bucket, flag a model when its MST is materially below peers.

Size buckets:

- tiny: `< 1B`
- small: `1B-2B`
- mid: `3B-5B`
- large: `7B-9B`
- xlarge: `13B-14B`

For each bucket, compute robust center:

- median `mst_rps`
- median `total_token_throughput` at confirmation

Flag when:

- row rate is below `bucket_median / ratio_threshold`, and
- absolute difference exceeds a scale-aware threshold.

Draft thresholds:

- `mst_rps >= 10`: ratio threshold `1.5x`, absolute delta `5 rps`
- `2 <= mst_rps < 10`: ratio threshold `1.5x`, absolute delta `1 rps`
- `mst_rps < 2`: ratio threshold `2.5x`, absolute delta `1 rps`

The low-rate rule prevents over-alerting on small absolute differences. A 2x difference between `0.3` and `0.6 rps` is much less actionable than a 2x difference between `8` and `16 rps`.

### 2. Larger-Model Inversion

Flag when a smaller model has MST comparable to or below a larger model under comparable settings.

Primary rule:

- model A has lower size than model B by at least `1.5x`
- `mst_A <= mst_B * 1.15`
- both `mst_A` and `mst_B` are above `2 rps`, or absolute difference is above `1 rps`

Examples:

- Gemma E4B at `5.64 rps` versus Qwen3-8B at `5.42 rps` should be contextual: similar rate to a larger model.
- Qwen3-4B-Thinking at `4.27 rps` versus Qwen3-8B at `5.42 rps` is suspicious if the workload is meant to compare pure serving behavior, but the model variant may have different generation behavior.

### 3. Same-Family Non-Monotonicity

Within the same model family, MST should usually decrease as model size increases, but not always monotonically because tokenization, MoE, quantization, SLOs, and architecture differ.

Flag when:

- a smaller model underperforms a larger model by more than `20%`, and
- both rates are above `2 rps`, and
- serving/SLO settings are comparable.

Examples:

- Qwen3-0.6B at `17.4 rps` is below Llama-1B but not below Qwen3-1.7B; this is more a cross-family peer anomaly than same-family non-monotonicity.
- Qwen3-4B-Thinking at `4.27 rps` versus Qwen3-8B at `5.42 rps` should be flagged as a same-family variant anomaly, with a note that "Thinking" may not be comparable to "Instruct".

### 4. Trace-Instability Suspect

Flag results where the final rate was selected after conflicting evidence:

- same rate was classified stable and unstable in different trials
- confirmation required majority pass
- many uncertain retries near the final bound
- result confidence is `low`
- termination was `max_request_rate_limited`, `no_confirmed_stable_open_loop_rate`, or `confirmation_inconclusive`

Examples:

- Qwen3-0.6B at `17.4 rps` had one unstable and one stable confirmation; that should increase rerun priority.
- Gemma E4B backed down from `8.46` after failed confirmation and had many uncertain trials between `5.99` and `7.05`; that should increase rerun priority.

### 5. SLO-Driven Disagreement

Flag results where apparent MST difference is mostly caused by different SLO policies rather than raw capacity.

Evidence:

- high-bound or confirmation failures are `slo_violation`
- TTFT/TPOT percentiles are near thresholds
- compared rows have different `ttft_slo_ms` or `tpot_slo_ms`

The report should phrase these as "not directly comparable" rather than "anomaly" when SLO mismatch dominates.

## Severity Scoring

Produce a score from 0 to 100:

- `+30` within-size outlier
- `+25` larger-model inversion
- `+20` same-family non-monotonicity
- `+15` trace-instability suspect
- `+10` low confidence result
- `-20` if all rates involved are below `1 rps`
- `-10` if SLO policies differ
- `-10` if model variants are known not directly comparable, e.g. Thinking vs Instruct

Severity labels:

- `high`: `>= 60`
- `medium`: `35-59`
- `low`: `< 35`

The exact weights should be tuned after a few runs.

## Report Format

Markdown report sections:

1. Summary table sorted by severity.
2. Model-size bucket table.
3. Larger-model inversion table.
4. Trace instability table.
5. Suggested reruns.

Each anomaly should include:

- model and MST
- comparator model(s)
- ratio and absolute difference
- whether SLO/serving settings are directly comparable
- confirmation trial id
- high-bound trial id if available
- short reason text
- paths to `search_trace.json` and relevant trial summaries

Example finding:

```text
Medium: google/gemma-4-E4B-it MST 5.64 rps is close to Qwen/Qwen3-8B MST 5.42 rps despite roughly half the nominal size.
Evidence: confirmation total token throughput 2700 tok/s, close to Qwen3-8B 2471 tok/s; multiple uncertain trials near 6-7 rps; final confidence low.
Suggested action: rerun Gemma E4B with Qwen3-4B-Instruct and Qwen3-8B controls using longer confirmation.
```

## Suggested Rerun Manifest

The analyzer should optionally emit a small manifest with only flagged models and comparator controls.

Selection rules:

- include flagged models
- include direct same-size comparator
- include nearest larger model comparator when the anomaly is a larger-model inversion
- include no more than 5-7 models by default
- copy the workload YAML to a distinct filename/name to avoid overwriting result dirs
- use longer trials/confirmation than the original run

Draft rerun search overrides:

```yaml
search:
  trial_min_duration_s: 180
  trial_max_duration_s: 300
  final_confirmation_duration_s: 300
```

## Implementation Phases

### Phase 1: Extractor

- Load orchestrator summary/state.
- Load each succeeded job's search trace.
- Extract normalized rows.
- Infer model size/family.
- Write `mst_rows.json`.

Focused tests:

- parses current orchestrator summary shape
- handles missing or failed jobs
- infers common model sizes including `0.6B`, `E4B`, `13B`, `14B`

### Phase 2: Rule Engine

- Implement comparability checks.
- Implement anomaly families.
- Implement severity scoring.
- Keep thresholds configurable.

Focused tests:

- flags Qwen3-0.6B vs Llama-1B-like fixture
- flags Gemma E4B vs Qwen3-4B/8B-like fixture
- does not over-alert for two sub-1-rps models with a 2x ratio
- labels SLO-mismatched comparisons as contextual

### Phase 3: Reporting and Rerun Manifest

- Write JSON and Markdown reports.
- Generate optional rerun manifest.
- Include trace/trial paths for manual inspection.

Focused tests:

- report includes evidence paths
- rerun manifest includes flagged models and controls only
- rerun workload copy uses distinct name/stem

## Open Questions

1. Should the analyzer treat "Thinking" variants as a separate family by default?
   - No if the output length is controled e.g. `from_dataset` or is simlar based on the search trace; yes if the generation behavior is very different and the report should explicitly note that they are not directly comparable to "Instruct" variants.

2. Should MoE or quantized models be excluded from size-bucket comparisons unless explicit model metadata is present?
  - Yes they should be treated separately. The analyzer can mention them in the report / analysis but should not give a verdict on whether they are anomalous without more detailed metadata.

3. Should the analyzer compare request-rate MST or token-throughput MST first?
   - Recommendation: flag on request-rate MST, but use token throughput as explanatory evidence.

4. Should sub-1-rps results be suppressed entirely or merely downgraded?
   - Recommendation: downgrade, because they can still matter for very expensive models/workloads.

