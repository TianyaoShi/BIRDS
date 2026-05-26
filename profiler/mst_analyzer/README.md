# MST Analyzer

This package reads a completed local orchestrator MST run and flags suspicious model results for review or rerun planning.

It is intended to sit after `local_orchestrator` / `llm_mst_finder` and before any follow-on workflows such as energy profiling.
Both analyzer and plot commands accept repeated orchestrator run roots. When multiple roots are provided, later roots override earlier succeeded rows with the same `(model, logical workload, endpoint, serving signature)` identity. The logical workload key normalizes rerun aliases such as `_mst_anomaly_rerun` and common `4k`/`8k` spellings, so targeted reruns replace the original row while TP1/TP2/TP4 serving configurations stay separate.

## What it does

- loads an orchestrator run from `summary.json` and `state.json`
- loads each succeeded job's `search_trace.json` and `final_report.json`
- extracts normalized MST result rows
- applies anomaly rules across comparable models
- assigns severity scores
- reports trace-only instability as diagnostics by default
- writes JSON and Markdown reports
- can emit a reusable size-vs-MST scatter plot with point annotations
- optionally emits a small rerun manifest for flagged models and controls

## Package layout

- `config.py` - analyzer threshold and suppression settings
- `extract.py` - orchestrator/result artifact loading and row normalization
- `rules.py` - anomaly families, comparability logic, severity scoring
- `reporting.py` - JSON/Markdown report generation and rerun manifest output
- `plotting.py` - reusable model-size scatter plotting helpers
- `cli.py` - command-line entrypoint

## CLI

Example:

```bash
PYTHONPATH=/path/to/BioLLM/profiler \
/path/to/venv/bin/python -m mst_analyzer.cli analyze \
  --orchestrator-run-root results/orchestrator/single-gpu-model-loop-run-sharegpt-000 \
  --output-dir results/analysis/single-gpu-model-loop-run-sharegpt-000
```

Aggregate a main run plus rerun artifacts by listing the main root first and the rerun root last:

```bash
PYTHONPATH=/path/to/BioLLM/profiler \
/path/to/venv/bin/python -m mst_analyzer.cli analyze \
  --orchestrator-run-root results/orchestrator/single-gpu-model-loop-run-sharegpt-000 \
  --orchestrator-run-root results/orchestrator/chat-anomaly-rerun-000 \
  --output-dir results/analysis/single-gpu-model-loop-run-sharegpt-000-with-rerun
```

With custom settings and explicit rerun-manifest generation:

```bash
PYTHONPATH=/path/to/BioLLM/profiler \
/path/to/venv/bin/python -m mst_analyzer.cli analyze \
  --orchestrator-run-root results/orchestrator/single-gpu-model-loop-run-sharegpt-000 \
  --output-dir results/analysis/single-gpu-model-loop-run-sharegpt-000-tuned \
  --emit-rerun-manifest \
  --settings-yaml profiler/mst_analyzer/settings_template.yaml
```

Plot the model-size vs MST scatter from either the analyzer rows JSON or a completed orchestrator run:

```bash
PYTHONPATH=/path/to/BioLLM/profiler \
/path/to/venv/bin/python -m mst_analyzer.cli plot \
  --mst-rows-json results/analysis/single-gpu-model-loop-run-sharegpt-000/mst_rows.json \
  --output-path results/analysis/single-gpu-model-loop-run-sharegpt-000/model_size_vs_mst.png
```

## Outputs

Each analysis run writes:

```text
<output-dir>/
  mst_rows.json
  mst_anomaly_report.json
  mst_anomaly_report.md
  suggested_rerun_manifest.yaml        # only with --emit-rerun-manifest
  <workload>_mst_anomaly_rerun.yaml    # only with --emit-rerun-manifest
```

`suggested_rerun_manifest.yaml` is only written when there are selected rerun targets.
By default, rerun manifests include every selected actionable target. Use `--max-rerun-models` only when you explicitly want to cap the manifest size.

## Settings

The analyzer supports threshold and suppression overrides from YAML, but the defaults should be treated as the calibrated interface for normal use. Avoid changing scoring internals to make a specific run agree with expectations.

The template file is:

- [settings_template.yaml](/path/to/BioLLM/profiler/mst_analyzer/settings_template.yaml)

Main settings groups:

- `outlier_bands`
- `larger_model_*`
- `same_family_*`
- `trace_instability_*`
- `include_trace_only_findings`
- `suppressions`

Supported suppressions:

- `disable_families`
- `suppress_trace_instability_below_rps`
- `suppress_contextual_only_findings`
- `suppress_quantized_bucket_verdicts`
- `suppress_moe_bucket_verdicts`

Merge-time filters:

```bash
--exclude-models <model> ...
--exclude-experiment-ids <experiment-id> ...
--min-model-size-b 3.0
```

These filters are applied while merging orchestrator roots, before anomaly rules
run. Use `--min-model-size-b 3.0` to keep stale `<3B` experiments from old run
state out of code, reasoning, and LongBench H100 analyses.

Valid family names for `disable_families`:

- `within_size_outlier`
- `larger_model_inversion`
- `same_family_non_monotonicity`
- `search_rate_cap_reached`
- `missing_confirmed_mst_rate`
- `trace_instability_suspect`
- `slo_driven_disagreement`

## Current comparison rules

Primary comparisons require matching:

- workload
- hardware
- endpoint type
- `max_num_seqs`
- `max_num_batched_tokens`
- `max_model_len`
- tensor parallel size
- dtype / quantization

When the same model appears multiple times with different tensor parallel sizes, each serving configuration is treated as a distinct result. Bucket summaries, anomaly comparators, Markdown tables, JSON report entries, rerun manifests, and plot annotations include the TP/GPU serving label so TP1, TP2, and TP4 MSTs are not visually collapsed under one model name.

SLO mismatches are still allowed for contextual comparisons, but they are penalized in severity and labeled accordingly.

Quantized and MoE models are suppressed from bucket summaries by default unless you relax those suppressions in settings. They are not globally removed from all comparison rules; if a comparison is otherwise compatible, the report can still use it with quantization/MoE metadata visible in the extracted rows.

Trace-only instability is reported under `trace_diagnostics` and in the Markdown diagnostics section by default. Set `include_trace_only_findings: true` only when you intentionally want noisy or low-confidence searches to count as anomaly candidates.

Terminal search outcomes are scored as blocking anomalies by default:
`search_rate_cap_reached` means the configured upper cap was too low to bound the
MST, and `missing_confirmed_mst_rate` means no final stable MST exists for
downstream energy profiling.

MST rates and bottleneck labels come from the saved MST finder artifacts. In
particular, `slo_limited` can be driven by an active-window SLO crossing in
`windows.csv`/`analysis.json` even when the aggregate trial percentiles in
`summary.json` are below the SLO. This analyzer preserves those labels for
completed experiments; review the high-bound and confirmation trial artifacts
before treating a near-threshold SLO label as a hard capacity limit.

## Development notes

- model-size inference reuses shared logic from `local_orchestrator.planning`
- the active settings are written into `mst_anomaly_report.json` under `analysis_config`
- the Markdown report is intended for quick review; use the JSON report when downstream tooling needs structured data
