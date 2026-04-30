# MST Analyzer

This package reads a completed local orchestrator MST run and flags suspicious model results for review or rerun planning.

It is intended to sit after `local_orchestrator` / `llm_mst_finder` and before any follow-on workflows such as energy profiling.

## What it does

- loads an orchestrator run from `summary.json` and `state.json`
- loads each succeeded job's `search_trace.json` and `final_report.json`
- extracts normalized MST result rows
- applies anomaly rules across comparable models
- assigns severity scores
- writes JSON and Markdown reports
- emits a small rerun manifest for flagged models and controls

## Package layout

- `config.py` - analyzer threshold and suppression settings
- `extract.py` - orchestrator/result artifact loading and row normalization
- `rules.py` - anomaly families, comparability logic, severity scoring
- `reporting.py` - JSON/Markdown report generation and rerun manifest output
- `cli.py` - command-line entrypoint

## CLI

Example:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m mst_analyzer.cli analyze \
  --orchestrator-run-root results/orchestrator/single-gpu-model-loop-run-sharegpt-000 \
  --output-dir results/analysis/single-gpu-model-loop-run-sharegpt-000
```

With custom settings:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m mst_analyzer.cli analyze \
  --orchestrator-run-root results/orchestrator/single-gpu-model-loop-run-sharegpt-000 \
  --output-dir results/analysis/single-gpu-model-loop-run-sharegpt-000-tuned \
  --settings-yaml profiler/mst_analyzer/settings_template.yaml
```

## Outputs

Each analysis run writes:

```text
<output-dir>/
  mst_rows.json
  mst_anomaly_report.json
  mst_anomaly_report.md
  suggested_rerun_manifest.yaml
  <workload>_mst_anomaly_rerun.yaml
```

`suggested_rerun_manifest.yaml` is only written when there are selected rerun targets.

## Settings

The analyzer now supports threshold and suppression overrides from YAML.

The template file is:

- [settings_template.yaml](/local/scratch/a/shi676/arr26/profiler/mst_analyzer/settings_template.yaml)

Main settings groups:

- `outlier_bands`
- `larger_model_*`
- `same_family_*`
- `trace_instability_*`
- `severity_weight_*`
- `severity_penalty_*`
- `suppressions`

Supported suppressions:

- `disable_families`
- `suppress_trace_instability_below_rps`
- `suppress_contextual_only_findings`
- `suppress_quantized_bucket_verdicts`
- `suppress_moe_bucket_verdicts`

Valid family names for `disable_families`:

- `within_size_outlier`
- `larger_model_inversion`
- `same_family_non_monotonicity`
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

SLO mismatches are still allowed for contextual comparisons, but they are penalized in severity and labeled accordingly.

Quantized and MoE models are suppressed from bucket verdicts by default unless you relax those suppressions in settings.

## Development notes

- model-size inference reuses shared logic from `local_orchestrator.planning`
- the active settings are written into `mst_anomaly_report.json` under `analysis_config`
- the Markdown report is intended for quick review; use the JSON report when downstream tooling needs structured data
