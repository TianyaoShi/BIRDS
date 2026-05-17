---
name: analyze-orchestrator-failures
description: Diagnose failed BioLLM/local_orchestrator/slurm_orchestrator/energy_profiler runs from run roots, summary.json/summary.md, state.json, manifests, Slurm logs, vLLM logs, MST logs, and profiler artifacts. Use when asked to analyze failed experiments, explain orchestrator failures, classify root causes, decide which failures are ignorable, or propose rerun/config fixes without rerunning successful jobs.
---

# Analyze Orchestrator Failures

## Workflow

1. Start from the run root. Inspect `summary.md`, `summary.json`, `state.json`, the copied/source manifest, group plans, and `logs/`. Prefer `rg`, `jq`, `sed`, and targeted file reads. Do not submit or cancel jobs unless the user explicitly asks.
2. Build a failed-job inventory: `experiment_id`/`job_id`, model, workload, tensor parallelism, GPU count, request rate, status, artifact paths, and the most specific stderr/stdout files.
3. Read logs from most specific to broadest:
   - MST jobs: `*.mst.stderr.log`, `final_report.*`, trial `summary.json`, `analysis.json`.
   - vLLM launch: `*.vllm.stderr.log`, Slurm array stderr, server stdout.
   - Energy jobs: profile stderr/stdout, job/repeat `summary.json`, `energy_summary.json`, `gpu_power.json`.
   - Scheduler issues: Slurm `.err`, array task ids, timeout/cancel messages, module load output.
4. Classify each failure by root cause and confidence. Use exact log snippets sparingly, with file paths and line numbers when possible.
5. Separate true workload/model failures from reporting/collection issues. If artifacts exist but status is failed, inspect finalization, summary/report writing, result sync, and stale `state.json`.
6. Check whether rerun is needed. Avoid rerunning succeeded experiments; prefer resume state updates, narrowed rerun manifests, or targeted config patches.
7. Report findings as a concise table: failed job, root cause, evidence, action. Include ignored failures explicitly, such as excluded `<3B` models or intentionally unsupported models.

## Failure Heuristics

- Treat tensor-parallel variants as distinct experiments. Never merge MST rates or anomalies across `tp1`, `tp2`, `tp4`, and `tp8`.
- If a rerun or analyzer output mixes multiple roots, verify workload identity and manifest expansion before comparing rows.
- `max_request_rate_limited` means the search hit the configured cap, not necessarily instability. Recommend raising caps or jump-starting open-loop trials when appropriate.
- `no_confirmed_stable_open_loop_rate` should be lifted to rerun recommendations when downstream energy planning needs a selected MST source.
- Launch OOM is usually config pressure: `max_model_len`, `max_num_seqs`, `gpu_memory_utilization`, tensor parallelism, or allocator fragmentation. Apply `PYTORCH_ALLOC_CONF=expandable_segments:True` only to the cases that need it.
- Low free memory on a Slurm GPU at startup usually needs GPU id mapping and `CUDA_VISIBLE_DEVICES` evidence, not just model config changes.
- vLLM/Transformers errors for new model architectures are dependency compatibility issues; suggest a separate venv when upgrading would disturb existing runs.
- If final MST artifacts exist but the job remains running or failed, inspect process cleanup, report blocking, collect/finalize logic, and whether the orchestrator exits after final report creation.
- For repeated energy jobs, aggregate-level raw trace paths may be `null`; use `artifacts.repeats[*]` for `request_records_jsonl`, `server_metrics_jsonl`, and `windows_csv`.
- If resume fails after manifest edits, inspect stale `state.json` and group plan task mappings before recommending fresh submission.

## Useful Checks

Use targeted commands like:

```bash
rg -n "ERROR|Traceback|OutOfMemory|timeout|KeyboardInterrupt|ValueError|failed|cancel|CANCELLED|TIMEOUT" <run-root>/logs
jq '.counts, .jobs[] | select(.status!="succeeded")' <run-root>/summary.json
jq '.jobs[] | {id:(.experiment_id // .job_id), status, last_error, artifacts}' <run-root>/state.json
```

For a longer taxonomy and artifact map, read `references/orchestrator-failure-taxonomy.md`.
