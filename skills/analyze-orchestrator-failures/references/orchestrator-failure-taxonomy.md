# Orchestrator Failure Taxonomy

## Artifact Map

- `summary.md`: fastest human overview; use it to list failed jobs and reported reasons.
- `summary.json`: canonical machine-readable job rows, rates, artifacts, and statuses.
- `state.json`: resume/collect state; inspect when status conflicts with artifacts.
- `plan.json`, group plan JSON, copied manifest: expansion, task ids, overrides, and Slurm grouping.
- `logs/*.err`: Slurm/module/scheduler failures and array task exits.
- `logs/*.vllm.stderr.log`: server launch, model load, CUDA/OOM, dependency errors.
- `logs/*.mst.stderr.log`: search loop, client failures, reporting, final report generation.
- Energy `jobs/*/{summary,energy_summary,gpu_power}.json`: profiler result and energy math.
- Energy `jobs/*/repeat_*/*`: raw repeat-level traces for repeated profiling jobs.

## Common Failure Classes

| Class | Evidence | Usual action |
| --- | --- | --- |
| Slurm/module setup | `module: unknown`, bad partition/QOS, missing CUDA | Patch Slurm config or module sequence. |
| venv/CUDA mismatch | vLLM/Torch CUDA import errors, wrong `cu12x` expectation | Use the matching uv venv or reinstall separate venv. |
| Unsupported architecture | Transformers/vLLM model config errors | Upgrade isolated venv or exclude until supported. |
| Launch OOM | `torch.OutOfMemoryError`, KV cache errors | Reduce `max_model_len`, `max_num_seqs`, utilization, or raise TP. |
| GPU contamination/mapping | startup free memory far below expectation | Check `SLURM_JOB_GPUS`, `CUDA_VISIBLE_DEVICES`, generated sbatch logging. |
| Timeout | Slurm `TIMEOUT`, server readiness timeout, long model load | Raise startup/job timeout only for affected models; compare artifact timestamps. |
| Client-limited search | client cannot issue enough requests | Add cooldown/retry or client capacity; do not treat as model MST. |
| Search cap reached | `max_request_rate_limited` / cap reached | Raise model-size cap or create targeted rerun with higher open-loop cap. |
| Trace instability | drift/SLO unstable across trials | Rerun if it prevents selected MST; otherwise report caveat. |
| Reporting/finalize issue | final artifacts exist but job marked failed | Patch collect/finalize/reporting; avoid rerunning completed GPU work. |
| Sync issue | local summaries exist, rsync failed | Fix publish filter/destination; rerun collect only. |
| Stale resume state | task mismatch, missing expanded id | Update state/plan coherently or create targeted rerun manifest. |

## Reporting Rules

- Prefer evidence over speculation. Include the file and exact matching phrase.
- Distinguish config fixes from code fixes and from ignorable exclusions.
- State when an issue only affects analysis/reporting and not GPU execution.
- When suggesting reruns, list only missing or invalid experiments and preserve successful results.
- For energy profiling, report both aggregate metrics and whether repeated raw traces are per-repeat only.
