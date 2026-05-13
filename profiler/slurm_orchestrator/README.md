# Slurm Orchestrator

This package is the cluster-facing companion to `local_orchestrator`. It does not lease GPUs, allocate local slots, or schedule work across a single host. Instead, it reuses the already-expanded experiment jobs from `local_orchestrator.matrix.expand_manifest(...)` and submits them to Slurm.

## Commands

Plan a run and materialize the job payloads, per-job state files, and sbatch scripts:

```bash
PYTHONPATH=/path/to/BioLLM/profiler \
  /path/to/venv/bin/python -m slurm_orchestrator.cli plan \
  --manifest /path/to/manifest.yaml \
  --run-id my-slurm-run
```

Submit the materialized plan with `sbatch`:

```bash
PYTHONPATH=/path/to/BioLLM/profiler \
  /path/to/venv/bin/python -m slurm_orchestrator.cli submit \
  --manifest /path/to/manifest.yaml \
  --run-id my-slurm-run
```

Resume a prior run. This refreshes non-succeeded job payloads from the manifest recorded in the run plan and submits only those Slurm array indices:

```bash
PYTHONPATH=/path/to/BioLLM/profiler \
  /path/to/venv/bin/python -m slurm_orchestrator.cli resume \
  --run-root /path/to/results/orchestrator/my-slurm-run
```

Use `--force` to rerun all jobs, or `--include-experiment` / `--exclude-experiment` with shell-style experiment ID patterns to target a subset.

Collect per-job JSON state back into `summary.json` and `summary.md`:

```bash
PYTHONPATH=/path/to/BioLLM/profiler \
  /path/to/venv/bin/python -m slurm_orchestrator.cli collect \
  --run-root /path/to/results/orchestrator/my-slurm-run
```

By default, `collect` also publishes a compact, shareable subset to the shared
results tree at `/depot/yiding/data/BioLLM-results/results`. For a run root such
as `/scratch/.../results/orchestrator/my-slurm-run`, the published files keep
their paths relative to the local `results/` root, for example:

```text
/depot/yiding/data/BioLLM-results/results/orchestrator/my-slurm-run
/depot/yiding/data/BioLLM-results/results/mst/<model>/<workload>/<server>
/depot/yiding/data/BioLLM-results/results/analysis/my-slurm-run
```

The default publish set is intentionally small:

- `orchestrator/<run_id>/summary.json`
- `orchestrator/<run_id>/summary.md`
- MST `final_report.json` and `final_report.md` files referenced by the
  collected run state
- MST `summary.json`, `summary.md`, and `analysis.json` files below those MST
  result directories
- image/PDF plots below `plots/`
- compact files under `analysis/<run_id>/`, when that directory exists

Raw per-request and raw metric files such as `request_records.jsonl`,
`server_metrics.jsonl`, and `windows.csv` are not published by the default
run-scoped sync. The implementation uses `rsync -a --partial
--chmod=D755,F644 --files-from <generated-list>` so it avoids copying unrelated
logs, Slurm scripts, per-job state files, and large intermediate trial records.

Useful sync controls:

```bash
# Disable publishing for one collect.
PYTHONPATH=/path/to/BioLLM/profiler \
  /path/to/venv/bin/python -m slurm_orchestrator.cli collect \
  --run-root /path/to/results/orchestrator/my-slurm-run \
  --no-sync-results

# Publish to a different shared results root.
PYTHONPATH=/path/to/BioLLM/profiler \
  /path/to/venv/bin/python -m slurm_orchestrator.cli collect \
  --run-root /path/to/results/orchestrator/my-slurm-run \
  --sync-results-to /path/to/shared/results

# Full-tree mirror. This copies logs and raw intermediates, so use it only when
# the destination has enough quota.
PYTHONPATH=/path/to/BioLLM/profiler \
  /path/to/venv/bin/python -m slurm_orchestrator.cli collect \
  --run-root /path/to/results/orchestrator/my-slurm-run \
  --sync-results-scope all

# Full-tree mirror that only copies files missing from the destination.
PYTHONPATH=/path/to/BioLLM/profiler \
  /path/to/venv/bin/python -m slurm_orchestrator.cli collect \
  --run-root /path/to/results/orchestrator/my-slurm-run \
  --sync-results-scope all \
  --sync-results-existing missing
```

`--sync-results-scope run` is the default compact publish mode.
`--sync-results-scope all` mirrors the nearest parent directory named `results`
and includes logs and raw intermediates. `--sync-results-existing update` is the
default and updates changed files; `--sync-results-existing missing` adds
`rsync --ignore-existing` and does not overwrite files already present in the
shared tree.

The same controls can be set with environment variables:

- `SLURM_ORCHESTRATOR_SYNC_RESULTS_TO`
- `SLURM_ORCHESTRATOR_SYNC_RESULTS_SCOPE=run|all`
- `SLURM_ORCHESTRATOR_SYNC_RESULTS_EXISTING=update|missing`
- `SLURM_ORCHESTRATOR_SYNC_RESULTS_ROOT`
- `SLURM_ORCHESTRATOR_DISABLE_RESULT_SYNC=1`

The shared results root must be a real directory, not a symlink into private
scratch storage. If it is currently a symlink, replace it once:

```bash
rm /depot/yiding/data/BioLLM-results/results
mkdir -p /depot/yiding/data/BioLLM-results/results
```

Submit a reviewed `energy_profiler` plan as Slurm arrays:

```bash
PYTHONPATH=/path/to/BioLLM/profiler \
  /path/to/venv/bin/python -m slurm_orchestrator.cli energy-submit \
  --plan /path/to/experiments/energy/<plan_id>.yaml \
  --run-id my-energy-slurm-run
```

Collect completed energy task state back into `state.json`, `summary.json`, and
`summary.md`:

```bash
PYTHONPATH=/path/to/BioLLM/profiler \
  /path/to/venv/bin/python -m slurm_orchestrator.cli energy-collect \
  --run-root /path/to/results/energy/<plan_id>/my-energy-slurm-run
```

## Separation From The Local Scheduler

- `local_orchestrator` still owns strict manifest parsing, experiment expansion, selector overrides, resource probing, vLLM command rendering, and MST search/report command construction.
- `slurm_orchestrator` only materializes expanded jobs into Slurm array groups and per-job status files.
- `run.max_active_gpus` remains local-only concurrency control. The Slurm adapter does not interpret it as cluster capacity.

## Manifest Fields

The Slurm adapter reads the optional top-level `slurm` section:

```yaml
slurm:
  partition: ai
  account: yiding
  qos: preemptible
  time: 04:00:00
  modules:
    - modtree/gpu
    - cuda/12.6.0
  setup_commands:
    - source /path/to/venv/bin/activate
  python_executable: /path/to/venv/bin/python
  cpus_per_gpu: 14
  cpus_per_task: null
  array_concurrency_limit: 4
  base_port: 8000
```

`run.python_executable` is still honored. If both are set, `slurm.python_executable` wins for the Slurm helper commands and MST invocations.

`--cpus-per-task` defaults to `slurm.cpus_per_gpu * launch.gpu_count`.
`slurm.cpus_per_gpu` defaults to `14` for backward compatibility with the original cluster.
Set `slurm.cpus_per_gpu: 32` for Anvil-style 4xA100 nodes with 128 CPUs, or set
`slurm.cpus_per_task` to force a fixed CPU count for every Slurm group.

## Submission Model

- Jobs are grouped by `launch.gpu_count`.
- Each group is submitted as one Slurm array with a generated `#SBATCH --array=...` clause.
- Every array task loads one expanded job payload, starts one vLLM server on localhost, waits for `/v1/models`, runs MST search and report, then tears the server down.
- Multi-GPU jobs request `launch.gpu_count` GPUs on one node with `#SBATCH --gres=gpu:<count>`.
- CPU allocation follows `slurm.cpus_per_gpu` unless `slurm.cpus_per_task` is set, so the default 4-GPU job renders `#SBATCH --cpus-per-task=56`, while `cpus_per_gpu: 32` renders `#SBATCH --cpus-per-task=128`.

## State And Logs

The run root contains:

- `plan.json` - materialized run plan
- `groups/<group>.json` - array task payloads
- `scripts/<group>.sbatch.sh` - generated sbatch scripts
- `jobs/<experiment_id>.json` - per-job mutable state
- `logs/*.vllm.stdout.log`
- `logs/*.vllm.stderr.log`
- `logs/*.mst.stdout.log`
- `logs/*.mst.stderr.log`
- `logs/slurm-<group>-%A_%a.out`
- `logs/slurm-<group>-%A_%a.err`
- `state.json` / `summary.json` / `summary.md` after `collect`

The adapter intentionally avoids one shared `state.json` while Slurm tasks are running. `collect` writes the aggregate `state.json` after reading per-job state files so downstream tools such as `mst_analyzer` and `energy_profiler` can consume Slurm runs through the same run-root contract as local orchestrator runs.

Energy Slurm runs live under:

```text
results/energy/<plan_id>/<energy_run_id>/
```

The energy run root contains:

- `plan.yaml` - copied `energy_profiler` plan
- `plan.json` - materialized Slurm energy run plan
- `groups/<group>.json` - array task payloads
- `scripts/<group>.sbatch.sh` - generated sbatch scripts
- `job-state/<energy_job_id>.json` - per-task mutable state while Slurm runs
- `jobs/<energy_job_id>/summary.json`
- `jobs/<energy_job_id>/request_records.jsonl`
- `jobs/<energy_job_id>/server_metrics.jsonl`
- `jobs/<energy_job_id>/windows.csv`
- `jobs/<energy_job_id>/gpu_power.json`
- `jobs/<energy_job_id>/energy_summary.json`
- `logs/*.vllm.stdout.log`
- `logs/*.vllm.stderr.log`
- `logs/*.profile.stdout.log`
- `logs/*.profile.stderr.log`
- `state.json` / `summary.json` / `summary.md` after `energy-collect`

Each energy array task starts one vLLM server, waits for readiness, and then
delegates the fixed-rate profiling work to
`python -m energy_profiler.cli run-live-trial`. This keeps benchmark execution,
GPU power sampling, `gpu_power.json`, and `energy_summary.json` generation on
the existing energy profiler path.

## Operational Notes

- The sbatch scripts use `set -euo pipefail`.
- `PYTHONPATH` is set to the repo’s `profiler/` directory inside each task.
- Result directories are cleaned before search to avoid stale `search_trace.json` failures on reruns.
- `resume` preserves succeeded job state, refreshes failed/planned/running job configs from the current manifest, and submits selected array indices with an `sbatch --array=...` override.
- The generated launch commands reuse `local_orchestrator.lifecycle.render_launch_command(...)`.
- The generated MST commands reuse `local_orchestrator.mst_adapter.build_search_command(...)` and `build_report_command(...)`.
- Probe output from manifest expansion is preserved in the per-job payloads and state files.

This adapter assumes one vLLM server per Slurm task and does not implement cross-node tensor parallelism.

## Slurm Energy Profiling Notes

- `energy-submit` loads the source MST run recorded in the energy plan and
  inherits the source manifest's `slurm` fields when available.
- Energy jobs are grouped by `EnergyPlanJob.launch.gpu_count`.
- Slurm owns GPU allocation. The task passes numeric IDs from `SLURM_JOB_GPUS`
  when present, then numeric IDs from `CUDA_VISIBLE_DEVICES`, and finally
  `0..gpu_count-1` as a fallback.
- The default energy Slurm time limit is `00:30:00` when the inherited Slurm
  config does not specify `time`.
- Failed energy jobs are represented as per-job state records with
  `status: failed` and a `last_error`. `energy-collect` writes an aggregate
  `state.json` and marks the run `failed` if any job failed.
- `energy-resume` is not implemented yet. Rerun support should follow the MST
  `resume` pattern by selecting non-succeeded array indices.
