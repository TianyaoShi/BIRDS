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
  array_concurrency_limit: 4
  base_port: 8000
```

`run.python_executable` is still honored. If both are set, `slurm.python_executable` wins for the Slurm helper commands and MST invocations.

`--cpus-per-task` is derived automatically as `14 * launch.gpu_count`, so the manifest does not expose a CPU override.

## Submission Model

- Jobs are grouped by `launch.gpu_count`.
- Each group is submitted as one Slurm array with a generated `#SBATCH --array=...` clause.
- Every array task loads one expanded job payload, starts one vLLM server on localhost, waits for `/v1/models`, runs MST search and report, then tears the server down.
- Multi-GPU jobs request `launch.gpu_count` GPUs on one node with `#SBATCH --gres=gpu:<count>`.
- CPU allocation follows the cluster convention `14 CPUs per GPU`, so a 4-GPU job renders `#SBATCH --cpus-per-task=56`.

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

## Operational Notes

- The sbatch scripts use `set -euo pipefail`.
- `PYTHONPATH` is set to the repo’s `profiler/` directory inside each task.
- Result directories are cleaned before search to avoid stale `search_trace.json` failures on reruns.
- `resume` preserves succeeded job state, refreshes failed/planned/running job configs from the current manifest, and submits selected array indices with an `sbatch --array=...` override.
- The generated launch commands reuse `local_orchestrator.lifecycle.render_launch_command(...)`.
- The generated MST commands reuse `local_orchestrator.mst_adapter.build_search_command(...)` and `build_report_command(...)`.
- Probe output from manifest expansion is preserved in the per-job payloads and state files.

This adapter assumes one vLLM server per Slurm task and does not implement cross-node tensor parallelism.

## Milestone: Slurm Energy Profiling

The current Slurm adapter submits MST search/report work only. `energy_profiler`
can consume completed orchestrator outputs and run fixed-rate energy profiling
locally, but there is not yet a Slurm backend for executing `EnergyPlanJob`
arrays on the cluster.

Target capability:

```bash
PYTHONPATH=/path/to/BioLLM/profiler \
  /path/to/venv/bin/python -m slurm_orchestrator.cli energy-submit \
  --plan /path/to/experiments/energy/<plan_id>.yaml
```

Planned implementation steps:

1. Make Slurm MST runs first-class inputs to `energy_profiler` plan generation.
   `slurm_orchestrator.collect_run()` should persist the aggregate state to
   `state.json` in addition to `summary.json` and `summary.md`, so
   `energy_profiler.planning` can read the source `manifest_path` through the
   same contract it already uses for `local_orchestrator` roots. A compatible
   fallback to `plan.json` is acceptable, but the preferred output contract is
   to write `state.json` after collection.

2. Add an `energy-submit --plan ...` Slurm command. This command should load an
   `EnergyPlan`, materialize `EnergyPlanJob` entries into Slurm array groups,
   and submit them with the same run-root structure, per-job JSON state,
   resume/collect semantics, module setup, venv setup, and diagnostic logging
   style used by MST Slurm jobs.

3. Reuse existing `energy_profiler` logic where possible. The Slurm array task
   should still run one energy profiling job per task: start one vLLM server,
   collect an idle power baseline, run one fixed-rate open-loop
   `llm_mst_finder.cli run-trial`, collect traffic power, write
   `gpu_power.json`, compute `energy_summary.json`, finalize state, and tear
   down the server. Avoid local GPU leasing inside a Slurm task; Slurm owns GPU
   allocation and the task should monitor the GPU IDs exposed to that task.

Open design questions before implementation:

- Should energy Slurm runs live under `results/energy/<plan_id>` to match
  `energy_profiler`, or under `results/orchestrator/<energy_run_id>` to match
  Slurm submission bookkeeping?
  A: `results/energy/<plan_id>/<energy_slurm_run_id>`
- Should the energy Slurm CLI be part of `slurm_orchestrator.cli` as
  `energy-submit` / `energy-collect` / `energy-resume`, or should it become a
  separate package/CLI such as `energy_slurm_orchestrator`?
  A: part of `slurm_orchestrator.cli` as `energy-submit` / `energy-collect` / `energy-resume`
- Should energy jobs group by `launch.gpu_count`, by server signature, or both?
  Grouping by server signature helps reason about repeated sweep rates, but
  one Slurm task per fixed-rate job is simpler and more failure-isolated.
  A: one Slurm task per fixed-rate job
- Should Slurm energy jobs restart vLLM for every fixed-rate point, or support
  an optional mode that runs multiple rates for the same server signature in
  one allocation to reduce startup overhead?
  A: For normal use case (mst_rounded), since we just do energy profiling on each model-workload pair's MST, there's no need to restart vLLM at all; there's only one fixed-rate point per job. For other use cases (sweep, explicit), we can consider an optional mode that runs multiple rates for the same server signature in one allocation with no restart, but only sleep between trials. The priority is to support mst_rounded first.
- Which GPU identifier should power monitoring use on this cluster:
  physical IDs from `nvidia-smi`, CUDA-local IDs after `CUDA_VISIBLE_DEVICES`,
  or `SLURM_JOB_GPUS` translated to physical IDs?
  A: `nvidia-smi` physical IDs
- Should energy plans carry Slurm fields directly, or should `energy-submit`
  accept `--manifest` / `--slurm-config` to reuse account, partition, qos,
  modules, setup commands, and time limits from the MST manifest?
  A: we will need a configurable time limit for energy jobs, but other Slurm fields can be inherited from the MST manifest. The cleanest way is still to keep everything in the energy plan, so we can have a self-contained plan that can be submitted without needing to reference the original MST manifest.
- What default Slurm time limit should a single energy job request? It should
  account for vLLM startup, idle baseline, trial duration, cooldown, and cleanup.
  A: 30 minutes for mst_rounded mode, which runs one trial per job. For sweep and explicit modes, we can consider a longer time limit or a configurable time limit that scales with the number of rates per job.
