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
    - modetree/gpu
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
- `summary.json` / `summary.md` after `collect`

The adapter intentionally avoids one shared `state.json` while Slurm tasks are running.

## Operational Notes

- The sbatch scripts use `set -euo pipefail`.
- `PYTHONPATH` is set to the repo’s `profiler/` directory inside each task.
- Result directories are cleaned before search to avoid stale `search_trace.json` failures on reruns.
- The generated launch commands reuse `local_orchestrator.lifecycle.render_launch_command(...)`.
- The generated MST commands reuse `local_orchestrator.mst_adapter.build_search_command(...)` and `build_report_command(...)`.
- Probe output from manifest expansion is preserved in the per-job payloads and state files.

This adapter assumes one vLLM server per Slurm task and does not implement cross-node tensor parallelism.
