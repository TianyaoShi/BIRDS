  We are extending `profiler/local_orchestrator` with a Slurm adapter layer for LLM inference profiling. The new package will be under `profiler/slurm_orchestrator`. Read
  `profiler/local_orchestrator/README.md` first.

  Goal:
  Implement the planned Slurm adapter as a thin submit-loop over already-expanded orchestrator jobs. Do not reimplement local GPU leasing/
  scheduling. Slurm owns GPU allocation; the adapter should reuse manifest parsing, experiment expansion, launch/search overrides, resource
  probing, vLLM command rendering, and MST command construction where practical.

  Important current architecture:
  - `local_orchestrator.manifest.load_manifest(path)` parses strict YAML.
  - `local_orchestrator.matrix.expand_manifest(manifest)` expands model/workload pairs into `ExpandedExperimentJob`.
  - `ExpandedExperimentJob` already contains:
    - `experiment_id`
    - `model`
    - `workload`
    - `endpoint`
    - resolved `launch`
    - resolved `search`
    - `hardware`
    - advisory `probe`
    - deterministic `result_dir`
    - `server_signature_key`
    - optional `server_metadata_file`
  - `launch.gpu_count` is the requested GPU count for a single vLLM server/job.
  - `run.max_active_gpus` is local-orchestrator concurrency only; do not interpret it as Slurm cluster capacity.
  - `probe.auto_gpu_count: true` may raise `launch.gpu_count` and `tensor_parallel_size` during expansion.
  - Probe is otherwise advisory.

  Key reusable functions/classes:
  - `local_orchestrator.lifecycle.render_launch_command(...)`
    - renders structured vLLM launch command, including:
      - `--tensor-parallel-size`
      - `--dtype`
      - `--gpu-memory-utilization`
      - `--max-model-len`
      - `--max-num-seqs`
      - `--max-num-batched-tokens`
  - `local_orchestrator.mst_adapter.MSTSearchAdapter._build_search_command(...)`
    - currently private, but useful logic. Consider refactoring command construction into public helpers instead of duplicating.
  - `MSTSearchAdapter` currently deletes `job.result_dir` before search to avoid stale `search_trace.json` overwrite errors. Preserve equivalent
  behavior in Slurm job scripts.
  - `RunStateStore` writes one JSON state file and is not safe for many Slurm jobs concurrently updating it. Avoid concurrent writes to
  `state.json`.

  Desired Slurm design:
  1. Add a Slurm-facing CLI, for example:
     - `python -m slurm_orchestrator.cli plan --manifest ...`
     - `python -m slurm_orchestrator.cli submit --manifest ... --run-id ...`
     - `python -m slurm_orchestrator.cli collect --run-root ...`
     Exact names can differ, but keep local `cli.py` intact.

  2. Add a `slurm` manifest section or CLI-only Slurm config.
     Current manifest parser is strict, so if adding YAML keys, update:
     - `manifest.py`
     - `models.py`
     - tests
     Likely Slurm config fields:
     - partition
     - account
     - qos
     - time
     - cpus_per_task
     - mem or mem_per_gpu
     - constraint
     - modules/setup commands
     - venv/python executable override
     - sbatch extra args
     - job array concurrency limit
     - base_port, default 8000 for one job per allocation

  3. Submission model:
     Prefer one Slurm job per expanded experiment, or an sbatch array over a serialized job-plan JSON.
     Each Slurm task should:
     - load one expanded job payload
     - request `launch.gpu_count` GPUs via Slurm
     - rely on Slurm-provided `CUDA_VISIBLE_DEVICES`
     - start vLLM on localhost
     - wait for readiness using `/v1/models`
     - run MST search
     - run MST report
     - terminate vLLM process group
     - write per-job status/log files

  4. State model:
     Do not have all array tasks write the same `state.json`.
     Use per-job files, e.g.:
     - `<run_root>/jobs/<experiment_id>.json`
     - `<run_root>/logs/<experiment_id>.vllm.stdout.log`
     - `<run_root>/logs/<experiment_id>.vllm.stderr.log`
     - `<run_root>/logs/<experiment_id>.mst.stdout.log`
     - `<run_root>/logs/<experiment_id>.mst.stderr.log`
     Then `collect` aggregates per-job JSON into `summary.json` / `summary.md`.
     You may reuse `RunStateStore.summarize` if you build a compatible aggregate state payload.

  5. Slurm job script behavior:
     - Use `set -euo pipefail`.
     - Export `PYTHONPATH=/path/to/arr26/profiler:$PYTHONPATH`.
     - Use the manifest’s `run.python_executable` if present.
     - Start vLLM in background and capture PID.
     - Use `trap` to kill the vLLM process group on exit.
     - Use `CUDA_VISIBLE_DEVICES` as provided by Slurm.
     - For one server per Slurm job, localhost port can usually be fixed, e.g. `8000`; no local port allocator needed unless supporting multiple servers per allocation. For different slurm tasks sharing a node, consider a simple port offset scheme based on Slurm task ID.

  6. Multi-GPU:
     - `launch.gpu_count` maps to Slurm GPU request.
     - `launch.tensor_parallel_size` maps to vLLM `--tensor-parallel-size`.
     - Do not implement cross-node tensor parallelism right now.
     - If `gpu_count > 1`, request that many GPUs on one node.

  7. Required tests:
     Add focused tests only. Do not run full suite by default.
     Suggested tests:
     - Slurm plan expands manifest into deterministic job payloads.
     - Generated sbatch script includes GPU count, output paths, vLLM launch, readiness wait, MST search/report, cleanup trap.
     - Array/concurrency option is rendered correctly.
     - Per-job status aggregation handles succeeded/failed/skipped.
     - Existing local orchestrator tests still pass for touched modules.

  8. Documentation:
     Create `profiler/slurm_orchestrator/README.md` with:
     - Slurm adapter commands
     - intended separation from local GPU scheduler
     - one-job-per-allocation assumption
     - state/log layout
     - cluster-specific manifest/CLI fields
     - note that Slurm adapter reuses expanded jobs and probe output

  9. Reference script
     You may find `BioLLM/profiler/deprecated/submit_enhanced_profiling_jobs.sh` useful as a reference for Slurm submission semantics and account information. However, there are lots of deprecated functionalities in that script that should not be reimplemented, such as model/dataset validation, job restart.

  Important recent behavior to preserve:
  - `launch.max_model_len` must be passed to vLLM as `--max-model-len`.
  - Workload `context_policy.max_model_len` only controls workload token filtering/truncation; it does not constrain vLLM allocation.
  - Hybrid MST search now uses closed-loop scouting more carefully:
    - `closed_loop_min_trials`
    - `max_closed_loop_concurrency`
    - `closed_loop_plateau_relative_gain`
  - MST output dirs should be cleaned before rerun to avoid stale `search_trace.json` overwrite failures.

  Deliverable:
  Implement a minimal but usable Slurm adapter layer and docs. Keep it thin: reuse expansion and command rendering; do not port local GPU
  leasing, local port allocation, or thread-per-slot scheduling into Slurm.