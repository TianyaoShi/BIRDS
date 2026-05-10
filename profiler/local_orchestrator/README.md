# Local Orchestrator (V1)

This package runs small-scale, single-node orchestration for LLM MST Finder experiments. It reads a YAML manifest, expands model/workload pairs into fully planned jobs, leases GPUs and ports, boots vLLM servers, invokes MST search/report, and writes durable run state + summaries.

## What this package is

- A pragmatic, local runner for MST Finder that assumes a single machine and a small GPU pool.
- A "run -> resume -> summarize" workflow with deterministic job expansion and persistent artifacts.
- A thin wrapper around `llm_mst_finder.cli` plus a vLLM lifecycle manager.

## What this package is not

- Not a distributed scheduler.
- Not a full experiment tracker.
- Not a general job orchestration framework.

## Quick start

1) Dry-run a manifest to validate expansion and resource usage:

```
PYTHONPATH=/path/to/arr26/profiler \
  /path/to/venv/bin/python -m local_orchestrator.cli dry-run \
  --manifest /path/to/manifest.yaml
```

2) Run a new orchestration:

```
PYTHONPATH=/path/to/arr26/profiler \
  /path/to/venv/bin/python -m local_orchestrator.cli run \
  --manifest /path/to/manifest.yaml \
  --run-id my-run-id
```

3) Resume (or force rerun) a prior run:

```
PYTHONPATH=/path/to/arr26/profiler \
  /path/to/venv/bin/python -m local_orchestrator.cli resume \
  --run-root /path/to/results/orchestrator/my-run-id \
  --force
```

4) Check status without running jobs:

```
PYTHONPATH=/path/to/arr26/profiler \
  /path/to/venv/bin/python -m local_orchestrator.cli status \
  --run-root /path/to/results/orchestrator/my-run-id
```

## Manifest overview

The manifest has these top-level sections: `run`, optional `slurm`, `hardware`, `probe`, `launch`, `search`, `overrides`, `experiments`.

- `run`: orchestrator output location, optional MST output root, GPU policy, ports, retry counts, and default endpoint.
- `slurm`: optional cluster submission settings for the separate `slurm_orchestrator` package; the local CLI ignores it.
- `hardware`: target accelerator profile used by the resource probe (`name`, `gpu_memory_gb`, `gpu_memory_utilization`).
- `probe`: conservative memory-estimation settings, including optional `auto_gpu_count`.
- `launch`: vLLM launch settings (structured flags or a raw template command).
- `search`: MST Finder search configuration.
- `overrides`: selector-based launch/search updates that apply after model/workload expansion.
- `experiments`: model/workload pairs, direct per-experiment overrides, experiment-local selector overrides, optional hardware/probe overrides, and metadata files.

The parser is strict: unknown keys fail validation, and mutually-exclusive launch styles cannot be mixed. If you need new fields, add them in `manifest.py` and `models.py` in lockstep.

### Model/workload/hardware-specific overrides

Use `overrides` when one global launch/search template is too blunt for a matrix. Overrides are applied in order after each model/workload pair is expanded. A rule matches only the selectors it declares; selector values use shell-style wildcards and are case-insensitive.

```
hardware:
  name: a100-80gb
  gpu_memory_gb: 80
  gpu_memory_utilization: 0.90

probe:
  enabled: true
  auto_gpu_count: true
  activation_memory_gb: 4
  memory_safety_factor: 1.20

launch:
  tensor_parallel_size: 1
  gpu_count: 1
  dtype: float16
  max_model_len: 32768

search:
  search_mode: hybrid
  initial_request_rate: 1
  max_request_rate: 8
  max_binary_steps: 8
  ttft_slo_ms: 1000
  ttft_slo_mode: static
  tpot_slo_ms: 125

overrides:
  - match:
      model: "*1B*"
    search:
      max_request_rate: 40
      max_binary_steps: 10
  - match:
      model: "*8B*"
    search:
      max_request_rate: 8
      max_binary_steps: 7
  - match:
      workload: "*long-context*"
    search:
      ttft_slo_mode: length_scaled
      tpot_slo_ms: 175

  - match:
      workload: "*longbench_short_answer_document_qa*"
    search:
      ttft_slo_mode: static
      longbench_ttft_static_preset: tight

experiments:
  - id: chat-matrix
    models:
      - meta-llama/Llama-3.1-8B-Instruct
      - Qwen/Qwen3-1.7B
    workloads:
      - workloads/sharegpt.yaml
      - workloads/long-context.yaml
```

Experiment-local `overrides` use the same shape and run after top-level `overrides`, so they can refine a broad policy for one experiment group.

### Resource probing

The probe estimates the minimum GPU count needed for model weights, an activation-memory allowance, and at least `probe.kv_cache_request_count` request's KV cache. It infers model size from names such as `1B`, `4B`, `8B`, and `E4B`; use `probe.model_size_overrides_b` for names that do not encode parameter count clearly.

When `probe.auto_gpu_count: true`, expansion raises `launch.gpu_count` to the estimated minimum and also raises `tensor_parallel_size` when it was tracking the old GPU count. When auto mode is off, the probe is advisory: dry-run and state output still expose the estimate, but the local scheduler only blocks on concrete configured resource constraints such as `launch.gpu_count > run.max_active_gpus`.

Dry-run output includes the final launch/search values and probe payload for each expanded job. This expanded job representation is the intended reuse point for a future Slurm adapter: Slurm should submit the already-expanded job plan and let the cluster manager allocate the requested GPUs.

### Local GPU scheduling

The local scheduler bin-packs pending jobs onto `run.max_active_gpus` from `run.allowed_gpu_ids`. Each running job leases `launch.gpu_count` GPUs and one base/metrics port pair. When multiple jobs fit, the scheduler starts the largest pending job that fits the currently free GPU pool, then backfills with smaller jobs where possible. Jobs whose `launch.gpu_count` exceeds `run.max_active_gpus` fail preflight without starting vLLM.

For example, with TP1, TP2, and TP4 jobs pending:

- 2 available GPUs: TP2 can run, then TP1 can backfill if 1 GPU remains free; TP4 waits or fails preflight when `max_active_gpus` is 2.
- 3 available GPUs: TP2 and TP1 can run concurrently; TP4 waits or fails preflight when `max_active_gpus` is 3.
- 4 available GPUs: TP4 runs when selected as the largest fitting job; TP2 and TP1 can co-run after enough GPUs are free.

For local raw launch templates, `{gpu_id}` expands to the first leased GPU and `{gpu_ids}` expands to the comma-separated leased set. The lifecycle manager also sets `CUDA_VISIBLE_DEVICES` to that same comma-separated set.

`launch.max_model_len` is passed to vLLM as `--max-model-len`. Keep this aligned with the workload context cap when using long-context model cards; workload `context_policy.max_model_len` filters or truncates requests for MST, but it does not constrain the vLLM engine allocation by itself.

### Search modes and reporting

`MSTSearchAdapter` runs `llm_mst_finder.cli search` followed by `report`. The report generation requires at least one **stable open-loop trial**. As a result:

- `search_mode: open-loop` works but can fail on unstable convergence.
- `search_mode: closed-loop` will run, but report generation fails (no open-loop trials).
- `search_mode: hybrid` is recommended for smoke runs because it produces closed-loop scouting and open-loop confirmation.

## Run outputs

Each run creates a root directory at:

```
<output_root>/<run_id>/
```

Key files:

- `state.json` — durable job state and artifacts
- `events.jsonl` — append-only event stream
- `summary.json` / `summary.md` — rollups and per-job summaries
- `logs/` — stdout/stderr for vLLM and MST steps

Each experiment writes MST artifacts under:

```
<mst_output_root>/<model_slug>/<dataset_slug>/<server_slug>/
```

For local orchestrator runs, the default MST output root is run-scoped:

```
<output_root_parent>/mst/<run_id>/
```

For example, `output_root: ../results/orchestrator` and `run_id: sharegpt-001`
will write MST artifacts under `../results/mst/sharegpt-001/`. This keeps
workload names semantic and avoids overwriting prior MST traces when the same
model/workload/server configuration is rerun. Set `run.mst_output_root` when a
manifest needs an explicit artifact root.

## Module layout

- `manifest.py` — strict YAML parsing and validation
- `matrix.py` — deterministic job expansion and result directory identity
- `resources.py` — GPU leasing and port allocation
- `lifecycle.py` — vLLM server lifecycle and readiness checks
- `mst_adapter.py` — MST CLI invocation and artifact validation
- `scheduler.py` — orchestration loop with retries, resume, and parallel slots
- `state_store.py` — durable state/events and summary output
- `cli.py` — `dry-run`, `run`, `resume`, `status`

## Deferred in V1 (explicit)

The V1 orchestrator intentionally omits:

- Multi-node or cluster scheduling (single host only; Slurm should be a thin submit-loop over expanded jobs).
- Cross-node tensor parallelism (not needed).
- Live GPU memory discovery from `nvidia-smi`; set `hardware.gpu_memory_gb` in the manifest for now.
- Adaptive backoff, cancellation, or preemption of jobs.
- Report-only or search-only modes (search+report is always run together).
- Automatic retries with exponential backoff or jitter.
- Server warmup or model-specific readiness probes beyond `/v1/models`.
- External state storage (state is JSON files only).

## Development notes

- `MSTSearchAdapter` relies on `PYTHONPATH` to find `llm_mst_finder` when invoked as a module.
- The scheduler uses a thread per active local job; each active job owns its GPU lease, port reservation, and lifecycle manager.
- Resume logic requires that the manifest expands to exactly the same experiment IDs as the stored run.

If you extend this package, update this README and keep manifest/model keys consistent across validation, search command construction, and summary output.
