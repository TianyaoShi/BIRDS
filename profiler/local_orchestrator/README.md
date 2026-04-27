# Local Orchestrator (V1)

This package runs small-scale, single-node orchestration for LLM MST Finder experiments. It reads a YAML manifest, expands model/workload pairs into jobs, leases GPUs and ports, boots vLLM servers, invokes MST search/report, and writes durable run state + summaries.

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

The manifest has four top-level sections: `run`, `launch`, `search`, `experiments`.

- `run`: output location, GPU policy, ports, retry counts, and default endpoint.
- `launch`: vLLM launch settings (structured flags or a raw template command).
- `search`: MST Finder search configuration.
- `experiments`: model/workload pairs and per-experiment overrides.

The parser is strict: unknown keys fail validation, and mutually-exclusive launch styles cannot be mixed. If you need new fields, add them in `manifest.py` and `models.py` in lockstep.

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
results/mst/<model_slug>/<dataset_slug>/<server_slug>/
```

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

- Multi-node or cluster scheduling (single host only).
- Multi-GPU per model (only `gpu_count=1` jobs).
- More than 3 active GPUs in a run (`max_active_gpus <= 3`).
- GPU memory awareness or preflight checks (no `nvidia-smi` integration).
- Adaptive backoff, cancellation, or preemption of jobs.
- Dynamic closed-loop search tuning (no closed-loop concurrency config in the manifest).
- Report-only or search-only modes (search+report is always run together).
- Automatic retries with exponential backoff or jitter.
- Server warmup or model-specific readiness probes beyond `/v1/models`.
- Metrics export beyond the local run artifacts (no Prometheus integration).
- External state storage (state is JSON files only).

## Development notes

- `MSTSearchAdapter` relies on `PYTHONPATH` to find `llm_mst_finder` when invoked as a module.
- The scheduler uses a thread-per-slot model for parallel jobs; each slot owns its own lifecycle manager.
- Resume logic requires that the manifest expands to exactly the same experiment IDs as the stored run.

If you extend this package, update this README and keep manifest/model keys consistent across validation, search command construction, and summary output.
