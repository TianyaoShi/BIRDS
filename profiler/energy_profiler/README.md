# Energy Profiler

`energy_profiler` is a post-MST profiling layer. It consumes completed
`local_orchestrator` result roots, generates reviewable fixed-rate energy
profiling plans, and executes those plans with GPU power monitoring.

The profiler does not change MST search behavior. MST search still finds the
maximum stable throughput under workload and SLO constraints. Energy profiling
uses those results as inputs and runs follow-up open-loop trials at selected
request rates.

## Workflow

1. Run an MST/orchestrator experiment.
2. Generate an energy plan from one or more orchestrator run roots.
3. Review or edit the YAML plan.
4. Run the plan.
5. Inspect per-job `summary.json`, `gpu_power.json`, and
   `energy_summary.json`.

## Plan Generation

Generate a rounded-MST plan from one orchestrator run:

```bash
PYTHONPATH=profiler:. python -m energy_profiler.cli plan-from-orchestrator \
  --orchestrator-run-root results/orchestrator/<run_id> \
  --output-plan experiments/energy/<plan_id>.yaml \
  --mode mst-rounded \
  --rate-source max_slo
```

When a main run and one or more reruns are both relevant, repeat
`--orchestrator-run-root` in priority order. Later roots override earlier roots
for the same `(model, workload, endpoint)` key:

```bash
PYTHONPATH=profiler:. python -m energy_profiler.cli plan-from-orchestrator \
  --orchestrator-run-root results/orchestrator/<main_run_id> \
  --orchestrator-run-root results/orchestrator/<rerun_id> \
  --output-plan experiments/energy/<plan_id>.yaml \
  --mode mst-rounded \
  --rate-source max_slo
```

This is the expected way to let the final energy plan digest a main loop plus
follow-up reruns without manually remembering which model's MST came from which
run ID.

Supported modes:

- `mst-rounded`: one fixed-rate profiling job per selected succeeded
  orchestrator job, using an MST rate floored to two decimal places by default.
- `sweep`: multiple fixed-rate jobs from low load through rounded MST for
  selected models or experiment IDs. Sweep mode uses preferred request-rate
  steps to keep the grid compact.
- `explicit`: fixed-rate jobs at request rates supplied with
  `--request-rates`.

Useful filters:

```bash
--models <model> ...
--workloads <workload-path> ...
--experiment-ids <experiment-id> ...
--exclude-models <model> ...
--exclude-workloads <workload-path> ...
--exclude-experiment-ids <experiment-id> ...
--min-model-size-b 3.0
--selection-yaml <path>
```

Exclusion filters are applied before merge validation and MST-rate selection. Use
`--min-model-size-b 3.0` for H100 plans that should ignore stale `<3B`
experiments left in old orchestrator state.

Dry-run a generated plan:

```bash
PYTHONPATH=profiler:. python -m energy_profiler.cli dry-run \
  --plan experiments/energy/<plan_id>.yaml
```

## Running A Managed Plan

Run a reviewed plan serially:

```bash
PYTHONPATH=profiler:. python -m energy_profiler.cli run \
  --plan experiments/energy/<plan_id>.yaml \
  --allowed-gpu-ids 0 1 \
  --max-active-gpus 1
```

Run a reviewed plan locally with GPU-aware parallel scheduling:

```bash
PYTHONPATH=profiler:. python -m local_orchestrator.cli energy-run \
  --plan experiments/energy/<plan_id>.yaml \
  --run-id <local_energy_run_id>
```

This path reuses the local orchestrator scheduler. It bin-packs energy jobs by
`launch.gpu_count`, starts TP2/TP4 vLLM servers with the leased physical GPU IDs,
and passes those same physical GPU IDs to the power monitors.

Resume an interrupted run:

```bash
PYTHONPATH=profiler:. python -m local_orchestrator.cli energy-resume \
  --run-root results/energy/<plan_id>/<local_energy_run_id>
```

Check run status:

```bash
PYTHONPATH=profiler:. python -m local_orchestrator.cli energy-status \
  --run-root results/energy/<plan_id>/<local_energy_run_id>
```

Managed plan execution reuses the same launch and resource primitives as
`local_orchestrator`: it leases the requested number of GPUs for each job,
allocates ports, starts or reuses a vLLM server for matching server signatures,
waits for readiness, runs fixed-rate open-loop trials, and releases resources at
the end. The `energy_profiler.cli run` command is serial; use
`local_orchestrator.cli energy-run` when local bin-packing across multiple GPUs is
desired.

Managed trials first measure an idle baseline, then run an unmeasured traffic
warmup at the target request rate, then start GPU power monitoring for the
measured trial. If `defaults.repeats` is greater than one, repeat artifacts are
stored under `repeat_001/`, `repeat_002/`, and so on, with aggregate repeat
statistics written at the job root.

Energy plans include separate runtime sections:

- `local_execution`: local-only GPU lease and port settings, including
  `allowed_gpu_ids` and `max_active_gpus`. Slurm execution ignores these fields.
- `slurm`: Slurm submission settings copied from the source orchestrator
  manifest, including `array_concurrency_limit`, `base_port`, partition/account,
  modules, and setup commands.

Runtime `run` and `resume` CLI flags may still override `local_execution`
explicitly. `slurm_orchestrator energy-submit` uses the plan's `slurm` section
when it is present, falling back to the source orchestrator manifest only for
older plans.

## Live-Server Trial

For a smoke test against an already-running OpenAI-compatible server, use
`run-live-trial`. This does not start vLLM and does not lease GPUs; the caller
must provide the GPU IDs to monitor.

Example:

```bash
PYTHONPATH=profiler:. python -m energy_profiler.cli run-live-trial \
  --trial-id live-gemma-4-e4b-it-synthetic-256-128-90s-gpu1 \
  --output-dir results/energy/live_gemma_4_e4b_it_synthetic_256_128_90s_gpu1 \
  --workload experiments/energy/synthetic_fixed_256_128.yaml \
  --model google/gemma-4-E4B-it \
  --base-url http://127.0.0.1:9300 \
  --endpoint /v1/completions \
  --metrics-url http://127.0.0.1:9300/metrics \
  --gpu-ids 1 \
  --duration-s 90 \
  --request-rate 1 \
  --request-timeout-s 300 \
  --metrics-interval-s 1 \
  --window-s 10 \
  --idle-monitor-duration-s 10 \
  --traffic-warmup-s 30 \
  --gpu-monitor-interval-s 1
```

Use `--force` only when intentionally replacing an existing output directory.

## Outputs

For each managed job or live trial, the profiler writes:

- `summary.json`: `llm_mst_finder` trial summary and benchmark metrics.
- `request_records.jsonl`: per-request latency and token records.
- `server_metrics.jsonl`: vLLM metrics samples.
- `windows.csv`: fixed-window latency, throughput, queue, and server metrics.
- `gpu_power.json`: idle and traffic GPU power traces.
- `energy_summary.json`: aggregate energy and per-token/request energy metrics.

For repeated jobs, the job root contains aggregate `summary.json`,
`gpu_power.json`, and `energy_summary.json`; per-repeat trial artifacts live in
`repeat_001/`, `repeat_002/`, etc.

Managed plan runs also keep run state and summaries under:

```text
results/energy/<plan_id>/
```

The run root includes `summary.json`, `summary.md`, and `summary_compact.csv`.
The compact CSV contains one row per energy job with model, workload, GPU count,
tensor parallel size, request rate, power statistics, aggregate energy, energy
per request, and energy per token.

## Running A Plan On Slurm

Reviewed plans can also be submitted through `slurm_orchestrator`:

```bash
PYTHONPATH=profiler:. python -m slurm_orchestrator.cli energy-submit \
  --plan experiments/energy/<plan_id>.yaml \
  --run-id <energy_slurm_run_id>
```

After the Slurm arrays finish, collect per-task state into the normal energy
summary files:

```bash
PYTHONPATH=profiler:. python -m slurm_orchestrator.cli energy-collect \
  --run-root results/energy/<plan_id>/<energy_slurm_run_id>
```

Slurm energy runs store artifacts under
`results/energy/<plan_id>/<energy_slurm_run_id>/jobs/<energy_job_id>/` with the
same per-job `summary.json`, `gpu_power.json`, and `energy_summary.json` files
as local managed execution.

## Metrics

Latency percentiles are available as `p50`, `p90`, `p95`, and `p99` for metric
families that collect percentiles:

- TTFT
- TPOT
- ITL
- E2E

Window CSVs expose the same percentile set for these latency families. Benchmark
length summaries include `mean`, `median`, `p90`, `p95`, and `p99`.

Energy summaries include:

- `energy_joules`
- `incremental_energy_joules`
- `energy_kwh`
- `avg_power_w`
- `idle_avg_power_w`
- `incremental_avg_power_w`
- `p50_power_w`
- `p90_power_w`
- `p95_power_w`
- `p99_power_w`
- energy per successful request
- energy per total request
- energy per total token
- incremental energy per corresponding unit

## Important Defaults

- Managed plan trial duration: `180s`
- Managed plan idle baseline duration: `warmup_s`, default `30s`
- Managed plan traffic warmup duration: `traffic_warmup_s`, default `30s`
- Managed plan cooldown: `15s`
- Managed plan repeats: `repeats`, default `1`
- Managed repeat cooldown: `repeat_cooldown_s`, default `15s`
- Managed GPU monitor interval: `0.025s`
- Live-trial duration: `90s`
- Live-trial request rate: `1 req/s`
- Live-trial readiness is the caller's responsibility because the server is
  already running.
- Server readiness timeout for managed launches defaults to `300s` through the
  local orchestrator launch model.

## Notes

- Use managed plan execution for real experiments. Use `run-live-trial` for
  quick validation against an already-running server.
- Always pass the actual GPU IDs used by a live server. The live path only
  monitors the IDs provided with `--gpu-ids`.
- For main-loop plus rerun MST collection, generate one plan with repeated
  `--orchestrator-run-root` arguments instead of creating disconnected energy
  plans with different run IDs.
