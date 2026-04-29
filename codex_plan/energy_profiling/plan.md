# Post-MST Energy Profiling Layer

Status: Draft for review on 2026-04-29.

## Goal

After an MST orchestrator run succeeds for a model/workload set, add a separate profiling layer that reads the orchestrator results, generates a human-reviewable energy profiling plan, and then lets the user manually launch that plan.

This layer should not change MST search behavior. MST remains responsible for finding a maximum stable request rate under workload/SLO constraints. The new layer consumes those results and runs fixed-rate open-loop profiling trials with GPU power monitoring.

## Main Use Cases

1. Profile each succeeded model at a rounded MST rate.
   - Input: `results/orchestrator/<run_id>/summary.json`.
   - For each succeeded job, read `max_slo_satisfying_request_rate` or `max_no_drift_request_rate`.
   - Round the selected MST downward to a human-preferable rate.
   - Generate one energy profiling trial per model/workload/server config.

2. Exhaustively profile selected models from low load to rounded MST.
   - Input: same orchestrator result plus a model allowlist.
   - Generate up to 20 fixed-rate points from near-zero load to the rounded MST.
   - This should be optional because it multiplies runtime.

3. Profile explicit comparison rates.
   - Input: selected models plus request rates supplied by the user.
   - Generate fixed-rate energy trials at exactly those rates.
   - This supports direct cross-model comparisons independent of MST-derived rates.

## Recommended Package Layout

Add a new package rather than extending `local_orchestrator` directly:

- `profiler/energy_profiler/__init__.py`
- `profiler/energy_profiler/models.py`
- `profiler/energy_profiler/planning.py`
- `profiler/energy_profiler/executor.py`
- `profiler/energy_profiler/cli.py`
- `profiler/energy_profiler/reporting.py`

Rationale:

- The layer consumes orchestrator outputs but has different semantics from MST search.
- It should be reusable for both local and Slurm-backed orchestrator runs.
- Plan generation and execution need to stay separable so the user can review or edit the plan before launching.

## Inputs

### Plan Generation Inputs

The planner should accept:

- `--orchestrator-run-root results/orchestrator/<run_id>`
- `--output-plan experiments/energy/<plan_id>.yaml`
- `--rate-source max_slo|max_no_drift`, default `max_slo`
- `--rounding-policy <name-or-json>`
- `--mode mst-rounded|sweep|explicit`
- optional model/workload filters
- optional explicit request rates
- optional sweep step count, default 10, max 20
- optional minimum request rate, default one rounded quantum

### Source Artifacts

Use these files:

- `results/orchestrator/<run_id>/summary.json`
  - high-level job index
  - job status
  - `result_dir`
  - `max_no_drift_request_rate`
  - `max_slo_satisfying_request_rate`

- `<job.result_dir>/search_trace.json`
  - authoritative search result and search config
  - useful for preserving `search_id`, `search_mode`, selected rate, and confirmation trial

- `<job.result_dir>/final_report.json`
  - optional richer context for reporting

- Existing manifest path from orchestrator state if needed:
  - `results/orchestrator/<run_id>/state.json`
  - useful for reconstructing model, workload path, endpoint, launch fields, and hardware.

## Energy Plan YAML

Generate a reviewable YAML file with all fixed-rate profiling jobs. Proposed shape:

```yaml
plan:
  plan_id: energy-sharegpt-l40-000
  source_orchestrator_run_root: results/orchestrator/single-gpu-model-loop-run-sharegpt-000
  output_root: results/energy
  python_executable: /local/scratch/a/shi676/.venv/bin/python
  mode: mst-rounded

defaults:
  duration_s: 180
  warmup_s: 30
  cooldown_s: 15
  metrics_interval_s: 1.0
  window_s: 10.0
  gpu_monitor_interval_s: 0.025
  gpu_monitor_truncate_s: 5
  monitor_clock: false
  request_timeout_s: 21600
  safety_max_outstanding: null

rounding:
  mode: floor_preferred
  preferred_steps: [0.1, 0.2, 0.25, 0.5, 1.0]
  minimum_rate: 0.1

jobs:
  - id: qwen3-8b-sharegpt-r4
    source_experiment_id: single-gpu-model-loop-...
    source_result_dir: results/mst/qwen-qwen3-8b/live-sharegpt-workload-context/server-...
    model: Qwen/Qwen3-8B
    workload: experiments/live_sharegpt_workload_context.yaml
    endpoint: /v1/chat/completions
    request_rate: 4.0
    mst_rate: 4.4375
    mst_rate_source: max_slo_satisfying_request_rate
    launch:
      executable: vllm
      dtype: float16
      max_model_len: 32768
      tensor_parallel_size: 1
      gpu_count: 1
    metadata:
      source_orchestrator_run_id: single-gpu-model-loop-run-sharegpt-000
      rounded_from_rate: 4.4375
      rounding_policy: floor_preferred
```

The generated plan should be deterministic and should include enough launch/search context that it can be executed later without depending on mutable orchestrator state.

## Rounding Policy

The selected MST should be rounded down to a human-preferable precision such as `0.1`, `0.2`, `0.25`, `0.5`, or `1`. This keeps the follow-up profiling rate at or below the measured SLO-valid MST.

Recommended default:

1. Choose a display quantum based on the MST scale.
   - `< 1 req/s`: use `0.1`
   - `1-3 req/s`: use `0.25`
   - `3-10 req/s`: use `0.5`
   - `>= 10 req/s`: use `1.0`

2. Floor to that quantum.

3. Clamp to `minimum_rate` only when the MST is positive but below the smallest useful open-loop rate.

Examples:

- `0.083` -> `0.1`, because positive rates below the minimum useful trial rate are clamped
- `0.61` -> `0.6`
- `1.84` -> `1.75`
- `4.37` -> `4.0`
- `17.4` -> `17`

This keeps rates readable while avoiding profiling above the measured MST.

## Sweep Policy

For selected models, generate up to 20 fixed-rate trials from low load to rounded MST.

Recommended defaults:

- `steps: 10`
- `max_steps: 20`
- `spacing: linear`
- include the rounded MST endpoint
- start at one rounding quantum, not exactly `0`, because open-loop rate `0` is not meaningful.
- run rates in ascending order while reusing the same warm vLLM server for a matching model/server signature.

Example for rounded MST `4.5` and `steps=10`:

```text
0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5
```

For very low MST values, use the smallest configured quantum and deduplicate rounded rates.

Optional future mode:

- logarithmic spacing for models with a large dynamic range, especially if the target rate exceeds 20 req/s.

## Explicit Comparison Policy

Allow the user to specify:

```yaml
explicit:
  models:
    - Qwen/Qwen3-4B-Instruct-2507
    - Qwen/Qwen3-8B
  request_rates: [0.5, 1.0, 2.0, 4.0]
```

The planner should validate that selected models exist in the source orchestrator run and have succeeded MST results unless the user passes an explicit override allowing missing MST.

## Execution Model

Execution should be a separate command from planning:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m energy_profiler.cli plan \
  --orchestrator-run-root results/orchestrator/single-gpu-model-loop-run-sharegpt-000 \
  --output-plan experiments/energy/sharegpt_l40_energy_000.yaml \
  --mode mst-rounded

PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m energy_profiler.cli run \
  --plan experiments/energy/sharegpt_l40_energy_000.yaml
```

The run command should:

1. Launch or reuse vLLM using the same lifecycle semantics as `local_orchestrator`.
2. Measure an idle-power baseline after the server reaches readiness and before request traffic starts.
3. Start GPU power monitors for the assigned GPU IDs.
4. Run one fixed-rate `llm_mst_finder.cli run-trial --mode open-loop` trial.
5. Stop power monitors after the trial.
6. Write one result directory per energy job.
7. Continue to the next job on failure.
8. Emit `summary.json` and `summary.md`.

For sweep jobs, group trials by model/server signature, reuse the same warm vLLM server, and execute rates in ascending order.

## Trial Runner Integration

Preferred implementation path:

1. Extend `llm_mst_finder` with an optional power monitor interface.
   - Add a `PowerMonitor` protocol similar to `MetricsPoller`.
   - Record power artifacts beside `request_records.jsonl`, `server_metrics.jsonl`, and `windows.csv`.
   - Add `energy_metrics` to `TrialSummary` or to a sidecar `energy_summary.json`.

2. Reuse `profiler/gpu_monitor.py`.
   - Current `GPUMonitor` returns:
     - `avg_power_mw`
     - `power_stats`
     - `power_trace_mw`
     - optional `clock_trace_mhz`
   - Keep milliwatts internally and expose watts/joules in summary fields with clear names.

3. Avoid using `benchmark_serving.py` as the executor.
   - It already has useful power-stat logic, but MST finder has the workload compatibility, request client, response parsing, and artifact structure we now depend on.
   - Reuse the concepts, not the benchmark CLI as the new core path.

Recommended energy output fields:

```json
{
  "energy_joules": 12345.6,
  "incremental_energy_joules": 2345.6,
  "energy_kwh": 0.003429,
  "avg_power_w": 274.3,
  "idle_avg_power_w": 222.2,
  "incremental_avg_power_w": 52.1,
  "min_power_w": 190.2,
  "p50_power_w": 276.1,
  "p95_power_w": 338.8,
  "max_power_w": 356.4,
  "energy_per_successful_request_j": 12.3,
  "energy_per_output_token_j": 0.04,
  "energy_per_total_token_j": 0.02,
  "monitor_duration_s": 180.0,
  "idle_monitor_duration_s": 30.0,
  "gpu_ids": [0]
}
```

For multi-GPU trials, compute:

- per-GPU power stats
- per-GPU idle baseline stats
- aggregate energy over all assigned GPUs
- aggregate incremental energy above idle baseline
- per-request/per-token metrics from aggregate energy

## Result Layout

Recommended local result layout:

```text
results/energy/<plan_id>/
  plan.yaml
  state.json
  summary.json
  summary.md
  logs/
    <job_id>.vllm.stdout.log
    <job_id>.vllm.stderr.log
    <job_id>.profile.stdout.log
    <job_id>.profile.stderr.log
  jobs/
    <job_id>/
      summary.json
      request_records.jsonl
      server_metrics.jsonl
      windows.csv
      gpu_power.json
      energy_summary.json
```

Each job result should preserve:

- source orchestrator run id
- source MST experiment id
- source MST result dir
- model/workload/endpoint
- launch fields
- selected request rate
- original MST rate and rounding policy
- duration and warmup/truncation settings
- raw power trace or a path to it

## Local vs Slurm

The plan format should be cluster-neutral.

Local execution can reuse:

- `local_orchestrator.lifecycle.render_launch_command`
- local GPU lease/port allocation if running multiple jobs locally
- `llm_mst_finder.cli run-trial`

Slurm execution later can reuse:

- the same generated energy plan YAML
- one Slurm task per energy job
- Slurm-owned GPU allocation
- fixed localhost port inside each allocation

Do not bake local GPU IDs into the generated plan unless the user explicitly supplies them. Prefer resolving GPU assignment at execution time.

## CLI Proposal

Plan commands:

```bash
python -m energy_profiler.cli plan-from-orchestrator \
  --orchestrator-run-root results/orchestrator/<run_id> \
  --output-plan experiments/energy/<plan_id>.yaml \
  --mode mst-rounded \
  --rate-source max_slo \
  --rounding-policy floor_preferred

python -m energy_profiler.cli plan-from-orchestrator \
  --orchestrator-run-root results/orchestrator/<run_id> \
  --output-plan experiments/energy/<plan_id>.yaml \
  --mode sweep \
  --models Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B \
  --sweep-steps 20

python -m energy_profiler.cli plan-explicit \
  --orchestrator-run-root results/orchestrator/<run_id> \
  --output-plan experiments/energy/<plan_id>.yaml \
  --models Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B \
  --request-rates 0.5 1 2 4
```

Execution commands:

```bash
python -m energy_profiler.cli dry-run --plan experiments/energy/<plan_id>.yaml
python -m energy_profiler.cli run --plan experiments/energy/<plan_id>.yaml
python -m energy_profiler.cli resume --run-root results/energy/<plan_id>
python -m energy_profiler.cli status --run-root results/energy/<plan_id>
```

## Implementation Phases

### Phase 1: Plan Schema and Generator

- Add strict dataclasses for energy plan defaults, rounding, jobs, and execution state.
- Parse orchestrator `summary.json` and `state.json`.
- Filter succeeded jobs.
- Extract MST rate and source artifacts.
- Generate deterministic plan YAML.
- Add dry-run rendering.

Focused tests:

- succeeded jobs are included, failed/skipped jobs are excluded
- `max_slo` vs `max_no_drift` rate source selection
- rounding behavior around low and high rates
- sweep generation deduplicates rates and caps at 20 steps
- sweep generation orders rates ascending for server reuse
- explicit plan validates model selection

### Phase 2: Power Monitor Integration in MST Trial Runner

- Add optional GPU power monitor support to `llm_mst_finder.cli run-trial`.
- Add CLI args:
  - `--gpu-id`, repeatable or comma-separated
  - `--gpu-monitor-interval-s`
  - `--gpu-monitor-truncate-s`
  - `--gpu-monitor-clock`
  - `--energy-summary-path`, optional
- Write `gpu_power.json` and `energy_summary.json`.
- Keep the default behavior unchanged when no GPU monitor args are supplied.

Focused tests:

- fake monitor produces deterministic energy summary
- no monitor preserves existing trial output schema
- zero successful requests avoids division by zero and marks per-request energy as null

### Phase 3: Energy Executor

- Reuse local orchestrator launch lifecycle to start vLLM.
- Run fixed-rate `llm_mst_finder.cli run-trial --mode open-loop`.
- Capture logs and job status.
- Continue on per-job failure.
- Write energy run summary.

Focused tests:

- command construction includes fixed `--request-rate`
- launch fields from plan are honored
- result directory overwrite policy is explicit
- failed jobs do not block later jobs

### Phase 4: Slurm Compatibility

- Add Slurm plan materialization for energy jobs.
- Use one task per fixed-rate profiling job.
- Reuse the same energy plan YAML.
- Collect per-job status into summary files.

This should wait until the Slurm MST adapter has settled.

## Important Design Decisions

1. Use MST finder, not `benchmark_serving.py`, as the fixed-rate executor.
   - MST finder already handles the current workload YAML, tokenizer/model context resolution, chat/completions response compatibility, and server metrics.
   - `benchmark_serving.py` should serve as the reference for power metric formulas.

2. Keep plan generation side-effect-free.
   - It should read artifacts and write a YAML plan only.
   - No vLLM launch, no trial execution.

3. Keep rounded MST profiling distinct from exhaustive sweeps.
   - One result per model is the default.
   - Sweeps are explicit because they are expensive.

4. Preserve raw power traces when practical.
   - For long runs, consider writing compressed JSONL or CSV instead of embedding a large array in summary JSON.

5. Keep the default rounded rate at or below MST.
   - Summaries must show both `mst_rate` and `profile_request_rate`.
   - This prevents misreading a human-rounded profiling rate as the exact MST.

6. Reuse warm servers within a matching model/server signature.
   - This reduces launch overhead and better reflects steady serving behavior.
   - Sweep rates should run in ascending order by default.

7. Measure idle baseline by default.
   - Report total GPU energy and incremental GPU energy above idle baseline.
   - This makes comparisons more interpretable across models with different static server power.

## Open Questions

1. How long should the idle-power baseline run?
   - Draft default: 30s.

2. Should idle baseline be measured once per server launch or before every fixed-rate trial?
   - Draft default: once per server launch, reused across sweep rates for the same server.

3. For sweep mode, should the default spacing be linear in request rate or logarithmic?
   - Linear is easier to compare.
   - Log spacing is more efficient when MST differs by an order of magnitude.

4. How much raw trace should be retained by default?
   - Full 25ms sampling traces are useful but can become large.
   - Summary-only output is smaller but less auditable.

5. For multi-GPU jobs, should energy be reported only as aggregate GPU energy or also normalized per GPU?

6. Should failed request classes like the gpt-oss Harmony stream parser error be excluded from energy-per-request denominators, treated as failed workload demand, or reported separately?
