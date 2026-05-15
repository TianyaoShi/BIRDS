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

selection:
  models: []
  workloads: []
  experiment_ids: []
  explicit_request_rates: []
  sweep:
    enabled: false
    models: []
    experiment_ids: []
    max_steps: 20

defaults:
  duration_s: 180
  idle_baseline_s: 30
  traffic_warmup_s: 30
  cooldown_s: 15
  repeats: 1
  repeat_cooldown_s: 15
  warmup_each_repeat: false
  metrics_interval_s: 1.0
  window_s: 10.0
  gpu_monitor_interval_s: 0.025
  gpu_monitor_truncate_s: 5
  monitor_clock: false
  request_timeout_s: 21600
  safety_max_outstanding: null

rounding:
  mst_mode: floor_decimal
  mst_decimal_places: 2
  sweep_mode: floor_preferred
  preferred_steps: [0.05, 0.1, 0.2, 0.25, 0.5, 1.0]
  minimum_rate: 0.1

jobs:
  - id: qwen3-8b-sharegpt-r4_25
    source_experiment_id: single-gpu-model-loop-...
    source_result_dir: results/mst/qwen-qwen3-8b/live-sharegpt-workload-context/server-...
    model: Qwen/Qwen3-8B
    workload: experiments/live_sharegpt_workload_context.yaml
    endpoint: /v1/chat/completions
    request_rate: 4.25
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
      rounding_step: 0.25
```

The generated plan should be deterministic and should include enough launch/search context that it can be executed later without depending on mutable orchestrator state.

## Rounding Policy

Use different rounding semantics for one-shot MST profiling and sweep
profiling.

For `mst-rounded`, the selected MST should be floored to fixed decimal
precision, not snapped to a preferred human step. The default should be
two decimal places:

```text
profile_rate = floor(MST * 100) / 100
```

This keeps the follow-up profiling rate at or below the measured SLO-valid MST
without unnecessarily moving a single-trial energy point away from the actual
MST. Clamp to `minimum_rate` only when the positive MST is below the smallest
useful open-loop rate.

Examples:

- `0.083` -> `0.1`, because positive rates below the minimum useful trial rate are clamped
- `0.619` -> `0.61`
- `1.849` -> `1.84`
- `4.379` -> `4.37`
- `17.499` -> `17.49`

Implementation note: `--rounding-policy floor_decimal` is the default for
`mst-rounded`; `--rounding-policy floor_preferred` remains available for the
older preferred-step behavior. JSON policies can set `mst_mode`,
`mst_decimal_places`, `sweep_mode`, `preferred_steps`, and `minimum_rate`.

For `sweep`, preferred-step rounding is still useful because the goal is to
construct a compact, readable grid from low load to MST. Recommended default:

1. Choose a display quantum based on the MST scale.
   - `< 2 req/s`: use `0.1` or `0.05`
   - `2-5 req/s`: use `0.25` or `0.2`
   - `5-10 req/s`: use `0.5`
   - `>= 10 req/s`: use `1.0`

2. Floor to that quantum.

3. Clamp to `minimum_rate` only when the MST is positive but below the smallest useful open-loop rate.

4. Prefer more total sweep points as long as the total step count stays within the configured cap, so that profiling and comparisons keep useful resolution.

Sweep examples:

- `0.083` -> `0.1`, because positive rates below the minimum useful trial rate are clamped, step = `0.05`
- `0.61` -> `0.6`, step = `0.05`
- `1.84` -> `1.8`, step = `0.1`
- `4.37` -> `4.25`, step = `0.25`
- `17.4` -> `17`, step = `1.0`

This keeps rates readable while avoiding profiling above the measured MST.

## Sweep Policy

For selected models, generate up to 20 fixed-rate trials from low load to rounded MST.

The sweep grid should be derived from the same floor-rounding policy used for the single rounded-MST profile. The rounding policy chooses a human-preferable step size for the model's MST scale; sweep mode then enumerates that grid from one step through the floored MST.

Recommended algorithm:

1. Select the finest preferred step that gives at most `max_steps` nonzero rates up to MST.
   - Candidate steps come from the rounding policy, e.g. `[0.05, 0.1, 0.2, 0.25, 0.5, 1.0]`.
   - Prefer the smallest step whose `floor(MST / step)` is `<= max_steps`.
   - If all fine steps exceed the cap, move to the next coarser step.

2. Compute `rounded_mst = floor(MST / step) * step`.

3. Generate all nonzero multiples of `step` through `rounded_mst`.

4. Deduplicate floating-point artifacts and preserve stable decimal formatting.

5. Run rates in ascending order while reusing the same warm vLLM server for a matching model/server signature.

Examples:

```text
MST 0.61, step 0.05, rounded MST 0.60:
0.05, 0.10, 0.15, ..., 0.60

MST 1.84, step 0.10, rounded MST 1.80:
0.10, 0.20, 0.30, ..., 1.80

MST 4.37, step 0.25, rounded MST 4.25:
0.25, 0.50, 0.75, ..., 4.25

MST 17.4, step 1.0, rounded MST 17.0:
1.0, 2.0, 3.0, ..., 17.0
```

For very low MST values below the minimum useful open-loop rate, generate one trial at `minimum_rate` and mark it as a clamped rate in the plan metadata.

Optional future mode:

- logarithmic spacing for models with a large dynamic range, especially if the target rate exceeds 20 req/s.

## Branch TODO: Natural Output Comparison Mode

The current MST and energy profiling path uses workload-specified output lengths
as an explicit generation budget. For `sampling.output_len.mode=from_dataset`,
the dataset reference/completion text is tokenized and sent as request
`max_tokens`; with `ignore_eos: true`, actual output length is expected to stay
close to that requested budget. This is appropriate for controlled serving
efficiency comparisons because prompt and output token demand are articulated
for every model.

This is not the right mode for comparing natural verbosity or reasoning-length
behavior between thinking and non-thinking variants. Add a separate small
comparison branch for that purpose:

- fix the input prompt dataset, GPU platform, server config, sampling settings,
  and request rate across the models being compared
- do not derive per-request `max_tokens` from dataset answer length
- either omit `max_tokens` if the serving API and client support it, or use a
  generous common safety cap that is clearly reported as a cap rather than a
  target
- set `ignore_eos: false` so each model can stop naturally
- report actual output length distributions as first-class outcomes, including
  p50/p90/p95/p99 and totals
- compare energy both per request and per actual total token, while clearly
  labeling the result as behavior-inclusive rather than controlled-token
  efficiency

Interpretation rule:

- controlled output-length profiling is best for architecture/kernel/hardware
  efficiency under the same token workload
- natural-output profiling is best for real-world model behavior and cost,
  because thinking models may spend additional tokens on reasoning and
  non-thinking variants may stop earlier or later

Do not mix these two result types in one decisive energy comparison table
without labeling them separately.

## Explicit Comparison Policy

Allow the user to specify comparison selections directly in the plan YAML:

```yaml
selection:
  models:
    - Qwen/Qwen3-4B-Instruct-2507
    - Qwen/Qwen3-8B
  workloads:
    - live_sharegpt_workload_context
  explicit_request_rates: [0.5, 1.0, 2.0, 4.0]
```

For sweep selection:

```yaml
selection:
  sweep:
    enabled: true
    models:
      - Qwen/Qwen3-4B-Instruct-2507
      - Qwen/Qwen3-8B
    max_steps: 20
```

The CLI should still support quick filters, but YAML selection should be the preferred review/edit path for human users. CLI filters should either:

- seed the generated YAML selection fields, or
- override them only when explicitly requested.

The planner should validate that selected models/workloads/experiment IDs exist in the source orchestrator run and have succeeded MST results unless the user passes an explicit override allowing missing MST.

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
3. Run an unmeasured pre-measurement traffic warmup at the same request rate,
   workload, endpoint, and launch settings as the measured trial.
4. Start GPU power monitors for the assigned GPU IDs.
5. Run one fixed-rate `llm_mst_finder.cli run-trial --mode open-loop` measured trial.
6. Stop power monitors after the measured trial.
7. Write one result directory per energy job/repeat.
8. Continue to the next job on failure.
9. Emit `summary.json` and `summary.md`.

For sweep jobs, group trials by model/server signature, reuse the same warm vLLM server, and execute rates in ascending order.

### Pre-Measurement Traffic Warmup

The current `warmup_s` behavior is only an idle-power baseline; it starts GPU
monitors and sleeps without sending requests. Add a separate traffic warmup
phase before the measured energy trial.

Required behavior:

- Run after vLLM readiness and after the idle baseline.
- Use the exact measured-trial workload, endpoint, model, request rate, request
  timeout, and safety limit.
- Do not include warmup request records, latency windows, server metrics, or GPU
  power samples in the reported energy trial artifacts.
- Write warmup artifacts under a separate subdirectory, for example
  `jobs/<job_id>/warmup/`, or discard them only if the subprocess succeeded and
  logs are preserved elsewhere.
- If warmup fails, fail the energy job before measured power monitoring starts.
- Default `traffic_warmup_s` should be nonzero for managed experiments; `30s` is
  a reasonable first default, with `0` disabling it for smoke tests.
- For repeated trials on the same server, apply traffic warmup before the first
  measured repeat by default. Optionally support `warmup_each_repeat: true` for
  highly variable workloads.

This separates three different concepts that should not share one field name:

- readiness: server API responds, usually `/v1/models`
- idle baseline: no traffic, used for incremental energy subtraction
- traffic warmup: unmeasured requests used to bring serving into a steadier
  cache/CUDA graph/scheduler state before the measured trial

## Run-Trial Reuse and External Energy Monitoring

Preferred implementation path:

1. Reuse `llm_mst_finder.cli run-trial` unchanged for traffic generation and latency artifacts.
   - Do not enable GPU energy collection by default during MST search.
   - Do not require `TrialRunner` to know about power monitoring for the first implementation.
   - The energy executor should call `run-trial --mode open-loop --request-rate ...` as a subprocess and treat the resulting `summary.json`, `request_records.jsonl`, `server_metrics.jsonl`, and `windows.csv` as normal trial artifacts.

2. Wrap the `run-trial` subprocess with energy monitoring in `energy_profiler.executor`.
   - Start `GPUMonitor` instances in the executor before launching the fixed-rate trial.
   - Stop monitors after the trial process exits.
   - Read the trial `summary.json` to compute energy per request/token.
   - Write `gpu_power.json` and `energy_summary.json` beside the trial artifacts.

3. Measure idle baseline outside MST finder.
   - After vLLM readiness and before traffic, start monitors for `idle_monitor_duration_s`.
   - Store idle power stats separately.
   - Use idle stats to compute incremental energy above idle for later traffic trials on the same server.

4. Run traffic warmup outside the measured monitor window.
   - Invoke the same `run-trial` path with a warmup trial ID and separate output
     directory.
   - Set `duration_s=traffic_warmup_s`.
   - Do not merge warmup artifacts into measured summaries.

5. Reuse `profiler/gpu_monitor.py`.
   - Current `GPUMonitor` returns:
     - `avg_power_mw`
     - `power_stats`
     - `power_trace_mw`
     - optional `clock_trace_mhz`
   - Keep milliwatts internally and expose watts/joules in summary fields with clear names.

6. Avoid using `benchmark_serving.py` as the executor.
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
- per-request and total-token-normalized metrics from aggregate energy

Do not use output-token-only energy as a primary metric for continuous-batching serving. Prompt prefill and decode work overlap across requests, so attributing total GPU energy only to generated tokens is misleading. Preferred token-normalized metrics:

- `energy_per_total_token_j`
- `incremental_energy_per_total_token_j`
- optionally `energy_per_request_j`

Generated-token-only metrics may be emitted only as an explicitly labeled compatibility field for older reports, and should be omitted by default.

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
      warmup/
        summary.json
        request_records.jsonl
        server_metrics.jsonl
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
- repeat index and repeat count when repeats are enabled
- raw power trace or a path to it

## Repeat Policy

Energy profiling should support repeats because single trials are sensitive to
cold-start effects, transient cluster noise, and workload randomness.

Recommended plan fields:

```yaml
defaults:
  repeats: 3
  repeat_cooldown_s: 15
  traffic_warmup_s: 30
  warmup_each_repeat: false
```

Execution behavior:

- Expand each logical energy job into `repeats` measured trials at execution
  time or during plan generation, but preserve a stable logical job ID for
  aggregation.
- Store repeat artifacts separately, for example
  `jobs/<job_id>/repeat_001/`, `repeat_002/`, and `repeat_003/`.
- Reuse the same vLLM server across repeats when the server signature matches.
- Run traffic warmup before the first repeat by default; if
  `warmup_each_repeat` is enabled, run warmup before every measured repeat.
- Compute per-repeat energy summaries and aggregate median/mean/stdev/min/max
  across repeats in the top-level job summary.
- Treat partial repeat failure as a failed job unless a future explicit
  `min_successful_repeats` setting is added.

For cold-start analysis, keep repeat order in metadata rather than only
reporting aggregate statistics. The first repeat can be compared against later
repeats to quantify how much warmup/reuse changed the energy result.

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
  --rounding-policy '{"mst_mode":"floor_decimal","mst_decimal_places":2,"sweep_mode":"floor_preferred"}'

python -m energy_profiler.cli plan-from-orchestrator \
  --orchestrator-run-root results/orchestrator/<run_id> \
  --output-plan experiments/energy/<plan_id>.yaml \
  --mode sweep \
  --selection-yaml experiments/energy/sharegpt_selection.yaml

python -m energy_profiler.cli plan-explicit \
  --orchestrator-run-root results/orchestrator/<run_id> \
  --output-plan experiments/energy/<plan_id>.yaml \
  --selection-yaml experiments/energy/sharegpt_selection.yaml
```

Example selection YAML:

```yaml
selection:
  models:
    - Qwen/Qwen3-4B-Instruct-2507
    - Qwen/Qwen3-8B
  explicit_request_rates: [0.5, 1.0, 2.0, 4.0]
  sweep:
    enabled: true
    models:
      - Qwen/Qwen3-8B
    max_steps: 20
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
- Apply YAML-native selection fields for models, workloads, experiment IDs, explicit rates, and sweep subsets.
- Generate deterministic plan YAML.
- Add dry-run rendering.

Focused tests:

- succeeded jobs are included, failed/skipped jobs are excluded
- `max_slo` vs `max_no_drift` rate source selection
- rounding behavior around low and high rates
- `mst-rounded` floors to fixed decimal precision and does not snap to preferred sweep steps
- sweep generation deduplicates rates and caps at 20 steps
- sweep generation orders rates ascending for server reuse
- explicit plan validates YAML-native model/workload/experiment selection

### Phase 2: External Power Monitoring Wrapper

- Add energy-profiler-owned monitor wrappers around `profiler/gpu_monitor.py`.
- Add idle baseline collection after vLLM readiness.
- Add unmeasured pre-measurement traffic warmup before measured power monitoring.
- Run `llm_mst_finder.cli run-trial` as an unchanged subprocess for the fixed-rate traffic trial.
- Stop monitors after the subprocess exits.
- Read the trial `summary.json`.
- Write `gpu_power.json` and `energy_summary.json` in the energy job directory.
- Keep MST search and ordinary `llm_mst_finder.cli run-trial` behavior unchanged when invoked outside the energy profiler.

Focused tests:

- fake monitor produces deterministic idle and traffic energy summaries
- traffic warmup artifacts are isolated from measured energy artifacts
- traffic warmup failure fails the energy job before power monitoring starts
- run-trial command construction remains a plain MST finder command with no energy-specific CLI args
- zero successful requests avoids division by zero and marks per-request energy as null
- generated-token-only energy is omitted or explicitly marked as a non-primary compatibility metric

### Phase 3: Energy Executor

- Reuse local orchestrator launch lifecycle to start vLLM.
- Run fixed-rate `llm_mst_finder.cli run-trial --mode open-loop` under the external power-monitor wrapper.
- Capture logs and job status.
- Continue on per-job failure.
- Write energy run summary.

Focused tests:

- command construction includes fixed `--request-rate`
- launch fields from plan are honored
- result directory overwrite policy is explicit
- failed jobs do not block later jobs
- repeat execution writes isolated repeat directories and aggregate repeat summaries
- warmup runs once per logical job by default and can be configured per repeat

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

8. Avoid output-token-only energy as a primary metric.
   - Continuous batching mixes prompt and decode work across requests.
   - Use request-normalized and total-token-normalized energy instead.

9. Prefer YAML-native selection for human-edited plans.
   - CLI filters are useful for quick generation, but selected models/rates/sweeps should be visible and editable in the plan YAML.

10. Use fixed decimal flooring for one-shot MST energy trials.
   - Preferred-step rounding is for sweep grid readability.
   - `mst-rounded` should stay as close as possible to the measured MST while
     never exceeding it.

11. Support repeats as first-class energy measurements.
   - Repeats reduce sensitivity to one unlucky run and reveal cold-start drift.
   - Summaries should expose both per-repeat values and aggregate statistics.

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
   - Both are useful.

6. Should failed request classes like the gpt-oss Harmony stream parser error be excluded from energy-per-request denominators, treated as failed workload demand, or reported separately?
   - They should still consume energy so do not exclude them from the denominator, but they should be marked as failed requests in the summary so users can interpret energy-per-successful-request vs energy-per-total-request.

7. Should traffic warmup duration be fixed in seconds or sized by request count?
   - Draft default: seconds, because it is simple and matches trial duration.
   - A future request-count mode may be better for very low request rates.

8. Should repeat expansion happen in the plan generator or executor?
   - Plan-time expansion makes Slurm arrays and artifact paths explicit.
   - Executor-time expansion keeps the YAML smaller but requires richer state
     handling.
