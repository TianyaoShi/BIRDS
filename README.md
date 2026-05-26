# Profiler Code Map

This published version keeps the core profiling and orchestration modules used
by the paper while omitting local planning artifacts, helper scripts, generated
results, and most experiment/config assets.

## Core File Tree

```text
profiler/
  gpu_monitor.py
  cpu_monitor.py
  benchmark_serving.py
  backend_request_func.py

  llm_mst_finder/
    cli.py
    trial_runner.py
    search.py
    workload.py
    request_client.py
    metrics_polling.py
    windowing.py
    analysis.py
    stability.py
    bottleneck.py
    reporting.py
    plotting.py
    records.py
    vllm_compat.py
    model_context.py

  local_orchestrator/
    cli.py
    manifest.py
    matrix.py
    planning.py
    scheduler.py
    lifecycle.py
    resources.py
    mst_adapter.py
    state_store.py
    models.py
    utils.py

  energy_profiler/
    cli.py
    planning.py
    executor.py
    reporting.py
    models.py

  output_quality_profiler/
    cli.py
    generation.py
    manifest.py
    materialization.py
    matrix.py
    mock_openai_server.py
    models.py
    scoring.py

  mst_analyzer/
    cli.py
    extract.py
    rules.py
    reporting.py
    config.py
    models.py

  slurm_orchestrator/
    cli.py
    planning.py
    state.py

tests/
  llm_mst_finder/
  local_orchestrator/
  energy_profiler/
  mst_analyzer/
  slurm_orchestrator/
```

## Profiler Modules

- `llm_mst_finder`: runs fixed-rate or closed-loop trials, searches for maximum sustainable throughput, classifies stability, and writes trial reports.
- `local_orchestrator`: expands experiment manifests, manages local GPU/port resources, launches vLLM servers, and runs MST searches across model/workload matrices.
- `energy_profiler`: consumes orchestrator MST outputs, creates reviewable energy profiling plans, runs fixed-rate energy trials, and summarizes GPU power and energy per request/token.
- `output_quality_profiler`: enforces the ShareGPT/WildChat response-quality profiling contract, including fixed decoding defaults, stratified materialization config validation, quality run manifest validation, and judge-score primitives.
- `mst_analyzer`: extracts and compares completed MST result directories using rule-based analysis and reporting.
- `slurm_orchestrator`: prepares Slurm-oriented orchestration state and planning for cluster execution.
- `gpu_monitor.py`: samples GPU power and optional clocks for energy accounting.
- `cpu_monitor.py`: CPU-side monitoring helper.
- `benchmark_serving.py` and `backend_request_func.py`: baseline serving benchmark utilities retained for direct benchmark compatibility.

The stripped publishable tree intentionally excludes repo-local skills,
operational helper scripts, and bundled experiment manifests.
