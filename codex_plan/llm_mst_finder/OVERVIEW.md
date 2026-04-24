Implement a Python package `llm_mst_finder` for estimating the max sustainable request rate of an OpenAI-compatible vLLM serving endpoint.

Reuse the design of vLLM’s official benchmark_serving_structured_output.py where appropriate, especially async request dispatch, request-rate generation, TTFT/TPOT/ITL/E2E measurement, and goodput/SLO calculation. However, do not just wrap the benchmark. Add open-loop and closed-loop load generators, per-request JSONL records, Prometheus `/metrics` polling, fixed-window aggregation, stability classification, bottleneck classification, hybrid search, and report generation.

The final estimator must use open-loop request-rate trials for max sustainable external arrival rate. Closed-loop concurrency sweep is only a scouting phase. Do not silently throttle open-loop request rate with max_concurrency. If a safety outstanding cap is used, abort and classify the trial as unstable/safety-aborted.

Implement milestones:
1. run one open-loop or closed-loop trial and save request/server/window data;
2. analyze one trial for stability and bottleneck class;
3. hybrid search for max sustainable rate;
4. optional config sweep over max_num_seqs and max_num_batched_tokens metadata;
5. Markdown/JSON report and plots.

Use robust windowed drift criteria:
- arrival vs completion rate;
- outstanding request slope;
- TTFT p90/p99 drift;
- TPOT p90/p99 drift;
- server waiting/running/swapped requests;
- KV-cache usage;
- preemption/swap metrics if available;
- error/timeout rate;
- TTFT/TPOT/E2E SLOs.

Write unit tests for load generation, windowing, stability classification, search convergence, and Prometheus parsing.

## Target outcome

Build a CLI tool, for example:

```bash
python -m llm_mst_finder.profile \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --workload-config workloads/sharegpt_512_128.yaml \
  --search-mode hybrid \
  --rate-precision 0.03 \
  --trial-min-duration-s 90 \
  --trial-max-duration-s 300 \
  --window-s 10 \
  --slo-ttft-ms 2000 \
  --slo-tpot-ms 80 \
  --max-error-rate 0.01 \
  --server-metrics-url http://127.0.0.1:8000/metrics \
  --output-dir results/run_001
```  

The tool should output:

```
max_no_drift_request_rate:  X req/s
max_slo_satisfying_rate:   Y req/s
closed_loop_plateau:       Z req/s, A output tok/s
bottleneck_class:          scheduler_cap | prefill_compute | decode_bandwidth | kv_cache | slo_limited | mixed
confidence:                high | medium | low
```

and save:

```
results/run_001/
  config.yaml
  search_trace.json
  trials/
    trial_000_closedloop_N16/
      request_records.jsonl
      server_metrics.jsonl
      windows.csv
      summary.json
      plots/
    trial_001_openloop_r8.0/
      ...
  final_report.md
  final_report.json
```

## High-level architecture

Implement the package as:

```
profiler/llm_mst_finder/
  __init__.py
  cli.py
  workload.py
  request_client.py
  loadgen.py
  metrics_polling.py
  records.py
  windowing.py
  stability.py
  search.py
  bottleneck.py
  reporting.py
  plotting.py
  vllm_compat.py
tests/llm_mst_finder/
  test_windowing.py
  test_stability.py
  test_search.py
  test_prometheus_parser.py
  test_loadgen.py
```

The architecture should be:

```
CLI
 └── SearchController
      ├── WorkloadProvider
      ├── TrialRunner
      │    ├── LoadGenerator
      │    ├── RequestClient
      │    └── MetricsPoller
      ├── WindowAggregator
      ├── StabilityClassifier
      ├── BottleneckClassifier
      └── Reporter
```

### Minimal CLI surface
```bash
# One open-loop trial
python -m llm_mst_finder.cli run-trial \
  --mode open-loop \
  --request-rate 8 \
  --duration-s 180 \
  --window-s 10 \
  --base-url http://127.0.0.1:8000 \
  --model MODEL \
  --workload-config workloads/fixed_512_128.yaml \
  --server-metrics-url http://127.0.0.1:8000/metrics \
  --output-dir results/trial_r8

# One closed-loop trial
python -m llm_mst_finder.cli run-trial \
  --mode closed-loop \
  --concurrency 64 \
  --duration-s 120 \
  ...

# Adaptive search
python -m llm_mst_finder.cli search \
  --search-mode hybrid \
  --rate-precision 0.03 \
  ...

# Analyze existing trial
python -m llm_mst_finder.cli analyze \
  --trial-dir results/trial_r8

# Generate report
python -m llm_mst_finder.cli report \
  --result-dir results/run_001
```
### Minimal final JSON schema
```JSON
{
  "workload": {
    "name": "fixed_512_128",
    "num_requests": 20000,
    "prompt_len_summary": {},
    "output_len_summary": {}
  },
  "server_config": {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "max_num_seqs": 256,
    "max_num_batched_tokens": 8192,
    "chunked_prefill": true
  },
  "search_result": {
    "max_no_drift_request_rate": 10.8,
    "max_slo_satisfying_request_rate": 9.6,
    "rate_precision": 0.03,
    "confidence": "high"
  },
  "closed_loop": {
    "peak_request_throughput": 12.1,
    "peak_output_token_throughput": 1450.0
  },
  "bottleneck": {
    "class": "scheduler_cap",
    "evidence": [
      "num_running near max_num_seqs",
      "num_waiting drifted upward",
      "TPOT stationary",
      "KV cache usage below 80%",
      "preemptions absent"
    ]
  },
  "trials": []
}
```