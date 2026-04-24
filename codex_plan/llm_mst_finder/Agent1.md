# Extract and modularize vLLM benchmark primitives
## Goal

Adapt vLLM’s official benchmark implementation into reusable components, without copying unnecessary structured-output-specific logic.

The official example already defines:

```
BenchmarkMetrics with TTFT, TPOT, ITL, and E2E fields;
SampleRequest;
get_request() for request-rate generation using Gamma/Poisson intervals;
calculate_metrics() for aggregate statistics;
benchmark() for async request dispatch.
```
The customized adapatation has added GPU power monitoring in `profiler/benchmark_serving.py`.

## Tasks

Create:

```
__init__.py
vllm_compat.py
request_client.py
loadgen.py
records.py
trial_runner.py
```
### Required abstractions
```Python
@dataclass
class SampleRequest:
    prompt: str
    prompt_len: int
    expected_output_len: int
    extra_body: dict | None = None
    metadata: dict = field(default_factory=dict)
```

```Python
@dataclass
class RequestRecord:
    request_id: str
    trial_id: str
    scheduled_send_ts: float
    actual_send_ts: float | None
    first_token_ts: float | None
    end_ts: float | None
    success: bool
    error: str | None

    prompt_len: int
    expected_output_len: int
    actual_output_len: int | None

    ttft_s: float | None
    e2e_s: float | None
    tpot_s: float | None
    itl_s: list[float]
    output_token_timestamps: list[float]
```
### Required load modes

Implement both:

```Python
OpenLoopLoadGenerator
ClosedLoopLoadGenerator
```

Open-loop mode (mostly reusing existing `get_request()`):


+ external arrival rate is fixed;
+ no concurrency throttling by default;
+ supports Poisson/Gamma arrivals like vLLM’s get_request(), where burstiness = 1 means Poisson/exponential inter-arrival time.
+ optional --safety-max-outstanding aborts trial if outstanding requests exceed threshold, but must not silently throttle.


Closed-loop mode:


+ fixed concurrency N;
+ each worker sends next request only after previous completion;
+ optional think time, default 0.


Important: do not use max_concurrency to estimate open-loop max sustainable rate. vLLM’s benchmark notes that when `--max-concurrency` is combined with `--request-rate`, actual request rate may be lower than specified if the server cannot keep up. That is useful for safety, but invalid for open-loop stability measurement unless reported explicitly.

## Local consistency constraints

+ Implement under `profiler/llm_mst_finder/`.
+ Reuse current profiler code as reference material, not by importing `benchmark_serving.py`.
+ `request_client.py` may adapt behavior from `profiler/backend_request_func.py`, but internal implementation errors must raise. Only real request/API failures become failed `RequestRecord`s.
+ `trial_runner.py` is the single orchestrator used by both `run-trial` modes and later search.
+ Follow `Rules.md` for `/local/scratch/a/shi676/.venv`, `PYTHONPATH`, and fail-fast behavior.
