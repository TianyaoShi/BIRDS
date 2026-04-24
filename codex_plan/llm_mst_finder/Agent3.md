# Implement server metrics polling
## Goal

Poll /metrics during each trial and save raw + normalized server metrics.

vLLM exposes Prometheus-compatible metrics with names including:

```
vllm:num_requests_running
vllm:num_requests_waiting
vllm:num_requests_swapped
vllm:kv_cache_usage_perc
vllm:prompt_tokens_total
vllm:generation_tokens_total
vllm:request_success_total
vllm:time_to_first_token_seconds
vllm:inter_token_latency_seconds
vllm:e2e_request_latency_seconds
vllm:request_queue_time_seconds
vllm:request_prefill_time_seconds
vllm:request_decode_time_seconds
```

The vLLM docs explicitly describe these as the relevant server/request metrics and state that server-level metrics help explain request-level metrics.

## Files
```
metrics_polling.py
```
Keep Prometheus parsing in `metrics_polling.py` unless it grows large enough to justify a split.
## Output record
```Python
@dataclass
class ServerMetricSample:
    ts: float
    raw: dict[str, Any]

    num_running: float | None
    num_waiting: float | None
    num_swapped: float | None
    kv_cache_usage: float | None

    prompt_tokens_total: float | None
    generation_tokens_total: float | None
    request_success_total: float | None
    request_abort_total: float | None
```
## Polling interval

Default:
```
1 second
```
Save:
```
server_metrics.jsonl
```
Also compute counter deltas later in windowing.

## Local consistency constraints

+ Polling failures during a live trial are recorded as metric-poller failures and should lower confidence; parser/schema bugs should raise.
+ Missing optional vLLM metrics should be represented as `None`, not invented as zero.
+ Do not add server health checks, restarts, or retries in v1.
+ Follow `Rules.md` for `/local/scratch/a/shi676/.venv`, `PYTHONPATH`, and fail-fast behavior.
