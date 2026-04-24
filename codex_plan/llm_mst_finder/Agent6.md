# Implement bottleneck classifier
## Goal

Classify why a trial failed or saturated.

This is essential for your case because observed max throughput may be limited by the vLLM scheduler cap, not by compute, memory bandwidth, or KV capacity.

vLLM’s optimization docs state that chunked prefill batches decode first, then uses the remaining `max_num_batched_tokens` budget for prefill; smaller `max_num_batched_tokens` improves ITL, while larger values improve TTFT and throughput. The same docs discuss preemption when there is not enough KV cache and recommend changing `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens`, or parallelism settings to alter KV pressure.

## Files
```
bottleneck.py
Bottleneck classes
scheduler_cap
prefill_compute_or_token_budget
decode_bandwidth
kv_cache
slo_limited
client_limited
mixed
unknown
```
## Classification rules
### Scheduler/config cap

Likely if:
```
num_running_mean ≈ max_num_seqs
num_waiting_mean or TTFT drifts upward
TPOT remains flat
kv_cache_usage_mean is below threshold, e.g. < 0.85
preemptions_delta ≈ 0
GPU utilization not clearly saturated
```
Interpretation:

`max_num_seqs` is lower than hardware-feasible active batch size.
### Prefill/token-budget wall

Likely if:
```
TTFT or queue time drifts upward
TPOT remains flat
prompt_tok_s plateaus
scheduled token budget near max_num_batched_tokens if observable
long prompts dominate workload
```
Interpretation:

prefill progress is insufficient; `max_num_batched_tokens` may be too low.
### Decode bandwidth wall

Likely if:
```
TPOT/ITL drifts upward
generation_tok_s plateaus
num_running high
KV not necessarily full
preemptions low
GPU memory bandwidth high if available
```
### KV-cache wall

Likely if:
```
kv_cache_usage_max near 1.0
num_swapped > 0 or preemptions increase
E2E p99 spikes
TTFT and TPOT may both degrade
```
### SLO-limited

Likely if:
```
stationary queue
stable completion rate
but TTFT or TPOT above SLO
```
### Client-limited

Likely if:
```
closed-loop sweep does not saturate server
client CPU/network saturated
actual arrival rate < configured arrival rate
many client-side scheduling delays
```

## Local consistency constraints

+ Bottleneck output must include evidence strings tied to observed window/server/request metrics.
+ Do not infer server config values that were not measured or supplied as metadata.
+ If a safety cap or client scheduling delay invalidates the configured open-loop arrival rate, prefer `client_limited` or `aborted_safety` over a server bottleneck class.
+ Follow `Rules.md` for `/local/scratch/a/shi676/.venv`, `PYTHONPATH`, and fail-fast behavior.
