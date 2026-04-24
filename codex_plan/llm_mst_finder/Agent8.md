# Implement configuration sweep support
## Goal

Separate:

$$\lambda^*(\theta)$$

from:
$$
\max_\theta \lambda^*(\theta)$$

where θ includes serving configuration:

```
max_num_seqs
max_num_batched_tokens
gpu_memory_utilization
max_model_len
tensor_parallel_size
pipeline_parallel_size
```
## Files

```
config_sweep.py
```
## First version

Do not auto-launch vLLM server yet. Start with externally supplied configs:

```YAML
server_configs:
  - name: seq128_tok4096
    metadata:
      max_num_seqs: 128
      max_num_batched_tokens: 4096
  - name: seq256_tok8192
    metadata:
      max_num_seqs: 256
      max_num_batched_tokens: 8192
```
The user manually starts the server for each config, and the profiler records config metadata.

## Explicitly out of v1

Do not add a server launcher, server restart, health-check loop, or cluster/job-manager integration. Those concerns are environment-specific and should remain manual until the estimator is correct.

## Config-sweep logic

For each server config:

```
run hybrid search
save result
classify bottleneck
```

Then report:

```
best config by no-drift throughput
best config by SLO throughput
best config by output tok/s
best config by energy consumption per token
```

## Local consistency constraints

+ Config sweep results are only comparable when workload, SLOs, duration, and search settings match; validate and report mismatches.
+ Treat each supplied config as metadata. Do not infer missing serving flags.
+ Follow `Rules.md` for `/local/scratch/a/shi676/.venv`, `PYTHONPATH`, and fail-fast behavior.
