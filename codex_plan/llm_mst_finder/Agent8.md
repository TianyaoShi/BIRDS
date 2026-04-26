# Agent 8 Retired

Do not dispatch Agent 8.

The original Agent 8 scope was a metadata-only `config-sweep` module. That scope is no longer part of MST finder. Server configuration search belongs to an upper orchestration layer that owns GPU allocation, live serving instances, server restarts, dataset/model scheduling, and non-default vLLM flags.

MST finder should still record and report server metadata such as:

```text
model
max_model_len
max_num_seqs
max_num_batched_tokens
gpu_memory_utilization
tensor_parallel_size
pipeline_parallel_size
```

But metadata recording is provenance, not configuration search. Reporting may compare result directories produced by an external orchestrator only when metadata proves those results are comparable.
