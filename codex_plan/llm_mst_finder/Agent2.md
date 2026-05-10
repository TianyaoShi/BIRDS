# Implement workload specification and sampling
## Goal

Make workload definition explicit and reproducible.

## Files
```
workload.py
model_context.py
workloads/*.yaml
```
## YAML schema

Example:
```YAML
name: sharegpt_512_128
dataset:
  type: sharegpt
  path: data/sharegpt.json
tokenizer: meta-llama/Llama-3.1-8B-Instruct
sampling:
  seed: 1
  num_requests: 20000
  conversation_mode: single_turn
  prompt_len:
    mode: fixed_or_bucketed
    target_mean: 512
  output_len:
    mode: fixed
    value: 128
request:
  temperature: 0.0
  ignore_eos: true
  stream: true
  extra_body:
    max_tokens: 128
context_policy:
  max_model_len: 4096
  tokenizer_source: vllm_model_config
  tokenizer: meta-llama/Llama-3.1-8B-Instruct
  over_limit: fail
```
Support at least:

```
synthetic-fixed
synthetic-distribution
jsonl
```

Add `sharegpt` after the synthetic and JSONL paths are working and tested. Reuse the existing benchmark sampling ideas only if they fit the new `SampleRequest` contract cleanly.

Conversation dataset modes:

```YAML
sampling:
  conversation_mode: single_turn | multi_turn_prefix | session_replay
  turn_selection: first_valid | random_valid | all_valid
  include_assistant_history: true
  min_prompt_turns: 1
  max_prompt_turns: 16
traffic:
  session_ordering: preserve_within_session
  session_interleaving: shuffled_sessions | round_robin
  per_session_think_time_s: 0
```

Mode rules:

- `single_turn` is the default for backward compatibility: first ordered user/human -> assistant/gpt pair only.
- `multi_turn_prefix` materializes independent prefix -> next assistant requests. It is not session-realistic unless traffic order is also preserved.
- `session_replay` is the default intended mode for realistic chatbot traffic once implemented. It emits all valid assistant-target turns, preserves order within each conversation, and interleaves sessions across the stream.
- `session_replay` samples must include `session_id`, `turn_index`, `preserve_order_key`, `preserve_order_index`, and `conversation_mode` metadata.

Each sample must include:

```
prompt
prompt_len
expected_output_len
sampling params
metadata
```
For `session_replay`, metadata is part of the scheduling contract, not optional decoration.
## Acceptance criteria


+ identical seed gives identical workload sequence;
+ actual tokenized prompt lengths are saved;
+ output length control supports ignore_eos=True for profiling when fixed OSL is needed.
+ context validation uses the serving/model-compatible tokenizer, not a convenience tokenizer such as whitespace;
+ default over-limit behavior is fail-fast before dispatch;
+ explicit `skip_sample` and `truncate_prompt` policies record skipped/truncated counts and source indexes.
+ chat session replay preserves per-session turn order under deterministic interleaving and records the conversation policy in workload/trial/report metadata.

## Context compatibility design

Add a pre-trial validation step:

```text
model_prompt_len = len(model_tokenizer.encode(prompt))
requested_total = model_prompt_len + expected_output_len
requested_total <= max_model_len
```

Supported policy:

```YAML
context_policy:
  max_model_len: 4096
  tokenizer_source: vllm_model_config | explicit | workload_tokenizer
  tokenizer: meta-llama/Llama-2-7b-chat-hf
  over_limit: fail | skip_sample | truncate_prompt
  truncation_side: left | right
```

`workload_tokenizer` is acceptable for synthetic/offline tests but should not be the production default for real model profiling. If `tokenizer_source=vllm_model_config` is selected and the tokenizer cannot be loaded locally through vLLM-compatible utilities, raise with a clear message. Do not download tokenizer artifacts in default tests.

## Local consistency constraints

+ Implement under `profiler/llm_mst_finder/workload.py`.
+ Put reusable model/context helpers in `profiler/llm_mst_finder/model_context.py`.
+ Malformed YAML, unknown workload types, missing files, or impossible length specs should raise immediately.
+ Unknown context policy values, missing `max_model_len`, or unavailable model-compatible tokenizer should raise before a trial starts.
+ Default unit tests must use synthetic local fixtures and must not download tokenizer or dataset artifacts.
+ Follow `Rules.md` for `/local/scratch/a/shi676/.venv`, `PYTHONPATH`, and fail-fast behavior.
