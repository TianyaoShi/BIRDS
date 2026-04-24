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

Each sample must include:

```
prompt
prompt_len
expected_output_len
sampling params
metadata
```
## Acceptance criteria


+ identical seed gives identical workload sequence;
+ actual tokenized prompt lengths are saved;
+ output length control supports ignore_eos=True for profiling when fixed OSL is needed.
+ context validation uses the serving/model-compatible tokenizer, not a convenience tokenizer such as whitespace;
+ default over-limit behavior is fail-fast before dispatch;
+ explicit `skip_sample` and `truncate_prompt` policies record skipped/truncated counts and source indexes.

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
