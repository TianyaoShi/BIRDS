# Implement workload specification and sampling
## Goal

Make workload definition explicit and reproducible.

## Files
```
workload.py
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

## Local consistency constraints

+ Implement under `profiler/llm_mst_finder/workload.py`.
+ Malformed YAML, unknown workload types, missing files, or impossible length specs should raise immediately.
+ Default unit tests must use synthetic local fixtures and must not download tokenizer or dataset artifacts.
+ Follow `Rules.md` for `/local/scratch/a/shi676/.venv`, `PYTHONPATH`, and fail-fast behavior.
