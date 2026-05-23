# SLO Criteria Summary for Orchestrator Manifests

This note summarizes the effective latency SLO criteria encoded in the current
local and Slurm experiment manifests.

Scope:
- local orchestrator manifests on L40
- Slurm manifests on H100
- Slurm manifests on A100-40GB x4

Unless otherwise noted, the MST stability check uses:
- `ttft_slo_field: ttft_p90_ms`
- `tpot_slo_field: tpot_p90_ms`

That means the acceptance test is based on p90 TTFT and p90 TPOT, not mean or
median latency.

## 1. Workload Families

There are four SLO families in the manifests:

| Workload family | Typical manifests | Default TTFT SLO | Default TPOT SLO | Notes |
| --- | --- | ---: | ---: | --- |
| ShareGPT / WildChat chat | `*sharegpt*`, `*wildchat*` | 1000 ms | 150 ms | Interactive chat target |
| Code workloads | `*code_workloads*` | 5000 ms | 75 ms | Workload-specific TTFT override for `crosscodeeval` vs `repobench` |
| Reasoning | `*reasoning*` | 2000 ms on H100/A100, 5000 ms on L40 | 200 ms | Some 3B/4B buckets tighten TTFT/TPOT |
| LongBench buckets | `*longbench*` | `length_scaled` | 150 ms | TTFT is computed per request from prompt length |

## 2. Chat Workloads: ShareGPT and WildChat

Default chat SLO:
- TTFT <= `1000 ms`
- TPOT <= `150 ms`

### L40 local chat buckets

| Model bucket | Representative models | TTFT SLO | TPOT SLO | Source pattern |
| --- | --- | ---: | ---: | --- |
| `<3B` | Qwen3-0.6B, Qwen3-1.7B, Gemma E2B, Llama 3.2 1B | 250 ms | 50 ms | local ShareGPT/WildChat |
| `3B/4B` | Qwen3-4B, Gemma E4B, Llama 3.2 3B | 500 ms | 100 ms | local ShareGPT/WildChat |
| `7B/8B/13B/14B` | Qwen3-8B/14B, Llama-2 7B/13B, Llama 3.1 8B | 1000 ms | 150 ms | inherits default |
| `~30B` | Qwen3-30B-A3B, Qwen3-32B, Gemma 26B-A4B, Gemma 31B, CodeLlama-34B | 1000 ms | 150 ms | inherits default |
| `70B+` | Llama-2/3 70B, gpt-oss-120b | 1000 ms | 150 ms | inherits default |

### H100 Slurm chat buckets

| Model bucket | Representative models | TTFT SLO | TPOT SLO | Source pattern |
| --- | --- | ---: | ---: | --- |
| `3B/4B` | Qwen3-4B, Gemma E4B, Llama 3.2 3B | 500 ms | 100 ms | H100 ShareGPT/WildChat override |
| `7B/8B` | Qwen3-8B, Llama-2 7B, Llama 3.1 8B | 1000 ms | 150 ms | inherits default |
| `13B/14B` | Qwen3-14B, Llama-2 13B | 1000 ms | 150 ms | inherits default |
| `~30B` | Qwen3-30B-A3B, Qwen3-32B, Gemma 26B-A4B, Gemma 31B, CodeLlama-34B | 1000 ms | 150 ms | inherits default |
| `70B+` | Llama-2/3 70B, gpt-oss-120b | 1000 ms | 150 ms | inherits default |
| `235B TP8` | Qwen3-235B-A22B instruct/thinking | 1000 ms | 150 ms | inherits default |

Note: the H100 chat manifests intentionally exclude `<3B` models.

### A100-40GB Slurm chat buckets

| Model bucket | Representative models | TTFT SLO | TPOT SLO | Source pattern |
| --- | --- | ---: | ---: | --- |
| `<3B` | Qwen3-0.6B, Qwen3-1.7B, Gemma E2B, Llama 3.2 1B | 500 ms | 100 ms | A100 ShareGPT/WildChat override |
| `3B/4B` | Qwen3-4B, Gemma E4B, Llama 3.2 3B | 750 ms | 125 ms | A100 ShareGPT/WildChat override |
| `7B/8B` | Qwen3-8B, Llama-2 7B, Llama 3.1 8B | 1000 ms | 150 ms | inherits default |
| `13B/14B` | Qwen3-14B, Llama-2 13B | 1000 ms | 150 ms | inherits default |
| `~30B` | Qwen3-30B-A3B, Qwen3-32B, Gemma 26B-A4B, Gemma 31B, CodeLlama-34B | 1000 ms | 150 ms | inherits default |
| `70B+` | gpt-oss-120b, multi-GPU large models | 1000 ms | 150 ms | inherits default |

## 3. Code Workloads

Code manifests use workload-first SLOs:

- `*crosscodeeval*`: TTFT <= `1500 ms`, TPOT <= `75 ms`
- `*repobench*`: TTFT <= `5000 ms`, TPOT <= `75 ms`

The model bucket then tightens some small-model cases.

### L40 local code buckets

| Workload | Model bucket | TTFT SLO | TPOT SLO |
| --- | --- | ---: | ---: |
| CrossCodeEval | `<3B` | 500 ms | 40 ms |
| CrossCodeEval | `3B/4B` | 750 ms | 50 ms |
| CrossCodeEval | `8B/14B` | 1500 ms | 75 ms |
| CrossCodeEval | `30B+`, `70B+`, `gpt-oss-120b` | 1500 ms | 75 ms |
| RepoBench | `<3B` | 5000 ms | 50 ms |
| RepoBench | `3B/4B` | 5000 ms | 50 ms |
| RepoBench | `8B/14B` | 5000 ms | 75 ms |
| RepoBench | `30B+`, `70B+`, `gpt-oss-120b` | 5000 ms | 75 ms |

### H100 Slurm code buckets

| Workload | Model bucket | TTFT SLO | TPOT SLO |
| --- | --- | ---: | ---: |
| CrossCodeEval | `3B/4B` | 750 ms | 50 ms |
| CrossCodeEval | `8B/14B` | 1500 ms | 75 ms |
| CrossCodeEval | sparse `~30B` MoE (`A3B` / `A4B`) | 1500 ms | 75 ms |
| CrossCodeEval | `70B+`, dense `30B+`, `235B TP8` | 1500 ms | 75 ms |
| RepoBench | `3B/4B` | 5000 ms | 50 ms |
| RepoBench | `8B/14B` | 5000 ms | 75 ms |
| RepoBench | sparse `~30B` MoE (`A3B` / `A4B`) | 5000 ms | 75 ms |
| RepoBench | `70B+`, dense `30B+`, `235B TP8` | 5000 ms | 75 ms |

Note: the H100 code manifests intentionally exclude `<3B` models.

### A100-40GB Slurm code buckets

| Workload | Model bucket | TTFT SLO | TPOT SLO |
| --- | --- | ---: | ---: |
| CrossCodeEval | `<3B` | 500 ms | 40 ms |
| CrossCodeEval | `3B/4B` | 750 ms | 50 ms |
| CrossCodeEval | `8B` | 1500 ms | 75 ms |
| CrossCodeEval | `14B+` and multi-GPU large buckets | 1500 ms | 75 ms |
| RepoBench | `<3B` | 5000 ms | 50 ms |
| RepoBench | `3B/4B` | 5000 ms | 50 ms |
| RepoBench | `8B` | 5000 ms | 75 ms |
| RepoBench | `14B+` and multi-GPU large buckets | 5000 ms | 75 ms |

## 4. Reasoning Workloads

Reasoning defaults:
- H100 / A100: TTFT <= `2000 ms`, TPOT <= `200 ms`
- L40 local: TTFT <= `5000 ms`, TPOT <= `200 ms`

### L40 local reasoning buckets

| Model bucket | Representative models | TTFT SLO | TPOT SLO |
| --- | --- | ---: | ---: |
| `<3B` | Qwen3-0.6B, Qwen3-1.7B, Gemma E2B, Llama 3.2 1B | 3000 ms | 150 ms |
| `3B/4B` | Qwen3-4B, Gemma E4B, Llama 3.2 3B | 4000 ms | 175 ms |
| `8B/14B` | Qwen3-8B/14B, Llama 3.1 8B, gpt-oss-20b | 5000 ms | 200 ms |
| `30B+`, `70B+`, `gpt-oss-120b` | multi-GPU local reasoning buckets | 5000 ms | 200 ms |

### H100 Slurm reasoning buckets

| Model bucket | Representative models | TTFT SLO | TPOT SLO |
| --- | --- | ---: | ---: |
| `3B/4B` non-thinking | Qwen3-4B-Instruct, Gemma E4B, Llama 3.2 3B | 1500 ms | 150 ms |
| `4B` thinking | Qwen3-4B-Thinking | 1500 ms | 175 ms |
| `8B/14B` | Qwen3-8B/14B, Llama 3.1 8B, gpt-oss-20b | 2000 ms | 200 ms |
| `~30B` | Qwen3-30B-A3B, Qwen3-32B, Gemma 26B/31B, CodeLlama-34B | 2000 ms | 200 ms |
| `70B+` | Llama 3.1 70B, gpt-oss-120b | 2000 ms | 200 ms |
| `235B TP8` | Qwen3-235B-A22B instruct/thinking | 2000 ms | 200 ms |

### A100-40GB Slurm reasoning buckets

| Model bucket | Representative models | TTFT SLO | TPOT SLO |
| --- | --- | ---: | ---: |
| `<3B` | Qwen3-0.6B, Qwen3-1.7B, Gemma E2B, Llama 3.2 1B | 1500 ms | 150 ms |
| `3B/4B` non-thinking | Qwen3-4B-Instruct, Gemma E4B, Llama 3.2 3B | 1500 ms | 150 ms |
| `4B` thinking | Qwen3-4B-Thinking | 1500 ms | 175 ms |
| `8B` | Qwen3-8B, Llama 3.1 8B, gpt-oss-20b | 2000 ms | 200 ms |
| `14B+` and multi-GPU large buckets | Qwen3-14B, 30B+, gpt-oss-120b | 2000 ms | 200 ms |

## 5. LongBench TTFT Policy

LongBench manifests do **not** use a single static TTFT threshold in practice.
They set:

- `ttft_slo_field: ttft_p90_ms`
- `tpot_slo_field: tpot_p90_ms`
- `ttft_slo_mode: length_scaled`
- `tpot_slo_ms: 150`

So the active policy is:

```text
TTFT_threshold_ms(record) =
  1000 * min(cap_s, base_s + per_1k_input_tokens_s * (prompt_len / 1000))
```

Where `prompt_len` is the request input token count.

The implementation is in
[profiler/llm_mst_finder/stability.py](/scratch/gautschi/shi676/BioLLM/profiler/llm_mst_finder/stability.py:890).

### LongBench length-scaled parameters

| LongBench bucket | base (s) | per 1k input tokens (s) | cap (s) |
| --- | ---: | ---: | ---: |
| `long_output_summarization` | 8.0 | 1.4 | 45.0 |
| `medium_output_summarization` | 6.0 | 1.2 | 40.0 |
| `medium_answer_rag_qa` | 5.0 | 0.9 | 30.0 |
| `short_answer_document_qa` | 4.0 | 0.8 | 20.0 |

These constants are defined in
[profiler/llm_mst_finder/stability.py](/scratch/gautschi/shi676/BioLLM/profiler/llm_mst_finder/stability.py:20).

### LongBench TPOT policy

For all current local and Slurm LongBench manifests:
- TPOT <= `150 ms`

### LongBench model-size buckets

The LongBench manifests mostly vary search ceilings and context limits by model
size, not the SLO thresholds themselves. So the compact view is:

| Platform | Model buckets with distinct TTFT/TPOT SLO | Effective TTFT / TPOT policy |
| --- | --- | --- |
| L40 local | all buckets | length-scaled TTFT + `150 ms` TPOT |
| H100 Slurm | all buckets, including TP8 Qwen3-235B | length-scaled TTFT + `150 ms` TPOT |
| A100-40GB Slurm | all buckets | length-scaled TTFT + `150 ms` TPOT |

The per-bucket manifest overrides for LongBench are about:
- request-rate cap
- closed-loop concurrency
- engine context / batching limits
- TP-specific launch constraints

They do not change the LongBench SLO equation.

## 6. Static LongBench Presets in Code

The stability code also defines static LongBench presets:

| Preset | long output | medium output | RAG QA | short QA |
| --- | ---: | ---: | ---: | ---: |
| `default` | 35 s | 30 s | 20 s | 15 s |
| `tight` | 25 s | 22 s | 15 s | 10 s |
| `relaxed` | 45 s | 40 s | 30 s | 20 s |

Current manifests do not use these presets; they use `ttft_slo_mode: length_scaled`.

## 7. Practical Reading Rule

When scanning a manifest:

1. Start from the workload-family default under `search`.
2. Apply workload-specific TTFT/TPOT overrides first.
3. Apply model-bucket TTFT/TPOT overrides next.
4. Ignore launch-only changes such as `max_model_len`, `dtype`, batching, or TP
   unless the question is about capacity rather than SLO.

For current manifests, that reduces to:
- chat: mostly `1000 / 150`, tighter only for small buckets
- code: `1500 / 75` for CrossCodeEval, `5000 / 75` for RepoBench, with tighter
  small-model TPOT and small-model CrossCodeEval TTFT
- reasoning: `2000 / 200` on H100/A100 and `5000 / 200` on L40, with tighter
  3B/4B buckets
- LongBench: length-scaled TTFT equation plus `150 ms` TPOT
