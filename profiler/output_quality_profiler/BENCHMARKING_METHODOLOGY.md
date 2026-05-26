# Benchmarking Methodology

This note describes the quality-profiler methodology for both the chat
LLM-as-judge track and the ground-truth benchmark track.

It reflects the implemented behavior in:

- `profiler/output_quality_profiler/scoring.py`
- `profiler/output_quality_profiler/benchmark_adapters/`
- `profiler/dataset_workload_materializer/datasets/code.py`
- user-supplied quality generation manifests

## 1. Chat quality score from win/tie rate

The chat quality score is `q_chat`:

```text
q_chat = (wins + 0.5 * ties) / (wins + ties + losses)
```

Rules:

- `win`: the judge prefers the candidate over the reference
- `tie`: counts as half credit
- `loss`: the judge prefers the reference
- `invalid`: excluded from the denominator

This is the implemented formula in
`profiler/output_quality_profiler/scoring.py`.

Operationally, `q_chat` is a pairwise score against a chosen reference model,
not an absolute benchmark score. A/B order is randomized during judge-batch
construction, and the aggregation output keeps a position breakdown so possible
judge-position bias can be checked.

## 2. Original benchmark metrics and what we report

### SuperGPQA

- Primary metric: accuracy
- Evaluation unit: final extracted answer label vs ground truth
- Breakdowns: by subject and by difficulty

Current local compatibility adapter extracts the final answer label from the
saved response text and scores unextractable successful responses as incorrect
instead of dropping them.

### LongBench v1 covered tasks

We score the original covered tasks with the original LongBench task-to-metric
mapping. The covered-task workload is grouped into four buckets:

1. `long_output_summarization`
   - Tasks: `gov_report`, `gov_report_e`
   - Metric family: ROUGE-style summarization
   - Task metric: `rouge`

2. `medium_output_summarization`
   - Tasks: `multi_news`, `multi_news_e`, `qmsum`, `vcsum`
   - Metric family: ROUGE-style summarization
   - Task metrics: `rouge` for English summarization tasks, `rouge_zh` for
     `vcsum`

3. `medium_answer_rag_qa`
   - Task: `dureader`
   - Metric family: Chinese ROUGE-style QA
   - Task metric: `rouge_zh`

4. `short_answer_document_qa`
   - Tasks: `multifieldqa_en`, `multifieldqa_en_e`, `multifieldqa_zh`,
     `qasper`, `qasper_e`
   - Metric family: F1-style QA
   - Task metrics: `qa_f1` / `qa_f1_zh`

Important interpretation:

- `by_bucket` is the primary report for the workload track.
- Top-level `overall_score` is only a macro summary across covered tasks.
- That top-level score mixes ROUGE-style and F1-style metrics, so it should not
  be treated as a clean single-family leaderboard metric.

### RepoBench

RepoBench local scoring emits:

- EM: exact match on whitespace-tokenized prediction vs reference
- ES: edit similarity (`fuzz.ratio`-compatible)
- optional CodeBLEU when dependencies are installed

For reporting, the main RepoBench pair is:

- `edit_similarity_percent` as the primary score
- `exact_match_percent` as the companion metric

CodeBLEU is retained only as an auxiliary field and is not the selected main
score.

### CrossCodeEval / CCEval

CrossCodeEval local scoring emits:

- Code Matching EM
- Code Matching ES
- Identifier Matching ID-EM
- Identifier precision / recall / F1

For reporting, the selected primary score is:

- `exact_match_percent` as the main score

and the supporting code-completion metrics are:

- `edit_similarity_percent`
- `identifier_f1_percent`

This matches the current adapter contract, where `overall_score` is Code
Matching EM.

## 3. Sampling parameters and token budgets used for benchmarks

The benchmark track uses deterministic single-sample decoding after prompt and
endpoint fixes. In practice that means:

- `temperature: 0.0`
- `n: 1`
- `min_p: 0.0`
- preserve request order
- include prompt text in saved response artifacts

The remaining settings are benchmark-specific.

### SuperGPQA

Endpoint:

- `/v1/chat/completions`

Decoding:

- `temperature: 0.0`
- `top_p: 1.0`
- `top_k: 20`
- `prompt_token_buffer: 128`

Completion budget:

- original hard-run setup: `max_tokens: 4096`
- current rerun setup for non-Llama-2 models: `max_tokens: 16384`
- Llama-2 models stay at `max_tokens: 4096`

Reason: the 4k cap frequently prevented a final answer from appearing,
especially for Qwen3 and always-thinking models.

### LongBench v1 covered

Endpoint:

- `/v1/chat/completions`

Decoding:

- `temperature: 0.0`
- `top_p: 1.0`
- `top_k: 20`
- `prompt_token_buffer: 128`

Completion budget:

- `max_tokens: 4096`

Operational notes:

- non-thinking Qwen3 models use `extra_body.chat_template_kwargs.enable_thinking:
  false`
- always-thinking models are run separately and their responses must be
  postprocessed to remove visible reasoning before scoring
- the answer-style rerun keeps the same deterministic budget but changes the
  prompt wording to ask for more direct answers

### RepoBench

Current corrected rerun path:

- endpoint: `/v1/chat/completions`
- decoding:
  - `temperature: 0.0`
  - `top_p: 0.95`
  - `top_k: 64`
  - `prompt_token_buffer: 128`
  - `max_tokens: 128`
  - structural stop strings only
- for pre-2507 Qwen3 models:
  - `extra_body.chat_template_kwargs.enable_thinking: false`

Earlier plain-prefix runs used `/v1/completions`, but the corrected chat rerun
is the preferred methodology because some instruction-tuned models produced
blank or malformed completions through the completions endpoint.

### CrossCodeEval / CCEval

Baseline path:

- endpoint: `/v1/completions`
- decoding:
  - `temperature: 0.0`
  - `top_p: 1.0`
  - `top_k: 20`
  - `prompt_token_buffer: 128`
  - `max_tokens: 128`
  - stop on triple newline

Gemma-friendly rerun path:

- endpoint: `/v1/chat/completions`
- decoding:
  - `temperature: 0.0`
  - `top_p: 0.95`
  - `top_k: 64`
  - `prompt_token_buffer: 128`
  - `max_tokens: 64`
  - structural stop strings only
  - `extra_body.chat_template_kwargs.enable_thinking: false`

Reason: Gemma models were much better aligned with a chat-style code-completion
template than with the plain completions endpoint.

## 4. How IDE-style code-completion retrieval context is prepared

### Retrieval is materialized offline, not executed live

For RepoBench and CrossCodeEval, the profiler does not run live retrieval,
tool-calling, or any agent loop during quality generation.

Instead, the materializer embeds the already-available retrieval context into
the single request payload:

- CrossCodeEval:
  - uses the row's `cross_file_context`
  - uses the current file prefix from `prompt` / `current_file_prefix`

- RepoBench:
  - uses the row's retrieved `context` snippets
  - keeps `import_statement`
  - keeps the current file prefix from `cropped_code`

This means each request remains an ordinary single-turn API call with the full
retrieval evidence serialized into prompt text. Nothing in the request requires
tool execution, function calling, or server-side retrieval.

### Plain-prefix template

The original code-completion workload path uses `prompt_template: plain_prefix`.

CrossCodeEval plain-prefix shape:

```text
Relevant repository context:
{cross_file_context}

{current_file_prefix}
```

If no cross-file context exists, the prompt is just the current file prefix.

RepoBench plain-prefix shape:

```text
Relevant repository context:
{serialized_context_snippets}

{import_statement}

{cropped_code_prefix}
```

For `in_file` RepoBench rows, the repository-context block is omitted.

This format is intentionally close to IDE completion: the prompt simply ends at
the code cursor.

### Enhanced-chat template

The corrected chat-based rerun path uses `prompt_template:
code_chat_completion` or `gemma_chat_completion`.

CrossCodeEval enhanced-chat user content:

```text
Complete the code at <CURSOR>.

<REPOSITORY_CONTEXT>
{cross_file_context}
</REPOSITORY_CONTEXT>

<TARGET_FILE path="{file_path}" language="{language}">
{current_file_prefix}<CURSOR>
</TARGET_FILE>

Return only the continuation after <CURSOR>.
```

RepoBench enhanced-chat user content:

```text
Complete the next line of code at <CURSOR>.

<REPOSITORY_CONTEXT>
{serialized_context_snippets}
</REPOSITORY_CONTEXT>

<IMPORTS>
{import_statement}
</IMPORTS>

<TARGET_FILE path="{file_path}" language="{language}">
{cropped_code_prefix}<CURSOR>
</TARGET_FILE>

Return only the next line after <CURSOR>.
```

The chat templates also attach a system prompt that constrains the model to act
as a code-completion engine and return only the raw code continuation.

### Why the chat template was introduced

The chat template was introduced for alignment, not to make the workload
agentic.

The plain-prefix `/v1/completions` path is closer to classic code completion,
but in practice several modern instruction-tuned models behaved poorly under
that interface:

- blank responses
- repeated punctuation or malformed text
- continuation of task instructions instead of code
- chat-role markup leaking into the output

The enhanced-chat template fixes that by:

- keeping the request single-turn and retrieval-free
- preserving the exact same embedded repository evidence
- making the cursor position explicit with `<CURSOR>`
- making file identity explicit with `<TARGET_FILE>`
- allowing a strict system prompt to enforce "return code only"
- using chat-native serving for models that are primarily trained/aligned as
  chat assistants

So the chat template is not a change to the benchmark semantics. It is a
delivery-format correction that improves compatibility with modern
instruction-tuned models while still evaluating ordinary next-line or
continuation completion.
