# Diagnosis

Your current setting is likely pathological for Gemma 4 IT:

```
endpoint: /v1/completions
template: plain_prefix
temperature: 0.0
top_p: 1.0
top_k: 20
max_tokens: 128
stop: ["\n\n"]
```

For Gemma 4 IT, this combines three problems:

- Wrong interaction format. Gemma 4 uses standard system, user, and assistant/model turns; Google’s prompt-formatting page shows the <|turn>system, <|turn>user, <|turn>model structure, and vLLM requires a chat template for chat models.
- Greedy decoding can lock into degenerate loops. In vLLM, temperature=0 means greedy sampling; then your top_p, top_k, and min_p settings are mostly irrelevant for escaping a bad high-probability token loop. vLLM explicitly defines zero temperature as greedy, while repetition/frequency penalties are the controls that discourage repeated tokens.
Gemma 4 31B has thinking/control-token behavior that can surface oddly if the template/parser path is wrong. Google notes that when thinking is disabled, most Gemma 4 models except E2B/E4B may still emit an empty thought block before the final answer. That makes raw completion especially fragile for 31B.

- The “official coding excellence” numbers are not a guarantee for prefix-only, next-statement, raw completion. The model card reports strong LiveCodeBench and Codeforces performance and says Gemma 4 supports coding, but those are instruction-style coding evaluations, not necessarily IDE-style editor completion.

## Recommended endpoint

Use:

`/v1/chat/completions`

for google/gemma-4-*-it.


## Recommended Gemma 4 IT prompt template

Avoid Markdown code fences in both prompt and output. They can prime the model to return fences. Use explicit sentinels instead.

```
SYSTEM:
You are a code completion engine.
Return only the exact code continuation after <CURSOR>.
Do not return Markdown, code fences, comments, explanations, XML tags, or natural language.
Do not repeat the provided context or prefix.
Complete the current statement or line only.
Begin immediately with code.

USER:
Complete the code at <CURSOR>.

<REPOSITORY_CONTEXT>
{retrieved_cross_file_context}
</REPOSITORY_CONTEXT>

<TARGET_FILE path="{path}" language="{language}">
{target_file_prefix}<CURSOR>
</TARGET_FILE>

Return only the continuation after <CURSOR>.
```

For your existing CrossCodeEval setup, I would convert the “retrieved code fragments as comments” into a separate `<REPOSITORY_CONTEXT>` block. Keeping them as comments inside one pseudo-file can encourage Gemma to continue comments or separator text.

## Recommended request parameters

Start with this stability-first configuration:

```json
{
  "endpoint": "/v1/chat/completions",
  "temperature": 0.0,
  "top_p": 0.95,
  "top_k": 64,
  "min_p": 0.0,
  "n": 1,
  "max_tokens": 64,
  "repetition_penalty": 1.05,
  "frequency_penalty": 0.05,
  "stop": [
    "</TARGET_FILE>",
    "</COMPLETION>",
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "\n\n\n"
  ],
  "seed": 1
}
```

Rationale:

Google’s official Gemma 4 recommendation is temperature=1.0, top_p=0.95, top_k=64, but that is a general-use recommendation, not a deterministic code-completion benchmark setting.
For CrossCodeEval, I would not start at temperature=1.0; it adds unnecessary variance. 
Keep min_p=0.0. CrossCodeEval often requires repository-specific identifiers; min_p > 0 can suppress rare but correct local API names.
Use top_k=64, not 20, because 64 matches Google’s Gemma 4 recommendation.
Reduce max_tokens from 128 to 64. CrossCodeEval references average roughly 13–17 tokens and around 1–1.7 lines depending on language, so 128 gives degenerate loops more room.
Add repetition_penalty and a small frequency_penalty because your failure signature is repeated-symbol/text runs. vLLM documents that repetition_penalty > 1 discourages repeated prompt/generated tokens, and positive frequency penalty discourages repeated generated tokens.

With the OpenAI Python client against vLLM, top_k, min_p, and repetition_penalty may need to go through extra_body:

```python
response = client.chat.completions.create(
    model="google/gemma-4-31B-it",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a code completion engine.\n"
                "Return only the exact code continuation after <CURSOR>.\n"
                "Do not return Markdown, code fences, comments, explanations, XML tags, or natural language.\n"
                "Do not repeat the provided context or prefix.\n"
                "Complete the current statement or line only.\n"
                "Begin immediately with code."
            ),
        },
        {
            "role": "user",
            "content": prompt_text,
        },
    ],
    max_tokens=64,
    temperature=0.0,
    top_p=0.95,
    frequency_penalty=0.05,
    stop=[
        "</TARGET_FILE>",
        "</COMPLETION>",
        "<|turn>",
        "<turn|>",
        "<|channel>",
        "<channel|>",
        "\n\n\n",
    ],
    seed=1,
    extra_body={
        "top_k": 64,
        "min_p": 0.0,
        "repetition_penalty": 1.05,
        "chat_template_kwargs": {"enable_thinking": False}
    },
)
```

## Post-processing additions

Your evaluator already strips fences and comments, but add a pre-eval sanitizer specifically for Gemma 4:

```python
def sanitize_gemma_completion(s: str) -> str:
    # Remove Gemma 4 reasoning/channel artifacts if they leak through.
    s = s.replace("<|channel>thought\n<channel|>", "")
    s = s.replace("<|channel>thought", "")
    s = s.replace("<channel|>", "")

    # Remove common accidental role/turn markers.
    for marker in ["<|turn>model", "<|turn>assistant", "<turn|>"]:
        s = s.replace(marker, "")

    # If the model echoes sentinels, keep only the area after cursor/completion marker.
    for marker in ["<COMPLETION>", "<CURSOR>"]:
        if marker in s:
            s = s.split(marker, 1)[-1]

    # Hard cut on XML-like closing tags.
    for marker in ["</COMPLETION>", "</TARGET_FILE>", "</REPOSITORY_CONTEXT>"]:
        if marker in s:
            s = s.split(marker, 1)[0]

    return s.strip("\r\n ")
```
Do not add "```" as a stop string. If the model begins with a code fence, that stop string can turn a recoverable fenced answer into a blank answer.