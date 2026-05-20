from __future__ import annotations

import json
from pathlib import Path

from output_quality_profiler.judge_batches import build_openai_judge_batch


def _write_response(path: Path, *, request_id: str, model: str, response_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "request_id": request_id,
        "model": model,
        "source": "sharegpt",
        "prompt_length_bucket": "short",
        "prompt": "Say hello.",
        "response_text": response_text,
        "success": True,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_build_openai_judge_batch_handles_aggregate_and_sharded_inputs(tmp_path: Path) -> None:
    responses_root = tmp_path / "responses"
    _write_response(
        responses_root / "candidate" / "responses.jsonl",
        request_id="request-1",
        model="candidate/model",
        response_text="Hello there.",
    )
    _write_response(
        responses_root / "reference" / "shards" / "shard-000" / "responses.jsonl",
        request_id="request-1",
        model="reference/model",
        response_text="Hi.",
    )
    template_path = tmp_path / "template.md"
    template_path.write_text(
        'Prompt: {prompt}\nA: {response_a}\nB: {response_b}\nReturn JSON only: {"winner":"A"}',
        encoding="utf-8",
    )

    result = build_openai_judge_batch(
        responses_root=responses_root,
        reference_model_slug="reference",
        candidate_model_slugs=("candidate",),
        judge_template_path=template_path,
        output_dir=tmp_path / "judge",
        evaluator_model="gpt-4.1-nano",
        max_comparisons=1,
        seed=1,
    )

    rows = [json.loads(line) for line in result.output_jsonl.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.request_count == 1
    assert rows[0]["method"] == "POST"
    assert rows[0]["url"] == "/v1/chat/completions"
    assert rows[0]["body"]["model"] == "gpt-4.1-nano"
    assert rows[0]["body"]["response_format"] == {"type": "json_object"}
    assert manifest["comparisons"][0]["candidate_model_slug"] == "candidate"
