from __future__ import annotations

import json
from pathlib import Path

from output_quality_profiler.response_preprocessing import (
    preprocess_response_tree,
    response_preprocessing_metadata,
    visible_response_text,
)


def test_visible_response_text_strips_completed_leading_think_block() -> None:
    assert visible_response_text("<think>hidden reasoning</think>\nFinal answer.") == "Final answer."


def test_visible_response_text_strips_orphan_think_close_prefix() -> None:
    assert visible_response_text("hidden reasoning</think>\nFinal answer.") == "Final answer."


def test_visible_response_text_handles_unclosed_leading_think_block() -> None:
    assert visible_response_text("<think>\nunfinished reasoning") == ""


def test_visible_response_text_strips_gpt_oss_joined_reasoning_prefix() -> None:
    text = "We need to summarize this carefully. Let's craft.\n\nWe need a concise answer.**Summary:** Done."
    assert visible_response_text(text, model_slug="openai-gpt-oss-20b") == "**Summary:** Done."


def test_visible_response_text_strips_qwen_thinking_untagged_reasoning_prefix() -> None:
    text = "I need to solve the task. Let's provide the final answer.\n\nThe answer is 42."
    assert (
        visible_response_text(text, model_slug="qwen-qwen3-4b-thinking-2507")
        == "The answer is 42."
    )


def test_response_preprocessing_metadata_records_hashes_and_flags() -> None:
    original = "<think>hidden</think> answer"
    rendered = visible_response_text(original, model_slug="qwen-qwen3-0-6b")
    metadata = response_preprocessing_metadata(
        original=original,
        rendered=rendered,
        model_slug="qwen-qwen3-0-6b",
    )
    assert metadata["stripped_leading_think_block"] is True
    assert metadata["removed_chars"] > 0
    assert len(metadata["original_sha256"]) == 64
    assert len(metadata["rendered_sha256"]) == 64


def test_preprocess_response_tree_writes_processed_copy(tmp_path: Path) -> None:
    responses_root = tmp_path / "responses"
    model_dir = responses_root / "qwen-qwen3-0-6b"
    model_dir.mkdir(parents=True)
    (model_dir / "summary.json").write_text('{"ok": true}\n', encoding="utf-8")
    (model_dir / "responses.jsonl").write_text(
        json.dumps(
            {
                "request_id": "r1",
                "success": True,
                "response_text": "<think>hidden</think>\nVisible.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = preprocess_response_tree(
        responses_root=responses_root,
        output_root=tmp_path / "processed",
    )

    processed_row = json.loads(
        (tmp_path / "processed" / "qwen-qwen3-0-6b" / "responses.jsonl").read_text(encoding="utf-8")
    )
    assert processed_row["response_text"] == "Visible."
    assert processed_row["metadata"]["response_preprocessing"]["stripped_leading_think_block"] is True
    assert (tmp_path / "processed" / "qwen-qwen3-0-6b" / "summary.json").is_file()
    assert result.changed_rows == 1
