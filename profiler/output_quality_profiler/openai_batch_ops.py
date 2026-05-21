from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


def split_judge_batch_by_candidate(
    *,
    batch_jsonl: str | Path,
    batch_manifest: str | Path,
    output_dir: str | Path,
    parts_per_candidate: int = 2,
    candidate_model_slugs: Sequence[str] = (),
) -> dict[str, Any]:
    if parts_per_candidate <= 0:
        raise ValueError("parts_per_candidate must be positive")
    batch_jsonl_path = Path(batch_jsonl).resolve()
    batch_manifest_path = Path(batch_manifest).resolve()
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    comparisons = manifest.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("batch manifest must contain non-empty comparisons")
    comparison_by_id = {str(item["custom_id"]): item for item in comparisons}
    candidate_filter = set(candidate_model_slugs)

    rows_by_candidate: dict[str, list[str]] = {}
    with batch_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str) or custom_id not in comparison_by_id:
                raise ValueError(f"JSONL row custom_id is missing from manifest: {custom_id!r}")
            candidate_slug = str(comparison_by_id[custom_id]["candidate_model_slug"])
            if candidate_filter and candidate_slug not in candidate_filter:
                continue
            rows_by_candidate.setdefault(candidate_slug, []).append(line)
    if candidate_filter:
        missing_candidates = sorted(candidate_filter - set(rows_by_candidate))
        if missing_candidates:
            raise ValueError(f"requested candidate slugs have no rows: {missing_candidates}")

    parts: list[dict[str, Any]] = []
    for candidate_slug, rows in sorted(rows_by_candidate.items()):
        chunks = _split_rows_by_size(rows, parts_per_candidate)
        for part_index, chunk in enumerate(chunks):
            if not chunk:
                continue
            part_name = f"{candidate_slug}.part-{part_index:02d}"
            part_jsonl = output_dir_path / f"{part_name}.jsonl"
            part_manifest = output_dir_path / f"{part_name}.manifest.json"
            custom_ids: list[str] = []
            with part_jsonl.open("w", encoding="utf-8") as handle:
                for row_text in chunk:
                    handle.write(row_text)
                    custom_ids.append(str(json.loads(row_text)["custom_id"]))
            part_comparisons = [comparison_by_id[custom_id] for custom_id in custom_ids]
            part_payload = dict(manifest)
            part_payload.update(
                {
                    "parent_manifest_path": str(batch_manifest_path),
                    "parent_output_jsonl": str(batch_jsonl_path),
                    "split_strategy": "by_candidate_size_balanced",
                    "candidate_model_slug": candidate_slug,
                    "split_part_index": part_index,
                    "split_part_count": len(chunks),
                    "request_count": len(chunk),
                    "request_counts_by_candidate": {candidate_slug: len(chunk)},
                    "output_jsonl": str(part_jsonl),
                    "comparisons": part_comparisons,
                }
            )
            part_manifest.write_text(
                json.dumps(part_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            parts.append(
                {
                    "candidate_model_slug": candidate_slug,
                    "part_index": part_index,
                    "jsonl": str(part_jsonl),
                    "manifest": str(part_manifest),
                    "requests": len(chunk),
                    "bytes": part_jsonl.stat().st_size,
                }
            )

    split_manifest = {
        "created_at": _now_utc_iso(),
        "parent_manifest_path": str(batch_manifest_path),
        "parent_output_jsonl": str(batch_jsonl_path),
        "parts_per_candidate": parts_per_candidate,
        "candidate_model_slugs": sorted(rows_by_candidate),
        "part_count": len(parts),
        "total_requests": sum(int(part["requests"]) for part in parts),
        "total_bytes": sum(int(part["bytes"]) for part in parts),
        "parts": parts,
    }
    split_manifest_path = output_dir_path / "split_manifest.json"
    split_manifest["split_manifest_path"] = str(split_manifest_path)
    split_manifest_path.write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return split_manifest


def submit_openai_batch_parts(
    *,
    split_manifest: str | Path,
    api_key_file: str | Path,
    ledger_path: str | Path | None = None,
    limit: int | None = None,
    wait_for_completion: bool = False,
    poll_interval_s: float = 60.0,
    completion_window: str = "24h",
) -> dict[str, Any]:
    from openai import OpenAI

    split_manifest_path = Path(split_manifest).resolve()
    split_payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    parts = list(split_payload.get("parts") or [])
    if not parts:
        raise ValueError("split manifest has no parts")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")

    resolved_ledger_path = (
        Path(ledger_path).resolve()
        if ledger_path is not None
        else split_manifest_path.with_name("submission_ledger.json")
    )
    ledger = _load_or_init_ledger(resolved_ledger_path, split_manifest_path=split_manifest_path)
    client = OpenAI(api_key=_read_api_key(api_key_file))

    processed = 0
    events: list[dict[str, Any]] = []
    for part in parts:
        if limit is not None and processed >= limit:
            break
        part_key = _part_key(part)
        entry = ledger["parts"].get(part_key)
        if entry is not None and entry.get("batch_status") in TERMINAL_BATCH_STATUSES:
            if _entry_result_retrieved(entry):
                continue
            entry = _retrieve_batch_result_files(client, entry)
            ledger["parts"][part_key] = entry
            ledger["updated_at"] = _now_utc_iso()
            _write_json(resolved_ledger_path, ledger)
            processed += 1
            events.append({"event": "retrieved_existing_terminal_batch", "part": part_key, "batch_id": entry.get("batch_id")})
            continue
        if entry is not None and entry.get("batch_id"):
            if wait_for_completion:
                entry = _wait_for_batch(client, entry, poll_interval_s=poll_interval_s)
                ledger["parts"][part_key] = entry
                _write_json(resolved_ledger_path, ledger)
            processed += 1
            events.append({"event": "existing_batch", "part": part_key, "batch_id": entry.get("batch_id")})
            continue

        jsonl_path = Path(str(part["jsonl"])).resolve()
        with jsonl_path.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={
                "candidate": str(part.get("candidate_model_slug", ""))[:512],
                "part_index": str(part.get("part_index", "")),
            },
        )
        entry = {
            **part,
            "submitted_at": _now_utc_iso(),
            "file_id": uploaded.id,
            "batch_id": batch.id,
            "batch_status": batch.status,
        }
        ledger["parts"][part_key] = entry
        ledger["updated_at"] = _now_utc_iso()
        _write_json(resolved_ledger_path, ledger)
        events.append({"event": "submitted", "part": part_key, "file_id": uploaded.id, "batch_id": batch.id})
        if wait_for_completion:
            entry = _wait_for_batch(client, entry, poll_interval_s=poll_interval_s)
            ledger["parts"][part_key] = entry
            ledger["updated_at"] = _now_utc_iso()
            _write_json(resolved_ledger_path, ledger)
        processed += 1

    return {
        "split_manifest": str(split_manifest_path),
        "ledger": str(resolved_ledger_path),
        "processed": processed,
        "events": events,
        "counts": _ledger_counts(ledger),
    }


def _split_rows_by_size(rows: Sequence[str], parts: int) -> list[list[str]]:
    target = sum(len(row.encode("utf-8")) for row in rows) / parts
    chunks: list[list[str]] = [[] for _ in range(parts)]
    chunk_index = 0
    chunk_bytes = 0
    for row in rows:
        row_bytes = len(row.encode("utf-8"))
        if chunk_index < parts - 1 and chunks[chunk_index] and chunk_bytes + row_bytes > target:
            chunk_index += 1
            chunk_bytes = 0
        chunks[chunk_index].append(row)
        chunk_bytes += row_bytes
    return chunks


def _wait_for_batch(client: Any, entry: dict[str, Any], *, poll_interval_s: float) -> dict[str, Any]:
    while True:
        batch = client.batches.retrieve(str(entry["batch_id"]))
        entry = {
            **entry,
            "batch_status": batch.status,
            "last_checked_at": _now_utc_iso(),
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
        }
        if batch.status in TERMINAL_BATCH_STATUSES:
            entry["completed_at"] = _now_utc_iso()
            return _retrieve_batch_result_files(client, entry)
        time.sleep(poll_interval_s)


def _retrieve_batch_result_files(client: Any, entry: dict[str, Any]) -> dict[str, Any]:
    if entry.get("output_file_id"):
        output_path = _result_path(entry, kind="output")
        if not output_path.is_file():
            _download_openai_file(client, str(entry["output_file_id"]), output_path)
        entry["output_jsonl"] = str(output_path)
    if entry.get("error_file_id"):
        error_path = _result_path(entry, kind="error")
        if not error_path.is_file():
            _download_openai_file(client, str(entry["error_file_id"]), error_path)
        entry["error_jsonl"] = str(error_path)
    entry["result_retrieved_at"] = _now_utc_iso()
    return entry


def _entry_result_retrieved(entry: dict[str, Any]) -> bool:
    if entry.get("output_file_id"):
        output_jsonl = entry.get("output_jsonl")
        if not output_jsonl or not Path(str(output_jsonl)).is_file():
            return False
    if entry.get("error_file_id"):
        error_jsonl = entry.get("error_jsonl")
        if not error_jsonl or not Path(str(error_jsonl)).is_file():
            return False
    return bool(entry.get("result_retrieved_at"))


def _result_path(entry: dict[str, Any], *, kind: str) -> Path:
    jsonl = Path(str(entry["jsonl"])).resolve()
    return jsonl.with_suffix(f".{kind}.jsonl")


def _download_openai_file(client: Any, file_id: str, path: Path) -> None:
    content = client.files.content(file_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(content, "write_to_file"):
        content.write_to_file(str(path))
        return
    if hasattr(content, "read"):
        data = content.read()
    elif isinstance(content, str):
        data = content.encode("utf-8")
    else:
        data = bytes(content)
    path.write_bytes(data)


def _part_key(part: dict[str, Any]) -> str:
    return f"{part.get('candidate_model_slug')}::part-{int(part.get('part_index', 0)):02d}"


def _load_or_init_ledger(path: Path, *, split_manifest_path: Path) -> dict[str, Any]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("parts", {})
        return payload
    return {
        "created_at": _now_utc_iso(),
        "updated_at": _now_utc_iso(),
        "split_manifest": str(split_manifest_path),
        "parts": {},
    }


def _ledger_counts(ledger: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in ledger.get("parts", {}).values():
        status = str(entry.get("batch_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _read_api_key(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"API key file is empty: {path}")
    if "=" in text and "\n" not in text:
        key, value = text.split("=", 1)
        if key.strip() in {"OPENAI_API_KEY", "OPENAI_KEY"}:
            return value.strip().strip('"').strip("'")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :]
        if "=" in stripped:
            key, value = stripped.split("=", 1)
            if key.strip() in {"OPENAI_API_KEY", "OPENAI_KEY"}:
                return value.strip().strip('"').strip("'")
        return stripped
    raise ValueError(f"API key file does not contain a key: {path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
