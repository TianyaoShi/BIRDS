from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable


DOWNLOADERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "gpqa_diamond": lambda output_dir: download_gpqa_diamond(output_dir),
    "gpqa_extended": lambda output_dir: download_gpqa_csv(output_dir, config_name="gpqa_extended"),
    "mmlu": lambda output_dir: download_mmlu(output_dir),
    "mmlu_pro": lambda output_dir: download_mmlu_pro(output_dir),
    "aime_2024_2026": lambda output_dir: download_aime_2024_2026(output_dir),
    "hard_reasoning_small_mixed": lambda output_dir: write_hard_reasoning_small_mixed(output_dir),
    "supergpqa": lambda output_dir: download_supergpqa(output_dir),
    "natural_reasoning": lambda output_dir: download_natural_reasoning(output_dir),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public reasoning benchmark sources as local JSONL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/reasoning"),
        help="Directory for normalized JSONL outputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DOWNLOADERS),
        default=list(DOWNLOADERS),
        help="Reasoning sources to download or derive.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {name: DOWNLOADERS[name](output_dir) for name in args.datasets}
    (output_dir / "download_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def download_gpqa_diamond(output_dir: Path) -> dict[str, Any]:
    return download_gpqa_csv(output_dir, config_name="gpqa_diamond")


def download_gpqa_csv(output_dir: Path, *, config_name: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    filename = f"{config_name}.csv"
    local_csv = hf_hub_download(
        repo_id="Wanfq/gpqa",
        repo_type="dataset",
        filename=filename,
    )
    rows: list[dict[str, Any]] = []
    with Path(local_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: value for key, value in row.items() if key is not None})
    if not rows:
        raise ValueError(f"Wanfq/gpqa {filename} produced no rows")
    output_path = output_dir / f"{config_name}.jsonl"
    write_jsonl(output_path, rows)
    return {
        "source": "Wanfq/gpqa",
        "source_file": filename,
        "output_path": str(output_path),
        "rows": len(rows),
    }


def download_mmlu(output_dir: Path) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset("cais/mmlu", "all", split="test")
    rows = [dict(row) for row in dataset]
    if not rows:
        raise ValueError("cais/mmlu all test produced no rows")
    output_path = output_dir / "mmlu.jsonl"
    write_jsonl(output_path, rows)
    return {
        "source": "cais/mmlu",
        "config": "all",
        "split": "test",
        "output_path": str(output_path),
        "rows": len(rows),
    }


def download_mmlu_pro(output_dir: Path) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = [dict(row) for row in dataset]
    if not rows:
        raise ValueError("TIGER-Lab/MMLU-Pro test produced no rows")
    output_path = output_dir / "mmlu_pro.jsonl"
    write_jsonl(output_path, rows)
    return {
        "source": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "output_path": str(output_path),
        "rows": len(rows),
    }


def download_supergpqa(output_dir: Path) -> dict[str, Any]:
    return download_hf_split_jsonl(
        output_dir / "supergpqa.jsonl",
        repo_id="m-a-p/SuperGPQA",
        split="train",
    )


def download_natural_reasoning(output_dir: Path) -> dict[str, Any]:
    return download_hf_split_jsonl(
        output_dir / "natural_reasoning.jsonl",
        repo_id="facebook/natural_reasoning",
        split="train",
    )


def download_hf_split_jsonl(
    output_path: Path,
    *,
    repo_id: str,
    split: str,
    config_name: str | None = None,
) -> dict[str, Any]:
    from datasets import load_dataset

    if config_name is None:
        dataset = load_dataset(repo_id, split=split)
    else:
        dataset = load_dataset(repo_id, config_name, split=split)
    rows = write_dataset_jsonl(output_path, dataset)
    if rows == 0:
        label = f"{repo_id} {config_name or ''} {split}".strip()
        raise ValueError(f"{label} produced no rows")
    metadata: dict[str, Any] = {
        "source": repo_id,
        "split": split,
        "output_path": str(output_path),
        "rows": rows,
    }
    if config_name is not None:
        metadata["config"] = config_name
    return metadata


def download_aime_2024_2026(output_dir: Path) -> dict[str, Any]:
    from datasets import load_dataset

    sources = [
        ("2024", "sea-snell/aime-2024", "test"),
        ("2025", "test-time-compute/aime_2025", "test"),
        ("2026", "MathArena/aime_2026", "train"),
    ]
    rows: list[dict[str, Any]] = []
    per_source: list[dict[str, Any]] = []
    for year, repo_id, split in sources:
        dataset = load_dataset(repo_id, split=split)
        source_rows = [normalize_aime_row(dict(row), year=year, repo_id=repo_id) for row in dataset]
        if not source_rows:
            raise ValueError(f"{repo_id} {split} produced no rows")
        rows.extend(source_rows)
        per_source.append({"source": repo_id, "split": split, "year": year, "rows": len(source_rows)})
    output_path = output_dir / "aime_2024_2026.jsonl"
    write_jsonl(output_path, rows)
    return {
        "sources": per_source,
        "output_path": str(output_path),
        "rows": len(rows),
    }


def write_hard_reasoning_small_mixed(output_dir: Path) -> dict[str, Any]:
    sources = [
        ("gpqa_extended", output_dir / "gpqa_extended.jsonl"),
        ("aime_2024_2026", output_dir / "aime_2024_2026.jsonl"),
    ]
    rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for source_name, source_path in sources:
        if not source_path.is_file():
            raise FileNotFoundError(f"mixed reasoning source missing: {source_path}")
        count = 0
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(f"{source_path} contains a non-object JSONL row")
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                row["metadata"] = {"mixed_source": source_name, **metadata}
                rows.append(row)
                count += 1
        source_counts[source_name] = count
    output_path = output_dir / "hard_reasoning_small_mixed.jsonl"
    write_jsonl(output_path, rows)
    return {
        "sources": source_counts,
        "output_path": str(output_path),
        "rows": len(rows),
    }


def normalize_aime_row(row: dict[str, Any], *, year: str, repo_id: str) -> dict[str, Any]:
    problem = first_string(row, ("question", "problem", "Problem"))
    answer = first_value(row, ("answer", "Answer"))
    if problem is None:
        prompt = row.get("prompt") or row.get("messages")
        problem = prompt_text(prompt)
    if problem is None:
        extra_info = row.get("extra_info") or row.get("metadata")
        if isinstance(extra_info, dict):
            problem = first_string(extra_info, ("raw_problem", "problem", "question"))
    if problem is None:
        raise ValueError(f"{repo_id} row has no problem/question field")
    if answer in (None, ""):
        reward_model = row.get("reward_model")
        if isinstance(reward_model, dict):
            answer = reward_model.get("ground_truth")
    if answer in (None, ""):
        raise ValueError(f"{repo_id} row has no answer field")
    record_id = first_value(row, ("id", "problem_idx", "raw_problem_id"))
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "Problem": problem,
        "Answer": str(answer),
        "year": int(year),
        "id": str(record_id) if record_id not in (None, "") else None,
        "metadata": {
            "source": repo_id,
            **metadata,
        },
    }


def prompt_text(prompt: Any) -> str | None:
    if isinstance(prompt, str) and prompt:
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for item in prompt:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
        if parts:
            return "\n\n".join(parts)
    return None


def first_string(row: dict[str, Any], keys: Iterable[str]) -> str | None:
    value = first_value(row, keys)
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        return str(value)
    return value


def first_value(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_dataset_jsonl(path, rows)


def write_dataset_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    main()
