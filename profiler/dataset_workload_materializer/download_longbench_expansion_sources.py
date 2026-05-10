from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SOURCES: dict[str, dict[str, Any]] = {
    "gov_report_original": {
        "repo_id": "launch/gov_report",
        "config": "plain_text",
        "splits": ("train", "valid", "test"),
    },
    "multi_news_original": {
        "repo_id": "tau/multi_news",
        "splits": ("train", "validation", "test"),
    },
    "qmsum_original": {
        "repo_id": "mattercalm/qmsum",
        "splits": None,
    },
    "meetingbank": {
        "repo_id": "huuuyeah/meetingbank",
        "splits": ("train", "validation", "test"),
    },
    "dureader_full": {
        "repo_id": "PaddlePaddle/dureader_robust",
        "splits": None,
        "trust_remote_code": True,
    },
    "qasper_full": {
        "repo_id": "allenai/qasper",
        "splits": ("train", "validation", "test"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download LongBench expansion datasets from Hugging Face as local JSONL exports."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/longbench_expansion"),
        help="Directory for JSONL exports consumed by the LongBench materializer.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(SOURCES),
        default=list(SOURCES),
        help="Expansion sources to download.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        dataset_name: download_source(dataset_name, SOURCES[dataset_name], output_dir=output_dir)
        for dataset_name in args.datasets
    }
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def download_source(name: str, source: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    from datasets import DatasetDict, load_dataset

    repo_id = str(source["repo_id"])
    config = source.get("config")
    kwargs: dict[str, Any] = {}
    if source.get("trust_remote_code") is True:
        kwargs["trust_remote_code"] = True
    splits = source.get("splits")
    output_path = output_dir / f"{name}.jsonl"
    if splits is None:
        dataset = load_dataset(repo_id, config, **kwargs) if config else load_dataset(repo_id, **kwargs)
        if not isinstance(dataset, DatasetDict):
            raise ValueError(f"{repo_id} did not return a DatasetDict; configure explicit splits")
        row_count = write_dataset_dict_jsonl(output_path, dataset)
        split_counts = {split_name: len(split_rows) for split_name, split_rows in dataset.items()}
    else:
        split_counts = {}
        row_count = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for split in splits:
                dataset = (
                    load_dataset(repo_id, config, split=split, **kwargs)
                    if config
                    else load_dataset(repo_id, split=split, **kwargs)
                )
                split_counts[str(split)] = len(dataset)
                for row in dataset:
                    payload = dict(row)
                    payload["_hf_split"] = split
                    handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
                    row_count += 1
    if row_count == 0:
        raise ValueError(f"{repo_id} produced no rows")
    return {
        "source": repo_id,
        "config": config,
        "output_path": str(output_path),
        "rows": row_count,
        "splits": split_counts,
        "trust_remote_code": source.get("trust_remote_code", False),
    }


def write_dataset_dict_jsonl(path: Path, dataset_dict: Any) -> int:
    def rows() -> Iterable[dict[str, Any]]:
        for split_name, dataset in dataset_dict.items():
            for row in dataset:
                payload = dict(row)
                payload["_hf_split"] = split_name
                yield payload

    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows():
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    main()
