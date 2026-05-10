from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable


SOURCES: dict[str, dict[str, Any]] = {
    "gov_report_original": {
        "repo_id": "launch/gov_report",
        "kind": "gov_report",
    },
    "multi_news_original": {
        "repo_id": "alexfabbri/multi_news",
        "kind": "multi_news_hf",
    },
    "qmsum_original": {
        "repo_id": "mattercalm/qmsum",
        "kind": "hf_files",
        "files": ("validation.jsonl", "test.jsonl"),
    },
    "meetingbank": {
        "repo_id": "huuuyeah/meetingbank",
        "kind": "hf_files",
        "files": ("train.json", "validation.json", "test.json"),
    },
    "dureader_full": {
        "repo_id": "baidu/DuReader",
        "kind": "baidu_dureader2",
    },
    "qasper_full": {
        "repo_id": "allenai/qasper",
        "kind": "qasper",
    },
}

DUREADER2_RAW_URL = "https://dataset-bj.cdn.bcebos.com/dureader/dureader_raw.zip"
DUREADER2_PREPROCESSED_URL = "https://dataset-bj.cdn.bcebos.com/dureader/dureader_preprocessed.zip"
QASPER_TRAIN_DEV_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
QASPER_TEST_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"


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
    output_path = output_dir / f"{name}.jsonl"
    kind = str(source["kind"])
    if kind == "gov_report":
        row_count, split_counts = download_gov_report(output_path)
    elif kind == "multi_news_hf":
        row_count, split_counts = download_multi_news_hf(output_path)
    elif kind == "hf_files":
        row_count, split_counts = download_hf_files(output_path, repo_id=str(source["repo_id"]), files=source["files"])
    elif kind == "baidu_dureader2":
        row_count, split_counts = download_baidu_dureader2(output_path)
    elif kind == "qasper":
        row_count, split_counts = download_qasper(output_path)
    else:
        raise ValueError(f"unsupported LongBench expansion source kind: {kind}")
    if row_count == 0:
        raise ValueError(f"{source['repo_id']} produced no rows")
    return {
        "source": str(source["repo_id"]),
        "output_path": str(output_path),
        "rows": row_count,
        "splits": split_counts,
    }


def download_hf_files(output_path: Path, *, repo_id: str, files: Iterable[str]) -> tuple[int, dict[str, int]]:
    from huggingface_hub import hf_hub_download

    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for file_name in files:
        local_path = Path(hf_hub_download(repo_id, file_name, repo_type="dataset"))
        split_name = local_path.stem
        file_rows = load_json_or_jsonl(local_path)
        split_counts[split_name] = len(file_rows)
        for row in file_rows:
            row["_hf_split"] = split_name
            rows.append(row)
    return write_jsonl(output_path, rows), split_counts


def download_gov_report(output_path: Path) -> tuple[int, dict[str, int]]:
    from huggingface_hub import hf_hub_download

    files = {
        "train": ("data/gao_train.jsonl", "data/crs_train.jsonl"),
        "validation": ("data/gao_valid.jsonl", "data/crs_valid.jsonl"),
        "test": ("data/gao_test.jsonl", "data/crs_test.jsonl"),
    }
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for split_name, split_files in files.items():
        before = len(rows)
        for file_name in split_files:
            source_type = "gao" if "gao_" in file_name else "crs"
            local_path = Path(hf_hub_download("launch/gov_report", file_name, repo_type="dataset"))
            for row in load_jsonl(local_path):
                rows.append(normalize_gov_report_row(row, source_type=source_type, split_name=split_name))
        split_counts[split_name] = len(rows) - before
    return write_jsonl(output_path, rows), split_counts


def normalize_gov_report_row(row: dict[str, Any], *, source_type: str, split_name: str) -> dict[str, Any]:
    if source_type == "gao":
        document_sections: list[dict[str, Any]] = []
        for section in row["report"]:
            document_sections.extend(recursive_gov_report_load(section, keep_letter=False, depth=1))
        summary_sections = [
            {
                "title": " ".join(section["section_title"].strip().split()),
                "paragraphs": "\n".join(" ".join(paragraph.strip().split()) for paragraph in section["paragraphs"]),
            }
            for section in row["highlight"]
        ]
        row_id = "GAO_" + str(row["id"])
        summary = " ".join(
            section["paragraphs"] for section in summary_sections if section["title"] != "What GAO Recommends"
        )
    elif source_type == "crs":
        document_sections = recursive_gov_report_load(row["reports"], keep_letter=True, depth=0)
        row_id = "CRS_" + str(row["id"])
        summary = " ".join(" ".join(paragraph.strip().split()) for paragraph in row["summary"])
    else:
        raise ValueError(f"unsupported GovReport source type: {source_type}")
    document = " ".join(
        section["title"] + " " + section["paragraphs"] if section["paragraphs"] else section["title"]
        for section in document_sections
    )
    return {
        "id": row_id,
        "document": document.replace("\n", " ").strip(),
        "summary": summary.replace("\n", " ").strip(),
        "_hf_split": split_name,
    }


def recursive_gov_report_load(section: dict[str, Any], *, keep_letter: bool, depth: int) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    section_title = str(section["section_title"])
    if section_title != "Letter" or keep_letter:
        sections.append(
            {
                "title": " ".join(section_title.strip().split()),
                "paragraphs": "\n".join(" ".join(paragraph.strip().split()) for paragraph in section["paragraphs"]),
                "depth": depth,
            }
        )
        next_depth = depth + 1
    else:
        next_depth = depth
    for subsection in section["subsections"]:
        sections.extend(recursive_gov_report_load(subsection, keep_letter=keep_letter, depth=next_depth))
    return sections


def download_multi_news_hf(output_path: Path) -> tuple[int, dict[str, int]]:
    from huggingface_hub import hf_hub_download

    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    files = (
        ("train", "data/train.src.cleaned", "data/train.tgt"),
        ("validation", "data/val.src.cleaned", "data/val.tgt"),
        ("test", "data/test.src.cleaned", "data/test.tgt"),
    )
    for split_name, src_file, tgt_file in files:
        src_path = Path(hf_hub_download("alexfabbri/multi_news", src_file, repo_type="dataset"))
        tgt_path = Path(hf_hub_download("alexfabbri/multi_news", tgt_file, repo_type="dataset"))
        before = len(rows)
        with src_path.open("r", encoding="utf-8") as src_handle, tgt_path.open("r", encoding="utf-8") as tgt_handle:
            for index, (source_line, target_line) in enumerate(zip(src_handle, tgt_handle)):
                rows.append(
                    {
                        "id": f"{split_name}-{index}",
                        "document": source_line.strip().replace("NEWLINE_CHAR", "\n"),
                        "summary": target_line.strip().lstrip("- "),
                        "_hf_split": split_name,
                    }
                )
        split_counts[split_name] = len(rows) - before
    return write_jsonl(output_path, rows), split_counts


def download_baidu_dureader2(output_path: Path) -> tuple[int, dict[str, int]]:
    archive_dir = output_path.parent / "baidu_dureader_2_0"
    archive_dir.mkdir(parents=True, exist_ok=True)
    raw_archive_path = archive_dir / "dureader_raw.zip"
    preprocessed_archive_path = archive_dir / "dureader_preprocessed.zip"
    download_url(DUREADER2_RAW_URL, raw_archive_path)
    download_url(DUREADER2_PREPROCESSED_URL, preprocessed_archive_path)

    with tempfile.TemporaryDirectory() as temp_dir_name:
        extract_dir = Path(temp_dir_name) / "preprocessed"
        extract_archive(preprocessed_archive_path, extract_dir)
        rows: list[dict[str, Any]] = []
        split_counts: dict[str, int] = {}
        for split_name, pattern in (
            ("train", "trainset/*.json"),
            ("validation", "devset/*.json"),
            ("test", "testset/*.json"),
        ):
            files = sorted((extract_dir / "preprocessed").glob(pattern))
            if not files:
                files = sorted(extract_dir.glob(f"**/{pattern}"))
            if not files:
                raise FileNotFoundError(f"DuReader 2.0 preprocessed split files missing: {pattern}")
            before = len(rows)
            for file_path in files:
                source_name = file_path.stem
                for row in load_jsonl(file_path):
                    normalized = normalize_baidu_dureader2_row(
                        row,
                        split_name=split_name,
                        source_name=source_name,
                    )
                    if normalized is not None:
                        rows.append(normalized)
            split_counts[split_name] = len(rows) - before
        return write_jsonl(output_path, rows), split_counts


def normalize_baidu_dureader2_row(
    row: dict[str, Any],
    *,
    split_name: str,
    source_name: str,
) -> dict[str, Any] | None:
    documents = row.get("documents")
    if not isinstance(documents, list) or not documents:
        return None
    question = row.get("question")
    if not isinstance(question, str) or not question:
        return None
    answers = row.get("answers")
    question_id = row.get("question_id") or row.get("id")
    return {
        "question_id": str(question_id) if question_id not in (None, "") else None,
        "question": question,
        "answers": answers if answers is not None else [],
        "documents": documents,
        "question_type": row.get("question_type"),
        "fact_or_opinion": row.get("fact_or_opinion"),
        "source": source_name,
        "_hf_split": split_name,
        "source_dataset": "baidu_dureader_2_0",
    }


def download_qasper(output_path: Path) -> tuple[int, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        train_dev_archive = temp_dir / "qasper-train-dev-v0.3.tgz"
        test_archive = temp_dir / "qasper-test-and-evaluator-v0.3.tgz"
        download_url(QASPER_TRAIN_DEV_URL, train_dev_archive)
        download_url(QASPER_TEST_URL, test_archive)
        train_dev_dir = temp_dir / "train_dev"
        test_dir = temp_dir / "test"
        extract_archive(train_dev_archive, train_dev_dir)
        extract_archive(test_archive, test_dir)
        for split_name, file_name, root in (
            ("train", "qasper-train-v0.3.json", train_dev_dir),
            ("validation", "qasper-dev-v0.3.json", train_dev_dir),
            ("test", "qasper-test-v0.3.json", test_dir),
        ):
            payload_path = next(root.rglob(file_name), None)
            if payload_path is None:
                raise FileNotFoundError(f"Qasper file not found after extraction: {file_name}")
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            before = len(rows)
            for paper_id, paper in payload.items():
                row = dict(paper)
                row["id"] = paper_id
                row["_hf_split"] = split_name
                rows.append(row)
            split_counts[split_name] = len(rows) - before
    return write_jsonl(output_path, rows), split_counts


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return load_jsonl(path)
    if isinstance(payload, list):
        return [expect_row(row, f"{path}[{index}]") for index, row in enumerate(payload)]
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return [expect_row(row, f"{path}.data[{index}]") for index, row in enumerate(payload["data"])]
        return [payload]
    raise ValueError(f"{path} must contain a JSON object/list or JSONL objects")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(expect_row(json.loads(stripped), f"{path}:{index + 1}"))
    return rows


def expect_row(row: Any, source: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{source} must be a JSON object")
    return dict(row)


def download_url(url: str, output_path: Path) -> None:
    import requests

    if output_path.is_file() and output_path.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def download_google_drive_file(url: str, output_path: Path) -> None:
    import requests

    session = requests.Session()
    response = session.get(url, stream=True, timeout=120)
    response.raise_for_status()
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token is not None:
        response.close()
        response = session.get(url, params={"confirm": token}, stream=True, timeout=120)
        response.raise_for_status()
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    response.close()


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            archive.extractall(output_dir)
        return
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(output_dir)
        return
    unpack_dir = output_dir / "unpacked"
    try:
        shutil.unpack_archive(str(archive_path), str(unpack_dir))
    except shutil.ReadError as exc:
        raise ValueError(f"downloaded file is not a supported archive: {archive_path}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    main()
