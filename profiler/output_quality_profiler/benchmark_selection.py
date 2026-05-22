from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from local_orchestrator.utils import slugify


BENCHMARK_TARGETS: dict[str, dict[str, Any]] = {
    "SuperGPQA": {
        "category": "Problem Solving",
        "benchmark_header": "Problem Solving Benchmark",
        "score_header": "Problem Solving Score",
        "accepted_aliases": ("supergpqa", "super gpqa"),
        "workload_group": "supergpqa_reasoning",
        "selection_policy": "workbook_missing",
        "is_full_benchmark": True,
        "decoding": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": 4096,
            "max_tokens_policy": "workload_expected_output_len",
        },
    },
    "SuperGPQA-hard": {
        "category": "Problem Solving",
        "benchmark_header": "Problem Solving Benchmark",
        "score_header": "Problem Solving Score",
        "accepted_aliases": ("supergpqa", "super gpqa"),
        "workload_group": "supergpqa_hard_reasoning",
        "selection_policy": "all_models",
        "is_full_benchmark": False,
        "decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0, "max_tokens": 4096},
    },
    "RepoBench": {
        "category": "Code Completion",
        "benchmark_header": "Code Completion Benchmark",
        "score_header": "Code Completion Score",
        "accepted_aliases": ("repobench", "repo bench"),
        "workload_group": "repobench_python_java_aggregate_cache_realistic",
        "selection_policy": "all_models",
        "exclude_model_substrings": ("gpt-oss",),
        "is_full_benchmark": True,
        "decoding": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": 512,
            "extra_body": {"stop": ["\n\n"]},
        },
    },
    "CrossCodeEval": {
        "category": "Code Completion",
        "benchmark_header": "Code Completion Benchmark",
        "score_header": "Code Completion Score",
        "accepted_aliases": ("crosscodeeval", "cross code eval", "cceval", "crosscode"),
        "workload_group": "crosscodeeval_rg1_unixcoder_cache_realistic",
        "selection_policy": "all_models",
        "exclude_model_substrings": ("gpt-oss",),
        "is_full_benchmark": True,
        "decoding": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": 512,
            "extra_body": {"stop": ["\n\n"]},
        },
    },
    "LongBench-v1-covered": {
        "category": "Long Context",
        "benchmark_header": "Long Context Benchmark",
        "score_header": "Long Context Score",
        "accepted_aliases": ("longbench", "longbench v1", "longbench-v1"),
        "rejected_aliases": ("longbench v2", "longbench-v2", "ruler", "mrcr"),
        "workload_group": "longbench_v1_covered",
        "selection_policy": "all_models",
        "exclude_model_substrings": ("llama-2",),
        "is_full_benchmark": False,
        "decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0, "max_tokens": 4096},
    },
}


@dataclass(frozen=True, slots=True)
class WorkloadGroup:
    name: str
    benchmark: str
    workload_paths: tuple[Path, ...]
    is_full_benchmark: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "benchmark": self.benchmark,
            "workload_paths": [str(path) for path in self.workload_paths],
            "is_full_benchmark": self.is_full_benchmark,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class MissingBenchmarkScore:
    model: str
    benchmark: str
    scorebook_category: str
    scorebook_benchmark_cell: str | None
    scorebook_score_cell: str | None
    reason: str
    workload_group: str
    workload_paths: tuple[Path, ...]
    is_full_benchmark: bool
    workbook_row: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "benchmark": self.benchmark,
            "scorebook_category": self.scorebook_category,
            "scorebook_benchmark_cell": self.scorebook_benchmark_cell,
            "scorebook_score_cell": self.scorebook_score_cell,
            "reason": self.reason,
            "workload_group": self.workload_group,
            "workload_paths": [str(path) for path in self.workload_paths],
            "is_full_benchmark": self.is_full_benchmark,
            "workbook_row": self.workbook_row,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkSelectionResult:
    scorebook: Path
    missing_scores_path: Path | None
    missing_scores_csv_path: Path | None
    parse_report_path: Path | None
    targets: tuple[str, ...]
    records: tuple[MissingBenchmarkScore, ...]
    skipped_rows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scorebook": str(self.scorebook),
            "missing_scores_path": None if self.missing_scores_path is None else str(self.missing_scores_path),
            "missing_scores_csv_path": (
                None if self.missing_scores_csv_path is None else str(self.missing_scores_csv_path)
            ),
            "parse_report_path": None if self.parse_report_path is None else str(self.parse_report_path),
            "targets": list(self.targets),
            "missing_count": len(self.records),
            "records": [record.to_dict() for record in self.records],
            "skipped_rows": list(self.skipped_rows),
        }


def select_missing_benchmark_scores(
    *,
    scorebook: str | Path,
    output_dir: str | Path | None = None,
    targets: Sequence[str] | None = None,
    repo_root: str | Path | None = None,
) -> BenchmarkSelectionResult:
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs
        raise RuntimeError("openpyxl is required to parse benchmark score workbooks") from exc

    scorebook_path = Path(scorebook).resolve()
    selected_targets = _normalize_targets(targets)
    repo = Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    workbook = openpyxl.load_workbook(scorebook_path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    header = next(rows, None)
    if header is None:
        raise ValueError(f"scorebook has no header row: {scorebook_path}")
    header_index = {str(value).strip(): index for index, value in enumerate(header) if value is not None}
    records: list[MissingBenchmarkScore] = []
    skipped_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        model = _cell_text(_cell(row, header_index, "Model"))
        if not model:
            continue
        if "/" not in model:
            skipped_rows.append({"row": row_number, "model_cell": model, "reason": "not a model id"})
            continue
        for target in selected_targets:
            spec = BENCHMARK_TARGETS[target]
            excluded_reason = _model_exclusion_reason(model, spec)
            if excluded_reason is not None:
                skipped_rows.append(
                    {
                        "row": row_number,
                        "model": model,
                        "benchmark": target,
                        "reason": excluded_reason,
                    }
                )
                continue
            benchmark_cell = _cell_text(_cell(row, header_index, str(spec["benchmark_header"])))
            score_cell = _cell_text(_cell(row, header_index, str(spec["score_header"])))
            missing, reason = _target_is_missing(
                target,
                benchmark_cell=benchmark_cell,
                score_cell=score_cell,
            )
            if not missing:
                continue
            workload_group = resolve_workload_group(target, repo_root=repo)
            records.append(
                MissingBenchmarkScore(
                    model=model,
                    benchmark=target,
                    scorebook_category=str(spec["category"]),
                    scorebook_benchmark_cell=benchmark_cell,
                    scorebook_score_cell=score_cell,
                    reason=reason,
                    workload_group=workload_group.name,
                    workload_paths=workload_group.workload_paths,
                    is_full_benchmark=workload_group.is_full_benchmark,
                    workbook_row=row_number,
                )
            )

    output_path = Path(output_dir).resolve() if output_dir is not None else None
    missing_json = missing_csv = parse_report = None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        missing_json = output_path / "missing_scores.json"
        missing_csv = output_path / "missing_scores.csv"
        parse_report = output_path / "workbook_parse_report.json"
        missing_json.write_text(
            json.dumps([record.to_dict() for record in records], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_missing_scores_csv(missing_csv, records)
        parse_report.write_text(
            json.dumps(
                {
                    "scorebook": str(scorebook_path),
                    "sheet": sheet.title,
                    "targets": list(selected_targets),
                    "missing_count": len(records),
                    "skipped_rows": skipped_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return BenchmarkSelectionResult(
        scorebook=scorebook_path,
        missing_scores_path=missing_json,
        missing_scores_csv_path=missing_csv,
        parse_report_path=parse_report,
        targets=selected_targets,
        records=tuple(records),
        skipped_rows=tuple(skipped_rows),
    )


def build_benchmark_generation_manifest(
    *,
    missing_plan: str | Path,
    base_manifest: str | Path,
    output_path: str | Path,
    run_id: str | None = None,
    include_benchmarks: Sequence[str] | None = None,
) -> dict[str, Any]:
    missing_records = _load_missing_plan(missing_plan)
    include = set(_normalize_targets(include_benchmarks)) if include_benchmarks else None
    selected = [record for record in missing_records if include is None or str(record["benchmark"]) in include]
    if not selected:
        raise ValueError("missing plan does not contain any records for the requested benchmark filter")

    base_path = Path(base_manifest).resolve()
    payload = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("base manifest must be a mapping")
    raw_experiments = payload.get("experiments")
    if not isinstance(raw_experiments, list):
        raise ValueError("base manifest must contain experiments")
    base_by_model = _base_experiment_index(raw_experiments)

    if run_id is not None:
        payload.setdefault("run", {})["run_id"] = run_id
    payload["experiments"] = []
    for record in selected:
        model = str(record["model"])
        benchmark = str(record["benchmark"])
        base_experiment = dict(base_by_model.get(model, {}))
        if not base_experiment:
            base_experiment = {"model": model}
        base_experiment.pop("workload", None)
        base_experiment.pop("workloads", None)
        base_experiment.pop("models", None)
        base_experiment["id"] = f"{slugify(model, max_length=42)}-{slugify(benchmark, max_length=28)}-responses"
        base_experiment["model"] = model
        base_experiment["workloads"] = list(record["workload_paths"])
        base_experiment["generation"] = _merge_benchmark_generation(
            base_experiment.get("generation"),
            BENCHMARK_TARGETS[benchmark]["decoding"],
        )
        payload["experiments"].append(base_experiment)

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return {
        "output": str(output),
        "base_manifest": str(base_path),
        "missing_plan": str(Path(missing_plan).resolve()),
        "experiment_count": len(payload["experiments"]),
        "benchmarks": sorted({str(record["benchmark"]) for record in selected}),
    }


def resolve_workload_group(benchmark: str, *, repo_root: str | Path | None = None) -> WorkloadGroup:
    target = _normalize_targets((benchmark,))[0]
    repo = Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    if target == "SuperGPQA":
        return _single_directory_group(
            repo,
            benchmark=target,
            name="supergpqa_reasoning",
            directory=Path("experiments/reasoning_workloads/supergpqa_reasoning/workload_yamls"),
            is_full_benchmark=True,
            notes="Existing SuperGPQA reasoning materialization.",
        )
    if target == "SuperGPQA-hard":
        return _single_directory_group(
            repo,
            benchmark=target,
            name="supergpqa_hard_reasoning",
            directory=Path("experiments/reasoning_workloads/supergpqa_hard_reasoning/workload_yamls"),
            is_full_benchmark=False,
            notes="Existing full hard-question subset materialization for MST/energy parity.",
        )
    if target == "RepoBench":
        return _single_directory_group(
            repo,
            benchmark=target,
            name="repobench_python_java_aggregate_cache_realistic",
            directory=Path(
                "experiments/code_workloads/repobench_python_java_aggregate_cache_realistic/workload_yamls"
            ),
            is_full_benchmark=True,
            notes="Existing Python/Java aggregate RepoBench materialization.",
        )
    if target == "CrossCodeEval":
        return _single_directory_group(
            repo,
            benchmark=target,
            name="crosscodeeval_rg1_unixcoder_cache_realistic",
            directory=Path(
                "experiments/code_workloads/crosscodeeval_rg1_unixcoder_cache_realistic/workload_yamls"
            ),
            is_full_benchmark=True,
            notes="Existing CrossCodeEval realistic materialization.",
        )
    if target == "LongBench-v1-covered":
        directories = (
            Path(
                "experiments/longbench_workloads/benchmark_original/"
                "longbench_long_output_summarization_original_official_qwen3_8b/workload_yamls"
            ),
            Path(
                "experiments/longbench_workloads/benchmark_original/"
                "longbench_medium_output_summarization_original_official_qwen3_8b/workload_yamls"
            ),
            Path(
                "experiments/longbench_workloads/benchmark_original/"
                "longbench_medium_answer_rag_qa_original_official_qwen3_8b/workload_yamls"
            ),
            Path(
                "experiments/longbench_workloads/benchmark_original/"
                "longbench_short_answer_document_qa_original_official_qwen3_8b/workload_yamls"
            ),
        )
        paths = tuple(path for directory in directories for path in _validated_longbench_paths(repo / directory))
        if not paths:
            raise FileNotFoundError(
                "no benchmark-original LongBench covered workload YAMLs found; "
                "materialize experiments/longbench_workloads/benchmark_original/*.yaml first"
            )
        return WorkloadGroup(
            name="longbench_v1_covered",
            benchmark=target,
            workload_paths=paths,
            is_full_benchmark=False,
            notes="Covered LongBench v1 original-task subset only; no latency/energy repeats or external expansion.",
        )
    raise ValueError(f"unsupported benchmark target: {benchmark}")


def _normalize_targets(targets: Sequence[str] | None) -> tuple[str, ...]:
    if targets is None:
        return tuple(BENCHMARK_TARGETS)
    normalized: list[str] = []
    aliases = {_normalize_text(name): name for name in BENCHMARK_TARGETS}
    aliases.update(
        {
            "supergpqahard": "SuperGPQA-hard",
            "longbenchv1covered": "LongBench-v1-covered",
            "longbench": "LongBench-v1-covered",
        }
    )
    for target in targets:
        key = _normalize_text(target)
        resolved = aliases.get(key)
        if resolved is None:
            raise ValueError(f"unsupported benchmark target: {target}")
        if resolved not in normalized:
            normalized.append(resolved)
    return tuple(normalized)


def _target_is_missing(
    target: str,
    *,
    benchmark_cell: str | None,
    score_cell: str | None,
) -> tuple[bool, str]:
    spec = BENCHMARK_TARGETS[target]
    if spec.get("selection_policy") == "all_models":
        return True, "target selected for all eligible models"
    if _is_empty_or_na(score_cell):
        return True, "score cell empty or N/A"
    if _is_empty_or_na(benchmark_cell):
        return True, "benchmark cell empty or N/A"
    normalized = _normalize_text(benchmark_cell)
    accepted_aliases = tuple(_normalize_text(str(alias)) for alias in spec["accepted_aliases"])
    if target == "LongBench-v1-covered" and "longbenchv1" in normalized:
        return False, "target benchmark present"
    for rejected in spec.get("rejected_aliases", ()):
        if _normalize_text(str(rejected)) in normalized:
            return True, "only rejected substitute benchmark present"
    if any(alias in normalized for alias in accepted_aliases):
        return False, "target benchmark present"
    return True, "target benchmark missing"


def _model_exclusion_reason(model: str, spec: dict[str, Any]) -> str | None:
    normalized = model.lower()
    for fragment in spec.get("exclude_model_substrings", ()):
        if str(fragment).lower() in normalized:
            return f"model excluded for benchmark target: contains {fragment}"
    return None


def _is_empty_or_na(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    return normalized in {"", "n/a", "na", "none", "-"}


def _normalize_text(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cell(row: Sequence[Any], header_index: dict[str, int], header: str) -> Any:
    index = header_index.get(header)
    if index is None or index >= len(row):
        return None
    return row[index]


def _cell_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _workload_paths(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        return ()
    return tuple(sorted(path.resolve() for path in directory.glob("*.yaml") if path.is_file()))


def _single_directory_group(
    repo: Path,
    *,
    benchmark: str,
    name: str,
    directory: Path,
    is_full_benchmark: bool,
    notes: str,
) -> WorkloadGroup:
    paths = _workload_paths(repo / directory)
    if not paths:
        raise FileNotFoundError(f"no workload YAMLs found for {benchmark}: {repo / directory}")
    return WorkloadGroup(
        name=name,
        benchmark=benchmark,
        workload_paths=paths,
        is_full_benchmark=is_full_benchmark,
        notes=notes,
    )


def _validated_longbench_paths(workload_dir: Path) -> tuple[Path, ...]:
    paths = _workload_paths(workload_dir)
    if not paths:
        raise FileNotFoundError(
            f"missing LongBench benchmark-original workload directory: {workload_dir}"
        )
    report_path = workload_dir.parent / "materialization_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing LongBench materialization report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sampling = report.get("sampling") if isinstance(report, dict) else None
    rows = report.get("rows") if isinstance(report, dict) else None
    if not isinstance(sampling, dict) or not isinstance(rows, dict):
        raise ValueError(f"LongBench materialization report is missing rows/sampling: {report_path}")
    prompt_template = report.get("prompt_template")
    if prompt_template != "longbench_official":
        raise ValueError(
            "LongBench benchmark target requires official LongBench prompts; "
            f"got prompt_template={prompt_template!r} in {report_path}. "
            "Rematerialize experiments/longbench_workloads/benchmark_original/*.yaml first."
        )
    repeat_policy = sampling.get("repeat_policy")
    materialized = int(rows.get("materialized", -1))
    expanded = int(sampling.get("expanded_sample_count", -2))
    unique = int(sampling.get("unique_sample_ids", -3))
    if repeat_policy is not None or expanded != unique or materialized != unique:
        raise ValueError(
            "LongBench benchmark target requires no-repeat original materialization; "
            f"got repeat_policy={repeat_policy!r}, materialized={materialized}, "
            f"expanded={expanded}, unique={unique} in {report_path}"
        )
    selected_tasks = report.get("selected_tasks")
    if not isinstance(selected_tasks, list) or not selected_tasks:
        raise ValueError(f"LongBench materialization report has no selected_tasks: {report_path}")
    if any(str(task).endswith("_original") for task in selected_tasks):
        raise ValueError(f"LongBench benchmark target must not use expanded external tasks: {report_path}")
    return paths


def _write_missing_scores_csv(path: Path, records: Iterable[MissingBenchmarkScore]) -> None:
    fieldnames = (
        "model",
        "benchmark",
        "scorebook_category",
        "scorebook_benchmark_cell",
        "scorebook_score_cell",
        "reason",
        "workload_group",
        "is_full_benchmark",
        "workbook_row",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            payload = record.to_dict()
            writer.writerow({field: payload[field] for field in fieldnames})


def _load_missing_plan(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("missing plan must be a JSON list")
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"missing plan record {index} must be a mapping")
        for field in ("model", "benchmark", "workload_paths"):
            if field not in record:
                raise ValueError(f"missing plan record {index} is missing {field}")
    return payload


def _base_experiment_index(experiments: Sequence[Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in experiments:
        if not isinstance(item, dict):
            continue
        models = []
        if isinstance(item.get("model"), str):
            models.append(str(item["model"]))
        if isinstance(item.get("models"), list):
            models.extend(str(model) for model in item["models"] if isinstance(model, str))
        for model in models:
            indexed[model] = dict(item)
    return indexed


def _merge_benchmark_generation(base_generation: Any, decoding_overrides: dict[str, Any]) -> dict[str, Any]:
    generation = dict(base_generation) if isinstance(base_generation, dict) else {}
    generation["concurrency_source"] = generation.get("concurrency_source", "explicit")
    if generation["concurrency_source"] != "explicit":
        generation["concurrency_source"] = "explicit"
    generation.setdefault("max_concurrency", 1)
    generation.setdefault("preserve_request_order", True)
    generation.setdefault("include_prompt_text", True)
    generation.setdefault("response_text_max_chars", 65536)
    decoding = dict(generation.get("decoding") or {})
    decoding.update({key: value for key, value in decoding_overrides.items() if key != "extra_body"})
    decoding["n"] = 1
    decoding.setdefault("max_tokens_policy", "model_context_minus_prompt_buffer")
    decoding.setdefault("prompt_token_buffer", 128)
    extra_body = dict(decoding.get("extra_body") or {})
    extra_body.update(dict(decoding_overrides.get("extra_body") or {}))
    decoding["extra_body"] = extra_body
    generation["decoding"] = decoding
    return generation
