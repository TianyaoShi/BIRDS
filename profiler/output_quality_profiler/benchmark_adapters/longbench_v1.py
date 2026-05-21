from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from .base import (
    ground_truth_values,
    load_response_rows,
    metadata,
    model_name_from_rows,
    normalize_text,
    write_score_artifacts,
)


try:  # pragma: no cover - depends on optional benchmark packages.
    import jieba as _jieba
except ImportError:  # pragma: no cover - covered through dependency report.
    _jieba = None

try:  # pragma: no cover - depends on optional benchmark packages.
    from rouge import Rouge as _Rouge
except ImportError:  # pragma: no cover - covered through dependency report.
    _Rouge = None


DATASET_METRICS: dict[str, str] = {
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "multifieldqa_zh": "qa_f1_zh",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "dureader": "rouge_zh",
    "gov_report": "rouge",
    "qmsum": "rouge",
    "multi_news": "rouge",
    "vcsum": "rouge_zh",
    "trec": "classification",
    "triviaqa": "qa_f1",
    "samsum": "rouge",
    "lsht": "classification",
    "passage_retrieval_en": "retrieval",
    "passage_count": "count",
    "passage_retrieval_zh": "retrieval_zh",
    "lcc": "code_sim",
    "repobench-p": "code_sim",
}

COVERED_TASKS = {
    "dureader",
    "gov_report",
    "gov_report_e",
    "multi_news",
    "multi_news_e",
    "multifieldqa_en",
    "multifieldqa_en_e",
    "multifieldqa_zh",
    "qasper",
    "qasper_e",
    "qmsum",
    "vcsum",
}

PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "long_output_summarization": {
        "label": "Long-output summarization",
        "metric_family": "ROUGE-style summarization",
        "tasks": {"gov_report", "gov_report_e"},
    },
    "medium_output_summarization": {
        "label": "Medium-output summarization",
        "metric_family": "ROUGE-style summarization",
        "tasks": {"multi_news", "multi_news_e", "qmsum", "vcsum"},
    },
    "medium_answer_rag_qa": {
        "label": "Medium-answer RAG QA",
        "metric_family": "LongBench Chinese ROUGE-style QA",
        "tasks": {"dureader"},
    },
    "short_answer_document_qa": {
        "label": "Short-answer document QA",
        "metric_family": "F1-style QA",
        "tasks": {"multifieldqa_en", "multifieldqa_en_e", "multifieldqa_zh", "qasper", "qasper_e"},
    },
}

TASK_TO_PROFILE = {
    task: profile
    for profile, spec in PROFILE_SPECS.items()
    for task in spec["tasks"]
}

FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum", "lsht"}


def score_longbench_v1_responses(
    *,
    responses_root: str | Path,
    output_dir: str | Path,
    raw_data_root: str | Path | None = None,
) -> dict[str, Any]:
    rows = load_response_rows(responses_root)
    raw_index = load_longbench_raw_index(raw_data_root or default_longbench_raw_data_root())
    per_item: list[dict[str, Any]] = []
    failed = invalid = scored = raw_matches = raw_mismatches = 0
    scores: list[float] = []
    by_task: dict[str, list[float]] = {}
    by_metric_task: dict[str, list[float]] = {}
    by_bucket_task: dict[str, dict[str, list[float]]] = {}
    for index, row in enumerate(rows):
        meta = metadata(row)
        request_id = str(row.get("request_id") or meta.get("sample_id") or index)
        task = _row_longbench_task(row, meta)
        metric_task = metric_task_name(task)
        bucket = _row_profile(meta, task=task)
        if not row.get("success", False):
            failed += 1
            per_item.append(
                {
                    "request_id": request_id,
                    "task": task,
                    "metric_task": metric_task,
                    "bucket": bucket,
                    "invalid_reason": "failed_generation",
                }
            )
            continue
        raw_entry = _resolve_raw_entry(meta, task=task, raw_index=raw_index)
        if raw_entry is None:
            references = ground_truth_values(row)
            all_classes: list[str] = []
            length = meta.get("longbench_length")
            raw_match = False
        else:
            references = raw_entry.answers
            all_classes = raw_entry.all_classes
            length = raw_entry.length
            raw_match = _raw_entry_matches_response(raw_entry, meta)
            raw_matches += int(raw_match)
            raw_mismatches += int(not raw_match)
        if not references:
            invalid += 1
            per_item.append(
                {
                    "request_id": request_id,
                    "task": task,
                    "metric_task": metric_task,
                    "bucket": bucket,
                    "invalid_reason": "missing_ground_truth",
                }
            )
            continue
        prediction = str(row.get("response_text") or "")
        item_score = score_longbench_item(
            prediction,
            references=references,
            task=task,
            all_classes=all_classes,
        )
        scored += 1
        scores.append(item_score)
        by_task.setdefault(task, []).append(item_score)
        by_metric_task.setdefault(metric_task, []).append(item_score)
        by_bucket_task.setdefault(bucket, {}).setdefault(task, []).append(item_score)
        per_item.append(
            {
                "request_id": request_id,
                "task": task,
                "metric_task": metric_task,
                "bucket": bucket,
                "prediction": prediction,
                "ground_truth": references,
                "length": length,
                "raw_reference_match": raw_match,
                "score": 100.0 * item_score,
            }
        )
    by_task_payload = {
        task: _score_group_payload(values, metric=DATASET_METRICS.get(metric_task_name(task)))
        for task, values in sorted(by_task.items())
    }
    task_scores = [payload["score"] for payload in by_task_payload.values() if payload["score"] is not None]
    score = {
        "benchmark": "LongBench-v1-covered",
        "model": model_name_from_rows(rows),
        "adapter": "longbench_v1_official_metrics_covered_v1",
        "metric": "covered_longbench_task_mean_percent",
        "overall_score": None if not task_scores else sum(task_scores) / len(task_scores),
        "item_weighted_score": None if not scores else 100.0 * sum(scores) / len(scores),
        "total_items": len(rows),
        "scored_items": scored,
        "failed_generations": failed,
        "invalid_items": invalid,
        "raw_reference_matches": raw_matches,
        "raw_reference_mismatches": raw_mismatches,
        "by_task": by_task_payload,
        "by_metric_task": {
            task: _score_group_payload(values, metric=DATASET_METRICS.get(task))
            for task, values in sorted(by_metric_task.items())
        },
        "by_bucket": _bucket_score_payload(by_bucket_task),
        "is_full_benchmark": False,
        "raw_data_root": str(Path(raw_data_root or default_longbench_raw_data_root()).resolve()),
        "covered_tasks": sorted(by_task),
        "unexpected_tasks": sorted(set(by_task) - COVERED_TASKS),
        "dependency_status": longbench_dependency_status(),
        "score_interpretation": (
            "Use by_bucket for the main workload-track interpretation. overall_score is a "
            "cross-bucket macro average retained for LongBench-style comparability and mixes "
            "ROUGE-style and F1-style metrics."
        ),
        "compatibility_note": (
            "Scores the materialized LongBench v1 covered-task subset with the original LongBench "
            "task-to-metric mapping. Overall score is the mean of covered task scores, not a full "
            "LongBench v1 leaderboard score."
        ),
    }
    return write_score_artifacts(
        output_dir=output_dir,
        score=score,
        per_item=per_item,
        markdown_title="LongBench v1 Covered Score",
    )


class LongBenchRawEntry:
    def __init__(
        self,
        *,
        task: str,
        row_index: int,
        row_id: str | None,
        answers: list[str],
        all_classes: list[str],
        length: int | None,
    ) -> None:
        self.task = task
        self.row_index = row_index
        self.row_id = row_id
        self.answers = answers
        self.all_classes = all_classes
        self.length = length


def default_longbench_raw_data_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "raw" / "longbench"


def load_longbench_raw_index(raw_data_root: str | Path) -> dict[tuple[str, int], LongBenchRawEntry]:
    root = Path(raw_data_root).resolve()
    if not root.is_dir():
        return {}
    index: dict[tuple[str, int], LongBenchRawEntry] = {}
    for path in sorted(root.glob("*.jsonl")):
        task = path.stem
        with path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{row_index + 1}: expected a JSON object")
                answers = row.get("answers")
                if not isinstance(answers, list):
                    answers = [] if answers is None else [answers]
                all_classes = row.get("all_classes")
                if not isinstance(all_classes, list):
                    all_classes = []
                length = row.get("length")
                index[(task, row_index)] = LongBenchRawEntry(
                    task=task,
                    row_index=row_index,
                    row_id=str(row["_id"]) if row.get("_id") is not None else None,
                    answers=[str(answer) for answer in answers if answer is not None],
                    all_classes=[str(value) for value in all_classes if value is not None],
                    length=int(length) if isinstance(length, int) else None,
                )
    return index


def score_longbench_item(
    prediction: str,
    *,
    references: Sequence[str],
    task: str,
    all_classes: Sequence[str] = (),
) -> float:
    metric_task = metric_task_name(task)
    if metric_task in FIRST_LINE_TASKS:
        prediction = prediction.lstrip("\n").split("\n")[0]
    scorer = _metric_scorer(metric_task)
    return max(scorer(prediction, reference, all_classes=all_classes) for reference in references)


def metric_task_name(task: str) -> str:
    return task[:-2] if task.endswith("_e") else task


def _metric_scorer(task: str) -> Callable[..., float]:
    metric = DATASET_METRICS.get(task)
    if metric == "qa_f1":
        return qa_f1_score
    if metric == "qa_f1_zh":
        return qa_f1_zh_score
    if metric == "rouge":
        return rouge_score
    if metric == "rouge_zh":
        return rouge_zh_score
    if metric == "classification":
        return classification_score
    if metric == "retrieval":
        return retrieval_score
    if metric == "retrieval_zh":
        return retrieval_zh_score
    if metric == "count":
        return count_score
    if metric == "code_sim":
        return code_sim_score
    raise ValueError(f"unsupported LongBench task for scoring: {task}")


def _row_longbench_task(row: dict[str, Any], meta: dict[str, Any]) -> str:
    raw = (
        meta.get("longbench_dataset")
        or meta.get("task")
        or meta.get("longbench_task")
        or row.get("dataset_task")
        or "unknown"
    )
    return str(raw)


def _row_profile(meta: dict[str, Any], *, task: str) -> str:
    raw_profile = meta.get("profile")
    if isinstance(raw_profile, str) and raw_profile:
        return raw_profile
    return TASK_TO_PROFILE.get(task, "unknown")


def _resolve_raw_entry(
    meta: dict[str, Any],
    *,
    task: str,
    raw_index: dict[tuple[str, int], LongBenchRawEntry],
) -> LongBenchRawEntry | None:
    row_index = meta.get("longbench_row_index")
    if not isinstance(row_index, int):
        return None
    return raw_index.get((task, row_index))


def _raw_entry_matches_response(raw_entry: LongBenchRawEntry, meta: dict[str, Any]) -> bool:
    response_id = meta.get("longbench_id")
    if response_id is not None and raw_entry.row_id is not None:
        return str(response_id) == raw_entry.row_id
    response_ground_truth = meta.get("ground_truth")
    if response_ground_truth is not None:
        return str(response_ground_truth) in raw_entry.answers
    return True


def _score_group_payload(values: Sequence[float], *, metric: str | None) -> dict[str, Any]:
    return {
        "count": len(values),
        "metric": metric,
        "score": 100.0 * sum(values) / len(values) if values else None,
    }


def _bucket_score_payload(by_bucket_task: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for bucket, task_values in sorted(by_bucket_task.items()):
        by_task = {
            task: _score_group_payload(values, metric=DATASET_METRICS.get(metric_task_name(task)))
            for task, values in sorted(task_values.items())
        }
        task_scores = [task_payload["score"] for task_payload in by_task.values() if task_payload["score"] is not None]
        item_scores = [score for values in task_values.values() for score in values]
        spec = PROFILE_SPECS.get(bucket, {})
        payload[bucket] = {
            "label": spec.get("label", bucket),
            "metric_family": spec.get("metric_family", "unknown"),
            "count": len(item_scores),
            "score": None if not task_scores else sum(task_scores) / len(task_scores),
            "score_type": "task_macro_percent",
            "item_weighted_score": None if not item_scores else 100.0 * sum(item_scores) / len(item_scores),
            "by_task": by_task,
        }
    return payload


def longbench_dependency_status() -> dict[str, bool]:
    return {
        "jieba": _jieba is not None,
        "rouge": _Rouge is not None,
        "fuzzywuzzy": _fuzzy_ratio_available(),
    }


def normalize_answer(s: str) -> str:
    """LongBench v1 English QA normalization."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def normalize_zh_answer(s: str) -> str:
    """LongBench v1 Chinese QA normalization."""

    cn_punctuation = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》"
        "「」『』〖〗〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    )
    all_punctuation = set(string.punctuation + cn_punctuation)
    return "".join(ch for ch in s.lower() if ch not in all_punctuation and not ch.isspace())


def count_score(prediction: str, ground_truth: str, **_: Any) -> float:
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth))
    return 0.0 if not numbers else float(right_num / len(numbers))


def retrieval_score(prediction: str, ground_truth: str, **_: Any) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    return count_score(prediction, matches[0])


def retrieval_zh_score(prediction: str, ground_truth: str, **_: Any) -> float:
    matches = re.findall(r"段落(\d+)", ground_truth)
    if not matches:
        return 0.0
    return count_score(prediction, matches[0])


def code_sim_score(prediction: str, ground_truth: str, **_: Any) -> float:
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            return _fuzzy_ratio(line, ground_truth) / 100.0
    return 0.0


def classification_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    all_classes = kwargs["all_classes"]
    em_match_list = [class_name for class_name in all_classes if class_name in prediction]
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def rouge_score(prediction: str, ground_truth: str, **_: Any) -> float:
    if _Rouge is not None:
        rouge = _Rouge()
        try:
            scores = rouge.get_scores([prediction], [ground_truth], avg=True)
        except Exception:
            return 0.0
        return float(scores["rouge-l"]["f"])
    return rouge_l_f1(prediction, ground_truth)


def rouge_zh_score(prediction: str, ground_truth: str, **_: Any) -> float:
    prediction_tokens = _zh_tokens(prediction)
    ground_truth_tokens = _zh_tokens(ground_truth)
    if _Rouge is not None:
        return rouge_score(" ".join(prediction_tokens), " ".join(ground_truth_tokens))
    return rouge_l_f1_tokens(prediction_tokens, ground_truth_tokens)


def f1_score(prediction: Sequence[str], ground_truth: Sequence[str]) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **_: Any) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction: str, ground_truth: str, **_: Any) -> float:
    prediction_tokens = [normalize_zh_answer(token) for token in _zh_tokens(prediction)]
    ground_truth_tokens = [normalize_zh_answer(token) for token in _zh_tokens(ground_truth)]
    prediction_tokens = [token for token in prediction_tokens if token]
    ground_truth_tokens = [token for token in ground_truth_tokens if token]
    return f1_score(prediction_tokens, ground_truth_tokens)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        count = ref_counts.get(token, 0)
        if count > 0:
            overlap += 1
            ref_counts[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    return rouge_l_f1_tokens(pred_tokens, ref_tokens)


def rouge_l_f1_tokens(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> float:
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _tokens(value: str) -> list[str]:
    normalized = normalize_text(value)
    return re.findall(r"[\w]+", normalized)


def _zh_tokens(value: str) -> list[str]:
    if _jieba is not None:
        return list(_jieba.cut(value, cut_all=False))
    return [char for char in value if not char.isspace()]


def _fuzzy_ratio_available() -> bool:
    try:
        import fuzzywuzzy.fuzz  # noqa: F401
    except ImportError:
        return False
    return True


def _fuzzy_ratio(prediction: str, ground_truth: str) -> int:
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        import difflib

        return int(round(100 * difflib.SequenceMatcher(None, prediction, ground_truth).ratio()))
    return int(fuzz.ratio(prediction, ground_truth))


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]
