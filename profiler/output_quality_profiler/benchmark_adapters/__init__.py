from __future__ import annotations

from pathlib import Path
from typing import Any

from .code_completion import score_code_completion_responses
from .longbench_v1 import score_longbench_v1_responses
from .supergpqa import score_supergpqa_responses


def score_benchmark_responses(
    *,
    benchmark: str,
    responses_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    normalized = _normalize_benchmark_name(benchmark)
    if normalized == "supergpqa":
        return score_supergpqa_responses(responses_root=responses_root, output_dir=output_dir)
    if normalized == "repobench":
        return score_code_completion_responses(
            responses_root=responses_root,
            output_dir=output_dir,
            benchmark_name="RepoBench",
        )
    if normalized in {"crosscodeeval", "cceval"}:
        return score_code_completion_responses(
            responses_root=responses_root,
            output_dir=output_dir,
            benchmark_name="CrossCodeEval",
        )
    if normalized in {"longbenchv1covered", "longbenchcovered", "longbench"}:
        return score_longbench_v1_responses(responses_root=responses_root, output_dir=output_dir)
    raise ValueError(f"unsupported benchmark adapter: {benchmark}")


def _normalize_benchmark_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())
