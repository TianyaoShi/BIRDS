from __future__ import annotations

from typing import Any, Callable

from ..models import DatasetLoadResult, MaterializationContext
from .code import load_code_dataset
from .longbench import load_longbench_dataset
from .reasoning import load_reasoning_dataset


DatasetLoader = Callable[[dict[str, Any], MaterializationContext], DatasetLoadResult]


DATASET_LOADERS: dict[str, DatasetLoader] = {
    "aime": load_reasoning_dataset,
    "crosscodeeval": load_code_dataset,
    "gpqa": load_reasoning_dataset,
    "longbench": load_longbench_dataset,
    "mmlu": load_reasoning_dataset,
    "mmlu_pro": load_reasoning_dataset,
    "natural_reasoning": load_reasoning_dataset,
    "repobench": load_code_dataset,
    "supergpqa": load_reasoning_dataset,
}


def load_dataset(dataset: dict[str, Any], ctx: MaterializationContext) -> DatasetLoadResult:
    try:
        loader = DATASET_LOADERS[ctx.dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_LOADERS))
        raise ValueError(f"supported dataset.name values: {supported}") from exc
    return loader(dataset, ctx)
