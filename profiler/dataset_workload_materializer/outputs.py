from __future__ import annotations

import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import yaml

from .common import optional_mapping
from .models import Counters, MaterializedSample


def cache_realistic_order(
    samples: list[MaterializedSample],
    *,
    seed: int,
    burst_size: int,
) -> list[MaterializedSample]:
    by_session: dict[str, list[MaterializedSample]] = defaultdict(list)
    for sample in samples:
        by_session[str(sample.metadata["session_id"])].append(sample)
    session_ids = sorted(by_session)
    rng = random.Random(seed)
    rng.shuffle(session_ids)
    queues: deque[tuple[str, deque[MaterializedSample]]] = deque()
    for session_id in session_ids:
        ordered = sorted(
            by_session[session_id],
            key=sample_order_key,
        )
        queues.append((session_id, deque(ordered)))
    output: list[MaterializedSample] = []
    while queues:
        session_id, queue = queues.popleft()
        del session_id
        for _ in range(burst_size):
            if not queue:
                break
            output.append(queue.popleft())
        if queue:
            queues.append(("", queue))
    return output


def epoch_shuffle_expand(
    samples: list[MaterializedSample],
    *,
    seed: int,
    target_samples: int,
) -> list[MaterializedSample]:
    if target_samples <= len(samples):
        return list(samples)
    unique_count = len(samples)
    expanded: list[MaterializedSample] = []
    epoch_index = 0
    while len(expanded) < target_samples:
        epoch = list(samples)
        rng = random.Random(seed + epoch_index * 1_000_003)
        rng.shuffle(epoch)
        for epoch_position, sample in enumerate(epoch):
            if len(expanded) >= target_samples:
                break
            original_sample_id = str(
                sample.metadata.get("original_sample_id", sample.sample_id)
            )
            materialized_sample_id = f"{original_sample_id}::epoch-{epoch_index:04d}"
            metadata = dict(sample.metadata)
            metadata.update(
                {
                    "sample_id": materialized_sample_id,
                    "original_sample_id": original_sample_id,
                    "repeat_policy": "epoch_shuffle",
                    "epoch_index": epoch_index,
                    "epoch_position": epoch_position,
                    "epoch_shuffle_seed": seed,
                    "unique_sample_count": unique_count,
                    "expanded_sample_count": target_samples,
                }
            )
            expanded.append(
                MaterializedSample(
                    sample_id=materialized_sample_id,
                    prompt=sample.prompt,
                    target=sample.target,
                    expected_output_len=sample.expected_output_len,
                    metadata=metadata,
                )
            )
        epoch_index += 1
    return expanded


def sample_order_key(sample: MaterializedSample) -> tuple[int, str, str]:
    return (
        int(sample.metadata["sequence_index"]),
        str(sample.metadata["file_path"]),
        str(sample.metadata["content_hash"]),
    )


def shard_samples(
    samples: list[MaterializedSample],
    *,
    samples_per_shard: int,
    requested_num_shards: int | None,
) -> list[list[MaterializedSample]]:
    shards = [
        samples[index : index + samples_per_shard]
        for index in range(0, len(samples), samples_per_shard)
    ]
    if requested_num_shards is not None and len(shards) > requested_num_shards:
        raise ValueError(
            "materialized samples exceed sharding capacity: "
            f"need {len(shards)} shards with samples_per_shard={samples_per_shard}, "
            f"but sharding.num_shards={requested_num_shards}"
        )
    return [shard for shard in shards if shard]


def write_outputs(
    output_dir: Path,
    *,
    name: str,
    shards: list[list[MaterializedSample]],
    tokenizer_name: str,
    request: dict[str, Any],
    context_policy: dict[str, Any],
    output_len: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(shards):
        shard_id = f"shard_{shard_index:03d}"
        shard_path = output_dir / "shards" / f"{shard_id}.runner.jsonl"
        workload_path = output_dir / "workload_yamls" / f"{shard_id}.yaml"
        with shard_path.open("w", encoding="utf-8") as handle:
            for sample in shard:
                metadata = dict(sample.metadata)
                metadata["shard_id"] = shard_id
                metadata["prompt_tokenizer_key"] = f"materialized:{tokenizer_name}"
                handle.write(
                    json.dumps(
                        {
                            "prompt": sample.prompt,
                            "prompt_len": int(sample.metadata["prompt_token_count"]),
                            "expected_output_len": sample.expected_output_len,
                            "metadata": metadata,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        workload_payload = workload_yaml_payload(
            name=name,
            shard_id=shard_id,
            shard_size=len(shard),
            tokenizer_name=tokenizer_name,
            request=request,
            context_policy=context_policy,
            output_len=output_len,
        )
        workload_path.write_text(
            yaml.safe_dump(workload_payload, sort_keys=False),
            encoding="utf-8",
        )
        entries.append(
            {
                "shard_id": shard_id,
                "num_samples": len(shard),
                "path": str(shard_path.relative_to(output_dir)),
                "workload_yaml_path": str(workload_path.relative_to(output_dir)),
            }
        )
    return entries


def workload_yaml_payload(
    *,
    name: str,
    shard_id: str,
    shard_size: int,
    tokenizer_name: str,
    request: dict[str, Any],
    context_policy: dict[str, Any],
    output_len: dict[str, Any],
) -> dict[str, Any]:
    request_payload: dict[str, Any] = {
        "stream": request.get("stream", True),
        "temperature": request.get("temperature", 0.0),
        "ignore_eos": request.get("ignore_eos", False),
    }
    if "top_p" in request:
        request_payload["top_p"] = request["top_p"]
    extra_body = dict(optional_mapping(request.get("extra_body"), "request.extra_body"))
    if "stop" in request:
        extra_body["stop"] = request["stop"]
    if extra_body:
        request_payload["extra_body"] = extra_body
    output_len_payload = dict(output_len) if output_len else {"mode": "from_dataset"}
    payload: dict[str, Any] = {
        "name": f"{name}-{shard_id}",
        "dataset": {
            "type": "jsonl",
            "path": f"../shards/{shard_id}.runner.jsonl",
        },
        "sampling": {
            "seed": 42,
            "num_requests": shard_size,
            "entry_selection": "sequential",
            "prompt_len": {"mode": "from_dataset"},
            "output_len": output_len_payload,
        },
        "request": request_payload,
    }
    if context_policy:
        payload["context_policy"] = context_policy
    return payload


def build_manifest(
    *,
    name: str,
    dataset_name: str,
    dataset_kind: str,
    task: str,
    profile: str | None,
    prompt_template: str,
    shards: list[list[MaterializedSample]],
    shard_entries: list[dict[str, Any]],
    samples_per_shard: int,
    selected_tasks: list[str] | None,
) -> dict[str, Any]:
    all_samples = [sample for shard in shards for sample in shard]
    manifest = {
        "workload_name": name,
        "dataset": dataset_name,
        "dataset_kind": dataset_kind,
        "task": task,
        "prompt_template": prompt_template,
        "num_shards": len(shards),
        "samples_per_shard": samples_per_shard,
        "shards": shard_entries,
        "selected_tasks": selected_tasks or sorted({str(sample.metadata["task"]) for sample in all_samples}),
        "language_counts": dict(Counter(str(sample.metadata["language"]) for sample in all_samples)),
        "prompt_tokens": summary([int(sample.metadata["prompt_token_count"]) for sample in all_samples]),
        "target_tokens": summary([int(sample.metadata["target_token_count"]) for sample in all_samples]),
        "profile_summaries": group_summaries(all_samples, group_key="profile"),
        "task_summaries": group_summaries(all_samples, group_key="task"),
        "unique_content_hashes": len({sample.metadata["content_hash"] for sample in all_samples}),
        "unique_group_ids": len(
            {
                sample.metadata["group_id"]
                for sample in all_samples
                if sample.metadata.get("group_id") not in (None, "")
            }
        ),
        "unique_sample_ids": len(
            {
                sample.metadata.get("original_sample_id", sample.sample_id)
                for sample in all_samples
            }
        ),
    }
    repeat_policy = sample_repeat_policy(all_samples)
    if repeat_policy is not None:
        manifest["repeat_policy"] = repeat_policy
        manifest["repeat_factor"] = len(all_samples) / manifest["unique_sample_ids"]
    if profile is not None:
        manifest["profile"] = profile
    return manifest


def build_report(
    *,
    name: str,
    dataset_name: str,
    dataset_kind: str,
    task: str,
    profile: str | None,
    prompt_template: str,
    raw_path: Path | None,
    output_dir: Path,
    tokenizer_name: str,
    samples: list[MaterializedSample],
    counters: Counters,
    selected_tasks: list[str] | None,
) -> dict[str, Any]:
    report = {
        "workload_name": name,
        "dataset": dataset_name,
        "dataset_kind": dataset_kind,
        "task": task,
        "prompt_template": prompt_template,
        "raw_path": str(raw_path) if raw_path is not None else None,
        "output_dir": str(output_dir),
        "selected_tasks": selected_tasks or sorted({str(sample.metadata["task"]) for sample in samples}),
        "tokenizer": {
            "name": tokenizer_name,
            "fallback_used": False,
        },
        "rows": {
            "total": counters.total_rows,
            "materialized": len(samples),
            "drops": dict(counters.drops),
        },
        "language_counts": dict(Counter(str(sample.metadata["language"]) for sample in samples)),
        "prompt_tokens": summary([int(sample.metadata["prompt_token_count"]) for sample in samples]),
        "target_tokens": summary([int(sample.metadata["target_token_count"]) for sample in samples]),
        "profile_summaries": group_summaries(samples, group_key="profile"),
        "task_summaries": group_summaries(samples, group_key="task"),
    }
    group_ids = [
        str(sample.metadata["group_id"])
        for sample in samples
        if sample.metadata.get("group_id") not in (None, "")
    ]
    if group_ids:
        group_counts = Counter(group_ids)
        report["group_id_reuse"] = {
            "unique_group_ids": len(group_counts),
            "max_reuse": max(group_counts.values()),
            "reused_group_ids": sum(1 for count in group_counts.values() if count > 1),
        }
    unique_sample_ids = len(
        {sample.metadata.get("original_sample_id", sample.sample_id) for sample in samples}
    )
    report["sampling"] = {
        "unique_sample_ids": unique_sample_ids,
        "expanded_sample_count": len(samples),
        "repeat_policy": sample_repeat_policy(samples),
        "repeat_factor": len(samples) / unique_sample_ids if unique_sample_ids else None,
    }
    if profile is not None:
        report["profile"] = profile
    return report


def sample_repeat_policy(samples: list[MaterializedSample]) -> str | None:
    policies = {
        str(sample.metadata["repeat_policy"])
        for sample in samples
        if sample.metadata.get("repeat_policy")
    }
    if not policies:
        return None
    if len(policies) != 1:
        raise ValueError(f"mixed repeat policies in materialized samples: {sorted(policies)}")
    return next(iter(policies))


def group_summaries(
    samples: list[MaterializedSample],
    *,
    group_key: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[MaterializedSample]] = defaultdict(list)
    for sample in samples:
        value = sample.metadata.get(group_key)
        if value in (None, ""):
            continue
        groups[str(value)].append(sample)
    summaries: dict[str, dict[str, Any]] = {}
    for group_name in sorted(groups):
        grouped_samples = groups[group_name]
        payload: dict[str, Any] = {
            "count": len(grouped_samples),
            "prompt_tokens": summary([int(sample.metadata["prompt_token_count"]) for sample in grouped_samples]),
            "target_tokens": summary([int(sample.metadata["target_token_count"]) for sample in grouped_samples]),
        }
        if group_key == "profile":
            payload["task_counts"] = dict(Counter(str(sample.metadata["task"]) for sample in grouped_samples))
        if group_key == "task":
            payload["language_counts"] = dict(Counter(str(sample.metadata["language"]) for sample in grouped_samples))
            profile_counts = Counter(
                str(sample.metadata["profile"])
                for sample in grouped_samples
                if sample.metadata.get("profile") not in (None, "")
            )
            if profile_counts:
                payload["profile_counts"] = dict(profile_counts)
            first_metadata = grouped_samples[0].metadata
            if "workload_type" in first_metadata:
                payload["workload_type"] = first_metadata["workload_type"]
            if "output_regime" in first_metadata:
                payload["output_regime"] = first_metadata["output_regime"]
        summaries[group_name] = payload
    return summaries


def summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p90": None, "p95": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
    }


def percentile(ordered: list[int], q: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
