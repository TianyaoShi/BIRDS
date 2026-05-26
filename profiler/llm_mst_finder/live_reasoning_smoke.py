from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


MODELS = [
    {
        "family": "qwen",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "gpu": 0,
        "port": 8500,
        "dtype": "float16",
    },
    {
        "family": "gemma",
        "model": "google/gemma-4-E4B-it",
        "gpu": 1,
        "port": 8501,
        "dtype": "float16",
    },
    {
        "family": "llama",
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "gpu": 2,
        "port": 8502,
        "dtype": "float16",
    },
    {
        "family": "gpt_oss",
        "model": "openai/gpt-oss-20b",
        "gpu": 3,
        "port": 8503,
        "dtype": "auto",
    },
]

def main() -> int:
    parser = argparse.ArgumentParser(description="Run short live reasoning run-trial smoke tests.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/live_reasoning_run_trials"),
    )
    parser.add_argument("--samples-per-workload", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--readiness-timeout-s", type=float, default=900.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--workload",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
        help="repeat with a logical workload name and workload YAML path",
    )
    args = parser.parse_args()

    if args.samples_per_workload <= 0:
        raise ValueError("--samples-per-workload must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")

    workloads = [{"name": name, "path": Path(path)} for name, path in args.workload]
    output_root = args.output_root.resolve()
    logs_dir = output_root / "logs"
    workloads_dir = output_root / "workloads"
    logs_dir.mkdir(parents=True, exist_ok=True)
    workloads_dir.mkdir(parents=True, exist_ok=True)

    prepared_workloads = {
        workload["name"]: _write_short_workload(
            source_path=workload["path"],
            output_path=workloads_dir / f"{workload['name']}.yaml",
            samples_per_workload=args.samples_per_workload,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
        )
        for workload in workloads
    }

    servers = []
    try:
        for spec in MODELS:
            servers.append(_start_server(spec, logs_dir=logs_dir, max_model_len=args.max_model_len))
        for spec, process in servers:
            _wait_ready(
                f"http://127.0.0.1:{spec['port']}",
                process=process,
                stderr_log=logs_dir / f"{_slug(spec['model'])}.stderr.log",
                timeout_s=args.readiness_timeout_s,
            )

        decoded_path = output_root / "decoded_responses.jsonl"
        summary: list[dict[str, Any]] = []
        source_rows = {
            workload_name: _load_workload_source_rows(workload_path)
            for workload_name, workload_path in prepared_workloads.items()
        }
        with decoded_path.open("w", encoding="utf-8") as decoded_handle:
            for spec, _ in servers:
                for workload_name, workload_path in prepared_workloads.items():
                    trial_dir = output_root / "trials" / _slug(spec["model"]) / workload_name
                    result = _run_trial(
                        spec=spec,
                        workload_name=workload_name,
                        workload_path=workload_path,
                        trial_dir=trial_dir,
                        duration_s=args.duration_s,
                    )
                    records = _load_jsonl(trial_dir / "request_records.jsonl")
                    stats = _summarize_records(records)
                    stats.update(
                        {
                            "family": spec["family"],
                            "model": spec["model"],
                            "workload": workload_name,
                            "trial_dir": str(trial_dir),
                            "run_trial_returncode": result.returncode,
                        }
                    )
                    summary.append(stats)
                    for record in records:
                        metadata = record.get("metadata", {})
                        source_row = source_rows[workload_name][int(metadata["source_index"])]
                        decoded_handle.write(
                            json.dumps(
                                {
                                    "actual_output_len": record.get("actual_output_len"),
                                    "error": record.get("error"),
                                    "family": spec["family"],
                                    "ground_truth": source_row["metadata"].get("ground_truth"),
                                    "model": spec["model"],
                                    "prompt": source_row.get("prompt"),
                                    "prompt_len": record.get("prompt_len"),
                                    "request_id": record.get("request_id"),
                                    "response_text": metadata.get("response_text", ""),
                                    "response_text_truncated": metadata.get("response_text_truncated"),
                                    "sample_id": metadata.get("sample_id"),
                                    "success": record.get("success"),
                                    "task": metadata.get("task"),
                                    "workload": workload_name,
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"decoded_responses": str(decoded_path), "summary": str(summary_path)}, sort_keys=True))
        for row in summary:
            print(json.dumps(row, sort_keys=True))
        return 0
    finally:
        for spec, process in reversed(servers):
            _stop_process(process, model=str(spec["model"]))


def _write_short_workload(
    *,
    source_path: Path,
    output_path: Path,
    samples_per_workload: int,
    max_tokens: int,
    max_model_len: int,
) -> Path:
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    source_dataset_path = (source_path.parent / payload["dataset"]["path"]).resolve()
    payload["dataset"]["path"] = str(source_dataset_path)
    payload["sampling"]["num_requests"] = samples_per_workload
    payload["sampling"]["output_len"] = {"mode": "natural_until_eos", "max_tokens": max_tokens}
    payload["context_policy"]["max_model_len"] = max_model_len
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def _start_server(spec: dict[str, Any], *, logs_dir: Path, max_model_len: int) -> tuple[dict[str, Any], subprocess.Popen]:
    slug = _slug(str(spec["model"]))
    stdout_log = logs_dir / f"{slug}.stdout.log"
    stderr_log = logs_dir / f"{slug}.stderr.log"
    command = [
        "vllm",
        "serve",
        str(spec["model"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(spec["port"]),
        "--served-model-name",
        str(spec["model"]),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        str(spec["dtype"]),
        "--gpu-memory-utilization",
        "0.88",
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        "16",
        "--max-num-batched-tokens",
        "8192",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
    env.setdefault("MPLCONFIGDIR", str(logs_dir / "matplotlib"))
    stdout_handle = stdout_log.open("w", encoding="utf-8")
    stderr_handle = stderr_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        env=env,
        preexec_fn=os.setsid,
    )
    process._stdout_handle = stdout_handle  # type: ignore[attr-defined]
    process._stderr_handle = stderr_handle  # type: ignore[attr-defined]
    return spec, process


def _wait_ready(
    base_url: str,
    *,
    process: subprocess.Popen,
    stderr_log: Path,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr_tail = _tail(stderr_log)
            raise RuntimeError(f"vLLM exited before readiness for {base_url}:\n{stderr_tail}")
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2.0)
    raise TimeoutError(f"timed out waiting for {base_url}/v1/models")


def _run_trial(
    *,
    spec: dict[str, Any],
    workload_name: str,
    workload_path: Path,
    trial_dir: Path,
    duration_s: float,
) -> subprocess.CompletedProcess:
    trial_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "llm_mst_finder.cli",
        "run-trial",
        "--trial-id",
        f"live-reasoning-{spec['family']}-{workload_name}",
        "--mode",
        "closed-loop",
        "--concurrency",
        "1",
        "--duration-s",
        str(duration_s),
        "--base-url",
        f"http://127.0.0.1:{spec['port']}",
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        str(spec["model"]),
        "--workload",
        str(workload_path),
        "--request-reuse-policy",
        "no-repeat-per-trial",
        "--request-timeout-s",
        "240",
        "--window-s",
        "5",
        "--output-dir",
        str(trial_dir),
    ]
    env = os.environ.copy()
    env["LLM_MST_FINDER_CAPTURE_RESPONSE_TEXT"] = "1"
    env["LLM_MST_FINDER_RESPONSE_TEXT_MAX_CHARS"] = "4096"
    return subprocess.run(command, check=True, text=True, env=env)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _load_workload_source_rows(workload_path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(workload_path.read_text(encoding="utf-8"))
    dataset_path = (workload_path.parent / payload["dataset"]["path"]).resolve()
    return _load_jsonl(dataset_path)


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [record for record in records if record.get("success")]
    nonempty = [
        record
        for record in successes
        if str(record.get("metadata", {}).get("response_text", "")).strip()
    ]
    return {
        "requests": len(records),
        "successes": len(successes),
        "nonempty_responses": len(nonempty),
        "failed": len(records) - len(successes),
        "first_response_prefix": (
            str(nonempty[0]["metadata"]["response_text"])[:240] if nonempty else ""
        ),
    }


def _stop_process(process: subprocess.Popen, *, model: str) -> None:
    try:
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=10)
    finally:
        for attr in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(process, attr, None)
            if handle is not None:
                handle.close()
        del model


def _tail(path: Path, *, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


if __name__ == "__main__":
    raise SystemExit(main())
