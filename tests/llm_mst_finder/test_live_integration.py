from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from llm_mst_finder.cli import main


LIVE_SERVER_ENV = "LLM_MST_FINDER_RUN_LIVE"
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DATASET_PATH = (
    "/local/scratch/a/shi676/llm_profiling/datasets/shareGPT/"
    "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
)
REQUEST_RATES = (1.0, 2.0)


pytestmark = [
    pytest.mark.skipif(
        os.environ.get(LIVE_SERVER_ENV) != "1",
        reason=(
            f"set {LIVE_SERVER_ENV}=1 to run live vLLM integration tests against "
            "localhost:8000 and the real ShareGPT dataset"
        ),
    ),
    pytest.mark.filterwarnings(
        "ignore:builtin type SwigPyPacked has no __module__ attribute:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:builtin type SwigPyObject has no __module__ attribute:DeprecationWarning"
    ),
]


def test_live_sharegpt_open_loop_trials_at_two_request_rates(tmp_path: Path, capsys) -> None:
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"live ShareGPT dataset not found: {dataset_path}")

    workload_path = tmp_path / "live_sharegpt_workload.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: live-sharegpt-open-loop",
                "dataset:",
                "  type: sharegpt",
                f"  path: {dataset_path}",
                f"tokenizer: {MODEL_NAME}",
                "sampling:",
                "  seed: 123",
                "  num_requests: 12",
                "  prompt_len:",
                "    mode: from_dataset",
                "  output_len:",
                "    mode: from_dataset",
                "request:",
                "  extra_body:",
                "    ignore_eos: false",
                "context_policy:",
                "  max_model_len: 4096",
                "  tokenizer_source: vllm_model_config",
                f"  tokenizer: {MODEL_NAME}",
                "  over_limit: truncate_prompt",
                "  truncation_side: left",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    for request_rate in REQUEST_RATES:
        trial_id = f"live-rate-{str(request_rate).replace('.', '-')}"
        output_dir = tmp_path / trial_id
        run_exit_code = main(
            [
                "run-trial",
                "--trial-id",
                trial_id,
                "--mode",
                "open-loop",
                "--request-rate",
                str(request_rate),
                "--duration-s",
                "6.0",
                "--window-s",
                "1.0",
                "--base-url",
                "http://127.0.0.1:8000",
                "--endpoint",
                "/v1/completions",
                "--model",
                MODEL_NAME,
                "--workload",
                str(workload_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        assert run_exit_code == 0
        run_stdout = json.loads(capsys.readouterr().out.strip())
        assert run_stdout["trial_id"] == trial_id
        assert run_stdout["status"] == "completed"

        summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        summary = summary_payload["summary"]
        assert summary_payload["config"]["model"] == MODEL_NAME
        assert summary_payload["config"]["request_rate"] == request_rate
        assert summary["trial_id"] == trial_id
        assert summary["started_requests"] >= 1
        assert summary["successful_requests"] >= 1
        assert summary["failed_requests"] == 0
        assert (output_dir / "request_records.jsonl").is_file()
        assert (output_dir / "windows.csv").is_file()

        analyze_exit_code = main(
            [
                "analyze",
                "--trial-dir",
                str(output_dir),
            ]
        )
        assert analyze_exit_code == 0
        analysis_stdout = json.loads(capsys.readouterr().out.strip())
        analysis_payload = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
        assert analysis_stdout["trial_id"] == trial_id
        assert analysis_payload["trial_id"] == trial_id
        assert analysis_payload["trial_validity"] in {"valid", "client_limited"}
        assert analysis_payload["trial_validity"] != "invalid_workload"
        assert analysis_payload["trial_validity"] != "metrics_invalid"
        if analysis_payload["trial_validity"] == "valid":
            assert analysis_payload["stability"] is not None
            assert analysis_payload["bottleneck"] is not None
        else:
            assert analysis_payload["stability"] is None
            assert analysis_payload["bottleneck"] is None
            assert analysis_payload["validity_reasons"]
