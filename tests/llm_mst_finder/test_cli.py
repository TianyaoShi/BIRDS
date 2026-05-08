from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from llm_mst_finder.cli import main
from llm_mst_finder.records import RequestRecord, SampleRequest


FIXTURES_ROOT = Path(__file__).parent / "fixtures"


class StubRequestClient:
    def __init__(self, **kwargs) -> None:
        del kwargs
        self.closed = False

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True

    async def send_request(
        self,
        sample_request: SampleRequest,
        *,
        request_id: str,
        trial_id: str,
        scheduled_send_ts: float,
    ) -> RequestRecord:
        await asyncio.sleep(0.002)
        actual_send_ts = scheduled_send_ts + 0.001
        first_token_ts = actual_send_ts + 0.001
        end_ts = actual_send_ts + 0.002
        return RequestRecord(
            request_id=request_id,
            trial_id=trial_id,
            scheduled_send_ts=scheduled_send_ts,
            actual_send_ts=actual_send_ts,
            first_token_ts=first_token_ts,
            end_ts=end_ts,
            success=True,
            error=None,
            prompt_len=sample_request.prompt_len,
            expected_output_len=sample_request.expected_output_len,
            actual_output_len=2,
            ttft_s=first_token_ts - actual_send_ts,
            e2e_s=end_ts - actual_send_ts,
            tpot_s=end_ts - first_token_ts,
            itl_s=[end_ts - first_token_ts],
            output_token_timestamps=[first_token_ts, end_ts],
            metadata=sample_request.metadata,
        )


def test_cli_run_trial_closed_loop_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("llm_mst_finder.cli.RequestClient", StubRequestClient)
    workload_path = FIXTURES_ROOT / "workloads" / "synthetic_fixed_512_128.yaml"
    output_dir = tmp_path / "cli-trial"

    exit_code = main(
        [
            "run-trial",
            "--trial-id",
            "cli-trial",
            "--mode",
            "closed-loop",
            "--concurrency",
            "1",
            "--duration-s",
            "0.01",
            "--model",
            "fake-model",
            "--workload",
            str(workload_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["summary"]["trial_id"] == "cli-trial"
    assert (output_dir / "request_records.jsonl").exists()
    assert (output_dir / "windows.csv").exists()
    captured = json.loads(capsys.readouterr().out.strip())
    assert captured["status"] == "completed"
    assert captured["trial_id"] == "cli-trial"


def test_cli_run_trial_records_context_policy_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("llm_mst_finder.cli.RequestClient", StubRequestClient)
    workload_path = tmp_path / "context_workload.yaml"
    workload_path.write_text(
        "\n".join(
            [
                "name: context-aware-synthetic",
                "dataset:",
                "  type: synthetic-fixed",
                "  prompt: alpha beta gamma delta epsilon",
                "tokenizer: character",
                "sampling:",
                "  seed: 1",
                "  num_requests: 1",
                "  prompt_len:",
                "    mode: fixed",
                "    value: 5",
                "  output_len:",
                "    mode: fixed",
                "    value: 2",
                "context_policy:",
                "  max_model_len: 5",
                "  tokenizer_source: workload_tokenizer",
                "  over_limit: truncate_prompt",
                "  truncation_side: left",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "cli-context-trial"

    exit_code = main(
        [
            "run-trial",
            "--trial-id",
            "cli-context-trial",
            "--mode",
            "closed-loop",
            "--concurrency",
            "1",
            "--duration-s",
            "0.01",
            "--model",
            "fake-model",
            "--workload",
            str(workload_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    context_policy = summary_payload["config"]["metadata"]["workload"]["context_policy"]
    assert context_policy["max_model_len"] == 5
    assert context_policy["truncated_samples"] == 1
    assert context_policy["skipped_samples"] == 0
    request_records = (output_dir / "request_records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert request_records
    assert all(json.loads(line)["prompt_len"] == 3 for line in request_records)


def test_cli_run_trial_records_server_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("llm_mst_finder.cli.RequestClient", StubRequestClient)
    workload_path = FIXTURES_ROOT / "workloads" / "synthetic_fixed_512_128.yaml"
    output_dir = tmp_path / "cli-server-metadata-trial"

    exit_code = main(
        [
            "run-trial",
            "--trial-id",
            "cli-server-metadata-trial",
            "--mode",
            "closed-loop",
            "--concurrency",
            "1",
            "--duration-s",
            "0.01",
            "--model",
            "fake-model",
            "--workload",
            str(workload_path),
            "--output-dir",
            str(output_dir),
            "--max-num-seqs",
            "64",
            "--max-num-batched-tokens",
            "4096",
        ]
    )

    assert exit_code == 0
    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    metadata = summary_payload["config"]["metadata"]
    assert "workload" in metadata
    assert metadata["server_metadata"]["max_num_seqs"] == 64.0
    assert metadata["server_metadata"]["max_num_batched_tokens"] == 4096.0
    assert summary_payload["summary"]["metadata"]["server_metadata"] == metadata["server_metadata"]


def test_cli_search_records_server_metadata_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    class StubSearchResult:
        def __init__(self, search_id: str) -> None:
            self.search_id = search_id

        def to_dict(self) -> dict[str, object]:
            return {"search_id": self.search_id, "max_no_drift_request_rate": 1.0}

    class StubSearchController:
        instances: list["StubSearchController"] = []

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.config = None
            StubSearchController.instances.append(self)

        async def search(self, config):
            self.config = config
            return StubSearchResult(config.search_id)

    monkeypatch.setattr("llm_mst_finder.cli.RequestClient", StubRequestClient)
    monkeypatch.setattr("llm_mst_finder.cli.SearchController", StubSearchController)
    workload_path = FIXTURES_ROOT / "workloads" / "synthetic_fixed_512_128.yaml"
    metadata_path = tmp_path / "server_metadata.json"
    metadata_path.write_text(
        json.dumps({"server_config": {"max_num_seqs": 128}, "engine": "vllm"}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "search",
            "--search-id",
            "cli-search-metadata",
            "--search-mode",
            "open-loop",
            "--output-dir",
            str(tmp_path / "search"),
            "--model",
            "fake-model",
            "--workload",
            str(workload_path),
            "--trial-min-duration-s",
            "10",
            "--trial-max-duration-s",
            "25",
            "--uncertain-trial-duration-multiplier",
            "3",
            "--closed-loop-min-trials",
            "4",
            "--ttft-slo-ms",
            "1500",
            "--ttft-slo-field",
            "ttft_p99_ms",
            "--tpot-slo-ms",
            "none",
            "--server-metadata-file",
            str(metadata_path),
            "--max-num-batched-tokens",
            "8192",
        ]
    )

    assert exit_code == 0
    captured = json.loads(capsys.readouterr().out.strip())
    assert captured["search_id"] == "cli-search-metadata"
    config = StubSearchController.instances[-1].config
    assert config.trial_duration_s == 10.0
    assert config.uncertain_trial_duration_s == 25.0
    assert config.uncertain_trial_duration_multiplier == 3.0
    assert config.final_confirmation_duration_s == 25.0
    assert config.closed_loop_min_trials == 4
    metadata = config.metadata
    assert metadata["stability_policy"]["ttft_slo_ms"] == 1500.0
    assert metadata["stability_policy"]["ttft_slo_field"] == "ttft_p99_ms"
    assert metadata["stability_policy"]["tpot_slo_ms"] is None
    assert "workload" in metadata
    server_metadata = metadata["server_metadata"]
    assert server_metadata["engine"] == "vllm"
    assert server_metadata["server_config"]["max_num_seqs"] == 128
    assert server_metadata["max_num_seqs"] == 128.0
    assert server_metadata["max_num_batched_tokens"] == 8192.0


def test_cli_rejects_conflicting_server_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("llm_mst_finder.cli.RequestClient", StubRequestClient)
    workload_path = FIXTURES_ROOT / "workloads" / "synthetic_fixed_512_128.yaml"
    metadata_path = tmp_path / "server_metadata.json"
    metadata_path.write_text(json.dumps({"max_num_seqs": 64}), encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting supplied server metadata values"):
        main(
            [
                "run-trial",
                "--trial-id",
                "cli-conflict-trial",
                "--mode",
                "closed-loop",
                "--concurrency",
                "1",
                "--duration-s",
                "0.01",
                "--model",
                "fake-model",
                "--workload",
                str(workload_path),
                "--output-dir",
                str(tmp_path / "conflict"),
                "--server-metadata-file",
                str(metadata_path),
                "--max-num-seqs",
                "32",
            ]
        )


def test_cli_analyze_writes_analysis_artifact(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("llm_mst_finder.cli.RequestClient", StubRequestClient)
    workload_path = FIXTURES_ROOT / "workloads" / "synthetic_fixed_512_128.yaml"
    output_dir = tmp_path / "cli-analyze-trial"

    run_exit_code = main(
        [
            "run-trial",
            "--trial-id",
            "cli-analyze-trial",
            "--mode",
            "closed-loop",
            "--concurrency",
            "1",
            "--duration-s",
            "0.01",
            "--model",
            "fake-model",
            "--workload",
            str(workload_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert run_exit_code == 0
    capsys.readouterr()

    analyze_exit_code = main(
        [
            "analyze",
            "--trial-dir",
            str(output_dir),
        ]
    )

    assert analyze_exit_code == 0
    analysis_payload = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
    assert analysis_payload["trial_validity"] == "valid"
    assert analysis_payload["stability"]["status"] in {"stable", "uncertain"}
    captured = json.loads(capsys.readouterr().out.strip())
    assert captured["trial_id"] == "cli-analyze-trial"
    assert captured["trial_validity"] == "valid"


def test_cli_report_writes_final_artifacts(tmp_path: Path, capsys) -> None:
    from tests.llm_mst_finder.test_reporting import _write_result_dir

    result_dir = tmp_path / "report-run"
    _write_result_dir(result_dir)

    exit_code = main(
        [
            "report",
            "--result-dir",
            str(result_dir),
            "--disable-plots",
        ]
    )

    assert exit_code == 0
    assert (result_dir / "final_report.json").is_file()
    assert (result_dir / "final_report.md").is_file()
    captured = json.loads(capsys.readouterr().out.strip())
    assert captured["comparison_included"] is False
