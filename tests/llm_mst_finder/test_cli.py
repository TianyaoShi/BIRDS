from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llm_mst_finder.cli import main
from llm_mst_finder.records import RequestRecord, SampleRequest


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
    workload_path = Path("profiler/llm_mst_finder/workloads/synthetic_fixed_512_128.yaml")
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
