from __future__ import annotations

from pathlib import Path

from llm_mst_finder.plotting import plot_search_results, plot_trial_windows


def test_plot_trial_windows_writes_placeholder_for_missing_optional_latency_series(
    tmp_path: Path,
) -> None:
    paths = plot_trial_windows(
        trial_dir=tmp_path / "trial",
        x_values=[0.0, 10.0],
        arrival_rate=[0.1, 0.2],
        completion_rate=[0.1, 0.2],
        outstanding=[0.0, 0.0],
        ttft_p50_ms=[None, None],
        ttft_p90_ms=[None, None],
        ttft_p95_ms=[None, None],
        ttft_p99_ms=[None, None],
        tpot_p50_ms=[None, None],
        tpot_p90_ms=[None, None],
        tpot_p95_ms=[None, None],
        tpot_p99_ms=[None, None],
        output_tok_s=[0.1, 0.2],
        kv_cache_usage=[0.0, 0.0],
        num_running=[0.0, 0.0],
        num_waiting=[0.0, 0.0],
        num_swapped=[None, None],
    )

    assert Path(paths["ttft_percentiles"]).is_file()
    assert Path(paths["tpot_percentiles"]).is_file()


def test_plot_search_results_writes_placeholder_for_missing_optional_latency_points(
    tmp_path: Path,
) -> None:
    paths = plot_search_results(
        output_dir=tmp_path,
        request_rates=[0.1, 0.2],
        classifications=["valid/stable", "valid/stable"],
        ttft_p90_ms=[None, None],
        tpot_p90_ms=[None, None],
        output_tok_s=[1.0, 2.0],
        queue_drift=[0.0, 0.0],
    )

    assert Path(paths["search_rate_vs_ttft_p90"]).is_file()
    assert Path(paths["search_rate_vs_tpot_p90"]).is_file()
