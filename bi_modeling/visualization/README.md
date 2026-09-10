# Paper Figure Map

This directory contains the visualization code used for the final BIRDS paper.
Figure numbers refer to [arXiv:2605.27480v3](https://arxiv.org/abs/2605.27480).

| Paper item | Generator | Subject |
| --- | --- | --- |
| Figure 3 | `plot_energy_bi_violins.py` | Per-request and per-token BI distributions and aggregate token-traffic impact |
| Figure 4 | `plot_combined_lifecycle_midpoint_1x2.py` | Lifecycle-stage and midpoint-to-endpoint contributions |
| Figure 5 | `plot_model_bi_per_request_bars.py` | ShareGPT BI and model quality |
| Figure 6 | `plot_model_qnbi_scatter.py` | ShareGPT QNBI by model family |
| Figure 7 | `plot_workload_bi_per_request_violin_length_box.py` | BI and request-length distributions across workloads |
| Figure 8 | `plot_qwen_reasoning_case_study.py` | Qwen3 Instruct versus Thinking on MMLU-Pro |
| Figure 9 | `plot_traffic_load_case_study.py` | Gemma-4 traffic-load, QNBI, and latency analysis |
| Figure 10 | `plot_gpu_type_case_study.py` | Qwen3 GPU-generation comparison |
| Figures 11-18 | `plot_model_bi_per_request_bars.py` | Appendix workload BI and quality results |
| Figures 19-26 | `plot_model_qnbi_scatter.py` | Appendix workload QNBI results |
| Figure 27 | `plot_gpu_type_case_study.py` | Gemma-4 GPU-generation comparison |
| Table 1 | `analyze_operational_region_energy_impacts.py` | Regional grid midpoint intensities and operational BI |

## Supporting Modules

- `energy_bi_calculator.py` builds the BI datasets consumed by the figure scripts.
- `analyze_h100_lifecycle_stage_breakdown.py`, `analyze_combined_midpoint_breakdown.py`, and `analyze_combined_midpoint_perspective_breakdown.py` prepare Figure 4 inputs. The lifecycle module is also shared by the model-level plots.
- `plot_workload_bi_per_request_violin.py` supplies shared workload data and plotting helpers used by Figure 7.
- `plot_style.py` and `VISUALIZATION_STYLE.md` define the common paper style.

The scripts expect the profiling result tree used by the paper. Large experiment artifacts are intentionally excluded from this public repository.
