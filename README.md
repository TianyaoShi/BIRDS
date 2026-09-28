# 🐦 BIRDS

### Characterizing and Understanding Biodiversity Impact of Large Language Model Serving

[![Paper](https://img.shields.io/badge/arXiv-2605.27480-b31b1b.svg)](https://arxiv.org/abs/2605.27480)
[![Venue](https://img.shields.io/badge/EMNLP_2026-Findings-2f855a.svg)](https://arxiv.org/abs/2605.27480)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](https://www.python.org/)

**BIRDS** is a framework for **Biodiversity Impact of Request-Driven LLM
Serving**. It measures ecosystem damage from LLM inference at the request level,
combining operational electricity use, embodied hardware impacts, serving
performance, and response quality.

The paper was accepted to **Findings of EMNLP 2026**.

📄 [Paper](https://arxiv.org/abs/2605.27480) · 💻 [Code](https://github.com/TianyaoShi/BIRDS) · 🌿 [Project Page](https://tianyaoshi.github.io/BIRDS/)

## 🌿 Why BIRDS?

The environmental cost of LLM serving extends beyond carbon emissions and water
consumption. Electricity generation and hardware lifecycles also affect
ecosystems through acidification, eutrophication, ecotoxicity, ozone formation,
and other pathways.

BIRDS provides a serving-oriented way to study these effects across workloads,
models, GPUs, operating points, and deployment regions.

## 🧭 Framework

BIRDS follows three steps:

1. **Define a request-level functional unit** under a workload, serving
   configuration, throughput, and latency SLO.
2. **Quantify biodiversity impact per request** by combining operational and
   embodied lifecycle impacts.
3. **Account for response quality** with Quality-Normalized Biodiversity Impact
   (QNBI), where lower values indicate less biodiversity impact per unit of
   achieved task quality.

## 🔎 Key Findings

- 📈 Small per-token impacts accumulate into meaningful ecosystem damage at
  trillion-token serving scales.
- ⚡ Operational electricity use contributes more than 95% of total impact on
  average, making serving efficiency a primary mitigation lever.
- 🌍 Biodiversity-aware deployment choices can differ from carbon- or
  water-aware choices because biodiversity integrates multiple ecological
  pathways.
- ⚖️ The model with the lowest impact per request is not always best after
  response quality is considered.
- 🧠 Intermediate dense models and sparse mid-sized MoE models often offer
  favorable quality-normalized impact, while long outputs, reasoning modes, older GPUs, and
  multi-GPU overhead can increase impact.

The evaluation covers conversational, reasoning, code-completion, and
long-context workloads; dense and MoE models from 0.6B to 235B parameters; and
NVIDIA L40, A100, and H100 GPUs.

## 🗂️ Repository Map

```text
bi_modeling/
  code/                  Biodiversity-impact and lifecycle models
  visualization/         Analysis and paper-figure utilities

profiler/
  llm_mst_finder/        Maximum sustainable throughput profiling
  local_orchestrator/    Local GPU scheduling and experiment execution
  slurm_orchestrator/    Slurm planning, submission, and collection
  energy_profiler/       GPU power and per-request energy measurement
  output_quality_profiler/  Response generation and quality evaluation
  dataset_workload_materializer/  Dataset-to-workload preparation
  mst_analyzer/          MST anomaly analysis and rerun planning

tests/                   Unit and integration tests
requirements/            Profiler and vLLM dependency sets
website/                 Project page, interactive results, and data sources
```

## 🚀 Getting Started

Install the profiler dependencies:

```bash
python -m pip install -r requirements/profiler.txt
export PYTHONPATH="$PWD/profiler:$PWD"
```

Each module has focused usage documentation:

- [`llm_mst_finder`](profiler/llm_mst_finder/README.md)
- [`local_orchestrator`](profiler/local_orchestrator/README.md)
- [`slurm_orchestrator`](profiler/slurm_orchestrator/README.md)
- [`energy_profiler`](profiler/energy_profiler/README.md)
- [`output_quality_profiler`](profiler/output_quality_profiler/README.md)
- [`dataset_workload_materializer`](profiler/dataset_workload_materializer/README.md)
- [`mst_analyzer`](profiler/mst_analyzer/README.md)

This public release contains the core modeling, profiling, orchestration, and
analysis modules. Private operational scripts, generated results, and
cluster-specific experiment manifests are intentionally excluded.

## 📚 Citation

If BIRDS is useful in your work, please cite:

```bibtex
@inproceedings{shi2026birds,
  title     = {{BIRDS}: Characterizing and Understanding Biodiversity Impact of Large Language Model Serving},
  author    = {Shi, Tianyao and Ding, Yi},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026},
  eprint    = {2605.27480},
  archivePrefix = {arXiv},
}
```

## 🙏 Acknowledgments

BIRDS builds on open-source LLM serving, evaluation, dataset, and lifecycle
assessment ecosystems. Please see the paper for the complete methodology,
datasets, tools, and references.

## 📜 License

BIRDS is released under the [Apache License 2.0](LICENSE). Portions of the
serving benchmark implementation are derived from the Apache-2.0-licensed vLLM
project. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
