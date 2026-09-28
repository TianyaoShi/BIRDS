"""Export only approved figure fields from the preserved EMNLP derived CSVs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

WORKLOADS = [
    ("sharegpt", "Everyday chat", "5, 6", "h100_sharegpt_model_qnbi_selected_configs.csv"),
    ("crosscodeeval", "Code completion / CrossCodeEval", "11, 19", "workload_comparison/h100_crosscodeeval_model_qnbi_selected_configs.csv"),
    ("repobench", "Code completion / RepoBench", "12, 20", "workload_comparison/h100_repobench_model_qnbi_selected_configs.csv"),
    ("mmlu-pro", "Reasoning / MMLU-Pro", "13, 21", "workload_comparison/h100_mmlu-pro_model_qnbi_selected_configs.csv"),
    ("supergpqa", "Reasoning / SuperGPQA", "14, 22", "workload_comparison/h100_supergpqa_model_qnbi_selected_configs.csv"),
    ("longbench_long_output_summarization", "LongBench / Long summarization", "15, 23", "workload_comparison/h100_longbench_long_output_summarization_model_qnbi_selected_configs.csv"),
    ("longbench_medium_output_summarization", "LongBench / Medium summarization", "16, 24", "workload_comparison/h100_longbench_medium_output_summarization_model_qnbi_selected_configs.csv"),
    ("longbench_medium_answer_rag_qa", "LongBench / RAG QA", "17, 25", "workload_comparison/h100_longbench_medium_answer_rag_qa_model_qnbi_selected_configs.csv"),
    ("longbench_short_answer_document_qa", "LongBench / Document QA", "18, 26", "workload_comparison/h100_longbench_short_answer_document_qa_model_qnbi_selected_configs.csv"),
]


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {key}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("derived", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "src/results.json")
    args = parser.parse_args()
    provenance = []

    def read(name):
        path = args.derived / name
        provenance.append({"file": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        with path.open(newline="") as source:
            return list(csv.DictReader(source))

    def selected(row):
        bi = number(row, "bi_per_request")
        qnbi = number(row, "qnbi_per_request")
        quality = number(row, "normalized_quality_score")
        if not 0 < quality <= 1 or not math.isclose(bi / quality, qnbi, rel_tol=1e-8):
            raise ValueError("BI / quality does not match QNBI")
        return {"model": row["model"].split("/")[-1], "gpu": row["accelerator"],
                "bi": bi, "qnbi": qnbi, "quality": quality}

    workloads = []
    for key, label, figures, filename in WORKLOADS:
        rows = []
        for row in read(filename):
            if not row.get("qnbi_per_request") or not row.get("normalized_quality_score"):
                continue
            rows.append({**selected(row), "family": row["family"],
                         "size": number(row, "size_b"), "moe": row["is_moe"].lower() == "true"})
        if not rows or len({r["model"] for r in rows}) != len(rows):
            raise ValueError(f"Empty or duplicate workload {key}")
        workloads.append({"key": key, "label": label, "figures": figures, "rows": rows})

    lifecycle = [{"label": r["stage"], "share": number(r, "mean")} for r in read("combined_h100_a100_lifecycle_stage_ratio_summary.csv")]
    midpoint = [{"horizon": int(r["time_horizon_years"]), "label": r["bucket"], "share": number(r, "mean")} for r in read("combined_h100_a100_midpoint_perspective_mean_ratio_summary.csv")]
    for group in [lifecycle] + [[r for r in midpoint if r["horizon"] == h] for h in [20, 100, 1000]]:
        if not math.isclose(sum(r["share"] for r in group), 1, abs_tol=1e-8):
            raise ValueError("Contribution shares do not sum to one")
    reasoning = [{**selected(r), "size": r["size_label"], "mode": r["series_label"]} for r in read("qwen_reasoning_mmlu_case_study_rows.csv")]
    gpus = {family: [selected(r) for r in read(f"sharegpt_{family}_gpu_type_case_study_rows.csv")] for family in ["qwen", "gemma"]}
    result = {"paper": "2605.27480v3", "archive": "visualization_emnlp", "units": "species-year per request",
              "selection": "Archived paper-selected configurations; H100 with archived A100 supplements where applicable",
              "energy_basis": "incremental (as recorded in archived figure inputs)",
              "workloads": workloads, "lifecycle": lifecycle, "midpoint": midpoint,
              "reasoning": reasoning, "gpus": gpus, "provenance": provenance}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Exported {sum(len(w['rows']) for w in workloads)} model/workload values, {len(reasoning)} reasoning values, and {sum(map(len, gpus.values()))} GPU values")


if __name__ == "__main__":
    main()
