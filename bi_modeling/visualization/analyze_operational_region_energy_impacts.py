"""Calculate midpoint and endpoint impacts for 1 kWh electricity by region."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bi_modeling.code import load_data, modeling  # noqa: E402

YEAR = 2024
ENERGY_KWH = 1.0
DATACENTER_PUE = 1.16
FALLBACK_DIRECT_WUE_L_PER_KWH = 0.36
DIRECT_WUE_PATH = REPO_ROOT / "bi_modeling" / "data" / "wet_bulb_temperature" / "regional_direct_wue_compact.json"
PERSPECTIVES = ("individualistic", "hierarchist", "egalitarian")
CORE_GRID_FACTORS = ("CO2e", "SOx", "NOx", "EWF")
REGION_SET = ("Norway", "France", "United Kingdom", "CA", "TX", "Japan")
DIRECT_WUE_REGION_MAP = {
    "Austria": "Frankfurt",
    "Belgium": "Brussel",
    "CA": "California",
    "France": "Paris",
    "Germany": "Frankfurt",
    "Italy": "Milan",
    "Japan": "Tokyo",
    "Korea": "Seoul",
    "Netherlands": "Amsterdam",
    "Spain": "Madrid",
    "Sweden": "Stockholm",
    "Switzerland": "Zurich",
    "Taiwan": "Taipei",
    "TX": "Texas",
    "VA": "Virginia",
}


def supported_regions() -> list[str]:
    """Use regions with explicit carbon, SOx/NOx, and water factors."""
    region_sets = [set(load_data.unified_emission_factors[factor]) for factor in CORE_GRID_FACTORS]
    return sorted(set.intersection(*region_sets))


def load_direct_wue_by_region() -> dict[str, tuple[float, str]]:
    with DIRECT_WUE_PATH.open(encoding="utf-8") as file:
        payload = json.load(file)
    documented_regions = payload["regions"]
    direct_wue = {}
    for region in supported_regions():
        documented_name = DIRECT_WUE_REGION_MAP.get(region)
        if documented_name is not None and documented_name in documented_regions:
            direct_wue[region] = (
                float(documented_regions[documented_name]["average_direct_wue_l_per_kwh"]),
                documented_name,
            )
        else:
            direct_wue[region] = (FALLBACK_DIRECT_WUE_L_PER_KWH, "microsoft_azure_americas_fy24_fy25_average")
    return direct_wue


def calculate_region_impacts(regions: list[str]) -> pd.DataFrame:
    records = []
    direct_wue_by_region = load_direct_wue_by_region()
    for region in regions:
        direct_wue, direct_wue_source = direct_wue_by_region[region]
        midpoint = modeling.calculate_operational_impacts_given_energy(
            ENERGY_KWH,
            YEAR,
            location=region,
            energy_unit="kWh",
            spatial_awareness=False,
            datacenter_wue=direct_wue,
            pue=DATACENTER_PUE,
        )
        record = {
            "region": region,
            "year": YEAR,
            "energy_kwh": ENERGY_KWH,
            "datacenter_pue": DATACENTER_PUE,
            "direct_wue_l_per_kwh": direct_wue,
            "direct_wue_source": direct_wue_source,
        }
        for midpoint_name in modeling.MIDPOINT_SCOPES:
            record[f"midpoint_{midpoint_name}"] = midpoint.get(midpoint_name, 0.0)
        for perspective in PERSPECTIVES:
            endpoint = modeling.midpoint_to_endpoint(midpoint, perspective=perspective)
            for endpoint_midpoint, value in endpoint.items():
                record[f"endpoint_{perspective}_{endpoint_midpoint}"] = value
            record[f"endpoint_{perspective}_total_species_yr"] = sum(endpoint.values())
        records.append(record)
    return pd.DataFrame.from_records(records)


def _best_region(rows: pd.DataFrame, metric: str) -> tuple[str, float]:
    idx = rows[metric].idxmin()
    return str(rows.loc[idx, "region"]), float(rows.loc[idx, metric])


def write_report(all_rows: pd.DataFrame, selected_rows: pd.DataFrame, output_path: Path) -> None:
    metrics = {
        "carbon": "midpoint_GWP",
        "water": "midpoint_WC",
        "biodiversity_hierarchist": "endpoint_hierarchist_total_species_yr",
    }
    all_best = {name: _best_region(all_rows, metric) for name, metric in metrics.items()}
    selected_best = {name: _best_region(selected_rows, metric) for name, metric in metrics.items()}

    selected_columns = [
        "region",
        "midpoint_GWP",
        "midpoint_WC",
        "endpoint_individualistic_total_species_yr",
        "endpoint_hierarchist_total_species_yr",
        "endpoint_egalitarian_total_species_yr",
    ]
    selected_table = selected_rows[selected_columns].copy()

    lines = [
        "# 1 kWh Operational Impact Region Sweep",
        "",
        f"- Year: {YEAR}",
        f"- Energy: {ENERGY_KWH} kWh",
        "- Spatial characterization: disabled",
        "- Supported region definition: explicit CO2e, SOx, NOx, and EWF factors",
        f"- Datacenter PUE: {DATACENTER_PUE}",
        f"- Direct WUE: documented regional values from `{DIRECT_WUE_PATH.name}`, falling back to {FALLBACK_DIRECT_WUE_L_PER_KWH} L/kWh",
        "- Endpoint total: sum of ecosystem endpoint midpoint contributions in species-yr",
        "",
        "## Global Optima Across Supported Regions",
        "",
    ]
    for name, (region, value) in all_best.items():
        lines.append(f"- {name}: {region} ({value:.6e})")

    lines.extend(
        [
            "",
        "## Selected Worldwide Region Set",
            "",
            ", ".join(REGION_SET),
            "",
            "Within this set, the carbon-, water-, and default biodiversity-optimal regions are distinct:",
            "",
        ]
    )
    for name, (region, value) in selected_best.items():
        lines.append(f"- {name}: {region} ({value:.6e})")

    lines.extend(["", "## Selected Region Values", "", selected_table.to_markdown(index=False), ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=REPO_ROOT / "bi_modeling" / "visualization" / "derived",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.derived_dir.mkdir(parents=True, exist_ok=True)

    all_rows = calculate_region_impacts(supported_regions())
    selected_rows = all_rows[all_rows["region"].isin(REGION_SET)].copy()
    selected_rows["region"] = pd.Categorical(selected_rows["region"], categories=REGION_SET, ordered=True)
    selected_rows = selected_rows.sort_values("region").reset_index(drop=True)

    all_path = args.derived_dir / "operational_1kwh_region_impacts_2024.csv"
    selected_path = args.derived_dir / "operational_1kwh_selected_region_impacts_2024.csv"
    report_path = args.derived_dir / "operational_1kwh_region_choice_report_2024.md"
    all_rows.to_csv(all_path, index=False)
    selected_rows.to_csv(selected_path, index=False)
    write_report(all_rows, selected_rows, report_path)
    print(f"Wrote {all_path}")
    print(f"Wrote {selected_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
