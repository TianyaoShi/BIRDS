import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from load_data import me_data, unified_emission_factors, impact_factors, cf_transportation, cf_eol
from sklearn.linear_model import LinearRegression

MIDPOINT_SCOPES = ['AP', 'FEP', 'MEP', 'POFP', 'TETP', 'FETP', 'METP', 'GWP', 'WC']   
global_distribution_ratio = {
    'US': 0.58,
    'CN': 0.26,
    'EU': 0.16
}
SHREDDING_ELECTRICITY_CONSUMPTION_KWH_PER_KG = 0.1/0.78
EOL_TRANSPORTATION_DISTANCE_KM = 100

transportation_average_distance = {
    "Ship": 12240.616001978953,
    "Air": 1337.4817996621982,
    "HeavyDutyTruck": 930.2868940029211,
    "Truck": 150.14649013500727,
}

# import Seagate HDD LCA results 

seagate_hdd_lca_results = me_data['seagate_hdd_lca_results']
for year in seagate_hdd_lca_results.keys():
    for key in seagate_hdd_lca_results[year].keys():
        if seagate_hdd_lca_results[year][key] is None:
            seagate_hdd_lca_results[year][key] = np.nan
    if 'photochemical oxidant formation (kg NMVOC)' in seagate_hdd_lca_results[year] and 'POFP' not in seagate_hdd_lca_results[year]:
            # ReCiPe 2008 Midpoint POFP assigned to photochemical oxidant formation (kg NMVOC) with a 1:1 ratio, as the POFP factor for NMVOC is 1.0 in ReCiPe 2008
            seagate_hdd_lca_results[year]['POFP'] = seagate_hdd_lca_results[year]['photochemical oxidant formation (kg NMVOC)'] * 1.0

hdd_carbon_footprint_records = me_data['hdd_carbon_footprint_records']

# Calculate the per-GB impacts for each year and each metric, after adjusting for the electricity-related emissions

results_per_gb = {}
hdd_embodied_impact_factors_by_year = {
    y: { stage:{midpoint:0.0 for midpoint in MIDPOINT_SCOPES} for stage in ['total', 'manufacturing', 'transportation',  'end-of-life'] } for y in ['2013','2016']
}

def _resolve_region_series(pollutant, region_code):
    aliases = {
        'US': ['US', 'United States'],
        'CN': ['CN', 'China'],
        'EU': ['EU', 'Europe'],
    }
    pollutant_data = unified_emission_factors[pollutant]
    for candidate in aliases.get(region_code, [region_code]):
        if candidate in pollutant_data:
            return pollutant_data[candidate]

    raise KeyError(
        f"No emission factor series found for pollutant '{pollutant}' and region '{region_code}'. "
        f"Available regions: {sorted(pollutant_data.keys())}"
    )


def get_mixed_grid_emission_factor(pollutant, year):
    idx = 0 if int(year) <= 2016 else int(year) - 2016
    us_series = _resolve_region_series(pollutant, 'US')
    cn_series = _resolve_region_series(pollutant, 'CN')
    eu_series = _resolve_region_series(pollutant, 'EU')

    return (
        us_series[idx] * global_distribution_ratio['US']
        + cn_series[idx] * global_distribution_ratio['CN']
        + eu_series[idx] * global_distribution_ratio['EU']
    )

def calculate_eol_impact_hdd(net_weight, total_weight, recycling_rate=0.25, PAPER_TO_PE_RATIO=2.1697866666666665):
    """
    Helper function, calculate eol impacts for HDD
    
    Parameters:
    mass: float - mass of the component
    recycling_rate: float - recycling rate (default 0.25 by most LCA reports)
    PAPER_TO_PE_RATIO: float - mass ratio of paper to PE in packaging inceneration (default 2.1697866666666665 based on Seagate LCA reports, average of 2017-2019)
    
    Returns:
    dict - impact values for each impact type
    """

    impacts = {k:0.0 for k in MIDPOINT_SCOPES}

    # EOL transportation
    for midpoint in MIDPOINT_SCOPES:
        if midpoint in cf_transportation['Truck'] and midpoint != 'GWP':
            impacts[midpoint] += cf_transportation['Truck'][midpoint] / cf_transportation['Truck']['reference_distance_km'] * EOL_TRANSPORTATION_DISTANCE_KM * total_weight

    # Shredding electricity consumption
    shredding_electricity_kwh = net_weight * SHREDDING_ELECTRICITY_CONSUMPTION_KWH_PER_KG
    for pollutant in ['SOx', 'NOx', 'NH3', 'airborne mercury', 'NMVOC']:
        for midpoint in MIDPOINT_SCOPES:
            if midpoint != 'GWP' and pollutant in impact_factors[midpoint]:
                impacts[midpoint] += shredding_electricity_kwh * get_mixed_grid_emission_factor(pollutant, '2016') / 1000 * impact_factors[midpoint][pollutant] # Convert to kg and then to impacts using the midpoint-specific impact factors for the pollutant

    # Landfill
    eol_landfill_factors = cf_eol['landfill']
    for midpoint in MIDPOINT_SCOPES:
        if midpoint in eol_landfill_factors and midpoint != 'GWP':
            impacts[midpoint] += net_weight * (1 - recycling_rate) * eol_landfill_factors[midpoint]

    # Energy credit from packaging incineration
    eol_packaging_factors_pe = cf_eol['energy_recycling_PE']
    eol_packaging_factors_paper = cf_eol['energy_recycling_paper']
    packaging_weight = total_weight - net_weight
    mass_paper = packaging_weight * PAPER_TO_PE_RATIO / (1 + PAPER_TO_PE_RATIO)
    mass_pe = packaging_weight / (1 + PAPER_TO_PE_RATIO)
    for midpoint in MIDPOINT_SCOPES:
        if midpoint in eol_packaging_factors_pe and midpoint in eol_packaging_factors_paper and midpoint != 'GWP':
            impacts[midpoint] += mass_pe * eol_packaging_factors_pe[midpoint] + mass_paper * eol_packaging_factors_paper[midpoint]
    
    return impacts

for year in seagate_hdd_lca_results.keys():
    if year == '2019' or year == '2014':
        # Skip 2019 as it only considers the MA part, not the entire HDD; 2014 already excluded operational electricity consumption, so the adjustment for electricity-related emissions is not applicable
        continue

    # Calculate mixed-grid pollutant emissions
    for pollutant in ['SOx', 'NOx', 'NH3', 'airborne mercury', 'NMVOC']:
        idx = 0 if int(year) <=2016 else int(year)-2016
        lifelong_emissions = seagate_hdd_lca_results[year]['lifecycle electricity consumption (kWh)'] * get_mixed_grid_emission_factor(pollutant, year) / 1000  # Convert to kg
        for midpoint in MIDPOINT_SCOPES:
            if midpoint not in ['GWP', 'WC', 'TETP'] and pollutant in impact_factors[midpoint].keys():
                # Subtract the electricity-related emissions from the original impact results for non-GWP and non-WC categories, as the GWP and WC impacts are calculated separately based on the electricity consumption and the corresponding emission factors
                # subtracting TETP leads to negative values, probably because the original LCA results did not consider the TETP impacts from electricity-related emissions, which can be significant; we thus also exclude TETP from the adjustment for electricity-related emissions
                if pollutant == 'NMVOC' and midpoint == 'POFP':
                    # ReCiPe 2008 Midpoint POFP assigned to photochemical oxidant formation (kg NMVOC) with a 1:1 ratio, as the POFP factor for NMVOC is 1.0 in ReCiPe 2008
                    seagate_hdd_lca_results[year][midpoint] -= lifelong_emissions * 1.0
                else:
                    seagate_hdd_lca_results[year][midpoint] -= lifelong_emissions * impact_factors[midpoint][pollutant]
                
                if seagate_hdd_lca_results[year][midpoint] < 0:
                    # If the adjusted impact value is negative, set it to zero, as the electricity-related emissions should not lead to negative impacts
                    seagate_hdd_lca_results[year][midpoint] = 0
                    print(f"Warning: Adjusted {midpoint} impact for year {year} is negative after subtracting electricity-related emissions for {pollutant}, set to zero.")

    hdd_embodied_impact_factors_by_year[year]['total'] = {midpoint: seagate_hdd_lca_results[year][midpoint] for midpoint in MIDPOINT_SCOPES}

    # Calculate the per-category distribution impacts based on the transportation mode ratio and average distance inferred from the LCA reports (analyze_hdd_carbon_trend.ipynb Cell 13 outputs)
    shipping_weight_kg = seagate_hdd_lca_results[year]['total weight'] 
    for midpoint in MIDPOINT_SCOPES:
        if midpoint != 'GWP':
            hdd_embodied_impact_factors_by_year[year]['transportation'][midpoint] = 0
            for mode in transportation_average_distance.keys():
                if mode in cf_transportation and midpoint in cf_transportation[mode]:
                    hdd_embodied_impact_factors_by_year[year]['transportation'][midpoint] += cf_transportation[mode][midpoint] / cf_transportation[mode]['reference_distance_km'] * transportation_average_distance[mode] * shipping_weight_kg
            if hdd_embodied_impact_factors_by_year[year]['transportation'][midpoint] > hdd_embodied_impact_factors_by_year[year]['total'][midpoint]:
                # If the calculated transportation impact is greater than the total impact, which is unlikely, set the transportation impact to be 0 and print a warning
                print(f"Warning: Calculated transportation {midpoint} impact for year {year} is greater than total {midpoint} impact: {hdd_embodied_impact_factors_by_year[year]['transportation'][midpoint]} > {hdd_embodied_impact_factors_by_year[year]['total'][midpoint]}, set transportation {midpoint} impact to be 0.")
                hdd_embodied_impact_factors_by_year[year]['transportation'][midpoint] = 0

    # Calculate EoL impacts based on weight and EoL factors from cf_eol: EOL_transportation (total weight) -> Shredding_Electricity_Consumption -> Landfill (net_weight * (1-recycling_rate)) + Energy_credit_from_packaging (total_weight - net_weight)
    hdd_embodied_impact_factors_by_year[year]['end-of-life'] = calculate_eol_impact_hdd(net_weight=seagate_hdd_lca_results[year]['net weight'], total_weight=seagate_hdd_lca_results[year]['total weight'],recycling_rate=0.25)
    for midpoint in MIDPOINT_SCOPES:
        if hdd_embodied_impact_factors_by_year[year]['end-of-life'][midpoint] > hdd_embodied_impact_factors_by_year[year]['total'][midpoint]:
            # If the calculated EoL impact is greater than the total impact, which is unlikely, set the EoL impact to be 0 and print a warning
            print(f"Warning: Calculated EoL {midpoint} impact for year {year} is greater than total {midpoint} impact: {hdd_embodied_impact_factors_by_year[year]['end-of-life'][midpoint]} > {hdd_embodied_impact_factors_by_year[year]['total'][midpoint]}, the corresponding transportation impact is {hdd_embodied_impact_factors_by_year[year]['transportation'][midpoint]}, set EoL {midpoint} impact to be 0.")
            hdd_embodied_impact_factors_by_year[year]['end-of-life'][midpoint] = 0

    hdd_embodied_impact_factors_by_year[year]['manufacturing'] = {midpoint: hdd_embodied_impact_factors_by_year[year]['total'][midpoint] - hdd_embodied_impact_factors_by_year[year]['transportation'][midpoint] - hdd_embodied_impact_factors_by_year[year]['end-of-life'][midpoint] for midpoint in MIDPOINT_SCOPES}

    # Check if manufacturing impact is negative after subtracting transportation and EoL impacts, if so, set manufacturing impact to zero and adjust transportation impact accordingly, as the transportation impact is more likely to be underestimated in LCA reports than the manufacturing impact to lead to negative manufacturing impact after subtraction
    for midpoint in MIDPOINT_SCOPES:
        if hdd_embodied_impact_factors_by_year[year]['manufacturing'][midpoint] < 0:
            print(f"Warning: Manufacturing {midpoint} impact for year {year} is negative after subtracting transportation and EoL impacts, set manufacturing impact to zero and adjust transportation impact accordingly.")
            hdd_embodied_impact_factors_by_year[year]['transportation'][midpoint] += hdd_embodied_impact_factors_by_year[year]['manufacturing'][midpoint]
            hdd_embodied_impact_factors_by_year[year]['manufacturing'][midpoint] = 0

    results_per_gb[int(year)] = {}
    capacity_gb = seagate_hdd_lca_results[year]['capacity (TB)'] * 1000  # Convert TB to GB
    for metric in MIDPOINT_SCOPES:
        if not np.isnan(hdd_embodied_impact_factors_by_year[year]['total'][metric]):
            results_per_gb[int(year)][metric] = { stage: hdd_embodied_impact_factors_by_year[year][stage][metric] / capacity_gb for stage in ['manufacturing', 'transportation', 'end-of-life','total'] }


STAGES = ['manufacturing', 'transportation', 'end-of-life']
TARGET_YEARS = range(2016, 2025)

Average_YoY_improvement_rates = {
    'manufacturing': 0.181928,
    'transportation': 0.102345,
    'end-of-life': 0.326759,
}  # analyze_hdd_carbon_trend.ipynb Cell 8 outputs for stage-wise per-GB carbon footprint (2016-2024)


def _resolve_project_root():
    project_home = os.environ.get('PROJECT_HOME')
    if project_home:
        return Path(project_home)
    return Path(__file__).resolve().parents[1]


def _fit_stage_yoy_rates_from_seagate_reports(csv_path):
    """Fit stage-level exponential decay rates for GWP/WC from raw product rows."""
    default_rates = dict(Average_YoY_improvement_rates)
    fallback = {'GWP': dict(default_rates), 'WC': dict(default_rates)}

    if not csv_path.exists():
        print(f'Warning: HDD LCA CSV not found at {csv_path}. Falling back to default stage YoY rates for GWP/WC.')
        return fallback

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f'Warning: Failed to read {csv_path}: {exc}. Falling back to default stage YoY rates for GWP/WC.')
        return fallback
    # filter out too old data points
    df = df[df['year'] >= 2016]

    df['capacity'] = pd.to_numeric(df.get('capacity'), errors='coerce')
    df['year'] = pd.to_numeric(df.get('year'), errors='coerce')
    df = df[(df['capacity'].notna()) & (df['capacity'] > 0) & (df['year'].notna())]

    metric_prefix = {
        'GWP': 'carbon_footprint',
        'WC': 'water_footprint',
    }
    stage_suffixes = {
        'manufacturing': ['bill_of_material', 'manufacturing', 'packaging'],
        'transportation': ['distribution'],
        'end-of-life': ['end_of_life'],
    }

    fitted_rates = {}
    for metric, prefix in metric_prefix.items():
        stage_rates = dict(default_rates)

        for stage, suffixes in stage_suffixes.items():
            stage_columns = [f'{prefix}_{suffix}' for suffix in suffixes]
            if any(column not in df.columns for column in stage_columns):
                continue

            stage_values = df[stage_columns].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=len(stage_columns))
            per_gb = stage_values / (df['capacity'] * 1000.0)

            fit_df = pd.DataFrame({'year': df['year'].astype(float), 'per_gb': per_gb})
            fit_df = fit_df[(fit_df['year'] >= 2016) & (fit_df['per_gb'].notna()) & np.isfinite(fit_df['per_gb']) & (fit_df['per_gb'] > 0)]
            if fit_df['year'].nunique() < 2:
                continue

            min_year = float(fit_df['year'].min())
            t = (fit_df['year'].to_numpy() - min_year).reshape(-1, 1)
            y_log = np.log(fit_df['per_gb'].to_numpy())

            model = LinearRegression().fit(t, y_log)
            b = float(model.coef_[0])
            stage_rates[stage] = 1 - float(np.exp(b))

        fitted_rates[metric] = stage_rates

    for metric in ('GWP', 'WC'):
        if metric not in fitted_rates:
            fitted_rates[metric] = dict(default_rates)

    return fitted_rates


def _build_hdd_impact_factors_by_stage_year(base_per_gb_2016, stage_rates_by_metric):
    nested = {}

    for year in TARGET_YEARS:
        year_data = {}
        year_delta = year - 2016

        for midpoint in MIDPOINT_SCOPES:
            base_midpoint = base_per_gb_2016.get(midpoint)
            stage_values = {}

            for stage in STAGES:
                base_value = 0.0
                if base_midpoint is not None and stage in base_midpoint:
                    try:
                        if not np.isnan(base_midpoint[stage]):
                            base_value = float(base_midpoint[stage])
                    except TypeError:
                        base_value = float(base_midpoint[stage])

                rate = Average_YoY_improvement_rates[stage]
                if midpoint in ('GWP', 'WC'):
                    rate = stage_rates_by_metric[midpoint][stage]

                stage_values[stage] = float(base_value * ((1 - rate) ** year_delta))

            stage_values['total'] = float(sum(stage_values[stage] for stage in STAGES))
            year_data[midpoint] = stage_values

        nested[year] = year_data

    return nested


def _serialize_stage_year_dict(nested):
    return {
        str(year): {
            midpoint: {
                stage: float(value)
                for stage, value in midpoint_data.items()
            }
            for midpoint, midpoint_data in year_data.items()
        }
        for year, year_data in nested.items()
    }


project_root = _resolve_project_root()
seagate_csv_path = project_root / 'data' / 'seagate_LCA_reports' / 'seagate_lca.csv'

if 2016 not in results_per_gb:
    raise ValueError('Base per-GB HDD results for year 2016 are required but were not found.')

base_values_2016 = results_per_gb[2016]
# Calibrate GWP for 2016, using the authentic by stage break down in LCA
base_values_2016['GWP'] = {
    'manufacturing': (22.017+31.146+0.537) / (seagate_hdd_lca_results['2016']['capacity (TB)'] * 1000),
    'transportation': 3.58 / (seagate_hdd_lca_results['2016']['capacity (TB)'] * 1000),
    'end-of-life': 7.16 / (seagate_hdd_lca_results['2016']['capacity (TB)'] * 1000),
}
GWP_WC_stage_yoy_rates = _fit_stage_yoy_rates_from_seagate_reports(seagate_csv_path)

hdd_impact_factors_by_stage_year = _build_hdd_impact_factors_by_stage_year(base_values_2016, GWP_WC_stage_yoy_rates)

# Keep backward-compatible output for existing modeling code paths.
hdd_impact_factors_by_year = {
    year: {
        midpoint: hdd_impact_factors_by_stage_year[year][midpoint]['total']
        for midpoint in MIDPOINT_SCOPES
    }
    for year in TARGET_YEARS
}

json_output_path = project_root / 'data' / 'hdd_impact_factors_by_stage_year.json'
json_output_path.parent.mkdir(parents=True, exist_ok=True)
try:
    with json_output_path.open('w', encoding='utf-8') as f:
        json.dump(_serialize_stage_year_dict(hdd_impact_factors_by_stage_year), f, indent=2, sort_keys=True)
except OSError as exc:  # pragma: no cover - defensive fallback
    print(f'Warning: Failed to write HDD impact JSON to {json_output_path}: {exc}')
