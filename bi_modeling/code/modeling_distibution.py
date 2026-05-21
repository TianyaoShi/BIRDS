from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import load_data as data
import modeling as legacy_model


RECEIVING_BUCKETS = (
    "manufacturing_site",
    "use_site",
    "ocean",
    "upstream_supply_chain_sites",
    "rest_of_world",
)
ENDPOINT_SCOPES = tuple(data.ECOSYSTEM_ENDPOINT_SCOPES)
MIDPOINT_SCOPES = tuple(legacy_model.MIDPOINT_SCOPES)
INTERCONTINENTAL_SPLIT_MODES = {"Air", "Ship", "Ship-Feeder", "HeavyDutyTruck"}
US_REGION_EMISSION_LOCATION = {
    "California": "CA",
    "Texas": "TX",
    "Virginia": "VA",
    "Iowa (Central US)": "MidWest",
}
WATER_LOCATION_NORMALIZATION = {
    "United States": "US",
    "USA": "US",
    "US": "US",
    "CA": "US",
    "TX": "US",
    "VA": "US",
    "MidWest": "US",
    "South Korea": "Korea",
    "Korea, Republic of": "Korea",
}
EMISSION_LOCATION_NORMALIZATION = {
    "United States": "US",
    "USA": "US",
    "US": "US",
    "South Korea": "Korea",
    "Korea, Republic of": "Korea",
}
EPSILON = 1e-20


def _append_unique_fallback(fallbacks: List[str], message: str) -> None:
    if message not in fallbacks:
        fallbacks.append(message)


def _zero_midpoints() -> Dict[str, float]:
    return {midpoint: 0.0 for midpoint in MIDPOINT_SCOPES}


def _zero_endpoint_summary() -> Dict[str, Any]:
    return {
        "total": 0.0,
        "by_scope": {scope: 0.0 for scope in ENDPOINT_SCOPES},
        "by_midpoint": {midpoint: 0.0 for midpoint in MIDPOINT_SCOPES},
    }


def _distribution_area_dataset() -> Dict[str, Any]:
    dataset = getattr(data, "receiving_area_distribution_data", None) or {}
    if not dataset.get("areas"):
        raise ValueError(
            "The receiving-area dataset is unavailable. Expected "
            f"{data.RECEIVING_AREA_SHARE_DATA_PATH}."
        )
    return dataset


def _bii_distribution_dataset() -> Dict[str, Any]:
    dataset = getattr(data, "bii_distribution_data", None) or {}
    if not dataset.get("area_bii_2023"):
        raise ValueError(
            "The BII distribution dataset is unavailable. Expected "
            f"{data.BII_DISTRIBUTION_LOOKUP_PATH}."
        )
    return dataset


def _normalize_perspective(perspective: str) -> str:
    perspective_key = data.RECIPE_PERSPECTIVE_ALIASES.get(str(perspective).strip().lower())
    if perspective_key is None:
        raise ValueError(
            f"Unsupported ReCiPe perspective {perspective!r}. "
            f"Available aliases: {sorted(data.RECIPE_PERSPECTIVE_ALIASES)}"
        )
    return perspective_key


def _resolve_supported_emission_location(location: Optional[str]) -> Optional[str]:
    if location is None:
        return None

    base_location = str(location).strip()
    if not base_location:
        return None

    candidates: List[str] = []
    for candidate in (
        base_location,
        EMISSION_LOCATION_NORMALIZATION.get(base_location, base_location),
        legacy_model._normalize_emission_factor_location(base_location),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if candidate in data.unified_emission_factors["CO2e"]:
            return candidate

    return None


def _normalize_water_location(location: Optional[str]) -> Optional[str]:
    if location is None:
        return None
    normalized = WATER_LOCATION_NORMALIZATION.get(str(location).strip(), str(location).strip())
    return normalized or None


def _distribution_location_context(location: Optional[str], fallbacks: List[str]) -> Dict[str, Any]:
    dataset = _distribution_area_dataset()
    actual_location = None if location is None else str(location).strip()
    if not actual_location:
        return {
            "actual_location": None,
            "emission_location": None,
            "resolved_location": None,
            "area_id": None,
            "direct_wue": None,
        }

    area_lookup = dataset.get("region_area_lookup", {})
    region_data = data.regional_direct_wue_data.get("regions", {}).get(actual_location)
    direct_wue = None

    if region_data is not None:
        direct_wue = region_data.get("average_direct_wue_l_per_kwh")
        if actual_location in US_REGION_EMISSION_LOCATION:
            emission_location = US_REGION_EMISSION_LOCATION[actual_location]
        else:
            emission_location = _resolve_supported_emission_location(region_data.get("country"))
        resolved_location = _normalize_water_location(region_data.get("country"))
        area_id = area_lookup.get(actual_location) or area_lookup.get(region_data.get("country", ""))
    else:
        emission_location = _resolve_supported_emission_location(actual_location)
        resolved_location = _normalize_water_location(actual_location)
        area_id = area_lookup.get(actual_location) or area_lookup.get(resolved_location or "")

    if emission_location is None:
        emission_location = "US"
        fallbacks.append(
            f"Fell back to US grid emission factors for unsupported location {actual_location!r}."
        )

    return {
        "actual_location": actual_location,
        "emission_location": emission_location,
        "resolved_location": resolved_location,
        "area_id": area_id if area_id in dataset.get("areas", {}) else None,
        "direct_wue": direct_wue,
    }


def _append_midpoint_record(
    records: List[Dict[str, Any]],
    *,
    stage: str,
    substage: str,
    component_type: str,
    process_name: str,
    midpoint: str,
    midpoint_value: float,
    actual_location: Optional[str] = None,
    emission_location: Optional[str] = None,
    resolved_location: Optional[str] = None,
    area_id: Optional[str] = None,
    local_receiving_bucket: Optional[str] = None,
    allocation_context: str = "local",
    pollutant: Optional[str] = None,
    pollutant_amount: Optional[float] = None,
    transport_mode: Optional[str] = None,
    transport_piece: Optional[str] = None,
    transport_distance_km: Optional[float] = None,
    intercontinental: bool = False,
    manufacturing_scope: Optional[str] = None,
    spatial_awareness: bool = False,
    notes: Optional[str] = None,
) -> None:
    if midpoint not in MIDPOINT_SCOPES or abs(midpoint_value) <= EPSILON:
        return

    records.append(
        {
            "stage": stage,
            "substage": substage,
            "component_type": component_type,
            "process_name": process_name,
            "midpoint": midpoint,
            "midpoint_value": float(midpoint_value),
            "actual_location": actual_location,
            "emission_location": emission_location,
            "resolved_location": resolved_location,
            "area_id": area_id,
            "local_receiving_bucket": local_receiving_bucket,
            "allocation_context": allocation_context,
            "pollutant": pollutant,
            "pollutant_amount": None if pollutant_amount is None else float(pollutant_amount),
            "transport_mode": transport_mode,
            "transport_piece": transport_piece,
            "transport_distance_km": None if transport_distance_km is None else float(transport_distance_km),
            "intercontinental": bool(intercontinental),
            "manufacturing_scope": manufacturing_scope,
            "spatial_awareness": bool(spatial_awareness),
            "notes": notes,
        }
    )


def _add_group_midpoint_records(
    records: List[Dict[str, Any]],
    impacts: Dict[str, float],
    *,
    stage: str,
    substage: str,
    component_type: str,
    process_name: str,
    actual_location: Optional[str],
    emission_location: Optional[str],
    resolved_location: Optional[str],
    area_id: Optional[str],
    local_receiving_bucket: Optional[str],
    manufacturing_scope: Optional[str],
    spatial_awareness: bool,
    notes: Optional[str] = None,
) -> None:
    for midpoint in MIDPOINT_SCOPES:
        value = impacts.get(midpoint, 0.0)
        _append_midpoint_record(
            records,
            stage=stage,
            substage=substage,
            component_type=component_type,
            process_name=process_name,
            midpoint=midpoint,
            midpoint_value=value,
            actual_location=actual_location,
            emission_location=emission_location,
            resolved_location=resolved_location,
            area_id=area_id,
            local_receiving_bucket=local_receiving_bucket,
            allocation_context="local",
            manufacturing_scope=manufacturing_scope,
            spatial_awareness=spatial_awareness,
            notes=notes,
        )


def _get_energy_emission_factor(pollutant: str, location: str, year_idx: int) -> float:
    normalized_location = legacy_model._normalize_emission_factor_location(location)
    if pollutant == "EWF" and normalized_location == "IA" and "MidWest" in data.unified_emission_factors[pollutant]:
        return data.unified_emission_factors[pollutant]["MidWest"][year_idx]
    if pollutant in ["NH3", "airborne mercury", "NMVOC"] and normalized_location in ["IN", "TX", "VA", "NE", "MO", "OH", "WY"]:
        return data.unified_emission_factors[pollutant]["US"][year_idx]
    if normalized_location in data.unified_emission_factors[pollutant]:
        return data.unified_emission_factors[pollutant][normalized_location][year_idx]
    return data.unified_emission_factors[pollutant]["US"][year_idx]


def _build_energy_midpoint_records(
    records: List[Dict[str, Any]],
    *,
    energy_kwh: float,
    year: int,
    location_context: Dict[str, Any],
    stage: str,
    substage: str,
    component_type: str,
    process_name: str,
    local_receiving_bucket: str,
    spatial_awareness: bool,
    excluded_pollutants: Optional[Iterable[str]] = None,
    datacenter_wue: Optional[float] = None,
    pue: float = 1.2,
    manufacturing_scope: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    if energy_kwh is None or energy_kwh <= 0:
        return
    if pue <= 0:
        raise ValueError("pue must be positive.")

    emission_location = location_context.get("emission_location") or "US"
    location_specific_cfs = legacy_model._get_location_specific_midpoint_cfs(
        emission_location,
        spatial_awareness=spatial_awareness,
    )
    air_pollutants = ["CO2e", "SOx", "NOx", "NMVOC", "NH3", "airborne mercury", "EWF"]
    for pollutant in excluded_pollutants or []:
        if pollutant in air_pollutants:
            air_pollutants.remove(pollutant)

    year_idx = min(year, 2024) - 2016
    for pollutant in air_pollutants:
        emission_factor = _get_energy_emission_factor(pollutant, emission_location, year_idx)
        if pollutant == "EWF" and datacenter_wue is not None:
            pollutant_amount = energy_kwh * (datacenter_wue + pue * emission_factor) / 1000
        else:
            pollutant_amount = energy_kwh * emission_factor / 1000

        for midpoint in MIDPOINT_SCOPES:
            midpoint_cf = legacy_model._get_midpoint_impact_factor(midpoint, pollutant, location_specific_cfs)
            if midpoint_cf is None:
                continue
            _append_midpoint_record(
                records,
                stage=stage,
                substage=substage,
                component_type=component_type,
                process_name=process_name,
                midpoint=midpoint,
                midpoint_value=midpoint_cf * pollutant_amount,
                actual_location=location_context.get("actual_location"),
                emission_location=emission_location,
                resolved_location=location_context.get("resolved_location") or emission_location,
                area_id=location_context.get("area_id"),
                local_receiving_bucket=local_receiving_bucket,
                allocation_context="local",
                pollutant=pollutant,
                pollutant_amount=pollutant_amount,
                manufacturing_scope=manufacturing_scope,
                spatial_awareness=spatial_awareness,
                notes=notes,
            )


def _transport_impacts_for_mode(
    mode: str,
    mass_kg: float,
    distance_km: float,
    fallbacks: List[str],
) -> Dict[str, float]:
    if mode not in data.cf_transportation:
        fallbacks.append(f"Missing transportation CF for mode {mode!r}; skipped transport calculation.")
        return _zero_midpoints()

    impacts = _zero_midpoints()
    cf_payload = data.cf_transportation[mode]
    reference_weight = cf_payload["reference_weight_kg"]
    reference_distance = cf_payload["reference_distance_km"]
    for midpoint in MIDPOINT_SCOPES:
        if midpoint in cf_payload:
            impacts[midpoint] = (
                cf_payload[midpoint]
                * mass_kg
                / reference_weight
                * distance_km
                / reference_distance
            )
    return impacts


def _build_transport_leg_midpoint_records(
    records: List[Dict[str, Any]],
    *,
    mass_kg: float,
    leg: Dict[str, Any],
    stage: str,
    component_type: str,
    spatial_awareness: bool,
    fallbacks: List[str],
) -> None:
    mode = leg["mode"]
    distance_km = leg["distance_km"]
    location_context = leg.get("location_context") or {}
    base_notes = leg.get("notes")

    if distance_km is None or distance_km <= 0:
        return

    if leg.get("intercontinental") and f"{mode}-NFP" in data.cf_transportation:
        full_impacts = _transport_impacts_for_mode(mode, mass_kg, distance_km, fallbacks)
        direct_impacts = _transport_impacts_for_mode(f"{mode}-NFP", mass_kg, distance_km, fallbacks)
        upstream_impacts = {
            midpoint: full_impacts.get(midpoint, 0.0) - direct_impacts.get(midpoint, 0.0)
            for midpoint in MIDPOINT_SCOPES
        }

        for midpoint in MIDPOINT_SCOPES:
            _append_midpoint_record(
                records,
                stage=stage,
                substage=leg["substage"],
                component_type=component_type,
                process_name=f"{leg['process_name']}:direct",
                midpoint=midpoint,
                midpoint_value=direct_impacts.get(midpoint, 0.0),
                actual_location=location_context.get("actual_location"),
                emission_location=location_context.get("emission_location"),
                resolved_location=location_context.get("resolved_location"),
                area_id=location_context.get("area_id"),
                local_receiving_bucket=None,
                allocation_context="intercontinental_transport_direct",
                transport_mode=mode,
                transport_piece="direct",
                transport_distance_km=distance_km,
                intercontinental=True,
                manufacturing_scope=leg.get("manufacturing_scope"),
                spatial_awareness=spatial_awareness,
                notes=base_notes,
            )
            _append_midpoint_record(
                records,
                stage=stage,
                substage=leg["substage"],
                component_type=component_type,
                process_name=f"{leg['process_name']}:upstream",
                midpoint=midpoint,
                midpoint_value=upstream_impacts.get(midpoint, 0.0),
                actual_location=None,
                emission_location=None,
                resolved_location=None,
                area_id=None,
                local_receiving_bucket="upstream_supply_chain_sites",
                allocation_context="intercontinental_transport_upstream",
                transport_mode=mode,
                transport_piece="upstream",
                transport_distance_km=distance_km,
                intercontinental=True,
                manufacturing_scope=leg.get("manufacturing_scope"),
                spatial_awareness=spatial_awareness,
                notes=base_notes,
            )
        return

    if leg.get("intercontinental") and f"{mode}-NFP" not in data.cf_transportation:
        fallbacks.append(
            f"Missing paired {mode}-NFP transport CF; treated the full {mode} burden as a local transport fallback."
        )

    full_impacts = _transport_impacts_for_mode(mode, mass_kg, distance_km, fallbacks)
    for midpoint in MIDPOINT_SCOPES:
        _append_midpoint_record(
            records,
            stage=stage,
            substage=leg["substage"],
            component_type=component_type,
            process_name=leg["process_name"],
            midpoint=midpoint,
            midpoint_value=full_impacts.get(midpoint, 0.0),
            actual_location=location_context.get("actual_location"),
            emission_location=location_context.get("emission_location"),
            resolved_location=location_context.get("resolved_location"),
            area_id=location_context.get("area_id"),
            local_receiving_bucket=leg.get("local_receiving_bucket"),
            allocation_context="local",
            transport_mode=mode,
            transport_piece="full",
            transport_distance_km=distance_km,
            intercontinental=bool(leg.get("intercontinental")),
            manufacturing_scope=leg.get("manufacturing_scope"),
            spatial_awareness=spatial_awareness,
            notes=base_notes,
        )


def _logic_manufacturing_context(fallbacks: List[str]) -> Dict[str, Any]:
    return _distribution_location_context("Taiwan", fallbacks)


def _memory_manufacturing_context(fallbacks: List[str]) -> Dict[str, Any]:
    return _distribution_location_context("Korea", fallbacks)


def _build_manufacture_to_use_transport_midpoint_records(
    records: List[Dict[str, Any]],
    working_specs: Dict[str, Any],
    *,
    use_region: Optional[str],
    transport_region: Optional[str],
    spatial_awareness: bool,
    fallbacks: List[str],
) -> None:
    use_region, _ = legacy_model._normalize_transport_regions(
        use_region=use_region,
        eol_region=None,
        transport_region=transport_region,
    )
    component_type = working_specs["component_type"]
    transport_family = legacy_model.get_device_transport_family(working_specs)
    origin_context = _logic_manufacturing_context(fallbacks) if transport_family == "logic" else _memory_manufacturing_context(fallbacks)
    mass_kg = working_specs["gross_weight"] / 1000

    if use_region is not None:
        route = legacy_model._get_transport_region_route(use_region)
        use_context = _distribution_location_context(use_region, fallbacks)
        fixed_origin_leg = data.regional_device_transport_routes["metadata"]["fixed_origin_legs"][transport_family]
        air_distance_key = "air_distance_from_taipei_km" if transport_family == "logic" else "air_distance_from_seoul_km"
        legs = [
            {
                "mode": "Truck",
                "distance_km": fixed_origin_leg["truck_km"],
                "substage": "manufacture_to_use_origin_truck",
                "process_name": "manufacture_to_use_origin_truck",
                "location_context": origin_context,
                "local_receiving_bucket": "manufacturing_site",
                "intercontinental": False,
            },
            {
                "mode": "Air",
                "distance_km": route.get(air_distance_key, 0),
                "substage": "manufacture_to_use_intercontinental_air",
                "process_name": "manufacture_to_use_intercontinental_air",
                "location_context": {
                    "actual_location": f"{origin_context['actual_location']} -> {use_context['actual_location']}",
                    "emission_location": None,
                    "resolved_location": None,
                    "area_id": None,
                },
                "local_receiving_bucket": None,
                "intercontinental": True,
            },
            {
                "mode": "Truck",
                "distance_km": route["last_mile_truck_km"],
                "substage": "manufacture_to_use_last_mile_truck",
                "process_name": "manufacture_to_use_last_mile_truck",
                "location_context": use_context,
                "local_receiving_bucket": "use_site",
                "intercontinental": False,
            },
        ]
    else:
        fallbacks.append(
            "Manufacture-to-use transport fell back to the legacy device distance map. "
            "Aggregated truck legs without regional routing are assigned to rest_of_world."
        )
        legs = []
        for mode, distance_km in legacy_model._copy_distance_map(working_specs["to_use_distance"]).items():
            if distance_km <= 0:
                continue
            legs.append(
                {
                    "mode": mode,
                    "distance_km": distance_km,
                    "substage": f"manufacture_to_use_{mode.lower()}",
                    "process_name": f"manufacture_to_use_{mode.lower()}",
                    "location_context": {} if mode in INTERCONTINENTAL_SPLIT_MODES else None,
                    "local_receiving_bucket": "rest_of_world" if mode == "Truck" else None,
                    "intercontinental": mode in INTERCONTINENTAL_SPLIT_MODES,
                    "notes": "Legacy manufacture-to-use routing lacks location-specific truck-leg separation.",
                }
            )

    for leg in legs:
        _build_transport_leg_midpoint_records(
            records,
            mass_kg=mass_kg,
            leg=leg,
            stage="transportation",
            component_type=component_type,
            spatial_awareness=spatial_awareness,
            fallbacks=fallbacks,
        )


def _build_use_to_eol_transport_midpoint_records(
    records: List[Dict[str, Any]],
    working_specs: Dict[str, Any],
    *,
    use_region: Optional[str],
    eol_region: Optional[str],
    transport_region: Optional[str],
    spatial_awareness: bool,
    fallbacks: List[str],
) -> Dict[str, Any]:
    use_region, eol_region = legacy_model._normalize_transport_regions(
        use_region=use_region,
        eol_region=eol_region,
        transport_region=transport_region,
    )

    if eol_region is not None:
        context = _distribution_location_context(eol_region, fallbacks)
        distance_map = {"Truck": legacy_model._get_transport_region_route(eol_region)["eol_truck_km"]}
    elif use_region is not None:
        context = _distribution_location_context(use_region, fallbacks)
        distance_map = legacy_model._copy_distance_map(
            legacy_model.resolve_transport_distances(
                working_specs,
                use_region=use_region,
                eol_region=None,
                transport_region=transport_region,
            )["use_to_eol"]
        )
    else:
        fallbacks.append(
            "Use-to-EoL transport fell back to the legacy device distance map without an explicit receiving-side region."
        )
        context = {
            "actual_location": None,
            "emission_location": None,
            "resolved_location": None,
            "area_id": None,
        }
        distance_map = legacy_model._copy_distance_map(
            working_specs.get("to_recycle_distance", legacy_model.distance["default"])
        )

    mass_kg = working_specs["net_weight"] / 1000
    for mode, distance_km in distance_map.items():
        if distance_km <= 0:
            continue
        _build_transport_leg_midpoint_records(
            records,
            mass_kg=mass_kg,
            leg={
                "mode": mode,
                "distance_km": distance_km,
                "substage": f"use_to_eol_{mode.lower()}",
                "process_name": f"use_to_eol_{mode.lower()}",
                "location_context": context,
                "local_receiving_bucket": "use_site" if context.get("actual_location") else "rest_of_world",
                "intercontinental": False,
            },
            stage="recycling",
            component_type=working_specs["component_type"],
            spatial_awareness=spatial_awareness,
            fallbacks=fallbacks,
        )
    return context


def _build_custom_transport_midpoint_records(
    records: List[Dict[str, Any]],
    *,
    distance_map: Dict[str, float],
    mass_kg: float,
    stage: str,
    substage_prefix: str,
    component_type: str,
    location_context: Optional[Dict[str, Any]],
    local_receiving_bucket: Optional[str],
    spatial_awareness: bool,
    fallbacks: List[str],
    manufacturing_scope: Optional[str] = None,
    notes: Optional[str] = None,
    intercontinental_modes: Optional[Sequence[str]] = None,
) -> None:
    if not distance_map:
        return
    modes_for_split = set(intercontinental_modes or INTERCONTINENTAL_SPLIT_MODES)
    for mode, distance_km in distance_map.items():
        if distance_km is None or distance_km <= 0:
            continue
        _build_transport_leg_midpoint_records(
            records,
            mass_kg=mass_kg,
            leg={
                "mode": mode,
                "distance_km": distance_km,
                "substage": f"{substage_prefix}_{mode.lower()}",
                "process_name": f"{substage_prefix}_{mode.lower()}",
                "location_context": location_context or {},
                "local_receiving_bucket": local_receiving_bucket,
                "intercontinental": mode in modes_for_split,
                "manufacturing_scope": manufacturing_scope,
                "notes": notes,
            },
            stage=stage,
            component_type=component_type,
            spatial_awareness=spatial_awareness,
            fallbacks=fallbacks,
        )


def _build_backend_midpoint_records(
    records: List[Dict[str, Any]],
    *,
    production_year: int,
    n_ic: int,
    component_type: str,
    spatial_awareness: bool,
    fallbacks: List[str],
) -> None:
    if n_ic <= 0:
        return
    backend_context = _logic_manufacturing_context(fallbacks)
    year_idx = legacy_model._get_year_index_from_series(data.spil_packaging_emissions, production_year)

    electricity_consumption = (
        data.spil_packaging_emissions["total_electricity_consumption_kWh_per_packaged_ic"][year_idx] * n_ic
    )
    _build_energy_midpoint_records(
        records,
        energy_kwh=electricity_consumption,
        year=production_year,
        location_context=backend_context,
        stage="manufacturing",
        substage="backend_packaging_testing_electricity",
        component_type=component_type,
        process_name="backend_packaging_testing_electricity",
        local_receiving_bucket="manufacturing_site",
        spatial_awareness=spatial_awareness,
        manufacturing_scope="scope2",
    )

    _append_midpoint_record(
        records,
        stage="manufacturing",
        substage="backend_packaging_testing_water",
        component_type=component_type,
        process_name="backend_packaging_testing_water",
        midpoint="WC",
        midpoint_value=data.spil_packaging_emissions["total_water_consumption_ton_per_packaged_ic"][year_idx] * n_ic,
        actual_location=backend_context["actual_location"],
        emission_location=backend_context["emission_location"],
        resolved_location=backend_context["resolved_location"],
        area_id=backend_context["area_id"],
        local_receiving_bucket="manufacturing_site",
        allocation_context="local",
        manufacturing_scope="scope1",
        spatial_awareness=spatial_awareness,
        notes="Backend water tons are treated as m^3, matching the legacy packaging/testing path.",
    )

    location_specific_cfs = legacy_model._get_location_specific_midpoint_cfs(
        backend_context["emission_location"],
        spatial_awareness=spatial_awareness,
    )
    for metric_key, pollutant in legacy_model.SPIL_BACKEND_DIRECT_POLLUTANT_MAP.items():
        pollutant_mass = data.spil_packaging_emissions[metric_key][year_idx] * n_ic
        if pollutant_mass == 0:
            continue
        for midpoint in MIDPOINT_SCOPES:
            midpoint_cf = legacy_model._get_midpoint_impact_factor(midpoint, pollutant, location_specific_cfs)
            if midpoint_cf is None:
                continue
            _append_midpoint_record(
                records,
                stage="manufacturing",
                substage="backend_packaging_testing_direct",
                component_type=component_type,
                process_name=f"backend_direct_{pollutant}",
                midpoint=midpoint,
                midpoint_value=midpoint_cf * pollutant_mass,
                actual_location=backend_context["actual_location"],
                emission_location=backend_context["emission_location"],
                resolved_location=backend_context["resolved_location"],
                area_id=backend_context["area_id"],
                local_receiving_bucket="manufacturing_site",
                allocation_context="local",
                pollutant=pollutant,
                pollutant_amount=pollutant_mass,
                manufacturing_scope="scope1",
                spatial_awareness=spatial_awareness,
            )


def _build_material_midpoint_records(
    records: List[Dict[str, Any]],
    device_specs: Dict[str, Any],
    *,
    bit_density: Optional[float] = None,
    silicon_wafer_region: str = "Japan",
    chemical_region: str = "Japan",
    silicon_wafer_transportation: Optional[Dict[str, float]] = None,
    chemical_transportation: Optional[Dict[str, float]] = None,
    spatial_awareness: bool,
    fallbacks: List[str],
) -> None:
    component_type = device_specs.get("component_type", device_specs.get("device_type"))
    base_impacts = legacy_model.calculate_manufacturing_material_impacts(
        device_specs,
        bit_density=bit_density,
        silicon_wafer_region=silicon_wafer_region,
        chemical_region=chemical_region,
        silicon_wafer_transportation=None,
        chemical_transportation=None,
        spatial_awareness=spatial_awareness,
    )

    silicon_context = _distribution_location_context(silicon_wafer_region, fallbacks)
    _add_group_midpoint_records(
        records,
        base_impacts["silicon"],
        stage="manufacturing",
        substage="scope3_silicon_supply_chain",
        component_type=component_type,
        process_name="silicon_supply_chain_energy",
        actual_location=silicon_context["actual_location"],
        emission_location=silicon_context["emission_location"],
        resolved_location=silicon_context["resolved_location"],
        area_id=silicon_context["area_id"],
        local_receiving_bucket="upstream_supply_chain_sites",
        manufacturing_scope="scope3",
        spatial_awareness=spatial_awareness,
    )

    chemical_context = _distribution_location_context(chemical_region, fallbacks)
    _add_group_midpoint_records(
        records,
        base_impacts["chemicals"],
        stage="manufacturing",
        substage="scope3_chemical_supply_chain",
        component_type=component_type,
        process_name="chemical_supply_chain_energy",
        actual_location=chemical_context["actual_location"],
        emission_location=chemical_context["emission_location"],
        resolved_location=chemical_context["resolved_location"],
        area_id=chemical_context["area_id"],
        local_receiving_bucket="upstream_supply_chain_sites",
        manufacturing_scope="scope3",
        spatial_awareness=spatial_awareness,
    )

    required_wafer_area_cm2 = (
        legacy_model.get_equivalent_wafers(device_specs, bit_density=bit_density)
        * legacy_model.EFFECTIVE_WAFER_AREA_MM2
        / 100
    )
    kg_silicon_per_cm2 = 0.34 / 2130

    if silicon_wafer_transportation is not None:
        _build_custom_transport_midpoint_records(
            records,
            distance_map=silicon_wafer_transportation,
            mass_kg=required_wafer_area_cm2 * kg_silicon_per_cm2,
            stage="manufacturing",
            substage_prefix="scope3_silicon_transport",
            component_type=component_type,
            location_context=silicon_context,
            local_receiving_bucket="upstream_supply_chain_sites",
            spatial_awareness=spatial_awareness,
            fallbacks=fallbacks,
            manufacturing_scope="scope3",
        )

    kg_chemicals_per_wafer_logic_45nm = (
        3.46e-2 + 3.3e-3 + 1.01 + 1.76e-1 + 2.05e-4 + 4.09e-4 + 7.5e-4 + 1.59e-1
        + 2.35e-3 + 2.84e-4 + 6.07e-4 + 7.1e-5 + 3.09e-2 + 1e-2 + 2.88e-1 + 5.9e-3
        + 1.68e-2 + 1.02e-1 + 3.27e-3 + 7.4e-3 + 4.31 + 1.92 + 1.66 + 23.6 + 23.3
        + 7.46e-1 + 7.46e-1 + 3.07e-2 + 5.45e-1 + 3.38e-2 + 2.62 + 1.53e-3 + 1.07e-7
        + 3.21e-1 + 1.22e-2 + 7.58e-5 + 1.31e-2 + 4.85e-2 + 1.15e-3 + 3e-3 + 2.62e-2
        + 3.35e-2 + 3.72e-2 + 2.02e-2 + 2.82e-1 + 3.16e-2 + 3.44e-1 + 3.27e-1
        + 3.52e-2 + 5.35e-5 + 1.2e-2 + 2.36e-2 + 1.15e-1 + 253 + 5.35 + 6.97 + 1.11
        + 5.03 + 2.26e-4 + 4.38e-3 + 1.2e-4 + 1.01e-2 + 2.35e-3 + 9.96e-4 + 4.81e-7
        + 7.16e-3 + 3.41e-3 + 2.43e-1
    ) / 1000
    kg_chemicals_per_wafer_dram_57nm = (
        2.19e-2 + 3.41e-3 + 6.48e-1 + 2.05e-1 + 2.56e-4 + 1.31e-3 + 3.77e-3 + 2.49e-3
        + 6.21e-4 + 7.71e-2 + 1.75e-2 + 2.88e-1 + 1.85e-2 + 2.13e-1 + 7.04e-1 + 1.46
        + 5.37e-1 + 13.1 + 2.18e-1 + 1.21e-1 + 10.5 + 1e-3 + 1.07e-7 + 8.48e-4
        + 9.23e-4 + 1.31e-2 + 1.15e-3 + 2.08e-3 + 1.32e-3 + 3.35e-2 + 3.72e-2
        + 2.02e-2 + 2.82e-1 + 3.16e-2 + 3.44e-1 + 3.27e-1 + 3.52e-2 + 1.07e-4
        + 1.3e-1 + 99.2 + 4.64 + 6.99 + 15.1 + 2.89e-4 + 5.6e-3 + 1.54e-4 + 1.29e-2
        + 2.35e-3 + 9.83e-4 + 4.81e-7 + 2.83e-4 + 2.43e-1
    ) / 1000

    if chemical_transportation is None:
        return

    def calc_dram_process_steps(local_bit_density: float) -> float:
        return 118 * legacy_model.np.log10(local_bit_density) + 356

    if component_type in ["CPU", "GPU"]:
        shipping_mass = (
            legacy_model.get_equivalent_wafers(device_specs, bit_density=bit_density)
            * kg_chemicals_per_wafer_logic_45nm
            * (
                data.node_to_layer_masks_map[device_specs["technology_node_nm"]]
                / data.node_to_layer_masks_map["45"]
            )
        )
    elif component_type in ["DRAM", "HBM"]:
        if bit_density is None:
            if component_type == "DRAM":
                bit_density = data.dram_bit_density_by_year[device_specs["production_year"] - 2016]
            else:
                bit_density = data.vram_bit_density.get(
                    device_specs.get("hbm_type"),
                    data.dram_bit_density_by_year[device_specs["production_year"] - 2016],
                )
        scaling_factor = calc_dram_process_steps(bit_density) / calc_dram_process_steps(0.022)
        shipping_mass = (
            scaling_factor
            * legacy_model.get_equivalent_wafers(device_specs, bit_density=bit_density)
            * kg_chemicals_per_wafer_dram_57nm
        )
    else:
        year_idx = min(max(device_specs["production_year"] - 2016, 0), len(data.ssd_bit_density_by_year) - 1)
        reference_scale = calc_dram_process_steps(data.dram_bit_density_by_year[year_idx]) / calc_dram_process_steps(0.022)
        dram_to_ssd_value_ratio = (
            (data.hynix_production_data["nand_revenue_ratio"][year_idx] / data.hynix_production_data["nand_k_wafers"][year_idx])
            / (
                data.hynix_production_data["dram_revenue_ratio"][year_idx]
                / data.hynix_production_data["estimated_dram_k_wafers"][year_idx]
            )
        )
        shipping_mass = (
            reference_scale
            * dram_to_ssd_value_ratio
            * legacy_model.get_equivalent_wafers(device_specs, bit_density=bit_density)
            * kg_chemicals_per_wafer_dram_57nm
        )

    _build_custom_transport_midpoint_records(
        records,
        distance_map=chemical_transportation,
        mass_kg=shipping_mass,
        stage="manufacturing",
        substage_prefix="scope3_chemical_transport",
        component_type=component_type,
        location_context=chemical_context,
        local_receiving_bucket="upstream_supply_chain_sites",
        spatial_awareness=spatial_awareness,
        fallbacks=fallbacks,
        manufacturing_scope="scope3",
    )


def _build_bom_midpoint_records(
    records: List[Dict[str, Any]],
    device_specs: Dict[str, Any],
    *,
    bom_template: Optional[Dict[str, float]],
    upstream_transportation: Optional[Dict[str, float]],
    spatial_awareness: bool,
    fallbacks: List[str],
) -> None:
    component_type = device_specs["component_type"]
    bom_impacts = legacy_model.calculate_manufacturing_bom_impacts(
        device_specs,
        bom_template=bom_template,
        upstream_transportation=None,
    )
    _add_group_midpoint_records(
        records,
        bom_impacts,
        stage="manufacturing",
        substage="scope3_bom_materials",
        component_type=component_type,
        process_name="bom_materials",
        actual_location=None,
        emission_location=None,
        resolved_location=None,
        area_id=None,
        local_receiving_bucket="upstream_supply_chain_sites",
        manufacturing_scope="scope3",
        spatial_awareness=spatial_awareness,
        notes="BOM production geography is unresolved; mapped to upstream_supply_chain_sites without an explicit area id.",
    )

    if upstream_transportation is not None:
        _build_custom_transport_midpoint_records(
            records,
            distance_map=upstream_transportation,
            mass_kg=device_specs["net_weight"] / 1000,
            stage="manufacturing",
            substage_prefix="scope3_bom_transport",
            component_type=component_type,
            location_context=None,
            local_receiving_bucket="upstream_supply_chain_sites",
            spatial_awareness=spatial_awareness,
            fallbacks=fallbacks,
            manufacturing_scope="scope3",
            notes="BOM transport geography is unresolved; mapped to upstream_supply_chain_sites unless transport allocation rules override it.",
        )


def _build_packaging_material_midpoint_records(
    records: List[Dict[str, Any]],
    working_specs: Dict[str, Any],
    *,
    spatial_awareness: bool,
) -> None:
    packaging_mass = working_specs["gross_weight"] - working_specs["net_weight"]
    impacts = legacy_model.calculate_packaging_material_impacts(packaging_mass, mass_unit="g")
    _add_group_midpoint_records(
        records,
        impacts,
        stage="manufacturing",
        substage="scope3_packaging_materials",
        component_type=working_specs["component_type"],
        process_name="packaging_materials",
        actual_location=None,
        emission_location=None,
        resolved_location=None,
        area_id=None,
        local_receiving_bucket="upstream_supply_chain_sites",
        manufacturing_scope="scope3",
        spatial_awareness=spatial_awareness,
        notes="Packaging material production geography is unresolved; mapped to upstream_supply_chain_sites without an explicit area id.",
    )


def _build_cpu_die_midpoint_records(
    records: List[Dict[str, Any]],
    cpu_specs: Dict[str, Any],
    *,
    production_yield: float,
    spatial_awareness: bool,
    calculate_upstream_materials: bool,
    bom_template: Optional[Dict[str, float]],
    fallbacks: List[str],
    **upstream_transportation_kwargs: Any,
) -> None:
    component_type = cpu_specs["component_type"]
    fab_context = _logic_manufacturing_context(fallbacks)
    year_idx = cpu_specs["production_year"] - 2016
    if year_idx < 0 or year_idx > 8:
        raise ValueError("Year must be between 2016 and 2024.")

    total_produce_units = (
        data.node_to_layer_masks_map[cpu_specs["technology_node_nm"]] * (cpu_specs["die_size_mm2"] / legacy_model.EFFECTIVE_WAFER_AREA_MM2)
        + data.node_to_layer_masks_map[cpu_specs["io_die_technology_node_nm"]] * (cpu_specs["io_die_size_mm2"] / legacy_model.EFFECTIVE_WAFER_AREA_MM2)
    ) / production_yield
    location_specific_cfs = legacy_model._get_location_specific_midpoint_cfs(
        fab_context["emission_location"],
        spatial_awareness=spatial_awareness,
    )

    for acid in data.tsmc_acid_emission_mix_ratio["2016"].keys():
        pollutant_mass = (
            data.tsmc_acid_emission_mix_ratio[str(year_idx + 2016)][acid]
            * data.tsmc_emissions_macro["per_unit_acid_g/wafer-mask-layer"][year_idx]
            * total_produce_units
            / 1000
        )
        for midpoint in MIDPOINT_SCOPES:
            midpoint_cf = legacy_model._get_midpoint_impact_factor(midpoint, acid, location_specific_cfs)
            if midpoint_cf is None:
                continue
            _append_midpoint_record(
                records,
                stage="manufacturing",
                substage="scope1_fab_direct_air",
                component_type=component_type,
                process_name=f"fab_direct_{acid}",
                midpoint=midpoint,
                midpoint_value=midpoint_cf * pollutant_mass,
                actual_location=fab_context["actual_location"],
                emission_location=fab_context["emission_location"],
                resolved_location=fab_context["resolved_location"],
                area_id=fab_context["area_id"],
                local_receiving_bucket="manufacturing_site",
                pollutant=acid,
                pollutant_amount=pollutant_mass,
                manufacturing_scope="scope1",
                spatial_awareness=spatial_awareness,
            )

    for pollutant in ["SOx", "NOx", "VOC"]:
        pollutant_mass = (
            data.tsmc_emissions_macro[f"{pollutant}_mt"][year_idx]
            * 1e6
            / (
                data.tsmc_emissions_macro["total_acid_mt"][year_idx]
                * 1e6
                / data.tsmc_emissions_macro["per_unit_acid_g/wafer-mask-layer"][year_idx]
            )
            * total_produce_units
            / 1000
        )
        for midpoint in MIDPOINT_SCOPES:
            midpoint_cf = legacy_model._get_midpoint_impact_factor(midpoint, pollutant, location_specific_cfs)
            if midpoint_cf is None:
                continue
            _append_midpoint_record(
                records,
                stage="manufacturing",
                substage="scope1_fab_direct_air",
                component_type=component_type,
                process_name=f"fab_direct_{pollutant}",
                midpoint=midpoint,
                midpoint_value=midpoint_cf * pollutant_mass,
                actual_location=fab_context["actual_location"],
                emission_location=fab_context["emission_location"],
                resolved_location=fab_context["resolved_location"],
                area_id=fab_context["area_id"],
                local_receiving_bucket="manufacturing_site",
                pollutant=pollutant,
                pollutant_amount=pollutant_mass,
                manufacturing_scope="scope1",
                spatial_awareness=spatial_awareness,
            )

    for pollutant in ["Cu2+", "NH4-N", "COD"]:
        discharge_liters = data.tsmc_emissions_macro["per_unit_wastewater_L/wafer-mask-layer"][year_idx] * total_produce_units
        pollutant_mass = discharge_liters * data.tsmc_emissions_macro[f"{pollutant}_ppm"][year_idx] * 1e-6
        for midpoint in MIDPOINT_SCOPES:
            midpoint_cf = legacy_model._get_midpoint_impact_factor(midpoint, pollutant, location_specific_cfs)
            if midpoint_cf is None:
                continue
            _append_midpoint_record(
                records,
                stage="manufacturing",
                substage="scope1_fab_wastewater",
                component_type=component_type,
                process_name=f"fab_wastewater_{pollutant}",
                midpoint=midpoint,
                midpoint_value=midpoint_cf * pollutant_mass,
                actual_location=fab_context["actual_location"],
                emission_location=fab_context["emission_location"],
                resolved_location=fab_context["resolved_location"],
                area_id=fab_context["area_id"],
                local_receiving_bucket="manufacturing_site",
                pollutant=pollutant,
                pollutant_amount=pollutant_mass,
                manufacturing_scope="scope1",
                spatial_awareness=spatial_awareness,
            )

    electricity_kwh = (
        data.tsmc_electricity_consumption["unit_consumption_kWh/wafer-mask-layer"][year_idx]
        * total_produce_units
        * (1 - data.tsmc_electricity_consumption["renewable_energy_ratio_%"][year_idx] * 1e-2)
    )
    _build_energy_midpoint_records(
        records,
        energy_kwh=electricity_kwh,
        year=year_idx + 2016,
        location_context=fab_context,
        stage="manufacturing",
        substage="scope2_fab_electricity",
        component_type=component_type,
        process_name="fab_electricity",
        local_receiving_bucket="manufacturing_site",
        spatial_awareness=spatial_awareness,
        excluded_pollutants=["CO2e"],
        manufacturing_scope="scope2",
        notes="Fab electricity excludes CO2e to preserve the legacy manufacturing scope split while keeping scope-2 water from electricity generation.",
    )

    carbon_scope_split = legacy_model._estimate_logic_fab_carbon_scope_split(
        cpu_specs,
        production_yield=production_yield,
    )
    for scope_name, midpoint_value in carbon_scope_split.items():
        if abs(midpoint_value) <= EPSILON:
            continue
        if scope_name == "scope3":
            actual_location = None
            emission_location = None
            resolved_location = None
            area_id = None
            local_receiving_bucket = "upstream_supply_chain_sites"
            notes = (
                "CPU/GPU scope-3 fab carbon follows the legacy logic carbon scope split; "
                "its upstream geography is unresolved and is therefore mapped to upstream_supply_chain_sites."
            )
        else:
            actual_location = fab_context["actual_location"]
            emission_location = fab_context["emission_location"]
            resolved_location = fab_context["resolved_location"]
            area_id = fab_context["area_id"]
            local_receiving_bucket = "manufacturing_site"
            notes = "CPU/GPU scope-1 and scope-2 fab carbon follows the legacy logic carbon scope split."
        _append_midpoint_record(
            records,
            stage="manufacturing",
            substage=f"{scope_name}_logic_fab_gwp_split",
            component_type=component_type,
            process_name=f"logic_fab_gwp_{scope_name}",
            midpoint="GWP",
            midpoint_value=midpoint_value,
            actual_location=actual_location,
            emission_location=emission_location,
            resolved_location=resolved_location,
            area_id=area_id,
            local_receiving_bucket=local_receiving_bucket,
            manufacturing_scope=scope_name,
            spatial_awareness=spatial_awareness,
            notes=notes,
        )

    direct_water = cpu_specs.get("manufacturing_water_m3", 0.0) or (
        cpu_specs.get("die_size_mm2", 0)
        * data.embodied_carbon_and_water["water_consumption_cmos_L_per_cm2_by_node"].get(str(cpu_specs["technology_node_nm"]), 0)
        / production_yield
        / 100000
    )
    _append_midpoint_record(
        records,
        stage="manufacturing",
        substage="scope1_fab_direct_water",
        component_type=component_type,
        process_name="fab_direct_water",
        midpoint="WC",
        midpoint_value=direct_water,
        actual_location=fab_context["actual_location"],
        emission_location=fab_context["emission_location"],
        resolved_location=fab_context["resolved_location"],
        area_id=fab_context["area_id"],
        local_receiving_bucket="manufacturing_site",
        manufacturing_scope="scope1",
        spatial_awareness=spatial_awareness,
    )

    if calculate_upstream_materials:
        _build_material_midpoint_records(
            records,
            {**cpu_specs, "production_yield": production_yield},
            spatial_awareness=spatial_awareness,
            fallbacks=fallbacks,
            **upstream_transportation_kwargs,
        )
        if component_type == "CPU":
            _build_bom_midpoint_records(
                records,
                cpu_specs,
                bom_template=bom_template,
                upstream_transportation=upstream_transportation_kwargs.get("upstream_transportation"),
                spatial_awareness=spatial_awareness,
                fallbacks=fallbacks,
            )

    _build_backend_midpoint_records(
        records,
        production_year=cpu_specs["production_year"],
        n_ic=1,
        component_type=component_type,
        spatial_awareness=spatial_awareness,
        fallbacks=fallbacks,
    )


def _build_storage_midpoint_records(
    records: List[Dict[str, Any]],
    storage_specs: Dict[str, Any],
    *,
    storage_type: str,
    production_yield: float,
    spatial_awareness: bool,
    calculate_upstream_materials: bool,
    fallbacks: List[str],
    bit_density: Optional[float] = None,
    **upstream_transportation_kwargs: Any,
) -> None:
    if storage_type == "HDD":
        raise NotImplementedError(
            "HDD is not yet supported in modeling_distibution.py because its current manufacturing path is pre-tabulated by stage."
        )

    fab_context = _memory_manufacturing_context(fallbacks)
    year = storage_specs["production_year"]
    year_idx = year - 2016
    equivalent_wafers = legacy_model.get_equivalent_wafers(
        {
            "device_type": storage_type,
            "capacity": storage_specs["capacity"],
            "production_year": year,
            "hbm_type": storage_specs.get("hbm_type"),
        },
        bit_density=bit_density,
        production_yield=production_yield,
    )

    location_specific_cfs = legacy_model._get_location_specific_midpoint_cfs(
        fab_context["emission_location"],
        spatial_awareness=spatial_awareness,
    )
    pollutant_series = data.dram_emissions
    if storage_type == "SSD":
        fallbacks.append(
            "SSD direct fab pollutant distribution mirrors the legacy storage midpoint path, which currently uses the DRAM/HBM pollutant series for parity."
        )
    for pollutant, per_wafer_mass in pollutant_series[year].items():
        pollutant_mass = per_wafer_mass * equivalent_wafers
        for midpoint in MIDPOINT_SCOPES:
            midpoint_cf = legacy_model._get_midpoint_impact_factor(midpoint, pollutant, location_specific_cfs)
            if midpoint_cf is None:
                continue
            _append_midpoint_record(
                records,
                stage="manufacturing",
                substage="scope1_memory_fab_direct",
                component_type=storage_type,
                process_name=f"memory_fab_direct_{pollutant}",
                midpoint=midpoint,
                midpoint_value=midpoint_cf * pollutant_mass,
                actual_location=fab_context["actual_location"],
                emission_location=fab_context["emission_location"],
                resolved_location=fab_context["resolved_location"],
                area_id=fab_context["area_id"],
                local_receiving_bucket="manufacturing_site",
                pollutant=pollutant,
                pollutant_amount=pollutant_mass,
                manufacturing_scope="scope1",
                spatial_awareness=spatial_awareness,
            )

    device_revenue_ratio = (
        data.hynix_production_data["dram_revenue_ratio"][year_idx]
        if storage_type in ["DRAM", "HBM"]
        else data.hynix_production_data["nand_revenue_ratio"][year_idx]
    )
    device_wafer_capacity = (
        data.hynix_production_data["estimated_dram_k_wafers"][year_idx] * 1000
        if storage_type in ["DRAM", "HBM"]
        else data.hynix_production_data["nand_k_wafers"][year_idx] * 1000
    )
    wafer_electricity_kwh = (
        data.hynix_production_data["electricity_consumption_GWh"][year_idx]
        * 1e6
        * device_revenue_ratio
        / device_wafer_capacity
        * (1 - data.hynix_production_data["renewable_energy_ratio_%"][year_idx] * 0.01)
    )
    _build_energy_midpoint_records(
        records,
        energy_kwh=wafer_electricity_kwh * equivalent_wafers,
        year=year,
        location_context=fab_context,
        stage="manufacturing",
        substage="scope2_memory_fab_electricity",
        component_type=storage_type,
        process_name="memory_fab_electricity",
        local_receiving_bucket="manufacturing_site",
        spatial_awareness=spatial_awareness,
        excluded_pollutants=["CO2e"],
        manufacturing_scope="scope2",
        notes="Fab electricity excludes CO2e to preserve the legacy manufacturing scope split while keeping scope-2 water from electricity generation.",
    )

    year_to_node_map = data.dram_year_to_node_map if storage_type in ["DRAM", "HBM"] else data.ssd_year_to_node_map
    carbon_per_wafer = (
        data.embodied_carbon_and_water["dram_carbon_emissions_kgCO2e_per_wafer_by_node"]
        if storage_type in ["DRAM", "HBM"]
        else data.embodied_carbon_and_water["ssd_carbon_emissions_kgCO2e_per_wafer_by_node"]
    )
    if storage_type == "HBM":
        node = data.vram_bit_density.get(f"{storage_specs.get('hbm_type')}_node", "2y")
    else:
        node = year_to_node_map.get(str(year), "")
    _append_midpoint_record(
        records,
        stage="manufacturing",
        substage="scope1_2_lumped_memory_gwp",
        component_type=storage_type,
        process_name="memory_fab_gwp_lumped",
        midpoint="GWP",
        midpoint_value=carbon_per_wafer.get(node, 0) * equivalent_wafers,
        actual_location=fab_context["actual_location"],
        emission_location=fab_context["emission_location"],
        resolved_location=fab_context["resolved_location"],
        area_id=fab_context["area_id"],
        local_receiving_bucket="manufacturing_site",
        manufacturing_scope="scope1_2_lumped",
        spatial_awareness=spatial_awareness,
        notes="DRAM/SSD/HBM manufacturing GWP follows the legacy lumped scope1-2 estimate.",
    )

    water_per_wafer = (
        data.hynix_production_data["total_water_consumed_1000_m3"][year_idx]
        * 1000
        * device_revenue_ratio
        / device_wafer_capacity
    )
    _append_midpoint_record(
        records,
        stage="manufacturing",
        substage="scope1_memory_fab_direct_water",
        component_type=storage_type,
        process_name="memory_fab_direct_water",
        midpoint="WC",
        midpoint_value=water_per_wafer * equivalent_wafers,
        actual_location=fab_context["actual_location"],
        emission_location=fab_context["emission_location"],
        resolved_location=fab_context["resolved_location"],
        area_id=fab_context["area_id"],
        local_receiving_bucket="manufacturing_site",
        manufacturing_scope="scope1",
        spatial_awareness=spatial_awareness,
    )

    if calculate_upstream_materials:
        _build_material_midpoint_records(
            records,
            {
                "device_type": storage_type,
                "component_type": storage_type,
                "capacity": storage_specs["capacity"],
                "production_year": year,
                "hbm_type": storage_specs.get("hbm_type"),
                "production_yield": production_yield,
            },
            bit_density=bit_density,
            spatial_awareness=spatial_awareness,
            fallbacks=fallbacks,
            **upstream_transportation_kwargs,
        )


def _build_manufacturing_midpoint_records(
    working_specs: Dict[str, Any],
    *,
    spatial_awareness: bool,
    calculate_upstream_materials: bool,
    bom_template: Optional[Dict[str, float]],
    fallbacks: List[str],
    **transportation_kwargs: Any,
) -> List[Dict[str, Any]]:
    component_type = working_specs["component_type"]
    records: List[Dict[str, Any]] = []

    if component_type == "CPU":
        _build_cpu_die_midpoint_records(
            records,
            working_specs,
            production_yield=working_specs.get("production_yield", 0.875),
            spatial_awareness=spatial_awareness,
            calculate_upstream_materials=calculate_upstream_materials,
            bom_template=bom_template,
            fallbacks=fallbacks,
            **transportation_kwargs,
        )
    elif component_type == "GPU":
        _build_cpu_die_midpoint_records(
            records,
            working_specs,
            production_yield=working_specs["die_production_yield"],
            spatial_awareness=spatial_awareness,
            calculate_upstream_materials=calculate_upstream_materials,
            bom_template=None,
            fallbacks=fallbacks,
            **transportation_kwargs,
        )
        _build_storage_midpoint_records(
            records,
            {
                "component_type": "HBM",
                "capacity": working_specs["hbm_capacity_GB"],
                "production_year": working_specs["production_year"],
                "hbm_type": working_specs["hbm_type"],
            },
            storage_type="HBM",
            production_yield=working_specs.get("memory_production_yield", 0.875),
            spatial_awareness=spatial_awareness,
            calculate_upstream_materials=calculate_upstream_materials,
            fallbacks=fallbacks,
            **transportation_kwargs,
        )
        if calculate_upstream_materials:
            _build_bom_midpoint_records(
                records,
                working_specs,
                bom_template=bom_template,
                upstream_transportation=transportation_kwargs.get("upstream_transportation"),
                spatial_awareness=spatial_awareness,
                fallbacks=fallbacks,
            )
        if transportation_kwargs.get("assembly_transportation_distance") is not None:
            _build_custom_transport_midpoint_records(
                records,
                distance_map=transportation_kwargs["assembly_transportation_distance"],
                mass_kg=working_specs.get("net_weight", 0) / 1000,
                stage="manufacturing",
                substage_prefix="scope1_internal_assembly_transport",
                component_type="GPU",
                location_context=None,
                local_receiving_bucket="manufacturing_site",
                spatial_awareness=spatial_awareness,
                fallbacks=fallbacks,
                manufacturing_scope="scope1",
                notes="GPU assembly transport location is ambiguous in the legacy API; local legs are treated as manufacturing-internal and intercontinental legs follow the transport split rules.",
            )
    elif component_type in ["DRAM", "SSD"]:
        _build_storage_midpoint_records(
            records,
            working_specs,
            storage_type=component_type,
            production_yield=working_specs.get("production_yield", 0.875),
            spatial_awareness=spatial_awareness,
            calculate_upstream_materials=calculate_upstream_materials,
            fallbacks=fallbacks,
            **transportation_kwargs,
        )
    elif component_type == "HDD":
        raise NotImplementedError(
            "HDD is not yet supported in modeling_distibution.py because the current manufacturing path is pre-tabulated by stage."
        )
    else:
        raise ValueError("Unsupported component type. Must be one of ['CPU', 'GPU', 'SSD', 'DRAM', 'HDD'].")

    _build_packaging_material_midpoint_records(
        records,
        working_specs,
        spatial_awareness=spatial_awareness,
    )
    return records


def _build_recycling_midpoint_records(
    working_specs: Dict[str, Any],
    *,
    use_region: Optional[str],
    eol_region: Optional[str],
    transport_region: Optional[str],
    spatial_awareness: bool,
    fallbacks: List[str],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    eol_context = _build_use_to_eol_transport_midpoint_records(
        records,
        working_specs,
        use_region=use_region,
        eol_region=eol_region,
        transport_region=transport_region,
        spatial_awareness=spatial_awareness,
        fallbacks=fallbacks,
    )

    packaging_mass = working_specs.get("gross_weight", 0) - working_specs.get("net_weight", 0)
    landfill_mass = working_specs["net_weight"] * (1 - working_specs.get("recycling_rate", 0.8232))
    incineration_impacts = legacy_model.calculate_recycling_impact(packaging_mass, mass_unit="g", pathway="inceneration")
    landfill_impacts = legacy_model.calculate_recycling_impact(landfill_mass, mass_unit="g", pathway="landfill")

    for impacts, substage, process_name in (
        (incineration_impacts, "packaging_incineration", "packaging_incineration"),
        (landfill_impacts, "device_landfill", "device_landfill"),
    ):
        _add_group_midpoint_records(
            records,
            impacts,
            stage="recycling",
            substage=substage,
            component_type=working_specs["component_type"],
            process_name=process_name,
            actual_location=eol_context.get("actual_location"),
            emission_location=eol_context.get("emission_location"),
            resolved_location=eol_context.get("resolved_location"),
            area_id=eol_context.get("area_id"),
            local_receiving_bucket="use_site" if eol_context.get("actual_location") else "rest_of_world",
            manufacturing_scope=None,
            spatial_awareness=spatial_awareness,
            notes="End-of-life impacts are folded into use_site because the requested bucket schema does not define a separate eol_site bucket.",
        )
    return records


def _build_operational_midpoint_records(
    device_specs: Dict[str, Any],
    *,
    load_ratio: float,
    years: int,
    location: str,
    use_region: Optional[str],
    emission_factor_year: Optional[int],
    spatial_awareness: bool,
    excluded_pollutants: Optional[Iterable[str]],
    datacenter_wue: Optional[float],
    pue: float,
    fallbacks: List[str],
) -> List[Dict[str, Any]]:
    actual_location = use_region or location
    location_context = _distribution_location_context(actual_location, fallbacks)
    effective_datacenter_wue = datacenter_wue
    if effective_datacenter_wue is None and location_context.get("direct_wue") is not None:
        effective_datacenter_wue = location_context["direct_wue"]

    if device_specs["component_type"] not in ["CPU", "GPU"]:
        achieved_power = (
            (load_ratio - 0.3) / 0.7 * (device_specs["peak_power"] - device_specs["idle_power"])
            + device_specs["idle_power"]
        )
        annual_energy = achieved_power * 24 * 365 / 1000
    else:
        annual_energy = device_specs["TDP"] * load_ratio * 24 * 365 / 1000

    records: List[Dict[str, Any]] = []
    if emission_factor_year is not None:
        energy_schedule = [(emission_factor_year, annual_energy * years)]
    else:
        energy_schedule = [
            (year, annual_energy)
            for year in range(device_specs["production_year"], device_specs["production_year"] + years)
        ]

    for year, energy_kwh in energy_schedule:
        _build_energy_midpoint_records(
            records,
            energy_kwh=energy_kwh,
            year=year,
            location_context=location_context,
            stage="operational",
            substage="use_phase",
            component_type=device_specs["component_type"],
            process_name="operational_energy",
            local_receiving_bucket="use_site",
            spatial_awareness=spatial_awareness,
            excluded_pollutants=excluded_pollutants,
            datacenter_wue=effective_datacenter_wue,
            pue=pue,
            notes="Operational direct WUE is sourced from the regional WUE dataset when use_region is supplied and datacenter_wue is omitted.",
        )
    return records


def _scale_midpoint_records(records: List[Dict[str, Any]], scalar: float) -> None:
    if scalar == 1.0:
        return
    for record in records:
        record["midpoint_value"] *= scalar
        if record.get("pollutant_amount") is not None:
            record["pollutant_amount"] *= scalar


def _build_area_share_context(midpoint_records: List[Dict[str, Any]], fallbacks: List[str]) -> Dict[str, Any]:
    dataset = _distribution_area_dataset()
    bucket_area_ids = {
        "manufacturing_site": set(),
        "use_site": set(),
        "upstream_supply_chain_sites": set(),
    }
    for record in midpoint_records:
        bucket = record.get("local_receiving_bucket")
        area_id = record.get("area_id")
        if bucket in bucket_area_ids and area_id in dataset.get("areas", {}):
            bucket_area_ids[bucket].add(area_id)

    explicit_area_ids = sorted(set().union(*bucket_area_ids.values())) if bucket_area_ids else []
    explicit_area_sq_km = sum(dataset["areas"][area_id]["area_sq_km"] for area_id in explicit_area_ids)
    bucket_area_sq_km = {bucket: 0.0 for bucket in RECEIVING_BUCKETS}
    bucket_area_breakdown_sq_km = {bucket: {} for bucket in RECEIVING_BUCKETS}

    for area_id in explicit_area_ids:
        owning_buckets = [bucket for bucket, area_ids in bucket_area_ids.items() if area_id in area_ids]
        if not owning_buckets:
            continue
        split_area = dataset["areas"][area_id]["area_sq_km"] / len(owning_buckets)
        for bucket in owning_buckets:
            bucket_area_sq_km[bucket] += split_area
            bucket_area_breakdown_sq_km[bucket][area_id] = split_area

    bucket_area_sq_km["ocean"] = dataset["metadata"]["ocean_area_sq_km"]
    residual_sq_km = dataset["metadata"]["world_surface_sq_km"] - dataset["metadata"]["ocean_area_sq_km"] - explicit_area_sq_km
    if residual_sq_km < 0:
        _append_unique_fallback(
            fallbacks,
            "Explicit modeled areas exceeded the non-ocean world surface area. rest_of_world was clipped to zero."
        )
        residual_sq_km = 0.0
    bucket_area_sq_km["rest_of_world"] = residual_sq_km
    bucket_area_breakdown_sq_km["rest_of_world"]["__residual_land__"] = residual_sq_km
    bucket_area_breakdown_sq_km["ocean"]["__ocean__"] = bucket_area_sq_km["ocean"]

    denominator = sum(bucket_area_sq_km.values())
    shares = {
        bucket: (bucket_area_sq_km[bucket] / denominator if denominator > 0 else 0.0)
        for bucket in RECEIVING_BUCKETS
    }
    land_denominator = denominator - bucket_area_sq_km["ocean"]
    land_shares = {
        bucket: (
            bucket_area_sq_km[bucket] / land_denominator
            if land_denominator > 0 and bucket != "ocean"
            else 0.0
        )
        for bucket in RECEIVING_BUCKETS
    }
    return {
        "bucket_area_sq_km": bucket_area_sq_km,
        "bucket_area_breakdown_sq_km": bucket_area_breakdown_sq_km,
        "bucket_area_ids": {bucket: sorted(area_ids) for bucket, area_ids in bucket_area_ids.items()},
        "explicit_area_ids": explicit_area_ids,
        "shares": shares,
        "land_shares": land_shares,
        "world_surface_sq_km": dataset["metadata"]["world_surface_sq_km"],
        "ocean_area_sq_km": dataset["metadata"]["ocean_area_sq_km"],
        "sources": dataset["metadata"].get("sources", []),
    }


def _normalize_positive_bii(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def _build_bii_context(area_share_context: Dict[str, Any], fallbacks: List[str]) -> Dict[str, Any]:
    dataset = _bii_distribution_dataset()
    area_bii_2023 = {
        area_id: float(value)
        for area_id, value in dataset.get("area_bii_2023", {}).items()
        if _normalize_positive_bii(value) is not None
    }
    bucket_defaults = {
        bucket: float(value)
        for bucket, value in dataset.get("bucket_default_bii_2023", {}).items()
        if _normalize_positive_bii(value) is not None
    }

    bucket_bii_2023: Dict[str, float] = {}
    bucket_bii_source: Dict[str, str] = {}
    for bucket in RECEIVING_BUCKETS:
        weighted_sum = 0.0
        weighted_area = 0.0
        missing_area_ids = []
        for area_id, area_sq_km in area_share_context.get("bucket_area_breakdown_sq_km", {}).get(bucket, {}).items():
            if area_id.startswith("__") or area_sq_km <= 0:
                continue
            bii_value = area_bii_2023.get(area_id)
            if bii_value is None:
                missing_area_ids.append(area_id)
                continue
            weighted_sum += bii_value * area_sq_km
            weighted_area += area_sq_km

        if weighted_area > 0:
            bucket_bii_2023[bucket] = weighted_sum / weighted_area
            bucket_bii_source[bucket] = "bucket_area_weighted"
            if missing_area_ids:
                _append_unique_fallback(
                    fallbacks,
                    f"BII weighting ignored unresolved area ids {sorted(set(missing_area_ids))} while averaging bucket {bucket!r}.",
                )
            continue

        fallback_bii = bucket_defaults.get(bucket)
        if fallback_bii is not None:
            bucket_bii_2023[bucket] = fallback_bii
            bucket_bii_source[bucket] = "bucket_default"
            if area_share_context.get("bucket_area_ids", {}).get(bucket):
                _append_unique_fallback(
                    fallbacks,
                    f"BII weighting fell back to the default divisor for bucket {bucket!r} because none of its explicit areas had a prepared BII constant.",
                )
            continue

        bucket_bii_2023[bucket] = 1.0
        bucket_bii_source[bucket] = "neutral_default"
        _append_unique_fallback(
            fallbacks,
            f"BII weighting used a neutral divisor of 1.0 for bucket {bucket!r} because no specific constant was available.",
        )

    return {
        "year": dataset.get("metadata", {}).get("year", 2023),
        "area_bii_2023": area_bii_2023,
        "bucket_bii_2023": bucket_bii_2023,
        "bucket_bii_source": bucket_bii_source,
        "bucket_default_bii_2023": bucket_defaults,
        "source_files": dataset.get("metadata", {}).get("source_files", []),
        "notes": dataset.get("metadata", {}).get("notes", []),
    }


def _resolve_bii_for_allocated_flow(
    midpoint_record: Dict[str, Any],
    *,
    receiving_bucket: str,
    allocation_method: str,
    bii_context: Dict[str, Any],
    fallbacks: List[str],
) -> Dict[str, Any]:
    bucket_bii_2023 = bii_context.get("bucket_bii_2023", {})
    area_bii_2023 = bii_context.get("area_bii_2023", {})

    if allocation_method in {
        "marine_to_ocean",
        "global_area_share",
        "intercontinental_transport_direct_global_area_share",
        "intercontinental_transport_upstream_full_minus_nfp",
        "unknown_location_fallback",
    }:
        bii_value = _normalize_positive_bii(bucket_bii_2023.get(receiving_bucket))
        if bii_value is None:
            _append_unique_fallback(
                fallbacks,
                f"BII weighting fell back to a neutral divisor for globally or bucket-allocated flows in {receiving_bucket!r}.",
            )
            bii_value = 1.0
            bii_source = f"bucket:{receiving_bucket}:neutral_default"
        else:
            bii_source = f"bucket:{receiving_bucket}:{bii_context.get('bucket_bii_source', {}).get(receiving_bucket, 'prepared')}"
        return {"bii_value": bii_value, "bii_source": bii_source}

    area_id = midpoint_record.get("area_id")
    area_bii = _normalize_positive_bii(area_bii_2023.get(area_id)) if area_id else None
    if area_bii is not None:
        return {"bii_value": area_bii, "bii_source": f"area:{area_id}"}

    bucket_bii = _normalize_positive_bii(bucket_bii_2023.get(receiving_bucket))
    if bucket_bii is not None:
        if area_id is None:
            _append_unique_fallback(
                fallbacks,
                f"BII weighting used the receiving-bucket constant for {receiving_bucket!r} because the flow location was unresolved.",
            )
        else:
            _append_unique_fallback(
                fallbacks,
                f"BII weighting used the receiving-bucket constant for area {area_id!r} in {receiving_bucket!r} because no area-specific constant was prepared.",
            )
        return {
            "bii_value": bucket_bii,
            "bii_source": f"bucket:{receiving_bucket}:{bii_context.get('bucket_bii_source', {}).get(receiving_bucket, 'prepared')}",
        }

    _append_unique_fallback(
        fallbacks,
        f"BII weighting used a neutral divisor of 1.0 for {receiving_bucket!r} because neither an area-specific nor bucket-level constant was available.",
    )
    return {"bii_value": 1.0, "bii_source": f"bucket:{receiving_bucket}:neutral_default"}


def _convert_record_to_endpoint_scopes(record: Dict[str, Any], perspective: str) -> Dict[str, float]:
    endpoint_values = {scope: 0.0 for scope in ENDPOINT_SCOPES}
    for scope in ENDPOINT_SCOPES:
        converted = legacy_model.midpoint_to_endpoint(
            {record["midpoint"]: record["midpoint_value"]},
            perspective=perspective,
            endpoint_scopes=(scope,),
            location=record.get("resolved_location") or record.get("actual_location"),
            spatial_awareness=record.get("spatial_awareness", False),
        )
        endpoint_values[scope] = converted.get(record["midpoint"], 0.0)
    return endpoint_values


def _allocation_shares_for_record(
    record: Dict[str, Any],
    endpoint_scope: str,
    area_share_context: Dict[str, Any],
) -> Dict[str, float]:
    if endpoint_scope == "marine":
        return {"ocean": 1.0}
    if record["allocation_context"] == "intercontinental_transport_upstream":
        return {"upstream_supply_chain_sites": 1.0}
    if record["midpoint"] == "GWP" or record["allocation_context"] == "intercontinental_transport_direct":
        return area_share_context["land_shares"]
    local_bucket = record.get("local_receiving_bucket") or "rest_of_world"
    return {local_bucket: 1.0}


def _allocation_method_for_record(record: Dict[str, Any], endpoint_scope: str) -> str:
    if endpoint_scope == "marine":
        return "marine_to_ocean"
    if record["allocation_context"] == "intercontinental_transport_upstream":
        return "intercontinental_transport_upstream_full_minus_nfp"
    if record["allocation_context"] == "intercontinental_transport_direct":
        return "intercontinental_transport_direct_global_area_share"
    if record["midpoint"] == "GWP":
        return "global_area_share"
    if record.get("local_receiving_bucket") is None:
        return "unknown_location_fallback"
    return "local_actual_location"


def _aggregate_stage_midpoints(midpoint_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_stage = defaultdict(_zero_midpoints)
    for record in midpoint_records:
        by_stage[record["stage"]][record["midpoint"]] += record["midpoint_value"]

    if by_stage:
        total = _zero_midpoints()
        for payload in by_stage.values():
            for midpoint in MIDPOINT_SCOPES:
                total[midpoint] += payload[midpoint]
        by_stage["total"] = total
    return {stage: dict(values) for stage, values in by_stage.items()}


def _midpoint_allocation_shares_for_record(
    record: Dict[str, Any],
    area_share_context: Dict[str, Any],
) -> Dict[str, float]:
    if record["allocation_context"] == "intercontinental_transport_upstream":
        return {"upstream_supply_chain_sites": 1.0}
    if record["midpoint"] in {"MEP", "METP"}:
        return {"ocean": 1.0}
    if record["midpoint"] == "GWP" or record["allocation_context"] == "intercontinental_transport_direct":
        return area_share_context["land_shares"]
    local_bucket = record.get("local_receiving_bucket") or "rest_of_world"
    return {local_bucket: 1.0}


def _aggregate_midpoint_by_location(
    midpoint_records: List[Dict[str, Any]],
    area_share_context: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    grouped = defaultdict(_zero_midpoints)
    for record in midpoint_records:
        shares = _midpoint_allocation_shares_for_record(record, area_share_context)
        for bucket, share in shares.items():
            if share <= 0:
                continue
            grouped[bucket][record["midpoint"]] += record["midpoint_value"] * share

    if grouped:
        total = _zero_midpoints()
        for payload in grouped.values():
            for midpoint in MIDPOINT_SCOPES:
                total[midpoint] += payload[midpoint]
        grouped["total"] = total
    return {bucket: dict(values) for bucket, values in grouped.items()}


def _aggregate_endpoint_summaries(
    flow_records: List[Dict[str, Any]],
    key_name: str,
    *,
    value_field: str = "endpoint_value",
) -> Dict[str, Dict[str, Any]]:
    grouped = defaultdict(_zero_endpoint_summary)
    for record in flow_records:
        value = record.get(value_field)
        if value is None:
            continue
        group_key = record[key_name]
        grouped[group_key]["total"] += value
        grouped[group_key]["by_scope"][record["endpoint_scope"]] += value
        grouped[group_key]["by_midpoint"][record["midpoint"]] += value

    if grouped:
        total = _zero_endpoint_summary()
        for payload in grouped.values():
            total["total"] += payload["total"]
            for scope in ENDPOINT_SCOPES:
                total["by_scope"][scope] += payload["by_scope"][scope]
            for midpoint in MIDPOINT_SCOPES:
                total["by_midpoint"][midpoint] += payload["by_midpoint"][midpoint]
        grouped["total"] = total

    return {
        group_key: {
            "total": payload["total"],
            "by_scope": dict(payload["by_scope"]),
            "by_midpoint": dict(payload["by_midpoint"]),
        }
        for group_key, payload in grouped.items()
    }


def _finalize_distribution(
    midpoint_records: List[Dict[str, Any]],
    *,
    perspective: str,
    fallbacks: List[str],
    bii_weighting: bool = False,
) -> Dict[str, Any]:
    area_share_context = _build_area_share_context(midpoint_records, fallbacks)
    bii_context = _build_bii_context(area_share_context, fallbacks) if bii_weighting else None
    flow_records: List[Dict[str, Any]] = []

    for midpoint_record in midpoint_records:
        endpoint_values = _convert_record_to_endpoint_scopes(midpoint_record, perspective)
        for endpoint_scope, endpoint_value in endpoint_values.items():
            if abs(endpoint_value) <= EPSILON:
                continue
            allocation_shares = _allocation_shares_for_record(
                midpoint_record,
                endpoint_scope,
                area_share_context,
            )
            allocation_method = _allocation_method_for_record(midpoint_record, endpoint_scope)
            for receiving_bucket, share in allocation_shares.items():
                if share <= 0:
                    continue
                allocated_endpoint_value = endpoint_value * share
                flow_record = {
                    "stage": midpoint_record["stage"],
                    "substage": midpoint_record["substage"],
                    "component_type": midpoint_record["component_type"],
                    "process_name": midpoint_record["process_name"],
                    "midpoint": midpoint_record["midpoint"],
                    "midpoint_value": midpoint_record["midpoint_value"],
                    "actual_location": midpoint_record["actual_location"],
                    "receiving_bucket": receiving_bucket,
                    "allocation_method": allocation_method,
                    "endpoint_scope": endpoint_scope,
                    "endpoint_value": allocated_endpoint_value,
                    "spatial_awareness": midpoint_record["spatial_awareness"],
                    "notes": midpoint_record["notes"],
                    "allocation_share": share,
                    "manufacturing_scope": midpoint_record.get("manufacturing_scope"),
                    "pollutant": midpoint_record.get("pollutant"),
                    "pollutant_amount": midpoint_record.get("pollutant_amount"),
                    "transport_mode": midpoint_record.get("transport_mode"),
                    "transport_piece": midpoint_record.get("transport_piece"),
                    "transport_distance_km": midpoint_record.get("transport_distance_km"),
                    "intercontinental": midpoint_record.get("intercontinental"),
                    "resolved_location": midpoint_record.get("resolved_location"),
                    "emission_location": midpoint_record.get("emission_location"),
                    "area_id": midpoint_record.get("area_id"),
                }
                if bii_weighting:
                    bii_payload = _resolve_bii_for_allocated_flow(
                        midpoint_record,
                        receiving_bucket=receiving_bucket,
                        allocation_method=allocation_method,
                        bii_context=bii_context or {},
                        fallbacks=fallbacks,
                    )
                    flow_record["bii_value"] = bii_payload["bii_value"]
                    flow_record["bii_source"] = bii_payload["bii_source"]
                    flow_record["bii_weighted_endpoint_value"] = allocated_endpoint_value / bii_payload["bii_value"]
                flow_records.append(flow_record)

    payload = {
        "flow_records": flow_records,
        "by_location_midpoint": _aggregate_midpoint_by_location(midpoint_records, area_share_context),
        "by_location_endpoint": _aggregate_endpoint_summaries(flow_records, "receiving_bucket"),
        "by_stage_midpoint": _aggregate_stage_midpoints(midpoint_records),
        "by_stage_endpoint": _aggregate_endpoint_summaries(flow_records, "stage"),
        "metadata": {
            "perspective": _normalize_perspective(perspective),
            "bii_weighting": bool(bii_weighting),
            "receiving_buckets": list(RECEIVING_BUCKETS),
            "endpoint_scopes": list(ENDPOINT_SCOPES),
            "midpoint_record_count": len(midpoint_records),
            "flow_record_count": len(flow_records),
            "area_share_context": area_share_context,
        },
        "fallbacks": fallbacks,
    }
    if bii_weighting:
        payload["by_location_endpoint_bii_weighted"] = _aggregate_endpoint_summaries(
            flow_records,
            "receiving_bucket",
            value_field="bii_weighted_endpoint_value",
        )
        payload["by_stage_endpoint_bii_weighted"] = _aggregate_endpoint_summaries(
            flow_records,
            "stage",
            value_field="bii_weighted_endpoint_value",
        )
        payload["metadata"]["bii_context"] = bii_context
    return payload


def calculate_manufacturing_distribution(
    device_specs: Dict[str, Any],
    *,
    spatial_awareness: bool = True,
    calculate_upstream_materials: bool = True,
    bom_template: Optional[Dict[str, float]] = None,
    perspective: str = data.DEFAULT_RECIPE_PERSPECTIVE,
    bii_weighting: bool = False,
    **transportation_kwargs: Any,
) -> Dict[str, Any]:
    working_specs = legacy_model._with_derived_transport_weights(device_specs)
    fallbacks: List[str] = []
    midpoint_records = _build_manufacturing_midpoint_records(
        working_specs,
        spatial_awareness=spatial_awareness,
        calculate_upstream_materials=calculate_upstream_materials,
        bom_template=bom_template,
        fallbacks=fallbacks,
        **transportation_kwargs,
    )
    return _finalize_distribution(
        midpoint_records,
        perspective=perspective,
        fallbacks=fallbacks,
        bii_weighting=bii_weighting,
    )


def calculate_transport_distribution(
    device_specs: Dict[str, Any],
    *,
    use_region: Optional[str] = None,
    transport_region: Optional[str] = None,
    spatial_awareness: bool = True,
    perspective: str = data.DEFAULT_RECIPE_PERSPECTIVE,
    bii_weighting: bool = False,
) -> Dict[str, Any]:
    working_specs = legacy_model._with_derived_transport_weights(device_specs)
    fallbacks: List[str] = []
    midpoint_records: List[Dict[str, Any]] = []
    _build_manufacture_to_use_transport_midpoint_records(
        midpoint_records,
        working_specs,
        use_region=use_region,
        transport_region=transport_region,
        spatial_awareness=spatial_awareness,
        fallbacks=fallbacks,
    )
    return _finalize_distribution(
        midpoint_records,
        perspective=perspective,
        fallbacks=fallbacks,
        bii_weighting=bii_weighting,
    )


def calculate_operational_distribution(
    device_specs: Dict[str, Any],
    load_ratio: float,
    *,
    years: int = 5,
    location: str = "US",
    use_region: Optional[str] = None,
    emission_factor_year: Optional[int] = None,
    spatial_awareness: bool = True,
    excluded_pollutants: Optional[Iterable[str]] = None,
    datacenter_wue: Optional[float] = None,
    pue: float = 1.2,
    perspective: str = data.DEFAULT_RECIPE_PERSPECTIVE,
    bii_weighting: bool = False,
) -> Dict[str, Any]:
    fallbacks: List[str] = []
    midpoint_records = _build_operational_midpoint_records(
        device_specs,
        load_ratio=load_ratio,
        years=years,
        location=location,
        use_region=use_region,
        emission_factor_year=emission_factor_year,
        spatial_awareness=spatial_awareness,
        excluded_pollutants=excluded_pollutants,
        datacenter_wue=datacenter_wue,
        pue=pue,
        fallbacks=fallbacks,
    )
    return _finalize_distribution(
        midpoint_records,
        perspective=perspective,
        fallbacks=fallbacks,
        bii_weighting=bii_weighting,
    )


def calculate_recycling_distribution(
    device_specs: Dict[str, Any],
    *,
    use_region: Optional[str] = None,
    eol_region: Optional[str] = None,
    transport_region: Optional[str] = None,
    spatial_awareness: bool = True,
    perspective: str = data.DEFAULT_RECIPE_PERSPECTIVE,
    bii_weighting: bool = False,
) -> Dict[str, Any]:
    working_specs = legacy_model._with_derived_transport_weights(device_specs)
    fallbacks: List[str] = []
    midpoint_records = _build_recycling_midpoint_records(
        working_specs,
        use_region=use_region,
        eol_region=eol_region,
        transport_region=transport_region,
        spatial_awareness=spatial_awareness,
        fallbacks=fallbacks,
    )
    return _finalize_distribution(
        midpoint_records,
        perspective=perspective,
        fallbacks=fallbacks,
        bii_weighting=bii_weighting,
    )


def calculate_total_impact_distribution(
    device_specs: Dict[str, Any],
    occupy_ratio: float = 1.0,
    *,
    manufacturing_only: bool = False,
    spatial_awareness: bool = True,
    calculate_upstream_materials: bool = True,
    bom_template: Optional[Dict[str, float]] = None,
    use_region: Optional[str] = None,
    eol_region: Optional[str] = None,
    transport_region: Optional[str] = None,
    perspective: str = data.DEFAULT_RECIPE_PERSPECTIVE,
    bii_weighting: bool = False,
    **transportation_kwargs: Any,
) -> Dict[str, Any]:
    working_specs = legacy_model._with_derived_transport_weights(device_specs)
    fallbacks: List[str] = []

    midpoint_records = _build_manufacturing_midpoint_records(
        working_specs,
        spatial_awareness=spatial_awareness,
        calculate_upstream_materials=calculate_upstream_materials,
        bom_template=bom_template,
        fallbacks=fallbacks,
        **transportation_kwargs,
    )

    if not manufacturing_only:
        _build_manufacture_to_use_transport_midpoint_records(
            midpoint_records,
            working_specs,
            use_region=use_region,
            transport_region=transport_region,
            spatial_awareness=spatial_awareness,
            fallbacks=fallbacks,
        )
        midpoint_records.extend(
            _build_recycling_midpoint_records(
                working_specs,
                use_region=use_region,
                eol_region=eol_region,
                transport_region=transport_region,
                spatial_awareness=spatial_awareness,
                fallbacks=fallbacks,
            )
        )

    _scale_midpoint_records(midpoint_records, occupy_ratio)
    return _finalize_distribution(
        midpoint_records,
        perspective=perspective,
        fallbacks=fallbacks,
        bii_weighting=bii_weighting,
    )
