

def ACT(device_specs, production_yield=0.875, storage_reference_capacity_GB=8):
    '''
    Embodied carbon = Area / Yield * (EPA * CI_fab + GPA + MPA) # logic device
                      Capacity * per_GB_carbon # memory device
                      + N_IC * packaging_carbon_per_IC # packaging carbon for all devices

    Returns embodied carbon in kgCO2
    '''
    epa_by_node_kWh_per_cm2 = {
        '28': 0.9,
        '20': 1.2,
        '14': 1.2,
        '10': 1.475,
        '7': 1.52,
        '7-EUV': 2.15,
        '5': 2.75,
        '3': 2.75
    }
    gpa_by_node_gCO2_per_cm2 = {
        '28': 137.5,
        '20': 150,
        '14': 162.5,
        '10': 195,
        '7': 300,
        '7-EUV': 275,
        '5': 327.5,
        '3': 327.5
    }
    mpa_gCO2_per_cm2 = 500
    ci_fab = 583 * 0.75 + 41 * 0.25 # 75% Taiwan grid, 25% renewable (solar), gCO2 per kWh
    packaging_carbon_per_IC_gCO2 = 150

    embodied_carbon_per_capcity_dram_by_node_gCO2_per_GB = {
        '5x': 600,
        '4x': 315,
        '3x': 230,
        '3x-LPDDR': 201,
        '2x-LPDDR2': 159,
        '2x-LPDDR3': 184,
        'LPDDR4': 48,
        '1x': 65
    }
    embodied_carbon_per_capcity_ssd_by_node_gCO2_per_GB = {
        '30nm': 30,
        '20nm': 15,
        '10nm': 10,
        '1z TLC': 5.6,
        'V3 TLC': 6.3,
    }
    area = device_specs['die_size_mm2'] / 100 # convert from mm^2 to cm^2

    if device_specs['component_type'] in ['CPU', 'GPU']:
        epa = epa_by_node_kWh_per_cm2[device_specs['technology_node']]
        gpa = gpa_by_node_gCO2_per_cm2[device_specs['technology_node']]
        embodied_carbon = area / production_yield * (epa * ci_fab + gpa + mpa_gCO2_per_cm2) + packaging_carbon_per_IC_gCO2
        if device_specs['component_type'] == 'GPU':
            embodied_carbon += device_specs['hbm_capacity_GB'] * embodied_carbon_per_capcity_dram_by_node_gCO2_per_GB['1x'] + packaging_carbon_per_IC_gCO2 * device_specs['hbm_capacity_GB'] / storage_reference_capacity_GB # add DRAM carbon for HBM, assuming 1x DRAM for HBM
    elif device_specs['component_type'] == 'DRAM':
        assert device_specs['technology_node'] in embodied_carbon_per_capcity_dram_by_node_gCO2_per_GB, f"DRAM technology node {device_specs['technology_node']} not found in carbon data"
        embodied_carbon = device_specs['capacity'] * embodied_carbon_per_capcity_dram_by_node_gCO2_per_GB[device_specs['technology_node']] + packaging_carbon_per_IC_gCO2 * device_specs['capacity'] / storage_reference_capacity_GB # add packaging carbon, assuming 1 IC per storage_reference_capacity_GB
    elif device_specs['component_type'] == 'SSD':
        assert device_specs['technology_node'] in embodied_carbon_per_capcity_ssd_by_node_gCO2_per_GB, f"SSD technology node {device_specs['technology_node']} not found in carbon data"
        embodied_carbon = device_specs['capacity'] * embodied_carbon_per_capcity_ssd_by_node_gCO2_per_GB[device_specs['technology_node']] + packaging_carbon_per_IC_gCO2 * device_specs['capacity'] / storage_reference_capacity_GB
    elif device_specs['component_type'] == 'HDD':
        embodied_carbon = device_specs['capacity'] * 1.33 # Exos x16
    else:
        raise ValueError(f"Unsupported component type {device_specs['component_type']}")

    return embodied_carbon / 1000 # convert from gCO2 to kgCO2

def ThirstyFLOPS(device_specs, production_yield=0.875, manufacturer='TSMC'):
    '''
    Embodied water = Area / Yield * (UPW + EPA * WUE + EPA * EWF * PUE) # logic device
                      Capacity * WPC # memory device
                      + N_IC * packaging_water_per_IC # packaging water for all devices
    Returns embodied water in m3
    '''
    wpc_by_type_liters_per_GB = {
        'DRAM': 0.8,
        'SSD': 0.0224,
        'HDD': 0.03318
    }
    storage_reference_capacity_GB = {
        'DRAM': 24, # 24GB LPDDR4X drive
        'SSD': 7680, # 7.68TB SSD
    }
    packaging_water_per_IC_liters = 0.6
    upw_by_node_liters_per_cm2 = {
        "28" : 5.9,
        "20" : 6.3,
        "14" : 6,
        "12" : 6.5,
        "10" : 7.8,
        "8"  : 6,
        "7"  : 10.5,
        "6"  : 12.5,
        "5"  : 12.5,
        "4"  : 13.5,
        "3"  : 14.2
    }
    epa_by_node_kWh_per_cm2 = {
        "28" : 0.90,
        "20" : 1.200,
        "14" : 1.250,
        "12" : 1.50,
        "10" : 1.475,
        "8"  : 1.520,
        "7"  : 2.150,
        "6"  : 2.45,
        "5"  : 2.750,
        "4"  : 2.900,
        "3"  : 3.250
    }

    PUE = 1.2
    if manufacturer == 'GlobalFoundries':
        ewf = 2.3
        WUE = 4.52
    elif manufacturer == 'TSMC':
        ewf = 1.4
        WUE = 7
    else:
        raise ValueError(f"Unsupported manufacturer {manufacturer}")
    
    area = device_specs['die_size_mm2'] / 100 # convert from mm^2 to cm^2
    upw = upw_by_node_liters_per_cm2[device_specs['technology_node']]
    epa = epa_by_node_kWh_per_cm2[device_specs['technology_node']]
    if device_specs['component_type'] in ['CPU', 'GPU']:
        embodied_water = area / production_yield * (upw + epa * WUE + epa * ewf * PUE) + packaging_water_per_IC_liters
        if device_specs['component_type'] == 'GPU':
            embodied_water += device_specs['hbm_capacity_GB'] * wpc_by_type_liters_per_GB['DRAM'] # add DRAM water for HBM, assuming 1x DRAM for HBM
            embodied_water += packaging_water_per_IC_liters * device_specs['hbm_capacity_GB'] / storage_reference_capacity_GB['DRAM'] # add packaging water for HBM, assuming 1 IC per storage_reference_capacity_GB
    elif device_specs['component_type'] in ['DRAM', 'SSD', 'HDD']:
        assert device_specs['component_type'] in wpc_by_type_liters_per_GB, f"Component type {device_specs['component_type']} not found in water data"
        embodied_water = device_specs['capacity'] * wpc_by_type_liters_per_GB[device_specs['component_type']] 
        if device_specs['component_type'] != 'HDD': 
            embodied_water += packaging_water_per_IC_liters * device_specs['capacity'] / storage_reference_capacity_GB[device_specs['component_type']]
    else:
        raise ValueError(f"Unsupported component type {device_specs['component_type']}")
    
    return embodied_water / 1000 # convert from liters to m3