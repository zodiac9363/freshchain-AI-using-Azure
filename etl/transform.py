"""
FreshChain AI — ETL Stage 3: Transformation & Feature Engineering
Standardizes labels, handles duplicates, and computes defensible biophysical features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


def calculate_vapor_pressure_deficit(temp_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    """
    Computes Vapor Pressure Deficit (VPD in kPa) using the Tetens equation for saturation vapor pressure.
    e_s(T) = 0.61078 * exp((17.27 * T) / (T + 237.3))
    VPD = e_s(T) * (1 - humidity_pct / 100)
    VPD directly determines transpirational moisture loss and shriveling rate in cold storage.
    """
    e_s = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    vpd = e_s * (1.0 - (humidity_pct / 100.0))
    return np.maximum(0.0, vpd)


def calculate_temp_humidity_index(temp_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    """
    Computes Temperature-Humidity Index (THI), a recognized psychrometric metric for thermal stress.
    THI = T - (0.55 - 0.0055 * H) * (T - 14.5)
    """
    thi = temp_c - (0.55 - 0.0055 * humidity_pct) * (temp_c - 14.5)
    return thi


def transform_sensor_data(
    df: pd.DataFrame,
    valid_mask: pd.Series,
    deduplicate_for_ml: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Transforms raw sensor dataset into standardized features with engineered biophysical indices.
    """
    # Filter to valid records
    clean = df[valid_mask].copy()
    initial_count = len(clean)

    # 1. Standardize column names
    rename_map = {
        "Fruit": "fruit",
        "Temp": "temperature_c",
        "Humid (%)": "humidity_pct",
        "Light (Fux)": "light_lux",
        "CO2 (pmm)": "co2_ppm",
        "Class": "raw_class"
    }
    clean = clean.rename(columns=rename_map)

    # 2. Standardize target label (Good -> 0, Bad/BAD -> 1)
    clean["raw_class"] = clean["raw_class"].astype(str).str.strip()
    clean["is_spoilage"] = clean["raw_class"].apply(lambda x: 1 if x.upper() == "BAD" else 0)

    # 3. Clean numeric types
    clean["temperature_c"] = clean["temperature_c"].astype(float)
    clean["humidity_pct"] = clean["humidity_pct"].astype(float)
    clean["light_lux"] = clean["light_lux"].astype(float)
    clean["co2_ppm"] = clean["co2_ppm"].astype(float)

    # 4. Biophysical Feature Engineering
    # A. Vapor Pressure Deficit (kPa)
    clean["vpd_kpa"] = calculate_vapor_pressure_deficit(clean["temperature_c"], clean["humidity_pct"]).round(4)

    # B. Temperature Humidity Index (THI)
    clean["thi_index"] = calculate_temp_humidity_index(clean["temperature_c"], clean["humidity_pct"]).round(4)

    # C. Commodity-specific CO2 baseline deviation (ppm)
    # Instead of assuming an arbitrary 400 ppm atmospheric baseline, calculate empirical median for each produce type
    fruit_median_co2 = clean.groupby("fruit")["co2_ppm"].transform("median")
    clean["co2_fruit_deviation"] = (clean["co2_ppm"] - fruit_median_co2).round(2)

    # D. High Light Stress Flag (>20 Lux indicates cold-room illumination or door seal breach)
    clean["high_light_risk"] = (clean["light_lux"] > 20.0).astype(int)

    # 5. Deterministic Batch Assignment (groups adjacent sensor readings into 50-reading lots)
    # Reproducible batch generator for cold-chain inventory management
    clean = clean.reset_index(drop=True)
    batch_size = 50
    clean["batch_id"] = clean.apply(
        lambda r: f"BATCH-{r['fruit'][:3].upper()}-{str(r.name // batch_size + 1).zfill(4)}",
        axis=1
    )

    duplicate_rows_count = int(clean.duplicated(subset=["fruit", "temperature_c", "humidity_pct", "light_lux", "co2_ppm"]).sum())
    
    if deduplicate_for_ml:
        clean = clean.drop_duplicates(subset=["fruit", "temperature_c", "humidity_pct", "light_lux", "co2_ppm"]).reset_index(drop=True)

    final_count = len(clean)

    report = {
        "initial_valid_records": initial_count,
        "final_transformed_records": final_count,
        "duplicates_removed": (initial_count - final_count) if deduplicate_for_ml else 0,
        "duplicate_rows_detected": duplicate_rows_count,
        "engineered_features": [
            {
                "name": "vpd_kpa",
                "unit": "kPa",
                "formula": "e_s(T) * (1 - H/100)",
                "description": "Vapor Pressure Deficit: physical force driving produce transpiration and moisture loss."
            },
            {
                "name": "thi_index",
                "unit": "dimensionless",
                "formula": "T - (0.55 - 0.0055*H)*(T - 14.5)",
                "description": "Temperature Humidity Index for psychrometric cold-storage thermal stress."
            },
            {
                "name": "co2_fruit_deviation",
                "unit": "ppm",
                "formula": "co2_ppm - median_fruit_co2",
                "description": "Commodity-specific respiration elevation above baseline storage equilibrium."
            },
            {
                "name": "high_light_risk",
                "unit": "binary flag {0,1}",
                "formula": "1 if light_lux > 20.0 else 0",
                "description": "Threshold flag for photolytic oxidation and chlorophyll degradation."
            },
            {
                "name": "batch_id",
                "unit": "identifier",
                "formula": "BATCH-{FRU}-{ID}",
                "description": "Synthetic reproducible batch ID for cold-chain lot simulation."
            }
        ],
        "class_balance": {
            "fresh_count": int((clean["is_spoilage"] == 0).sum()),
            "spoiled_count": int((clean["is_spoilage"] == 1).sum()),
            "spoilage_rate_pct": float(round((clean["is_spoilage"] == 1).mean() * 100, 2))
        }
    }

    return clean, report


def transform_fao_data(df: pd.DataFrame, valid_mask: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans and standardizes the UN FAO Food Loss dataset for risk analytics and loss stage benchmarks.
    """
    clean = df[valid_mask].copy()
    initial_count = len(clean)

    # 1. Select and rename core analytic columns
    core_cols = {
        "country": "country",
        "commodity": "commodity",
        "year": "year",
        "loss_percentage": "loss_percentage",
        "food_supply_stage": "supply_stage",
        "cause_of_loss": "cause_of_loss"
    }
    available_cols = {k: v for k, v in core_cols.items() if k in clean.columns}
    clean = clean[list(available_cols.keys())].rename(columns=available_cols)

    # 2. Standardize loss percentage numeric values
    clean["loss_percentage"] = pd.to_numeric(clean["loss_percentage"], errors="coerce").round(2)
    clean = clean.dropna(subset=["loss_percentage"])

    # 3. Standardize supply chain stages
    stage_mapping = {
        "Whole supply chain": "Whole Chain",
        "Retail": "Retail",
        "Farm": "Farm",
        "Storage": "Storage",
        "Wholesale": "Wholesale",
        "Harvest": "Harvest",
        "Households": "Consumer",
        "Transport": "Transport",
        "Processing": "Processing",
        "Post-harvest": "Post-Harvest",
        "Trader": "Wholesale/Trader",
        "Market": "Retail/Market",
        "Packing": "Packing",
        "Distribution": "Distribution"
    }
    clean["supply_stage_standardized"] = clean["supply_stage"].map(
        lambda s: stage_mapping.get(str(s).strip(), "Other")
    )

    # 4. Commodity name clean up
    clean["commodity_clean"] = clean["commodity"].astype(str).str.strip().str.title()

    # 5. Fill cause_of_loss missing with "Unspecified / Multi-factor"
    clean["cause_of_loss"] = clean["cause_of_loss"].fillna("Unspecified / Multi-factor").astype(str).str.strip()

    final_count = len(clean)

    report = {
        "initial_valid_records": initial_count,
        "final_transformed_records": final_count,
        "mean_loss_percentage": float(round(clean["loss_percentage"].mean(), 2)),
        "median_loss_percentage": float(round(clean["loss_percentage"].median(), 2)),
        "top_stages": clean["supply_stage_standardized"].value_counts().head(5).to_dict()
    }

    return clean, report


if __name__ == "__main__":
    from etl.extract import extract_data
    from etl.validate import validate_sensor_dataset, validate_fao_dataset
    
    s_raw, f_raw, _ = extract_data()
    _, s_rep, s_mask = validate_sensor_dataset(s_raw)
    _, f_rep, f_mask = validate_fao_dataset(f_raw)
    
    s_clean, s_tf_rep = transform_sensor_data(s_raw, s_mask)
    f_clean, f_tf_rep = transform_fao_data(f_raw, f_mask)
    
    print(f"[TRANSFORM] Sensor transformed: {s_clean.shape}, spoilage rate: {s_tf_rep['class_balance']['spoilage_rate_pct']}%")
    print(f"[TRANSFORM] FAO transformed: {f_clean.shape}, mean loss: {f_tf_rep['mean_loss_percentage']}%")
