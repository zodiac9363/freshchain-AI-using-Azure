"""
FreshChain AI — ETL Stage 2: Validation
Enforces schema constraints, domain-specific ranges, and missing/corrupt checks.
Invalid records are explicitly counted, documented, and diagnosed.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

SENSOR_REQUIRED_COLUMNS = ["Fruit", "Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)", "Class"]
FAO_REQUIRED_COLUMNS = ["country", "commodity", "year", "loss_percentage", "food_supply_stage"]

VALID_FRUITS = {"Tomato", "Orange", "Pineapple", "Banana"}
VALID_SENSOR_CLASSES = {"Good", "Bad", "BAD"}

# Biophysical cold-chain domain ranges
SENSOR_RANGE_BOUNDS = {
    "Temp": (-5.0, 60.0),          # Physical cold chain storage bounds in Celsius
    "Humid (%)": (0.0, 100.0),       # Relative humidity percentage
    "Light (Fux)": (0.0, 2000.0),    # Luminous flux exposure (Lux)
    "CO2 (pmm)": (0.0, 5000.0)       # Respiration CO2 gas concentration (ppm)
}


def validate_sensor_dataset(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any], pd.DataFrame]:
    """
    Validates IoT sensor observations.
    Returns:
        (is_valid, report_dict, valid_df_mask)
    """
    report: Dict[str, Any] = {
        "dataset": "IoT_Sensor_Dataset",
        "total_records": len(df),
        "missing_columns": [],
        "null_counts": {},
        "range_violations": {},
        "invalid_fruit_labels": 0,
        "invalid_class_labels": 0,
        "duplicate_rows": int(df.duplicated().sum()),
        "invalid_records_count": 0,
        "valid_records_count": 0,
        "status": "PASSED"
    }

    # 1. Column presence check
    missing_cols = [c for c in SENSOR_REQUIRED_COLUMNS if c not in df.columns]
    report["missing_columns"] = missing_cols
    if missing_cols:
        report["status"] = "FAILED_SCHEMA"
        report["invalid_records_count"] = len(df)
        return False, report, pd.Series(False, index=df.index)

    # 2. Null values check
    null_counts = {c: int(df[c].isnull().sum()) for c in SENSOR_REQUIRED_COLUMNS}
    report["null_counts"] = null_counts

    valid_mask = pd.Series(True, index=df.index)
    for col, count in null_counts.items():
        if count > 0:
            valid_mask &= df[col].notnull()

    # 3. Valid Fruit check
    fruit_invalid_mask = ~df["Fruit"].astype(str).str.strip().isin(VALID_FRUITS)
    report["invalid_fruit_labels"] = int(fruit_invalid_mask.sum())
    valid_mask &= ~fruit_invalid_mask

    # 4. Valid Class check
    class_invalid_mask = ~df["Class"].astype(str).str.strip().isin(VALID_SENSOR_CLASSES)
    report["invalid_class_labels"] = int(class_invalid_mask.sum())
    valid_mask &= ~class_invalid_mask

    # 5. Numerical Range check
    for col, (min_val, max_val) in SENSOR_RANGE_BOUNDS.items():
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        nan_or_oob = numeric_series.isnull() | (numeric_series < min_val) | (numeric_series > max_val)
        violations = int(nan_or_oob.sum())
        report["range_violations"][col] = {
            "expected_range": [min_val, max_val],
            "violations_count": violations
        }
        if violations > 0:
            valid_mask &= ~nan_or_oob

    valid_count = int(valid_mask.sum())
    invalid_count = len(df) - valid_count
    report["valid_records_count"] = valid_count
    report["invalid_records_count"] = invalid_count

    if invalid_count > 0:
        report["status"] = "PASSED_WITH_WARNINGS"
    else:
        report["status"] = "PASSED"

    return (invalid_count == 0), report, valid_mask


def validate_fao_dataset(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any], pd.DataFrame]:
    """
    Validates FAO Food Loss dataset.
    Returns:
        (is_valid, report_dict, valid_df_mask)
    """
    report: Dict[str, Any] = {
        "dataset": "UN_FAO_Food_Loss_Dataset",
        "total_records": len(df),
        "missing_columns": [],
        "null_counts": {},
        "invalid_loss_percentages": 0,
        "invalid_years": 0,
        "duplicate_rows": int(df.duplicated().sum()),
        "valid_records_count": 0,
        "invalid_records_count": 0,
        "status": "PASSED"
    }

    missing_cols = [c for c in FAO_REQUIRED_COLUMNS if c not in df.columns]
    report["missing_columns"] = missing_cols
    if missing_cols:
        report["status"] = "FAILED_SCHEMA"
        report["invalid_records_count"] = len(df)
        return False, report, pd.Series(False, index=df.index)

    null_counts = {c: int(df[c].isnull().sum()) for c in FAO_REQUIRED_COLUMNS}
    report["null_counts"] = null_counts

    valid_mask = pd.Series(True, index=df.index)

    # Validate loss percentage
    loss_num = pd.to_numeric(df["loss_percentage"], errors="coerce")
    invalid_loss = loss_num.isnull() | (loss_num < 0.0) | (loss_num > 100.0)
    report["invalid_loss_percentages"] = int(invalid_loss.sum())
    valid_mask &= ~invalid_loss

    # Validate year
    year_num = pd.to_numeric(df["year"], errors="coerce")
    invalid_years = year_num.isnull() | (year_num < 1990) | (year_num > 2030)
    report["invalid_years"] = int(invalid_years.sum())
    valid_mask &= ~invalid_years

    # Check commodity and food_supply_stage not empty
    empty_commodity = df["commodity"].isnull() | (df["commodity"].astype(str).str.strip() == "")
    empty_stage = df["food_supply_stage"].isnull() | (df["food_supply_stage"].astype(str).str.strip() == "")
    valid_mask &= ~empty_commodity & ~empty_stage

    valid_count = int(valid_mask.sum())
    invalid_count = len(df) - valid_count
    report["valid_records_count"] = valid_count
    report["invalid_records_count"] = invalid_count

    if invalid_count > 0:
        report["status"] = "PASSED_WITH_WARNINGS"
    else:
        report["status"] = "PASSED"

    return (invalid_count == 0), report, valid_mask


if __name__ == "__main__":
    from etl.extract import extract_data
    s_raw, f_raw, _ = extract_data()
    s_ok, s_rep, _ = validate_sensor_dataset(s_raw)
    f_ok, f_rep, _ = validate_fao_dataset(f_raw)
    print(f"[VALIDATE] Sensor validation: {s_rep['status']} (Valid: {s_rep['valid_records_count']}, Invalid: {s_rep['invalid_records_count']})")
    print(f"[VALIDATE] FAO validation: {f_rep['status']} (Valid: {f_rep['valid_records_count']}, Invalid: {f_rep['invalid_records_count']})")
