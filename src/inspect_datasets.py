"""
FreshChain AI — Dataset Inspection & Schema Verification
Phase 1: Foundation and Data Preparation
"""

import os
import json
import pandas as pd
import numpy as np

DATASET_SENSOR_PATH = os.path.join("dataset", "Dataset.csv")
DATASET_FAO_PATH = os.path.join("dataset", "Data (1).csv")
OUTPUT_REPORT_PATH = os.path.join("report", "phase1_dataset_inspection.json")


def inspect_sensor_dataset(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sensor dataset not found at {path}")

    df = pd.read_csv(path)
    
    # Basic dimensions
    n_rows, n_cols = df.shape
    
    # Missing and duplicates
    missing = df.isnull().sum().to_dict()
    duplicates_count = int(df.duplicated().sum())
    
    # Column details
    col_details = {}
    for col in df.columns:
        col_details[col] = {
            "dtype": str(df[col].dtype),
            "sample_values": [str(x) for x in df[col].dropna().unique()[:5]],
            "n_unique": int(df[col].nunique()),
            "missing_count": int(df[col].isnull().sum())
        }
    
    # Fruit distribution
    fruit_counts = {str(k): int(v) for k, v in df['Fruit'].value_counts().to_dict().items()} if 'Fruit' in df.columns else {}
    
    # Class distribution (raw)
    class_counts = {str(k): int(v) for k, v in df['Class'].value_counts().to_dict().items()} if 'Class' in df.columns else {}
    
    # Numerical summaries
    num_summary = {}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        num_summary[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std())
        }
        
    # Check for timestamps / batch identifiers
    has_timestamp = any("time" in c.lower() or "date" in c.lower() for c in df.columns)
    has_batch_id = any("batch" in c.lower() or "lot" in c.lower() or "id" in c.lower() for c in df.columns)

    return {
        "file": path,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "column_details": col_details,
        "missing_values": missing,
        "duplicate_rows": duplicates_count,
        "fruit_counts": fruit_counts,
        "raw_class_counts": class_counts,
        "numerical_summary": num_summary,
        "has_timestamp": has_timestamp,
        "has_batch_id": has_batch_id
    }


def inspect_fao_dataset(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAO dataset not found at {path}")

    df = pd.read_csv(path, low_memory=False)
    
    n_rows, n_cols = df.shape
    missing = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
    duplicates_count = int(df.duplicated().sum())
    
    col_details = {}
    for col in df.columns:
        col_details[col] = {
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "missing_count": int(df[col].isnull().sum())
        }
    
    # Commodities
    top_commodities = {str(k): int(v) for k, v in df['commodity'].value_counts().head(10).to_dict().items()} if 'commodity' in df.columns else {}
    
    # Supply stages
    supply_stages = {str(k): int(v) for k, v in df['food_supply_stage'].value_counts().to_dict().items()} if 'food_supply_stage' in df.columns else {}
    
    # Loss percentage stats
    loss_stats = {}
    if 'loss_percentage' in df.columns:
        loss_numeric = pd.to_numeric(df['loss_percentage'], errors='coerce').dropna()
        loss_stats = {
            "valid_loss_count": int(len(loss_numeric)),
            "min": float(loss_numeric.min()),
            "max": float(loss_numeric.max()),
            "mean": float(loss_numeric.mean()),
            "median": float(loss_numeric.median()),
            "std": float(loss_numeric.std())
        }
        
    # Check for timestamps / years
    years = [int(y) for y in sorted(df['year'].dropna().unique().tolist())] if 'year' in df.columns else []

    return {
        "file": path,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "column_details": col_details,
        "missing_values": missing,
        "duplicate_rows": duplicates_count,
        "top_10_commodities": top_commodities,
        "supply_stages": supply_stages,
        "loss_percentage_stats": loss_stats,
        "years_covered": years,
        "n_countries": int(df['country'].nunique()) if 'country' in df.columns else 0
    }


def main():
    print("=" * 60)
    print("FreshChain AI — Phase 1 Dataset Inspection")
    print("=" * 60)
    
    sensor_res = inspect_sensor_dataset(DATASET_SENSOR_PATH)
    print(f"\n[+] IoT Sensor Dataset: {sensor_res['n_rows']} rows, {sensor_res['n_cols']} columns")
    print(f"    Columns: {sensor_res['columns']}")
    print(f"    Raw Class Distribution: {sensor_res['raw_class_counts']}")
    print(f"    Fruit Distribution: {sensor_res['fruit_counts']}")
    print(f"    Duplicates: {sensor_res['duplicate_rows']}")
    print(f"    Has Timestamps: {sensor_res['has_timestamp']}, Has Batch IDs: {sensor_res['has_batch_id']}")
    
    fao_res = inspect_fao_dataset(DATASET_FAO_PATH)
    print(f"\n[+] FAO Food Loss Dataset: {fao_res['n_rows']} rows, {fao_res['n_cols']} columns")
    print(f"    Countries: {fao_res['n_countries']}, Years: {min(fao_res['years_covered'])} - {max(fao_res['years_covered'])}")
    print(f"    Top Commodities: {list(fao_res['top_10_commodities'].keys())[:5]}")
    print(f"    Supply Chain Stages: {list(fao_res['supply_stages'].keys())}")
    print(f"    Loss Percentage Mean: {fao_res['loss_percentage_stats'].get('mean', 0.0):.2f}%")
    
    combined = {
        "phase": 1,
        "sensor_dataset": sensor_res,
        "fao_dataset": fao_res
    }
    
    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump(combined, f, indent=2)
        
    print(f"\n[OK] Inspection report saved to: {OUTPUT_REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
