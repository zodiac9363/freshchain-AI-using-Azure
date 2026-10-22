"""
FreshChain AI — ETL Stage 4: Loading & Data Lake Export
Persists cleaned datasets, leakage-safe splits, and audit manifests.
Supports local data lake structure with Azure Blob Storage sync capabilities.
"""

import os
import json
import pandas as pd
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Optional

PROCESSED_DIR = os.path.join("data", "processed")
SPLITS_DIR = os.path.join("data", "splits")


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Creates leakage-safe stratified splits on both fruit category and spoilage target.
    Stratification key: 'strata' = fruit + '_' + is_spoilage
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    # Create compound strata for joint balance
    df_copy = df.copy()
    strata = df_copy["fruit"].astype(str) + "_" + df_copy["is_spoilage"].astype(str)
    
    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df_copy,
        test_size=val_test_ratio,
        stratify=strata,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_share_of_val_test = val_ratio / val_test_ratio
    val_strata = val_test_df["fruit"].astype(str) + "_" + val_test_df["is_spoilage"].astype(str)
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=(1.0 - val_share_of_val_test),
        stratify=val_strata,
        random_state=random_state
    )
    
    return {
        "train": train_df.reset_index(drop=True),
        "validation": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True)
    }


def save_processed_datasets(
    sensor_cleaned: pd.DataFrame,
    fao_cleaned: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    etl_report: Dict[str, Any],
    blob_connection_string: Optional[str] = None,
    processed_container_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Saves clean datasets, train/val/test splits, and comprehensive audit manifest.
    Optionally syncs to Azure Blob Storage if configured.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # 1. Save cleaned datasets
    sensor_path = os.path.join(PROCESSED_DIR, "sensor_cleaned.csv")
    fao_path = os.path.join(PROCESSED_DIR, "fao_cleaned.csv")
    sensor_cleaned.to_csv(sensor_path, index=False)
    fao_cleaned.to_csv(fao_path, index=False)

    # 2. Save splits
    split_paths = {}
    split_summary = {}
    for name, split_df in splits.items():
        path = os.path.join(SPLITS_DIR, f"{name}.csv")
        split_df.to_csv(path, index=False)
        split_paths[name] = path
        split_summary[name] = {
            "rows": len(split_df),
            "percentage": round(len(split_df) / len(sensor_cleaned) * 100, 2),
            "spoilage_rate_pct": round(float((split_df["is_spoilage"] == 1).mean() * 100), 2),
            "fruit_breakdown": split_df["fruit"].value_counts().to_dict()
        }

    # 3. Compile full ETL audit manifest
    manifest = {
        "pipeline_version": "1.0.0",
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "files_generated": {
            "sensor_cleaned": sensor_path,
            "fao_cleaned": fao_path,
            "splits": split_paths
        },
        "splits_summary": split_summary,
        "etl_audit_details": etl_report,
        "azure_blob_sync": {
            "attempted": False,
            "success": False
        }
    }

    # Optional Azure Blob sync
    if blob_connection_string and processed_container_name:
        manifest["azure_blob_sync"]["attempted"] = True
        try:
            from azure.storage.blob import BlobServiceClient
            client = BlobServiceClient.from_connection_string(blob_connection_string)
            container = client.get_container_client(processed_container_name)
            
            # Upload files
            for local_file, blob_name in [
                (sensor_path, "sensor_cleaned.csv"),
                (fao_path, "fao_cleaned.csv")
            ]:
                with open(local_file, "rb") as data:
                    container.upload_blob(name=blob_name, data=data, overwrite=True)
            manifest["azure_blob_sync"]["success"] = True
        except Exception as e:
            manifest["azure_blob_sync"]["error"] = str(e)

    # Save manifest JSON
    manifest_path = os.path.join(PROCESSED_DIR, "etl_report.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    from etl.extract import extract_data
    from etl.validate import validate_sensor_dataset, validate_fao_dataset
    from etl.transform import transform_sensor_data, transform_fao_data
    
    s_raw, f_raw, ext_meta = extract_data()
    _, s_val, s_mask = validate_sensor_dataset(s_raw)
    _, f_val, f_mask = validate_fao_dataset(f_raw)
    
    s_clean, s_tf = transform_sensor_data(s_raw, s_mask)
    f_clean, f_tf = transform_fao_data(f_raw, f_mask)
    
    splits = create_stratified_splits(s_clean)
    
    combined_report = {
        "extract": ext_meta,
        "validate_sensor": s_val,
        "validate_fao": f_val,
        "transform_sensor": s_tf,
        "transform_fao": f_tf
    }
    
    manifest = save_processed_datasets(s_clean, f_clean, splits, combined_report)
    print(f"[LOAD] Successfully loaded all processed outputs and manifest: {manifest['files_generated']}")
