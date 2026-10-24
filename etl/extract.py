"""
FreshChain AI — ETL Stage 1: Extraction
Reads local raw datasets or downloads from Azure Blob Storage if credentials are configured.
"""

import os
import shutil
import pandas as pd
from typing import Tuple, Dict, Any, Optional

DEFAULT_SENSOR_RAW_PATH = os.path.join("dataset", "Dataset.csv")
DEFAULT_FAO_RAW_PATH = os.path.join("dataset", "Data (1).csv")
DATA_RAW_DIR = os.path.join("data", "raw")


def extract_data(
    sensor_path: str = DEFAULT_SENSOR_RAW_PATH,
    fao_path: str = DEFAULT_FAO_RAW_PATH,
    blob_connection_string: Optional[str] = None,
    blob_container_name: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Extracts raw sensor and FAO datasets.
    If Azure Blob credentials are provided and azure-storage-blob is installed,
    attempts extraction from Azure Blob Storage with fallback to local files.
    """
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    metadata = {
        "source_type": "local_filesystem",
        "sensor_source_path": sensor_path,
        "fao_source_path": fao_path,
        "azure_blob_attempted": False,
        "azure_blob_success": False
    }

    # Azure Blob Storage check
    if blob_connection_string and blob_container_name:
        metadata["azure_blob_attempted"] = True
        try:
            from azure.storage.blob import BlobServiceClient
            client = BlobServiceClient.from_connection_string(blob_connection_string)
            container = client.get_container_client(blob_container_name)
            
            # Download sensor dataset
            sensor_blob = container.get_blob_client("sensor_raw.csv")
            dest_sensor = os.path.join(DATA_RAW_DIR, "sensor_raw.csv")
            with open(dest_sensor, "wb") as f:
                f.write(sensor_blob.download_blob().readall())
            sensor_path = dest_sensor
            
            # Download FAO dataset
            fao_blob = container.get_blob_client("fao_raw.csv")
            dest_fao = os.path.join(DATA_RAW_DIR, "fao_raw.csv")
            with open(dest_fao, "wb") as f:
                f.write(fao_blob.download_blob().readall())
            fao_path = dest_fao
            
            metadata["source_type"] = "azure_blob_storage"
            metadata["azure_blob_success"] = True
        except Exception as e:
            metadata["azure_blob_error"] = str(e)
            metadata["source_type"] = "local_filesystem_fallback"

    # Validate local files exist with fallback to data/raw mirror
    if not os.path.exists(sensor_path):
        alt_sensor = os.path.join(DATA_RAW_DIR, "sensor_raw.csv")
        if os.path.exists(alt_sensor):
            sensor_path = alt_sensor
        else:
            raise FileNotFoundError(f"Raw sensor dataset not found at {sensor_path} or {alt_sensor}")

    if not os.path.exists(fao_path):
        alt_fao = os.path.join(DATA_RAW_DIR, "fao_raw.csv")
        if os.path.exists(alt_fao):
            fao_path = alt_fao
        else:
            raise FileNotFoundError(f"Raw FAO dataset not found at {fao_path} or {alt_fao}")

    # Mirror to data/raw for standardized lake architecture
    local_sensor_raw = os.path.join(DATA_RAW_DIR, "sensor_raw.csv")
    local_fao_raw = os.path.join(DATA_RAW_DIR, "fao_raw.csv")
    if os.path.abspath(sensor_path) != os.path.abspath(local_sensor_raw):
        shutil.copyfile(sensor_path, local_sensor_raw)
    if os.path.abspath(fao_path) != os.path.abspath(local_fao_raw):
        shutil.copyfile(fao_path, local_fao_raw)

    sensor_df = pd.read_csv(local_sensor_raw)
    fao_df = pd.read_csv(local_fao_raw, low_memory=False)

    metadata["sensor_raw_shape"] = list(sensor_df.shape)
    metadata["fao_raw_shape"] = list(fao_df.shape)

    return sensor_df, fao_df, metadata


if __name__ == "__main__":
    s_df, f_df, meta = extract_data()
    print(f"[EXTRACT] Success: Sensor shape={s_df.shape}, FAO shape={f_df.shape}")
    print(f"[EXTRACT] Source metadata: {meta}")
