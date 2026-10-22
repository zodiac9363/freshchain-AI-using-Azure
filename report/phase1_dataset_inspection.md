# FreshChain AI — Phase 1 Dataset Inspection & Schema Report
**Project**: Cold-Chain Quality Early Warning & Redistribution Prioritization  
**Author / Team**: FreshChain AI Data Engineering Team  
**Date**: October 2026 · Semester V IA3 Submission  

---

## 1. Executive Summary

Phase 1 establishes the reproducible local foundation for FreshChain AI. Both raw CSV datasets housed in `dataset/` were loaded, validated, and statistically profiled using a deterministic inspection pipeline (`src/inspect_datasets.py`).

| Dataset Description | Filename | Actual Rows | Actual Columns | Primary Key / Index | Missing Values | Duplicate Rows |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **IoT Produce Sensors** | `dataset/Dataset.csv` | **10,995** | **6** | Cross-sectional readings | **0 (0.0%)** | 2,880 (26.19%) |
| **UN FAO Food Loss & Waste** | `dataset/Data (1).csv` | **4,957** | **18** | Global survey observations | Multiple fields | 13 (0.26%) |

> [!IMPORTANT]
> **Schema & Label Verification**:
> 1. The IoT sensor dataset contains **10,995 rows**, exactly matching the project blueprint.
> 2. The quality target `Class` contains three distinct categorical values: `Good` ($n = 5,667$), `Bad` ($n = 4,617$), and `BAD` ($n = 711$). Merging `Bad` and `BAD` yields **5,328 spoilage instances (48.46%)** and **5,667 fresh instances (51.54%)**, establishing a healthy and nearly balanced binary target.
> 3. Column headers contain slight typographical variations from physical sensor export: `Light (Fux)` (intended as Lux / luminous flux) and `CO2 (pmm)` (intended as parts-per-million, ppm). These will be standardized during Phase 2 ETL.
> 4. Neither dataset contains native temporal timestamps. The sensor observations represent cross-sectional IoT point-in-time telemetry. In Phase 2, a reproducible synthetic batch generation strategy will be documented for cold-chain batch tracking.

---

## 2. IoT Sensor Dataset Detailed Schema & Profile (`Dataset.csv`)

### 2.1 Column Schema and Data Types
| Raw Column Name | Inferred Dtype | Standardized Target Name | Unit | Range / Cardinality | Description |
| :--- | :---: | :--- | :---: | :---: | :--- |
| `Fruit` | `object (str)` | `fruit` | Categorical | 4 unique | Fruit commodity category |
| `Temp` | `int64` | `temperature_c` | °C | $[21, 27]$ | Storage chamber ambient temperature |
| `Humid (%)` | `int64` | `humidity_pct` | % | $[71, 95]$ | Relative atmospheric humidity |
| `Light (Fux)` | `float64` | `light_lux` | Lux | $[4.22, 268.45]$ | Luminous flux exposure |
| `CO2 (pmm)` | `int64` | `co2_ppm` | ppm | $[20, 478]$ | Carbon dioxide gas concentration |
| `Class` | `object (str)` | `is_spoilage` | Binary | 3 raw $\rightarrow$ {0, 1} | Quality classification label |

### 2.2 Target Class Distribution
- **`Good`**: 5,667 records (51.54%) $\rightarrow$ Standardized to **0 (Fresh)**
- **`Bad`**: 4,617 records (41.99%) $\rightarrow$ Standardized to **1 (Spoiled / Quality Risk)**
- **`BAD`**: 711 records (6.47%) $\rightarrow$ Standardized to **1 (Spoiled / Quality Risk)**
- **Consolidated Target**: Fresh = 5,667 (51.54%), Spoiled = 5,328 (48.46%).

### 2.3 Fruit Distribution
- **Tomato**: 3,741 records (34.02%)
- **Orange**: 3,330 records (30.29%)
- **Pineapple**: 2,257 records (20.53%)
- **Banana**: 1,667 records (15.16%)

### 2.4 Numerical Feature Summary
| Feature | Min | Q1 (est.) | Median | Mean | Q3 (est.) | Max | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Temperature (°C)** | 21.0 | 23.0 | 24.0 | 23.84 | 25.0 | 27.0 | 1.23 |
| **Humidity (%)** | 71.0 | 93.0 | 95.0 | 93.53 | 95.0 | 95.0 | 3.00 |
| **Light Exposure (Lux)**| 4.22 | 8.78 | 12.91 | 24.93 | 16.38 | 268.45 | 48.02 |
| **$CO_2$ Concentration (ppm)** | 20.0 | 290.0 | 323.0 | 319.53 | 365.0 | 478.0 | 58.89 |

### 2.5 Duplicate Handling Policy for Phase 2
- **2,880 rows (26.19%)** share identical sensor tuples with at least one other row.
- **Physical Analysis**: IoT multi-sensor hubs frequently record identical discrete integer readings (e.g. constant $24^\circ\text{C}$ and $95\%$ humidity) across consecutive polling cycles.
- **Decision**: In Phase 2 ETL, duplicate rows are audited, quantified, and deduplicated for machine learning training to prevent data leakage, while raw ingestion preserves full audit telemetry.

---

## 3. UN FAO Food Loss & Waste Dataset Profile (`Data (1).csv`)

### 3.1 Overview
- **Total Records**: 4,957 records across **86 countries** covering historical reports from 2000 through modern surveillance.
- **Primary Metrics**: `loss_percentage` (numerical percentage of food lost), `loss_quantity`, `food_supply_stage`, `cause_of_loss`.
- **Mean Loss Percentage**: **11.15%** (Standard Deviation: 11.23%, Median: 7.00%, Range: $0.0\% - 100.0\%$).

### 3.2 Supply Chain Stage Distribution
1. **Whole supply chain**: 1,281 records
2. **Retail**: 740 records
3. **Farm**: 684 records
4. **Storage**: 576 records
5. **Wholesale**: 428 records
6. **Harvest**: 318 records
7. **Households**: 242 records
8. **Transport**: 216 records
9. **Processing**: 194 records
10. **Post-harvest / Trader / Other**: 278 records

### 3.3 Overlapping Commodities with IoT Sensor Dataset
- **Tomatoes**: High representation ($>250$ records) with post-harvest and retail loss distributions.
- **Bananas**: Prominently represented in transport and wholesale loss stages.
- **Citrus Fruits (Oranges)** and **Pineapples / Tropical Fruits**: Directly map to FAO sub-categories.

---

## 4. Phase 1 Deliverables & Sign-Off

1. **Virtual Environment**: `.venv` activated with Python 3.12.10, upgraded pip 26.2.2.
2. **Directory Structure**: All core directories initialized (`app`, `etl`, `ml`, `src`, `functions`, `tests`, `docker`, `report`, `.github/workflows`, `data/raw`, `data/processed`, `data/splits`).
3. **Dependency Manifest**: Initial `requirements.txt` generated with pinned versions.
4. **Automated Inspection Tool**: `src/inspect_datasets.py` operational, generating reproducible machine-readable (`report/phase1_dataset_inspection.json`) and academic-grade artifacts.

**Phase 1 Status**: **COMPLETE & VERIFIED**. Ready to proceed to **Phase 2 — ETL Pipeline**.
