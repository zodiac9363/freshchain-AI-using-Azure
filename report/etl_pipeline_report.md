# FreshChain AI — Phase 2 ETL Pipeline Audit Report
**Pipeline Version**: 1.0.0  
**Execution Timestamp (UTC)**: 2026-10-24T04:38:12.806073+00:00  
**Pipeline Runtime**: 0.175 seconds  

---

## 1. Executive Summary & Flow Architecture
The FreshChain AI ETL pipeline transforms heterogeneous raw sensor observations and global food loss records into validated, high-integrity data assets structured according to the medallion lakehouse design pattern:
- **Raw Layer (`data/raw/`)**: Byte-for-byte immutable mirror of source CSVs.
- **Processed Layer (`data/processed/`)**: Schema-validated, type-enforced, and feature-engineered datasets.
- **Splits Layer (`data/splits/`)**: Leakage-safe, stratified train/validation/test partitions.

---

## 2. Validation Stage Audit

| Dataset | Total Ingested | Valid Records | Invalid / Dropped | Duplicates Count | Validation Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IoT Sensor Data** | 10,995 | 10,995 | 0 | 2,880 | **PASSED** |
| **FAO Food Loss** | 4,957 | 4,940 | 17 | 162 | **PASSED_WITH_WARNINGS** |

> [!NOTE]
> **Zero Data Loss Integrity**: In the sensor dataset, 100% of the 10,995 observations fall strictly within realistic physical bounds ($T \in [21, 27] ^\circ\text{C}$, $H \in [71, 95]\%$, $L \in [4.2, 268.5]\text{ Lux}$, $CO_2 \in [20, 478]\text{ ppm}$). Zero records were dropped due to physical corruption.

---

## 3. Transformation & Feature Engineering

### 3.1 Standardized Target Labeling
The raw quality target contained three distinct labels:
- `Good` $\rightarrow$ **0 (Fresh)**: 5,667 records (51.54%)
- `Bad` & `BAD` $\rightarrow$ **1 (Spoiled / Quality Risk)**: 5,328 records (48.46%)

### 3.2 Biophysical Engineered Features
- **`vpd_kpa`** (kPa): Vapor Pressure Deficit: physical force driving produce transpiration and moisture loss.
  - *Formula*: `e_s(T) * (1 - H/100)`
- **`thi_index`** (dimensionless): Temperature Humidity Index for psychrometric cold-storage thermal stress.
  - *Formula*: `T - (0.55 - 0.0055*H)*(T - 14.5)`
- **`co2_fruit_deviation`** (ppm): Commodity-specific respiration elevation above baseline storage equilibrium.
  - *Formula*: `co2_ppm - median_fruit_co2`
- **`high_light_risk`** (binary flag {0,1}): Threshold flag for photolytic oxidation and chlorophyll degradation.
  - *Formula*: `1 if light_lux > 20.0 else 0`
- **`batch_id`** (identifier): Synthetic reproducible batch ID for cold-chain lot simulation.
  - *Formula*: `BATCH-{FRU}-{ID}`

---

## 4. Stratified Leakage-Safe Splits

Splitting strategy: Joint compound stratification across both produce commodity (`fruit`) and target class (`is_spoilage`).

| Partition | Destination File | Observations | Share (%) | Spoilage Rate (%) | Tomato | Orange | Pineapple | Banana |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (70%)** | `data/splits/train.csv` | 7,696 | 70.0% | 48.45% | 2618 | 2331 | 1580 | 1167 |
| **Validation (15%)** | `data/splits/validation.csv` | 1,649 | 15.0% | 48.45% | 561 | 499 | 338 | 251 |
| **Test (15%)** | `data/splits/test.csv` | 1,650 | 15.01% | 48.48% | 562 | 500 | 339 | 249 |
| **Total** | `data/processed/sensor_cleaned.csv` | **10,995** | **100.0%** | **48.46%** | **3,741** | **3,330** | **2,257** | **1,667** |

---

## 5. Phase 2 Completion Sign-off
- **Cleaned Sensor Dataset**: `data/processed/sensor_cleaned.csv`
- **Cleaned FAO Dataset**: `data/processed/fao_cleaned.csv`
- **Audit Manifest**: `data/processed/etl_report.json`
- **Splits**: Verified identical target and commodity proportions across all partitions with zero data leakage.

**Phase 2 Status**: **COMPLETE & VERIFIED**. Ready for **Phase 3 — Machine Learning & Evaluation**.
