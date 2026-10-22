# FreshChain AI: IoT-Driven Cold-Chain Spoilage Early Warning & Triage Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Azure](https://img.shields.io/badge/Cloud-Azure%20Serverless-0078D4.svg)](https://azure.microsoft.com/)
[![Tests](https://img.shields.io/badge/Tests-18%2F18%20Passed-brightgreen.svg)](tests/)

FreshChain AI is an enterprise cloud data engineering, machine learning, and IoT telemetry platform designed for **cold-chain produce spoilage early warning, biophysical stress decomposition, and financial Value at Risk (VaR) redistribution triage**.

Built for **Cloud Computing Project Assessment 3 (IA3)**, this project demonstrates an end-to-end cloud pipeline covering:
1. **Medallion Data Lakehouse** via Azure Blob Storage (`raw` $\rightarrow$ `processed` $\rightarrow$ `analytics`).
2. **Leakage-Safe Machine Learning & Evaluation** with MLflow experiment tracking (Random Forest Champion: **100.0% Recall** on held-out test data).
3. **Serverless REST API & Urgency Prioritization Engine** via Azure Functions v2 & FastAPI.
4. **Interactive 5-Page Streamlit Intelligence Dashboard**.
5. **Zero-Credit-Burn Cloud Architecture** preserving student credit allowances.

---

## 🏛️ System Architecture

```
+---------------------------------------------------------------------------------------------------------+
|                                        FRESHCHAIN AI ARCHITECTURE                                       |
|                                                                                                         |
|  [ IoT Sensors ]               [ Azure Blob Storage ]              [ Medallion Lakehouse ]              |
|  • Temp, Humid, CO2, Light  -> • freshchain-raw        (Bronze) -> • Schema Validation & Cleaning       |
|                                • freshchain-processed  (Silver) -> • Biophysical Feature Engineering    |
|                                • freshchain-analytics  (Gold)   -> • Leakage-Safe Stratified Splits     |
|                                                                                |                        |
|                                                                                v                        |
|  [ Streamlit UI Dashboard ]    [ Serverless API / Functions ]      [ Managed ML & MLflow Registry ]     |
|  1. Executive Overview      <- • FastAPI / Azure Func v2        <- • Logistic Reg vs RF vs GBDT         |
|  2. Batch Inspection           • Urgency Engine: U in [0, 100]     • MLflow Run Lineage Tracking        |
|  3. Priority Queue             • Financial Exposure: VaR in $      • Champion: Random Forest (1.00 AUC) |
|  4. FAO Loss Analytics         • Sub-millisecond Inference SLA     • Joblib Artifact Bundle Serialization|
|  5. ML Model Evaluation                                                                                 |
+---------------------------------------------------------------------------------------------------------+
```

---

## 🚀 Key Features

* **Biophysical Feature Engineering**: Automatically derives Vapor Pressure Deficit ($VPD$ in kPa), Temperature-Humidity Index ($THI$), and Commodity Respiration Elevation ($\Delta\text{CO}_2$).
* **Empirical Multi-Variable Urgency Score ($U$)**:
  $$U = 100 \times \left(0.50 P_s + 0.20 S_T + 0.20 S_{\text{CO}_2} + 0.10 S_H\right)$$
  Classifies batches into `CRITICAL`, `HIGH`, `MEDIUM`, and `LOW` operational dispatch tiers.
* **Financial Risk Exposure ($\text{VaR}$)**: Calculates estimated monetary exposure ($ \text{VaR} = \text{kg} \times \text{Price} \times P_s $).
* **Dual-Target Cloud Storage**: Seamlessly switches between live Azure Blob Storage and local in-process lake emulation without requiring credentials.
* **5-Page Streamlit Dashboard**: Executive Overview, Single-Batch Inspection, Warehouse Priority Queue (with CSV export), Global FAO Food Loss Analytics, and ML Model Evaluation.
* **Production CI/CD**: Automated GitHub Actions workflow testing code, running lints, building Docker containers, and validating APIs.

---

## 📂 Project Repository Structure

```
├── app/
│   ├── main.py                          # Page 1: Executive Overview & Live Telemetry
│   ├── components/
│   │   ├── styles.py                    # Design System Tokens, CSS & Plotly Dark Theme
│   │   └── header.py                    # Telemetry Status Bar & Live Cloud Badge
│   └── pages/
│       ├── 1_Batch_Inspection.py         # Page 2: Single-Batch Testing & Scenarios
│       ├── 2_Priority_Queue.py          # Page 3: Warehouse Triage Queue & CSV Export
│       ├── 3_FAO_Loss_Analytics.py      # Page 4: Global FAO Food Loss Benchmarks
│       └── 4_ML_Evaluation.py           # Page 5: MLflow Benchmark & Held-Out Test Audit
├── dataset/
│   ├── Dataset.csv                      # IoT Produce Sensor Telemetry (10,995 rows)
│   └── Data (1).csv                     # United Nations FAO Food Loss Records (4,957 rows)
├── data/
│   ├── raw/                             # Immutable bronze source mirror
│   ├── processed/                       # Cleaned, standardized datasets & ETL report
│   ├── splits/                          # Leakage-safe 70/15/15 train/val/test splits
│   └── azure_blob_emulator/             # Local offline lake storage emulator
├── etl/
│   ├── extract.py                       # Dual-source data extraction (Local & Azure Blob)
│   ├── validate.py                      # Schema & physical range validator
│   ├── transform.py                     # Label standardizer & biophysical feature engine
│   ├── load.py                          # Multi-tier data loader & manifest generator
│   └── run_pipeline.py                  # End-to-end pipeline orchestrator
├── ml/
│   ├── train.py                         # Model training (Logistic Reg, RF, GBDT)
│   ├── evaluate.py                      # Metric comparison, confusion matrices, ROC curves
│   ├── predict.py                       # Standalone inference service
│   ├── azureml_run.py                   # Azure ML & MLflow experiment logger
│   └── saved_models/                    # Champion model joblib bundle
├── functions/
│   ├── function_app.py                  # Official Azure Functions v2 Python runtime
│   ├── local_api.py                     # High-performance FastAPI local mirror
│   └── requirements.txt                 # Serverless dependencies
├── docker/
│   ├── Dockerfile                       # Multi-stage production container
│   ├── docker-compose.yml               # Multi-service stack (API + Dashboard)
│   └── .dockerignore                    # Build optimization rules
├── tests/
│   ├── test_etl.py                      # Data validation & transformation tests
│   ├── test_ml.py                       # Model accuracy, latency & inference tests
│   ├── test_priority.py                 # Urgency Score & VaR boundary tests
│   └── test_api.py                      # REST endpoint HTTP contract tests
├── report/
│   ├── FreshChain_AI_Report.md          # Comprehensive Academic & Technical Report
│   └── model_comparison_results.json    # Verified benchmark metrics
├── .github/workflows/
│   └── ci_cd.yml                        # Automated GitHub Actions pipeline
├── requirements.txt                     # Pinned project dependencies
└── ARCHITECTURE.md                      # IA3 Rubric Mapping & Budget Architecture
```

---

## ⚡ Quick Start & Local Execution

### 1. Clone the Repository & Set Up Virtual Environment
```bash
git clone https://github.com/zodiac9363/freshchain-AI-using-Azure.git
cd freshchain-AI-using-Azure

python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the End-to-End ETL Pipeline
```bash
python etl/run_pipeline.py
```

### 3. Run the Automated Test Suite (18 Tests)
```bash
pytest -v
```

### 4. Launch the Streamlit Analytics Dashboard
```bash
streamlit run app/main.py --server.port 8501
```
*Access in browser at:* `http://localhost:8501`

### 5. Launch the REST Prediction API
```bash
uvicorn functions.local_api:app --host 0.0.0.0 --port 8000
```
*Interactive Swagger documentation:* `http://localhost:8000/docs`

---

## 🐳 Docker Deployment

Run both the REST API and Streamlit Dashboard using Docker Compose:
```bash
cd docker
docker-compose up --build
```
- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

---

## 🧪 Model Performance Summary

| Algorithm | Validation Accuracy | Spoilage Recall | Spoilage Precision | F1-Score | ROC-AUC | Inference Latency | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $91.69\%$ | $93.87\%$ | $89.50\%$ | $91.63\%$ | $0.9765$ | $0.0037\text{ ms}$ | Baseline |
| **Random Forest** | **99.88%** | **99.87%** | **99.87%** | **99.87%** | **1.0000** | $0.0356\text{ ms}$ | 🏆 **CHAMPION** |
| **Gradient Boosting** | $99.70\%$ | $99.75\%$ | $99.62\%$ | $99.69\%$ | $1.0000$ | $0.0076\text{ ms}$ | Candidate |

### Final Held-Out Test Set Verification (1,650 Unseen Samples)
* **Test Accuracy**: **$99.94\%$**
* **Test Spoilage Recall**: **100.00%** ($\text{TP}=800,\ \text{FN}=0$ — zero spoiled batches missed)
* **Test F1-Score**: **$99.94\%$**
* **Test ROC-AUC**: **$1.0000$**

---

## 📄 Academic Report
The complete 10-section technical report is available in [report/FreshChain_AI_Report.md](report/FreshChain_AI_Report.md).

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
