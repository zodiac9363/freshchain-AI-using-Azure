# FreshChain AI — Complete IA3 Project Blueprint & Architecture
**Course**: Cloud Computing for Data Analytics (23DC2043) · B.Tech AI & Data Science · Semester V  
**Target**: **25 / 25 Marks** · Zero Out-of-Pocket Azure Student Credit Burn (<$0.50)

---

## 1. Executive Summary & Problem Context
FreshChain AI is an end-to-end cloud data engineering and managed machine learning platform designed for real-time cold-chain spoilage early warning and redistribution prioritization.

### Datasets Utilized (From `dataset/` directory)
1. **IoT Cold-Chain Sensor Dataset (`dataset/Dataset.csv`)**:
   - **10,995 observations** across 4 produce commodities (Tomato, Orange, Pineapple, Banana).
   - Features: Temperature (°C), Humidity (%), Light Exposure (Flux/Lux), $CO_2$ Concentration (ppm).
   - Quality Labels: `Good` (5,667), `Bad` (4,617), `BAD` (711) — standardizes to binary 0/1.
2. **United Nations FAO Food Loss and Waste Database (`dataset/Data (1).csv`)**:
   - **4,957 empirical records** across global supply chain stages (Farm, Harvest, Transport, Storage, Wholesale, Retail).
   - Establishes baseline loss distributions, loss causes, and economic Value at Risk (VaR).

---

## 2. Rubric Mapping (Target: 25 / 25)

| Assessment Criterion | Mark Weight | Required Excellence | FreshChain AI Implementation |
| :--- | :---: | :--- | :--- |
| **Problem Statement & Dataset with Cloud Visualization** | **5 Marks** | Real-time problem, cloud ingested dataset (Azure Blob), interactive dashboard. | Azure Blob Storage (`freshchain-raw/` $\rightarrow$ `processed/`), multi-tier lake with SAS token & manifest, Streamlit interactive app with live cloud telemetry. |
| **Cloud Architecture & Pipeline Implementation** | **10 Marks** | Docker containerization, serverless function (Azure Functions), ETL workflow, CI/CD (GitHub Actions). | End-to-end 5-stage ETL, Azure Functions v2 Python runtime (HTTP & Blob triggers), multi-stage Dockerfile, automated GitHub Actions pipeline. |
| **Managed Cloud ML & Result Analysis** | **5 Marks** | Managed ML service (Azure ML Studio), comparison of metrics, dynamic web app/notebook. | Azure ML Studio experiment tracking (MLflow), 3-way model comparison (Baseline vs Random Forest vs Gradient Boosting), model registered in Azure ML Registry, live inference UI. |
| **Technical Report** | **5 Marks** | <10% plagiarism, high quality figures, well-formatted tables, all references cited. | IEEE-format comprehensive report with Mermaid diagrams, mathematical models, confusion matrices, and academic citations. |

---

## 3. Azure for Students Zero-Burn Budget Architecture
To protect your $100 student credit, FreshChain AI uses exclusively free-tier or pennies-per-run services:
- **Azure Blob Storage**: 5 GB free on Azure for Students (our datasets consume <15 MB).
- **Azure Functions Consumption Plan**: 1 million free invocations per month ($0.00).
- **Azure ML Studio MLflow Tracking**: Basic workspace ($0.00). Experiments execute with remote tracking or 1-minute serverless CPU jobs (<$0.02).
- **No 24/7 Managed Endpoints**: Model inference is served via serverless Azure Functions rather than paid $100/mo Azure ML Managed Online Endpoints.
- **GitHub Actions**: 2,000 free runner minutes per month.

---

## 4. Repository Structure & Next Steps
- `etl/`: Data extraction, validation, cleaning, feature engineering, and loading.
- `ml/`: Training scripts, metric evaluation, Azure ML MLflow logger, and inference engine.
- `src/`: Multi-variable spoilage prioritization algorithm & FAO analytics.
- `functions/`: Serverless Azure Function v2 (`etl_trigger` and `prediction_api`).
- `app/`: 5-page interactive Streamlit dashboard.
- `docker/`: Multi-stage Docker container & compose files.
- `.github/workflows/`: Automated CI/CD workflow.
- `tests/`: Pytest unit and integration test suite.
- `report/`: Formal academic report for submission.
