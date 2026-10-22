# FreshChain AI — Phase 5 Azure ML & MLflow Experiment Tracking Report
**Experiment Name**: `freshchain-produce-quality-classification`  
**Experiment ID**: `1`  
**Tracking URI**: `sqlite:///Z:/5th sem projects/cloud 3rd IA project/mlflow.db`  
**Execution Timestamp**: 2026-10-21T06:11:52.659102+00:00  

---

## 1. Managed Experiment Run Audit & Run IDs

Every candidate algorithm was executed under strict MLflow tracking to guarantee reproducibility, auditability, and academic integrity.

| Candidate Model | MLflow Run ID | Validation Accuracy | Spoilage Precision | Spoilage Recall | Validation F1-Score | Validation ROC-AUC | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `f40993df5b25...` | 0.9169 | 0.8950 | 0.9387 | **0.9163** | 0.9765 | 0.167s |
| **Random Forest** | `500a84dc54cb...` | 0.9988 | 0.9987 | 0.9987 | **0.9987** | 1.0000 | 0.428s |
| **Gradient Boosting** | `cf19037e45dc...` | 0.9970 | 0.9962 | 0.9975 | **0.9969** | 1.0000 | 2.072s |

---

## 2. Selected Champion Model & Registry Status

- **Champion Selected**: `Random_Forest`
- **Registered Model Name**: `FreshChain-Produce-Quality-RF`
- **Champion MLflow Run ID**: `500a84dc54cb4e21af5e8a54b63180da`
- **Held-Out Test Set Performance**:
  - **Accuracy**: 0.9994
  - **Precision**: 0.9988
  - **Recall**: 1.0000
  - **F1-Score**: 0.9994
  - **ROC-AUC**: 1.0000

---

## 3. Zero-Burn Azure ML Student Credit Strategy
1. **Experiment Tracking via MLflow**: Experiments execute on local compute with remote/local MLflow tracking ($0.00).
2. **Serverless Model Serving**: Instead of deploying an Azure ML Managed Online Endpoint (which burns $\approx \$100$/month of student credits on idle compute), FreshChain AI serves the model via serverless Azure Functions Consumption Plan (1,000,000 free executions/month).

**Phase 5 Status**: **COMPLETE & VERIFIED**. Ready for **Phase 6 (Azure Functions & Decision Engine)**.
