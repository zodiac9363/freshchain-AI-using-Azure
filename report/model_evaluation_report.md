# FreshChain AI — Phase 3 Machine Learning Evaluation Report
**Champion Model**: `Random_Forest`  
**Evaluation Protocol**: Stratified 70/15/15 train/val/test split. Preprocessors fit strictly on train split.

---

## 1. Candidate Algorithm Comparison (Validation Set)

| Candidate Model | Accuracy | Precision (Spoilage) | Recall (Spoilage) | F1-Score | ROC-AUC | Train Time (s) | Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9169 | 0.8950 | 0.9387 | **0.9163** | 0.9765 | 0.117 | 0.0030 |
| **Random Forest** **(Champion)** | 0.9988 | 0.9987 | 0.9987 | **0.9987** | 1.0000 | 0.201 | 0.0188 |
| **Gradient Boosting** | 0.9970 | 0.9962 | 0.9975 | **0.9969** | 1.0000 | 0.935 | 0.0042 |

---

## 2. Champion Model Performance on Held-Out Test Set (`Random_Forest`)

The champion model was evaluated **exactly once** on the pristine held-out test partition ($n = 1,650$ observations):

- **Test Accuracy**: **0.9994** (99.94%)
- **Test Precision (Spoilage)**: **0.9988**
- **Test Recall (Spoilage)**: **1.0000**
- **Test F1-Score**: **0.9994**
- **Test ROC-AUC**: **1.0000**
- **Prediction Latency**: **0.0195 ms** per inference call

### Confusion Matrix Breakdown (Test Set)
- **True Negatives (Fresh correctly flagged)**: 849
- **False Positives (Fresh falsely flagged as spoiled)**: 1
- **False Negatives (Spoiled missed - critical risk)**: 0
- **True Positives (Spoiled correctly detected)**: 800

---

## 3. Visual Artifacts Generated
- Confusion Matrix: `report/figures/confusion_matrix.png`
- ROC Curve: `report/figures/roc_curve.png`
- Candidate Comparison Barplot: `report/figures/model_comparison.png`

**Phase 3 Status**: **COMPLETE & VERIFIED**. Ready for **Phase 4 & 5 (Cloud Storage & Model Tracking)**.
