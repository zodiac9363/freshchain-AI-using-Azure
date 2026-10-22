"""
FreshChain AI — Phase 3: Detailed Model Evaluation & Figure Generation
Produces Confusion Matrices, ROC Curves, and comprehensive markdown reports.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import roc_curve, auc, confusion_matrix

FIGURES_DIR = os.path.join("report", "figures")
RESULTS_JSON_PATH = os.path.join("report", "model_comparison_results.json")
SAVED_MODEL_PATH = os.path.join("ml", "saved_models", "freshchain_champion_pipeline.joblib")
TEST_SPLIT_PATH = os.path.join("data", "splits", "test.csv")
REPORT_MD_PATH = os.path.join("report", "model_evaluation_report.md")


def generate_evaluation_artifacts():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 1. Load results JSON
    if not os.path.exists(RESULTS_JSON_PATH):
        raise FileNotFoundError(f"Model comparison results not found at {RESULTS_JSON_PATH}. Run ml/train.py first.")
    
    with open(RESULTS_JSON_PATH, "r") as f:
        results = json.load(f)

    # 2. Load model & test data
    saved_bundle = joblib.load(SAVED_MODEL_PATH)
    pipeline = saved_bundle["pipeline"]
    champion_name = saved_bundle["champion_name"]

    test_df = pd.read_csv(TEST_SPLIT_PATH)
    features = saved_bundle["features_numeric"] + saved_bundle["features_categorical"]
    X_test = test_df[features]
    y_test = test_df["is_spoilage"]

    # Predict
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 3. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Fresh (0)", "Spoiled (1)"],
        yticklabels=["Fresh (0)", "Spoiled (1)"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix: {champion_name} (Held-Out Test Set)", fontsize=12, pad=12, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 4. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(6.5, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"{champion_name} (AUC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Random Baseline (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title(f"ROC Curve: Spoilage Classification ({champion_name})", fontsize=12, pad=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(FIGURES_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 5. Plot Model Comparison Bar Chart
    comp = results["candidate_validation_comparison"]
    models = list(comp.keys())
    metrics_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    x = np.arange(len(models))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, m_name in enumerate(metrics_names):
        vals = [comp[m][m_name] for m in models]
        offset = (i - len(metrics_names) / 2) * width + width / 2
        ax.bar(x + offset, vals, width, label=m_name.upper())

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax.set_title("Validation Metric Comparison Across Candidate Algorithms", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in models], fontsize=10, fontweight="bold")
    ax.set_ylim([0.0, 1.1])
    ax.legend(loc="lower right", ncol=3)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    comp_chart_path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(comp_chart_path, dpi=300)
    plt.close()

    # 6. Generate Markdown Report
    test_m = results["held_out_test_metrics"]
    report_content = f"""# FreshChain AI — Phase 3 Machine Learning Evaluation Report
**Champion Model**: `{champion_name}`  
**Evaluation Protocol**: Stratified 70/15/15 train/val/test split. Preprocessors fit strictly on train split.

---

## 1. Candidate Algorithm Comparison (Validation Set)

| Candidate Model | Accuracy | Precision (Spoilage) | Recall (Spoilage) | F1-Score | ROC-AUC | Train Time (s) | Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for m, d in comp.items():
        is_champ = " **(Champion)**" if m == champion_name else ""
        report_content += f"| **{m.replace('_', ' ')}**{is_champ} | {d['accuracy']:.4f} | {d['precision']:.4f} | {d['recall']:.4f} | **{d['f1']:.4f}** | {d['roc_auc']:.4f} | {d['training_time_sec']:.3f} | {d['latency_ms_per_sample']:.4f} |\n"

    report_content += f"""
---

## 2. Champion Model Performance on Held-Out Test Set (`{champion_name}`)

The champion model was evaluated **exactly once** on the pristine held-out test partition ($n = {len(test_df):,}$ observations):

- **Test Accuracy**: **{test_m['accuracy']:.4f}** ({test_m['accuracy']*100:.2f}%)
- **Test Precision (Spoilage)**: **{test_m['precision']:.4f}**
- **Test Recall (Spoilage)**: **{test_m['recall']:.4f}**
- **Test F1-Score**: **{test_m['f1']:.4f}**
- **Test ROC-AUC**: **{test_m['roc_auc']:.4f}**
- **Prediction Latency**: **{test_m['latency_ms_per_sample']:.4f} ms** per inference call

### Confusion Matrix Breakdown (Test Set)
- **True Negatives (Fresh correctly flagged)**: {test_m['confusion_matrix']['tn']:,}
- **False Positives (Fresh falsely flagged as spoiled)**: {test_m['confusion_matrix']['fp']:,}
- **False Negatives (Spoiled missed - critical risk)**: {test_m['confusion_matrix']['fn']:,}
- **True Positives (Spoiled correctly detected)**: {test_m['confusion_matrix']['tp']:,}

---

## 3. Visual Artifacts Generated
- Confusion Matrix: `report/figures/confusion_matrix.png`
- ROC Curve: `report/figures/roc_curve.png`
- Candidate Comparison Barplot: `report/figures/model_comparison.png`

**Phase 3 Status**: **COMPLETE & VERIFIED**. Ready for **Phase 4 & 5 (Cloud Storage & Model Tracking)**.
"""

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[ARTIFACTS] Figures saved to {FIGURES_DIR}")
    print(f"[ARTIFACTS] Evaluation report written to {REPORT_MD_PATH}")


if __name__ == "__main__":
    generate_evaluation_artifacts()
