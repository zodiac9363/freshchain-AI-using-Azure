"""
FreshChain AI — Phase 3: Machine Learning Training & Model Comparison
Trains Logistic Regression (Baseline), Random Forest, and Gradient Boosting.
Fits all preprocessing strictly on training data to prevent leakage.
Compares models on the validation split and selects the champion for held-out testing.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

TRAIN_SPLIT_PATH = os.path.join("data", "splits", "train.csv")
VAL_SPLIT_PATH = os.path.join("data", "splits", "validation.csv")
TEST_SPLIT_PATH = os.path.join("data", "splits", "test.csv")
MODEL_DIR = os.path.join("ml", "saved_models")
REPORT_DIR = os.path.join("report")

FEATURE_NUMERIC = ["temperature_c", "humidity_pct", "light_lux", "co2_ppm", "vpd_kpa", "thi_index", "co2_fruit_deviation", "high_light_risk"]
FEATURE_CATEGORICAL = ["fruit"]
TARGET_COL = "is_spoilage"


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer preprocessing pipeline.
    Applies StandardScaler to numerical biophysical features
    and OneHotEncoder with handle_unknown='ignore' to categorical fruit types.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURE_CATEGORICAL)
        ]
    )
    return preprocessor


def get_candidate_models() -> Dict[str, Any]:
    """
    Returns the three required candidate classification algorithms.
    """
    return {
        "Logistic_Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        ),
        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.9,
            random_state=42
        )
    }


def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Evaluates a trained model pipeline on a dataset partition and measures prediction latency.
    """
    t0 = time.time()
    y_pred = pipeline.predict(X)
    t_predict = time.time() - t0
    
    y_proba = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, "predict_proba") else None
    
    latency_ms_per_sample = (t_predict / len(X)) * 1000.0 if len(X) > 0 else 0.0
    
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "accuracy": float(round(accuracy_score(y, y_pred), 4)),
        "precision": float(round(precision_score(y, y_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y, y_pred, zero_division=0), 4)),
        "f1": float(round(f1_score(y, y_pred, zero_division=0), 4)),
        "roc_auc": float(round(roc_auc_score(y, y_proba), 4)) if y_proba is not None else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "latency_ms_per_sample": float(round(latency_ms_per_sample, 4))
    }
    return metrics


def train_and_compare_models() -> Dict[str, Any]:
    """
    Main training workflow:
    1. Loads train, val, and test splits.
    2. Trains 3 candidate models on train split.
    3. Evaluates candidates on validation split.
    4. Selects champion model based on validation F1-score.
    5. Evaluates champion on test split.
    6. Persists champion model pipeline.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("FreshChain AI — Phase 3 Machine Learning Model Comparison")
    print("=" * 70)

    # 1. Load splits
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    val_df = pd.read_csv(VAL_SPLIT_PATH)
    test_df = pd.read_csv(TEST_SPLIT_PATH)

    X_train = train_df[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y_train = train_df[TARGET_COL]

    X_val = val_df[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y_val = val_df[TARGET_COL]

    X_test = test_df[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y_test = test_df[TARGET_COL]

    print(f"Features: Numerical={FEATURE_NUMERIC}, Categorical={FEATURE_CATEGORICAL}")
    print(f"Datasets: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")

    candidates = get_candidate_models()
    comparison_results = {}
    trained_pipelines = {}

    # 2. Train and evaluate each model
    for name, clf in candidates.items():
        print(f"\n[Training] {name}...")
        preprocessor = build_preprocessor()
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        t_start = time.time()
        pipe.fit(X_train, y_train)
        train_time_sec = time.time() - t_start

        val_metrics = evaluate_model(pipe, X_val, y_val)
        val_metrics["training_time_sec"] = float(round(train_time_sec, 4))

        comparison_results[name] = val_metrics
        trained_pipelines[name] = pipe

        print(f"  -> Train Time: {train_time_sec:.3f}s | Latency: {val_metrics['latency_ms_per_sample']:.4f} ms/sample")
        print(f"  -> Validation Acc: {val_metrics['accuracy']:.4f} | Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f}")

    # 3. Model selection based on validation F1 score (prioritizing balanced spoilage detection)
    champion_name = max(comparison_results, key=lambda k: comparison_results[k]["f1"])
    champion_val_f1 = comparison_results[champion_name]["f1"]
    print(f"\n[Selection] Champion Model Selected: {champion_name} (Val F1: {champion_val_f1:.4f})")

    # 4. Final Evaluation on Held-Out Test Set
    champion_pipe = trained_pipelines[champion_name]
    test_metrics = evaluate_model(champion_pipe, X_test, y_test)
    print(f"\n[Final Held-Out Test Set Evaluation — {champion_name}]")
    print(f"  -> Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  -> Test Precision: {test_metrics['precision']:.4f}")
    print(f"  -> Test Recall:    {test_metrics['recall']:.4f}")
    print(f"  -> Test F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  -> Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  -> Confusion Matrix: TN={test_metrics['confusion_matrix']['tn']}, FP={test_metrics['confusion_matrix']['fp']}, FN={test_metrics['confusion_matrix']['fn']}, TP={test_metrics['confusion_matrix']['tp']}")

    # 5. Save champion pipeline
    model_save_path = os.path.join(MODEL_DIR, "freshchain_champion_pipeline.joblib")
    joblib.dump({
        "pipeline": champion_pipe,
        "champion_name": champion_name,
        "features_numeric": FEATURE_NUMERIC,
        "features_categorical": FEATURE_CATEGORICAL,
        "val_metrics": comparison_results[champion_name],
        "test_metrics": test_metrics
    }, model_save_path)
    print(f"\n[Persisted] Champion pipeline saved to: {model_save_path}")

    # 6. Save results JSON
    summary_report = {
        "candidate_validation_comparison": comparison_results,
        "champion_model": champion_name,
        "held_out_test_metrics": test_metrics,
        "features_used": {
            "numeric": FEATURE_NUMERIC,
            "categorical": FEATURE_CATEGORICAL
        }
    }
    results_json_path = os.path.join(REPORT_DIR, "model_comparison_results.json")
    with open(results_json_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"[Persisted] Model comparison results saved to: {results_json_path}")

    return summary_report


if __name__ == "__main__":
    train_and_compare_models()
