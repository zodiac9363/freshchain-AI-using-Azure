"""
FreshChain AI — Phase 3: Standalone Model Inference Service
Loads the champion trained pipeline and predicts on fresh produce sensor observations.
Automatically derives biophysical features if only raw physical readings are provided.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from etl.transform import calculate_vapor_pressure_deficit, calculate_temp_humidity_index

DEFAULT_MODEL_PATH = os.path.join("ml", "saved_models", "freshchain_champion_pipeline.joblib")

# Empirical commodity baseline medians computed from dataset
FRUIT_MEDIAN_CO2 = {
    "Tomato": 344.0,
    "Orange": 286.0,
    "Pineapple": 331.0,
    "Banana": 298.0
}


class FreshChainPredictor:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Champion model bundle not found at {model_path}. Train model first via ml/train.py.")
        
        self.bundle = joblib.load(model_path)
        self.pipeline = self.bundle["pipeline"]
        self.champion_name = self.bundle["champion_name"]
        self.features_numeric = self.bundle["features_numeric"]
        self.features_categorical = self.bundle["features_categorical"]

    def _prepare_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures all required engineered features exist.
        If missing from raw input, computes them on the fly.
        """
        df = input_df.copy()

        # Map common alias column names
        rename_map = {
            "Fruit": "fruit",
            "Temp": "temperature_c",
            "Humid (%)": "humidity_pct",
            "Light (Fux)": "light_lux",
            "CO2 (pmm)": "co2_ppm"
        }
        df = df.rename(columns=rename_map)

        # Compute engineered features if missing
        if "vpd_kpa" not in df.columns:
            df["vpd_kpa"] = calculate_vapor_pressure_deficit(df["temperature_c"], df["humidity_pct"]).round(4)

        if "thi_index" not in df.columns:
            df["thi_index"] = calculate_temp_humidity_index(df["temperature_c"], df["humidity_pct"]).round(4)

        if "co2_fruit_deviation" not in df.columns:
            median_co2 = df["fruit"].map(FRUIT_MEDIAN_CO2).fillna(320.0)
            df["co2_fruit_deviation"] = (df["co2_ppm"] - median_co2).round(2)

        if "high_light_risk" not in df.columns:
            df["high_light_risk"] = (df["light_lux"] > 20.0).astype(int)

        feature_cols = self.features_numeric + self.features_categorical
        return df[feature_cols]

    def predict(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Performs inference on input dictionary, list of dictionaries, or DataFrame.
        Returns:
            List of prediction results with class, label, probability, and risk level.
        """
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame(input_data)
        else:
            input_df = input_data.copy()

        prepared_df = self._prepare_features(input_df)

        preds = self.pipeline.predict(prepared_df)
        probas = self.pipeline.predict_proba(prepared_df)[:, 1] if hasattr(self.pipeline, "predict_proba") else [None] * len(preds)

        results = []
        for i in range(len(prepared_df)):
            pred_class = int(preds[i])
            spoilage_prob = float(round(probas[i], 4)) if probas[i] is not None else (1.0 if pred_class == 1 else 0.0)
            
            # Risk status categorization
            if spoilage_prob < 0.30:
                risk_level = "LOW"
            elif spoilage_prob < 0.70:
                risk_level = "MODERATE"
            else:
                risk_level = "HIGH"

            results.append({
                "fruit": str(prepared_df.iloc[i]["fruit"]),
                "predicted_class": pred_class,
                "predicted_label": "Spoiled / Quality Risk" if pred_class == 1 else "Fresh / Quality OK",
                "spoilage_probability": spoilage_prob,
                "freshness_probability": float(round(1.0 - spoilage_prob, 4)),
                "risk_level": risk_level,
                "model_used": self.champion_name
            })

        return results


def run_prediction_demo():
    print("=" * 70)
    print("FreshChain AI — Phase 3 Standalone Inference Demonstration")
    print("=" * 70)

    predictor = FreshChainPredictor()
    print(f"Loaded Champion Model: {predictor.champion_name}")

    # Test cases: One fresh profile and one high-stress spoiled profile
    test_samples = [
        {
            "fruit": "Tomato",
            "temperature_c": 22.0,
            "humidity_pct": 90.0,
            "light_lux": 6.5,
            "co2_ppm": 310.0
        },
        {
            "fruit": "Tomato",
            "temperature_c": 26.0,
            "humidity_pct": 95.0,
            "light_lux": 16.5,
            "co2_ppm": 420.0
        },
        {
            "fruit": "Orange",
            "temperature_c": 21.0,
            "humidity_pct": 85.0,
            "light_lux": 5.0,
            "co2_ppm": 240.0
        },
        {
            "fruit": "Banana",
            "temperature_c": 25.5,
            "humidity_pct": 94.0,
            "light_lux": 25.0,
            "co2_ppm": 395.0
        }
    ]

    print("\nRunning inference on 4 fresh sample inputs without retraining:")
    predictions = predictor.predict(test_samples)

    for i, res in enumerate(predictions, 1):
        sample = test_samples[i - 1]
        print(f"\n[Sample {i}] Produce: {sample['fruit']} | Temp: {sample['temperature_c']}°C | Humid: {sample['humidity_pct']}% | CO2: {sample['co2_ppm']} ppm")
        print(f"  -> Predicted Quality: {res['predicted_label']}")
        print(f"  -> Spoilage Prob P_s: {res['spoilage_probability'] * 100:.1f}% (Risk: {res['risk_level']})")
        print(f"  -> Model Identifier:  {res['model_used']}")

    print("\n[OK] Standalone inference completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_prediction_demo()
