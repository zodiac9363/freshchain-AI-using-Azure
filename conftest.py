"""
FreshChain AI — Pytest configuration
Ensures workspace root is in sys.path during test execution and guarantees model compatibility.
"""

import sys
import os
import pytest

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session", autouse=True)
def ensure_environment():
    """Guarantees trained model and splits exist and match active sklearn runtime."""
    model_path = os.path.join(PROJECT_ROOT, "ml", "saved_models", "freshchain_champion_pipeline.joblib")
    need_train = False
    if not os.path.exists(model_path):
        need_train = True
    else:
        try:
            import joblib
            bundle = joblib.load(model_path)
            pipe = bundle["pipeline"]
            import pandas as pd
            dummy = pd.DataFrame([{
                "fruit": "Orange",
                "temperature_c": 22.0,
                "humidity_pct": 88.0,
                "light_lux": 8.0,
                "co2_ppm": 290.0,
                "vpd_kpa": 0.5,
                "thi_index": 20.0,
                "co2_fruit_deviation": 4.0,
                "high_light_risk": 0
            }])
            pipe.predict(dummy)
        except Exception as e:
            print(f"[conftest] Model needs re-fitting for active runtime ({e}). Training champion...")
            need_train = True

    if need_train:
        from ml.train import train_and_compare_models
        train_and_compare_models()
