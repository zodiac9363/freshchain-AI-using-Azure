"""
FreshChain AI — Phase 7: Automated Tests for Machine Learning Components
Validates model artifact loading, schema input robustness, inference output ranges, and latency.
"""

import os
import time
import pytest
from ml.predict import FreshChainPredictor


@pytest.fixture(scope="module")
def predictor():
    return FreshChainPredictor()


def test_model_loading(predictor):
    assert predictor.pipeline is not None
    assert predictor.champion_name in {"Random_Forest", "Gradient_Boosting", "Logistic_Regression"}
    assert len(predictor.features_numeric) > 0
    assert "fruit" in predictor.features_categorical


def test_model_inference_probability_range(predictor):
    sample = {
        "fruit": "Tomato",
        "temperature_c": 24.0,
        "humidity_pct": 92.0,
        "light_lux": 12.0,
        "co2_ppm": 340.0
    }
    results = predictor.predict(sample)
    assert len(results) == 1
    res = results[0]
    
    assert res["predicted_class"] in [0, 1]
    assert 0.0 <= res["spoilage_probability"] <= 1.0
    assert 0.0 <= res["freshness_probability"] <= 1.0
    assert abs(res["spoilage_probability"] + res["freshness_probability"] - 1.0) < 1e-4
    assert res["risk_level"] in ["LOW", "MODERATE", "HIGH"]


def test_model_auto_derives_engineered_features(predictor):
    # Pass only raw readings; engine should calculate vpd_kpa, thi_index, co2_fruit_deviation automatically
    sample = {
        "fruit": "Banana",
        "temperature_c": 26.5,
        "humidity_pct": 95.0,
        "light_lux": 28.0,
        "co2_ppm": 450.0
    }
    results = predictor.predict(sample)
    assert len(results) == 1
    assert results[0]["predicted_class"] in [0, 1]


def test_model_inference_latency(predictor):
    sample = {
        "fruit": "Orange",
        "temperature_c": 22.0,
        "humidity_pct": 88.0,
        "light_lux": 8.0,
        "co2_ppm": 290.0
    }
    t0 = time.time()
    for _ in range(50):
        predictor.predict(sample)
    elapsed = time.time() - t0
    avg_latency_ms = (elapsed / 50.0) * 1000.0
    # Production SLA: single-item inference latency under 200 ms
    assert avg_latency_ms < 200.0
