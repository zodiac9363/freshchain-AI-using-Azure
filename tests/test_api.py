"""
FreshChain AI — Phase 7: Automated Integration Tests for Prediction API
Uses FastAPI TestClient to test health, single prediction, validation errors, and batch queue ranking.
"""

import pytest
from starlette.testclient import TestClient
from functions.local_api import app

client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "champion_model" in data


def test_api_single_prediction_valid():
    payload = {
        "fruit": "Tomato",
        "temperature_c": 24.5,
        "humidity_pct": 92.0,
        "light_lux": 14.0,
        "co2_ppm": 360.0,
        "batch_id": "BATCH-TOM-001",
        "quantity_kg": 1500.0,
        "price_per_kg": 2.80
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["fruit"] == "Tomato"
    assert data["batch_id"] == "BATCH-TOM-001"
    assert data["predicted_class"] in [0, 1]
    assert 0.0 <= data["spoilage_probability"] <= 1.0
    assert 0.0 <= data["urgency_score"] <= 100.0
    assert data["priority_tier"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    assert data["value_at_risk_usd"] >= 0.0
    assert "recommended_action" in data


def test_api_validation_error_invalid_temperature():
    # Out of bounds temperature (>60 C)
    payload = {
        "fruit": "Tomato",
        "temperature_c": 150.0,
        "humidity_pct": 92.0,
        "light_lux": 14.0,
        "co2_ppm": 360.0
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_api_validation_error_unsupported_fruit():
    # Unsupported fruit 'Apple'
    payload = {
        "fruit": "Apple",
        "temperature_c": 22.0,
        "humidity_pct": 90.0,
        "light_lux": 10.0,
        "co2_ppm": 320.0
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_api_batch_prediction_ranking():
    payload = {
        "readings": [
            {
                "fruit": "Orange",
                "temperature_c": 21.0,
                "humidity_pct": 88.0,
                "light_lux": 6.0,
                "co2_ppm": 240.0,
                "batch_id": "BATCH-ORA-GOOD"
            },
            {
                "fruit": "Tomato",
                "temperature_c": 26.5,
                "humidity_pct": 95.0,
                "light_lux": 18.0,
                "co2_ppm": 440.0,
                "batch_id": "BATCH-TOM-HOT"
            }
        ]
    }
    response = client.post("/api/batch_predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["total_batches"] == 2
    assert len(data["ranked_queue"]) == 2
    # Verify sorted descending by urgency_score
    q = data["ranked_queue"]
    assert q[0]["urgency_score"] >= q[1]["urgency_score"]
