"""
FreshChain AI — Phase 6: Local & Docker REST API Service (FastAPI)
Exposes the same endpoints as Azure Functions for local development, automated testing, and Docker execution.
Includes auto-generated Swagger documentation at /docs.
"""

import os
import sys
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.schemas import SensorReading, PredictionResponse, BatchPredictionPayload, BatchPredictionResponse
from src.priority_score import evaluate_sensor_batch, rank_priority_queue
from ml.predict import FreshChainPredictor

app = FastAPI(
    title="FreshChain AI — Produce Quality & Spoilage Early Warning API",
    description="Serverless REST API providing real-time quality classification, biophysical Urgency Scoring (U), and Value at Risk (VaR) triage.",
    version="1.0.0"
)

# Enable CORS for Streamlit and external web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
try:
    predictor = FreshChainPredictor()
except Exception as e:
    predictor = None
    print(f"[API WARNING] Could not load champion model: {e}")


@app.get("/api/health", tags=["Monitoring"])
def health_check():
    """Returns service health status and champion model metadata."""
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model predictor not initialized")
    return {
        "status": "HEALTHY",
        "service": "FreshChain AI REST API",
        "champion_model": predictor.champion_name,
        "features_tracked": len(predictor.features_numeric) + len(predictor.features_categorical)
    }


@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single_batch(reading: SensorReading):
    """
    Evaluates a single produce sensor observation.
    Returns quality classification, spoilage probability P_s, Urgency Score U, and Value at Risk (VaR).
    """
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Champion model not available")
    try:
        return evaluate_sensor_batch(reading, predictor.predict)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/batch_predict", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_queue(payload: BatchPredictionPayload):
    """
    Evaluates multiple produce batches and returns an actionable priority queue sorted by Urgency Score.
    """
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Champion model not available")
    try:
        return rank_priority_queue(payload.readings, predictor.predict)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/etl_trigger", tags=["Pipeline"])
def trigger_cloud_etl():
    """Triggers end-to-end data lake ETL pipeline."""
    try:
        from etl.run_pipeline import run_etl_pipeline
        manifest = run_etl_pipeline()
        return {
            "status": "ETL_COMPLETED_SUCCESSFULLY",
            "execution_timestamp_utc": manifest["execution_timestamp_utc"],
            "records_processed": manifest["etl_audit_details"]["transform_sensor"]["final_transformed_records"]
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("functions.local_api:app", host="127.0.0.1", port=8000, reload=True)
