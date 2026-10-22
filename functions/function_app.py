"""
FreshChain AI — Phase 6: Azure Functions Serverless Application (Python v2 Model)
Provides serverless endpoints for real-time produce quality prediction and priority triage.

Endpoints:
  - GET  /api/health: System health and model registry status
  - POST /api/predict: Single produce batch inference & Urgency Score
  - POST /api/batch_predict: Multi-batch ranking & Priority Queue
  - POST /api/etl_trigger: Serverless cloud trigger to execute the ETL pipeline
"""

import os
import sys
import json
import logging
from typing import Dict, Any

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import azure.functions as func
    app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
except (ImportError, Exception):
    app = None

from src.schemas import SensorReading, BatchPredictionPayload
from src.priority_score import evaluate_sensor_batch, rank_priority_queue
from ml.predict import FreshChainPredictor

# Lazy loaded predictor
_PREDICTOR = None


def get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = FreshChainPredictor()
    return _PREDICTOR


# 1. Health Endpoint
if app is not None:
    @app.route(route="health", methods=["GET"])
    def health_check(req: func.HttpRequest) -> func.HttpResponse:
        logging.info("Health check requested.")
        try:
            pred = get_predictor()
            payload = {
                "status": "HEALTHY",
                "service": "FreshChain AI Serverless API",
                "champion_model": pred.champion_name,
                "environment": os.getenv("ENVIRONMENT", "cloud_production")
            }
            return func.HttpResponse(json.dumps(payload), mimetype="application/json", status_code=200)
        except Exception as e:
            return func.HttpResponse(json.dumps({"status": "DEGRADED", "error": str(e)}), mimetype="application/json", status_code=500)

    # 2. Single Batch Prediction Endpoint
    @app.route(route="predict", methods=["POST"])
    def predict_endpoint(req: func.HttpRequest) -> func.HttpResponse:
        logging.info("Prediction requested.")
        try:
            req_body = req.get_json()
            reading = SensorReading(**req_body)
            pred = get_predictor()
            response_obj = evaluate_sensor_batch(reading, pred.predict)
            return func.HttpResponse(response_obj.model_dump_json(), mimetype="application/json", status_code=200)
        except ValueError as val_err:
            return func.HttpResponse(json.dumps({"error": "Validation Error", "details": str(val_err)}), mimetype="application/json", status_code=422)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return func.HttpResponse(json.dumps({"error": "Internal Server Error", "details": str(e)}), mimetype="application/json", status_code=500)

    # 3. Multi-Batch Ranking Endpoint
    @app.route(route="batch_predict", methods=["POST"])
    def batch_predict_endpoint(req: func.HttpRequest) -> func.HttpResponse:
        logging.info("Batch priority queue evaluation requested.")
        try:
            req_body = req.get_json()
            payload = BatchPredictionPayload(**req_body)
            pred = get_predictor()
            batch_resp = rank_priority_queue(payload.readings, pred.predict)
            return func.HttpResponse(batch_resp.model_dump_json(), mimetype="application/json", status_code=200)
        except ValueError as val_err:
            return func.HttpResponse(json.dumps({"error": "Validation Error", "details": str(val_err)}), mimetype="application/json", status_code=422)
        except Exception as e:
            return func.HttpResponse(json.dumps({"error": "Internal Server Error", "details": str(e)}), mimetype="application/json", status_code=500)

    # 4. ETL Trigger Endpoint
    @app.route(route="etl_trigger", methods=["POST"])
    def etl_trigger_endpoint(req: func.HttpRequest) -> func.HttpResponse:
        logging.info("Cloud ETL execution triggered.")
        try:
            from etl.run_pipeline import run_etl_pipeline
            manifest = run_etl_pipeline()
            return func.HttpResponse(json.dumps({
                "status": "ETL_COMPLETED_SUCCESSFULLY",
                "execution_timestamp": manifest["execution_timestamp_utc"],
                "records_processed": manifest["etl_audit_details"]["transform_sensor"]["final_transformed_records"]
            }), mimetype="application/json", status_code=200)
        except Exception as e:
            return func.HttpResponse(json.dumps({"error": "ETL Pipeline Failed", "details": str(e)}), mimetype="application/json", status_code=500)
