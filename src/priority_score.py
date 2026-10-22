"""
FreshChain AI — Phase 6: Multi-Variable Urgency Prioritization & Decision Engine
Calculates defensible stress factors, Urgency Score (U), and Value at Risk (VaR).

Mathematical Model:
  U = 100 * (w1 * P_s + w2 * S_T + w3 * S_CO2 + w4 * S_H)
  Weights: w1=0.50, w2=0.20, w3=0.20, w4=0.10
  All stress metrics normalized to [0.0, 1.0].

Financial Exposure:
  VaR = quantity_kg * price_per_kg * P_s (Estimated Exposure, not confirmed loss)
"""

import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.schemas import SensorReading, PredictionResponse, StressFactors, BatchPredictionResponse, PriorityTier


# Weight configuration defined in blueprint
W_SPOILAGE = 0.50
W_TEMP = 0.20
W_CO2 = 0.20
W_HUMID = 0.10


def compute_thermal_stress(temp_c: float) -> float:
    """
    Computes normalized thermal stress S_T in [0.0, 1.0].
    Produce storage baseline is ~18°C. Degradation accelerates towards 28°C.
    S_T = clip((T - 18.0) / (28.0 - 18.0), 0.0, 1.0)
    """
    s_t = (temp_c - 18.0) / 10.0
    return float(max(0.0, min(1.0, round(s_t, 4))))


def compute_co2_stress(co2_ppm: float) -> float:
    """
    Computes normalized respiration stress S_CO2 in [0.0, 1.0].
    Chamber baseline respiration is ~300 ppm. Elevated respiration caps at 480 ppm.
    S_CO2 = clip((CO2 - 300.0) / (480.0 - 300.0), 0.0, 1.0)
    """
    s_co2 = (co2_ppm - 300.0) / 180.0
    return float(max(0.0, min(1.0, round(s_co2, 4))))


def compute_humidity_stress(humidity_pct: float) -> float:
    """
    Computes normalized humidity stress S_H in [0.0, 1.0].
    Optimal produce relative humidity is 90%.
    Deviation towards dry (<70%) causes shriveling; condensation (>98%) causes mould.
    S_H = clip(abs(H - 90.0) / 20.0, 0.0, 1.0)
    """
    s_h = abs(humidity_pct - 90.0) / 20.0
    return float(max(0.0, min(1.0, round(s_h, 4))))


def compute_urgency_score(
    p_s: float,
    s_t: float,
    s_co2: float,
    s_h: float
) -> float:
    """
    Computes the final bounded Urgency Score U in [0.0, 100.0].
    U = 100 * (0.50 * P_s + 0.20 * S_T + 0.20 * S_CO2 + 0.10 * S_H)
    """
    u = 100.0 * (W_SPOILAGE * p_s + W_TEMP * s_t + W_CO2 * s_co2 + W_HUMID * s_h)
    return float(max(0.0, min(100.0, round(u, 2))))


def compute_value_at_risk(quantity_kg: float, price_per_kg: float, p_s: float) -> float:
    """
    Computes financial exposure in USD.
    VaR = quantity * price * P_s
    """
    var = quantity_kg * price_per_kg * p_s
    return float(round(var, 2))


def classify_priority(urgency_score: float) -> Tuple[PriorityTier, str]:
    """
    Maps Urgency Score U to triage priority tier and operational action.
    """
    if urgency_score >= 70.0:
        return "CRITICAL", "Immediate local clearance, flash discount, or industrial processing (within 6 hrs)"
    elif urgency_score >= 45.0:
        return "HIGH", "Reroute to nearest wholesale / retail market; increase cold-room circulation"
    elif urgency_score >= 20.0:
        return "MEDIUM", "Maintain standard cold-chain transit schedule; inspect at next waypoint"
    else:
        return "LOW", "Optimal fresh quality preserved; certified for premium long-haul delivery"


def evaluate_sensor_batch(
    reading: SensorReading,
    predictor_fn
) -> PredictionResponse:
    """
    Evaluates a single produce batch reading through the ML model and prioritization algorithm.
    """
    batch_dict = {
        "fruit": reading.fruit,
        "temperature_c": reading.temperature_c,
        "humidity_pct": reading.humidity_pct,
        "light_lux": reading.light_lux,
        "co2_ppm": reading.co2_ppm
    }

    pred_res = predictor_fn([batch_dict])[0]
    p_s = pred_res["spoilage_probability"]

    # Stress Factors
    s_t = compute_thermal_stress(reading.temperature_c)
    s_co2 = compute_co2_stress(reading.co2_ppm)
    s_h = compute_humidity_stress(reading.humidity_pct)

    u_score = compute_urgency_score(p_s, s_t, s_co2, s_h)
    tier, action = classify_priority(u_score)

    var_usd = compute_value_at_risk(
        quantity_kg=reading.quantity_kg or 1000.0,
        price_per_kg=reading.price_per_kg or 2.50,
        p_s=p_s
    )

    batch_label = reading.batch_id or f"BATCH-{reading.fruit[:3].upper()}-AUTO"

    return PredictionResponse(
        fruit=reading.fruit,
        batch_id=batch_label,
        predicted_class=pred_res["predicted_class"],
        predicted_label=pred_res["predicted_label"],
        spoilage_probability=p_s,
        urgency_score=u_score,
        priority_tier=tier,
        recommended_action=action,
        value_at_risk_usd=var_usd,
        stress_factors=StressFactors(
            thermal_stress_s_t=s_t,
            respiration_stress_s_co2=s_co2,
            humidity_stress_s_h=s_h,
            spoilage_prob_p_s=p_s
        ),
        model_used=pred_res.get("model_used", "FreshChain_Champion_Model"),
        timestamp_utc=datetime.now(timezone.utc).isoformat()
    )


def rank_priority_queue(
    readings: List[SensorReading],
    predictor_fn
) -> BatchPredictionResponse:
    """
    Evaluates multiple batches and returns a ranked priority queue sorted descending by Urgency Score.
    """
    results = [evaluate_sensor_batch(r, predictor_fn) for r in readings]
    results.sort(key=lambda x: x.urgency_score, reverse=True)

    critical_count = sum(1 for r in results if r.priority_tier == "CRITICAL")
    high_count = sum(1 for r in results if r.priority_tier == "HIGH")
    total_var = round(sum(r.value_at_risk_usd for r in results), 2)

    return BatchPredictionResponse(
        total_batches=len(results),
        critical_count=critical_count,
        high_count=high_count,
        total_value_at_risk_usd=total_var,
        ranked_queue=results
    )


if __name__ == "__main__":
    from ml.predict import FreshChainPredictor
    
    predictor = FreshChainPredictor()
    
    sample_1 = SensorReading(
        fruit="Tomato",
        temperature_c=26.0,
        humidity_pct=95.0,
        light_lux=16.5,
        co2_ppm=430.0,
        batch_id="BATCH-TOM-0101",
        quantity_kg=2500.0,
        price_per_kg=3.20
    )
    
    sample_2 = SensorReading(
        fruit="Orange",
        temperature_c=21.0,
        humidity_pct=88.0,
        light_lux=6.0,
        co2_ppm=250.0,
        batch_id="BATCH-ORA-0202",
        quantity_kg=5000.0,
        price_per_kg=1.80
    )

    batch_resp = rank_priority_queue([sample_1, sample_2], predictor.predict)
    print(f"[PRIORITY ENGINE] Evaluated {batch_resp.total_batches} batches:")
    for b in batch_resp.ranked_queue:
        print(f"  * {b.batch_id} ({b.fruit}) | U-Score: {b.urgency_score} [{b.priority_tier}] | VaR: ${b.value_at_risk_usd:,.2f} | Action: {b.recommended_action[:50]}...")
