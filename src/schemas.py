"""
FreshChain AI — Phase 6: Pydantic Validation Schemas
Enforces strict domain boundaries, physical ranges, and schema validation.
"""

from typing import Optional, List, Dict, Literal
from pydantic import BaseModel, Field, field_validator


ProduceType = Literal["Tomato", "Orange", "Pineapple", "Banana"]
PriorityTier = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]


class SensorReading(BaseModel):
    fruit: ProduceType = Field(..., description="Produce commodity type")
    temperature_c: float = Field(..., ge=-10.0, le=60.0, description="Chamber temperature in Celsius")
    humidity_pct: float = Field(..., ge=0.0, le=100.0, description="Relative humidity percentage")
    light_lux: float = Field(..., ge=0.0, le=5000.0, description="Luminous flux in Lux")
    co2_ppm: float = Field(..., ge=0.0, le=5000.0, description="Carbon dioxide concentration in ppm")
    batch_id: Optional[str] = Field(default=None, description="Lot/batch tracking identifier")
    quantity_kg: Optional[float] = Field(default=1000.0, gt=0.0, description="Batch total weight in kilograms")
    price_per_kg: Optional[float] = Field(default=2.50, gt=0.0, description="Estimated price per kilogram in USD")

    @field_validator("fruit", mode="before")
    @classmethod
    def normalize_fruit(cls, v: str) -> str:
        if isinstance(v, str):
            clean = v.strip().capitalize()
            if clean in {"Tomato", "Orange", "Pineapple", "Banana"}:
                return clean
        raise ValueError(f"Invalid fruit '{v}'. Supported fruits: Tomato, Orange, Pineapple, Banana")


class StressFactors(BaseModel):
    thermal_stress_s_t: float = Field(..., ge=0.0, le=1.0)
    respiration_stress_s_co2: float = Field(..., ge=0.0, le=1.0)
    humidity_stress_s_h: float = Field(..., ge=0.0, le=1.0)
    spoilage_prob_p_s: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    fruit: str
    batch_id: str
    predicted_class: int
    predicted_label: str
    spoilage_probability: float
    urgency_score: float
    priority_tier: PriorityTier
    recommended_action: str
    value_at_risk_usd: float
    stress_factors: StressFactors
    model_used: str
    timestamp_utc: str


class BatchPredictionPayload(BaseModel):
    readings: List[SensorReading]


class BatchPredictionResponse(BaseModel):
    total_batches: int
    critical_count: int
    high_count: int
    total_value_at_risk_usd: float
    ranked_queue: List[PredictionResponse]
