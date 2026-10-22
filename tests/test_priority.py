"""
FreshChain AI — Phase 7: Automated Tests for Urgency Prioritization Engine
Tests mathematical boundaries of Urgency Score U, stress factors, and Value at Risk.
"""

import pytest
from src.priority_score import (
    compute_thermal_stress,
    compute_co2_stress,
    compute_humidity_stress,
    compute_urgency_score,
    compute_value_at_risk,
    classify_priority
)


def test_stress_factor_bounds():
    # Low stress tests
    assert compute_thermal_stress(15.0) == 0.0
    assert compute_co2_stress(250.0) == 0.0
    assert compute_humidity_stress(90.0) == 0.0

    # High stress tests
    assert compute_thermal_stress(35.0) == 1.0
    assert compute_co2_stress(600.0) == 1.0
    assert compute_humidity_stress(65.0) == 1.0


def test_urgency_score_extreme_boundaries():
    # Complete freshness (P_s=0, S_T=0, S_CO2=0, S_H=0) -> U must be exactly 0.0
    u_min = compute_urgency_score(p_s=0.0, s_t=0.0, s_co2=0.0, s_h=0.0)
    assert u_min == 0.0
    tier_min, _ = classify_priority(u_min)
    assert tier_min == "LOW"

    # Extreme spoilage (P_s=1.0, S_T=1.0, S_CO2=1.0, S_H=1.0) -> U must be exactly 100.0
    u_max = compute_urgency_score(p_s=1.0, s_t=1.0, s_co2=1.0, s_h=1.0)
    assert u_max == 100.0
    tier_max, _ = classify_priority(u_max)
    assert tier_max == "CRITICAL"


def test_urgency_score_intermediate_weights():
    # Only spoilage is high (P_s=1.0, stress=0): U = 100 * (0.50 * 1.0) = 50.0
    u_mid = compute_urgency_score(p_s=1.0, s_t=0.0, s_co2=0.0, s_h=0.0)
    assert u_mid == 50.0
    tier, _ = classify_priority(u_mid)
    assert tier == "HIGH"


def test_value_at_risk_calculation():
    # 2,000 kg produce at $3.00/kg with 50% spoilage probability
    var = compute_value_at_risk(quantity_kg=2000.0, price_per_kg=3.00, p_s=0.50)
    assert var == 3000.00

    # Zero risk -> VaR = 0
    var_zero = compute_value_at_risk(quantity_kg=5000.0, price_per_kg=2.50, p_s=0.0)
    assert var_zero == 0.00
