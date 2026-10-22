"""
FreshChain AI — Enterprise Cold Chain Analytics & Early Warning Platform
Page 2: Batch Inspection & Live Sensor Evaluation
"""

import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.components.styles import CUSTOM_CSS, apply_plotly_theme, render_kpi_card, render_badge
from app.components.header import render_header
from ml.predict import FreshChainPredictor
from src.priority_score import (
    compute_thermal_stress,
    compute_co2_stress,
    compute_humidity_stress,
    compute_urgency_score,
    compute_value_at_risk,
    classify_priority
)
from etl.transform import calculate_vapor_pressure_deficit, calculate_temp_humidity_index

st.set_page_config(
    page_title="Batch Inspection — FreshChain AI",
    page_icon="🔬",
    layout="wide"
)

render_header("Single-Batch Quality Inspection", "Real-Time Sensor Evaluation, Urgency Scoring & Biophysical Decomposition")

# Load model predictor with caching
@st.cache_resource
def get_predictor():
    try:
        return FreshChainPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

predictor = get_predictor()

# ----------------- SIDEBAR CONTROLS & PRESETS -----------------
with st.sidebar:
    st.markdown("### 🎛️ Simulation Presets")
    preset = st.selectbox(
        "Select an operational scenario:",
        [
            "Custom Input",
            "Optimal Cold Storage (Safe)",
            "Chamber Respiration Spike (Tomato)",
            "Thermal Cooling Failure (Orange)",
            "High Humidity Mould Danger (Pineapple)"
        ]
    )
    
    # Preset definitions
    defaults = {
        "fruit": "Tomato",
        "temp": 20.0,
        "humid": 88.0,
        "light": 10.0,
        "co2": 320.0,
        "quantity": 500.0,
        "price": 1.50
    }
    
    if preset == "Optimal Cold Storage (Safe)":
        defaults = {"fruit": "Banana", "temp": 18.0, "humid": 90.0, "light": 8.0, "co2": 290.0, "quantity": 1200.0, "price": 0.95}
    elif preset == "Chamber Respiration Spike (Tomato)":
        defaults = {"fruit": "Tomato", "temp": 24.5, "humid": 93.0, "light": 15.0, "co2": 420.0, "quantity": 800.0, "price": 1.80}
    elif preset == "Thermal Cooling Failure (Orange)":
        defaults = {"fruit": "Orange", "temp": 28.5, "humid": 70.0, "light": 32.0, "co2": 395.0, "quantity": 1500.0, "price": 1.25}
    elif preset == "High Humidity Mould Danger (Pineapple)":
        defaults = {"fruit": "Pineapple", "temp": 22.0, "humid": 99.0, "light": 6.0, "co2": 350.0, "quantity": 600.0, "price": 2.20}

# ----------------- INPUT FORM -----------------
col_inputs, col_results = st.columns([1, 1.2])

with col_inputs:
    st.markdown('<div class="fc-card">', unsafe_allow_html=True)
    st.markdown("### 📦 Produce & Chamber Telemetry")
    
    fruit_list = ["Tomato", "Banana", "Orange", "Pineapple"]
    fruit_idx = fruit_list.index(defaults["fruit"]) if defaults["fruit"] in fruit_list else 0
    fruit = st.selectbox("Fruit Commodity", fruit_list, index=fruit_idx)
    
    c_temp, c_humid = st.columns(2)
    with c_temp:
        temp_c = st.slider("Chamber Temp (°C)", 10.0, 35.0, float(defaults["temp"]), 0.1, help="Produce storage temperature in Celsius")
    with c_humid:
        humid_pct = st.slider("Relative Humidity (%)", 30.0, 100.0, float(defaults["humid"]), 0.5, help="Chamber relative humidity percentage")
        
    c_light, c_co2 = st.columns(2)
    with c_light:
        light_lux = st.slider("Illuminance (Lux)", 0.0, 60.0, float(defaults["light"]), 1.0, help="Storage light intensity in Lux")
    with c_co2:
        co2_ppm = st.slider("Respiration CO₂ (ppm)", 150.0, 600.0, float(defaults["co2"]), 5.0, help="Respiration gas concentration in parts per million")
        
    st.markdown("---")
    st.markdown("### 💰 Financial Exposure Parameters")
    c_qty, c_price = st.columns(2)
    with c_qty:
        quantity_kg = st.number_input("Batch Quantity (kg)", min_value=1.0, max_value=50000.0, value=float(defaults["quantity"]), step=50.0)
    with c_price:
        price_per_kg = st.number_input("Wholesale Price ($/kg)", min_value=0.1, max_value=20.0, value=float(defaults["price"]), step=0.10)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- LIVE PREDICTION & SCORING -----------------
with col_results:
    if predictor is not None:
        sample = {
            "fruit": fruit,
            "temperature_c": temp_c,
            "humidity_pct": humid_pct,
            "light_lux": light_lux,
            "co2_ppm": co2_ppm
        }
        
        preds = predictor.predict(sample)
        pred = preds[0]
        
        p_s = pred["spoilage_probability"]
        predicted_class = pred["predicted_class"]
        
        # Calculate stress factors
        s_t = compute_thermal_stress(temp_c)
        s_co2 = compute_co2_stress(co2_ppm)
        s_h = compute_humidity_stress(humid_pct)
        u_score = compute_urgency_score(p_s, s_t, s_co2, s_h)
        tier, action_rec = classify_priority(u_score)
        var_usd = compute_value_at_risk(quantity_kg, price_per_kg, p_s)
        
        # Engineered biophysical features
        vpd_val = calculate_vapor_pressure_deficit(temp_c, humid_pct)
        thi_val = calculate_temp_humidity_index(temp_c, humid_pct)
        
        # Display Decision Card
        card_border = "#f43f5e" if predicted_class == 1 else "#10b981"
        card_bg = "rgba(244, 63, 94, 0.08)" if predicted_class == 1 else "rgba(16, 185, 129, 0.08)"
        class_text = "SPOILING / HIGH RISK (BAD)" if predicted_class == 1 else "FRESH / OPTIMAL (GOOD)"
        
        st.markdown(f"""
        <div class="fc-card" style="border-left: 5px solid {card_border}; background: {card_bg};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">
                        ML Quality Classification
                    </div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: {card_border};">
                        {class_text}
                    </div>
                </div>
                <div>
                    {render_badge(tier.name)}
                </div>
            </div>
            <div style="margin-top: 10px; font-size: 0.9rem; color: #cbd5e1;">
                <strong>Action Protocol:</strong> {action_rec}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 3 Score KPIs
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(render_kpi_card("Spoilage Prob (P_s)", f"{p_s * 100:.1f}%", "Champion Model Estimate", "#f43f5e" if p_s > 0.5 else "#10b981"), unsafe_allow_html=True)
        with r2:
            st.markdown(render_kpi_card("Urgency Score (U)", f"{u_score:.1f} / 100", f"Tier: {tier.name}", "#fbbf24" if u_score > 60 else "#38bdf8"), unsafe_allow_html=True)
        with r3:
            st.markdown(render_kpi_card("Value at Risk (VaR)", f"${var_usd:,.2f}", f"Batch: {quantity_kg:,.0f} kg", "#c084fc"), unsafe_allow_html=True)
            
        st.write("")
        
        # Radar & Decomposition Chart
        st.markdown('<div class="fc-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Multi-Variable Stress Factor Decomposition")
        
        stress_categories = ["Model Spoilage (Ps)", "Thermal Stress (S_T)", "Respiration Stress (S_CO2)", "Humidity Stress (S_H)"]
        stress_values = [p_s, s_t, s_co2, s_h]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=stress_values + [stress_values[0]],
            theta=stress_categories + [stress_categories[0]],
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.25)',
            line=dict(color='#38bdf8', width=2),
            name="Stress Profile"
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#94a3b8")),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#cbd5e1", size=11)),
                bgcolor="rgba(15,23,42,0.4)"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Biophysical diagnostics
        st.markdown(f"""
        <div style="display: flex; justify-content: space-around; background: rgba(30, 41, 59, 0.6); padding: 10px; border-radius: 8px; font-size: 0.85rem; color: #94a3b8;">
            <span>VPD: <strong style="color:#f8fafc;">{vpd_val:.3f} kPa</strong></span>
            <span>THI Index: <strong style="color:#f8fafc;">{thi_val:.2f}</strong></span>
            <span>CO₂ Delta: <strong style="color:#f8fafc;">{co2_ppm - 300.0:+.1f} ppm</strong></span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------- OPERATIONAL LIMITATIONS DISCLAIMER -----------------
st.markdown("""
<div class="fc-card" style="border: 1px solid rgba(245, 158, 11, 0.3); background: rgba(245, 158, 11, 0.05);">
    <div style="font-size: 0.95rem; font-weight: 700; color: #fbbf24; margin-bottom: 6px;">
        ⚠️ Operational Constraints & Real-World Decision Boundaries
    </div>
    <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
        • <strong>Experimental Baseline:</strong> The underlying classification models are trained on controlled laboratory chamber observations. In commercial cold-storage environments, natural air stratification, produce pack density, and sensor calibration drift require operational safety margins.<br>
        • <strong>Exposure vs Confirmed Loss:</strong> The calculated <em>Value at Risk (VaR)</em> represents financial exposure based on spoilage probability, not confirmed invoice loss. It must be paired with physical QA sampling before executing insurance write-offs.
    </div>
</div>
""", unsafe_allow_html=True)
