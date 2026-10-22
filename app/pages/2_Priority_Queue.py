"""
FreshChain AI — Enterprise Cold Chain Analytics & Early Warning Platform
Page 3: Warehouse Triage Priority Queue & Financial Risk Optimization
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.components.styles import CUSTOM_CSS, apply_plotly_theme, render_kpi_card, render_badge
from app.components.header import render_header
from src.priority_score import (
    compute_thermal_stress,
    compute_co2_stress,
    compute_humidity_stress,
    compute_urgency_score,
    compute_value_at_risk,
    classify_priority
)
from ml.predict import FreshChainPredictor

st.set_page_config(
    page_title="Priority Queue — FreshChain AI",
    page_icon="⚡",
    layout="wide"
)

render_header("Warehouse Triage Priority Queue", "Dynamic Urgency Ranking, Expiry Mitigation & Financial Exposure (VaR) Management")

# Wholesale baseline prices ($/kg)
COMMODITY_PRICES = {
    "Tomato": 1.75,
    "Banana": 0.95,
    "Orange": 1.40,
    "Pineapple": 2.20
}

@st.cache_data
def generate_inventory_queue():
    """Generates an operational warehouse inventory batch queue from processed telemetry."""
    sensor_path = os.path.join(PROJECT_ROOT, "data", "processed", "sensor_cleaned.csv")
    if not os.path.exists(sensor_path):
        return pd.DataFrame()
        
    df = pd.read_csv(sensor_path)
    
    # Sample 100 representative warehouse lots
    np.random.seed(42)
    sample_df = df.sample(100, random_state=42).copy().reset_index(drop=True)
    
    # Assign realistic lot IDs and quantities
    lots = []
    for idx, row in sample_df.iterrows():
        fruit = row["fruit"]
        lot_id = f"LOT-{fruit[:3].upper()}-{2026000 + idx:04d}"
        qty_kg = int(np.random.choice([250, 500, 750, 1000, 1500, 2000]))
        price = COMMODITY_PRICES.get(fruit, 1.50)
        
        # Calculate stress
        temp = row["temperature_c"]
        co2 = row["co2_ppm"]
        humid = row["humidity_pct"]
        
        s_t = compute_thermal_stress(temp)
        s_co2 = compute_co2_stress(co2)
        s_h = compute_humidity_stress(humid)
        
        # In the cleaned dataset, 'is_spoilage' is 0 (Good) or 1 (Bad)
        # For realistic queue, simulate model predicted probability around observed class
        is_spoil = row.get("is_spoilage", row.get("class", 0))
        if is_spoil == 1:
            p_s = round(float(np.random.uniform(0.70, 0.99)), 4)
        else:
            p_s = round(float(np.random.uniform(0.01, 0.25)), 4)
            
        u_score = compute_urgency_score(p_s, s_t, s_co2, s_h)
        tier, action = classify_priority(u_score)
        var_usd = compute_value_at_risk(qty_kg, price, p_s)
        
        lots.append({
            "lot_id": lot_id,
            "fruit": fruit,
            "temp_c": round(temp, 1),
            "humidity_pct": round(humid, 1),
            "co2_ppm": round(co2, 1),
            "spoilage_prob": p_s,
            "urgency_score": u_score,
            "priority_tier": tier.name,
            "quantity_kg": qty_kg,
            "unit_price": price,
            "var_usd": var_usd,
            "action_protocol": action
        })
        
    queue_df = pd.DataFrame(lots).sort_values("urgency_score", ascending=False).reset_index(drop=True)
    return queue_df

queue_df = generate_inventory_queue()

# ----------------- QUEUE KPIS -----------------
k1, k2, k3, k4 = st.columns(4)

total_lots = len(queue_df)
crit_lots = len(queue_df[queue_df["priority_tier"] == "CRITICAL"])
high_lots = len(queue_df[queue_df["priority_tier"] == "HIGH"])
total_var = queue_df["var_usd"].sum() if not queue_df.empty else 0.0

with k1:
    st.markdown(render_kpi_card("Active Chamber Lots", f"{total_lots}", "Monitored Pallet Units", "#38bdf8"), unsafe_allow_html=True)
with k2:
    st.markdown(render_kpi_card("Critical Action Required", f"{crit_lots}", "Urgency U ≥ 75 (Expedite)", "#f43f5e"), unsafe_allow_html=True)
with k3:
    st.markdown(render_kpi_card("Elevated Attention", f"{high_lots}", "Urgency U: 50–74 (Repack)", "#fbbf24"), unsafe_allow_html=True)
with k4:
    st.markdown(render_kpi_card("Total Value at Risk", f"${total_var:,.2f}", "Estimated Exposure (VaR)", "#c084fc"), unsafe_allow_html=True)

st.write("")

# ----------------- QUEUE FILTER CONTROLS -----------------
st.markdown('<div class="fc-card">', unsafe_allow_html=True)
st.markdown("### 🔍 Inventory Filter & Dispatch Controls")

fc1, fc2, fc3, fc4 = st.columns([1, 1, 1.2, 1.5])

with fc1:
    selected_fruit = st.selectbox("Filter Commodity:", ["All Commodities"] + sorted(queue_df["fruit"].unique().tolist()))
with fc2:
    selected_tier = st.selectbox("Priority Tier:", ["All Tiers", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
with fc3:
    min_urgency = st.slider("Minimum Urgency Score (U):", 0.0, 100.0, 0.0, 5.0)
with fc4:
    search_lot = st.text_input("Search Lot ID:", "", placeholder="e.g. LOT-TOM-20260012")

# Apply filters
filtered_df = queue_df.copy()

if selected_fruit != "All Commodities":
    filtered_df = filtered_df[filtered_df["fruit"] == selected_fruit]
if selected_tier != "All Tiers":
    filtered_df = filtered_df[filtered_df["priority_tier"] == selected_tier]
if min_urgency > 0:
    filtered_df = filtered_df[filtered_df["urgency_score"] >= min_urgency]
if search_lot.strip():
    filtered_df = filtered_df[filtered_df["lot_id"].str.contains(search_lot.strip(), case=False, na=False)]

st.markdown('</div>', unsafe_allow_html=True)

# ----------------- CHARTS & QUEUE TABLE -----------------
col_chart, col_exp = st.columns([1, 1.5])

with col_chart:
    st.markdown('<div class="fc-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Urgency Score Distribution")
    
    tier_colors = {"CRITICAL": "#f43f5e", "HIGH": "#fbbf24", "MEDIUM": "#60a5fa", "LOW": "#34d399"}
    fig_hist = px.histogram(
        filtered_df,
        x="urgency_score",
        color="priority_tier",
        color_discrete_map=tier_colors,
        nbins=20,
        labels={"urgency_score": "Urgency Score (U)", "count": "Lot Count"}
    )
    apply_plotly_theme(fig_hist)
    fig_hist.update_layout(height=320, barmode="stack")
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_exp:
    st.markdown('<div class="fc-card">', unsafe_allow_html=True)
    st.markdown("#### 💰 Financial Exposure (VaR) by Commodity")
    
    if not filtered_df.empty:
        var_by_fruit = filtered_df.groupby("fruit")["var_usd"].sum().reset_index()
        fig_var = px.bar(
            var_by_fruit,
            x="fruit",
            y="var_usd",
            color="fruit",
            color_discrete_sequence=["#38bdf8", "#818cf8", "#f472b6", "#fbbf24"],
            labels={"fruit": "Commodity", "var_usd": "Cumulative Exposure ($)"}
        )
        apply_plotly_theme(fig_var)
        fig_var.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_var, use_container_width=True)
    else:
        st.info("No records match the current filter selection.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- INTERACTIVE QUEUE TABLE -----------------
st.markdown('<div class="fc-card">', unsafe_allow_html=True)
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown(f"#### 📋 Triage Dispatch Queue ({len(filtered_df)} Lots Matching)")
with header_col2:
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Queue CSV",
        data=csv_bytes,
        file_name="freshchain_triage_queue.csv",
        mime="text/csv",
        use_container_width=True
    )

display_cols = [
    "lot_id", "fruit", "temp_c", "humidity_pct", "co2_ppm", 
    "spoilage_prob", "urgency_score", "priority_tier", "quantity_kg", "var_usd", "action_protocol"
]

format_dict = {
    "temp_c": "{:.1f}°C",
    "humidity_pct": "{:.1f}%",
    "co2_ppm": "{:.0f} ppm",
    "spoilage_prob": "{:.2%}",
    "urgency_score": "{:.1f}",
    "quantity_kg": "{:,.0f} kg",
    "var_usd": "${:,.2f}"
}

st.dataframe(
    filtered_df[display_cols].style.format(format_dict).background_gradient(
        subset=["urgency_score"], cmap="YlOrRd", vmin=0, vmax=100
    ),
    use_container_width=True,
    height=420
)

st.markdown('</div>', unsafe_allow_html=True)
