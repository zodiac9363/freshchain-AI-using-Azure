"""
FreshChain AI — Enterprise Cold Chain Analytics & Early Warning Platform
Page 4: Global FAO Food Loss Analytics & Supply Chain Benchmarking
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.components.styles import CUSTOM_CSS, apply_plotly_theme, render_kpi_card, render_badge
from app.components.header import render_header

st.set_page_config(
    page_title="FAO Loss Analytics — FreshChain AI",
    page_icon="🌍",
    layout="wide"
)

render_header("Global Food Loss Benchmarks (FAO)", "Macro-Level Supply Chain Inefficiencies, Vulnerable Tiers & Empirical Causes")

@st.cache_data
def load_fao_data():
    fao_path = os.path.join(PROJECT_ROOT, "data", "processed", "fao_cleaned.csv")
    if not os.path.exists(fao_path):
        return pd.DataFrame()
    return pd.read_csv(fao_path)

fao_df = load_fao_data()

# ----------------- TOP KPI ROW -----------------
k1, k2, k3, k4 = st.columns(4)

avg_loss = fao_df["loss_percentage"].mean() if not fao_df.empty else 0.0
highest_stage = (fao_df.groupby("supply_stage_standardized")["loss_percentage"].mean().idxmax() 
                 if not fao_df.empty else "Post-Harvest / Storage")
top_commodity = (fao_df.groupby("commodity_clean")["loss_percentage"].mean().idxmax() 
                 if not fao_df.empty else "Perishables")
total_records = len(fao_df)

with k1:
    st.markdown(render_kpi_card("Global Mean Food Loss", f"{avg_loss:.2f}%", "United Nations FAO Benchmark", "#f43f5e"), unsafe_allow_html=True)
with k2:
    st.markdown(render_kpi_card("Most Vulnerable Tier", f"{highest_stage}", "Peak supply-chain shrinkage", "#fbbf24"), unsafe_allow_html=True)
with k3:
    st.markdown(render_kpi_card("Highest Shrink Commodity", f"{top_commodity}", "Highest average reported loss", "#38bdf8"), unsafe_allow_html=True)
with k4:
    st.markdown(render_kpi_card("Curated FAO Records", f"{total_records:,}", "Cross-country empirical reports", "#10b981"), unsafe_allow_html=True)

st.write("")

# ----------------- FILTER SECTION -----------------
st.markdown('<div class="fc-card">', unsafe_allow_html=True)
st.markdown("### 🔍 Macro Supply Chain Filters")
f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    stage_options = ["All Stages"] + sorted(fao_df["supply_stage_standardized"].unique().tolist())
    selected_stage = st.selectbox("Supply Chain Tier:", stage_options)

with f_col2:
    all_commodities = sorted(fao_df["commodity_clean"].dropna().unique().tolist())
    selected_commodity = st.selectbox("Commodity Filter:", ["All Commodities"] + all_commodities)

with f_col3:
    all_countries = sorted(fao_df["country"].dropna().unique().tolist())
    selected_country = st.selectbox("Country / Jurisdiction:", ["All Countries"] + all_countries)

with f_col4:
    min_year = int(fao_df["year"].min()) if not fao_df.empty else 1990
    max_year = int(fao_df["year"].max()) if not fao_df.empty else 2024
    year_range = st.slider("Observation Year Range:", min_year, max_year, (min_year, max_year))

# Filter dataset
filtered_fao = fao_df.copy()
if selected_stage != "All Stages":
    filtered_fao = filtered_fao[filtered_fao["supply_stage_standardized"] == selected_stage]
if selected_commodity != "All Commodities":
    filtered_fao = filtered_fao[filtered_fao["commodity_clean"] == selected_commodity]
if selected_country != "All Countries":
    filtered_fao = filtered_fao[filtered_fao["country"] == selected_country]
filtered_fao = filtered_fao[(filtered_fao["year"] >= year_range[0]) & (filtered_fao["year"] <= year_range[1])]

st.markdown('</div>', unsafe_allow_html=True)

# ----------------- VISUALIZATIONS ROW 1 -----------------
col_stage, col_cause = st.columns([1.2, 1])

with col_stage:
    st.markdown('<div class="fc-card">', unsafe_allow_html=True)
    st.markdown("#### 🏭 Mean Loss Percentage by Supply Chain Tier")
    
    stage_loss = (filtered_fao.groupby("supply_stage_standardized")["loss_percentage"]
                  .agg(["mean", "count"]).reset_index()
                  .sort_values("mean", ascending=False))
    
    fig_stage = px.bar(
        stage_loss,
        x="supply_stage_standardized",
        y="mean",
        color="mean",
        color_continuous_scale="Reds",
        labels={"supply_stage_standardized": "Supply Chain Tier", "mean": "Average Reported Loss (%)"},
        text_auto=".1f"
    )
    apply_plotly_theme(fig_stage)
    fig_stage.update_layout(height=340, coloraxis_showscale=False)
    st.plotly_chart(fig_stage, use_container_width=True)
    st.caption("Post-Harvest handling and Storage consistently demonstrate high loss rates, validating the need for FreshChain AI cold-chain interventions.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_cause:
    st.markdown('<div class="fc-card">', unsafe_allow_html=True)
    st.markdown("#### ⚠️ Primary Causes of Shrinkage")
    
    # Clean top causes
    top_causes = (filtered_fao["cause_of_loss"]
                  .value_counts().head(8).reset_index())
    top_causes.columns = ["Cause", "Reports"]
    
    fig_cause = px.pie(
        top_causes,
        names="Cause",
        values="Reports",
        hole=0.45,
        color_discrete_sequence=["#38bdf8", "#818cf8", "#f472b6", "#fbbf24", "#34d399", "#a78bfa", "#f87171", "#94a3b8"]
    )
    apply_plotly_theme(fig_cause)
    fig_cause.update_layout(height=340, showlegend=True, legend=dict(orientation="v", x=1.0, y=0.5))
    st.plotly_chart(fig_cause, use_container_width=True)
    st.caption("Biological deterioration, improper storage temperature, and post-harvest delays represent majority loss drivers.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- VISUALIZATIONS ROW 2 -----------------
col_comm, col_trend = st.columns([1.2, 1])

with col_comm:
    st.markdown('<div class="fc-card">', unsafe_allow_html=True)
    st.markdown("#### 🍎 Top 12 Commodities by Average Loss Rate")
    
    comm_agg = (filtered_fao.groupby("commodity_clean")["loss_percentage"]
                .agg(["mean", "count"])
                .query("count >= 5")
                .sort_values("mean", ascending=False)
                .head(12).reset_index())
    
    fig_comm = px.bar(
        comm_agg,
        x="mean",
        y="commodity_clean",
        orientation="h",
        color="mean",
        color_continuous_scale="Viridis",
        labels={"mean": "Average Loss (%)", "commodity_clean": "Produce Commodity"}
    )
    apply_plotly_theme(fig_comm)
    fig_comm.update_layout(height=340, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig_comm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_trend:
    st.markdown('<div class="fc-card">', unsafe_allow_html=True)
    st.markdown("#### 📈 Historical Loss Trends Over Time")
    
    yearly_agg = (filtered_fao.groupby("year")["loss_percentage"]
                  .mean().reset_index())
    
    fig_trend = px.line(
        yearly_agg,
        x="year",
        y="loss_percentage",
        markers=True,
        line_shape="spline",
        labels={"year": "Year", "loss_percentage": "Mean Loss (%)"}
    )
    fig_trend.update_traces(line=dict(color="#38bdf8", width=3), marker=dict(size=6, color="#818cf8"))
    apply_plotly_theme(fig_trend)
    fig_trend.update_layout(height=340)
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- MACRO IMPACT BANNER -----------------
st.markdown("""
<div class="fc-card" style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%); border-left: 4px solid #10b981;">
    <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">
        🌱 Strategic Supply Chain Alignment: UN SDG 12.3 & Economic Resilience
    </div>
    <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
        According to the Food and Agriculture Organization (FAO), global post-harvest and storage shrinkage across fresh fruits and vegetables accounts for upwards of <strong>14% to 22%</strong> of total agricultural production. By providing continuous telemetry-driven respiration detection and automated triage scoring, FreshChain AI directly bridges the gap between field harvest and supermarket shelf, preventing high-risk produce batches from entering catastrophic decomposition.
    </div>
</div>
""", unsafe_allow_html=True)
