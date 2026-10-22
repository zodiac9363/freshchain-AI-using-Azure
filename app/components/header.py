"""
FreshChain AI — Header & Navigation Component
Renders the top telemetry bar, cloud connectivity badge, and page metadata.
"""

import streamlit as st
from app.components.styles import CUSTOM_CSS


def render_header(page_title: str, subtitle: str):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 28px;">
        <div>
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span class="cloud-pulse"></span>
                <span style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.1em;">FreshChain AI · Live Telemetry</span>
            </div>
            <h1 style="margin: 0; font-size: 2.3rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.03em;">{page_title}</h1>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">{subtitle}</p>
        </div>
        <div style="text-align: right; background: rgba(15, 23, 42, 0.6); padding: 8px 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
            <div style="font-size: 0.75rem; color: #64748b;">Cloud Architecture</div>
            <div style="font-size: 0.85rem; font-weight: 700; color: #10b981;">Azure Blob · Functions v2</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
