"""
FreshChain AI — Design System & Custom CSS Tokens
Provides rich dark-mode glassmorphic styling, custom card components, and vibrant visual accents.
"""

from typing import Any, Optional

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Dark glassmorphic background */
.stApp {
    background: radial-gradient(circle at 15% 15%, #0f172a 0%, #030712 100%);
    color: #f8fafc;
}

/* Card Container */
.fc-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.fc-card:hover {
    border-color: rgba(56, 189, 248, 0.3);
    transform: translateY(-2px);
}

/* Glowing KPI Box */
.fc-kpi-container {
    display: flex;
    flex-direction: column;
    padding: 18px;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    position: relative;
    overflow: hidden;
}

.fc-kpi-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
}

.fc-kpi-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 6px;
}

.fc-kpi-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.03em;
    line-height: 1.1;
}

.fc-kpi-sub {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 6px;
}

/* Status Badges */
.badge-critical {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
}

.badge-high {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.4);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
}

.badge-medium {
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.4);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
}

.badge-low {
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.4);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
}

/* Telemetry Indicator */
.cloud-pulse {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse-green 2s infinite;
    margin-right: 8px;
}

@keyframes pulse-green {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Tables styling */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* Metric styling */
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
}

/* Custom buttons */
.stButton>button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
    border: 1px solid rgba(255, 255, 255, 0.12);
}
.stButton>button:hover {
    border-color: #38bdf8;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.3);
    transform: translateY(-1px);
}
</style>
"""

def apply_plotly_theme(fig):
    """Applies premium dark glassmorphic styling to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#e2e8f0", size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            bgcolor="rgba(15,23,42,0.6)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(color="#94a3b8")
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.1)",
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.1)",
            tickfont=dict(color="#94a3b8")
        )
    )
    return fig


def render_kpi_card(title: str, value: str, subtext: str = "", accent_color: str = "#38bdf8") -> str:
    """Returns HTML for a sleek glowing KPI box."""
    return f"""
    <div class="fc-kpi-container" style="border-top: 2px solid {accent_color};">
        <div class="fc-kpi-title">{title}</div>
        <div class="fc-kpi-value">{value}</div>
        {f'<div class="fc-kpi-sub">{subtext}</div>' if subtext else ''}
    </div>
    """


def render_badge(tier: Any, text: str = None) -> str:
    """Returns an HTML badge for a priority tier."""
    t = getattr(tier, "name", str(tier)).upper()
    display_text = text or t
    if t == "CRITICAL":
        return f'<span class="badge-critical">● {display_text}</span>'
    elif t == "HIGH":
        return f'<span class="badge-high">▲ {display_text}</span>'
    elif t == "MEDIUM":
        return f'<span class="badge-medium">■ {display_text}</span>'
    else:
        return f'<span class="badge-low">✓ {display_text}</span>'

