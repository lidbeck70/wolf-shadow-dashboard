"""
Sentiment & Flow Page — Cyberpunk Dashboard Module
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

try:
    from sentiment.fear_greed import compute_fear_greed, get_retail_flow, get_fear_greed_history
except ImportError:
    from dashboard.sentiment.fear_greed import compute_fear_greed, get_retail_flow, get_fear_greed_history

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
BG       = "#050510"
BG2      = "#0a0a1e"
CYAN     = "#00ffff"
MAGENTA  = "#ff00ff"
GREEN    = "#00ff88"
RED      = "#ff3355"
YELLOW   = "#ffdd00"
TEXT     = "#e0e0ff"
DIM      = "#4a4a6a"
ORANGE   = "#ff8800"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_to_rgba(score: float, alpha: float = 1.0) -> str:
    """
    Map a 0-100 score to an rgba() color string.
    Red (0) → Orange (25) → Yellow (50) → Light Green (75) → Green (100)
    """
    if score < 25:
        t = score / 25.0
        r = int(255)
        g = int(51 + (136 - 51) * t)   # 51 → 136
        b = int(85 * (1 - t))           # 85 → 0
    elif score < 50:
        t = (score - 25) / 25.0
        r = int(255)
        g = int(136 + (221 - 136) * t)  # 136 → 221
        b = 0
    elif score < 75:
        t = (score - 50) / 25.0
        r = int(255 * (1 - t))          # 255 → 0
        g = int(221 + (255 - 221) * t)  # 221 → 255
        b = 0
    else:
        t = (score - 75) / 25.0
        r = 0
        g = int(255)
        b = int(136 * t)                # 0 → 136
    return f"rgba({r},{g},{b},{alpha})"


def _component_bar_html(name: str, score: float) -> str:
    """Render a single component card as an HTML snippet."""
    fill_color = _score_to_rgba(score, 0.85)
    border_color = _score_to_rgba(score, 0.5)
    pct = int(score)
    return f"""
<div style="
    background:{BG2};
    border:1px solid {border_color};
    border-radius:8px;
    padding:10px 14px;
    margin-bottom:10px;
    font-family:'Courier New',monospace;
">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span style="color:{TEXT};font-size:0.82rem;letter-spacing:0.05em;">{name.upper()}</span>
        <span style="color:{fill_color};font-size:1rem;font-weight:700;">{score:.0f}</span>
    </div>
    <div style="background:{BG};border-radius:4px;height:8px;overflow:hidden;">
        <div style="
            width:{pct}%;
            height:100%;
            background:linear-gradient(90deg,{_score_to_rgba(max(score-30,0),0.7)},{fill_color});
            border-radius:4px;
            transition:width 0.4s ease;
        "></div>
    </div>
</div>
"""


def _section_header(title: str, subtitle: str = "") -> None:
    sub_html = f"<p style='color:{DIM};font-size:0.8rem;margin:0;letter-spacing:0.1em;'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
<div style="margin:28px 0 14px 0;border-left:3px solid {CYAN};padding-left:12px;">
    <h3 style="color:{CYAN};font-family:'Courier New',monospace;margin:0;letter-spacing:0.12em;text-transform:uppercase;font-size:1rem;">{title}</h3>
    {sub_html}
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_gauge(score: float, label: str, color: str) -> None:
    """Section 1: Fear & Greed Gauge."""
    _section_header("Fear & Greed Index", "Composite sentiment score — updated every 30 min")

    # Gauge figure
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "font": {"size": 56, "color": color, "family": "Courier New"},
            "suffix": "",
        },
        title={
            "text": label,
            "font": {"size": 22, "color": color, "family": "Courier New"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": DIM,
                "tickfont": {"color": TEXT, "size": 11, "family": "Courier New"},
                "nticks": 6,
            },
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": BG2,
            "borderwidth": 0,
            "steps": [
                {"range": [0,   25], "color": "rgba(255,51,85,0.18)"},
                {"range": [25,  45], "color": "rgba(255,136,0,0.15)"},
                {"range": [45,  55], "color": "rgba(255,221,0,0.13)"},
                {"range": [55,  75], "color": "rgba(136,221,0,0.15)"},
                {"range": [75, 100], "color": "rgba(0,255,136,0.18)"},
            ],
            "threshold": {
                "line": {"color": CYAN, "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG2,
        font={"color": TEXT, "family": "Courier New"},
        height=340,
        margin={"t": 30, "b": 10, "l": 30, "r": 30},
    )

    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        st.plotly_chart(fig, use_container_width=True, key="fear_greed_gauge")

    # Colored label badge
    badge_bg = _score_to_rgba(score, 0.18)
    badge_border = _score_to_rgba(score, 0.7)
    st.markdown(f"""
<div style="text-align:center;margin-top:-10px;margin-bottom:18px;">
    <span style="
        display:inline-block;
        background:{badge_bg};
        border:1.5px solid {badge_border};
        color:{color};
        font-family:'Courier New',monospace;
        font-size:1.1rem;
        font-weight:700;
        letter-spacing:0.18em;
        padding:6px 28px;
        border-radius:24px;
        text-transform:uppercase;
    ">{label}</span>
</div>
""", unsafe_allow_html=True)


def _render_components(components: dict[str, float]) -> None:
    """Section 2: Component Breakdown."""
    _section_header("Component Breakdown", "Six sub-indicators — each scored 0-100")

    items = list(components.items())
    left_items = items[:3]
    right_items = items[3:]

    col_l, col_r = st.columns(2)
    with col_l:
        for name, score in left_items:
            st.markdown(_component_bar_html(name, score), unsafe_allow_html=True)
    with col_r:
        for name, score in right_items:
            st.markdown(_component_bar_html(name, score), unsafe_allow_html=True)


def _render_history() -> None:
    """Section 3: Fear & Greed History — VIX-based proxy, last 60 trading days."""
    _section_header("Fear & Greed Trend", "VIX-based proxy — last 60 trading days")

    try:
        df = get_fear_greed_history(days=60)
    except Exception as exc:
        st.markdown(
            f"<p style='color:{DIM};font-family:Courier New,monospace;'>History data unavailable: {exc}</p>",
            unsafe_allow_html=True,
        )
        return

    if df is None or df.empty:
        st.markdown(
            f"<p style='color:{DIM};font-family:Courier New,monospace;'>No historical data available.</p>",
            unsafe_allow_html=True,
        )
        return

    try:
        _render_history_chart(df)
    except Exception as exc:
        st.markdown(
            f"<p style='color:{DIM};font-family:Courier New,monospace;'>Chart render error: {exc}</p>",
            unsafe_allow_html=True,
        )


def _render_history_chart(df: "pd.DataFrame") -> None:
    """Build and display the Fear & Greed history chart."""
    fig = go.Figure()

    # Horizontal fill bands (zone backgrounds)
    x_range = [df["date"].min(), df["date"].max()]

    fig.add_hrect(y0=0,  y1=25,  fillcolor="rgba(255,51,85,0.10)",   line_width=0, layer="below")
    fig.add_hrect(y0=25, y1=45,  fillcolor="rgba(255,136,0,0.08)",   line_width=0, layer="below")
    fig.add_hrect(y0=45, y1=55,  fillcolor="rgba(255,221,0,0.07)",   line_width=0, layer="below")
    fig.add_hrect(y0=55, y1=75,  fillcolor="rgba(136,221,0,0.08)",   line_width=0, layer="below")
    fig.add_hrect(y0=75, y1=100, fillcolor="rgba(0,255,136,0.10)",   line_width=0, layer="below")

    # Zone label annotations
    zone_labels = [
        (12.5, "EXTREME FEAR"),
        (35.0, "FEAR"),
        (50.0, "NEUTRAL"),
        (65.0, "GREED"),
        (87.5, "EXTREME GREED"),
    ]
    for y_pos, zone_name in zone_labels:
        fig.add_annotation(
            x=df["date"].iloc[0],
            y=y_pos,
            text=zone_name,
            showarrow=False,
            font={"size": 9, "color": DIM, "family": "Courier New"},
            xanchor="left",
            yanchor="middle",
            opacity=0.7,
        )

    # Main line
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["score"],
        mode="lines",
        name="F&G Proxy",
        line={"color": CYAN, "width": 2.5, "shape": "spline"},
        fill="tozeroy",
        fillcolor="rgba(0,255,255,0.07)",
    ))

    # Horizontal reference lines
    for level, clr in [(25, "rgba(255,51,85,0.4)"), (50, "rgba(255,221,0,0.35)"), (75, "rgba(0,255,136,0.4)")]:
        fig.add_hline(
            y=level,
            line_dash="dot",
            line_color=clr,
            line_width=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG2,
        font={"color": TEXT, "family": "Courier New"},
        height=320,
        margin={"t": 20, "b": 40, "l": 50, "r": 20},
        xaxis={
            "showgrid": True,
            "gridcolor": "rgba(74,74,106,0.3)",
            "tickfont": {"size": 10, "color": DIM},
            "title": "",
        },
        yaxis={
            "range": [0, 100],
            "showgrid": True,
            "gridcolor": "rgba(74,74,106,0.3)",
            "tickfont": {"size": 10, "color": DIM},
            "title": {"text": "Score", "font": {"size": 11, "color": DIM}},
        },
        showlegend=False,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, key="fear_greed_history")


def _render_retail_flow() -> None:
    """Section 4: Retail Flow — placeholder with dim mock chart."""
    _section_header("Retail Flow", "Nasdaq API — placeholder")

    # Info box
    st.markdown(f"""
<div style="
    background:{BG2};
    border:1.5px solid {CYAN};
    border-radius:10px;
    padding:18px 22px;
    margin-bottom:16px;
    font-family:'Courier New',monospace;
">
    <div style="color:{CYAN};font-size:1rem;font-weight:700;letter-spacing:0.15em;margin-bottom:8px;">
        RETAIL FLOW (NASDAQ API)
    </div>
    <div style="color:{TEXT};font-size:0.85rem;line-height:1.6;margin-bottom:10px;">
        Integration coming soon — this module will track retail buy/sell flow from
        Nasdaq's retail order flow data.
    </div>
    <div style="
        display:inline-block;
        background:rgba(74,74,106,0.25);
        border:1px solid {DIM};
        color:{DIM};
        font-size:0.78rem;
        letter-spacing:0.12em;
        padding:3px 14px;
        border-radius:12px;
    ">STATUS: PLACEHOLDER</div>
</div>
""", unsafe_allow_html=True)

    # Grayed-out mock bar chart
    mock_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    mock_values = [+1.2, -0.8, +2.1, -1.5, +0.6]
    bar_colors = [
        f"rgba(74,74,106,0.6)" for _ in mock_values
    ]

    fig = go.Figure(go.Bar(
        x=mock_labels,
        y=mock_values,
        marker_color=bar_colors,
        marker_line_width=0,
        name="Net Retail Flow (mock)",
    ))

    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color=DIM,
        line_width=1,
    )

    # Watermark
    fig.add_annotation(
        x=2,
        y=0,
        text="DEMO DATA — NOT REAL",
        showarrow=False,
        font={"size": 22, "color": "rgba(74,74,106,0.25)", "family": "Courier New"},
        textangle=-20,
        xref="x",
        yref="y",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG2,
        font={"color": DIM, "family": "Courier New"},
        height=220,
        margin={"t": 15, "b": 30, "l": 40, "r": 20},
        xaxis={
            "showgrid": False,
            "tickfont": {"size": 10, "color": DIM},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "rgba(74,74,106,0.2)",
            "tickfont": {"size": 10, "color": DIM},
            "title": {"text": "Net Flow ($B)", "font": {"size": 10, "color": DIM}},
            "zeroline": False,
        },
        showlegend=False,
        title={
            "text": "5-DAY RETAIL FLOW (PLACEHOLDER)",
            "font": {"size": 11, "color": DIM, "family": "Courier New"},
            "x": 0.02,
            "xanchor": "left",
        },
    )

    st.plotly_chart(fig, use_container_width=True, key="retail_flow_placeholder")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_sentiment_page() -> None:
    """Render the full Sentiment & Flow Streamlit page."""

    # Page-level CSS injection for consistent cyberpunk feel
    st.markdown(f"""
<style>
    /* Streamlit container overrides */
    .block-container {{
        background: {BG};
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }}
    section[data-testid="stSidebar"] {{
        background: {BG2};
    }}
    /* Remove default Streamlit expander/metric border highlights */
    .stMetric label {{
        color: {DIM} !important;
        font-family: 'Courier New', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.1em !important;
    }}
    .stMetric [data-testid="metric-container"] {{
        background: {BG2};
        border: 1px solid {DIM};
        border-radius: 8px;
        padding: 8px 14px;
    }}
    /* Plotly chart background alignment */
    .js-plotly-plot .plotly {{
        background: {BG} !important;
    }}
</style>
""", unsafe_allow_html=True)

    # Page title
    st.markdown(f"""
<div style="
    border-bottom:1px solid {DIM};
    padding-bottom:14px;
    margin-bottom:4px;
    font-family:'Courier New',monospace;
">
    <h1 style="
        color:{CYAN};
        font-size:1.6rem;
        letter-spacing:0.2em;
        text-transform:uppercase;
        margin:0;
        text-shadow:0 0 18px rgba(0,255,255,0.5);
    ">SENTIMENT &amp; FLOW</h1>
    <p style="color:{DIM};font-size:0.78rem;margin:4px 0 0 0;letter-spacing:0.1em;">
        Synthetic Fear &amp; Greed · Retail Flow (placeholder) · Real-time via yfinance
    </p>
</div>
""", unsafe_allow_html=True)

    # Load data with spinner
    with st.spinner("Computing Fear & Greed index…"):
        fg = compute_fear_greed()

    score      = fg["score"]
    label      = fg["label"]
    color      = fg["color"]
    components = fg["components"]

    # -----------------------------------------------------------------------
    # Section 1: Gauge
    # -----------------------------------------------------------------------
    _render_gauge(score, label, color)

    st.markdown("<hr style='border:1px solid rgba(74,74,106,0.4);margin:6px 0 6px 0;'>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 2: Component Breakdown
    # -----------------------------------------------------------------------
    _render_components(components)

    st.markdown("<hr style='border:1px solid rgba(74,74,106,0.4);margin:6px 0 6px 0;'>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 3: History trend
    # -----------------------------------------------------------------------
    _render_history()

    st.markdown("<hr style='border:1px solid rgba(74,74,106,0.4);margin:6px 0 6px 0;'>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 4: Retail Flow (placeholder)
    # -----------------------------------------------------------------------
    _render_retail_flow()

    # Footer timestamp
    st.markdown(
        f"<p style='color:{DIM};font-family:Courier New,monospace;font-size:0.7rem;text-align:right;margin-top:20px;'>"
        f"SENTIMENT MODULE · DATA: YFINANCE · CACHE TTL: 30 MIN</p>",
        unsafe_allow_html=True,
    )
