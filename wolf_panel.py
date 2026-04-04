#!/usr/bin/env python3
"""
SweWolf Panel — Lidbeck Edition v2.0
====================================
Cyberpunk dark-theme dashboard for Nordic swing trading intelligence.

Run:
    streamlit run wolf_panel.py
"""

import sys
import os
import warnings
import time
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATH SETUP — find screener/backtester modules
# Looks in: 1) same folder as dashboard  2) sibling folders  3) parent folder
# ---------------------------------------------------------------------------
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(DASHBOARD_DIR)

# Add all possible locations to sys.path
for p in [
    DASHBOARD_DIR,                                    # same folder
    os.path.join(WORKSPACE_DIR, "screener"),           # sibling: ../screener/
    os.path.join(WORKSPACE_DIR, "backtester"),          # sibling: ../backtester/
    WORKSPACE_DIR,                                     # parent folder
]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# CAGR Strategy module
try:
    from cagr.cagr_streamlit import render_cagr_page
    CAGR_AVAILABLE = True
except ImportError:
    CAGR_AVAILABLE = False

# Long-Term Trend & Drawdowns module
try:
    from long_trend.long_trend_streamlit import render_long_trend_page
    LONG_TREND_AVAILABLE = True
except ImportError:
    LONG_TREND_AVAILABLE = False

# Sector & Global Regime module
try:
    from sector_cycle.sector_cycle_streamlit import render_sector_cycle_page
    SECTOR_CYCLE_AVAILABLE = True
except ImportError:
    SECTOR_CYCLE_AVAILABLE = False

# Sentiment & Flow module
try:
    from sentiment.sentiment_streamlit import render_sentiment_page
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# Heatmap module
try:
    from heatmap.heatmap_streamlit import render_heatmap_page
    HEATMAP_AVAILABLE = True
except ImportError:
    HEATMAP_AVAILABLE = False

# RS Backtest module
try:
    from rs_backtest.rs_backtest_streamlit import render_rs_backtest_page
    RS_BACKTEST_AVAILABLE = True
except ImportError:
    RS_BACKTEST_AVAILABLE = False

# OVTLYR module
try:
    from ovtlyr.ui.layout import render_ovtlyr_page
    OVTLYR_AVAILABLE = True
except ImportError:
    OVTLYR_AVAILABLE = False

# Rules page
try:
    from ovtlyr.ui.rules_page import render_rules_page
    RULES_AVAILABLE = True
except ImportError:
    RULES_AVAILABLE = False

# OVTLYR Screener
try:
    from screener_ovtlyr import run_ovtlyr_screener
    OVTLYR_SCREENER_AVAILABLE = True
except ImportError:
    OVTLYR_SCREENER_AVAILABLE = False

# Unified backtest engine
try:
    from backtest_engine import run_batch_backtest, run_backtest
    BACKTEST_ENGINE_AVAILABLE = True
except ImportError:
    BACKTEST_ENGINE_AVAILABLE = False

# Long Regime Monitor
try:
    from long_regime_monitor import render_long_regime_monitor
    LONG_REGIME_AVAILABLE = True
except ImportError:
    LONG_REGIME_AVAILABLE = False
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# PAGE CONFIG — must be very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="SweWolf Panel — Lidbeck Edition",
    page_icon="🐺",
)

# PWA / Mobile meta tags for iPad home screen
st.markdown(
    """
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="SweWolf">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <meta name="theme-color" content="#050510">
    <link rel="apple-touch-icon" href="https://em-content.zobj.net/source/apple/391/wolf_1f43a.png">
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# CYBERPUNK CSS
# ---------------------------------------------------------------------------
CYBERPUNK_CSS = """
<style>
/* ── Global Reset ────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Courier New', Courier, monospace;
}

/* ── Background ──────────────────────────────────────────────── */
.stApp {
    background-color: #050510;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(0, 255, 255, 0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 80%, rgba(255, 0, 255, 0.04) 0%, transparent 60%),
        repeating-linear-gradient(
            0deg, transparent, transparent 39px, rgba(0,255,255,0.03) 40px
        ),
        repeating-linear-gradient(
            90deg, transparent, transparent 39px, rgba(0,255,255,0.03) 40px
        );
}

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #07071a;
    border-right: 1px solid rgba(0,255,255,0.15);
}

/* ── Tab styling ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #07071a;
    border-bottom: 2px solid rgba(0,255,255,0.2);
    gap: 4px;
    padding: 0 8px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border: 1px solid rgba(0,255,255,0.15);
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    color: rgba(0,255,255,0.5);
    font-family: 'Courier New', monospace;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 10px 20px;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(0,255,255,0.12) 0%, transparent 100%);
    border: 1px solid #00ffff;
    border-bottom: none;
    color: #00ffff;
    text-shadow: 0 0 10px #00ffff, 0 0 20px rgba(0,255,255,0.5);
}

.stTabs [data-baseweb="tab"]:hover {
    color: #00ffff;
    border-color: rgba(0,255,255,0.4);
}

/* ── Metric cards ────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(0,255,255,0.06) 0%, rgba(255,0,255,0.03) 100%);
    border: 1px solid rgba(0,255,255,0.2);
    border-radius: 8px;
    padding: 16px;
    position: relative;
    overflow: hidden;
}

[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00ffff, #ff00ff);
}

[data-testid="stMetricLabel"] {
    color: rgba(0,255,255,0.6) !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
    color: #00ffff !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    text-shadow: 0 0 12px rgba(0,255,255,0.5);
}

[data-testid="stMetricDelta"] > div {
    font-size: 13px !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,255,255,0.15) 0%, rgba(255,0,255,0.1) 100%);
    border: 1px solid #00ffff;
    border-radius: 4px;
    color: #00ffff;
    font-family: 'Courier New', monospace;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 12px 32px;
    transition: all 0.2s ease;
    box-shadow: 0 0 12px rgba(0,255,255,0.2), inset 0 0 12px rgba(0,255,255,0.05);
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,255,255,0.25) 0%, rgba(255,0,255,0.2) 100%);
    border-color: #ff00ff;
    color: #ff00ff;
    box-shadow: 0 0 20px rgba(255,0,255,0.4), 0 0 40px rgba(0,255,255,0.2);
    text-shadow: 0 0 8px #ff00ff;
}

.stButton > button:active {
    box-shadow: 0 0 30px rgba(255,0,255,0.6);
}

/* ── Inputs ──────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background-color: #0a0a1f !important;
    border: 1px solid rgba(0,255,255,0.25) !important;
    border-radius: 4px !important;
    color: #00ffff !important;
    font-family: 'Courier New', monospace !important;
}

.stSelectbox > div > div:focus,
.stTextInput > div > div > input:focus {
    border-color: #00ffff !important;
    box-shadow: 0 0 8px rgba(0,255,255,0.3) !important;
}

/* ── Slider ──────────────────────────────────────────────────── */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #00ffff, #ff00ff) !important;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #00ffff !important;
    box-shadow: 0 0 10px rgba(0,255,255,0.7) !important;
}

/* ── DataFrames / tables ─────────────────────────────────────── */
.stDataFrame {
    border: 1px solid rgba(0,255,255,0.15) !important;
    border-radius: 6px !important;
}

/* ── Headers ─────────────────────────────────────────────────── */
h1, h2, h3 {
    color: #00ffff !important;
    text-shadow: 0 0 20px rgba(0,255,255,0.4);
    letter-spacing: 3px;
}

/* ── Progress bar ────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00ffff, #ff00ff);
}

/* ── Alerts/info ─────────────────────────────────────────────── */
.stAlert {
    background-color: rgba(0,255,255,0.07) !important;
    border: 1px solid rgba(0,255,255,0.25) !important;
    color: #00ffff !important;
}

/* ── Dividers ────────────────────────────────────────────────── */
hr {
    border-color: rgba(0,255,255,0.15) !important;
}

/* ── Custom banner ───────────────────────────────────────────── */
.wolf-banner {
    background: linear-gradient(135deg, #07071a 0%, #0a0020 50%, #07071a 100%);
    border: 1px solid rgba(0,255,255,0.2);
    border-top: 3px solid #00ffff;
    border-radius: 8px;
    padding: 20px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.wolf-banner::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #ff00ff, transparent);
}

.wolf-banner h1 {
    font-size: 36px !important;
    font-weight: 900 !important;
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
    background: linear-gradient(90deg, #00ffff, #ff00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none !important;
    letter-spacing: 6px !important;
}

.wolf-banner p {
    color: rgba(0,255,255,0.5);
    font-size: 11px;
    letter-spacing: 4px;
    margin: 0;
    text-transform: uppercase;
}

/* ── Score badges ────────────────────────────────────────────── */
.score-green  { color: #00ff88; font-weight: 700; text-shadow: 0 0 8px rgba(0,255,136,0.5); }
.score-yellow { color: #ffdd00; font-weight: 700; text-shadow: 0 0 8px rgba(255,221,0,0.5); }
.score-red    { color: #ff3366; font-weight: 700; text-shadow: 0 0 8px rgba(255,51,102,0.5); }
.entry-yes    { color: #00ff88; font-weight: 700; letter-spacing: 1px; }
.entry-no     { color: rgba(255,255,255,0.25); }

/* ── Regime gauge label ──────────────────────────────────────── */
.regime-score {
    font-size: 72px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(180deg, #00ffff, #ff00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    text-shadow: none;
}

.regime-label {
    font-size: 11px;
    letter-spacing: 4px;
    text-transform: uppercase;
    text-align: center;
    color: rgba(0,255,255,0.5);
    margin-top: 4px;
}

/* ── Section titles ──────────────────────────────────────────── */
.section-title {
    font-size: 11px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: rgba(0,255,255,0.4);
    border-bottom: 1px solid rgba(0,255,255,0.1);
    padding-bottom: 6px;
    margin-bottom: 16px;
}

/* ── Status pill ─────────────────────────────────────────────── */
.status-active {
    background: rgba(0,255,136,0.15);
    border: 1px solid #00ff88;
    border-radius: 20px;
    color: #00ff88;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 4px 14px;
    text-transform: uppercase;
    display: inline-block;
    text-shadow: 0 0 8px rgba(0,255,136,0.5);
}

.status-inactive {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    color: rgba(255,255,255,0.3);
    font-size: 11px;
    letter-spacing: 2px;
    padding: 4px 14px;
    text-transform: uppercase;
    display: inline-block;
}

/* ── iPad & Mobile Responsive ─────────────────────────────────── */
@media (max-width: 1024px) {
    .stTabs [data-baseweb="tab"] {
        font-size: 10px !important;
        letter-spacing: 1px !important;
        padding: 8px 12px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        flex-wrap: nowrap;
    }
    [data-testid="stMetricValue"] {
        font-size: 20px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 9px !important;
    }
}

@media (max-width: 768px) {
    .stTabs [data-baseweb="tab"] {
        font-size: 9px !important;
        padding: 6px 8px !important;
        letter-spacing: 0.5px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 16px !important;
    }
    [data-testid="column"] {
        min-width: 100% !important;
    }
}

/* Touch-friendly targets (44px minimum per Apple HIG) */
.stButton > button,
.stSelectbox > div {
    min-height: 44px;
}

/* Smooth scrolling iOS */
[data-testid="stAppViewContainer"] {
    -webkit-overflow-scrolling: touch;
}

/* Scrollable tab bar on iPad */
.stTabs [data-baseweb="tab-list"] {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}
.stTabs [data-baseweb="tab"] {
    white-space: nowrap;
    flex-shrink: 0;
}

</style>
"""

# ─── Plotly dark template ─────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(5,5,16,0)",
    plot_bgcolor="rgba(5,5,16,0)",
    font=dict(family="Courier New, monospace", color="#00ffff", size=11),
    xaxis=dict(
        gridcolor="rgba(0,255,255,0.08)",
        zerolinecolor="rgba(0,255,255,0.15)",
        tickfont=dict(color="rgba(0,255,255,0.6)"),
    ),
    yaxis=dict(
        gridcolor="rgba(0,255,255,0.08)",
        zerolinecolor="rgba(0,255,255,0.15)",
        tickfont=dict(color="rgba(0,255,255,0.6)"),
    ),
    margin=dict(l=50, r=20, t=50, b=40),
)


# =============================================================================
# HELPERS
# =============================================================================

def inject_css():
    st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)


def wolf_banner():
    st.markdown("""
    <div class="wolf-banner">
        <h1>🐺 SweWolf Panel</h1>
        <p>Lidbeck Edition &nbsp;|&nbsp; Nordic Swing Trading Intelligence &nbsp;|&nbsp; Multi-Layer Analysis</p>
    </div>
    """, unsafe_allow_html=True)


def section_title(text):
    st.markdown(f'<p class="section-title">{text}</p>', unsafe_allow_html=True)


def color_score(val):
    if val >= 70:
        return f'<span class="score-green">{val}</span>'
    elif val >= 50:
        return f'<span class="score-yellow">{val}</span>'
    else:
        return f'<span class="score-red">{val}</span>'


def color_entry(val):
    if val == "YES":
        return f'<span class="entry-yes">✦ YES</span>'
    return f'<span class="entry-no">·no·</span>'


# =============================================================================
# PLOTLY CHART BUILDERS
# =============================================================================

def build_equity_chart(eq_df, ticker):
    """Plotly equity curve with gradient fill."""
    fig = go.Figure()

    # Fill above initial
    fig.add_trace(go.Scatter(
        x=eq_df.index, y=eq_df["equity"],
        fill="tozeroy",
        fillcolor="rgba(0,255,255,0.05)",
        line=dict(color="#00ffff", width=1.5),
        name="Equity",
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
    ))

    # Initial capital line
    fig.add_hline(
        y=100000,
        line=dict(color="rgba(255,0,255,0.4)", width=1, dash="dot"),
        annotation_text="Initial $100k",
        annotation_font=dict(color="rgba(255,0,255,0.6)", size=10),
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"EQUITY CURVE — {ticker}", font=dict(size=13, color="#00ffff")),
        height=320,
        showlegend=False,
    )
    return fig


def build_drawdown_chart(dd_series):
    """Plotly drawdown chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd_series.index,
        y=dd_series.values * 100,
        fill="tozeroy",
        fillcolor="rgba(255,0,255,0.12)",
        line=dict(color="#ff00ff", width=1.5),
        name="Drawdown %",
        hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>",
    ))
    layout_copy = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"}
    yaxis_base = PLOTLY_LAYOUT.get("yaxis", {})
    fig.update_layout(
        **layout_copy,
        title=dict(text="DRAWDOWN %", font=dict(size=13, color="#ff00ff")),
        height=220,
        yaxis=dict(**yaxis_base, ticksuffix="%"),
        showlegend=False,
    )
    return fig


def build_monthly_heatmap(returns_series, eq_df):
    """Plotly monthly returns heatmap."""
    dates = eq_df.index
    if len(dates) > len(returns_series):
        dates = dates[:len(returns_series)]
    elif len(returns_series) > len(dates):
        returns_series = returns_series[:len(dates)]

    monthly = pd.DataFrame({"date": dates, "return": returns_series.values})
    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    monthly_ret = monthly.groupby(["year", "month"])["return"].sum().unstack() * 100

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    existing_months = [m for m in range(1, 13) if m in monthly_ret.columns]
    display_labels = [month_labels[m-1] for m in existing_months]
    z_data = monthly_ret[existing_months].values
    y_labels = [str(y) for y in monthly_ret.index.tolist()]

    # Text annotations
    text_data = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z_data]

    fig = go.Figure(go.Heatmap(
        z=z_data,
        x=display_labels,
        y=y_labels,
        text=text_data,
        texttemplate="%{text}",
        textfont=dict(size=10, family="Courier New"),
        colorscale=[
            [0.0,  "#ff0044"],
            [0.35, "#4d0022"],
            [0.5,  "#0a0a1f"],
            [0.65, "#003322"],
            [1.0,  "#00ff88"],
        ],
        zmid=0,
        zmin=-10,
        zmax=10,
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="rgba(0,255,255,0.6)", size=10),
            ticksuffix="%",
            outlinewidth=0,
        ),
        hovertemplate="<b>%{x} %{y}</b><br>%{z:.2f}%<extra></extra>",
    ))

    layout_copy2 = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("yaxis", "xaxis")}
    fig.update_layout(
        **layout_copy2,
        title=dict(text="MONTHLY RETURNS HEATMAP", font=dict(size=13, color="#00ffff")),
        height=max(180, len(y_labels) * 35 + 80),
        xaxis=dict(**PLOTLY_LAYOUT.get("xaxis", {}), side="bottom"),
        yaxis=dict(**PLOTLY_LAYOUT.get("yaxis", {}), autorange="reversed"),
    )
    return fig


def build_gauge(value, max_val, label, color_cyan=True):
    """Plotly gauge/indicator."""
    bar_color = "rgba(0,255,255,0.9)" if color_cyan else "rgba(255,0,255,0.9)"
    bg_color = "rgba(0,255,255,0.08)" if color_cyan else "rgba(255,0,255,0.08)"

    pct = value / max_val
    green_thresh = 0.67 * max_val
    yellow_thresh = 0.4 * max_val

    if value >= green_thresh:
        bar_color = "rgba(0,255,136,0.9)"
    elif value >= yellow_thresh:
        bar_color = "rgba(255,221,0,0.9)"
    else:
        bar_color = "rgba(255,51,102,0.9)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(
            font=dict(family="Courier New", color="#00ffff", size=28),
            suffix=f"/{max_val}",
        ),
        title=dict(
            text=label,
            font=dict(family="Courier New", color="rgba(0,255,255,0.6)", size=11),
        ),
        gauge=dict(
            axis=dict(
                range=[0, max_val],
                tickfont=dict(color="rgba(0,255,255,0.5)", size=9),
                tickcolor="rgba(0,255,255,0.3)",
            ),
            bar=dict(color=bar_color, thickness=0.25),
            bgcolor=bg_color,
            borderwidth=1,
            bordercolor="rgba(0,255,255,0.2)",
            steps=[
                dict(range=[0, yellow_thresh], color="rgba(255,51,102,0.07)"),
                dict(range=[yellow_thresh, green_thresh], color="rgba(255,221,0,0.07)"),
                dict(range=[green_thresh, max_val], color="rgba(0,255,136,0.07)"),
            ],
            threshold=dict(
                line=dict(color="#ff00ff", width=2),
                thickness=0.6,
                value=green_thresh,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Courier New"),
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


# =============================================================================
# TAB 1 — SCREENER
# =============================================================================

def tab_screener():
    inject_css()
    section_title("Market Scanner — 4-Layer Regime Scoring")

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([1, 1, 1, 0.6])

    with col1:
        market_opt = st.selectbox(
            "SELECT MARKET",
            ["All", "Commodity", "S&P 500", "US Mid Cap", "US Small Cap",
             "Stockholm", "Oslo", "Copenhagen", "Helsinki",
             "Europe", "Canada", "Junior Miners"],
            key="screener_market",
        )

    with col2:
        min_score = st.slider(
            "MINIMUM REGIME SCORE",
            min_value=0, max_value=125,
            value=50,
            key="screener_min_score",
        )

    with col3:
        screener_preset = st.selectbox(
            "PRESET",
            ["Auto-detect", "Universal",
             "XLE", "XLB", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLU", "XLC",
             "OMX Stockholm", "OMX Copenhagen", "Oslo OSEBX", "OMX Helsinki",
             "OXY", "GOLD", "NEM", "XOM", "GLD"],
            key="screener_preset",
            help="Select parameter preset or Auto-detect based on ticker",
        )

    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("⚡ SCAN", key="screener_run", use_container_width=True)

    st.markdown("---")

    # ── Status / Legend ───────────────────────────────────────────────────────
    lcol, rcol = st.columns([3, 1])
    with rcol:
        st.markdown("""
        <div style="border:1px solid rgba(0,255,255,0.15);border-radius:6px;padding:12px;font-size:11px;letter-spacing:1px;">
            <div style="color:rgba(0,255,255,0.5);letter-spacing:3px;margin-bottom:8px;">SCORE LEGEND</div>
            <div><span class="score-green">■</span> &nbsp;STRONG &nbsp;&ge; 70</div>
            <div><span class="score-yellow">■</span> &nbsp;MODERATE &nbsp;&ge; 50</div>
            <div><span class="score-red">■</span> &nbsp;WEAK &nbsp;&lt; 50</div>
            <div style="margin-top:8px;"><span class="entry-yes">✦ YES</span> &nbsp;= Entry Signal</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Run screener ──────────────────────────────────────────────────────────
    if run_btn:
        try:
            from wolf_shadow_screener import run_screener, MARKETS

            market_map = {
                "All": None,
                "Commodity": ["commodity"],
                "S&P 500": ["sp500"],
                "US Mid Cap": ["us_midcap"],
                "US Small Cap": ["us_smallcap"],
                "Stockholm": ["stockholm"],
                "Oslo": ["oslo"],
                "Copenhagen": ["copenhagen"],
                "Helsinki": ["helsinki"],
                "Europe": ["europe"],
                "Canada": ["canada"],
                "Junior Miners": ["junior_miners"],
            }
            selected_markets = market_map[market_opt]

            with st.spinner("🐺 WOLF IS HUNTING... scanning markets..."):
                df_results = run_screener(markets=selected_markets, min_score=min_score)

        except Exception as e:
            st.error(f"Screener error: {e}")
            return

        if df_results is None or df_results.empty:
            st.warning("No stocks matched the criteria. Try lowering the minimum score.")
            return

        # ── Summary KPIs ──────────────────────────────────────────────────────
        n_total   = len(df_results)
        n_strong  = len(df_results[df_results["Total Score"] >= 70])
        n_entry   = len(df_results[df_results["Entry Signal"] == "YES"])
        top_score = int(df_results["Total Score"].max())
        top_tick  = df_results.iloc[0]["Ticker"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("STOCKS SCANNED", n_total)
        k2.metric("STRONG REGIME (≥70)", n_strong,
                  delta=f"{n_strong/n_total*100:.0f}% of results" if n_total else None)
        k3.metric("ENTRY SIGNALS", n_entry,
                  delta="🐺 Active" if n_entry > 0 else "None found")
        k4.metric("TOP SCORE", f"{top_score} — {top_tick}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Build display table ───────────────────────────────────────────────
        _adx_cols = ["ADX"] if "ADX" in df_results.columns else []
        _preset_cols = ["Preset"] if "Preset" in df_results.columns else []
        display_cols = (
            ["Ticker", "Name"] + _preset_cols +
            ["Total Score", "Market(30)", "Sector(30)",
             "Stock(50)", "Ichi(15)", "EMA Stack", "Entry Signal", "RSI"]
            + _adx_cols + ["Close"]
        )
        df_display = df_results[display_cols].copy()

        # Style rows by score and entry signal
        def style_screener_row(row):
            total = row["Total Score"]
            is_entry = row["Entry Signal"] == "YES"
            if is_entry:
                bg = "background-color: rgba(0,255,136,0.12)"
            elif total >= 70:
                bg = "background-color: rgba(0,255,136,0.06)"
            elif total >= 50:
                bg = "background-color: rgba(255,221,0,0.05)"
            else:
                bg = "background-color: rgba(255,51,102,0.04)"
            return [bg] * len(row)

        def style_score_cell(val):
            if val >= 70:
                return "color: #00ff88; font-weight: bold"
            elif val >= 50:
                return "color: #ffdd00; font-weight: bold"
            else:
                return "color: #ff3366; font-weight: bold"

        def style_entry_cell(val):
            if val == "YES":
                return "color: #00ff88; font-weight: bold"
            return "color: rgba(255,255,255,0.3)"

        _styler = df_display.style
        _map_fn = _styler.map if hasattr(_styler, "map") else _styler.applymap
        styled_df = (
            _styler
            .apply(style_screener_row, axis=1)
            .format({"RSI": "{:.1f}", "Close": "{:.2f}"})
        )
        styled_df = _map_fn(style_score_cell, subset=["Total Score"])
        styled_df = _map_fn(style_entry_cell, subset=["Entry Signal"])

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            height=min(600, 40 + len(df_display) * 35),
            column_config={
                "Total Score": st.column_config.NumberColumn(
                    "TOTAL SCORE",
                    help="Composite regime score (max 125)",
                    format="%d",
                ),
                "Ticker":       st.column_config.TextColumn("TICKER"),
                "Name":         st.column_config.TextColumn("NAME"),
                "Market(30)":   st.column_config.NumberColumn("MKT(30)",  format="%d"),
                "Sector(30)":   st.column_config.NumberColumn("SEC(30)",  format="%d"),
                "Stock(50)":    st.column_config.NumberColumn("STK(50)",  format="%d"),
                "Ichi(15)":     st.column_config.NumberColumn("ICHI(15)", format="%d"),
                "EMA Stack":    st.column_config.TextColumn("EMA STACK"),
                "Entry Signal": st.column_config.TextColumn("ENTRY"),
                "RSI":          st.column_config.NumberColumn("RSI",    format="%.1f"),
                "ADX":          st.column_config.NumberColumn("ADX",    format="%.1f"),
                "Preset":       st.column_config.TextColumn("PRESET"),
                "Close":        st.column_config.NumberColumn("CLOSE",  format="%.2f"),
            },
        )

        # ── Entry signals detail ──────────────────────────────────────────────
        entries = df_results[df_results["Entry Signal"] == "YES"]
        if not entries.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            section_title(f"Active Entry Signals ({len(entries)})")
            entry_display = entries[[
                "Ticker", "Name", "Total Score", "Entry Zone",
                "SL (1.5 ATR)", "TP1 (2R)", "TP2 (3R)", "RSI", "ATR"
            ]].copy()
            st.dataframe(
                entry_display.style
                .set_properties(**{
                    "background-color": "rgba(0,255,136,0.08)",
                    "color": "#00ff88",
                    "font-family": "Courier New, monospace",
                    "font-size": "12px",
                })
                .set_table_styles([{
                    "selector": "thead th",
                    "props": [
                        ("background-color", "#07071a"),
                        ("color", "rgba(0,255,255,0.6)"),
                        ("font-size", "10px"),
                        ("letter-spacing", "2px"),
                    ]
                }]),
                use_container_width=True,
                hide_index=True,
            )
    else:
        # Placeholder before first scan
        st.markdown("""
        <div style="
            text-align:center;
            padding:80px 20px;
            border:1px dashed rgba(0,255,255,0.15);
            border-radius:8px;
            margin-top:20px;
        ">
            <div style="font-size:48px;margin-bottom:16px;">🐺</div>
            <div style="color:rgba(0,255,255,0.5);letter-spacing:4px;font-size:14px;margin-bottom:8px;">
                READY TO HUNT
            </div>
            <div style="color:rgba(0,255,255,0.3);font-size:11px;letter-spacing:2px;">
                SELECT MARKET · SET MIN SCORE · CLICK SCAN
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 2 — BACKTEST
# =============================================================================

def tab_backtest():
    section_title("Strategy Backtester — Historical Performance Analysis")

    col1, col2, col3, col4 = st.columns([1.2, 0.8, 1, 1.2])

    with col1:
        ticker_input = st.text_input(
            "TICKER SYMBOL",
            value="XOM",
            key="bt_ticker",
            placeholder="e.g. XOM, NVDA, HEXA-B.ST",
        ).strip().upper()

    with col2:
        years_opt = st.selectbox(
            "BACKTEST PERIOD",
            [1, 2, 3, 5],
            index=2,
            format_func=lambda x: f"{x} Year{'s' if x > 1 else ''}",
            key="bt_years",
        )

    with col3:
        sector_etf = st.selectbox(
            "SECTOR ETF (REGIME)",
            ["XLE", "XLB", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLU", "XLC"],
            key="bt_sector",
        )

    with col4:
        bt_preset = st.selectbox(
            "PRESET",
            ["Auto-detect", "Universal",
             "XLE", "XLB", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLU", "XLC",
             "OMX Stockholm", "OMX Copenhagen", "Oslo OSEBX", "OMX Helsinki",
             "OXY", "GOLD", "NEM", "XOM", "GLD"],
            key="bt_preset",
            help="Select parameter preset or Auto-detect based on ticker",
        )

    run_btn = st.button("▶ RUN BACKTEST", key="bt_run", use_container_width=False)
    st.markdown("---")

    if run_btn:
        if not ticker_input:
            st.warning("Please enter a ticker symbol.")
            return

        # ── v3 preset parameters (21 presets) ─────────────────────────
        _PRESET_PARAMS_BT = {
            # US Sector ETFs
            "XLE":  {"atr_mult":2.1, "adx_thresh":5,  "tp1_r":1.75, "tp1_pct":0.10, "tp2_r":5.5,  "tp2_pct":0.10, "core_pct":0.40},
            "XLB":  {"atr_mult":1.5, "adx_thresh":21, "tp1_r":2.25, "tp1_pct":0.10, "tp2_r":6.0,  "tp2_pct":0.15, "core_pct":0.55},
            "XLF":  {"atr_mult":3.2, "adx_thresh":28, "tp1_r":2.5,  "tp1_pct":0.25, "tp2_r":3.5,  "tp2_pct":0.15, "core_pct":0.55},
            "XLK":  {"atr_mult":2.3, "adx_thresh":3,  "tp1_r":2.5,  "tp1_pct":0.20, "tp2_r":5.25, "tp2_pct":0.05, "core_pct":0.70},
            "XLV":  {"atr_mult":2.0, "adx_thresh":2,  "tp1_r":4.0,  "tp1_pct":0.05, "tp2_r":4.5,  "tp2_pct":0.20, "core_pct":0.40},
            "XLI":  {"atr_mult":1.8, "adx_thresh":6,  "tp1_r":3.75, "tp1_pct":0.10, "tp2_r":4.0,  "tp2_pct":0.10, "core_pct":0.60},
            "XLY":  {"atr_mult":2.0, "adx_thresh":2,  "tp1_r":3.0,  "tp1_pct":0.20, "tp2_r":3.75, "tp2_pct":0.25, "core_pct":0.55},
            "XLP":  {"atr_mult":2.3, "adx_thresh":0,  "tp1_r":2.25, "tp1_pct":0.05, "tp2_r":5.75, "tp2_pct":0.25, "core_pct":0.50},
            "XLRE": {"atr_mult":2.5, "adx_thresh":1,  "tp1_r":3.5,  "tp1_pct":0.05, "tp2_r":5.25, "tp2_pct":0.20, "core_pct":0.40},
            "XLU":  {"atr_mult":1.1, "adx_thresh":13, "tp1_r":3.5,  "tp1_pct":0.15, "tp2_r":4.5,  "tp2_pct":0.20, "core_pct":0.45},
            "XLC":  {"atr_mult":3.1, "adx_thresh":14, "tp1_r":3.25, "tp1_pct":0.25, "tp2_r":3.5,  "tp2_pct":0.05, "core_pct":0.60},
            # Nordic Exchanges
            "OMX Stockholm":  {"atr_mult":2.0, "adx_thresh":18, "tp1_r":3.15, "tp1_pct":0.20, "tp2_r":4.8,  "tp2_pct":0.22, "core_pct":0.46},
            "OMX Copenhagen": {"atr_mult":2.1, "adx_thresh":12, "tp1_r":2.2,  "tp1_pct":0.12, "tp2_r":5.15, "tp2_pct":0.12, "core_pct":0.48},
            "Oslo OSEBX":     {"atr_mult":2.3, "adx_thresh":20, "tp1_r":3.55, "tp1_pct":0.16, "tp2_r":4.8,  "tp2_pct":0.19, "core_pct":0.49},
            "OMX Helsinki":   {"atr_mult":2.1, "adx_thresh":11, "tp1_r":3.05, "tp1_pct":0.21, "tp2_r":5.55, "tp2_pct":0.15, "core_pct":0.54},
            # Individual stocks
            "OXY":  {"atr_mult":2.8, "adx_thresh":27, "tp1_r":3.0,  "tp1_pct":0.20, "tp2_r":5.5,  "tp2_pct":0.25, "core_pct":0.70},
            "GOLD": {"atr_mult":1.5, "adx_thresh":16, "tp1_r":1.75, "tp1_pct":0.20, "tp2_r":5.75, "tp2_pct":0.05, "core_pct":0.50},
            "NEM":  {"atr_mult":3.1, "adx_thresh":17, "tp1_r":3.0,  "tp1_pct":0.05, "tp2_r":4.25, "tp2_pct":0.25, "core_pct":0.60},
            "XOM":  {"atr_mult":2.7, "adx_thresh":27, "tp1_r":3.5,  "tp1_pct":0.10, "tp2_r":5.5,  "tp2_pct":0.10, "core_pct":0.70},
            "GLD":  {"atr_mult":2.6, "adx_thresh":7,  "tp1_r":1.75, "tp1_pct":0.10, "tp2_r":5.0,  "tp2_pct":0.20, "core_pct":0.60},
            # Universal fallback
            "Universal": {"atr_mult":2.5, "adx_thresh":19, "tp1_r":2.6,  "tp1_pct":0.13, "tp2_r":5.2,  "tp2_pct":0.17, "core_pct":0.62},
        }
        if bt_preset == "Auto-detect":
            if ticker_input in _PRESET_PARAMS_BT:
                _bt_pkey = ticker_input
            elif ticker_input.endswith(".ST"):
                _bt_pkey = "OMX Stockholm"
            elif ticker_input.endswith(".OL"):
                _bt_pkey = "Oslo OSEBX"
            elif ticker_input.endswith(".CO"):
                _bt_pkey = "OMX Copenhagen"
            elif ticker_input.endswith(".HE"):
                _bt_pkey = "OMX Helsinki"
            else:
                _bt_pkey = "Universal"
        else:
            _bt_pkey = bt_preset
        _bt_p = _PRESET_PARAMS_BT.get(_bt_pkey, _PRESET_PARAMS_BT["Universal"])
        V2_CONFIG = {
            "atr_mult":        _bt_p["atr_mult"],
            "adx_threshold":   _bt_p["adx_thresh"],
            "tp1_r":           _bt_p["tp1_r"],
            "tp1_pct":         _bt_p["tp1_pct"],
            "tp2_r":           _bt_p["tp2_r"],
            "tp2_pct":         _bt_p["tp2_pct"],
            "daily_breaker":  -0.08,
            "core_exit_bars":   3,
            "trail_exit":      "kijun_ema10",
        }

        _backtest_module = None
        try:
            import wolf_shadow_backtest as _bt_mod
            # Override CONFIG values with v2.2 preset parameters
            if hasattr(_bt_mod, "CONFIG"):
                _bt_mod.CONFIG["atr_mult"]       = V2_CONFIG["atr_mult"]
                _bt_mod.CONFIG["adx_threshold"]  = V2_CONFIG["adx_threshold"]
                _bt_mod.CONFIG["tp1_rr"]         = V2_CONFIG["tp1_r"]
                _bt_mod.CONFIG["tp1_pct"]        = V2_CONFIG["tp1_pct"]
                _bt_mod.CONFIG["tp2_rr"]         = V2_CONFIG["tp2_r"]
                _bt_mod.CONFIG["tp2_pct"]        = V2_CONFIG["tp2_pct"]
                _bt_mod.CONFIG["daily_breaker"]  = V2_CONFIG["daily_breaker"]
                _bt_mod.CONFIG["core_exit_bars"] = V2_CONFIG["core_exit_bars"]
            _backtest_module = _bt_mod
            from wolf_shadow_backtest import (
                add_indicators, calc_regime_scores,
                fetch_regime_data, Backtester, validate_criteria,
                ACCEPT_CRITERIA,
            )
            _use_module = True
        except ImportError:
            _use_module = False

        with st.spinner(f"🐺 Running backtest for {ticker_input} ({years_opt}y)..."):
            try:
                # Fetch data
                end = datetime.now()
                start = end - timedelta(days=years_opt * 365 + 100)
                df_raw = yf.download(ticker_input, start=start, end=end, progress=False, auto_adjust=True)

                if isinstance(df_raw.columns, pd.MultiIndex):
                    df_raw.columns = df_raw.columns.get_level_values(0)

                if df_raw is None or len(df_raw) < 50:
                    st.error(f"Insufficient data for {ticker_input} ({len(df_raw) if df_raw is not None else 0} bars). Need at least 50 bars.")
                    return

                if _use_module and len(df_raw) >= 250:
                    # ── Module path with v2 CONFIG already applied above ──
                    spy_df, sec_df = fetch_regime_data("SPY", sector_etf, years_opt)
                    df_proc = add_indicators(df_raw.copy())
                    df_proc = calc_regime_scores(spy_df, sec_df, df_proc)
                    df_proc = df_proc.dropna()

                    bt = Backtester(df_proc)
                    results = bt.run()

                    if results is None:
                        st.warning("No trades were generated for this period/ticker.")
                        return

                else:
                    # ── Simplified inline backtest using v2 logic ─────────────
                    if not _use_module:
                        st.info("⚠️ Backtester module not found — using simplified inline backtest with v2 parameters.")

                    df = df_raw.copy()
                    c = df["Close"]

                    # Indicators
                    df["ema10"]  = c.ewm(span=10,  adjust=False).mean()
                    df["ema21"]  = c.ewm(span=21,  adjust=False).mean()
                    df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
                    df["ema200"] = c.ewm(span=200, adjust=False).mean()

                    delta = c.diff()
                    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
                    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
                    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

                    tr = pd.concat([
                        df["High"] - df["Low"],
                        (df["High"] - c.shift()).abs(),
                        (df["Low"]  - c.shift()).abs(),
                    ], axis=1).max(axis=1)
                    df["atr"] = tr.ewm(com=13, adjust=False).mean()

                    # Kijun (26-period)
                    df["kijun"] = (df["High"].rolling(26).max() + df["Low"].rolling(26).min()) / 2

                    df.dropna(inplace=True)
                    df = df.iloc[years_opt * -252:] if len(df) > years_opt * 252 else df

                    # v2 backtest loop
                    capital   = 100_000.0
                    equity    = [capital]
                    trades_list = []
                    position  = None
                    atr_m     = V2_CONFIG["atr_mult"]
                    tp1_r     = V2_CONFIG["tp1_r"]
                    tp1_pct   = V2_CONFIG["tp1_pct"]
                    tp2_r     = V2_CONFIG["tp2_r"]
                    tp2_pct   = V2_CONFIG["tp2_pct"]
                    daily_brk = V2_CONFIG["daily_breaker"]
                    core_bars = V2_CONFIG["core_exit_bars"]

                    bars_below_ema50 = 0
                    daily_block      = False

                    for i in range(1, len(df)):
                        row   = df.iloc[i]
                        prev  = df.iloc[i - 1]
                        price = float(row["Close"])

                        # Daily breaker: only blocks new entries
                        day_ret = (price / float(prev["Close"]) - 1)
                        daily_block = day_ret <= daily_brk

                        if position is not None:
                            # Track bars below EMA50 for core exit (v2: 3 bars)
                            if price < float(row["ema50"]):
                                bars_below_ema50 += 1
                            else:
                                bars_below_ema50 = 0

                            pos = position
                            exit_price = None
                            exit_reason = ""

                            # TP1 hit
                            if not pos.get("tp1_hit") and price >= pos["tp1"]:
                                trim_shares = pos["shares"] * tp1_pct
                                capital += trim_shares * pos["tp1"]
                                pos["shares"] -= trim_shares
                                pos["tp1_hit"] = True

                            # TP2 hit
                            if not pos.get("tp2_hit") and price >= pos["tp2"]:
                                trim_shares = pos["shares"] * tp2_pct
                                capital += trim_shares * pos["tp2"]
                                pos["shares"] -= trim_shares
                                pos["tp2_hit"] = True

                            # Stop loss
                            if price <= pos["sl"]:
                                exit_price  = pos["sl"]
                                exit_reason = "STOP"

                            # CORE exit: close < EMA50 for 3 bars (v2)
                            elif bars_below_ema50 >= core_bars:
                                exit_price  = price
                                exit_reason = "EMA50_EXIT"

                            # TRIM trail: close < kijun AND close < ema10
                            elif price < float(row["kijun"]) and price < float(row["ema10"]):
                                exit_price  = price
                                exit_reason = "TRAIL_EXIT"

                            if exit_price is not None:
                                capital += pos["shares"] * exit_price
                                pnl = capital - pos["capital_at_entry"]
                                pnl_pct = (exit_price / pos["entry"] - 1) * 100
                                trades_list.append({
                                    "entry_date":  pos["entry_date"],
                                    "exit_date":   df.index[i],
                                    "entry_price": pos["entry"],
                                    "exit_price":  exit_price,
                                    "pnl":         round(pnl, 2),
                                    "pnl_pct":     round(pnl_pct, 2),
                                    "exit_reason": exit_reason,
                                    "bars_held":   i - pos["entry_idx"],
                                })
                                position = None
                                bars_below_ema50 = 0

                        # Entry: EMA stack + RSI 45-70, not blocked by daily breaker
                        if (
                            position is None
                            and not daily_block
                            and float(row["ema10"]) > float(row["ema21"]) > float(row["ema50"]) > float(row["ema200"])
                            and 45 < float(row["rsi"]) < 70
                            and price > float(row["ema50"])
                        ):
                            atr_v = float(row["atr"])
                            sl    = price - atr_m * atr_v
                            risk  = price - sl
                            shares = (capital * 0.02) / risk if risk > 0 else 0
                            if shares > 0:
                                cost = shares * price
                                if cost <= capital:
                                    capital -= cost
                                    position = {
                                        "entry":            price,
                                        "entry_date":       df.index[i],
                                        "entry_idx":        i,
                                        "shares":           shares,
                                        "sl":               sl,
                                        "tp1":              price + tp1_r * risk,
                                        "tp2":              price + tp2_r * risk,
                                        "tp1_hit":          False,
                                        "tp2_hit":          False,
                                        "capital_at_entry": capital + cost,
                                    }

                        equity.append(capital + (position["shares"] * price if position else 0))

                    # Close any open position at end
                    if position is not None:
                        final_price = float(df.iloc[-1]["Close"])
                        capital += position["shares"] * final_price
                        pnl = capital - position["capital_at_entry"]
                        trades_list.append({
                            "entry_date":  position["entry_date"],
                            "exit_date":   df.index[-1],
                            "entry_price": position["entry"],
                            "exit_price":  final_price,
                            "pnl":         round(pnl, 2),
                            "pnl_pct":     round((final_price / position["entry"] - 1) * 100, 2),
                            "exit_reason": "END",
                            "bars_held":   len(df) - position["entry_idx"],
                        })

                    trades = pd.DataFrame(trades_list)
                    eq_arr = np.array(equity)
                    eq_idx = df.index[:len(eq_arr)]
                    if len(eq_arr) > len(eq_idx):
                        eq_arr = eq_arr[:len(eq_idx)]
                    eq_df  = pd.DataFrame({"equity": eq_arr}, index=eq_idx)

                    peak  = eq_df["equity"].cummax()
                    dd_ser = (eq_df["equity"] - peak) / peak

                    total_ret  = (eq_arr[-1] / 100_000 - 1) * 100
                    daily_ret  = eq_df["equity"].pct_change().dropna()
                    returns    = daily_ret

                    if len(trades) > 0:
                        wins     = trades[trades["pnl"] > 0]
                        losses   = trades[trades["pnl"] <= 0]
                        winrate  = len(wins) / len(trades) * 100
                        avg_win  = wins["pnl"].mean() if len(wins) > 0 else 0
                        avg_loss = losses["pnl"].mean() if len(losses) > 0 else -1
                        pf       = abs(wins["pnl"].sum() / losses["pnl"].sum()) if losses["pnl"].sum() != 0 else 99
                        avg_rr   = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                        avg_bars = trades["bars_held"].mean() if "bars_held" in trades.columns else 0
                    else:
                        winrate = pf = avg_rr = avg_bars = 0

                    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
                    neg    = daily_ret[daily_ret < 0]
                    sortino = float(daily_ret.mean() / neg.std() * np.sqrt(252)) if len(neg) > 1 and neg.std() > 0 else 0
                    max_dd = float(dd_ser.min() * 100)
                    cagr   = ((eq_arr[-1] / 100_000) ** (252 / max(len(eq_arr), 1)) - 1) * 100
                    calmar = abs(cagr / max_dd) if max_dd < 0 else 0

                    metrics = {
                        "Total Return %":  round(total_ret, 2),
                        "Sharpe Ratio":    round(sharpe,  2),
                        "Profit Factor":   round(pf,      2),
                        "Winrate %":       round(winrate, 1),
                        "Max Drawdown %":  round(max_dd,  2),
                        "Calmar Ratio":    round(calmar,  2),
                        "CAGR %":          round(cagr,    2),
                        "Sortino Ratio":   round(sortino, 2),
                        "Avg R:R":         round(avg_rr,  2),
                        "Total Trades":    len(trades),
                        "Avg Bars Held":   round(avg_bars, 1),
                        "Final Equity":    round(float(eq_arr[-1]), 0),
                    }

                    results = {
                        "metrics":  metrics,
                        "equity":   eq_df,
                        "drawdown": dd_ser,
                        "trades":   trades,
                        "returns":  returns,
                    }

            except Exception as e:
                st.error(f"Backtest error: {e}")
                import traceback; st.code(traceback.format_exc())
                return

        metrics = results["metrics"]
        eq_df   = results["equity"]
        dd_ser  = results["drawdown"]
        trades  = results["trades"]
        returns = results["returns"]

        # ── Metric cards (row 1) ───────────────────────────────────────────────
        section_title("Performance Metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)

        ret_val   = metrics["Total Return %"]
        sharpe    = metrics["Sharpe Ratio"]
        pf        = metrics["Profit Factor"]
        winrate   = metrics["Winrate %"]
        maxdd     = metrics["Max Drawdown %"]
        calmar    = metrics["Calmar Ratio"]

        m1.metric("TOTAL RETURN",   f"{ret_val:+.1f}%",
                  delta="▲ PROFIT" if ret_val > 0 else "▼ LOSS")
        m2.metric("SHARPE RATIO",   f"{sharpe:.2f}",
                  delta="✓ PASS" if sharpe >= 1.5 else "✗ FAIL")
        m3.metric("PROFIT FACTOR",  f"{pf:.2f}",
                  delta="✓ PASS" if pf >= 1.5 else "✗ FAIL")
        m4.metric("WIN RATE",       f"{winrate:.1f}%",
                  delta="✓ PASS" if winrate >= 45 else "✗ FAIL")
        m5.metric("MAX DRAWDOWN",   f"{maxdd:.1f}%",
                  delta="✓ PASS" if maxdd >= -15 else "✗ FAIL")
        m6.metric("CALMAR RATIO",   f"{calmar:.2f}",
                  delta="✓ PASS" if calmar >= 1.0 else "✗ FAIL")

        # ── Row 2 KPIs ────────────────────────────────────────────────────────
        x1, x2, x3, x4, x5, x6 = st.columns(6)
        x1.metric("CAGR",          f"{metrics['CAGR %']:.1f}%")
        x2.metric("SORTINO",       f"{metrics['Sortino Ratio']:.2f}")
        x3.metric("AVG R:R",       f"{metrics['Avg R:R']:.2f}")
        x4.metric("TOTAL TRADES",  f"{metrics['Total Trades']}")
        x5.metric("AVG BARS HELD", f"{metrics['Avg Bars Held']:.1f}")
        x6.metric("FINAL EQUITY",  f"${metrics['Final Equity']:,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────────────────
        section_title("Equity & Drawdown")
        st.plotly_chart(build_equity_chart(eq_df, ticker_input),
                        use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(build_drawdown_chart(dd_ser),
                        use_container_width=True, config={"displayModeBar": False})

        # ── Monthly Heatmap ───────────────────────────────────────────────────
        section_title("Monthly Returns")
        if len(returns) > 20:
            st.plotly_chart(build_monthly_heatmap(returns, eq_df),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Insufficient data for monthly heatmap.")

        # ── Accept criteria table ─────────────────────────────────────────────
        section_title("Accept Criteria Validation")

        # Inline fallback for validate_criteria when module not available
        if not _use_module or "validate_criteria" not in dir():
            _thresholds = [
                ("Total Return %",  metrics.get("Total Return %", 0),  ">= 20%",   lambda v: v >= 20),
                ("Sharpe Ratio",    metrics.get("Sharpe Ratio",   0),  ">= 1.5",   lambda v: v >= 1.5),
                ("Profit Factor",   metrics.get("Profit Factor",  0),  ">= 1.5",   lambda v: v >= 1.5),
                ("Winrate %",       metrics.get("Winrate %",      0),  ">= 45%",   lambda v: v >= 45),
                ("Max Drawdown %",  metrics.get("Max Drawdown %", 0),  ">= -15%",  lambda v: v >= -15),
                ("Calmar Ratio",    metrics.get("Calmar Ratio",   0),  ">= 1.0",   lambda v: v >= 1.0),
            ]
            criteria_df = pd.DataFrame([
                {"Metric": m, "Value": round(float(v), 2), "Threshold": t, "Status": "PASS" if fn(v) else "FAIL"}
                for m, v, t, fn in _thresholds
            ])
        else:
            criteria_df = validate_criteria(metrics)

        passed_count = len(criteria_df[criteria_df["Status"] == "PASS"])
        total_count = len(criteria_df)
        st.write(f"**Result: {passed_count}/{total_count} PASSED**")

        def color_criteria(row):
            if row["Status"] == "PASS":
                return ["background-color: rgba(0,180,80,0.15); color: #00cc66"] * len(row)
            return ["background-color: rgba(200,50,80,0.15); color: #ff6666"] * len(row)

        st.dataframe(
            criteria_df.style.apply(color_criteria, axis=1),
            use_container_width=True,
            hide_index=True,
            height=280,
        )

        # ── Trade log ─────────────────────────────────────────────────────────
        if not trades.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            section_title(f"Trade Log ({len(trades)} trades)")

            # Color trades by P&L
            trade_display = trades.copy()
            if "entry_date" in trade_display.columns:
                trade_display["entry_date"] = pd.to_datetime(trade_display["entry_date"]).dt.strftime("%Y-%m-%d")
            if "exit_date" in trade_display.columns:
                trade_display["exit_date"] = pd.to_datetime(trade_display["exit_date"]).dt.strftime("%Y-%m-%d")

            for col in ["entry_price", "exit_price", "pnl"]:
                if col in trade_display.columns:
                    trade_display[col] = trade_display[col].round(2)
            if "pnl_pct" in trade_display.columns:
                trade_display["pnl_pct"] = trade_display["pnl_pct"].round(2)

            def style_trades(row):
                if "pnl" in row.index:
                    if row["pnl"] > 0:
                        return ["background-color: rgba(0,255,136,0.07); color: #00ff88"] * len(row)
                    else:
                        return ["background-color: rgba(255,51,102,0.07); color: #ff6688"] * len(row)
                return [""] * len(row)

            st.dataframe(
                trade_display.style.apply(style_trades, axis=1)
                .set_properties(**{
                    "font-family": "Courier New, monospace",
                    "font-size": "11px",
                }),
                use_container_width=True,
                hide_index=True,
                height=350,
            )
    else:
        st.markdown("""
        <div style="
            text-align:center;
            padding:80px 20px;
            border:1px dashed rgba(255,0,255,0.15);
            border-radius:8px;
            margin-top:20px;
        ">
            <div style="font-size:48px;margin-bottom:16px;">📊</div>
            <div style="color:rgba(255,0,255,0.5);letter-spacing:4px;font-size:14px;margin-bottom:8px;">
                AWAITING COMMAND
            </div>
            <div style="color:rgba(255,0,255,0.3);font-size:11px;letter-spacing:2px;">
                ENTER TICKER · SELECT PERIOD · RUN BACKTEST
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 3 — REGIME MONITOR
# =============================================================================

def tab_regime():
    section_title("Real-Time Regime Monitor — Live Market Intelligence")

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1.5])
    with ctrl1:
        watch_ticker = st.text_input("STOCK TICKER", value="XOM", key="reg_ticker").strip().upper()
    with ctrl2:
        watch_sector = st.selectbox(
            "SECTOR ETF",
            ["XLE", "XLB", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLU", "XLC"],
            key="reg_sector",
        )
    with ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_btn = st.button("🔄 REFRESH REGIME", key="reg_refresh", use_container_width=True)

    st.markdown("---")

    # Session state for regime data
    if "regime_data" not in st.session_state:
        st.session_state.regime_data = None
    if "regime_ticker" not in st.session_state:
        st.session_state.regime_ticker = None

    # Load on first visit or refresh
    should_load = (
        refresh_btn
        or st.session_state.regime_data is None
        or st.session_state.regime_ticker != watch_ticker
    )

    if should_load:
        with st.spinner(f"📡 Fetching live regime data for {watch_ticker}..."):
            try:
                # ── Inline helper functions (no external module needed) ────────
                def _ema(series, span):
                    return series.ewm(span=span, adjust=False).mean()

                def _rsi(series, period=14):
                    delta = series.diff()
                    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
                    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
                    rs = gain / loss.replace(0, np.nan)
                    return 100 - (100 / (1 + rs))

                def _atr(df, period=14):
                    high, low, close = df["High"], df["Low"], df["Close"]
                    tr = pd.concat([
                        high - low,
                        (high - close.shift()).abs(),
                        (low  - close.shift()).abs(),
                    ], axis=1).max(axis=1)
                    return tr.ewm(com=period - 1, adjust=False).mean()

                def _adx(df, period=14):
                    high, low, close = df["High"], df["Low"], df["Close"]
                    plus_dm  = high.diff()
                    minus_dm = -low.diff()
                    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
                    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
                    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
                    atr_v    = tr.ewm(span=period, adjust=False).mean()
                    plus_di  = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_v)
                    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_v)
                    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
                    return dx.ewm(span=period, adjust=False).mean()

                def _ichimoku(df):
                    """Returns (tenkan, kijun, span_a, span_b, chikou)."""
                    hi9  = df["High"].rolling(9).max()
                    lo9  = df["Low"].rolling(9).min()
                    hi26 = df["High"].rolling(26).max()
                    lo26 = df["Low"].rolling(26).min()
                    hi52 = df["High"].rolling(52).max()
                    lo52 = df["Low"].rolling(52).min()
                    tenkan  = (hi9  + lo9)  / 2
                    kijun   = (hi26 + lo26) / 2
                    span_a  = ((tenkan + kijun) / 2).shift(26)
                    span_b  = ((hi52 + lo52) / 2).shift(26)
                    chikou  = df["Close"].shift(-26)
                    return tenkan, kijun, span_a, span_b, chikou

                def _fetch(ticker, period="1y"):
                    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    return df if len(df) >= 50 else None

                def _score_market(spy_df):
                    """Score SPY market regime (max 30)."""
                    if spy_df is None or len(spy_df) < 60:
                        return 0
                    c = spy_df["Close"]
                    score = 0
                    e20  = _ema(c, 20).iloc[-1]
                    e50  = _ema(c, 50).iloc[-1]
                    e200 = _ema(c, 200).iloc[-1]
                    last = c.iloc[-1]
                    rsi  = _rsi(c).iloc[-1]
                    if last > e200:          score += 10
                    if e50  > e200:          score += 8
                    if last > e50:           score += 7
                    if rsi  > 50:            score += 5
                    return min(score, 30)

                def _score_sector(sec_df):
                    """Score sector ETF (max 30)."""
                    if sec_df is None or len(sec_df) < 60:
                        return 0
                    c = sec_df["Close"]
                    score = 0
                    e20  = _ema(c, 20).iloc[-1]
                    e50  = _ema(c, 50).iloc[-1]
                    e200 = _ema(c, 200).iloc[-1]
                    last = c.iloc[-1]
                    rsi  = _rsi(c).iloc[-1]
                    if last > e200:          score += 10
                    if e50  > e200:          score += 8
                    if last > e50:           score += 7
                    if rsi  > 50:            score += 5
                    return min(score, 30)

                def _score_stock(stk_df):
                    """Score individual stock and return full result dict (max 50 stock + 15 ichi)."""
                    if stk_df is None or len(stk_df) < 60:
                        return None
                    c = stk_df["Close"]
                    e10  = _ema(c, 10)
                    e21  = _ema(c, 21)
                    e50  = _ema(c, 50)
                    e200 = _ema(c, 200)
                    rsi_s = _rsi(c)
                    atr_s = _atr(stk_df)
                    tenkan, kijun, span_a, span_b, chikou = _ichimoku(stk_df)

                    adx_s    = _adx(stk_df)
                    last     = c.iloc[-1]
                    rsi_val  = float(rsi_s.iloc[-1])
                    atr_val  = float(atr_s.iloc[-1])
                    adx_val  = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0.0
                    e10_last = float(e10.iloc[-1])
                    e21_last = float(e21.iloc[-1])
                    e50_last = float(e50.iloc[-1])
                    e200_last= float(e200.iloc[-1])

                    # Stock score (max 50)
                    stk_score = 0
                    if last > e200_last:                         stk_score += 10
                    if e50_last > e200_last:                     stk_score += 8
                    if last > e50_last:                          stk_score += 8
                    ema_stack = (e10_last > e21_last > e50_last > e200_last)
                    if ema_stack:                                stk_score += 10
                    ema_trend = last > e21_last
                    if ema_trend:                                stk_score += 7
                    if 40 < rsi_val < 75:                        stk_score += 7
                    stk_score = min(stk_score, 50)

                    # Ichimoku score (max 15)
                    ichi_score = 0
                    sa = float(span_a.iloc[-1]) if not pd.isna(span_a.iloc[-1]) else 0
                    sb = float(span_b.iloc[-1]) if not pd.isna(span_b.iloc[-1]) else 0
                    tk = float(tenkan.iloc[-1]) if not pd.isna(tenkan.iloc[-1]) else 0
                    kj = float(kijun.iloc[-1])  if not pd.isna(kijun.iloc[-1])  else 0
                    cloud_top = max(sa, sb)
                    cloud_bot = min(sa, sb)
                    if last > cloud_top:                         ichi_score += 6
                    elif last > cloud_bot:                       ichi_score += 2
                    if tk > kj:                                  ichi_score += 5
                    if sa > sb:                                  ichi_score += 4
                    ichi_score = min(ichi_score, 15)

                    # Entry signal
                    sl_dist   = 1.5 * atr_val
                    entry_zone = last
                    sl_level  = last - sl_dist
                    risk      = sl_dist
                    tp1       = last + 2.0 * risk
                    tp2       = last + 3.0 * risk
                    has_entry = ema_stack and ema_trend and 45 < rsi_val < 70

                    return {
                        "stock_score": stk_score,
                        "ichi_score":  ichi_score,
                        "close":       float(last),
                        "rsi":         rsi_val,
                        "atr":         atr_val,
                        "adx":         round(adx_val, 1),
                        "ema_stack":   ema_stack,
                        "ema_trend":   ema_trend,
                        "has_entry":   has_entry,
                        "entry_zone":  entry_zone,
                        "sl_level":    sl_level,
                        "tp1_2R":      tp1,
                        "tp2_3R":      tp2,
                    }

                # ── Fetch data ────────────────────────────────────────────────
                spy_df   = _fetch("SPY",         period="1y")
                sec_df   = _fetch(watch_sector,  period="1y")
                stock_df = _fetch(watch_ticker,  period="1y")

                mkt_score  = _score_market(spy_df)
                sec_score  = _score_sector(sec_df)
                stk_result = _score_stock(stock_df)

                stk_score  = stk_result["stock_score"] if stk_result else 0
                ichi_score = stk_result["ichi_score"]   if stk_result else 0
                total      = mkt_score + sec_score + stk_score + ichi_score

                close_price = stk_result["close"]      if stk_result else 0
                rsi_val     = stk_result["rsi"]        if stk_result else 0
                ema_stack   = stk_result["ema_stack"]  if stk_result else False
                ema_trend   = stk_result["ema_trend"]  if stk_result else False
                has_entry   = stk_result["has_entry"]  if stk_result else False
                entry_zone  = stk_result["entry_zone"] if stk_result else 0
                sl_level    = stk_result["sl_level"]   if stk_result else 0
                tp1         = stk_result["tp1_2R"]     if stk_result else 0
                tp2         = stk_result["tp2_3R"]     if stk_result else 0
                atr_val     = stk_result["atr"]        if stk_result else 0
                adx_val_reg = stk_result["adx"]        if stk_result else 0

                st.session_state.regime_data = {
                    "mkt_score":   mkt_score,
                    "sec_score":   sec_score,
                    "stk_score":   stk_score,
                    "ichi_score":  ichi_score,
                    "total":       total,
                    "close":       close_price,
                    "rsi":         rsi_val,
                    "adx":         adx_val_reg,
                    "ema_stack":   ema_stack,
                    "ema_trend":   ema_trend,
                    "has_entry":   has_entry,
                    "entry_zone":  entry_zone,
                    "sl_level":    sl_level,
                    "tp1":         tp1,
                    "tp2":         tp2,
                    "atr":         atr_val,
                    "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state.regime_ticker = watch_ticker

            except Exception as e:
                st.error(f"Regime fetch error: {e}")
                import traceback; st.code(traceback.format_exc())
                return

    data = st.session_state.regime_data
    if data is None:
        return

    # ── Big regime score display ──────────────────────────────────────────────
    total = data["total"]
    if total >= 85:
        regime_label = "BULL REGIME"
        regime_color = "#00ff88"
    elif total >= 65:
        regime_label = "MODERATE BULL"
        regime_color = "#ffdd00"
    elif total >= 45:
        regime_label = "NEUTRAL"
        regime_color = "#ff9900"
    else:
        regime_label = "BEAR / AVOID"
        regime_color = "#ff3366"

    score_col, gauge_col = st.columns([1, 3])

    with score_col:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(0,255,255,0.06) 0%, rgba(255,0,255,0.03) 100%);
            border: 1px solid {regime_color};
            border-radius: 12px;
            padding: 32px 16px;
            text-align: center;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute; top: 0; left: 0; right: 0; height: 3px;
                background: linear-gradient(90deg, #00ffff, #ff00ff);
            "></div>
            <div style="font-size:11px;letter-spacing:4px;color:rgba(0,255,255,0.5);margin-bottom:8px;">
                REGIME SCORE
            </div>
            <div style="
                font-size: 80px;
                font-weight: 900;
                line-height: 1;
                color: {regime_color};
                text-shadow: 0 0 30px {regime_color}80, 0 0 60px {regime_color}30;
                font-family: 'Courier New', monospace;
            ">{total}</div>
            <div style="font-size:11px;letter-spacing:3px;color:rgba(0,255,255,0.4);margin-top:4px;">
                / 125 MAX
            </div>
            <div style="margin-top:16px;">
                <span style="
                    background: {regime_color}1a;
                    border: 1px solid {regime_color};
                    border-radius: 20px;
                    color: {regime_color};
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 3px;
                    padding: 5px 16px;
                    text-shadow: 0 0 8px {regime_color}80;
                ">{regime_label}</span>
            </div>
            <div style="
                margin-top: 20px;
                font-size: 10px;
                color: rgba(0,255,255,0.3);
                letter-spacing: 2px;
            ">
                {watch_ticker} &nbsp;·&nbsp; {data['timestamp']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 4 Gauges ──────────────────────────────────────────────────────────────
    with gauge_col:
        g1, g2, g3, g4 = st.columns(4)
        g1.plotly_chart(build_gauge(data["mkt_score"],  30, "MARKET (SPY)", color_cyan=True),
                        use_container_width=True, config={"displayModeBar": False})
        g2.plotly_chart(build_gauge(data["sec_score"],  30, f"SECTOR ({watch_sector})", color_cyan=False),
                        use_container_width=True, config={"displayModeBar": False})
        g3.plotly_chart(build_gauge(data["stk_score"],  50, f"STOCK ({watch_ticker})", color_cyan=True),
                        use_container_width=True, config={"displayModeBar": False})
        g4.plotly_chart(build_gauge(data["ichi_score"], 15, "ICHIMOKU", color_cyan=False),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Status indicators ─────────────────────────────────────────────────────
    section_title("System Status")

    s1, s2, s3, s4 = st.columns(4)

    def status_pill(label, active):
        if active:
            return f'<div class="status-active">{label}</div>'
        return f'<div class="status-inactive">{label}</div>'

    with s1:
        st.markdown("**EMA STACK**")
        st.markdown(status_pill("EMA STACK FULL" if data["ema_stack"] else "EMA STACK OFF",
                                data["ema_stack"]), unsafe_allow_html=True)

    with s2:
        st.markdown("**EMA TREND**")
        st.markdown(status_pill("TREND ACTIVE" if data["ema_trend"] else "NO TREND",
                                data["ema_trend"]), unsafe_allow_html=True)

    with s3:
        st.markdown("**ENTRY SIGNAL**")
        st.markdown(status_pill("ENTRY LIVE" if data["has_entry"] else "NO ENTRY",
                                data["has_entry"]), unsafe_allow_html=True)

    with s4:
        st.markdown("**REGIME GATE**")
        gate_ok = total >= 40
        st.markdown(status_pill("GATE OPEN" if gate_ok else "GATE CLOSED", gate_ok),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Live price info ───────────────────────────────────────────────────────
    section_title(f"Live Levels — {watch_ticker}")

    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("CLOSE PRICE",  f"{data['close']:.2f}")
    p2.metric("RSI (14)",     f"{data['rsi']:.1f}",
              delta="Overbought" if data['rsi'] > 70 else ("Oversold" if data['rsi'] < 30 else "Neutral"))
    _adx_disp = data.get("adx", 0)
    _adx_thresh_disp = 19  # Universal default
    p3.metric("ADX (14)",     f"{_adx_disp:.1f}",
              delta="Trending" if _adx_disp >= _adx_thresh_disp else "Weak trend")
    p4.metric("ATR (14)",     f"{data['atr']:.2f}")
    p5.metric("ENTRY ZONE",   f"{data['entry_zone']:.2f}")
    p6.metric("STOP LOSS",    f"{data['sl_level']:.2f}")

    t1, t2 = st.columns(2)
    t1.metric("TP1 (2R)",     f"{data['tp1']:.2f}",
              delta=f"+{((data['tp1']/data['close'])-1)*100:.1f}%" if data['close'] > 0 else None)
    t2.metric("TP2 (3R)",     f"{data['tp2']:.2f}",
              delta=f"+{((data['tp2']/data['close'])-1)*100:.1f}%" if data['close'] > 0 else None)

    # ── Mini score breakdown table ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_title("Score Breakdown")

    breakdown = [
        ("Market Regime", "SPY",          data["mkt_score"],  30),
        ("Sector ETF",    watch_sector,   data["sec_score"],  30),
        ("Stock Score",   watch_ticker,   data["stk_score"],  50),
        ("Ichimoku",      "Cloud/TK/CK",  data["ichi_score"], 15),
    ]

    bd_df = pd.DataFrame([
        {"Layer": label, "Instrument": sub, "Score": score, "Max": max_s, "Pct": round(score / max_s * 100)}
        for label, sub, score, max_s in breakdown
    ])
    bd_df.loc[len(bd_df)] = {"Layer": "TOTAL", "Instrument": "", "Score": total, "Max": 125, "Pct": round(total / 125 * 100)}
    bd_df["Score Display"] = bd_df.apply(lambda r: f"{int(r['Score'])}/{int(r['Max'])}", axis=1)
    bd_df["Fill %"] = bd_df["Pct"].astype(str) + "%"

    display_df = bd_df[["Layer", "Instrument", "Score Display", "Fill %"]].copy()

    # Map colors based on the Pct values (kept in bd_df)
    pct_values = bd_df["Pct"].tolist()

    def color_bd(row):
        idx = row.name
        pct = pct_values[idx] if idx < len(pct_values) else 0
        if pct >= 67:
            return ["background-color: rgba(0,180,80,0.15); color: #00cc66"] * len(row)
        elif pct >= 40:
            return ["background-color: rgba(200,200,0,0.1); color: #cccc00"] * len(row)
        return ["background-color: rgba(200,50,80,0.1); color: #ff6666"] * len(row)

    st.dataframe(
        display_df.style.apply(color_bd, axis=1),
        use_container_width=True,
        hide_index=True,
        height=230,
    )


# =============================================================================
# MAIN APP
# =============================================================================

def _tab_not_found(module_name: str, folder: str):
    """Show a friendly message when a module is not found."""
    st.warning(f"{module_name} module not found. Make sure the `{folder}/` folder is in the dashboard directory.")
    st.code(f"Expected: dashboard/{folder}/")


# =============================================================================
# CONSOLIDATED SCREENER TAB
# =============================================================================

def tab_screener_consolidated():
    """Unified Screener tab with dropdown: Swing / Long / OVTLYR."""
    mode = st.selectbox(
        "SCREENER MODE",
        ["Swing Screener", "Long Screener", "OVTLYR Screener"],
        key="screener_mode_select",
    )

    if mode == "Swing Screener":
        tab_screener()  # Existing swing screener — UNTOUCHED

    elif mode == "Long Screener":
        if CAGR_AVAILABLE:
            render_cagr_page()  # Existing long screener — UNTOUCHED
        else:
            _tab_not_found("Long Screener", "cagr")

    elif mode == "OVTLYR Screener":
        _render_ovtlyr_screener_ui()


def _render_ovtlyr_screener_ui():
    """OVTLYR Screener with z-score weighted scoring."""
    st.markdown(
        "<h2 style='color:#00ffff;letter-spacing:0.1em;'>"
        "OVTLYR SCREENER</h2>"
        "<p style='color:#4a4a6a;font-size:0.7rem;'>Z-score normalized · "
        "Weighted composite · Trend 30% + Momentum 25% + Vol 15% + Volume 15% + ADX 15%</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        universe = st.selectbox("Universe", ["Nordic", "US", "Canada", "All"], key="ovtlyr_univ")
    with col2:
        min_vol = st.number_input("Min Avg Volume", value=100_000, step=50_000, key="ovtlyr_minvol")
    with col3:
        top_n = st.number_input("Top N for Test", value=10, min_value=3, max_value=50, key="ovtlyr_topn")

    col_scan, col_test = st.columns(2)
    with col_scan:
        scan_clicked = st.button("↻  SCAN", key="ovtlyr_scan", use_container_width=True)
    with col_test:
        test_clicked = st.button("⚡ TEST TOP N", key="ovtlyr_test_topn", use_container_width=True)

    if scan_clicked or test_clicked:
        if not OVTLYR_SCREENER_AVAILABLE:
            st.error("OVTLYR Screener module not found.")
            return

        with st.spinner("🐺 Scanning universe..."):
            if universe == "All":
                dfs = []
                for u in ["Nordic", "US", "Canada"]:
                    df = run_ovtlyr_screener(u, min_vol)
                    if not df.empty:
                        dfs.append(df)
                results = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
                if not results.empty:
                    results = results.sort_values("Composite", ascending=False).reset_index(drop=True)
                    results["Rank"] = range(1, len(results) + 1)
            else:
                results = run_ovtlyr_screener(universe, min_vol)

        if results.empty:
            st.warning("No results. Try a different universe or lower the volume filter.")
            return

        # Store in session state
        st.session_state["ovtlyr_results"] = results

        # KPI cards
        total = len(results)
        strong_buy = len(results[results["Signal"] == "STRONG BUY"])
        buy = len(results[results["Signal"] == "BUY"])
        hold = len(results[results["Signal"] == "HOLD"])
        sell = len(results[results["Signal"] == "SELL"])

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Scanned", total)
        k2.metric("STRONG BUY", strong_buy)
        k3.metric("BUY", buy)
        k4.metric("HOLD", hold)
        k5.metric("SELL", sell)

        # Color styling
        def _signal_color(val):
            colors = {"STRONG BUY": "color:#00ffff", "BUY": "color:#00ff88",
                      "HOLD": "color:#ffdd00", "SELL": "color:#ff3355"}
            return colors.get(val, "")

        styled = results.style
        _map = styled.map if hasattr(styled, "map") else styled.applymap
        styled = _map(_signal_color, subset=["Signal"])
        st.dataframe(styled, use_container_width=True, hide_index=True,
                     height=min(600, 38 + 35 * len(results)))

        # Test Top N
        if test_clicked:
            top_tickers = results.head(int(top_n))["Ticker"].tolist()
            st.session_state["test_topn_tickers"] = top_tickers
            st.session_state["test_topn_mode"] = "ovtlyr"
            st.session_state["auto_run_backtest"] = True
            st.success(f"Top {len(top_tickers)} tickers queued for backtest: {', '.join(top_tickers)}")
            st.info("→ Switch to the BACKTEST tab to see results.")

    elif "ovtlyr_results" in st.session_state:
        results = st.session_state["ovtlyr_results"]
        st.dataframe(results, use_container_width=True, hide_index=True,
                     height=min(600, 38 + 35 * len(results)))


# =============================================================================
# CONSOLIDATED BACKTEST TAB
# =============================================================================

def tab_backtest_consolidated():
    """Unified Backtest tab with dropdown: Swing / Long / OVTLYR."""
    mode = st.selectbox(
        "BACKTEST MODE",
        ["Swing", "Long", "OVTLYR", "RS Sector"],
        key="backtest_mode_select",
    )

    if mode == "Swing":
        tab_backtest()  # Existing backtest — UNTOUCHED

    elif mode == "Long":
        if LONG_TREND_AVAILABLE:
            render_long_trend_page()  # Existing long-term trend — UNTOUCHED
        else:
            _tab_not_found("Long-Term Trend", "long_trend")

    elif mode == "RS Sector":
        if RS_BACKTEST_AVAILABLE:
            render_rs_backtest_page()  # Existing RS backtest — UNTOUCHED
        else:
            _tab_not_found("RS Backtest", "rs_backtest")

    elif mode == "OVTLYR":
        _render_ovtlyr_backtest_ui()


def _render_ovtlyr_backtest_ui():
    """OVTLYR backtest with Test Top N integration."""
    st.markdown(
        "<h2 style='color:#00ffff;letter-spacing:0.1em;'>"
        "OVTLYR BACKTEST</h2>"
        "<p style='color:#4a4a6a;font-size:0.7rem;'>EMA 10/20 crossover + ADX filter + Volume confirmation</p>",
        unsafe_allow_html=True,
    )

    # Check for auto-run from Test Top N
    auto_run = st.session_state.pop("auto_run_backtest", False)
    auto_tickers = st.session_state.get("test_topn_tickers", [])
    auto_mode = st.session_state.get("test_topn_mode", "ovtlyr")

    col1, col2 = st.columns(2)
    with col1:
        ticker_input = st.text_input(
            "Tickers (comma-separated)",
            value=", ".join(auto_tickers) if auto_tickers else "VOLV-B.ST, EQNR.OL, BOL.ST",
            key="ovtlyr_bt_tickers",
        )
    with col2:
        years = st.selectbox("Period", [1, 2, 3, 5], index=2, key="ovtlyr_bt_years")

    run_bt = st.button("▶  RUN BACKTEST", key="ovtlyr_bt_run", use_container_width=True)

    if run_bt or auto_run:
        if not BACKTEST_ENGINE_AVAILABLE:
            st.error("Backtest engine not found.")
            return

        tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
        if not tickers:
            st.warning("Enter at least one ticker.")
            return

        with st.spinner(f"🐺 Backtesting {len(tickers)} tickers ({years}y)..."):
            summary = run_batch_backtest(tickers, years, auto_mode)

        if summary.empty:
            st.warning("No backtest results. Check tickers or period.")
            return

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tickers Tested", len(summary))
        k2.metric("Avg Return", f"{summary['Total Return %'].mean():.1f}%")
        k3.metric("Avg Win Rate", f"{summary['Win Rate %'].mean():.1f}%")
        k4.metric("Avg Max DD", f"{summary['Max DD %'].mean():.1f}%")

        # Results table
        def _ret_color(val):
            try:
                v = float(val)
                return "color:#00ff88" if v > 0 else "color:#ff3355"
            except (TypeError, ValueError):
                return ""

        styled = summary.style
        _map = styled.map if hasattr(styled, "map") else styled.applymap
        for col in ["Total Return %", "CAGR %", "Max DD %", "Avg Return %"]:
            if col in summary.columns:
                styled = _map(_ret_color, subset=[col])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Individual trade details
        with st.expander("Individual Trades", expanded=False):
            for ticker in tickers[:5]:
                bt = run_backtest(ticker, years, auto_mode)
                trades = bt.get("trades", [])
                if trades:
                    st.markdown(f"**{ticker}** — {len(trades)} trades")
                    st.dataframe(pd.DataFrame(trades), use_container_width=True, hide_index=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    inject_css()
    wolf_banner()

    tab_labels = [
        "  SCREENER  ",
        "  BACKTEST  ",
        "  SWING REGIME  ",
        "  LONG REGIME  ",
        "  OVTLYR REGIME  ",
        "  SECTOR & REGIME  ",
        "  SENTIMENT  ",
        "  HEATMAP  ",
        "  RULES  ",
    ]
    (tab1, tab2, tab_swing_regime, tab_long_regime, tab_ovtlyr,
     tab6, tab7, tab8, tab_rules) = st.tabs(tab_labels)

    with tab1:
        tab_screener_consolidated()

    with tab2:
        tab_backtest_consolidated()

    with tab_swing_regime:
        tab_regime()  # Existing swing regime monitor — UNTOUCHED

    with tab_long_regime:
        if LONG_REGIME_AVAILABLE:
            render_long_regime_monitor()
        else:
            _tab_not_found("Long Regime Monitor", "long_regime_monitor")

    with tab_ovtlyr:
        if OVTLYR_AVAILABLE:
            render_ovtlyr_page()
        else:
            _tab_not_found("OVTLYR", "ovtlyr")

    with tab6:
        if SECTOR_CYCLE_AVAILABLE:
            render_sector_cycle_page()
        else:
            _tab_not_found("Sector & Global Regime", "sector_cycle")

    with tab7:
        if SENTIMENT_AVAILABLE:
            render_sentiment_page()
        else:
            _tab_not_found("Sentiment & Flow", "sentiment")

    with tab8:
        if HEATMAP_AVAILABLE:
            render_heatmap_page()
        else:
            _tab_not_found("Heatmap", "heatmap")

    with tab_rules:
        if RULES_AVAILABLE:
            render_rules_page()
        else:
            _tab_not_found("Rules", "ovtlyr/ui")


if __name__ == "__main__":
    main()
