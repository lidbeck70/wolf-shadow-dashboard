"""
sector_cycle_streamlit.py
Streamlit page: Sector & Global Regime Analysis (Ovtlyr / Vermeulen style).

Entry point: render_sector_cycle_page()
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .sector_data import (
    compute_sector_summary,
    trend_color_rgba,
    SECTOR_TICKERS,
    GLOBAL_TICKERS,
    NORDIC_TICKERS,
    TREND_UP,
    TREND_DOWN,
    TREND_NEUTRAL,
)

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

BG      = "#0c0c12"
BG2     = "#14141e"
CYAN    = "#c9a84c"
MAGENTA = "#8b7340"
GREEN   = "#2d8a4e"
RED     = "#c44545"
YELLOW  = "#d4943a"
TEXT    = "#e8e4dc"
DIM     = "#8a8578"

PLOTLY_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=BG2,
    plot_bgcolor=BG,
    font=dict(family="monospace", color=TEXT),
    margin=dict(l=8, r=8, t=32, b=8),
)

# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

PAGE_CSS = f"""
<style>
/* ---- base ---- */
[data-testid="stAppViewContainer"] {{
    background: {BG};
    color: {TEXT};
}}
[data-testid="stSidebar"] {{
    background: {BG2};
}}
/* ---- section headers ---- */
.regime-section-header {{
    font-family: monospace;
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    color: {CYAN};
    text-transform: uppercase;
    border-bottom: 1px solid {DIM};
    padding-bottom: 4px;
    margin-bottom: 12px;
    margin-top: 24px;
}}
/* ---- KPI card ---- */
.kpi-card {{
    background: {BG2};
    border: 1px solid {DIM};
    border-radius: 6px;
    padding: 16px 20px;
    text-align: center;
}}
.kpi-label {{
    font-family: monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: {DIM};
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.kpi-value {{
    font-family: monospace;
    font-size: 1.6rem;
    font-weight: bold;
    line-height: 1;
}}
/* ---- regime badge ---- */
.regime-badge {{
    display: inline-block;
    font-family: monospace;
    font-size: 1.1rem;
    font-weight: bold;
    letter-spacing: 0.12em;
    padding: 6px 18px;
    border-radius: 4px;
    margin-top: 4px;
}}
/* ---- index card ---- */
.index-card {{
    background: {BG2};
    border: 1px solid {DIM};
    border-radius: 6px;
    padding: 10px 12px 4px 12px;
    margin-bottom: 8px;
    min-height: 120px;
}}
.index-name {{
    font-family: monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    color: {TEXT};
    text-transform: uppercase;
    margin-bottom: 4px;
}}
.trend-badge {{
    display: inline-block;
    font-family: monospace;
    font-size: 0.6rem;
    font-weight: bold;
    letter-spacing: 0.12em;
    padding: 2px 8px;
    border-radius: 3px;
    margin-bottom: 6px;
    text-transform: uppercase;
}}
.pct-row {{
    font-family: monospace;
    font-size: 0.65rem;
    color: {DIM};
    margin-top: 4px;
}}
.pct-pos {{ color: {GREEN}; }}
.pct-neg {{ color: {RED}; }}
.pct-neu {{ color: {YELLOW}; }}
</style>
"""


# ---------------------------------------------------------------------------
# Helper: mini sparkline figure
# ---------------------------------------------------------------------------

def _sparkline_fig(data: list) -> go.Figure:
    """Return a minimal Plotly line figure for embedding as a chart."""
    if not data or len(data) < 2:
        fig = go.Figure()
        fig.update_layout(
            height=60,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    vals = np.array(data, dtype=float)
    color = "rgba(45,138,78,0.9)" if vals[-1] >= vals[0] else "rgba(196,69,69,0.9)"

    fig = go.Figure(
        go.Scatter(
            y=vals,
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=color.replace("0.9", "0.12"),
        )
    )
    fig.update_layout(
        height=60,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: format pct with sign & color class
# ---------------------------------------------------------------------------

def _pct_html(val: float) -> str:
    if np.isnan(val):
        return '<span class="pct-neu">—</span>'
    cls = "pct-pos" if val > 0 else ("pct-neg" if val < 0 else "pct-neu")
    sign = "+" if val > 0 else ""
    return f'<span class="{cls}">{sign}{val:.1f}%</span>'


# ---------------------------------------------------------------------------
# Helper: trend badge HTML
# ---------------------------------------------------------------------------

def _trend_badge_html(trend: str) -> str:
    if trend == TREND_UP:
        bg, fg = "rgba(45,138,78,0.15)", GREEN
    elif trend == TREND_DOWN:
        bg, fg = "rgba(196,69,69,0.15)", RED
    else:
        bg, fg = "rgba(212,148,58,0.15)", YELLOW
    return (
        f'<span class="trend-badge" '
        f'style="background:{bg};color:{fg};border:1px solid {fg};">'
        f'{trend}</span>'
    )


# ---------------------------------------------------------------------------
# Section 1: Sector Pie Wheel
# ---------------------------------------------------------------------------

def _render_sector_pie(summary: pd.DataFrame) -> None:
    st.markdown('<p class="regime-section-header">▣ Sector Cycle Wheel — US Sector ETFs</p>', unsafe_allow_html=True)

    sectors = summary[summary["category"] == "Sector"].copy()

    if sectors.empty:
        st.warning("No sector data available.")
        return

    bullish_count = int((sectors["trend_state"] == TREND_UP).sum())

    marker_colors = [
        trend_color_rgba(t, 0.85) for t in sectors["trend_state"]
    ]
    line_colors = [
        trend_color_rgba(t, 0.4) for t in sectors["trend_state"]
    ]

    fig = go.Figure(
        go.Pie(
            labels=sectors["name"].tolist(),
            values=[1] * len(sectors),           # equal slices
            marker=dict(
                colors=marker_colors,
                line=dict(color=line_colors, width=2),
            ),
            textinfo="label",
            textfont=dict(size=11, color=TEXT, family="monospace"),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Trend: %{customdata[0]}<br>"
                "Close: %{customdata[1]:.2f}<br>"
                "1M: %{customdata[2]:.1f}%<br>"
                "<extra></extra>"
            ),
            customdata=sectors[["trend_state", "last_close", "pct_change_1m"]].values,
            hole=0.48,
            direction="clockwise",
            sort=False,
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=420,
        showlegend=True,
        legend=dict(
            font=dict(size=10, color=TEXT, family="monospace"),
            bgcolor="rgba(0,0,0,0)",
            bordercolor=DIM,
            borderwidth=1,
        ),
        annotations=[
            dict(
                text=f"<b>{bullish_count}/11</b><br><span style='font-size:10px'>BULLISH</span>",
                x=0.5, y=0.5,
                font=dict(size=18, color=GREEN, family="monospace"),
                showarrow=False,
                align="center",
            )
        ],
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Section 2: Global Index Grid
# ---------------------------------------------------------------------------

def _render_global_grid(summary: pd.DataFrame) -> None:
    st.markdown('<p class="regime-section-header">▣ Global & Nordic Index Monitor</p>', unsafe_allow_html=True)

    indices = summary[summary["category"].isin(["Global", "Nordic"])].copy()

    if indices.empty:
        st.warning("No index data available.")
        return

    cols = st.columns(4)

    for i, (_, row) in enumerate(indices.iterrows()):
        col = cols[i % 4]
        with col:
            close_str = f"{row['last_close']:,.0f}" if not np.isnan(row["last_close"]) else "—"
            card_html = f"""
<div class="index-card">
  <div class="index-name">{row['name']}</div>
  {_trend_badge_html(row['trend_state'])}
  <div style="font-family:monospace;font-size:0.9rem;color:{TEXT};margin-bottom:2px;">{close_str}</div>
  <div class="pct-row">1M: {_pct_html(row['pct_change_1m'])} &nbsp; 3M: {_pct_html(row['pct_change_3m'])}</div>
</div>
"""
            st.markdown(card_html, unsafe_allow_html=True)
            # Sparkline inside the card area
            spark_fig = _sparkline_fig(row["sparkline_data"])
            st.plotly_chart(
                spark_fig,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"spark_{row['ticker']}",
            )


# ---------------------------------------------------------------------------
# Section 3: Sector Detail Table
# ---------------------------------------------------------------------------

def _render_sector_table(summary: pd.DataFrame) -> None:
    st.markdown('<p class="regime-section-header">▣ Sector Detail Table</p>', unsafe_allow_html=True)

    sectors = summary[summary["category"] == "Sector"].copy()

    if sectors.empty:
        st.warning("No sector data available.")
        return

    table_df = sectors[["name", "trend_state", "pct_change_1m", "pct_change_3m", "last_close"]].copy()
    table_df.columns = ["Sector", "Trend", "1M %", "3M %", "Close"]
    table_df = table_df.reset_index(drop=True)

    def _color_trend(val: str):
        if val == TREND_UP:
            return f"color: {GREEN}; font-weight: bold;"
        if val == TREND_DOWN:
            return f"color: {RED}; font-weight: bold;"
        return f"color: {YELLOW}; font-weight: bold;"

    def _color_pct(val):
        if pd.isna(val):
            return f"color: {DIM};"
        if val > 0:
            return f"color: {GREEN};"
        if val < 0:
            return f"color: {RED};"
        return f"color: {YELLOW};"

    styled = (
        table_df.style
        .map(_color_trend, subset=["Trend"])
        .map(_color_pct, subset=["1M %", "3M %"])
        .format({"1M %": "{:+.2f}%", "3M %": "{:+.2f}%", "Close": "{:.2f}"}, na_rep="—")
        .set_properties(**{
            "font-family": "monospace",
            "font-size": "0.8rem",
        })
        .set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("background-color", BG2),
                    ("color", CYAN),
                    ("font-family", "monospace"),
                    ("font-size", "0.7rem"),
                    ("letter-spacing", "0.1em"),
                    ("text-transform", "uppercase"),
                    ("border-bottom", f"1px solid {DIM}"),
                ],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "rgba(201,168,76,0.04)")],
            },
            {
                "selector": "tbody tr",
                "props": [("background-color", BG)],
            },
        ])
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Section 4: Regime Summary
# ---------------------------------------------------------------------------

def _render_regime_summary(summary: pd.DataFrame) -> None:
    st.markdown('<p class="regime-section-header">▣ Regime Summary</p>', unsafe_allow_html=True)

    sectors = summary[summary["category"] == "Sector"]
    indices = summary[summary["category"].isin(["Global", "Nordic"])]

    bullish_sectors   = int((sectors["trend_state"] == TREND_UP).sum())
    total_sectors     = len(sectors)
    uptrend_indices   = int((indices["trend_state"] == TREND_UP).sum())
    total_indices     = len(indices)

    # Determine overall regime
    if bullish_sectors > 7 and uptrend_indices > 4:
        regime_label = "RISK-ON"
        regime_color = GREEN
        regime_bg    = "rgba(45,138,78,0.12)"
        regime_border = GREEN
    elif bullish_sectors < 4 or uptrend_indices < 2:
        regime_label = "RISK-OFF"
        regime_color = RED
        regime_bg    = "rgba(196,69,69,0.12)"
        regime_border = RED
    else:
        regime_label = "CAUTIOUS"
        regime_color = YELLOW
        regime_bg    = "rgba(212,148,58,0.12)"
        regime_border = YELLOW

    # Build regime bar chart (sector uptrend count, index uptrend count)
    bar_fig = go.Figure()

    bar_fig.add_trace(go.Bar(
        x=["Bullish Sectors", "Uptrend Indices"],
        y=[bullish_sectors, uptrend_indices],
        marker=dict(
            color=[
                trend_color_rgba(TREND_UP if bullish_sectors > total_sectors // 2 else TREND_DOWN, 0.8),
                trend_color_rgba(TREND_UP if uptrend_indices > total_indices // 2 else TREND_DOWN, 0.8),
            ],
            line=dict(color=DIM, width=1),
        ),
        text=[f"{bullish_sectors}/{total_sectors}", f"{uptrend_indices}/{total_indices}"],
        textposition="outside",
        textfont=dict(family="monospace", size=12, color=TEXT),
        width=0.45,
    ))

    # Threshold reference lines
    bar_fig.add_hline(y=7,  line=dict(color="rgba(45,138,78,0.4)",  width=1, dash="dot"),  annotation_text="Risk-On (sectors)", annotation_font=dict(color=GREEN, size=9, family="monospace"))
    bar_fig.add_hline(y=4,  line=dict(color="rgba(196,69,69,0.4)",  width=1, dash="dot"),  annotation_text="Risk-Off (sectors)", annotation_font=dict(color=RED,   size=9, family="monospace"), annotation_position="bottom right")
    bar_fig.add_hline(y=4,  line=dict(color="rgba(45,138,78,0.4)",  width=1, dash="dot"))
    bar_fig.add_hline(y=2,  line=dict(color="rgba(196,69,69,0.4)",  width=1, dash="dot"))

    bar_fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=220,
        yaxis=dict(
            range=[0, max(total_sectors, total_indices) + 2],
            gridcolor=f"rgba(138,133,120,0.3)",
            tickfont=dict(family="monospace", size=9, color=DIM),
        ),
        xaxis=dict(tickfont=dict(family="monospace", size=10, color=TEXT)),
        showlegend=False,
    )

    # KPI cards row
    c1, c2, c3 = st.columns(3)

    with c1:
        bull_color = GREEN if bullish_sectors > total_sectors // 2 else (RED if bullish_sectors < 4 else YELLOW)
        st.markdown(
            f"""<div class="kpi-card">
                  <div class="kpi-label">Bullish Sectors</div>
                  <div class="kpi-value" style="color:{bull_color};">{bullish_sectors}/{total_sectors}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    with c2:
        idx_color = GREEN if uptrend_indices > total_indices // 2 else (RED if uptrend_indices < 2 else YELLOW)
        st.markdown(
            f"""<div class="kpi-card">
                  <div class="kpi-label">Global Uptrend</div>
                  <div class="kpi-value" style="color:{idx_color};">{uptrend_indices}/{total_indices}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""<div class="kpi-card">
                  <div class="kpi-label">Overall Regime</div>
                  <div>
                    <span class="regime-badge"
                          style="background:{regime_bg};color:{regime_color};border:1px solid {regime_border};">
                      {regime_label}
                    </span>
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Sector trend distribution bar (bonus visual)
# ---------------------------------------------------------------------------

def _render_trend_distribution(summary: pd.DataFrame) -> None:
    """Stacked horizontal bar: Uptrend | Neutral | Downtrend per category."""
    st.markdown('<p class="regime-section-header">▣ Trend Distribution by Category</p>', unsafe_allow_html=True)

    categories = ["Sector", "Global", "Nordic"]
    up_counts   = []
    neu_counts  = []
    dn_counts   = []

    for cat in categories:
        sub = summary[summary["category"] == cat]
        up_counts.append(int((sub["trend_state"] == TREND_UP).sum()))
        neu_counts.append(int((sub["trend_state"] == TREND_NEUTRAL).sum()))
        dn_counts.append(int((sub["trend_state"] == TREND_DOWN).sum()))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Uptrend",
        x=up_counts,
        y=categories,
        orientation="h",
        marker=dict(color="rgba(45,138,78,0.8)", line=dict(color="rgba(45,138,78,0.4)", width=1)),
        text=[f"{v}" for v in up_counts],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(family="monospace", size=11, color=BG),
    ))
    fig.add_trace(go.Bar(
        name="Neutral",
        x=neu_counts,
        y=categories,
        orientation="h",
        marker=dict(color="rgba(212,148,58,0.75)", line=dict(color="rgba(212,148,58,0.3)", width=1)),
        text=[f"{v}" for v in neu_counts],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(family="monospace", size=11, color=BG),
    ))
    fig.add_trace(go.Bar(
        name="Downtrend",
        x=dn_counts,
        y=categories,
        orientation="h",
        marker=dict(color="rgba(196,69,69,0.8)", line=dict(color="rgba(196,69,69,0.4)", width=1)),
        text=[f"{v}" for v in dn_counts],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(family="monospace", size=11, color=BG),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        barmode="stack",
        height=180,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="monospace", size=10, color=TEXT),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor="rgba(138,133,120,0.3)",
            tickfont=dict(family="monospace", size=9, color=DIM),
        ),
        yaxis=dict(
            tickfont=dict(family="monospace", size=10, color=TEXT),
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_sector_cycle_page() -> None:
    """Render the full Sector & Global Regime Analysis page."""

    # Inject cyberpunk CSS
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    # Page header
    st.markdown(
        f"""
        <h2 style="font-family:monospace;color:{CYAN};letter-spacing:0.15em;
                   text-transform:uppercase;border-bottom:1px solid {DIM};
                   padding-bottom:8px;margin-bottom:4px;">
            ◈ Sector &amp; Global Regime
        </h2>
        <p style="font-family:monospace;font-size:0.72rem;color:{DIM};margin-bottom:0;">
            EMA-50 / EMA-200 trend classification · yfinance · 1-year lookback · cache 60 min
        </p>
        """,
        unsafe_allow_html=True,
    )

    # --- Data loading with progress feedback ---
    with st.spinner("Fetching market data…"):
        summary = compute_sector_summary()

    if summary.empty:
        st.error("Failed to retrieve market data. Check your internet connection and try again.")
        return

    # ------------------------------------------------------------------ #
    #  Controls sidebar                                                    #
    # ------------------------------------------------------------------ #
    # ── Controls in main area ────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])

    with ctrl_col1:
        # Category filter for the global grid section
        show_categories = st.multiselect(
            "Show categories",
            options=["Global", "Nordic"],
            default=["Global", "Nordic"],
            key="sector_cycle_cat_filter",
        )

    with ctrl_col2:
        # Quick stats
        total_up = int((summary["trend_state"] == TREND_UP).sum())
        total    = len(summary)
        total_dn = int((summary["trend_state"] == TREND_DOWN).sum())
        total_ne = total - total_up - total_dn

        st.markdown(
            f"""
            <div style="font-family:monospace;font-size:0.68rem;color:{DIM};">
              <p>All instruments: {total}</p>
              <p style="color:{GREEN};">▲ Uptrend: {total_up}</p>
              <p style="color:{YELLOW};">◆ Neutral: {total_ne}</p>
              <p style="color:{RED};">▼ Downtrend: {total_dn}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ctrl_col3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("↺ Refresh Data", key="sector_cycle_refresh"):
            compute_sector_summary.clear()
            st.rerun()

    # ------------------------------------------------------------------ #
    #  Layout: Pie (left) + Trend distribution (right)                    #
    # ------------------------------------------------------------------ #
    pie_col, dist_col = st.columns([3, 2], gap="medium")

    with pie_col:
        _render_sector_pie(summary)

    with dist_col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        _render_trend_distribution(summary)

        # Compact sector status list
        st.markdown(
            f'<p style="font-family:monospace;font-size:0.65rem;color:{DIM};'
            f'letter-spacing:0.1em;text-transform:uppercase;margin-top:12px;">Sector States</p>',
            unsafe_allow_html=True,
        )
        for _, row in summary[summary["category"] == "Sector"].iterrows():
            color = GREEN if row["trend_state"] == TREND_UP else (RED if row["trend_state"] == TREND_DOWN else YELLOW)
            arrow = "▲" if row["trend_state"] == TREND_UP else ("▼" if row["trend_state"] == TREND_DOWN else "◆")
            st.markdown(
                f'<div style="font-family:monospace;font-size:0.7rem;color:{color};">'
                f'{arrow} {row["name"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    #  Section 4: Regime Summary (high up for prominence)                 #
    # ------------------------------------------------------------------ #
    _render_regime_summary(summary)

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    #  Section 2: Global Index Grid (filtered by sidebar)                 #
    # ------------------------------------------------------------------ #
    filtered_summary = summary[summary["category"].isin(show_categories)].copy() if show_categories else summary[summary["category"].isin(["Global", "Nordic"])].copy()
    _render_global_grid(filtered_summary)

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    #  Section 3: Sector Detail Table                                     #
    # ------------------------------------------------------------------ #
    _render_sector_table(summary)

    # Footer
    st.markdown(
        f'<p style="font-family:monospace;font-size:0.62rem;color:{DIM};'
        f'text-align:center;margin-top:32px;border-top:1px solid {DIM};'
        f'padding-top:12px;">Data: Yahoo Finance · EMA-50/200 trend classification · '
        f'Regime thresholds: >7 sectors & >4 indices → RISK-ON; &lt;4 sectors or &lt;2 indices → RISK-OFF</p>',
        unsafe_allow_html=True,
    )
