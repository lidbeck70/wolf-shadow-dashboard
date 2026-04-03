"""
long_trend_streamlit.py
Streamlit page for Long-Term Trend & Drawdown analysis.

Entry point: render_long_trend_page()

Sections:
  1. Price chart with EMA50/EMA200, drawdown zones, Rick Rule markers
  2. Drawdown analysis table
  3. Drawdown summary box
  4. "Where Are We Now?" — trend phase, cycle position, Rick Rule verdict
  5. Rick Rule backtest summary
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module imports (dual-path for both workspace root and Streamlit contexts)
# ---------------------------------------------------------------------------
try:
    from long_trend.long_trend_loader import (
        run_long_trend_analysis,
        CYCLE_PHASES,
        cycle_phase_index,
    )
    from cagr.cagr_loader import NORDIC_TICKERS
except ImportError:
    try:
        from dashboard.long_trend.long_trend_loader import (
            run_long_trend_analysis,
            CYCLE_PHASES,
            cycle_phase_index,
        )
        from dashboard.cagr.cagr_loader import NORDIC_TICKERS
    except ImportError:
        from long_trend_loader import (  # type: ignore
            run_long_trend_analysis,
            CYCLE_PHASES,
            cycle_phase_index,
        )
        try:
            import sys, os
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from cagr.cagr_loader import NORDIC_TICKERS
        except ImportError:
            NORDIC_TICKERS = {}

# ---------------------------------------------------------------------------
# Extended ticker universes for Long-Term Trend analysis
# ---------------------------------------------------------------------------

# Nordic mid & small caps (Sweden)
NORDIC_MIDSMALL_SE: dict = {
    "BEIA-B.ST": {"name": "Beijer Alma B", "country": "Sweden", "sector": "Industrials"},
    "BURE.ST":   {"name": "Bure Equity", "country": "Sweden", "sector": "Financials"},
    "CAST.ST":   {"name": "Castellum", "country": "Sweden", "sector": "Real Estate"},
    "CATE.ST":   {"name": "Catena", "country": "Sweden", "sector": "Real Estate"},
    "DIOS.ST":   {"name": "Dios Fastigheter", "country": "Sweden", "sector": "Real Estate"},
    "ELUX-B.ST": {"name": "Electrolux B", "country": "Sweden", "sector": "Consumer Discretionary"},
    "FABG.ST":   {"name": "Fabege", "country": "Sweden", "sector": "Real Estate"},
    "GETI-B.ST": {"name": "Getinge B", "country": "Sweden", "sector": "Healthcare"},
    "HUFV-A.ST": {"name": "Hufvudstaden A", "country": "Sweden", "sector": "Real Estate"},
    "HUSQ-B.ST": {"name": "Husqvarna B", "country": "Sweden", "sector": "Industrials"},
    "KINV-B.ST": {"name": "Kinnevik B", "country": "Sweden", "sector": "Financials"},
    "LATO-B.ST": {"name": "Latour B", "country": "Sweden", "sector": "Financials"},
    "LUND-B.ST": {"name": "Lundin Mining B", "country": "Sweden", "sector": "Materials"},
    "LUMI.ST":   {"name": "Loomis", "country": "Sweden", "sector": "Industrials"},
    "SCA-B.ST":  {"name": "SCA B", "country": "Sweden", "sector": "Materials"},
    "SAGA-B.ST": {"name": "Sagax B", "country": "Sweden", "sector": "Real Estate"},
    "SECU-B.ST": {"name": "Securitas B", "country": "Sweden", "sector": "Industrials"},
    "SKA-B.ST":  {"name": "Skanska B", "country": "Sweden", "sector": "Industrials"},
    "SSAB-B.ST": {"name": "SSAB B", "country": "Sweden", "sector": "Materials"},
    "SWMA.ST":   {"name": "Swedish Match", "country": "Sweden", "sector": "Consumer Staples"},
    "TEL2-B.ST": {"name": "Tele2 B", "country": "Sweden", "sector": "Communication"},
    "TELIA.ST":  {"name": "Telia Company", "country": "Sweden", "sector": "Communication"},
    "THULE.ST":  {"name": "Thule Group", "country": "Sweden", "sector": "Consumer Discretionary"},
    "TREL-B.ST": {"name": "Trelleborg B", "country": "Sweden", "sector": "Industrials"},
    "WIHL.ST":   {"name": "Wihlborgs", "country": "Sweden", "sector": "Real Estate"},
    "ADDV-B.ST": {"name": "Addvise B", "country": "Sweden", "sector": "Healthcare"},
    "BALD-B.ST": {"name": "Balder B", "country": "Sweden", "sector": "Real Estate"},
    "BILL.ST":   {"name": "Billerud", "country": "Sweden", "sector": "Materials"},
    "CLAR-B.ST": {"name": "Clas Ohlson B", "country": "Sweden", "sector": "Consumer Discretionary"},
    "NOLA-B.ST": {"name": "Nolato B", "country": "Sweden", "sector": "Industrials"},
}

# Nordic mid & small caps (Norway)
NORDIC_MIDSMALL_NO: dict = {
    "ORK.OL":    {"name": "Orkla", "country": "Norway", "sector": "Consumer Staples"},
    "YAR.OL":    {"name": "Yara International", "country": "Norway", "sector": "Materials"},
    "GOGL.OL":   {"name": "Golden Ocean", "country": "Norway", "sector": "Industrials"},
    "FRO.OL":    {"name": "Frontline", "country": "Norway", "sector": "Energy"},
    "VAR.OL":    {"name": "Vår Energi", "country": "Norway", "sector": "Energy"},
    "HAFNI.OL":  {"name": "Hafnia", "country": "Norway", "sector": "Energy"},
    "TGS.OL":    {"name": "TGS NOPEC", "country": "Norway", "sector": "Energy"},
    "HAUTO.OL":  {"name": "Höegh Autoliners", "country": "Norway", "sector": "Industrials"},
    "AKER.OL":   {"name": "Aker ASA", "country": "Norway", "sector": "Financials"},
    "AKSO.OL":   {"name": "Aker Solutions", "country": "Norway", "sector": "Energy"},
    "NOD.OL":    {"name": "Nordic Semiconductor", "country": "Norway", "sector": "Technology"},
    "NEL.OL":    {"name": "Nel ASA", "country": "Norway", "sector": "Energy"},
    "KOG.OL":    {"name": "Kongsberg Gruppen", "country": "Norway", "sector": "Industrials"},
    "SCHB.OL":   {"name": "Schibsted B", "country": "Norway", "sector": "Communication"},
    "BWLPG.OL":  {"name": "BW LPG", "country": "Norway", "sector": "Energy"},
    "FLNG.OL":   {"name": "Flex LNG", "country": "Norway", "sector": "Energy"},
}

# Nordic mid & small caps (Denmark)
NORDIC_MIDSMALL_DK: dict = {
    "AMBU-B.CO": {"name": "Ambu B", "country": "Denmark", "sector": "Healthcare"},
    "DEMANT.CO": {"name": "Demant", "country": "Denmark", "sector": "Healthcare"},
    "GN.CO":     {"name": "GN Audio", "country": "Denmark", "sector": "Technology"},
    "RBREW.CO":  {"name": "Royal Unibrew", "country": "Denmark", "sector": "Consumer Staples"},
    "JYSK.CO":   {"name": "Jyske Bank", "country": "Denmark", "sector": "Financials"},
    "DANSKE.CO": {"name": "Danske Bank", "country": "Denmark", "sector": "Financials"},
    "ISS.CO":    {"name": "ISS A/S", "country": "Denmark", "sector": "Industrials"},
    "PNDORA.CO": {"name": "Pandora", "country": "Denmark", "sector": "Consumer Discretionary"},
    "FLS.CO":    {"name": "FLSmidth", "country": "Denmark", "sector": "Industrials"},
    "ROCK-B.CO": {"name": "Rockwool B", "country": "Denmark", "sector": "Industrials"},
    "SYDB.CO":   {"name": "Sydbank", "country": "Denmark", "sector": "Financials"},
    "SIM.CO":    {"name": "SimCorp", "country": "Denmark", "sector": "Technology"},
}

# Nordic mid & small caps (Finland)
NORDIC_MIDSMALL_FI: dict = {
    "ORNBV.HE":  {"name": "Orion", "country": "Finland", "sector": "Healthcare"},
    "TIETO.HE":  {"name": "TietoEVRY", "country": "Finland", "sector": "Technology"},
    "VALMT.HE":  {"name": "Valmet", "country": "Finland", "sector": "Industrials"},
    "CGCBV.HE":  {"name": "Cargotec", "country": "Finland", "sector": "Industrials"},
    "METSB.HE":  {"name": "Metsä Board", "country": "Finland", "sector": "Materials"},
    "HUH1V.HE":  {"name": "Huhtamäki", "country": "Finland", "sector": "Industrials"},
    "ELISA.HE":  {"name": "Elisa", "country": "Finland", "sector": "Communication"},
    "KEMIRA.HE":  {"name": "Kemira", "country": "Finland", "sector": "Materials"},
    "OLVAS.HE":  {"name": "Olvi", "country": "Finland", "sector": "Consumer Staples"},
    "OUT1V.HE":  {"name": "Outokumpu", "country": "Finland", "sector": "Materials"},
    "TOKMAN.HE": {"name": "Tokmanni", "country": "Finland", "sector": "Consumer Discretionary"},
}

# UCITS ETFs & SPY
ETF_TREND_TICKERS: dict = {
    "SPY":       {"name": "SPDR S&P 500 ETF", "country": "US", "sector": "ETF"},
    "QQQ":       {"name": "Invesco QQQ (Nasdaq 100)", "country": "US", "sector": "ETF"},
    "IWM":       {"name": "iShares Russell 2000 ETF", "country": "US", "sector": "ETF"},
    "GLD":       {"name": "SPDR Gold Trust", "country": "US", "sector": "ETF"},
    "SLV":       {"name": "iShares Silver Trust", "country": "US", "sector": "ETF"},
    "GDX":       {"name": "VanEck Gold Miners ETF", "country": "US", "sector": "ETF"},
    "GDXJ":      {"name": "VanEck Junior Gold Miners ETF", "country": "US", "sector": "ETF"},
    "XLE":       {"name": "Energy Select SPDR", "country": "US", "sector": "ETF"},
    "XLB":       {"name": "Materials Select SPDR", "country": "US", "sector": "ETF"},
    "XLF":       {"name": "Financial Select SPDR", "country": "US", "sector": "ETF"},
    "XLK":       {"name": "Technology Select SPDR", "country": "US", "sector": "ETF"},
    "IWDA.AS":   {"name": "iShares MSCI World UCITS", "country": "IE", "sector": "ETF"},
    "IEMA.AS":   {"name": "iShares MSCI EM UCITS", "country": "IE", "sector": "ETF"},
    "CSPX.AS":   {"name": "iShares S&P 500 UCITS", "country": "IE", "sector": "ETF"},
    "VWRL.AS":   {"name": "Vanguard FTSE All-World UCITS", "country": "IE", "sector": "ETF"},
    "VUSA.AS":   {"name": "Vanguard S&P 500 UCITS", "country": "IE", "sector": "ETF"},
    "EUNL.DE":   {"name": "iShares MSCI World UCITS (EUR)", "country": "IE", "sector": "ETF"},
    "IUSQ.DE":   {"name": "iShares MSCI ACWI UCITS", "country": "IE", "sector": "ETF"},
    "IQQQ.DE":   {"name": "iShares Nasdaq 100 UCITS", "country": "IE", "sector": "ETF"},
    "SXRV.DE":   {"name": "iShares MSCI Europe UCITS", "country": "IE", "sector": "ETF"},
    "IQQH.DE":   {"name": "iShares Global Clean Energy UCITS", "country": "IE", "sector": "ETF"},
    "MEUD.PA":   {"name": "Lyxor STOXX 600 UCITS", "country": "FR", "sector": "ETF"},
    "XDWD.DE":   {"name": "Xtrackers MSCI World UCITS", "country": "LU", "sector": "ETF"},
    "BTCE.DE":   {"name": "ETC Group Bitcoin ETP", "country": "DE", "sector": "ETF"},
}


def _build_trend_ticker_registry() -> dict:
    """Combine all ticker sources into a categorised registry for the UI dropdown."""
    registry: dict = {}

    def _add(source: dict, category: str) -> None:
        for ticker, meta in source.items():
            registry[ticker] = {**meta, "_category": category}

    _add(NORDIC_TICKERS, "🇸🇪 Nordic Large Cap") if NORDIC_TICKERS else None
    _add(NORDIC_MIDSMALL_SE, "🇸🇪 Sweden Mid/Small")
    _add(NORDIC_MIDSMALL_NO, "🇳🇴 Norway Mid/Small")
    _add(NORDIC_MIDSMALL_DK, "🇩🇰 Denmark Mid/Small")
    _add(NORDIC_MIDSMALL_FI, "🇫🇮 Finland Mid/Small")
    _add(ETF_TREND_TICKERS, "📊 ETFs & Index")

    return registry


# ---------------------------------------------------------------------------
# Cyberpunk theme constants
# ---------------------------------------------------------------------------
BG = "#050510"
BG2 = "#0a0a1e"
CYAN = "#00ffff"
MAGENTA = "#ff00ff"
GREEN = "#00ff88"
RED = "#ff3355"
YELLOW = "#ffdd00"
TEXT = "#e0e0ff"
DIM = "#4a4a6a"

# rgba versions for Plotly
CYAN_RGBA = "rgba(0,255,255,1)"
MAGENTA_RGBA = "rgba(255,0,255,1)"
GREEN_RGBA = "rgba(0,255,136,1)"
RED_RGBA = "rgba(255,51,85,1)"
YELLOW_RGBA = "rgba(255,221,0,1)"
RED_ZONE_RGBA = "rgba(255,51,85,0.12)"
RED_ZONE_LINE_RGBA = "rgba(255,51,85,0.35)"
TEXT_RGBA = "rgba(224,224,255,1)"
DIM_RGBA = "rgba(74,74,106,1)"
PRICE_RGBA = "rgba(0,255,255,0.8)"

# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def _badge(text: str, color: str) -> str:
    """Return an HTML badge with cyberpunk styling."""
    return (
        f'<span style="'
        f'background:transparent;'
        f'border:1px solid {color};'
        f'color:{color};'
        f'padding:3px 12px;'
        f'border-radius:4px;'
        f'font-family:monospace;'
        f'font-size:0.85rem;'
        f'font-weight:700;'
        f'letter-spacing:1px;'
        f'text-shadow:0 0 8px {color};'
        f'box-shadow:0 0 6px {color}44;'
        f'">{text}</span>'
    )


def _metric_box(label: str, value: str, color: str = CYAN) -> str:
    """Return an HTML metric box."""
    return (
        f'<div style="'
        f'background:{BG2};'
        f'border:1px solid {color}44;'
        f'border-left:3px solid {color};'
        f'padding:12px 16px;'
        f'border-radius:4px;'
        f'margin-bottom:8px;'
        f'">'
        f'<div style="color:{DIM};font-size:0.75rem;font-family:monospace;'
        f'text-transform:uppercase;letter-spacing:1px;">{label}</div>'
        f'<div style="color:{color};font-size:1.2rem;font-family:monospace;'
        f'font-weight:700;margin-top:4px;">{value}</div>'
        f'</div>'
    )


def _trend_color(phase: str) -> str:
    return {
        "Bullish": GREEN,
        "Bearish": RED,
        "Neutral": YELLOW,
    }.get(phase, CYAN)


def _cycle_color(phase: str) -> str:
    colors = {
        "Accumulation": YELLOW,
        "Early Uptrend": CYAN,
        "Strong Uptrend": GREEN,
        "Late Uptrend": MAGENTA,
        "Early Downtrend": YELLOW,
        "Capitulation": RED,
        "Recovery": CYAN,
    }
    return colors.get(phase, DIM)


def _verdict_color(verdict: str) -> str:
    return {
        "BUY zone": GREEN,
        "SELL zone": RED,
        "HOLD": YELLOW,
    }.get(verdict, CYAN)


# ---------------------------------------------------------------------------
# Section 1: Price chart
# ---------------------------------------------------------------------------

def _build_price_chart(
    df: pd.DataFrame,
    drawdowns: list,
    buy_signals: pd.DataFrame,
    sell_signals: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """
    Build the long-term price chart with:
      - Candlestick or close line
      - EMA50 (yellow dotted)
      - EMA200 (magenta solid)
      - Drawdown zones (red shaded)
      - Rick Rule buy (green △) / sell (red ▽) markers
    """
    fig = go.Figure()

    close = df["Close"].dropna()
    dates = close.index

    # ── Price line ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates,
        y=close.values,
        name="Price",
        line=dict(color=PRICE_RGBA, width=1.5),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Price: %{y:.2f}<extra></extra>",
    ))

    # ── EMA50 (yellow, dotted) ────────────────────────────────────────────
    if "EMA50" in df.columns:
        ema50 = df["EMA50"].dropna()
        fig.add_trace(go.Scatter(
            x=ema50.index,
            y=ema50.values,
            name="EMA 50",
            line=dict(color=YELLOW_RGBA, width=1.5, dash="dot"),
            hovertemplate="EMA50: %{y:.2f}<extra></extra>",
        ))

    # ── EMA200 (magenta, solid) ───────────────────────────────────────────
    if "EMA200" in df.columns:
        ema200 = df["EMA200"].dropna()
        fig.add_trace(go.Scatter(
            x=ema200.index,
            y=ema200.values,
            name="EMA 200",
            line=dict(color=MAGENTA_RGBA, width=2.0),
            hovertemplate="EMA200: %{y:.2f}<extra></extra>",
        ))

    # ── Drawdown zones (red shaded) ───────────────────────────────────────
    for dd in drawdowns:
        fig.add_vrect(
            x0=dd["start"],
            x1=dd["end"],
            fillcolor=RED_ZONE_RGBA,
            line_color=RED_ZONE_LINE_RGBA,
            line_width=1,
            annotation_text=f"{dd['max_drop_pct']:.0%}",
            annotation_position="top left",
            annotation=dict(
                font=dict(color=RED, size=9, family="monospace"),
                bgcolor="rgba(5,5,16,0.7)",
            ),
        )

    # ── Rick Rule BUY markers (green △) ──────────────────────────────────
    if not buy_signals.empty:
        # Align buy prices to df close on that date (or nearest)
        buy_dates = buy_signals["Date"].values
        buy_prices = buy_signals["Price"].values
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_prices,
            name="BUY Signal",
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                color=GREEN_RGBA,
                size=12,
                line=dict(color=GREEN_RGBA, width=1),
            ),
            hovertemplate="<b>BUY</b><br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
        ))

    # ── Rick Rule SELL markers (red ▽) ───────────────────────────────────
    if not sell_signals.empty:
        sell_dates = sell_signals["Date"].values
        sell_prices = sell_signals["Price"].values
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_prices,
            name="SELL Signal",
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                color=RED_RGBA,
                size=12,
                line=dict(color=RED_RGBA, width=1),
            ),
            hovertemplate="<b>SELL</b><br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG2,
        plot_bgcolor=BG,
        title=dict(
            text=f"<span style='font-family:monospace;color:{CYAN}'>{ticker} — Long-Term Price & EMAs</span>",
            font=dict(size=16, color=CYAN, family="monospace"),
        ),
        legend=dict(
            bgcolor="rgba(10,10,30,0.8)",
            bordercolor=DIM,
            borderwidth=1,
            font=dict(color=TEXT, family="monospace", size=11),
        ),
        xaxis=dict(
            gridcolor=f"rgba(74,74,106,0.3)",
            tickfont=dict(color=DIM, family="monospace"),
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            gridcolor=f"rgba(74,74,106,0.3)",
            tickfont=dict(color=DIM, family="monospace"),
        ),
        hovermode="x unified",
        height=520,
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig


# ---------------------------------------------------------------------------
# Section 2: Drawdown table
# ---------------------------------------------------------------------------

def _render_drawdown_table(drawdowns: list) -> None:
    """Render drawdown analysis as st.dataframe with classification coloring."""
    if not drawdowns:
        st.info("No drawdowns exceeding 10% detected in this period.")
        return

    rows = []
    for dd in drawdowns:
        ebitda_str = f"{dd['ebitda_delta']:.1%}" if dd.get("ebitda_delta") is not None else "N/A"
        margin_str = f"{dd['margin_delta']:+.1f}pp" if dd.get("margin_delta") is not None else "N/A"
        debt_str = f"{dd['debt_delta']:.1%}" if dd.get("debt_delta") is not None else "N/A"
        rows.append({
            "Start": dd["start"].strftime("%Y-%m-%d") if hasattr(dd["start"], "strftime") else str(dd["start"])[:10],
            "End": dd["end"].strftime("%Y-%m-%d") if hasattr(dd["end"], "strftime") else str(dd["end"])[:10],
            "Duration (days)": int(dd["duration_days"]),
            "Max Drop %": f"{dd['max_drop_pct']:.1%}",
            "Classification": dd.get("classification", "Noise/Temporary"),
            "EBITDA Delta": ebitda_str,
            "Margin Delta": margin_str,
            "Debt Delta": debt_str,
        })

    display_df = pd.DataFrame(rows)

    # Color mapping for classification
    classification_colors = {
        "Noise/Temporary": "#1a3a1a",
        "Fundamental Deterioration": "#3a1a1a",
        "Macro/Geopolitical": "#1a1a3a",
        "Sector-Wide": "#2a2a1a",
    }

    def _style_classification(val: str) -> str:
        bg = classification_colors.get(val, "#0a0a1e")
        return f"background-color: {bg}; color: {TEXT}; font-family: monospace;"

    def _style_drop(val: str) -> str:
        try:
            v = float(val.replace("%", "")) / 100
            if v <= -0.30:
                return f"color: {RED}; font-weight: bold; font-family: monospace;"
            if v <= -0.20:
                return f"color: {MAGENTA}; font-family: monospace;"
            return f"color: {YELLOW}; font-family: monospace;"
        except Exception:
            return ""

    styled = (
        display_df.style
        .map(_style_classification, subset=["Classification"])
        .map(_style_drop, subset=["Max Drop %"])
        .set_properties(**{
            "background-color": BG2,
            "color": TEXT,
            "font-family": "monospace",
            "font-size": "0.82rem",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", BG),
                ("color", CYAN),
                ("font-family", "monospace"),
                ("font-size", "0.80rem"),
                ("text-transform", "uppercase"),
                ("letter-spacing", "1px"),
                ("border-bottom", f"1px solid {DIM}"),
            ]},
            {"selector": "td", "props": [
                ("border-bottom", f"1px solid rgba(74,74,106,0.3)"),
            ]},
        ])
    )

    st.dataframe(styled, use_container_width=True, height=min(50 + len(rows) * 38, 480))


# ---------------------------------------------------------------------------
# Section 3: Drawdown summary
# ---------------------------------------------------------------------------

def _render_drawdown_summary(summary: dict, total: int) -> None:
    """Render summary box showing breakdown of drawdown classifications."""
    st.markdown(
        f'<div style="'
        f'background:{BG2};'
        f'border:1px solid {DIM};'
        f'border-top:2px solid {CYAN};'
        f'border-radius:6px;'
        f'padding:16px 20px;'
        f'font-family:monospace;'
        f'">'
        f'<div style="color:{CYAN};font-size:0.75rem;text-transform:uppercase;'
        f'letter-spacing:2px;margin-bottom:12px;">Drawdown Classification Breakdown</div>'
        f'<div style="display:flex;gap:24px;flex-wrap:wrap;">'
        f'<div><span style="color:{DIM};">Total drawdowns:</span> '
        f'<span style="color:{TEXT};font-weight:700;">{total}</span></div>'
        f'<div><span style="color:{DIM};">Noise/Temporary:</span> '
        f'<span style="color:{GREEN};font-weight:700;">{summary.get("noise_pct", 0)}%</span></div>'
        f'<div><span style="color:{DIM};">Fundamental:</span> '
        f'<span style="color:{RED};font-weight:700;">{summary.get("fundamental_pct", 0)}%</span></div>'
        f'<div><span style="color:{DIM};">Macro/Geopolitical:</span> '
        f'<span style="color:{MAGENTA};font-weight:700;">{summary.get("macro_pct", 0)}%</span></div>'
        f'<div><span style="color:{DIM};">Sector-Wide:</span> '
        f'<span style="color:{YELLOW};font-weight:700;">{summary.get("sector_pct", 0)}%</span></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 4: Where Are We Now?
# ---------------------------------------------------------------------------

def _render_cycle_bar(current_phase: str) -> None:
    """Render a horizontal cycle position bar using Plotly."""
    n = len(CYCLE_PHASES)
    current_idx = cycle_phase_index(current_phase)

    bar_colors = []
    for i, phase in enumerate(CYCLE_PHASES):
        if i == current_idx:
            bar_colors.append(_cycle_color(phase))
        else:
            bar_colors.append("rgba(74,74,106,0.3)")

    fig = go.Figure()

    for i, phase in enumerate(CYCLE_PHASES):
        is_active = (i == current_idx)
        color = _cycle_color(phase) if is_active else DIM
        opacity = 1.0 if is_active else 0.4

        fig.add_trace(go.Bar(
            x=[phase],
            y=[1],
            name=phase,
            marker=dict(
                color=bar_colors[i],
                opacity=opacity,
                line=dict(color=BG2, width=2),
            ),
            text=phase if is_active else "",
            textposition="inside",
            textfont=dict(color=BG, size=11, family="monospace"),
            hoverinfo="x",
            showlegend=False,
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG2,
        plot_bgcolor=BG2,
        barmode="stack",
        height=80,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            tickfont=dict(color=DIM, family="monospace", size=9),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        bargap=0.05,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_where_are_we(
    trend_phase: str,
    cycle_position: str,
    rick_verdict: str,
) -> None:
    """Render the 'Where Are We Now?' panel."""
    tc = _trend_color(trend_phase)
    cc = _cycle_color(cycle_position)
    vc = _verdict_color(rick_verdict)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div style="'
            f'background:{BG2};border:1px solid {tc}44;border-top:2px solid {tc};'
            f'border-radius:6px;padding:16px;text-align:center;">'
            f'<div style="color:{DIM};font-size:0.7rem;font-family:monospace;'
            f'text-transform:uppercase;letter-spacing:2px;">Trend Phase</div>'
            f'<div style="margin-top:8px;">{_badge(trend_phase, tc)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f'<div style="'
            f'background:{BG2};border:1px solid {cc}44;border-top:2px solid {cc};'
            f'border-radius:6px;padding:16px;text-align:center;">'
            f'<div style="color:{DIM};font-size:0.7rem;font-family:monospace;'
            f'text-transform:uppercase;letter-spacing:2px;">Cycle Position</div>'
            f'<div style="margin-top:8px;">{_badge(cycle_position, cc)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f'<div style="'
            f'background:{BG2};border:1px solid {vc}44;border-top:2px solid {vc};'
            f'border-radius:6px;padding:16px;text-align:center;">'
            f'<div style="color:{DIM};font-size:0.7rem;font-family:monospace;'
            f'text-transform:uppercase;letter-spacing:2px;">Rick Rule Verdict</div>'
            f'<div style="margin-top:8px;">{_badge(rick_verdict, vc)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    _render_cycle_bar(cycle_position)


# ---------------------------------------------------------------------------
# Section 5: Rick Rule backtest
# ---------------------------------------------------------------------------

def _render_backtest(backtest: pd.DataFrame) -> None:
    """Render the Rick Rule backtest trades table and total return."""
    if backtest.empty:
        st.info("No completed Rick Rule trades in this period.")
        return

    # Compute total return from last SELL row
    completed_sells = backtest[backtest["Action"] == "SELL"]
    total_return = float(completed_sells["Cumulative_Return"].iloc[-1]) if not completed_sells.empty else 0.0
    num_trades = len(completed_sells)

    # Summary metric
    tr_color = GREEN if total_return >= 0 else RED
    sign = "+" if total_return >= 0 else ""

    st.markdown(
        f'<div style="'
        f'background:{BG2};border:1px solid {tr_color}44;border-left:3px solid {tr_color};'
        f'border-radius:4px;padding:14px 20px;margin-bottom:16px;'
        f'display:flex;justify-content:space-between;align-items:center;">'
        f'<div>'
        f'<span style="color:{DIM};font-size:0.75rem;font-family:monospace;'
        f'text-transform:uppercase;letter-spacing:1px;">Total Return (Rick Rule Strategy)</span><br>'
        f'<span style="color:{tr_color};font-size:1.6rem;font-family:monospace;'
        f'font-weight:700;">{sign}{total_return:.1f}%</span>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<span style="color:{DIM};font-size:0.75rem;font-family:monospace;">Completed Trades</span><br>'
        f'<span style="color:{CYAN};font-size:1.2rem;font-family:monospace;font-weight:700;">'
        f'{num_trades}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Format display DataFrame
    display = backtest.copy()
    if "Date" in display.columns:
        display["Date"] = display["Date"].apply(
            lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
        )
    display["Return_pct"] = display["Return_pct"].apply(
        lambda v: f"{v:+.2f}%" if pd.notna(v) and v is not None else "—"
    )
    display["Cumulative_Return"] = display["Cumulative_Return"].apply(
        lambda v: f"{v:+.1f}%" if pd.notna(v) and v is not None else "—"
    )
    display["Price"] = display["Price"].apply(
        lambda v: f"{v:.2f}" if pd.notna(v) else "—"
    )

    def _style_action(val: str) -> str:
        if val == "BUY":
            return f"color: {GREEN}; font-weight: bold; font-family: monospace;"
        if val == "SELL":
            return f"color: {RED}; font-weight: bold; font-family: monospace;"
        return ""

    def _style_return(val: str) -> str:
        if val == "—":
            return f"color: {DIM}; font-family: monospace;"
        try:
            v = float(val.replace("%", "").replace("+", ""))
            color = GREEN if v >= 0 else RED
            return f"color: {color}; font-family: monospace;"
        except Exception:
            return ""

    styled = (
        display.style
        .map(_style_action, subset=["Action"])
        .map(_style_return, subset=["Return_pct", "Cumulative_Return"])
        .set_properties(**{
            "background-color": BG2,
            "color": TEXT,
            "font-family": "monospace",
            "font-size": "0.82rem",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", BG),
                ("color", CYAN),
                ("font-family", "monospace"),
                ("font-size", "0.80rem"),
                ("text-transform", "uppercase"),
                ("letter-spacing", "1px"),
                ("border-bottom", f"1px solid {DIM}"),
            ]},
            {"selector": "td", "props": [
                ("border-bottom", f"1px solid rgba(74,74,106,0.3)"),
            ]},
        ])
    )

    st.dataframe(styled, use_container_width=True, height=min(60 + len(display) * 38, 480))


# ---------------------------------------------------------------------------
# Section heading helper
# ---------------------------------------------------------------------------
def _section_heading(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div style="margin:28px 0 12px 0;">'
        f'<div style="color:{CYAN};font-family:monospace;font-size:0.7rem;'
        f'text-transform:uppercase;letter-spacing:3px;color:{DIM};">'
        f'Long-Term Analysis</div>'
        f'<div style="color:{CYAN};font-family:monospace;font-size:1.1rem;'
        f'font-weight:700;letter-spacing:1px;">{title}</div>'
        f'{"<div style=" + chr(34) + "color:" + DIM + ";font-size:0.8rem;font-family:monospace;" + chr(34) + ">" + subtitle + "</div>" if subtitle else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main page render
# ---------------------------------------------------------------------------
def render_long_trend_page() -> None:
    """
    Main entry point for the Long-Term Trend & Drawdown Streamlit page.
    Call this from your main app router.
    """
    # ── Page config ───────────────────────────────────────────────────────
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {BG};
        }}
        .block-container {{
            padding-top: 1rem;
        }}
        div[data-testid="stDataFrame"] {{
            background-color: {BG2};
            border: 1px solid {DIM};
            border-radius: 4px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        f'<h1 style="'
        f'font-family:monospace;'
        f'color:{CYAN};'
        f'font-size:1.6rem;'
        f'letter-spacing:2px;'
        f'text-transform:uppercase;'
        f'border-bottom:1px solid {DIM};'
        f'padding-bottom:12px;'
        f'margin-bottom:4px;'
        f'text-shadow:0 0 20px {CYAN}66;'
        f'">Long-Term Trend &amp; Drawdown Analysis</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{DIM};font-family:monospace;font-size:0.8rem;'
        f'margin-bottom:24px;">EMA cross strategy · Rick Rule signals · Cycle position</p>',
        unsafe_allow_html=True,
    )

    # ── Sidebar controls ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f'<div style="color:{CYAN};font-family:monospace;font-size:0.8rem;'
            f'text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">'
            f'Long-Term Analysis</div>',
            unsafe_allow_html=True,
        )

        # Ticker selector — categorised dropdown with all universes
        full_registry = _build_trend_ticker_registry()

        if full_registry:
            # Group by category for the dropdown
            categories = sorted(set(v.get("_category", "") for v in full_registry.values()))
            selected_cat = st.selectbox(
                "Category",
                ["All"] + categories,
                index=0,
                key="lt_category",
            )

            # Filter registry by category
            if selected_cat != "All":
                filtered_reg = {k: v for k, v in full_registry.items() if v.get("_category") == selected_cat}
            else:
                filtered_reg = full_registry

            ticker_options = {
                f"{v.get('name', k)} ({k})": k
                for k, v in sorted(filtered_reg.items(), key=lambda x: x[1].get("name", x[0]))
            }
            display_names = list(ticker_options.keys())
            default_idx = 0
            for i, name in enumerate(display_names):
                if "VOLV" in name or "Volvo" in name:
                    default_idx = i
                    break
            selected_display = st.selectbox(
                "Stock / ETF",
                display_names,
                index=default_idx,
                key="lt_ticker",
            )
            ticker = ticker_options[selected_display]

            st.caption(f"{len(filtered_reg)} instruments in this category")
        else:
            ticker = st.text_input("Ticker (yfinance)", value="VOLV-B.ST")

        period = st.radio(
            "Analysis Period",
            options=["5y", "10y", "20y"],
            index=1,
            horizontal=True,
        )

        st.markdown("---")
        st.markdown(
            f'<div style="color:{DIM};font-size:0.72rem;font-family:monospace;">'
            f'Data: yfinance (price) + Börsdata (fundamentals if key set)</div>',
            unsafe_allow_html=True,
        )

    # ── Data loading ──────────────────────────────────────────────────────
    with st.spinner(f"Loading {ticker} — {period} history..."):
        analysis = run_long_trend_analysis(ticker, period)

    if analysis.get("error"):
        st.error(f"Data error: {analysis['error']}")
        return

    df = analysis["df"]
    drawdowns = analysis["drawdowns"]
    buy_signals = analysis["buy_signals"]
    sell_signals = analysis["sell_signals"]
    backtest = analysis["backtest"]
    trend_phase = analysis["trend_phase"]
    cycle_position = analysis["cycle_position"]
    rick_verdict = analysis["rick_verdict"]
    drawdown_summary = analysis["drawdown_summary"]

    if df.empty:
        st.warning(f"No price data available for {ticker}.")
        return

    # ── Quick stats bar ───────────────────────────────────────────────────
    last = df.dropna(subset=["Close"]).iloc[-1]
    first = df.dropna(subset=["Close"]).iloc[0]
    total_return_pct = (float(last["Close"]) - float(first["Close"])) / float(first["Close"]) * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_metric_box("Current Price", f"{last['Close']:.2f}", CYAN), unsafe_allow_html=True)
    with c2:
        clr = GREEN if last.get("EMA50", 0) > last.get("EMA200", 0) else RED
        ema50_val = f"{last['EMA50']:.2f}" if "EMA50" in df.columns and not pd.isna(last.get("EMA50")) else "N/A"
        st.markdown(_metric_box("EMA 50", ema50_val, clr), unsafe_allow_html=True)
    with c3:
        ema200_val = f"{last['EMA200']:.2f}" if "EMA200" in df.columns and not pd.isna(last.get("EMA200")) else "N/A"
        st.markdown(_metric_box("EMA 200", ema200_val, MAGENTA), unsafe_allow_html=True)
    with c4:
        tr_clr = GREEN if total_return_pct >= 0 else RED
        sign = "+" if total_return_pct >= 0 else ""
        st.markdown(_metric_box(f"Total Return ({period})", f"{sign}{total_return_pct:.1f}%", tr_clr), unsafe_allow_html=True)

    # ── Section 1: Price chart ─────────────────────────────────────────────
    _section_heading("Price Chart", f"EMA crossover · Drawdown zones · Rick Rule signals")
    fig = _build_price_chart(df, drawdowns, buy_signals, sell_signals, ticker)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    # ── Section 2: Drawdown table ──────────────────────────────────────────
    _section_heading(
        "Drawdown Analysis",
        f"All drawdowns > 10% — classified by cause",
    )
    _render_drawdown_table(drawdowns)

    # ── Section 3: Drawdown summary ────────────────────────────────────────
    if drawdowns:
        _render_drawdown_summary(drawdown_summary, len(drawdowns))

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

    # ── Section 4: Where Are We Now? ───────────────────────────────────────
    _section_heading("Where Are We Now?", "Trend phase · Cycle position · Rick Rule verdict")
    _render_where_are_we(trend_phase, cycle_position, rick_verdict)

    # Description of current cycle phase
    cycle_descriptions = {
        "Accumulation": "Price consolidating near EMA200. Smart money accumulating. Volume declining. Potential reversal ahead.",
        "Early Uptrend": "Price has just crossed above EMA200. EMA50 crossing EMA200. Trend reversal confirmed — early entry zone.",
        "Strong Uptrend": "Price well above EMA200. Both EMAs rising. Risk/reward favors holding positions.",
        "Late Uptrend": "Price stretched far above EMA200 (>20%). RSI elevated. Consider tightening stops.",
        "Early Downtrend": "Price has crossed below EMA200. EMA50 turning down. Exit or reduce exposure.",
        "Capitulation": "Steep decline. Price well below EMA200. High volatility. Potential wash-out event.",
        "Recovery": "Price still below EMA200 but stabilising. EMA50 flattening. Watch for re-accumulation.",
    }
    desc = cycle_descriptions.get(cycle_position, "")
    if desc:
        cc = _cycle_color(cycle_position)
        st.markdown(
            f'<div style="'
            f'background:{BG2};border:1px solid {cc}33;border-left:2px solid {cc};'
            f'border-radius:4px;padding:10px 16px;margin-top:8px;">'
            f'<span style="color:{DIM};font-size:0.78rem;font-family:monospace;">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Section 5: Rick Rule backtest ─────────────────────────────────────
    _section_heading(
        "Rick Rule Strategy Backtest",
        "Simulated buy/sell based on EMA200 crossover + fundamentals",
    )
    _render_backtest(backtest)

    # Footer
    st.markdown(
        f'<div style="margin-top:32px;padding-top:16px;border-top:1px solid {DIM};'
        f'color:{DIM};font-size:0.72rem;font-family:monospace;">'
        f'Strategy: BUY when price > EMA200 &amp; EMA50 > EMA200 &amp; fundamentals intact. '
        f'SELL when price below EMA200 for 10+ consecutive days. '
        f'Not financial advice.'
        f'</div>',
        unsafe_allow_html=True,
    )
