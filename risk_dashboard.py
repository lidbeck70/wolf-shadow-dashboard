"""
risk_dashboard.py — Risk Dashboard for Holdings tab

Shows total exposure, worst-case scenario, sector concentration,
and strategy allocation for all open positions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Nordic Gold theme
BG = "#0c0c12"
BG2 = "#14141e"
CYAN = "#c9a84c"
MAGENTA = "#8b7340"
GREEN = "#2d8a4e"
RED = "#c44545"
YELLOW = "#d4943a"
TEXT = "#e8e4dc"
DIM = "#8a8578"

DEFAULT_CAPITAL = 500_000

STRATEGY_MAP = {
    "swing": "Wolf",
    "ovtlyr": "Viking",
    "long": "Alpha",
}

STRATEGY_COLORS = {
    "Wolf": CYAN,
    "Viking": MAGENTA,
    "Alpha": GREEN,
}


@st.cache_data(ttl=3600, show_spinner=False, max_entries=100)
def _fetch_atr(ticker: str, period: str = "3mo") -> float:
    """Fetch 14-day ATR for a ticker. Returns 0 on failure."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, auto_adjust=True)
        if df.empty or len(df) < 15:
            return 0.0
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        return atr
    except Exception:
        return 0.0


@st.cache_data(ttl=900, show_spinner=False, max_entries=100)
def _fetch_price(ticker: str) -> float:
    """Fetch current price for a ticker."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="5d", auto_adjust=True)
        if df.empty:
            return 0.0
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return float(df["Close"].iloc[-1])
    except Exception:
        return 0.0


def _build_positions(holdings_data: dict) -> List[dict]:
    """Build a flat list of positions with computed fields."""
    positions = []
    for pkey, holdings in holdings_data.items():
        strategy = STRATEGY_MAP.get(pkey, pkey)
        for h in holdings:
            ticker = h.get("ticker", "")
            entry = h.get("entry_price", 0)
            sector = h.get("sector", "Unknown")
            shares = h.get("shares", 1)
            price = _fetch_price(ticker)
            atr = _fetch_atr(ticker)
            sl = entry - 0.5 * atr if entry > 0 and atr > 0 else 0
            sl_distance = entry - sl if sl > 0 else 0
            risk_amount = sl_distance * shares
            position_value = price * shares if price > 0 else entry * shares
            positions.append({
                "ticker": ticker,
                "strategy": strategy,
                "sector": sector,
                "entry_price": entry,
                "current_price": price,
                "shares": shares,
                "atr": atr,
                "sl": sl,
                "sl_distance": sl_distance,
                "risk_amount": risk_amount,
                "position_value": position_value,
            })
    return positions


def _render_kpi_card(label: str, value: str, sub: str = "", color: str = CYAN):
    """Render a styled KPI card."""
    st.markdown(
        f"<div style='background:{BG2};border-radius:8px;padding:16px 12px;text-align:center;"
        f"border:1px solid rgba(201,168,76,0.15);'>"
        f"<div style='color:{DIM};font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;'>{label}</div>"
        f"<div style='color:{color};font-size:1.4rem;font-weight:700;margin:4px 0;'>{value}</div>"
        f"<div style='color:{DIM};font-size:0.6rem;'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_risk_dashboard(holdings_data: dict):
    """Render the Risk Dashboard section within Holdings tab."""
    try:
        # Section header
        st.markdown(
            f"<div style='border-bottom:2px solid {CYAN};padding-bottom:6px;margin-bottom:16px;'>"
            f"<h3 style='color:{CYAN};margin:0;letter-spacing:0.08em;'>RISK DASHBOARD</h3>"
            f"<span style='color:{DIM};font-size:0.7rem;'>Totalexponering, sektor\u00adkoncentration & worst-case</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        positions = _build_positions(holdings_data)
        if not positions:
            st.info("Inga innehav att analysera.")
            return

        # Capital input
        capital = st.number_input(
            "Totalt kapital (SEK)", value=DEFAULT_CAPITAL,
            min_value=10000, step=10000,
            key="risk_capital_input",
            help="Används f\u00f6r att ber\u00e4kna risk i % av kapital",
        )

        # ── 1. Total Exposure KPIs ──────────────────────────────────
        total_invested = sum(p["entry_price"] * p["shares"] for p in positions if p["entry_price"] > 0)
        total_at_risk = sum(p["risk_amount"] for p in positions)
        risk_pct = (total_at_risk / capital * 100) if capital > 0 else 0
        n_positions = len(positions)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _render_kpi_card(
                "Total investerat",
                f"{total_invested:,.0f} SEK",
                f"{n_positions} positioner",
            )
        with c2:
            _render_kpi_card(
                "Total risk (SL)",
                f"{total_at_risk:,.0f} SEK",
                "Summa SL-avst\u00e5nd \u00d7 andelar",
                color=RED if total_at_risk > capital * 0.1 else CYAN,
            )
        with c3:
            _render_kpi_card(
                "Risk % av kapital",
                f"{risk_pct:.1f}%",
                f"av {capital:,.0f} SEK",
                color=RED if risk_pct > 10 else YELLOW if risk_pct > 5 else GREEN,
            )
        with c4:
            _render_kpi_card(
                "\u00d6ppna positioner",
                str(n_positions),
                f"Wolf {sum(1 for p in positions if p['strategy']=='Wolf')} · "
                f"Viking {sum(1 for p in positions if p['strategy']=='Viking')} · "
                f"Alpha {sum(1 for p in positions if p['strategy']=='Alpha')}",
            )

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # ── 2. Worst Case Scenario ──────────────────────────────────
        try:
            worst_case_loss = sum(
                p["sl_distance"] * p["shares"]
                for p in positions
                if p["sl"] > 0
            )
            worst_pct = (worst_case_loss / capital * 100) if capital > 0 else 0
            wc_color = RED if worst_pct > 10 else YELLOW if worst_pct > 5 else GREEN

            st.markdown(
                f"<div style='background:{BG2};border:1px solid rgba(196,69,69,0.3);border-radius:8px;"
                f"padding:16px 20px;margin-bottom:16px;'>"
                f"<div style='color:{DIM};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;"
                f"margin-bottom:4px;'>WORST CASE SCENARIO</div>"
                f"<div style='color:{wc_color};font-size:1.2rem;font-weight:700;'>"
                f"Om alla SL triggar: -{worst_case_loss:,.0f} SEK (-{worst_pct:.1f}%)</div>"
                f"<div style='color:{DIM};font-size:0.6rem;margin-top:4px;'>"
                f"SL = entry - 0.5 \u00d7 ATR(14) f\u00f6r varje inneh\u00e5ll</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass

        # ── 3. Sector Concentration ─────────────────────────────────
        try:
            sector_values = {}
            for p in positions:
                s = p["sector"] if p["sector"] and p["sector"] != "Unknown" else "Okänd"
                sector_values[s] = sector_values.get(s, 0) + p["position_value"]

            total_value = sum(sector_values.values())
            if total_value > 0 and sector_values:
                sectors = sorted(sector_values.keys(), key=lambda s: sector_values[s], reverse=True)
                pcts = [sector_values[s] / total_value * 100 for s in sectors]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=sectors,
                    x=pcts,
                    orientation="h",
                    marker=dict(color="rgba(201,168,76,0.8)"),
                    text=[f"{v:.1f}%" for v in pcts],
                    textposition="auto",
                    textfont=dict(color=TEXT, size=11),
                    hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
                ))

                # 25% threshold line
                fig.add_vline(
                    x=25, line_dash="dash",
                    line_color="rgba(196,69,69,0.7)", line_width=2,
                    annotation_text="25% gräns",
                    annotation_font_color=RED,
                    annotation_font_size=10,
                )

                fig.update_layout(
                    title=dict(
                        text="SEKTORKONCENTRATION",
                        font=dict(color=CYAN, size=14),
                    ),
                    paper_bgcolor=BG,
                    plot_bgcolor=BG,
                    font=dict(color=TEXT),
                    xaxis=dict(
                        title="% av portföljvärde",
                        tickfont=dict(color=DIM, size=10),
                        gridcolor="rgba(201,168,76,0.1)",
                        range=[0, max(max(pcts) * 1.15, 30)],
                    ),
                    yaxis=dict(
                        tickfont=dict(color=CYAN, size=10),
                        gridcolor="rgba(201,168,76,0.05)",
                    ),
                    height=max(200, 40 * len(sectors) + 80),
                    margin=dict(l=120, r=40, t=50, b=40),
                    showlegend=False,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Warnings for over-concentrated sectors
                for s, pct in zip(sectors, pcts):
                    if pct > 25:
                        st.warning(f"Sektor \u00f6verkoncentrerad: {s} = {pct:.1f}% (gr\u00e4ns 25%)")
        except Exception:
            pass

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # ── 4. Strategy Allocation ──────────────────────────────────
        try:
            strat_data = {}
            for s_name in ["Wolf", "Viking", "Alpha"]:
                s_positions = [p for p in positions if p["strategy"] == s_name]
                s_value = sum(p["position_value"] for p in s_positions)
                strat_data[s_name] = {
                    "value": s_value,
                    "count": len(s_positions),
                }

            sc1, sc2, sc3 = st.columns(3)
            for col, (s_name, data) in zip([sc1, sc2, sc3], strat_data.items()):
                with col:
                    color = STRATEGY_COLORS.get(s_name, CYAN)
                    _render_kpi_card(
                        f"{s_name} Portfolio",
                        f"{data['value']:,.0f} SEK",
                        f"{data['count']} positioner",
                        color=color,
                    )
        except Exception:
            pass

    except Exception as e:
        logger.warning("Risk dashboard error: %s", e)
        st.warning(f"Risk Dashboard kunde inte laddas: {e}")
