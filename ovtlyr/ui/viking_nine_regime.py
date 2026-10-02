"""
ovtlyr/ui/viking_nine_regime.py — REGIME → Marknad → Arc Regime → ⚔️ Viking Nine Regime.

Beslutssidan för EN aktie i Viking Nine-systemet — screenern (⚔️ Viking Nine)
hittar kandidaterna, här tas beslutet:

  MARKET (SPY + bredd)  →  OVTLYR NINE (setup)  →  graf  →
  VIKING EXECUTION + RISK ENGINE (GOLDEN TICKET / WAIT / NO TRADE)  →
  VIKING EXIT ENGINE (öppen position)  →  signallogg

Samma motorer som screenern: ovtlyr_nine, viking_execution, viking_exit.
Den gamla Viking Regime lämnas orörd.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ovtlyr_nine as on
import storage
import viking_execution as vx
import viking_screen as vs
from ovtlyr.ui.execution_card import render_execution
from ovtlyr.ui.exit_card import render_exit_section
from ovtlyr.ui.nine_card import render_nine_card
from ovtlyr.ui.viking_screens import _ROWS, _earnings, _market_banner, _render_log, _sector
from ui.charts import PLOTLY_LAYOUT
from ui.components import note, page_header
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, RED, TEXT

_TICKER = "vnr_ticker"


def _section(title: str) -> None:
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;"
                f"margin:14px 0 6px 0;'>{title}</div>", unsafe_allow_html=True)


def _scanned() -> list:
    """[(ticker, kategori)] från senaste ⚔️ Viking Nine-skanningen, bästa först."""
    data = st.session_state.get(_ROWS) or {}
    rows = vs.results(data.get("rows") or [])
    return [(r["ticker"], r["category"]) for r in rows if r.get("nine") is not None]


def _load(ticker: str) -> Optional[pd.DataFrame]:
    try:
        from market_prices import ohlcv
        df = ohlcv(ticker, on.PERIOD)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def price_chart(ticker: str, df: pd.DataFrame, decision: Optional[vx.EntryDecision]) -> go.Figure:
    d = df.tail(130)
    c = df["Close"].astype(float)
    fig = go.Figure(go.Candlestick(x=list(d.index), open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                   name=ticker, increasing_line_color=GREEN, decreasing_line_color=RED))
    for n, col in ((10, CYAN), (20, AMBER), (50, GOLD)):
        e = c.ewm(span=n, adjust=False).mean().tail(130)
        fig.add_trace(go.Scatter(x=list(e.index), y=list(e.values), name=f"EMA{n}", line=dict(color=col, width=1.2)))
    if decision is not None and decision.position is not None:
        p = decision.position
        lines = [(p.entry, f"entry {p.entry:,.2f}", TEXT, "solid"), (p.stop, f"stopp {p.stop:,.2f}", RED, "dash")]
        if decision.resistance is not None:
            lines.append((decision.resistance, f"motstånd {decision.resistance:,.2f}", AMBER, "dot"))
        for y, label, col, dash in lines:
            fig.add_hline(y=y, line=dict(color=col, width=1, dash=dash), annotation_text=label,
                          annotation_font=dict(color=col, size=10), annotation_position="top left")
    layout = dict(PLOTLY_LAYOUT)
    layout.update(height=380, showlegend=True, xaxis_rangeslider_visible=False,
                  title=dict(text=f"{ticker} — DAGLIG, EMA10/20/50", font=dict(size=12, color=CYAN)),
                  legend=dict(orientation="h", y=1.02, x=0))
    fig.update_layout(**layout)
    return fig


def render_viking_nine_regime_page() -> None:
    page_header("⚔️ Viking Nine Regime", "Beslutet för en aktie: OVTLYR Nine (setup) → Viking Execution och "
                                         "risk (entry) → exit. Kandidaterna hittas i SCREENING → ⚔️ Viking Nine.")
    _market_banner()

    scanned = _scanned()
    st.session_state.setdefault(_TICKER, "SPY")
    c1, c2 = st.columns([2, 3])
    if scanned:
        labels = ["—"] + [f"{t} · {cat}" for t, cat in scanned]
        pick = c2.selectbox("Från senaste skanningen", labels, key="vnr_pick")
        if pick != st.session_state.get("vnr_last_pick"):          # bara när valet ändras — annars vinner fritext
            st.session_state["vnr_last_pick"] = pick
            if pick != "—":
                st.session_state[_TICKER] = pick.split(" · ")[0]
    ticker = (c1.text_input("TICKER", key=_TICKER) or "").strip().upper()
    if not scanned:
        note("Kör ⚔️ Viking Nine under SCREENING för att få kandidaterna i en lista här — eller skriv en ticker.")
    if not ticker:
        return

    df = _load(ticker)
    if df is None or len(df) < on.MIN_BARS:
        note(f"DATA UNAVAILABLE — ingen kurshistorik för {ticker} (kräver {on.MIN_BARS} dagar).")
        return
    ob = vs._ob_analysis(df)
    nine = on.evaluate(ticker, stock_df=df, ob_analysis=ob, sector_getter=_sector)

    _section("OVTLYR NINE — SETUP")
    render_nine_card(nine)

    _section("VIKING EXECUTION · RISK ENGINE · FINAL DECISION")
    k1, k2 = st.columns(2)
    capital = k1.number_input("Kapital (SEK)", min_value=0.0, value=100000.0, step=10000.0, key="vnr_capital")
    max_pos = k2.number_input("Max position (% av kapitalet)", min_value=5.0, max_value=100.0,
                              value=float(vx.MAX_POSITION_PCT), step=5.0, key="vnr_max_pos")
    try:
        from trade_journal import load_journal
        trades = load_journal()
    except Exception:
        trades = []
    earnings = _earnings(ticker)
    decision = vx.evaluate_entry(ticker, df, nine=nine, capital=capital, ob_analysis=ob, earnings_date=earnings,
                                 earnings_known=earnings is not None, trades=trades, max_position_pct=max_pos)
    try:
        st.plotly_chart(price_chart(ticker, df, decision), use_container_width=True,
                        config={"displayModeBar": False}, key="vnr_chart")
    except Exception as exc:
        note(f"Grafen kunde inte ritas: {exc}")
    render_execution(decision)

    category = vx.watchlist_category(nine.passed, decision.status)
    log = storage.session_load(vs.LOG_STORE, [])
    if category in vs.LOGGED_CATEGORIES:
        new_log, n = vs.append_log(log, [{"ticker": ticker, "nine": nine, "decision": decision,
                                          "category": category}], source="viking_nine_regime")
        if n:
            st.session_state[vs.LOG_STORE] = new_log
            log = new_log

    render_exit_section(ticker, df, nine, ob, earnings)

    with st.expander(f"📜 Signallogg ({len(log)} rader)"):
        _render_log(log)
    note("Appen förutsäger ingenting — den visar hur många av systemets villkor som är uppfyllda just nu. "
         f"Allt i OVTLYR Nine är {on.APPROXIMATION}: riktiga priser, panelens definitioner.")
