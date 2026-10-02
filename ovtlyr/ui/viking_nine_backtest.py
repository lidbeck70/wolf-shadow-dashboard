"""
ovtlyr/ui/viking_nine_backtest.py — PORTFOLIO → Backtest → ⚔️ Viking Nine.

Kör viking_backtest över en lista tickers (förval: senaste ⚔️ Viking Nine-
skanningen) och visar nyckeltalen i R: antal affärer, win rate, snitt- och
median-R, profit factor, max drawdown, snittvinnare/-förlorare, expectancy,
flest förluster i rad och snittinnehav — plus R-kurvan, exitorsakerna och
varje affär. Utan look-ahead; begränsningarna står under resultatet.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import viking_backtest as vb
import viking_screen as vs
from ovtlyr.ui.viking_screens import _ROWS, _sector
from ui.charts import PLOTLY_LAYOUT
from ui.components import big_card, note, page_header
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, RED, TEXT

_RES = "vnb_result"


def _default_tickers() -> str:
    data = st.session_state.get(_ROWS) or {}
    rows = [r for r in vs.results(data.get("rows") or []) if r.get("nine") is not None]
    return ", ".join(r["ticker"] for r in rows[:20])


def _fmt(v, f="{:+.2f}R") -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    return f.format(v)


def curve_chart(m: dict) -> go.Figure:
    pts = m.get("curve") or []
    fig = go.Figure(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines",
                               line=dict(color=CYAN, width=1.6), name="Summa R"))
    fig.add_hline(y=0, line=dict(color=DIM, width=1, dash="dot"))
    layout = dict(PLOTLY_LAYOUT)
    layout.update(height=300, showlegend=False, title=dict(text="R-KURVA (STÄNGDA AFFÄRER)",
                                                           font=dict(size=12, color=CYAN)),
                  yaxis=dict(title="R", gridcolor="rgba(255,255,255,0.05)"))
    fig.update_layout(**layout)
    return fig


def render_viking_nine_backtest() -> None:
    page_header("⚔️ Viking Nine — backtest", "OVTLYR Nine + Viking Execution + exitmotorn bakåt i tiden, "
                                            "rapporterat i R. Inga data som inte var kända vid entry.")
    with st.form("vnb_form", clear_on_submit=False):
        raw = st.text_area("Tickers (förval: senaste ⚔️ Viking Nine-skanningen)", _default_tickers(),
                           key="vnb_tickers", height=70, placeholder="t.ex. NVDA, MSFT, VOLV-B.ST")
        c1, c2, c3 = st.columns(3)
        years = c1.selectbox("Period (år)", [1, 2, 3, 5], index=2, key="vnb_years")
        min_nine = c2.selectbox("Minsta Nine för entry", [9, 8, 7], index=0, key="vnb_min_nine",
                                help="9 = systemets regel (GOLDEN TICKET). 8/7 visar vad en lösare regel hade gett.")
        need_vol = c3.checkbox("Kräv relativ volym", value=False, key="vnb_vol")
        go_ = st.form_submit_button("⚔️ Kör backtest")
    tickers = vs.parse_tickers(raw)
    if go_ and tickers:
        bar = st.progress(0.0, text="Hämtar SPY och sektor-ETF:er …")
        res = vb.run(tickers, sector_getter=_sector,
                     cfg=vb.Config(min_nine=int(min_nine), require_volume=bool(need_vol), years=int(years)),
                     progress=lambda i, n, t: bar.progress(i / n, text=f"{t} ({i}/{n})"))
        bar.empty()
        st.session_state[_RES] = res
    res = st.session_state.get(_RES)
    if not res:
        note("Välj tickers och tryck ⚔️ Kör backtest. Tips: kör ⚔️ Viking Nine-skanningen först så fylls "
             "kandidaterna i här.")
        return
    render_result(res)


def render_result(res: dict) -> None:
    m, cfg = res["metrics"], res["config"]
    st.markdown(f"<div style='color:{DIM};font-size:0.78rem;'>{len(res['per_ticker'])} tickers · {cfg.years} år · "
                f"entry vid Nine ≥ {cfg.min_nine}/9 · volymkrav {'på' if cfg.require_volume else 'av'}</div>",
                unsafe_allow_html=True)
    if not m.get("trades"):
        note("Inga stängda affärer under perioden. Prova fler tickers, längre period eller lägre minsta Nine "
             "för att se hur ofta setupen uppstår.")
    else:
        exp = m["expectancy"]
        cards = [
            ("AFFÄRER", f"{m['trades']}", f"snittinnehav {m['avg_holding_days']:g} handelsdagar", CYAN),
            ("WIN RATE", f"{m['win_rate']:g} %", f"flest förluster i rad: {m['max_consecutive_losses']}", CYAN),
            ("EXPECTANCY", _fmt(exp), f"{m['win_rate'] / 100:.2f} × {m['avg_winner']:.2f} − "
                                       f"{1 - m['win_rate'] / 100:.2f} × {abs(m['avg_loser']):.2f}",
             GREEN if exp > 0 else RED),
            ("PROFIT FACTOR", _fmt(m["profit_factor"], "{:.2f}"), "vinster i R / förluster i R",
             GREEN if (m["profit_factor"] or 0) > 1 else RED),
            ("SNITT / MEDIAN", f"{_fmt(m['avg_r'])} / {_fmt(m['median_r'])}", f"summa {_fmt(m['total_r'])}", GOLD),
            ("MAX DRAWDOWN", _fmt(-m["max_drawdown_r"]), f"snittvinnare {_fmt(m['avg_winner'])} · snittförlorare "
                                                         f"{_fmt(m['avg_loser'])}", AMBER),
        ]
        cols = st.columns(3)
        for k, (title, big, sub, col) in enumerate(cards):
            cols[k % 3].markdown(big_card(title, big, sub, col), unsafe_allow_html=True)
        try:
            st.plotly_chart(curve_chart(m), use_container_width=True, config={"displayModeBar": False},
                            key="vnb_curve")
        except Exception:
            pass
        trades = [t for t in res["trades"] if not t.open]
        by = pd.DataFrame([{"Exit": t.exit_reason, "R": t.r} for t in trades]).groupby("Exit")["R"]
        rows = "".join(f"<tr><td style='text-align:left;'>{k}</td><td>{int(v.count())}</td>"
                       f"<td>{v.mean():+.2f}R</td></tr>" for k, v in by)
        st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:8px;'>EXITORSAKER"
                    f"</div><table style='font-size:0.78rem;color:{TEXT};text-align:right;'><tr style='color:{DIM};'>"
                    f"<th style='text-align:left;'>Exit</th><th>Antal</th><th>Snitt</th></tr>{rows}</table>",
                    unsafe_allow_html=True)
        with st.expander(f"Alla affärer ({len(res['trades'])})"):
            st.dataframe(pd.DataFrame([{
                "Ticker": t.ticker, "Signal": t.signal_date, "Entry": t.entry_date, "Pris in": round(t.entry, 2),
                "Stopp": round(t.stop, 2), "Exit": t.exit_date, "Pris ut": round(t.exit, 2), "Orsak": t.exit_reason,
                "R": t.r, "Dagar": t.days, "Nine": f"{t.nine}/9"} for t in res["trades"]]),
                hide_index=True, width="stretch")
    with st.expander("Per ticker"):
        st.dataframe(pd.DataFrame([{
            "Ticker": p["ticker"], "Sektor-ETF": p.get("sector_etf") or "—", "Signaler": p["signals"],
            "Affärer": len(p["trades"]), "No chase": p["no_chase"], "R/R < 2": p["low_rr"],
            "Data": p.get("error") or "ok"} for p in res["per_ticker"]]), hide_index=True, width="stretch")
    note("Expectancy = win rate × snittvinnare − förlustandel × snittförlorare, i R. " + " ".join(res["notes"]))
