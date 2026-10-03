"""
swing_backtest_ui.py — PORTFOLIO → Backtest → 📈 Momentum Swing.

Kör swing_backtest över svenska Large + Mid Cap (Börsdata) med veckorutinens
regler, som en portfölj: kontokurva mot OMXSPI, CAGR, max drawdown,
vinstandel, payoff-kvot (regelns mål > 2,0), exitorsaker och varje affär.
'Jämför varianter' kör samma data utan setup-krav och utan regimfilter.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import swing_backtest as sb
from ui.charts import PLOTLY_LAYOUT
from ui.components import big_card, note, page_header
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, GREY, RED, TEXT

_DATA, _RES = "sbt_data", "sbt_result"
REGIME_COLOR = {sb.GREEN: GREEN, sb.YELLOW: AMBER, sb.RED: RED, sb.UNKNOWN: GREY}


def _fmt(v, f="{:+.1f} %") -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    return f.format(v)


def _data(years: int) -> dict:
    """Kursdata för perioden — hämtas en gång per session och period (alla varianter delar den)."""
    hit = st.session_state.get(_DATA)
    if hit and hit.get("years") == years:
        return hit["data"]
    bar = st.progress(0.0, text="Hämtar Large + Mid Cap …")
    data = sb.load_data(years, progress=lambda i, n, t: bar.progress(i / max(n, 1), text=f"{t} ({i}/{n})"))
    bar.empty()
    st.session_state[_DATA] = {"years": years, "data": data}
    return data


def comparison_rows(runs: dict) -> list:
    rows = []
    for name, r in runs.items():
        m, b = r["metrics"], r.get("benchmark") or {}
        rows.append({"Variant": name, "Affärer": m.get("trades", 0), "Win rate %": m.get("win_rate"),
                     "Payoff": m.get("payoff"), "Profit factor": m.get("profit_factor"),
                     "Total %": m.get("total_return_pct"), "CAGR %": m.get("cagr_pct"),
                     "Max DD %": m.get("max_dd_pct"), "Exponering %": m.get("exposure_pct"),
                     "OMXSPI CAGR %": b.get("cagr_pct")})
    return rows


def equity_chart(r: dict) -> go.Figure:
    eq, b = r["equity"], (r.get("benchmark") or {}).get("curve")
    fig = go.Figure()
    if eq is not None and len(eq):
        fig.add_trace(go.Scatter(x=eq.index, y=(eq / eq.iloc[0] * 100).values, name="Momentum Swing",
                                 line=dict(color=CYAN, width=1.8)))
    if b is not None and len(b):
        fig.add_trace(go.Scatter(x=b.index, y=(b / b.iloc[0] * 100).values, name="OMXSPI",
                                 line=dict(color=GREY, width=1.4, dash="dot")))
    lay = dict(PLOTLY_LAYOUT)
    lay.update(height=320, title=dict(text="KONTOT MOT OMXSPI (START = 100)", font=dict(size=12, color=CYAN)),
               legend=dict(orientation="h", y=-0.15), yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False))
    fig.update_layout(**lay)
    return fig


def render_swing_backtest_page() -> None:
    page_header("📈 Momentum Swing — backtest", "Veckorutinen bakåt i tiden på Large + Mid Cap: ranking, setup, "
                                                "regim och de tre säljreglerna — som en portfölj, utan look-ahead.")
    with st.form("sbt_form"):
        c1, c2, c3 = st.columns(3)
        years = c1.selectbox("Period (år)", [3, 5, 10], index=1, key="sbt_years")
        size = c2.slider("Positionsstorlek GRÖN regim (%)", 12, 20, 16, key="sbt_size",
                         help="Regeln: 12–20 % av swing-kapitalet. GUL regim = halva.")
        fee = c3.number_input("Courtage per transaktion (%)", 0.0, 0.5, 0.1, 0.05, key="sbt_fee")
        compare = st.checkbox("Jämför varianter (utan setup-krav · utan regimfilter)", value=True, key="sbt_compare")
        go_ = st.form_submit_button("📈 Kör backtest")
    if go_:
        data = _data(int(years))
        pn = sb.panels(data["prices"], data["index"])
        names = list(sb.VARIANTS) if compare else ["Reglerna"]
        runs = {}
        with st.spinner("Simulerar veckorutinen …"):
            for name in names:
                cfg = sb.Config(years=int(years), size_green=float(size), size_yellow=float(size) / 2,
                                fee_pct=float(fee), **sb.VARIANTS[name])
                runs[name] = sb.run(pn, data["index"], cfg)
        st.session_state[_RES] = {"runs": runs, "source": data["source"], "missing": data.get("missing", 0),
                                  "universe": data.get("universe", 0)}
    res = st.session_state.get(_RES)
    if not res:
        note("Välj period och tryck 📈 Kör backtest. Första körningen hämtar kurser för hela Large + Mid Cap "
             "(några hundra bolag) och tar en stund; varianterna återanvänder samma data.")
        return
    note(f"Data: {res['source']}" + (f" · {res['missing']} bolag utan kurser" if res.get("missing") else ""))
    runs = res["runs"]
    if len(runs) > 1:
        rows = comparison_rows(runs)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        note("Samma data och regler — 'Utan setup-krav' köper topp 20 utan A/B-krav, 'Utan regimfilter' köper "
             "även i RÖD och GUL regim med full storlek. Visar om reglerna gör nytta.")
        shown = st.selectbox("Visa detaljer för", list(runs), key="sbt_show")
    else:
        shown = next(iter(runs))
    render_result(runs[shown])


def render_result(r: dict) -> None:
    m, b = r["metrics"], r.get("benchmark") or {}
    if not m.get("trades"):
        note("Inga stängda affärer under perioden.")
    cards = [
        ("CAGR", _fmt(m.get("cagr_pct")), f"OMXSPI {_fmt(b.get('cagr_pct'))}",
         GREEN if (m.get("cagr_pct") or 0) > (b.get("cagr_pct") or 0) else AMBER),
        ("TOTALT", _fmt(m.get("total_return_pct")), f"OMXSPI {_fmt(b.get('total_return_pct'))}", GOLD),
        ("MAX DRAWDOWN", _fmt(-(m.get("max_dd_pct") or 0)), f"OMXSPI {_fmt(-(b.get('max_dd_pct') or 0))}", AMBER),
        ("AFFÄRER", f"{m.get('trades', 0)}", f"win rate {_fmt(m.get('win_rate'), '{:.0f} %')} · snitt "
                                             f"{_fmt(m.get('avg_days'), '{:.0f}')} dagar", CYAN),
        ("PAYOFF-KVOT", _fmt(m.get("payoff"), "{:.2f}"), f"mål > 2,0 · snittvinst {_fmt(m.get('avg_win_pct'))} / "
                                                          f"förlust {_fmt(m.get('avg_loss_pct'))}",
         GREEN if (m.get("payoff") or 0) >= 2 else AMBER),
        ("EXPONERING", _fmt(m.get("exposure_pct"), "{:.0f} %"), f"profit factor {_fmt(m.get('profit_factor'), '{:.2f}')} "
                                                                 f"· halvsålt {_fmt(m.get('half_sold_share'), '{:.0f} %')}",
         CYAN),
    ]
    cols = st.columns(3)
    for k, (title, big, sub, col) in enumerate(cards):
        cols[k % 3].markdown(big_card(title, big, sub, col), unsafe_allow_html=True)
    try:
        st.plotly_chart(equity_chart(r), use_container_width=True, config={"displayModeBar": False}, key="sbt_curve")
    except Exception:
        pass
    share = r.get("regime_share") or {}
    if share:
        st.markdown("<div style='font-size:0.78rem;margin:4px 0;'>Regim under perioden: " + " · ".join(
            f"<span style='color:{REGIME_COLOR.get(k, GREY)};font-weight:700;'>{k} {v * 100:.0f} %</span>"
            for k, v in share.items()) + "</div>", unsafe_allow_html=True)
    ex = sb.exit_table(r["trades"])
    if ex:
        body = "".join(f"<tr><td style='text-align:left;'>{e['Exit']}</td><td>{e['Antal']}</td>"
                       f"<td>{e['Snitt %']:+.2f} %</td><td>{e['Summa %']:+.1f} %</td></tr>" for e in ex)
        st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:8px;'>EXITORSAKER</div>"
                    f"<table style='font-size:0.78rem;color:{TEXT};text-align:right;'><tr style='color:{DIM};'>"
                    f"<th style='text-align:left;'>Exit</th><th>Antal</th><th>Snitt</th><th>Summa</th></tr>{body}</table>",
                    unsafe_allow_html=True)
    with st.expander(f"Alla affärer ({len(r['trades'])})"):
        st.dataframe(pd.DataFrame([{
            "Ticker": t.ticker, "Entry": t.entry_date, "Pris in": round(t.entry, 2), "Setup": t.setup, "Regim": t.regime,
            "Exit": t.exit_date, "Pris ut": None if t.exit is None else round(t.exit, 2),
            "Orsak": "öppen" if t.open else t.reason, "Avkastning %": t.ret_pct, "Dagar": t.days,
            "Halva såld": "ja" if t.half_sold else ""} for t in r["trades"]]), hide_index=True, width="stretch")
    note(" ".join(sb.NOTES) + " Regelns tumregel: dra inga slutsatser på färre än 15–20 affärer; normal vinstandel "
         "är 40–55 %.")
