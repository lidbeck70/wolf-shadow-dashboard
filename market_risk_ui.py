"""
market_risk_ui.py — REGIME → Marknad → 🌩️ Marknadsrisk.

Överst nivån för SPY och OMXS30 sida vid sida. För vald marknad: vilka
varningar som lyser, index och poäng över tid, och den historiska
träffbilden — träffandel per nivå mot basfrekvensen, nedgångarna (varnade
eller missade) och larmen (träffar och falsklarm). En riskspärr, ingen
förutsägelse.
"""

from __future__ import annotations

import time

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import market_risk as mr
from ui.charts import PLOTLY_LAYOUT
from ui.components import big_card, note, page_header
from ui.tokens import AMBER, CYAN, DIM, GREEN, RED, TEXT

_CACHE = "mr_cache"
_TTL_S = 6 * 3600
LEVEL_COLOR = {"LÅG": GREEN, "FÖRHÖJD": AMBER, "HÖG": RED}


def _section(title: str, sub: str = "") -> None:
    st.markdown(f"<div style='color:{CYAN};font-family:Courier New;letter-spacing:2px;font-size:0.85rem;"
                f"margin:18px 0 6px;'>{title}" + (f" <span style='color:{DIM};letter-spacing:0;'>— {sub}</span>"
                                                  if sub else "") + "</div>", unsafe_allow_html=True)


def _load(market: str) -> mr.MarketRisk:
    cache = st.session_state.setdefault(_CACHE, {})
    hit = cache.get(market)
    if hit and time.time() - hit[0] < _TTL_S:
        return hit[1]
    with st.spinner(f"Räknar {mr.MARKETS[market]['label']} …"):
        res = mr.evaluate(market)
    cache[market] = (time.time(), res)
    return res


def _summary(r: mr.MarketRisk) -> str:
    if r.error:
        return big_card(r.label.upper(), "DATA UNAVAILABLE", r.error, DIM)
    cal = r.calibration
    lv = cal.by_level.get(r.level) if cal else None
    sub = f"{r.points} av {r.possible} varningar · {r.date}"
    if cal and lv and lv["hit_rate"] is not None:
        sub += f" · −{int(mr.DRAWDOWN * 100)} % inom 3 mån följde {lv['hit_rate']:g} % (normalt {cal.base_rate:g} %)"
    return big_card(r.label.upper(), r.level, sub, LEVEL_COLOR.get(r.level, DIM))


def history_chart(r: mr.MarketRisk) -> go.Figure:
    close, pts = r.close, r.history
    start = r.calibration.start if r.calibration else str(close.index[0].date())
    close, pts = close[close.index >= start], pts[pts.index >= start]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=list(close.index), y=list(close.values), name=r.label,
                             line=dict(color=CYAN, width=1.3)), row=1, col=1)
    cols = [LEVEL_COLOR[mr.level_of(int(p))] for p in pts.values]
    fig.add_trace(go.Bar(x=list(pts.index), y=list(pts.values), marker_color=cols, name="Varningar"), row=2, col=1)
    for _n, lo in mr.LEVELS[1:]:
        fig.add_hline(y=lo - 0.5, line=dict(color=DIM, width=1, dash="dot"), row=2, col=1)
    layout = dict(PLOTLY_LAYOUT)
    layout.update(height=440, showlegend=False, bargap=0,
                  title=dict(text=f"{r.label.upper()} OCH ANTAL VARNINGAR", font=dict(size=12, color=CYAN)))
    fig.update_layout(**layout)
    fig.update_yaxes(type="log", row=1, col=1)
    return fig


def render_market_risk_page() -> None:
    page_header("🌩️ Marknadsrisk", "Hur ofta har marknaden fallit minst 10 % inom tre månader när de här "
                                   "varningarna lyst? En riskspärr — ingen förutsägelse.")
    c_r, _ = st.columns([1, 4])
    if c_r.button("🔄 Uppdatera", key="mr_refresh"):
        st.session_state.pop(_CACHE, None)
    results = {m: _load(m) for m in mr.MARKETS}
    cols = st.columns(len(results))
    for col, r in zip(cols, results.values()):
        col.markdown(_summary(r), unsafe_allow_html=True)

    market = st.radio("Marknad", list(mr.MARKETS), horizontal=True, key="mr_market",
                      format_func=lambda m: mr.MARKETS[m]["label"])
    r = results[market]
    if r.error:
        note(f"⚠ {r.error}")
        return
    _signals(r)
    try:
        st.plotly_chart(history_chart(r), use_container_width=True, config={"displayModeBar": False},
                        key=f"mr_chart_{market}")
    except Exception as exc:
        note(f"Grafen kunde inte ritas: {exc}")
    _calibration(r)
    note("Gränserna är satta i förväg och inte optimerade mot historiken. Varningarna använder bara data fram "
         "till varje dag; målet tittar framåt per definition. Nedgångar är få — läs alltid antalet fall. "
         "Använd nivån som riskspärr (färre nya affärer, mindre positioner, tätare stopp), inte som "
         "blankningssignal. OMXS30 får de globala varningarna men ingen bredd.")


def _signals(r: mr.MarketRisk) -> None:
    _section("⚠️ VARNINGAR NU", f"{r.points} av {r.possible} lyser · {r.date}")
    rows = []
    for s in r.signals:
        if not s["available"]:
            mark, col, state = "?", DIM, "DATA UNAVAILABLE"
        elif s["active"]:
            mark, col, state = "●", RED, "LYSER"
        else:
            mark, col, state = "○", GREEN, "av"
        rows.append(f"<div style='font-size:0.82rem;color:{TEXT};line-height:1.6;'><span style='color:{col};"
                    f"font-weight:700;'>{mark}</span> <b>{s['label']}</b> <span style='color:{col};font-size:0.7rem;'>"
                    f"{state}</span><div style='color:{DIM};font-size:0.72rem;margin-left:16px;'>{s['why']}</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)
    lv = r.level
    st.markdown(f"<div style='margin-top:6px;color:{LEVEL_COLOR.get(lv, DIM)};font-weight:700;letter-spacing:0.1em;'>"
                f"NIVÅ: {lv}</div><div style='color:{DIM};font-size:0.75rem;'>LÅG 0–1 · FÖRHÖJD 2–3 · HÖG 4+ "
                f"varningar</div>", unsafe_allow_html=True)


def _calibration(r: mr.MarketRisk) -> None:
    cal = r.calibration
    _section("📊 HISTORISK TRÄFFBILD", f"−{int(mr.DRAWDOWN * 100)} % inom {mr.HORIZON} handelsdagar")
    if cal is None:
        note("Kalibreringen kräver att alla varningar har data — serierna saknas eller är för korta.")
        return
    st.markdown(f"<div style='color:{DIM};font-size:0.78rem;'>{cal.start} – {cal.end} · {cal.days} dagar med känt "
                f"utfall · basfrekvens <b style='color:{TEXT};'>{cal.base_rate:g} %</b></div>", unsafe_allow_html=True)
    rows = []
    for name, v in cal.by_level.items():
        hr = v["hit_rate"]
        hit = "—" if hr is None else f"{hr:g} %"
        rel = "—" if hr is None or not cal.base_rate else f"{hr / cal.base_rate:.1f}×"
        bold = "font-weight:700;" if name == r.level else ""
        rows.append(f"<tr style='{bold}'><td style='text-align:left;color:{LEVEL_COLOR[name]};'>{name}"
                    f"{' (nu)' if name == r.level else ''}</td><td>{v['days']}</td><td>{hit}</td><td>{rel}</td></tr>")
    rows = "".join(rows)
    st.markdown(f"<table style='font-size:0.8rem;color:{TEXT};text-align:right;'><tr style='color:{DIM};'>"
                f"<th style='text-align:left;'>Nivå</th><th>Dagar</th><th>Följdes av −10 %</th><th>Mot normalt</th>"
                f"</tr>{rows}</table>", unsafe_allow_html=True)
    a = cal.alarms
    warned = sum(e["warned"] for e in cal.episodes)
    st.markdown(f"<div style='color:{TEXT};font-size:0.82rem;margin-top:8px;'>Nedgångar ≥ {int(mr.DRAWDOWN * 100)} %: "
                f"<b>{len(cal.episodes)}</b> · varnade i förväg (FÖRHÖJD eller HÖG inom tre månader före): "
                f"<b>{warned}</b> · missade: <b>{len(cal.episodes) - warned}</b><br>Larm (nivå HÖG startar): "
                f"<b>{a['total']}</b> · följdes av −10 %: <b>{a['hits']}</b> · falsklarm: <b>{a['false']}</b></div>",
                unsafe_allow_html=True)
    if cal.episodes:
        with st.expander("Varje nedgång"):
            body = "".join(
                f"<tr><td style='text-align:left;'>{e['peak']}</td><td>{e['cross']}</td><td>{e['trough']}</td>"
                f"<td>{e['depth']:g} %</td><td style='color:{GREEN if e['warned'] else RED};'>"
                f"{'HÖG' if e['high'] else 'FÖRHÖJD' if e['warned'] else 'missad'}</td></tr>" for e in cal.episodes)
            st.markdown(f"<table style='font-size:0.78rem;color:{TEXT};text-align:right;'><tr style='color:{DIM};'>"
                        f"<th style='text-align:left;'>Topp</th><th>−10 % nåddes</th><th>Botten</th><th>Djup</th>"
                        f"<th>Varning före</th></tr>{body}</table>", unsafe_allow_html=True)
