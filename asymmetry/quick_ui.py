"""
asymmetry/quick_ui.py — Snabbkollen i Wolf Asymmetry: skriv en ticker, få
Survival, Margin of Safety och Confidence (auto) som mätare med kort under,
ett poängkort i Viking Regime-stil och fyra grafer. Helt automatiskt —
inget matas in, inget sparas förrän du trycker "Lägg till i arket".
"""

from __future__ import annotations

import time
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from ui.charts import PLOTLY_LAYOUT, build_gauge
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, GREY, RED, TEXT

from asymmetry import quick, quick_data

CACHE_KEY = "asym_quick_cache"
CACHE_TTL_S = 3600
_STATUS_COLOR = {"GREEN": GREEN, "AMBER": AMBER, "RED": RED, "DATA_GAP": GREY}
_STATUS_ICON = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴", "DATA_GAP": "⚪"}
_VERDICT_COLOR = {"GRÖN": GREEN, "GUL": AMBER, "RÖD": RED, "DATA_GAP": GREY}
_VERDICT_LABEL = {"GRÖN": "STARK ASYMMETRI", "GUL": "BLANDAD BILD", "RÖD": "SVAG ASYMMETRI",
                  "DATA_GAP": "FÖR LITE DATA"}


def _cached(ticker: str, force: bool = False) -> dict:
    cache = st.session_state.setdefault(CACHE_KEY, {})
    hit = cache.get(ticker)
    if hit and not force and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    data = quick_data.fetch(ticker)
    cache[ticker] = (time.time(), data)
    return data


def render_quick(add_to_sheet=None) -> None:
    """add_to_sheet(ticker, name) -> bool: kopplas av Wolf Asymmetry-sidan."""
    with st.form("asym_quick_form", border=False):
        c1, c2, c3 = st.columns([3, 1, 1])
        ticker = c1.text_input("Ticker", key="asym_quick_ticker", placeholder="T.ex. BOL.ST, LUG.ST, NEM, AEM.TO, IPT.V",
                               label_visibility="collapsed")
        go_btn = c2.form_submit_button("🔍 Analysera", use_container_width=True)
        refresh = c3.form_submit_button("🔄 Uppdatera", use_container_width=True)
    t = str(ticker or "").strip().upper()
    if not t:
        st.caption("Skriv en ticker. Börsdata först (nordiskt, sedan globalt), Yahoo som reserv. "
                   "Survival · Margin of Safety · Confidence räknas helt automatiskt.")
        return
    with st.spinner(f"Hämtar {t} …"):
        data = _cached(t, force=bool(refresh))
    if not data.get("prices") and data.get("source") in ("—", "", None):
        st.warning(f"Hittade inget för {t} i Börsdata eller Yahoo. Kontrollera tickern (t.ex. suffix .ST, .OL, .TO).")
        return
    res = quick.score(data)
    _score_card(res, data)
    _gauges(res)
    _volatility(res)
    _charts(data)
    extra = []
    if data.get("filled_yahoo"):
        extra.append("Ur Yahoo (Börsdata tomt): " + ", ".join(data["filled_yahoo"]))
    if extra:
        st.caption(" · ".join(extra))
    if add_to_sheet is not None:
        if st.button("➕ Lägg till i arket (Analys/Ark)", key=f"asym_quick_add_{t}"):
            ok = add_to_sheet(data.get("yf_ticker") or t, data.get("name") or t)
            (st.success if ok else st.info)(
                f"{t} ligger i arket — öppna Ark för råvara och stage." if ok else f"{t} finns redan i arket.")


# ── Poängkortet (Viking Regime-stil) ────────────────────────────────────────
def _score_card(res: quick.QuickResult, data: dict) -> None:
    col = _VERDICT_COLOR.get(res.verdict, GREY)
    total = res.total
    total_txt = f"{total:.0f}" if total is not None else "—"
    measured = sum(len(g.measured) for g in res.groups)
    n_cards = sum(len(g.scored) for g in res.groups)
    st.markdown(
        f"""<div style="background:linear-gradient(135deg, rgba(0,229,255,0.06) 0%, rgba(139,115,64,0.03) 100%);
        border:1px solid {col};border-radius:12px;padding:28px 16px;text-align:center;position:relative;
        overflow:hidden;margin-bottom:10px;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{CYAN},#00A8BF);"></div>
        <div style="font-size:11px;letter-spacing:4px;color:rgba(0,229,255,0.5);margin-bottom:8px;">WOLF ASYMMETRY SCORE</div>
        <div style="font-size:76px;font-weight:900;line-height:1;color:{col};font-family:'Courier New',monospace;">{total_txt}</div>
        <div style="font-size:11px;letter-spacing:3px;color:rgba(0,229,255,0.4);margin-top:4px;">/ 300 MAX</div>
        <div style="margin-top:14px;"><span style="background:{col}1a;border:1px solid {col};border-radius:20px;
        color:{col};font-size:11px;font-weight:700;letter-spacing:3px;padding:5px 16px;">
        {res.verdict} · {_VERDICT_LABEL.get(res.verdict, "")}</span></div>
        <div style="margin-top:12px;font-size:0.8rem;color:{TEXT};">{" · ".join(res.reasons)}</div>
        <div style="margin-top:14px;font-size:10px;color:rgba(0,229,255,0.35);letter-spacing:2px;">
        {res.ticker} · {res.name} · {res.source} · {measured}/{n_cards} kort mätta · {data.get("fetched", "")}</div>
        </div>""",
        unsafe_allow_html=True)


# ── Mätare + kort ────────────────────────────────────────────────────────────
def _gauges(res: quick.QuickResult) -> None:
    cols = st.columns(3)
    for col, g in zip(cols, res.groups):
        with col:
            if g.score is None:
                st.markdown(f"<div style='text-align:center;color:{DIM};font-family:Courier New;"
                            f"letter-spacing:2px;margin:40px 0 8px;'>{g.label.upper()}<br>"
                            f"<span style='font-size:1.6rem;color:{GREY};'>DATA_GAP</span></div>",
                            unsafe_allow_html=True)
            else:
                st.plotly_chart(build_gauge(round(g.score), 100, g.label.upper(), color_cyan=True),
                                use_container_width=True, config={"displayModeBar": False},
                                key=f"asym_quick_gauge_{g.key}")
            st.caption(f"{g.coverage}")
            for p in g.pillars:
                st.markdown(_card(p), unsafe_allow_html=True)


def _card(p) -> str:
    c = _STATUS_COLOR.get(p.status, GREY)
    return (f"<div style='border-left:3px solid {c};background:#14141e;border-radius:6px;"
            f"padding:7px 10px;margin-bottom:6px;'>"
            f"<div style='display:flex;justify-content:space-between;gap:8px;'>"
            f"<span style='color:{TEXT};font-size:0.82rem;'>{_STATUS_ICON.get(p.status, '')} {p.label}</span>"
            f"<span style='color:{c};font-size:0.82rem;font-weight:700;'>{p.value}</span></div>"
            f"<div style='color:{DIM};font-size:0.7rem;margin-top:2px;'>{p.why}</div></div>")


def _volatility(res: quick.QuickResult) -> None:
    """Cykelvolatilitet — egen dimension under mätarna, räknas inte i 300."""
    g = res.volatility
    if g is None or not g.pillars:
        return
    st.markdown(f"<div style='color:{CYAN};font-family:Courier New;letter-spacing:2px;font-size:0.8rem;"
                f"margin:6px 0 4px;'>📉 CYKELVOLATILITET <span style='color:{DIM};letter-spacing:0;'>"
                f"— info, räknas inte i 300. Svängande resultat och FCF är cykeln, inte sämre data.</span></div>",
                unsafe_allow_html=True)
    cols = st.columns(len(g.pillars))
    for col, p in zip(cols, g.pillars):
        col.markdown(_card(p), unsafe_allow_html=True)


# ── Grafer ───────────────────────────────────────────────────────────────────
def _layout(title: str, height: int = 260) -> dict:
    base = {k: v for k, v in PLOTLY_LAYOUT.items()}
    base.update(title=dict(text=title, font=dict(size=12, color=CYAN)), height=height,
                legend=dict(orientation="h", y=-0.25, font=dict(color=DIM, size=10)))
    return base


def price_chart(data: dict) -> Optional[go.Figure]:
    px = data.get("prices") or []
    if len(px) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[d for d, _ in px], y=[v for _, v in px], name="Kurs",
                             line=dict(color=CYAN, width=1.8)))
    sma = [(d, v) for d, v in (data.get("sma200") or []) if v is not None]
    if sma:
        fig.add_trace(go.Scatter(x=[d for d, _ in sma], y=[v for _, v in sma], name="SMA200",
                                 line=dict(color=GOLD, width=1.4, dash="dot")))
    fig.update_layout(**_layout("KURS 2 ÅR MOT 200-DAGARS SNITT"))
    return fig


def ev_ebitda_chart(data: dict) -> Optional[go.Figure]:
    s = [(y, v) for y, v in (data.get("ev_ebitda_series") or []) if v is not None and 0 < v < 100]
    if len(s) < 2:
        return None
    from statistics import median
    med = median(v for _, v in s)
    fig = go.Figure(go.Scatter(x=[y for y, _ in s], y=[v for _, v in s], mode="lines+markers", name="EV/EBITDA",
                               line=dict(color=CYAN, width=2), marker=dict(size=7)))
    fig.add_hline(y=med, line=dict(color=GOLD, width=1, dash="dot"), annotation_text=f"median {med:.1f}×",
                  annotation_font=dict(color=GOLD, size=10))
    fig.update_layout(**_layout("EV/EBITDA MOT EGEN MEDIAN"))
    return fig


def _bars(series, title: str, name: str, signed: bool = False) -> Optional[go.Figure]:
    s = [(y, v) for y, v in (series or []) if v is not None]
    if len(s) < 2:
        return None
    colors = [(GREEN if v >= 0 else RED) if signed else CYAN for _, v in s]
    fig = go.Figure(go.Bar(x=[y for y, _ in s], y=[v for _, v in s], name=name, marker=dict(color=colors)))
    fig.update_layout(**_layout(title), showlegend=False)
    return fig


def _charts(data: dict) -> None:
    figs = [(price_chart(data), "price"), (ev_ebitda_chart(data), "ev"),
            (_bars(data.get("shares_series"), "ANTAL AKTIER (M) — UTSPÄDNING", "Aktier"), "shares"),
            (_bars(data.get("fcf_series"), "FRITT KASSAFLÖDE PER ÅR (M)", "FCF", signed=True), "fcf")]
    for row in (figs[:2], figs[2:]):
        cols = st.columns(2)
        for col, (fig, key) in zip(cols, row):
            with col:
                if fig is None:
                    st.caption("Grafen kan inte ritas: för lite historik.")
                else:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                                    key=f"asym_quick_chart_{key}")
