"""
berserk/regime_ui.py — REGIME → Råvaror → 🪓 BERSERK Regime.

Var i råvarucykeln varje tema står idag och vilka BERSERK-setups det tillåter:
marknaderna (regionernas index mot SMA200), en karta per komplex med temats
fas (BAISSE · VÄNDER / BAISSE / STARK / UPPTREND / SVAG), drivarens 63-dagars-
avkastning, läge i femårsintervallet, avstånd från 52-veckorshögsta och hur
producentbolagen ligger mot sin råvara.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from berserk import live
from berserk import signals as sg
from berserk import themes as th
from berserk import universe as uv
from ui.components import note, page_header
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, GREY, RED, TEXT

_STATE = "bz_regime"
PHASE_COLOR = {"BAISSE · VÄNDER": GOLD, "BAISSE": RED, "STARK": GREEN, "UPPTREND": CYAN, "SVAG": AMBER,
               "INGEN DRIVARE": GREY}
SHORT = {sg.S1: "S1", sg.S2: "S2", sg.S3: "S3"}


def load(getter=None, nordic_provider=None, today=None) -> dict:
    """Allt regimen behöver: drivare, index och producenternas kurser (en batch)."""
    drivers = live.load_drivers(list(th.THEMES), getter, today)
    markets = live.load_markets(list(uv.REGIONS), getter, nordic_provider)
    try:
        from market_prices import closes
        prod = closes(list(uv.PRODUCERS), "1y") if getter is None else {
            t: live._close(getter(t, "1y")) for t in uv.PRODUCERS}
    except Exception:
        prod = {}
    return {"themes": live.theme_states(drivers, prod), "markets": live.market_states(markets),
            "when": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}


def _fmt(v, f="{:+.1f}") -> str:
    return "—" if v is None else f.format(v)


def theme_card(r: dict) -> str:
    col = PHASE_COLOR.get(r["phase"], GREY)
    chips = " ".join(f"<span style='border:1px solid {CYAN};border-radius:3px;padding:0 4px;color:{CYAN};"
                     f"font-size:0.66rem;'>{SHORT.get(s, s)}</span>" for s in r["setups"]) or \
        f"<span style='color:{DIM};font-size:0.66rem;'>inga setups</span>"
    rng = "—" if r["range5y"] is None else f"{r['range5y']:.0f} %"
    return (f"<div style='border:1px solid {col};border-left:4px solid {col};border-radius:4px;padding:6px 8px;"
            f"min-width:0;'><div style='display:flex;justify-content:space-between;gap:6px;'>"
            f"<b style='color:{TEXT};font-size:0.82rem;'>{r['label']}</b>"
            f"<span style='color:{col};font-size:0.68rem;font-weight:700;'>{r['phase']}</span></div>"
            f"<div style='color:{DIM};font-size:0.7rem;line-height:1.5;'>"
            f"{r['driver'] or 'ingen prisserie'} · 63 d {_fmt(r['ret63'])} % · 5 år {rng} · "
            f"topp {_fmt(r['dd252'])} %<br>producenter mot råvaran {_fmt(r['divergence'])} pe · "
            f"{r['lagging']} av {r['producers']} efter ≥ 10 pe</div><div style='margin-top:3px;'>{chips}</div></div>")


def market_chip(m: dict) -> str:
    col = GREEN if m["ok"] else RED if m["ok"] is False else GREY
    val = "—" if m["vs_sma200"] is None else f"{m['vs_sma200']:+.1f} %"
    return (f"<div style='border:1px solid {col};border-radius:4px;padding:3px 8px;font-size:0.76rem;"
            f"color:{TEXT};'>{m['region']} <span style='color:{DIM};'>{m['index']}</span> "
            f"<b style='color:{col};'>{val}</b></div>")


def render_berserk_regime_page() -> None:
    page_header("🪓 BERSERK Regime", "Var i råvarucykeln varje tema står och vilka setups det tillåter idag.")
    c1, c2 = st.columns([1, 3])
    if c1.button("📡 Uppdatera", key="bz_regime_refresh") or _STATE not in st.session_state:
        with st.spinner("Hämtar råvaruterminer, index och producenter …"):
            st.session_state[_STATE] = load()
    data = st.session_state[_STATE]
    c2.markdown(f"<div style='color:{DIM};font-size:0.78rem;padding-top:8px;'>Uppdaterad {data['when']}</div>",
                unsafe_allow_html=True)
    render(data)


def render(data: dict) -> None:
    mk = data.get("markets") or []
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:6px;'>MARKNADERNA "
                f"(GRINDEN: INDEX ÖVER SMA200)</div>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 10px;'>"
                + "".join(market_chip(m) for m in mk) + "</div>", unsafe_allow_html=True)
    rows = data.get("themes") or []
    turning = [r for r in rows if r["phase"] == "BAISSE · VÄNDER"]
    if turning:
        note("Cykelvändning (S2) möjlig nu: " + ", ".join(r["label"] for r in turning)
             + " — råvaran har varit i baisse och vänder. Det är här BERSERK letar hatade bolag.")
    for cx, title in th.COMPLEXES.items():
        part = [r for r in rows if r["complex"] == cx]
        if not part:
            continue
        st.markdown(f"<div style='color:{GOLD};font-size:0.72rem;letter-spacing:0.1em;margin-top:10px;'>{title}</div>",
                    unsafe_allow_html=True)
        st.markdown("<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:8px;"
                    "margin-top:4px;'>" + "".join(theme_card(r) for r in part) + "</div>", unsafe_allow_html=True)
    note("Faser: BAISSE · VÄNDER = råvaran ≥ 30 % under toppen (eller i nedre 20 % av femårsintervallet) senaste "
         "halvåret och nu över EMA50 med stigande EMA20 (S2) · BAISSE = utan vändning · STARK = över SMA50 och "
         "SMA200 med positiv 63 d (S1 när producenter halkat efter) · UPPTREND = över SMA200 (S3) · SVAG = under "
         "SMA200. Producenter mot råvaran = medianen av aktiernas 63-dagarsavkastning minus råvarans, i "
         "procentenheter. Setups är vad temat tillåter — aktierna avgör i 🪓 BERSERK-skannern.")
