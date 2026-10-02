"""
ovtlyr/ui/nine_card.py — kortet OVTLYR NINE · SLINGSHOT SETUP (Viking Regime).

Tre lager (MARKET 40 % · SECTOR 30 % · STOCK 30 %), rå poäng n/9, viktad
poäng och status. Varje faktor visas som ✓ / ✗ / ? (DATA UNAVAILABLE eller
STALE DATA — räknas aldrig som PASS). Allt är märkt WOLF APPROXIMATION.
"""

from __future__ import annotations

import streamlit as st

import ovtlyr_nine as on
from ui.components import note

_GREEN, _RED, _DIM, _AMBER, _CYAN, _TEXT, _BG2 = ("#00e676", "#ff1744", "#8a8578", "#ffb300", "#00E5FF",
                                                  "#e8e4d8", "#14161c")
_LAYER_TITLES = {"market": "MARKET", "sector": "SECTOR", "stock": "STOCK"}


def _mark(f: on.Factor) -> tuple:
    if f.status == on.PASS:
        return "✓", _GREEN
    if f.status == on.FAIL:
        return "✗", _RED
    return "?", _DIM


def status_color(status: str) -> str:
    return _GREEN if status == on.FULL else _RED if status == on.NOT_ALIGNED else _DIM


def card_html(nine: on.NineResult) -> str:
    blocks = []
    for name in ("market", "sector", "stock"):
        rows = "".join(
            f"<div style='font-size:0.8rem;color:{_TEXT};line-height:1.6;'>"
            f"<span style='color:{c};font-weight:700;'>{m}</span> {f.label}"
            + (f" <span style='color:{_DIM};font-size:0.68rem;'>{f.status}</span>"
               if f.status not in (on.PASS, on.FAIL) else "") + "</div>"
            for f in nine.layer(name) for m, c in [_mark(f)])
        n, size = nine.layer_passed(name), on.LAYER_SIZE[name]
        sub = f" · {nine.sector_etf}" if name == "sector" and nine.sector_etf else ""
        col = _GREEN if n == size else _AMBER if n else _RED
        blocks.append(
            f"<div style='flex:1 1 150px;background:{_BG2};border:1px solid #2a2d36;border-radius:6px;"
            f"padding:8px 10px;'><div style='color:{_CYAN};font-size:0.72rem;letter-spacing:0.1em;'>"
            f"{_LAYER_TITLES[name]} {on.WEIGHTS[name]} %{sub}</div>{rows}"
            f"<div style='color:{col};font-weight:700;margin-top:4px;'>{n} / {size}</div></div>")
    calc = " · ".join(f"{_LAYER_TITLES[n].title()} {nine.layer_passed(n)}/{on.LAYER_SIZE[n]} × {on.WEIGHTS[n]} "
                      f"= {nine.layer_points(n):g}" for n in ("market", "sector", "stock"))
    st_col = status_color(nine.status)
    missing = ""
    if nine.status != on.FULL:
        names = [f"{_LAYER_TITLES[f.layer].title()} {f.label.lower()}" for f in nine.failed]
        gaps = [f"{_LAYER_TITLES[f.layer].title()} {f.label.lower()}" for f in nine.unavailable]
        if names:
            missing += f"<div style='color:{_RED};font-size:0.75rem;'>Failed: {', '.join(names)}</div>"
        if gaps:
            missing += (f"<div style='color:{_DIM};font-size:0.75rem;'>Data saknas (räknas inte som PASS): "
                        f"{', '.join(gaps)}</div>")
    return (
        f"<div style='border:1px solid {st_col};border-radius:8px;padding:10px 12px;margin:8px 0;'>"
        f"<div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;'>"
        f"<div style='color:{_CYAN};font-weight:700;letter-spacing:0.12em;'>OVTLYR NINE "
        f"<span style='color:{_DIM};font-weight:400;'>· SLINGSHOT SETUP</span></div>"
        f"<div style='color:{_AMBER};font-size:0.68rem;border:1px solid {_AMBER};border-radius:4px;"
        f"padding:1px 6px;'>{on.APPROXIMATION}</div></div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin:8px 0;'>{''.join(blocks)}</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:16px;align-items:baseline;'>"
        f"<div style='color:{_TEXT};font-size:1.1rem;font-weight:700;'>TOTAL {nine.passed} / {on.NINE_TOTAL}</div>"
        f"<div style='color:{_TEXT};'>Viktat <b>{nine.weighted:g} / 100</b></div>"
        f"<div style='color:{st_col};font-weight:700;letter-spacing:0.08em;'>STATUS: {nine.status}</div></div>"
        f"<div style='color:{_DIM};font-size:0.72rem;margin-top:4px;'>{calc}</div>{missing}"
        f"<div style='color:{_DIM};font-size:0.7rem;margin-top:6px;'>Nine = setup, inte entry. Riktiga priser "
        f"(SPY, sektor-ETF:er, aktien) men panelens egna definitioner — inte OVTLYR:s data.</div></div>")


def render_nine_card(nine: on.NineResult) -> None:
    st.markdown(card_html(nine), unsafe_allow_html=True)
    with st.expander("OVTLYR Nine — varje faktor (värde, tid, källa, status)"):
        rows = "".join(
            f"<tr><td style='text-align:left;'>{_LAYER_TITLES[f.layer]}</td><td style='text-align:left;'>{f.label}"
            f"</td><td style='color:{_mark(f)[1]};'>{f.status}</td><td>{'—' if f.value is None else f'{f.value:g}'}"
            f"</td><td>{f.timestamp or '—'}</td><td style='text-align:left;'>{f.source}</td>"
            f"<td style='text-align:left;color:{_DIM};'>{f.detail}</td></tr>" for f in nine.factors)
        st.markdown(
            f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.72rem;color:{_TEXT};"
            f"text-align:right;'><tr style='color:{_DIM};'><th style='text-align:left;'>Lager</th>"
            f"<th style='text-align:left;'>Faktor</th><th>Status</th><th>Värde</th><th>Tid</th>"
            f"<th style='text-align:left;'>Källa</th><th style='text-align:left;'>Uträkning</th></tr>{rows}"
            f"</table></div>", unsafe_allow_html=True)
        note("Definitionerna står i ovtlyr_nine.py. DATA UNAVAILABLE och STALE DATA (senaste stängning "
             f"äldre än {on.STALE_BDAYS} handelsdagar) räknas aldrig som PASS.")
