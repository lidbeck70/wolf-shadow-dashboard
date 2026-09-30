"""
gold_silver/miners.py — ⛏️ silverbolag genom Snabbkollens motor (PR B).

Kedjan per kvotscenario, utan egen räknelogik:
  guldpris / kvot = silverpris
  → bolagets egna linjer mot silverpriset (intäkt, EBITDA, FCF — quick_leverage)
  → EV med egen median-EV/EBITDA → − nettoskuld → mot börsvärdet → kurs
    (asymmetry.quick_scenarios.at_prices)

Bolagets data hämtas med asymmetry.quick_data.fetch, med råvaran låst till
silver. NAV räknas inte automatiskt — värderingen går via EV/EBITDA.
"""

from __future__ import annotations

import time

import streamlit as st

from gold_silver import config as gc
from gold_silver import engine as ge
from ui.components import big_card, note
from ui.tokens import AMBER, CYAN, DIM, GREEN, GREY, RED, TEXT

_CACHE = "gs_miner_cache"
_TTL_S = 3600


def fetch_miner(ticker: str, fetcher=None) -> dict:
    """Snabbkollens datadict med råvaran låst till silver (SI=F)."""
    if fetcher is None:
        from asymmetry import quick_data
        fetcher = quick_data.fetch
    return fetcher(ticker, theme_getter=lambda s: "silver")


def miner_scenarios(d: dict, gold, silver) -> tuple:
    """(LeverageEstimate, EngineResult, [(namn, kvot, PricePoint)])."""
    from asymmetry import quick_leverage as ql
    from asymmetry import quick_scenarios as qs
    lev = ql.from_data(d)
    pts = ge.miner_price_points(gold, silver)
    res, rows = qs.at_prices(d, [(n, p) for n, _r, p in pts], lev)
    by_name = {r.name: r for r in rows}
    return lev, res, [(n, r, by_name[n]) for n, r, _p in pts if n in by_name]


def _ratio_color(r: float) -> str:
    return GREEN if r >= 2 else CYAN if r >= 1 else AMBER if r >= 0.7 else RED


def render_miner_section(gold, silver, cur_ratio) -> None:
    from asymmetry import quick_config as qc
    from asymmetry import quick_leverage as ql
    st.markdown(f"<div style='color:{CYAN};font-family:Courier New;letter-spacing:2px;font-size:0.85rem;"
                f"margin:18px 0 6px;'>⛏️ SILVERBOLAG — HÄVSTÅNG PER KVOT <span style='color:{DIM};"
                f"letter-spacing:0;'>— Snabbkollens motor, råvaran låst till silver</span></div>",
                unsafe_allow_html=True)
    with st.form("gs_miner_form", clear_on_submit=False):
        c1, c2 = st.columns([3, 1])
        t = c1.text_input("Silverbolag (ticker)", key="gs_miner_ticker", placeholder="t.ex. PAAS, HL, AG, FRES.L")
        run_it = c2.form_submit_button("⛏️ Räkna")
    t = (t or "").strip().upper()
    if not t:
        note("Skriv ett silverbolag. Bolagets intäkt, EBITDA och FCF ställs mot silverpriset år för år; "
             "varje kvotscenario ger ett silverpris som går genom samma kedja som 5×-motorn.")
        return
    if ge._pos(gold) is None or ge._pos(silver) is None:
        note("Kräver guld- och silverpris > 0 överst på sidan.")
        return
    cache = st.session_state.setdefault(_CACHE, {})
    hit = cache.get(t)
    if run_it or not hit or time.time() - hit["t"] > _TTL_S:
        with st.spinner(f"Hämtar {t} …"):
            cache[t] = {"t": time.time(), "data": fetch_miner(t)}
    d = cache[t]["data"]
    lev, res, rows = miner_scenarios(d, gold, silver)
    name = d.get("name") or t
    if res.error:
        note(f"⚠ {name}: DATA_GAP — {res.error}. Motorn kräver att bolagets EBITDA följer silverpriset "
             f"(minst {qc.LEV_MIN_YEARS} år, R² ≥ {qc.LEV_MIN_R2:g}), egen EV/EBITDA-historik, nettoskuld "
             f"och börsvärde.")
        return
    c1, c2 = st.columns(2)
    lc = GREEN if lev.downside_label == "STARK" else AMBER if lev.downside_label == "MÅTTLIG" else RED
    c1.markdown(big_card("COMMODITY LEVERAGE (SILVER)", "—" if lev.score is None else f"{lev.score}/10",
                         "—" if lev.response_pct is None else
                         f"silver +20 % → {lev.basis} {lev.response_pct:+.0f} % · {lev.flag}", lc),
                unsafe_allow_html=True)
    be = ("EJ MÄTBAR" if lev.break_even_margin_pct is None else
          f"{lev.break_even_margin_pct:.0f} % · break-even {ql.fmt_price(lev.break_even_price)} USD/oz")
    c2.markdown(big_card("BREAK-EVEN (SILVER)", be.split(" · ")[0], lev.break_even_note or be, GREY),
                unsafe_allow_html=True)
    ccy = res.price_ccy
    head = (f"<tr style='color:{DIM};'><th style='text-align:left;'>Scenario</th><th>Kvot</th><th>Silver</th>"
            f"<th>Intäkt M</th><th>EBITDA M</th><th>FCF M</th><th>Eget kapital M</th><th>× börsvärde</th>"
            f"<th>Kurs {ccy}</th></tr>")

    def _v(x, fmt="{:,.0f}"):
        return "—" if x is None else fmt.format(x)
    body = "".join(
        f"<tr><td style='text-align:left;'>{n}{' ⚠' if p.outside_history else ''}</td><td>{r:g}</td>"
        f"<td>{p.price:,.2f} ({p.price_pct:+.0f} %)</td><td>{_v(p.revenue)}</td><td>{_v(p.ebitda)}</td>"
        f"<td>{_v(p.fcf)}</td><td>{_v(p.equity)}</td>"
        f"<td style='color:{_ratio_color(p.ratio)};font-weight:700;'>{p.ratio:.2f}×</td>"
        f"<td>{_v(p.share_price, '{:,.2f}')}</td></tr>" for n, r, p in rows)
    st.markdown(f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.76rem;color:{TEXT};"
                f"text-align:right;'>{head}{body}</table></div>", unsafe_allow_html=True)
    f = lev.fits.get("EBITDA")
    note(f"{name}: guld {gold:,.0f} / kvot = silver → bolagets egen EBITDA-linje mot silver (R² {f.r2:.2f}, "
         f"{f.n} år) → EV med egen median {res.multiples['median']:g}× EV/EBITDA → − nettoskuld "
         f"{res.net_debt:,.0f} M → mot börsvärdet {res.mcap:,.0f} M {res.report_ccy} → kurs. Antalet aktier hålls "
         f"fast. Kvot {gc.REFERENCE_RATIO:g} är ett REFERENCE SCENARIO, inte ett fair value.")
    if any(p.outside_history for _n, _r, p in rows):
        note("⚠ = silverpriset ligger utanför tio års årssnitt — bolagets linje extrapoleras och är osäkrare "
             "ju längre bort den dras.")
    note("Bara silverpriset ändras: guld-, zink- och blyintäkter ligger kvar som i historiken. NAV räknas "
         "inte automatiskt — värderingen går via EV/EBITDA.")
