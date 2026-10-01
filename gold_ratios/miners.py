"""
gold_ratios/miners.py — ⛏️ bolag per kvotscenario genom Snabbkollens motor (PR B).

Samma kedja som silverbolagen i Guld/Silver, för valt pars råvara:
  guldpris / kvot = råvarupris  (kvoterna = parets egna percentiler)
  → bolagets egna linjer mot råvaran (intäkt, EBITDA, FCF — quick_leverage)
  → EV med egen median-EV/EBITDA → − nettoskuld → mot börsvärdet → kurs
    (asymmetry.quick_scenarios.at_prices)

Bara par vars ticker är samma serie som Snabbkollen använder för temat
(asymmetry.quick_config.LEV_PRICE_TICKERS) — annars skulle bolagets linje
och scenariopriset mäta olika saker (t.ex. Brent mot WTI).
"""

from __future__ import annotations

import time

import streamlit as st

from gold_ratios import config as rc
from gold_ratios import engine as gre
from gold_silver import engine as ge
from ui.components import big_card, note
from ui.tokens import AMBER, CYAN, DIM, GREEN, GREY, RED, TEXT

_CACHE = "gr_miner_cache"
_TTL_S = 3600


def minable(pair: dict) -> bool:
    from asymmetry import quick_config as qc
    theme = pair.get("theme")
    return pair["kind"] == rc.COMMODITY and bool(theme) and qc.LEV_PRICE_TICKERS.get(theme) == pair["ticker"]


def fetch_miner(ticker: str, theme: str, fetcher=None) -> dict:
    """Snabbkollens datadict med råvaran låst till parets tema."""
    if fetcher is None:
        from asymmetry import quick_data
        fetcher = quick_data.fetch
    return fetcher(ticker, theme_getter=lambda s: theme)


def price_points(gold, other, full) -> list:
    """[(namn, kvot, pris i visad enhet)] — BASE = dagens pris, övriga guld / percentilkvot."""
    if full is None:
        return []
    scen = [(name, None if attr is None else getattr(full, attr)) for name, attr in rc.MINER_SCENARIOS]
    return ge.miner_price_points(gold, other, scen)


def miner_scenarios(pair: dict, d: dict, gold, other, full) -> tuple:
    """(LeverageEstimate, EngineResult, [(namn, kvot, pris, PricePoint)]).
    Priset i visad enhet (USD); motorn får Yahoos enhet (US-cent för vete och kaffe)."""
    from asymmetry import quick_leverage as ql
    from asymmetry import quick_scenarios as qs
    lev = ql.from_data(d)
    pts = price_points(gold, other, full)
    scale = pair.get("scale", 1.0) or 1.0
    res, rows = qs.at_prices(d, [(n, p / scale) for n, _r, p in pts], lev)
    by_name = {r.name: r for r in rows}
    return lev, res, [(n, r, p, by_name[n]) for n, r, p in pts if n in by_name]


def _ratio_color(r: float) -> str:
    return GREEN if r >= 2 else CYAN if r >= 1 else AMBER if r >= 0.7 else RED


def render_miner_section(pair: dict, gold, other, full) -> None:
    from asymmetry import quick_config as qc
    from asymmetry import quick_leverage as ql
    label = pair["label"].lower()
    st.markdown(f"<div style='color:{CYAN};font-family:Courier New;letter-spacing:2px;font-size:0.85rem;"
                f"margin:18px 0 6px;'>⛏️ BOLAG — HÄVSTÅNG PER KVOT <span style='color:{DIM};"
                f"letter-spacing:0;'>— Snabbkollens motor, råvaran låst till {label}</span></div>",
                unsafe_allow_html=True)
    if not minable(pair):
        note(f"Inte för {pair['label']}: Snabbkollen mäter bolagen mot en annan prisserie för det här temat "
             f"(eller inget tema alls) — välj t.ex. Olja (WTI) i stället för Brent.")
        return
    k = pair["key"]
    with st.form("gr_miner_form", clear_on_submit=False):
        c1, c2 = st.columns([3, 1])
        t = c1.text_input(f"Bolag ({label})", key=f"gr_miner_ticker_{k}",
                          placeholder=f"t.ex. {rc.MINER_EXAMPLES.get(pair['theme'], '')}")
        run_it = c2.form_submit_button("⛏️ Räkna")
    t = (t or "").strip().upper()
    if not t:
        note(f"Skriv ett bolag. Bolagets intäkt, EBITDA och FCF ställs mot priset på {label} år för år; varje "
             f"kvotscenario (parets egna percentiler) ger ett pris som går genom samma kedja som 5×-motorn.")
        return
    if ge._pos(gold) is None or ge._pos(other) is None or full is None:
        note(f"Kräver guldpris, pris på {label} och kvothistorik.")
        return
    cache = st.session_state.setdefault(_CACHE, {})
    ck = f"{k}:{t}"
    hit = cache.get(ck)
    if run_it or not hit or time.time() - hit["t"] > _TTL_S:
        with st.spinner(f"Hämtar {t} …"):
            cache[ck] = {"t": time.time(), "data": fetch_miner(t, pair["theme"])}
    d = cache[ck]["data"]
    lev, res, rows = miner_scenarios(pair, d, gold, other, full)
    name = d.get("name") or t
    if res.error:
        note(f"⚠ {name}: DATA_GAP — {res.error}. Motorn kräver att bolagets EBITDA följer priset på {label} "
             f"(minst {qc.LEV_MIN_YEARS} år, R² ≥ {qc.LEV_MIN_R2:g}), egen EV/EBITDA-historik, nettoskuld "
             f"och börsvärde.")
        return
    up = pair["label"].upper()
    c1, c2 = st.columns(2)
    lc = GREEN if lev.downside_label == "STARK" else AMBER if lev.downside_label == "MÅTTLIG" else RED
    c1.markdown(big_card(f"COMMODITY LEVERAGE ({up})", "—" if lev.score is None else f"{lev.score}/10",
                         "—" if lev.response_pct is None else
                         f"{label} +20 % → {lev.basis} {lev.response_pct:+.0f} % · {lev.flag}", lc),
                unsafe_allow_html=True)
    scale = pair.get("scale", 1.0) or 1.0
    be = ("EJ MÄTBAR" if lev.break_even_margin_pct is None else
          f"{lev.break_even_margin_pct:.0f} % · break-even "
          f"{ql.fmt_price(lev.break_even_price * scale if lev.break_even_price else None)} {pair['unit']}")
    c2.markdown(big_card(f"BREAK-EVEN ({up})", be.split(" · ")[0], lev.break_even_note or be, GREY),
                unsafe_allow_html=True)
    ccy = res.price_ccy
    head = (f"<tr style='color:{DIM};'><th style='text-align:left;'>Scenario</th><th>Kvot</th>"
            f"<th>{pair['label']} ({pair['unit']})</th><th>Intäkt M</th><th>EBITDA M</th><th>FCF M</th>"
            f"<th>Eget kapital M</th><th>× börsvärde</th><th>Kurs {ccy}</th></tr>")

    def _v(x, fmt="{:,.0f}"):
        return "—" if x is None else fmt.format(x)
    body = "".join(
        f"<tr><td style='text-align:left;'>{n}{' ⚠' if p.outside_history else ''}</td><td>{_fmt(r)}</td>"
        f"<td>{_fmt(price)} ({p.price_pct:+.0f} %)</td><td>{_v(p.revenue)}</td><td>{_v(p.ebitda)}</td>"
        f"<td>{_v(p.fcf)}</td><td>{_v(p.equity)}</td>"
        f"<td style='color:{_ratio_color(p.ratio)};font-weight:700;'>{p.ratio:.2f}×</td>"
        f"<td>{_v(p.share_price, '{:,.2f}')}</td></tr>" for n, r, price, p in rows)
    st.markdown(f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.76rem;color:{TEXT};"
                f"text-align:right;'>{head}{body}</table></div>", unsafe_allow_html=True)
    f = lev.fits.get("EBITDA")
    note(f"{name}: guld {gold:,.0f} / kvot = {label} → bolagets egen EBITDA-linje mot {label} (R² {f.r2:.2f}, "
         f"{f.n} år) → EV med egen median {res.multiples['median']:g}× EV/EBITDA → − nettoskuld "
         f"{res.net_debt:,.0f} M → mot börsvärdet {res.mcap:,.0f} M {res.report_ccy} → kurs. Antalet aktier hålls "
         f"fast. Kvoterna är parets egna percentiler — inte ett fair value.")
    if any(p.outside_history for _n, _r, _pr, p in rows):
        note("⚠ = priset ligger utanför tio års årssnitt — bolagets linje extrapoleras och är osäkrare ju längre "
             "bort den dras.")
    note(f"Bara priset på {label} ändras: andra intäkter ligger kvar som i historiken. NAV räknas inte "
         f"automatiskt — värderingen går via EV/EBITDA.")


def _fmt(v) -> str:
    return "—" if v is None else f"{v:,.{gre.decimals(v)}f}"
