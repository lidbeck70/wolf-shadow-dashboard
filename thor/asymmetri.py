"""
thor/asymmetri.py — Asymmetri-fliken i Thors analysblock: Wolf Asymmetrys Snabbkoll för aktien, för alla
strategier, plus tre tillägg som räknas ur SAMMA data (asymmetry.quick_data.fetch) och SAMMA kedjor som
Snabbkollens motorer (quick_value, quick_scenarios) — inga nya regler, inget sparas:

  RIKTPRIS            medianen av oberoende skattningar ur bolagets egen historik, varje formel synlig:
                        omvärdering   EBITDA nu × egen median-EV/EBITDA − nettoskuld
                        normal        omsättning × egen median-marginal × median-multipel − nettoskuld
                        kassaflöde    fritt kassaflöde × egen median-P/FCF
                        tillväxt      fem år egen tillväxt (kapad) × median-marginal × median-multipel − nettoskuld,
                                      diskonterad DISCOUNT_RATE per år och justerad för utspädningstakten
                        råvarubas     (råvarubolag) 5×-motorns BASE — dagens råvarupris, egen median-multipel
                      Medianen tål en skattning som skenar; spannet (lägsta–högsta) visar hur osäkert det är.
  KURSHÄVSTÅNG        aktiekursen när drivaren går ±10/20/30 %: råvarupriset (egen EBITDA-linje) för råvarubolag,
                      EBITDA för övriga — multipeln hålls fast, så det är rörelse- och skuldhävstången som syns.
  TILLVÄXTSCENARIER   fem år med egen tillväxt −5 … +15 procentenheter → omsättning → EBITDA (median-marginal)
                      → EV (median-multipel) → − nettoskuld → kurs, och vad det ger per år.
Belopp i miljoner i rapportvalutan; kursen i handelsvalutan via dagens börsvärde (antalet aktier hålls fast).
Rena funktioner; Snabbkollens egna block ritas av asymmetry.quick_ui.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from html import escape
from statistics import median
from typing import Callable, Optional

from thor import cards
from ui.tokens import AMBER, CYAN, DIM, GREEN, GREY, RED, TEXT

CACHE_KEY, CACHE_TTL_S = "asym_quick_cache", 3600       # samma cache som 🐺 Wolf Asymmetrys Snabbkoll
DISCOUNT_RATE = 0.10                                    # avkastningskrav per år i tillväxtskattningen
HORIZON_YEARS = 5
LEVERAGE_STEPS = (-30.0, -20.0, -10.0, 10.0, 20.0, 30.0)
GROWTH_DELTAS_PP = (-5.0, 0.0, 5.0, 10.0, 15.0)
MIN_ESTIMATES = 2


def _n(v) -> Optional[float]:
    try:
        return None if v is None or v != v else float(v)
    except (TypeError, ValueError):
        return None


def _clamp_growth(g: Optional[float]) -> float:
    from asymmetry import quick_config as qc
    lo, hi = qc.GROWTH_CAP
    return min(max(g if g is not None else 0.0, lo), hi)


def _pfcf_median(d: dict) -> Optional[float]:
    from asymmetry import quick_config as qc
    vals = [x for x in (_n(v) for v in (d.get("p_fcf_hist") or [])) if x is not None and 0 < x < 100]
    return median(vals) if len(vals) >= qc.MIN_HISTORY else None


def _dilution_factor(d: dict, years: int = HORIZON_YEARS) -> tuple:
    """(faktor per aktie efter `years` år, takt %/år) — tre års aktieökning antas fortsätta; minskning räknas inte."""
    g3 = _n(d.get("shares_growth_3y_pct"))
    if g3 is None or g3 <= 0:
        return 1.0, 0.0
    rate = (1 + g3 / 100) ** (1 / 3) - 1
    return 1 / (1 + rate) ** years, round(rate * 100, 2)


# ── Riktpris ────────────────────────────────────────────────────────────────
@dataclass
class Estimate:
    key: str
    label: str
    price: Optional[float]
    formula: str
    note: str = ""


@dataclass
class Riktpris:
    mode: str = ""
    price_now: Optional[float] = None
    target: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    upside_pct: Optional[float] = None
    method: str = ""
    ccy: str = ""
    estimates: list = field(default_factory=list)
    error: Optional[str] = None

    @property
    def used(self) -> list:
        return [e for e in self.estimates if e.price is not None and e.price > 0]


def riktpris(d: dict, mode: Optional[str] = None) -> Riktpris:
    """Riktpriset: medianen av de skattningar som går att räkna ur bolagets egen historik."""
    from asymmetry import quick_config as qc
    from asymmetry import quick_scenarios as qs
    from asymmetry import quick_value as qv
    mode = mode or qv.detect_mode(d)
    out = Riktpris(mode=mode, price_now=_n(d.get("price")),
                   ccy=str(d.get("price_currency") or d.get("currency") or "").upper())
    if out.price_now is None or out.price_now <= 0:
        out.error = "kursen saknas — inget riktpris utan dagens kurs"
        return out
    px = out.price_now
    b, err = qv.base(d)
    mcap, mcap_note = qs.market_cap_report(d)
    if b is not None:
        med_x, med_m, nd = b.multiples["median"], b.margin_q["median"], b.net_debt

        def at(equity: float) -> float:
            return round(px * max(equity / b.mcap, 0.0), 2)
        out.estimates.append(Estimate(
            "rerating", "Omvärdering", at(b.ebitda * med_x - nd),
            f"EBITDA nu {b.ebitda:,.0f} × egen median {med_x:g}× − nettoskuld {nd:,.0f} = "
            f"{b.ebitda * med_x - nd:,.0f} M mot börsvärdet {b.mcap:,.0f} M",
            f"dagens multipel {b.implied_multiple:g}×"))
        norm = b.revenue * med_m / 100
        out.estimates.append(Estimate(
            "normal", "Normal marginal", at(norm * med_x - nd),
            f"omsättning {b.revenue:,.0f} × median-marginal {med_m:.1f} % = EBITDA {norm:,.0f} × median {med_x:g}× "
            f"− nettoskuld {nd:,.0f}", f"marginal nu {b.margin:.1f} %"))
        g = _clamp_growth(b.growth)
        rev5 = b.revenue * (1 + g) ** HORIZON_YEARS
        eq5 = rev5 * med_m / 100 * med_x - nd
        disc = eq5 / (1 + DISCOUNT_RATE) ** HORIZON_YEARS
        dil, dil_rate = _dilution_factor(d)
        out.estimates.append(Estimate(
            "growth", f"Tillväxt {HORIZON_YEARS} år", round(at(disc) * dil, 2),
            f"omsättning {b.revenue:,.0f} × (1 {g * 100:+.1f} %)^{HORIZON_YEARS} = {rev5:,.0f} × median-marginal "
            f"{med_m:.1f} % × median {med_x:g}× − nettoskuld = {eq5:,.0f} M, diskonterat {DISCOUNT_RATE:.0%}/år = "
            f"{disc:,.0f} M" + (f", utspädning {dil_rate:g} %/år" if dil_rate else ""),
            "egen tillväxt kapad till " + f"{qc.GROWTH_CAP[0]:.0%}…{qc.GROWTH_CAP[1]:.0%}"
            if b.growth is not None and not qc.GROWTH_CAP[0] <= b.growth <= qc.GROWTH_CAP[1] else "egen tillväxt"))
    fcf, med_pf = _n(d.get("fcf")), _pfcf_median(d)
    if fcf is not None and fcf > 0 and med_pf is not None and mcap:
        out.estimates.append(Estimate(
            "fcf", "Kassaflöde", round(px * fcf * med_pf / mcap, 2),
            f"fritt kassaflöde {fcf:,.0f} × egen median-P/FCF {med_pf:g}× = {fcf * med_pf:,.0f} M mot börsvärdet "
            f"{mcap:,.0f} M ({mcap_note})", f"P/FCF nu {_n(d.get('p_fcf')) or '—'}"))
    if mode == qc.MODE_COMMODITY:
        eng = qs.run(d)
        base = next((s for s in eng.scenarios if s.name == "BASE"), None) if not eng.error else None
        if base is not None and base.share_price:
            out.estimates.append(Estimate(
                "base", "Råvarubas", base.share_price,
                f"{eng.commodity} vid dagens pris → EBITDA {base.ebitda:,.0f} (egen linje) × median {base.multiple:g}× "
                f"− nettoskuld {eng.net_debt:,.0f} = {base.equity:,.0f} M mot börsvärdet {eng.mcap:,.0f} M",
                "5×-motorns BASE"))
        elif eng.error:
            out.estimates.append(Estimate("base", "Råvarubas", None, "", f"DATA_GAP: {eng.error}"))
    used = out.used
    if len(used) < MIN_ESTIMATES:
        out.error = (f"för få skattningar ({len(used)} av minst {MIN_ESTIMATES})"
                     + (f" — {err}" if err else ""))
        return out
    prices = [e.price for e in used]
    out.target, out.low, out.high = round(median(prices), 2), round(min(prices), 2), round(max(prices), 2)
    out.upside_pct = round((out.target / px - 1) * 100, 1)
    years = b.years if b is not None else "egen historik"
    out.method = (f"median av {len(used)} skattningar ({', '.join(e.label.lower() for e in used)}) ur egen "
                  f"historik {years} · spannet är lägsta–högsta skattning")
    return out


# ── Kurshävstång ────────────────────────────────────────────────────────────
@dataclass
class Leverage:
    driver: str = ""
    unit: str = ""
    price_now: Optional[float] = None
    rows: list = field(default_factory=list)      # [{pct, driver, price, change_pct, elasticity}]
    up: Optional[float] = None                     # snitt-elasticitet på plusstegen
    down: Optional[float] = None                   # … på minusstegen (positivt tal)
    financial: Optional[float] = None              # nettoskuld / börsvärde
    note: str = ""
    error: Optional[str] = None


def share_leverage(d: dict, mode: Optional[str] = None, steps=LEVERAGE_STEPS) -> Leverage:
    """Kursen när drivaren går ±steg: råvarupriset via 5×-motorns kedja (at_prices) för råvarubolag,
    EBITDA vid dagens multipel för övriga. Elasticitet = kursens % / drivarens %."""
    from asymmetry import quick_config as qc
    from asymmetry import quick_scenarios as qs
    from asymmetry import quick_value as qv
    mode = mode or qv.detect_mode(d)
    out = Leverage(price_now=_n(d.get("price")))
    if out.price_now is None or out.price_now <= 0:
        out.error = "kursen saknas"
        return out
    px = out.price_now
    if mode == qc.MODE_COMMODITY:
        cp = d.get("commodity_px") or {}
        p0, fx = cp.get("p0"), cp.get("fx_now") or 1.0
        if not p0:
            out.error = "råvarupriset saknas"
            return out
        usd = p0 / fx
        res, pts = qs.at_prices(d, [(f"{pct:+.0f} %", usd * (1 + pct / 100)) for pct in steps])
        if res.error or not pts:
            out.error = res.error or "kedjan gav inga punkter"
            return out
        out.driver, out.unit = (res.commodity or "råvaran").capitalize(), res.unit
        out.financial = round(res.net_debt / res.mcap, 2) if res.mcap else None
        for pct, p in zip(steps, pts):
            out.rows.append(_row(pct, p.price, p.share_price, px))
        out.note = (f"{out.driver}-pris → EBITDA (egen linje) → EV med egen median {res.multiples['median']:g}× "
                    f"− nettoskuld {res.net_debt:,.0f} M → mot börsvärdet {res.mcap:,.0f} M")
    else:
        b, err = qv.base(d)
        if err:
            out.error = err
            return out
        out.driver, out.unit = "EBITDA", f"M {b.report_ccy}".strip()
        out.financial = round(b.net_debt / b.mcap, 2) if b.mcap else None
        for pct in steps:
            eb = b.ebitda * (1 + pct / 100)
            eq = eb * b.implied_multiple - b.net_debt
            out.rows.append(_row(pct, round(eb, 0), round(px * max(eq / b.mcap, 0.0), 2), px))
        out.note = (f"EBITDA {b.ebitda:,.0f} M × dagens multipel {b.implied_multiple:g}× − nettoskuld "
                    f"{b.net_debt:,.0f} M → mot börsvärdet {b.mcap:,.0f} M — multipeln hålls fast")
    ups = [r["elasticity"] for r in out.rows if r["pct"] > 0 and r["elasticity"] is not None]
    downs = [r["elasticity"] for r in out.rows if r["pct"] < 0 and r["elasticity"] is not None]
    out.up = round(sum(ups) / len(ups), 2) if ups else None
    out.down = round(sum(downs) / len(downs), 2) if downs else None
    return out


def _row(pct: float, driver, price: Optional[float], px: float) -> dict:
    change = None if price is None else round((price / px - 1) * 100, 1)
    return {"pct": pct, "driver": driver, "price": price, "change_pct": change,
            "elasticity": None if change is None or not pct else round(change / pct, 2)}


# ── Tillväxtscenarier ───────────────────────────────────────────────────────
@dataclass
class Growth:
    years: int = HORIZON_YEARS
    own_pct: Optional[float] = None                # egen tillväxt (kapad), %/år
    margin: Optional[float] = None                 # median-marginal %
    multiple: Optional[float] = None               # median EV/EBITDA
    price_now: Optional[float] = None
    ccy: str = ""
    rows: list = field(default_factory=list)      # [{growth_pct, own, revenue, ebitda, equity, ratio, price, cagr_pct}]
    note: str = ""
    error: Optional[str] = None


def growth_table(d: dict, years: int = HORIZON_YEARS, deltas=GROWTH_DELTAS_PP) -> Growth:
    """Egen tillväxt (kapad) plus/minus procentenheter i `years` år → kurs via median-marginal och median-multipel."""
    from asymmetry import quick_value as qv
    out = Growth(years=years, price_now=_n(d.get("price")))
    b, err = qv.base(d)
    if err:
        out.error = err
        return out
    if out.price_now is None or out.price_now <= 0:
        out.error = "kursen saknas"
        return out
    g = _clamp_growth(b.growth)
    out.own_pct, out.margin, out.multiple = round(g * 100, 1), b.margin_q["median"], b.multiples["median"]
    out.ccy = b.price_ccy
    for delta in deltas:
        rate = g + delta / 100
        rev = b.revenue * (1 + rate) ** years
        eb = rev * out.margin / 100
        eq = eb * out.multiple - b.net_debt
        ratio = eq / b.mcap
        price = round(out.price_now * max(ratio, 0.0), 2)
        cagr = round((ratio ** (1 / years) - 1) * 100, 1) if ratio > 0 else None
        out.rows.append({"growth_pct": round(rate * 100, 1), "own": delta == 0, "revenue": round(rev, 0),
                         "ebitda": round(eb, 0), "equity": round(eq, 0), "ratio": round(ratio, 2), "price": price,
                         "cagr_pct": cagr})
    out.note = (f"omsättning {b.revenue:,.0f} M × (1 + tillväxt)^{years} × median-marginal {out.margin:.1f} % × median "
                f"{out.multiple:g}× EV/EBITDA − nettoskuld {b.net_debt:,.0f} M → mot börsvärdet {b.mcap:,.0f} M "
                f"{b.report_ccy} → kurs · egen tillväxt {out.own_pct:+.1f} %/år ({b.years})")
    return out


# ── HTML (rena funktioner) ──────────────────────────────────────────────────
def _f(v, fmt: str = "{:,.0f}") -> str:
    return "—" if v is None else fmt.format(v)


def _cell(tag: str, i: int, text) -> str:
    return f"<{tag} style='text-align:{'left' if i == 0 else 'right'};'>{escape(str(text))}</{tag}>"


def _table(head: list, rows: list, bold_rows: Optional[set] = None) -> str:
    th = "".join(_cell("th", i, h) for i, h in enumerate(head))
    body = []
    for i, r in enumerate(rows):
        style = f"font-weight:700;color:{CYAN};" if bold_rows and i in bold_rows else ""
        body.append(f"<tr style='{style}'>" + "".join(_cell("td", j, c) for j, c in enumerate(r)) + "</tr>")
    return (f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.76rem;color:{TEXT};'>"
            f"<tr style='color:{DIM};'>{th}</tr>{''.join(body)}</table></div>")


def _upside_color(pct: Optional[float]) -> str:
    if pct is None:
        return GREY
    return GREEN if pct >= 30 else CYAN if pct >= 0 else AMBER if pct >= -15 else RED


def riktpris_html(r: Riktpris) -> str:
    if r.error:
        return cards.kpi_row([("Riktpris", "—", GREY, f"DATA_GAP: {r.error}")]) + _estimates_html(r)
    items = [("Riktpris", f"{r.target:,.2f} {r.ccy}".strip(), _upside_color(r.upside_pct),
              f"{r.upside_pct:+.0f} % mot kursen {r.price_now:,.2f}"),
             ("Spann", f"{r.low:,.2f}–{r.high:,.2f}", DIM, "lägsta–högsta skattning"),
             ("Metod", f"median av {len(r.used)}", CYAN, r.mode)]
    return cards.kpi_row(items) + _estimates_html(r) + _foot(r.method)


def _foot(text: str) -> str:
    return f"<div style='color:{DIM};font-size:0.7rem;'>{escape(text)}</div>"


def _estimates_html(r: Riktpris) -> str:
    if not r.estimates:
        return ""
    px = r.price_now or 0
    rows = [(e.label, _f(e.price, "{:,.2f}"), _f((e.price / px - 1) * 100 if e.price and px else None, "{:+.0f} %"),
             e.formula or e.note) for e in r.estimates]
    return _table(["Skattning", "Kurs", "Mot i dag", "Formel"], rows)


def leverage_html(lv: Leverage) -> str:
    if lv.error:
        return cards.kpi_row([("Kurshävstång", "—", GREY, f"DATA_GAP: {lv.error}")])
    items = [("Hävstång upp", f"{lv.up:.2f}×" if lv.up is not None else "—", GREEN,
              f"{lv.driver} +10 % → kurs {lv.up * 10:+.0f} %" if lv.up is not None else ""),
             ("Hävstång ned", f"{lv.down:.2f}×" if lv.down is not None else "—", RED,
              f"{lv.driver} −10 % → kurs {-lv.down * 10:+.0f} %" if lv.down is not None else ""),
             ("Finansiell hävstång", f"{lv.financial:.2f}×" if lv.financial is not None else "—", AMBER,
              "nettoskuld / börsvärde — skulden förstärker båda hållen")]
    rows = []
    for r in lv.rows:
        small = isinstance(r["driver"], float) and r["driver"] < 1000
        rows.append((f"{r['pct']:+.0f} %", _f(r["driver"], "{:,.2f}" if small else "{:,.0f}"),
                     _f(r["price"], "{:,.2f}"), _f(r["change_pct"], "{:+.1f} %"), _f(r["elasticity"], "{:.2f}×")))
    head = [lv.driver, f"Nivå ({lv.unit})" if lv.unit else "Nivå", "Kurs", "Mot i dag", "Hävstång"]
    return cards.kpi_row(items) + _table(head, rows) + _foot(lv.note)


def growth_html(g: Growth) -> str:
    if g.error:
        return cards.kpi_row([("Tillväxtscenarier", "—", GREY, f"DATA_GAP: {g.error}")])
    rows = [(f"{r['growth_pct']:+.1f} %/år" + (" (egen takt)" if r["own"] else ""), _f(r["revenue"]), _f(r["ebitda"]),
             _f(r["equity"]), _f(r["ratio"], "{:.2f}×"), _f(r["price"], "{:,.2f}"), _f(r["cagr_pct"], "{:+.1f} %"))
            for r in g.rows]
    bold = {i for i, r in enumerate(g.rows) if r["own"]}
    head = ["Tillväxt", f"Omsättning år {g.years}", "EBITDA", "Eget kapital", "× börsvärde", f"Kurs {g.ccy}", "Per år"]
    return _table(head, rows, bold) + _foot(g.note)


# ── Fliken ──────────────────────────────────────────────────────────────────
def cached(st, ticker: str, api=None, fetch_fn: Optional[Callable] = None, force: bool = False) -> dict:
    """Snabbkollens data per aktie, cachad en timme i sessionen — samma cache som 🐺 Wolf Asymmetry."""
    cache = st.session_state.setdefault(CACHE_KEY, {})
    hit = cache.get(ticker)
    if hit and not force and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    if fetch_fn is None:
        from asymmetry import quick_data
        fetch_fn = quick_data.fetch
    data = fetch_fn(ticker, api=api)
    empty = isinstance(data, dict) and "source" in data and not data.get("source") \
        and not any(data.get(k) for k in ("prices", "price", "close", "last"))
    if empty:                                         # ett tomt svar (källorna nere) cachas inte (granskning 2, Q1)
        cache.pop(ticker, None)
    else:
        cache[ticker] = (time.time(), data)
    return data


def _section(st, title: str, sub: str = "") -> None:
    tail = f" <span style='color:{DIM};text-transform:none;letter-spacing:0;'>— {escape(sub)}</span>" if sub else ""
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;"
                f"margin:12px 0 6px;'>{escape(title)}{tail}</div>", unsafe_allow_html=True)


def render(st, ident, strategy: str, api=None, fetch_fn: Optional[Callable] = None) -> None:
    """Snabbkollen för aktien (poängkortet, mätarna, cykelvolatiliteten, motorn och graferna) med Thors tre tillägg."""
    from asymmetry import quick
    from asymmetry import quick_config as qc
    from asymmetry import quick_ui as qui
    from asymmetry import quick_value as qv
    from ui.components import note
    force = st.button("↺ Hämta igen", key="thor_asym_refresh")
    with st.spinner(f"Snabbkollen hämtar {ident.yf} …"):
        d = cached(st, ident.yf, api, fetch_fn, force)
    if not d.get("prices") and d.get("source") in ("—", "", None):
        note(f"Hittade inget för {ident.yf} i Börsdata eller Yahoo — Snabbkollen kan inte räkna.")
        return
    detected = qv.detect_mode(d)
    pick = st.radio("Läge", [qc.MODE_AUTO, qc.MODE_COMMODITY, qc.MODE_VALUE], horizontal=True, key="thor_asym_mode",
                    help="Auto = Råvara när bolaget har ett råvarutema, annars Värde.")
    mode = detected if pick == qc.MODE_AUTO else pick
    res = quick.score(d, mode)
    qui._score_card(res, d)
    qui._gauges(res)
    qui._volatility(res)
    _section(st, "RIKTPRIS", "median av skattningar ur egen historik — inte en prognos")
    st.markdown(riktpris_html(riktpris(d, mode)), unsafe_allow_html=True)
    _section(st, "HÄVSTÅNG PÅ AKTIEKURSEN", "drivaren ±10/20/30 % med multipeln fast")
    st.markdown(leverage_html(share_leverage(d, mode)), unsafe_allow_html=True)
    _section(st, f"TILLVÄXTSCENARIER — {HORIZON_YEARS} ÅR", "egen tillväxt plus/minus procentenheter")
    st.markdown(growth_html(growth_table(d)), unsafe_allow_html=True)
    if mode == qc.MODE_VALUE:
        qui._value(d)
        qui._value_engine(d)
    else:
        lev = qui._leverage(d)
        qui._engine(d, lev)
    qui._charts(d)
    if d.get("filled_yahoo"):
        note("Ur Yahoo (Börsdata tomt): " + ", ".join(d["filled_yahoo"]))
    note("Samma Snabbkoll som 🐺 Wolf Asymmetry (Survival · Margin of Safety · Confidence, 5×-motorn, graferna). "
         "Riktpriset, kurshävstången och tillväxtscenarierna är Thors tillägg ur samma data och samma kedjor — "
         "egen historik, inga antaganden utifrån, inget sparas. Ett riktpris är en skattning, inte en prognos.")
