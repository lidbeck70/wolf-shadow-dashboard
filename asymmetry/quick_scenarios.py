"""
asymmetry/quick_scenarios.py — 5×-motorn i Snabbkollen, helt automatisk
(Asymmetry 1.1 §27–31). Utanför 300.

Kedjan, med varje antagande synligt:
  råvarupris → EBITDA   bolagets egen linje (EBITDA = a + b × pris, quick_leverage)
  EBITDA → EV           bolagets egen EV/EBITDA-historik (25:e percentil / median / 75:e)
  EV − nettoskuld       senaste rapporten (Börsdata)
  → eget kapital        mot dagens börsvärde (omräknat till rapportvalutan)
  → kurs                dagens kurs × (eget kapital / börsvärde) — antalet aktier hålls fast

Baklänges: vilket råvarupris krävs för 2×/3×/5×/10× vid egen medianmultipel?
Jämförs med råvarans högsta årssnitt under tio år → 5× POTENTIAL JA/VILLKORAT/NEJ.
Thesis killers bara ur uppmätta tal — inga sannolikheter. Rena funktioner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from asymmetry import quick_config as qc
from asymmetry import quick_leverage as ql


@dataclass
class Scenario:
    name: str
    price_pct: float
    price: float                     # råvarans enhet (USD)
    multiple_key: str
    multiple: float
    ebitda: float                    # rapportvalutan, miljoner
    ev: float
    equity: float
    ratio: float                     # eget kapital / dagens börsvärde
    share_price: Optional[float]     # handelsvalutan
    outside_history: bool = False


@dataclass
class Requirement:
    multiple: int                    # 2, 3, 5, 10
    price: Optional[float]           # råvarans enhet
    price_pct: Optional[float]
    verdict: str                     # JA | VILLKORAT | NEJ | —


@dataclass
class Killer:
    label: str
    detail: str
    measured: bool = True            # False = kan inte mätas automatiskt


@dataclass
class EngineResult:
    commodity: str = ""
    unit: str = ""
    multiples: dict = field(default_factory=dict)      # {"low", "median", "high"}
    net_debt: Optional[float] = None
    mcap: Optional[float] = None                       # rapportvalutan
    mcap_note: str = ""
    report_ccy: str = ""
    price_ccy: str = ""
    share_price: Optional[float] = None
    price_high: Optional[float] = None                 # högsta årssnittet, råvarans enhet
    price_low: Optional[float] = None
    price_pctl: Optional[float] = None                 # dagens pris mot tio års årssnitt (%)
    cycle_top: bool = False
    scenarios: list = field(default_factory=list)
    requirements: list = field(default_factory=list)
    five_x: str = ""                                   # JA | VILLKORAT | NEJ
    five_x_text: str = ""
    stress: list = field(default_factory=list)         # [(pct, {key: (ratio, kurs)})]
    killers: list = field(default_factory=list)
    error: Optional[str] = None


def _n(v) -> Optional[float]:
    try:
        return None if v is None or v != v else float(v)
    except (TypeError, ValueError):
        return None


def _quantile(vals: list, q: float) -> float:
    s = sorted(vals)
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def own_multiples(hist) -> dict:
    vals = [x for x in (_n(v) for v in (hist or [])) if x is not None and 0 < x < 100]
    if len(vals) < qc.MIN_HISTORY:
        return {}
    return {"min": round(min(vals), 2), "low": round(_quantile(vals, 0.25), 2),
            "median": round(_quantile(vals, 0.5), 2), "high": round(_quantile(vals, 0.75), 2)}


def price_percentile(p0: float, prices: dict) -> Optional[float]:
    """Andel av årssnitten (%) som ligger under eller på dagens pris."""
    vals = [v for v in (prices or {}).values() if v is not None]
    if not vals or not p0:
        return None
    return round(sum(1 for v in vals if v <= p0) / len(vals) * 100, 0)


def market_cap_report(d: dict) -> tuple:
    """(börsvärde i rapportvalutan, miljoner; not). Börsdata först, Yahoo sist.
    Valutan räknas om med den fasta FX-tabellen (sheets_refresh.FX_TO_USD)."""
    from sheets_refresh import FX_TO_USD
    rep = str(d.get("currency") or "USD").upper()
    for mc, ccy, src in ((d.get("mcap_bd"), d.get("price_currency"), "Börsdata"),
                         (d.get("mcap_yahoo"), d.get("mcap_yahoo_ccy"), "Yahoo")):
        mc = _n(mc)
        if mc is None or mc <= 0:
            continue
        ccy = str(ccy or rep).upper()
        if ccy == rep:
            return mc, f"börsvärde {src}"
        if ccy in FX_TO_USD and rep in FX_TO_USD:
            return mc * FX_TO_USD[ccy] / FX_TO_USD[rep], f"börsvärde {src}, {ccy}→{rep} ur fast FX-tabell"
    return None, ""


def run(d: dict, lev: Optional[ql.LeverageEstimate] = None) -> EngineResult:
    lev = lev or ql.from_data(d)
    res = EngineResult(commodity=lev.commodity, unit=lev.unit, report_ccy=str(d.get("currency") or "").upper(),
                       price_ccy=str(d.get("price_currency") or d.get("currency") or "").upper())
    cp = d.get("commodity_px") or {}
    p0, fx = cp.get("p0"), cp.get("fx_now") or 1.0
    f = (lev.fits or {}).get("EBITDA")
    if lev.error and not f:
        res.error = lev.error
        return res
    if not (f and f.n >= qc.LEV_MIN_YEARS and f.b > 0 and f.r2 >= qc.LEV_MIN_R2 and p0 and f.at(p0) > 0):
        res.error = ("EBITDA följer inte råvarupriset tydligt nog (5×-motorn värderar på EBITDA)"
                     + (f" — R² {f.r2:.2f}" if f else ""))
        return res
    res.multiples = own_multiples(d.get("ev_ebitda_hist"))
    if not res.multiples:
        res.error = f"för kort EV/EBITDA-historik (< {qc.MIN_HISTORY} år) — ingen egen multipel att värdera med"
        return res
    res.net_debt = _n(d.get("net_debt"))
    if res.net_debt is None:
        res.error = "nettoskulden saknas"
        return res
    res.mcap, res.mcap_note = market_cap_report(d)
    if not res.mcap:
        res.error = "börsvärdet saknas"
        return res
    res.share_price = _n(d.get("price"))
    prices = cp.get("prices") or {}
    if prices:
        res.price_high = round(max(prices.values()) / fx, 2)
        res.price_low = round(min(prices.values()) / fx, 2)

    def equity_at(pct: float, mkey: str) -> tuple:
        p = p0 * (1 + pct / 100)
        eb = f.at(p)
        ev = eb * res.multiples[mkey]
        eq = ev - res.net_debt
        return p, eb, ev, eq, eq / res.mcap

    res.price_pctl = price_percentile(p0, prices)
    res.cycle_top = res.price_pctl is not None and res.price_pctl >= qc.CYCLE_TOP_PCTL
    for name, pct, mkey in qc.SCENARIOS:
        if name == "BEAR" and res.cycle_top:
            mkey = qc.BEAR_AT_TOP_MULTIPLE          # i toppen: sämsta egna multipeln i bear
        p, eb, ev, eq, ratio = equity_at(pct, mkey)
        res.scenarios.append(Scenario(
            name, pct, round(p / fx, 2), mkey, res.multiples[mkey], round(eb, 0), round(ev, 0), round(eq, 0),
            round(ratio, 2), round(res.share_price * max(ratio, 0), 2) if res.share_price else None,
            outside_history=bool(prices) and not (min(prices.values()) <= p <= max(prices.values()))))

    med = res.multiples["median"]
    high_rep = max(prices.values()) if prices else None
    for k in qc.TARGET_MULTIPLES:
        eb_req = (k * res.mcap + res.net_debt) / med
        p_req = (eb_req - f.a) / f.b
        if p_req <= 0:
            res.requirements.append(Requirement(k, 0.0, -100.0, "JA"))
            continue
        verdict = "—"
        if high_rep:
            verdict = ("JA" if p_req <= high_rep else
                       "VILLKORAT" if p_req <= high_rep * qc.FIVE_X_CONDITIONAL_FACTOR else "NEJ")
        res.requirements.append(Requirement(k, round(p_req / fx, 2), round((p_req / p0 - 1) * 100, 0), verdict))
    five = next((r for r in res.requirements if r.multiple == qc.FIVE_X), None)
    if five:
        res.five_x = five.verdict
        name = (res.commodity or "råvaran").capitalize()
        res.five_x_text = (f"{name} {five.price_pct:+.0f} % ({five.price:,.2f} {res.unit}) vid egen median "
                           f"{med:g}× EV/EBITDA" + (f" · högsta årssnitt 10 år {res.price_high:,.2f}"
                                                    if res.price_high else ""))
    for pct in qc.STRESS_PRICE_PCT:
        row = {}
        for mkey in qc.STRESS_MULTIPLES:
            _p, _eb, _ev, _eq, ratio = equity_at(pct, mkey)
            row[mkey] = (round(ratio, 2), round(res.share_price * max(ratio, 0), 2) if res.share_price else None)
        res.stress.append((pct, row))
    res.killers = killers(d, lev, res)
    return res


def killers(d: dict, lev: ql.LeverageEstimate, res: EngineResult) -> list:
    """☠️ Vad dödar caset? Bara ur uppmätta tal, plus två som inte går att mäta
    automatiskt och sägs rakt ut. Inga sannolikheter (spec §31)."""
    out = []
    name = (lev.commodity or "råvaran")
    lo = lev.downside or {}
    if lev.downside_label == "SKÖR":
        out.append(Killer("Råvarukollaps", f"{lev.downside_basis} negativt redan vid {name} −20 % "
                                           f"({lo.get(-20.0, 0):,.0f} M)"))
    elif lev.downside_label == "MÅTTLIG":
        out.append(Killer("Råvarukollaps", f"{lev.downside_basis} negativt vid {name} −30 % ({lo.get(-30.0, 0):,.0f} M)"))
    nd = _n(d.get("nd_ebitda"))
    if nd is not None and nd > qc.KILL_ND_EBITDA:
        out.append(Killer("Skuld", f"{nd:.1f}× nettoskuld/EBITDA — ett prisfall kan tvinga fram nyemission"))
    dil = _n(d.get("shares_growth_3y_pct"))
    if dil is not None and dil > qc.KILL_DILUTION_3Y_PCT:
        out.append(Killer("Utspädning", f"+{dil:.0f} % fler aktier på tre år — uppsidan delas på fler"))
    if lev.r2 is not None and lev.r2 < qc.KILL_WEAK_R2:
        out.append(Killer("Annat än priset styr", f"R² {lev.r2:.2f} — volym, kostnader eller förvärv förklarar "
                                                  f"mycket av resultatet"))
    if res.report_ccy and res.report_ccy != "USD":
        out.append(Killer("Valuta", f"{name.capitalize()} prissätts i USD, rapporterna i {res.report_ccy} — "
                                    f"en starkare {res.report_ccy} äter marginalen"))
    ev_now = _n(d.get("ev_ebitda"))
    if ev_now is not None and res.multiples \
            and ev_now > res.multiples["median"] * (1 + qc.KILL_EV_PREMIUM_PCT / 100):
        prem = (ev_now / res.multiples["median"] - 1) * 100
        out.append(Killer("Värderingen redan hög", f"EV/EBITDA {ev_now:.1f}× är {prem:.0f} % över egen median "
                                                   f"{res.multiples['median']:g}×"))
    if res.cycle_top:
        out.append(Killer("Cykeltopp", f"{name.capitalize()} ligger över {res.price_pctl:.0f} % av tio års "
                                       f"årssnitt — marknaden brukar sätta en lägre multipel på toppvinster, så "
                                       f"medianen {res.multiples['median']:g}× är troligen generös. BEAR räknas med "
                                       f"lägsta egna {res.multiples['min']:g}×"))
    if any(s.outside_history for s in res.scenarios if s.price_pct > 0):
        out.append(Killer("Bull kräver nya pristoppar", "Bull-scenarierna ligger över tio års högsta årssnitt — "
                                                        "linjen extrapoleras"))
    if lev.band == qc.BREAK_EVEN_NOT_MEASURABLE:
        out.append(Killer("En råvara räknas", f"{lev.break_even_note} — exponeringen per metall är okänd här"))
    else:
        out.append(Killer("En råvara räknas", "Biprodukter och övriga metaller ingår inte — exponeringen per metall "
                                              "är okänd här", measured=False))
    out.append(Killer("Capex, produktion, tillstånd", "Kostnadsöverdrag, produktionsstörningar och tillstånd "
                                                      "syns inte i siffrorna — läs rapporterna", measured=False))
    return out


# ── Värdering vid givna råvarupriser (Guld/Silver-fliken, PR B) ───────────────
@dataclass
class PricePoint:
    name: str
    price: float                     # råvarans enhet (USD)
    price_pct: float                 # mot dagens pris
    revenue: Optional[float]         # rapportvalutan, miljoner (egen linje, None utan samband)
    ebitda: float
    fcf: Optional[float]
    ev: float
    equity: float
    ratio: float                     # eget kapital / dagens börsvärde
    share_price: Optional[float]
    outside_history: bool = False


def at_prices(d: dict, points, lev: Optional[ql.LeverageEstimate] = None,
              multiple_key: str = "median") -> tuple:
    """(EngineResult, [PricePoint]) — samma kedja som 5×-motorn, men vid
    råvarupriser som anroparen väljer (USD i råvarans enhet). Ingen egen
    räknelogik: EBITDA ur bolagets linje, EV med egen multipel, − nettoskuld."""
    lev = lev or ql.from_data(d)
    res = run(d, lev)
    if res.error:
        return res, []
    cp = d.get("commodity_px") or {}
    p0, fx = cp.get("p0"), cp.get("fx_now") or 1.0
    prices = cp.get("prices") or {}
    fits = lev.fits or {}
    f = fits["EBITDA"]
    mult = res.multiples[multiple_key]
    out = []
    for name, usd in points:
        if not usd or usd <= 0:
            continue
        p = usd * fx
        eb = f.at(p)
        ev = eb * mult
        eq = ev - res.net_debt
        ratio = eq / res.mcap
        out.append(PricePoint(
            name, round(usd, 2), round((p / p0 - 1) * 100, 1),
            round(fits["Intäkt"].at(p), 0) if fits.get("Intäkt") else None, round(eb, 0),
            round(fits["FCF"].at(p), 0) if fits.get("FCF") else None, round(ev, 0), round(eq, 0), round(ratio, 2),
            round(res.share_price * max(ratio, 0), 2) if res.share_price else None,
            outside_history=bool(prices) and not (min(prices.values()) <= p <= max(prices.values()))))
    return res, out
