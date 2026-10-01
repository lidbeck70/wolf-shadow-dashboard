"""
asymmetry/quick_value.py — Wolf Asymmetry för värdebolag (inte råvaror).

Asymmetrin i ett värdebolag kommer inte ur ett råvarupris utan ur bolagets
egen historik, tre drivare med uträkningen synlig:

  Omvärdering           EBITDA nu × egen median-EV/EBITDA − nettoskuld
  Marginalåterhämtning  omsättning × egen median-marginal × dagens multipel − nettoskuld
  Dubbel hävstång       båda samtidigt
  Tillväxt              egen omsättnings-CAGR (visas, används i 5×-motorn)

"Dagens multipel" = (börsvärde + nettoskuld) / EBITDA, så allt räknas mot
samma börsvärde. Värdefälle-kontrollerna läses ur samma tal. Banker,
försäkring och fastighet ger DATA_GAP — EBITDA bär inte deras värdering.
Rena funktioner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Optional

from asymmetry import quick_config as qc


def _n(v) -> Optional[float]:
    try:
        return None if v is None or v != v else float(v)
    except (TypeError, ValueError):
        return None


def detect_mode(d: dict) -> str:
    """Råvara när bolaget har ett råvarutema, annars Värde."""
    theme = ((d.get("commodity_px") or {}).get("commodity") or "").lower()
    return qc.MODE_COMMODITY if theme in qc.COMMODITY_THEMES else qc.MODE_VALUE


def is_financial(d: dict) -> Optional[str]:
    """Förklaring när bolaget är bank/försäkring/fastighet, annars None."""
    try:
        if int(d.get("bd_branch_id") or 0) in qc.FINANCIAL_BRANCH_IDS:
            return "Börsdata-branschen är bank, kredit, investmentbolag, försäkring eller fastighet"
    except (TypeError, ValueError):
        pass
    text = str(d.get("sector_text") or "").lower()
    hit = next((k for k in qc.FINANCIAL_KEYWORDS if k in text), None)
    return f"Yahoo: {d.get('sector_text')}" if hit else None


def _dict(series) -> dict:
    return {int(y): float(v) for y, v in (series or []) if _n(v) is not None}


def _quantile(vals: list, q: float) -> float:
    s = sorted(vals)
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def cagr(series: dict, years: Optional[int] = None) -> Optional[float]:
    """Årlig tillväxt mellan första och sista året (eller de sista `years`)."""
    ys = sorted(y for y, v in series.items() if v and v > 0)
    if len(ys) < 2:
        return None
    if years is not None:
        ys = [y for y in ys if y >= ys[-1] - years]
        if len(ys) < 2 or ys[-1] - ys[0] < years:
            return None
    n = ys[-1] - ys[0]
    return (series[ys[-1]] / series[ys[0]]) ** (1 / n) - 1 if n > 0 else None


def slope(series: dict, years: int) -> Optional[float]:
    """OLS-lutning (enheter per år) över de sista `years` åren."""
    pts = sorted((y, v) for y, v in series.items())[-years:]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx, my = sum(y for y, _ in pts) / n, sum(v for _, v in pts) / n
    sxx = sum((y - mx) ** 2 for y, _ in pts)
    return sum((y - mx) * (v - my) for y, v in pts) / sxx if sxx else None


@dataclass
class ValueBase:
    """Det gemensamma underlaget för drivarna och 5×-motorn."""
    revenue: float                     # senaste årets omsättning, rapportvalutan M
    ebitda: float
    margin: float                      # dagens EBITDA-marginal, %
    margins: dict                      # {år: marginal %}
    margin_q: dict                     # {"min","p25","median","p75","max"}
    multiples: dict                    # own_multiples: min/low/median/high
    implied_multiple: float            # (börsvärde + nettoskuld) / EBITDA
    net_debt: float
    mcap: float                        # rapportvalutan
    mcap_note: str
    share_price: Optional[float]
    growth: Optional[float]            # egen CAGR, hela historiken
    growth_3y: Optional[float]
    years: str                         # "2016–2025"
    report_ccy: str = ""
    price_ccy: str = ""


def base(d: dict) -> tuple:
    """(ValueBase | None, fel | None)."""
    from asymmetry import quick_scenarios as qs
    fin = is_financial(d)
    if fin:
        return None, (f"EBITDA bär inte värderingen för banker, försäkring och fastighet ({fin}) — "
                      f"värdeläget mäter inte dem")
    rev, eb = _dict(d.get("revenue_series")), _dict(d.get("ebitda_series"))
    years = sorted(set(rev) & set(eb))
    if len(years) < qc.VALUE_MIN_YEARS:
        return None, f"för få år med både omsättning och EBITDA ({len(years)} < {qc.VALUE_MIN_YEARS})"
    margins = {y: eb[y] / rev[y] * 100 for y in years if rev[y] > 0}
    mvals = list(margins.values())
    med = median(mvals)
    if med <= 0:
        return None, "bolagets EBITDA-marginal är normalt negativ — värdedrivarna kräver en lönsam historik"
    last = years[-1]
    revenue, ebitda = rev[last], eb[last]
    cur_m = _n(d.get("ebitda_margin_pct"))
    margin = cur_m if cur_m is not None else margins.get(last, ebitda / revenue * 100)
    ebitda_now = revenue * margin / 100
    multiples = qs.own_multiples(d.get("ev_ebitda_hist"))
    if not multiples:
        return None, f"för kort EV/EBITDA-historik (< {qc.MIN_HISTORY} år)"
    nd = _n(d.get("net_debt"))
    if nd is None:
        return None, "nettoskulden saknas"
    mcap, note = qs.market_cap_report(d)
    if not mcap:
        return None, "börsvärdet saknas"
    if ebitda_now <= 0:
        return None, "dagens EBITDA ≤ 0 — omvärdering på EBITDA går inte att räkna"
    q = {"min": min(mvals), "p25": _quantile(mvals, 0.25), "median": med,
         "p75": _quantile(mvals, 0.75), "max": max(mvals)}
    g = cagr(rev)
    return ValueBase(revenue, ebitda_now, round(margin, 2), margins, {k: round(v, 2) for k, v in q.items()},
                     multiples, round((mcap + nd) / ebitda_now, 2), nd, mcap, note, _n(d.get("price")),
                     None if g is None else round(g, 4),
                     None if cagr(rev, 3) is None else round(cagr(rev, 3), 4), f"{years[0]}–{last}",
                     str(d.get("currency") or "").upper(),
                     str(d.get("price_currency") or d.get("currency") or "").upper()), None


@dataclass
class Driver:
    key: str
    label: str
    ratio: Optional[float]             # eget kapital / börsvärde
    upside_pct: Optional[float]
    formula: str
    note: str = ""


@dataclass
class Trap:
    label: str
    flagged: Optional[bool]            # True = varning, False = ok, None = kan inte mätas
    detail: str


@dataclass
class ValueResult:
    base: Optional[ValueBase] = None
    drivers: list = field(default_factory=list)
    traps: list = field(default_factory=list)
    error: Optional[str] = None


def _equity_ratio(b: ValueBase, ebitda: float, multiple: float) -> float:
    return (ebitda * multiple - b.net_debt) / b.mcap


def analyze(d: dict) -> ValueResult:
    b, err = base(d)
    if err:
        return ValueResult(error=err)
    res = ValueResult(base=b)
    med_mult, med_m = b.multiples["median"], b.margin_q["median"]
    nd = f"{b.net_debt:,.0f}"

    def drv(key, label, eb, mult, formula, note=""):
        r = _equity_ratio(b, eb, mult)
        res.drivers.append(Driver(key, label, round(r, 2), round((r - 1) * 100, 0), formula, note))
    drv("rerating", "Omvärdering", b.ebitda, med_mult,
        f"EBITDA {b.ebitda:,.0f} × egen median {med_mult:g}× − nettoskuld {nd} mot börsvärdet {b.mcap:,.0f}",
        f"dagens multipel {b.implied_multiple:g}× (börsvärde + nettoskuld / EBITDA)")
    norm_eb = b.revenue * med_m / 100
    drv("margin", "Marginalåterhämtning", norm_eb, b.implied_multiple,
        f"omsättning {b.revenue:,.0f} × egen median-marginal {med_m:.1f} % = EBITDA {norm_eb:,.0f} × dagens "
        f"{b.implied_multiple:g}× − nettoskuld {nd}", f"marginal nu {b.margin:.1f} % mot median {med_m:.1f} %")
    drv("double", "Dubbel hävstång", norm_eb, med_mult,
        f"EBITDA {norm_eb:,.0f} (median-marginal) × median {med_mult:g}× − nettoskuld {nd}",
        "omvärdering och marginalåterhämtning samtidigt")
    g = b.growth
    res.drivers.append(Driver("growth", "Tillväxt", None, None if g is None else round(g * 100, 1),
                              f"omsättningens årliga tillväxt {b.years}" + (
                                  "" if b.growth_3y is None else f" · senaste 3 år {b.growth_3y * 100:+.1f} %/år"),
                              "används i 5×-motorn"))
    res.traps = traps(d, b)
    return res


def traps(d: dict, b: ValueBase) -> list:
    """⚠️ Värdefälla? Billigt är ofta billigt av ett skäl — bara uppmätta tal."""
    out = []
    g3 = b.growth_3y
    out.append(Trap("Krympande omsättning", None if g3 is None else g3 < qc.TRAP_REVENUE_CAGR_3Y,
                    "—" if g3 is None else f"{g3 * 100:+.1f} %/år senaste 3 år"))
    sl = slope(b.margins, qc.TRAP_MARGIN_SLOPE_YEARS)
    falling = None if sl is None else (sl < 0 and b.margin < b.margin_q["median"])
    out.append(Trap("Fallande marginal", falling, "—" if sl is None else
                    f"{sl:+.2f} procentenheter/år senaste {qc.TRAP_MARGIN_SLOPE_YEARS} år, nu {b.margin:.1f} % "
                    f"mot median {b.margin_q['median']:.1f} %"))
    roic = [v for _, v in (d.get("roic_series") or []) if _n(v) is not None]
    rm = median(roic) if roic else None
    out.append(Trap("Låg avkastning på kapitalet", None if rm is None else rm < qc.TRAP_ROIC_MIN,
                    "—" if rm is None else f"median-ROIC {rm:.1f} % ({len(roic)} år) · under "
                                           f"{qc.TRAP_ROIC_MIN:g} % förstör troligen kapital"))
    nd = _n(d.get("nd_ebitda"))
    out.append(Trap("Hög skuld", None if nd is None else nd > qc.TRAP_ND_EBITDA,
                    "—" if nd is None else f"{nd:.1f}× nettoskuld/EBITDA"))
    dil = _n(d.get("shares_growth_3y_pct"))
    out.append(Trap("Utspädning", None if dil is None else dil > qc.TRAP_DILUTION_3Y,
                    "—" if dil is None else f"{dil:+.0f} % fler aktier på tre år"))
    return out
