"""
asymmetry/quick_value_scenarios.py — 5×-motorn för värdebolag (PR 2).

Samma kedja som råvarumotorn, men drivarna är bolagets egna:
  omsättning × (1 + egen tillväxt)^år → × marginal (egen historik) = EBITDA
  → × multipel (egen EV/EBITDA-historik) = EV → − nettoskuld = eget kapital
  → mot dagens börsvärde → kurs (antalet aktier hålls fast)

Baklänges: vilken årlig tillväxt i fem år krävs för 2×/3×/5×/10× vid egen
median-marginal och median-multipel — mot bolagets egen historiska takt.
Thesis killers = värdefällorna som slog till + det som inte kan mätas.
Rena funktioner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from asymmetry import quick_config as qc
from asymmetry import quick_value as qv


@dataclass
class ValueScenario:
    name: str
    margin: float
    margin_key: str
    multiple: float
    multiple_key: str
    years: int
    revenue: float
    ebitda: float
    equity: float
    ratio: float
    share_price: Optional[float]


@dataclass
class GrowthRequirement:
    multiple: int
    growth_pct: float                  # krävd årlig tillväxt
    revenue: float
    verdict: str                       # JA | VILLKORAT | NEJ


@dataclass
class ValueEngine:
    base: Optional[qv.ValueBase] = None
    growth_used: Optional[float] = None
    scenarios: list = field(default_factory=list)
    requirements: list = field(default_factory=list)
    five_x: str = ""
    five_x_text: str = ""
    stress: list = field(default_factory=list)       # [(marginalnyckel, marginal, {multipelnyckel: (ratio, kurs)})]
    killers: list = field(default_factory=list)      # [(rubrik, detalj, uppmätt)]
    error: Optional[str] = None


def _growth(b: qv.ValueBase) -> float:
    g = b.growth if b.growth is not None else 0.0
    lo, hi = qc.GROWTH_CAP
    return min(max(g, lo), hi)


def _chain(b: qv.ValueBase, revenue: float, margin: float, multiple: float) -> tuple:
    eb = revenue * margin / 100
    eq = eb * multiple - b.net_debt
    ratio = eq / b.mcap
    return eb, eq, ratio, (round(b.share_price * max(ratio, 0), 2) if b.share_price else None)


def run(d: dict) -> ValueEngine:
    b, err = qv.base(d)
    if err:
        return ValueEngine(error=err)
    out = ValueEngine(base=b)
    g = _growth(b)
    out.growth_used = round(g, 4)

    for name, mkey, xkey, years, cmp in qc.VALUE_SCENARIOS:
        rev = b.revenue * (1 + g) ** years
        m, x = (b.margin if mkey == "now" else b.margin_q[mkey]), b.multiples[xkey]
        if cmp == "min" and b.margin < m:
            m, mkey = b.margin, "now"
        elif cmp == "max" and b.margin > m:
            m, mkey = b.margin, "now"
        eb, eq, ratio, px = _chain(b, rev, m, x)
        out.scenarios.append(ValueScenario(name, round(m, 2), mkey, x, xkey, years, round(rev, 0), round(eb, 0),
                                           round(eq, 0), round(ratio, 2), px))

    med_m, med_x = b.margin_q["median"], b.multiples["median"]
    own = max(g, 0.0) * 100
    for k in qc.TARGET_MULTIPLES:
        rev_req = (k * b.mcap + b.net_debt) / med_x / (med_m / 100)
        req = 0.0 if rev_req <= b.revenue else ((rev_req / b.revenue) ** (1 / qc.VALUE_REQ_YEARS) - 1) * 100
        verdict = ("JA" if req <= own else "VILLKORAT" if req <= own + qc.VALUE_COND_GAP_PP else "NEJ")
        out.requirements.append(GrowthRequirement(k, round(req, 1), round(rev_req, 0), verdict))
    five = next(r for r in out.requirements if r.multiple == qc.FIVE_X)
    out.five_x = five.verdict
    out.five_x_text = (f"{five.growth_pct:.1f} %/år i {qc.VALUE_REQ_YEARS} år vid egen median-marginal "
                       f"{med_m:.1f} % och median {med_x:g}× EV/EBITDA — egen tillväxt {own:.1f} %/år")

    for mkey in qc.VALUE_STRESS_MARGINS:
        row = {}
        for xkey in qc.VALUE_STRESS_MULTIPLES:
            _eb, _eq, ratio, px = _chain(b, b.revenue, b.margin_q[mkey], b.multiples[xkey])
            row[xkey] = (round(ratio, 2), px)
        out.stress.append((mkey, round(b.margin_q[mkey], 2), row))

    for t in qv.traps(d, b):
        if t.flagged:
            out.killers.append((t.label, t.detail, True))
    if b.growth is not None and b.growth > qc.GROWTH_CAP[1]:
        out.killers.append(("Tillväxten kapad", f"egen tillväxt {b.growth * 100:.0f} %/år räknas som "
                                                f"{qc.GROWTH_CAP[1] * 100:.0f} % — hög takt håller sällan", True))
    out.killers.append(("Multipeln kan stanna låg", "ett hatat bolag kan handlas under sin median i åratal — "
                                                    "omvärderingen kräver en katalysator", False))
    out.killers.append(("Konkurrens, teknik, ledning", "syns inte i siffrorna — läs rapporterna", False))
    return out
