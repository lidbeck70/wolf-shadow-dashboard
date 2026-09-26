"""
asymmetry/safety.py — Margin of Safety Score 0–10.

Hur mycket kan verkligheten försämras innan caset går sönder? Fem delar om
0–2: prismarginal mot break-even, capex-marginal (utvecklare), kostnads-
marginal, balansräkning under prisfall, värdering under prisfall. En del
utan underlag är DATA_MISSING och räknas inte som noll: totalen visas mot
det som gick att mäta ("6,5 av 8 mätbara"). Varje del förklarar sig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from asymmetry.config import ASYMMETRY_CONFIG as CFG
from asymmetry.model import Inputs, step_table


@dataclass
class Component:
    key: str
    label: str
    points: Optional[float]              # None = DATA_MISSING / ej tillämpligt
    max: float
    steps: list = field(default_factory=list)
    not_applicable: bool = False

    @property
    def measured(self) -> bool:
        return self.points is not None


@dataclass
class SafetyResult:
    components: list
    total: Optional[float]               # summa av mätta delar, None om ingen
    measurable_max: float                # summan av max för mätta delar
    max: float                           # 10
    missing: list = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.total is None:
            return "DATA_MISSING"
        if self.measurable_max < self.max:
            return f"{self.total:g}/{self.measurable_max:g} mätbara (av {self.max:g})"
        return f"{self.total:g}/{self.max:g}"

    def component(self, key: str) -> Optional[Component]:
        return next((c for c in self.components if c.key == key), None)


def _fmt(v) -> str:
    return "–" if v is None else f"{v:,.0f}"


# ── A. prismarginal ──────────────────────────────────────────────────────────
def price_margin(inp: Inputs) -> Component:
    m = CFG["margin_of_safety"]
    c = Component("price", "Prismarginal mot break-even", None, m["max_per_component"])
    if inp.price is None or inp.be is None:
        c.steps.append("DATA_MISSING: råvarupris eller break-even/AISC saknas")
        return c
    margin = (inp.price - inp.be) / inp.price * 100
    c.points = step_table(margin, m["price"], default=0.0)
    c.steps.append(f"(pris {inp.price:g} − break-even {inp.be:g}) / pris = {margin:+.1f} %")
    c.steps.append("tabell: " + " · ".join(f"≥ {f:g} % → {p:g}" for f, p in m["price"]) + " · under 0 → 0")
    return c


# ── B. capex-marginal ────────────────────────────────────────────────────────
def capex_margin(inp: Inputs) -> Component:
    m = CFG["margin_of_safety"]
    c = Component("capex", "CapEx-marginal", None, m["max_per_component"])
    if not inp.pre_revenue:
        c.not_applicable = True
        c.steps.append("Ej tillämpligt: producent utan byggprojekt — capex ligger i AISC")
        return c
    if inp.npv is None or inp.capex is None:
        c.steps.append("DATA_MISSING: NPV eller capex saknas")
        return c
    robust_to = 0.0
    for step in m["capex_steps_pct"]:
        p = inp.point(0.0, capex_pct=step)
        nav = p.value_musd
        c.steps.append(f"CapEx +{step:g} %: NAV {_fmt(nav)} MUSD"
                       + (f", IRR ≈ {p.irr_pct:.0f} %" if p.irr_pct is not None else "")
                       + (f", aktie {p.share_price:,.3g}" if p.share_price is not None else ""))
        if nav is not None and nav > 0:
            robust_to = step
        else:
            break
    c.points = step_table(robust_to, m["capex"], default=0.0) if robust_to else 0.0
    c.steps.append(f"NPV håller sig positivt upp till +{robust_to:g} % capex → {c.points:g} p")
    return c


# ── C. kostnadsmarginal ──────────────────────────────────────────────────────
def opex_margin(inp: Inputs) -> Component:
    m = CFG["margin_of_safety"]
    c = Component("opex", "Kostnadsmarginal (AISC/opex)", None, m["max_per_component"])
    base = inp.point(0.0)
    if base.fcf_musd is None:
        c.steps.append("DATA_MISSING: FCF kan inte räknas (produktion, pris eller AISC/break-even saknas)")
        return c
    if base.fcf_musd <= 0:
        c.points = 0.0
        c.steps.append(f"FCF {base.fcf_musd:,.0f} MUSD ≤ 0 redan vid dagens kostnader → 0 p")
        return c
    robust_to = 0.0
    for step in m["opex_steps_pct"]:
        p = inp.point(0.0, cost_pct=step)
        c.steps.append(f"kostnad +{step:g} %: EBITDA {_fmt(p.ebitda_musd)}, FCF {_fmt(p.fcf_musd)} MUSD")
        if p.fcf_musd is not None and p.fcf_musd > 0:
            robust_to = step
        else:
            break
    c.points = step_table(robust_to, m["opex"], default=m["opex_base_positive"]) if robust_to \
        else m["opex_base_positive"]
    c.steps.append(f"FCF positivt upp till +{robust_to:g} % kostnad → {c.points:g} p")
    return c


# ── D. balansräkning ─────────────────────────────────────────────────────────
def balance_margin(inp: Inputs) -> Component:
    m = CFG["margin_of_safety"]
    c = Component("balance", "Balansräkning under prisfall", None, m["max_per_component"])
    survived = 0
    any_measured = False
    for step in m["balance_steps_pct"]:
        p = inp.point(step)
        if p.fcf_musd is None or p.ebitda_musd is None:
            c.steps.append(f"pris {step:+g} %: DATA_MISSING")
            continue
        any_measured = True
        ok, why = True, []
        if p.fcf_musd < 0:
            need = -p.fcf_musd * m["balance_cash_cover_years"]
            if inp.cash < need:
                ok = False
                why.append(f"kassa {inp.cash:,.0f} < {need:,.0f} MUSD ({m['balance_cash_cover_years']:g} års underskott)")
            else:
                why.append(f"kassan täcker {m['balance_cash_cover_years']:g} års underskott")
        net_debt = inp.debt - inp.cash
        if p.ebitda_musd > 0 and net_debt > 0:
            nd = net_debt / p.ebitda_musd
            if nd > m["balance_max_nd_ebitda"]:
                ok = False
                why.append(f"nettoskuld/EBITDA {nd:.1f}× > {m['balance_max_nd_ebitda']:g}×")
        elif p.ebitda_musd <= 0 and net_debt > 0:
            ok = False
            why.append("nettoskuld utan EBITDA")
        survived += 1 if ok else 0
        c.steps.append(f"pris {step:+g} %: FCF {p.fcf_musd:,.0f} MUSD → "
                       + ("överlever" if ok else "ÖVERLEVER INTE") + (f" ({'; '.join(why)})" if why else ""))
    if not any_measured:
        c.steps.append("DATA_MISSING: inget prissteg gick att räkna")
        return c
    c.points = step_table(survived, m["balance"], default=0.0)
    c.steps.append(f"{survived} av {len(m['balance_steps_pct'])} steg överlevda → {c.points:g} p")
    return c


# ── E. värdering ─────────────────────────────────────────────────────────────
def valuation_margin(inp: Inputs) -> Component:
    m = CFG["margin_of_safety"]
    c = Component("valuation", "Värdering under prisfall", None, m["max_per_component"])
    p = inp.point(m["valuation_probe_pct"])
    if p.upside_pct is None:
        c.steps.append("DATA_MISSING: uppsida vid prisfall kan inte räknas (börsvärde eller ekonomi saknas)")
        return c
    c.points = step_table(p.upside_pct, m["valuation"], default=0.0)
    c.steps.append(f"pris {m['valuation_probe_pct']:+g} %: equity {p.equity_musd:,.0f} MUSD mot börsvärde "
                   f"{inp.mcap:,.0f} = {p.upside_pct:+.0f} %")
    c.steps.append("tabell: " + " · ".join(f"≥ {f:g} % → {pts:g}" for f, pts in m["valuation"]) + " · annars 0")
    return c


def margin_of_safety(inp: Inputs) -> SafetyResult:
    comps = [price_margin(inp), capex_margin(inp), opex_margin(inp), balance_margin(inp), valuation_margin(inp)]
    measured = [c for c in comps if c.measured]
    total = round(sum(c.points for c in measured), 2) if measured else None
    return SafetyResult(comps, total, sum(c.max for c in measured),
                        sum(c.max for c in comps if not c.not_applicable), inp.missing)
