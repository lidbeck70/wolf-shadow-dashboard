"""
asymmetry/model.py — en punkt i ekonomin vid ett givet pris, capex och
kostnadsläge, med samma formler som confidence.scenarios.engine.

Producent:  EBITDA = produktion × (pris − AISC × (1 + kostnad %)) / 1e6
            FCF    = EBITDA × (1 − skatt) när positivt, annars EBITDA
            EV     = EBITDA × mål-EV/EBITDA · equity = EV − skuld + kassa
Utvecklare: NAV = NPV + lutning × pris % − capex × capex %   (lutning ur
            FS-känsligheten eller annuitet, samma som scenario-motorn)
            equity = max(NAV, 0) × P/NAV + kassa − skuld
            Steady state-EBITDA/FCF via break-even när den finns.

Saknas ett fält blir punkten ofullständig (DATA_MISSING i stegen) — aldrig
ett påhittat tal. Inputs bokför antaganden och saknade fält.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from confidence import config as cfg
from confidence.data.models import CompanyInput
from confidence.scenarios import engine as se


@dataclass
class Point:
    price_pct: float
    capex_pct: float = 0.0
    cost_pct: float = 0.0
    price: Optional[float] = None
    revenue_musd: Optional[float] = None
    ebitda_musd: Optional[float] = None
    fcf_musd: Optional[float] = None
    margin_pct: Optional[float] = None       # EBITDA / intäkt
    value_musd: Optional[float] = None       # EV (producent) eller NAV (utvecklare)
    irr_pct: Optional[float] = None          # utvecklare: skalad ur studiens IRR (MODELLED)
    equity_musd: Optional[float] = None
    share_price: Optional[float] = None
    upside_pct: Optional[float] = None
    steps: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.equity_musd is not None


class Inputs:
    """Läser bolaget en gång; punkter räknas sedan utan nya uppslag.
    Bygger på scenario-motorns _Inputs så antaganden och saknade fält
    bokförs på samma sätt (och med samma default-tabell i config)."""

    def __init__(self, company: CompanyInput):
        self.c = company
        self.inp = se._Inputs(company)
        inp = self.inp
        self.stage = company.stage
        self.pre_revenue = company.stage in cfg.PRE_REVENUE
        self.royalty = company.stage == "royalty"
        self.price = inp.num("commodity_price")
        self.tax = inp.default("tax_rate_pct", cfg.DEFAULT_TAX_RATE_PCT, "Skattesats", "%") / 100
        self.debt = inp.default("debt_musd", 0.0, "Skuld", "MUSD")
        self.cash = inp.default("cash_musd", 0.0, "Kassa", "MUSD")
        self.mcap, self.shares, self.share_price = se._anchor(inp)
        if not self.shares:
            self.shares = inp.num("basic_shares_m", required=False)
        self.unit = str(company.get("production_unit").value) if company.has("production_unit") else "enheter"
        # produktion: årsproduktion, annars "produktion nu" (Durrett-arkets fält)
        self.prod = inp.num("annual_production", required=False)
        if self.prod is None:
            self.prod = inp.num("production_current", required=not self.pre_revenue)
        if self.pre_revenue:
            self.npv = inp.num("npv_musd")
            self.capex = inp.num("capex_musd")
            self.be = inp.num("breakeven_price", required=False)
            self.irr = inp.num("irr_pct", required=False)
            self.pnav = inp.default("target_p_nav", cfg.DEFAULT_TARGET_P_NAV, "P/NAV vid omvärdering", "×")
            self.slope, self.slope_text = se._nav_slope(inp, self.npv, self.prod, self.tax)
            self.aisc, self.mult, self.margin_pct = None, None, None
        else:
            self.npv = self.capex = self.irr = self.slope = None
            self.slope_text, self.pnav = "", None
            self.mult = inp.default("target_ev_ebitda", cfg.DEFAULT_TARGET_EV_EBITDA, "EV/EBITDA i scenariot", "×")
            self.aisc = None if self.royalty else inp.num("aisc")
            self.margin_pct = inp.num("ebitda_margin_pct") if self.royalty else None
            self.be = self.aisc if self.aisc is not None else inp.num("breakeven_price", required=False)

    @property
    def missing(self) -> list:
        return list(self.inp.missing)

    @property
    def assumptions(self) -> list:
        return list(self.inp.assumptions)

    # ── en punkt ────────────────────────────────────────────────────────
    def point(self, price_pct: float = 0.0, capex_pct: float = 0.0, cost_pct: float = 0.0) -> Point:
        p = Point(price_pct, capex_pct, cost_pct)
        if self.price is None:
            p.steps.append("DATA_MISSING: råvarupris saknas")
            return p
        p.price = self.price * (1 + price_pct / 100)
        p.steps.append(f"pris {self.price:g} × (1 {price_pct:+g} %) = {p.price:,.4g}")
        if self.pre_revenue:
            self._developer(p, capex_pct, cost_pct)
        else:
            self._producer(p, cost_pct)
        self._finish(p)
        return p

    def _producer(self, p: Point, cost_pct: float) -> None:
        if self.prod is None:
            p.steps.append("DATA_MISSING: produktion saknas")
            return
        p.revenue_musd = self.prod * p.price / 1e6
        p.steps.append(f"intäkt = {self.prod:,.4g} {self.unit} × {p.price:,.4g} = {p.revenue_musd:,.0f} MUSD")
        if self.royalty:
            if self.margin_pct is None:
                p.steps.append("DATA_MISSING: EBITDA-marginal saknas (royalty)")
                return
            fixed = self.prod * self.price / 1e6 * (1 - self.margin_pct / 100) * (1 + cost_pct / 100)
            p.ebitda_musd = p.revenue_musd - fixed
            p.steps.append(f"EBITDA = intäkt − fasta kostnader {fixed:,.0f} MUSD = {p.ebitda_musd:,.0f} MUSD")
        else:
            if self.aisc is None:
                p.steps.append("DATA_MISSING: AISC saknas")
                return
            aisc = self.aisc * (1 + cost_pct / 100)
            p.ebitda_musd = self.prod * (p.price - aisc) / 1e6
            p.steps.append(f"EBITDA ≈ {self.prod:,.4g} × ({p.price:,.4g} − AISC {aisc:,.4g}"
                           + (f" [+{cost_pct:g} %]" if cost_pct else "") + f") = {p.ebitda_musd:,.0f} MUSD")
        p.fcf_musd = p.ebitda_musd * (1 - self.tax) if p.ebitda_musd > 0 else p.ebitda_musd
        p.steps.append(f"FCF = EBITDA × (1 − skatt {self.tax * 100:g} %) = {p.fcf_musd:,.0f} MUSD")
        p.value_musd = p.ebitda_musd * self.mult
        p.equity_musd = p.value_musd - self.debt + self.cash
        p.steps.append(f"EV = EBITDA × {self.mult:g} = {p.value_musd:,.0f}; equity = EV − skuld {self.debt:g} "
                       f"+ kassa {self.cash:g} = {p.equity_musd:,.0f} MUSD")

    def _developer(self, p: Point, capex_pct: float, cost_pct: float) -> None:
        if self.prod is not None:
            p.revenue_musd = self.prod * p.price / 1e6
            p.steps.append(f"steady state-intäkt = {self.prod:,.4g} {self.unit} × {p.price:,.4g} = "
                           f"{p.revenue_musd:,.0f} MUSD")
            if self.be is not None:
                be = self.be * (1 + cost_pct / 100)
                p.ebitda_musd = self.prod * (p.price - be) / 1e6
                p.fcf_musd = p.ebitda_musd * (1 - self.tax) if p.ebitda_musd > 0 else p.ebitda_musd
                p.steps.append(f"marginal ≈ {self.prod:,.4g} × ({p.price:,.4g} − break-even {be:,.4g}) = "
                               f"{p.ebitda_musd:,.0f} MUSD; FCF = {p.fcf_musd:,.0f} MUSD")
        if self.npv is None:
            p.steps.append("DATA_MISSING: NPV saknas — scenarier kräver minst en PEA")
            return
        if self.slope is None:
            p.steps.append(f"DATA_MISSING: kan inte flytta NAV med priset ({self.slope_text})")
            return
        nav = self.npv + self.slope * p.price_pct
        p.steps.append(f"NAV = NPV {self.npv:,.0f} + {self.slope:,.1f} MUSD per % × {p.price_pct:+g} % = "
                       f"{nav:,.0f} MUSD ({self.slope_text})")
        if abs(p.price_pct) > abs(cfg.STRESS_PRICE_PCT):
            p.steps.append("OBS: linjär extrapolation utanför FS-känslighetens intervall")
        if capex_pct and self.capex is not None:
            nav -= self.capex * capex_pct / 100
            p.steps.append(f"CapEx {capex_pct:+g} % → NAV − {self.capex * capex_pct / 100:,.0f} MUSD = "
                           f"{nav:,.0f} MUSD (odiskonterat, konservativt)")
        elif capex_pct:
            p.steps.append("DATA_MISSING: capex saknas — capex-stressen kan inte räknas")
            return
        if cost_pct and self.be is None:
            p.steps.append("OBS: kostnadsstress utan break-even påverkar inte NAV")
        p.value_musd = nav
        if self.irr is not None and self.npv > 0:
            p.irr_pct = self.irr * max(nav, 0.0) / self.npv
            p.steps.append(f"IRR ≈ {self.irr:g} % × NAV/NPV = {p.irr_pct:.1f} % (MODELLED, proportionell skalning)")
        p.equity_musd = max(nav, 0.0) * self.pnav + self.cash - self.debt
        p.steps.append(f"equity = max(NAV, 0) × P/NAV {self.pnav:g} + kassa {self.cash:g} − skuld {self.debt:g} "
                       f"= {p.equity_musd:,.0f} MUSD")

    def _finish(self, p: Point) -> None:
        if p.revenue_musd and p.ebitda_musd is not None and p.revenue_musd > 0:
            p.margin_pct = p.ebitda_musd / p.revenue_musd * 100
        if p.equity_musd is None:
            return
        if self.shares:
            p.share_price = p.equity_musd / self.shares
            p.steps.append(f"aktie = {p.equity_musd:,.0f} / {self.shares:g} M = {p.share_price:,.3g}")
        if self.mcap:
            p.upside_pct = (p.equity_musd / self.mcap - 1) * 100
            p.steps.append(f"mot börsvärde {self.mcap:,.0f} MUSD: {p.upside_pct:+.0f} %")
        else:
            p.steps.append("DATA_MISSING: börsvärde saknas — ingen upside")
        for attr in ("price", "revenue_musd", "ebitda_musd", "fcf_musd", "margin_pct", "value_musd", "irr_pct",
                     "equity_musd", "share_price", "upside_pct"):
            v = getattr(p, attr)
            if v is not None:
                setattr(p, attr, round(v, 6))


def pct_change(new: Optional[float], base: Optional[float]) -> Optional[float]:
    """% förändring, None när basen är noll/negativ eller något saknas."""
    if new is None or base is None or base <= 0:
        return None
    return round((new / base - 1) * 100, 6)


def step_table(value: Optional[float], table, default: float = 0.0, reverse: bool = False) -> Optional[float]:
    """Första raden (gräns, poäng) vars gräns är uppfylld: value ≥ gräns
    (reverse=False) eller value ≤ gräns. None när value saknas."""
    if value is None:
        return None
    for floor, pts in table:
        if (value <= floor) if reverse else (value >= floor):
            return pts
    return default
