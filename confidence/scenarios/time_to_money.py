"""
confidence/scenarios/time_to_money.py — steg kvar till första kassaflöde,
förväntad tid och konfidens (SPEC).

Stegen kommer ur config.TTM_STAGE_YEARS (branschtypiska år per mognads-
steg, VAL). Bolagets eget årtal (first_cashflow_year) jämförs med det
typiska: en plan under 60 % av typisk tid flaggas som aggressiv och drar
konfidensen till "låg". Producenter/royalty: 0 år, hög konfidens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from confidence import config as cfg
from confidence.data.models import CompanyInput
from confidence.scoring.growth import years_to_money

_RANK = {"låg": 0, "medel": 1, "hög": 2}


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    years: float
    confidence: str


@dataclass
class TimeToMoney:
    stages: list                             # [Stage] kvar
    typical_years: float                     # summa av stegen
    company_years: Optional[float]           # ur first_cashflow_year, None = saknas
    years: float                             # det som används (bolagets om det finns)
    confidence: str                          # låg | medel | hög
    basis: str                               # "bolagets plan" | "MODELLED (stage-typiskt)" | "i produktion"
    notes: list = field(default_factory=list)
    flags: list = field(default_factory=list)


def remaining_stages(maturity: str) -> list:
    keys = [k for k, _l, _y, _c in cfg.TTM_STAGE_YEARS]
    if maturity not in keys:
        return []
    return [Stage(k, lbl, y, c) for k, lbl, y, c in cfg.TTM_STAGE_YEARS[keys.index(maturity):]]


def time_to_money(company: CompanyInput, today: Optional[date] = None) -> TimeToMoney:
    if company.stage not in cfg.PRE_REVENUE or company.maturity == "production":
        return TimeToMoney([], 0.0, years_to_money(company, today), 0.0, "hög", "i produktion",
                           notes=[f"{company.stage}: kassaflöde nu"])
    stages = remaining_stages(company.maturity)
    typical = sum(s.years for s in stages)
    conf = min((s.confidence for s in stages), key=lambda c: _RANK[c]) if stages else "låg"
    company_years = years_to_money(company, today)
    notes = [f"{s.label}: ~{s.years:g} år ({s.confidence} konfidens)" for s in stages]
    flags: list = []
    if company_years is None:
        notes.append(f"first_cashflow_year saknas → stage-typiskt {typical:g} år (MODELLED)")
        return TimeToMoney(stages, typical, None, typical, conf, "MODELLED (stage-typiskt)", notes, flags)
    if typical > 0 and company_years < typical * cfg.TTM_AGGRESSIVE_RATIO:
        flags.append(f"aggressiv plan: bolaget säger {company_years:g} år, stage-typiskt {typical:g} år")
        conf = "låg"
    elif typical > 0 and company_years > typical * 1.5:
        notes.append(f"bolagets plan ({company_years:g} år) är långsammare än typiskt ({typical:g} år)")
    if company.truth("timeline_documented") is False:
        flags.append("tidsplanen är inte dokumenterad/finansierad")
        conf = "låg" if conf != "låg" else conf
    delays = company.num("historical_delays")
    if delays is not None and delays == 0:
        flags.append("upprepade förseningar historiskt")
        conf = "låg"
    return TimeToMoney(stages, typical, company_years, company_years, conf, "bolagets plan", notes, flags)
