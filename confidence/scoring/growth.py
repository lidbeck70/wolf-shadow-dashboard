"""
confidence/scoring/growth.py — Production & Growth 10 p (SPEC).

Time-to-money = år till första kassaflöde, läst mot config.TIME_TO_MONEY_TABLE.
Producenter och royaltybolag har kassaflöde nu (0 år) om inget annat anges.
Discovery Option 0–5 (explorers) hålls separat från huvudscoren.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from confidence import config as cfg
from confidence.data.models import CompanyInput, PillarScore
from confidence.scoring._steps import finish, note, read, table_lt


def years_to_money(company: CompanyInput, today: Optional[date] = None) -> Optional[float]:
    """År till första kassaflöde; None = DATA_MISSING (pre-revenue utan årtal)."""
    today = today or date.today()
    yr = company.num("first_cashflow_year")
    if yr is not None:
        return max(0.0, yr - today.year)
    if company.stage not in cfg.PRE_REVENUE:
        return 0.0
    return None


def score(company: CompanyInput, today: Optional[date] = None) -> PillarScore:
    p = PillarScore("production_growth", "Production & Growth", 0.0, cfg.PILLAR_MAX["production_growth"])
    yrs = years_to_money(company, today)
    if yrs is None:
        p.missing.append("first_cashflow_year")
        note(p, "Time-to-money", 0, p.max, "DATA_MISSING (första kassaflöde-år saknas)")
    else:
        if company.has("first_cashflow_year"):
            pt = company.get("first_cashflow_year")
            why = f"första kassaflöde {int(company.num('first_cashflow_year'))} ({pt.kind}" + \
                  (f", {pt.source}" if pt.source else "") + ")"
        else:
            why = f"{company.stage}: kassaflöde nu"
        note(p, "Time-to-money", table_lt(yrs, cfg.TIME_TO_MONEY_TABLE, cfg.TIME_TO_MONEY_BEYOND), p.max,
             f"{yrs:g} år · {why}")
    return finish(p)


def discovery_option(company: CompanyInput) -> Optional[float]:
    """Explorers: 0–5 separat. None när det inte gäller eller saknas."""
    if company.stage != "explorer":
        return None
    scratch = PillarScore("discovery", "Discovery Option", 0.0, cfg.DISCOVERY_OPTION_MAX)
    v = read(company, "discovery_option", scratch)
    return None if v is None else min(v, float(cfg.DISCOVERY_OPTION_MAX))
