"""
confidence/scoring/case.py — Case Score 0–100: åtta pelare → summa → betyg.

case_score(company, commodity, today) räknar varje pelare, summerar, sätter
betyg enligt config.RATING_BANDS och samlar alla DATA_MISSING och flaggor.
Confidence Score påverkar aldrig detta tal (SPEC).
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from confidence import commodities as com
from confidence import config as cfg
from confidence.data.models import CaseScore, CompanyInput
from confidence.scoring import (balance, economics, growth, management, resource, scarcity,
                                strategic, valuation)


def rating_for(total: float) -> str:
    for floor, label in cfg.RATING_BANDS:
        if total >= floor:
            return label
    return cfg.RATING_BANDS[-1][1]


def case_score(company: CompanyInput, commodity: Optional[com.Commodity] = None,
               today: Optional[date] = None, commodity_overrides: Optional[dict] = None) -> CaseScore:
    """Alla pelare för ett bolag. commodity kan ges direkt, annars slås
    company.commodity upp i registret (med ev. överlagringar)."""
    if commodity is None:
        commodity = com.get(company.commodity, commodity_overrides)
    pillars = [
        strategic.score(company, commodity),
        resource.score(company),
        scarcity.score(commodity),
        economics.score(company),
        growth.score(company, today),
        balance.score(company),
        valuation.score(company),
        management.score(company),
    ]
    assert [p.key for p in pillars] == [k for k, _l, _m in cfg.PILLARS]
    total = round(sum(p.points for p in pillars), 1)
    missing: list = []
    for p in pillars:
        for k in p.missing:
            if k not in missing:
                missing.append(k)
    flags = [f"{p.label}: {name} → tak {cap_v:g} p" for p in pillars for name, cap_v in p.caps]
    if commodity is None:
        flags.append(f"råvara {company.commodity!r} okänd — Strategic Commodity och Demand & Scarcity 0 p")
    if missing:
        flags.append(f"{len(missing)} fält saknas (DATA_MISSING) — poängen är ett golv, inte ett betyg")
    return CaseScore(total=total, rating=rating_for(total), pillars=pillars,
                     discovery_option=growth.discovery_option(company), missing=missing, flags=flags)
