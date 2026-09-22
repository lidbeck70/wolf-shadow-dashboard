"""
confidence/scoring/strategic.py — Strategic Commodity 20 p (SPEC).

Strategisk betydelse 0–10, efterfrågetillväxt 0–5, geopolitisk knapphet 0–5.
Läser råvaruregistret (seedad betydelse som ESTIMATE, resten null tills
sourcat). Saknat tal ger 0 p och DATA_MISSING — aldrig ett gissat värde.
"""

from __future__ import annotations

from typing import Optional

from confidence.commodities import Commodity
from confidence.config import PILLAR_MAX, STRATEGIC_SUB
from confidence.data.models import CompanyInput, PillarScore
from confidence.data.provenance import describe, num
from confidence.scoring._steps import finish, note

_LABEL = {"strategic_significance": "Strategisk betydelse", "demand_growth": "Efterfrågetillväxt",
          "geopolitical_scarcity": "Geopolitisk knapphet"}


def score(company: CompanyInput, commodity: Optional[Commodity]) -> PillarScore:
    p = PillarScore("strategic_commodity", "Strategic Commodity", 0.0, PILLAR_MAX["strategic_commodity"])
    if commodity is None:
        p.missing.append("commodity")
        p.notes.append(f"råvara {company.commodity!r} okänd i registret — 0 p (DATA_MISSING)")
        return finish(p)
    for key, mx in STRATEGIC_SUB.items():
        point = getattr(commodity, key)
        v = num(point)
        if v is None:
            p.missing.append(f"commodity.{key}")
            note(p, _LABEL[key], 0, mx, f"{commodity.label}: DATA_MISSING")
            continue
        pts = min(max(v, 0.0), float(mx))
        note(p, _LABEL[key], pts, mx, describe(point, commodity.label))
    return finish(p)
