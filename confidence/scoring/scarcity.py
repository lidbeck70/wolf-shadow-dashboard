"""
confidence/scoring/scarcity.py — Demand & Scarcity 15 p (SPEC).

Utbudsbalans (underskott i % av efterfrågan, negativt = överskott) läses
mot config.SUPPLY_BALANCE_TABLE; varje gällande justering (substitution,
återvinning, teknikskifte, sekundärt utbud, projektförseningar) drar 1 p.
Balansen kommer ur råvaruregistrets överlagring — null → 0 p, DATA_MISSING.
"""

from __future__ import annotations

from typing import Optional

from confidence.commodities import Commodity
from confidence.config import (PILLAR_MAX, SUPPLY_ADJUSTMENTS, SUPPLY_BALANCE_MAX_POINTS,
                               SUPPLY_BALANCE_TABLE)
from confidence.data.models import PillarScore
from confidence.data.provenance import describe, num
from confidence.scoring._steps import finish, note, table_lt

ADJUSTMENT_POINTS = 1.0     # VAL — per justering
_ADJ_LABEL = {"adj_substitution": "substitution", "adj_recycling": "återvinning",
              "adj_tech_change": "teknikskifte", "adj_secondary_supply": "sekundärt utbud",
              "adj_project_delays": "projektförseningar (höjer)"}


def score(commodity: Optional[Commodity]) -> PillarScore:
    p = PillarScore("demand_scarcity", "Demand & Scarcity", 0.0, PILLAR_MAX["demand_scarcity"])
    if commodity is None:
        p.missing.append("commodity")
        p.notes.append("råvara okänd — 0 p (DATA_MISSING)")
        return finish(p)
    bal = num(commodity.supply_balance_pct)
    if bal is None:
        p.missing.append("commodity.supply_balance_pct")
        note(p, "Utbudsbalans", 0, SUPPLY_BALANCE_MAX_POINTS, f"{commodity.label}: DATA_MISSING")
        return finish(p)
    base = table_lt(bal, SUPPLY_BALANCE_TABLE, SUPPLY_BALANCE_MAX_POINTS)
    side = "underskott" if bal > 0 else "överskott"
    note(p, "Utbudsbalans", base, SUPPLY_BALANCE_MAX_POINTS,
         f"{side} {abs(bal):g} % · {describe(commodity.supply_balance_pct)}")
    adj = 0.0
    for a in commodity.adjustments:
        if a not in SUPPLY_ADJUSTMENTS:
            p.notes.append(f"okänd justering {a!r} ignoreras")
            continue
        sign = +1.0 if a == "adj_project_delays" else -1.0
        adj += sign * ADJUSTMENT_POINTS
        p.notes.append(f"justering {_ADJ_LABEL[a]}: {sign * ADJUSTMENT_POINTS:+g} p")
    if adj:
        p.components["Justeringar"] = adj
        p.notes.append(f"Justeringar {adj:+g} p — summa av raderna ovan (VAL: ±{ADJUSTMENT_POINTS:g} per justering)")
    return finish(p)
