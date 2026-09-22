"""
confidence/scoring/resource.py — Resource & Project Quality 15 p (SPEC).

Storlek 2, halt 2, metallurgi 1, gruvliv 1, geologi 2, infrastruktur 4,
expansion 3. Delpoängen är manuella bedömningar (AQS-förifyllda i UI:t).
Gruvliv härleds ur mine_life_years (> 10 år → 1) när delpoängen saknas.
"""

from __future__ import annotations

from confidence.config import FIELD_BY_KEY, PILLAR_MAX, RESOURCE_SUB
from confidence.data.models import CompanyInput, PillarScore
from confidence.scoring._steps import finish, note, read, src

MINE_LIFE_FULL_POINT_YEARS = 10.0     # SPEC-hint: > 10 år → 1 p


def score(company: CompanyInput) -> PillarScore:
    p = PillarScore("resource_quality", "Resource & Project Quality", 0.0, PILLAR_MAX["resource_quality"])
    for key, mx in RESOURCE_SUB.items():
        label = FIELD_BY_KEY[key].label
        v = read(company, key, p)
        if v is None and key == "res_mine_life":
            yrs = read(company, "mine_life_years", p)
            if yrs is not None:
                p.missing.remove(key)
                pts = float(mx) if yrs > MINE_LIFE_FULL_POINT_YEARS else 0.0
                note(p, label, pts, mx, f"härledd ur {src(company, 'mine_life_years', 'gruvliv')}")
                continue
        if v is None:
            note(p, label, 0, mx, "DATA_MISSING")
            continue
        note(p, label, min(v, float(mx)), mx, src(company, key))
    return finish(p)
