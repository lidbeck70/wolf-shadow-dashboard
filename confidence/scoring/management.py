"""
confidence/scoring/management.py — Management 5 p (SPEC listar bedömningarna;
VAL: fem delar à 1 p). Byggmeriter, kapitalallokering, utspädningshistorik,
insynsägande (≥ 10 % → 1, ≥ 3 % → 0,5), alignment & leverans mot löften.
"""

from __future__ import annotations

from confidence import config as cfg
from confidence.data.models import CompanyInput, PillarScore
from confidence.scoring._steps import finish, note, read, src, step_ge


def score(company: CompanyInput) -> PillarScore:
    p = PillarScore("management", "Management", 0.0, cfg.PILLAR_MAX["management"])
    for key, mx in cfg.MANAGEMENT_SUB.items():
        label = cfg.FIELD_BY_KEY[key].label if key in cfg.FIELD_BY_KEY else "Insynsägande"
        if key == "mgmt_insider_ownership":
            v = read(company, "insider_ownership_pct", p)
            note(p, label, step_ge(v, cfg.INSIDER_OWNERSHIP_STEPS), mx,
                 "DATA_MISSING" if v is None else f"{v:g} % · {src(company, 'insider_ownership_pct')}")
            continue
        v = read(company, key, p)
        note(p, label, 0 if v is None else min(v, float(mx)), mx,
             "DATA_MISSING" if v is None else src(company, key))
    return finish(p)
