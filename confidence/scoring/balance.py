"""
confidence/scoring/balance.py — Balance Sheet & Financing 10 p, stage-routad (SPEC).

Producenter/royalty: nettoskuld/EBITDA mot config.ND_EBITDA_TABLE.
Developers: finansieringsgap 4 (CapEx − kassa − åtagen, andel av CapEx),
runway 3 (kassa / kvartalsburn), åtaganden 3 (partner, offtake, åtagen
finansiering ≥ 25 % av CapEx). Explorers: gap och åtagen finansiering
gäller inte (ingen CapEx) — skalas pro rata. DS ur kontrollerna drar
1–2 p (config.DILUTION_PENALTY_STEPS).
"""

from __future__ import annotations

from confidence import config as cfg
from confidence.data.models import CompanyInput, PillarScore
from confidence.scoring._steps import (finish, note, read, read_bool, src, step_ge, step_le,
                                       table_lt)


def score(company: CompanyInput) -> PillarScore:
    p = PillarScore("balance_sheet", "Balance Sheet & Financing", 0.0, cfg.PILLAR_MAX["balance_sheet"])
    if company.stage in cfg.PRE_REVENUE:
        applicable = _developer(company, p)
    else:
        applicable = None
        nd = read(company, "net_debt_ebitda", p)
        note(p, "Nettoskuld/EBITDA", table_lt(nd, cfg.ND_EBITDA_TABLE, cfg.ND_EBITDA_BEYOND), p.max,
             "DATA_MISSING" if nd is None else f"{nd:.2f}× · {src(company, 'net_debt_ebitda')}")
    ds = read(company, "dilution_score", p) if company.has("dilution_score") else None   # frivilligt
    if ds is not None:
        pen = step_ge(ds, cfg.DILUTION_PENALTY_STEPS)
        if pen:
            p.components["Utspädning (DS)"] = -pen
            p.notes.append(f"Utspädning (DS) −{pen:g} p — DS {ds:g}/10 · {src(company, 'dilution_score')}")
    return finish(p, applicable)


def _developer(company: CompanyInput, p: PillarScore) -> float:
    gap_max = cfg.FUNDING_GAP_STEPS[0][1]
    runway_max = cfg.RUNWAY_STEPS[0][1]
    applicable = float(p.max)
    cash = read(company, "cash_musd", p)
    burn = read(company, "quarterly_burn_musd", p)
    committed = company.num("committed_financing_musd") if company.has("committed_financing_musd") else None
    capex = read(company, "capex_musd", p) if company.stage == "developer" else None

    if company.stage == "explorer":
        applicable -= gap_max + cfg.COMMITMENT_POINTS["committed_financing_ok"]
        p.notes.append("explorer: finansieringsgap och åtagen finansiering gäller inte (ingen CapEx)")
    elif capex is not None and capex > 0 and cash is not None:
        gap = max(0.0, capex - cash - (committed or 0.0)) / capex
        note(p, "Finansieringsgap", step_le(gap, cfg.FUNDING_GAP_STEPS), gap_max,
             f"{gap * 100:.0f} % av CapEx ({capex:g} − kassa {cash:g} − åtagen {committed or 0:g})")
        ok = committed is not None and committed / capex >= cfg.COMMITTED_FINANCING_MIN_SHARE
        note(p, "Åtagen finansiering", cfg.COMMITMENT_POINTS["committed_financing_ok"] if ok else 0,
             cfg.COMMITMENT_POINTS["committed_financing_ok"],
             f"{(committed or 0) / capex * 100:.0f} % av CapEx (krav ≥ {cfg.COMMITTED_FINANCING_MIN_SHARE * 100:g} %)")
    else:
        note(p, "Finansieringsgap", 0, gap_max, "DATA_MISSING")
        note(p, "Åtagen finansiering", 0, cfg.COMMITMENT_POINTS["committed_financing_ok"], "DATA_MISSING")

    if cash is not None and burn is not None and burn > 0:
        q = cash / burn
        note(p, "Runway", step_ge(q, cfg.RUNWAY_STEPS), runway_max, f"{q:.1f} kvartal (kassa {cash:g} / burn {burn:g})")
    elif cash is not None and burn == 0:
        note(p, "Runway", runway_max, runway_max, "ingen burn rapporterad")
    else:
        note(p, "Runway", 0, runway_max, "DATA_MISSING")

    for key, name in (("has_strategic_partner", "Strategisk partner"), ("has_offtake", "Offtake")):
        b = read_bool(company, key, p)
        note(p, name, cfg.COMMITMENT_POINTS[key] if b else 0, cfg.COMMITMENT_POINTS[key],
             "DATA_MISSING" if b is None else src(company, key))
    return applicable
