"""
confidence/scoring/valuation.py — Valuation 10 p, stage-routad (SPEC).

Producenter/royalty: EV/EBITDA 3, FCF-yield 3, P/E 1, EV/EBIT 1, P/NAV 2.
Developers/explorers: P/NAV mot config.P_NAV_TABLE (< 0,30 → 10 … > 1,50 → 0).
P/NAV läses ur fältet p_nav, annars börsvärde / NAV.
"""

from __future__ import annotations

from typing import Optional

from confidence import config as cfg
from confidence.data.models import CompanyInput, PillarScore
from confidence.scoring._steps import finish, note, read, src, step_ge, step_le, table_lt


def p_nav(company: CompanyInput, p: PillarScore) -> Optional[float]:
    v = company.num("p_nav") if company.has("p_nav") else None
    if v is not None:
        return v
    mc = read(company, "market_cap_musd", p)
    nav = read(company, "nav_musd", p)
    if mc is not None and nav is not None and nav > 0:
        return mc / nav
    if "p_nav" not in p.missing:
        p.missing.append("p_nav")
    return None


def _pnav_src(company: CompanyInput) -> str:
    if company.has("p_nav"):
        return src(company, "p_nav")
    return f"{src(company, 'market_cap_musd', 'börsvärde')} / {src(company, 'nav_musd', 'NAV')}"


def score(company: CompanyInput) -> PillarScore:
    p = PillarScore("valuation", "Valuation", 0.0, cfg.PILLAR_MAX["valuation"])
    if company.stage in cfg.PRE_REVENUE:
        pn = p_nav(company, p)
        note(p, "P/NAV", table_lt(pn, cfg.P_NAV_TABLE, cfg.P_NAV_BEYOND), p.max,
             "DATA_MISSING" if pn is None else f"{pn:.2f}× · {_pnav_src(company)}")
        return finish(p)

    for key, name, steps, mx in (("ev_ebitda", "EV/EBITDA", cfg.EV_EBITDA_STEPS, 3),
                                 ("pe", "P/E", cfg.PE_STEPS, 1),
                                 ("ev_ebit", "EV/EBIT", cfg.EV_EBIT_STEPS, 1)):
        v = read(company, key, p)
        pts = 0.0 if v is None or v <= 0 else step_le(v, steps)
        note(p, name, pts, mx, "DATA_MISSING" if v is None else f"{v:.1f}× · {src(company, key)}")
    fy = read(company, "fcf_yield_pct", p)
    note(p, "FCF-yield", step_ge(fy, cfg.VAL_FCF_YIELD_STEPS), 3,
         "DATA_MISSING" if fy is None else f"{fy:.1f} % · {src(company, 'fcf_yield_pct')}")
    pn = p_nav(company, p)
    note(p, "P/NAV", step_le(pn, cfg.NAV_PROD_STEPS) if pn else 0, 2,
         "DATA_MISSING" if pn is None else f"{pn:.2f}× · {_pnav_src(company)}")
    return finish(p)
