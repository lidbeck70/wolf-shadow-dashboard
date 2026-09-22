"""
engines/durrett/financing.py — STEP 7b finansieringsrisk (SPEC §11–12).

Funding coverage = (kassa + åtagen finansiering) / CapEx. Kravet ställs
mot bolagets storlek: stort projekt + liten kassa + litet bolag = hög
finansieringsrisk. Partners/offtake sänker risken. Funding cliff när
finansiering krävs före nästa milstolpe.
"""

from __future__ import annotations

from typing import Optional

from engines.durrett import config as dc
from engines.durrett._base import Ctx, score_from, steps_ge, steps_le
from engines.durrett.models import Score


def coverage(cash: Optional[float], committed: Optional[float], capex: Optional[float]) -> Optional[float]:
    if capex is None or capex <= 0:
        return None
    return ((cash or 0.0) + (committed or 0.0)) / capex


def funding_cliff(ctx: Ctx, ss: dict) -> Optional[dict]:
    """{cliff: bool, why} — kräver finansiering före nästa milstolpe? None om okänt."""
    cash = ss.get("cash_usd")
    cost = ctx.num("milestone_cost_musd", required=False)
    ms = ctx.text("next_milestone")
    if cash is None or cost is None:
        return None
    committed = ctx.num("committed_financing_musd", required=False) or 0.0
    gap = cost - cash - committed
    return {"cliff": gap > 0, "gap_musd": gap, "milestone": ms or "nästa milstolpe",
            "why": f"kostnad till {ms or 'milstolpen'} {cost:g} MUSD mot kassa {cash:g} + åtagen {committed:g}"}


def score(ctx: Ctx, company_type: str, ss: dict) -> Score:
    st = ctx.cfg["score_steps"]
    comps: dict = {}
    cash = ss.get("cash_usd")
    mcap = ss.get("mcap_usd")
    if company_type in (dc.PRODUCER, dc.ROYALTY, dc.HYBRID):
        fcf = ctx.usd("free_cash_flow_musd", required=False)
        exp_capex = ctx.num("expansion_capex_musd", required=False)
        if fcf is not None:
            comps["coverage"] = (steps_ge(fcf, ((0.0, 80), (-1e18, 30)), 30) if fcf >= 0 else 30.0,
                                 f"FCF {fcf:+.0f} MUSD — {'självfinansierad' if fcf >= 0 else 'kassaflödesnegativ'}")
        elif cash is not None and mcap:
            comps["coverage"] = (steps_ge(cash / mcap * 100, ((20.0, 90), (10.0, 75), (5.0, 60), (0.0, 45))),
                                 f"kassa {cash:g} MUSD = {cash / mcap * 100:.0f} % av börsvärdet (FCF okänt)")
        else:
            comps["coverage"] = (None, "FCF och kassa saknas")
        if exp_capex is not None and cash is not None:
            cov = cash / exp_capex if exp_capex > 0 else 1.0
            comps["requirement"] = (steps_ge(cov, st["funding_coverage"]), f"expansions-capex {exp_capex:g} MUSD mot kassa {cash:g}")
        else:
            comps["requirement"] = (80.0, "ingen expansions-capex angiven") if exp_capex is None and company_type != dc.HYBRID else (None, "kassa saknas")
        comps["partners"] = (None, "gäller developers")
        return score_from("financing", "Financing", comps, {"coverage": 60, "requirement": 40}, ctx)

    capex = ctx.num("capex_musd", required=False)
    committed = ctx.num("committed_financing_musd", required=False)
    cov = coverage(cash, committed, capex)
    if cov is not None:
        ctx.metric("funding_coverage", cov, "×", f"(kassa {cash or 0:g} + åtagen {committed or 0:g}) / CapEx {capex:g}")
        comps["coverage"] = (steps_ge(cov, st["funding_coverage"]), f"täckning {cov * 100:.0f} % av CapEx")
    elif company_type == dc.EXPLORER and capex is None:
        comps["coverage"] = (None, "ingen CapEx än (explorer)")
    else:
        comps["coverage"] = (None, "CapEx eller kassa saknas")
    if capex is not None and mcap:
        ratio = capex / mcap
        comps["requirement"] = (steps_le(ratio, st["capex_to_mcap"], ctx.cfg["score_beyond"]["capex_to_mcap"]),
                                f"CapEx {capex:g} MUSD = {ratio:.1f}× börsvärdet — " +
                                ("hög finansieringsrisk" if ratio >= ctx.cfg["red_flag_thresholds"]["capex_to_mcap_large"] else "hanterbart"))
    elif company_type == dc.EXPLORER:
        burn = ctx.num("quarterly_burn_musd", required=False)
        if cash is not None and burn:
            q = cash / burn
            comps["requirement"] = (steps_ge(q / 4, st["runway_years"]), f"runway {q:.1f} kvartal — nästa emission")
        else:
            comps["requirement"] = (None, "kassa/burn saknas")
    else:
        comps["requirement"] = (None, "CapEx eller börsvärde saknas")
    partner = ctx.truth("has_strategic_partner", required=False)
    offtake = ctx.truth("has_offtake", required=False)
    if partner is None and offtake is None:
        comps["partners"] = (None, "partner/offtake okänt")
    else:
        v = 40.0 + (30.0 if partner else 0.0) + (30.0 if offtake else 0.0)
        comps["partners"] = (v, ", ".join(x for x in (("strategisk partner" if partner else None),
                                                      ("offtake" if offtake else None)) if x) or "varken partner eller offtake")
    s = score_from("financing", "Financing", comps, ctx.cfg["sub_weights"]["financing"], ctx, min_coverage=0.4)
    cliff = funding_cliff(ctx, ss)
    if cliff and cliff["cliff"]:
        s.negative.append(f"− FUNDING CLIFF: {cliff['why']} (gap {cliff['gap_musd']:g} MUSD)")
        ctx.metric("funding_cliff_gap", cliff["gap_musd"], "MUSD", cliff["milestone"])
    return s
