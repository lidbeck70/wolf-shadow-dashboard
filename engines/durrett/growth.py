"""
engines/durrett/growth.py — STEP 5 projected growth (SPEC §9).

Production Growth Multiple = framtida / nuvarande produktion; CAGR över
åren dit; resurs-/reservtillväxt mot 3 år sedan; pipeline (expansion,
nya fyndigheter, tillstånd) ur bedömningsfält. Pre-revenue-bolag utan
produktion nu får multipeln N/A — framtida produktion syns i Upside.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from engines.durrett._base import Ctx, score_from, steps_ge
from engines.durrett.models import Score


def growth_multiple(current: Optional[float], future: Optional[float]) -> Optional[float]:
    if current is None or future is None or current <= 0:
        return None
    return future / current


def cagr_pct(current: Optional[float], future: Optional[float], years: Optional[float]) -> Optional[float]:
    if current is None or future is None or current <= 0 or future <= 0 or years is None or years <= 0:
        return None
    return ((future / current) ** (1.0 / years) - 1.0) * 100.0


def score(ctx: Ctx, res: dict, today: Optional[date] = None) -> Score:
    st = ctx.cfg["score_steps"]
    today = today or date.today()
    comps: dict = {}
    cur = ctx.quantity("production_current", required=False)
    fut = ctx.quantity("production_future", required=False)
    yr = ctx.num("production_future_year", required=False)
    gm = growth_multiple(cur, fut)
    years = (yr - today.year) if yr is not None else None
    cg = cagr_pct(cur, fut, years)
    if gm is not None:
        ctx.metric("production_growth_multiple", gm, "×", f"{fut:g} / {cur:g}")
        if cg is not None:
            ctx.metric("production_cagr", cg, "%", f"över {years:g} år")
        comps["production"] = (steps_ge(gm, st["growth_multiple"]),
                               f"produktion {cur:,.0f} → {fut:,.0f} = {gm:.1f}×" + (f" ({cg:.0f} % CAGR till {int(yr)})" if cg is not None else ""))
    elif fut is not None and cur is None:
        comps["production"] = (None, f"framtida produktion {fut:,.0f} men ingen nu — multipeln odefinierad (se Upside)")
    else:
        prev = ctx.quantity("production_3y_ago", required=False)
        hist = cagr_pct(prev, cur, 3) if cur is not None and prev else None
        if hist is not None:
            comps["production"] = (steps_ge(hist, st["production_cagr_pct"]), f"historisk CAGR 3 år {hist:.0f} %")
        elif cur is not None and cur <= 0:
            comps["production"] = (None, "produktion nu är 0 — tillväxtmultipeln odefinierad")
        else:
            comps["production"] = (None, "produktion nu/framtid saknas")

    f = res["attributable_factor"]
    for key, base, name in (("resource_3y_ago", "resource_total", "resource"), ("reserve_3y_ago", "reserve_total", "reserve")):
        then = ctx.quantity(key, required=False)
        now = res.get(base)
        if then and now is not None:
            g = (now / then - 1) * 100
            ctx.metric(f"{name}_growth_3y", g, "%")
            comps[name] = (steps_ge(g, st["resource_growth_pct"], 30), f"{name} {then:,.0f} → {now:,.0f} ({g:+.0f} % på 3 år)")
        else:
            comps[name] = (None, f"{name}-historik saknas")
    _ = f

    exp = ctx.num("res_expansion", required=False)               # 0–3
    exp_capex = ctx.num("expansion_capex_musd", required=False)
    zones = ctx.truth("multiple_zones", required=False)
    parts, val = [], None
    if exp is not None:
        val = exp / 3 * 100
        parts.append(f"expansionspotential {exp:g}/3")
    if exp_capex is not None:
        parts.append(f"expansions-capex {exp_capex:g} MUSD planerad")
        val = (val if val is not None else 50.0) + 10
    if zones is not None:
        parts.append("flera zoner" if zones else "en zon")
        val = (val if val is not None else 50.0) + (10 if zones else -5)
    comps["pipeline"] = (None if val is None else round(min(100.0, val), 1), ", ".join(parts) or "pipeline okänd")
    return score_from("growth", "Growth", comps, ctx.cfg["sub_weights"]["growth"], ctx, min_coverage=0.4)
