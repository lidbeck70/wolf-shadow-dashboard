"""
engines/durrett/upside.py — STEP 10 upside / rating (SPEC §15).

Upside Multiple = uppskattat framtida börsvärde / nuvarande FD-börsvärde.
Framtida börsvärde = framtida EV (produktion × (pris − AISC) × multipel)
− nettoskuld. Visar Upside %, multipel, potentiellt och nuvarande
börsvärde. Upside är skilt från Quality: hög kvalitet kan ha låg upside
och tvärtom.
"""

from __future__ import annotations

from typing import Optional

from engines.durrett._base import Ctx, score_from, steps_ge
from engines.durrett.models import Score


def upside_multiple(future_mcap: Optional[float], current_fd_mcap: Optional[float]) -> Optional[float]:
    if future_mcap is None or current_fd_mcap is None or current_fd_mcap <= 0:
        return None
    return future_mcap / current_fd_mcap


def compute(ctx: Ctx, ss: dict, fe: dict) -> dict:
    cur = ss.get("fd_mcap_usd") or ss.get("mcap_usd")
    basis = "FD-börsvärde" if ss.get("fd_mcap_usd") else "börsvärde (FD okänt)"
    fut = fe.get("future_mcap")
    mult = upside_multiple(fut, cur)
    out = {"current_mcap": cur, "current_basis": basis, "potential_mcap": fut, "upside_multiple": mult,
           "upside_pct": (mult - 1) * 100 if mult is not None else None}
    if mult is not None:
        ctx.metric("upside_multiple", mult, "×", f"potentiellt {fut:,.0f} / {basis} {cur:,.0f} MUSD")
        ctx.metric("upside_pct", out["upside_pct"], "%")
    return out


def score(ctx: Ctx, up: dict) -> Score:
    st = ctx.cfg["score_steps"]
    comps: dict = {}
    m = up.get("upside_multiple")
    if m is not None:
        comps["upside_multiple"] = (steps_ge(m, st["upside_multiple"]),
                                    f"{m:.1f}× ({up['upside_pct']:+.0f} %) — potentiellt {up['potential_mcap']:,.0f} mot {up['current_basis']} {up['current_mcap']:,.0f} MUSD")
    else:
        comps["upside_multiple"] = (None, "framtida börsvärde eller nuvarande FD-börsvärde saknas")
    disc = ctx.num("discovery_option", required=False)
    if disc is not None:
        comps["optionality"] = (disc / 5 * 100, f"Discovery Option {disc:g}/5")
    else:
        comps["optionality"] = (None, "discovery option ej bedömd")
    return score_from("upside", "Upside", comps, {"upside_multiple": 80, "optionality": 20}, ctx, min_coverage=0.2)
