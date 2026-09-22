"""
engines/durrett/explorer.py — DURRETT EXPLORER ENGINE (SPEC §17–19).

Två huvudtyper: OPTIONALITY PLAY (metall i marken × implicit värde per
enhet mot EV) och DISCOVERY PLAY (storlek, halt, bredd, kontinuitet,
zoner, step-outs, geologisk modell, borrtäthet). Positionerar bolaget på
Lassonde-kurvan. Ingen ny värderingsmodell — bara det som går att räkna
ur angivna tal; resten N/A.
"""

from __future__ import annotations

from engines.durrett import config as dc
from engines.durrett._base import Ctx, scale_0_2, steps_le


def lassonde_position(ctx: Ctx) -> tuple:
    """(steg, källa) — lassonde_stage-fältet, annars mognad → Lassonde."""
    ls = ctx.text("lassonde_stage")
    if ls in dc.LASSONDE_STAGES:
        return ls, "lassonde_stage"
    mapped = dc.MATURITY_TO_LASSONDE.get(ctx.c.maturity)
    if mapped:
        return mapped, f"härledd ur mognad {ctx.c.maturity}"
    return "exploration", "okänd — antas exploration"


def optionality(ctx: Ctx, ss: dict, res: dict) -> dict:
    """Metall i marken × implicit värde/enhet mot EV → Optionality Score 0–100."""
    qty = res.get("resource_total")
    if qty is None:
        qty = res.get("reserve_total")
    f = res.get("attributable_factor", 1.0)
    ev = ss.get("ev_usd")
    ipu = ctx.num("implied_value_per_unit", required=False)
    out = {"metal_in_ground": qty * f if qty is not None else None, "unit": res.get("unit"),
           "ev_usd": ev, "ev_per_unit": None, "implied_value_musd": None, "implied_vs_ev": None,
           "score": None, "reason": ""}
    if qty is None or ev is None:
        out["reason"] = "resurs eller EV saknas"
        return out
    out["ev_per_unit"] = ev * 1e6 / (qty * f) if qty * f > 0 else None
    cheap = ctx.cfg["explorer"]["optionality_ev_per_unit_cheap"].get(ctx.price_unit)
    exp = ctx.cfg["explorer"]["optionality_ev_per_unit_expensive"].get(ctx.price_unit)
    if ipu is not None:
        out["implied_value_musd"] = qty * f * ipu / 1e6
        out["implied_vs_ev"] = out["implied_value_musd"] / ev if ev > 0 else None
        r = out["implied_vs_ev"]
        out["score"] = None if r is None else float(min(100.0, max(0.0, 50.0 + (r - 1.0) * 25.0)))
        out["reason"] = f"implicit värde {out['implied_value_musd']:,.0f} MUSD ({ipu:g} USD/{out['unit']}) mot EV {ev:,.0f}"
    elif cheap is not None and exp is not None and out["ev_per_unit"] is not None:
        v = out["ev_per_unit"]
        out["score"] = float(steps_le(v, ((cheap, 100), (cheap * 2, 80), ((cheap + exp) / 2, 60), (exp, 40)), 15))
        out["reason"] = f"EV {v:,.2f} USD/{out['unit']} i marken (billigt ≤ {cheap:g}, dyrt ≥ {exp:g}, VAL) — implicit värde/enhet ej angivet"
    else:
        out["reason"] = f"ingen trappa för {ctx.price_unit} och inget implicit värde angivet (TODO)"
    return out


def discovery(ctx: Ctx) -> dict:
    """Discovery Score 0–100 ur borresultat och geologi (bedömningar 0–2 + intercept)."""
    parts, vals = [], []
    ic = ctx.num("best_intercept_gram_m", required=False)
    if ic is not None:
        thr = ctx.cfg["explorer"]["discovery_play_min_intercept"]
        vals.append(min(100.0, ic / thr * 50.0))
        parts.append(f"bästa intercept {ic:g} (halt×m; discovery-tröskel {thr:g})")
    width = ctx.num("intercept_width_m", required=False)
    if width is not None:
        vals.append(float(steps_le(-width, ((-20.0, 90), (-10.0, 70), (-5.0, 50), (-2.0, 30)), 15)))
        parts.append(f"bredd {width:g} m")
    for key, label in (("continuity", "kontinuitet"), ("step_out_success", "step-outs"), ("geological_model", "geologisk modell")):
        v = ctx.num(key, required=False)
        if v is not None:
            vals.append(scale_0_2(v))
            parts.append(f"{label} {int(v)}/2")
    zones = ctx.truth("multiple_zones", required=False)
    if zones is not None:
        vals.append(80.0 if zones else 40.0)
        parts.append("flera zoner" if zones else "en zon")
    holes = ctx.num("drill_holes", required=False)
    if holes is not None:
        vals.append(float(steps_le(-holes, ((-100.0, 90), (-40.0, 70), (-15.0, 50), (-5.0, 30)), 15)))
        parts.append(f"{int(holes)} borrhål")
    score = round(sum(vals) / len(vals), 1) if vals else None
    return {"score": score, "reason": ", ".join(parts) or "inga borrresultat/geologiska bedömningar",
            "is_discovery_play": ic is not None and ic >= ctx.cfg["explorer"]["discovery_play_min_intercept"]}


def profile(ctx: Ctx, ss: dict, res: dict) -> dict:
    stage, stage_src = lassonde_position(ctx)
    opt = optionality(ctx, ss, res)
    disc = discovery(ctx)
    if disc["is_discovery_play"]:
        play = "DISCOVERY PLAY"
    elif opt["score"] is not None:
        play = "OPTIONALITY PLAY"
    else:
        play = "UNKNOWN"
    land = ctx.num("land_package_km2", required=False)
    return {"play": play, "lassonde": stage, "lassonde_label": dc.LASSONDE_LABEL.get(stage, stage),
            "lassonde_source": stage_src, "optionality": opt, "discovery": disc,
            "land_package_km2": land,
            "optionality_score": opt["score"], "discovery_score": disc["score"]}
