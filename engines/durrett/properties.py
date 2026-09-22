"""
engines/durrett/properties.py — STEP 1 properties/ownership (SPEC §5).

Reserver (Proven + Probable) och resurser (Measured + Indicated + Inferred)
hålls isär och varje tal säger vilken kategori det kommer från.
Nyckeltal: NPV/CAPEX, MCap/resurs, MCap/reserv, EV/resurs, EV/reserv,
$/enhet i marken. Poängen väger projektkvalitet (NPV/CAPEX, IRR, gruvliv,
recovery, halt, infrastruktur, storlek, prospekteringsuppsida).
"""

from __future__ import annotations

from typing import Optional

from engines.durrett._base import Ctx, scale_0_2, score_from, steps_ge
from engines.durrett.models import Score

_RES_KEYS = ("resource_measured", "resource_indicated", "resource_inferred")
_REV_KEYS = ("reserve_proven", "reserve_probable")


def resources(ctx: Ctx) -> dict:
    """{reserve_total, reserve_categories, resource_total, resource_categories, mi_total, unit,
       attributable_factor} i basenhet (oz/lb/t). None när inget är angivet."""
    rev = {k: ctx.quantity(k, required=False) for k in _REV_KEYS}
    res = {k: ctx.quantity(k, required=False) for k in _RES_KEYS}
    own = ctx.num("ownership_pct", required=False)
    factor = (own / 100.0) if own is not None else 1.0
    if own is None:
        ctx.log.append("ownership_pct saknas — 100 % antas (ASSUMPTION)")

    def _sum(d):
        vals = [v for v in d.values() if v is not None]
        return sum(vals) if vals else None

    out = {
        "reserve_total": _sum(rev), "reserve_categories": [k.split("_")[1] for k, v in rev.items() if v is not None],
        "resource_total": _sum(res), "resource_categories": [k.split("_")[1] for k, v in res.items() if v is not None],
        "mi_total": _sum({k: res[k] for k in ("resource_measured", "resource_indicated")}),
        "unit": ctx.base_unit or (ctx.text("resource_unit") or ""), "attributable_factor": factor,
    }
    if out["reserve_total"] is None and out["resource_total"] is None:
        for k in ("reserve_proven", "reserve_probable", "resource_measured", "resource_indicated"):
            if k not in ctx.missing:
                ctx.missing.append(k)
    for k in ("reserve_total", "resource_total", "mi_total"):
        if out[k] is not None:
            cats = out["reserve_categories"] if k == "reserve_total" else \
                (["measured", "indicated"] if k == "mi_total" else out["resource_categories"])
            ctx.metric(k, out[k] * factor, out["unit"], "kategorier: " + "+".join(cats) +
                       (f" · {own:g} % ägarandel" if own is not None else ""))
    return out


def per_unit(value_musd: Optional[float], quantity: Optional[float]) -> Optional[float]:
    """USD per enhet i marken (MUSD × 1e6 / enheter)."""
    if value_musd is None or quantity is None or quantity <= 0:
        return None
    return value_musd * 1e6 / quantity


def valuation_metrics(ctx: Ctx, ss: dict, res: dict) -> dict:
    """MCap/EV per resurs/reserv-enhet (attributable) — SPEC §5 och §13 A/B."""
    f = res["attributable_factor"]
    out = {}
    for label, value in (("mcap", ss.get("mcap_usd")), ("ev", ss.get("ev_usd"))):
        for base, qty in (("resource", res["resource_total"]), ("reserve", res["reserve_total"]),
                          ("mi", res["mi_total"])):
            v = per_unit(value, qty * f if qty is not None else None)
            out[f"{label}_per_{base}_unit"] = v
            if v is not None:
                ctx.metric(f"{label}_per_{base}_unit", v, f"USD/{res['unit']}",
                           "attributable · " + ("+".join(res["reserve_categories"]) if base == "reserve" else
                                                ("measured+indicated" if base == "mi" else "+".join(res["resource_categories"]))))
    npv = ctx.num("npv_musd", required=False)
    capex = ctx.num("capex_musd", required=False)
    out["npv_capex"] = npv / capex if npv is not None and capex else None
    if out["npv_capex"] is not None:
        ctx.metric("npv_capex", out["npv_capex"], "×", f"NPV {npv:g} / CapEx {capex:g}")
    return out


def score(ctx: Ctx, metrics: dict, res: dict) -> Score:
    st = ctx.cfg["score_steps"]
    thr = ctx.cfg["red_flag_thresholds"]
    comps: dict = {}
    nc = metrics.get("npv_capex")
    comps["npv_capex"] = (steps_ge(nc, st["npv_capex"]), f"NPV/CapEx {nc:.2f}×" if nc is not None else "NPV eller CapEx saknas")
    irr = ctx.num("irr_pct", required=False)
    comps["irr"] = (steps_ge(irr, st["irr_pct"]), f"IRR {irr:g} %" if irr is not None else "IRR saknas")
    life = ctx.num("mine_life_years", required=False)
    comps["mine_life"] = (steps_ge(life, st["mine_life_years"]), f"gruvliv {life:g} år" if life is not None else "gruvliv saknas")
    rec = ctx.num("recovery_pct", required=False)
    comps["recovery"] = (steps_ge(rec, st["recovery_pct"]), f"recovery {rec:g} %" if rec is not None else "recovery saknas")
    grade = ctx.num("grade", required=False)
    gunit = ctx.text("grade_unit") or ""
    if grade is not None:
        low = thr["grade_low_by_unit"].get(ctx.price_unit)
        if low is not None and low > 0:
            ratio = grade / low
            comps["grade"] = (steps_ge(ratio, ((3.0, 100), (2.0, 80), (1.5, 65), (1.0, 50), (0.7, 30), (0.0, 15))),
                              f"halt {grade:g} {gunit} mot lågtröskel {low:g}")
        else:
            comps["grade"] = (None, f"halt {grade:g} {gunit} — ingen tröskel för {ctx.price_unit or 'råvaran'} (TODO)")
    else:
        comps["grade"] = (None, "halt saknas")
    infra = ctx.num("res_infrastructure", required=False)      # 0–4 ur Confidence-fältet
    comps["infrastructure"] = (None if infra is None else infra / 4 * 100,
                               f"infrastruktur {infra:g}/4" if infra is not None else "infrastruktur saknas")
    size = ctx.num("res_size", required=False)                  # 0–2
    comps["resource_size"] = (scale_0_2(size), f"storlek {size:g}/2" if size is not None else "storleksbedömning saknas")
    exp = ctx.num("res_expansion", required=False)              # 0–3
    comps["exploration_upside"] = (None if exp is None else exp / 3 * 100,
                                   f"expansion {exp:g}/3" if exp is not None else "prospekteringsuppsida saknas")
    s = score_from("properties", "Properties", comps, ctx.cfg["sub_weights"]["properties"], ctx)
    if res["reserve_total"] is None and res["resource_total"] is not None:
        s.unknown.append("? inga reserver angivna — bara resurser (" + "+".join(res["resource_categories"]) + ")")
    return s
