"""
engines/durrett/valuation.py — STEP 9 low valuation (SPEC §13–14).

A. Resursvärde: MCap/EV per resursenhet.   B. Reservvärde: per reservenhet.
C. Framtida kassaflöde: produktion × (pris − AISC) = framtida rörelse-
   kassaflöde; × multipel (input, default 5×, 10× som alternativ) = framtida EV.
D. EV/NPV och MCap/framtida vinst (Durretts 10×-regel ur scoring.py).
Peer-median finns inte i repot (TODO) — trapporna är absoluta och märkta VAL.
"""

from __future__ import annotations

from typing import Optional

from engines.durrett import config as dc
from engines.durrett._base import Ctx, score_from, steps_le
from engines.durrett.models import Score


def future_operating_cf(production: Optional[float], price: Optional[float], aisc: Optional[float]) -> Optional[float]:
    """MUSD: produktion (enheter) × (pris − AISC) / 1e6."""
    if production is None or price is None or aisc is None:
        return None
    return production * (price - aisc) / 1e6


def future_ev(future_fcf: Optional[float], multiple: Optional[float]) -> Optional[float]:
    if future_fcf is None or multiple is None:
        return None
    return future_fcf * multiple


def future_earnings(ctx: Ctx, ss: dict) -> dict:
    """Framtida vinst enligt Durrett: framtida (eller nuvarande) produktion × (pris − AISC)."""
    price = ctx.num("commodity_price", required=False)
    aisc = ctx.num("aisc", required=False)
    if aisc is None:
        aisc = ctx.num("breakeven_price", required=False)
        if aisc is not None:
            ctx.log.append("AISC saknas — break-even-priset används som all-in-kostnad (MODELLED)")
        elif "aisc" not in ctx.missing:
            ctx.missing.append("aisc")
    prod = ctx.quantity("production_future", required=False)
    basis = "framtida produktion"
    if prod is None:
        prod = ctx.quantity("production_current", required=False)
        basis = "nuvarande produktion"
    if prod is None:
        prod = ctx.quantity("annual_production", required=False)
        basis = "årsproduktion (studie)"
    ocf = future_operating_cf(prod, price, aisc)
    tax = ctx.num("tax_rate_pct", required=False)
    tax_f = (tax if tax is not None else ctx.cfg["default_tax_rate_pct"]) / 100
    fcf = ocf * (1 - tax_f) if ocf is not None and ocf > 0 else ocf
    mult = ctx.num("target_ev_ebitda", required=False)
    mult_src = "fält"
    if mult is None:
        mult, mult_src = ctx.cfg["valuation_multiples"]["default"], "config default"
    ev = future_ev(fcf, mult)
    out = {"production": prod, "basis": basis, "price": price, "aisc": aisc, "operating_cf": ocf,
           "tax_pct": tax_f * 100, "tax_assumed": tax is None, "future_fcf": fcf, "multiple": mult,
           "multiple_source": mult_src, "future_ev": ev,
           "future_mcap": (ev - (ss.get("net_debt_usd") or 0.0)) if ev is not None else None}
    if ocf is not None:
        ctx.metric("future_operating_cf", ocf, "MUSD", f"{basis} {prod:,.0f} × ({price:g} − {aisc:g})")
        ctx.metric("future_fcf", fcf, "MUSD", f"× (1 − skatt {tax_f * 100:g} %{' ASSUMPTION' if tax is None else ''})")
        ctx.metric("future_ev", ev, "MUSD", f"× multipel {mult:g} ({mult_src})")
    return out


def score(ctx: Ctx, company_type: str, ss: dict, pm: dict, fe: dict) -> Score:
    st, beyond = ctx.cfg["score_steps"], ctx.cfg["score_beyond"]
    comps: dict = {}
    unit = ctx.base_unit
    gold_like = ctx.price_unit == "USD/oz"

    ev_res = pm.get("ev_per_resource_unit")
    if ev_res is not None:
        if gold_like:
            comps["resource_value"] = (steps_le(ev_res, st["ev_per_oz_usd"], beyond["ev_per_oz_usd"]), f"EV {ev_res:,.0f} USD/oz resurs")
        else:
            comps["resource_value"] = (None, f"EV {ev_res:,.2f} USD/{unit} resurs — ingen trappa för {ctx.price_unit} (TODO)")
    else:
        comps["resource_value"] = (None, "EV eller resurs saknas")
    ev_rev = pm.get("ev_per_reserve_unit")
    if ev_rev is not None:
        if gold_like:
            comps["reserve_value"] = (steps_le(ev_rev / 2, st["ev_per_oz_usd"], beyond["ev_per_oz_usd"]),
                                      f"EV {ev_rev:,.0f} USD/oz reserv (reserv-uns värderas dubbelt mot resurs, VAL)")
        else:
            comps["reserve_value"] = (None, f"EV {ev_rev:,.2f} USD/{unit} reserv — ingen trappa (TODO)")
    else:
        comps["reserve_value"] = (None, "EV eller reserv saknas")

    npv = ctx.num("npv_musd", required=False)
    ev = ss.get("ev_usd")
    if npv is not None and npv > 0 and ev is not None:
        r = ev / npv
        ctx.metric("ev_npv", r, "×")
        comps["ev_npv"] = (steps_le(r, st["ev_npv"], beyond["ev_npv"]), f"EV/NPV {r:.2f}×")
    elif npv is not None and npv <= 0:
        comps["ev_npv"] = (5.0, f"NPV {npv:g} MUSD ≤ 0")
    else:
        comps["ev_npv"] = (None, "NPV eller EV saknas")

    mcap = ss.get("mcap_usd")
    if fe.get("future_fcf") is not None and mcap:
        if fe["future_fcf"] > 0:
            r = mcap / fe["future_fcf"]
            ctx.metric("mcap_future_earnings", r, "×", "Durretts köpregel < 10×")
            comps["future_earnings"] = (steps_le(r, st["mcap_future_earnings"], beyond["mcap_future_earnings"]),
                                        f"MCap/framtida vinst {r:.1f}× ({fe['basis']})")
        else:
            comps["future_earnings"] = (5.0, f"framtida vinst ≤ 0 ({fe['basis']} × negativ marginal)")
    else:
        comps["future_earnings"] = (None, "framtida vinst eller börsvärde saknas")

    weights = dict(ctx.cfg["sub_weights"]["valuation"])
    if company_type == dc.EXPLORER:
        weights = {"resource_value": 60, "reserve_value": 0, "ev_npv": 20, "future_earnings": 20}
    elif company_type == dc.ROYALTY:
        weights = {"resource_value": 0, "reserve_value": 0, "ev_npv": 40, "future_earnings": 60}
    weights = {k: w for k, w in weights.items() if w > 0}
    return score_from("valuation", "Valuation", comps, weights, ctx, min_coverage=0.3)
