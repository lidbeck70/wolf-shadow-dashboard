"""
engines/durrett/costs.py — STEP 7 cost structure (SPEC §11).

Producenter: marginal/enhet = metallpris − AISC, AISC/pris, kassamarginal
(cash cost), sustaining capex. Developers: IRR, NPV/CAPEX, payback,
capex-intensitet (CapEx mot börsvärde). Royalty: kostnadsstrukturen är
inte relevant → N/A med skäl (profilen väger bort den).
"""

from __future__ import annotations

from typing import Optional

from engines.durrett import config as dc
from engines.durrett._base import Ctx, score_from, steps_ge, steps_le
from engines.durrett.models import Score


def margin_per_unit(price: Optional[float], aisc: Optional[float]) -> Optional[float]:
    if price is None or aisc is None:
        return None
    return price - aisc


def aisc_to_price(price: Optional[float], aisc: Optional[float]) -> Optional[float]:
    if price is None or aisc is None or price <= 0:
        return None
    return aisc / price


def score(ctx: Ctx, company_type: str, ss: dict) -> Score:
    st = ctx.cfg["score_steps"]
    comps: dict = {}
    if company_type == dc.ROYALTY:
        s = Score("costs", "Costs", None, reason="royalty/stream — ingen egen kostnadsstruktur")
        return s
    price = ctx.num("commodity_price", required=False)
    if company_type in (dc.PRODUCER, dc.HYBRID):
        aisc = ctx.num("aisc")                                  # kärnfält för producenter — bokförs som saknat
        m = margin_per_unit(price, aisc)
        r = aisc_to_price(price, aisc)
        if m is not None:
            ctx.metric("operating_margin_per_unit", m, ctx.price_unit, f"pris {price:g} − AISC {aisc:g}")
            ctx.metric("aisc_to_price", r, "×")
            pct = m / price * 100
            comps["margin"] = (steps_ge(pct, st["aisc_margin_pct"]), f"marginal {m:g} {ctx.price_unit} = {pct:.0f} % av priset")
        else:
            comps["margin"] = (None, "AISC eller råvarupris saknas")
        cc = ctx.num("cash_cost", required=False)
        if cc is not None and price:
            cpct = (price - cc) / price * 100
            comps["cost_position"] = (steps_ge(cpct, st["aisc_margin_pct"]), f"kassamarginal (C1/cash cost {cc:g}) {cpct:.0f} %")
        elif aisc is not None and price:
            comps["cost_position"] = (steps_ge((price - aisc) / price * 100, st["aisc_margin_pct"]),
                                      "cash cost saknas — AISC-marginalen används")
        else:
            comps["cost_position"] = (None, "cash cost / C1 saknas")
        sus = ctx.num("sustaining_capex_musd", required=False)
        prod = ctx.quantity("production_current", required=False)
        if sus is not None and prod and price:
            rev = prod * price / 1e6
            share = sus / rev * 100 if rev else None
            comps["sustaining"] = (None if share is None else steps_le(share, ((5.0, 100), (10.0, 80), (15.0, 60), (25.0, 40)), 20),
                                   f"sustaining capex {sus:g} MUSD = {share:.0f} % av intäkten" if share is not None else "intäkt 0")
        else:
            comps["sustaining"] = (None, "sustaining capex saknas (ingår ofta i AISC)")
        return score_from("costs", "Costs", comps, ctx.cfg["sub_weights"]["costs"], ctx)

    # developer / explorer / unknown
    irr = ctx.num("irr_pct", required=False)
    npv = ctx.num("npv_musd", required=False)
    capex = ctx.num("capex_musd", required=False)
    nc = npv / capex if npv is not None and capex else None
    parts = []
    vals = []
    if irr is not None:
        vals.append(steps_ge(irr, st["irr_pct"]))
        parts.append(f"IRR {irr:g} %")
    if nc is not None:
        vals.append(steps_ge(nc, st["npv_capex"]))
        parts.append(f"NPV/CapEx {nc:.2f}×")
    pb = ctx.num("payback_years", required=False)
    if pb is not None:
        vals.append(steps_le(pb, ((1.5, 100), (2.5, 80), (3.5, 60), (5.0, 40)), 20))
        parts.append(f"payback {pb:g} år")
    comps["margin"] = (round(sum(vals) / len(vals), 1) if vals else None, ", ".join(parts) or "IRR/NPV/CapEx/payback saknas")
    be = ctx.num("breakeven_price", required=False)
    if be is not None and price:
        cover = price / be
        comps["cost_position"] = (steps_ge(cover, ((1.6, 100), (1.4, 85), (1.25, 70), (1.15, 55), (1.05, 35)), 15),
                                  f"pris/break-even {cover:.2f}×")
    else:
        comps["cost_position"] = (None, "break-even-pris saknas")
    mcap = ss.get("mcap_usd")
    if capex is not None and mcap:
        ratio = capex / mcap
        ctx.metric("capex_to_mcap", ratio, "×")
        comps["sustaining"] = (steps_le(ratio, st["capex_to_mcap"], ctx.cfg["score_beyond"]["capex_to_mcap"]),
                               f"initial CapEx {capex:g} MUSD = {ratio:.1f}× börsvärdet")
    else:
        comps["sustaining"] = (None, "CapEx eller börsvärde saknas")
    return score_from("costs", "Costs", comps, ctx.cfg["sub_weights"]["costs"], ctx)
