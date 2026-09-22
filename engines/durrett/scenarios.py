"""
engines/durrett/scenarios.py — BEAR / BASE / BULL (SPEC §29–30, §34).

Per scenario kan användaren justera: råvarupris, produktion, AISC, CapEx,
FX, multipel, recovery, gruvliv. Defaults ur config.scenario_defaults
(metallpris per råvara när tabellen finns, annars procentuell ändring
mot dagens pris — och det sägs). Varje fall listar sina antaganden och
steg. Fair value/aktie = framtida börsvärde / FD-aktier.

Ingen dold spot-prognos: Base använder pristabellen om den finns; annars
dagens pris, märkt ASSUMPTION "spot som långsiktigt pris".
"""

from __future__ import annotations

from typing import Optional

from engines.durrett._base import Ctx
from engines.durrett.models import Assumption, ScenarioCase

_LABEL = {"bear": "Bear", "base": "Base", "bull": "Bull"}
_ADJ_KEYS = ("commodity_price", "production", "aisc", "capex_musd", "fx_to_usd", "multiple", "recovery_pct",
             "mine_life_years")


def defaults_for(ctx: Ctx, key: str) -> dict:
    """Scenariots default-justeringar {price, price_source, production_pct, aisc_pct, capex_pct, fx_pct,
    multiple, recovery_pct, mine_life_pct}."""
    sd = ctx.cfg["scenario_defaults"]
    ck = ctx.c.commodity
    table = sd["commodity_prices"].get(ck)
    spot = ctx.num("commodity_price", required=False)
    if table and table.get(key) is not None:
        price, src = table[key], f"config pristabell {ck} ({table.get('unit', '')})"
    elif spot is not None:
        price, src = spot * (1 + sd["price_change_pct"][key] / 100), \
            f"spot {spot:g} {sd['price_change_pct'][key]:+g} % (ASSUMPTION — ingen pristabell för {ck})"
    else:
        price, src = None, "råvarupris saknas"
    return {"price": price, "price_source": src, "production_pct": sd["production_change_pct"][key],
            "aisc_pct": sd["aisc_change_pct"][key], "capex_pct": sd["capex_change_pct"][key],
            "fx_pct": sd["fx_change_pct"][key], "multiple": sd["multiple"][key],
            "recovery_pct": sd["recovery_change_pct"][key], "mine_life_pct": sd["mine_life_change_pct"][key]}


def run_case(ctx: Ctx, key: str, ss: dict, fe: dict, overrides: Optional[dict] = None) -> ScenarioCase:
    """overrides: {commodity_price, production, aisc, capex_musd, fx_to_usd, multiple, recovery_pct,
    mine_life_years} — absoluta värden som ersätter defaults för scenariot."""
    ov = overrides or {}
    d = defaults_for(ctx, key)
    s = ScenarioCase(key, _LABEL[key])
    A = s.assumptions.append

    price = ov.get("commodity_price", d["price"])
    if price is None:
        s.reason = "råvarupris saknas"
        return s
    A(Assumption("Råvarupris", price, ctx.price_unit, "ASSUMPTION", "override" if "commodity_price" in ov else d["price_source"]))

    base_prod = fe.get("production")
    if base_prod is None:
        s.reason = "produktion saknas (nu, framtida eller studie)"
        return s
    prod = ov.get("production", base_prod * (1 + d["production_pct"] / 100))
    A(Assumption("Produktion", prod, ctx.base_unit + "/år", "ASSUMPTION",
                 "override" if "production" in ov else f"{fe.get('basis')} {d['production_pct']:+g} %"))
    rec_adj = d["recovery_pct"]
    if "recovery_pct" in ov:
        base_rec = ctx.num("recovery_pct", required=False)
        rec_adj = (ov["recovery_pct"] / base_rec - 1) * 100 if base_rec else 0.0
    if rec_adj:
        prod = prod * (1 + rec_adj / 100)
        A(Assumption("Recovery-justering", rec_adj, "%", "ASSUMPTION", "produktionen skalas"))

    aisc_base = fe.get("aisc")
    if aisc_base is None:
        s.reason = "AISC/break-even saknas"
        return s
    aisc = ov.get("aisc", aisc_base * (1 + d["aisc_pct"] / 100))
    A(Assumption("AISC", aisc, ctx.price_unit, "ASSUMPTION", "override" if "aisc" in ov else f"{aisc_base:g} {d['aisc_pct']:+g} %"))
    mult = ov.get("multiple", d["multiple"])
    A(Assumption("Multipel", mult, "×", "ASSUMPTION", "override" if "multiple" in ov else "config scenario_defaults"))
    tax = fe.get("tax_pct", ctx.cfg["default_tax_rate_pct"])
    A(Assumption("Skattesats", tax, "%", "ASSUMPTION" if fe.get("tax_assumed", True) else "ACTUAL", "config" if fe.get("tax_assumed", True) else "fält"))
    fx_pct = d["fx_pct"]
    if "fx_to_usd" in ov:
        base_fx = ctx.num("fx_to_usd", required=False)
        fx_pct = (ov["fx_to_usd"] / base_fx - 1) * 100 if base_fx else 0.0
    if fx_pct:
        A(Assumption("FX-justering av kostnader", fx_pct, "%", "ASSUMPTION", "AISC i lokal valuta skalas"))
        aisc = aisc * (1 + fx_pct / 100)

    s.commodity_price, s.production, s.aisc, s.multiple = price, prod, aisc, mult
    s.operating_cf_musd = prod * (price - aisc) / 1e6
    s.steps.append(f"rörelsekassaflöde = {prod:,.0f} × ({price:g} − {aisc:g}) = {s.operating_cf_musd:,.0f} MUSD")
    s.future_fcf_musd = s.operating_cf_musd * (1 - tax / 100) if s.operating_cf_musd > 0 else s.operating_cf_musd
    s.steps.append(f"FCF = × (1 − skatt {tax:g} %) = {s.future_fcf_musd:,.0f} MUSD")
    s.future_ev_musd = s.future_fcf_musd * mult
    s.steps.append(f"framtida EV = FCF × {mult:g} = {s.future_ev_musd:,.0f} MUSD")

    capex_base = ctx.num("capex_musd", required=False)
    pre = ctx.c.stage in ("explorer", "developer")
    net_debt = ss.get("net_debt_usd") or 0.0
    capex = None
    if pre and capex_base is not None:
        capex = ov.get("capex_musd", capex_base * (1 + d["capex_pct"] / 100))
        s.capex_musd = capex
        A(Assumption("Initial CapEx", capex, "MUSD", "ASSUMPTION", "override" if "capex_musd" in ov else f"{capex_base:g} {d['capex_pct']:+g} %"))
        committed = ctx.num("committed_financing_musd", required=False) or 0.0
        unfunded = max(0.0, capex - (ss.get("cash_usd") or 0.0) - committed)
        s.future_mcap_musd = s.future_ev_musd - net_debt - unfunded
        s.steps.append(f"framtida börsvärde = EV − nettoskuld {net_debt:,.0f} − ofinansierad CapEx {unfunded:,.0f} = {s.future_mcap_musd:,.0f} MUSD "
                       f"(ofinansierad del antas lånas/emitteras till nuvarande värde, konservativt)")
    else:
        s.future_mcap_musd = s.future_ev_musd - net_debt
        s.steps.append(f"framtida börsvärde = EV − nettoskuld {net_debt:,.0f} = {s.future_mcap_musd:,.0f} MUSD")

    life = ctx.num("mine_life_years", required=False)
    if life is not None and d["mine_life_pct"]:
        A(Assumption("Gruvliv", life * (1 + d["mine_life_pct"] / 100), "år", "ASSUMPTION",
                     f"{life:g} {d['mine_life_pct']:+g} % — påverkar inte multipelvärderingen, visas för spårbarhet"))

    cur = ss.get("fd_mcap_usd") or ss.get("mcap_usd")
    fd = ss.get("fd_m")
    if fd:
        s.fair_value_per_share = s.future_mcap_musd / fd
        s.steps.append(f"fair value/aktie = {s.future_mcap_musd:,.0f} / {fd:g} M FD-aktier = {s.fair_value_per_share:,.3g} USD")
    if cur:
        s.upside_multiple = s.future_mcap_musd / cur
        s.upside_pct = (s.upside_multiple - 1) * 100
        s.steps.append(f"upside = {s.future_mcap_musd:,.0f} / {cur:,.0f} = {s.upside_multiple:.2f}× ({s.upside_pct:+.0f} %)")
    else:
        s.reason = "börsvärde saknas — ingen upside"
    return s


def run_all(ctx: Ctx, ss: dict, fe: dict, overrides: Optional[dict] = None) -> dict:
    """{bear, base, bull} → ScenarioCase. overrides: {scenario: {param: värde}}."""
    ov = overrides or {}
    return {k: run_case(ctx, k, ss, fe, ov.get(k)) for k in ("bear", "base", "bull")}


def robustness(cases: dict) -> Optional[float]:
    """Scenariorobusthet 0–100 för Model Confidence: hur mycket av Base-upsiden
    överlever i Bear (VAL). None när Bear/Base saknas."""
    bear, base = cases.get("bear"), cases.get("base")
    if not bear or not base or bear.upside_pct is None or base.upside_pct is None:
        return None
    if base.upside_multiple is None or base.upside_multiple <= 0:
        return 0.0
    ratio = (bear.upside_multiple or 0.0) / base.upside_multiple
    return round(max(0.0, min(100.0, ratio * 100.0)), 1)
