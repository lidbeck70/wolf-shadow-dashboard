"""
engines/durrett/jurisdiction.py — STEP 4 location (SPEC §8).

COUNTRY RISK ur repots jurisdiktionstabell (contrarian_alpha.resource_scoring,
region före land före börs). PROJECT RISK ur åtta 0–2-bedömningar
(tillstånd, infrastruktur, kraft/vatten, samhälle, säkerhet, miljö,
nationalisering, valuta) plus statens andel. Projektet väger 60 %, så ett
konkret bra projekt i ett sämre land straffas inte automatiskt.
"""

from __future__ import annotations

from engines.durrett._base import Ctx, score_from, steps_le
from engines.durrett.models import Score

try:
    from contrarian_alpha.resource_scoring import score_jurisdiction
except Exception:                                                   # pragma: no cover
    score_jurisdiction = None

_PROJECT_FIELDS = (("risk_permitting", "tillstånd", 20), ("risk_infrastructure", "infrastruktur", 15),
                   ("risk_power_water", "kraft/vatten", 12), ("risk_community", "samhälle/urfolk", 15),
                   ("risk_security", "säkerhet", 10), ("risk_environment", "miljö", 10),
                   ("risk_nationalization", "nationalisering/ägande", 10), ("risk_currency", "valuta", 8))


def country_risk(ctx: Ctx) -> tuple:
    """(poäng 0–100 | None, förklaring)."""
    c = ctx.c
    if score_jurisdiction is None:
        return None, "resource_scoring saknas"
    sc, conf, flags = score_jurisdiction(c.country or "", c.exchange or "", c.jurisdiction or "")
    if "JURISDICTION_UNKNOWN" in flags:
        return None, f"{c.country or c.jurisdiction or 'land'} finns inte i repots tabell"
    where = c.jurisdiction or c.country or c.exchange
    note = f"{where}: {sc:g}/100 (repots tabell, konfidens {conf:g})"
    if "JURISDICTION_FROM_EXCHANGE" in flags:
        note += " — härledd ur börsen"
    return float(sc), note


def project_risk(ctx: Ctx) -> tuple:
    """Viktat snitt av 0–2-bedömningarna (0 → 10, 1 → 50, 2 → 90) + statens andel."""
    acc, wsum, why = 0.0, 0.0, []
    for key, label, w in _PROJECT_FIELDS:
        v = ctx.num(key, required=False)
        if v is None:
            continue
        acc += {0: 10.0, 1: 50.0, 2: 90.0}.get(int(v), 50.0) * w
        wsum += w
        why.append(f"{label} {int(v)}/2")
    take = ctx.num("government_take_pct", required=False)
    if take is not None:
        acc += steps_le(take, ((30.0, 90), (40.0, 70), (50.0, 50), (60.0, 30)), 10) * 10
        wsum += 10
        why.append(f"statens andel {take:g} %")
    if wsum == 0:
        return None, "inga projektbedömningar angivna"
    cover = wsum / (sum(w for _k, _l, w in _PROJECT_FIELDS) + 10)
    tail = f" ({cover * 100:.0f} % av bedömningarna gjorda)" if cover < 1 else ""
    return round(acc / wsum, 1), ", ".join(why) + tail


def score(ctx: Ctx) -> Score:
    cv, cwhy = country_risk(ctx)
    pv, pwhy = project_risk(ctx)
    comps = {"country": (cv, cwhy), "project": (pv, pwhy)}
    s = score_from("jurisdiction", "Jurisdiction", comps, ctx.cfg["sub_weights"]["jurisdiction"], ctx,
                   min_coverage=0.4)
    if cv is not None:
        ctx.metric("country_risk_score", cv, "0–100")
    if pv is not None:
        ctx.metric("project_risk_score", pv, "0–100")
    return s
