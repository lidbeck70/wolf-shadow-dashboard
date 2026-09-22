"""
engines/durrett/developer.py — DURRETT DEVELOPER FIT (SPEC §16).

Sex separata kriterier: Strong Project, High Upside Potential, Good
Location, Strong Management, Path to Production, Strong Insiders.
Checklistan 0–6 visas som 5/6 etc. och kallas aldrig BUY. Ett kriterium
som inte kan bedömas blir None ("?"), inte False.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from engines.durrett._base import Ctx


def checklist(ctx: Ctx, scores: dict, up: dict, today: Optional[date] = None) -> dict:
    cfg = ctx.cfg["developer_checklist"]
    today = today or date.today()
    items = []

    npv = ctx.num("npv_musd", required=False)
    capex = ctx.num("capex_musd", required=False)
    irr = ctx.num("irr_pct", required=False)
    nc = npv / capex if npv is not None and capex else None
    if nc is None and irr is None:
        items.append(("Strong Project", None, "NPV/CapEx och IRR saknas"))
    else:
        ok = (nc is not None and nc >= cfg["strong_project_npv_capex_min"]) and \
             (irr is None or irr >= cfg["strong_project_irr_min"])
        items.append(("Strong Project", ok, f"NPV/CapEx {nc:.2f}× (krav ≥ {cfg['strong_project_npv_capex_min']:g})" if nc is not None else "NPV/CapEx saknas"
                      + (f", IRR {irr:g} % (krav ≥ {cfg['strong_project_irr_min']:g})" if irr is not None else "")))

    m = up.get("upside_multiple")
    if m is None:
        items.append(("High Upside Potential", None, "upside-multipel saknas"))
    else:
        items.append(("High Upside Potential", m >= cfg["high_upside_multiple_min"],
                      f"{m:.1f}× (krav ≥ {cfg['high_upside_multiple_min']:g}×)"))

    j = scores["jurisdiction"].value
    items.append(("Good Location", None if j is None else j >= cfg["good_location_min_score"],
                  f"Jurisdiction {j:g} (krav ≥ {cfg['good_location_min_score']:g})" if j is not None else "jurisdiktion N/A"))

    mg = scores["management"].value
    items.append(("Strong Management", None if mg is None else mg >= cfg["strong_management_min_score"],
                  f"Management {mg:g} (krav ≥ {cfg['strong_management_min_score']:g})" if mg is not None else "management N/A"))

    yr = ctx.num("first_cashflow_year", required=False)
    permits = ctx.truth("permits_granted", required=False)
    fin = ctx.truth("financing_committed", required=False)
    if yr is None:
        items.append(("Path to Production", None, "första kassaflöde-år saknas"))
    else:
        years = yr - today.year
        ok = years <= cfg["path_to_production_max_years"] and permits is not False
        why = f"{years:g} år till kassaflöde (krav ≤ {cfg['path_to_production_max_years']:g})"
        why += ", tillstånd beviljade" if permits else (", tillstånd saknas" if permits is False else ", tillstånd okänt")
        why += ", finansiering åtagen" if fin else ""
        items.append(("Path to Production", ok, why))

    own = ctx.num("insider_ownership_pct", required=False)
    buy = ctx.num("insider_buying_12m_musd", required=False)
    if own is None:
        items.append(("Strong Insiders", None, "insynsägande saknas"))
    else:
        ok = own >= cfg["strong_insiders_min_pct"]
        items.append(("Strong Insiders", ok, f"insynsägande {own:g} % (krav ≥ {cfg['strong_insiders_min_pct']:g})"
                      + (f", insiderköp {buy:g} MUSD" if buy else "")))

    passed = sum(1 for _n, ok, _w in items if ok)
    unknown = sum(1 for _n, ok, _w in items if ok is None)
    return {"items": items, "passed": passed, "unknown": unknown, "total": len(items),
            "label": f"DURRETT DEVELOPER FIT {passed}/{len(items)}" + (f" ({unknown} okända)" if unknown else "")}
