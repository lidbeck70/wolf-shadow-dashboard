"""
engines/durrett/thesis.py — INVESTMENT THESIS + CATALYST ENGINE (SPEC §35–36).

Neutral text ur poängen: why it could work / fail, key catalysts, key
risks, what must go right, what would invalidate, next milestones.
Aldrig BUY/SELL. Katalysatorer läses ur CompanyInput.catalysts och
sorteras på väntat datum, vikt, påverkan, konfidens.
"""

from __future__ import annotations

from engines.durrett import config as dc
from engines.durrett.models import Catalyst

_IMPORTANCE_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
_CONF_RANK = {"high": 0, "medium": 1, "low": 2}


def catalysts(company) -> list:
    out = []
    for raw in company.catalysts or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        t = str(raw.get("type") or "")
        out.append(Catalyst(name=str(raw["name"]), type=t if t in dc.CATALYST_TYPES else "",
                            expected=str(raw.get("expected") or ""),
                            importance=str(raw.get("importance") or "MEDIUM").upper(),
                            impact=str(raw.get("impact") or ""),
                            confidence=str(raw.get("confidence") or "medium").lower(),
                            source=str(raw.get("source") or "")))
    out.sort(key=lambda c: (c.expected or "9999", _IMPORTANCE_RANK.get(c.importance, 1),
                            _CONF_RANK.get(c.confidence, 1)))
    return out


def build(a) -> dict:
    """a: DurrettAnalysis (utan thesis). Returnerar dict med sju listor."""
    work, fail, must, invalid, next_ms = [], [], [], [], []
    for s in a.scores():
        if s.value is None:
            continue
        if s.value >= 70:
            work += [p[2:] for p in s.positive[:2]]
        elif s.value < 40:
            fail += [n[2:] for n in s.negative[:2]]
    base = a.base_case
    if base and base.upside_pct is not None:
        (work if base.upside_pct > 0 else fail).append(
            f"Base-scenariot ger {base.upside_pct:+.0f} % vid {base.commodity_price:g} {a.commodity and ''}pris, "
            f"multipel {base.multiple:g}")
        must.append(f"råvarupriset håller Base-nivån ({base.commodity_price:g}) och produktionen når {base.production:,.0f}")
    bear = a.bear_case
    if bear and bear.upside_pct is not None:
        invalid.append(f"Bear-scenariot ({bear.commodity_price:g}, AISC {bear.aisc:g}) ger {bear.upside_pct:+.0f} %")
    for f in a.red_flags:
        if f.severity in (dc.CRITICAL, dc.HIGH):
            invalid.append(f"{f.flag}: {f.reason}")
    if a.developer_checklist:
        for name, ok, why in a.developer_checklist["items"]:
            if ok is False:
                must.append(f"{name}: {why}")
    if a.explorer_profile:
        must.append(f"{a.explorer_profile['play']}: nästa steg på Lassonde-kurvan efter {a.explorer_profile['lassonde_label']}")
    for c in a.catalysts[:5]:
        next_ms.append(f"{c.expected or 'datum okänt'} · {c.name}" + (f" ({c.importance})" if c.importance else ""))
    if not next_ms and a.missing_data:
        next_ms.append("inga katalysatorer registrerade — lägg in under Indata")
    key_risks = [f"{f.severity}: {f.flag} — {f.reason}" for f in a.red_flags[:6]]
    return {"why_it_could_work": work[:6] or ["inga delpoäng över 70 ännu"],
            "why_it_could_fail": fail[:6] or ["inga delpoäng under 40 i det som är angivet"],
            "key_catalysts": [f"{c.name} ({c.expected or '?'}, {c.importance})" for c in a.catalysts[:6]],
            "key_risks": key_risks or ["inga red flags i det som är angivet"],
            "what_must_go_right": must[:6],
            "what_would_invalidate": invalid[:6] or ["Bear-scenario eller red flags saknas — kan inte formuleras"],
            "next_milestones": next_ms[:6],
            "note": "Neutral sammanställning ur poängen — ingen köp- eller säljrekommendation."}
