"""
confidence/regional.py — Regional knapphet 0–100 (SPEC).

Hur knapp är råvaran ur ett västligt/regionalt perspektiv, och sitter
tillgången där det betyder något?
  jurisdiktion   50  repots tabell (contrarian_alpha.resource_scoring)
  koncentration  30  största producentlandets andel av världsutbudet —
                     full poäng bara om tillgången ligger i en säker
                     jurisdiktion (annars halveras: den är inte "västlig")
  västligt gap   20  andel av utbudet ur västliga jurisdiktioner
Koncentration och västlig andel kommer ur råvaruregistrets överlagring
(null tills sourcat → 0 + DATA_MISSING, skalas pro rata).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from confidence import config as cfg
from confidence.commodities import Commodity
from confidence.data.models import CompanyInput
from confidence.data.provenance import describe, num
from confidence.scoring._steps import step_ge, step_le

try:
    from contrarian_alpha.resource_scoring import score_jurisdiction
except Exception:                                                   # pragma: no cover
    score_jurisdiction = None


@dataclass
class RegionalScarcity:
    score: float
    band: str
    components: dict
    notes: list
    missing: list
    coverage: float
    jurisdiction_score: Optional[float] = None
    jurisdiction_confidence: Optional[float] = None
    flags: list = field(default_factory=list)


def band_for(score: float) -> str:
    return next((lbl for floor, lbl in cfg.REGIONAL_BANDS if score >= floor), cfg.REGIONAL_BANDS[-1][1])


def jurisdiction(company: CompanyInput) -> tuple:
    """(poäng 0–100, konfidens 0–1, flaggor) ur repots jurisdiktionstabell."""
    if score_jurisdiction is None:
        return None, None, ["resource_scoring saknas"]
    return score_jurisdiction(company.country or "", company.exchange or "", company.jurisdiction or "")


def regional_scarcity(company: CompanyInput, commodity: Optional[Commodity]) -> RegionalScarcity:
    sub = cfg.REGIONAL_SUB
    comp: dict = {}
    notes: list = []
    missing: list = []
    flags: list = []
    covered = 0.0

    j_score, j_conf, j_flags = jurisdiction(company)
    if j_score is not None and "JURISDICTION_UNKNOWN" not in j_flags:
        comp["Jurisdiktion"] = round(j_score / 100 * sub["jurisdiction"], 2)
        covered += sub["jurisdiction"]
        where = company.jurisdiction or company.country or company.exchange
        notes.append(f"Jurisdiktion {comp['Jurisdiktion']:g}/{sub['jurisdiction']} — {where}: {j_score:g}/100 "
                     f"(konfidens {j_conf:g}, repots tabell)")
        flags += [f for f in j_flags if f != "JURISDICTION_FROM_EXCHANGE"]
        if "JURISDICTION_FROM_EXCHANGE" in j_flags:
            notes.append("jurisdiktion härledd ur börsen — ange land/region för full konfidens")
    else:
        missing.append("jurisdiction")
        given = ", ".join(x for x in (company.country, company.jurisdiction, company.exchange) if x)
        notes.append(f"Jurisdiktion 0/{sub['jurisdiction']} — DATA_MISSING ("
                     + (f"{given} finns inte i repots jurisdiktionstabell" if given else "land/region saknas") + ")")
    safe = j_score is not None and j_score >= cfg.SAFE_JURISDICTION_MIN

    conc = num(commodity.supply_concentration_pct) if commodity else None
    if conc is not None:
        pts = step_ge(conc, cfg.CONCENTRATION_STEPS)
        who = f" ({commodity.top_supplier})" if commodity.top_supplier else ""
        if not safe:
            pts = pts / 2
            tail = f" — halverad: tillgången ligger inte i en säker jurisdiktion (< {cfg.SAFE_JURISDICTION_MIN:g})"
        else:
            tail = ""
        comp["Koncentration"] = round(pts, 2)
        covered += sub["concentration"]
        notes.append(f"Koncentration {pts:g}/{sub['concentration']} — största land{who} {conc:g} % · "
                     f"{describe(commodity.supply_concentration_pct)}{tail}")
    else:
        missing.append("commodity.supply_concentration_pct")
        notes.append(f"Koncentration 0/{sub['concentration']} — DATA_MISSING")

    west = num(commodity.western_share_pct) if commodity else None
    if west is not None:
        pts = step_le(west, cfg.WESTERN_GAP_STEPS) or cfg.WESTERN_GAP_BEYOND
        comp["Västligt gap"] = pts
        covered += sub["western_gap"]
        notes.append(f"Västligt gap {pts:g}/{sub['western_gap']} — västlig andel {west:g} % · "
                     f"{describe(commodity.western_share_pct)}")
    else:
        missing.append("commodity.western_share_pct")
        notes.append(f"Västligt gap 0/{sub['western_gap']} — DATA_MISSING")

    total_max = sum(sub.values())
    coverage = covered / total_max
    raw = sum(comp.values())
    if covered <= 0:
        score = 0.0
        flags.append("regional knapphet saknar underlag — 0")
    elif coverage < cfg.REGIONAL_MIN_COVERAGE:
        score = raw                                  # jurisdiktion ensam säger inget om knapphet
        flags.append(f"underlag {coverage * 100:.0f} % — råpoäng utan uppskalning; saknas: {', '.join(missing)}")
    else:
        score = raw / covered * 100
        if coverage < 1:
            notes.append(f"skalad {covered:g} → {total_max} p: delar utan data räknas inte (täckning {coverage * 100:.0f} %)")
            flags.append(f"täckning {coverage * 100:.0f} % — saknas: {', '.join(missing)}")
    return RegionalScarcity(round(score, 1), band_for(score), comp, notes, missing, round(coverage, 2),
                            j_score, j_conf, flags)
