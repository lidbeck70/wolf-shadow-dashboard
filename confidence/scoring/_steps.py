"""
confidence/scoring/_steps.py — trappor och tabeller, en gång.

Tre läsordningar räcker för alla config-tabeller:
  step_ge   (tröskel fallande, v ≥ tröskel → p)   marginaler, IRR, yield …
  step_le   (tröskel stigande, v ≤ tröskel → p)   payback, EV/EBITDA, gap …
  table_lt  (gräns stigande,  v < gräns  → p, annars beyond)   spec-tabellerna
Plus läsning av ett fält med DATA_MISSING-bokföring och pro-rata-skalning
när en del inte gäller ett stage.
"""

from __future__ import annotations

from typing import Optional

from confidence.data.models import CompanyInput, PillarScore
from confidence.data.provenance import describe
from confidence.data.validation import usable


def step_ge(v: Optional[float], steps) -> float:
    if v is None:
        return 0.0
    for thr, pts in steps:
        if v >= thr:
            return float(pts)
    return 0.0


def step_le(v: Optional[float], steps) -> float:
    if v is None:
        return 0.0
    for thr, pts in steps:
        if v <= thr:
            return float(pts)
    return 0.0


def table_lt(v: Optional[float], table, beyond) -> float:
    if v is None:
        return 0.0
    for limit, pts in table:
        if v < limit:
            return float(pts)
    return float(beyond)


def read(company: CompanyInput, key: str, pillar: PillarScore) -> Optional[float]:
    """Talet om fältet är giltigt för stage, annars None + bokfört som saknat."""
    if usable(company, key):
        return company.num(key)
    if key not in pillar.missing:
        pillar.missing.append(key)
    return None


def read_bool(company: CompanyInput, key: str, pillar: PillarScore) -> Optional[bool]:
    if usable(company, key):
        return company.truth(key)
    if key not in pillar.missing:
        pillar.missing.append(key)
    return None


def note(pillar: PillarScore, name: str, pts: float, max_pts: float, text: str) -> None:
    """Delpoäng + en förklarande rad: 'AISC-marginal 3/4 — 31 % (…)'."""
    pillar.components[name] = round(pts, 2)
    pillar.notes.append(f"{name} {pts:g}/{max_pts:g} — {text}")


def src(company: CompanyInput, key: str, label: str = "") -> str:
    """Källraden för ett fält, för notes."""
    return describe(company.get(key), label)


def finish(pillar: PillarScore, applicable_max: Optional[float] = None) -> PillarScore:
    """Summera komponenter, applicera tak, skala pro rata om delar inte gäller stage."""
    total = sum(pillar.components.values())
    if applicable_max is not None and applicable_max < pillar.max and applicable_max > 0:
        total = total * pillar.max / applicable_max
        pillar.notes.append(f"skalad {applicable_max:g} → {pillar.max:g} p: delar som inte gäller "
                            f"stage räknas inte (VAL)")
    for _name, cap in pillar.caps:
        if total > cap:
            total = float(cap)
    pillar.points = round(min(max(total, 0.0), pillar.max), 2)
    return pillar


def cap(pillar: PillarScore, name: str, value: float, text: str) -> None:
    pillar.caps.append((name, value))
    pillar.notes.append(f"TAK {value:g} p — {name}: {text}")
