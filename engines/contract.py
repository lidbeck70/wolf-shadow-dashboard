"""
engines/contract.py — det standardiserade svaret från varje motor.

{
    "engine": "durrett", "ticker": "XYZ",
    "quality_score": 0-100, "risk_score": 0-100 (100 = lägst risk),
    "valuation_score": 0-100, "upside_score": 0-100, "confidence_score": 0-100,
    "red_flags": [ {flag, severity, reason, source, data, date} ],
    "catalysts": [ {name, type, expected, importance, impact, confidence, source} ],
    "missing_data": [ fältnyckel ],
    "explanations": { poängnamn: {"score", "positive", "negative", "unknown", "drivers"} },
    "scenarios": { "bear"|"base"|"bull": {fair_value, upside_pct, assumptions} },
    "extras": motor-specifikt (t.ex. Durretts alla delpoäng)
}

Alla poäng är 0–100 där 0 = extremt svagt, 50 = neutralt, 100 = extremt
starkt. None betyder N/A (kunde inte beräknas) — aldrig 0 för okänt.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

ENGINE_FIELDS = ("quality_score", "risk_score", "valuation_score", "upside_score", "confidence_score",
                 "red_flags", "catalysts", "missing_data", "explanations", "scenarios")


@dataclass
class EngineResult:
    engine: str
    ticker: str
    quality_score: Optional[float]
    risk_score: Optional[float]
    valuation_score: Optional[float]
    upside_score: Optional[float]
    confidence_score: Optional[float]
    red_flags: list = field(default_factory=list)
    catalysts: list = field(default_factory=list)
    missing_data: list = field(default_factory=list)
    explanations: dict = field(default_factory=dict)
    scenarios: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)
    generated: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def is_complete(self) -> bool:
        return all(getattr(self, k) is not None for k in ENGINE_FIELDS[:5])


def validate_result(d: Any) -> list:
    """Fel i ett motorsvar (tom lista = giltigt). För multi-model-vyn."""
    errs = []
    if not isinstance(d, dict):
        return ["inte ett dict"]
    for k in ENGINE_FIELDS:
        if k not in d:
            errs.append(f"saknar {k}")
    for k in ENGINE_FIELDS[:5]:
        v = d.get(k)
        if v is not None and not (isinstance(v, (int, float)) and 0 <= v <= 100):
            errs.append(f"{k} utanför 0–100: {v!r}")
    for k in ("red_flags", "catalysts", "missing_data"):
        if k in d and not isinstance(d[k], list):
            errs.append(f"{k} är inte en lista")
    for k in ("explanations", "scenarios"):
        if k in d and not isinstance(d[k], dict):
            errs.append(f"{k} är inte ett dict")
    return errs
