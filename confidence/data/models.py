"""
confidence/data/models.py — analysens objekt.

CompanyInput  identitet + ett fält-dict {nyckel: Datapoint} (nycklarna är
              config.FIELDS). Det är det enda som lagras (inmatningar).
PillarScore   en pelare/del: poäng, max, komponenter, förklaringar, saknat.
CaseScore     åtta pelare → 0–100 + betyg. Discovery Option separat.
ConfidenceScore  sju delar → råpoäng → kill-caps → 0–100 + band.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from confidence.config import FIELD_BY_KEY, STAGES
from confidence.data.provenance import Datapoint, from_dict, num, truth


@dataclass
class CompanyInput:
    ticker: str
    name: str = ""
    commodity: str = ""            # nyckel i confidence.commodities (t.ex. "copper")
    secondary_commodity: str = ""
    country: str = ""
    jurisdiction: str = ""         # region/stat/provins, t.ex. "Quebec"
    stage: str = "developer"       # config.STAGES
    maturity: str = "pea"          # config.MATURITY
    exchange: str = ""
    ins_id: Optional[int] = None   # Börsdata-id när bolaget finns där
    fields: dict = field(default_factory=dict)   # {key: Datapoint}
    notes: str = ""

    # ── läsning ───────────────────────────────────────────────────────────
    def get(self, key: str) -> Optional[Datapoint]:
        p = self.fields.get(key)
        return from_dict(p) if p is not None else None

    def num(self, key: str) -> Optional[float]:
        return num(self.get(key))

    def truth(self, key: str) -> Optional[bool]:
        return truth(self.get(key))

    def has(self, key: str) -> bool:
        p = self.get(key)
        return p is not None and not p.missing

    def set(self, key: str, point: Datapoint) -> None:
        if key not in FIELD_BY_KEY:
            raise KeyError(f"okänt fält {key!r} — lägg till det i config.FIELDS")
        self.fields[key] = point

    # ── lagring ───────────────────────────────────────────────────────────
    def as_dict(self) -> dict:
        d = asdict(self)
        d["fields"] = {k: (v.as_dict() if isinstance(v, Datapoint) else v)
                       for k, v in self.fields.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CompanyInput":
        d = dict(d or {})
        raw_fields = d.pop("fields", {}) or {}
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        obj.fields = {k: from_dict(v) for k, v in raw_fields.items()}
        if obj.stage not in STAGES:
            obj.stage = "developer"
        return obj


@dataclass
class PillarScore:
    key: str
    label: str
    points: float
    max: float
    components: dict = field(default_factory=dict)   # delpoäng {namn: poäng}
    notes: list = field(default_factory=list)        # förklaring i klartext, en rad per komponent
    missing: list = field(default_factory=list)      # fältnycklar som saknades
    caps: list = field(default_factory=list)         # tak som slog till (stress etc.)

    @property
    def pct(self) -> float:
        return round(self.points / self.max * 100, 1) if self.max else 0.0


@dataclass
class CaseScore:
    total: float
    rating: str
    pillars: list                                    # [PillarScore] i config.PILLARS ordning
    discovery_option: Optional[float] = None         # explorers, separat
    missing: list = field(default_factory=list)      # alla saknade fält, unika
    flags: list = field(default_factory=list)

    def pillar(self, key: str) -> Optional[PillarScore]:
        return next((p for p in self.pillars if p.key == key), None)


@dataclass
class ConfidenceScore:
    total: float                                     # efter kill-caps
    band: str
    raw_total: float                                 # före caps
    parts: list                                      # [PillarScore] i config.CONFIDENCE_PARTS ordning
    caps_applied: list = field(default_factory=list)  # [(nyckel, tak, text)]
    missing: list = field(default_factory=list)
    flags: list = field(default_factory=list)

    def part(self, key: str) -> Optional[PillarScore]:
        return next((p for p in self.parts if p.key == key), None)


def to_jsonable(obj: Any) -> Any:
    """dataclass-träd → dict/list för JSON (Gist, tester)."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
