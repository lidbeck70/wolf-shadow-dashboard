"""
confidence/data/provenance.py — en datapunkt bär sitt ursprung.

Datapoint(value, kind, source, source_type, pub_date, data_date, confidence)
är minsta enheten i modellen. kind skiljer ACTUAL / ESTIMATE / GUIDANCE /
MODELLED / ASSUMPTION; source_type mappar till Data Quality-poängen.
value=None betyder DATA_MISSING — aldrig ett påhittat tal.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Any, Optional

from confidence.config import (FRESHNESS_OLD, FRESHNESS_TABLE, FRESHNESS_UNKNOWN,
                               KINDS, SOURCE_QUALITY)


@dataclass(frozen=True)
class Datapoint:
    value: Any = None
    kind: str = "ASSUMPTION"       # ACTUAL | ESTIMATE | GUIDANCE | MODELLED | ASSUMPTION
    source: str = ""               # "Företagets FS 2025", "Börsdata", "43-101 (SRK)"
    source_type: str = ""          # primary | independent | secondary | mixed | weak | unsupported
    pub_date: Optional[str] = None   # ÅÅÅÅ-MM-DD
    data_date: Optional[str] = None  # ÅÅÅÅ-MM-DD (perioden talet avser)
    confidence: str = ""           # high | medium | low (t.ex. från extraktionen)
    unit: str = ""
    note: str = ""

    @property
    def missing(self) -> bool:
        return self.value is None or self.value == ""

    def as_dict(self) -> dict:
        return asdict(self)


def dp(value: Any = None, **meta) -> Datapoint:
    """Kort konstruktor: dp(180_000, kind="GUIDANCE", source="Årsredovisning", ...)."""
    kind = str(meta.pop("kind", "ASSUMPTION")).upper()
    if kind not in KINDS:
        raise ValueError(f"okänd datatyp {kind!r}, giltiga: {KINDS}")
    return Datapoint(value=value, kind=kind, **meta)


def from_dict(d: Any) -> Datapoint:
    """Datapoint ur lagrad dict (eller ett rått värde utan proveniens →
    ASSUMPTION utan källa, så det syns i Confidence)."""
    if isinstance(d, Datapoint):
        return d
    if isinstance(d, dict) and "value" in d:
        keys = {"value", "kind", "source", "source_type", "pub_date", "data_date",
                "confidence", "unit", "note"}
        clean = {k: v for k, v in d.items() if k in keys}
        kind = str(clean.get("kind") or "ASSUMPTION").upper()
        clean["kind"] = kind if kind in KINDS else "ASSUMPTION"
        return Datapoint(**clean)
    return Datapoint(value=d)


def num(p: Optional[Datapoint]) -> Optional[float]:
    """Talet i datapunkten, eller None (aldrig ett gissat tal)."""
    if p is None or p.missing:
        return None
    try:
        v = float(p.value)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def truth(p: Optional[Datapoint]) -> Optional[bool]:
    """Booleskt fält: True/False, eller None när det inte är bedömt."""
    if p is None or p.missing:
        return None
    v = p.value
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "ja", "yes", "1"):
        return True
    if s in ("false", "nej", "no", "0"):
        return False
    return None


def months_old(p: Optional[Datapoint], today: Optional[date] = None) -> Optional[float]:
    """Ålder i månader för datapunkten (data_date före pub_date). None = okänd."""
    if p is None:
        return None
    raw = p.data_date or p.pub_date
    if not raw:
        return None
    try:
        d = date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None
    today = today or date.today()
    return max(0.0, (today - d).days / 30.44)


def freshness_points(p: Optional[Datapoint], today: Optional[date] = None) -> int:
    """Data Quality 'Freshness' 0–5 enligt config.FRESHNESS_TABLE."""
    m = months_old(p, today)
    if m is None:
        return FRESHNESS_UNKNOWN
    for limit, pts in FRESHNESS_TABLE:
        if m < limit:
            return pts
    return FRESHNESS_OLD


def source_points(p: Optional[Datapoint]) -> int:
    """Data Quality 'Source quality' 0–5 enligt config.SOURCE_QUALITY."""
    if p is None or p.missing:
        return 0
    return SOURCE_QUALITY.get(str(p.source_type or "").lower(), 0)


def describe(p: Optional[Datapoint], label: str = "") -> str:
    """Spårbar klartext: 'Kopparproduktion: 180 000 t (GUIDANCE, Årsredovisning, 2026-03-15, high)'."""
    if p is None or p.missing:
        return f"{label}: DATA_MISSING" if label else "DATA_MISSING"
    bits = [p.kind]
    if p.source:
        bits.append(p.source)
    if p.data_date or p.pub_date:
        bits.append(str(p.data_date or p.pub_date))
    if p.confidence:
        bits.append(p.confidence)
    v = p.value
    vs = f"{v:,.6g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
    unit = f" {p.unit}" if p.unit else ""
    head = f"{label}: " if label else ""
    return f"{head}{vs}{unit} ({', '.join(bits)})"
