"""
engines/durrett/_base.py — gemensamma byggstenar för stegen.

Ctx         läser bolagets fält med bokföring av saknat, normaliserar
            enheter (oz/lb/t, valuta → USD) och håller beräknade nyckeltal.
score_from  väger komponenter (0–100 eller None) till en Score: vikterna
            omfördelas över de komponenter som finns; är för lite känt
            blir poängen N/A med reason — aldrig 0 för okänt.
steps_ge/le trappor ur config.score_steps.
"""

from __future__ import annotations

from typing import Optional

from confidence import commodities as com
from confidence.data.models import CompanyInput
from confidence.data.provenance import describe
from confidence.data.validation import usable
from engines.durrett.config import DURRETT_CONFIG
from engines.durrett.models import Score

MIN_COMPONENT_COVERAGE = 0.5      # VAL: under 50 % av vikten känd → N/A

# Enheter → basenhet (oz, lb, t)
_UNIT_FACTOR = {"oz": ("oz", 1.0), "koz": ("oz", 1e3), "Moz": ("oz", 1e6),
                "lb": ("lb", 1.0), "Mlb": ("lb", 1e6),
                "t": ("t", 1.0), "kt": ("t", 1e3), "Mt": ("t", 1e6)}
# Råvarans prisenhet → basenhet för mängd
_PRICE_UNIT_BASE = {"USD/oz": "oz", "USD/lb": "lb", "USD/lb U3O8": "lb", "USD/t": "t", "USD/t LCE": "t",
                    "USD/kg NdPr": "kg", "USD/boe": "boe", "USD/MMBtu": "MMBtu", "USD/ct": "ct"}


class Ctx:
    """Läsning med bokföring. missing samlar fältnycklar som saknades."""

    def __init__(self, company: CompanyInput, config: Optional[dict] = None):
        self.c = company
        self.cfg = config or DURRETT_CONFIG
        self.missing: list = []
        self.metrics: dict = {}
        self.log: list = []
        self.commodity = com.get(company.commodity)
        self.price_unit = self.commodity.unit if self.commodity else ""
        self.base_unit = _PRICE_UNIT_BASE.get(self.price_unit, "")

    # ── fält ────────────────────────────────────────────────────────────
    def num(self, key: str, required: bool = True) -> Optional[float]:
        if usable(self.c, key):
            return self.c.num(key)
        if required and key not in self.missing:
            self.missing.append(key)
        return None

    def truth(self, key: str, required: bool = True) -> Optional[bool]:
        if usable(self.c, key):
            return self.c.truth(key)
        if required and key not in self.missing:
            self.missing.append(key)
        return None

    def text(self, key: str) -> Optional[str]:
        if usable(self.c, key):
            return str(self.c.get(key).value)
        return None

    def src(self, key: str, label: str = "") -> str:
        return describe(self.c.get(key), label)

    def metric(self, name: str, value, unit: str = "", note: str = "") -> None:
        self.metrics[name] = {"value": value, "unit": unit, "note": note}

    # ── enheter ─────────────────────────────────────────────────────────
    def quantity(self, key: str, required: bool = True) -> Optional[float]:
        """Fältet i basenhet (oz/lb/t) enligt resource_unit."""
        v = self.num(key, required)
        if v is None:
            return None
        unit = self.text("resource_unit")
        if unit is None:
            pu = (self.text("production_unit") or "").strip()
            unit = pu if pu in _UNIT_FACTOR else None
        if unit is None:
            if "resource_unit" not in self.missing:
                self.missing.append("resource_unit")
            return None
        base, f = _UNIT_FACTOR.get(unit, (None, None))
        if base is None:
            self.log.append(f"okänd enhet {unit!r} för {key}")
            return None
        if self.base_unit and base != self.base_unit:
            self.log.append(f"enhetskonflikt: {key} i {unit} men råvaran prissätts i {self.price_unit}")
            return None
        return v * f

    def usd(self, key: str, required: bool = True) -> Optional[float]:
        """Belopp i USD: fält × fx_to_usd när market_currency ≠ USD."""
        v = self.num(key, required)
        if v is None:
            return None
        cur = self.text("market_currency") or "USD"
        if cur == "USD":
            return v
        fx = self.num("fx_to_usd", required=False)
        if fx is None:
            if "fx_to_usd" not in self.missing:
                self.missing.append("fx_to_usd")
            self.log.append(f"{key} i {cur} utan växelkurs — N/A")
            return None
        return v * fx


# ── trappor ─────────────────────────────────────────────────────────────────
def steps_ge(v: Optional[float], table, beyond: float = 0.0) -> Optional[float]:
    """Tröskel fallande: v ≥ tröskel → p; under alla → beyond."""
    if v is None:
        return None
    for thr, pts in table:
        if v >= thr:
            return float(pts)
    return float(beyond)


def steps_le(v: Optional[float], table, beyond: float = 0.0) -> Optional[float]:
    """Tröskel stigande: v ≤ tröskel → p; över alla → beyond."""
    if v is None:
        return None
    for thr, pts in table:
        if v <= thr:
            return float(pts)
    return float(beyond)


def steps_lt(v: Optional[float], table, beyond: float = 0.0) -> Optional[float]:
    if v is None:
        return None
    for thr, pts in table:
        if v < thr:
            return float(pts)
    return float(beyond)


def scale_0_2(v: Optional[float]) -> Optional[float]:
    """0/1/2-bedömning → 10/50/90 (VAL)."""
    if v is None:
        return None
    return {0: 10.0, 1: 50.0, 2: 90.0}.get(int(max(0, min(2, v))), 50.0)


# ── sammanvägning ───────────────────────────────────────────────────────────
def score_from(key: str, label: str, components: dict, weights: dict, ctx: Optional[Ctx] = None,
               min_coverage: float = MIN_COMPONENT_COVERAGE) -> Score:
    """components {namn: (0–100 | None, förklaring)}; weights {namn: vikt}."""
    s = Score(key, label, None)
    known_w = 0.0
    total_w = 0.0
    acc = 0.0
    for name, w in weights.items():
        total_w += w
        val, why = components.get(name, (None, "DATA_MISSING"))
        s.components[name] = (val, w)
        if val is None:
            s.unknown.append(f"? {name}: {why}")
            continue
        known_w += w
        acc += val * w
        (s.positive if val >= 50 else s.negative).append(f"{'+' if val >= 50 else '−'} {name} {val:g}: {why}")
    if total_w <= 0 or known_w / total_w < min_coverage:
        s.reason = f"för lite känt ({known_w / total_w * 100:.0f} % av vikten)" if total_w else "inga komponenter"
    else:
        s.value = round(acc / known_w, 1)
        if known_w < total_w:
            s.unknown.append(f"vikter omfördelade över {known_w / total_w * 100:.0f} % känd vikt")
    if ctx is not None:
        s.missing = [k for k in ctx.missing]
    return s


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))
