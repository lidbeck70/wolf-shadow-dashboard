"""
asymmetry/stress.py — stressmatrisen pris × capex, break-even-marginalen
och den confidence-justerade uppsidan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from asymmetry import config as acfg
from asymmetry.config import ASYMMETRY_CONFIG as CFG
from asymmetry.model import Inputs, Point


@dataclass
class StressMatrix:
    price_pct: tuple
    capex_pct: tuple
    cells: dict                          # {(price_pct, capex_pct): Point}
    note: str = ""

    def cell(self, price_pct: float, capex_pct: float) -> Optional[Point]:
        return self.cells.get((price_pct, capex_pct))


def stress_matrix(inp: Inputs) -> StressMatrix:
    s = CFG["stress_matrix"]
    cells = {}
    for pp in s["price_pct"]:
        for cp in s["capex_pct"]:
            # producenter har ingen capex-axel: samma punkt i varje capex-kolumn
            cells[(pp, cp)] = inp.point(pp, capex_pct=cp if inp.pre_revenue else 0.0)
    note = "" if inp.pre_revenue else "Producent: capex-axeln påverkar inte (capex ligger i AISC)."
    return StressMatrix(tuple(s["price_pct"]), tuple(s["capex_pct"]), cells, note)


@dataclass
class BreakEven:
    price: Optional[float]
    break_even: Optional[float]
    margin_pct: Optional[float]
    band: str
    steps: list = field(default_factory=list)


def break_even_margin(inp: Inputs) -> BreakEven:
    b = CFG["break_even"]
    if inp.price is None or inp.be is None:
        return BreakEven(inp.price, inp.be, None, "DATA_MISSING",
                         ["DATA_MISSING: råvarupris eller break-even/AISC saknas"])
    margin = round((inp.price - inp.be) / inp.price * 100, 6)
    band = (acfg.BREAK_EVEN_STRONG if margin >= b["strong"] else
            acfg.BREAK_EVEN_MODERATE if margin >= b["moderate"] else acfg.BREAK_EVEN_WEAK)
    return BreakEven(inp.price, inp.be, margin, band,
                     [f"(pris {inp.price:g} − break-even {inp.be:g}) / pris = {margin:+.1f} %",
                      f"band: ≥ {b['strong']:g} % stark · ≥ {b['moderate']:g} % måttlig · annars svag"])


# ── Confidence-justerad uppsida ──────────────────────────────────────────────
def _linear(upside_pct: float, confidence: float) -> float:
    return upside_pct * max(0.0, min(100.0, confidence)) / 100.0


FORMULAS = {"linear": _linear}


def adjusted_upside(upside_pct: Optional[float], confidence: Optional[float]) -> Optional[float]:
    """Base-uppsida × confidence/100 (formeln väljs i config). None när något saknas."""
    if upside_pct is None or confidence is None:
        return None
    fn = FORMULAS[CFG["adjusted_upside"]["formula"]]
    return round(fn(float(upside_pct), float(confidence)), 6)
