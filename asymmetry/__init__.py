"""
asymmetry — Wolf Asymmetry: Commodity Leverage, Margin of Safety,
break-even-marginal, stressmatris och confidence-justerad uppsida ovanpå
confidence-lagrets bolag och scenario-motor.

    from asymmetry import analyze
    r = analyze(company, confidence_total=78)
    r.leverage.score, r.safety.total, r.break_even.band, r.matrix.cell(-20, 20)

Inga egna inmatningar: allt läses ur CompanyInput (data/confidence.json).
Saknat underlag blir DATA_MISSING, aldrig noll.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from confidence.data.models import CompanyInput

from asymmetry.config import ASYMMETRY_CONFIG
from asymmetry.leverage import LeverageResult, commodity_leverage
from asymmetry.model import Inputs
from asymmetry.safety import SafetyResult, margin_of_safety
from asymmetry.stress import (BreakEven, StressMatrix, adjusted_upside, break_even_margin, scenario_asymmetry,
                              scenarios, stress_matrix)


@dataclass
class AsymmetryResult:
    ticker: str
    stage: str
    leverage: LeverageResult
    safety: SafetyResult
    break_even: BreakEven
    matrix: StressMatrix
    base_upside_pct: Optional[float]
    confidence: Optional[float]
    adjusted_upside_pct: Optional[float]
    assumptions: list
    missing: list
    scenarios: list = field(default_factory=list)   # [confidence.scenarios.engine.Scenario]
    asymmetry: object = None                        # confidence.scenarios.engine.Asymmetry | None

    def scenario(self, key: str):
        return next((s for s in self.scenarios if s.key == key), None)


def analyze(company: CompanyInput, confidence_total: Optional[float] = None) -> AsymmetryResult:
    inp = Inputs(company)
    lev = commodity_leverage(inp)
    saf = margin_of_safety(inp)
    be = break_even_margin(inp)
    mx = stress_matrix(inp)
    scen = scenarios(inp)
    base = inp.point(0.0)
    seen, uniq = set(), []
    for a in inp.assumptions:
        if a.name not in seen:
            seen.add(a.name)
            uniq.append(a)
    return AsymmetryResult(company.ticker, company.stage, lev, saf, be, mx, base.upside_pct, confidence_total,
                           adjusted_upside(base.upside_pct, confidence_total), uniq, inp.missing,
                           scen, scenario_asymmetry(scen))


__all__ = ["analyze", "AsymmetryResult", "ASYMMETRY_CONFIG", "Inputs"]
