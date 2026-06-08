"""
market_cycle/engine.py
======================
Evaluate all 14 cycle phases against computed indicators and return
the winning phase with confidence score and matched/unmatched conditions.
"""

from __future__ import annotations

import math
from typing import Any

from market_cycle.rules import MARKET_CYCLE_RULES, PHASE_ORDER


def _eval_condition(indicator_val: Any, op: str, value: Any) -> bool:
    """Return True if indicator_val satisfies the condition."""
    if indicator_val is None:
        return False
    if isinstance(indicator_val, float) and math.isnan(indicator_val):
        return False
    if op == "gt":
        return indicator_val > value
    if op == "gte":
        return indicator_val >= value
    if op == "lt":
        return indicator_val < value
    if op == "lte":
        return indicator_val <= value
    if op == "between":
        lo, hi = value
        return lo <= indicator_val <= hi
    return False


def detect_market_cycle(indicators: dict) -> dict:
    """
    Score all 14 phases against the indicator dict.

    Returns
    -------
    {
        "phase":        str,          # winning phase name
        "confidence":   float,        # 0-100, % of conditions met (weighted)
        "phase_scores": dict,         # score per phase
        "matched_rules": {
            "matched":   list,        # conditions met for winning phase
            "unmatched": list,        # conditions not met (includes actual value)
        },
    }
    """
    phase_scores: dict[str, float] = {}
    all_breakdowns: dict[str, dict] = {}

    for phase in PHASE_ORDER:
        config = MARKET_CYCLE_RULES[phase]
        conditions = config["conditions"]
        weight = config.get("weight", 1.0)
        matched, unmatched = [], []

        for cond in conditions:
            field = cond["field"]
            op = cond["op"]
            value = cond["value"]
            actual = indicators.get(field)

            if _eval_condition(actual, op, value):
                matched.append(dict(cond))
            else:
                entry = dict(cond)
                if actual is not None:
                    entry["actual"] = round(actual, 4)
                else:
                    entry["actual"] = None
                unmatched.append(entry)

        n_total = len(conditions)
        raw_score = (len(matched) / n_total * 100) if n_total > 0 else 0.0
        phase_scores[phase] = round(raw_score * weight, 2)
        all_breakdowns[phase] = {"matched": matched, "unmatched": unmatched}

    best_phase = max(phase_scores, key=lambda p: phase_scores[p])
    raw_confidence = phase_scores[best_phase]
    # Normalise to 0-100 accounting for weight > 1.0
    max_weight = max(c.get("weight", 1.0) for c in MARKET_CYCLE_RULES.values())
    confidence = min(raw_confidence / max_weight, 100.0)

    return {
        "phase": best_phase,
        "confidence": round(confidence, 1),
        "phase_scores": phase_scores,
        "matched_rules": all_breakdowns[best_phase],
    }
