"""
strength.py — Fundamental Strength Score (0-100).
Measures underlying business quality despite market hate.

6 components:
- FCF 3y consistency (max 30p)
- EBITDA margin (max 15p)
- FCF yield (max 15p)
- Debt/Equity (max 20p)
- Revenue growth (max 15p)
- EV/EBITDA bonus (max 5p)
"""
import logging

logger = logging.getLogger(__name__)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def calculate_strength_score(fundamentals: dict) -> tuple:
    """Calculate fundamental strength score (0-100).

    Args:
        fundamentals: Dict with fcf, fcf_history, ebitda, ebitda_margin,
                      fcf_yield, debt_to_equity, revenue_growth, ev_ebitda.

    Returns:
        (strength_score, breakdown_dict)
    """
    if not fundamentals:
        return 0.0, {}

    breakdown = {}

    # 1. FCF 3-year consistency (max 30p)
    fcf_score = 0.0
    fcf_history = fundamentals.get("fcf_history", [])
    fcf = fundamentals.get("fcf")

    if fcf_history and len(fcf_history) >= 3:
        # All 3 years positive = 30p, 2/3 = 20p, 1/3 = 10p
        positive_years = sum(1 for f in fcf_history if f and f > 0)
        fcf_score = positive_years / 3 * 30
    elif fcf_history and len(fcf_history) >= 1:
        # Single year available — partial credit
        if fcf_history[0] and fcf_history[0] > 0:
            fcf_score = 15.0  # Half credit for single positive year
    elif fcf and fcf > 0:
        fcf_score = 10.0  # Minimal credit
    breakdown["fcf_consistency"] = round(fcf_score, 1)

    # 2. EBITDA margin (max 15p)
    ebitda_margin_score = 0.0
    ebitda_margin = fundamentals.get("ebitda_margin")
    if ebitda_margin is not None:
        # 0% = 0p, 15% = 7.5p, 30%+ = 15p
        ebitda_margin_score = _clamp(ebitda_margin / 30 * 15, 0, 15)
    breakdown["ebitda_margin"] = round(ebitda_margin_score, 1)

    # 3. FCF yield (max 15p)
    fcf_yield_score = 0.0
    fcf_yield = fundamentals.get("fcf_yield")
    if fcf_yield is not None and fcf_yield > 0:
        # 0% = 0p, 5% = 7.5p, 10%+ = 15p
        fcf_yield_score = _clamp(fcf_yield / 10 * 15, 0, 15)
    breakdown["fcf_yield"] = round(fcf_yield_score, 1)

    # 4. Debt/Equity (max 20p) — lower debt = higher score
    de_score = 0.0
    de = fundamentals.get("debt_to_equity")
    if de is not None:
        if de <= 0.3:
            de_score = 20.0  # Very low debt
        elif de <= 0.5:
            de_score = 16.0
        elif de <= 1.0:
            de_score = 12.0
        elif de <= 1.5:
            de_score = 8.0
        elif de <= 2.0:
            de_score = 4.0
        else:
            de_score = 0.0  # Heavily leveraged
    else:
        de_score = 5.0  # Unknown — small benefit of doubt
    breakdown["debt_equity"] = round(de_score, 1)

    # 5. Revenue growth (max 15p)
    rev_growth_score = 0.0
    rev_growth = fundamentals.get("revenue_growth")
    if rev_growth is not None:
        if rev_growth > 0:
            # 0% = 0p, 10% = 7.5p, 20%+ = 15p
            rev_growth_score = _clamp(rev_growth / 20 * 15, 0, 15)
        elif rev_growth > -10:
            # Slight decline — small penalty
            rev_growth_score = 2.0
    breakdown["revenue_growth"] = round(rev_growth_score, 1)

    # 6. EV/EBITDA bonus (max 5p) — low = undervalued
    ev_ebitda_score = 0.0
    ev_ebitda = fundamentals.get("ev_ebitda")
    if ev_ebitda is not None and ev_ebitda > 0:
        if ev_ebitda <= 6:
            ev_ebitda_score = 5.0  # Deep value
        elif ev_ebitda <= 10:
            ev_ebitda_score = 3.0
        elif ev_ebitda <= 15:
            ev_ebitda_score = 1.0
    breakdown["ev_ebitda_bonus"] = round(ev_ebitda_score, 1)

    total = (
        fcf_score + ebitda_margin_score + fcf_yield_score +
        de_score + rev_growth_score + ev_ebitda_score
    )
    total = _clamp(total, 0, 100)

    return round(total, 1), breakdown
