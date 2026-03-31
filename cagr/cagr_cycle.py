"""
cagr_cycle.py
Cycle scoring module — 0 to 3 points.

The three binary inputs represent macro/cycle assessments:
  1. sector_undervalued  → 1 point
  2. underinvestment     → 1 point (capex cycle near trough)
  3. sentiment_low       → 1 point (crowd pessimism = contrarian buy)
"""

from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Default cycle assessments per sector
# These are opinionated starting values; the UI lets users override them.
# ---------------------------------------------------------------------------

DEFAULT_CYCLE: Dict[str, dict] = {
    "Energy": {
        "sector_undervalued": True,
        "underinvestment": True,
        "sentiment_low": True,
    },
    "Materials": {
        "sector_undervalued": True,
        "underinvestment": True,
        "sentiment_low": False,
    },
    "Industrials": {
        "sector_undervalued": False,
        "underinvestment": False,
        "sentiment_low": False,
    },
    "Consumer Discretionary": {
        "sector_undervalued": False,
        "underinvestment": False,
        "sentiment_low": False,
    },
    "Consumer Staples": {
        "sector_undervalued": False,
        "underinvestment": False,
        "sentiment_low": False,
    },
    "Healthcare": {
        "sector_undervalued": False,
        "underinvestment": False,
        "sentiment_low": False,
    },
    "Financials": {
        "sector_undervalued": True,
        "underinvestment": False,
        "sentiment_low": True,
    },
    "Technology": {
        "sector_undervalued": False,
        "underinvestment": False,
        "sentiment_low": False,
    },
    "Communication Services": {
        "sector_undervalued": False,
        "underinvestment": False,
        "sentiment_low": False,
    },
    "Utilities": {
        "sector_undervalued": True,
        "underinvestment": False,
        "sentiment_low": True,
    },
    "Real Estate": {
        "sector_undervalued": True,
        "underinvestment": False,
        "sentiment_low": True,
    },
    "Unknown": {
        "sector_undervalued": False,
        "underinvestment": False,
        "sentiment_low": False,
    },
}

# Human-readable labels for the UI
CYCLE_LABELS: Dict[str, str] = {
    "sector_undervalued": "Sector Undervalued",
    "underinvestment": "Underinvestment Cycle",
    "sentiment_low": "Sentiment Low",
}

CYCLE_DESCRIPTIONS: Dict[str, str] = {
    "sector_undervalued": (
        "Sector trades at a discount to its historical average "
        "on EV/EBITDA or P/B basis."
    ),
    "underinvestment": (
        "Capital expenditure has been below replacement for 2+ years, "
        "creating future supply constraints."
    ),
    "sentiment_low": (
        "Analyst consensus is bearish/neutral and fund positioning is light "
        "— classic contrarian setup."
    ),
}


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_cycle(
    sector_undervalued: bool,
    underinvestment: bool,
    sentiment_low: bool,
) -> dict:
    """
    Score a stock/sector on three binary cycle criteria.

    Parameters
    ----------
    sector_undervalued : bool
    underinvestment    : bool
    sentiment_low      : bool

    Returns
    -------
    dict with keys:
      - cycle_score : int (0-3)
      - details     : dict with criterion → {"value": bool, "pass": bool}
    """
    criteria = {
        "Sector Undervalued": sector_undervalued,
        "Underinvestment Cycle": underinvestment,
        "Sentiment Low": sentiment_low,
    }

    score = sum(1 for v in criteria.values() if v)

    details = {
        k: {"value": v, "pass": v}
        for k, v in criteria.items()
    }

    return {
        "cycle_score": score,
        "details": details,
    }


def get_default_cycle_for_sector(sector: str) -> dict:
    """
    Return the default cycle assessment dict for a given sector.
    Falls back to "Unknown" if sector not found.
    """
    return dict(DEFAULT_CYCLE.get(sector, DEFAULT_CYCLE["Unknown"]))


def score_cycle_for_sector(sector: str, overrides: dict | None = None) -> dict:
    """
    Convenience: apply default cycle for sector, then apply any manual overrides.

    Parameters
    ----------
    sector    : str  — sector name matching DEFAULT_CYCLE keys
    overrides : dict — optional partial override of the three boolean keys

    Returns
    -------
    Same structure as score_cycle()
    """
    base = get_default_cycle_for_sector(sector)
    if overrides:
        base.update(overrides)
    return score_cycle(
        sector_undervalued=base["sector_undervalued"],
        underinvestment=base["underinvestment"],
        sentiment_low=base["sentiment_low"],
    )
