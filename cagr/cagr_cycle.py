"""
cagr_cycle.py
Cycle scoring module — 0 to 3 points.

Redesigned: replaces 36 manual checkboxes with a clean per-sector
assessment using a single slider (0–3) per sector.

Auto-compute option uses market data to estimate cycle position:
  - Sector P/E vs 10-year average (undervalued?)
  - Capex trend (underinvestment?)
  - Sector ETF distance from 52-week high (sentiment?)

Manual override always available.
"""

from __future__ import annotations

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Sector definitions with ETF proxies for auto-compute
# ---------------------------------------------------------------------------

SECTOR_CONFIG: Dict[str, dict] = {
    "Energy": {
        "etf": "XLE",
        "default_score": 3,
        "description": "Oil & gas, energy services",
        "thesis": "Underinvestment cycle, supply constrained",
    },
    "Materials": {
        "etf": "XLB",
        "default_score": 2,
        "description": "Mining, chemicals, metals",
        "thesis": "Commodity supercycle thesis, China reopening",
    },
    "Industrials": {
        "etf": "XLI",
        "default_score": 1,
        "description": "Machinery, defence, transport",
        "thesis": "Infrastructure spending, defence budgets",
    },
    "Consumer Discretionary": {
        "etf": "XLY",
        "default_score": 0,
        "description": "Retail, autos, luxury",
        "thesis": "Consumer under pressure from rates",
    },
    "Consumer Staples": {
        "etf": "XLP",
        "default_score": 1,
        "description": "Food, beverages, household",
        "thesis": "Defensive, stable but limited upside",
    },
    "Healthcare": {
        "etf": "XLV",
        "default_score": 1,
        "description": "Pharma, biotech, medtech",
        "thesis": "Demographic tailwind, innovation",
    },
    "Financials": {
        "etf": "XLF",
        "default_score": 2,
        "description": "Banks, insurance, fintech",
        "thesis": "Higher rates support NIM, undervalued",
    },
    "Technology": {
        "etf": "XLK",
        "default_score": 0,
        "description": "Software, semiconductors, hardware",
        "thesis": "Fully valued, AI hype cycle",
    },
    "Communication Services": {
        "etf": "XLC",
        "default_score": 0,
        "description": "Telecom, media, social",
        "thesis": "Mixed — telecom defensive, media cyclical",
    },
    "Utilities": {
        "etf": "XLU",
        "default_score": 2,
        "description": "Power, water, renewables",
        "thesis": "Undervalued, nuclear renaissance",
    },
    "Real Estate": {
        "etf": "XLRE",
        "default_score": 2,
        "description": "REITs, property developers",
        "thesis": "Rate-sensitive, bottoming cycle",
    },
    "ETF": {
        "etf": "SPY",
        "default_score": 1,
        "description": "Broad market ETFs",
        "thesis": "Market-neutral assessment",
    },
    "Unknown": {
        "etf": "SPY",
        "default_score": 1,
        "description": "Unclassified",
        "thesis": "Default neutral",
    },
}

# Legacy compat — old code references DEFAULT_CYCLE with bool keys
DEFAULT_CYCLE: Dict[str, dict] = {}
for _sector, _cfg in SECTOR_CONFIG.items():
    _score = _cfg["default_score"]
    DEFAULT_CYCLE[_sector] = {
        "sector_undervalued": _score >= 2,
        "underinvestment":    _score >= 3,
        "sentiment_low":      _score >= 1,
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
# Score labels
# ---------------------------------------------------------------------------

SCORE_LABELS = {
    0: ("BEARISH",  "#ff3355"),
    1: ("NEUTRAL",  "#ffdd00"),
    2: ("BULLISH",  "#00ff88"),
    3: ("STRONG",   "#00ffff"),
}

def score_label(score: int) -> tuple:
    """Return (label, color) for a cycle score 0-3."""
    return SCORE_LABELS.get(max(0, min(3, score)), ("NEUTRAL", "#ffdd00"))


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


def score_cycle_from_int(score_int: int) -> dict:
    """
    Convert a single integer score (0-3) into the standard cycle result dict.
    Maps integer back to the three boolean criteria for compatibility.
    """
    score_int = max(0, min(3, score_int))
    return score_cycle(
        sector_undervalued=(score_int >= 2),
        underinvestment=(score_int >= 3),
        sentiment_low=(score_int >= 1),
    )


def get_default_cycle_for_sector(sector: str) -> dict:
    """
    Return the default cycle assessment dict for a given sector.
    Falls back to "Unknown" if sector not found.
    """
    return dict(DEFAULT_CYCLE.get(sector, DEFAULT_CYCLE["Unknown"]))


def score_cycle_for_sector(sector: str, overrides: dict | None = None) -> dict:
    """
    Convenience: apply default cycle for sector, then apply any manual overrides.

    Supports two override styles:
      1. Old-style: {"sector_undervalued": bool, "underinvestment": bool, "sentiment_low": bool}
      2. New-style: {"cycle_score": int}  (0-3 slider value)

    Returns same structure as score_cycle().
    """
    # New-style: direct integer score
    if overrides and "cycle_score" in overrides:
        return score_cycle_from_int(overrides["cycle_score"])

    # Old-style: boolean overrides
    base = get_default_cycle_for_sector(sector)
    if overrides:
        base.update(overrides)
    return score_cycle(
        sector_undervalued=base["sector_undervalued"],
        underinvestment=base["underinvestment"],
        sentiment_low=base["sentiment_low"],
    )
