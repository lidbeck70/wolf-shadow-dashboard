"""
models.py — Data models for Odin's Blindspot Index.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BlindspotReport:
    """Complete contrarian opportunity report for a single ticker."""
    ticker: str = ""
    timestamp: str = ""
    sector: str = ""
    industry: str = ""
    market: str = "US"

    # Raw price data
    close: float = 0.0
    sma50: float = 0.0
    sma200: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0
    perf_6m: float = 0.0
    perf_12m: float = 0.0

    # Scores
    hat_score: float = 0.0
    necessity_score: float = 0.0
    strength_score: float = 0.0
    catalyst_score: float = 0.0
    opportunity_score: float = 0.0

    # Fundamental data
    fcf: Optional[float] = None
    ebitda: Optional[float] = None
    debt_to_equity: Optional[float] = None
    revenue_growth: Optional[float] = None
    ev_ebitda: Optional[float] = None
    fcf_yield: Optional[float] = None
    ebitda_margin: Optional[float] = None

    # Confidence
    price_confidence: float = 0.0
    fundamentals_confidence: float = 0.0
    necessity_confidence: float = 0.0
    overall_confidence: float = 0.0

    # Flags
    flags: dict = field(default_factory=dict)

    # Sub-score breakdown
    hat_breakdown: dict = field(default_factory=dict)
    strength_breakdown: dict = field(default_factory=dict)
    catalyst_breakdown: dict = field(default_factory=dict)
