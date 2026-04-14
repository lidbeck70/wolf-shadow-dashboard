"""
models.py — Data models for the Retail Sentiment Engine.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourceResult:
    """Result from a single data source fetch."""
    data: Optional[dict] = None
    confidence: float = 0.0
    source: str = ""
    error: Optional[str] = None


@dataclass
class TickerReport:
    """Aggregated sentiment report for a single ticker."""
    ticker: str = ""
    timestamp: str = ""
    scores: dict = field(default_factory=dict)
    confidences: dict = field(default_factory=dict)
    data_sources_available: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    market: str = "US"
