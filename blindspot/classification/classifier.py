"""
classifier.py — Industry classification and necessity scoring.
Uses TICKER_OVERRIDES first, then maps via NECESSITY_MAP keywords.
"""
import logging

from blindspot.classification.sector_map import NECESSITY_MAP, TICKER_OVERRIDES

logger = logging.getLogger(__name__)


def get_industry(ticker: str, fundamentals: dict) -> tuple:
    """Get sector/industry for a ticker.

    Returns (sector_label, industry_label) from override or fundamentals.
    """
    if ticker in TICKER_OVERRIDES:
        label, _ = TICKER_OVERRIDES[ticker]
        return label, label

    fund = fundamentals.get(ticker, {})
    sector = fund.get("sector", "")
    industry = fund.get("industry", "")
    return sector, industry


def map_necessity(ticker: str, fundamentals: dict) -> tuple:
    """Map a ticker to its necessity score (0-100) and confidence.

    Returns (necessity_score, confidence, sector_label).
    """
    # Check ticker override first
    if ticker in TICKER_OVERRIDES:
        label, score = TICKER_OVERRIDES[ticker]
        return score, 1.0, label

    # Try to match from fundamentals
    fund = fundamentals.get(ticker, {})
    sector = fund.get("sector", "").lower()
    industry = fund.get("industry", "").lower()

    # Try industry first (more specific), then sector
    for text in [industry, sector]:
        if not text:
            continue
        for keyword, score in NECESSITY_MAP.items():
            if keyword in text:
                label = fund.get("industry", fund.get("sector", "Unknown"))
                return score, 0.8, label

    # Unknown — default to 30 with low confidence
    label = fund.get("industry", fund.get("sector", "Unmapped"))
    return 30, 0.3, label
