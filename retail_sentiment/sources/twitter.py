"""
twitter.py — Twitter/X sentiment source (PLACEHOLDER).
snscrape is deprecated since X changed their API in 2023.
Structure ready for future API integration (SocialGrep, Apify, etc).
"""
import logging

from retail_sentiment.models import SourceResult

logger = logging.getLogger(__name__)


def fetch_twitter() -> SourceResult:
    """Placeholder: always returns confidence=0.
    The composite scorer will auto-reweight around this missing source.
    """
    logger.info("Twitter source unavailable (snscrape deprecated). Returning confidence=0.")
    return SourceResult(
        data={"tickers": {}, "note": "snscrape deprecated, awaiting replacement API"},
        confidence=0.0,
        source="twitter",
    )


def get_ticker_sentiment(twitter_result: SourceResult, ticker: str) -> dict:
    """Extract sentiment data for a specific ticker (currently empty)."""
    return {}
