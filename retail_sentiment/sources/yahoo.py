"""
yahoo.py — Yahoo Finance trending tickers source.
"""
import logging
import requests

from retail_sentiment.models import SourceResult
from retail_sentiment.cache import get_cached, set_cached

logger = logging.getLogger(__name__)

YAHOO_TRENDING_URL = "https://query1.finance.yahoo.com/v1/finance/trending/US"
CACHE_KEY = "yahoo_trending"


def fetch_yahoo_trending() -> SourceResult:
    """Fetch trending tickers from Yahoo Finance API."""
    cached = get_cached(CACHE_KEY)
    if cached is not None:
        return cached

    try:
        r = requests.get(
            YAHOO_TRENDING_URL,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        quotes = r.json().get("finance", {}).get("result", [])
        tickers = []
        if quotes:
            tickers = [q["symbol"] for q in quotes[0].get("quotes", []) if "symbol" in q]

        result = SourceResult(
            data={"trending_tickers": tickers[:30]},
            confidence=1.0 if tickers else 0.0,
            source="yahoo",
        )
        set_cached(CACHE_KEY, result)
        return result

    except Exception as e:
        logger.warning("Yahoo trending fetch failed: %s", e)
        return SourceResult(
            data={"trending_tickers": []},
            confidence=0.0,
            source="yahoo",
            error=str(e),
        )


def is_ticker_trending(yahoo_result: SourceResult, ticker: str) -> bool:
    """Check if a ticker appears in Yahoo trending list."""
    if not yahoo_result.data:
        return False
    return ticker in yahoo_result.data.get("trending_tickers", [])
