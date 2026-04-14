"""
reddit.py — ApeWisdom Reddit mentions source.
Fetches trending tickers from Reddit (WSB + all-stocks) via the free ApeWisdom API.
"""
import logging
import requests

from retail_sentiment.models import SourceResult
from retail_sentiment.cache import get_cached, set_cached

logger = logging.getLogger(__name__)

APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"
CACHE_KEY = "reddit_apewisdom"


def fetch_reddit() -> SourceResult:
    """Fetch trending tickers from ApeWisdom. Returns SourceResult with ranked ticker data."""
    cached = get_cached(CACHE_KEY)
    if cached is not None:
        return cached

    try:
        r = requests.get(
            APEWISDOM_URL,
            timeout=15,
            headers={"User-Agent": "NordicAlpha/2.0"},
        )
        r.raise_for_status()
        data = r.json().get("results", [])
        if not data:
            result = SourceResult(data={"tickers": {}}, confidence=0.0, source="reddit")
            set_cached(CACHE_KEY, result)
            return result

        tickers = {}
        for item in data[:50]:
            ticker = item.get("ticker", "")
            if ticker:
                tickers[ticker] = {
                    "mentions": item.get("mentions", 0),
                    "upvotes": item.get("upvotes", 0),
                    "rank": item.get("rank", 0),
                    "rank_24h_ago": item.get("rank_24h_ago", 0),
                    "name": item.get("name", ""),
                }

        result = SourceResult(
            data={"tickers": tickers, "raw_results": data[:50]},
            confidence=1.0 if tickers else 0.0,
            source="reddit",
        )
        set_cached(CACHE_KEY, result)
        return result

    except Exception as e:
        logger.warning("Reddit fetch failed: %s", e)
        return SourceResult(data={"tickers": {}}, confidence=0.0, source="reddit", error=str(e))


def get_ticker_mentions(reddit_result: SourceResult, ticker: str) -> dict:
    """Extract mention data for a specific ticker from reddit results."""
    if not reddit_result.data:
        return {}
    return reddit_result.data.get("tickers", {}).get(ticker, {})
