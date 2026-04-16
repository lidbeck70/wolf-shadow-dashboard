"""
stocktwits.py — StockTwits sentiment source (replaces broken Twitter/snscrape).
Free API, no auth needed. Covers both US and some Nordic tickers.
API: https://api.stocktwits.com/api/2/streams/symbol/{TICKER}.json
Rate limit: 200 requests/hour. Cache results for 15 min.
"""
import logging
import requests
from retail_sentiment.models import SourceResult

logger = logging.getLogger(__name__)

STOCKTWITS_BASE = "https://api.stocktwits.com/api/2/streams/symbol"


def fetch_stocktwits() -> SourceResult:
    """Fetch StockTwits trending/overview data.
    Returns a SourceResult with trending symbols data.
    Includes retry logic and fallback to per-ticker fetch.
    """
    import time

    # Try trending endpoint with 1 retry
    for attempt in range(2):
        try:
            r = requests.get(
                f"{STOCKTWITS_BASE}/trending.json",
                timeout=10,
                headers={"User-Agent": "NordicAlpha/1.0"},
            )
            if r.status_code == 200:
                data = r.json()
                return SourceResult(
                    data={"trending": data},
                    confidence=1.0,
                    source="stocktwits",
                )
            elif r.status_code == 429 and attempt == 0:
                logger.info("StockTwits trending rate-limited, retrying in 2s...")
                time.sleep(2)
                continue
        except Exception as e:
            logger.warning("StockTwits trending fetch attempt %d failed: %s", attempt + 1, e)
            if attempt == 0:
                time.sleep(2)
                continue

    # Fallback: try a known ticker to verify API is up
    try:
        test = fetch_ticker_sentiment("SPY")
        if test.get("confidence", 0) > 0:
            logger.info("StockTwits trending failed but per-ticker API works — degraded mode")
            return SourceResult(
                data={"fallback": True},
                confidence=0.5,
                source="stocktwits",
            )
    except Exception as e:
        logger.warning("StockTwits fallback fetch also failed: %s", e)

    return SourceResult(data={}, confidence=0.0, source="stocktwits")


def fetch_ticker_sentiment(ticker: str) -> dict:
    """Fetch StockTwits sentiment for a specific ticker.

    StockTwits tickers don't use exchange suffixes.
    Strip .ST, .OL, .CO, .HE before querying.

    Returns dict with:
        - message_count: int
        - bullish: int
        - bearish: int
        - neutral: int
        - bull_ratio: float (0-1)
        - bear_ratio: float (0-1)
        - watchlist_count: int
        - sentiment_label: str ('Bullish'/'Bearish'/'Neutral')
    """
    # Strip Nordic exchange suffixes for StockTwits
    clean_ticker = ticker.split(".")[0]  # "EQNR.OL" -> "EQNR"
    # Also handle tickers like "SSAB-A" -> "SSAB"
    clean_ticker = clean_ticker.split("-")[0] if "-" in clean_ticker else clean_ticker

    try:
        r = requests.get(
            f"{STOCKTWITS_BASE}/{clean_ticker}.json",
            timeout=10,
            headers={"User-Agent": "NordicAlpha/1.0"},
        )
        if r.status_code == 200:
            data = r.json()
            symbol = data.get("symbol", {})
            messages = data.get("messages", [])

            bullish = 0
            bearish = 0
            for msg in messages:
                entities = msg.get("entities")
                if entities is None:
                    continue
                sentiment = entities.get("sentiment")
                if sentiment is None:
                    continue
                basic = sentiment.get("basic", "")
                if basic == "Bullish":
                    bullish += 1
                elif basic == "Bearish":
                    bearish += 1

            total = len(messages)
            neutral = total - bullish - bearish
            bull_ratio = bullish / total if total > 0 else 0.0
            bear_ratio = bearish / total if total > 0 else 0.0

            # Determine overall sentiment label
            if bull_ratio > 0.5:
                label = "Bullish"
            elif bear_ratio > 0.3:
                label = "Bearish"
            else:
                label = "Neutral"

            return {
                "message_count": total,
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
                "bull_ratio": round(bull_ratio, 3),
                "bear_ratio": round(bear_ratio, 3),
                "watchlist_count": symbol.get("watchlist_count", 0),
                "sentiment_label": label,
                "confidence": 1.0 if total >= 10 else (0.5 if total > 0 else 0.0),
            }
        elif r.status_code == 404:
            logger.debug("StockTwits: ticker %s not found", clean_ticker)
    except Exception as e:
        logger.warning("StockTwits fetch failed for %s: %s", ticker, e)

    return {
        "message_count": 0, "bullish": 0, "bearish": 0, "neutral": 0,
        "bull_ratio": 0.0, "bear_ratio": 0.0, "watchlist_count": 0,
        "sentiment_label": "Unknown", "confidence": 0.0,
    }


# Keep old function names for backward compatibility
fetch_twitter = fetch_stocktwits

def get_ticker_sentiment(twitter_result: SourceResult, ticker: str) -> dict:
    """Bridge function for backward compat with engine.py."""
    return fetch_ticker_sentiment(ticker)
