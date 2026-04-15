"""
hat.py — Hat Score (0-100): How hated/neglected is this stock?
Higher = more hated = potentially better contrarian opportunity.

9 components with gradual scoring:
- SMA200 gap (max 25p)
- SMA50 gap (max 10p)
- 52w low proximity (max 15p)
- 6m performance (max 5p)
- 12m performance (max 5p)
- Volume drought (max 10p)
- Reddit neglect (max 10p)
- Twitter negativity (max 10p)
- Retail apathy (max 10p)
"""
import logging

logger = logging.getLogger(__name__)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def calculate_hat_score(price_data: dict, sentiment_data: dict = None) -> tuple:
    """Calculate the hat score (0-100) for a ticker.

    Args:
        price_data: Dict with close, sma50, sma200, high_52w, low_52w, etc.
        sentiment_data: Optional dict with reddit_mentions, twitter_sentiment, etc.

    Returns:
        (hat_score, breakdown_dict)
    """
    if not price_data:
        return 0.0, {}

    breakdown = {}

    close = price_data.get("close", 0)
    sma50 = price_data.get("sma50", 0)
    sma200 = price_data.get("sma200", 0)
    high_52w = price_data.get("high_52w", 0)
    low_52w = price_data.get("low_52w", 0)
    perf_6m = price_data.get("perf_6m", 0)
    perf_12m = price_data.get("perf_12m", 0)
    current_volume = price_data.get("current_volume", 0)
    avg_volume_20d = price_data.get("avg_volume_20d", 0)
    std_volume_20d = price_data.get("std_volume_20d", 0)

    # 1. SMA200 gap (max 25p) — how far below SMA200
    sma200_gap = 0.0
    if sma200 > 0 and close > 0:
        pct_below = (sma200 - close) / sma200 * 100
        # 0% below = 0p, 5% = 6.25p, 10% = 12.5p, 20%+ = 25p
        sma200_gap = _clamp(pct_below / 20 * 25, 0, 25)
    breakdown["sma200_gap"] = round(sma200_gap, 1)

    # 2. SMA50 gap (max 10p) — how far below SMA50
    sma50_gap = 0.0
    if sma50 > 0 and close > 0:
        pct_below = (sma50 - close) / sma50 * 100
        sma50_gap = _clamp(pct_below / 15 * 10, 0, 10)
    breakdown["sma50_gap"] = round(sma50_gap, 1)

    # 3. 52w low proximity (max 15p) — how close to 52w low
    low_proximity = 0.0
    if high_52w > low_52w > 0 and close > 0:
        position = (close - low_52w) / (high_52w - low_52w)
        # At 52w low (position=0) = 15p, at mid (0.5) = 0p
        low_proximity = _clamp((0.5 - position) / 0.5 * 15, 0, 15)
    breakdown["low_52w_proximity"] = round(low_proximity, 1)

    # 4. 6m performance (max 5p) — negative = hated
    perf_6m_score = 0.0
    if perf_6m < 0:
        perf_6m_score = _clamp(abs(perf_6m) / 30 * 5, 0, 5)
    breakdown["perf_6m"] = round(perf_6m_score, 1)

    # 5. 12m performance (max 5p) — negative = hated
    perf_12m_score = 0.0
    if perf_12m < 0:
        perf_12m_score = _clamp(abs(perf_12m) / 40 * 5, 0, 5)
    breakdown["perf_12m"] = round(perf_12m_score, 1)

    # 6. Volume drought (max 10p) — low volume = neglected
    vol_drought = 0.0
    if avg_volume_20d > 0 and current_volume > 0:
        vol_ratio = current_volume / avg_volume_20d
        # Ratio < 0.5 = max neglect (10p), ratio 1.0 = 0p
        if vol_ratio < 1.0:
            vol_drought = _clamp((1.0 - vol_ratio) / 0.5 * 10, 0, 10)
    breakdown["vol_drought"] = round(vol_drought, 1)

    # 7-9. Sentiment components (max 10p each)
    reddit_neglect = 0.0
    twitter_negativity = 0.0
    retail_apathy = 0.0

    if sentiment_data:
        # Reddit neglect — not mentioned = hated/forgotten
        reddit_mentions = sentiment_data.get("reddit_mentions", 0)
        if reddit_mentions == 0:
            reddit_neglect = 10.0
        elif reddit_mentions < 5:
            reddit_neglect = 7.0
        elif reddit_mentions < 20:
            reddit_neglect = 3.0
        breakdown["reddit_neglect"] = round(reddit_neglect, 1)

        # Twitter negativity — placeholder, use 5p default if no data
        twitter_sent = sentiment_data.get("twitter_sentiment")
        if twitter_sent is not None:
            if twitter_sent < -0.3:
                twitter_negativity = 10.0
            elif twitter_sent < 0:
                twitter_negativity = _clamp(abs(twitter_sent) / 0.3 * 10, 0, 10)
        else:
            twitter_negativity = 5.0  # Default: assume moderate neglect
        breakdown["twitter_negativity"] = round(twitter_negativity, 1)

        # Retail apathy — no retail interest from sources
        retail_flow = sentiment_data.get("retail_flow_score", 50)
        if retail_flow < 30:
            retail_apathy = 10.0
        elif retail_flow < 50:
            retail_apathy = _clamp((50 - retail_flow) / 20 * 10, 0, 10)
        breakdown["retail_apathy"] = round(retail_apathy, 1)
    else:
        # No sentiment data — use moderate defaults
        reddit_neglect = 7.0
        twitter_negativity = 5.0
        retail_apathy = 5.0
        breakdown["reddit_neglect"] = reddit_neglect
        breakdown["twitter_negativity"] = twitter_negativity
        breakdown["retail_apathy"] = retail_apathy

    total = (
        sma200_gap + sma50_gap + low_proximity +
        perf_6m_score + perf_12m_score + vol_drought +
        reddit_neglect + twitter_negativity + retail_apathy
    )
    total = _clamp(total, 0, 100)

    return round(total, 1), breakdown
