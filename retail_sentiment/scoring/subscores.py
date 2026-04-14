"""
subscores.py — Individual source score calculations.
Each function returns a (score, confidence) tuple on a 0-100 scale.
"""
from retail_sentiment.scoring.normalize import normalize_z_to_100, normalize_value
from retail_sentiment.models import SourceResult


def reddit_score(reddit_result: SourceResult, ticker: str) -> tuple:
    """Score based on Reddit mention rank and volume.
    Returns (score: 0-100, confidence: 0-1).
    """
    if reddit_result.confidence == 0.0 or not reddit_result.data:
        return 0.0, 0.0

    tickers = reddit_result.data.get("tickers", {})
    info = tickers.get(ticker, {})
    if not info:
        return 0.0, 0.0

    mentions = info.get("mentions", 0)
    rank = info.get("rank", 50)

    # Rank score: rank 1 = 100, rank 50 = 0
    rank_score = max(0, 100 - (rank - 1) * 2)

    # Mentions score: normalize against typical range
    mention_score = normalize_value(mentions, 0, 500)

    # Blend: 60% rank, 40% mentions
    score = rank_score * 0.6 + mention_score * 0.4
    return round(score, 1), 1.0


def twitter_score(twitter_result: SourceResult, ticker: str) -> tuple:
    """Score based on Twitter/X sentiment (currently placeholder).
    Returns (score: 0-100, confidence: 0-1).
    """
    return 0.0, 0.0


def retail_flow_score(volume_data: dict, options_data: dict) -> tuple:
    """Score based on volume anomalies and options flow.
    Combines volume z-score with options PCR.
    Returns (score: 0-100, confidence: 0-1).
    """
    if not volume_data and not options_data:
        return 0.0, 0.0

    components = []
    confidences = []

    # Volume component
    if volume_data:
        vol_z = volume_data.get("volume_z", 0.0)
        vol_score = normalize_z_to_100(vol_z)
        vol_conf = volume_data.get("confidence", 0.5)
        components.append(("volume", vol_score, vol_conf))
        confidences.append(vol_conf)

    # Options component (PCR inverted: low PCR = bullish = high score)
    if options_data:
        pcr = options_data.get("pcr_volume", 0.0)
        # PCR range: 0.3 (very bullish) to 1.5 (very bearish)
        # Invert: low PCR = high score
        opts_score = normalize_value(1.5 - pcr, 0, 1.2)
        components.append(("options", opts_score, 1.0))
        confidences.append(1.0)

    if not components:
        return 0.0, 0.0

    # Weighted average of available components
    total_weight = sum(c[2] for c in components)
    if total_weight == 0:
        return 0.0, 0.0

    score = sum(c[1] * c[2] for c in components) / total_weight
    avg_conf = sum(confidences) / len(confidences)

    return round(score, 1), round(avg_conf, 2)


def yahoo_score(yahoo_result: SourceResult, ticker: str) -> tuple:
    """Score based on Yahoo trending presence.
    Binary: in trending list = 80, not in list = 20.
    Returns (score: 0-100, confidence: 0-1).
    """
    if yahoo_result.confidence == 0.0 or not yahoo_result.data:
        return 0.0, 0.0

    trending = yahoo_result.data.get("trending_tickers", [])
    is_trending = ticker in trending

    if is_trending:
        # Higher rank = higher score
        try:
            rank = trending.index(ticker)
            score = max(60, 100 - rank * 2)
        except ValueError:
            score = 80
    else:
        score = 20.0

    return round(score, 1), yahoo_result.confidence


def hype_overlap_score(
    reddit_result: SourceResult,
    yahoo_result: SourceResult,
    volume_data: dict,
    ticker: str,
) -> tuple:
    """Detect cross-source overlap — ticker appearing in multiple sources.
    More overlap sources = higher hype score.
    Returns (score: 0-100, confidence: 0-1).
    """
    overlap_count = 0
    total_sources = 0

    # Reddit presence
    if reddit_result.confidence > 0:
        total_sources += 1
        tickers = reddit_result.data.get("tickers", {}) if reddit_result.data else {}
        if ticker in tickers and tickers[ticker].get("rank", 999) <= 25:
            overlap_count += 1

    # Yahoo trending presence
    if yahoo_result.confidence > 0:
        total_sources += 1
        trending = yahoo_result.data.get("trending_tickers", []) if yahoo_result.data else []
        if ticker in trending:
            overlap_count += 1

    # Volume anomaly presence
    if volume_data:
        total_sources += 1
        if volume_data.get("volume_ratio", 0) > 1.5:
            overlap_count += 1

    if total_sources == 0:
        return 0.0, 0.0

    # Score based on fraction of sources with presence
    score = (overlap_count / max(total_sources, 1)) * 100
    confidence = min(1.0, total_sources / 3)

    return round(score, 1), round(confidence, 2)
