"""
engine.py — Orchestration for the Retail Sentiment Engine.
Uses ThreadPoolExecutor for parallel data fetching (Streamlit-compatible).
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from retail_sentiment.config import DEFAULT_TICKERS, detect_market
from retail_sentiment.models import SourceResult, TickerReport
from retail_sentiment.sources.reddit import fetch_reddit, get_ticker_mentions
from retail_sentiment.sources.twitter import fetch_twitter
from retail_sentiment.sources.yahoo import fetch_yahoo_trending
from retail_sentiment.sources.volume import fetch_volume, get_ticker_volume
from retail_sentiment.sources.options import fetch_options, get_ticker_options
from retail_sentiment.scoring.subscores import (
    reddit_score,
    twitter_score,
    retail_flow_score,
    yahoo_score,
    hype_overlap_score,
)
from retail_sentiment.scoring.composite import calculate_composite
from retail_sentiment.history import append_report

logger = logging.getLogger(__name__)

MAX_WORKERS = 5


def _fetch_all_sources(tickers: list) -> dict:
    """Fetch all data sources in parallel using ThreadPoolExecutor."""
    results = {}

    def _fetch_reddit():
        return "reddit", fetch_reddit()

    def _fetch_twitter():
        return "twitter", fetch_twitter()

    def _fetch_yahoo():
        return "yahoo", fetch_yahoo_trending()

    def _fetch_volume():
        return "volume", fetch_volume(tickers)

    def _fetch_options():
        return "options", fetch_options(tickers)

    tasks = [_fetch_reddit, _fetch_twitter, _fetch_yahoo, _fetch_volume, _fetch_options]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fn): fn.__name__ for fn in tasks}
        for future in as_completed(futures):
            try:
                name, result = future.result()
                results[name] = result
            except Exception as e:
                fn_name = futures[future]
                logger.warning("Source fetch %s failed: %s", fn_name, e)
                results[fn_name.replace("_fetch_", "")] = SourceResult(
                    confidence=0.0, source=fn_name, error=str(e)
                )

    return results


def build_ticker_report(ticker: str, sources: dict) -> TickerReport:
    """Build a complete sentiment report for a single ticker."""
    market = detect_market(ticker)
    timestamp = datetime.utcnow().isoformat()

    reddit_res = sources.get("reddit", SourceResult(source="reddit"))
    twitter_res = sources.get("twitter", SourceResult(source="twitter"))
    yahoo_res = sources.get("yahoo", SourceResult(source="yahoo"))
    volume_res = sources.get("volume", SourceResult(source="volume"))
    options_res = sources.get("options", SourceResult(source="options"))

    # Get per-ticker data from bulk results
    vol_data = get_ticker_volume(volume_res, ticker)
    opts_data = get_ticker_options(options_res, ticker)

    # Calculate subscores
    r_score, r_conf = reddit_score(reddit_res, ticker)
    t_score, t_conf = twitter_score(twitter_res, ticker)
    rf_score, rf_conf = retail_flow_score(vol_data, opts_data)
    y_score, y_conf = yahoo_score(yahoo_res, ticker)
    h_score, h_conf = hype_overlap_score(reddit_res, yahoo_res, vol_data, ticker)

    scores = {
        "reddit": r_score,
        "twitter": t_score,
        "retail_flow": rf_score,
        "yahoo": y_score,
        "hype_overlap": h_score,
    }
    confidences = {
        "reddit": r_conf,
        "twitter": t_conf,
        "retail_flow": rf_conf,
        "yahoo": y_conf,
        "hype_overlap": h_conf,
    }

    composite = calculate_composite(scores, confidences)
    scores["composite"] = composite

    # Track which sources are available
    available = [s for s, c in confidences.items() if c > 0]

    # Metadata
    metadata = {
        "volume_ratio": vol_data.get("volume_ratio", 0.0),
        "price": vol_data.get("price", 0.0),
        "price_change_pct": vol_data.get("price_change_pct", 0.0),
        "reddit_mentions": get_ticker_mentions(reddit_res, ticker).get("mentions", 0),
        "reddit_rank": get_ticker_mentions(reddit_res, ticker).get("rank", 0),
    }

    report = TickerReport(
        ticker=ticker,
        timestamp=timestamp,
        scores=scores,
        confidences=confidences,
        data_sources_available=available,
        metadata=metadata,
        market=market,
    )

    # Persist to history
    append_report(ticker, composite, scores, market)

    return report


def build_all_reports(tickers: list = None) -> dict:
    """Build sentiment reports for all tickers.

    Returns dict with:
        - reports: list of TickerReport
        - sources: raw source results for UI display
        - timestamp: when the run completed
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    logger.info("Building sentiment reports for %d tickers", len(tickers))
    sources = _fetch_all_sources(tickers)

    reports = []
    for ticker in tickers:
        try:
            report = build_ticker_report(ticker, sources)
            reports.append(report)
        except Exception as e:
            logger.warning("Failed to build report for %s: %s", ticker, e)

    # Sort by composite score descending
    reports.sort(key=lambda r: r.scores.get("composite", 0), reverse=True)

    return {
        "reports": reports,
        "sources": sources,
        "timestamp": datetime.utcnow().isoformat(),
    }
