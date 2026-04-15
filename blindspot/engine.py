"""
engine.py — Orchestration for Odin's Blindspot Index.
Uses ThreadPoolExecutor for parallel data fetching (Streamlit-compatible).
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from blindspot.config import BLINDSPOT_TICKERS
from blindspot.models import BlindspotReport
from blindspot.sources.price import fetch_price_data
from blindspot.sources.fundamentals import fetch_fundamentals
from blindspot.classification.classifier import map_necessity, get_industry
from blindspot.scoring.hat import calculate_hat_score
from blindspot.scoring.strength import calculate_strength_score
from blindspot.scoring.catalyst import calculate_catalyst_score
from blindspot.scoring.opportunity import calculate_opportunity
from blindspot.history import append_report
from blindspot.config import detect_market

logger = logging.getLogger(__name__)

MAX_WORKERS = 4


def _fetch_sentiment_data(ticker: str) -> dict:
    """Try to get sentiment data from retail_sentiment sources."""
    sentiment = {}
    try:
        from retail_sentiment.sources.reddit import fetch_reddit, get_ticker_mentions
        reddit_result = fetch_reddit()
        mentions = get_ticker_mentions(reddit_result, ticker)
        sentiment["reddit_mentions"] = mentions.get("mentions", 0)
    except Exception:
        sentiment["reddit_mentions"] = 0

    try:
        from retail_sentiment.sources.volume import fetch_volume, get_ticker_volume
        vol_result = fetch_volume([ticker])
        vol_data = get_ticker_volume(vol_result, ticker)
        sentiment["retail_flow_score"] = 50  # Neutral default
        if vol_data:
            vol_ratio = vol_data.get("volume_ratio", 1.0)
            if vol_ratio < 0.5:
                sentiment["retail_flow_score"] = 10
            elif vol_ratio < 0.8:
                sentiment["retail_flow_score"] = 30
    except Exception:
        pass

    return sentiment


def _fetch_all_data(tickers: list) -> dict:
    """Fetch all data sources in parallel."""
    results = {"price": {}, "fundamentals": {}, "sentiment": {}}

    def _fetch_price():
        return "price", fetch_price_data(tickers)

    def _fetch_fund():
        return "fundamentals", fetch_fundamentals(tickers)

    tasks = [_fetch_price, _fetch_fund]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fn): fn.__name__ for fn in tasks}
        for future in as_completed(futures):
            try:
                name, data = future.result()
                results[name] = data
            except Exception as e:
                fn_name = futures[future]
                logger.warning("Data fetch %s failed: %s", fn_name, e)

    # Fetch sentiment per-ticker (lightweight, uses cached reddit data)
    for ticker in tickers:
        try:
            results["sentiment"][ticker] = _fetch_sentiment_data(ticker)
        except Exception:
            results["sentiment"][ticker] = {}

    return results


def build_ticker_report(ticker: str, all_data: dict) -> BlindspotReport:
    """Build a complete blindspot report for a single ticker."""
    timestamp = datetime.utcnow().isoformat()
    market = detect_market(ticker)

    price_data = all_data.get("price", {}).get(ticker, {})
    fund_data = all_data.get("fundamentals", {}).get(ticker, {})
    sent_data = all_data.get("sentiment", {}).get(ticker, {})

    # Classification
    necessity, necessity_conf, sector_label = map_necessity(
        ticker, all_data.get("fundamentals", {})
    )
    _, industry = get_industry(ticker, all_data.get("fundamentals", {}))

    # Scoring
    hat, hat_breakdown = calculate_hat_score(price_data, sent_data)
    strength, strength_breakdown = calculate_strength_score(fund_data)
    catalyst, catalyst_breakdown = calculate_catalyst_score(price_data)

    # Confidence
    price_conf = price_data.get("confidence", 0.0)
    fund_conf = fund_data.get("confidence", 0.0)
    overall_conf = (price_conf + fund_conf + necessity_conf) / 3

    # Opportunity
    opportunity, flags = calculate_opportunity(
        hat, necessity, strength, catalyst, overall_conf, necessity_conf
    )

    report = BlindspotReport(
        ticker=ticker,
        timestamp=timestamp,
        sector=sector_label,
        industry=industry,
        market=market,
        close=price_data.get("close", 0),
        sma50=price_data.get("sma50", 0),
        sma200=price_data.get("sma200", 0),
        high_52w=price_data.get("high_52w", 0),
        low_52w=price_data.get("low_52w", 0),
        perf_6m=price_data.get("perf_6m", 0),
        perf_12m=price_data.get("perf_12m", 0),
        hat_score=hat,
        necessity_score=necessity,
        strength_score=strength,
        catalyst_score=catalyst,
        opportunity_score=opportunity,
        fcf=fund_data.get("fcf"),
        ebitda=fund_data.get("ebitda"),
        debt_to_equity=fund_data.get("debt_to_equity"),
        revenue_growth=fund_data.get("revenue_growth"),
        ev_ebitda=fund_data.get("ev_ebitda"),
        fcf_yield=fund_data.get("fcf_yield"),
        ebitda_margin=fund_data.get("ebitda_margin"),
        price_confidence=price_conf,
        fundamentals_confidence=fund_conf,
        necessity_confidence=necessity_conf,
        overall_confidence=overall_conf,
        flags=flags,
        hat_breakdown=hat_breakdown,
        strength_breakdown=strength_breakdown,
        catalyst_breakdown=catalyst_breakdown,
    )

    # Persist to history
    try:
        append_report(
            ticker, opportunity, hat, strength, catalyst, necessity, sector_label
        )
    except Exception:
        pass

    return report


def build_all_reports(tickers: list = None) -> dict:
    """Build blindspot reports for all tickers.

    Returns dict with:
        - reports: list of BlindspotReport sorted by opportunity desc
        - timestamp: when the run completed
    """
    if tickers is None:
        tickers = BLINDSPOT_TICKERS

    logger.info("Building blindspot reports for %d tickers", len(tickers))
    all_data = _fetch_all_data(tickers)

    reports = []
    for ticker in tickers:
        try:
            report = build_ticker_report(ticker, all_data)
            reports.append(report)
        except Exception as e:
            logger.warning("Failed to build blindspot report for %s: %s", ticker, e)

    reports.sort(key=lambda r: r.opportunity_score, reverse=True)

    return {
        "reports": reports,
        "timestamp": datetime.utcnow().isoformat(),
    }
