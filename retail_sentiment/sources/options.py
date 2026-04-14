"""
options.py — EODHD options data source (optional).
Requires EODHD_API_KEY in st.secrets or os.environ.
If no key: returns confidence=0 immediately.
Nordic tickers: always confidence=0 (no options data available).
"""
import logging
import os
import requests

from retail_sentiment.models import SourceResult
from retail_sentiment.config import detect_market
from retail_sentiment.cache import get_cached, set_cached

logger = logging.getLogger(__name__)

EODHD_OPTIONS_URL = "https://eodhd.com/api/options/{ticker}.US"


def _get_api_key() -> str:
    """Try st.secrets first, then os.environ."""
    try:
        import streamlit as st
        return st.secrets.get("EODHD_API_KEY", "")
    except Exception:
        pass
    return os.environ.get("EODHD_API_KEY", "")


def fetch_options(tickers: list) -> SourceResult:
    """Fetch options data from EODHD for US tickers.
    Returns PCR, options volume, and z-scores.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.info("EODHD_API_KEY not set. Options source returning confidence=0.")
        return SourceResult(
            data={"tickers": {}},
            confidence=0.0,
            source="options",
            error="No EODHD_API_KEY configured",
        )

    cache_key = f"options_{'_'.join(sorted(tickers[:10]))}"
    cached = get_cached(cache_key)
    if cached is not None:
        return cached

    ticker_data = {}
    for ticker in tickers:
        if detect_market(ticker) == "NORDIC":
            continue

        try:
            url = EODHD_OPTIONS_URL.format(ticker=ticker)
            r = requests.get(
                url,
                params={"api_token": api_key, "fmt": "json"},
                timeout=10,
            )
            r.raise_for_status()
            options_data = r.json()

            if not options_data:
                continue

            # Calculate PCR and volume from options chain
            total_call_vol = 0
            total_put_vol = 0
            total_call_oi = 0
            total_put_oi = 0

            # EODHD returns nested structure with expirations
            if isinstance(options_data, dict):
                for _exp_date, chain in options_data.items():
                    if not isinstance(chain, dict):
                        continue
                    for option in chain.get("calls", []):
                        total_call_vol += option.get("volume", 0) or 0
                        total_call_oi += option.get("openInterest", 0) or 0
                    for option in chain.get("puts", []):
                        total_put_vol += option.get("volume", 0) or 0
                        total_put_oi += option.get("openInterest", 0) or 0

            total_vol = total_call_vol + total_put_vol
            pcr_volume = total_put_vol / total_call_vol if total_call_vol > 0 else 0.0
            pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0.0

            ticker_data[ticker] = {
                "pcr_volume": round(pcr_volume, 3),
                "pcr_oi": round(pcr_oi, 3),
                "total_options_volume": total_vol,
                "total_call_volume": total_call_vol,
                "total_put_volume": total_put_vol,
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
            }

        except Exception as e:
            logger.debug("Options fetch failed for %s: %s", ticker, e)
            continue

    overall_conf = 1.0 if ticker_data else 0.0
    result = SourceResult(
        data={"tickers": ticker_data},
        confidence=overall_conf,
        source="options",
    )
    set_cached(cache_key, result)
    return result


def get_ticker_options(options_result: SourceResult, ticker: str) -> dict:
    """Extract options data for a specific ticker."""
    if not options_result.data:
        return {}
    return options_result.data.get("tickers", {}).get(ticker, {})
