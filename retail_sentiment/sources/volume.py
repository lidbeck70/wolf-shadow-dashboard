"""
volume.py — yfinance volume data source.
Fetches 1-month volume history and computes volume ratios / z-scores.
"""
import logging

from retail_sentiment.models import SourceResult
from retail_sentiment.cache import get_cached, set_cached

logger = logging.getLogger(__name__)


def fetch_volume(tickers: list) -> SourceResult:
    """Fetch volume data for a list of tickers via yfinance.
    Returns volume ratios and z-scores per ticker.
    """
    cache_key = f"volume_{'_'.join(sorted(tickers[:10]))}"
    cached = get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np

        ticker_data = {}
        # Download in batch for efficiency
        raw = yf.download(tickers, period="1mo", auto_adjust=True, progress=False, threads=True)

        if raw is None or raw.empty:
            return SourceResult(data={"tickers": {}}, confidence=0.0, source="volume")

        is_multi = isinstance(raw.columns, pd.MultiIndex)

        for ticker in tickers:
            try:
                if is_multi:
                    vol = raw.xs(ticker, level=1, axis=1)["Volume"].dropna()
                    close = raw.xs(ticker, level=1, axis=1)["Close"].dropna()
                else:
                    if len(tickers) == 1:
                        vol = raw["Volume"].dropna()
                        close = raw["Close"].dropna()
                    else:
                        continue

                if len(vol) < 5:
                    continue

                current_vol = float(vol.iloc[-1])
                avg_vol_20d = float(vol.tail(20).mean())
                std_vol_20d = float(vol.tail(20).std())

                vol_ratio = current_vol / avg_vol_20d if avg_vol_20d > 0 else 0.0
                vol_z = (current_vol - avg_vol_20d) / std_vol_20d if std_vol_20d > 0 else 0.0

                # Price change data
                price_change_pct = 0.0
                if len(close) >= 2:
                    price_change_pct = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)

                confidence = 1.0 if len(vol) >= 20 else 0.5

                ticker_data[ticker] = {
                    "current_volume": current_vol,
                    "avg_volume_20d": avg_vol_20d,
                    "std_volume_20d": std_vol_20d,
                    "volume_ratio": round(vol_ratio, 2),
                    "volume_z": round(vol_z, 2),
                    "price": round(float(close.iloc[-1]), 2) if len(close) > 0 else 0.0,
                    "price_change_pct": round(price_change_pct, 2),
                    "confidence": confidence,
                }
            except (KeyError, ValueError, IndexError) as e:
                logger.debug("Volume fetch failed for %s: %s", ticker, e)
                continue

        overall_conf = 1.0 if ticker_data else 0.0
        result = SourceResult(
            data={"tickers": ticker_data},
            confidence=overall_conf,
            source="volume",
        )
        set_cached(cache_key, result)
        return result

    except Exception as e:
        logger.warning("Volume fetch failed: %s", e)
        return SourceResult(data={"tickers": {}}, confidence=0.0, source="volume", error=str(e))


def get_ticker_volume(volume_result: SourceResult, ticker: str) -> dict:
    """Extract volume data for a specific ticker."""
    if not volume_result.data:
        return {}
    return volume_result.data.get("tickers", {}).get(ticker, {})
