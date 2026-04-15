"""
price.py — yfinance price data source for Odin's Blindspot Index.
Fetches 1y history and computes: close, SMA50/200, 52w high/low,
perf 6m/12m, volume stats, ATR14, SMA50 slope.
"""
import logging

import numpy as np

from blindspot.cache import get_cached, set_cached

logger = logging.getLogger(__name__)


def fetch_price_data(tickers: list) -> dict:
    """Fetch price data for all tickers. Returns {ticker: price_dict}."""
    cache_key = f"blindspot_price_{'_'.join(sorted(tickers[:10]))}"
    cached = get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        import yfinance as yf

        results = {}
        raw = yf.download(
            tickers, period="1y", auto_adjust=True, progress=False, threads=True
        )

        if raw is None or raw.empty:
            return results

        import pandas as pd
        is_multi = isinstance(raw.columns, pd.MultiIndex)

        for ticker in tickers:
            try:
                if is_multi:
                    close = raw.xs(ticker, level=1, axis=1)["Close"].dropna()
                    high = raw.xs(ticker, level=1, axis=1)["High"].dropna()
                    low = raw.xs(ticker, level=1, axis=1)["Low"].dropna()
                    vol = raw.xs(ticker, level=1, axis=1)["Volume"].dropna()
                elif len(tickers) == 1:
                    close = raw["Close"].dropna()
                    high = raw["High"].dropna()
                    low = raw["Low"].dropna()
                    vol = raw["Volume"].dropna()
                else:
                    continue

                if len(close) < 20:
                    continue

                current_close = float(close.iloc[-1])

                # SMAs
                sma50 = float(close.tail(50).mean()) if len(close) >= 50 else current_close
                sma200 = float(close.tail(200).mean()) if len(close) >= 200 else current_close

                # 52-week high/low
                high_52w = float(high.max())
                low_52w = float(low.min())

                # Performance
                perf_6m = 0.0
                if len(close) >= 126:
                    perf_6m = float((current_close / close.iloc[-126] - 1) * 100)

                perf_12m = 0.0
                if len(close) >= 252:
                    perf_12m = float((current_close / close.iloc[0] - 1) * 100)

                # Volume stats
                current_volume = float(vol.iloc[-1]) if len(vol) > 0 else 0.0
                avg_volume_20d = float(vol.tail(20).mean()) if len(vol) >= 20 else 0.0
                std_volume_20d = float(vol.tail(20).std()) if len(vol) >= 20 else 0.0

                # ATR 14
                atr_14 = 0.0
                if len(close) >= 15 and len(high) >= 15 and len(low) >= 15:
                    tr_vals = []
                    for i in range(-14, 0):
                        h = float(high.iloc[i])
                        l = float(low.iloc[i])
                        prev_c = float(close.iloc[i - 1])
                        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
                        tr_vals.append(tr)
                    atr_14 = float(np.mean(tr_vals))

                # SMA50 slope (linear regression over last 10 days)
                sma50_slope = 0.0
                if len(close) >= 60:
                    sma50_series = close.rolling(50).mean().dropna()
                    if len(sma50_series) >= 10:
                        recent_sma = sma50_series.tail(10).values
                        x = np.arange(10)
                        try:
                            slope = float(np.polyfit(x, recent_sma, 1)[0])
                            sma50_slope = slope
                        except (np.linalg.LinAlgError, ValueError):
                            pass

                confidence = 1.0 if len(close) >= 200 else 0.7

                results[ticker] = {
                    "close": round(current_close, 2),
                    "sma50": round(sma50, 2),
                    "sma200": round(sma200, 2),
                    "high_52w": round(high_52w, 2),
                    "low_52w": round(low_52w, 2),
                    "perf_6m": round(perf_6m, 2),
                    "perf_12m": round(perf_12m, 2),
                    "current_volume": current_volume,
                    "avg_volume_20d": avg_volume_20d,
                    "std_volume_20d": std_volume_20d,
                    "atr_14": round(atr_14, 4),
                    "sma50_slope": round(sma50_slope, 4),
                    "confidence": confidence,
                }
            except (KeyError, ValueError, IndexError) as e:
                logger.debug("Price fetch failed for %s: %s", ticker, e)
                continue

        set_cached(cache_key, results)
        return results

    except Exception as e:
        logger.warning("Price fetch failed: %s", e)
        return {}
