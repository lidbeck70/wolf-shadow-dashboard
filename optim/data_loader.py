"""
data_loader.py — WOLF x SHADOW Optimization Pipeline
=====================================================
Loads OHLCV data from CSV files or yfinance.
Supports resampling, data quality validation, and joblib caching.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache setup
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(os.environ.get("WOLF_CACHE_DIR", "/tmp/wolf_cache"))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
memory = Memory(location=str(_CACHE_DIR), verbose=0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_and_clean(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalise column names, set DatetimeIndex, forward-fill minor gaps."""
    df.columns = [c.strip().lower() for c in df.columns]

    # Accept 'datetime' or 'date' as the timestamp column
    for ts_col in ("datetime", "date", "timestamp", "time"):
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=False)
            df = df.set_index(ts_col)
            break

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"[{ticker}] Could not find a datetime column in CSV.")

    df = df.sort_index()

    # Standardise OHLCV column names
    rename_map = {
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
        "adj close": "close", "adj_close": "close",
    }
    df = df.rename(columns=rename_map)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{ticker}] Missing columns: {missing}")

    df = df[required].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def _validate(df: pd.DataFrame, ticker: str, max_gap_bars: int = 5) -> pd.DataFrame:
    """
    Data quality checks:
      1. Drop rows with NaN in OHLCV
      2. Warn about price anomalies (zero / negative)
      3. Detect calendar gaps > max_gap_bars and forward-fill
    """
    n_before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning("[%s] Dropped %d NaN rows", ticker, n_dropped)

    # Price sanity
    bad_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    if bad_price:
        logger.warning("[%s] %d bars with zero/negative prices — removing", ticker, bad_price)
        df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]

    # OHLC consistency: high >= low
    inconsistent = (df["high"] < df["low"]).sum()
    if inconsistent:
        logger.warning("[%s] %d bars where high < low — swapping", ticker, inconsistent)
        mask = df["high"] < df["low"]
        df.loc[mask, ["high", "low"]] = df.loc[mask, ["low", "high"]].values

    # Volume: allow zero but not negative
    df["volume"] = df["volume"].clip(lower=0)
    df["volume"] = df["volume"].fillna(0)

    # Forward-fill gaps up to max_gap_bars
    if len(df) > 1:
        freq = pd.infer_freq(df.index[:50]) if len(df) >= 50 else None
        if freq:
            full_idx = pd.date_range(df.index[0], df.index[-1], freq=freq)
            n_gaps = len(full_idx) - len(df)
            if n_gaps > 0 and n_gaps < max_gap_bars * 100:
                df = df.reindex(full_idx)
                df["close"] = df["close"].ffill()
                df["open"] = df["open"].ffill()
                df["high"] = df["high"].ffill()
                df["low"] = df["low"].ffill()
                df["volume"] = df["volume"].fillna(0)
                logger.info("[%s] Forward-filled %d gap bars", ticker, n_gaps)

    logger.info("[%s] Loaded %d clean bars from %s to %s",
                ticker, len(df), df.index[0].date(), df.index[-1].date())
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv(filepath: str | Path, ticker: str = "TICKER") -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file.

    Expected columns (case-insensitive): date/datetime, open, high, low, close, volume
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")

    df = pd.read_csv(filepath)
    df = _parse_and_clean(df, ticker)
    df = _validate(df, ticker)
    return df


@memory.cache
def _download_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Cached yfinance download."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance>=0.2.36")

    logger.info("[%s] Downloading from yfinance (period=%s, interval=%s)", ticker, period, interval)
    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")

    # yfinance returns MultiIndex columns when downloading multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    raw.columns = [c.lower() for c in raw.columns]
    raw.index.name = "date"
    df = _parse_and_clean(raw.reset_index(), ticker)
    df = _validate(df, ticker)
    return df


def load_yfinance(
    ticker: str,
    years: int = 5,
    interval: str = "1h",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV data via yfinance with caching.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. "XOM", "EQNR.OL")
    years : int
        Lookback in years (converted to yfinance period string)
    interval : str
        Data frequency: "1h", "4h", "1d", etc.
    force_download : bool
        Bypass joblib cache

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
    """
    # yfinance period string
    period = f"{years}y"

    # yfinance only supports hourly data for last ~730 days
    if interval in ("1h", "60m") and years > 2:
        logger.warning(
            "[%s] yfinance limits 1h data to ~2 years; requesting max available", ticker
        )
        period = "730d"

    if force_download:
        _download_yfinance.cache_clear()

    return _download_yfinance(ticker, period, interval)


def load_data(
    ticker: str,
    csv_dir: Optional[str | Path] = None,
    years: int = 5,
    interval: str = "1h",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Unified loader: tries CSV first, falls back to yfinance.

    CSV filename patterns tried (in order):
      {csv_dir}/{ticker}.csv
      {csv_dir}/{ticker.lower()}.csv
      {csv_dir}/{ticker}_{interval}.csv

    Parameters
    ----------
    ticker : str
    csv_dir : path to directory containing CSVs (optional)
    years : int
    interval : str
    force_download : bool

    Returns
    -------
    pd.DataFrame
    """
    if csv_dir is not None:
        csv_dir = Path(csv_dir)
        candidates = [
            csv_dir / f"{ticker}.csv",
            csv_dir / f"{ticker.lower()}.csv",
            csv_dir / f"{ticker}_{interval}.csv",
            csv_dir / f"{ticker.upper()}.csv",
        ]
        for path in candidates:
            if path.exists():
                logger.info("[%s] Loading from CSV: %s", ticker, path)
                return load_csv(path, ticker)
        logger.info("[%s] No CSV found in %s — falling back to yfinance", ticker, csv_dir)

    return load_yfinance(ticker, years=years, interval=interval, force_download=force_download)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

_RESAMPLE_OHLCV = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Resample OHLCV DataFrame to a coarser timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        Source data with DatetimeIndex
    target_tf : str
        Target timeframe: "4H", "1D", "1W", etc.
        Accepts Pine-Script-style strings: "4h" -> "4H", "1d" -> "1D"

    Returns
    -------
    pd.DataFrame  (resampled, NaN rows dropped)
    """
    # Normalise string
    _map = {"1h": "1H", "4h": "4H", "1d": "1D", "1D": "1D", "1w": "1W", "1W": "1W",
            "D": "1D", "W": "1W", "H": "1H"}
    tf = _map.get(target_tf, target_tf)

    resampled = df.resample(tf).agg(_RESAMPLE_OHLCV)
    resampled = resampled.dropna(subset=["close"])
    return resampled


# ---------------------------------------------------------------------------
# Market data helpers for regime scoring (SPY + sector ETF)
# ---------------------------------------------------------------------------

def load_market_data(
    tickers: list[str],
    years: int = 5,
    interval: str = "1h",
    csv_dir: Optional[str | Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load multiple tickers and return a dict {ticker: DataFrame}.
    Used to load SPY and sector ETFs alongside stock data.
    """
    result: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            result[t] = load_data(t, csv_dir=csv_dir, years=years, interval=interval)
        except Exception as exc:
            logger.error("[%s] Failed to load: %s", t, exc)
    return result


def align_to_stock(stock_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex ref_df to match stock_df's DatetimeIndex using forward-fill.
    This mirrors Pine Script's barmerge.gaps_off / lookahead_off behaviour.
    """
    aligned = ref_df.reindex(stock_df.index, method="ffill")
    return aligned


# ---------------------------------------------------------------------------
# Utility: cache key
# ---------------------------------------------------------------------------

def df_hash(df: pd.DataFrame) -> str:
    """Compute a short hash of a DataFrame for cache keying."""
    h = hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:12]
    return h


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick smoke test
    df = load_yfinance("XOM", years=2, interval="1h")
    print(df.tail())
    print("Shape:", df.shape)
    df4h = resample(df, "4H")
    print("4H shape:", df4h.shape)
    df1d = resample(df, "1D")
    print("1D shape:", df1d.shape)
