"""
screener_ovtlyr.py
OVTLYR-style quantitative screener with z-score normalization.

Scoring model (weighted composite):
  Trend      30%  — EMA alignment (10/20/50/200) + MACD signal
  Momentum   25%  — RSI zone + 5/20-day price change
  Volatility 15%  — ATR relative to price + hist vol percentile
  Volume     15%  — Volume vs 20-day SMA ratio
  ADX        15%  — Trend strength (14-period)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try Börsdata first
try:
    from borsdata_api import get_api, BorsdataAPI, _get_nordic_tickers
    _HAS_BORSDATA = True
except ImportError:
    try:
        from dashboard.borsdata_api import get_api, BorsdataAPI, _get_nordic_tickers
        _HAS_BORSDATA = True
    except ImportError:
        _HAS_BORSDATA = False
        _get_nordic_tickers = None

# ── Ticker universes ──────────────────────────────────────────────────
# Import from existing modules (used as fallback)
try:
    from cagr.cagr_loader import NORDIC_TICKERS as _FALLBACK_NORDIC
except ImportError:
    _FALLBACK_NORDIC = {}

try:
    from heatmap.heatmap_streamlit import US_TICKERS, CANADA_TICKERS
except ImportError:
    US_TICKERS = {}
    CANADA_TICKERS = {}


def _build_nordic_universe() -> Dict[str, dict]:
    """Build Nordic universe from Börsdata API with fallback to hardcoded list."""
    try:
        if _get_nordic_tickers is not None:
            dynamic_tickers = _get_nordic_tickers()
            if dynamic_tickers:
                # Build a dict compatible with the existing universe format
                return {t: {"name": t.split(".")[0], "sector": "Unknown", "country": "Nordic"} for t in dynamic_tickers}
    except Exception:
        pass
    return _FALLBACK_NORDIC


# Backwards-compatible alias
NORDIC_TICKERS = _FALLBACK_NORDIC

UNIVERSES = {
    "Nordic": None,  # Resolved dynamically in run_ovtlyr_screener
    "US": US_TICKERS,
    "Canada": CANADA_TICKERS,
}


# ── Indicator computation ─────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    fast = _ema(close, 12)
    slow = _ema(close, 26)
    macd_line = fast - slow
    signal = _ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return (100 - (100 / (1 + rs))).fillna(50)

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=close.index)

    plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, float("nan"))
    minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, float("nan"))

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, float("nan"))
    adx = dx.rolling(period).mean()
    return adx.fillna(0)


# ── Scoring functions ─────────────────────────────────────────────────

def _score_trend(df: pd.DataFrame) -> float:
    """Score 0-100 based on EMA alignment + MACD."""
    close = df["Close"]
    score = 0

    ema10 = _ema(close, 10).iloc[-1]
    ema20 = _ema(close, 20).iloc[-1]
    ema50 = _ema(close, 50).iloc[-1]
    ema200 = _ema(close, 200).iloc[-1] if len(close) >= 200 else _ema(close, len(close)).iloc[-1]
    price = close.iloc[-1]

    # EMA stack alignment (0-60)
    if price > ema10 > ema20 > ema50 > ema200:
        score += 60  # Perfect bullish stack
    elif price > ema50 > ema200:
        score += 40  # Strong bullish
    elif price > ema200:
        score += 20  # Above 200

    # MACD (0-40)
    _, _, macd_hist = _macd(close)
    if len(macd_hist) > 1:
        hist_now = macd_hist.iloc[-1]
        hist_prev = macd_hist.iloc[-2]
        if hist_now > 0 and hist_now > hist_prev:
            score += 40  # Bullish and accelerating
        elif hist_now > 0:
            score += 25  # Bullish but decelerating
        elif hist_now > hist_prev:
            score += 15  # Bearish but improving

    return min(100, max(0, score))

def _score_momentum(df: pd.DataFrame) -> float:
    """Score 0-100 based on RSI + price change."""
    close = df["Close"]
    rsi_val = _rsi(close, 14).iloc[-1]

    score = 0

    # RSI zone (0-50)
    if 50 < rsi_val < 70:
        score += 50  # Sweet spot
    elif 40 < rsi_val <= 50:
        score += 35  # Neutral-bullish
    elif rsi_val >= 70:
        score += 20  # Overbought — risky
    elif 30 < rsi_val <= 40:
        score += 25  # Neutral-bearish
    else:
        score += 10  # Oversold

    # Price change (0-50)
    if len(close) >= 20:
        chg_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100
        chg_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100

        if chg_5d > 0 and chg_20d > 0:
            score += 50  # Both positive
        elif chg_20d > 0:
            score += 30  # Longer trend positive
        elif chg_5d > 0:
            score += 20  # Short-term bounce

    return min(100, max(0, score))

def _score_volatility(df: pd.DataFrame) -> float:
    """Score 0-100: lower volatility = higher score (safer for entries)."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # ATR relative to price
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1)),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]
    atr_pct = atr14 / close.iloc[-1] * 100 if close.iloc[-1] > 0 else 5

    # Lower ATR% = higher score
    if atr_pct < 1.5:
        score = 90
    elif atr_pct < 2.5:
        score = 70
    elif atr_pct < 3.5:
        score = 50
    elif atr_pct < 5:
        score = 30
    else:
        score = 10

    return score

def _score_volume(df: pd.DataFrame) -> float:
    """Score 0-100 based on volume vs 20-day average."""
    if "Volume" not in df.columns:
        return 50
    vol = df["Volume"]
    vol_sma = vol.rolling(20).mean()
    if vol_sma.iloc[-1] == 0:
        return 50
    ratio = vol.iloc[-1] / vol_sma.iloc[-1]

    # Higher ratio on up days = bullish volume
    is_up = df["Close"].iloc[-1] > df["Open"].iloc[-1]

    if ratio > 1.5 and is_up:
        return 95  # Strong volume on up day
    elif ratio > 1.2 and is_up:
        return 80
    elif ratio > 1.0 and is_up:
        return 65
    elif ratio > 1.0:
        return 45  # Volume but down day
    elif ratio > 0.7:
        return 35  # Below average
    else:
        return 15  # Very low volume

def _score_adx(df: pd.DataFrame) -> float:
    """Score 0-100 based on ADX trend strength."""
    adx = _adx(df["High"], df["Low"], df["Close"], 14).iloc[-1]

    if adx > 40:
        return 95  # Very strong trend
    elif adx > 30:
        return 80  # Strong trend
    elif adx > 25:
        return 65  # Moderate trend
    elif adx > 20:
        return 50  # Weak trend
    elif adx > 15:
        return 30  # Very weak
    else:
        return 10  # No trend / consolidation


# ── Composite scoring with z-score normalization ──────────────────────

WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.25,
    "volatility": 0.15,
    "volume": 0.15,
    "adx": 0.15,
}

def score_single_ticker(df: pd.DataFrame) -> dict:
    """Score a single ticker's DataFrame. Returns dict of raw sub-scores."""
    if df is None or df.empty or len(df) < 50:
        return None
    try:
        return {
            "trend": _score_trend(df),
            "momentum": _score_momentum(df),
            "volatility": _score_volatility(df),
            "volume": _score_volume(df),
            "adx": _score_adx(df),
        }
    except Exception as e:
        logger.warning("score_single_ticker error: %s", e)
        return None


def _zscore_normalize(values: pd.Series) -> pd.Series:
    """Z-score normalization: (x - mean) / std, then scale to 0-100."""
    mean = values.mean()
    std = values.std()
    if std == 0:
        return pd.Series(50.0, index=values.index)
    z = (values - mean) / std
    # Map z-scores to 0-100 (z of -2 → 0, z of +2 → 100)
    normalized = (z + 2) / 4 * 100
    return normalized.clip(0, 100)


@st.cache_data(ttl=1800, show_spinner=False)
def run_ovtlyr_screener(
    universe: str = "Nordic",
    min_volume: int = 100_000,
    period: str = "1y",
    ticker_list: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Run the full OVTLYR screener on a universe of tickers.

    Args:
        universe: Legacy universe string ("Nordic", "US", "Canada", "All")
                  or "custom" when ticker_list is provided.
        min_volume: Minimum average 20-day volume filter.
        period: yfinance download period (default "1y").
        ticker_list: Explicit list of yfinance tickers. When provided,
                     overrides the universe parameter.

    Returns DataFrame with columns:
      Ticker, Name, Sector, Country,
      Trend, Momentum, Volatility, Volume, ADX,
      Composite (z-score weighted), Signal, Rank
    """
    try:
        if ticker_list is not None and len(ticker_list) > 0:
            # New path: explicit ticker list from ticker_universe
            tickers_meta = {
                t: {"name": t.split(".")[0], "sector": "Unknown", "country": "Intl"}
                for t in ticker_list
            }
        elif universe == "Nordic":
            tickers_meta = _build_nordic_universe()
        else:
            tickers_meta = UNIVERSES.get(universe, _FALLBACK_NORDIC)

        if not tickers_meta:
            return pd.DataFrame()

        tickers = list(tickers_meta.keys())
    except Exception as e:
        logger.warning("run_ovtlyr_screener universe build error: %s", e)
        return pd.DataFrame()

    # Batch download
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, threads=True)
    except Exception:
        raw = None

    results = []
    for ticker in tickers:
        meta = tickers_meta.get(ticker, {})
        try:
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    df = raw.xs(ticker, level=1, axis=1).dropna()
                elif len(tickers) == 1:
                    df = raw.dropna()
                else:
                    continue
            else:
                tk = yf.Ticker(ticker)
                df = tk.history(period=period, auto_adjust=True)

            if df.empty or len(df) < 50:
                continue

            # Check minimum volume
            avg_vol = df["Volume"].tail(20).mean() if "Volume" in df.columns else 0
            if avg_vol < min_volume:
                continue

            scores = score_single_ticker(df)
            if scores is None:
                continue

            results.append({
                "Ticker": ticker,
                "Name": meta.get("name", ticker),
                "Sector": meta.get("sector", "Unknown"),
                "Country": meta.get("country", "Unknown"),
                **scores,
                "_close": float(df["Close"].iloc[-1]),
                "_avg_vol": float(avg_vol),
            })
        except Exception as e:
            logger.warning("Screener error for %s: %s", ticker, e)
            continue

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)

    # Z-score normalize each component across the universe
    for col in ["trend", "momentum", "volatility", "volume", "adx"]:
        df_results[f"{col}_z"] = _zscore_normalize(df_results[col])

    # Weighted composite
    df_results["Composite"] = (
        df_results["trend_z"] * WEIGHTS["trend"] +
        df_results["momentum_z"] * WEIGHTS["momentum"] +
        df_results["volatility_z"] * WEIGHTS["volatility"] +
        df_results["volume_z"] * WEIGHTS["volume"] +
        df_results["adx_z"] * WEIGHTS["adx"]
    ).round(1)

    # Rank
    df_results["Rank"] = df_results["Composite"].rank(ascending=False, method="min").astype(int)

    # Signal
    df_results["Signal"] = df_results["Composite"].apply(
        lambda x: "STRONG BUY" if x >= 75 else ("BUY" if x >= 60 else ("HOLD" if x >= 40 else "SELL"))
    )

    # Round display columns
    for col in ["trend", "momentum", "volatility", "volume", "adx", "Composite"]:
        df_results[col] = df_results[col].round(1)

    # Sort by composite descending
    df_results = df_results.sort_values("Composite", ascending=False).reset_index(drop=True)

    # Select display columns
    display_cols = [
        "Rank", "Ticker", "Name", "Sector", "Country",
        "trend", "momentum", "volatility", "volume", "adx",
        "Composite", "Signal",
    ]
    return df_results[display_cols]
