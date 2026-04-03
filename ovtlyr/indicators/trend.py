"""
Trend indicators for OVTLYR.
Pure functions — no Streamlit imports.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True Range → ATR."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def compute_trend(df: pd.DataFrame) -> dict:
    """
    Compute trend indicators from an OHLCV DataFrame.

    Returns a dict with:
        ema50          : pd.Series
        ema200         : pd.Series
        trend_state    : "Bullish" | "Bearish" | "Neutral"
        regime_color   : "green" | "orange" | "red"
        ema50_above_200: bool
        price_above_200: bool
        ema200_rising  : bool

    Logic
    -----
    Bullish : price > EMA200 AND EMA50 > EMA200
    Bearish : price < EMA200 AND EMA50 < EMA200
    Neutral : mixed

    Regime color
        green  : Bullish + low/normal volatility (ATR14/close < 0.03)
        orange : price within 2% of EMA200  (regardless of trend state)
        red    : Bearish OR high volatility (ATR14/close > 0.04)
    """
    close = df["Close"].dropna()

    if len(close) < 2:
        empty = pd.Series(dtype=float)
        return {
            "ema50": empty,
            "ema200": empty,
            "trend_state": "Neutral",
            "regime_color": "orange",
            "ema50_above_200": False,
            "price_above_200": False,
            "ema200_rising": False,
        }

    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    atr14 = _atr(df.loc[close.index], 14)

    last_close = float(close.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    last_ema200 = float(ema200.iloc[-1])
    last_atr = float(atr14.iloc[-1]) if not atr14.empty else 0.0

    ema50_above_200 = last_ema50 > last_ema200
    price_above_200 = last_close > last_ema200

    # EMA200 rising: compare last value to value 10 bars ago
    if len(ema200) >= 11:
        ema200_rising = float(ema200.iloc[-1]) > float(ema200.iloc[-11])
    else:
        ema200_rising = False

    # Trend state
    if price_above_200 and ema50_above_200:
        trend_state = "Bullish"
    elif not price_above_200 and not ema50_above_200:
        trend_state = "Bearish"
    else:
        trend_state = "Neutral"

    # ATR/close ratio
    atr_ratio = last_atr / last_close if last_close > 0 else 0.0

    # Price proximity to EMA200 (within 2%)
    price_near_ema200 = abs(last_close - last_ema200) / last_ema200 < 0.02 if last_ema200 > 0 else False

    # Regime color
    if price_near_ema200:
        regime_color = "orange"
    elif trend_state == "Bearish" or atr_ratio > 0.04:
        regime_color = "red"
    elif trend_state == "Bullish" and atr_ratio < 0.03:
        regime_color = "green"
    else:
        # Neutral or Bullish with moderate volatility
        regime_color = "orange"

    return {
        "ema50": ema50,
        "ema200": ema200,
        "trend_state": trend_state,
        "regime_color": regime_color,
        "ema50_above_200": ema50_above_200,
        "price_above_200": price_above_200,
        "ema200_rising": ema200_rising,
    }
