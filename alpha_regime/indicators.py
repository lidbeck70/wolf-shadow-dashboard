"""
alpha_regime/indicators.py
Compute market-cycle indicator fields from an OHLCV DataFrame.
All values match the field names expected by market_cycle.rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> float:
    """RSI of the last `period`+lookback bars, returning the most recent value."""
    s = series.dropna()
    if len(s) < period + 1:
        return float("nan")
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _momentum_pct(series: pd.Series, days: int) -> float:
    """% change over last `days` bars."""
    s = series.dropna()
    if len(s) < days + 1:
        return float("nan")
    old = float(s.iloc[-days - 1])
    new = float(s.iloc[-1])
    if old == 0:
        return float("nan")
    return (new - old) / abs(old) * 100.0


def _macd_diff(series: pd.Series) -> float:
    """MACD(12,26) − Signal(9): positive means bullish crossover."""
    s = series.dropna()
    if len(s) < 35:
        return float("nan")
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return float((macd_line - signal).iloc[-1])


def compute_market_indicators(df: pd.DataFrame) -> dict:
    """
    Compute all 9 indicator fields consumed by detect_market_cycle().

    Input: OHLCV DataFrame indexed by Date with at least a Close column.
    Returns dict with keys matching market_cycle.rules field names.
    """
    if df.empty or "Close" not in df.columns:
        return {}

    close = df["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()

    if len(close) < 30:
        return {}

    price = float(close.iloc[-1])

    # Moving averages
    ma200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
    ma50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

    price_vs_ma200 = (price / ma200 - 1) * 100 if ma200 != 0 else float("nan")
    price_vs_ma50 = (price / ma50 - 1) * 100 if ma50 != 0 else float("nan")

    # 90-day drawdown: % below the rolling 90-day high
    window = min(90, len(close))
    high_90 = float(close.rolling(window).max().iloc[-1])
    drawdown_90 = (price / high_90 - 1) * 100 if high_90 != 0 else float("nan")

    # Volume ratio
    volume_vs_avg20 = float("nan")
    vol_col = "Volume" if "Volume" in df.columns else None
    if vol_col:
        vol = df[vol_col].squeeze()
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        vol = vol.dropna()
        if len(vol) >= 21:
            avg20 = float(vol.iloc[-21:-1].mean())
            current_vol = float(vol.iloc[-1])
            if avg20 > 0:
                volume_vs_avg20 = current_vol / avg20

    return {
        "rsi": _rsi(close),
        "momentum_30": _momentum_pct(close, 30),
        "momentum_60": _momentum_pct(close, 60),
        "momentum_90": _momentum_pct(close, 90),
        "price_vs_ma200": price_vs_ma200,
        "price_vs_ma50": price_vs_ma50,
        "drawdown_90": drawdown_90,
        "macd_diff": _macd_diff(close),
        "volume_vs_avg20": volume_vs_avg20,
    }
