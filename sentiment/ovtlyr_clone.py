"""
sentiment/ovtlyr_clone.py
=========================
OVTLYR-style Fear & Greed composite sentiment plugin.

Replicates the five-component weighting used by the OVTLYR engine:

  Component          Weight   Source
  -----------------  ------   -------------------------------------------
  Momentum (RSI)      25 %    RSI-14 on Close, normalised to 0-100
  Trend               25 %    Price position relative to EMA20 / EMA50
  Volatility          20 %    ATR14 as % of price vs 20-session baseline
  Volume              15 %    Current volume vs 20-session average volume
  Breadth             15 %    Fraction of last 20 bars that were up-closes

All inputs are derived from the OHLCV DataFrame; no external API calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(v)))


# ---------------------------------------------------------------------------
# Component calculators
# ---------------------------------------------------------------------------

def _comp_momentum(close: pd.Series) -> float:
    """RSI-14 expressed directly as 0-100 Fear/Greed value."""
    if len(close) < 15:
        return 50.0
    return _clamp(_rsi(close))


def _comp_trend(close: pd.Series) -> float:
    """
    Price position relative to EMA20 and EMA50.
    Both above → 100, one above → 60, both below → 0.
    """
    if len(close) < 51:
        return 50.0
    price   = float(close.iloc[-1])
    e20     = float(_ema(close, 20).iloc[-1])
    e50     = float(_ema(close, 50).iloc[-1])
    above20 = price > e20
    above50 = price > e50
    if above20 and above50:
        return 85.0
    if above20 or above50:
        return 55.0
    return 15.0


def _comp_volatility(df: pd.DataFrame) -> float:
    """
    ATR14 as % of price vs its own 20-session rolling mean.
    Rising ATR (expanding volatility) → fear (lower score).
    Compressed ATR → greed (higher score).
    """
    if len(df) < 35:
        return 50.0
    atr_series = _atr(df)
    close      = df["Close"]
    atr_pct    = atr_series / close          # ATR as % of price
    baseline   = atr_pct.rolling(20).mean()
    ratio      = float(atr_pct.iloc[-1]) / float(baseline.iloc[-1])
    # ratio < 1 → compressed vol → greed; ratio > 1 → expanded → fear
    score = _clamp(100.0 - (ratio - 1.0) * 80.0)
    return score


def _comp_volume(df: pd.DataFrame) -> float:
    """
    Current volume vs 20-session average volume.
    Above-average volume on up-bars → greed; on down-bars stays neutral.
    """
    if len(df) < 21:
        return 50.0
    vol    = df["Volume"].astype(float)
    close  = df["Close"]
    avg_vol = float(vol.rolling(20).mean().iloc[-1])
    if avg_vol <= 0:
        return 50.0
    ratio   = float(vol.iloc[-1]) / avg_vol
    up_bar  = float(close.iloc[-1]) >= float(close.iloc[-2])
    if up_bar:
        return _clamp(50.0 + (ratio - 1.0) * 35.0)
    return _clamp(50.0 - (ratio - 1.0) * 20.0)


def _comp_breadth(close: pd.Series, window: int = 20) -> float:
    """Fraction of last *window* bars that closed higher than the previous bar."""
    if len(close) < window + 1:
        return 50.0
    recent = close.iloc[-(window + 1):]
    up_bars = (recent.diff().dropna() > 0).sum()
    return _clamp((up_bars / window) * 100.0)


# ---------------------------------------------------------------------------
# Plugin interface
# ---------------------------------------------------------------------------

def compute_score(df: pd.DataFrame) -> float:
    """
    Compute OVTLYR-style Fear & Greed score from OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex and columns
        Open, High, Low, Close, Volume.

    Returns
    -------
    float in [0, 100]  — 0 = Extreme Fear, 100 = Extreme Greed
    """
    if df is None or len(df) < 20:
        return 50.0

    close = df["Close"].astype(float)

    momentum   = _comp_momentum(close)
    trend      = _comp_trend(close)
    volatility = _comp_volatility(df)
    volume     = _comp_volume(df)
    breadth    = _comp_breadth(close)

    score = (
        0.25 * momentum
        + 0.25 * trend
        + 0.20 * volatility
        + 0.15 * volume
        + 0.15 * breadth
    )
    return round(_clamp(score), 2)


def compute_signal(df: pd.DataFrame) -> dict:
    """
    Derive a directional signal from the OVTLYR Fear & Greed score.

    Returns
    -------
    dict with keys:
      bias       : "bullish" | "neutral" | "bearish"
      confidence : float 0-1  (how far the score is from neutral)
      score      : float 0-100
      label      : human-readable label
      components : dict of individual component scores
    """
    if df is None or len(df) < 20:
        return {"bias": "neutral", "confidence": 0.0, "score": 50.0,
                "label": "Insufficient data", "components": {}}

    close = df["Close"].astype(float)

    components = {
        "momentum":   round(_comp_momentum(close),   2),
        "trend":      round(_comp_trend(close),       2),
        "volatility": round(_comp_volatility(df),     2),
        "volume":     round(_comp_volume(df),         2),
        "breadth":    round(_comp_breadth(close),     2),
    }

    score = compute_score(df)

    if score >= 65:
        bias, label = "bullish", "Greed" if score < 80 else "Extreme Greed"
    elif score <= 35:
        bias, label = "bearish", "Fear" if score > 20 else "Extreme Fear"
    else:
        bias, label = "neutral", "Neutral"

    # Confidence = distance from 50, scaled to 0-1
    confidence = round(abs(score - 50.0) / 50.0, 3)

    return {
        "bias":       bias,
        "confidence": confidence,
        "score":      score,
        "label":      label,
        "components": components,
    }


# ---------------------------------------------------------------------------
# Plugin descriptor
# ---------------------------------------------------------------------------

SENTIMENT_PLUGIN: dict = {
    "key":         "ovtlyr_fg",
    "name":        "OVTLYR Fear & Greed",
    "description": (
        "OVTLYR-style composite sentiment: Momentum(25%) + Trend(25%) "
        "+ Volatility(20%) + Volume(15%) + Breadth(15%). "
        "Derived entirely from OHLCV data."
    ),
    "color":        "#c9a84c",
    "compute_score":  compute_score,
    "compute_signal": compute_signal,
}
