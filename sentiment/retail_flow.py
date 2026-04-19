"""
sentiment/retail_flow.py
========================
Retail Participation Flow sentiment plugin.

Retail traders exhibit measurable OHLCV footprints:
  • They chase momentum — buy after multi-day runs
  • They prefer round-number strikes / levels → volume spikes near EMA20
  • They capitulate on high-volume down bars
  • They exit positions intraday → close skews toward low on down-days

Four components approximate the retail sentiment pressure:

  Component              Weight   Signal
  ---------------------  ------   ---------------------------------------------
  Volume Surge           30 %    Current vol / 20-day avg vol; up-bar = greed
  Price vs EMA20         25 %    Above EMA20 and rising = retail FOMO
  Consecutive Momentum   25 %    Streak of up/down closes (retail herding)
  Close Position Range   20 %    Close near high of day → retail buying; near
                                  low → retail selling / fear

All inputs derived from the OHLCV DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(v)))


# ---------------------------------------------------------------------------
# Component calculators
# ---------------------------------------------------------------------------

def _comp_volume_surge(df: pd.DataFrame, window: int = 20) -> float:
    """
    Volume surge on directional bars.
    Up-bar + high volume → retail FOMO (greedy, high score).
    Down-bar + high volume → retail panic (fearful, low score).
    Flat volume → neutral (50).
    """
    if len(df) < window + 1:
        return 50.0

    vol   = df["Volume"].astype(float)
    close = df["Close"].astype(float)
    avg   = float(vol.rolling(window).mean().iloc[-1])
    if avg <= 0:
        return 50.0

    ratio  = float(vol.iloc[-1]) / avg
    up_bar = float(close.iloc[-1]) > float(close.iloc[-2])

    # Surge multiplier: ratio 1→3 maps to 0→50 bonus/penalty
    surge = _clamp((ratio - 1.0) * 25.0, 0.0, 50.0)
    if up_bar:
        return _clamp(50.0 + surge)
    else:
        return _clamp(50.0 - surge)


def _comp_price_ema20(close: pd.Series) -> float:
    """
    Retail FOMO gauge: price distance above/below EMA20.

    % above EMA20 mapped linearly:
      +5 % above → score 80
      At EMA20   → score 50
      -5 % below → score 20
    """
    if len(close) < 21:
        return 50.0

    price   = float(close.iloc[-1])
    e20     = float(_ema(close, 20).iloc[-1])
    if e20 <= 0:
        return 50.0
    pct     = (price - e20) / e20
    score   = 50.0 + pct * 600.0     # ±5 % → ±30 points
    return _clamp(score)


def _comp_consecutive_momentum(close: pd.Series, max_streak: int = 5) -> float:
    """
    Retail herding: count the current streak of up or down closes.
    Positive streak → greed; negative → fear.
    Streak of ±max_streak saturates at 90/10.
    """
    if len(close) < 3:
        return 50.0

    changes = close.diff().dropna()
    streak  = 0
    for chg in reversed(changes.values):
        if chg > 0:
            if streak < 0:
                break
            streak += 1
        elif chg < 0:
            if streak > 0:
                break
            streak -= 1
        else:
            break

    score = 50.0 + (streak / max_streak) * 40.0
    return _clamp(score)


def _comp_close_position(df: pd.DataFrame, window: int = 5) -> float:
    """
    Where the close sits within the day's High-Low range, averaged over
    *window* bars.  Close near High → retail buying pressure (greedy).
    Close near Low → retail selling / stop-hunting (fear).
    """
    if len(df) < window:
        return 50.0

    recent  = df.iloc[-window:]
    high    = recent["High"].astype(float)
    low     = recent["Low"].astype(float)
    close   = recent["Close"].astype(float)
    rng     = high - low
    valid   = rng > 0
    if not valid.any():
        return 50.0
    pos = ((close - low) / rng.where(valid, np.nan)).dropna()
    return _clamp(float(pos.mean()) * 100.0)


# ---------------------------------------------------------------------------
# Plugin interface
# ---------------------------------------------------------------------------

def compute_score(df: pd.DataFrame) -> float:
    """
    Compute Retail Flow sentiment score from OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex and columns
        Open, High, Low, Close, Volume.

    Returns
    -------
    float in [0, 100]  — 0 = strong retail selling, 100 = strong retail buying
    """
    if df is None or len(df) < 10:
        return 50.0

    close = df["Close"].astype(float)

    vol_surge   = _comp_volume_surge(df)
    price_ema   = _comp_price_ema20(close)
    consec_mom  = _comp_consecutive_momentum(close)
    close_pos   = _comp_close_position(df)

    score = (
        0.30 * vol_surge
        + 0.25 * price_ema
        + 0.25 * consec_mom
        + 0.20 * close_pos
    )
    return round(_clamp(score), 2)


def compute_signal(df: pd.DataFrame) -> dict:
    """
    Derive a directional signal from the Retail Flow score.

    Returns
    -------
    dict with keys:
      bias           : "bullish" | "neutral" | "bearish"
      confidence     : float 0-1
      score          : float 0-100
      label          : human-readable label
      components     : dict of individual component scores
      retail_pressure: "accumulating" | "distributing" | "neutral"
    """
    if df is None or len(df) < 10:
        return {"bias": "neutral", "confidence": 0.0, "score": 50.0,
                "label": "Insufficient data", "components": {},
                "retail_pressure": "neutral"}

    close = df["Close"].astype(float)

    components = {
        "volume_surge":          round(_comp_volume_surge(df),              2),
        "price_vs_ema20":        round(_comp_price_ema20(close),             2),
        "consecutive_momentum":  round(_comp_consecutive_momentum(close),    2),
        "close_position_range":  round(_comp_close_position(df),             2),
    }

    score = compute_score(df)

    if score >= 62:
        bias, label, pressure = "bullish", "Retail Accumulating", "accumulating"
    elif score <= 38:
        bias, label, pressure = "bearish", "Retail Distributing", "distributing"
    else:
        bias, label, pressure = "neutral", "Retail Neutral", "neutral"

    confidence = round(abs(score - 50.0) / 50.0, 3)

    return {
        "bias":            bias,
        "confidence":      confidence,
        "score":           score,
        "label":           label,
        "components":      components,
        "retail_pressure": pressure,
    }


# ---------------------------------------------------------------------------
# Plugin descriptor
# ---------------------------------------------------------------------------

SENTIMENT_PLUGIN: dict = {
    "key":         "retail_flow",
    "name":        "Retail Flow",
    "description": (
        "Retail participation pressure: Volume Surge(30%) + Price vs EMA20(25%) "
        "+ Consecutive Momentum(25%) + Close Position in Range(20%). "
        "Identifies retail FOMO and panic via OHLCV footprints."
    ),
    "color":        "#2d8a4e",
    "compute_score":  compute_score,
    "compute_signal": compute_signal,
}
