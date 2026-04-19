"""
sentiment/options_flow.py
=========================
Options Flow sentiment plugin — OHLCV-derived proxy.

Without a live options feed, we construct synthetic options-flow signals
from the price / volume patterns that options activity leaves behind in the
underlying equity:

  Component           Weight   Proxy rationale
  ------------------  ------   ------------------------------------------------
  IV Proxy (ATR%)     30 %    ATR14 / Close vs 20-day baseline.  Expanding ATR
                               implies elevated IV → put-buying pressure (fear).
  Put/Call Skew       25 %    Daily H-L asymmetry: up-wick dominant → call
                               sellers capping the move (bearish skew); down-wick
                               dominant → put sellers supporting (bullish skew).
  Smart-Money Flow    25 %    Large-range bars on above-avg volume signal
                               institutional positioning (options hedgers).
  Momentum Skew       20 %    EMA10 vs EMA21 spread as a % of price.  Positive
                               spread → call flow; negative → put flow.

Score interpretation
--------------------
  0  – 30  : Elevated put flow / bearish options positioning
  31 – 50  : Mild put bias / cautious
  51 – 70  : Mild call bias / constructive
  71 – 100 : Elevated call flow / bullish options positioning
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


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

def _comp_iv_proxy(df: pd.DataFrame, window: int = 20) -> float:
    """
    Implied-volatility proxy via ATR% vs its own 20-session mean.

    ratio < 1 → compressed IV → call bias (high score, constructive).
    ratio > 1 → expanded IV  → put bias (low score, fearful).
    """
    if len(df) < window + 15:
        return 50.0

    close      = df["Close"].astype(float)
    atr_series = _atr(df)
    atr_pct    = atr_series / close
    baseline   = float(atr_pct.rolling(window).mean().iloc[-1])
    current    = float(atr_pct.iloc[-1])
    if baseline <= 0:
        return 50.0

    ratio = current / baseline
    # ratio 0.5→1.5 maps to score 90→10
    score = _clamp(100.0 - (ratio - 0.5) * 80.0)
    return score


def _comp_wick_skew(df: pd.DataFrame, window: int = 5) -> float:
    """
    Intraday wick asymmetry averaged over *window* bars.

    Each bar:
      body_mid   = (Open + Close) / 2
      up_wick    = High - max(Open, Close)
      down_wick  = min(Open, Close) - Low
      skew       = up_wick - down_wick  (positive → upper wick dominates)

    Dominant upper wicks → sellers capping rally → put pressure (bearish).
    Dominant lower wicks → buyers absorbing dips → call pressure (bullish).
    """
    if len(df) < window:
        return 50.0

    recent     = df.iloc[-window:].copy()
    open_      = recent["Open"].astype(float)
    high       = recent["High"].astype(float)
    low        = recent["Low"].astype(float)
    close      = recent["Close"].astype(float)

    up_wick    = high - pd.concat([open_, close], axis=1).max(axis=1)
    down_wick  = pd.concat([open_, close], axis=1).min(axis=1) - low
    total_rng  = high - low
    valid      = total_rng > 0

    # Normalised skew: +1 = all upper wick, -1 = all lower wick
    skew_norm  = ((down_wick - up_wick) / total_rng.where(valid, np.nan)).dropna()
    if skew_norm.empty:
        return 50.0

    avg_skew   = float(skew_norm.mean())    # +1 = bullish (lower wicks), -1 bearish
    score      = _clamp(50.0 + avg_skew * 50.0)
    return score


def _comp_smart_money(df: pd.DataFrame, window: int = 20) -> float:
    """
    Smart-money (institutional / options-hedger) proxy.

    Large-range bars on above-average volume indicate hedging activity.
    Direction (up vs down) determines bullish vs bearish positioning.
    """
    if len(df) < window + 1:
        return 50.0

    recent   = df.iloc[-(window + 1):]
    high     = recent["High"].astype(float)
    low      = recent["Low"].astype(float)
    close    = recent["Close"].astype(float)
    vol      = recent["Volume"].astype(float)

    bar_rng  = high - low
    avg_rng  = float(bar_rng.rolling(window).mean().iloc[-1])
    avg_vol  = float(vol.rolling(window).mean().iloc[-1])

    if avg_rng <= 0 or avg_vol <= 0:
        return 50.0

    rng_ratio = float(bar_rng.iloc[-1]) / avg_rng
    vol_ratio = float(vol.iloc[-1]) / avg_vol

    # Large-range + high-volume = smart-money signal
    signal_strength = (rng_ratio * vol_ratio - 1.0)
    up_bar          = float(close.iloc[-1]) >= float(close.iloc[-2])

    if up_bar:
        return _clamp(50.0 + signal_strength * 25.0)
    else:
        return _clamp(50.0 - signal_strength * 25.0)


def _comp_momentum_skew(close: pd.Series) -> float:
    """
    EMA10 vs EMA21 spread as % of price.
    Positive spread (EMA10 > EMA21) → call-flow momentum.
    Negative spread → put-flow / defensive.
    """
    if len(close) < 22:
        return 50.0

    price  = float(close.iloc[-1])
    e10    = float(_ema(close, 10).iloc[-1])
    e21    = float(_ema(close, 21).iloc[-1])
    if price <= 0:
        return 50.0

    spread_pct = (e10 - e21) / price      # typically ±0.02
    score      = _clamp(50.0 + spread_pct * 2000.0)
    return score


# ---------------------------------------------------------------------------
# Plugin interface
# ---------------------------------------------------------------------------

def compute_score(df: pd.DataFrame) -> float:
    """
    Compute Options Flow sentiment score from OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex and columns
        Open, High, Low, Close, Volume.

    Returns
    -------
    float in [0, 100]  — 0 = heavy put flow, 100 = heavy call flow
    """
    if df is None or len(df) < 15:
        return 50.0

    close = df["Close"].astype(float)

    iv_proxy    = _comp_iv_proxy(df)
    wick_skew   = _comp_wick_skew(df)
    smart_money = _comp_smart_money(df)
    mom_skew    = _comp_momentum_skew(close)

    score = (
        0.30 * iv_proxy
        + 0.25 * wick_skew
        + 0.25 * smart_money
        + 0.20 * mom_skew
    )
    return round(_clamp(score), 2)


def compute_signal(df: pd.DataFrame) -> dict:
    """
    Derive a directional signal from the Options Flow score.

    Returns
    -------
    dict with keys:
      bias          : "bullish" | "neutral" | "bearish"
      confidence    : float 0-1
      score         : float 0-100
      label         : human-readable label
      components    : dict of individual component scores
      flow_type     : "call_dominated" | "put_dominated" | "mixed"
      iv_regime     : "elevated" | "compressed" | "normal"
    """
    if df is None or len(df) < 15:
        return {"bias": "neutral", "confidence": 0.0, "score": 50.0,
                "label": "Insufficient data", "components": {},
                "flow_type": "mixed", "iv_regime": "normal"}

    close = df["Close"].astype(float)

    iv_proxy    = _comp_iv_proxy(df)
    components  = {
        "iv_proxy":    round(iv_proxy,                    2),
        "wick_skew":   round(_comp_wick_skew(df),         2),
        "smart_money": round(_comp_smart_money(df),       2),
        "mom_skew":    round(_comp_momentum_skew(close),  2),
    }

    score = compute_score(df)

    if score >= 65:
        bias, label, flow_type = "bullish", "Call-Dominated Flow", "call_dominated"
    elif score <= 35:
        bias, label, flow_type = "bearish", "Put-Dominated Flow", "put_dominated"
    else:
        bias, label, flow_type = "neutral", "Mixed Flow", "mixed"

    # IV regime from the IV proxy component directly
    if iv_proxy <= 35:
        iv_regime = "elevated"
    elif iv_proxy >= 65:
        iv_regime = "compressed"
    else:
        iv_regime = "normal"

    confidence = round(abs(score - 50.0) / 50.0, 3)

    return {
        "bias":       bias,
        "confidence": confidence,
        "score":      score,
        "label":      label,
        "components": components,
        "flow_type":  flow_type,
        "iv_regime":  iv_regime,
    }


# ---------------------------------------------------------------------------
# Plugin descriptor
# ---------------------------------------------------------------------------

SENTIMENT_PLUGIN: dict = {
    "key":         "options_flow",
    "name":        "Options Flow",
    "description": (
        "OHLCV-derived options-flow proxy: IV Proxy/ATR%(30%) + "
        "Wick Skew/Put-Call(25%) + Smart-Money bars(25%) + "
        "EMA Momentum Skew(20%). Identifies call vs put pressure without "
        "requiring a live options feed."
    ),
    "color":        "#8b7340",
    "compute_score":  compute_score,
    "compute_signal": compute_signal,
}
