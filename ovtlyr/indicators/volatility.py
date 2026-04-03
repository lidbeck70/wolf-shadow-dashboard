"""
Volatility indicators for OVTLYR.
Pure functions — no Streamlit imports.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True Range → ATR (EMA-smoothed)."""
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


def _hist_vol(close: pd.Series, window: int, trading_days: int = 252) -> float:
    """Annualised historical volatility over *window* days."""
    log_returns = np.log(close / close.shift(1)).dropna()
    if len(log_returns) < window:
        return float("nan")
    rolling_std = log_returns.rolling(window).std()
    annualised = rolling_std * np.sqrt(trading_days)
    return float(annualised.iloc[-1])


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def compute_volatility(df: pd.DataFrame) -> dict:
    """
    Compute volatility metrics from an OHLCV DataFrame.

    Returns a dict with:
        atr14       : float  – current ATR(14)
        hist_vol_20 : float  – 20-day annualised historical volatility
        hist_vol_60 : float  – 60-day annualised historical volatility
        risk_score  : int    – 0-100
        risk_level  : "Low" | "Normal" | "High"

    Risk score derivation
    ---------------------
    Based on ATR14 / close ratio:
        < 0.02  → maps to score  0–30  (Low)
        0.02–0.04 → maps to score 30–70 (Normal)
        > 0.04  → maps to score 70–100 (High)
    The ratio is clipped and linearly scaled to 0–100.
    """
    close = df["Close"].dropna()

    if len(close) < 2:
        return {
            "atr14": float("nan"),
            "hist_vol_20": float("nan"),
            "hist_vol_60": float("nan"),
            "risk_score": 50,
            "risk_level": "Normal",
        }

    atr_series = _atr(df.loc[close.index], 14)
    last_atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
    last_close = float(close.iloc[-1])

    hist_vol_20 = _hist_vol(close, 20)
    hist_vol_60 = _hist_vol(close, 60)

    # ATR/close ratio → risk score 0-100
    atr_ratio = last_atr / last_close if last_close > 0 else 0.0
    # Clamp ratio to [0, 0.06] range and scale linearly to 0-100
    ratio_min = 0.0
    ratio_max = 0.06
    ratio_clipped = max(ratio_min, min(ratio_max, atr_ratio))
    risk_score = int(round((ratio_clipped / ratio_max) * 100))

    # Risk level thresholds
    if atr_ratio < 0.02:
        risk_level = "Low"
    elif atr_ratio > 0.04:
        risk_level = "High"
    else:
        risk_level = "Normal"

    return {
        "atr14": last_atr,
        "hist_vol_20": hist_vol_20,
        "hist_vol_60": hist_vol_60,
        "risk_score": risk_score,
        "risk_level": risk_level,
    }
