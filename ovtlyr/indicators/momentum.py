"""
Momentum indicators for OVTLYR.
Pure functions — no Streamlit imports.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # First average using simple mean for the seed
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _roc(close: pd.Series, period: int) -> float:
    """Rate of Change over *period* bars."""
    if len(close) <= period:
        return float("nan")
    past = float(close.iloc[-(period + 1)])
    if past == 0:
        return float("nan")
    return float((close.iloc[-1] - past) / past * 100)


def _zscore_50(close: pd.Series) -> float:
    """Z-score of the latest close vs. 50-day rolling mean/std."""
    if len(close) < 50:
        return float("nan")
    window = close.rolling(50)
    mean = window.mean()
    std = window.std()
    z = (close - mean) / std.replace(0, np.nan)
    return float(z.iloc[-1])


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def compute_momentum(df: pd.DataFrame) -> dict:
    """
    Compute momentum metrics from an OHLCV DataFrame.

    Returns a dict with:
        rsi            : float  – RSI(14) current value
        rsi_series     : pd.Series
        roc_10         : float  – Rate of Change (10-day), percent
        roc_20         : float  – Rate of Change (20-day), percent
        zscore         : float  – z-score of close vs 50-day mean/std
        ob_os_flag     : "Overbought" | "Oversold" | "Neutral"
        momentum_score : int    – 0-100

    OB/OS Rules
    -----------
    Overbought : RSI > 70 OR z-score > 2
    Oversold   : RSI < 30 OR z-score < -2
    Neutral    : otherwise

    Momentum score
    --------------
    RSI value (0-100) is used directly as the score baseline.
    """
    close = df["Close"].dropna()

    if len(close) < 2:
        empty = pd.Series(dtype=float)
        return {
            "rsi": float("nan"),
            "rsi_series": empty,
            "roc_10": float("nan"),
            "roc_20": float("nan"),
            "zscore": float("nan"),
            "ob_os_flag": "Neutral",
            "momentum_score": 50,
        }

    rsi_series = _rsi(close, 14)
    last_rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
    last_rsi = last_rsi if not np.isnan(last_rsi) else 50.0

    roc_10 = _roc(close, 10)
    roc_20 = _roc(close, 20)
    zscore = _zscore_50(close)

    # OB/OS flag
    z_val = zscore if not np.isnan(zscore) else 0.0
    if last_rsi > 70 or z_val > 2:
        ob_os_flag = "Overbought"
    elif last_rsi < 30 or z_val < -2:
        ob_os_flag = "Oversold"
    else:
        ob_os_flag = "Neutral"

    # Momentum score: RSI as primary (0-100)
    momentum_score = int(round(max(0, min(100, last_rsi))))

    return {
        "rsi": last_rsi,
        "rsi_series": rsi_series,
        "roc_10": roc_10,
        "roc_20": roc_20,
        "zscore": zscore,
        "ob_os_flag": ob_os_flag,
        "momentum_score": momentum_score,
    }
