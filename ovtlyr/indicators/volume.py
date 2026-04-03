"""
Volume analysis indicators for OVTLYR.
Pure functions — no Streamlit imports.
"""

import pandas as pd
import numpy as np


def compute_volume(df: pd.DataFrame) -> dict:
    """
    Compute volume metrics from an OHLCV DataFrame.

    Returns a dict with:
        current_volume : float
        avg_volume_20  : float  – 20-bar simple moving average of volume
        volume_ratio   : float  – current / avg_volume_20
        volume_spike   : bool   – ratio > 2.0
        volume_trend   : "Rising" | "Falling" | "Flat"
        volume_series  : pd.Series  – last 60 bars of volume

    Volume trend logic
    ------------------
    Compare the 10-bar EMA of volume to the 30-bar EMA of volume:
        Rising  : short EMA > long EMA by >5 %
        Falling : short EMA < long EMA by >5 %
        Flat    : within ±5 %
    """
    vol = df["Volume"].dropna()

    if vol.empty:
        return {
            "current_volume": float("nan"),
            "avg_volume_20": float("nan"),
            "volume_ratio": float("nan"),
            "volume_spike": False,
            "volume_trend": "Flat",
            "volume_series": pd.Series(dtype=float),
        }

    current_volume = float(vol.iloc[-1])

    # 20-bar simple average
    if len(vol) >= 20:
        avg_volume_20 = float(vol.rolling(20).mean().iloc[-1])
    else:
        avg_volume_20 = float(vol.mean())

    # Volume ratio
    if avg_volume_20 and avg_volume_20 > 0:
        volume_ratio = current_volume / avg_volume_20
    else:
        volume_ratio = float("nan")

    volume_spike = bool(volume_ratio > 2.0) if not np.isnan(volume_ratio) else False

    # Volume trend via short vs long EMA
    ema10 = vol.ewm(span=10, adjust=False).mean()
    ema30 = vol.ewm(span=30, adjust=False).mean()
    last_ema10 = float(ema10.iloc[-1])
    last_ema30 = float(ema30.iloc[-1])

    if last_ema30 > 0:
        ratio = last_ema10 / last_ema30
        if ratio > 1.05:
            volume_trend = "Rising"
        elif ratio < 0.95:
            volume_trend = "Falling"
        else:
            volume_trend = "Flat"
    else:
        volume_trend = "Flat"

    # Last 60 bars of volume
    volume_series = vol.iloc[-60:]

    return {
        "current_volume": current_volume,
        "avg_volume_20": avg_volume_20,
        "volume_ratio": volume_ratio,
        "volume_spike": volume_spike,
        "volume_trend": volume_trend,
        "volume_series": volume_series,
    }
