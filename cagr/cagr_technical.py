"""
cagr_technical.py
Technical scoring module — 0 to 4 points.

Scoring criteria (all computed from OHLCV DataFrame):
  1. MA200 slope positive (EMA200 today > EMA200 20 bars ago)  → 1 point
  2. Price > EMA200                                             → 1 point
  3. EMA50  > EMA200                                           → 1 point
  4. Momentum positive (RSI > 50 AND RSI rising over 5 bars)  → 1 point
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indicator helpers (pure pandas — no external TA libraries)
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder/EWM method).
    Returns a Series of RSI values (0-100).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_technical(df: pd.DataFrame) -> dict:
    """
    Score a stock on 4 technical criteria.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a DatetimeIndex and at least a 'Close' column.
        Output of cagr_loader.fetch_price_data()[ticker].

    Returns
    -------
    dict with keys:
      - tech_score     : int (0-4)
      - details        : dict with criterion → {"value": ..., "pass": bool}
      - sparkline_data : list of last 120 Close prices (float) for mini chart
    """
    empty_result = {
        "tech_score": 0,
        "details": {
            "MA200 Slope Positive": {"value": "N/A", "pass": False},
            "Price > EMA200":       {"value": "N/A", "pass": False},
            "EMA50 > EMA200":       {"value": "N/A", "pass": False},
            "Momentum (RSI>50↑)":   {"value": "N/A", "pass": False},
        },
        "sparkline_data": [],
    }

    if df is None or df.empty:
        return empty_result

    # Ensure we have a Close column
    close_col = None
    for candidate in ("Close", "Adj Close", "close", "adj close"):
        if candidate in df.columns:
            close_col = candidate
            break
    if close_col is None:
        logger.warning("DataFrame has no Close column: %s", list(df.columns))
        return empty_result

    close = df[close_col].dropna()

    if len(close) < 30:
        logger.debug("Not enough data for technical scoring (%d bars)", len(close))
        return empty_result

    score = 0
    details = {}

    # Compute indicators
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    rsi = _rsi(close, 14)

    last_close = close.iloc[-1]
    last_ema50 = ema50.iloc[-1]
    last_ema200 = ema200.iloc[-1]
    last_rsi = rsi.iloc[-1]

    # ---- Criterion 1: EMA200 slope positive (today vs 20 bars ago) ----
    if len(ema200.dropna()) >= 21:
        ema200_now = ema200.iloc[-1]
        ema200_past = ema200.iloc[-21]
        slope_positive = bool(ema200_now > ema200_past)
        details["MA200 Slope Positive"] = {
            "value": f"EMA200: {ema200_now:.2f} vs {ema200_past:.2f} (20 bars ago)",
            "pass": slope_positive,
        }
        if slope_positive:
            score += 1
    else:
        details["MA200 Slope Positive"] = {"value": "Insufficient data", "pass": False}

    # ---- Criterion 2: Price > EMA200 ----
    if not np.isnan(last_ema200):
        above_ema200 = bool(last_close > last_ema200)
        details["Price > EMA200"] = {
            "value": f"Price {last_close:.2f} vs EMA200 {last_ema200:.2f}",
            "pass": above_ema200,
        }
        if above_ema200:
            score += 1
    else:
        details["Price > EMA200"] = {"value": "EMA200 N/A", "pass": False}

    # ---- Criterion 3: EMA50 > EMA200 (Golden Cross) ----
    if not np.isnan(last_ema50) and not np.isnan(last_ema200):
        golden = bool(last_ema50 > last_ema200)
        details["EMA50 > EMA200"] = {
            "value": f"EMA50 {last_ema50:.2f} vs EMA200 {last_ema200:.2f}",
            "pass": golden,
        }
        if golden:
            score += 1
    else:
        details["EMA50 > EMA200"] = {"value": "N/A", "pass": False}

    # ---- Criterion 4: Momentum — RSI > 50 AND RSI rising ----
    if not np.isnan(last_rsi):
        rsi_above_50 = bool(last_rsi > 50)
        # "rising" = current RSI > RSI 5 bars ago
        if len(rsi.dropna()) >= 6:
            rsi_rising = bool(last_rsi > rsi.dropna().iloc[-6])
        else:
            rsi_rising = False
        momentum = rsi_above_50 and rsi_rising
        trend = "rising" if rsi_rising else "falling"
        details["Momentum (RSI>50↑)"] = {
            "value": f"RSI {last_rsi:.1f} ({trend})",
            "pass": momentum,
        }
        if momentum:
            score += 1
    else:
        details["Momentum (RSI>50↑)"] = {"value": "RSI N/A", "pass": False}

    # ---- Sparkline data: last 120 Close prices ----
    sparkline_data = (
        close.iloc[-120:].round(4).tolist()
        if len(close) >= 2
        else []
    )

    return {
        "tech_score": score,
        "details": details,
        "sparkline_data": sparkline_data,
    }


def get_indicator_series(df: pd.DataFrame) -> dict:
    """
    Return full indicator series for charting purposes.
    Used by the detail expander in the Streamlit page.

    Returns
    -------
    dict with keys:
      close, ema50, ema200, rsi, dates
    All are lists (JSON-serialisable).
    """
    if df is None or df.empty:
        return {}

    close_col = next(
        (c for c in ("Close", "Adj Close", "close") if c in df.columns), None
    )
    if close_col is None:
        return {}

    close = df[close_col].dropna()
    if len(close) < 2:
        return {}

    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    rsi_vals = _rsi(close, 14)

    dates = close.index.strftime("%Y-%m-%d").tolist()

    return {
        "dates": dates,
        "close": close.round(4).tolist(),
        "ema50": ema50.round(4).tolist(),
        "ema200": ema200.round(4).tolist(),
        "rsi": rsi_vals.round(2).tolist(),
    }
