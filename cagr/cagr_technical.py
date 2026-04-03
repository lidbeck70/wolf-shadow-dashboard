"""
cagr_technical.py
Technical scoring module — 0 to 7 points.

Scoring criteria (all computed from OHLCV DataFrame):

── Trend Confirmation (0-3) ── (rules 2, 3)
  1. Price > EMA200                                   → 1 point
  2. EMA50 > EMA200 (Golden Cross)                    → 1 point
  3. EMA200 slope positive (today vs 20 bars ago)     → 1 point

── Stage Analysis (0-2) ── (Weinstein Stage 2)
  4. Stage 2 Confirmed                                → 1 point
     price > EMA200 AND EMA50 > EMA200 AND EMA200 rising AND
     price > 52-week low * 1.25 AND price within 30% of 52-week high
  5. Not overextended                                 → 1 point
     price < EMA50 * 1.20 AND RSI < 75

── Momentum (0-2) ──
  6. RSI > 50 AND RSI rising over 5 bars              → 1 point
  7. 6-month (126-day) relative strength positive     → 1 point
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
# Fear & Greed approximation (informational gate — NOT part of tech_score)
# ---------------------------------------------------------------------------

def compute_fear_greed_gate(df: pd.DataFrame) -> dict:
    """
    Simple Fear & Greed approximation from price data alone.

    Returns dict with:
      - fear_greed_score : float 0-100
      - label            : "Extreme Fear" / "Fear" / "Neutral" / "Greed" / "Extreme Greed"
      - buy_ok           : bool (True if score < 60 — user's rule 5)

    Composite:
      RSI component   : rsi_score = RSI value (0-100)
      EMA200 distance : ema_score = 50 + (pct_above_ema200 * 200), clamped 0-100
      Volatility      : vol_score = 100 - (recent_vol / historical_vol * 50), clamped 0-100
                        where recent_vol  = std(daily returns, 10 days)
                              historical_vol = std(daily returns, 60 days)
      Composite = 0.4 * rsi_score + 0.3 * ema_score + 0.3 * vol_score
    """
    fallback = {
        "fear_greed_score": 50.0,
        "label": "Neutral",
        "buy_ok": True,
    }

    if df is None or df.empty:
        return fallback

    close_col = next(
        (c for c in ("Close", "Adj Close", "close") if c in df.columns), None
    )
    if close_col is None:
        return fallback

    close = df[close_col].dropna()
    if len(close) < 30:
        return fallback

    # RSI component
    rsi_series = _rsi(close, 14)
    last_rsi = rsi_series.iloc[-1]
    rsi_score = float(last_rsi) if not np.isnan(last_rsi) else 50.0

    # EMA200 distance component
    ema200 = _ema(close, 200)
    last_ema200 = ema200.iloc[-1]
    last_close = close.iloc[-1]
    if not np.isnan(last_ema200) and last_ema200 > 0:
        pct_above = (last_close - last_ema200) / last_ema200
        ema_score = 50.0 + pct_above * 200.0
        ema_score = float(np.clip(ema_score, 0.0, 100.0))
    else:
        ema_score = 50.0

    # Volatility component
    returns = close.pct_change().dropna()
    if len(returns) >= 60:
        recent_vol = returns.iloc[-10:].std()
        historical_vol = returns.iloc[-60:].std()
        if historical_vol > 0 and not np.isnan(recent_vol):
            vol_score = 100.0 - (recent_vol / historical_vol * 50.0)
            vol_score = float(np.clip(vol_score, 0.0, 100.0))
        else:
            vol_score = 50.0
    else:
        vol_score = 50.0

    composite = 0.4 * rsi_score + 0.3 * ema_score + 0.3 * vol_score
    composite = float(np.clip(composite, 0.0, 100.0))

    if composite < 25:
        label = "Extreme Fear"
    elif composite < 45:
        label = "Fear"
    elif composite < 55:
        label = "Neutral"
    elif composite < 75:
        label = "Greed"
    else:
        label = "Extreme Greed"

    return {
        "fear_greed_score": round(composite, 1),
        "label": label,
        "buy_ok": composite < 60,
    }


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_technical(df: pd.DataFrame) -> dict:
    """
    Score a stock on 7 technical criteria (0-7 points).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a DatetimeIndex and at least a 'Close' column.
        Output of cagr_loader.fetch_price_data()[ticker].

    Returns
    -------
    dict with keys:
      - tech_score     : int (0-7)
      - details        : dict with criterion → {"value": str, "pass": bool}
      - sparkline_data : list of last 120 Close prices (float) for mini chart
    """
    empty_result = {
        "tech_score": 0,
        "tech_max": 7,
        "details": {
            "Price > EMA200":       {"value": "N/A", "pass": False},
            "EMA50 > EMA200":       {"value": "N/A", "pass": False},
            "EMA200 Slope Positive":{"value": "N/A", "pass": False},
            "Stage 2 Confirmed":    {"value": "N/A", "pass": False},
            "Not Overextended":     {"value": "N/A", "pass": False},
            "Momentum (RSI>50↑)":  {"value": "N/A", "pass": False},
            "RS 6-Month Positive":  {"value": "N/A", "pass": False},
        },
        "sparkline_data": [],
    }

    if df is None or df.empty:
        return empty_result

    # Resolve Close column
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

    # Compute core indicators
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    rsi = _rsi(close, 14)

    last_close = close.iloc[-1]
    last_ema50 = ema50.iloc[-1]
    last_ema200 = ema200.iloc[-1]
    last_rsi = rsi.iloc[-1]

    # ── Trend Confirmation ──────────────────────────────────────────────────

    # Criterion 1: Price > EMA200
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

    # Criterion 2: EMA50 > EMA200 (Golden Cross)
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

    # Criterion 3: EMA200 slope positive (today vs 20 bars ago)
    ema200_clean = ema200.dropna()
    if len(ema200_clean) >= 21:
        ema200_now = ema200_clean.iloc[-1]
        ema200_past = ema200_clean.iloc[-21]
        slope_positive = bool(ema200_now > ema200_past)
        details["EMA200 Slope Positive"] = {
            "value": f"EMA200: {ema200_now:.2f} vs {ema200_past:.2f} (20 bars ago)",
            "pass": slope_positive,
        }
        if slope_positive:
            score += 1
    else:
        details["EMA200 Slope Positive"] = {"value": "Insufficient data", "pass": False}

    # ── Stage Analysis (Weinstein Stage 2) ─────────────────────────────────

    # Precompute 52-week high/low
    lookback = min(252, len(close))
    week52_high = close.iloc[-lookback:].max()
    week52_low = close.iloc[-lookback:].min()

    # Criterion 4: Stage 2 Confirmed
    #   price > EMA200 AND EMA50 > EMA200 AND EMA200 rising
    #   AND price > 52-week low * 1.25 AND price within 30% of 52-week high
    ema200_rising = (
        bool(ema200_clean.iloc[-1] > ema200_clean.iloc[-21])
        if len(ema200_clean) >= 21
        else False
    )
    above_ema200_flag = (
        not np.isnan(last_ema200) and bool(last_close > last_ema200)
    )
    golden_flag = (
        not np.isnan(last_ema50) and not np.isnan(last_ema200)
        and bool(last_ema50 > last_ema200)
    )
    above_low_threshold = bool(last_close > week52_low * 1.25)
    within_high_threshold = bool(last_close >= week52_high * 0.70)  # within 30% of 52w high

    stage2 = (
        above_ema200_flag
        and golden_flag
        and ema200_rising
        and above_low_threshold
        and within_high_threshold
    )
    pct_from_high = (last_close / week52_high - 1) * 100
    pct_from_low = (last_close / week52_low - 1) * 100
    details["Stage 2 Confirmed"] = {
        "value": (
            f"Price {last_close:.2f} | "
            f"52w H {week52_high:.2f} ({pct_from_high:+.1f}%) | "
            f"52w L {week52_low:.2f} ({pct_from_low:+.1f}%)"
        ),
        "pass": stage2,
    }
    if stage2:
        score += 1

    # Criterion 5: Not overextended
    #   price < EMA50 * 1.20 AND RSI < 75
    if not np.isnan(last_ema50) and not np.isnan(last_rsi):
        not_extended_price = bool(last_close < last_ema50 * 1.20)
        not_extended_rsi = bool(last_rsi < 75)
        not_overextended = not_extended_price and not_extended_rsi
        pct_above_ema50 = (last_close / last_ema50 - 1) * 100
        details["Not Overextended"] = {
            "value": (
                f"Price {pct_above_ema50:+.1f}% vs EMA50 | RSI {last_rsi:.1f}"
            ),
            "pass": not_overextended,
        }
        if not_overextended:
            score += 1
    else:
        details["Not Overextended"] = {"value": "N/A", "pass": False}

    # ── Momentum ────────────────────────────────────────────────────────────

    # Criterion 6: RSI > 50 AND RSI rising over 5 bars
    if not np.isnan(last_rsi):
        rsi_above_50 = bool(last_rsi > 50)
        rsi_clean = rsi.dropna()
        if len(rsi_clean) >= 6:
            rsi_rising = bool(last_rsi > rsi_clean.iloc[-6])
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

    # Criterion 7: 6-month (126-day) relative strength positive
    if len(close) >= 127:
        rs_return = float(close.iloc[-1] / close.iloc[-127] - 1)
        rs_positive = rs_return > 0
        details["RS 6-Month Positive"] = {
            "value": f"6M return: {rs_return * 100:+.1f}%",
            "pass": rs_positive,
        }
        if rs_positive:
            score += 1
    else:
        details["RS 6-Month Positive"] = {"value": "Insufficient data (<127 bars)", "pass": False}

    # ---- Sparkline data: last 120 Close prices ----
    sparkline_data = (
        close.iloc[-120:].round(4).tolist()
        if len(close) >= 2
        else []
    )

    return {
        "tech_score": score,
        "tech_max": 7,
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
