"""
orderblocks.py — OVTLYR-style Order Block Detection

Ported from the Deepthought v3.4.1 Pine Script logic:
  - Uses Rate of Change (ROC) crossover to detect momentum shifts
  - Bullish OB = last RED candle before ROC crosses UP above sensitivity
  - Bearish OB = last GREEN candle before ROC crosses DOWN below -sensitivity
  - Two sensitivity levels (28% and 50%) for different-sized OBs
  - Mitigation: close breaks through OB → marked as mitigated
  - Volume analysis: relative volume strength at OB creation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OrderBlock:
    """Single detected Order Block."""
    type: str               # "bullish" or "bearish"
    start_idx: int          # bar index in the DataFrame where OB candle sits
    high: float             # OB zone top
    low: float              # OB zone bottom
    date: str               # date string of OB candle
    volume: float           # volume at OB candle
    vol_strength: float     # relative volume (volume / 20-bar avg)
    sensitivity: int        # which sensitivity set (1 or 2)
    status: str             # "Active" / "Mitigated"
    mitigation_date: str = ""
    mitigation_idx: int = -1


# ---------------------------------------------------------------------------
# Core detection — OVTLYR / Deepthought method
# ---------------------------------------------------------------------------

def detect_orderblocks(
    df: pd.DataFrame,
    sens1: float = 0.28,
    sens2: float = 0.50,
    lookback_candles: int = 15,
    min_gap: int = 5,
    max_blocks: int = 40,
    mitigation_type: str = "close",
) -> List[OrderBlock]:
    """
    Detect Order Blocks using the OVTLYR Rate-of-Change method.

    Logic (from Deepthought Pine Script):
      1. Compute ROC = (open - open[4]) / open[4] * 100
      2. When ROC crosses BELOW -sensitivity → Bearish OB
         Find the last GREEN (close > open) candle within 4-15 bars back
         That candle's high/low = the OB zone
      3. When ROC crosses ABOVE +sensitivity → Bullish OB
         Find the last RED (close < open) candle within 4-15 bars back
      4. Min 5 bars between OBs of the same sensitivity set
      5. Mitigation: if close (or wick) breaks through OB → mark as mitigated

    Parameters
    ----------
    df             : OHLCV DataFrame with DatetimeIndex
    sens1          : Sensitivity level 1 (default 0.28 = 28%)
    sens2          : Sensitivity level 2 (default 0.50 = 50%)
    lookback_candles : How far back to search for the OB candle (4-15)
    min_gap        : Minimum bars between OBs of same set
    max_blocks     : Maximum OBs to keep
    mitigation_type: "close" = close must break OB; "wick" = high/low can break

    Returns list of OrderBlock objects, most recent first.
    """
    if df is None or df.empty or len(df) < 20:
        return []

    # Ensure we have the right columns
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    v = df["Volume"].values.astype(float) if "Volume" in df.columns else np.zeros(len(df))

    n = len(df)

    # Get dates
    if isinstance(df.index, pd.DatetimeIndex):
        dates = [str(d.date()) for d in df.index]
    elif "Date" in df.columns:
        dates = [str(d) for d in pd.to_datetime(df["Date"])]
    else:
        dates = [str(i) for i in range(n)]

    # Compute ROC: (open - open[4]) / open[4] * 100
    roc = np.full(n, 0.0)
    for i in range(4, n):
        if o[i - 4] != 0:
            roc[i] = (o[i] - o[i - 4]) / o[i - 4] * 100

    # Volume relative strength (20-bar SMA)
    vol_avg = np.full(n, 1.0)
    for i in range(20, n):
        avg = np.mean(v[i - 20:i])
        vol_avg[i] = v[i] / avg if avg > 0 else 1.0

    # Helper: detect crossover/crossunder
    def crossover(series, level):
        """True at bar i if series[i] > level and series[i-1] <= level."""
        result = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if series[i] > level and series[i - 1] <= level:
                result[i] = True
        return result

    def crossunder(series, level):
        result = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if series[i] < level and series[i - 1] >= level:
                result[i] = True
        return result

    all_obs: List[OrderBlock] = []

    # Process each sensitivity level
    for sens_num, sens_val in [(1, sens1), (2, sens2)]:
        bearish_cross = crossunder(roc, -sens_val)
        bullish_cross = crossover(roc, sens_val)

        last_cross_idx = -min_gap - 1  # track gap between OBs

        for i in range(lookback_candles + 1, n):
            # ── Bearish OB: ROC crosses under -sensitivity ──
            if bearish_cross[i] and (i - last_cross_idx) > min_gap:
                # Find the last GREEN candle (close > open) within 4-15 bars back
                ob_idx = -1
                for j in range(4, min(lookback_candles + 1, i)):
                    if c[i - j] > o[i - j]:  # green candle
                        ob_idx = i - j
                        break

                if ob_idx >= 0:
                    all_obs.append(OrderBlock(
                        type="bearish",
                        start_idx=ob_idx,
                        high=h[ob_idx],
                        low=l[ob_idx],
                        date=dates[ob_idx],
                        volume=v[ob_idx],
                        vol_strength=round(vol_avg[i], 2),
                        sensitivity=sens_num,
                        status="Active",
                    ))
                    last_cross_idx = i

            # ── Bullish OB: ROC crosses over +sensitivity ──
            if bullish_cross[i] and (i - last_cross_idx) > min_gap:
                # Find the last RED candle (close < open) within 4-15 bars back
                ob_idx = -1
                for j in range(4, min(lookback_candles + 1, i)):
                    if c[i - j] < o[i - j]:  # red candle
                        ob_idx = i - j
                        break

                if ob_idx >= 0:
                    all_obs.append(OrderBlock(
                        type="bullish",
                        start_idx=ob_idx,
                        high=h[ob_idx],
                        low=l[ob_idx],
                        date=dates[ob_idx],
                        volume=v[ob_idx],
                        vol_strength=round(vol_avg[i], 2),
                        sensitivity=sens_num,
                        status="Active",
                    ))
                    last_cross_idx = i

    # ── Validate status: check mitigation ──
    for ob in all_obs:
        idx = ob.start_idx
        for i in range(idx + 1, n):
            if ob.type == "bearish":
                # Bearish OB mitigated when close (or high) goes above OB top
                test_val = c[i] if mitigation_type == "close" else h[i]
                if test_val > ob.high:
                    ob.status = "Mitigated"
                    ob.mitigation_date = dates[i]
                    ob.mitigation_idx = i
                    break
            else:  # bullish
                # Bullish OB mitigated when close (or low) goes below OB bottom
                test_val = c[i] if mitigation_type == "close" else l[i]
                if test_val < ob.low:
                    ob.status = "Mitigated"
                    ob.mitigation_date = dates[i]
                    ob.mitigation_idx = i
                    break

    # Sort by date descending, limit count
    all_obs.sort(key=lambda x: x.start_idx, reverse=True)
    return all_obs[:max_blocks]


# ---------------------------------------------------------------------------
# Price-vs-OB analysis
# ---------------------------------------------------------------------------

def classify_price_vs_ob(
    current_price: float,
    orderblocks: List[OrderBlock],
    proximity_pct: float = 0.02,
) -> dict:
    """
    Classify current price position relative to active order blocks.

    Returns dict with:
      nearest_bullish_ob : OrderBlock or None
      nearest_bearish_ob : OrderBlock or None
      approaching_bullish: bool (price within proximity_pct of bullish OB high)
      approaching_bearish: bool (price within proximity_pct of bearish OB low)
      inside_ob          : bool (price is inside an active OB zone)
      signal_bias        : "BUY" / "SELL" / "HOLD" / "REDUCE"
      active_count       : int
      mitigated_count    : int
    """
    active_bullish = [ob for ob in orderblocks if ob.type == "bullish" and ob.status == "Active"]
    active_bearish = [ob for ob in orderblocks if ob.type == "bearish" and ob.status == "Active"]

    active_count = len(active_bullish) + len(active_bearish)
    mitigated_count = sum(1 for ob in orderblocks if ob.status == "Mitigated")

    # Find nearest OBs
    nearest_bull = None
    nearest_bear = None
    min_bull_dist = float("inf")
    min_bear_dist = float("inf")

    for ob in active_bullish:
        dist = abs(current_price - ob.high)
        if dist < min_bull_dist:
            min_bull_dist = dist
            nearest_bull = ob

    for ob in active_bearish:
        dist = abs(current_price - ob.low)
        if dist < min_bear_dist:
            min_bear_dist = dist
            nearest_bear = ob

    # Proximity checks
    approaching_bull = False
    approaching_bear = False
    inside_ob = False

    if nearest_bull and current_price > 0:
        if abs(current_price - nearest_bull.high) / current_price < proximity_pct:
            approaching_bull = True
        if nearest_bull.low <= current_price <= nearest_bull.high:
            inside_ob = True

    if nearest_bear and current_price > 0:
        if abs(current_price - nearest_bear.low) / current_price < proximity_pct:
            approaching_bear = True
        if nearest_bear.low <= current_price <= nearest_bear.high:
            inside_ob = True

    # Signal bias
    if approaching_bull and not approaching_bear:
        signal_bias = "BUY"
    elif approaching_bear and not approaching_bull:
        signal_bias = "SELL"
    elif inside_ob:
        signal_bias = "HOLD"
    elif active_bearish and current_price > active_bearish[0].high:
        signal_bias = "REDUCE"  # broke through bearish OB
    else:
        signal_bias = "HOLD"

    return {
        "nearest_bullish_ob": nearest_bull,
        "nearest_bearish_ob": nearest_bear,
        "approaching_bullish": approaching_bull,
        "approaching_bearish": approaching_bear,
        "inside_ob": inside_ob,
        "signal_bias": signal_bias,
        "active_count": active_count,
        "mitigated_count": mitigated_count,
    }
