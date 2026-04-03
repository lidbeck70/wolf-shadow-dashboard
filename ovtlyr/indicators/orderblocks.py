"""
Order Block detection for OVTLYR.

An Order Block is the last counter-trend candle before a strong impulse move
that creates a Break of Structure (BOS). These zones represent institutional
supply/demand and often act as future support/resistance.

Pure functions — no Streamlit imports.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class OrderBlock:
    type: str            # "bullish" or "bearish"
    start_idx: int       # integer bar index (iloc position) of the OB candle
    high: float          # OB zone high
    low: float           # OB zone low
    date: str            # ISO date string of OB candle
    volume: float        # volume at OB candle
    status: str          # "Active" | "Mitigated" | "Invalidated"
    mitigation_date: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR using EWM smoothing aligned to df index."""
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


def _impulse_move(df: pd.DataFrame, start_iloc: int, direction: str, max_bars: int = 3) -> float:
    """
    Return the total net move (signed, in price units) across up to *max_bars*
    starting at *start_iloc* in the given *direction* ("up" or "down").
    """
    end_iloc = min(start_iloc + max_bars, len(df))
    slice_df = df.iloc[start_iloc:end_iloc]
    if slice_df.empty:
        return 0.0
    if direction == "up":
        return float(slice_df["Close"].iloc[-1] - slice_df["Open"].iloc[0])
    else:
        return float(slice_df["Open"].iloc[0] - slice_df["Close"].iloc[-1])


def _avg_volume(df: pd.DataFrame, end_iloc: int, window: int = 20) -> float:
    """Simple average volume over *window* bars ending at *end_iloc* (exclusive)."""
    start = max(0, end_iloc - window)
    vol = df["Volume"].iloc[start:end_iloc]
    if vol.empty:
        return float("nan")
    return float(vol.mean())


def _swing_high(df: pd.DataFrame, end_iloc: int, lookback: int = 10) -> float:
    """Highest High in the *lookback* bars before *end_iloc*."""
    start = max(0, end_iloc - lookback)
    return float(df["High"].iloc[start:end_iloc].max())


def _swing_low(df: pd.DataFrame, end_iloc: int, lookback: int = 10) -> float:
    """Lowest Low in the *lookback* bars before *end_iloc*."""
    start = max(0, end_iloc - lookback)
    return float(df["Low"].iloc[start:end_iloc].min())


def _validate_status(
    ob: OrderBlock,
    df: pd.DataFrame,
    ob_iloc: int,
) -> OrderBlock:
    """
    Update ob.status and ob.mitigation_date by scanning price action
    after the OB candle (all bars from ob_iloc + 1 onwards).

    Mitigated   : price has wicked into the OB zone (Low <= zone_high and High >= zone_low)
    Invalidated :
        Bullish OB – a candle closed below ob.low
        Bearish OB – a candle closed above ob.high
    Active      : none of the above
    """
    future = df.iloc[ob_iloc + 1:]
    if future.empty:
        return ob

    for i, row in future.iterrows():
        high_i = row["High"]
        low_i = row["Low"]
        close_i = row["Close"]
        date_str = str(i.date()) if hasattr(i, "date") else str(i)

        if ob.type == "bullish":
            # Invalidated: close below OB low
            if close_i < ob.low:
                ob.status = "Invalidated"
                ob.mitigation_date = date_str
                return ob
            # Mitigated: price has touched the OB zone
            if low_i <= ob.high and high_i >= ob.low:
                ob.status = "Mitigated"
                ob.mitigation_date = date_str
                # Do NOT return — invalidation later overrides mitigation

        else:  # bearish
            # Invalidated: close above OB high
            if close_i > ob.high:
                ob.status = "Invalidated"
                ob.mitigation_date = date_str
                return ob
            # Mitigated
            if high_i >= ob.low and low_i <= ob.high:
                ob.status = "Mitigated"
                ob.mitigation_date = date_str

    return ob


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def detect_orderblocks(
    df: pd.DataFrame,
    lookback: int = 100,
    impulse_factor: float = 1.5,
) -> List[OrderBlock]:
    """
    Detect Bullish and Bearish Order Blocks in the OHLCV DataFrame.

    Bullish OB — last bearish (red) candle before a strong bullish impulse:
        1. Candle closes below its open.
        2. Over the next 1-3 candles, total upward move > impulse_factor × ATR(14).
        3. Average volume of impulse candles > 1.2× 20-bar average volume.
        4. BOS: impulse closes above the highest High of the preceding 10 bars.

    Bearish OB — last bullish (green) candle before a strong bearish impulse:
        1. Candle closes above its open.
        2. Over the next 1-3 candles, total downward move > impulse_factor × ATR(14).
        3. Volume confirmation (same as above).
        4. BOS: impulse closes below the lowest Low of the preceding 10 bars.

    Status assigned per _validate_status rules.

    Returns a list of OrderBlock objects, most-recent first.
    """
    if df is None or df.empty:
        return []

    # Work on a clean, reset-indexed copy of the last *lookback* bars
    df_work = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    df_work = df_work.iloc[-lookback:].reset_index(drop=False)
    # Keep original index (dates) in a column called "orig_index"
    orig_col = df_work.columns[0]   # first column is the original index

    atr_series = _atr(df_work.rename(columns={orig_col: "_idx"}).set_index("_idx")[["High", "Low", "Close", "Open", "Volume"]])
    # Realign ATR to integer positions
    atr_values = atr_series.values

    n = len(df_work)
    max_impulse_bars = 3
    bos_lookback = 10
    vol_confirm_factor = 1.2
    order_blocks: List[OrderBlock] = []

    for i in range(n - max_impulse_bars - 1):
        row = df_work.iloc[i]
        open_i = row["Open"]
        close_i = row["Close"]
        high_i = row["High"]
        low_i = row["Low"]
        vol_i = row["Volume"]
        atr_i = float(atr_values[i]) if not np.isnan(atr_values[i]) else 0.0
        date_i = str(row[orig_col])
        if hasattr(row[orig_col], "date"):
            date_i = str(row[orig_col].date())

        avg_vol = _avg_volume(df_work, i, window=20)
        min_impulse = impulse_factor * atr_i

        # ---- Bullish OB: bearish candle → bullish impulse ----
        if close_i < open_i:
            # Check impulse over next 1-3 bars
            for bars in range(1, max_impulse_bars + 1):
                end = i + bars + 1
                if end > n:
                    break
                impulse = _impulse_move(df_work, i + 1, "up", bars)
                if impulse < min_impulse:
                    continue

                # Volume check on impulse bars
                impulse_vol = float(df_work["Volume"].iloc[i + 1: i + 1 + bars].mean())
                if avg_vol > 0 and impulse_vol < vol_confirm_factor * avg_vol:
                    continue

                # BOS: impulse close must exceed swing high of 10 bars before OB
                swing_hi = _swing_high(df_work, i, bos_lookback)
                impulse_close = float(df_work["Close"].iloc[i + bars])
                if impulse_close <= swing_hi:
                    continue

                # Valid bullish OB found
                ob = OrderBlock(
                    type="bullish",
                    start_idx=i,
                    high=float(high_i),
                    low=float(low_i),
                    date=date_i,
                    volume=float(vol_i),
                    status="Active",
                )
                ob = _validate_status(ob, df_work.set_index(orig_col)[["Open", "High", "Low", "Close", "Volume"]], i)
                order_blocks.append(ob)
                break  # one OB per candle

        # ---- Bearish OB: bullish candle → bearish impulse ----
        elif close_i > open_i:
            for bars in range(1, max_impulse_bars + 1):
                end = i + bars + 1
                if end > n:
                    break
                impulse = _impulse_move(df_work, i + 1, "down", bars)
                if impulse < min_impulse:
                    continue

                impulse_vol = float(df_work["Volume"].iloc[i + 1: i + 1 + bars].mean())
                if avg_vol > 0 and impulse_vol < vol_confirm_factor * avg_vol:
                    continue

                swing_lo = _swing_low(df_work, i, bos_lookback)
                impulse_close = float(df_work["Close"].iloc[i + bars])
                if impulse_close >= swing_lo:
                    continue

                ob = OrderBlock(
                    type="bearish",
                    start_idx=i,
                    high=float(high_i),
                    low=float(low_i),
                    date=date_i,
                    volume=float(vol_i),
                    status="Active",
                )
                ob = _validate_status(ob, df_work.set_index(orig_col)[["Open", "High", "Low", "Close", "Volume"]], i)
                order_blocks.append(ob)
                break

    # Most recent first
    order_blocks.sort(key=lambda ob: ob.start_idx, reverse=True)
    return order_blocks


def classify_price_vs_ob(
    current_price: float,
    orderblocks: List[OrderBlock],
) -> dict:
    """
    Classify current price position relative to active order blocks.

    Returns a dict with:
        nearest_bullish_ob  : OrderBlock or None – nearest active bullish OB below price
        nearest_bearish_ob  : OrderBlock or None – nearest active bearish OB above price
        approaching_bullish : bool – price within 2% of bullish OB high
        approaching_bearish : bool – price within 2% of bearish OB low
        reacting_to         : "bullish_ob" | "bearish_ob" | None
        signal_bias         : "BUY" | "SELL" | "HOLD" | "REDUCE"

    Signal bias logic
    -----------------
    BUY    : price bouncing from bullish OB (low touched zone, close above OB high)
    SELL   : price bouncing from bearish OB (high touched zone, close below OB low)
    HOLD   : price between OBs, no immediate reaction
    REDUCE : price breaking through an OB against expected direction (OB invalidated)
    """
    active_bullish = [ob for ob in orderblocks if ob.type == "bullish" and ob.status == "Active"]
    active_bearish = [ob for ob in orderblocks if ob.type == "bearish" and ob.status == "Active"]
    invalidated_any = [ob for ob in orderblocks if ob.status == "Invalidated"]

    # Nearest bullish OB below current price
    below_bullish = [ob for ob in active_bullish if ob.high < current_price]
    nearest_bullish_ob: Optional[OrderBlock] = (
        max(below_bullish, key=lambda ob: ob.high) if below_bullish else None
    )

    # Nearest bearish OB above current price
    above_bearish = [ob for ob in active_bearish if ob.low > current_price]
    nearest_bearish_ob: Optional[OrderBlock] = (
        min(above_bearish, key=lambda ob: ob.low) if above_bearish else None
    )

    # Approaching flags (within 2%)
    approaching_bullish = False
    if nearest_bullish_ob is not None and nearest_bullish_ob.high > 0:
        approaching_bullish = abs(current_price - nearest_bullish_ob.high) / nearest_bullish_ob.high < 0.02

    approaching_bearish = False
    if nearest_bearish_ob is not None and nearest_bearish_ob.low > 0:
        approaching_bearish = abs(nearest_bearish_ob.low - current_price) / nearest_bearish_ob.low < 0.02

    # Reacting to
    reacting_to: Optional[str] = None
    if nearest_bullish_ob is not None:
        # Price is within the bullish OB zone
        if nearest_bullish_ob.low <= current_price <= nearest_bullish_ob.high:
            reacting_to = "bullish_ob"
    if nearest_bearish_ob is not None:
        if nearest_bearish_ob.low <= current_price <= nearest_bearish_ob.high:
            reacting_to = "bearish_ob"

    # Signal bias
    signal_bias = "HOLD"

    # REDUCE: most-recent OB got invalidated (price broke through)
    if invalidated_any:
        signal_bias = "REDUCE"

    # BUY: approaching or reacting to bullish OB
    if nearest_bullish_ob is not None and approaching_bullish:
        signal_bias = "BUY"
    if reacting_to == "bullish_ob":
        signal_bias = "BUY"

    # SELL: approaching or reacting to bearish OB (overrides BUY if both simultaneously)
    if nearest_bearish_ob is not None and approaching_bearish:
        signal_bias = "SELL"
    if reacting_to == "bearish_ob":
        signal_bias = "SELL"

    return {
        "nearest_bullish_ob": nearest_bullish_ob,
        "nearest_bearish_ob": nearest_bearish_ob,
        "approaching_bullish": approaching_bullish,
        "approaching_bearish": approaching_bearish,
        "reacting_to": reacting_to,
        "signal_bias": signal_bias,
    }
