"""
indicators.py — WOLF x SHADOW Optimization Pipeline
====================================================
Fully vectorized indicator calculations using numpy/pandas.
NO per-bar Python loops.  All functions accept a params dict for easy
sweep by the optimizer.

Regime score exactly mirrors Pine Script WOLF x SHADOW v2:
  market (30) + sector (30) + stock (50) + ichimoku (15) = 125 max
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Memory

from data_loader import _CACHE_DIR

logger = logging.getLogger(__name__)

memory = Memory(location=str(_CACHE_DIR), verbose=0)

# ---------------------------------------------------------------------------
# Default parameter dict (mirrors Pine Script defaults)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict = {
    # Ichimoku
    "tenkan_len": 9,
    "kijun_len": 26,
    "spanb_len": 52,
    "displacement": 26,
    # EMA stack
    "ema_pulse": 10,    # EMA10
    "ema_fast": 20,     # EMA20
    "ema_slow": 50,     # EMA50
    "ema_macro": 200,   # EMA200
    # RSI / momentum
    "rsi_len": 14,
    "rsi_hot": 70,
    # Extension
    "ext_pct": 2.7,
    # ATR
    "atr_len": 14,
    "atr_mult": 2.5,
    # Order block
    "ob_lookback": 5,
    # Exit / TP
    "tp1_rr": 2.5,
    "tp1_pct": 0.15,
    "tp2_rr": 4.0,
    "tp2_pct": 0.15,
    # Position sizing
    "core_pct": 0.50,
    "trim_pct": 0.50,
    "add_pct": 0.10,
    # Entry gate
    "entry_min_regime": 40,
    "add_min_regime": 50,
    # Cooldown
    "cooldown_bars": 4,
}


# ---------------------------------------------------------------------------
# Low-level vectorized primitives
# ---------------------------------------------------------------------------

def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average — pandas ewm (matches Pine Script ta.ema)."""
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Wilder RSI matching Pine Script ta.rsi (RMA smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range using Wilder RMA (matches Pine Script ta.atr)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def donchian(high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    """(highest_high + lowest_low) / 2 over 'length' bars — Pine's donchian."""
    return (high.rolling(length).max() + low.rolling(length).min()) / 2.0


def volume_ma(volume: pd.Series, length: int = 20) -> pd.Series:
    """Simple moving average of volume."""
    return volume.rolling(length).mean()


def rolling_highest(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).max()


def rolling_lowest(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).min()


# ---------------------------------------------------------------------------
# EMA Stack indicators
# ---------------------------------------------------------------------------

def compute_ema_stack(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Compute EMA10/20/50/200 and derived boolean signals.

    Returns DataFrame with columns:
      ema_pulse, ema_fast, ema_slow, ema_macro,
      ema_trend, ema_stack_full, above_macro
    """
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out["ema_pulse"] = ema(close, params.get("ema_pulse", 10))
    out["ema_fast"]  = ema(close, params.get("ema_fast",  20))
    out["ema_slow"]  = ema(close, params.get("ema_slow",  50))
    out["ema_macro"] = ema(close, params.get("ema_macro", 200))

    # emaTrend: ema10>ema20>ema50 AND close > ema50
    out["ema_trend"] = (
        (out["ema_pulse"] > out["ema_fast"]) &
        (out["ema_fast"]  > out["ema_slow"]) &
        (close            > out["ema_slow"])
    )
    # Full stack: all 4 stacked
    out["ema_stack_full"] = (
        (out["ema_pulse"] > out["ema_fast"]) &
        (out["ema_fast"]  > out["ema_slow"]) &
        (out["ema_slow"]  > out["ema_macro"])
    )
    out["above_macro"] = close > out["ema_macro"]
    return out


# ---------------------------------------------------------------------------
# Ichimoku
# ---------------------------------------------------------------------------

def compute_ichimoku(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Compute Ichimoku components and sub-scores.

    Displacement shifts senkouA/B forward — Pine uses plotted offset.
    For backtesting we use the current-bar cloud (displaced back by i_displacement
    so that cloud[displacement] aligns with current bar).

    Returns DataFrame with columns:
      tenkan, kijun, senkouA, senkouB, kumo_top, kumo_bottom,
      ichi_above_kumo, ichi_tk_bull, ichi_chikou_ok, ichi_bull_twist,
      ichi_score  (0-15)
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    disp  = int(params.get("displacement", 26))

    out = pd.DataFrame(index=df.index)
    out["tenkan"]  = donchian(high, low, int(params.get("tenkan_len", 9)))
    out["kijun"]   = donchian(high, low, int(params.get("kijun_len",  26)))
    out["senkouA"] = (out["tenkan"] + out["kijun"]) / 2.0
    out["senkouB"] = donchian(high, low, int(params.get("spanb_len", 52)))

    # The cloud used for entry checks: shift senkouA/B backward by displacement
    # so that we compare current price against the cloud that Pine displays
    # at offset=displacement from current bar
    sA_shifted = out["senkouA"].shift(disp)
    sB_shifted = out["senkouB"].shift(disp)

    out["kumo_top"]    = sA_shifted.combine(sB_shifted, np.maximum)
    out["kumo_bottom"] = sA_shifted.combine(sB_shifted, np.minimum)

    # Ichimoku sub-scores (matching Pine Script)
    out["ichi_above_kumo"] = close > out["kumo_top"]
    out["ichi_tk_bull"]    = out["tenkan"] > out["kijun"]
    # Chikou: current close vs close[displacement] bars ago
    out["ichi_chikou_ok"]  = close > close.shift(disp)
    out["ichi_bull_twist"] = out["senkouA"] > out["senkouB"]

    out["ichi_score"] = (
        out["ichi_above_kumo"].astype(int) * 5 +
        out["ichi_tk_bull"].astype(int)    * 5 +
        out["ichi_chikou_ok"].astype(int)  * 3 +
        out["ichi_bull_twist"].astype(int) * 2
    )
    return out


# ---------------------------------------------------------------------------
# RSI / Momentum / Extension
# ---------------------------------------------------------------------------

def compute_momentum(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Returns:
      rsi_val, over_extended, momentum_up, momentum_down, weakness_seq
    """
    close = df["close"]
    rsi_len  = int(params.get("rsi_len",  14))
    rsi_hot  = float(params.get("rsi_hot", 70))
    ext_pct  = float(params.get("ext_pct", 2.7))
    ema_pulse_col = ema(close, int(params.get("ema_pulse", 10)))

    out = pd.DataFrame(index=df.index)
    out["rsi_val"] = rsi(close, rsi_len)

    ext_decimal = ext_pct / 100.0
    out["over_extended"] = (
        (close > ema_pulse_col * (1.0 + ext_decimal)) |
        (out["rsi_val"] > rsi_hot)
    )

    # RSI acceleration (three rising bars)
    out["momentum_up"]   = (
        (out["rsi_val"] > out["rsi_val"].shift(1)) &
        (out["rsi_val"].shift(1) > out["rsi_val"].shift(2))
    )
    out["momentum_down"] = (
        (out["rsi_val"] < out["rsi_val"].shift(1)) &
        (out["rsi_val"].shift(1) < out["rsi_val"].shift(2))
    )
    out["weakness_seq"] = (
        (close < ema_pulse_col) &
        (close.shift(1) < ema_pulse_col.shift(1))
    )
    return out


# ---------------------------------------------------------------------------
# Order Blocks (vectorized)
# ---------------------------------------------------------------------------

def compute_order_blocks(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Vectorized order block detection.

    Bullish OB: low <= lowest_low of prev ob_lookback bars AND close > open
    Bearish OB: high >= highest_high of prev ob_lookback bars AND close < open
    """
    lookback = int(params.get("ob_lookback", 5))
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    open_ = df["open"]

    out = pd.DataFrame(index=df.index)
    # Pine Script: ta.lowest(low, lookback)[1] — lowest of PREVIOUS lookback bars
    prev_lowest  = low.shift(1).rolling(lookback).min()
    prev_highest = high.shift(1).rolling(lookback).max()

    out["bull_ob"] = (low <= prev_lowest)  & (close > open_)
    out["bear_ob"] = (high >= prev_highest) & (close < open_)
    return out


# ---------------------------------------------------------------------------
# EMA Events
# ---------------------------------------------------------------------------

def compute_ema_events(
    df: pd.DataFrame,
    ema_stack: pd.DataFrame,
) -> pd.DataFrame:
    """
    EMA crossover and reclaim events.

    emaCrossUp: ema_pulse crosses above ema_fast AND close > ema_slow
    emaReclaim:  close crosses above ema_pulse (was below, now above)
    """
    close     = df["close"]
    ema_pulse = ema_stack["ema_pulse"]
    ema_fast  = ema_stack["ema_fast"]
    ema_slow  = ema_stack["ema_slow"]

    out = pd.DataFrame(index=df.index)
    # crossover: current bar pulse > fast, previous bar pulse <= fast
    out["ema_cross_up"] = (
        (ema_pulse > ema_fast) &
        (ema_pulse.shift(1) <= ema_fast.shift(1)) &
        (close > ema_slow)
    )
    # reclaim: close crosses above ema_pulse
    out["ema_reclaim"] = (
        (close > ema_pulse) &
        (close.shift(1) <= ema_pulse.shift(1))
    )
    return out


# ---------------------------------------------------------------------------
# Regime Score — 4-layer (market + sector + stock + ichimoku)
# ---------------------------------------------------------------------------

def compute_market_score(spy_df: pd.DataFrame) -> pd.Series:
    """
    Market layer (max 30) using SPY data aligned to stock index.
    Matches Pine Script marketScore logic.
    """
    spy_close  = spy_df["close"]
    spy_ema50  = ema(spy_close, 50)
    spy_ema200 = ema(spy_close, 200)
    spy_rsi    = rsi(spy_close, 14)

    # ATR-based volatility check
    spy_atr    = atr(spy_df["high"], spy_df["low"], spy_close, 14)
    spy_atr_pct = spy_atr / spy_close.replace(0, np.nan) * 100.0

    score = (
        (spy_close > spy_ema50).astype(int)  * 10 +
        (spy_close > spy_ema200).astype(int) * 10 +
        (spy_rsi   > 50).astype(int)         * 5  +
        ((spy_atr_pct > 0.3) & (spy_atr_pct < 4.0)).astype(int) * 5
    )
    return score.rename("market_score")


def compute_sector_score(sector_df: pd.DataFrame) -> pd.Series:
    """
    Sector layer (max 30).
    Matches Pine Script sectorScore logic.
    """
    sec_close  = sector_df["close"]
    sec_ema50  = ema(sec_close, 50)
    sec_ema200 = ema(sec_close, 200)
    sec_rsi    = rsi(sec_close, 14)

    score = (
        (sec_close > sec_ema50).astype(int)  * 10 +
        (sec_close > sec_ema200).astype(int) * 10 +
        (sec_rsi   > 50).astype(int)         * 10
    )
    return score.rename("sector_score")


def compute_stock_score(
    df: pd.DataFrame,
    ema_stack: pd.DataFrame,
    momentum_df: pd.DataFrame,
) -> pd.Series:
    """
    Stock layer (max 50).
    Matches Pine Script stockScore.
    """
    close = df["close"]
    score = (
        (ema_stack["ema_pulse"] > ema_stack["ema_fast"]).astype(int)  * 8 +
        (ema_stack["ema_fast"]  > ema_stack["ema_slow"]).astype(int)  * 8 +
        (close                  > ema_stack["ema_slow"]).astype(int)  * 8 +
        (close                  > ema_stack["ema_macro"]).astype(int) * 8 +
        (momentum_df["rsi_val"] > 50).astype(int)                     * 8 +
        momentum_df["momentum_up"].astype(int)                        * 10
    )
    return score.rename("stock_score")


def compute_regime_score(
    df: pd.DataFrame,
    params: dict,
    spy_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Full 4-layer regime score (0-125).

    Parameters
    ----------
    df : stock OHLCV
    params : parameter dict
    spy_df : SPY OHLCV aligned to df.index (optional — if None, market score = 0)
    sector_df : Sector ETF OHLCV aligned to df.index (optional — if None, sector score = 0)

    Returns DataFrame with all sub-scores and total regime_score.
    """
    ema_stack  = compute_ema_stack(df, params)
    ichi       = compute_ichimoku(df, params)
    momentum_d = compute_momentum(df, params)

    if spy_df is not None:
        market_sc = compute_market_score(spy_df).reindex(df.index).ffill().fillna(0)
    else:
        market_sc = pd.Series(0, index=df.index, name="market_score")

    if sector_df is not None:
        sector_sc = compute_sector_score(sector_df).reindex(df.index).ffill().fillna(0)
    else:
        sector_sc = pd.Series(0, index=df.index, name="sector_score")

    stock_sc  = compute_stock_score(df, ema_stack, momentum_d)
    ichi_sc   = ichi["ichi_score"]

    regime = (
        market_sc.values +
        sector_sc.values +
        stock_sc.values +
        ichi_sc.values
    ).clip(0, 125)

    ob_df    = compute_order_blocks(df, params)
    ema_ev   = compute_ema_events(df, ema_stack)
    atr_vals = atr(df["high"], df["low"], df["close"], int(params.get("atr_len", 14)))

    out = pd.DataFrame(index=df.index)
    out["regime_score"] = regime
    out["market_score"] = market_sc.values
    out["sector_score"] = sector_sc.values
    out["stock_score"]  = stock_sc.values
    out["ichi_score"]   = ichi_sc.values

    # EMA stack
    for col in ema_stack.columns:
        out[col] = ema_stack[col].values

    # Ichimoku
    for col in ["tenkan", "kijun", "senkouA", "senkouB", "kumo_top", "kumo_bottom"]:
        out[col] = ichi[col].values

    # Momentum
    for col in ["rsi_val", "over_extended", "momentum_up", "momentum_down", "weakness_seq"]:
        out[col] = momentum_d[col].values

    # OB
    out["bull_ob"] = ob_df["bull_ob"].values
    out["bear_ob"] = ob_df["bear_ob"].values

    # EMA events
    out["ema_cross_up"] = ema_ev["ema_cross_up"].values
    out["ema_reclaim"]  = ema_ev["ema_reclaim"].values

    # ATR
    out["atr_val"] = atr_vals.values

    # Dynamic sizing helpers (from Pine Script lines 212-216)
    reg = out["regime_score"].values
    out["add_mult"] = np.where(
        reg >= 90, 2.0,
        np.where(reg >= 70, 1.25,
                 np.where(reg >= 50, 0.75, 0.3))
    )
    core_pct = float(params.get("core_pct", 0.50)) * 100
    trim_pct = (1.0 - float(params.get("core_pct", 0.50))) * 100
    out["core_size_adj"] = np.where(
        reg >= 70, core_pct,
        np.where(reg >= 50, core_pct * 0.7, core_pct * 0.4)
    )
    out["trim_size_adj"] = np.where(
        reg >= 70, trim_pct,
        np.where(reg >= 50, trim_pct * 0.7, trim_pct * 0.4)
    )

    return out


# ---------------------------------------------------------------------------
# Cached wrapper for optimization loops
# ---------------------------------------------------------------------------

def _params_key(params: dict) -> str:
    """Short hash of parameter dict for cache keying."""
    s = str(sorted(params.items()))
    return hashlib.md5(s.encode()).hexdigest()[:12]


@memory.cache
def compute_indicators_cached(
    df_hash: str,
    params_key: str,
    # actual data must be passed as values for cache to work
    df_values: np.ndarray,
    df_columns: list,
    df_index: np.ndarray,
    params: dict,
    spy_values: Optional[np.ndarray] = None,
    spy_columns: Optional[list] = None,
    spy_index: Optional[np.ndarray] = None,
    sector_values: Optional[np.ndarray] = None,
    sector_columns: Optional[list] = None,
    sector_index: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, list, np.ndarray]:
    """Joblib-cached indicator computation. Returns (values, columns, index)."""
    df = pd.DataFrame(df_values, columns=df_columns,
                      index=pd.DatetimeIndex(df_index))

    spy_df = None
    if spy_values is not None:
        spy_df = pd.DataFrame(spy_values, columns=spy_columns,
                               index=pd.DatetimeIndex(spy_index))

    sector_df = None
    if sector_values is not None:
        sector_df = pd.DataFrame(sector_values, columns=sector_columns,
                                  index=pd.DatetimeIndex(sector_index))

    result = compute_regime_score(df, params, spy_df=spy_df, sector_df=sector_df)
    return result.values, list(result.columns), result.index.values


def get_indicators(
    df: pd.DataFrame,
    params: dict,
    spy_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None,
    df_hash_str: str = "",
) -> pd.DataFrame:
    """
    Public entry point with joblib caching.
    Pass df_hash_str from data_loader.df_hash(df) to enable cache hits.
    """
    pk = _params_key(params)

    spy_v = spy_df.values if spy_df is not None else None
    spy_c = list(spy_df.columns) if spy_df is not None else None
    spy_i = spy_df.index.values if spy_df is not None else None

    sec_v = sector_df.values if sector_df is not None else None
    sec_c = list(sector_df.columns) if sector_df is not None else None
    sec_i = sector_df.index.values if sector_df is not None else None

    vals, cols, idx = compute_indicators_cached(
        df_hash_str, pk,
        df.values, list(df.columns), df.index.values,
        params,
        spy_v, spy_c, spy_i,
        sec_v, sec_c, sec_i,
    )
    return pd.DataFrame(vals, columns=cols, index=pd.DatetimeIndex(idx))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_loader import load_yfinance

    df   = load_yfinance("XOM",  years=2, interval="1h")
    spy  = load_yfinance("SPY",  years=2, interval="1h")
    sec  = load_yfinance("XLE",  years=2, interval="1h")

    from data_loader import align_to_stock
    spy_a = align_to_stock(df, spy)
    sec_a = align_to_stock(df, sec)

    ind = compute_regime_score(df, DEFAULT_PARAMS, spy_df=spy_a, sector_df=sec_a)
    print(ind[["regime_score", "ema_trend", "rsi_val", "bull_ob"]].tail(10))
    print("Regime score range:", ind["regime_score"].min(), "–", ind["regime_score"].max())
