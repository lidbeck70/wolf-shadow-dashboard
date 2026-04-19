"""
Wolf x Shadow Strategy
======================
Swing-trading strategy combining a 4-layer regime score (Market / Sector /
Stock / Ichimoku) with EMA-stack entries, ATR-based stop-losses, and a
Kijun-sen + EMA10 trailing exit.

Entry (all required):
  1. EMA10 > EMA21 > EMA50 > EMA200 (full stack bullish)
  2. RSI 14 between entry_rsi_low and entry_rsi_high (default 45–70)
  3. ADX 14 >= adx_threshold (default 19)
  4. Price > EMA50

Exit (first trigger wins):
  • Stop-loss:    price <= entry − atr_mult × ATR14
  • Core exit:    price < EMA50 for core_exit_bars consecutive bars
  • Trail exit:   price < Kijun AND price < EMA10

Risk / sizing:
  • Risk per trade = capital × risk_pct (default 2%)
  • Stop distance  = atr_mult × ATR14
  • Shares         = risk_amount / stop_distance
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict = {
    # EMA periods
    "ema_pulse": 10,
    "ema_fast": 21,
    "ema_slow": 50,
    "ema_macro": 200,
    # Ichimoku
    "kijun_period": 26,
    # RSI
    "rsi_period": 14,
    "entry_rsi_low": 45,
    "entry_rsi_high": 70,
    # ADX
    "adx_period": 14,
    "adx_threshold": 19,
    # ATR
    "atr_period": 14,
    "atr_mult": 2.5,
    # Partial exits
    "tp1_r": 2.6,
    "tp1_pct": 0.13,
    "tp2_r": 5.2,
    "tp2_pct": 0.17,
    # Position
    "core_pct": 0.62,
    "risk_pct": 0.02,
    # State machine
    "core_exit_bars": 3,
    "daily_breaker": -0.08,
    # Minimum regime score to enter (out of 125)
    "entry_min_score": 40,
}


# ---------------------------------------------------------------------------
# Internal indicator helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    plus_dm  = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > (-l.diff()).clip(lower=0), 0.0)
    minus_dm = minus_dm.where(minus_dm > h.diff().clip(lower=0), 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_v    = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_v)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_v)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(span=period, adjust=False).mean()


def _kijun(df: pd.DataFrame, period: int = 26) -> pd.Series:
    return (df["High"].rolling(period).max() + df["Low"].rolling(period).min()) / 2


# ---------------------------------------------------------------------------
# entry_fn
# ---------------------------------------------------------------------------

def entry_fn(df: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Evaluate Wolf x Shadow entry conditions on the latest bar of *df*.

    Parameters
    ----------
    df     : OHLCV DataFrame (DatetimeIndex, columns: Open High Low Close Volume)
    params : override dict; merged with DEFAULT_PARAMS

    Returns
    -------
    dict with keys:
      signal        : "BUY" | "HOLD"
      confidence    : int 0-100   (proportion of gates that passed × 100)
      entry_price   : float | None
      stop_loss     : float | None  (entry − atr_mult × ATR)
      tp1_price     : float | None
      tp2_price     : float | None
      gates         : list of {rule: str, passed: bool, value: str}
      reasons       : list[str]
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    if df is None or len(df) < 210:
        return {"signal": "HOLD", "confidence": 0, "entry_price": None,
                "stop_loss": None, "tp1_price": None, "tp2_price": None,
                "gates": [], "reasons": ["Insufficient data"]}

    close = df["Close"]
    e10  = _ema(close, p["ema_pulse"])
    e21  = _ema(close, p["ema_fast"])
    e50  = _ema(close, p["ema_slow"])
    e200 = _ema(close, p["ema_macro"])
    rsi  = _rsi(close, p["rsi_period"])
    atr  = _atr(df, p["atr_period"])
    adx  = _adx(df, p["adx_period"])

    price    = float(close.iloc[-1])
    e10_v    = float(e10.iloc[-1])
    e21_v    = float(e21.iloc[-1])
    e50_v    = float(e50.iloc[-1])
    e200_v   = float(e200.iloc[-1])
    rsi_v    = float(rsi.iloc[-1])
    atr_v    = float(atr.iloc[-1])
    adx_v    = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0

    ema_stack = e10_v > e21_v > e50_v > e200_v

    gates = [
        {"rule": "EMA Stack (10>21>50>200)", "passed": ema_stack,
         "value": f"10={e10_v:.2f} 21={e21_v:.2f} 50={e50_v:.2f} 200={e200_v:.2f}"},
        {"rule": f"Price > EMA50", "passed": price > e50_v,
         "value": f"Price={price:.2f} EMA50={e50_v:.2f}"},
        {"rule": f"RSI {p['entry_rsi_low']}–{p['entry_rsi_high']}",
         "passed": p["entry_rsi_low"] < rsi_v < p["entry_rsi_high"],
         "value": f"RSI={rsi_v:.1f}"},
        {"rule": f"ADX >= {p['adx_threshold']}", "passed": adx_v >= p["adx_threshold"],
         "value": f"ADX={adx_v:.1f}"},
    ]

    passed = sum(g["passed"] for g in gates)
    confidence = int(passed / len(gates) * 100)
    signal = "BUY" if all(g["passed"] for g in gates) else "HOLD"

    stop_loss  = price - p["atr_mult"] * atr_v if signal == "BUY" else None
    risk       = (price - stop_loss) if stop_loss else 0.0
    tp1_price  = (price + p["tp1_r"] * risk) if risk > 0 else None
    tp2_price  = (price + p["tp2_r"] * risk) if risk > 0 else None

    reasons = [("✓ " if g["passed"] else "✗ ") + g["rule"] + f" ({g['value']})" for g in gates]
    reasons.append(f"→ Signal: {signal} (confidence {confidence}%)")

    return {
        "signal":      signal,
        "confidence":  confidence,
        "entry_price": price if signal == "BUY" else None,
        "stop_loss":   stop_loss,
        "tp1_price":   tp1_price,
        "tp2_price":   tp2_price,
        "gates":       gates,
        "reasons":     reasons,
    }


# ---------------------------------------------------------------------------
# exit_fn
# ---------------------------------------------------------------------------

def exit_fn(position: dict, df: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Evaluate Wolf x Shadow exit conditions on the latest bar of *df*.

    Parameters
    ----------
    position : dict with keys:
                 entry_price     – float
                 stop_loss       – float
                 bars_below_ema50 – int (caller must track and pass this)
                 tp1_hit         – bool
                 tp2_hit         – bool
    df       : OHLCV DataFrame
    params   : override dict

    Returns
    -------
    dict with keys:
      exit        : bool
      reason      : str | None
      exit_price  : float | None
      partial_tp1 : bool  (TP1 triggered this bar)
      partial_tp2 : bool  (TP2 triggered this bar)
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    if df is None or len(df) < 2:
        return {"exit": False, "reason": None, "exit_price": None,
                "partial_tp1": False, "partial_tp2": False}

    close  = df["Close"]
    price  = float(close.iloc[-1])
    atr    = float(_atr(df, p["atr_period"]).iloc[-1])
    e10    = float(_ema(close, p["ema_pulse"]).iloc[-1])
    e50    = float(_ema(close, p["ema_slow"]).iloc[-1])
    kijun_v = float(_kijun(df, p["kijun_period"]).iloc[-1])

    entry        = float(position.get("entry_price", price))
    stop_loss    = float(position.get("stop_loss", entry - p["atr_mult"] * atr))
    bars_below   = int(position.get("bars_below_ema50", 0))
    tp1_hit      = bool(position.get("tp1_hit", False))
    tp2_hit      = bool(position.get("tp2_hit", False))

    risk = entry - stop_loss if entry > stop_loss else p["atr_mult"] * atr
    tp1_price = entry + p["tp1_r"] * risk
    tp2_price = entry + p["tp2_r"] * risk

    partial_tp1 = not tp1_hit and price >= tp1_price
    partial_tp2 = tp1_hit and not tp2_hit and price >= tp2_price

    # Hard exits
    if price <= stop_loss:
        return {"exit": True, "reason": "STOP_LOSS", "exit_price": stop_loss,
                "partial_tp1": False, "partial_tp2": False}

    if bars_below >= p["core_exit_bars"]:
        return {"exit": True, "reason": "EMA50_EXIT", "exit_price": price,
                "partial_tp1": False, "partial_tp2": False}

    if price < kijun_v and price < e10:
        return {"exit": True, "reason": "TRAIL_EXIT", "exit_price": price,
                "partial_tp1": partial_tp1, "partial_tp2": partial_tp2}

    return {"exit": False, "reason": None, "exit_price": None,
            "partial_tp1": partial_tp1, "partial_tp2": partial_tp2}


# ---------------------------------------------------------------------------
# risk_fn
# ---------------------------------------------------------------------------

def risk_fn(df: pd.DataFrame, capital: float, params: dict | None = None) -> dict:
    """
    Calculate Wolf x Shadow position size for the latest bar.

    Parameters
    ----------
    df      : OHLCV DataFrame
    capital : total available capital
    params  : override dict

    Returns
    -------
    dict with keys:
      shares           : int
      position_value   : float
      position_pct     : float   (fraction of capital)
      risk_amount      : float   (capital × risk_pct)
      stop_distance    : float   (atr_mult × ATR)
      stop_loss        : float
      entry_price      : float
      atr              : float
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    if df is None or len(df) < 20:
        return {"shares": 0, "position_value": 0.0, "position_pct": 0.0,
                "risk_amount": 0.0, "stop_distance": 0.0, "stop_loss": 0.0,
                "entry_price": 0.0, "atr": 0.0}

    price = float(df["Close"].iloc[-1])
    atr   = float(_atr(df, p["atr_period"]).iloc[-1])

    stop_distance = p["atr_mult"] * atr
    risk_amount   = capital * p["risk_pct"]
    shares        = int(risk_amount / stop_distance) if stop_distance > 0 else 0
    position_value = shares * price
    position_pct   = position_value / capital if capital > 0 else 0.0

    return {
        "shares":         shares,
        "position_value": position_value,
        "position_pct":   position_pct,
        "risk_amount":    risk_amount,
        "stop_distance":  stop_distance,
        "stop_loss":      price - stop_distance,
        "entry_price":    price,
        "atr":            atr,
    }


# ---------------------------------------------------------------------------
# Strategy descriptor
# ---------------------------------------------------------------------------

STRATEGY: dict = {
    "key":         "wolf",
    "name":        "Wolf x Shadow",
    "description": (
        "Swing strategy: 4-layer regime score (Market/Sector/Stock/Ichimoku), "
        "EMA10>21>50>200 stack entry, RSI 45–70, ADX filter, "
        "ATR stop-loss, Kijun+EMA10 trailing exit."
    ),
    "color":            "#c9a84c",
    "params":           DEFAULT_PARAMS,
    "entry_fn":         entry_fn,
    "exit_fn":          exit_fn,
    "risk_fn":          risk_fn,
    "sentiment_plugins":  ["ovtlyr_fg", "retail_flow"],
    "alerts_enabled":     True,
    "alert_channels":     ["discord"],
}
