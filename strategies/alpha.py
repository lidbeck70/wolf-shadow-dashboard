"""
Alpha Strategy
==============
Long-term position strategy based on the CAGR scoring model.

Entry (all required):
  1. Green regime   — cycle_score >= 2
  2. Price > EMA200 — stock in long-term uptrend
  3. EMA50 > EMA200 — golden cross confirmed
  4. Score >= BUY threshold (55% of max_score)
  5. All hard gates pass

Signal levels:
  STRONG BUY — score_pct >= 70% AND all gates pass
  BUY        — score_pct >= 55% AND all gates pass

Exit triggers (rules 6-7):
  SELL       — price closes below EMA200
  SELL       — regime turns red (cycle_score == 0)

Risk / sizing:
  • Position sized to risk_pct of capital per trade
  • Stop distance = EMA200 breach (price – EMA200)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_PARAMS: dict = {
    "ema_slow":        50,
    "ema_macro":       200,
    "rsi_period":      14,
    "atr_period":      14,
    "atr_mult":        2.0,
    "cycle_min":       2,       # minimum cycle_score to enter
    "strong_buy_pct":  0.70,
    "buy_pct":         0.55,
    "hold_pct":        0.35,
    "risk_pct":        0.015,   # 1.5% risk per trade
    "tp1_r":           3.0,
    "tp2_r":           6.0,
    "tp1_pct":         0.30,
    "tp2_pct":         0.30,
    "core_pct":        0.40,
}


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def entry_fn(df: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Evaluate Alpha long-term entry conditions on the latest bar of *df*.

    This is a simplified standalone evaluator using technical gates only.
    For full CAGR scoring with fundamentals, use cagr/cagr_scoring.py directly.

    Returns
    -------
    dict with keys:
      signal        : "STRONG BUY" | "BUY" | "HOLD" | "SELL"
      confidence    : int 0-100
      entry_price   : float | None
      stop_loss     : float | None
      tp1_price     : float | None
      tp2_price     : float | None
      gates         : list of {rule, passed, value}
      reasons       : list[str]
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    if df is None or len(df) < 210:
        return {"signal": "HOLD", "confidence": 0, "entry_price": None,
                "stop_loss": None, "tp1_price": None, "tp2_price": None,
                "gates": [], "reasons": ["Insufficient data"]}

    close  = df["Close"]
    e50    = _ema(close, p["ema_slow"])
    e200   = _ema(close, p["ema_macro"])
    atr    = _atr(df, p["atr_period"])
    rsi    = _rsi(close, p["rsi_period"])

    price   = float(close.iloc[-1])
    e50_v   = float(e50.iloc[-1])
    e200_v  = float(e200.iloc[-1])
    atr_v   = float(atr.iloc[-1])
    rsi_v   = float(rsi.iloc[-1])

    # EMA200 slope: compare current vs 10 bars ago
    e200_slope_ok = float(e200.iloc[-1]) > float(e200.iloc[-11]) if len(e200) > 11 else False
    # RS proxy: RSI above 50 (rising relative strength)
    rs_ok = rsi_v > 50

    gates = [
        {"rule": "Price > EMA200",
         "passed": price > e200_v,
         "value": f"Price={price:.2f} EMA200={e200_v:.2f}"},
        {"rule": "EMA50 > EMA200 (golden cross)",
         "passed": e50_v > e200_v,
         "value": f"EMA50={e50_v:.2f} EMA200={e200_v:.2f}"},
        {"rule": "EMA200 slope rising",
         "passed": e200_slope_ok,
         "value": f"EMA200 now={e200_v:.2f} vs 10-bar-ago={e200.iloc[-11]:.2f}" if len(e200) > 11 else "N/A"},
        {"rule": "RSI > 50 (RS positive)",
         "passed": rs_ok,
         "value": f"RSI={rsi_v:.1f}"},
    ]

    passed     = sum(g["passed"] for g in gates)
    score_pct  = passed / len(gates)
    confidence = int(score_pct * 100)

    all_pass = all(g["passed"] for g in gates)

    if score_pct >= p["strong_buy_pct"] and all_pass:
        signal = "STRONG BUY"
    elif score_pct >= p["buy_pct"] and all_pass:
        signal = "BUY"
    elif score_pct >= p["hold_pct"]:
        signal = "HOLD"
    else:
        signal = "SELL"

    is_buy = signal in ("STRONG BUY", "BUY")
    stop_loss = price - p["atr_mult"] * atr_v if is_buy else None
    risk      = (price - stop_loss) if stop_loss else 0.0
    tp1_price = (price + p["tp1_r"] * risk) if risk > 0 else None
    tp2_price = (price + p["tp2_r"] * risk) if risk > 0 else None

    reasons = [("✓ " if g["passed"] else "✗ ") + g["rule"] + f" ({g['value']})" for g in gates]
    reasons.append(f"→ Signal: {signal} (confidence {confidence}%)")

    return {
        "signal":      signal,
        "confidence":  confidence,
        "entry_price": price if is_buy else None,
        "stop_loss":   stop_loss,
        "tp1_price":   tp1_price,
        "tp2_price":   tp2_price,
        "gates":       gates,
        "reasons":     reasons,
    }


def exit_fn(position: dict, df: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Evaluate Alpha exit conditions on the latest bar of *df*.

    Exit rules (CAGR rules 6-7):
      • Price closes below EMA200 → SELL
      • Hard stop-loss breach → SELL

    Returns
    -------
    dict with keys:
      exit        : bool
      reason      : str | None
      exit_price  : float | None
      partial_tp1 : bool
      partial_tp2 : bool
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    if df is None or len(df) < 2:
        return {"exit": False, "reason": None, "exit_price": None,
                "partial_tp1": False, "partial_tp2": False}

    close   = df["Close"]
    price   = float(close.iloc[-1])
    atr_v   = float(_atr(df, p["atr_period"]).iloc[-1])
    e200_v  = float(_ema(close, p["ema_macro"]).iloc[-1])

    entry     = float(position.get("entry_price", price))
    stop_loss = float(position.get("stop_loss", entry - p["atr_mult"] * atr_v))
    tp1_hit   = bool(position.get("tp1_hit", False))
    tp2_hit   = bool(position.get("tp2_hit", False))

    risk      = entry - stop_loss if entry > stop_loss else p["atr_mult"] * atr_v
    tp1_price = entry + p["tp1_r"] * risk
    tp2_price = entry + p["tp2_r"] * risk

    partial_tp1 = not tp1_hit and price >= tp1_price
    partial_tp2 = tp1_hit and not tp2_hit and price >= tp2_price

    if price <= stop_loss:
        return {"exit": True, "reason": "STOP_LOSS", "exit_price": stop_loss,
                "partial_tp1": False, "partial_tp2": False}

    if price < e200_v:
        return {"exit": True, "reason": "EMA200_BREACH", "exit_price": price,
                "partial_tp1": False, "partial_tp2": False}

    return {"exit": False, "reason": None, "exit_price": None,
            "partial_tp1": partial_tp1, "partial_tp2": partial_tp2}


def risk_fn(df: pd.DataFrame, capital: float, params: dict | None = None) -> dict:
    """
    Calculate Alpha position size for the latest bar.

    Returns
    -------
    dict with keys:
      shares, position_value, position_pct, risk_amount,
      stop_distance, stop_loss, entry_price, atr
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    if df is None or len(df) < 20:
        return {"shares": 0, "position_value": 0.0, "position_pct": 0.0,
                "risk_amount": 0.0, "stop_distance": 0.0, "stop_loss": 0.0,
                "entry_price": 0.0, "atr": 0.0}

    price = float(df["Close"].iloc[-1])
    atr   = float(_atr(df, p["atr_period"]).iloc[-1])

    stop_distance  = p["atr_mult"] * atr
    risk_amount    = capital * p["risk_pct"]
    shares         = int(risk_amount / stop_distance) if stop_distance > 0 else 0
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


STRATEGY: dict = {
    "key":         "alpha",
    "name":        "Alpha Long-Term",
    "description": (
        "Long-term position strategy: CAGR-based scoring (fundamentals + cycle + "
        "technical), Price>EMA200, EMA50>EMA200 golden cross, green regime gate, "
        "ATR stop-loss, EMA200 breach exit."
    ),
    "color":            "#2d8a4e",
    "params":           DEFAULT_PARAMS,
    "entry_fn":         entry_fn,
    "exit_fn":          exit_fn,
    "risk_fn":          risk_fn,
    "sentiment_plugins":  ["retail_flow", "options_flow"],
    "alerts_enabled":     True,
    "alert_channels":     ["discord"],
}
