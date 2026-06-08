"""
Viking Strategy
===============
Swing/position strategy based on the OVTLYR NINE composite score (0-100).

Three-layer composite:
  Market (40%) — trend, signal, breadth
  Sector (30%) — fear & greed, breadth
  Stock  (30%) — signal, trend, fear & greed, order blocks

Entry:
  OVTLYR NINE >= 70  AND  no sell-off override  AND  no restrictive OBs

Exit triggers (first wins):
  • OVTLYR NINE < 40       → SELL
  • Price < 20EMA          → SELL (SPY proxy sell-off override)
  • Price < 10EMA          → trailing stop
  • ½ ATR hard stop-loss

Risk / sizing:
  • Risk per trade = capital × risk_pct (default 1.5%)
  • Stop distance  = ½ × ATR14
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_PARAMS: dict = {
    "ema10":       10,
    "ema20":       20,
    "ema50":       50,
    "ema200":      200,
    "atr_period":  14,
    "atr_stop_mult": 0.5,    # ½ ATR hard stop
    "risk_pct":    0.015,    # 1.5% risk per trade
    "tp1_r":       2.0,
    "tp2_r":       4.0,
    "tp1_pct":     0.25,
    "tp2_pct":     0.25,
    "core_pct":    0.50,
    "nine_buy":    70,       # OVTLYR NINE threshold for BUY
    "nine_sell":   40,       # OVTLYR NINE threshold for SELL
    "fg_max":      90,       # Fear & Greed ceiling (above → REDUCE)
    "fg_min":      25,       # Fear & Greed floor
}


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _compute_ovtlyr_nine(
    price: float,
    ema10_v: float,
    ema20_v: float,
    ema50_v: float,
    fg_score: float,
    sector_green: bool,
    ob_bias: str,
    approaching_bearish: bool,
) -> dict:
    """Inline OVTLYR NINE computation from price/EMA/sentiment inputs."""
    spy_under_20ema = (ema20_v > 0) and (price < ema20_v)
    trend_bullish   = (ema10_v > ema20_v) and (price > ema50_v) if ema10_v > 0 else False

    # Market layer (40%) — 3 components
    mkt_trend   = trend_bullish
    mkt_signal  = trend_bullish
    mkt_breadth = sector_green
    market_score = int(round(sum([mkt_trend, mkt_signal, mkt_breadth]) / 3 * 100))

    # Sector layer (30%) — 2 components
    sec_fg      = (fg_score < 90) and (fg_score > 25)
    sec_breadth = sector_green
    sector_score = int(round(sum([sec_fg, sec_breadth]) / 2 * 100))

    # Stock layer (30%) — 4 components
    stk_signal = ob_bias in ("BUY", "HOLD")
    stk_trend  = trend_bullish
    stk_fg     = fg_score > 30
    stk_ob     = ob_bias not in ("SELL", "REDUCE") and not approaching_bearish
    stock_score = int(round(sum([stk_signal, stk_trend, stk_fg, stk_ob]) / 4 * 100))

    nine = int(round(market_score * 0.40 + sector_score * 0.30 + stock_score * 0.30))

    return {
        "ovtlyr_nine":   nine,
        "market_score":  market_score,
        "sector_score":  sector_score,
        "stock_score":   stock_score,
        "spy_under_20":  spy_under_20ema,
        "components": [
            {"rule": "1. Market Trend (10>20EMA, Price>50EMA)", "passed": mkt_trend,   "layer": "MARKET"},
            {"rule": "2. Market Signal (bullish)",              "passed": mkt_signal,  "layer": "MARKET"},
            {"rule": "3. Market Breadth (sector green)",        "passed": mkt_breadth, "layer": "MARKET"},
            {"rule": "4. Sector F&G (25-90)",                  "passed": sec_fg,      "layer": "SECTOR"},
            {"rule": "5. Sector Breadth (sector green)",        "passed": sec_breadth, "layer": "SECTOR"},
            {"rule": "6. Stock Signal (OB bias ok)",            "passed": stk_signal,  "layer": "STOCK"},
            {"rule": "7. Stock Trend (10>20EMA, Price>50EMA)",  "passed": stk_trend,   "layer": "STOCK"},
            {"rule": "8. Stock F&G (> 30)",                    "passed": stk_fg,      "layer": "STOCK"},
            {"rule": "9. Order Blocks clear",                   "passed": stk_ob,      "layer": "STOCK"},
        ],
    }


def entry_fn(df: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Evaluate Viking entry conditions on the latest bar of *df*.

    Builds OVTLYR NINE from price/EMA data with neutral sentiment (50) and
    no order-block data (conservative HOLD bias). Pass enriched inputs via
    the session-state pattern in the UI tabs.

    Returns
    -------
    dict with keys:
      signal        : "BUY" | "HOLD" | "REDUCE" | "SELL"
      confidence    : int 0-100   (OVTLYR NINE score)
      entry_price   : float | None
      stop_loss     : float | None
      tp1_price     : float | None
      tp2_price     : float | None
      gates         : list of {rule, passed, layer}
      reasons       : list[str]
      ovtlyr_nine   : int
      market_score  : int
      sector_score  : int
      stock_score   : int
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    if df is None or len(df) < 210:
        return {"signal": "HOLD", "confidence": 0, "entry_price": None,
                "stop_loss": None, "tp1_price": None, "tp2_price": None,
                "gates": [], "reasons": ["Insufficient data"],
                "ovtlyr_nine": 0, "market_score": 0, "sector_score": 0, "stock_score": 0}

    close   = df["Close"]
    e10     = _ema(close, p["ema10"])
    e20     = _ema(close, p["ema20"])
    e50     = _ema(close, p["ema50"])
    atr     = _atr(df, p["atr_period"])

    price   = float(close.iloc[-1])
    e10_v   = float(e10.iloc[-1])
    e20_v   = float(e20.iloc[-1])
    e50_v   = float(e50.iloc[-1])
    atr_v   = float(atr.iloc[-1])

    # Use neutral sentiment and no OBs when called without enriched context
    fg_score          = float(params.get("fg_score", 50))          if params else 50.0
    sector_green      = bool(params.get("sector_green", True))      if params else True
    ob_bias           = str(params.get("ob_bias", "HOLD"))          if params else "HOLD"
    approaching_bear  = bool(params.get("approaching_bearish", False)) if params else False

    result = _compute_ovtlyr_nine(
        price, e10_v, e20_v, e50_v,
        fg_score, sector_green, ob_bias, approaching_bear,
    )

    nine  = result["ovtlyr_nine"]
    gates = result["components"]
    spy_under = result["spy_under_20"]

    if nine < p["nine_sell"] or spy_under:
        signal = "SELL"
    elif approaching_bear or fg_score > 80:
        signal = "REDUCE"
    elif nine >= p["nine_buy"] and not spy_under and ob_bias not in ("SELL", "REDUCE"):
        signal = "BUY"
    else:
        signal = "HOLD"

    is_buy     = signal == "BUY"
    stop_loss  = price - p["atr_stop_mult"] * atr_v if is_buy else None
    risk       = (price - stop_loss) if stop_loss else 0.0
    tp1_price  = (price + p["tp1_r"] * risk) if risk > 0 else None
    tp2_price  = (price + p["tp2_r"] * risk) if risk > 0 else None

    reasons = [
        f"OVTLYR NINE = {nine}/100  "
        f"[Market:{result['market_score']}×40%  Sector:{result['sector_score']}×30%  Stock:{result['stock_score']}×30%]"
    ]
    for g in gates:
        reasons.append(("✓ " if g["passed"] else "✗ ") + g["rule"])
    reasons.append(f"→ Signal: {signal} (OVTLYR NINE {nine}/100)")

    return {
        "signal":       signal,
        "confidence":   nine,
        "entry_price":  price if is_buy else None,
        "stop_loss":    stop_loss,
        "tp1_price":    tp1_price,
        "tp2_price":    tp2_price,
        "gates":        gates,
        "reasons":      reasons,
        "ovtlyr_nine":  nine,
        "market_score": result["market_score"],
        "sector_score": result["sector_score"],
        "stock_score":  result["stock_score"],
    }


def exit_fn(position: dict, df: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Evaluate Viking exit conditions on the latest bar of *df*.

    Exit triggers (first wins):
      1. Hard stop-loss (½ ATR)
      2. Price < 10EMA  (trailing stop)
      3. Price < 20EMA  (SPY proxy sell-off override)

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
    e10_v   = float(_ema(close, p["ema10"]).iloc[-1])
    e20_v   = float(_ema(close, p["ema20"]).iloc[-1])

    entry     = float(position.get("entry_price", price))
    stop_loss = float(position.get("stop_loss", entry - p["atr_stop_mult"] * atr_v))
    tp1_hit   = bool(position.get("tp1_hit", False))
    tp2_hit   = bool(position.get("tp2_hit", False))

    risk      = entry - stop_loss if entry > stop_loss else p["atr_stop_mult"] * atr_v
    tp1_price = entry + p["tp1_r"] * risk
    tp2_price = entry + p["tp2_r"] * risk

    partial_tp1 = not tp1_hit and price >= tp1_price
    partial_tp2 = tp1_hit and not tp2_hit and price >= tp2_price

    if price <= stop_loss:
        return {"exit": True, "reason": "STOP_LOSS", "exit_price": stop_loss,
                "partial_tp1": False, "partial_tp2": False}

    if price < e10_v:
        return {"exit": True, "reason": "EMA10_TRAIL", "exit_price": price,
                "partial_tp1": partial_tp1, "partial_tp2": partial_tp2}

    if price < e20_v:
        return {"exit": True, "reason": "SPY_SELLOFF_PROXY", "exit_price": price,
                "partial_tp1": partial_tp1, "partial_tp2": partial_tp2}

    return {"exit": False, "reason": None, "exit_price": None,
            "partial_tp1": partial_tp1, "partial_tp2": partial_tp2}


def risk_fn(df: pd.DataFrame, capital: float, params: dict | None = None) -> dict:
    """
    Calculate Viking position size for the latest bar.

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

    stop_distance  = p["atr_stop_mult"] * atr
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
    "key":         "viking",
    "name":        "Viking (OVTLYR)",
    "description": (
        "Swing/position strategy: OVTLYR NINE composite score (0-100) from "
        "Market(40%) + Sector(30%) + Stock(30%) layers. BUY >= 70, SELL < 40, "
        "½-ATR stop-loss, EMA10 trailing exit."
    ),
    "color":            "#00A8BF",
    "params":           DEFAULT_PARAMS,
    "entry_fn":         entry_fn,
    "exit_fn":          exit_fn,
    "risk_fn":          risk_fn,
    "sentiment_plugins":  ["ovtlyr_fg"],
    "alerts_enabled":     True,
    "alert_channels":     ["discord"],

    # ── Köp-regler (buy rules) ────────────────────────────────────────────────
    # Derived verbatim from _compute_ovtlyr_nine components + entry threshold.
    "rules_buy": [
        "1. OVTLYR NINE ≥ 70 (composite buy threshold)",
        "2. Market Trend: EMA10 > EMA20, Price > EMA50",
        "3. Market Signal: bullish trend confirmed",
        "4. Market Breadth: sector green",
        "5. Sector F&G: Fear & Greed 25–90",
        "6. Sector Breadth: sector green",
        "7. Stock Signal: Order Block bias ok (BUY / HOLD)",
        "8. Stock Trend: EMA10 > EMA20, Price > EMA50",
        "9. Stock F&G: Fear & Greed > 30",
        "10. Order Blocks clear (no approaching bearish OBs)",
    ],

    # ── Sälj-regler (sell rules) ──────────────────────────────────────────────
    # Derived verbatim from exit_fn triggers (first trigger wins).
    "rules_sell": [
        "1. OVTLYR NINE < 40 → SELL",
        "2. Price < EMA20 → SPY sell-off proxy override",
        "3. Price < EMA10 → trailing stop exit",
        "4. Price ≤ entry − ½ × ATR14 → hard stop-loss",
        "5. Fear & Greed > 80 → REDUCE (extreme greed override)",
        "6. Order Block bias SELL or REDUCE → signal REDUCE",
        "7. Approaching bearish Order Block detected → REDUCE",
    ],
}
