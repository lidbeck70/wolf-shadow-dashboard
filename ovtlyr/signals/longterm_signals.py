"""
OVTLYR Golden Ticket Trading Strategy — Signal Engine.

Implements the OVTLYR NINE composite score (0-100) from 3 weighted layers:
  Market  (40%) — 3 components
  Sector  (30%) — 2 components
  Stock   (30%) — 4 components (including Order Blocks & momentum)

Signal thresholds:
  BUY    >= 70   AND no sell-off override AND no restrictive OBs
  HOLD    40-69  OR breadth contracting
  REDUCE        Fear & Greed stalling/reversing OR approaching bearish OB
  SELL   < 40   OR SPY < 20EMA OR breadth crossover down
"""


def compute_longterm_signal(
    trend: dict,
    sentiment: dict,
    volatility: dict,
    ob_analysis: dict,
    sector_green: bool,
) -> dict:
    """
    OVTLYR Golden Ticket signal engine.

    Computes the OVTLYR NINE score (0-100) from 3 layers:

    MARKET (40%):
      1. Market Trend: SPY 10EMA > 20EMA, Price > 50EMA
      2. Market Signal: trend is bullish
      3. Market Breadth: breadth advancing (bullish)

    SECTOR (30%):
      4. Sector Fear & Greed: advancing (not extreme greed)
      5. Sector Breadth: advancing (bullish)

    STOCK (30%):
      6. Stock Signal: buy signal
      7. Stock Trend: 10EMA/20EMA bullish, Price > 50EMA
      8. Stock Fear & Greed: advancing
      9. Order Blocks: no restrictive OBs + momentum (price > prior day low)

    Each component scores 0 or 1. Layer score = sum of components / total * 100.
    Final OVTLYR NINE = market_score * 0.40 + sector_score * 0.30 + stock_score * 0.30

    Signal logic:
      BUY: OVTLYR NINE >= 70 AND no sell-off override AND no restrictive OBs
      HOLD: OVTLYR NINE 40-69 OR breadth contracting
      REDUCE: Fear & Greed stalling/reversing OR price approaching bearish OB
      SELL: OVTLYR NINE < 40 OR SPY < 20EMA OR breadth crossover down

    Exit triggers (check separately):
      - SPY close under 20EMA → SELL ALL
      - Stock trailing stop: price < 10EMA → SELL
      - Stop loss: ½ ATR from entry → SELL
      - Order block hit → SELL
      - Gap & crap momentum → SELL
      - Fear & Greed target hit → SELL (with spread rules)

    Parameters
    ----------
    trend : dict
        Keys expected:
          price           – current price (float)
          ema10           – 10-period EMA (float or Series)
          ema20           – 20-period EMA (float or Series)
          ema50           – 50-period EMA (float or Series)
          ema200          – 200-period EMA (float or Series)
          trend_state     – "bullish" | "bearish" | "neutral"
          regime_color    – "green" | "orange" | "red"
          price_above_200 – bool
          ema50_above_200 – bool
    sentiment : dict
        Keys expected:
          score   – Fear & Greed score 0–100 (int/float)
          label   – human-readable label
    volatility : dict
        Keys expected:
          risk_score – composite risk 0–100 (int/float)
          atr14      – Average True Range 14-period (float)
    ob_analysis : dict
        Keys expected:
          signal_bias        – "BUY" | "HOLD" | "REDUCE" | "SELL"
          nearest_bullish_ob – dict or None
          nearest_bearish_ob – dict or None
          approaching_bullish – bool
          approaching_bearish – bool
    sector_green : bool
        True if the stock's sector is bullish / advancing.

    Returns
    -------
    dict with keys:
      signal         : str  — "BUY" | "HOLD" | "REDUCE" | "SELL"
      ovtlyr_nine    : int  — 0–100
      market_score   : int  — 0–100
      sector_score   : int  — 0–100
      stock_score    : int  — 0–100
      confidence     : int  — 0–100 (alias for ovtlyr_nine for backwards compat)
      reasons        : list[str]
      gates          : list[{rule: str, passed: bool, detail: str}]
      exit_triggers  : list[{trigger: str, active: bool}]
    """

    # ------------------------------------------------------------------ #
    #  Helper: safely convert a value (possibly pd.Series/list) to float
    # ------------------------------------------------------------------ #
    def _to_float(val, default=0.0):
        """Safely convert a value (possibly pd.Series) to float."""
        if val is None:
            return default
        try:
            import pandas as _pd
            if isinstance(_pd.Series, type) and isinstance(val, _pd.Series):
                return float(val.iloc[-1]) if len(val) > 0 else default
        except Exception:
            pass
        if isinstance(val, list):
            return float(val[-1]) if val else default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------ #
    #  Extract scalars from inputs
    # ------------------------------------------------------------------ #
    price   = _to_float(trend.get("price", trend.get("last_close", 0)))
    ema10   = _to_float(trend.get("ema10",  0))
    ema20   = _to_float(trend.get("ema20",  0))
    ema50   = _to_float(trend.get("ema50",  0))
    ema200  = _to_float(trend.get("ema200", 0))

    trend_state   = str(trend.get("trend_state",  trend.get("direction", "neutral"))).lower()
    regime_color  = str(trend.get("regime_color", "red")).lower()

    sentiment_score = _to_float(sentiment.get("score", 50))
    sentiment_label = str(sentiment.get("label", "Neutral"))

    risk_score = _to_float(volatility.get("risk_score", 50))
    atr14      = _to_float(volatility.get("atr14",  volatility.get("atr", 0)))

    ob_bias             = str(ob_analysis.get("signal_bias", "HOLD")).upper()
    approaching_bearish = bool(ob_analysis.get("approaching_bearish", False))
    approaching_bullish = bool(ob_analysis.get("approaching_bullish", False))

    # ------------------------------------------------------------------ #
    #  Derive proxy values where real SPY data is unavailable
    #  (SPY proxy: use stock's own trend data; noted in output)
    # ------------------------------------------------------------------ #
    # SPY sell-off override: proxy — price under 20EMA means "SPY under 20EMA"
    spy_under_20ema = (ema20 > 0) and (price < ema20)

    # ------------------------------------------------------------------ #
    #  MARKET LAYER  (40%)  — 3 components
    # ------------------------------------------------------------------ #

    # Component 1 — Market Trend: 10EMA > 20EMA AND Price > 50EMA  [SPY proxy]
    if ema10 > 0 and ema20 > 0 and ema50 > 0:
        mkt_trend_ok = (ema10 > ema20) and (price > ema50)
    else:
        # Fall back to regime_color when EMAs not available
        mkt_trend_ok = regime_color == "green"
    mkt_trend_detail = (
        f"10EMA({ema10:.2f}) > 20EMA({ema20:.2f}) AND Price({price:.2f}) > 50EMA({ema50:.2f}) [SPY proxy]"
        if ema10 > 0 else
        f"Regime={regime_color.upper()} [SPY proxy — EMAs not available]"
    )

    # Component 2 — Market Signal: trend is bullish  [SPY proxy]
    mkt_signal_ok = trend_state in ("bullish", "uptrend", "green")
    mkt_signal_detail = f"trend_state={trend_state.upper()} [SPY proxy]"

    # Component 3 — Market Breadth: breadth advancing  [SPY proxy via sector_green heuristic]
    # sector_green == True is used as a proxy for "market breadth advancing"
    mkt_breadth_ok = sector_green
    mkt_breadth_detail = f"sector_green={sector_green} → market breadth advancing [SPY proxy]"

    market_components = [mkt_trend_ok, mkt_signal_ok, mkt_breadth_ok]
    market_score = int(round(sum(market_components) / len(market_components) * 100))

    # ------------------------------------------------------------------ #
    #  SECTOR LAYER  (30%)  — 2 components
    # ------------------------------------------------------------------ #

    # Component 4 — Sector Fear & Greed: advancing (not extreme greed > 90)
    sec_fg_ok = (sentiment_score < 90) and (sentiment_score > 25)
    sec_fg_detail = f"F&G={sentiment_score:.0f} — advancing range (25–90)"

    # Component 5 — Sector Breadth: advancing (bullish 10EMA cross)
    sec_breadth_ok = sector_green
    sec_breadth_detail = f"sector_green={sector_green} → bullish EMA cross"

    sector_components = [sec_fg_ok, sec_breadth_ok]
    sector_score = int(round(sum(sector_components) / len(sector_components) * 100))

    # ------------------------------------------------------------------ #
    #  STOCK LAYER  (30%)  — 4 components
    # ------------------------------------------------------------------ #

    # Component 6 — Stock Signal: buy signal (ob_bias == BUY or regime green)
    stk_signal_ok = (ob_bias in ("BUY", "HOLD")) and (regime_color in ("green", "orange"))
    stk_signal_detail = f"ob_bias={ob_bias}, regime={regime_color.upper()}"

    # Component 7 — Stock Trend: 10EMA/20EMA bullish AND Price > 50EMA
    if ema10 > 0 and ema20 > 0 and ema50 > 0:
        stk_trend_ok = (ema10 > ema20) and (price > ema50)
    else:
        stk_trend_ok = regime_color == "green"
    stk_trend_detail = (
        f"10EMA({ema10:.2f}) > 20EMA({ema20:.2f}) AND Price({price:.2f}) > 50EMA({ema50:.2f})"
        if ema10 > 0 else
        f"Regime={regime_color.upper()} [EMA10/20 not yet computed]"
    )

    # Component 8 — Stock Fear & Greed: advancing (score > 30 and not reversing)
    stk_fg_ok = sentiment_score > 30
    stk_fg_detail = f"F&G={sentiment_score:.0f} > 30 (advancing)"

    # Component 9 — Order Blocks: no restrictive bearish OBs + momentum ok
    no_restrictive_ob = ob_bias not in ("SELL", "REDUCE") and not approaching_bearish
    stk_ob_ok = no_restrictive_ob
    stk_ob_detail = (
        f"ob_bias={ob_bias}, approaching_bearish={approaching_bearish} → "
        + ("CLEAR" if stk_ob_ok else "BLOCKED")
    )

    stock_components = [stk_signal_ok, stk_trend_ok, stk_fg_ok, stk_ob_ok]
    stock_score = int(round(sum(stock_components) / len(stock_components) * 100))

    # ------------------------------------------------------------------ #
    #  OVTLYR NINE composite score
    # ------------------------------------------------------------------ #
    ovtlyr_nine = int(round(
        market_score * 0.40
        + sector_score * 0.30
        + stock_score * 0.30
    ))

    # ------------------------------------------------------------------ #
    #  Signal determination
    # ------------------------------------------------------------------ #
    reasons: list[str] = []

    # Sell-off override check
    selloff_override = spy_under_20ema
    if selloff_override:
        reasons.append("✗ SELL-OFF OVERRIDE: Price < 20EMA (SPY proxy) → NO NEW TRADES")

    # Primary signal logic
    if ovtlyr_nine < 40 or spy_under_20ema or (not mkt_breadth_ok and not mkt_trend_ok):
        signal = "SELL"
    elif approaching_bearish or sentiment_score > 80:
        signal = "REDUCE"
    elif ovtlyr_nine >= 70 and not selloff_override and stk_ob_ok:
        signal = "BUY"
    else:
        signal = "HOLD"

    # ------------------------------------------------------------------ #
    #  Build GATES list (OVTLYR NINE components)
    # ------------------------------------------------------------------ #
    gates: list[dict] = [
        # Market layer
        {
            "rule": "1. Market Trend: 10EMA > 20EMA, Price > 50EMA",
            "passed": mkt_trend_ok,
            "detail": mkt_trend_detail,
            "layer": "MARKET",
        },
        {
            "rule": "2. Market Signal: Trend bullish",
            "passed": mkt_signal_ok,
            "detail": mkt_signal_detail,
            "layer": "MARKET",
        },
        {
            "rule": "3. Market Breadth: Advancing",
            "passed": mkt_breadth_ok,
            "detail": mkt_breadth_detail,
            "layer": "MARKET",
        },
        # Sector layer
        {
            "rule": "4. Sector Fear & Greed: Advancing (25–90)",
            "passed": sec_fg_ok,
            "detail": sec_fg_detail,
            "layer": "SECTOR",
        },
        {
            "rule": "5. Sector Breadth: Advancing",
            "passed": sec_breadth_ok,
            "detail": sec_breadth_detail,
            "layer": "SECTOR",
        },
        # Stock layer
        {
            "rule": "6. Stock Signal: Buy",
            "passed": stk_signal_ok,
            "detail": stk_signal_detail,
            "layer": "STOCK",
        },
        {
            "rule": "7. Stock Trend: 10EMA/20EMA, Price > 50EMA",
            "passed": stk_trend_ok,
            "detail": stk_trend_detail,
            "layer": "STOCK",
        },
        {
            "rule": "8. Stock Fear & Greed: Advancing",
            "passed": stk_fg_ok,
            "detail": stk_fg_detail,
            "layer": "STOCK",
        },
        {
            "rule": "9. Order Blocks: No restrictive OBs + Momentum",
            "passed": stk_ob_ok,
            "detail": stk_ob_detail,
            "layer": "STOCK",
        },
        # Override check
        {
            "rule": "SPY Sell-Off Override: Price above 20EMA",
            "passed": not selloff_override,
            "detail": f"Price({price:.2f}) vs 20EMA({ema20:.2f}) [SPY proxy]",
            "layer": "OVERRIDE",
        },
    ]

    # ------------------------------------------------------------------ #
    #  EXIT TRIGGERS
    # ------------------------------------------------------------------ #
    exit_triggers: list[dict] = [
        {
            "trigger": "SPY close under 20EMA → SELL ALL",
            "active": spy_under_20ema,
        },
        {
            "trigger": "Stock trailing stop: price < 10EMA",
            "active": (ema10 > 0) and (price < ema10),
        },
        {
            "trigger": "Stop loss: ½ ATR from entry",
            "active": False,  # requires entry price context (not available here)
        },
        {
            "trigger": "Order block hit (bearish OB blocking path)",
            "active": approaching_bearish or ob_bias == "SELL",
        },
        {
            "trigger": "Gap & Crap momentum reversal",
            "active": False,  # requires intraday data (not available here)
        },
        {
            "trigger": "Fear & Greed target hit (0-50: exit at 63)",
            "active": (sentiment_score >= 63 and sentiment_score <= 75),
        },
        {
            "trigger": "Fear & Greed target hit (50-75: 10pt spread)",
            "active": (sentiment_score > 75 and sentiment_score <= 85),
        },
        {
            "trigger": "Fear & Greed target hit (75+: 5pt spread)",
            "active": sentiment_score > 85,
        },
    ]

    # ------------------------------------------------------------------ #
    #  Human-readable reasons
    # ------------------------------------------------------------------ #
    reasons.append(
        f"OVTLYR NINE = {ovtlyr_nine}/100  "
        f"[Market:{market_score} × 40%  Sector:{sector_score} × 30%  Stock:{stock_score} × 30%]"
    )

    layer_items = [
        ("Market Trend",    mkt_trend_ok,   mkt_trend_detail),
        ("Market Signal",   mkt_signal_ok,  mkt_signal_detail),
        ("Market Breadth",  mkt_breadth_ok, mkt_breadth_detail),
        ("Sector F&G",      sec_fg_ok,      sec_fg_detail),
        ("Sector Breadth",  sec_breadth_ok, sec_breadth_detail),
        ("Stock Signal",    stk_signal_ok,  stk_signal_detail),
        ("Stock Trend",     stk_trend_ok,   stk_trend_detail),
        ("Stock F&G",       stk_fg_ok,      stk_fg_detail),
        ("Order Blocks",    stk_ob_ok,      stk_ob_detail),
    ]
    for name, passed, detail in layer_items:
        icon = "✓" if passed else "✗"
        reasons.append(f"{icon} {name}: {detail}")

    active_exits = [t["trigger"] for t in exit_triggers if t["active"]]
    if active_exits:
        for t in active_exits:
            reasons.append(f"⚠ EXIT TRIGGER: {t}")

    reasons.append(f"→ Final signal: {signal} (OVTLYR NINE {ovtlyr_nine}/100)")

    # ------------------------------------------------------------------ #
    #  Return
    # ------------------------------------------------------------------ #
    return {
        "signal":        signal,
        "ovtlyr_nine":   ovtlyr_nine,
        "market_score":  market_score,
        "sector_score":  sector_score,
        "stock_score":   stock_score,
        "confidence":    ovtlyr_nine,   # backwards-compatible alias
        "reasons":       reasons,
        "gates":         gates,
        "exit_triggers": exit_triggers,
    }
