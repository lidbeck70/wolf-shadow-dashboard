"""
Swing signal computation based on the user's 10 swing rules.

LONG  — trend bullish + OB support + volume confirmation
SHORT — trend bearish + OB resistance + volume confirmation
FLAT  — mixed signals or consolidation
"""


def compute_swing_signal(
    trend: dict,
    momentum: dict,
    volume: dict,
    ob_analysis: dict,
) -> dict:
    """
    Compute swing signal based on the user's 10 swing rules.

    Parameters
    ----------
    trend : dict
        Keys expected:
          direction       – "bullish" | "bearish" | "neutral"
          in_consolidation – bool  (True = range/squeeze, no trades)
          price           – current price (float)
          ema50           – 50-period EMA (float)
          ema200          – 200-period EMA (float)
          regime_color    – "green" | "orange" | "red"
          pullback_to_ema – bool (price near EMA, potential entry zone)
    momentum : dict
        Keys expected:
          rsi             – RSI 14 value (float)
          roc             – Rate of Change % (float)
          z_score         – price Z-score (float)
          ob_os_flag      – "overbought" | "oversold" | "neutral"
    volume : dict
        Keys expected:
          confirms        – bool (current volume > average)
          ratio           – float (current vol / avg vol, e.g. 1.4 = 40% above avg)
          trend           – "rising" | "falling" | "flat"
    ob_analysis : dict
        Keys expected:
          signal_bias         – "BUY" | "HOLD" | "REDUCE" | "SELL"
          nearest_bullish_ob  – dict or None  {high, low, status, volume}
          nearest_bearish_ob  – dict or None  {high, low, status, volume}
          price_in_bullish_ob – bool
          price_in_bearish_ob – bool

    Returns
    -------
    dict with keys:
      signal     : str        — "LONG" | "SHORT" | "FLAT"
      entry_type : str | None — "Pullback to EMA" | "OB Bounce" | "Breakout" | None
      confidence : int        — 0–100
      reasons    : list[str]  — plain-language rule evaluations
      gates      : list[dict] — [{rule: str, passed: bool}, ...]
    """

    # ------------------------------------------------------------------ #
    #  Extract inputs with safe defaults
    # ------------------------------------------------------------------ #
    def _to_float(val, default=0.0):
        if val is None:
            return default
        try:
            import pandas as _pd
            if isinstance(val, _pd.Series):
                return float(val.iloc[-1]) if len(val) > 0 else default
        except Exception:
            pass
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    # Map trend_state to direction
    ts = trend.get("trend_state", trend.get("direction", "neutral"))
    direction = str(ts).lower() if ts else "neutral"
    if direction == "bullish":
        direction = "bullish"
    elif direction == "bearish":
        direction = "bearish"
    else:
        direction = "neutral"
    in_consolidation = bool(trend.get("in_consolidation", False))
    pullback_to_ema = bool(trend.get("pullback_to_ema", False))
    price = _to_float(trend.get("price", trend.get("last_close", 0)))
    ema50 = _to_float(trend.get("ema50", 0))

    rsi = _to_float(momentum.get("rsi", 50))
    ob_os_flag = momentum.get("ob_os_flag", "neutral")

    vol_confirms = bool(volume.get("confirms", False))
    vol_ratio = float(volume.get("ratio", 1.0))

    ob_bias = ob_analysis.get("signal_bias", "HOLD")
    price_in_bullish_ob = bool(ob_analysis.get("price_in_bullish_ob", False))
    price_in_bearish_ob = bool(ob_analysis.get("price_in_bearish_ob", False))
    nearest_bullish_ob = ob_analysis.get("nearest_bullish_ob")
    nearest_bearish_ob = ob_analysis.get("nearest_bearish_ob")

    # ------------------------------------------------------------------ #
    #  FLAT gate — consolidation blocks all trades (Rule 2)
    # ------------------------------------------------------------------ #
    if in_consolidation:
        return {
            "signal": "FLAT",
            "entry_type": None,
            "confidence": 0,
            "reasons": [
                "✗ Rule 2: Market is in consolidation/squeeze — no trades.",
                "→ Final signal: FLAT",
            ],
            "gates": [
                {"rule": "Rule 2: Not in consolidation", "passed": False},
            ],
        }

    # ------------------------------------------------------------------ #
    #  LONG gates (Rules 1, 2, 3, 6)
    # ------------------------------------------------------------------ #
    long_gates = [
        {
            "rule": "Rule 1: Trend is bullish",
            "passed": direction == "bullish",
        },
        {
            "rule": "Rule 2: Not in consolidation/squeeze",
            "passed": not in_consolidation,
        },
        {
            "rule": "Rule 3: Bullish OB key level nearby or price in OB",
            "passed": (
                price_in_bullish_ob
                or nearest_bullish_ob is not None
                or ob_bias in ("BUY", "HOLD")
            ),
        },
        {
            "rule": "Rule 6: Volume confirms move",
            "passed": vol_confirms and vol_ratio >= 1.1,
        },
        {
            "rule": "Rule 4: Pullback entry (not chasing impulse)",
            "passed": pullback_to_ema or price_in_bullish_ob,
        },
        {
            "rule": "Momentum not overbought",
            "passed": ob_os_flag != "overbought" and rsi < 75,
        },
    ]

    # ------------------------------------------------------------------ #
    #  SHORT gates (mirror of LONG, bearish direction)
    # ------------------------------------------------------------------ #
    short_gates = [
        {
            "rule": "Rule 1: Trend is bearish",
            "passed": direction == "bearish",
        },
        {
            "rule": "Rule 2: Not in consolidation/squeeze",
            "passed": not in_consolidation,
        },
        {
            "rule": "Rule 3: Bearish OB key level nearby or price in OB",
            "passed": (
                price_in_bearish_ob
                or nearest_bearish_ob is not None
                or ob_bias in ("SELL", "REDUCE")
            ),
        },
        {
            "rule": "Rule 6: Volume confirms move",
            "passed": vol_confirms and vol_ratio >= 1.1,
        },
        {
            "rule": "Rule 4: Pullback entry (not chasing dump)",
            "passed": (
                (ema50 > 0 and price >= ema50 * 0.99)  # near EMA from below
                or price_in_bearish_ob
            ),
        },
        {
            "rule": "Momentum not oversold",
            "passed": ob_os_flag != "oversold" and rsi > 25,
        },
    ]

    # ------------------------------------------------------------------ #
    #  Score each direction
    # ------------------------------------------------------------------ #
    long_score = sum(1 for g in long_gates if g["passed"])
    short_score = sum(1 for g in short_gates if g["passed"])

    # Require at least 4/6 gates for a directional signal
    THRESHOLD = 4

    # ------------------------------------------------------------------ #
    #  Entry type detection
    # ------------------------------------------------------------------ #
    def _entry_type_long() -> str:
        if price_in_bullish_ob:
            return "OB Bounce"
        if pullback_to_ema:
            return "Pullback to EMA"
        return "Breakout"

    def _entry_type_short() -> str:
        if price_in_bearish_ob:
            return "OB Bounce"
        if ema50 > 0 and abs(price - ema50) / ema50 < 0.005:
            return "Pullback to EMA"
        return "Breakout"

    # ------------------------------------------------------------------ #
    #  Final signal
    # ------------------------------------------------------------------ #
    if long_score >= THRESHOLD and long_score >= short_score:
        signal = "LONG"
        entry_type = _entry_type_long()
        gates = long_gates
        confidence = int(round(long_score / len(long_gates) * 100))
    elif short_score >= THRESHOLD:
        signal = "SHORT"
        entry_type = _entry_type_short()
        gates = short_gates
        confidence = int(round(short_score / len(short_gates) * 100))
    else:
        signal = "FLAT"
        entry_type = None
        gates = long_gates  # show long gates as reference
        confidence = 0

    # ------------------------------------------------------------------ #
    #  Human-readable reasons
    # ------------------------------------------------------------------ #
    reasons: list[str] = []
    for g in gates:
        status = "✓" if g["passed"] else "✗"
        reasons.append(f"{status} {g['rule']}")

    reasons.append(
        f"→ Final signal: {signal}"
        + (f" via {entry_type}" if entry_type else "")
        + f" (confidence {confidence}%)"
    )

    # Extra context
    reasons.append(f"  RSI: {rsi:.1f} | Volume ratio: {vol_ratio:.2f}x | OB bias: {ob_bias}")

    return {
        "signal": signal,
        "entry_type": entry_type,
        "confidence": confidence,
        "reasons": reasons,
        "gates": gates,
    }
