"""
Long-term signal computation based on the user's 10 rules.

BUY  — all 6 gates pass
HOLD — regime orange OR sentiment 50-70 (partial pass)
REDUCE — ob_analysis signal_bias == "REDUCE", or risk > 70, or regime degrading
SELL — any single hard exit triggers
"""


def compute_longterm_signal(
    trend: dict,
    sentiment: dict,
    volatility: dict,
    ob_analysis: dict,
    sector_green: bool,
) -> dict:
    """
    Compute long-term signal based on the user's 10 rules.

    Parameters
    ----------
    trend : dict
        Keys expected:
          price           – current price (float)
          ema50           – 50-period EMA (float)
          ema200          – 200-period EMA (float)
          regime_color    – "green" | "orange" | "red"
          regime_prev     – previous regime_color (for degrading detection)
    sentiment : dict
        Keys expected:
          score           – Fear & Greed score 0–100 (int/float)
    volatility : dict
        Keys expected:
          risk_score      – composite risk 0–100 (int/float)
          atr             – Average True Range (float)
          hist_vol        – historical volatility % (float)
    ob_analysis : dict
        Keys expected:
          signal_bias     – "BUY" | "HOLD" | "REDUCE" | "SELL"
    sector_green : bool
        True if the stock's sector is in a green / bullish regime.

    Returns
    -------
    dict with keys:
      signal     : str  — "BUY" | "HOLD" | "REDUCE" | "SELL"
      confidence : int  — 0–100
      reasons    : list[str]  — plain-language explanation of each rule
      gates      : list[dict] — [{rule: str, passed: bool}, ...]
    """

    # ------------------------------------------------------------------ #
    #  Extract inputs with safe defaults
    # ------------------------------------------------------------------ #
    price = float(trend.get("price", 0))
    ema50 = float(trend.get("ema50", 0))
    ema200 = float(trend.get("ema200", 0))
    regime = trend.get("regime_color", "red")
    regime_prev = trend.get("regime_prev", regime)

    sentiment_score = float(sentiment.get("score", 50))
    risk_score = float(volatility.get("risk_score", 50))
    ob_bias = ob_analysis.get("signal_bias", "HOLD")

    # ------------------------------------------------------------------ #
    #  SELL gates — any single one triggers SELL immediately
    # ------------------------------------------------------------------ #
    sell_gates = [
        {
            "rule": "Rule 6: Price < EMA200 (hard exit)",
            "passed": price >= ema200,
        },
        {
            "rule": "Rule 7: Regime is not red",
            "passed": regime != "red",
        },
        {
            "rule": "Rule 4+7: Sector NOT green while regime != green",
            "passed": sector_green or regime == "green",
        },
        {
            "rule": "OB signal_bias != SELL",
            "passed": ob_bias != "SELL",
        },
    ]

    sell_triggered = any(not g["passed"] for g in sell_gates)

    # ------------------------------------------------------------------ #
    #  BUY gates — ALL six must pass
    # ------------------------------------------------------------------ #
    buy_gates = [
        {
            "rule": "Rule 2: Price > EMA200",
            "passed": price > ema200,
        },
        {
            "rule": "Rule 3: EMA50 > EMA200 (golden cross)",
            "passed": ema50 > ema200,
        },
        {
            "rule": "Rule 1: Regime is green",
            "passed": regime == "green",
        },
        {
            "rule": "Rule 4: Sector is green",
            "passed": sector_green,
        },
        {
            "rule": "Rule 5: Fear & Greed < 60 (no euphoria)",
            "passed": sentiment_score < 60,
        },
        {
            "rule": "OB signal_bias is BUY or HOLD",
            "passed": ob_bias in ("BUY", "HOLD"),
        },
    ]

    all_buy_gates_pass = all(g["passed"] for g in buy_gates)

    # ------------------------------------------------------------------ #
    #  REDUCE conditions
    # ------------------------------------------------------------------ #
    regime_degrading = regime_prev == "green" and regime == "orange"
    reduce_conditions = [
        ob_bias == "REDUCE",
        risk_score > 70,
        regime_degrading,
    ]
    reduce_triggered = any(reduce_conditions)

    # ------------------------------------------------------------------ #
    #  HOLD conditions
    # ------------------------------------------------------------------ #
    hold_conditions = regime == "orange" or (50 <= sentiment_score <= 70)

    # ------------------------------------------------------------------ #
    #  Final signal determination
    # ------------------------------------------------------------------ #
    if sell_triggered:
        signal = "SELL"
    elif reduce_triggered:
        signal = "REDUCE"
    elif all_buy_gates_pass:
        signal = "BUY"
    elif hold_conditions:
        signal = "HOLD"
    else:
        # Some buy gates fail but no sell/reduce triggered — lean HOLD
        signal = "HOLD"

    # ------------------------------------------------------------------ #
    #  Confidence score (percentage of gates that passed)
    # ------------------------------------------------------------------ #
    all_gates = buy_gates + sell_gates
    passed_count = sum(1 for g in all_gates if g["passed"])
    confidence = int(round(passed_count / len(all_gates) * 100))

    # ------------------------------------------------------------------ #
    #  Human-readable reasons
    # ------------------------------------------------------------------ #
    reasons: list[str] = []

    # Buy gate reasons
    for g in buy_gates:
        status = "✓" if g["passed"] else "✗"
        reasons.append(f"{status} {g['rule']}")

    # Sell gate reasons (only report failures — they're the critical ones)
    for g in sell_gates:
        if not g["passed"]:
            reasons.append(f"✗ SELL TRIGGER — {g['rule']}")

    # Reduce reasons
    if ob_bias == "REDUCE":
        reasons.append("⚠ OB analysis recommends REDUCE")
    if risk_score > 70:
        reasons.append(f"⚠ Risk score elevated: {risk_score:.0f}/100")
    if regime_degrading:
        reasons.append("⚠ Regime degrading: green → orange")

    # Summary
    reasons.append(f"→ Final signal: {signal} (confidence {confidence}%)")

    # ------------------------------------------------------------------ #
    #  Combined gates list for the UI gates table
    # ------------------------------------------------------------------ #
    gates = buy_gates + [
        {
            "rule": "No SELL trigger active",
            "passed": not sell_triggered,
        },
        {
            "rule": "No REDUCE condition active",
            "passed": not reduce_triggered,
        },
        {
            "rule": "Risk score ≤ 70",
            "passed": risk_score <= 70,
        },
        {
            "rule": "Sentiment not euphoric (< 60)",
            "passed": sentiment_score < 60,
        },
    ]

    return {
        "signal": signal,
        "confidence": confidence,
        "reasons": reasons,
        "gates": gates,
    }
