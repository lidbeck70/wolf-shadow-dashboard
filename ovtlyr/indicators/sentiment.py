"""
Synthetic Fear & Greed sentiment score for OVTLYR.
Pure functions — no Streamlit imports.
"""

import math


def compute_sentiment(
    trend: dict,
    volatility: dict,
    momentum: dict,
    volume: dict,
    breadth: dict = None,
) -> dict:
    """
    Compute a synthetic Fear & Greed score (0-100) from indicator dicts.

    Component weights
    -----------------
    Volatility   20 % : 100 - risk_score  (low vol = greedy)
    Volume       15 % : volume_ratio clamped to [0, 3], scaled to 0-100
    Breadth      15 % : pct_bullish * 100 if available, else 50
    Trend        25 % : 100 if Bullish, 50 if Neutral, 0 if Bearish
    Momentum     25 % : RSI value directly (0-100)

    Returns a dict with:
        score : int    – 0-100
        label : "Extreme Fear" | "Fear" | "Neutral" | "Greed" | "Extreme Greed"
        color : str    – rgba color string

    Label thresholds
    ----------------
    0-20   : Extreme Fear
    21-40  : Fear
    41-60  : Neutral
    61-80  : Greed
    81-100 : Extreme Greed
    """

    # --- Volatility component (20%) ---
    risk_score = volatility.get("risk_score", 50)
    vol_component = 100 - risk_score  # low risk → high greed

    # --- Volume component (15%) ---
    volume_ratio = volume.get("volume_ratio", 1.0)
    if volume_ratio is None or (isinstance(volume_ratio, float) and math.isnan(volume_ratio)):
        volume_ratio = 1.0
    # Clamp to [0, 3] then scale to 0-100
    volume_clamped = max(0.0, min(3.0, volume_ratio))
    vol_score = (volume_clamped / 3.0) * 100

    # --- Breadth component (15%) ---
    if breadth is not None:
        pct_bullish = breadth.get("pct_bullish", 0.5)
        if pct_bullish is None:
            pct_bullish = 0.5
        breadth_score = float(pct_bullish) * 100
    else:
        breadth_score = 50.0

    # --- Trend component (25%) ---
    trend_state = trend.get("trend_state", "Neutral")
    if trend_state == "Bullish":
        trend_score = 100.0
    elif trend_state == "Bearish":
        trend_score = 0.0
    else:
        trend_score = 50.0

    # --- Momentum component (25%) ---
    rsi = momentum.get("rsi", 50.0)
    if rsi is None or (isinstance(rsi, float) and math.isnan(rsi)):
        rsi = 50.0
    momentum_score = max(0.0, min(100.0, float(rsi)))

    # --- Weighted aggregate ---
    score_float = (
        0.20 * vol_component
        + 0.15 * vol_score
        + 0.15 * breadth_score
        + 0.25 * trend_score
        + 0.25 * momentum_score
    )
    score = int(round(max(0, min(100, score_float))))

    # --- Label ---
    if score <= 20:
        label = "Extreme Fear"
    elif score <= 40:
        label = "Fear"
    elif score <= 60:
        label = "Neutral"
    elif score <= 80:
        label = "Greed"
    else:
        label = "Extreme Greed"

    # --- Color (rgba, no 8-digit hex) ---
    # Gradient: red (Extreme Fear) → orange → yellow (Neutral) → green (Extreme Greed)
    if score <= 20:
        color = "rgba(220, 38, 38, 0.9)"    # red
    elif score <= 40:
        color = "rgba(234, 88, 12, 0.9)"    # orange
    elif score <= 60:
        color = "rgba(202, 138, 4, 0.9)"    # amber/yellow
    elif score <= 80:
        color = "rgba(22, 163, 74, 0.85)"   # green
    else:
        color = "rgba(5, 150, 105, 0.9)"    # emerald

    return {
        "score": score,
        "label": label,
        "color": color,
    }
