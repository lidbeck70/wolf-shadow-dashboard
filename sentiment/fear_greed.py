"""
Fear & Greed Index — Synthetic composite built from freely available yfinance data.
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _rsi(series: pd.Series, period: int = 14) -> float:
    """Compute RSI(period) from a price series. Returns the most recent value."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def _fetch_close(ticker: str, period: str = "1y") -> pd.Series:
    """Return adjusted-close series for a ticker, or empty Series on failure."""
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return pd.Series(dtype=float)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna()
    except Exception:
        return pd.Series(dtype=float)


def _score_label_color(score: float) -> tuple[str, str]:
    """Return (label, hex_color) for a 0-100 fear/greed score."""
    if score < 25:
        return "Extreme Fear", "#ff3355"
    elif score < 45:
        return "Fear", "#ff8800"
    elif score < 55:
        return "Neutral", "#ffdd00"
    elif score < 75:
        return "Greed", "#88dd00"
    else:
        return "Extreme Greed", "#00ff88"


# ---------------------------------------------------------------------------
# Component calculators
# ---------------------------------------------------------------------------

def _comp_volatility() -> float:
    """
    VIX proxy: Score = 100 - min(VIX * 3, 100).
    High VIX → fear (low score).
    """
    vix = _fetch_close("^VIX", period="1mo")
    if vix.empty:
        return 50.0
    vix_val = float(vix.iloc[-1])
    return _clamp(100.0 - vix_val * 3.0)


def _comp_breadth() -> float:
    """
    Market Breadth: % of last 50 trading days that were up days for S&P 500.
    Score = up_pct * 100.
    """
    gspc = _fetch_close("^GSPC", period="6mo")
    if len(gspc) < 51:
        return 50.0
    last_50 = gspc.iloc[-51:]
    up_days = (last_50.diff().dropna() > 0).sum()
    up_pct = up_days / 50.0
    return _clamp(up_pct * 100.0)


def _comp_ema200_distance() -> float:
    """
    Distance from EMA200 for S&P 500.
    Score = 50 + (pct_above_ema200 * 200), clamped 0-100.
    """
    gspc = _fetch_close("^GSPC", period="2y")
    if len(gspc) < 200:
        return 50.0
    ema200 = gspc.ewm(span=200, adjust=False).mean()
    latest_price = float(gspc.iloc[-1])
    latest_ema = float(ema200.iloc[-1])
    pct_above = (latest_price - latest_ema) / latest_ema if latest_ema != 0 else 0.0
    return _clamp(50.0 + pct_above * 200.0)


def _comp_sector_trends() -> float:
    """
    Sector Trend Distribution: how many of 11 US sector ETFs trade above their EMA200.
    Score = (count / 11) * 100.
    """
    sectors = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLB", "XLC", "XLY", "XLP", "XLU", "XLRE"]
    above_count = 0
    for ticker in sectors:
        prices = _fetch_close(ticker, period="2y")
        if len(prices) < 200:
            above_count += 1  # treat missing as neutral (above)
            continue
        ema200 = prices.ewm(span=200, adjust=False).mean()
        if float(prices.iloc[-1]) > float(ema200.iloc[-1]):
            above_count += 1
    return _clamp((above_count / 11.0) * 100.0)


def _comp_safe_haven() -> float:
    """
    Safe Haven Demand: compare gold (GLD) vs S&P 500 over 20 trading days.
    Gold outperforming → fear. Score = 50 - (gold_outperformance * 500), clamped 0-100.
    """
    gold = _fetch_close("GLD", period="6mo")
    gspc = _fetch_close("^GSPC", period="6mo")
    if len(gold) < 21 or len(gspc) < 21:
        return 50.0
    gold_ret = (float(gold.iloc[-1]) - float(gold.iloc[-21])) / float(gold.iloc[-21])
    gspc_ret = (float(gspc.iloc[-1]) - float(gspc.iloc[-21])) / float(gspc.iloc[-21])
    outperformance = gold_ret - gspc_ret  # positive = gold winning = fear
    return _clamp(50.0 - outperformance * 500.0)


def _comp_momentum() -> float:
    """
    RSI(14) of S&P 500. Score = RSI value directly (high RSI = greed).
    """
    gspc = _fetch_close("^GSPC", period="6mo")
    if len(gspc) < 15:
        return 50.0
    return _clamp(_rsi(gspc, 14))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

WEIGHTS = {
    "Volatility (VIX)": 0.20,
    "Market Breadth": 0.15,
    "EMA200 Distance": 0.20,
    "Sector Trends": 0.20,
    "Safe Haven Demand": 0.10,
    "Momentum (RSI)": 0.15,
}

COMPONENT_FUNCS = {
    "Volatility (VIX)": _comp_volatility,
    "Market Breadth": _comp_breadth,
    "EMA200 Distance": _comp_ema200_distance,
    "Sector Trends": _comp_sector_trends,
    "Safe Haven Demand": _comp_safe_haven,
    "Momentum (RSI)": _comp_momentum,
}


@st.cache_data(ttl=1800)
def compute_fear_greed() -> dict:
    """
    Compute the synthetic Fear & Greed index.

    Returns:
        {
            "score": float (0–100),
            "label": str,
            "color": str (hex),
            "components": {name: score, ...},
        }
    """
    components: dict[str, float] = {}
    for name, fn in COMPONENT_FUNCS.items():
        try:
            components[name] = round(fn(), 1)
        except Exception:
            components[name] = 50.0

    score = sum(components[name] * WEIGHTS[name] for name in WEIGHTS)
    score = round(_clamp(score), 1)
    label, color = _score_label_color(score)

    return {
        "score": score,
        "label": label,
        "color": color,
        "components": components,
    }


def get_retail_flow() -> dict:
    """
    Placeholder for Nasdaq retail order flow data.
    """
    return {
        "status": "placeholder",
        "message": "Nasdaq API integration coming soon",
        "net_flow_5d": None,
        "net_flow_30d": None,
    }


@st.cache_data(ttl=1800)
def get_fear_greed_history(days: int = 60) -> pd.DataFrame:
    """
    Build a rolling Fear & Greed trend using VIX as proxy for the last `days` trading days.
    Score = 100 - min(VIX * 3, 100), with a smoothed rolling average for visual appeal.

    Returns a DataFrame with columns: ['date', 'score']
    """
    vix = _fetch_close("^VIX", period="1y")
    if vix.empty:
        # Return flat neutral line as fallback
        dates = pd.date_range(end=datetime.today(), periods=days, freq="B")
        return pd.DataFrame({"date": dates, "score": [50.0] * days})

    vix_recent = vix.iloc[-days:]
    raw_scores = (100.0 - (vix_recent * 3.0)).clip(0, 100)
    # Smooth with a 5-day rolling mean to look more like a real index
    smoothed = raw_scores.rolling(window=5, min_periods=1).mean().round(1)

    df = pd.DataFrame({
        "date": smoothed.index,
        "score": smoothed.values,
    })
    return df
