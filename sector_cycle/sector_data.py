"""
sector_data.py
Data module for Sector & Global Regime analysis.
Fetches ETF/index data via yfinance and computes trend states.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List

# ---------------------------------------------------------------------------
# Instrument registry
# ---------------------------------------------------------------------------

INSTRUMENTS = [
    # ----- Nordic -----
    {"ticker": "^OMX",     "name": "OMXS30 (Sweden)",  "category": "Nordic",  "color": "rgba(0,255,255,0.85)"},
    {"ticker": "^OSEAX",   "name": "OSEBX (Norway)",   "category": "Nordic",  "color": "rgba(0,220,255,0.85)"},
    {"ticker": "^OMXC25",  "name": "OMXC25 (Denmark)", "category": "Nordic",  "color": "rgba(0,180,255,0.85)"},
    {"ticker": "^OMXH25",  "name": "OMXH25 (Finland)", "category": "Nordic",  "color": "rgba(0,140,255,0.85)"},

    # ----- Global indices -----
    {"ticker": "^GSPC",    "name": "S&P 500",          "category": "Global",  "color": "rgba(255,0,255,0.85)"},
    {"ticker": "^IXIC",    "name": "Nasdaq",            "category": "Global",  "color": "rgba(220,0,255,0.85)"},
    {"ticker": "^GDAXI",   "name": "DAX",               "category": "Global",  "color": "rgba(180,0,255,0.85)"},
    {"ticker": "^FTSE",    "name": "FTSE 100",          "category": "Global",  "color": "rgba(140,0,255,0.85)"},
    {"ticker": "^N225",    "name": "Nikkei 225",        "category": "Global",  "color": "rgba(255,100,0,0.85)"},
    {"ticker": "^HSI",     "name": "Hang Seng",         "category": "Global",  "color": "rgba(255,140,0,0.85)"},

    # ----- US Sector ETFs -----
    {"ticker": "XLE",  "name": "Energy",            "category": "Sector", "color": "rgba(255,221,0,0.85)"},
    {"ticker": "XLF",  "name": "Financials",        "category": "Sector", "color": "rgba(255,200,0,0.85)"},
    {"ticker": "XLK",  "name": "Technology",        "category": "Sector", "color": "rgba(0,255,136,0.85)"},
    {"ticker": "XLV",  "name": "Healthcare",        "category": "Sector", "color": "rgba(0,220,120,0.85)"},
    {"ticker": "XLI",  "name": "Industrials",       "category": "Sector", "color": "rgba(0,180,100,0.85)"},
    {"ticker": "XLB",  "name": "Materials",         "category": "Sector", "color": "rgba(0,255,200,0.85)"},
    {"ticker": "XLC",  "name": "Communication",     "category": "Sector", "color": "rgba(0,200,255,0.85)"},
    {"ticker": "XLY",  "name": "Consumer Disc.",    "category": "Sector", "color": "rgba(255,100,100,0.85)"},
    {"ticker": "XLP",  "name": "Consumer Staples",  "category": "Sector", "color": "rgba(200,80,80,0.85)"},
    {"ticker": "XLU",  "name": "Utilities",         "category": "Sector", "color": "rgba(160,60,255,0.85)"},
    {"ticker": "XLRE", "name": "Real Estate",       "category": "Sector", "color": "rgba(120,40,200,0.85)"},
]

# Handy lookup maps
TICKER_MAP: Dict[str, dict] = {inst["ticker"]: inst for inst in INSTRUMENTS}

NORDIC_TICKERS  = [i["ticker"] for i in INSTRUMENTS if i["category"] == "Nordic"]
GLOBAL_TICKERS  = [i["ticker"] for i in INSTRUMENTS if i["category"] == "Global"]
SECTOR_TICKERS  = [i["ticker"] for i in INSTRUMENTS if i["category"] == "Sector"]
ALL_TICKERS     = [i["ticker"] for i in INSTRUMENTS]

# ---------------------------------------------------------------------------
# Trend-state constants
# ---------------------------------------------------------------------------

TREND_UP      = "Uptrend"
TREND_DOWN    = "Downtrend"
TREND_NEUTRAL = "Neutral"


def _compute_trend_state(close: pd.Series, ema50: pd.Series, ema200: pd.Series) -> pd.Series:
    """
    Row-wise trend classification:
      Close > EMA200 and EMA50 > EMA200  → Uptrend
      Close < EMA200 and EMA50 < EMA200  → Downtrend
      else                               → Neutral
    """
    conditions = [
        (close > ema200) & (ema50 > ema200),
        (close < ema200) & (ema50 < ema200),
    ]
    choices = [TREND_UP, TREND_DOWN]
    return pd.Series(
        np.select(conditions, choices, default=TREND_NEUTRAL),
        index=close.index,
        name="trend_state",
    )


# ---------------------------------------------------------------------------
# Core data fetch
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False, max_entries=10)
def fetch_regime_data(tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV for each ticker and attach EMA50, EMA200, trend_state.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are tickers; each DataFrame has columns:
        Close, EMA50, EMA200, trend_state
    """
    result: Dict[str, pd.DataFrame] = {}

    try:
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return result

    # yf returns MultiIndex columns when multiple tickers are requested
    single = len(tickers) == 1

    for ticker in tickers:
        try:
            if single:
                close_series = raw["Close"]
            else:
                close_series = raw["Close"][ticker]

            close_series = close_series.dropna()
            if len(close_series) < 10:
                continue

            ema50  = close_series.ewm(span=50,  adjust=False).mean()
            ema200 = close_series.ewm(span=200, adjust=False).mean()

            df = pd.DataFrame({
                "Close":       close_series,
                "EMA50":       ema50,
                "EMA200":      ema200,
                "trend_state": _compute_trend_state(close_series, ema50, ema200),
            })
            result[ticker] = df

        except Exception:
            continue

    return result


# ---------------------------------------------------------------------------
# Sector summary builder
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False, max_entries=10)
def compute_sector_summary() -> pd.DataFrame:
    """
    Build a summary DataFrame for all instruments.

    Columns
    -------
    ticker, name, category, trend_state, last_close,
    pct_change_1m, pct_change_3m, sparkline_data
    """
    data = fetch_regime_data(ALL_TICKERS, period="1y")

    rows = []
    for inst in INSTRUMENTS:
        ticker = inst["ticker"]
        df = data.get(ticker)

        if df is None or df.empty:
            rows.append({
                "ticker":         ticker,
                "name":           inst["name"],
                "category":       inst["category"],
                "color":          inst["color"],
                "trend_state":    TREND_NEUTRAL,
                "last_close":     float("nan"),
                "pct_change_1m":  float("nan"),
                "pct_change_3m":  float("nan"),
                "sparkline_data": [],
            })
            continue

        closes = df["Close"]
        last_close   = float(closes.iloc[-1])
        trend_state  = str(df["trend_state"].iloc[-1])

        # Approximate trading days: 1M ≈ 21 days, 3M ≈ 63 days
        def _pct(n: int) -> float:
            if len(closes) > n:
                return float((closes.iloc[-1] / closes.iloc[-n] - 1) * 100)
            return float("nan")

        pct_1m = _pct(21)
        pct_3m = _pct(63)

        # Last 60 closes for sparkline
        sparkline = closes.iloc[-60:].tolist()

        rows.append({
            "ticker":         ticker,
            "name":           inst["name"],
            "category":       inst["category"],
            "color":          inst["color"],
            "trend_state":    trend_state,
            "last_close":     last_close,
            "pct_change_1m":  pct_1m,
            "pct_change_3m":  pct_3m,
            "sparkline_data": sparkline,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers used by the UI layer
# ---------------------------------------------------------------------------

def trend_color_rgba(trend: str, opacity: float = 0.9) -> str:
    """Return an rgba() color string for a given trend_state."""
    mapping = {
        TREND_UP:      f"rgba(0,255,136,{opacity})",
        TREND_DOWN:    f"rgba(255,51,85,{opacity})",
        TREND_NEUTRAL: f"rgba(255,221,0,{opacity})",
    }
    return mapping.get(trend, f"rgba(74,74,106,{opacity})")
