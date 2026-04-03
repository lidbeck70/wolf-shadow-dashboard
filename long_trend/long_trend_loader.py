"""
long_trend_loader.py
Data loading and analysis for Long-Term Trend & Drawdown module.

Features:
  - 10-20 year price history via yfinance
  - EMA50 / EMA200 computation
  - Drawdown detection (peak-to-trough > 10%)
  - Drawdown classification using fundamental data (Börsdata or yfinance)
  - Trend phase classification (Bullish / Neutral / Bearish)
  - Cycle position classifier (7 phases)
  - Rick Rule buy/sell signal generation
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Streamlit import
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    st = None  # type: ignore[assignment]
    _STREAMLIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Börsdata import (with dual-path fallback)
# ---------------------------------------------------------------------------
_BORSDATA_OK = False
_get_api = None  # type: ignore
_KPI: dict = {}

for _attempt in range(1):
    try:
        from borsdata_api import get_api as _get_api, KPI as _KPI  # type: ignore
        _BORSDATA_OK = True
        break
    except ImportError:
        pass
    try:
        from dashboard.borsdata_api import get_api as _get_api, KPI as _KPI  # type: ignore
        _BORSDATA_OK = True
        break
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Cache helper
# ---------------------------------------------------------------------------
def _cache(ttl: int = 3600):
    def decorator(fn):
        if _STREAMLIT_AVAILABLE and st is not None:
            return st.cache_data(ttl=ttl)(fn)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PERIOD_MAP = {
    "5y": "5y",
    "10y": "10y",
    "20y": "20y",
}

# Market indices for macro check
_MACRO_INDEX = "^GSPC"
_NORDIC_INDEX = "^OMX"  # OMX Stockholm 30


def _safe_pct(new_val, old_val) -> Optional[float]:
    """Return percentage change or None if inputs invalid."""
    try:
        if old_val is None or old_val == 0:
            return None
        return (new_val - old_val) / abs(old_val)
    except Exception:
        return None


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI for a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------
# Map period strings to approximate bar counts for Börsdata
_PERIOD_TO_BARS = {
    "5y": 1260,
    "10y": 2520,
    "20y": 5040,
    "max": 5040,
}


@_cache(ttl=3600)
def fetch_long_history(ticker: str, period: str = "10y") -> pd.DataFrame:
    """
    Fetch long-term OHLCV data for a ticker.

    Data source priority:
      1. Börsdata API (up to 20 years, reliable, no rate limit issues)
      2. yfinance (fallback for tickers not in Börsdata, e.g. US/global)

    Returns DataFrame with columns: Open, High, Low, Close, Volume
    indexed by Date.
    """
    # Skip Börsdata for index tickers and non-stock instruments
    _skip_borsdata = (
        ticker.startswith("^")              # Index (^GSPC, ^OMX, etc)
        or ticker.endswith(".AS")           # Amsterdam ETFs (UCITS)
        or ticker.endswith(".DE")           # Xetra ETFs (UCITS)
        or ticker.endswith(".PA")           # Paris ETFs
        or ticker.endswith(".L")            # London ETFs
        or ticker in ("SPY", "QQQ", "IWM", "GLD", "SLV", "GDX", "GDXJ",
                      "XLE", "XLB", "XLF", "XLK", "XLV", "XLI", "XLY",
                      "XLP", "XLU", "XLC", "XLRE")
    )

    # --- Try Börsdata first (only for stocks it covers) ---
    if _BORSDATA_OK and _get_api is not None and not _skip_borsdata:
        try:
            api = _get_api()
            if api.is_configured:
                max_bars = _PERIOD_TO_BARS.get(PERIOD_MAP.get(period, period), 2520)
                df = api.get_stockprices_df(ticker, max_count=max_bars)
                if not df.empty and len(df) >= 20:
                    logger.info(
                        "fetch_long_history(%s): Börsdata OK, %d bars",
                        ticker, len(df),
                    )
                    return df
        except Exception as exc:
            logger.warning("fetch_long_history(%s) Börsdata failed: %s", ticker, exc)

    # --- Fallback to yfinance ---
    yf_period = PERIOD_MAP.get(period, period)
    fallback_periods = [yf_period]
    if yf_period in ("20y", "max"):
        fallback_periods += ["10y", "5y"]
    elif yf_period == "10y":
        fallback_periods += ["5y", "2y"]
    elif yf_period == "5y":
        fallback_periods += ["2y", "1y"]

    import time as _time
    for attempt_period in fallback_periods:
        for attempt in range(2):
            try:
                tk = yf.Ticker(ticker)
                df = tk.history(period=attempt_period, auto_adjust=True, progress=False)
                df = df.dropna(how="all")
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.index.name = "Date"
                if not df.empty and len(df) >= 20:
                    return df
            except Exception as exc:
                logger.warning(
                    "fetch_long_history(%s, %s) yfinance attempt %d: %s",
                    ticker, attempt_period, attempt + 1, exc,
                )
                if attempt == 0:
                    _time.sleep(1)

    logger.error("fetch_long_history(%s): all sources failed", ticker)
    return pd.DataFrame()


@_cache(ttl=3600)
def _fetch_index_history(index_ticker: str, period: str = "10y") -> pd.Series:
    """Fetch close series for a market index."""
    try:
        tk = yf.Ticker(index_ticker)
        df = tk.history(period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.Series(dtype=float)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df["Close"]
    except Exception:
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------
def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EMA50, EMA200, RSI14, and Volume_MA20 columns to price DataFrame.
    Returns the enriched DataFrame (copy).
    """
    if df.empty or "Close" not in df.columns:
        return df

    out = df.copy()
    out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean()
    out["EMA200"] = out["Close"].ewm(span=200, adjust=False).mean()
    out["RSI14"] = _rsi(out["Close"], 14)

    # Volume moving average (handle missing volume gracefully)
    if "Volume" in out.columns:
        out["Volume_MA20"] = out["Volume"].rolling(20).mean()
    else:
        out["Volume_MA20"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Drawdown detection
# ---------------------------------------------------------------------------
def detect_drawdowns(
    price: pd.Series,
    min_drop: float = 0.10,
) -> List[dict]:
    """
    Detect peak-to-trough drawdowns exceeding `min_drop` (default 10%).

    Returns list of dicts:
      start, end, peak_price, trough_price, max_drop_pct, duration_days
    """
    if price.empty:
        return []

    drawdowns: List[dict] = []
    prices = price.dropna()
    n = len(prices)
    dates = prices.index.tolist()
    vals = prices.values

    peak_idx = 0
    peak_val = vals[0]
    in_drawdown = False
    trough_idx = 0
    trough_val = vals[0]

    for i in range(1, n):
        v = vals[i]
        if v > peak_val:
            if in_drawdown:
                drop = (trough_val - peak_val) / peak_val
                if drop <= -min_drop:
                    drawdowns.append({
                        "start": dates[peak_idx],
                        "end": dates[trough_idx],
                        "peak_price": float(peak_val),
                        "trough_price": float(trough_val),
                        "max_drop_pct": float(drop),
                        "duration_days": (dates[trough_idx] - dates[peak_idx]).days,
                        "classification": "Noise/Temporary",
                        "ebitda_delta": None,
                        "margin_delta": None,
                        "debt_delta": None,
                    })
                in_drawdown = False
            peak_idx = i
            peak_val = v
            trough_idx = i
            trough_val = v
        else:
            if not in_drawdown and (v - peak_val) / peak_val <= -min_drop:
                in_drawdown = True
            if in_drawdown and v < trough_val:
                trough_val = v
                trough_idx = i

    # Capture open drawdown at end of series
    if in_drawdown:
        drop = (trough_val - peak_val) / peak_val
        if drop <= -min_drop:
            drawdowns.append({
                "start": dates[peak_idx],
                "end": dates[trough_idx],
                "peak_price": float(peak_val),
                "trough_price": float(trough_val),
                "max_drop_pct": float(drop),
                "duration_days": (dates[trough_idx] - dates[peak_idx]).days,
                "classification": "Noise/Temporary",
                "ebitda_delta": None,
                "margin_delta": None,
                "debt_delta": None,
            })

    return drawdowns


# ---------------------------------------------------------------------------
# Fundamental delta for a drawdown period
# ---------------------------------------------------------------------------
def _get_borsdata_kpi_at(
    ins_id: int,
    kpi_id: int,
    target_year: int,
    report_type: str = "year",
) -> Optional[float]:
    """Get a KPI value near `target_year` from Börsdata history."""
    if not _BORSDATA_OK or _get_api is None:
        return None
    try:
        api = _get_api()
        hist = api.get_kpi_history(ins_id, kpi_id, report_type, "mean")
        if not hist:
            return None
        # Find the closest year
        best = None
        best_diff = 999
        for entry in hist:
            y = entry.get("y") or entry.get("year")
            v = entry.get("v")
            if y is not None and v is not None:
                diff = abs(int(y) - target_year)
                if diff < best_diff:
                    best_diff = diff
                    best = v
        return best
    except Exception as exc:
        logger.debug("_get_borsdata_kpi_at(ins_id=%s, kpi=%s): %s", ins_id, kpi_id, exc)
        return None


def _get_yf_annual_fundamental(
    ticker: str,
    field: str,
    target_year: int,
) -> Optional[float]:
    """
    Extract an annual fundamental field from yfinance financials
    closest to `target_year`.
    """
    try:
        tk = yf.Ticker(ticker)
        fin = tk.financials  # columns are dates
        if fin is None or fin.empty:
            return None
        if field not in fin.index:
            return None
        row = fin.loc[field]
        # Convert column index to years
        best_val = None
        best_diff = 999
        for col, val in row.items():
            try:
                yr = pd.Timestamp(col).year
                diff = abs(yr - target_year)
                if diff < best_diff and pd.notna(val):
                    best_diff = diff
                    best_val = float(val)
            except Exception:
                continue
        return best_val
    except Exception:
        return None


def enrich_drawdown_fundamentals(
    drawdown: dict,
    ticker: str,
    ins_id: Optional[int] = None,
) -> dict:
    """
    Fetch fundamental metrics before and after the drawdown period and
    compute deltas. Modifies and returns the drawdown dict.
    """
    dd = drawdown.copy()
    start_year = dd["start"].year
    end_year = dd["end"].year

    ebitda_before = None
    ebitda_after = None
    margin_before = None
    margin_after = None
    debt_before = None
    debt_after = None

    if _BORSDATA_OK and ins_id is not None and _get_api is not None:
        try:
            ebitda_before = _get_borsdata_kpi_at(ins_id, _KPI["ebitda_m"], start_year)
            ebitda_after = _get_borsdata_kpi_at(ins_id, _KPI["ebitda_m"], end_year)
            margin_before = _get_borsdata_kpi_at(ins_id, _KPI["operating_margin"], start_year)
            margin_after = _get_borsdata_kpi_at(ins_id, _KPI["operating_margin"], end_year)
            debt_before = _get_borsdata_kpi_at(ins_id, _KPI["debt_to_equity"], start_year)
            debt_after = _get_borsdata_kpi_at(ins_id, _KPI["debt_to_equity"], end_year)
        except Exception as exc:
            logger.debug("Börsdata fundamental fetch failed: %s", exc)
    else:
        # Fall back to yfinance annual financials
        ebitda_before = _get_yf_annual_fundamental(ticker, "EBITDA", start_year)
        ebitda_after = _get_yf_annual_fundamental(ticker, "EBITDA", end_year)
        # Operating margin proxy: EBIT / Revenue
        rev_before = _get_yf_annual_fundamental(ticker, "Total Revenue", start_year)
        rev_after = _get_yf_annual_fundamental(ticker, "Total Revenue", end_year)
        ebit_before = _get_yf_annual_fundamental(ticker, "EBIT", start_year)
        ebit_after = _get_yf_annual_fundamental(ticker, "EBIT", end_year)
        if rev_before and ebit_before and rev_before != 0:
            margin_before = ebit_before / rev_before * 100
        if rev_after and ebit_after and rev_after != 0:
            margin_after = ebit_after / rev_after * 100
        # Debt/equity proxy via yfinance balance sheet
        try:
            tk = yf.Ticker(ticker)
            bs = tk.balance_sheet
            if bs is not None and not bs.empty:
                def _get_bs(field, year):
                    if field not in bs.index:
                        return None
                    row = bs.loc[field]
                    best_val = None
                    best_diff = 999
                    for col, val in row.items():
                        try:
                            yr = pd.Timestamp(col).year
                            diff = abs(yr - year)
                            if diff < best_diff and pd.notna(val):
                                best_diff = diff
                                best_val = float(val)
                        except Exception:
                            continue
                    return best_val

                td_before = _get_bs("Total Debt", start_year) or _get_bs("Long Term Debt", start_year)
                te_before = _get_bs("Stockholders Equity", start_year) or _get_bs("Total Equity Gross Minority Interest", start_year)
                td_after = _get_bs("Total Debt", end_year) or _get_bs("Long Term Debt", end_year)
                te_after = _get_bs("Stockholders Equity", end_year) or _get_bs("Total Equity Gross Minority Interest", end_year)
                if td_before and te_before and te_before != 0:
                    debt_before = td_before / te_before * 100
                if td_after and te_after and te_after != 0:
                    debt_after = td_after / te_after * 100
        except Exception as exc:
            logger.debug("yf balance sheet for %s: %s", ticker, exc)

    # Compute deltas
    ebitda_delta = _safe_pct(ebitda_after, ebitda_before) if (ebitda_before and ebitda_after) else None
    margin_delta = (margin_after - margin_before) if (margin_before is not None and margin_after is not None) else None
    debt_delta = _safe_pct(debt_after, debt_before) if (debt_before and debt_after) else None

    dd["ebitda_delta"] = ebitda_delta
    dd["margin_delta"] = margin_delta
    dd["debt_delta"] = debt_delta

    return dd


# ---------------------------------------------------------------------------
# Drawdown classification
# ---------------------------------------------------------------------------
def classify_drawdown(
    drawdown: dict,
    market_close: pd.Series,
    price: pd.Series,
) -> str:
    """
    Classify a drawdown based on:
      1. Fundamental deterioration signals
      2. Macro/market-wide drop
      3. Sector peer drop proxy (broad market as proxy)
    Returns one of: "Noise/Temporary", "Fundamental Deterioration",
                    "Macro/Geopolitical", "Sector-Wide"
    """
    start = drawdown["start"]
    end = drawdown["end"]

    # ── Fundamental checks ──────────────────────────────────────────────
    ebitda_delta = drawdown.get("ebitda_delta")
    margin_delta = drawdown.get("margin_delta")
    debt_delta = drawdown.get("debt_delta")

    if ebitda_delta is not None and ebitda_delta < -0.15:
        return "Fundamental Deterioration"
    if margin_delta is not None and margin_delta < -5.0:
        return "Fundamental Deterioration"
    if debt_delta is not None and debt_delta > 0.30:
        return "Fundamental Deterioration"

    # ── Macro check ─────────────────────────────────────────────────────
    if not market_close.empty:
        try:
            mkt_slice = market_close.loc[
                (market_close.index >= start) & (market_close.index <= end)
            ]
            if len(mkt_slice) >= 2:
                mkt_drop = (mkt_slice.iloc[-1] - mkt_slice.iloc[0]) / mkt_slice.iloc[0]
                if mkt_drop < -0.08:
                    return "Macro/Geopolitical"
        except Exception:
            pass

    # ── Sector-wide (simple: if market dropped 4–8%, treat as sector) ───
    if not market_close.empty:
        try:
            mkt_slice = market_close.loc[
                (market_close.index >= start) & (market_close.index <= end)
            ]
            if len(mkt_slice) >= 2:
                mkt_drop = (mkt_slice.iloc[-1] - mkt_slice.iloc[0]) / mkt_slice.iloc[0]
                if mkt_drop < -0.04:
                    return "Sector-Wide"
        except Exception:
            pass

    return "Noise/Temporary"


# ---------------------------------------------------------------------------
# Trend phase classification
# ---------------------------------------------------------------------------
def classify_trend_phase(
    price: float,
    ema50: float,
    ema200: float,
) -> str:
    """
    Classify current trend phase.

    Rules:
      Bullish  → price > EMA200 AND EMA50 > EMA200
      Bearish  → price < EMA200 AND EMA50 < EMA200
      Neutral  → otherwise
    """
    if pd.isna(price) or pd.isna(ema50) or pd.isna(ema200):
        return "Neutral"
    if price > ema200 and ema50 > ema200:
        return "Bullish"
    if price < ema200 and ema50 < ema200:
        return "Bearish"
    return "Neutral"


# ---------------------------------------------------------------------------
# Cycle position classifier (7 phases)
# ---------------------------------------------------------------------------
def classify_cycle_position(df: pd.DataFrame) -> str:
    """
    Classify current market cycle position using the most recent data.

    Uses last 20 rows to assess EMA direction, price-to-EMA distance,
    RSI momentum, and volume trends.

    Returns one of:
      "Accumulation", "Early Uptrend", "Strong Uptrend", "Late Uptrend",
      "Early Downtrend", "Capitulation", "Recovery"
    """
    required = {"Close", "EMA50", "EMA200", "RSI14"}
    if df.empty or not required.issubset(df.columns):
        return "Neutral"

    recent = df.dropna(subset=["Close", "EMA50", "EMA200"]).tail(20)
    if len(recent) < 5:
        return "Neutral"

    price = float(recent["Close"].iloc[-1])
    ema50 = float(recent["EMA50"].iloc[-1])
    ema200 = float(recent["EMA200"].iloc[-1])
    rsi = float(recent["RSI14"].iloc[-1]) if "RSI14" in recent.columns and not pd.isna(recent["RSI14"].iloc[-1]) else 50.0

    # EMA50 slope (last 10 days)
    ema50_slope = float(recent["EMA50"].diff(10).iloc[-1]) if len(recent) >= 10 else 0.0
    ema200_slope = float(recent["EMA200"].diff(10).iloc[-1]) if len(recent) >= 10 else 0.0

    # EMA50 was below EMA200 previously
    ema50_crossed_above = (
        float(recent["EMA50"].iloc[-1]) > float(recent["EMA200"].iloc[-1])
        and float(recent["EMA50"].iloc[max(0, -5)]) < float(recent["EMA200"].iloc[max(0, -5)])
    )
    price_to_ema200_pct = (price - ema200) / ema200 if ema200 != 0 else 0.0

    # Volume trend (declining = accumulation / late uptrend)
    vol_declining = False
    if "Volume_MA20" in recent.columns:
        vol_vals = recent["Volume_MA20"].dropna()
        if len(vol_vals) >= 10:
            vol_declining = float(vol_vals.iloc[-1]) < float(vol_vals.iloc[-10])

    # ── Decision tree ────────────────────────────────────────────────────
    # Accumulation: price near EMA200 (within ±5%), EMA50 turning up from below
    if (
        abs(price_to_ema200_pct) < 0.05
        and ema50 < ema200
        and ema50_slope > 0
        and vol_declining
    ):
        return "Accumulation"

    # Early Uptrend: price just crossed above EMA200, EMA50 crossing EMA200
    if (
        price > ema200
        and ema50_crossed_above
        and price_to_ema200_pct < 0.10
    ):
        return "Early Uptrend"

    # Late Uptrend: price far above EMA200 (>20%), momentum weakening
    if (
        price > ema200
        and ema50 > ema200
        and price_to_ema200_pct > 0.20
        and (rsi > 70 or ema50_slope < ema50_slope * 0.5)
    ):
        return "Late Uptrend"

    # Strong Uptrend: price well above EMA200, both EMAs rising
    if (
        price > ema200
        and ema50 > ema200
        and ema50_slope > 0
        and ema200_slope > 0
    ):
        return "Strong Uptrend"

    # Recovery: price below EMA200 but bouncing, EMA50 flattening
    if (
        price < ema200
        and ema50 < ema200
        and ema50_slope > -0.001 * ema200  # flattening
        and price > float(recent["Close"].iloc[-5]) if len(recent) >= 5 else True
    ):
        return "Recovery"

    # Capitulation: price well below EMA200, sharp decline
    if (
        price < ema200
        and ema50 < ema200
        and price_to_ema200_pct < -0.15
        and rsi < 30
    ):
        return "Capitulation"

    # Early Downtrend: price crossed below EMA200, EMA50 turning down
    if (
        price < ema200
        and ema50_slope < 0
    ):
        return "Early Downtrend"

    # Default: if price above EMA200 but mixed signals → Early Uptrend-ish
    if price > ema200:
        return "Strong Uptrend"
    return "Recovery"


# ---------------------------------------------------------------------------
# Rick Rule buy/sell signal generation
# ---------------------------------------------------------------------------
def _check_fundamentals_intact(ticker: str, ins_id: Optional[int] = None) -> bool:
    """
    Return True if fundamentals are intact:
      - ROE > 10%
      - No significant debt spike recently
    Uses Börsdata if available, otherwise yfinance.
    """
    try:
        if _BORSDATA_OK and ins_id is not None and _get_api is not None:
            api = _get_api()
            roe_hist = api.get_kpi_history(ins_id, _KPI["roe"], "r12", "mean")
            if roe_hist:
                roe = roe_hist[-1].get("v")
                if roe is not None and roe < 10:
                    return False
            return True
        else:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            roe = info.get("returnOnEquity")
            if roe is not None and roe < 0.10:
                return False
            return True
    except Exception:
        return True  # Assume intact if data unavailable


def generate_rick_rule_signals(
    df: pd.DataFrame,
    ticker: str,
    ins_id: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate Rick Rule BUY and SELL signals.

    BUY criteria:
      - Confirmed uptrend: price > EMA200 AND EMA50 > EMA200
      - Fundamentals intact: ROE > 10%, no debt spike

    SELL criteria:
      - Price below EMA200 for > 10 consecutive days
      - OR fundamental deterioration detected

    Returns (buy_signals, sell_signals) as DataFrames with Date, Price columns.
    """
    required = {"Close", "EMA50", "EMA200"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=["Date", "Price"]), pd.DataFrame(columns=["Date", "Price"])

    data = df.dropna(subset=["Close", "EMA50", "EMA200"]).copy()
    fundamentals_intact = _check_fundamentals_intact(ticker, ins_id)

    buys: List[dict] = []
    sells: List[dict] = []

    above_ema200 = data["Close"] > data["EMA200"]
    ema50_above_200 = data["EMA50"] > data["EMA200"]
    below_ema200_streak = 0
    in_buy_zone = False

    for i, (idx, row) in enumerate(data.iterrows()):
        price = float(row["Close"])
        is_above = bool(above_ema200.iloc[i])
        ema50_up = bool(ema50_above_200.iloc[i])

        # BUY signal: uptrend confirmed + fundamentals intact
        if is_above and ema50_up and fundamentals_intact and not in_buy_zone:
            buys.append({"Date": idx, "Price": price})
            in_buy_zone = True
            below_ema200_streak = 0

        # Track consecutive days below EMA200
        if not is_above:
            below_ema200_streak += 1
        else:
            below_ema200_streak = 0

        # SELL signal
        if in_buy_zone:
            if below_ema200_streak >= 10:
                sells.append({"Date": idx, "Price": price})
                in_buy_zone = False
                below_ema200_streak = 0

    buy_df = pd.DataFrame(buys) if buys else pd.DataFrame(columns=["Date", "Price"])
    sell_df = pd.DataFrame(sells) if sells else pd.DataFrame(columns=["Date", "Price"])
    return buy_df, sell_df


# ---------------------------------------------------------------------------
# Rick Rule backtest
# ---------------------------------------------------------------------------
def backtest_rick_rule(
    df: pd.DataFrame,
    buy_signals: pd.DataFrame,
    sell_signals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulate Rick Rule buy/sell trades and compute returns.

    Returns DataFrame with columns:
      Action, Date, Price, Return_pct, Cumulative_Return
    """
    if buy_signals.empty:
        return pd.DataFrame(columns=["Action", "Date", "Price", "Return_pct", "Cumulative_Return"])

    trades: List[dict] = []
    buy_list = buy_signals.sort_values("Date").to_dict("records")
    sell_list = sell_signals.sort_values("Date").to_dict("records") if not sell_signals.empty else []

    cumulative = 1.0
    used_sells = set()

    for buy in buy_list:
        buy_date = buy["Date"]
        buy_price = buy["Price"]
        trades.append({
            "Action": "BUY",
            "Date": buy_date,
            "Price": round(buy_price, 2),
            "Return_pct": None,
            "Cumulative_Return": None,
        })
        # Find next sell after this buy
        for j, sell in enumerate(sell_list):
            if j in used_sells:
                continue
            if sell["Date"] > buy_date:
                sell_price = sell["Price"]
                ret = (sell_price - buy_price) / buy_price
                cumulative *= (1 + ret)
                used_sells.add(j)
                trades.append({
                    "Action": "SELL",
                    "Date": sell["Date"],
                    "Price": round(sell_price, 2),
                    "Return_pct": round(ret * 100, 2),
                    "Cumulative_Return": round((cumulative - 1) * 100, 2),
                })
                break

    return pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["Action", "Date", "Price", "Return_pct", "Cumulative_Return"]
    )


# ---------------------------------------------------------------------------
# Master analysis function
# ---------------------------------------------------------------------------
@_cache(ttl=3600)
def run_long_trend_analysis(ticker: str, period: str = "10y") -> dict:
    """
    Full long-trend analysis pipeline for a ticker.

    Returns a dict with keys:
      df              - enriched price DataFrame (Close, EMA50, EMA200, RSI14, Volume_MA20)
      drawdowns       - list of drawdown dicts (classified)
      buy_signals     - DataFrame of Rick Rule BUY signals
      sell_signals    - DataFrame of Rick Rule SELL signals
      backtest        - DataFrame of backtest trades
      trend_phase     - str: "Bullish" / "Neutral" / "Bearish"
      cycle_position  - str: one of 7 cycle phases
      rick_verdict    - str: "BUY zone" / "HOLD" / "SELL zone"
      drawdown_summary - dict: pct noise, fundamental, macro
      error           - str or None
    """
    result = {
        "df": pd.DataFrame(),
        "drawdowns": [],
        "buy_signals": pd.DataFrame(columns=["Date", "Price"]),
        "sell_signals": pd.DataFrame(columns=["Date", "Price"]),
        "backtest": pd.DataFrame(columns=["Action", "Date", "Price", "Return_pct", "Cumulative_Return"]),
        "trend_phase": "Neutral",
        "cycle_position": "Recovery",
        "rick_verdict": "HOLD",
        "drawdown_summary": {"noise_pct": 0, "fundamental_pct": 0, "macro_pct": 0, "sector_pct": 0},
        "error": None,
    }

    # 1. Fetch price data
    df = fetch_long_history(ticker, period)
    if df.empty:
        result["error"] = f"No price data found for {ticker}"
        return result

    # 2. Compute technicals
    df = compute_technicals(df)
    result["df"] = df

    # 3. Resolve Börsdata instrument ID (if available)
    ins_id: Optional[int] = None
    if _BORSDATA_OK and _get_api is not None:
        try:
            api = _get_api()
            ins_id = api.resolve_instrument_id(ticker)
        except Exception:
            pass

    # 4. Detect drawdowns
    drawdowns = detect_drawdowns(df["Close"], min_drop=0.10)

    # 5. Fetch market index for macro classification
    market_close = _fetch_index_history(_MACRO_INDEX, period)
    # Align index to same timezone-naive format
    if not market_close.empty and market_close.index.tz is not None:
        market_close.index = market_close.index.tz_localize(None)

    # 6. Enrich + classify each drawdown
    enriched_drawdowns = []
    for dd in drawdowns:
        dd = enrich_drawdown_fundamentals(dd, ticker, ins_id)
        dd["classification"] = classify_drawdown(dd, market_close, df["Close"])
        enriched_drawdowns.append(dd)

    result["drawdowns"] = enriched_drawdowns

    # 7. Drawdown summary
    if enriched_drawdowns:
        total = len(enriched_drawdowns)
        counts = {
            "Noise/Temporary": 0,
            "Fundamental Deterioration": 0,
            "Macro/Geopolitical": 0,
            "Sector-Wide": 0,
        }
        for dd in enriched_drawdowns:
            c = dd.get("classification", "Noise/Temporary")
            counts[c] = counts.get(c, 0) + 1
        result["drawdown_summary"] = {
            "noise_pct": round(counts["Noise/Temporary"] / total * 100),
            "fundamental_pct": round(counts["Fundamental Deterioration"] / total * 100),
            "macro_pct": round(counts["Macro/Geopolitical"] / total * 100),
            "sector_pct": round(counts["Sector-Wide"] / total * 100),
        }

    # 8. Trend phase
    last_valid = df.dropna(subset=["Close", "EMA50", "EMA200"])
    if not last_valid.empty:
        last = last_valid.iloc[-1]
        result["trend_phase"] = classify_trend_phase(
            float(last["Close"]),
            float(last["EMA50"]),
            float(last["EMA200"]),
        )

    # 9. Cycle position
    result["cycle_position"] = classify_cycle_position(df)

    # 10. Rick Rule signals
    buy_signals, sell_signals = generate_rick_rule_signals(df, ticker, ins_id)
    result["buy_signals"] = buy_signals
    result["sell_signals"] = sell_signals

    # 11. Rick Rule verdict (current)
    if not last_valid.empty:
        last_row = last_valid.iloc[-1]
        price = float(last_row["Close"])
        ema50 = float(last_row["EMA50"])
        ema200 = float(last_row["EMA200"])
        fundamentals_intact = _check_fundamentals_intact(ticker, ins_id)

        below_streak = 0
        close_series = df["Close"].dropna()
        ema200_series = df["EMA200"].dropna()
        for i in range(min(15, len(close_series))):
            idx = -(i + 1)
            if close_series.iloc[idx] < ema200_series.iloc[idx]:
                below_streak += 1
            else:
                break

        if price > ema200 and ema50 > ema200 and fundamentals_intact:
            result["rick_verdict"] = "BUY zone"
        elif below_streak >= 10:
            result["rick_verdict"] = "SELL zone"
        else:
            result["rick_verdict"] = "HOLD"

    # 12. Backtest
    result["backtest"] = backtest_rick_rule(df, buy_signals, sell_signals)

    return result


# ---------------------------------------------------------------------------
# Cycle ordering (for UI bar)
# ---------------------------------------------------------------------------
CYCLE_PHASES = [
    "Accumulation",
    "Early Uptrend",
    "Strong Uptrend",
    "Late Uptrend",
    "Early Downtrend",
    "Capitulation",
    "Recovery",
]


def cycle_phase_index(phase: str) -> int:
    """Return 0-based index of phase in CYCLE_PHASES list."""
    try:
        return CYCLE_PHASES.index(phase)
    except ValueError:
        return 0
