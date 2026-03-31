"""
cagr_fundamentals.py
Fundamental scoring module — 0 to 10 points (upgraded from 6).

Scoring criteria (Börsdata Pro+ data):
  ── Valuation (0-2) ──
  1. EV/EBITDA  < 12          → 1 point
  2. P/B        < 3           → 1 point

  ── Quality (0-3) ──
  3. ROE        > 12%         → 1 point
  4. FCF Stability > 50       → 1 point  (Börsdata KPI 179, scale 0-100)
  5. F-Score    >= 6          → 1 point  (Piotroski, Börsdata KPI 167)

  ── Growth (0-2) ──
  6. Revenue Growth > 5% YoY  → 1 point
  7. Earnings Growth > 5% YoY → 1 point

  ── Financial Health (0-2) ──
  8. Net Debt/EBITDA < 3      → 1 point
  9. Current Ratio   > 1      → 1 point

  ── Insider (0-1) ──
  10. Insider buying (last 6 months) → 1 point

When Börsdata data is unavailable, falls back gracefully to yfinance
approximations with the original 6-point scale.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value, default=None) -> Optional[float]:
    """Convert a value to float safely, returning default on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_pct(val: Optional[float]) -> str:
    """Format a decimal as percentage string."""
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def _fmt_num(val: Optional[float], decimals: int = 2) -> str:
    """Format a number for display."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Legacy helpers (yfinance fallback)
# ---------------------------------------------------------------------------

def _fcf_yield(info: dict) -> Optional[float]:
    """Calculate Free Cash Flow yield = FCF / Market Cap."""
    fcf = _safe_float(info.get("freeCashflow"))
    market_cap = _safe_float(info.get("marketCap"))
    if fcf is not None and market_cap and market_cap > 0:
        return fcf / market_cap
    return _safe_float(info.get("fcf_yield"))


def _roic(info: dict) -> Optional[float]:
    """Get ROIC — prefer Börsdata direct value, fall back to approximation."""
    roic = _safe_float(info.get("roic_approx"))
    if roic is not None:
        return roic

    roce = _safe_float(info.get("returnOnCapitalEmployed"))
    if roce is not None:
        return roce

    ebitda = _safe_float(info.get("ebitda"))
    total_debt = _safe_float(info.get("totalDebt")) or 0.0
    bvps = _safe_float(info.get("bookValue"))
    shares = _safe_float(info.get("sharesOutstanding"))
    if bvps is not None and shares is not None:
        book_value = bvps * shares
    else:
        book_value = None

    if ebitda is not None and book_value is not None:
        invested_capital = total_debt + max(book_value, 1)
        nopat = ebitda * 0.65
        if invested_capital > 0:
            return nopat / invested_capital

    roe = _safe_float(info.get("returnOnEquity"))
    return roe


def _debt_equity(info: dict) -> Optional[float]:
    """Return Debt/Equity ratio, normalised to decimal."""
    de = _safe_float(info.get("de_ratio"))
    if de is not None:
        return de

    de = _safe_float(info.get("debtToEquity"))
    if de is None:
        return None
    if de > 10:
        return de / 100.0
    return de


# ---------------------------------------------------------------------------
# Insider buying check
# ---------------------------------------------------------------------------

def _has_recent_insider_buying(insider_df: Optional[pd.DataFrame], months: int = 6) -> bool:
    """Return True if there is at least one insider buy within `months` months."""
    if insider_df is None or insider_df.empty:
        return False
    try:
        cutoff = datetime.utcnow() - timedelta(days=months * 30)
        df = insider_df.copy()
        date_col = next(
            (c for c in ("Start Date", "Date", "startDate") if c in df.columns),
            None,
        )
        if date_col is None:
            return False

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        recent = df[df[date_col] >= cutoff]

        type_col = next(
            (c for c in ("Transaction", "transaction", "Type") if c in recent.columns),
            None,
        )
        if type_col is None:
            return len(recent) > 0

        buys = recent[recent[type_col].str.contains("Buy|Purchase|buy|purchase", na=False)]
        return len(buys) > 0
    except Exception as exc:
        logger.debug("Insider buy check failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Börsdata-enhanced scoring (10 points)
# ---------------------------------------------------------------------------

def _score_borsdata(ticker_info: dict, insider_df: Optional[pd.DataFrame] = None) -> dict:
    """
    Score using Börsdata Pro+ data — 10 criteria, 10 points max.
    """
    details = {}
    score = 0

    # ── 1. EV/EBITDA < 12 ────────────────────────────────────────────────
    ev_ebitda = _safe_float(ticker_info.get("enterpriseToEbitda"))
    if ev_ebitda is not None:
        passed = ev_ebitda < 12
        details["EV/EBITDA"] = {
            "value": _fmt_num(ev_ebitda),
            "threshold": "< 12",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["EV/EBITDA"] = {"value": "N/A", "threshold": "< 12", "pass": False}

    # ── 2. P/B < 3 ───────────────────────────────────────────────────────
    pb = _safe_float(ticker_info.get("priceToBook"))
    if pb is not None:
        passed = pb < 3
        details["P/B"] = {
            "value": _fmt_num(pb),
            "threshold": "< 3",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["P/B"] = {"value": "N/A", "threshold": "< 3", "pass": False}

    # ── 3. ROE > 12% ─────────────────────────────────────────────────────
    roe = _safe_float(ticker_info.get("returnOnEquity"))
    if roe is not None:
        # Börsdata returns as decimal (0.15 = 15%)
        roe_pct = roe if abs(roe) < 5 else roe / 100
        passed = roe_pct > 0.12
        details["ROE"] = {
            "value": _fmt_pct(roe_pct),
            "threshold": "> 12%",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["ROE"] = {"value": "N/A", "threshold": "> 12%", "pass": False}

    # ── 4. FCF Stability > 50 ────────────────────────────────────────────
    fcf_stab = _safe_float(ticker_info.get("fcf_stability"))
    if fcf_stab is not None:
        passed = fcf_stab > 50
        details["FCF Stability"] = {
            "value": _fmt_num(fcf_stab, 0),
            "threshold": "> 50",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["FCF Stability"] = {"value": "N/A", "threshold": "> 50", "pass": False}

    # ── 5. F-Score >= 6 (Piotroski) ──────────────────────────────────────
    f_score = _safe_float(ticker_info.get("f_score"))
    if f_score is not None:
        passed = f_score >= 6
        details["F-Score"] = {
            "value": _fmt_num(f_score, 0),
            "threshold": ">= 6",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["F-Score"] = {"value": "N/A", "threshold": ">= 6", "pass": False}

    # ── 6. Revenue Growth > 5% ───────────────────────────────────────────
    rev_growth = _safe_float(ticker_info.get("revenue_growth"))
    if rev_growth is not None:
        rev_g = rev_growth if abs(rev_growth) < 5 else rev_growth / 100
        passed = rev_g > 0.05
        details["Revenue Growth"] = {
            "value": _fmt_pct(rev_g),
            "threshold": "> 5%",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Revenue Growth"] = {"value": "N/A", "threshold": "> 5%", "pass": False}

    # ── 7. Earnings Growth > 5% ──────────────────────────────────────────
    earn_growth = _safe_float(ticker_info.get("earnings_growth"))
    if earn_growth is not None:
        earn_g = earn_growth if abs(earn_growth) < 5 else earn_growth / 100
        passed = earn_g > 0.05
        details["Earnings Growth"] = {
            "value": _fmt_pct(earn_g),
            "threshold": "> 5%",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Earnings Growth"] = {"value": "N/A", "threshold": "> 5%", "pass": False}

    # ── 8. Net Debt/EBITDA < 3 ───────────────────────────────────────────
    nd_ebitda = _safe_float(ticker_info.get("net_debt_ebitda"))
    if nd_ebitda is not None:
        passed = nd_ebitda < 3
        details["Net Debt/EBITDA"] = {
            "value": _fmt_num(nd_ebitda),
            "threshold": "< 3",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Net Debt/EBITDA"] = {"value": "N/A", "threshold": "< 3", "pass": False}

    # ── 9. Current Ratio > 1 ─────────────────────────────────────────────
    cr = _safe_float(ticker_info.get("current_ratio"))
    if cr is not None:
        passed = cr > 1.0
        details["Current Ratio"] = {
            "value": _fmt_num(cr),
            "threshold": "> 1.0",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Current Ratio"] = {"value": "N/A", "threshold": "> 1.0", "pass": False}

    # ── 10. Insider buying ───────────────────────────────────────────────
    insider_buy = ticker_info.get("insider_buying", False)
    if not insider_buy and insider_df is not None:
        insider_buy = _has_recent_insider_buying(insider_df, months=6)
    details["Insider Buying"] = {
        "value": "Yes" if insider_buy else "No/N/A",
        "threshold": "Any buy in 6m",
        "pass": insider_buy,
    }
    if insider_buy:
        score += 1

    return {
        "fund_score": score,
        "fund_max": 10,
        "details": details,
        "_data_source": "borsdata",
    }


# ---------------------------------------------------------------------------
# yfinance fallback scoring (6 points — original)
# ---------------------------------------------------------------------------

def _score_yfinance(ticker_info: dict, insider_df: Optional[pd.DataFrame] = None) -> dict:
    """
    Score using yfinance data — 6 criteria (original scale).
    """
    details = {}
    score = 0

    # 1. EV/EBITDA < 8
    ev_ebitda = _safe_float(ticker_info.get("enterpriseToEbitda"))
    if ev_ebitda is not None:
        passed = ev_ebitda < 8
        details["EV/EBITDA"] = {"value": _fmt_num(ev_ebitda), "threshold": "< 8", "pass": passed}
        if passed:
            score += 1
    else:
        details["EV/EBITDA"] = {"value": "N/A", "threshold": "< 8", "pass": False}

    # 2. P/B < 1.5
    pb = _safe_float(ticker_info.get("priceToBook"))
    if pb is not None:
        passed = pb < 1.5
        details["P/B"] = {"value": _fmt_num(pb), "threshold": "< 1.5", "pass": passed}
        if passed:
            score += 1
    else:
        details["P/B"] = {"value": "N/A", "threshold": "< 1.5", "pass": False}

    # 3. FCF Yield > 8%
    fcf_yield = _fcf_yield(ticker_info)
    if fcf_yield is not None:
        passed = fcf_yield > 0.08
        details["FCF Yield"] = {"value": _fmt_pct(fcf_yield), "threshold": "> 8%", "pass": passed}
        if passed:
            score += 1
    else:
        details["FCF Yield"] = {"value": "N/A", "threshold": "> 8%", "pass": False}

    # 4. ROIC > 10%
    roic = _roic(ticker_info)
    if roic is not None:
        passed = roic > 0.10
        details["ROIC"] = {"value": _fmt_pct(roic), "threshold": "> 10%", "pass": passed}
        if passed:
            score += 1
    else:
        details["ROIC"] = {"value": "N/A", "threshold": "> 10%", "pass": False}

    # 5. Debt/Equity < 0.5
    de = _debt_equity(ticker_info)
    if de is not None:
        passed = de < 0.5
        details["Debt/Equity"] = {"value": _fmt_num(de), "threshold": "< 0.5", "pass": passed}
        if passed:
            score += 1
    else:
        details["Debt/Equity"] = {"value": "N/A", "threshold": "< 0.5", "pass": False}

    # 6. Insider buying
    insider_buy = ticker_info.get("insider_buying", False)
    if not insider_buy and insider_df is not None:
        insider_buy = _has_recent_insider_buying(insider_df, months=6)
    details["Insider Buying"] = {
        "value": "Yes" if insider_buy else "No/N/A",
        "threshold": "Any buy in 6m",
        "pass": insider_buy,
    }
    if insider_buy:
        score += 1

    return {
        "fund_score": score,
        "fund_max": 6,
        "details": details,
        "_data_source": "yfinance",
    }


# ---------------------------------------------------------------------------
# Main scoring function (auto-selects based on data source)
# ---------------------------------------------------------------------------

def score_fundamentals(
    ticker_info: dict,
    insider_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Score a stock on fundamental criteria.

    Automatically selects the scoring model based on data source:
      • Börsdata data → 10-point model (richer criteria)
      • yfinance data → 6-point model (original criteria)

    The output always includes:
      - fund_score : int
      - fund_max   : int (10 or 6 — tells scoring layer the scale)
      - details    : dict with criterion name → {value, threshold, pass}
      - _data_source : str ("borsdata" or "yfinance")
    """
    data_source = ticker_info.get("_data_source", "yfinance")

    if data_source == "borsdata":
        return _score_borsdata(ticker_info, insider_df)
    else:
        return _score_yfinance(ticker_info, insider_df)
