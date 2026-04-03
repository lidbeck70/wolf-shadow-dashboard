"""
cagr_fundamentals.py
Fundamental scoring module — 0 to 20 points (Börsdata) / 6 points (yfinance fallback).

Scoring criteria (Börsdata Pro+ data) — 20 points:

  ── Earnings Acceleration (0-4) ──
  1. Revenue Growth > 15% YoY         → 1 point
  2. Earnings Growth > 20% YoY        → 1 point
  3. Revenue Growth > Earnings Growth  → 1 point  (real-sales growth, not cost-cutting)
  4. Revenue CAGR 5y > 10%            → 1 point

  ── Quality (0-4) ──
  5. ROE > 17%                        → 1 point
  6. FCF Stability > 70               → 1 point  (Börsdata KPI 179, scale 0-100)
  7. F-Score >= 7                     → 1 point  (Piotroski, Börsdata KPI 167)
  8. Operating Margin > 15%           → 1 point

  ── Valuation vs Growth (0-3) ──
  9.  EV/EBITDA < 15                  → 1 point
  10. P/B < 4                         → 1 point
  11. PEG-like: PE / (earnings_growth*100) < 1.5 → 1 point

  ── Relative Strength (0-3) ──
  12. RS Rank >= 80 (top 20%)         → 1 point  (Börsdata 1-100, 100 = best)
  13. Momentum: RS Rank >= 50         → 1 point  (proxy for 6M positive momentum)
  14. Magic Formula rank <= 30        → 1 point  (Börsdata rank, lower = better)

  ── Financial Health (0-3) ──
  15. Net Debt/EBITDA < 2.5           → 1 point
  16. Current Ratio > 1.2             → 1 point
  17. Equity Ratio > 30%              → 1 point

  ── Cycle & Insider (0-3) ──
  18. Earnings Stability > 60         → 1 point  (scale 0-100)
  19. Dividend Yield > 1%             → 1 point  (signals cash generation)
  20. Insider buying (last 6 months)  → 1 point

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


def _normalize_pct(val: Optional[float]) -> Optional[float]:
    """
    Normalize a value that might be a decimal (0.15) or a percentage (15.0).
    If abs(val) > 5, assume it's expressed as a percentage and divide by 100.
    """
    if val is None:
        return None
    return val if abs(val) <= 5 else val / 100.0


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
# Börsdata-enhanced scoring (20 points)
# ---------------------------------------------------------------------------

def _score_borsdata(ticker_info: dict, insider_df: Optional[pd.DataFrame] = None) -> dict:
    """
    Score using Börsdata Pro+ data — 20 criteria, 20 points max.
    """
    details = {}
    score = 0

    # ── Earnings Acceleration (0-4) ─────────────────────────────────────────

    # 1. Revenue Growth > 15%
    rev_growth = _safe_float(ticker_info.get("revenue_growth"))
    rev_g = _normalize_pct(rev_growth)
    if rev_g is not None:
        passed = rev_g > 0.15
        details["Revenue Growth"] = {
            "value": _fmt_pct(rev_g),
            "threshold": "> 15%",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Revenue Growth"] = {"value": "N/A", "threshold": "> 15%", "pass": False}

    # 2. Earnings Growth > 20%
    earn_growth = _safe_float(ticker_info.get("earnings_growth"))
    earn_g = _normalize_pct(earn_growth)
    if earn_g is not None:
        passed = earn_g > 0.20
        details["Earnings Growth"] = {
            "value": _fmt_pct(earn_g),
            "threshold": "> 20%",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Earnings Growth"] = {"value": "N/A", "threshold": "> 20%", "pass": False}

    # 3. Revenue Growth > Earnings Growth (real-sales-driven growth)
    if rev_g is not None and earn_g is not None:
        passed = rev_g > earn_g
        details["Rev > Earn Growth"] = {
            "value": f"{_fmt_pct(rev_g)} vs {_fmt_pct(earn_g)}",
            "threshold": "Revenue > Earnings growth",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Rev > Earn Growth"] = {
            "value": "N/A",
            "threshold": "Revenue > Earnings growth",
            "pass": False,
        }

    # 4. Revenue CAGR 5y > 10%
    rev_cagr_5y = _safe_float(ticker_info.get("revenue_cagr_5y"))
    cagr_5y = _normalize_pct(rev_cagr_5y)
    if cagr_5y is not None:
        passed = cagr_5y > 0.10
        details["Revenue CAGR 5y"] = {
            "value": _fmt_pct(cagr_5y),
            "threshold": "> 10%",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Revenue CAGR 5y"] = {"value": "N/A", "threshold": "> 10%", "pass": False}

    # ── Quality (0-4) ───────────────────────────────────────────────────────

    # 5. ROE > 17%
    roe = _safe_float(ticker_info.get("returnOnEquity"))
    roe_pct = _normalize_pct(roe)
    if roe_pct is not None:
        passed = roe_pct > 0.17
        details["ROE"] = {
            "value": _fmt_pct(roe_pct),
            "threshold": "> 17%",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["ROE"] = {"value": "N/A", "threshold": "> 17%", "pass": False}

    # 6. FCF Stability > 70
    fcf_stab = _safe_float(ticker_info.get("fcf_stability"))
    if fcf_stab is not None:
        passed = fcf_stab > 70
        details["FCF Stability"] = {
            "value": _fmt_num(fcf_stab, 0),
            "threshold": "> 70",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["FCF Stability"] = {"value": "N/A", "threshold": "> 70", "pass": False}

    # 7. F-Score >= 7 (Piotroski)
    f_score = _safe_float(ticker_info.get("f_score"))
    if f_score is not None:
        passed = f_score >= 7
        details["F-Score"] = {
            "value": _fmt_num(f_score, 0),
            "threshold": ">= 7",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["F-Score"] = {"value": "N/A", "threshold": ">= 7", "pass": False}

    # 8. Operating Margin > 15%
    op_margin = _safe_float(ticker_info.get("operating_margin"))
    op_m = _normalize_pct(op_margin)
    if op_m is not None:
        passed = op_m > 0.15
        details["Operating Margin"] = {
            "value": _fmt_pct(op_m),
            "threshold": "> 15%",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Operating Margin"] = {"value": "N/A", "threshold": "> 15%", "pass": False}

    # ── Valuation vs Growth (0-3) ────────────────────────────────────────────

    # 9. EV/EBITDA < 15
    ev_ebitda = _safe_float(ticker_info.get("enterpriseToEbitda"))
    if ev_ebitda is not None:
        passed = ev_ebitda < 15
        details["EV/EBITDA"] = {
            "value": _fmt_num(ev_ebitda),
            "threshold": "< 15",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["EV/EBITDA"] = {"value": "N/A", "threshold": "< 15", "pass": False}

    # 10. P/B < 4
    pb = _safe_float(ticker_info.get("priceToBook"))
    if pb is not None:
        passed = pb < 4
        details["P/B"] = {
            "value": _fmt_num(pb),
            "threshold": "< 4",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["P/B"] = {"value": "N/A", "threshold": "< 4", "pass": False}

    # 11. PEG-like: PE / (earnings_growth * 100) < 1.5
    pe = _safe_float(ticker_info.get("trailingPE") or ticker_info.get("forwardPE"))
    if pe is not None and earn_g is not None and earn_g > 0:
        peg_approx = pe / (earn_g * 100)
        passed = peg_approx < 1.5
        details["PEG (approx)"] = {
            "value": _fmt_num(peg_approx),
            "threshold": "< 1.5",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["PEG (approx)"] = {"value": "N/A", "threshold": "< 1.5", "pass": False}

    # ── Relative Strength (0-3) ──────────────────────────────────────────────

    # 12. RS Rank >= 80 (top 20%; Börsdata scale: 100 = best)
    rs_rank = _safe_float(ticker_info.get("rs_rank"))
    if rs_rank is not None:
        passed = rs_rank >= 80
        details["RS Rank"] = {
            "value": _fmt_num(rs_rank, 0),
            "threshold": ">= 80 (top 20%)",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["RS Rank"] = {"value": "N/A", "threshold": ">= 80 (top 20%)", "pass": False}

    # 13. Momentum proxy: RS Rank >= 50
    if rs_rank is not None:
        passed = rs_rank >= 50
        details["Momentum (RS >= 50)"] = {
            "value": _fmt_num(rs_rank, 0),
            "threshold": ">= 50",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Momentum (RS >= 50)"] = {"value": "N/A", "threshold": ">= 50", "pass": False}

    # 14. Magic Formula rank <= 30 (Börsdata rank, lower = better)
    magic_formula = _safe_float(ticker_info.get("magic_formula"))
    if magic_formula is not None:
        passed = magic_formula <= 30
        details["Magic Formula"] = {
            "value": _fmt_num(magic_formula, 0),
            "threshold": "<= 30 (top 30%)",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Magic Formula"] = {"value": "N/A", "threshold": "<= 30 (top 30%)", "pass": False}

    # ── Financial Health (0-3) ───────────────────────────────────────────────

    # 15. Net Debt/EBITDA < 2.5
    nd_ebitda = _safe_float(ticker_info.get("net_debt_ebitda"))
    if nd_ebitda is not None:
        passed = nd_ebitda < 2.5
        details["Net Debt/EBITDA"] = {
            "value": _fmt_num(nd_ebitda),
            "threshold": "< 2.5",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Net Debt/EBITDA"] = {"value": "N/A", "threshold": "< 2.5", "pass": False}

    # 16. Current Ratio > 1.2
    cr = _safe_float(ticker_info.get("current_ratio"))
    if cr is not None:
        passed = cr > 1.2
        details["Current Ratio"] = {
            "value": _fmt_num(cr),
            "threshold": "> 1.2",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Current Ratio"] = {"value": "N/A", "threshold": "> 1.2", "pass": False}

    # 17. Equity Ratio > 30%
    eq_ratio = _safe_float(ticker_info.get("equity_ratio"))
    eq_r = _normalize_pct(eq_ratio)
    if eq_r is not None:
        passed = eq_r > 0.30
        details["Equity Ratio"] = {
            "value": _fmt_pct(eq_r),
            "threshold": "> 30%",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Equity Ratio"] = {"value": "N/A", "threshold": "> 30%", "pass": False}

    # ── Cycle & Insider (0-3) ────────────────────────────────────────────────

    # 18. Earnings Stability > 60
    earn_stab = _safe_float(ticker_info.get("earnings_stability"))
    if earn_stab is not None:
        passed = earn_stab > 60
        details["Earnings Stability"] = {
            "value": _fmt_num(earn_stab, 0),
            "threshold": "> 60",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Earnings Stability"] = {"value": "N/A", "threshold": "> 60", "pass": False}

    # 19. Dividend Yield > 1%
    div_yield = _safe_float(ticker_info.get("dividend_yield"))
    div_y = _normalize_pct(div_yield)
    if div_y is not None:
        passed = div_y > 0.01
        details["Dividend Yield"] = {
            "value": _fmt_pct(div_y),
            "threshold": "> 1%",
            "pass": passed,
        }
        if passed:
            score += 1
    else:
        details["Dividend Yield"] = {"value": "N/A", "threshold": "> 1%", "pass": False}

    # 20. Insider buying (last 6 months)
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
        "fund_max": 20,
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
      • Börsdata data → 20-point model (richer criteria)
      • yfinance data → 6-point model (original criteria)

    The output always includes:
      - fund_score : int
      - fund_max   : int (20 or 6 — tells scoring layer the scale)
      - details    : dict with criterion name → {value, threshold, pass}
      - _data_source : str ("borsdata" or "yfinance")
    """
    data_source = ticker_info.get("_data_source", "yfinance")

    if data_source == "borsdata":
        return _score_borsdata(ticker_info, insider_df)
    else:
        return _score_yfinance(ticker_info, insider_df)
