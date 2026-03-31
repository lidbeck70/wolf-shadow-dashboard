"""
cagr_loader.py
Data loader for Nordic stocks and UCITS ETFs.

Supports two data sources:
  1. Börsdata API (primary) — activated when BORSDATA_API_KEY is set.
     Provides 20-year fundamental history, 200+ KPIs, report data.
  2. yfinance (fallback) — used when Börsdata key is absent or for
     tickers not found in Börsdata (ETFs, US stocks).

All public functions are cached with st.cache_data (TTL 3600 s) when
Streamlit is available; otherwise a simple in-process joblib memory cache
is used as a fallback.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Streamlit import (dashboard context)
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    st = None  # type: ignore[assignment]
    _STREAMLIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Optional joblib cache (CLI / script context)
# ---------------------------------------------------------------------------
try:
    from joblib import Memory as _JoblibMemory
    _joblib_memory = _JoblibMemory(location=os.path.join(os.path.dirname(__file__), ".cache"), verbose=0)
    _JOBLIB_AVAILABLE = True
except ImportError:
    _joblib_memory = None  # type: ignore[assignment]
    _JOBLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Börsdata API import
# ---------------------------------------------------------------------------
_BORSDATA_MODULE = False
_get_borsdata_api = None  # type: ignore

# Try multiple import paths to work in all contexts:
#   1. Absolute: "from dashboard.borsdata_api" (pytest / workspace root)
#   2. Relative: "from borsdata_api" (Streamlit Cloud runs from dashboard/)
#   3. sys.path fallback (edge cases)

for _import_attempt in range(1):
    try:
        from dashboard.borsdata_api import BorsdataAPI, get_api as _get_borsdata_api, KPI
        _BORSDATA_MODULE = True
        break
    except ImportError:
        pass
    try:
        from borsdata_api import BorsdataAPI, get_api as _get_borsdata_api, KPI
        _BORSDATA_MODULE = True
        break
    except ImportError:
        pass
    try:
        import sys as _sys
        _parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if _parent not in _sys.path:
            _sys.path.insert(0, _parent)
        from borsdata_api import BorsdataAPI, get_api as _get_borsdata_api, KPI
        _BORSDATA_MODULE = True
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Cache decorator factory
# ---------------------------------------------------------------------------

def _cache(ttl: int = 3600):
    """Return a cache decorator appropriate for the current runtime."""
    def decorator(fn):
        if _STREAMLIT_AVAILABLE:
            return st.cache_data(ttl=ttl)(fn)
        if _JOBLIB_AVAILABLE and _joblib_memory is not None:
            return _joblib_memory.cache(fn)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Data source detection
# ---------------------------------------------------------------------------

def _borsdata_available() -> bool:
    """Check if Börsdata API is configured and ready."""
    if not _BORSDATA_MODULE:
        return False
    try:
        api = _get_borsdata_api()
        return api.is_configured
    except Exception:
        return False


def get_data_source() -> str:
    """Return current data source name for UI display."""
    return "Börsdata Pro+" if _borsdata_available() else "yfinance"


# ---------------------------------------------------------------------------
# Ticker registries
# ---------------------------------------------------------------------------

NORDIC_TICKERS: Dict[str, dict] = {
    # ── Sweden (OMXS30 components & large caps) ──────────────────────────
    "VOLV-B.ST": {
        "name": "Volvo B", "country": "Sweden",
        "sector": "Industrials", "exchange": "OMXS",
    },
    "AZN.ST": {
        "name": "AstraZeneca", "country": "Sweden",
        "sector": "Healthcare", "exchange": "OMXS",
    },
    "SAND.ST": {
        "name": "Sandvik", "country": "Sweden",
        "sector": "Industrials", "exchange": "OMXS",
    },
    "SEB-A.ST": {
        "name": "SEB A", "country": "Sweden",
        "sector": "Financials", "exchange": "OMXS",
    },
    "HEXA-B.ST": {
        "name": "Hexagon B", "country": "Sweden",
        "sector": "Technology", "exchange": "OMXS",
    },
    "ERIC-B.ST": {
        "name": "Ericsson B", "country": "Sweden",
        "sector": "Technology", "exchange": "OMXS",
    },
    "ATCO-A.ST": {
        "name": "Atlas Copco A", "country": "Sweden",
        "sector": "Industrials", "exchange": "OMXS",
    },
    "ABB.ST": {
        "name": "ABB", "country": "Sweden",
        "sector": "Industrials", "exchange": "OMXS",
    },
    "INVE-B.ST": {
        "name": "Investor B", "country": "Sweden",
        "sector": "Financials", "exchange": "OMXS",
    },
    "ASSA-B.ST": {
        "name": "Assa Abloy B", "country": "Sweden",
        "sector": "Industrials", "exchange": "OMXS",
    },
    "ESSITY-B.ST": {
        "name": "Essity B", "country": "Sweden",
        "sector": "Consumer Staples", "exchange": "OMXS",
    },
    "HM-B.ST": {
        "name": "H&M B", "country": "Sweden",
        "sector": "Consumer Discretionary", "exchange": "OMXS",
    },
    "SWED-A.ST": {
        "name": "Swedbank A", "country": "Sweden",
        "sector": "Financials", "exchange": "OMXS",
    },
    "SHB-A.ST": {
        "name": "Handelsbanken A", "country": "Sweden",
        "sector": "Financials", "exchange": "OMXS",
    },
    "ALFA.ST": {
        "name": "Alfa Laval", "country": "Sweden",
        "sector": "Industrials", "exchange": "OMXS",
    },
    "SKF-B.ST": {
        "name": "SKF B", "country": "Sweden",
        "sector": "Industrials", "exchange": "OMXS",
    },
    "EVO.ST": {
        "name": "Evolution", "country": "Sweden",
        "sector": "Consumer Discretionary", "exchange": "OMXS",
    },
    "NIBE-B.ST": {
        "name": "NIBE B", "country": "Sweden",
        "sector": "Industrials", "exchange": "OMXS",
    },
    "SINCH.ST": {
        "name": "Sinch", "country": "Sweden",
        "sector": "Technology", "exchange": "OMXS",
    },
    "BOL.ST": {
        "name": "Boliden", "country": "Sweden",
        "sector": "Materials", "exchange": "OMXS",
    },
    # ── Norway (OBX components) ──────────────────────────────────────────
    "EQNR.OL": {
        "name": "Equinor", "country": "Norway",
        "sector": "Energy", "exchange": "OSE",
    },
    "DNB.OL": {
        "name": "DNB Bank", "country": "Norway",
        "sector": "Financials", "exchange": "OSE",
    },
    "NHY.OL": {
        "name": "Norsk Hydro", "country": "Norway",
        "sector": "Materials", "exchange": "OSE",
    },
    "MOWI.OL": {
        "name": "Mowi", "country": "Norway",
        "sector": "Consumer Staples", "exchange": "OSE",
    },
    "AKRBP.OL": {
        "name": "Aker BP", "country": "Norway",
        "sector": "Energy", "exchange": "OSE",
    },
    "TEL.OL": {
        "name": "Telenor", "country": "Norway",
        "sector": "Communication Services", "exchange": "OSE",
    },
    "SUBC.OL": {
        "name": "Subsea 7", "country": "Norway",
        "sector": "Energy", "exchange": "OSE",
    },
    "SGSN.OL": {
        "name": "Storebrand", "country": "Norway",
        "sector": "Financials", "exchange": "OSE",
    },
    "SALM.OL": {
        "name": "SalMar", "country": "Norway",
        "sector": "Consumer Staples", "exchange": "OSE",
    },
    "BAKKA.OL": {
        "name": "Bakkafrost", "country": "Norway",
        "sector": "Consumer Staples", "exchange": "OSE",
    },
    # ── Denmark ─────────────────────────────────────────────────────────
    "NOVO-B.CO": {
        "name": "Novo Nordisk B", "country": "Denmark",
        "sector": "Healthcare", "exchange": "CSE",
    },
    "DSV.CO": {
        "name": "DSV", "country": "Denmark",
        "sector": "Industrials", "exchange": "CSE",
    },
    "MAERSK-B.CO": {
        "name": "Maersk B", "country": "Denmark",
        "sector": "Industrials", "exchange": "CSE",
    },
    "CARL-B.CO": {
        "name": "Carlsberg B", "country": "Denmark",
        "sector": "Consumer Staples", "exchange": "CSE",
    },
    "VWS.CO": {
        "name": "Vestas Wind", "country": "Denmark",
        "sector": "Energy", "exchange": "CSE",
    },
    "ORSTED.CO": {
        "name": "Ørsted", "country": "Denmark",
        "sector": "Utilities", "exchange": "CSE",
    },
    "NZYM-B.CO": {
        "name": "Novozymes B", "country": "Denmark",
        "sector": "Materials", "exchange": "CSE",
    },
    "GMAB.CO": {
        "name": "Genmab", "country": "Denmark",
        "sector": "Healthcare", "exchange": "CSE",
    },
    "TRYG.CO": {
        "name": "Tryg", "country": "Denmark",
        "sector": "Financials", "exchange": "CSE",
    },
    "COLOB.CO": {
        "name": "Coloplast B", "country": "Denmark",
        "sector": "Healthcare", "exchange": "CSE",
    },
    # ── Finland ─────────────────────────────────────────────────────────
    "NOKIA.HE": {
        "name": "Nokia", "country": "Finland",
        "sector": "Technology", "exchange": "HSE",
    },
    "SAMPO.HE": {
        "name": "Sampo", "country": "Finland",
        "sector": "Financials", "exchange": "HSE",
    },
    "NESTE.HE": {
        "name": "Neste", "country": "Finland",
        "sector": "Energy", "exchange": "HSE",
    },
    "UPM.HE": {
        "name": "UPM-Kymmene", "country": "Finland",
        "sector": "Materials", "exchange": "HSE",
    },
    "FORTUM.HE": {
        "name": "Fortum", "country": "Finland",
        "sector": "Utilities", "exchange": "HSE",
    },
    "KNEBV.HE": {
        "name": "Kone", "country": "Finland",
        "sector": "Industrials", "exchange": "HSE",
    },
    "STERV.HE": {
        "name": "Stora Enso R", "country": "Finland",
        "sector": "Materials", "exchange": "HSE",
    },
    "METSO.HE": {
        "name": "Metso", "country": "Finland",
        "sector": "Industrials", "exchange": "HSE",
    },
    "KESKOB.HE": {
        "name": "Kesko B", "country": "Finland",
        "sector": "Consumer Staples", "exchange": "HSE",
    },
    "WRT1V.HE": {
        "name": "Wärtsilä", "country": "Finland",
        "sector": "Industrials", "exchange": "HSE",
    },
}

ETF_TICKERS: Dict[str, dict] = {
    "IWDA.AS": {
        "name": "iShares Core MSCI World UCITS ETF",
        "country": "IE", "sector": "ETF", "exchange": "AEX",
        "category": "Global Equity",
    },
    "IEMA.AS": {
        "name": "iShares MSCI EM UCITS ETF",
        "country": "IE", "sector": "ETF", "exchange": "AEX",
        "category": "Emerging Markets",
    },
    "XDWD.DE": {
        "name": "Xtrackers MSCI World Swap UCITS ETF",
        "country": "LU", "sector": "ETF", "exchange": "XETRA",
        "category": "Global Equity",
    },
    "CSPX.AS": {
        "name": "iShares Core S&P 500 UCITS ETF",
        "country": "IE", "sector": "ETF", "exchange": "AEX",
        "category": "US Equity",
    },
    "MEUD.PA": {
        "name": "Lyxor Core STOXX 600 UCITS ETF",
        "country": "FR", "sector": "ETF", "exchange": "EPA",
        "category": "European Equity",
    },
    "EUNL.DE": {
        "name": "iShares Core MSCI World UCITS ETF (EUR)",
        "country": "IE", "sector": "ETF", "exchange": "XETRA",
        "category": "Global Equity",
    },
    "VWRL.AS": {
        "name": "Vanguard FTSE All-World UCITS ETF",
        "country": "IE", "sector": "ETF", "exchange": "AEX",
        "category": "Global Equity",
    },
    "IUSQ.DE": {
        "name": "iShares MSCI ACWI UCITS ETF",
        "country": "IE", "sector": "ETF", "exchange": "XETRA",
        "category": "Global Equity",
    },
    "VUSA.AS": {
        "name": "Vanguard S&P 500 UCITS ETF",
        "country": "IE", "sector": "ETF", "exchange": "AEX",
        "category": "US Equity",
    },
    "IQQQ.DE": {
        "name": "iShares Nasdaq 100 UCITS ETF",
        "country": "IE", "sector": "ETF", "exchange": "XETRA",
        "category": "US Technology",
    },
    "SXRV.DE": {
        "name": "iShares Core MSCI Europe UCITS ETF",
        "country": "IE", "sector": "ETF", "exchange": "XETRA",
        "category": "European Equity",
    },
    "SADM.DE": {
        "name": "iShares MSCI EM Asia UCITS ETF",
        "country": "IE", "sector": "ETF", "exchange": "XETRA",
        "category": "EM Asia",
    },
    "XGLE.DE": {
        "name": "Xtrackers MSCI Eurozone Swap UCITS ETF",
        "country": "LU", "sector": "ETF", "exchange": "XETRA",
        "category": "Eurozone Equity",
    },
    "IQQH.DE": {
        "name": "iShares Global Clean Energy UCITS ETF",
        "country": "IE", "sector": "ETF", "exchange": "XETRA",
        "category": "Thematic",
    },
    "BTCE.DE": {
        "name": "ETC Group Physical Bitcoin ETP",
        "country": "DE", "sector": "ETF", "exchange": "XETRA",
        "category": "Crypto",
    },
}


# ---------------------------------------------------------------------------
# Public API — ticker registry
# ---------------------------------------------------------------------------

def load_nordic_tickers() -> Dict[str, dict]:
    """Return registry of Nordic stock tickers with metadata."""
    return dict(NORDIC_TICKERS)


def load_etf_tickers() -> Dict[str, dict]:
    """Return registry of UCITS ETF tickers with metadata."""
    return dict(ETF_TICKERS)


def load_all_tickers() -> Dict[str, dict]:
    """Return combined Nordic stocks + UCITS ETF ticker dict."""
    combined = {}
    combined.update(NORDIC_TICKERS)
    combined.update(ETF_TICKERS)
    return combined


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value, default=None) -> Optional[float]:
    """Convert value to float safely; return default on failure."""
    if value is None:
        return default
    try:
        f = float(value)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _get_close_col(df: pd.DataFrame) -> Optional[str]:
    """Return the name of the Close column in a DataFrame."""
    for candidate in ("Close", "Adj Close", "close", "adj close"):
        if candidate in df.columns:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Price data (always via yfinance — Börsdata is end-of-day only)
# ---------------------------------------------------------------------------

@_cache(ttl=3600)
def fetch_price_data(
    tickers: list,
    period: str = "2y",
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data for a list of tickers via yfinance.

    Returns dict ticker → DataFrame(Date index, Open, High, Low, Close, Volume).
    """
    result: Dict[str, pd.DataFrame] = {}

    try:
        raw = yf.download(
            tickers=tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.warning("Batch download failed (%s); falling back to individual.", exc)
        raw = None

    if raw is not None and not raw.empty:
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    df = raw.xs(ticker, level=1, axis=1).copy()
                    df = df.dropna(how="all")
                    df.index.name = "Date"
                    result[ticker] = df if not df.empty else pd.DataFrame()
                except Exception:
                    result[ticker] = pd.DataFrame()
        else:
            if len(tickers) == 1:
                df = raw.copy()
                df.index.name = "Date"
                result[tickers[0]] = df if not df.empty else pd.DataFrame()

    missing = [t for t in tickers if t not in result or result[t].empty]
    for ticker in missing:
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(period=period, auto_adjust=True)
            df = df.dropna(how="all")
            df.index.name = "Date"
            result[ticker] = df if not df.empty else pd.DataFrame()
        except Exception as exc:
            logger.warning("fetch_price_data(%s): %s", ticker, exc)
            result[ticker] = pd.DataFrame()

    return result


# ---------------------------------------------------------------------------
# Börsdata fundamentals extraction
# ---------------------------------------------------------------------------

def _extract_borsdata_fundamentals(ticker: str) -> dict:
    """
    Pull fundamental data from Börsdata API.

    Returns a normalised dict compatible with the scoring layer.
    Falls back to empty defaults if the ticker can't be resolved.
    """
    data = _make_empty_fundamentals(ticker)

    if not _borsdata_available():
        return data

    api = _get_borsdata_api()
    ins_id = api.resolve_instrument_id(ticker)

    if ins_id is None:
        logger.info(
            "Börsdata: could not resolve instrument ID for %s, falling back to yfinance.",
            ticker,
        )
        return data

    data["_data_source"] = "borsdata"
    data["_borsdata_ins_id"] = ins_id

    # Get instrument metadata
    inst_info = api.get_instrument_info(ins_id)
    if inst_info:
        data["shortName"] = inst_info.get("name", ticker)

    # ── Fetch key KPIs via screener (single-instrument KPI history) ──────
    kpi_mapping = {
        # output_key:         (kpi_id, report_type, divisor)
        "enterpriseToEbitda": (KPI["ev_ebitda"], "r12", 1),
        "priceToBook":        (KPI["pb"], "r12", 1),
        "trailingPE":         (KPI["pe"], "r12", 1),
        "returnOnEquity":     (KPI["roe"], "r12", 0.01),     # % → decimal
        "returnOnAssets":     (KPI["roa"], "r12", 0.01),
        "roic_approx":        (KPI["roic"], "r12", 0.01),
        "de_ratio":           (KPI["debt_to_equity"], "r12", 0.01),
        "equity_ratio":       (KPI["equity_ratio"], "r12", 0.01),
        "gross_margin":       (KPI["gross_margin"], "r12", 0.01),
        "operating_margin":   (KPI["operating_margin"], "r12", 0.01),
        "profit_margin":      (KPI["profit_margin"], "r12", 0.01),
        "fcf_margin":         (KPI["fcf_margin"], "r12", 0.01),
        "ebitda_margin":      (KPI["ebitda_margin"], "r12", 0.01),
        "net_debt_ebitda":    (KPI["net_debt_ebitda"], "r12", 1),
        "current_ratio":      (KPI["current_ratio"], "r12", 1),
        "dividend_yield":     (KPI["dividend_yield"], "r12", 0.01),
        "revenue_growth":     (KPI["revenue_growth"], "r12", 0.01),
        "earnings_growth":    (KPI["earnings_growth"], "r12", 0.01),
        "earnings_stability": (KPI["earnings_stab"], "r12", 1),
        "fcf_stability":      (KPI["fcf_stab"], "r12", 1),
        "f_score":            (KPI["f_score"], "r12", 1),
        "magic_formula":      (KPI["magic_formula"], "r12", 1),
        "rs_rank":            (KPI["rs_rank"], "r12", 1),
    }

    for out_key, (kpi_id, rtype, divisor) in kpi_mapping.items():
        try:
            hist = api.get_kpi_history(ins_id, kpi_id, rtype, "mean")
            if hist:
                val = hist[-1].get("v")
                if val is not None:
                    data[out_key] = val * divisor if divisor != 1 else val
        except Exception as exc:
            logger.debug("Börsdata KPI %s for %s: %s", out_key, ticker, exc)

    # ── Absolute values from reports ─────────────────────────────────────
    try:
        reports = api.get_reports(ins_id, "r12", max_count=2)
        if reports:
            latest = reports[-1]
            data["freeCashflow"] = _safe_float(latest.get("freeCashFlow"))
            data["totalRevenue"] = _safe_float(latest.get("revenues") or latest.get("netSales"))
            data["ebitda"] = None  # Not directly in reports, use margins
            data["totalDebt"] = None
            data["totalCash"] = _safe_float(latest.get("cashAndEquivalents"))
            data["totalEquity"] = _safe_float(latest.get("totalEquity"))
            data["netDebt"] = _safe_float(latest.get("netDebt"))

            # Compute market cap for FCF yield
            # Use KPI if available
            mc_hist = api.get_kpi_history(ins_id, KPI["market_cap"], "r12", "mean")
            if mc_hist:
                mc_val = mc_hist[-1].get("v")
                if mc_val is not None:
                    data["marketCap"] = mc_val * 1_000_000  # Börsdata reports in millions

            # FCF yield
            if data["freeCashflow"] and data["marketCap"] and data["marketCap"] > 0:
                data["fcf_yield"] = data["freeCashflow"] / (data["marketCap"] / 1_000_000)
            elif data.get("fcf_margin") and data.get("ps"):
                # Alternative: FCF margin / P/S
                pass

    except Exception as exc:
        logger.debug("Börsdata reports for %s: %s", ticker, exc)

    # ── Growth history (CAGR) ────────────────────────────────────────────
    try:
        growth = api.get_growth_history(ins_id, years=10)
        data["revenue_cagr_5y"] = growth.get("revenue_cagr_5y")
        data["revenue_cagr_10y"] = growth.get("revenue_cagr_10y")
        data["earnings_cagr_5y"] = growth.get("earnings_cagr_5y")
        data["earnings_cagr_10y"] = growth.get("earnings_cagr_10y")
    except Exception as exc:
        logger.debug("Börsdata growth for %s: %s", ticker, exc)

    return data


# ---------------------------------------------------------------------------
# yfinance fundamentals extraction (fallback)
# ---------------------------------------------------------------------------

def _extract_yfinance_fundamentals(ticker: str) -> dict:
    """
    Pull fundamental data from yfinance .info dict.
    Returns a normalised dict. All numeric fields default to None.
    """
    data = _make_empty_fundamentals(ticker)
    data["_data_source"] = "yfinance"

    try:
        tk = yf.Ticker(ticker)
        info: dict = tk.info or {}

        if not info or info.get("regularMarketPrice") is None and not info.get("marketCap"):
            return data

        for field in ("shortName", "sector", "industry", "country", "currency"):
            v = info.get(field)
            if v and isinstance(v, str):
                data[field] = v

        # Valuation
        data["enterpriseToEbitda"] = _safe_float(info.get("enterpriseToEbitda"))
        data["priceToBook"] = _safe_float(info.get("priceToBook"))
        data["trailingPE"] = _safe_float(info.get("trailingPE"))
        data["forwardPE"] = _safe_float(info.get("forwardPE"))

        # Cash flow
        data["freeCashflow"] = _safe_float(info.get("freeCashflow"))
        data["marketCap"] = _safe_float(info.get("marketCap"))
        data["operatingCashflow"] = _safe_float(info.get("operatingCashflow"))
        if data["freeCashflow"] and data["marketCap"] and data["marketCap"] > 0:
            data["fcf_yield"] = data["freeCashflow"] / data["marketCap"]

        # Profitability
        data["returnOnEquity"] = _safe_float(info.get("returnOnEquity"))
        data["returnOnAssets"] = _safe_float(info.get("returnOnAssets"))

        # Approximate ROIC
        roce = _safe_float(info.get("returnOnCapitalEmployed"))
        if roce is not None:
            data["roic_approx"] = roce
        else:
            ebitda = _safe_float(info.get("ebitda"))
            total_debt = _safe_float(info.get("totalDebt")) or 0.0
            bvps = _safe_float(info.get("bookValue"))
            shares = _safe_float(info.get("sharesOutstanding"))
            if ebitda is not None and bvps is not None and shares is not None:
                book_equity = bvps * shares
                invested_cap = total_debt + max(book_equity, 1.0)
                if invested_cap > 0:
                    data["roic_approx"] = (ebitda * 0.65) / invested_cap
            elif data["returnOnEquity"] is not None:
                data["roic_approx"] = data["returnOnEquity"]

        # Leverage
        de_raw = _safe_float(info.get("debtToEquity"))
        if de_raw is not None:
            data["debtToEquity"] = de_raw
            data["de_ratio"] = de_raw / 100.0 if de_raw > 10 else de_raw
        data["totalDebt"] = _safe_float(info.get("totalDebt"))
        data["totalCash"] = _safe_float(info.get("totalCash"))

        # Income
        data["ebitda"] = _safe_float(info.get("ebitda"))
        data["totalRevenue"] = _safe_float(info.get("totalRevenue"))

        data["_raw_info"] = {
            k: v for k, v in info.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }

    except Exception as exc:
        logger.warning("_extract_yfinance_fundamentals(%s): %s", ticker, exc)

    return data


def _make_empty_fundamentals(ticker: str) -> dict:
    """Create an empty fundamentals dict with all expected keys."""
    return {
        # Identifiers
        "ticker": ticker,
        "shortName": ticker,
        "sector": "Unknown",
        "industry": "Unknown",
        "country": "Unknown",
        "currency": "Unknown",
        "_data_source": "none",
        "_borsdata_ins_id": None,
        # Valuation
        "enterpriseToEbitda": None,
        "priceToBook": None,
        "trailingPE": None,
        "forwardPE": None,
        # Cash flow & yield
        "freeCashflow": None,
        "marketCap": None,
        "fcf_yield": None,
        "operatingCashflow": None,
        # Profitability
        "returnOnEquity": None,
        "returnOnAssets": None,
        "roic_approx": None,
        "gross_margin": None,
        "operating_margin": None,
        "profit_margin": None,
        "fcf_margin": None,
        "ebitda_margin": None,
        # Leverage
        "debtToEquity": None,
        "de_ratio": None,
        "equity_ratio": None,
        "net_debt_ebitda": None,
        "current_ratio": None,
        "totalDebt": None,
        "totalCash": None,
        "totalEquity": None,
        "netDebt": None,
        # Income
        "ebitda": None,
        "totalRevenue": None,
        # Growth (Börsdata only)
        "revenue_growth": None,
        "earnings_growth": None,
        "revenue_cagr_5y": None,
        "revenue_cagr_10y": None,
        "earnings_cagr_5y": None,
        "earnings_cagr_10y": None,
        # Stability (Börsdata only)
        "earnings_stability": None,
        "fcf_stability": None,
        # Quality scores (Börsdata only)
        "f_score": None,
        "magic_formula": None,
        "rs_rank": None,
        # Dividend
        "dividend_yield": None,
        # Insider
        "insider_buying": False,
        # Raw passthrough
        "_raw_info": {},
    }


# ---------------------------------------------------------------------------
# Insider transactions (yfinance — Börsdata has Holdings API for Pro+)
# ---------------------------------------------------------------------------

@_cache(ttl=3600)
def fetch_insider_transactions(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch insider transaction data via yfinance."""
    try:
        tk = yf.Ticker(ticker)
        insider = tk.insider_transactions
        if insider is not None and not insider.empty:
            return insider
    except Exception as exc:
        logger.warning("fetch_insider_transactions(%s): %s", ticker, exc)
    return None


def _check_insider_buying(ticker: str, days: int = 180) -> bool:
    """Return True if there is at least one insider BUY in the last `days` days."""
    try:
        insider_df = fetch_insider_transactions(ticker)
        if insider_df is None or insider_df.empty:
            return False

        cutoff = datetime.utcnow() - timedelta(days=days)
        df = insider_df.copy()

        date_col = next(
            (c for c in ("Start Date", "Date", "startDate") if c in df.columns),
            None,
        )
        if date_col is None:
            return False

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        recent = df[df[date_col] >= cutoff].dropna(subset=[date_col])

        type_col = next(
            (c for c in ("Transaction", "transaction", "Type") if c in recent.columns),
            None,
        )
        if type_col is None:
            return len(recent) > 0

        buys = recent[
            recent[type_col].str.contains("Buy|Purchase|buy|purchase", na=False)
        ]
        return len(buys) > 0

    except Exception as exc:
        logger.debug("_check_insider_buying(%s): %s", ticker, exc)
        return False


# ---------------------------------------------------------------------------
# Public fundamentals API
# ---------------------------------------------------------------------------

@_cache(ttl=3600)
def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental data for a single ticker.

    Data source priority:
      1. Börsdata API (if key is configured and ticker resolves)
      2. yfinance (fallback)

    Returns normalised dict with all scoring-relevant fields.
    """
    # Try Börsdata first
    if _borsdata_available():
        data = _extract_borsdata_fundamentals(ticker)
        if data.get("_data_source") == "borsdata":
            # Enrich with insider data from yfinance (Börsdata doesn't expose this easily)
            data["insider_buying"] = _check_insider_buying(ticker, days=180)
            return data

    # Fall back to yfinance
    data = _extract_yfinance_fundamentals(ticker)
    data["insider_buying"] = _check_insider_buying(ticker, days=180)
    return data


# ---------------------------------------------------------------------------
# Convenience batch loader
# ---------------------------------------------------------------------------

def fetch_fundamentals_batch(
    tickers: List[str],
) -> Dict[str, dict]:
    """
    Fetch fundamentals for multiple tickers.
    Returns dict of ticker → fundamentals dict.
    """
    results = {}
    for ticker in tickers:
        results[ticker] = fetch_fundamentals(ticker)
    return results
