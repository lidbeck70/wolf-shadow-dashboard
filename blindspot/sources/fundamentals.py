"""
fundamentals.py — Fundamental data source for Odin's Blindspot Index.
Tries EODHD first (if API key available), falls back to yfinance.
"""
import logging
import os

from blindspot.cache import get_cached, set_cached

logger = logging.getLogger(__name__)


def _get_eodhd_key() -> str:
    """Get EODHD API key from st.secrets or environment."""
    try:
        import streamlit as st
        key = st.secrets.get("EODHD_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("EODHD_API_KEY", "")


def _fetch_eodhd(ticker: str, api_key: str) -> dict:
    """Fetch fundamentals from EODHD API."""
    import requests

    # EODHD uses . notation differently — convert Nordic suffixes
    eodhd_ticker = ticker
    if ticker.endswith(".ST"):
        eodhd_ticker = ticker.replace(".ST", ".STO")
    elif ticker.endswith(".OL"):
        eodhd_ticker = ticker.replace(".OL", ".OSL")

    url = f"https://eodhd.com/api/fundamentals/{eodhd_ticker}"
    params = {"api_token": api_key, "fmt": "json"}

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    # Extract financials
    result = {
        "sector": "",
        "industry": "",
        "fcf": None,
        "fcf_history": [],
        "ebitda": None,
        "revenue": None,
        "debt_to_equity": None,
        "ev_ebitda": None,
        "revenue_growth": None,
        "ebitda_margin": None,
        "market_cap": None,
        "enterprise_value": None,
    }

    # General info
    general = data.get("General", {})
    result["sector"] = general.get("GicsSector", general.get("Sector", ""))
    result["industry"] = general.get("GicsSubIndustry", general.get("Industry", ""))

    # Highlights
    highlights = data.get("Highlights", {})
    result["market_cap"] = highlights.get("MarketCapitalization")
    result["ebitda"] = highlights.get("EBITDA")
    result["revenue"] = highlights.get("Revenue")

    # Valuation
    valuation = data.get("Valuation", {})
    result["ev_ebitda"] = valuation.get("EnterpriseValueEbitda")
    result["enterprise_value"] = valuation.get("EnterpriseValue")

    # Financial ratios
    financials = data.get("Financials", {})

    # Cash flow — get FCF for last 3 years
    cash_flow = financials.get("Cash_Flow", {}).get("yearly", {})
    if isinstance(cash_flow, dict):
        fcf_values = []
        for period_key in sorted(cash_flow.keys(), reverse=True)[:3]:
            period = cash_flow[period_key]
            fcf_val = period.get("freeCashFlow")
            if fcf_val is not None:
                try:
                    fcf_values.append(float(fcf_val))
                except (ValueError, TypeError):
                    pass
        if fcf_values:
            result["fcf"] = fcf_values[0]
            result["fcf_history"] = fcf_values

    # Balance sheet for D/E
    balance = financials.get("Balance_Sheet", {}).get("yearly", {})
    if isinstance(balance, dict):
        latest_key = sorted(balance.keys(), reverse=True)
        if latest_key:
            latest = balance[latest_key[0]]
            total_debt = None
            equity = None
            for k in ["totalDebt", "shortLongTermDebt", "longTermDebt"]:
                if latest.get(k) is not None:
                    try:
                        total_debt = float(latest[k])
                        break
                    except (ValueError, TypeError):
                        pass
            for k in ["totalStockholderEquity", "totalShareholderEquity"]:
                if latest.get(k) is not None:
                    try:
                        equity = float(latest[k])
                        break
                    except (ValueError, TypeError):
                        pass
            if total_debt is not None and equity and equity > 0:
                result["debt_to_equity"] = round(total_debt / equity, 2)

    # Income statement for revenue growth and EBITDA margin
    income = financials.get("Income_Statement", {}).get("yearly", {})
    if isinstance(income, dict):
        sorted_years = sorted(income.keys(), reverse=True)
        if len(sorted_years) >= 2:
            curr_rev = income[sorted_years[0]].get("totalRevenue")
            prev_rev = income[sorted_years[1]].get("totalRevenue")
            if curr_rev and prev_rev:
                try:
                    curr_rev = float(curr_rev)
                    prev_rev = float(prev_rev)
                    if prev_rev > 0:
                        result["revenue_growth"] = round((curr_rev / prev_rev - 1) * 100, 2)
                    if result["ebitda"] and curr_rev > 0:
                        result["ebitda_margin"] = round(float(result["ebitda"]) / curr_rev * 100, 2)
                except (ValueError, TypeError):
                    pass

    # FCF yield
    if result["fcf"] and result["market_cap"] and result["market_cap"] > 0:
        result["fcf_yield"] = round(result["fcf"] / result["market_cap"] * 100, 2)

    return result


def _fetch_yfinance(ticker: str) -> dict:
    """Fallback: fetch fundamentals from yfinance .info dict."""
    import yfinance as yf

    info = yf.Ticker(ticker).info
    if not info:
        return {}

    result = {
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "fcf": info.get("freeCashflow"),
        "fcf_history": [],
        "ebitda": info.get("ebitda"),
        "revenue": info.get("totalRevenue"),
        "debt_to_equity": None,
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "revenue_growth": None,
        "ebitda_margin": None,
        "market_cap": info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
        "fcf_yield": None,
    }

    # D/E from yfinance (already a ratio * 100 in some cases)
    de = info.get("debtToEquity")
    if de is not None:
        try:
            de = float(de)
            # yfinance sometimes returns as percentage
            result["debt_to_equity"] = de / 100 if de > 10 else de
        except (ValueError, TypeError):
            pass

    # Revenue growth
    rg = info.get("revenueGrowth")
    if rg is not None:
        try:
            result["revenue_growth"] = round(float(rg) * 100, 2)
        except (ValueError, TypeError):
            pass

    # EBITDA margin
    if result["ebitda"] and result["revenue"] and result["revenue"] > 0:
        result["ebitda_margin"] = round(result["ebitda"] / result["revenue"] * 100, 2)

    # FCF yield
    if result["fcf"] and result["market_cap"] and result["market_cap"] > 0:
        result["fcf_yield"] = round(result["fcf"] / result["market_cap"] * 100, 2)

    # Single year FCF as history
    if result["fcf"]:
        result["fcf_history"] = [result["fcf"]]

    return result


def fetch_fundamentals(tickers: list) -> dict:
    """Fetch fundamentals for all tickers. Returns {ticker: {data, confidence}}."""
    cache_key = f"blindspot_fund_{'_'.join(sorted(tickers[:10]))}"
    cached = get_cached(cache_key)
    if cached is not None:
        return cached

    api_key = _get_eodhd_key()
    results = {}

    for ticker in tickers:
        try:
            if api_key:
                try:
                    data = _fetch_eodhd(ticker, api_key)
                    data["confidence"] = 1.0
                    data["source"] = "eodhd"
                    results[ticker] = data
                    continue
                except Exception as e:
                    logger.debug("EODHD failed for %s, falling back: %s", ticker, e)

            # yfinance fallback
            data = _fetch_yfinance(ticker)
            if data:
                data["confidence"] = 0.6
                data["source"] = "yfinance"
                results[ticker] = data
        except Exception as e:
            logger.debug("Fundamentals fetch failed for %s: %s", ticker, e)
            continue

    set_cached(cache_key, results)
    return results
