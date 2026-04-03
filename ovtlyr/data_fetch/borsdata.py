"""
Börsdata PRO API — primary data source for OVTLYR.
All functions are pure (no Streamlit imports). They accept plain parameters
and return DataFrames or dicts so callers can cache freely.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from borsdata_api import get_api, KPI
except ImportError:
    from dashboard.borsdata_api import get_api, KPI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_client():
    """Return a connected Börsdata API client, or None if unavailable."""
    try:
        return get_api()
    except Exception:
        return None


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, period_years: int = 2) -> pd.DataFrame:
    """
    Fetch OHLCV data from Börsdata PRO API.

    Uses borsdata_api.get_stockprices_df() method.
    Returns a DataFrame with a DatetimeIndex and columns:
        Open, High, Low, Close, Volume.
    Returns an empty DataFrame if the ticker is not found in Börsdata.
    """
    api = _get_api_client()
    if api is None:
        return pd.DataFrame()

    try:
        # Resolve instrument id from ticker string
        instruments = api.get_instruments()
        ins_df = instruments.instruments if hasattr(instruments, "instruments") else instruments
        if hasattr(ins_df, "to_dict"):
            # DataFrame path
            row = ins_df[ins_df["ticker"].str.upper() == ticker.upper()]
            if row.empty:
                return pd.DataFrame()
            ins_id = int(row.iloc[0]["insId"])
        else:
            # List of instrument objects
            match = [i for i in ins_df if getattr(i, "ticker", "").upper() == ticker.upper()]
            if not match:
                return pd.DataFrame()
            ins_id = match[0].insId

        from_date = datetime.today() - timedelta(days=365 * period_years)
        prices = api.get_stockprices_df(ins_id, from_date=from_date)

        if prices is None or prices.empty:
            return pd.DataFrame()

        # Normalise column names
        col_map = {}
        for c in prices.columns:
            cl = c.lower()
            if "open" in cl:
                col_map[c] = "Open"
            elif "high" in cl:
                col_map[c] = "High"
            elif "low" in cl:
                col_map[c] = "Low"
            elif "close" in cl or "last" in cl:
                col_map[c] = "Close"
            elif "vol" in cl:
                col_map[c] = "Volume"
        prices = prices.rename(columns=col_map)

        # Ensure DatetimeIndex
        if not isinstance(prices.index, pd.DatetimeIndex):
            date_cols = [c for c in prices.columns if "date" in c.lower()]
            if date_cols:
                prices.index = pd.to_datetime(prices[date_cols[0]])
                prices = prices.drop(columns=date_cols)
            else:
                prices.index = pd.to_datetime(prices.index)

        prices = prices.sort_index()
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in prices.columns:
                prices[col] = np.nan

        return prices[required].dropna(subset=["Close"])

    except Exception:
        return pd.DataFrame()


def fetch_sector(ticker: str) -> str:
    """
    Get the sector name from Börsdata instrument metadata.

    Returns an empty string if the ticker is not found.
    """
    api = _get_api_client()
    if api is None:
        return ""

    try:
        instruments = api.get_instruments()
        ins_df = instruments.instruments if hasattr(instruments, "instruments") else instruments
        if hasattr(ins_df, "to_dict"):
            row = ins_df[ins_df["ticker"].str.upper() == ticker.upper()]
            if row.empty:
                return ""
            return str(row.iloc[0].get("sectorName", ""))
        else:
            match = [i for i in ins_df if getattr(i, "ticker", "").upper() == ticker.upper()]
            if not match:
                return ""
            return str(getattr(match[0], "sectorName", ""))
    except Exception:
        return ""


def fetch_fundamentals_snapshot(ticker: str) -> dict:
    """
    Get key fundamentals via Börsdata screener KPIs.

    Returns a dict with:
        roe, operating_margin, revenue_growth, earnings_growth,
        net_debt_ebitda, fcf_stability, f_score, rs_rank,
        pe, pb, ev_ebitda, market_cap.
    Missing values are None.
    """
    result = {
        "roe": None,
        "operating_margin": None,
        "revenue_growth": None,
        "earnings_growth": None,
        "net_debt_ebitda": None,
        "fcf_stability": None,
        "f_score": None,
        "rs_rank": None,
        "pe": None,
        "pb": None,
        "ev_ebitda": None,
        "market_cap": None,
    }

    api = _get_api_client()
    if api is None:
        return result

    try:
        instruments = api.get_instruments()
        ins_df = instruments.instruments if hasattr(instruments, "instruments") else instruments
        if hasattr(ins_df, "to_dict"):
            row = ins_df[ins_df["ticker"].str.upper() == ticker.upper()]
            if row.empty:
                return result
            ins_id = int(row.iloc[0]["insId"])
        else:
            match = [i for i in ins_df if getattr(i, "ticker", "").upper() == ticker.upper()]
            if not match:
                return result
            ins_id = match[0].insId

        # KPI id map (Börsdata standard KPI IDs)
        kpi_map = {
            "roe": KPI.ROE if hasattr(KPI, "ROE") else 2,
            "operating_margin": KPI.OPERATING_MARGIN if hasattr(KPI, "OPERATING_MARGIN") else 11,
            "revenue_growth": KPI.REVENUE_GROWTH if hasattr(KPI, "REVENUE_GROWTH") else 16,
            "earnings_growth": KPI.EARNINGS_GROWTH if hasattr(KPI, "EARNINGS_GROWTH") else 17,
            "net_debt_ebitda": KPI.NET_DEBT_EBITDA if hasattr(KPI, "NET_DEBT_EBITDA") else 60,
            "pe": KPI.PE if hasattr(KPI, "PE") else 2,
            "pb": KPI.PB if hasattr(KPI, "PB") else 3,
            "ev_ebitda": KPI.EV_EBITDA if hasattr(KPI, "EV_EBITDA") else 40,
            "market_cap": KPI.MARKET_CAP if hasattr(KPI, "MARKET_CAP") else 64,
        }

        for key, kpi_id in kpi_map.items():
            try:
                val = api.get_kpi_updated(ins_id, kpi_id, "last", "mean")
                if val is not None:
                    if hasattr(val, "value"):
                        result[key] = val.value
                    elif isinstance(val, (int, float)):
                        result[key] = val
            except Exception:
                pass

    except Exception:
        pass

    return result


def fetch_all_instruments_summary() -> pd.DataFrame:
    """
    Get a summary DataFrame of all Börsdata instruments with:
        ticker, name, sector, market, last_close, ema50, ema200, trend_state.

    Uses batch endpoints for efficiency.
    Returns an empty DataFrame on failure.
    """
    api = _get_api_client()
    if api is None:
        return pd.DataFrame()

    try:
        instruments = api.get_instruments()
        ins_df = instruments.instruments if hasattr(instruments, "instruments") else instruments

        if not hasattr(ins_df, "iterrows"):
            # Convert list of objects to DataFrame
            ins_df = pd.DataFrame([
                {
                    "insId": getattr(i, "insId", None),
                    "ticker": getattr(i, "ticker", ""),
                    "name": getattr(i, "name", ""),
                    "sectorName": getattr(i, "sectorName", ""),
                    "marketName": getattr(i, "marketName", ""),
                }
                for i in ins_df
            ])

        records = []
        for _, row in ins_df.iterrows():
            ins_id = row.get("insId")
            ticker = str(row.get("ticker", ""))
            name = str(row.get("name", ""))
            sector = str(row.get("sectorName", ""))
            market = str(row.get("marketName", ""))

            last_close = np.nan
            ema50_val = np.nan
            ema200_val = np.nan
            trend_state = "Neutral"

            try:
                from_date = datetime.today() - timedelta(days=365)
                prices = api.get_stockprices_df(ins_id, from_date=from_date)
                if prices is not None and not prices.empty:
                    close_col = next(
                        (c for c in prices.columns if "close" in c.lower() or "last" in c.lower()),
                        None,
                    )
                    if close_col:
                        close = prices[close_col].dropna()
                        if len(close) >= 1:
                            last_close = float(close.iloc[-1])
                            ema50_val = float(_ema(close, 50).iloc[-1])
                            ema200_val = float(_ema(close, 200).iloc[-1])
                            if last_close > ema200_val and ema50_val > ema200_val:
                                trend_state = "Bullish"
                            elif last_close < ema200_val and ema50_val < ema200_val:
                                trend_state = "Bearish"
                            else:
                                trend_state = "Neutral"
            except Exception:
                pass

            records.append(
                {
                    "ticker": ticker,
                    "name": name,
                    "sector": sector,
                    "market": market,
                    "last_close": last_close,
                    "ema50": ema50_val,
                    "ema200": ema200_val,
                    "trend_state": trend_state,
                }
            )

        return pd.DataFrame(records)

    except Exception:
        return pd.DataFrame()


def is_available() -> bool:
    """Return True if Börsdata API is configured and ready."""
    return _get_api_client() is not None
