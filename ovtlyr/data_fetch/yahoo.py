"""
Yahoo Finance (yfinance) — fallback data source for OVTLYR.
All functions are pure (no Streamlit imports).
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance.

    Returns a DataFrame with a DatetimeIndex and columns:
        Open, High, Low, Close, Volume.
    Returns an empty DataFrame if the ticker is not found or on error.
    Same return format as borsdata.fetch_ohlcv.
    """
    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        df = tk.history(period=period, auto_adjust=True)

        if df is None or df.empty:
            return pd.DataFrame()

        # Normalise column names (yfinance already returns capitalised names)
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if "open" in cl:
                col_map[c] = "Open"
            elif "high" in cl:
                col_map[c] = "High"
            elif "low" in cl:
                col_map[c] = "Low"
            elif "close" in cl:
                col_map[c] = "Close"
            elif "vol" in cl:
                col_map[c] = "Volume"
        df = df.rename(columns=col_map)

        # Ensure DatetimeIndex (timezone-naive for consistency)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                import numpy as np
                df[col] = np.nan

        return df[required].dropna(subset=["Close"]).sort_index()

    except Exception:
        return pd.DataFrame()


def fetch_sector(ticker: str) -> str:
    """
    Get the sector for a ticker from yfinance .info dict.

    Returns an empty string if the sector cannot be determined.
    """
    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        info = tk.info or {}
        return str(info.get("sector", ""))
    except Exception:
        return ""
