"""
Market breadth computation for OVTLYR.
Pure functions — no Streamlit imports.
"""

import pandas as pd


def compute_breadth(instruments_df: pd.DataFrame = None) -> dict:
    """
    Compute market breadth from a DataFrame of instruments.

    The DataFrame must contain (at minimum) a ``trend_state`` column with
    values "Bullish", "Bearish", or "Neutral". An optional ``sector``
    column enables per-sector breakdown.

    If *instruments_df* is None, the function attempts to fetch it from
    Börsdata via :func:`data_fetch.borsdata.fetch_all_instruments_summary`.

    Returns a dict with:
        total      : int
        bullish    : int
        bearish    : int
        neutral    : int
        pct_bullish: float  – fraction in [0, 1]
        sectors    : dict   – sector_name -> {bullish, bearish, neutral, total}
    """
    if instruments_df is None:
        try:
            try:
                from borsdata_api import get_api  # noqa: F401 – availability check
                from dashboard.ovtlyr.data_fetch.borsdata import fetch_all_instruments_summary
            except ImportError:
                from dashboard.ovtlyr.data_fetch.borsdata import fetch_all_instruments_summary
            instruments_df = fetch_all_instruments_summary()
        except Exception:
            instruments_df = pd.DataFrame()

    empty_result = {
        "total": 0,
        "bullish": 0,
        "bearish": 0,
        "neutral": 0,
        "pct_bullish": 0.5,
        "sectors": {},
    }

    if instruments_df is None or instruments_df.empty or "trend_state" not in instruments_df.columns:
        return empty_result

    df = instruments_df.copy()

    # Normalise trend_state to title case
    df["trend_state"] = df["trend_state"].astype(str).str.strip().str.title()

    total = len(df)
    bullish = int((df["trend_state"] == "Bullish").sum())
    bearish = int((df["trend_state"] == "Bearish").sum())
    neutral = int((df["trend_state"] == "Neutral").sum())
    pct_bullish = bullish / total if total > 0 else 0.5

    # Per-sector breakdown
    sectors: dict = {}
    if "sector" in df.columns:
        for sector_name, group in df.groupby("sector"):
            s_total = len(group)
            s_bullish = int((group["trend_state"] == "Bullish").sum())
            s_bearish = int((group["trend_state"] == "Bearish").sum())
            s_neutral = int((group["trend_state"] == "Neutral").sum())
            sectors[str(sector_name)] = {
                "bullish": s_bullish,
                "bearish": s_bearish,
                "neutral": s_neutral,
                "total": s_total,
            }

    return {
        "total": total,
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "pct_bullish": pct_bullish,
        "sectors": sectors,
    }
