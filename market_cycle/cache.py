"""
market_cycle/cache.py
=====================
Streamlit-cached wrappers for indicator computation and cycle detection.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from market_cycle.indicators import compute_indicators, download_ohlcv, _compute_from_df
from market_cycle.engine import detect_market_cycle


@st.cache_data(ttl=3600, show_spinner=False)
def cached_market_cycle_analysis(ticker: str, period: str) -> dict:
    """Return {indicators, result} for the current state of ticker."""
    indicators = compute_indicators(ticker, period)
    if not indicators:
        return {}
    result = detect_market_cycle(indicators)
    return {"indicators": indicators, "result": result}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_market_cycle_history(ticker: str, period: str) -> list[dict]:
    """
    Compute rolling phase detection over time.

    Downloads up to 2y of data and samples N evenly spaced points.
    Returns list of {date, phase, confidence, phase_index}.
    """
    # Download raw data — always use enough history for rolling indicators
    dl_period = "2y" if period in ("1mo", "3mo", "6mo", "1y") else "5y"
    df = download_ohlcv(ticker, dl_period)
    if df is None or df.empty or len(df) < 60:
        return []

    # Map period to number of rows to look back from the end
    period_rows = {
        "1mo":  21,
        "3mo":  63,
        "6mo":  126,
        "1y":   252,
        "2y":   504,
    }
    lookback_rows = period_rows.get(period, 252)
    # We need at least 30 rows of prior history to compute indicators
    min_history = 50

    # Build index positions: evenly spaced N points within [min_history, end]
    n_points = 60
    end_idx = len(df)
    start_idx = max(min_history, end_idx - lookback_rows)

    if end_idx - start_idx < 5:
        return []

    import numpy as np
    positions = np.linspace(start_idx, end_idx - 1, n_points, dtype=int)
    positions = sorted(set(positions))  # deduplicate

    from market_cycle.rules import PHASE_ORDER

    history = []
    for pos in positions:
        slice_df = df.iloc[: pos + 1].copy()
        if len(slice_df) < min_history:
            continue
        try:
            ind = _compute_from_df(slice_df)
            if not ind:
                continue
            res = detect_market_cycle(ind)
            phase = res["phase"]
            history.append({
                "date": df.index[pos],
                "phase": phase,
                "confidence": res["confidence"],
                "phase_index": PHASE_ORDER.index(phase),
            })
        except Exception:
            continue

    return history
