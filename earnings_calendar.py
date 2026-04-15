"""
earnings_calendar.py — Earnings Calendar for Holdings tab

Shows upcoming earnings dates for all open positions,
with warnings for holdings reporting within 3 or 7 days.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Nordic Gold theme
BG = "#0c0c12"
BG2 = "#14141e"
CYAN = "#c9a84c"
MAGENTA = "#8b7340"
GREEN = "#2d8a4e"
RED = "#c44545"
YELLOW = "#d4943a"
TEXT = "#e8e4dc"
DIM = "#8a8578"

STRATEGY_MAP = {
    "swing": "Wolf",
    "ovtlyr": "Viking",
    "long": "Alpha",
}


@st.cache_data(ttl=86400, show_spinner=False, max_entries=50)
def _fetch_earnings_date(ticker: str) -> dict:
    """Fetch next earnings date for a ticker. Cached 24h."""
    try:
        tk = yf.Ticker(ticker)

        # Try .calendar first
        try:
            cal = tk.calendar
            if cal is not None:
                if isinstance(cal, pd.DataFrame) and not cal.empty:
                    # calendar can be a DataFrame with column 0 or "Earnings Date"
                    for col_name in ["Earnings Date", 0]:
                        if col_name in cal.columns:
                            val = cal[col_name].iloc[0]
                            if pd.notna(val):
                                dt = pd.Timestamp(val)
                                if dt.tzinfo is not None:
                                    dt = dt.tz_localize(None)
                                return {"ticker": ticker, "date": dt.to_pydatetime(), "source": "calendar"}
                    # Try index-based access
                    if "Earnings Date" in cal.index:
                        val = cal.loc["Earnings Date"].iloc[0]
                        if pd.notna(val):
                            dt = pd.Timestamp(val)
                            if dt.tzinfo is not None:
                                dt = dt.tz_localize(None)
                            return {"ticker": ticker, "date": dt.to_pydatetime(), "source": "calendar"}
                elif isinstance(cal, dict):
                    for key in ["Earnings Date", "earningsDate"]:
                        if key in cal:
                            val = cal[key]
                            if isinstance(val, list) and val:
                                val = val[0]
                            if val is not None:
                                dt = pd.Timestamp(val)
                                if dt.tzinfo is not None:
                                    dt = dt.tz_localize(None)
                                return {"ticker": ticker, "date": dt.to_pydatetime(), "source": "calendar"}
        except Exception:
            pass

        # Fallback: .earnings_dates
        try:
            ed = tk.earnings_dates
            if ed is not None and not ed.empty:
                now = pd.Timestamp.now()
                # Filter to future dates
                future = ed[ed.index >= now]
                if not future.empty:
                    next_date = future.index[0]
                    if next_date.tzinfo is not None:
                        next_date = next_date.tz_localize(None)
                    return {"ticker": ticker, "date": next_date.to_pydatetime(), "source": "earnings_dates"}
                # If no future dates, take the most recent
                last_date = ed.index[0]
                if last_date.tzinfo is not None:
                    last_date = last_date.tz_localize(None)
                return {"ticker": ticker, "date": last_date.to_pydatetime(), "source": "earnings_dates"}
        except Exception:
            pass

        return {"ticker": ticker, "date": None, "source": None}
    except Exception:
        return {"ticker": ticker, "date": None, "source": None}


def _color_days(days: int) -> str:
    """Return color based on days until earnings."""
    if days < 3:
        return RED
    elif days < 7:
        return YELLOW
    return DIM


def render_earnings_calendar(holdings_data: dict):
    """Render the Earnings Calendar section within Holdings tab."""
    try:
        # Section header
        st.markdown(
            f"<div style='border-bottom:2px solid {CYAN};padding-bottom:6px;margin-bottom:16px;'>"
            f"<h3 style='color:{CYAN};margin:0;letter-spacing:0.08em;'>EARNINGS CALENDAR</h3>"
            f"<span style='color:{DIM};font-size:0.7rem;'>Kommande rapporter f\u00f6r dina inneh\u00e5ll (30 dagar)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Collect all tickers with their strategies
        ticker_strategy = {}
        for pkey, holdings in holdings_data.items():
            strategy = STRATEGY_MAP.get(pkey, pkey)
            for h in holdings:
                ticker_strategy[h.get("ticker", "")] = strategy

        if not ticker_strategy:
            st.info("Inga inneh\u00e5ll att kontrollera.")
            return

        # Fetch earnings dates
        now = datetime.now()
        cutoff = now + timedelta(days=30)
        earnings_rows = []
        imminent = []  # < 3 days

        for ticker, strategy in ticker_strategy.items():
            try:
                result = _fetch_earnings_date(ticker)
                if result["date"] is not None:
                    dt = result["date"]
                    days_until = (dt - now).days
                    if 0 <= days_until <= 30:
                        earnings_rows.append({
                            "Ticker": ticker,
                            "Rapportdatum": dt.strftime("%Y-%m-%d"),
                            "Dagar kvar": days_until,
                            "Strategi": strategy,
                        })
                        if days_until < 3:
                            imminent.append((ticker, days_until))
            except Exception:
                continue

        # ── Warning Banner ──────────────────────────────────────────
        if imminent:
            for ticker, days in imminent:
                day_text = "idag" if days == 0 else f"om {days} dag" if days == 1 else f"om {days} dagar"
                st.warning(f"{ticker} rapporterar {day_text} \u2014 \u00f6verv\u00e4g att minska position")

        # ── Earnings Table ──────────────────────────────────────────
        if earnings_rows:
            df = pd.DataFrame(earnings_rows).sort_values("Dagar kvar").reset_index(drop=True)

            def _highlight_row(row):
                days = row["Dagar kvar"]
                if days < 3:
                    return [f"background-color: rgba(196,69,69,0.15); color: {RED}"] * len(row)
                elif days < 7:
                    return [f"background-color: rgba(212,148,58,0.1); color: {YELLOW}"] * len(row)
                return [f"color: {TEXT}"] * len(row)

            styled = df.style.apply(_highlight_row, axis=1).format(
                {"Dagar kvar": "{:.0f}"}
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Summary
            st.markdown(
                f"<div style='color:{DIM};font-size:0.65rem;text-align:right;'>"
                f"{len(earnings_rows)} rapport(er) inom 30 dagar"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:{BG2};border-radius:8px;padding:20px;text-align:center;"
                f"border:1px solid rgba(201,168,76,0.1);'>"
                f"<div style='color:{DIM};font-size:0.85rem;'>"
                f"Inga rapporter inom 30 dagar f\u00f6r dina inneh\u00e5ll</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    except Exception as e:
        logger.warning("Earnings calendar error: %s", e)
        st.warning(f"Earnings Calendar kunde inte laddas: {e}")
