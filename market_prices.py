"""
market_prices.py — EN cachad väg till kurshistorik ur yfinance.

Panelen hade minst tio nästan identiska nedladdningswrappers utan gemensam
cache: temabordet laddade samma ticker tre gånger och SPY nio gånger,
kvotmodulen 23 nedladdningar för 13 symboler, Ember DXY tre gånger, och
Wolf Regime två okachade nedladdningar per klick. Här finns en funktion
per behov, cachad i Streamlit (6 h, begränsat antal poster) och med en
process-cache när Streamlit inte finns (Actions-jobben).

  ohlcv(ticker, period)  → DataFrame Open/High/Low/Close/Volume (tom vid fel)
  close(ticker, period)  → Series Close
  closes(tickers, period)→ dict ticker → Series (en batch-nedladdning)

Ingen modul behöver ändra sin logik: samma period-strängar som yfinance.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)

TTL_SECONDS = 6 * 3600
MAX_ENTRIES = 300

try:
    import streamlit as st
    _cache = st.cache_data(ttl=TTL_SECONDS, max_entries=MAX_ENTRIES, show_spinner=False)
except Exception:                                   # pragma: no cover — headless
    st = None

    def _cache(fn):
        store: dict = {}

        def wrapped(*args):
            now = time.time()
            hit = store.get(args)
            if hit and now - hit[0] < TTL_SECONDS:
                return hit[1]
            val = fn(*args)
            if len(store) >= MAX_ENTRIES:
                store.pop(next(iter(store)))
            store[args] = (now, val)
            return val
        wrapped.clear = store.clear                   # samma API som st.cache_data
        return wrapped


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


class _NoData(Exception):
    """Tom eller misslyckad nedladdning. Kastas INNE i den cachade funktionen så
    att ett tillfälligt fel (t.ex. Yahoos 'Too Many Requests') aldrig cachas i
    sex timmar — nästa anrop försöker igen."""


@_cache
def _ohlcv_raw(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    df = _flatten(yf.download(ticker, period=period, auto_adjust=True, progress=False, threads=False))
    df = df.dropna(how="all")
    if df.empty:
        raise _NoData(f"{ticker} {period}: tom")
    return df


def ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Kurshistorik för EN ticker. Tom DataFrame (aldrig undantag) vid fel —
    och felet cachas inte."""
    try:
        return _ohlcv_raw(ticker, period)
    except Exception as exc:
        log.warning("market_prices: %s %s misslyckades: %s", ticker, period, exc)
        return pd.DataFrame()


def close(ticker: str, period: str = "1y") -> pd.Series:
    df = ohlcv(ticker, period)
    if df.empty or "Close" not in df:
        return pd.Series(dtype=float)
    return df["Close"].dropna()


@_cache
def _batch_raw(tickers: tuple, period: str) -> dict:
    import yfinance as yf
    raw = yf.download(list(tickers), period=period, auto_adjust=True, progress=False,
                      group_by="ticker", threads=True)
    out: dict = {}
    if raw is None or len(raw) == 0:
        raise _NoData(f"batch {period}: tom")
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                s = raw[t]["Close"].dropna() if "Close" in raw[t] else pd.Series(dtype=float)
                out[t] = s
    elif len(tickers) == 1 and "Close" in raw:
        out[tickers[0]] = raw["Close"].dropna()
    if not any(len(s) for s in out.values()):
        raise _NoData(f"batch {period}: inga kurser")
    return out


def _batch(tickers: tuple, period: str) -> dict:
    try:
        return _batch_raw(tickers, period)
    except Exception as exc:
        log.warning("market_prices: batch %s misslyckades: %s", period, exc)
        return {}


def closes(tickers: Iterable[str], period: str = "1y") -> dict:
    """{ticker: Close-serie} ur en batch-nedladdning (saknade tickers utelämnas)."""
    tup = tuple(dict.fromkeys(str(t).strip() for t in tickers if t))
    return dict(_batch(tup, period)) if tup else {}


def clear() -> None:
    """Töm BARA prisdatacachen (inte hela st.cache_data)."""
    for fn in (_ohlcv_raw, _batch_raw):
        try:
            fn.clear()
        except Exception:
            pass
