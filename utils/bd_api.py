"""
utils/bd_api.py
===============
Börsdata PRO+ API client — lightweight facade over the REST v1 API.

Base URL : https://apiservice.borsdata.se/v1
Rate limit: 100 calls / 10 s  (this client targets ≤ 90 calls / 10 s)
Docs      : https://github.com/Borsdata-Sweden/API

Quick start
-----------
    from utils.bd_api import BDClient, load_api_key

    client = BDClient(load_api_key())
    df     = client.get_price_history("VOLV-B.ST", period="1y")
    fund   = client.get_fundamentals("VOLV-B.ST")
    meta   = client.get_metadata("VOLV-B.ST")

Environment variable
--------------------
    BD_API_KEY=<your-key>
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL     = "https://apiservice.borsdata.se/v1"
_RATE_WINDOW  = 10.0   # seconds
_RATE_MAX     = 90     # calls per window (hard limit is 100; leave headroom)
_TIMEOUT      = 15     # HTTP timeout in seconds
_MAX_RETRIES  = 3


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def load_api_key() -> str:
    """
    Read the Börsdata API key from the environment.

    Checks (in order):
      1. BD_API_KEY environment variable
      2. BORSDATA_API_KEY environment variable (legacy name)
      3. Streamlit secrets (key "BD_API_KEY" or "BORSDATA_API_KEY")

    Returns an empty string if no key is found.
    """
    key = os.environ.get("BD_API_KEY", "") or os.environ.get("BORSDATA_API_KEY", "")
    if key:
        return key
    try:
        import streamlit as st
        return (
            st.secrets.get("BD_API_KEY", "")
            or st.secrets.get("BORSDATA_API_KEY", "")
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Period helper
# ---------------------------------------------------------------------------

def _period_to_from_date(period: str) -> Optional[str]:
    """Convert a period string to a 'YYYY-MM-DD' from_date for the API."""
    _map = {
        "1w":  7,
        "1m":  30,
        "3m":  90,
        "6m":  180,
        "1y":  365,
        "2y":  730,
        "3y":  1095,
        "5y":  1825,
        "10y": 3650,
        "20y": 7300,
        "max": None,
    }
    days = _map.get(period.lower())
    if days is None:
        return None  # "max" → no date filter
    return (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Rate limiter (thread-safe token bucket)
# ---------------------------------------------------------------------------

class _RateLimiter:
    def __init__(self, max_calls: int = _RATE_MAX, window: float = _RATE_WINDOW):
        self._max    = max_calls
        self._window = window
        self._calls: list[float] = []
        self._lock   = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                self._calls = [t for t in self._calls if now - t < self._window]
                if len(self._calls) < self._max:
                    self._calls.append(now)
                    return
                oldest = self._calls[0]
                wait   = self._window - (now - oldest) + 0.05
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Module-level LRU-cached data fetchers
# These are keyed by api_key so multi-key usage is safe.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _cached_instruments(api_key: str) -> List[dict]:
    """Fetch and cache the full instrument list for a given API key."""
    try:
        resp = requests.get(
            f"{_BASE_URL}/instruments",
            params={"authKey": api_key},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("instruments", [])
    except Exception as exc:
        logger.warning("bd_api: instruments fetch failed: %s", exc)
        return []


@functools.lru_cache(maxsize=8)
def _cached_sectors(api_key: str) -> List[dict]:
    """Fetch and cache the sector list for a given API key."""
    try:
        resp = requests.get(
            f"{_BASE_URL}/sectors",
            params={"authKey": api_key},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("sectors", [])
    except Exception as exc:
        logger.warning("bd_api: sectors fetch failed: %s", exc)
        return []


@functools.lru_cache(maxsize=256)
def _cached_kpi_screener(api_key: str, kpi_id: int) -> Dict[int, float]:
    """Fetch one screener KPI for ALL instruments; returns {ins_id: value}."""
    try:
        resp = requests.get(
            f"{_BASE_URL}/instruments/kpis/{kpi_id}/last/latest",
            params={"authKey": api_key},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        result: Dict[int, float] = {}
        for entry in resp.json().get("values", []):
            iid = entry.get("i")
            val = entry.get("n")
            if iid is not None and val is not None:
                result[iid] = val
        return result
    except Exception as exc:
        logger.warning("bd_api: KPI screener %d failed: %s", kpi_id, exc)
        return {}


# Module-level rate limiter shared by all cached fetchers so concurrent
# ThreadPoolExecutor workers respect the same 90-call/10-s budget.
_module_limiter = _RateLimiter()

# Batch-result TTL cache: {(api_key, frozenset_key, period): {"ts": float, "data": dict}}
_batch_cache: Dict[tuple, dict] = {}
_BATCH_CACHE_TTL = 3600  # seconds


@functools.lru_cache(maxsize=512)
def _cached_stockprices(
    api_key: str,
    ins_id: int,
    from_date: Optional[str],
) -> Optional[List[dict]]:
    """
    Fetch raw stockprice records for one Börsdata instrument.
    LRU-cached for the process lifetime; call clear_cache() to invalidate.

    Parameters
    ----------
    api_key   : Börsdata API key (part of cache key)
    ins_id    : Börsdata instrument ID
    from_date : 'YYYY-MM-DD' start date, or None for maximum history

    Returns
    -------
    List of raw bar dicts {d, o, h, l, c, v} or None on failure.
    """
    _module_limiter.acquire()
    try:
        params: Dict[str, Any] = {"authKey": api_key, "maxcount": 5040}
        if from_date:
            params["from"] = from_date
        resp = requests.get(
            f"{_BASE_URL}/instruments/{ins_id}/stockprices",
            params=params,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        bars = resp.json().get("stockPricesList", [])
        return bars if bars else None
    except Exception as exc:
        logger.warning("bd_api: stockprices fetch failed ins_id=%d: %s", ins_id, exc)
        return None


# ---------------------------------------------------------------------------
# KPI ID reference (subset used by BDClient)
# ---------------------------------------------------------------------------

_KPI = {
    "pe":               2,
    "ps":               3,
    "pb":               4,
    "ev_ebitda":        11,
    "ev_ebit":          10,
    "p_fcf":            76,
    "roe":              33,
    "roa":              34,
    "roic":             37,
    "gross_margin":     28,
    "operating_margin": 29,
    "profit_margin":    30,
    "fcf_margin":       31,
    "revenue_growth":   94,
    "earnings_growth":  97,
    "equity_ratio":     39,
    "debt_to_equity":   40,
    "net_debt_ebitda":  42,
    "current_ratio":    44,
    "dividend_yield":   1,
    "f_score":          167,
    "rs_rank":          192,
    "market_cap":       50,
    "revenue_m":        53,
    "fcf_m":            63,
    "short_selling":    207,
}

# KPIs stored as percentage in API (divide by 100 to get decimal)
_KPI_PCT = {"roe", "roa", "roic", "gross_margin", "operating_margin",
            "profit_margin", "fcf_margin", "revenue_growth", "earnings_growth",
            "equity_ratio", "debt_to_equity", "dividend_yield"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bars_to_df(raw: List[dict], ticker: str = "") -> Optional["pd.DataFrame"]:
    """Convert a raw Börsdata stockPricesList to a pandas OHLCV DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        return None
    if not raw:
        return None
    rows = [
        {
            "Date":   bar.get("d"),
            "Open":   bar.get("o"),
            "High":   bar.get("h"),
            "Low":    bar.get("l"),
            "Close":  bar.get("c"),
            "Volume": bar.get("v", 0),
        }
        for bar in raw
    ]
    try:
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"])
        df = df.set_index("Date").sort_index()
        return df if not df.empty else None
    except Exception as exc:
        logger.warning("bd_api: _bars_to_df failed for %r: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# BDClient
# ---------------------------------------------------------------------------

class BDClient:
    """
    Börsdata PRO+ API client.

    Parameters
    ----------
    api_key : str
        Your Börsdata API key.  Use ``load_api_key()`` to read from env.

    Example
    -------
    >>> client = BDClient(load_api_key())
    >>> df = client.get_price_history("VOLV-B.ST", period="1y")
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key or ""
        self._limiter = _RateLimiter()
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        GET a Börsdata endpoint with rate-limiting and retries.
        Returns parsed JSON or None on failure.
        """
        if not self.api_key:
            logger.error("bd_api: no API key configured")
            return None

        full_params = {"authKey": self.api_key}
        if params:
            full_params.update(params)

        url = f"{_BASE_URL}{path}"

        for attempt in range(_MAX_RETRIES):
            try:
                self._limiter.acquire()
                resp = self._session.get(url, params=full_params, timeout=_TIMEOUT)

                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 5))
                    logger.warning("bd_api: 429 rate-limited, waiting %d s", wait)
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning("bd_api: %d server error, retry in %d s",
                                   resp.status_code, wait)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.Timeout:
                logger.warning("bd_api: timeout on %s (attempt %d)", path, attempt + 1)
                time.sleep(2 ** attempt)
            except Exception as exc:
                logger.warning("bd_api: request failed on %s: %s", path, exc)
                return None

        logger.error("bd_api: all %d retries failed for %s", _MAX_RETRIES, path)
        return None

    # ------------------------------------------------------------------
    # Instrument resolution
    # ------------------------------------------------------------------

    def _resolve_id(self, ticker: str) -> Optional[int]:
        """
        Resolve a ticker string to a Börsdata instrument ID.

        Handles yfinance suffixes (.ST, .OL, .CO, .HE) and dash→space
        conversions (VOLV-B → VOLV B).

        Returns None if the ticker cannot be matched.
        """
        instruments = _cached_instruments(self.api_key)
        if not instruments:
            return None

        # Build lookup on first call (cached alongside instruments list)
        ticker_map: Dict[str, int] = {}
        for inst in instruments:
            iid    = inst.get("insId")
            t      = inst.get("ticker", "")
            name   = inst.get("name", "")
            if iid is None:
                continue
            if t:
                ticker_map[t.upper()] = iid
            if name:
                ticker_map[name.upper()] = iid

        query = ticker.upper().strip()

        if query in ticker_map:
            return ticker_map[query]

        # Strip yfinance exchange suffixes
        for suffix in (".ST", ".OL", ".CO", ".HE", ".AS", ".DE", ".PA", ".L"):
            if query.endswith(suffix):
                base   = query[: -len(suffix)]
                spaced = base.replace("-", " ")
                for candidate in (base, spaced):
                    if candidate in ticker_map:
                        return ticker_map[candidate]
                # Strip share class (VOLV-B → VOLV)
                parts = base.split("-")
                if len(parts) == 2 and len(parts[1]) <= 2:
                    if parts[0] in ticker_map:
                        return ticker_map[parts[0]]
                break

        # Partial match fallback
        for key, iid in ticker_map.items():
            if query in key or key in query:
                return iid

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_price_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> Optional["pd.DataFrame"]:
        """
        Fetch OHLCV price history for a ticker.

        Parameters
        ----------
        ticker   : yfinance-style ticker string (e.g. "VOLV-B.ST", "EQNR.OL")
        period   : lookback window — "1w", "1m", "3m", "6m", "1y", "2y",
                   "3y", "5y", "10y", "20y", "max"
        interval : "1d" (daily) — Börsdata REST v1 only provides daily OHLCV

        Returns
        -------
        pd.DataFrame with DatetimeIndex and columns Open, High, Low, Close, Volume
        Returns None if the ticker cannot be resolved or data fetch fails.

        Note
        ----
        The ``interval`` parameter is accepted for API compatibility but the
        Börsdata REST v1 endpoint only supplies end-of-day (daily) bars.
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("bd_api: pandas is required for get_price_history")
            return None

        if interval != "1d":
            logger.warning(
                "bd_api: Börsdata REST v1 only provides daily bars; "
                "interval=%r ignored", interval
            )

        ins_id = self._resolve_id(ticker)
        if ins_id is None:
            logger.warning("bd_api: could not resolve ticker %r", ticker)
            return None

        from_date = _period_to_from_date(period)
        raw = _cached_stockprices(self.api_key, ins_id, from_date)
        if not raw:
            return None

        return _bars_to_df(raw, ticker)

    def get_price_history_batch(
        self,
        tickers: List[str],
        period: str = "1y",
        interval: str = "1d",
        max_workers: int = 8,
    ) -> Dict[str, "pd.DataFrame"]:
        """
        Fetch OHLCV price history for multiple tickers in parallel.

        Resolves all tickers to Börsdata instrument IDs in one pass (using the
        cached instrument list), then fires concurrent HTTP requests via a
        ThreadPoolExecutor.  Each per-ticker response is stored in the
        ``_cached_stockprices`` LRU cache so subsequent single calls are free.

        Tickers that Börsdata cannot resolve (e.g., US equities not in the
        Nordic database) are batched to yfinance as a fallback.

        The assembled result dict is stored in ``_batch_cache`` with a 1-hour
        TTL to avoid redundant round-trips on Streamlit reruns.

        Parameters
        ----------
        tickers     : list of yfinance-style ticker strings
        period      : lookback window — same values as ``get_price_history()``
        interval    : "1d" only (Börsdata v1 is end-of-day)
        max_workers : max concurrent threads for Börsdata HTTP requests

        Returns
        -------
        dict ``{ticker: pd.DataFrame}`` — only tickers with ≥ 10 bars are
        included.  Missing tickers are silently omitted.
        """
        try:
            import pandas as pd
        except ImportError:
            return {}

        if not tickers:
            return {}

        tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order
        from_date = _period_to_from_date(period)

        # --- Batch-level TTL cache -------------------------------------------
        _cache_key = (self.api_key, frozenset(tickers), period)
        _entry = _batch_cache.get(_cache_key)
        if _entry and time.time() - _entry["ts"] < _BATCH_CACHE_TTL:
            logger.debug("bd_api: batch cache hit (%d tickers)", len(tickers))
            return _entry["data"]

        # --- Phase 1: resolve ticker → ins_id in one pass --------------------
        bd_map: Dict[str, int] = {}    # ticker → ins_id  (BD can serve these)
        missing: List[str]     = []    # tickers BD cannot resolve

        for ticker in tickers:
            ins_id = self._resolve_id(ticker)
            if ins_id is not None:
                bd_map[ticker] = ins_id
            else:
                missing.append(ticker)

        logger.debug(
            "bd_api: batch — %d via BD, %d via yfinance fallback",
            len(bd_map), len(missing),
        )

        # --- Phase 2: concurrent Börsdata fetch ------------------------------
        result: Dict[str, pd.DataFrame] = {}

        def _fetch_one(ticker: str, ins_id: int) -> Optional[pd.DataFrame]:
            raw = _cached_stockprices(self.api_key, ins_id, from_date)
            return _bars_to_df(raw, ticker) if raw else None

        if bd_map:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_map = {
                    pool.submit(_fetch_one, t, iid): t
                    for t, iid in bd_map.items()
                }
                for future in concurrent.futures.as_completed(future_map):
                    ticker = future_map[future]
                    try:
                        df = future.result()
                        if df is not None and len(df) >= 10:
                            result[ticker] = df
                        else:
                            missing.append(ticker)  # BD returned empty → try yfinance
                    except Exception as exc:
                        logger.warning("bd_api: batch worker failed %r: %s", ticker, exc)
                        missing.append(ticker)

        # --- Phase 3: yfinance fallback for unresolvable / empty tickers -----
        if missing:
            try:
                import yfinance as yf

                yf_raw = yf.download(
                    missing,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    progress=False,
                    threads=True,
                    auto_adjust=True,
                )

                if missing and isinstance(yf_raw.columns, pd.MultiIndex):
                    for ticker in missing:
                        try:
                            df = yf_raw[ticker].dropna(subset=["Close"])
                            if len(df) >= 10:
                                if df.index.tz is not None:
                                    df.index = df.index.tz_localize(None)
                                result[ticker] = df
                        except (KeyError, TypeError):
                            pass
                elif not yf_raw.empty and len(missing) == 1:
                    # Single-ticker download — no MultiIndex
                    df = yf_raw.dropna(subset=["Close"])
                    if len(df) >= 10:
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        result[missing[0]] = df

            except Exception as exc:
                logger.warning("bd_api: yfinance fallback failed: %s", exc)

        # --- Store in batch-level TTL cache ----------------------------------
        _batch_cache[_cache_key] = {"ts": time.time(), "data": result}
        logger.debug("bd_api: batch complete — %d/%d tickers have data",
                     len(result), len(tickers))
        return result

    def get_fundamentals(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a fundamental snapshot for a ticker.

        Uses the screener KPI endpoint (one call per KPI returns all instruments,
        so values are served from the module-level LRU cache after the first call).

        Returns
        -------
        dict with normalised keys:
          pe, ps, pb, ev_ebitda, ev_ebit, p_fcf,
          roe, roa, roic, gross_margin, operating_margin, profit_margin,
          fcf_margin, revenue_growth, earnings_growth, equity_ratio,
          debt_to_equity, net_debt_ebitda, current_ratio, dividend_yield,
          f_score, rs_rank, market_cap, revenue_m, fcf_m, short_selling
        Returns None if ticker cannot be resolved.
        """
        ins_id = self._resolve_id(ticker)
        if ins_id is None:
            logger.warning("bd_api: could not resolve ticker %r for fundamentals", ticker)
            return None

        result: Dict[str, Any] = {"ticker": ticker, "ins_id": ins_id}

        for name, kpi_id in _KPI.items():
            try:
                all_vals = _cached_kpi_screener(self.api_key, kpi_id)
                raw_val  = all_vals.get(ins_id)
                if raw_val is not None and name in _KPI_PCT:
                    result[name] = raw_val / 100.0
                else:
                    result[name] = raw_val
            except Exception as exc:
                logger.debug("bd_api: KPI %s failed for %r: %s", name, ticker, exc)
                result[name] = None

        return result

    def get_metadata(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch instrument metadata for a ticker.

        Returns
        -------
        dict with keys:
          ins_id, ticker, name, isin, market_id, sector_id, country_id,
          instrument_type, stock_price_currency, report_currency
        Returns None if the ticker cannot be resolved.
        """
        ins_id = self._resolve_id(ticker)
        if ins_id is None:
            logger.warning("bd_api: could not resolve ticker %r for metadata", ticker)
            return None

        instruments = _cached_instruments(self.api_key)
        for inst in instruments:
            if inst.get("insId") == ins_id:
                return {
                    "ins_id":               inst.get("insId"),
                    "ticker":               inst.get("ticker"),
                    "name":                 inst.get("name"),
                    "isin":                 inst.get("isin"),
                    "market_id":            inst.get("marketId"),
                    "sector_id":            inst.get("sectorId"),
                    "country_id":           inst.get("countryId"),
                    "instrument_type":      inst.get("instrumentType"),
                    "stock_price_currency": inst.get("stockPriceCurrency"),
                    "report_currency":      inst.get("reportCurrency"),
                }

        return None

    def get_sectors(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch the list of sectors from Börsdata.

        Returns
        -------
        list of dicts with keys: id, name
        Returns None if the API call fails.
        """
        sectors = _cached_sectors(self.api_key)
        if not sectors:
            return None
        return [{"id": s.get("id"), "name": s.get("name")} for s in sectors]

    def get_etf_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch ETF-specific data for a ticker.

        Tries the Börsdata ETF endpoint first; falls back to regular
        instrument metadata + latest price if the ETF endpoint is unavailable.

        Returns
        -------
        dict with keys:
          ticker, ins_id, name, last_price, currency,
          nav (Net Asset Value, if available), etf_type (if available)
        Returns None if the ticker cannot be resolved.
        """
        ins_id = self._resolve_id(ticker)
        if ins_id is None:
            logger.warning("bd_api: could not resolve ticker %r for ETF data", ticker)
            return None

        result: Dict[str, Any] = {"ticker": ticker, "ins_id": ins_id}

        # Try ETF-specific endpoint
        etf_data = self._get(f"/instruments/{ins_id}/stockprices/last")
        if etf_data:
            prices = etf_data.get("stockPricesList", [])
            if prices:
                latest = prices[-1]
                result["last_price"] = latest.get("c")
                result["last_date"]  = latest.get("d")

        # Enrich with instrument metadata
        meta = self.get_metadata(ticker)
        if meta:
            result["name"]     = meta.get("name")
            result["currency"] = meta.get("stock_price_currency")
            result["market_id"] = meta.get("market_id")

        # Try to fetch NAV/ETF details from the dedicated endpoint (Pro+ feature)
        etf_detail = self._get(f"/instruments/etf/{ins_id}")
        if etf_detail:
            result["nav"]      = etf_detail.get("nav")
            result["etf_type"] = etf_detail.get("etfType") or etf_detail.get("type")

        return result if result else None

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """
        Invalidate all module-level caches (LRU + batch TTL).
        Call this to force fresh data after an API key change or on demand.
        """
        _cached_instruments.cache_clear()
        _cached_sectors.cache_clear()
        _cached_kpi_screener.cache_clear()
        _cached_stockprices.cache_clear()
        _batch_cache.clear()
        logger.info("bd_api: all caches cleared")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "configured" if self.api_key else "NO KEY"
        return f"BDClient(status={status})"
