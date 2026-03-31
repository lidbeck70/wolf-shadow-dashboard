"""
borsdata_api.py
Central Börsdata REST-API wrapper with authentication, rate limiting and caching.

Börsdata Pro+ API — https://github.com/Borsdata-Sweden/API
Base URL: https://apiservice.borsdata.se/v1
Rate limit: 100 calls / 10 seconds, target < 10K calls / 24h.

Usage:
    from borsdata_api import BorsdataAPI
    api = BorsdataAPI()                     # reads key from env / secrets
    instruments = api.get_instruments()     # list of all instruments
    reports = api.get_reports(ins_id=77)    # annual reports for Ericsson
    pe_hist = api.get_kpi_history(77, 2)    # P/E history for Ericsson
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://apiservice.borsdata.se/v1"

# Rate-limit guard: max 90 calls per 10-second window (leave headroom)
_RATE_WINDOW = 10.0     # seconds
_RATE_MAX_CALLS = 90     # slightly below the hard 100 limit


# ---------------------------------------------------------------------------
# KPI ID reference  (most useful — full list at wiki/Kpi-Screener-List)
# ---------------------------------------------------------------------------

KPI = {
    # ── Valuation ──
    "pe":               2,
    "ps":               3,
    "pb":               4,
    "ev_ebit":          10,
    "ev_ebitda":        11,
    "ev_s":             15,
    "ev_fcf":           13,
    "p_fcf":            76,
    "peg":              19,
    # ── Profitability ──
    "gross_margin":     28,
    "operating_margin": 29,
    "profit_margin":    30,
    "fcf_margin":       31,
    "ebitda_margin":    32,
    "roe":              33,
    "roa":              34,
    "roic":             37,
    "roc":              36,
    # ── Growth ──
    "revenue_growth":   94,
    "earnings_growth":  97,
    "dividend_growth":  98,
    "fcf_growth":       23,   # FCF / Aktie (used as proxy)
    # ── Debt & Leverage ──
    "equity_ratio":     39,
    "debt_to_equity":   40,
    "net_debt_pct":     41,
    "net_debt_ebitda":  42,
    "current_ratio":    44,
    # ── Per-share ──
    "revenue_per_share": 5,
    "eps":              6,
    "dividend_ps":      7,
    "bvps":             8,
    # ── Absolute values (millions) ──
    "revenue_m":        53,
    "ebitda_m":         54,
    "ebit_m":           55,
    "earnings_m":       56,
    "total_assets_m":   57,
    "total_equity_m":   58,
    "net_debt_m":       60,
    "ocf_m":            62,
    "fcf_m":            63,
    "market_cap":       50,
    "ev":               49,
    # ── Dividend ──
    "dividend_yield":   1,
    "dividend_payout":  20,
    # ── Stability ──
    "earnings_stab":    174,
    "ebit_stab":        175,
    "ebitda_stab":      176,
    "dividend_stab":    177,
    "ocf_stab":         178,
    "fcf_stab":         179,
    # ── Technicals / Strategies ──
    "magic_formula":    163,
    "f_score":          167,
    "graham":           164,
    "rs_rank":          192,
    "ma200":            157,
    "rsi":              159,
    # ── Stock price perf ──
    "stock_perf":       151,
    # ── Short selling ──
    "short_selling":    207,
}


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

class _DiskCache:
    """Simple JSON file cache keyed on URL hash.  TTL in seconds."""

    def __init__(self, cache_dir: str, default_ttl: int = 3600):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = default_ttl

    def _key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def get(self, url: str) -> Optional[Any]:
        path = self._dir / f"{self._key(url)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ts = data.get("_ts", 0)
            if time.time() - ts > self._ttl:
                path.unlink(missing_ok=True)
                return None
            return data.get("payload")
        except Exception:
            return None

    def set(self, url: str, payload: Any) -> None:
        path = self._dir / f"{self._key(url)}.json"
        try:
            path.write_text(
                json.dumps({"_ts": time.time(), "payload": payload}, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Cache write failed: %s", exc)

    def clear(self) -> int:
        """Remove all cached files. Returns count of removed files."""
        count = 0
        for f in self._dir.glob("*.json"):
            f.unlink(missing_ok=True)
            count += 1
        return count


# ---------------------------------------------------------------------------
# Rate limiter (token-bucket, thread-safe)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Token-bucket rate limiter: max `max_calls` per `window` seconds."""

    def __init__(self, max_calls: int = _RATE_MAX_CALLS, window: float = _RATE_WINDOW):
        self._max = max_calls
        self._window = window
        self._calls: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a call slot is available."""
        while True:
            with self._lock:
                now = time.time()
                # Purge old entries
                self._calls = [t for t in self._calls if now - t < self._window]
                if len(self._calls) < self._max:
                    self._calls.append(now)
                    return
                # Calculate wait
                oldest = self._calls[0]
                wait = self._window - (now - oldest) + 0.05
            logger.debug("Rate limit: sleeping %.2f s", wait)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def _resolve_api_key() -> str:
    """
    Resolve Börsdata API key from (in order):
      1. Environment variable BORSDATA_API_KEY
      2. Streamlit secrets
    Returns empty string if not found.
    """
    key = os.environ.get("BORSDATA_API_KEY", "")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("BORSDATA_API_KEY", "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main API class
# ---------------------------------------------------------------------------

class BorsdataAPI:
    """
    Börsdata REST-API v1 client.

    Features:
      • Automatic API key resolution (env → Streamlit secrets)
      • Rate limiting (90 calls / 10 s)
      • Disk cache (default 1 h TTL)
      • Retry with back-off on 429 / 5xx
      • Instrument lookup by ticker symbol
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | None = None,
        cache_ttl: int = 3600,
    ):
        self.api_key = api_key or _resolve_api_key()
        if not self.api_key:
            logger.warning(
                "No Börsdata API key found. Set BORSDATA_API_KEY env var or "
                "add it to Streamlit secrets."
            )

        cache_path = cache_dir or os.path.join(
            os.path.dirname(__file__), ".borsdata_cache"
        )
        self._cache = _DiskCache(cache_path, default_ttl=cache_ttl)
        self._limiter = _RateLimiter()
        self._session = requests.Session()
        self._session.headers.update({"accept": "application/json"})

        # Lazy-loaded instrument registry
        self._instruments: Optional[List[dict]] = None
        self._ticker_map: Optional[Dict[str, int]] = None
        self._id_map: Optional[Dict[int, dict]] = None

    @property
    def is_configured(self) -> bool:
        """True if an API key is available."""
        return bool(self.api_key)

    # ─── Low-level HTTP ───────────────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None, use_cache: bool = True) -> Any:
        """
        GET a Börsdata API endpoint.  Returns parsed JSON payload.
        Handles rate limiting, caching, and retries.
        """
        if not self.api_key:
            raise RuntimeError("Börsdata API key not configured")

        full_params = {"authKey": self.api_key}
        if params:
            full_params.update(params)

        url = f"{BASE_URL}{path}"
        cache_key = f"{url}?{'&'.join(f'{k}={v}' for k,v in sorted(full_params.items()) if k != 'authKey')}"

        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        self._limiter.acquire()

        for attempt in range(3):
            try:
                resp = self._session.get(url, params=full_params, timeout=15)

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.warning("429 rate limited, waiting %d s", retry_after)
                    time.sleep(retry_after)
                    self._limiter.acquire()
                    continue

                if resp.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning("%d server error, retry in %d s", resp.status_code, wait)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                self._cache.set(cache_key, data)
                return data

            except requests.exceptions.Timeout:
                logger.warning("Timeout on %s (attempt %d)", path, attempt + 1)
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Failed after 3 attempts: {path}")

    # ─── Instruments ──────────────────────────────────────────────────────

    def get_instruments(self, force_refresh: bool = False) -> List[dict]:
        """
        Fetch all instruments.  Cached in-memory for the session.

        Returns list of dicts with keys:
          insId, name, ticker, isin, stockPriceCurrency, reportCurrency,
          instrumentType, marketId, branchId, sectorId, countryId
        """
        if self._instruments and not force_refresh:
            return self._instruments

        data = self._get("/instruments", use_cache=not force_refresh)
        self._instruments = data.get("instruments", [])

        # Build lookup maps
        self._ticker_map = {}
        self._id_map = {}
        for inst in self._instruments:
            ins_id = inst.get("insId")
            ticker = inst.get("ticker", "")
            name = inst.get("name", "")
            self._id_map[ins_id] = inst
            # Map by ticker (case-insensitive)
            if ticker:
                self._ticker_map[ticker.upper()] = ins_id
            # Also map by name for fuzzy matching
            if name:
                self._ticker_map[name.upper()] = ins_id

        return self._instruments

    def get_markets(self) -> List[dict]:
        """Fetch market metadata (id → name mapping)."""
        data = self._get("/markets")
        return data.get("markets", [])

    def get_sectors(self) -> List[dict]:
        """Fetch sector metadata."""
        data = self._get("/sectors")
        return data.get("sectors", [])

    def get_countries(self) -> List[dict]:
        """Fetch country metadata."""
        data = self._get("/countries")
        return data.get("countries", [])

    def get_branches(self) -> List[dict]:
        """Fetch branch/industry metadata."""
        data = self._get("/branches")
        return data.get("branches", [])

    def resolve_instrument_id(self, ticker_or_name: str) -> Optional[int]:
        """
        Look up Börsdata instrument ID from a ticker symbol or company name.

        Handles common Nordic suffixes:
          VOLV-B.ST → strips .ST and tries VOLV B, VOLV-B
          EQNR.OL  → strips .OL and tries EQNR
          NOVO-B.CO → strips .CO and tries NOVO B
          NOKIA.HE  → strips .HE and tries NOKIA

        Returns None if not found.
        """
        if self._ticker_map is None:
            self.get_instruments()

        assert self._ticker_map is not None

        query = ticker_or_name.upper().strip()

        # Direct match
        if query in self._ticker_map:
            return self._ticker_map[query]

        # Strip yfinance suffixes (.ST, .OL, .CO, .HE, .AS, .DE, .PA)
        for suffix in (".ST", ".OL", ".CO", ".HE", ".AS", ".DE", ".PA"):
            if query.endswith(suffix):
                base = query[:-len(suffix)]
                # Try direct
                if base in self._ticker_map:
                    return self._ticker_map[base]
                # Replace dash with space (VOLV-B → VOLV B)
                spaced = base.replace("-", " ")
                if spaced in self._ticker_map:
                    return self._ticker_map[spaced]
                # Try just the base without share class
                parts = base.split("-")
                if len(parts) == 2 and len(parts[1]) == 1:
                    if parts[0] in self._ticker_map:
                        return self._ticker_map[parts[0]]
                break

        # Fuzzy: try partial match
        for key, ins_id in self._ticker_map.items():
            if query in key or key in query:
                return ins_id

        return None

    def get_instrument_info(self, ins_id: int) -> Optional[dict]:
        """Get cached instrument info by ID."""
        if self._id_map is None:
            self.get_instruments()
        assert self._id_map is not None
        return self._id_map.get(ins_id)

    # ─── Reports ──────────────────────────────────────────────────────────

    def get_reports(
        self,
        ins_id: int,
        report_type: str = "year",
        max_count: int = 20,
        original: bool = True,
    ) -> List[dict]:
        """
        Fetch report data for an instrument.

        Parameters
        ----------
        ins_id      : Börsdata instrument ID
        report_type : "year", "r12", or "quarter"
        max_count   : max reports (20 for year, 40 for r12/quarter)
        original    : True = original report currency, False = converted

        Returns list of report dicts with fields:
          revenues, grossIncome, operatingIncome, profitBeforeTax,
          profitToEquityHolders, earningsPerShare, numberOfShares,
          dividend, netDebt, freeCashFlow, totalEquity, totalAssets, etc.
        """
        params = {
            "maxcount": max_count,
            "original": 1 if original else 0,
        }
        if report_type in ("year", "r12", "quarter"):
            data = self._get(f"/instruments/{ins_id}/reports/{report_type}", params)
        else:
            data = self._get(f"/instruments/{ins_id}/reports", params)

        # Response keys vary by type
        for key in ("reportsYear", "reportsR12", "reportsQuarter", "reports"):
            if key in data:
                return data[key]
        return []

    def get_reports_batch(
        self,
        ins_ids: List[int],
        max_count: int = 20,
        original: bool = True,
    ) -> Dict[int, List[dict]]:
        """
        Fetch reports for multiple instruments (max 50 per call).
        Returns dict of ins_id → list of report dicts.
        """
        results: Dict[int, List[dict]] = {}

        # Batch in groups of 50
        for i in range(0, len(ins_ids), 50):
            batch = ins_ids[i:i+50]
            id_list = ",".join(str(x) for x in batch)
            params = {
                "instList": id_list,
                "maxcount": max_count,
                "original": 1 if original else 0,
            }
            data = self._get("/instruments/reports", params)

            # Parse — data contains all reports grouped by instrument
            for key in ("reportsYear", "reportsR12", "reportsQuarter", "reports"):
                if key in data:
                    for report in data[key]:
                        iid = report.get("instrumentId") or report.get("instrument")
                        if iid is not None:
                            results.setdefault(iid, []).append(report)
                    break

        return results

    # ─── Stock prices ─────────────────────────────────────────────────────

    def get_stockprices(
        self,
        ins_id: int,
        from_date: str | None = None,
        max_count: int = 0,
    ) -> List[dict]:
        """
        Fetch end-of-day stock prices for an instrument.

        Parameters
        ----------
        ins_id    : instrument ID
        from_date : "YYYY-MM-DD" start date (optional)
        max_count : 0 = default (~1y), up to 5040 (20 years)

        Returns list of dicts: {d: date, c: close, h: high, l: low, o: open, v: volume}
        """
        params = {}
        if from_date:
            params["from"] = from_date
        if max_count > 0:
            params["maxcount"] = max_count

        data = self._get(f"/instruments/{ins_id}/stockprices", params)
        return data.get("stockPricesList", [])

    def get_last_stockprices(self) -> List[dict]:
        """Get the latest stock price for ALL instruments (single API call)."""
        data = self._get("/instruments/stockprices/last")
        return data.get("stockPricesList", [])

    # ─── KPI Screener ─────────────────────────────────────────────────────

    def get_kpi_screener(
        self,
        kpi_id: int,
        calc_group: str = "last",
        calc: str = "latest",
    ) -> List[dict]:
        """
        Get a specific KPI for ALL instruments (screener endpoint).

        Parameters
        ----------
        kpi_id     : KPI ID (see KPI dict above)
        calc_group : "last", "1month", "3month", "6month", "1year", "3year", "5year"
        calc       : "latest", "mean", "low", "high", "sum", "count", "return"

        Returns list of {i: instrument_id, n: numeric_value, s: string_value}
        """
        data = self._get(f"/instruments/kpis/{kpi_id}/{calc_group}/{calc}")
        return data.get("values", [])

    def get_kpi_screener_dict(
        self,
        kpi_id: int,
        calc_group: str = "last",
        calc: str = "latest",
    ) -> Dict[int, float]:
        """
        Convenience: get KPI screener as {instrument_id: value} dict.
        Filters out None/NaN values.
        """
        raw = self.get_kpi_screener(kpi_id, calc_group, calc)
        result = {}
        for entry in raw:
            ins_id = entry.get("i")
            val = entry.get("n")
            if ins_id is not None and val is not None:
                result[ins_id] = val
        return result

    # ─── KPI History ──────────────────────────────────────────────────────

    def get_kpi_history(
        self,
        ins_id: int,
        kpi_id: int,
        report_type: str = "year",
        price_type: str = "mean",
    ) -> List[dict]:
        """
        Get historical KPI values for a single instrument.

        Parameters
        ----------
        ins_id      : instrument ID
        kpi_id      : KPI ID
        report_type : "year", "r12", "quarter"
        price_type  : "mean", "low", "high"

        Returns list of {y: year, p: period, v: value}
        """
        data = self._get(
            f"/instruments/{ins_id}/kpis/{kpi_id}/{report_type}/{price_type}/history"
        )
        return data.get("values", [])

    def get_kpi_history_batch(
        self,
        ins_ids: List[int],
        kpi_id: int,
        report_type: str = "year",
        price_type: str = "mean",
    ) -> Dict[int, List[dict]]:
        """
        Get historical KPI for multiple instruments (max 50 per call).
        Returns dict of ins_id → list of {y, p, v}.
        """
        results: Dict[int, List[dict]] = {}

        for i in range(0, len(ins_ids), 50):
            batch = ins_ids[i:i+50]
            id_list = ",".join(str(x) for x in batch)
            data = self._get(
                f"/instruments/kpis/{kpi_id}/{report_type}/{price_type}/history",
                params={"instList": id_list},
            )
            # Response has values grouped by instrument
            for entry in data.get("values", []):
                iid = entry.get("i")
                if iid is not None:
                    results.setdefault(iid, []).append(entry)

        return results

    # ─── Convenience: Multi-KPI snapshot ──────────────────────────────────

    def get_fundamentals_snapshot(self, ins_id: int) -> dict:
        """
        Fetch a comprehensive fundamental snapshot for one instrument.
        Combines screener KPIs and latest report data.

        Returns a flat dict with normalised keys matching CAGR scoring needs:
          roe, roa, roic, ev_ebitda, pb, pe, ps,
          gross_margin, operating_margin, profit_margin, fcf_margin,
          revenue_growth, earnings_growth,
          equity_ratio, debt_to_equity, net_debt_ebitda, current_ratio,
          earnings_stability, fcf_stability,
          market_cap, ev, revenue_m, ebitda_m, ebit_m, fcf_m,
          dividend_yield, dividend_payout, f_score, magic_formula,
          rs_rank, short_selling
        """
        # We'll fetch the key KPIs from the screener (one call per KPI
        # returns ALL instruments — very efficient for batch).
        # For a single instrument, use the single-instrument KPI history
        # endpoint for the "last" value.

        snapshot: dict = {"ins_id": ins_id}

        # Map of output key → (kpi_id, divisor)
        # divisor: 1 = raw value, 100 = percent→decimal
        kpi_fetch = {
            "roe":                  (KPI["roe"], 100),
            "roa":                  (KPI["roa"], 100),
            "roic":                 (KPI["roic"], 100),
            "roc":                  (KPI["roc"], 100),
            "ev_ebitda":            (KPI["ev_ebitda"], 1),
            "pb":                   (KPI["pb"], 1),
            "pe":                   (KPI["pe"], 1),
            "ps":                   (KPI["ps"], 1),
            "ev_ebit":              (KPI["ev_ebit"], 1),
            "p_fcf":                (KPI["p_fcf"], 1),
            "gross_margin":         (KPI["gross_margin"], 100),
            "operating_margin":     (KPI["operating_margin"], 100),
            "profit_margin":        (KPI["profit_margin"], 100),
            "fcf_margin":           (KPI["fcf_margin"], 100),
            "ebitda_margin":        (KPI["ebitda_margin"], 100),
            "revenue_growth":       (KPI["revenue_growth"], 100),
            "earnings_growth":      (KPI["earnings_growth"], 100),
            "equity_ratio":         (KPI["equity_ratio"], 100),
            "debt_to_equity":       (KPI["debt_to_equity"], 100),
            "net_debt_ebitda":      (KPI["net_debt_ebitda"], 1),
            "current_ratio":        (KPI["current_ratio"], 1),
            "dividend_yield":       (KPI["dividend_yield"], 100),
            "dividend_payout":      (KPI["dividend_payout"], 100),
            "earnings_stability":   (KPI["earnings_stab"], 1),
            "fcf_stability":        (KPI["fcf_stab"], 1),
            "f_score":              (KPI["f_score"], 1),
            "magic_formula":        (KPI["magic_formula"], 1),
            "rs_rank":              (KPI["rs_rank"], 1),
            "market_cap":           (KPI["market_cap"], 1),
            "ev":                   (KPI["ev"], 1),
            "revenue_m":            (KPI["revenue_m"], 1),
            "ebitda_m":             (KPI["ebitda_m"], 1),
            "ebit_m":               (KPI["ebit_m"], 1),
            "fcf_m":                (KPI["fcf_m"], 1),
            "ocf_m":                (KPI["ocf_m"], 1),
            "net_debt_m":           (KPI["net_debt_m"], 1),
            "short_selling":        (KPI["short_selling"], 1),
        }

        for key, (kpi_id, divisor) in kpi_fetch.items():
            try:
                hist = self.get_kpi_history(ins_id, kpi_id, "r12", "mean")
                if hist:
                    # Take the latest value (last entry)
                    latest = hist[-1]
                    val = latest.get("v")
                    if val is not None:
                        snapshot[key] = val / divisor if divisor != 1 else val
                    else:
                        snapshot[key] = None
                else:
                    snapshot[key] = None
            except Exception as exc:
                logger.debug("KPI %s (id=%d) failed for ins %d: %s", key, kpi_id, ins_id, exc)
                snapshot[key] = None

        return snapshot

    def get_fundamentals_snapshot_fast(self, ins_ids: List[int]) -> Dict[int, dict]:
        """
        Batch-fetch fundamental snapshots using screener endpoints.
        Much more efficient than per-instrument calls — one API call per KPI
        returns values for ALL instruments.

        Returns dict of ins_id → snapshot dict.
        """
        # Initialise empty snapshots
        snapshots: Dict[int, dict] = {iid: {"ins_id": iid} for iid in ins_ids}
        id_set = set(ins_ids)

        # KPIs to fetch with divisor
        kpi_fetch = {
            "roe":                  (KPI["roe"], 100),
            "roa":                  (KPI["roa"], 100),
            "roic":                 (KPI["roic"], 100),
            "ev_ebitda":            (KPI["ev_ebitda"], 1),
            "pb":                   (KPI["pb"], 1),
            "pe":                   (KPI["pe"], 1),
            "ps":                   (KPI["ps"], 1),
            "p_fcf":                (KPI["p_fcf"], 1),
            "ev_ebit":              (KPI["ev_ebit"], 1),
            "gross_margin":         (KPI["gross_margin"], 100),
            "operating_margin":     (KPI["operating_margin"], 100),
            "profit_margin":        (KPI["profit_margin"], 100),
            "fcf_margin":           (KPI["fcf_margin"], 100),
            "ebitda_margin":        (KPI["ebitda_margin"], 100),
            "revenue_growth":       (KPI["revenue_growth"], 100),
            "earnings_growth":      (KPI["earnings_growth"], 100),
            "equity_ratio":         (KPI["equity_ratio"], 100),
            "debt_to_equity":       (KPI["debt_to_equity"], 100),
            "net_debt_ebitda":      (KPI["net_debt_ebitda"], 1),
            "current_ratio":        (KPI["current_ratio"], 1),
            "dividend_yield":       (KPI["dividend_yield"], 100),
            "earnings_stability":   (KPI["earnings_stab"], 1),
            "fcf_stability":        (KPI["fcf_stab"], 1),
            "f_score":              (KPI["f_score"], 1),
            "magic_formula":        (KPI["magic_formula"], 1),
            "rs_rank":              (KPI["rs_rank"], 1),
            "market_cap":           (KPI["market_cap"], 1),
            "revenue_m":            (KPI["revenue_m"], 1),
            "fcf_m":                (KPI["fcf_m"], 1),
            "net_debt_m":           (KPI["net_debt_m"], 1),
            "short_selling":        (KPI["short_selling"], 1),
        }

        for key, (kpi_id, divisor) in kpi_fetch.items():
            try:
                all_vals = self.get_kpi_screener(kpi_id, "last", "latest")
                for entry in all_vals:
                    iid = entry.get("i")
                    if iid in id_set:
                        val = entry.get("n")
                        if val is not None:
                            snapshots[iid][key] = val / divisor if divisor != 1 else val
                        else:
                            snapshots[iid][key] = None
            except Exception as exc:
                logger.warning("Screener KPI %s failed: %s", key, exc)
                for iid in ins_ids:
                    snapshots[iid].setdefault(key, None)

        return snapshots

    # ─── Revenue / Earnings growth history ────────────────────────────────

    def get_growth_history(
        self,
        ins_id: int,
        years: int = 10,
    ) -> dict:
        """
        Compute multi-year revenue and earnings growth from report data.

        Returns:
          revenue_cagr_5y, revenue_cagr_10y,
          earnings_cagr_5y, earnings_cagr_10y,
          revenue_history: [(year, value), ...],
          earnings_history: [(year, value), ...]
        """
        reports = self.get_reports(ins_id, "year", max_count=min(years + 1, 20))

        # Sort by year
        reports.sort(key=lambda r: r.get("year", 0))

        rev_hist = []
        earn_hist = []
        for r in reports:
            y = r.get("year")
            rev = r.get("revenues") or r.get("netSales")
            earn = r.get("profitToEquityHolders") or r.get("earningsPerShare")
            if y:
                if rev is not None:
                    rev_hist.append((y, rev))
                if earn is not None:
                    earn_hist.append((y, earn))

        def _cagr(values: list, n_years: int) -> Optional[float]:
            if len(values) < n_years + 1:
                return None
            end_val = values[-1][1]
            start_val = values[-(n_years + 1)][1]
            if start_val is None or end_val is None or start_val <= 0 or end_val <= 0:
                return None
            return (end_val / start_val) ** (1 / n_years) - 1

        return {
            "revenue_cagr_5y":  _cagr(rev_hist, 5),
            "revenue_cagr_10y": _cagr(rev_hist, 10),
            "earnings_cagr_5y": _cagr(earn_hist, 5),
            "earnings_cagr_10y": _cagr(earn_hist, 10),
            "revenue_history":  rev_hist,
            "earnings_history": earn_hist,
        }

    # ─── Cache management ─────────────────────────────────────────────────

    def clear_cache(self) -> int:
        """Clear the disk cache. Returns number of files removed."""
        self._instruments = None
        self._ticker_map = None
        self._id_map = None
        return self._cache.clear()

    # ─── Repr ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "configured" if self.is_configured else "NO KEY"
        return f"BorsdataAPI(status={status})"


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init)
# ---------------------------------------------------------------------------

_instance: Optional[BorsdataAPI] = None


def get_api() -> BorsdataAPI:
    """Get or create the module-level BorsdataAPI singleton."""
    global _instance
    if _instance is None:
        _instance = BorsdataAPI()
    return _instance
