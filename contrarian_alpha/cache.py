"""
cache.py — Multi-domain TTL cache + Gist persistence for Contrarian Alpha (Fas 8).

Four named cache buckets with distinct TTLs:
  fundamentals  24 h  FCF, D/E, EBITDA, Altman Z
  price          1 h  SMA50/200, 52w range, volume stats
  sentiment      6 h  analyst upgrades, short interest, sector ETF flow
  regime         1 h  Viking/OVTLYR regime colour

Gist persistence:
  Screener results are serialised to JSON and stored in a GitHub Gist so they
  survive Streamlit redeploys.  Uses the same Gist + auth pattern as
  gist_storage.py (holdings).  Falls back to a local .json file when the Gist
  is unavailable.

Auto-scan scheduler:
  should_auto_scan()  — True once per calendar day after 07:30 CEST
  mark_auto_scanned() — persist the timestamp after a successful scan
  send_morning_alert() — push top-3 results via alerts.engine.send_alert
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─── TTL constants (seconds) ─────────────────────────────────────────────────

TTL_FUNDAMENTALS = 86_400   # 24 h
TTL_PRICE        =  3_600   # 1 h
TTL_SENTIMENT    = 21_600   # 6 h
TTL_REGIME       =  3_600   # 1 h

# ─── Core TTL cache ──────────────────────────────────────────────────────────

class TTLCache:
    """In-memory key/value store with per-entry expiry and LRU-ish eviction."""

    def __init__(self, ttl: int, max_entries: int = 500):
        self._store: dict[str, tuple[Any, float]] = {}
        self._ttl         = ttl
        self._max_entries = max_entries

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        if len(self._store) >= self._max_entries and key not in self._store:
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        self._store[key] = (value, time.time())

    def clear(self) -> None:
        self._store.clear()

    def size(self) -> int:
        return len(self._store)

    def ttl_remaining(self, key: str) -> float:
        entry = self._store.get(key)
        if entry is None:
            return 0.0
        _, ts = entry
        remaining = self._ttl - (time.time() - ts)
        return max(0.0, remaining)


# ─── Named bucket singletons ─────────────────────────────────────────────────

_fundamentals: TTLCache = TTLCache(TTL_FUNDAMENTALS)
_price:        TTLCache = TTLCache(TTL_PRICE)
_sentiment:    TTLCache = TTLCache(TTL_SENTIMENT)
_regime:       TTLCache = TTLCache(TTL_REGIME)

# ─── Delisted-ticker blacklist (session-scoped, never expires) ─────────────────
_DELISTED_BLACKLIST: set[str] = set()


def is_delisted(ticker: str) -> bool:
    return ticker in _DELISTED_BLACKLIST


def mark_delisted(ticker: str) -> None:
    _DELISTED_BLACKLIST.add(ticker)


def delisted_count() -> int:
    return len(_DELISTED_BLACKLIST)

_BUCKETS: dict[str, TTLCache] = {
    "fundamentals": _fundamentals,
    "price":        _price,
    "sentiment":    _sentiment,
    "regime":       _regime,
}

# ─── Public accessor helpers ─────────────────────────────────────────────────

def get_fundamentals(key: str) -> Optional[Any]:
    return _fundamentals.get(key)

def set_fundamentals(key: str, value: Any) -> None:
    _fundamentals.set(key, value)

def get_price(key: str) -> Optional[Any]:
    return _price.get(key)

def set_price(key: str, value: Any) -> None:
    _price.set(key, value)

def get_sentiment(key: str) -> Optional[Any]:
    return _sentiment.get(key)

def set_sentiment(key: str, value: Any) -> None:
    _sentiment.set(key, value)

def get_regime(key: str) -> Optional[Any]:
    return _regime.get(key)

def set_regime(key: str, value: Any) -> None:
    _regime.set(key, value)


def clear_all() -> None:
    for c in _BUCKETS.values():
        c.clear()

def cache_stats() -> dict[str, dict]:
    return {
        name: {"size": cache.size(), "ttl_s": cache._ttl}
        for name, cache in _BUCKETS.items()
    }


# ─── @st.cache_data wrappers ─────────────────────────────────────────────────
# Only wraps functions whose inputs are hashable primitives (strings/ints).
# Used in hate.py and catalyst.py as the outermost layer; TTLCache above is
# the inner layer for non-Streamlit call paths (e.g., background threads).

try:
    import streamlit as _st

    @_st.cache_data(ttl=TTL_SENTIMENT, show_spinner=False)
    def cached_analyst_data(ticker: str, api_key: str = "") -> dict:
        """Streamlit-cached wrapper for hate.fetch_analyst_data (6 h TTL)."""
        from contrarian_alpha.hate import fetch_analyst_data
        return fetch_analyst_data(ticker, api_key or None)

    @_st.cache_data(ttl=TTL_SENTIMENT, show_spinner=False)
    def cached_short_data(ticker: str, api_key: str = "") -> dict:
        """Streamlit-cached wrapper for hate.fetch_short_data (6 h TTL)."""
        from contrarian_alpha.hate import fetch_short_data
        return fetch_short_data(ticker, api_key or None)

    @_st.cache_data(ttl=TTL_SENTIMENT, show_spinner=False)
    def cached_sector_data(sector_etf: str, market_etf: str = "SPY") -> dict:
        """Streamlit-cached wrapper for hate.fetch_sector_data (6 h TTL)."""
        from contrarian_alpha.hate import fetch_sector_data
        return fetch_sector_data(sector_etf, market_etf)

    @_st.cache_data(ttl=TTL_REGIME, show_spinner=False)
    def cached_viking_regime(ticker: str) -> tuple[bool, str, None]:
        """
        Streamlit-cached Viking Regime lookup via yfinance (1 h TTL).
        Returns (is_green, color, ovtlyr_nine=None).
        """
        from contrarian_alpha.catalyst import get_viking_regime
        return get_viking_regime(ticker=ticker)

    @_st.cache_data(ttl=TTL_PRICE, show_spinner=False)
    def cached_price_metrics(ticker: str, period: str = "1y") -> dict:
        """
        Streamlit-cached price-metric dict (1 h TTL).
        Fetches via yfinance and runs _compute_price_metrics.
        """
        if is_delisted(ticker):
            return {}
        try:
            import yfinance as yf
            import logging as _logging
            import pandas as pd
            from contrarian_alpha.engine import _compute_price_metrics

            _yf_log = _logging.getLogger("yfinance")
            _prev = _yf_log.level
            _yf_log.setLevel(_logging.CRITICAL)
            try:
                df = yf.Ticker(ticker).history(period=period, auto_adjust=True, progress=False)
            finally:
                _yf_log.setLevel(_prev)
            if df is None or df.empty:
                mark_delisted(ticker)
                return {}
            df.index  = df.index.tz_localize(None) if hasattr(df.index, "tz") and df.index.tz else df.index
            df.columns = [c.capitalize() for c in df.columns]
            return _compute_price_metrics(df)
        except Exception as e:
            mark_delisted(ticker)
            logger.debug("cached_price_metrics failed for %s: %s", ticker, e)
            return {}

    _ST_CACHE_AVAILABLE = True

except ImportError:
    _ST_CACHE_AVAILABLE = False

    def cached_analyst_data(ticker: str, api_key: str = "") -> dict:
        cached = get_sentiment(f"analyst:{ticker}")
        if cached is not None:
            return cached
        from contrarian_alpha.hate import fetch_analyst_data
        result = fetch_analyst_data(ticker, api_key or None)
        set_sentiment(f"analyst:{ticker}", result)
        return result

    def cached_short_data(ticker: str, api_key: str = "") -> dict:
        cached = get_sentiment(f"short:{ticker}")
        if cached is not None:
            return cached
        from contrarian_alpha.hate import fetch_short_data
        result = fetch_short_data(ticker, api_key or None)
        set_sentiment(f"short:{ticker}", result)
        return result

    def cached_sector_data(sector_etf: str, market_etf: str = "SPY") -> dict:
        cached = get_sentiment(f"sector:{sector_etf}:{market_etf}")
        if cached is not None:
            return cached
        from contrarian_alpha.hate import fetch_sector_data
        result = fetch_sector_data(sector_etf, market_etf)
        set_sentiment(f"sector:{sector_etf}:{market_etf}", result)
        return result

    def cached_viking_regime(ticker: str) -> tuple[bool, str, None]:
        cached = get_regime(f"viking:{ticker}")
        if cached is not None:
            return cached
        from contrarian_alpha.catalyst import get_viking_regime
        result = get_viking_regime(ticker=ticker)
        set_regime(f"viking:{ticker}", result)
        return result

    def cached_price_metrics(ticker: str, period: str = "1y") -> dict:
        if is_delisted(ticker):
            return {}
        cached = get_price(f"price:{ticker}")
        if cached is not None:
            return cached
        try:
            import yfinance as yf
            import logging as _logging
            from contrarian_alpha.engine import _compute_price_metrics
            _yf_log = _logging.getLogger("yfinance")
            _prev = _yf_log.level
            _yf_log.setLevel(_logging.CRITICAL)
            try:
                df = yf.Ticker(ticker).history(period=period, auto_adjust=True, progress=False)
            finally:
                _yf_log.setLevel(_prev)
            if df is None or df.empty:
                mark_delisted(ticker)
                return {}
            df.columns = [c.capitalize() for c in df.columns]
            result = _compute_price_metrics(df)
            set_price(f"price:{ticker}", result)
            return result
        except Exception:
            mark_delisted(ticker)
            return {}


# ─── Gist persistence for screener results ───────────────────────────────────

_GIST_ID          = os.environ.get("CA_GIST_ID", "50348cb5b9e325c8ae91439763d5f144")
_GIST_FILENAME    = "contrarian_alpha_results.json"
_GIST_API_URL     = f"https://api.github.com/gists/{_GIST_ID}"
_LOCAL_FALLBACK   = os.path.join(os.path.dirname(__file__), ".ca_results.json")

_EMPTY_RESULTS: dict = {"results": [], "timestamp": "", "universe_count": 0}


def _get_github_token() -> Optional[str]:
    try:
        import streamlit as st
        token = st.secrets.get("GITHUB_TOKEN", None)
        if token:
            return str(token).strip()
        token = st.secrets["GITHUB_TOKEN"]
        if token:
            return str(token).strip()
    except Exception:
        pass
    return os.environ.get("GITHUB_TOKEN")


def _auth_header(token: str) -> dict:
    prefix = "Bearer" if token.startswith("github_pat_") else "token"
    return {
        "Authorization": f"{prefix} {token}",
        "Accept":        "application/vnd.github.v3+json",
    }


def _result_to_dict(r) -> dict:
    """Serialise a ContrairianAlphaResult to a JSON-safe dict."""
    return {
        "ticker":           r.ticker,
        "name":             r.name,
        "market":           r.market,
        "sector":           r.sector,
        "composite_score":  r.composite_score,
        "rank":             r.rank,
        "necessity_score":  r.necessity_score,
        "hat_score":        r.hat_score,
        "strength_score":   r.strength_score,
        "catalyst_score":   r.catalyst_score,
        "viking_bonus_pts": r.viking_bonus_pts,
        "close":            r.close,
        "sma200":           r.sma200,
        "fcf_m":            r.fcf_m,
        "ebitda_pct":       r.ebitda_pct,
        "debt_equity":      r.debt_equity,
        "altman_z":         r.altman_z,
        "viking_color":     (r.catalyst_result.viking_regime_color
                             if r.catalyst_result else "unknown"),
        "all_flags":        r.all_flags,
        "timestamp":        r.timestamp,
    }


def save_screener_results(pipeline_result) -> bool:
    """
    Persist top pipeline results to Gist + local fallback.
    pipeline_result is a PipelineResult from engine.run_pipeline().
    Returns True if Gist write succeeded.
    """
    payload = {
        "timestamp":      pipeline_result.timestamp,
        "universe_count": pipeline_result.universe_count,
        "run_duration_s": pipeline_result.run_duration_s,
        "results": [_result_to_dict(r) for r in pipeline_result.results],
    }

    # Always write local fallback first
    try:
        with open(_LOCAL_FALLBACK, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as e:
        logger.debug("Local CA results write failed: %s", e)

    token = _get_github_token()
    if not token:
        return False

    try:
        import requests
        headers = _auth_header(token)
        body = {
            "files": {
                _GIST_FILENAME: {
                    "content": json.dumps(payload, indent=2, default=str)
                }
            }
        }
        r = requests.patch(_GIST_API_URL, headers=headers, json=body, timeout=15)
        if r.status_code == 200:
            logger.info("CA results saved to Gist (%d results)", len(payload["results"]))
            return True
        logger.warning("Gist CA save failed: HTTP %d", r.status_code)
    except Exception as e:
        logger.warning("Gist CA save exception: %s", e)
    return False


def load_screener_results() -> dict:
    """
    Load last-saved screener results.
    Priority: Gist → local file → empty.
    Returns dict with keys: results (list[dict]), timestamp, universe_count.
    """
    # Try Gist (no auth needed for public gist)
    try:
        import requests
        r = requests.get(_GIST_API_URL, timeout=10)
        if r.status_code == 200:
            gist = r.json()
            content = gist.get("files", {}).get(_GIST_FILENAME, {}).get("content", "")
            if content:
                data = json.loads(content)
                if data.get("results"):
                    return data
    except Exception as e:
        logger.debug("Gist CA load failed: %s", e)

    # Local fallback
    try:
        if os.path.exists(_LOCAL_FALLBACK):
            with open(_LOCAL_FALLBACK, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("results"):
                return data
    except Exception as e:
        logger.debug("Local CA load failed: %s", e)

    return dict(_EMPTY_RESULTS)


def gist_storage_status() -> str:
    """Return 'cloud_ok' | 'cloud_error_<code>' | 'local_only'."""
    token = _get_github_token()
    if not token:
        return "local_only"
    try:
        import requests
        r = requests.get(_GIST_API_URL, headers=_auth_header(token), timeout=5)
        return "cloud_ok" if r.status_code == 200 else f"cloud_error_{r.status_code}"
    except Exception as e:
        return f"cloud_error_{e}"


# ─── Auto-scan scheduler (07:30 CEST daily) ──────────────────────────────────

_CEST_OFFSET   = timedelta(hours=2)   # CEST = UTC+2 (summer); change to +1 for CET
_SCAN_HOUR     = 7
_SCAN_MINUTE   = 30
_AUTO_SCAN_KEY = ".ca_auto_scan_date"


def _today_scan_dt_utc() -> datetime:
    """Return today's 07:30 CEST expressed in UTC."""
    now_cest = datetime.now(tz=timezone.utc) + _CEST_OFFSET
    trigger   = now_cest.replace(hour=_SCAN_HOUR, minute=_SCAN_MINUTE,
                                  second=0, microsecond=0)
    return trigger - _CEST_OFFSET   # back to UTC


def should_auto_scan() -> bool:
    """
    Return True if:
      - Current CEST time is >= 07:30 today, AND
      - Auto-scan has NOT already been triggered today.
    Checks a local marker file to persist the state across page reloads.
    """
    now_utc   = datetime.now(tz=timezone.utc)
    trigger   = _today_scan_dt_utc()

    if now_utc < trigger:
        return False

    # Read last-scanned date
    try:
        if os.path.exists(_AUTO_SCAN_KEY):
            with open(_AUTO_SCAN_KEY) as f:
                last_date = f.read().strip()
            today_cest = (now_utc + _CEST_OFFSET).strftime("%Y-%m-%d")
            if last_date == today_cest:
                return False   # already ran today
    except Exception:
        pass

    return True


def mark_auto_scanned() -> None:
    """Persist today's CEST date as the last auto-scan date."""
    try:
        today_cest = (datetime.now(tz=timezone.utc) + _CEST_OFFSET).strftime("%Y-%m-%d")
        with open(_AUTO_SCAN_KEY, "w") as f:
            f.write(today_cest)
    except Exception as e:
        logger.debug("mark_auto_scanned write failed: %s", e)


def send_morning_alert(pipeline_result, channels: list[str] | None = None) -> dict:
    """
    Send morning alert via alerts.engine with top-3 contrarian picks.
    channels defaults to ["discord"] if not specified.
    Returns per-channel result dict from send_alert.
    """
    if channels is None:
        channels = ["discord"]

    top3 = pipeline_result.results[:3]
    if not top3:
        return {}

    lines = ["🐺 **Contrarian Alpha — Morgonscan**", ""]
    for r in top3:
        regime_icon = "🟢" if (r.catalyst_result and r.catalyst_result.viking_regime_green) else "⚪"
        lines.append(
            f"{regime_icon} **{r.ticker}** {r.name}  "
            f"— Score {r.composite_score:.1f}  "
            f"(N{r.necessity_score:.0f} H{r.hat_score:.0f} "
            f"S{r.strength_score:.0f} C{r.catalyst_score:.0f})"
        )

    lines += [
        "",
        f"Universum: {pipeline_result.universe_count} bolag  "
        f"| Passerade: {pipeline_result.composite_ranked}  "
        f"| Tid: {pipeline_result.run_duration_s:.1f}s",
        f"_{pipeline_result.timestamp[:16]} UTC_",
    ]
    message = "\n".join(lines)

    metadata = {
        "source":   "contrarian_alpha",
        "top_3":    [r.ticker for r in top3],
        "scores":   [r.composite_score for r in top3],
        "universe": pipeline_result.universe_count,
    }

    try:
        # alerts module lives in wolf-shadow-dashboard (dashboard root or dev sibling)
        import sys
        from pathlib import Path
        for _bd_candidate in [
            Path(__file__).parent.parent,
            Path(__file__).parent.parent.parent / "Documents" / "wolf-shadow-dashboard",
        ]:
            if _bd_candidate.exists() and str(_bd_candidate) not in sys.path:
                sys.path.insert(0, str(_bd_candidate))
        from alerts.engine import send_alert
        return send_alert(message, channels=channels, metadata=metadata)
    except Exception as e:
        logger.warning("send_morning_alert failed: %s", e)
        return {}
