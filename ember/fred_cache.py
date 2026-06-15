"""
ember/fred_cache.py
Disk-cached FRED T10Y2Y fetch.

fetch_t10y2y_values() -> (vals: list[float], is_stale: bool)

- 12h TTL disk cache (daily FRED updates don't warrant more frequent fetches)
- Timeout 20s, 1 retry with 2s backoff
- Returns stale cache on fetch failure rather than raising
- Raises only when no cache exists and all fetch attempts fail
"""
from __future__ import annotations

import io
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(tempfile.gettempdir()) / "ember_fred_t10y2y.json"
_CACHE_TTL  = 604_800  # 7 days (T10Y2Y moves slowly; weekly refresh is plenty)
_TIMEOUT    = 12       # seconds per attempt (fail fast, serve cache)
_RETRIES    = 2        # extra retry attempts after first failure


def _read_cache() -> Optional[tuple[list[float], float]]:
    """Return (vals, timestamp) from disk, or None if absent/corrupt."""
    try:
        if _CACHE_FILE.exists():
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            return data["vals"], float(data["ts"])
    except Exception:
        pass
    return None


def _write_cache(vals: list[float]) -> None:
    try:
        _CACHE_FILE.write_text(
            json.dumps({"ts": time.time(), "vals": vals}), encoding="utf-8"
        )
    except Exception as exc:
        logger.debug("fred_cache: write failed: %s", exc)


def _fetch_from_fred(url: str) -> list[float]:
    import requests
    resp = requests.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), skiprows=1, names=["date", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if df.empty:
        raise ValueError("T10Y2Y: empty result from FRED")
    return df["value"].tolist()


def fetch_t10y2y_values(url: str) -> tuple[list[float], bool]:
    """
    Return (values, is_stale).

    is_stale=False  — fresh fetch or unexpired cache
    is_stale=True   — stale cache served because FRED was unreachable
    Raises RuntimeError only if no cache exists and all fetches fail.
    """
    # Return fresh cache if still valid
    cached = _read_cache()
    if cached is not None:
        vals, ts = cached
        if time.time() - ts < _CACHE_TTL:
            return vals, False

    # Attempt fetch with one retry
    last_exc: Optional[Exception] = None
    for attempt in range(_RETRIES + 1):
        try:
            vals = _fetch_from_fred(url)
            _write_cache(vals)
            return vals, False
        except Exception as exc:
            last_exc = exc
            logger.debug("fred_cache: fetch attempt %d failed: %s", attempt + 1, exc)
            if attempt < _RETRIES:
                time.sleep(2)

    # Serve stale cache if available
    if cached is not None:
        logger.debug("fred_cache: serving stale cache (age %.0fh)",
                     (time.time() - cached[1]) / 3600)
        return cached[0], True

    raise RuntimeError(f"FRED T10Y2Y ej tillgänglig och ingen cachad data: {last_exc}")
