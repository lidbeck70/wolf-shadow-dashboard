"""
ember/cache.py
Persist EMBER scan results to a GitHub Gist (+ local fallback), mirroring
the contrarian_alpha.cache pattern so scheduled GitHub Actions can publish
results that the (local or cloud) panel reads instantly.
"""
from __future__ import annotations
import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_GIST_ID        = os.environ.get("EMBER_GIST_ID", "")
_GIST_FILENAME  = "ember_results.json"
_GIST_API_URL   = f"https://api.github.com/gists/{_GIST_ID}" if _GIST_ID else ""
_LOCAL_FALLBACK = os.path.join(os.path.dirname(__file__), ".ember_results.json")


def _get_github_token() -> Optional[str]:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GIST_TOKEN")


def _auth_header(token: str) -> dict:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def _setup_to_dict(r) -> dict:
    """Serialize an EmberSetupResult to a compact JSON-safe dict."""
    return {
        "ticker": r.ticker, "typ": r.typ, "sektor": r.sektor,
        "trend_pass": r.trend_pass, "entry_pass": r.entry_pass,
        "notrade_clear": r.notrade_clear, "eligible": r.eligible,
        "price": r.price, "entry": r.entry, "stop": r.stop,
        "t1": r.t1, "t2": r.t2, "rr": r.rr, "shares": r.shares,
        "atr14": r.atr14, "ema20": r.ema20,
        "cykel_label": r.cykel_label, "percentile_10y": r.percentile_10y,
        "hat_score": r.hat_score, "necessity": r.necessity,
        "candle_pattern": r.candle_pattern,
        "asymmetry_score": r.asymmetry_score, "setup_quality": r.setup_quality,
        "macro_total": (r.macro.total if r.macro else None),
        "sentiment_total": (r.sentiment.total if r.sentiment else None),
        "error": r.error,
    }


def save_ember_results(scan_result) -> bool:
    """Persist EMBER scan to Gist + local fallback. Returns True on Gist success."""
    payload = {
        "timestamp": scan_result.timestamp.isoformat() if hasattr(scan_result.timestamp, "isoformat") else str(scan_result.timestamp),
        "eligible":    [_setup_to_dict(r) for r in scan_result.eligible],
        "near_misses": [_setup_to_dict(r) for r in scan_result.near_misses],
    }
    try:
        with open(_LOCAL_FALLBACK, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as e:
        logger.debug("Local EMBER write failed: %s", e)

    token = _get_github_token()
    if not token or not _GIST_API_URL:
        return False
    try:
        import requests
        body = {"files": {_GIST_FILENAME: {"content": json.dumps(payload, indent=2, default=str)}}}
        r = requests.patch(_GIST_API_URL, headers=_auth_header(token), json=body, timeout=15)
        if r.status_code == 200:
            logger.info("EMBER results saved to Gist (%d eligible)", len(payload["eligible"]))
            return True
        logger.warning("EMBER Gist save failed: HTTP %d", r.status_code)
    except Exception as e:
        logger.warning("EMBER Gist save exception: %s", e)
    return False


def load_ember_results() -> dict:
    """Load last-saved EMBER results. Priority: Gist -> local -> empty."""
    if _GIST_API_URL:
        try:
            import requests
            r = requests.get(_GIST_API_URL, timeout=10)
            if r.status_code == 200:
                content = r.json().get("files", {}).get(_GIST_FILENAME, {}).get("content", "")
                if content:
                    data = json.loads(content)
                    if data.get("eligible") is not None:
                        return data
        except Exception as e:
            logger.debug("EMBER Gist load failed: %s", e)
    try:
        if os.path.exists(_LOCAL_FALLBACK):
            with open(_LOCAL_FALLBACK, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.debug("EMBER local load failed: %s", e)
    return {"eligible": [], "near_misses": [], "timestamp": None}
