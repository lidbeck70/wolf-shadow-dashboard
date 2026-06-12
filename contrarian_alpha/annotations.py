"""
contrarian_alpha/annotations.py
Persistent user-entered discipline annotations per ticker.

Stores: invalidation_text, invalidation_price, probable_catalyst.
Backend: local JSON (.ca_annotations.json) + optional Gist sidecar.
Backward-compatible: missing keys default to empty string / None.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Storage config ────────────────────────────────────────────────────────────

_LOCAL_PATH   = os.path.join(os.path.dirname(__file__), ".ca_annotations.json")
_GIST_ID      = "50348cb5b9e325c8ae91439763d5f144"
_GIST_FILE    = "ca_annotations.json"
_GIST_API_URL = f"https://api.github.com/gists/{_GIST_ID}"

# Fields stored per ticker
_FIELDS = ("invalidation_text", "invalidation_price", "probable_catalyst")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_token() -> Optional[str]:
    try:
        import streamlit as st
        token = st.secrets.get("GITHUB_TOKEN", None)
        if token:
            return str(token).strip()
    except Exception:
        pass
    return None


def _auth_header(token: str) -> dict:
    prefix = "Bearer" if token.startswith("github_pat_") else "token"
    return {"Authorization": f"{prefix} {token}",
            "Accept": "application/vnd.github.v3+json"}


# ── Public API ────────────────────────────────────────────────────────────────

def load_annotations() -> dict:
    """
    Load all annotations.
    Returns dict: {ticker: {invalidation_text, invalidation_price, probable_catalyst}}.
    """
    # 1. Local file
    if os.path.exists(_LOCAL_PATH):
        try:
            with open(_LOCAL_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.debug("load_annotations local: %s", exc)

    # 2. Gist fallback
    try:
        import requests
        r = requests.get(_GIST_API_URL, timeout=8)
        if r.status_code == 200:
            content = r.json().get("files", {}).get(_GIST_FILE, {}).get("content", "")
            if content:
                data = json.loads(content)
                if isinstance(data, dict):
                    # cache locally
                    try:
                        with open(_LOCAL_PATH, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                    except Exception:
                        pass
                    return data
    except Exception as exc:
        logger.debug("load_annotations gist: %s", exc)

    return {}


def save_annotation(
    ticker: str,
    invalidation_text: str = "",
    invalidation_price: Optional[float] = None,
    probable_catalyst: str = "",
) -> None:
    """
    Upsert annotation for a single ticker and persist.
    """
    data = load_annotations()
    data[ticker] = {
        "invalidation_text":  invalidation_text,
        "invalidation_price": invalidation_price,
        "probable_catalyst":  probable_catalyst,
    }

    # Save locally
    try:
        with open(_LOCAL_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning("save_annotation local write: %s", exc)

    # Push to Gist (best-effort)
    token = _get_token()
    if token:
        try:
            import requests
            payload = {"files": {_GIST_FILE: {"content": json.dumps(data, indent=2)}}}
            requests.patch(_GIST_API_URL, headers=_auth_header(token),
                           json=payload, timeout=8)
        except Exception as exc:
            logger.debug("save_annotation gist push: %s", exc)


def get_annotation(ticker: str) -> dict:
    """Return annotation dict for one ticker, with safe defaults."""
    ann = load_annotations().get(ticker, {})
    return {
        "invalidation_text":  ann.get("invalidation_text", ""),
        "invalidation_price": ann.get("invalidation_price"),
        "probable_catalyst":  ann.get("probable_catalyst", ""),
    }
