"""
gist_storage.py — Persistent holdings storage via GitHub Gist.
Falls back to local JSON if gist is unreachable.

READ:  No auth needed (gist is accessible by URL).
WRITE: Requires GITHUB_TOKEN in Streamlit secrets.
"""
import json
import os
import requests
import streamlit as st
from typing import Optional

GIST_ID = "50348cb5b9e325c8ae91439763d5f144"
GIST_FILENAME = "holdings_init.json"
GIST_API_URL = f"https://api.github.com/gists/{GIST_ID}"
LOCAL_FALLBACK = ".holdings_data.json"
_EMPTY = {"swing": [], "ovtlyr": [], "long": []}


def _get_github_token() -> Optional[str]:
    """Get GitHub token from Streamlit secrets (multiple access methods)."""
    try:
        token = st.secrets.get("GITHUB_TOKEN", None)
        if token:
            return str(token).strip()
    except Exception:
        pass
    try:
        token = st.secrets["GITHUB_TOKEN"]
        if token:
            return str(token).strip()
    except Exception:
        pass
    return None


def _auth_header(token: str) -> dict:
    """Build Authorization header — Bearer for fine-grained PATs, token for classic."""
    prefix = "Bearer" if token.startswith("github_pat_") else "token"
    return {
        "Authorization": f"{prefix} {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def load_holdings() -> dict:
    """
    Load holdings. Priority: session_state > Gist > local file > empty.
    Gist READ works without auth.
    """
    if "holdings_data" in st.session_state:
        return st.session_state["holdings_data"]

    data = None

    # Try gist (no auth needed for reading)
    try:
        r = requests.get(GIST_API_URL, timeout=10)
        if r.status_code == 200:
            gist = r.json()
            content = gist.get("files", {}).get(GIST_FILENAME, {}).get("content", "")
            if content:
                parsed = json.loads(content)
                if any(parsed.get(k) for k in ("swing", "ovtlyr", "long")):
                    data = parsed
    except Exception:
        pass

    # Fallback to local file
    if data is None:
        try:
            if os.path.exists(LOCAL_FALLBACK):
                with open(LOCAL_FALLBACK) as f:
                    data = json.load(f)
        except Exception:
            pass

    if data is None:
        data = dict(_EMPTY)

    for key in ("swing", "ovtlyr", "long"):
        if key not in data:
            data[key] = []

    st.session_state["holdings_data"] = data
    return data


def save_holdings(data: dict) -> bool:
    """
    Save holdings to Gist + local file + session_state.
    Returns True if gist write succeeded.
    """
    st.session_state["holdings_data"] = data

    try:
        with open(LOCAL_FALLBACK, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass

    token = _get_github_token()
    if not token:
        return False

    try:
        headers = _auth_header(token)
        payload = {
            "files": {
                GIST_FILENAME: {
                    "content": json.dumps(data, indent=2, default=str)
                }
            }
        }
        r = requests.patch(GIST_API_URL, headers=headers, json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def get_storage_status() -> str:
    """Return storage status string for UI display."""
    token = _get_github_token()
    if token:
        try:
            headers = _auth_header(token)
            r = requests.get(GIST_API_URL, headers=headers, timeout=5)
            if r.status_code == 200:
                return "cloud_ok"
            else:
                return f"cloud_error_{r.status_code}"
        except Exception as e:
            return f"cloud_error_{e}"
    return "local_only"
