"""
gist_storage.py — Persistent holdings storage via GitHub Gist.
Falls back to local JSON if no GitHub token is configured.
"""
import json
import os
import streamlit as st
import requests
from typing import Optional

GIST_ID = "50348cb5b9e325c8ae91439763d5f144"
GIST_FILENAME = "holdings_init.json"
GIST_API_URL = f"https://api.github.com/gists/{GIST_ID}"
LOCAL_FALLBACK = ".holdings_data.json"


def _get_github_token() -> Optional[str]:
    """Get GitHub token from Streamlit secrets."""
    try:
        return st.secrets.get("GITHUB_TOKEN", None)
    except Exception:
        return None


def load_holdings() -> dict:
    """
    Load holdings from GitHub Gist.
    Fallback chain: Gist → local JSON file → empty default.
    Also caches in session_state for fast repeated reads.
    """
    # Fast path: already loaded this session
    if "holdings_data" in st.session_state:
        return st.session_state["holdings_data"]

    data = None

    # Try GitHub Gist first
    token = _get_github_token()
    if token:
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
            }
            r = requests.get(GIST_API_URL, headers=headers, timeout=10)
            if r.status_code == 200:
                gist = r.json()
                content = gist.get("files", {}).get(GIST_FILENAME, {}).get("content", "")
                if content:
                    data = json.loads(content)
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

    # Default empty structure
    if data is None:
        data = {"swing": [], "ovtlyr": [], "long": []}

    # Ensure all keys exist
    for key in ("swing", "ovtlyr", "long"):
        if key not in data:
            data[key] = []

    st.session_state["holdings_data"] = data
    return data


def save_holdings(data: dict) -> bool:
    """
    Save holdings to GitHub Gist + local file + session_state.
    Returns True if gist save succeeded.
    """
    # Always update session_state immediately
    st.session_state["holdings_data"] = data

    # Always save local fallback
    try:
        with open(LOCAL_FALLBACK, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass

    # Save to GitHub Gist
    token = _get_github_token()
    if not token:
        return False

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
        }
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
