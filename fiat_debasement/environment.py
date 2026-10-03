"""
fiat_debasement/environment.py — FIAT ENVIRONMENT: en liten indikator för
gruvbolagens granskningssidor (Low / Neutral / Elevated) ur Wolf Debasement
Index för USD.

Indikatorn är bakgrund — den används ALDRIG som köp- eller säljsignal och
påverkar inga regler. Cachas i processen sex timmar (fel: tio minuter).
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from fiat_debasement import config as cfg
from fiat_debasement import index as fx

TTL_OK_S, TTL_FAIL_S = 6 * 3600, 600
_CACHE: dict = {}
NA = "DATA UNAVAILABLE"
COLOR = {"Low": "#2d8a4e", "Neutral": "#d4943a", "Elevated": "#c44545", NA: "#6b7280"}


def level_of(value: Optional[float]) -> str:
    if value is None:
        return NA
    for limit, name in cfg.ENVIRONMENT_LEVELS:
        if value < limit:
            return name
    return cfg.ENVIRONMENT_LEVELS[-1][1]


def current(ccy: str = cfg.ENVIRONMENT_CCY, compute: Optional[Callable] = None) -> dict:
    """{currency, level, value, as_of} — cachad. compute(ccy) → IndexResult (för tester)."""
    hit = _CACHE.get(ccy)
    if hit and time.time() - hit[0] < (TTL_OK_S if hit[1]["value"] is not None else TTL_FAIL_S):
        return hit[1]
    try:
        r = (compute or fx.for_currency)(ccy)
        out = {"currency": ccy, "value": r.value, "as_of": r.as_of, "level": level_of(r.value)}
    except Exception:
        out = {"currency": ccy, "value": None, "as_of": None, "level": NA}
    _CACHE[ccy] = (time.time(), out)
    return out


def badge_html(env: dict) -> str:
    lvl = env.get("level", NA)
    col = COLOR.get(lvl, COLOR[NA])
    val = "" if env.get("value") is None else f" ({env['value']:.0f}/100)"
    return (f"<div style='font-size:0.78rem;margin:2px 0 8px;'><span style='color:#8a8578;letter-spacing:1px;'>"
            f"FIAT ENVIRONMENT {env.get('currency', '')}</span> "
            f"<span style='background:{col}22;color:{col};border:1px solid {col};border-radius:4px;"
            f"padding:1px 7px;font-weight:700;'>{lvl}{val}</span> "
            f"<span style='color:#8a8578;'>· bakgrund ur REGIME → Makro → 🐺 Fiat Debasement — "
            f"ingen köp- eller säljsignal</span></div>")


def render_badge(ccy: str = cfg.ENVIRONMENT_CCY) -> None:
    """Ritar indikatorn; ett fel får aldrig stoppa sidan den ligger på."""
    try:
        import streamlit as st
        st.markdown(badge_html(current(ccy)), unsafe_allow_html=True)
    except Exception:
        pass


def clear_cache() -> None:
    _CACHE.clear()
