"""
tabs/home.py
============
Home tab — Mission Control layout for Nordic Arc Systems.

Shows: system pulse, four navigation zones, and recent alerts.
"""

from __future__ import annotations

import streamlit as st
from datetime import datetime

from ui.theme import inject_css, section_title, PALETTE as _P

_DIM   = _P["text_dim"]
_TEXT  = _P["text"]

_CYAN   = "#00E5FF"
_PURPLE = "#B400FF"
_EMBER  = "#FF6B3D"
_GREEN2 = "#2d8a4e"


# ── Color logic ───────────────────────────────────────────────────────────────

def _status_color(status: str) -> str:
    s = status.upper()
    if any(k in s for k in ("BULL", "OPTIMISM", "BELIEF", "HOPE")):
        return _GREEN2
    if any(k in s for k in ("BEAR", "PANIC", "CAPITULATION")):
        return _EMBER
    return _EMBER


# ── HTML builders ─────────────────────────────────────────────────────────────

def _pulse_card(label: str, value: str, color: str) -> str:
    return (
        f'<div style="background:#1A1F25;border:1px solid rgba(255,255,255,0.06);'
        f'border-left:3px solid {color};border-radius:8px;padding:14px 16px;">'
        f'<div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;'
        f'color:{_DIM};margin-bottom:6px;">{label}</div>'
        f'<div style="font-size:16px;font-weight:700;color:{color};">{value}</div>'
        f'</div>'
    )


def _zone_label(text: str, color: str) -> str:
    return (
        f'<div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;'
        f'color:{color};border-left:2px solid {color};padding-left:8px;'
        f'margin-bottom:12px;">{text}</div>'
    )


def _nav_card(title: str, desc: str, color: str) -> str:
    return (
        f'<div style="background:#1A1F25;border:1px solid rgba(255,255,255,0.06);'
        f'border-left:2px solid {color};border-radius:8px;padding:14px;">'
        f'<div style="font-size:13px;font-weight:700;color:#E8EDF2;margin-bottom:4px;">{title}</div>'
        f'<div style="font-size:11px;color:#6B7280;line-height:1.5;">{desc}</div>'
        f'</div>'
    )


# ── System Pulse ──────────────────────────────────────────────────────────────

def _render_system_pulse() -> None:
    wolf   = st.session_state.get("wolf_regime_status", "UNKNOWN")
    viking = st.session_state.get("viking_regime_status", "UNKNOWN")
    cycle  = st.session_state.get("market_cycle_phase", "UNKNOWN")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_pulse_card("Wolf Regime", wolf, _status_color(wolf)), unsafe_allow_html=True)
    with c2:
        st.markdown(_pulse_card("Viking Regime", viking, _status_color(viking)), unsafe_allow_html=True)
    with c3:
        st.markdown(_pulse_card("Market Cycle Phase", cycle, _status_color(cycle)), unsafe_allow_html=True)


# ── Navigation Zones ──────────────────────────────────────────────────────────

def _render_zones() -> None:
    # ZONE 1 — SIGNAL
    st.markdown(_zone_label("SIGNAL — HITTA KANDIDATER", _CYAN), unsafe_allow_html=True)
    z1 = st.columns(3)
    zone1 = [
        ("Arc Screener",     "Skanna nordiska + US-aktier mot alla strategier"),
        ("Contrarian Alpha", "Hatade, nödvändiga bolag med stark balansräkning"),
        ("Market Cycle",     "14-fas psykologicykel för valfri ticker"),
    ]
    for col, (title, desc) in zip(z1, zone1):
        with col:
            st.markdown(_nav_card(title, desc, _CYAN), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ZONE 2 — REGIME
    st.markdown(_zone_label("REGIME — FÖRSTÅ MARKNADEN", _PURPLE), unsafe_allow_html=True)
    z2 = st.columns(4)
    zone2 = [
        ("Wolf Regime",     "EMA-stack för swing-trades"),
        ("Alpha Regime",    "Långsiktigt positionscykel"),
        ("Viking Regime",   "OVTLYR NINE + order blocks"),
        ("Flow Divergence", "Global sektorsbredd och makrocykel"),
    ]
    for col, (title, desc) in zip(z2, zone2):
        with col:
            st.markdown(_nav_card(title, desc, _PURPLE), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ZONE 3 — INTELLIGENCE
    st.markdown(_zone_label("INTELLIGENCE — TOLKA SIGNALERNA", _EMBER), unsafe_allow_html=True)
    z3 = st.columns(4)
    zone3 = [
        ("Odin's Blindspot", "Contrarian sektorintelligens"),
        ("Sentiment",        "Fear, greed och kapitalflöden"),
        ("Retail Pulse",     "Reddit, StockTwits, retail-flöde"),
        ("Heatmap",          "Visuell marknadsvy"),
    ]
    for col, (title, desc) in zip(z3, zone3):
        with col:
            st.markdown(_nav_card(title, desc, _EMBER), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ZONE 4 — PORTFOLIO
    st.markdown(_zone_label("PORTFOLIO — HANTERA POSITIONER", _GREEN2), unsafe_allow_html=True)
    z4 = st.columns(4)
    zone4 = [
        ("Holdings",      "Positioner och riskexponering"),
        ("Trade Journal", "Logga trades och granska P&L"),
        ("Backtest",      "Historisk signalvalidering"),
        ("Alerts",        "Konfigurera och hantera notifieringar"),
    ]
    for col, (title, desc) in zip(z4, zone4):
        with col:
            st.markdown(_nav_card(title, desc, _GREEN2), unsafe_allow_html=True)


# ── Recent alerts ─────────────────────────────────────────────────────────────

def _render_recent_alerts(n: int = 5) -> None:
    section_title("Recent Alerts", "🔔")

    try:
        from alerts.engine import ALERT_LOG
    except Exception:
        ALERT_LOG = []

    if not ALERT_LOG:
        st.markdown(
            f'<div style="background:#1A1F25;border:1px solid rgba(255,255,255,0.06);'
            f'border-radius:8px;padding:16px;color:{_DIM};font-size:0.76rem;text-align:center;">'
            f'No alerts fired yet.</div>',
            unsafe_allow_html=True,
        )
        return

    for entry in reversed(ALERT_LOG[-n:]):
        ts    = entry.get("timestamp", "")
        msg   = entry.get("message", "")
        chans = entry.get("channels", [])
        meta  = entry.get("metadata", {}) or {}
        sig   = meta.get("signal", "")

        sig_color = {
            "BUY":    _GREEN2,
            "SELL":   _EMBER,
            "REDUCE": _EMBER,
        }.get(sig, _DIM)

        chan_pills = "&nbsp;".join(
            f'<span style="background:rgba(255,255,255,0.04);border:1px solid {_DIM}33;'
            f'border-radius:3px;padding:1px 5px;font-size:0.62rem;color:{_DIM};">{c}</span>'
            for c in chans
        )

        st.markdown(
            f'<div style="background:#1A1F25;border:1px solid rgba(255,255,255,0.06);'
            f'border-radius:8px;padding:10px 14px;margin-bottom:6px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
            f'  <div style="flex:1;min-width:0;">'
            f'    <div style="color:{_TEXT};font-size:0.76rem;margin-bottom:4px;">{msg}</div>'
            f'    <div>{chan_pills}</div>'
            f'  </div>'
            f'  <div style="text-align:right;flex-shrink:0;margin-left:12px;">'
            + (f'<div style="color:{sig_color};font-size:0.7rem;font-weight:700;'
               f'margin-bottom:4px;">{sig}</div>' if sig else "")
            + f'<div style="color:{_DIM};font-size:0.63rem;">{ts[:16] if ts else ""}</div>'
            f'  </div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ── Main entry point ──────────────────────────────────────────────────────────

def tab_home() -> None:
    inject_css()
    section_title("Mission Control", "🔱")

    st.markdown(
        f'<p style="color:{_DIM};font-size:0.8rem;margin:-8px 0 20px;">'
        f'Nordic Arc Systems — See What the Market Can’t.  '
        f'<span style="font-size:0.72rem;">{datetime.now().strftime("%A %d %B %Y")}</span></p>',
        unsafe_allow_html=True,
    )

    section_title("System Pulse", "📡")
    _render_system_pulse()
    st.markdown("<br>", unsafe_allow_html=True)

    _render_zones()
    st.markdown("<br>", unsafe_allow_html=True)

    _render_recent_alerts()
