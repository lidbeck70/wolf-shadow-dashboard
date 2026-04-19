"""
tabs/home.py
============
Home tab — dashboard overview for Nordic Alpha Systems.

Shows: system status, active strategies summary, recent alerts,
quick-access navigation cards, and market pulse (Thor-Index).
"""

from __future__ import annotations

import streamlit as st
from datetime import datetime

from ui.theme import inject_css, section_title, card as _card, PALETTE as _P

_BG2   = _P["bg2"]
_BG3   = _P["bg3"]
_GOLD  = _P["gold"]
_DIM   = _P["text_dim"]
_TEXT  = _P["text"]
_GREEN = _P["green"]
_RED   = _P["red"]
_AMBER = _P["amber"]


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _pill(text: str, color: str = _GOLD, bg: str = "rgba(201,168,76,0.08)") -> str:
    return (
        f'<span style="background:{bg};border:1px solid {color}44;'
        f'border-radius:4px;padding:2px 8px;font-size:0.67rem;'
        f'color:{color};white-space:nowrap;">{text}</span>'
    )


def _stat_tile(label: str, value: str, color: str = _GOLD, sub: str = "") -> str:
    return (
        f'<div style="color:{_DIM};font-size:0.62rem;text-transform:uppercase;'
        f'letter-spacing:0.12em;margin-bottom:4px;">{label}</div>'
        f'<div style="color:{color};font-size:1.15rem;font-weight:700;">{value}</div>'
        + (f'<div style="color:{_DIM};font-size:0.68rem;margin-top:3px;">{sub}</div>' if sub else "")
    )


def _nav_card(icon: str, title: str, desc: str, color: str = _GOLD) -> str:
    return (
        f'<div style="text-align:center;padding:6px 0;">'
        f'  <div style="font-size:1.6rem;margin-bottom:6px;">{icon}</div>'
        f'  <div style="color:{color};font-size:0.8rem;font-weight:700;'
        f'  letter-spacing:0.05em;margin-bottom:4px;">{title}</div>'
        f'  <div style="color:{_DIM};font-size:0.68rem;line-height:1.4;">{desc}</div>'
        f'</div>'
    )


# ── Strategy status summary ───────────────────────────────────────────────────

def _render_strategy_pulse() -> None:
    section_title("Strategy Pulse", "⚡")

    try:
        from strategies.registry import STRATEGIES
    except Exception:
        st.warning("Strategy registry unavailable.")
        return

    cols = st.columns(len(STRATEGIES))
    for col, (name, strat) in zip(cols, STRATEGIES.items()):
        color       = strat.get("color", _GOLD)
        alerts_on   = strat.get("alerts_enabled", False)
        plugins     = strat.get("sentiment_plugins", [])
        channels    = strat.get("alert_channels", [])
        alert_dot   = f'<span style="color:{_GREEN};">●</span>' if alerts_on else f'<span style="color:{_DIM};">○</span>'

        with col:
            _card(
                content=(
                    f'<div style="color:{color};font-size:0.9rem;font-weight:700;'
                    f'letter-spacing:0.06em;margin-bottom:8px;">{name}</div>'
                    f'<div style="color:{_DIM};font-size:0.68rem;margin-bottom:6px;">'
                    f'{alert_dot}&nbsp;Alerts '
                    + ("ON" if alerts_on else "OFF")
                    + f'&nbsp;&nbsp;·&nbsp;&nbsp;{len(plugins)} plugin{"s" if len(plugins) != 1 else ""}'
                    + f'</div>'
                    + (
                        f'<div style="font-size:0.65rem;color:{_DIM};">'
                        + "&nbsp;".join(
                            f'<span style="background:{color}11;border:1px solid {color}33;'
                            f'border-radius:3px;padding:1px 6px;color:{color};">{p}</span>'
                            for p in plugins
                        )
                        + '</div>'
                        if plugins else ""
                    )
                ),
                border_color=f"{color}44",
                accent_color=color,
                padding="14px 16px",
            )


# ── Thor-Index (market pulse composite) ──────────────────────────────────────

def _render_thor_index() -> None:
    section_title("Thor-Index", "⚡")

    st.markdown(
        f'<p style="color:{_DIM};font-size:0.76rem;margin:-8px 0 14px;">'
        f'Nordic market composite — breadth, momentum, and regime signals aggregated '
        f'across the Wolf, Alpha, and Viking strategy layers.</p>',
        unsafe_allow_html=True,
    )

    # Read from session state if available (set by regime/sector tabs)
    thor = st.session_state.get("thor_index", None)

    if thor is None:
        _card(
            content=(
                f'<div style="color:{_DIM};font-size:0.78rem;text-align:center;padding:12px 0;">'
                f'Thor-Index not yet computed.<br>'
                f'<span style="font-size:0.68rem;">Run a scan in the SCREENER or VIKING REGIME tab '
                f'to populate the index.</span>'
                f'</div>'
            ),
            border_color=_P["border"],
            padding="16px",
        )
        return

    score       = int(thor.get("score", 0))
    label       = thor.get("label", "NEUTRAL")
    trend       = thor.get("trend", "—")
    breadth     = thor.get("breadth", "—")
    momentum    = thor.get("momentum", "—")
    updated     = thor.get("updated", "—")

    color = _GREEN if score >= 65 else (_RED if score <= 35 else _AMBER)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _card(_stat_tile("THOR-INDEX", f"{score}/100", color, label),
              border_color=f"{color}44", accent_color=color, padding="14px 16px")
    with c2:
        _card(_stat_tile("TREND", trend, _GOLD), padding="14px 16px")
    with c3:
        _card(_stat_tile("BREADTH", breadth, _GOLD), padding="14px 16px")
    with c4:
        _card(_stat_tile("MOMENTUM", momentum, _GOLD, f"Updated {updated}"),
              padding="14px 16px")


# ── Recent alerts ─────────────────────────────────────────────────────────────

def _render_recent_alerts(n: int = 5) -> None:
    section_title("Recent Alerts", "🔔")

    try:
        from alerts.engine import ALERT_LOG
    except Exception:
        ALERT_LOG = []

    if not ALERT_LOG:
        _card(
            f'<div style="color:{_DIM};font-size:0.76rem;text-align:center;padding:8px 0;">'
            f'No alerts fired yet.</div>',
            padding="14px 16px",
        )
        return

    for entry in reversed(ALERT_LOG[-n:]):
        ts      = entry.get("timestamp", "")
        msg     = entry.get("message", "")
        chans   = entry.get("channels", [])
        meta    = entry.get("metadata", {}) or {}
        sig     = meta.get("signal", "")

        sig_color = {
            "BUY":    _GREEN,
            "SELL":   _RED,
            "REDUCE": _AMBER,
        }.get(sig, _DIM)

        chan_pills = "&nbsp;".join(
            f'<span style="background:rgba(255,255,255,0.04);border:1px solid {_DIM}33;'
            f'border-radius:3px;padding:1px 5px;font-size:0.62rem;color:{_DIM};">{c}</span>'
            for c in chans
        )

        _card(
            content=(
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
                f'</div>'
            ),
            padding="10px 14px",
            margin_bottom="6px",
        )


# ── Quick-nav cards ───────────────────────────────────────────────────────────

def _render_quick_nav() -> None:
    section_title("Quick Access", "🗺")

    nav_items = [
        ("📡", "Screener",        "Scan Nordic + US equities across all strategies"),
        ("📊", "Backtest",        "Replay strategy signals on historical price data"),
        ("💼", "Holdings",        "Portfolio positions, risk exposure, earnings calendar"),
        ("📓", "Trade Journal",   "Log trades, review P&L, tag patterns"),
        ("🐺", "Wolf Regime",     "EMA-stack regime monitor for swing setups"),
        ("🦅", "Alpha Regime",    "Long-term cycle monitor for position setups"),
        ("⚔", "Viking Regime",   "OVTLYR NINE score + order-block overlay"),
        ("🌍", "Sector & Regime", "Global sector breadth and macro cycle"),
        ("🔔", "Alerts",          "Configure alert channels and view alert history"),
        ("📋", "Strategies",      "Entry/exit rules and risk model for every strategy"),
    ]

    cols = st.columns(5)
    for i, (icon, title, desc) in enumerate(nav_items):
        with cols[i % 5]:
            _card(
                _nav_card(icon, title, desc),
                padding="14px 12px",
                margin_bottom="8px",
            )


# ── System status ─────────────────────────────────────────────────────────────

def _render_system_status() -> None:
    section_title("System Status", "⚙")

    import os

    checks = [
        ("Börsdata API",   bool(os.environ.get("BORSDATA_API_KEY")),   "BD_API_KEY set"),
        ("Discord Alerts", bool(os.environ.get("DISCORD_WEBHOOK_URL")), "DISCORD_WEBHOOK_URL set"),
        ("Email Alerts",   bool(os.environ.get("EMAIL_FROM")),          "EMAIL_FROM set"),
        ("Webhook Alerts", bool(os.environ.get("ALERT_WEBHOOK_URL")),   "ALERT_WEBHOOK_URL set"),
    ]

    cols = st.columns(len(checks))
    for col, (label, ok, hint) in zip(cols, checks):
        dot   = f'<span style="color:{_GREEN};">●</span>' if ok else f'<span style="color:{_RED};">●</span>'
        state = "CONNECTED" if ok else "NOT SET"
        color = _GREEN if ok else _RED
        with col:
            _card(
                f'<div style="color:{_DIM};font-size:0.62rem;text-transform:uppercase;'
                f'letter-spacing:0.1em;margin-bottom:4px;">{label}</div>'
                f'<div style="font-size:0.9rem;">{dot}&nbsp;'
                f'<span style="color:{color};font-size:0.8rem;font-weight:700;">{state}</span></div>'
                f'<div style="color:{_DIM};font-size:0.62rem;margin-top:4px;">{hint}</div>',
                padding="12px 14px",
                margin_bottom="8px",
            )


# ── Main entry point ──────────────────────────────────────────────────────────

def tab_home() -> None:
    inject_css()
    section_title("Dashboard Overview", "🐺")

    st.markdown(
        f'<p style="color:{_DIM};font-size:0.8rem;margin:-8px 0 20px;">'
        f'Nordic Alpha Systems — Born of Wolves, Made for Markets.&nbsp;&nbsp;'
        f'<span style="font-size:0.72rem;">{datetime.now().strftime("%A %d %B %Y")}</span></p>',
        unsafe_allow_html=True,
    )

    _render_strategy_pulse()
    st.markdown("<br>", unsafe_allow_html=True)

    _render_thor_index()
    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([2, 1])

    with left:
        _render_quick_nav()

    with right:
        _render_recent_alerts()
        st.markdown("<br>", unsafe_allow_html=True)
        _render_system_status()
