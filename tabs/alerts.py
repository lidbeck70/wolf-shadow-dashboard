"""
tabs/alerts.py
==============
Alert Center — UI for configuring per-strategy alert settings,
viewing the in-memory alert log, and sending test alerts.

Settings written here mutate the live STRATEGY dicts (in-memory),
so they are immediately visible to tabs/backtest.py and tabs/screener.py
within the same Streamlit process without any file I/O.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import streamlit as st

from ui.theme import inject_css, section_title, card as _theme_card, PALETTE as _PAL

# ── Palette (matches project-wide cyberpunk theme) ──────────────────────────
_CYAN   = "#c9a84c"
_GREEN  = "#2d8a4e"
_RED    = "#c44545"
_YELLOW = "#d4943a"
_DIM    = "#8a8578"
_TEXT   = "#e8e4dc"
_BG     = "#0c0c12"
_BG2    = "#14141e"
_BG3    = "#1a1a28"
_BORDER = "rgba(201,168,76,0.15)"

_ALL_CHANNELS = ["discord", "email", "webhook"]

_CHANNEL_ICONS = {
    "discord": "💬",
    "email":   "✉️",
    "webhook": "🔗",
}

_SIGNAL_ICONS = {
    "entry":             "🟢",
    "exit":              "🔴",
    "STOP":              "🛑",
    "TRAIL_EXIT":        "📉",
    "EMA50_EXIT":        "📊",
    "END":               "🏁",
    "sentiment_extreme": "⚡",
    "regime_shift":      "🌊",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _card(content: str, border_color: str = _BORDER) -> None:
    _theme_card(content, border_color=border_color, padding="14px 16px", margin_bottom="8px")


def _label(text: str) -> str:
    return (
        f'<span style="color:{_DIM};font-size:0.68rem;text-transform:uppercase;'
        f'letter-spacing:0.08em;">{text}</span>'
    )


def _pill(text: str, color: str = _CYAN) -> str:
    return (
        f'<span style="background:rgba(0,0,0,0.35);border:1px solid {color}33;'
        f'border-radius:4px;padding:1px 7px;font-size:0.68rem;color:{color};">'
        f'{text}</span>'
    )


def _result_dot(ok: bool) -> str:
    color = _GREEN if ok else _RED
    return f'<span style="color:{color};font-size:0.8rem;">{"●" if ok else "○"}</span>'


def _signal_icon(signal: str) -> str:
    for k, icon in _SIGNAL_ICONS.items():
        if k.lower() in signal.lower():
            return icon
    return "📢"


# ── Strategy settings block ──────────────────────────────────────────────────

def _render_strategy_settings(strategies: dict) -> None:
    section_title("Strategy Alert Settings")

    st.markdown(
        f'<p style="color:{_DIM};font-size:0.78rem;margin-bottom:16px;">'
        f'Changes take effect immediately for Backtest and Screener within '
        f'this session. They are not persisted to disk.</p>',
        unsafe_allow_html=True,
    )

    for name, strat in strategies.items():
        color  = strat.get("color", _CYAN)
        s_key  = name.lower()

        with st.expander(
            f"{strat.get('name', name)}",
            expanded=True,
        ):
            st.markdown(
                f'<div style="color:{color};font-size:0.7rem;text-transform:uppercase;'
                f'letter-spacing:0.12em;margin-bottom:10px;">'
                f'{strat.get("description", "")[:120]}…</div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns([1, 2, 1])

            with c1:
                enabled = st.toggle(
                    "Alerts enabled",
                    value=bool(strat.get("alerts_enabled", False)),
                    key=f"al_enabled_{s_key}",
                )
                strat["alerts_enabled"] = enabled

            with c2:
                channels = st.multiselect(
                    "Channels",
                    _ALL_CHANNELS,
                    default=[
                        ch for ch in strat.get("alert_channels", ["discord"])
                        if ch in _ALL_CHANNELS
                    ],
                    format_func=lambda ch: f"{_CHANNEL_ICONS.get(ch, '')} {ch}",
                    key=f"al_channels_{s_key}",
                    disabled=not enabled,
                )
                strat["alert_channels"] = channels

            with c3:
                # Sentiment plugins badge
                plugins = strat.get("sentiment_plugins", [])
                st.markdown(
                    _label("Sentiment plugins") + "<br>"
                    + " ".join(_pill(p) for p in plugins),
                    unsafe_allow_html=True,
                )

            if enabled and not channels:
                st.warning("No channels selected — alerts will be skipped.", icon="⚠️")


# ── Alert-type toggles ───────────────────────────────────────────────────────

def _render_alert_type_settings(strategies: dict) -> None:
    section_title("Alert Type Toggles")

    st.markdown(
        f'<p style="color:{_DIM};font-size:0.78rem;margin-bottom:12px;">'
        f'These flags are stored on every strategy dict so the backend '
        f'can gate specific alert categories independently.</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        on_entry = st.toggle(
            "Entry / Exit signals",
            value=st.session_state.get("al_type_entry", True),
            key="al_type_entry",
            help="Fire an alert when a new entry or exit signal is detected.",
        )

    with col2:
        on_sentiment = st.toggle(
            "Sentiment extremes",
            value=st.session_state.get("al_type_sentiment", True),
            key="al_type_sentiment",
            help="Fire when any sentiment plugin score exceeds 90 or falls below 10.",
        )

    with col3:
        on_regime = st.toggle(
            "Regime shifts",
            value=st.session_state.get("al_type_regime", True),
            key="al_type_regime",
            help="Fire when a ticker's regime category flips between bull and bear.",
        )

    # Propagate to every live STRATEGY dict so backtest/screener can gate alerts.
    for strat in strategies.values():
        strat["alerts_on_entry"]              = on_entry
        strat["alerts_on_sentiment_extreme"]  = on_sentiment
        strat["alerts_on_regime_shift"]       = on_regime

    # Status summary
    active_types = [
        label for label, flag in [
            ("Entry/Exit", on_entry),
            ("Sentiment", on_sentiment),
            ("Regime",    on_regime),
        ] if flag
    ]
    st.markdown(
        f'<div style="margin-top:8px;">'
        + _label("Active alert types") + "&nbsp;&nbsp;"
        + ("&nbsp;".join(_pill(t, _GREEN) for t in active_types)
           if active_types else _pill("none", _RED))
        + "</div>",
        unsafe_allow_html=True,
    )


# ── Test alert ───────────────────────────────────────────────────────────────

def _render_test_alert(send_fn) -> None:
    section_title("Send Test Alert")

    c1, c2, c3 = st.columns([3, 2, 1])

    with c1:
        test_msg = st.text_input(
            "Message",
            value="Wolf-Shadow test alert — system check",
            key="al_test_msg",
            placeholder="Enter alert message…",
        )

    with c2:
        test_channels = st.multiselect(
            "Channels",
            _ALL_CHANNELS,
            default=["discord"],
            format_func=lambda ch: f"{_CHANNEL_ICONS.get(ch, '')} {ch}",
            key="al_test_channels",
        )

    with c3:
        ticker_tag = st.text_input("Ticker (optional)", value="TEST", key="al_test_ticker")

    send_btn = st.button(
        "▶ SEND TEST ALERT",
        key="al_send_test",
        use_container_width=True,
    )

    if send_btn:
        if not test_channels:
            st.warning("Select at least one channel.")
            return

        if send_fn is None:
            st.error("alerts.engine not available.")
            return

        with st.spinner("Sending…"):
            results = send_fn(
                test_msg,
                test_channels,
                metadata={
                    "ticker": ticker_tag or "TEST",
                    "signal": "test",
                    "title":  "Wolf-Shadow Test Alert",
                    "color":  0xC9A84C,
                },
            )

        all_ok = all(results.values()) if results else False
        if all_ok:
            st.success(f"Sent to: {', '.join(test_channels)}")
        else:
            failed = [ch for ch, ok in results.items() if not ok]
            ok_chs = [ch for ch, ok in results.items() if ok]
            if ok_chs:
                st.success(f"Delivered: {', '.join(ok_chs)}")
            st.error(
                f"Failed: {', '.join(failed)} — "
                f"check environment variables (e.g. DISCORD_WEBHOOK_URL)."
            )


# ── Alert log ────────────────────────────────────────────────────────────────

def _render_alert_log(log: list) -> None:
    n = len(log)
    section_title(f"Alert Log  ({n} entries this session)")

    if not log:
        st.markdown(
            f'<div style="text-align:center;padding:40px 20px;'
            f'border:1px dashed {_BORDER};border-radius:8px;color:{_DIM};">'
            f'<div style="font-size:32px;margin-bottom:10px;">📭</div>'
            f'No alerts fired yet this session.</div>',
            unsafe_allow_html=True,
        )
        return

    # Controls row
    fc1, fc2, _ = st.columns([1, 1, 3])
    with fc1:
        filter_signal = st.selectbox(
            "Filter by type",
            ["All"] + sorted({
                e["metadata"].get("signal", "other")
                for e in log
                if isinstance(e.get("metadata"), dict)
            }),
            key="al_log_filter",
        )
    with fc2:
        show_n = st.selectbox("Show last", [25, 50, 100, 200], key="al_log_show_n")

    if st.button("Clear log", key="al_log_clear"):
        log.clear()
        st.rerun()

    # Filter and paginate
    visible = [
        e for e in log
        if filter_signal == "All"
        or (isinstance(e.get("metadata"), dict)
            and e["metadata"].get("signal", "") == filter_signal)
    ]
    visible = list(reversed(visible[-show_n:]))

    st.markdown(f'<div style="color:{_DIM};font-size:0.7rem;margin:4px 0 10px;">'
                f'Showing {len(visible)} of {n} entries (newest first)</div>',
                unsafe_allow_html=True)

    for entry in visible:
        _render_log_entry(entry)


def _render_log_entry(entry: Dict[str, Any]) -> None:
    ts      = entry.get("timestamp", "")
    message = entry.get("message", "")
    channels = entry.get("channels", [])
    results  = entry.get("results", {})
    meta     = entry.get("metadata", {})

    signal  = meta.get("signal", "")
    ticker  = meta.get("ticker", "")
    icon    = _signal_icon(signal)

    all_ok  = all(results.values()) if results else False
    any_ok  = any(results.values()) if results else False
    status_color = _GREEN if all_ok else (_YELLOW if any_ok else _RED)

    channel_dots = " ".join(
        f'{_CHANNEL_ICONS.get(ch, "?")} {_result_dot(results.get(ch, False))}'
        for ch in channels
    )
    title_part = meta.get("title", "")

    content = (
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'  <div style="flex:1;min-width:0;">'
        f'    <span style="font-size:1rem;">{icon}</span>'
        f'    &nbsp;<span style="color:{_TEXT};font-size:0.8rem;font-weight:600;">'
        f'{title_part or message[:60]}</span>'
        + (f'&nbsp;&nbsp;{_pill(ticker, _CYAN)}' if ticker and ticker != "TEST" else "")
        + f'    <div style="color:{_DIM};font-size:0.72rem;margin-top:3px;'
        f'word-break:break-word;">{message[:180]}</div>'
        f'  </div>'
        f'  <div style="text-align:right;white-space:nowrap;padding-left:12px;">'
        f'    <div style="color:{_DIM};font-size:0.65rem;">{ts}</div>'
        f'    <div style="margin-top:4px;">{channel_dots}</div>'
        f'  </div>'
        f'</div>'
    )

    _card(content, border_color=f"{status_color}44")


# ── Channel status panel ─────────────────────────────────────────────────────

def _render_channel_status() -> None:
    section_title("Channel Configuration Status")

    import os
    checks = [
        ("discord", "DISCORD_WEBHOOK_URL",  "Discord webhook URL"),
        ("email",   "EMAIL_TO",             "Recipient email address"),
        ("webhook", "ALERT_WEBHOOK_URL",    "Generic webhook target URL"),
    ]

    cols = st.columns(len(checks))
    for col, (ch, env_key, label) in zip(cols, checks):
        val      = os.environ.get(env_key, "").strip()
        is_set   = bool(val)
        color    = _GREEN if is_set else _RED
        icon     = _CHANNEL_ICONS.get(ch, "?")
        status   = "CONFIGURED" if is_set else "NOT SET"
        preview  = (
            val[:12] + "…" + val[-4:] if len(val) > 20
            else val[:20]
        ) if is_set else f"set {env_key}"

        with col:
            _card(
                f'<div style="font-size:1.2rem;margin-bottom:4px;">{icon}</div>'
                f'<div style="color:{color};font-size:0.68rem;text-transform:uppercase;'
                f'letter-spacing:0.1em;font-weight:700;">{status}</div>'
                f'<div style="color:{_TEXT};font-size:0.78rem;margin-top:2px;">'
                f'{ch.title()}</div>'
                f'<div style="color:{_DIM};font-size:0.62rem;margin-top:3px;">'
                f'{label}</div>'
                f'<div style="color:{_DIM};font-size:0.6rem;font-family:monospace;'
                f'margin-top:4px;">{preview}</div>',
                border_color=f"{color}44",
            )


# ── Main entry point ─────────────────────────────────────────────────────────

def tab_alerts() -> None:
    inject_css()
    section_title("Alert Center")

    st.markdown(
        f'<p style="color:{_DIM};font-size:0.8rem;margin:-8px 0 20px;">'
        f'Configure per-strategy alert delivery, set alert-type filters, '
        f'send test alerts, and review the in-session alert log.</p>',
        unsafe_allow_html=True,
    )

    # Load dependencies — graceful degradation if packages are missing.
    try:
        from strategies.registry import STRATEGIES
    except Exception as exc:
        st.error(f"Cannot load strategies: {exc}")
        return

    send_fn = None
    log: list = []
    try:
        from alerts.engine import send_alert, ALERT_LOG
        send_fn = send_alert
        log     = ALERT_LOG
    except Exception as exc:
        st.warning(f"alerts.engine not available: {exc}")

    # ── Tabs within the Alerts page ──────────────────────────────────────────
    inner_tabs = st.tabs([
        "  SETTINGS  ",
        "  ALERT LOG  ",
        "  CHANNELS  ",
    ])

    with inner_tabs[0]:
        _render_strategy_settings(STRATEGIES)
        st.markdown("<br>", unsafe_allow_html=True)
        _render_alert_type_settings(STRATEGIES)
        st.markdown("<br>", unsafe_allow_html=True)
        _render_test_alert(send_fn)

    with inner_tabs[1]:
        _render_alert_log(log)

    with inner_tabs[2]:
        _render_channel_status()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:{_DIM};font-size:0.75rem;border:1px solid {_BORDER};'
            f'border-radius:6px;padding:12px;">'
            f'<div style="color:{_CYAN};font-size:0.7rem;text-transform:uppercase;'
            f'letter-spacing:0.1em;margin-bottom:8px;">Environment Variables</div>'
            f'<code style="color:{_TEXT};">DISCORD_WEBHOOK_URL</code>'
            f'&nbsp;— Discord incoming webhook URL (from Server Settings → Integrations)<br>'
            f'<code style="color:{_TEXT};">EMAIL_FROM</code>'
            f'&nbsp;— Sender address (placeholder — no SMTP wired yet)<br>'
            f'<code style="color:{_TEXT};">EMAIL_TO</code>'
            f'&nbsp;— Recipient address(es), comma-separated<br>'
            f'<code style="color:{_TEXT};">ALERT_WEBHOOK_URL</code>'
            f'&nbsp;— Generic POST target (Slack, Teams, n8n, custom API…)<br>'
            f'<code style="color:{_TEXT};">ALERT_WEBHOOK_TOKEN</code>'
            f'&nbsp;— Optional Bearer token for the webhook endpoint<br>'
            f'</div>',
            unsafe_allow_html=True,
        )
