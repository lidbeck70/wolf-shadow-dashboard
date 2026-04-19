"""
tabs/strategy_overview.py
==========================
Strategy Overview — one card per strategy with entry/exit/risk summaries,
sentiment plugin badges, alert status, and navigation shortcuts.
"""

from __future__ import annotations

import streamlit as st

from ui.theme import inject_css, section_title, card as _theme_card, PALETTE as _PAL

# ── Palette ──────────────────────────────────────────────────────────────────
_BG     = "#0c0c12"
_BG2    = "#14141e"
_BG3    = "#1a1a28"
_TEXT   = "#e8e4dc"
_DIM    = "#8a8578"
_GREEN  = "#2d8a4e"
_RED    = "#c44545"
_CYAN   = "#c9a84c"
_YELLOW = "#d4943a"

# ── Per-strategy static metadata (derived from module docstrings) ─────────────
_META: dict = {
    "wolf": {
        "timeframe": "Swing — days to weeks",
        "universe":  "Nordic + US equities",
        "entry": [
            "EMA10 > EMA21 > EMA50 > EMA200  (full bull stack)",
            "RSI(14) in range 45 – 70",
            "ADX(14) ≥ 19",
            "Price above EMA50",
        ],
        "exit": [
            "Stop-loss: price ≤ entry − 2.5× ATR14",
            "Core exit: price < EMA50 for 3 consecutive bars",
            "Trail exit: price < Kijun-sen AND price < EMA10",
        ],
        "risk": [
            "Risk per trade: 2% of capital",
            "Stop distance: 2.5× ATR14",
            "Shares: risk_amount ÷ stop_distance",
            "TP1 @ 2.6R (13% core)  ·  TP2 @ 5.2R (17% core)",
        ],
    },
    "alpha": {
        "timeframe": "Position — weeks to months",
        "universe":  "Nordic + US large-caps",
        "entry": [
            "Green regime: cycle_score ≥ 2",
            "Price > EMA200  (long-term uptrend)",
            "EMA50 > EMA200  (golden cross confirmed)",
            "CAGR composite score ≥ 55% of maximum",
        ],
        "exit": [
            "Price closes below EMA200",
            "Regime turns red (cycle_score = 0)",
        ],
        "risk": [
            "Risk per trade: 1.5% of capital",
            "Stop distance: price − EMA200",
            "TP1 @ 3R (30% core)  ·  TP2 @ 6R (30% core)",
        ],
    },
    "viking": {
        "timeframe": "Swing — days to weeks",
        "universe":  "Nordic + US equities",
        "entry": [
            "OVTLYR NINE composite score ≥ 70",
            "No SPY sell-off override active",
            "No restrictive order blocks present",
        ],
        "exit": [
            "NINE score < 40  → SELL",
            "Price < EMA20   → sell-off override",
            "Price < EMA10   → trailing stop",
            "½× ATR14 hard stop-loss floor",
        ],
        "risk": [
            "Risk per trade: 1.5% of capital",
            "Stop distance: ½× ATR14",
            "TP1 @ 2R (25% core)  ·  TP2 @ 4R (25% core)",
        ],
    },
}

# Map plugin key → display label
_PLUGIN_LABELS: dict = {
    "ovtlyr_fg":    "OVTLYR F&G",
    "retail_flow":  "Retail Flow",
    "options_flow": "Options Flow",
}


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _h(tag: str, text: str, style: str = "") -> str:
    return f"<{tag} style='{style}'>{text}</{tag}>"


def _pill(text: str, bg: str = "rgba(255,255,255,0.06)",
          color: str = _CYAN, border: str = "rgba(201,168,76,0.25)") -> str:
    return (
        f'<span style="background:{bg};border:1px solid {border};border-radius:4px;'
        f'padding:2px 8px;font-size:0.68rem;color:{color};white-space:nowrap;">'
        f'{text}</span>'
    )


def _icon_pill(icon: str, text: str, color: str) -> str:
    return _pill(f"{icon} {text}", color=color,
                 border=f"{color}44", bg=f"{color}11")


def _section_label(text: str) -> str:
    return (
        f'<div style="color:{_DIM};font-size:0.63rem;text-transform:uppercase;'
        f'letter-spacing:0.12em;margin:10px 0 5px;">{text}</div>'
    )


def _bullet_list(items: list[str], bullet_color: str = _CYAN) -> str:
    rows = "".join(
        f'<div style="display:flex;gap:6px;margin-bottom:3px;">'
        f'<span style="color:{bullet_color};flex-shrink:0;">›</span>'
        f'<span style="color:{_TEXT};font-size:0.76rem;">{item}</span>'
        f'</div>'
        for item in items
    )
    return rows


def _divider(color: str = "rgba(201,168,76,0.12)") -> str:
    return f'<hr style="border:none;border-top:1px solid {color};margin:12px 0;">'


# ── Single strategy card ──────────────────────────────────────────────────────

def _render_strategy_card(display_name: str, strat: dict) -> None:
    key   = strat.get("key", display_name.lower())
    color = strat.get("color", _CYAN)
    name  = strat.get("name", display_name)
    desc  = strat.get("description", "")
    meta  = _META.get(key, {})

    plugins        = strat.get("sentiment_plugins", [])
    alerts_enabled = strat.get("alerts_enabled", False)
    alert_channels = strat.get("alert_channels", [])
    params         = strat.get("params", {})

    # ── Outer card container ─────────────────────────────────────────────────
    st.markdown(
        f'<div style="border:1px solid {color}44;border-left:3px solid {color};'
        f'border-radius:8px;background:{_BG2};padding:20px 22px;margin-bottom:24px;">',
        unsafe_allow_html=True,
    )

    # Header row: name + badges
    timeframe = meta.get("timeframe", "")
    universe  = meta.get("universe", "")

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        f'  <span style="color:{color};font-size:1.15rem;font-weight:700;'
        f'  letter-spacing:0.04em;">{name}</span>'
        + (_icon_pill("⏱", timeframe, color) if timeframe else "")
        + (_icon_pill("🌍", universe,  _DIM)  if universe  else "")
        + (f'&nbsp;&nbsp;<span style="color:{_GREEN};font-size:0.68rem;">'
           f'● Alerts ON</span>' if alerts_enabled else
           f'&nbsp;&nbsp;<span style="color:{_DIM};font-size:0.68rem;">'
           f'○ Alerts OFF</span>')
        + f'</div>'
        f'<div style="color:{_DIM};font-size:0.77rem;margin-top:6px;">{desc}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Logic summary ────────────────────────────────────────────────────────
    rules_buy  = strat.get("rules_buy")
    rules_sell = strat.get("rules_sell")

    if rules_buy or rules_sell:
        # Strategy carries explicit Köp/Sälj rule lists — render them separately.
        col_buy, col_sell, col_risk = st.columns(3)

        with col_buy:
            section_title("Köp-regler")
            st.markdown(
                _bullet_list(rules_buy or ["—"], bullet_color=_GREEN),
                unsafe_allow_html=True,
            )

        with col_sell:
            section_title("Sälj-regler")
            st.markdown(
                _bullet_list(rules_sell or ["—"], bullet_color=_RED),
                unsafe_allow_html=True,
            )

        with col_risk:
            st.markdown(
                _section_label("Risk model")
                + _bullet_list(meta.get("risk", ["—"]), bullet_color=_YELLOW),
                unsafe_allow_html=True,
            )
    else:
        # Default three-column layout (entry / exit / risk).
        col_entry, col_exit, col_risk = st.columns(3)

        with col_entry:
            st.markdown(
                _section_label("Entry conditions")
                + _bullet_list(meta.get("entry", ["—"]), bullet_color=_GREEN),
                unsafe_allow_html=True,
            )

        with col_exit:
            st.markdown(
                _section_label("Exit triggers")
                + _bullet_list(meta.get("exit", ["—"]), bullet_color=_RED),
                unsafe_allow_html=True,
            )

        with col_risk:
            st.markdown(
                _section_label("Risk model")
                + _bullet_list(meta.get("risk", ["—"]), bullet_color=_YELLOW),
                unsafe_allow_html=True,
            )

    st.markdown(_divider(), unsafe_allow_html=True)

    # ── Bottom row: plugins + params + alerts ────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        plugin_pills = "&nbsp;".join(
            _pill(_PLUGIN_LABELS.get(p, p)) for p in plugins
        ) if plugins else _pill("none", color=_DIM)

        channel_pills = "&nbsp;".join(
            _pill(ch, color=_GREEN if alerts_enabled else _DIM,
                  border=f"{'#2d8a4e' if alerts_enabled else _DIM}44")
            for ch in alert_channels
        ) if alert_channels else _pill("—", color=_DIM)

        risk_pct = params.get("risk_pct", 0)
        atr_mult = params.get("atr_mult", params.get("atr_stop_mult", "—"))

        st.markdown(
            _section_label("Sentiment plugins")
            + f'<div style="margin-bottom:8px;">{plugin_pills}</div>'
            + _section_label("Alert channels")
            + f'<div style="margin-bottom:8px;">{channel_pills}</div>'
            + _section_label("Key parameters")
            + f'<div style="color:{_TEXT};font-size:0.74rem;">'
            + (f'Risk/trade: <b style="color:{_CYAN};">'
               f'{risk_pct*100:.1f}%</b>&nbsp;&nbsp;' if risk_pct else "")
            + (f'ATR mult: <b style="color:{_CYAN};">{atr_mult}</b>' if atr_mult else "")
            + '</div>',
            unsafe_allow_html=True,
        )

    with right:
        # Navigation buttons
        st.markdown(_section_label("Quick actions"), unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        with b1:
            if st.button(
                "📡 Screener",
                key=f"ov_screener_{key}",
                use_container_width=True,
                help="Open this strategy in the Screener tab",
            ):
                st.session_state["overview_goto_tab"]     = "screener"
                st.session_state["overview_active_strat"] = key
                st.toast(f"Navigate to the SCREENER tab to scan with {name}.", icon="📡")

        with b2:
            if st.button(
                "📊 Backtest",
                key=f"ov_backtest_{key}",
                use_container_width=True,
                help="Open this strategy in the Backtest tab",
            ):
                st.session_state["overview_goto_tab"]     = "backtest"
                st.session_state["overview_active_strat"] = key
                st.toast(f"Navigate to the BACKTEST tab to test {name}.", icon="📊")

        if st.button(
            "🔔 Alert Settings",
            key=f"ov_alerts_{key}",
            use_container_width=True,
            help="Open this strategy's alert configuration",
        ):
            st.session_state["overview_goto_tab"]     = "alerts"
            st.session_state["overview_active_strat"] = key
            st.toast(f"Navigate to the ALERTS tab to configure {name} alerts.", icon="🔔")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Premium placeholders ─────────────────────────────────────────────────
    with st.expander(f"📖  Full Rulebook — {name}", expanded=False):
        st.info(
            "Premium feature: Full rulebook — step-by-step entry checklist, "
            "annotated chart examples, parameter tuning guide, and historical "
            "regime context for this strategy.",
            icon="🔒",
        )

    with st.expander(f"⚡  Advanced Alerts — {name}", expanded=False):
        st.info(
            "Premium feature: Advanced alerts — conditional alert chains, "
            "multi-ticker watchlist monitoring, Telegram / SMS delivery, "
            "and AI-generated alert summaries.",
            icon="🔒",
        )


# ── Summary header ────────────────────────────────────────────────────────────

def _render_header(strategies: dict) -> None:
    n_enabled = sum(1 for s in strategies.values() if s.get("alerts_enabled"))
    n_plugins  = len({p for s in strategies.values()
                      for p in s.get("sentiment_plugins", [])})

    cols = st.columns(4)
    stats = [
        ("STRATEGIES",        str(len(strategies)),      _CYAN),
        ("ALERTS ACTIVE",     f"{n_enabled} / {len(strategies)}", _GREEN),
        ("SENTIMENT PLUGINS", str(n_plugins),             _YELLOW),
        ("CHANNELS",          "discord · email · webhook", _DIM),
    ]
    for col, (label, value, color) in zip(cols, stats):
        with col:
            _theme_card(
                f'<div style="color:{_DIM};font-size:0.63rem;text-transform:uppercase;'
                f'letter-spacing:0.1em;">{label}</div>'
                f'<div style="color:{color};font-size:1.1rem;font-weight:700;'
                f'margin-top:4px;">{value}</div>',
                border_color="rgba(255,255,255,0.06)",
                padding="12px 14px",
            )


# ── Main entry point ──────────────────────────────────────────────────────────

def tab_strategy_overview() -> None:
    inject_css()
    section_title("Strategy Overview")

    st.markdown(
        f'<p style="color:{_DIM};font-size:0.8rem;margin:-8px 0 20px;">'
        f'All active strategies — entry logic, exit rules, risk model, '
        f'sentiment plugins, and alert configuration at a glance.</p>',
        unsafe_allow_html=True,
    )

    try:
        from strategies.registry import STRATEGIES
    except Exception as exc:
        st.error(f"Cannot load strategies: {exc}")
        return

    _render_header(STRATEGIES)
    st.markdown("<br>", unsafe_allow_html=True)

    # One card per strategy
    for display_name, strat in STRATEGIES.items():
        _render_strategy_card(display_name, strat)

    # ── Session navigation hint ──────────────────────────────────────────────
    goto = st.session_state.get("overview_goto_tab")
    if goto:
        tab_labels = {
            "screener": "SCREENER",
            "backtest": "BACKTEST",
            "alerts":   "ALERTS",
        }
        label = tab_labels.get(goto, goto.upper())
        strat_name = st.session_state.get("overview_active_strat", "")
        st.markdown(
            f'<div style="border:1px solid {_CYAN}33;border-radius:6px;'
            f'background:{_BG3};padding:10px 14px;margin-top:8px;'
            f'font-size:0.78rem;color:{_DIM};">'
            f'<span style="color:{_CYAN};">↗ Shortcut</span>&nbsp; '
            f'Click the <b style="color:{_TEXT};">{label}</b> tab above '
            f'to continue with <b style="color:{_TEXT};">{strat_name}</b>.'
            f'</div>',
            unsafe_allow_html=True,
        )
