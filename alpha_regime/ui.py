"""
alpha_regime/ui.py
Streamlit UI for the dual-mode Alpha Regime confirmation system.

Entry point: render_alpha_regime()
Modes: Quality (Buffett/KAP) · Deep Contrarian (Rule/Sprott)
"""
from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

from alpha_regime.engine import RegimeResult, run_regime_analysis
from alpha_regime.quality_signals import SignalResult
from alpha_regime.contrarian_signals import (
    CYCLE_STRIP_ORDER,
    CYCLE_PHASE_COLORS,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_MARKET_OPTIONS = {
    "SPY — S&P 500 (US)":    "SPY",
    "^OMX — OMX Stockholm 30": "^OMX",
    "^OMXSPI — OMX All Share":  "^OMXSPI",
    "QQQ — Nasdaq 100":       "QQQ",
}

_VERDICT_COLORS = {
    "BUY":   "#1aaa5a",
    "WATCH": "#e8a020",
    "WAIT":  "#607080",
}

_TREND_COLORS = {
    "Bullish": "#1aaa5a",
    "Neutral": "#e8a020",
    "Bearish": "#cc3333",
}

_SIGNAL_ICONS = {
    True:  "✅",
    False: "❌",
    None:  "⚪",
}


# ── Helper renderers ──────────────────────────────────────────────────────────

def _css_badge(text: str, bg: str, fg: str = "#FFFFFF") -> str:
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;'
        f'border-radius:4px;font-size:0.82rem;font-weight:700;'
        f'letter-spacing:0.05em;">{text}</span>'
    )


def _signal_card(sig: SignalResult) -> None:
    """Render a single Quality signal card."""
    border = "#1aaa5a" if sig.passed else ("#cc3333" if sig.label != "NO DATA" else "#607080")
    icon = "✅" if sig.passed else ("⚪" if sig.label == "NO DATA" else "❌")

    st.markdown(
        f"""
        <div style="border:1px solid {border};border-radius:6px;padding:12px 14px;
                    background:#0D1117;margin-bottom:4px;">
            <div style="font-size:0.72rem;color:#8899aa;letter-spacing:0.08em;
                        text-transform:uppercase;margin-bottom:4px;">{sig.name}</div>
            <div style="font-size:1.15rem;font-weight:700;color:{border};
                        margin-bottom:6px;">{icon} {sig.label}</div>
            <div style="font-size:0.8rem;color:#aabbcc;">{sig.detail}</div>
            {f'<div style="font-size:0.75rem;color:#e8a020;margin-top:4px;">⚠ {sig.warning}</div>'
             if sig.warning else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_verdict_banner(verdict: str, passed: int, total: int) -> None:
    color = _VERDICT_COLORS.get(verdict, "#607080")
    subtitle = {
        "BUY":   f"All {total} signals confirmed — quality entry conditions met",
        "WATCH": f"{passed}/{total} signals confirmed — monitor for full alignment",
        "WAIT":  f"{passed}/{total} signals confirmed — wait for better setup",
    }.get(verdict, "")
    st.markdown(
        f"""
        <div style="border:2px solid {color};border-radius:8px;padding:20px 24px;
                    background:rgba({_hex_to_rgb(color)},0.08);margin:16px 0;">
            <div style="font-size:0.72rem;color:#8899aa;letter-spacing:0.1em;
                        text-transform:uppercase;">Quality Mode Verdict</div>
            <div style="font-size:2rem;font-weight:900;color:{color};
                        letter-spacing:0.06em;">{verdict}</div>
            <div style="font-size:0.88rem;color:#aabbcc;margin-top:4px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"
    return "100,150,200"


def _render_market_context(r: RegimeResult) -> None:
    """Shared market context strip."""
    with st.expander("Market Context", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            phase_color = CYCLE_PHASE_COLORS.get(r.market_phase, "#607080")
            st.markdown(
                f"**Market Phase**<br>"
                f"{_css_badge(r.market_phase, phase_color)}",
                unsafe_allow_html=True,
            )
        with c2:
            st.metric("Cycle Confidence", f"{r.market_confidence:.0f}%")
        with c3:
            pct = r.price_vs_ma200
            color = "#1aaa5a" if pct > 0 else "#cc3333"
            st.markdown(
                f"**Price vs 200D MA**<br>"
                f'<span style="color:{color};font-size:1.1rem;font-weight:700;">'
                f'{pct:+.1f}%</span>',
                unsafe_allow_html=True,
            )
        with c4:
            trend_color = _TREND_COLORS.get(r.trend_phase, "#607080")
            st.markdown(
                f"**Trend Phase**<br>"
                f"{_css_badge(r.trend_phase, trend_color)}",
                unsafe_allow_html=True,
            )

        if r.sentiment_score is not None:
            sent = r.sentiment_score
            sent_label = "Bearish" if sent < 30 else ("Bullish" if sent > 70 else "Neutral")
            sent_color = "#cc3333" if sent < 30 else ("#1aaa5a" if sent > 70 else "#607080")
            st.markdown(
                f"Retail Sentiment: {_css_badge(f'{sent:.0f}/100 — {sent_label}', sent_color)}",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Retail sentiment: unavailable")


def _render_cycle_strip(current_phase: str) -> None:
    """Render the 14-phase cycle strip with current phase highlighted."""
    cells = []
    for ph in CYCLE_STRIP_ORDER:
        col = CYCLE_PHASE_COLORS.get(ph, "#607080")
        active = ph == current_phase
        border = "2px solid #ffffff" if active else "1px solid transparent"
        scale = "1.08" if active else "1"
        cells.append(
            f'<div style="flex:1;text-align:center;padding:6px 2px;'
            f'background:{col};border-radius:4px;border:{border};'
            f'transform:scale({scale});font-size:0.62rem;font-weight:{"700" if active else "400"};'
            f'color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
            f'title="{ph}">{ph.replace("_", " ")}</div>'
        )
    st.markdown(
        f'<div style="display:flex;gap:3px;margin:10px 0;">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


# ── Mode renderers ────────────────────────────────────────────────────────────

def _render_quality_mode(r: RegimeResult) -> None:
    if r.error:
        st.error(r.error)
        return

    # Signal grid — 4 columns
    cols = st.columns(4)
    for i, sig in enumerate(r.signals):
        with cols[i % 4]:
            _signal_card(sig)

    # Verdict banner
    total_available = sum(1 for s in r.signals if s.label != "NO DATA")
    _render_verdict_banner(r.quality_verdict, r.signals_passed, total_available)

    # Market context
    _render_market_context(r)


def _render_contrarian_mode(r: RegimeResult) -> None:
    if r.error:
        st.error(r.error)
        return

    c = r.contrarian
    if c is None:
        st.warning("Contrarian stage could not be determined.")
        return

    # Large staged signal chip
    st.markdown(
        f"""
        <div style="border:2px solid {c.color};border-radius:8px;padding:22px 28px;
                    background:rgba({_hex_to_rgb(c.color)},0.10);margin-bottom:16px;">
            <div style="font-size:0.72rem;color:#8899aa;letter-spacing:0.1em;
                        text-transform:uppercase;">Deep Contrarian Signal</div>
            <div style="font-size:2rem;font-weight:900;color:{c.color};
                        letter-spacing:0.06em;">{c.label}</div>
            <div style="margin-top:6px;">
                {_css_badge(f"Confidence: {c.confidence}", "#2a3040")}
                &nbsp;{_css_badge(f"Phase: {r.market_phase}", CYCLE_PHASE_COLORS.get(r.market_phase, "#607080"))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Three metrics row
    m1, m2, m3 = st.columns(3)
    with m1:
        phase_color = CYCLE_PHASE_COLORS.get(r.market_phase, "#607080")
        st.markdown(
            f"**Market Phase**<br>{_css_badge(r.market_phase, phase_color)}"
            f"<br><span style='font-size:0.78rem;color:#8899aa;'>"
            f"{r.market_confidence:.0f}% confidence</span>",
            unsafe_allow_html=True,
        )
    with m2:
        pct = r.price_vs_ma200
        color = "#1aaa5a" if pct > 0 else "#cc3333"
        st.markdown(
            f"**Price vs 200D MA**<br>"
            f'<span style="color:{color};font-size:1.4rem;font-weight:700;">{pct:+.1f}%</span>',
            unsafe_allow_html=True,
        )
    with m3:
        if r.sentiment_score is not None:
            sent = r.sentiment_score
            sent_label = "Bearish" if sent < 30 else ("Bullish" if sent > 70 else "Neutral")
            sent_color = "#cc3333" if sent < 30 else ("#1aaa5a" if sent > 70 else "#aabbcc")
            st.markdown(
                f"**Retail Sentiment**<br>"
                f'<span style="color:{sent_color};font-size:1.4rem;font-weight:700;">'
                f"{sent:.0f}/100</span><br>"
                f"<span style='font-size:0.78rem;color:#8899aa;'>{sent_label}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("**Retail Sentiment**<br><span style='color:#607080;'>N/A</span>", unsafe_allow_html=True)

    # Rationale
    st.markdown("**Rationale**")
    for bullet in c.rationale:
        st.markdown(f"- {bullet}")
    if c.sentiment_note:
        st.caption(c.sentiment_note)

    # Cycle strip
    st.markdown("**14-Phase Cycle Position**")
    _render_cycle_strip(r.market_phase)


# ── Entry point ───────────────────────────────────────────────────────────────

def render_alpha_regime() -> None:
    """Main entry point — renders the full Alpha Regime tab."""
    st.markdown(
        "<h2 style='color:#00E5FF;letter-spacing:0.05em;margin-bottom:4px;'>"
        "ALPHA REGIME</h2>"
        "<div style='color:#607080;font-size:0.85rem;margin-bottom:16px;'>"
        "Dual-mode market confirmation system · Quality (Buffett/KAP) · "
        "Deep Contrarian (Rule/Sprott)</div>",
        unsafe_allow_html=True,
    )

    # ── Controls ─────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1.5, 0.8])
    with ctrl1:
        mode = st.radio(
            "Mode",
            options=["quality", "contrarian"],
            format_func=lambda x: "Quality" if x == "quality" else "Deep Contrarian",
            horizontal=True,
            key="ar_mode",
        )
    with ctrl2:
        ticker = st.text_input(
            "Stock ticker",
            value=st.session_state.get("ar_ticker", "ATCO-A.ST"),
            key="ar_ticker",
        ).strip().upper()
    with ctrl3:
        market_display = st.selectbox(
            "Market benchmark",
            options=list(_MARKET_OPTIONS.keys()),
            key="ar_market",
        )
        market_ticker = _MARKET_OPTIONS[market_display]
    with ctrl4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶ ANALYSE", key="ar_run", use_container_width=True)

    # ── State management ──────────────────────────────────────────────────────
    cache_key = f"ar_result_{ticker}_{mode}_{market_ticker}"
    result: Optional[RegimeResult] = st.session_state.get(cache_key)

    if run_btn or result is None:
        with st.spinner(f"Analysing {ticker} …"):
            result = run_regime_analysis(
                ticker=ticker,
                mode=mode,
                market_ticker=market_ticker,
            )
            st.session_state[cache_key] = result

    if result is None:
        st.info("Press ▶ ANALYSE to run.")
        return

    # ── Render ────────────────────────────────────────────────────────────────
    if result.mode == "quality":
        _render_quality_mode(result)
    else:
        _render_contrarian_mode(result)
