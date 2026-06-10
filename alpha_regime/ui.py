"""
alpha_regime/ui.py
Streamlit UI for the dual-mode Alpha Regime confirmation system.

Entry point: render_alpha_regime()
Modes: Quality (Buffett/KAP) · Deep Contrarian (Rule/Sprott)
"""
from __future__ import annotations

import logging
from typing import Optional

import plotly.graph_objects as go
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


# ── Contrarian verdict constants ──────────────────────────────────────────────

_STAGE_VERDICT: dict[str, dict] = {
    "ACCUMULATE_1": {"n": 1, "type": "ACCUMULATE", "sub": "Deploy first tranche of your planned position now"},
    "ACCUMULATE_2": {"n": 2, "type": "ACCUMULATE", "sub": "Deploy second tranche — despair/disbelief still present"},
    "ACCUMULATE_3": {"n": 3, "type": "ACCUMULATE", "sub": "Final accumulation window — hope returning, price near 200D"},
    "HOLD":         {"n": 0, "type": "HOLD",        "sub": "Hold existing positions · no new accumulation at these levels"},
    "DISTRIBUTE_1": {"n": 1, "type": "DISTRIBUTE",  "sub": "Trim 25–33% of position into current strength"},
    "DISTRIBUTE_2": {"n": 2, "type": "DISTRIBUTE",  "sub": "Distribute 50–75% of position · sell into greed"},
    "DISTRIBUTE_3": {"n": 3, "type": "DISTRIBUTE",  "sub": "Exit remaining long exposure · preserve capital"},
}

_STATUS_COLORS = {
    "RUBBER_BAND_STRETCHED": "#00E5FF",
    "TENSION_BUILDING":      "#c9a84c",
    "NEUTRAL":               "#607080",
    "DATA_GAP":              "#4a5060",
}


# ── Contrarian helpers ────────────────────────────────────────────────────────

def _render_verdict_banner_contrarian(r: RegimeResult, c) -> None:
    info = _STAGE_VERDICT.get(c.stage, {"n": 0, "type": "WAIT", "sub": "Conditions not met — await clearer signal"})
    t, n = info["type"], info["n"]
    if t == "ACCUMULATE":
        icon, headline = "🟢", f"ACCUMULATE — TRANCHE {n} OF 3"
    elif t == "DISTRIBUTE":
        icon, headline = "🔴", f"DISTRIBUTE — TRANCHE {n} OF 3"
    elif t == "HOLD":
        icon, headline = "🔵", "HOLD — MONITOR"
    else:
        icon, headline = "⚪", "WAIT — CONDITIONS NOT MET"

    phase_badge = _css_badge(r.market_phase, CYCLE_PHASE_COLORS.get(r.market_phase, "#607080"))
    conf_badge  = _css_badge(f"Confidence: {c.confidence}", "#2a3040")
    color = c.color

    st.markdown(
        f"""<div style="border:3px solid {color};border-radius:8px;padding:22px 28px;
            background:rgba({_hex_to_rgb(color)},0.10);margin-bottom:16px;">
          <div style="font-size:0.72rem;color:#8899aa;letter-spacing:0.1em;
              text-transform:uppercase;">Deep Contrarian Signal</div>
          <div style="font-size:2.2rem;font-weight:900;color:{color};
              letter-spacing:0.06em;margin:4px 0;">{icon} {headline}</div>
          <div style="font-size:0.9rem;color:#aabbcc;margin-bottom:8px;">{info['sub']}</div>
          {conf_badge}&nbsp;{phase_badge}
        </div>""",
        unsafe_allow_html=True,
    )


def _render_next_trigger(r: RegimeResult, c) -> None:
    stage = c.stage
    if stage == "ACCUMULATE_1":
        label = "ACCUMULATE 2/3"
        text  = (f"Market cycle must shift out of {r.market_phase} phase into DISBELIEF or ANGER "
                 f"(currently <b>{r.market_phase}</b> at {r.market_confidence:.0f}% confidence)")
    elif stage == "ACCUMULATE_2":
        label = "ACCUMULATE 3/3"
        text  = (f"Market must enter HOPE phase (currently: <b>{r.market_phase}</b>). "
                 f"Price {r.price_vs_ma200:+.1f}% vs 200D MA — watch for upward recross toward 0%")
    elif stage == "ACCUMULATE_3":
        label = "HOLD"
        low3  = f"3m low: <b>{r.price_3m_low:.2f}</b>" if r.price_3m_low else ""
        low6  = f", prior 3m low: <b>{r.price_6m_low:.2f}</b>" if r.price_6m_low else ""
        text  = (f"Market must confirm OPTIMISM/BELIEF phase (currently: <b>{r.market_phase}</b>). "
                 + (f"{low3}{low6} — watch for higher low formation" if low3 else ""))
    elif stage == "HOLD":
        label = "DISTRIBUTE 1/3"
        text  = (f"Cycle shift to THRILL triggers first distribution tranche. "
                 f"Price currently <b>{r.price_vs_ma200:+.1f}%</b> above 200D MA — "
                 f"monitor cycle confidence ({r.market_confidence:.0f}%) and sentiment extremes")
    elif stage == "DISTRIBUTE_1":
        label = "DISTRIBUTE 2/3"
        text  = "Cycle must enter EUPHORIA or COMPLACENCY. <b>Trim 25–33%</b> of position into current strength now."
    elif stage == "DISTRIBUTE_2":
        label = "DISTRIBUTE 3/3"
        text  = "Cycle must enter ANXIETY or DENIAL. <b>Distribute 50–75%</b> of position now."
    elif stage == "DISTRIBUTE_3":
        label = "EXIT"
        text  = (f"<b>Exit remaining long exposure</b> — PANIC/CAPITULATION approaching. "
                 f"Price {r.price_vs_ma200:+.1f}% vs 200D MA.")
    else:
        return

    st.markdown(
        f"""<div style="border:1px solid #c9a84c;border-radius:6px;padding:14px 18px;
            background:#100e08;margin:12px 0 16px;">
          <div style="font-size:0.7rem;color:#8899aa;letter-spacing:0.1em;
              text-transform:uppercase;margin-bottom:4px;">Next Trigger → {label}</div>
          <div style="font-size:0.88rem;color:#e8e4dc;">{text}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_rubber_band_gauge(r: RegimeResult, exposure_override) -> None:
    """Render horizontal percentile gauge + 10y sparkline for the relevant commodity ratio."""
    # Determine which ratio key to display
    if exposure_override is None:          # Auto
        ratio_key = r.detected_exposure
    elif exposure_override == "":          # user chose "None"
        return
    else:
        ratio_key = exposure_override

    if not ratio_key:
        st.caption("Commodity exposure not detected — select manually in the control above.")
        return

    if not r.commodity_ratios:
        st.caption("Commodity ratio data unavailable (module not loaded).")
        return

    ratio = r.commodity_ratios.get(ratio_key)
    if ratio is None:
        st.caption(f"No ratio data for key: {ratio_key}")
        return

    if ratio.status == "DATA_GAP":
        st.warning(f"Commodity data gap — {ratio.label}: {ratio.error or 'data unavailable'}")
        return

    status_color = _STATUS_COLORS.get(ratio.status, "#607080")

    st.markdown(
        f'<p style="color:#c9a84c;font-weight:600;margin-bottom:6px;">'
        f'Rubber Band Gauge — {ratio.label}</p>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"**Ratio value** `{ratio.current:.5f}`")
    with m2:
        st.markdown(
            f"**10y Percentile** {_css_badge(f'{ratio.percentile:.1f}th', status_color)}",
            unsafe_allow_html=True,
        )
    with m3:
        z_color = "#1aaa5a" if ratio.zscore > 1.5 else ("#cc3333" if ratio.zscore < -1.5 else "#aabbcc")
        st.markdown(
            f"**Z-score** {_css_badge(f'{ratio.zscore:+.2f}σ', z_color)}",
            unsafe_allow_html=True,
        )

    _STATUS_LABELS = {
        "RUBBER_BAND_STRETCHED": f"RUBBER BAND STRETCHED — {ratio.denominator_label} historically stretched cheap",
        "TENSION_BUILDING":      f"TENSION BUILDING — {ratio.denominator_label} approaching extreme cheap",
        "NEUTRAL":               "NEUTRAL — no extreme commodity reading",
    }
    st.markdown(
        f'<div style="color:{status_color};font-size:0.85rem;font-weight:700;margin:4px 0 10px;">'
        f'{_STATUS_LABELS.get(ratio.status, ratio.status)}</div>',
        unsafe_allow_html=True,
    )

    # ── Horizontal percentile gauge ───────────────────────────────────────────
    pct = ratio.percentile
    if ratio.cheap_direction == "high":
        # Stretched zone on the right (high = cheap)
        tension_zone  = '<div style="position:absolute;left:80%;width:10%;top:0;bottom:0;background:rgba(201,168,76,0.22);"></div>'
        stretched_zone = '<div style="position:absolute;left:90%;right:0;top:0;bottom:0;background:rgba(0,229,255,0.28);"></div>'
        cheap_label, cheap_align = "CHEAP ZONE →", "right"
    else:
        # Stretched zone on the left (low = cheap for copper_gold)
        tension_zone  = '<div style="position:absolute;left:10%;width:10%;top:0;bottom:0;background:rgba(201,168,76,0.22);"></div>'
        stretched_zone = '<div style="position:absolute;left:0;width:10%;top:0;bottom:0;background:rgba(0,229,255,0.28);"></div>'
        cheap_label, cheap_align = "← CHEAP ZONE", "left"

    bar_color_hex = status_color
    gauge_html = f"""
    <div style="margin:8px 0 2px;">
      <div style="position:relative;background:#1a2030;border-radius:6px;height:28px;overflow:hidden;">
        {tension_zone}
        {stretched_zone}
        <div style="position:absolute;left:0;width:{pct:.1f}%;top:0;bottom:0;
            background:linear-gradient(to right,#2a3545,{bar_color_hex}55);pointer-events:none;"></div>
        <div style="position:absolute;left:{pct:.1f}%;top:2px;bottom:2px;width:3px;
            background:#ffffff;border-radius:2px;transform:translateX(-50%);pointer-events:none;"></div>
        <span style="position:absolute;left:4px;top:50%;transform:translateY(-50%);
            font-size:0.62rem;color:#6677aa;">0</span>
        <span style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
            font-size:0.62rem;color:#6677aa;pointer-events:none;">50</span>
        <span style="position:absolute;right:4px;top:50%;transform:translateY(-50%);
            font-size:0.62rem;color:#6677aa;">100</span>
      </div>
      <div style="font-size:0.65rem;color:#00E5FF;text-align:{cheap_align};margin-top:2px;">{cheap_label}</div>
    </div>"""
    st.markdown(gauge_html, unsafe_allow_html=True)

    # ── 10y sparkline ─────────────────────────────────────────────────────────
    if ratio.sparkline_dates and ratio.sparkline_values:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ratio.sparkline_dates,
            y=ratio.sparkline_values,
            mode="lines",
            line=dict(color="#00E5FF", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0,229,255,0.06)",
            name=ratio.label,
        ))
        fig.add_hline(
            y=ratio.current,
            line_dash="dot", line_color="#c9a84c", line_width=1,
            annotation_text=f"  current {ratio.current:.4f}",
            annotation_font_color="#c9a84c",
            annotation_font_size=10,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0D1117",
            font=dict(color="#aabbcc", size=10),
            xaxis=dict(gridcolor="rgba(138,133,120,0.1)", title=""),
            yaxis=dict(gridcolor="rgba(138,133,120,0.1)", title=""),
            margin=dict(l=40, r=20, t=8, b=28),
            height=130,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"rb_spark_{ratio_key}")


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


def _render_contrarian_mode(r: RegimeResult, exposure_override=None) -> None:
    if r.error:
        st.error(r.error)
        return

    c = r.contrarian
    if c is None:
        st.warning("Contrarian stage could not be determined.")
        return

    # 1. Verdict banner
    _render_verdict_banner_contrarian(r, c)

    # 2. Next trigger box
    _render_next_trigger(r, c)

    # 3. Rubber band gauge
    _render_rubber_band_gauge(r, exposure_override)

    # 4. Checklist
    if c.rationale:
        st.markdown(
            '<p style="color:#c9a84c;font-weight:600;margin-bottom:4px;">Checklist</p>',
            unsafe_allow_html=True,
        )
        for bullet in c.rationale:
            st.markdown(f"- {bullet}")
    if c.sentiment_note:
        st.caption(c.sentiment_note)

    # 5. Market context + cycle strip
    _render_market_context(r)
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

    # ── Commodity exposure override (contrarian mode only) ────────────────────
    exposure_override: Optional[str] = None
    if mode == "contrarian":
        _exp_col, _ = st.columns([2, 5])
        with _exp_col:
            _EXP_OPTIONS = ["Auto", "Silver", "Gold Miners", "Oil", "Copper", "None"]
            _EXP_TO_KEY: dict[str, Optional[str]] = {
                "Auto":        None,
                "Silver":      "gold_silver",
                "Gold Miners": "metal_miners",
                "Oil":         "gold_oil",
                "Copper":      "copper_gold",
                "None":        "",
            }
            _exp_sel = st.selectbox(
                "Commodity exposure",
                options=_EXP_OPTIONS,
                key="ar_exposure",
                help="Override auto-detected commodity sector for the rubber band gauge",
            )
            exposure_override = _EXP_TO_KEY.get(_exp_sel)

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
        _render_contrarian_mode(result, exposure_override=exposure_override)
