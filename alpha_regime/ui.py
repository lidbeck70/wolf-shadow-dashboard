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

# ── Optional: direct commodity ratio helpers for custom builder ───────────────
_CR_OK = False
try:
    from alpha_regime.commodity_ratios import (
        _download_close as _cr_download_close,
        _percentile_of  as _cr_percentile_of,
        _classify       as _cr_classify,
        RatioResult     as _CRRatioResult,
    )
    _CR_OK = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_MARKET_OPTIONS = {
    "SPY — S&P 500 (US)":              "SPY",
    "QQQ — Nasdaq 100 (US)":           "QQQ",
    "^OMX — OMXS30 (Stockholm)":       "^OMX",
    "OBX — Oslo Børs":                 "OBX.OL",
    "^OMXC25 — OMX Copenhagen 25":     "^OMXC25",
    "^OMXH25 — OMX Helsinki 25":       "^OMXH25",
    "GDX — Gold Miners ETF":           "GDX",
    "GDXJ — Junior Gold Miners ETF":   "GDXJ",
    "GLD — Gold":                      "GLD",
    "SLV — Silver":                    "SLV",
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

# Thresholds mirrored from commodity_ratios.py (avoid circular import)
_GAUGE_STRETCHED_PCT  = 90.0
_GAUGE_TENSION_PCT    = 80.0
_GAUGE_CHEAP_LOW_MAX  = 10.0
_GAUGE_CHEAP_LOW_TENS = 20.0


def _ratio_verdict(ratio) -> tuple:
    """Return (badge_text, color, explanation) for a RatioResult."""
    pct = ratio.percentile
    d   = ratio.cheap_direction
    lbl = ratio.label
    den = ratio.denominator_label

    if d == "high":
        if pct >= _GAUGE_STRETCHED_PCT:
            return ("🟢 KÖP-STÖD",      "#1aaa5a",
                    f"{lbl} vid {pct:.0f}:e percentilen — {den} historiskt max-billig")
        if pct >= _GAUGE_TENSION_PCT:
            return ("⚡ SPÄNNING BYGGS", "#c9a84c",
                    f"{lbl} vid {pct:.0f}:e percentilen — {den} närmar sig billighetszonen")
        if pct <= 10.0:
            return ("🔴 SÄLJ-STÖD",      "#cc3333",
                    f"{lbl} vid {pct:.0f}:e percentilen — {den} historiskt max-dyr")
        return ("⚪ NEUTRAL", "#607080",
                f"{lbl} vid {pct:.0f}:e percentilen — ingen extrem")
    else:  # "low" direction (copper_gold, gdx_spy)
        if pct <= _GAUGE_CHEAP_LOW_MAX:
            return ("🟢 KÖP-STÖD",      "#1aaa5a",
                    f"{lbl} vid {pct:.0f}:e percentilen — {den} historiskt max-billig vs motvikten")
        if pct <= _GAUGE_CHEAP_LOW_TENS:
            return ("⚡ SPÄNNING BYGGS", "#c9a84c",
                    f"{lbl} vid {pct:.0f}:e percentilen — {den} närmar sig billighetszonen")
        if pct >= 90.0:
            return ("🔴 SÄLJ-STÖD",      "#cc3333",
                    f"{lbl} vid {pct:.0f}:e percentilen — {den} historiskt max-dyr vs motvikten")
        return ("⚪ NEUTRAL", "#607080",
                f"{lbl} vid {pct:.0f}:e percentilen — ingen extrem")


def _compute_ratio_summary(commodity_ratios: dict, keys: list) -> str:
    """Return e.g. 'Gummisnoddar: 2 KÖP-STÖD, 1 NEUTRAL' for the exposure's ratio list."""
    counts: dict[str, int] = {"KÖP-STÖD": 0, "SÄLJ-STÖD": 0, "SPÄNNING": 0, "NEUTRAL": 0, "DATA_GAP": 0}
    for key in keys:
        r = commodity_ratios.get(key)
        if r is None or r.status == "DATA_GAP":
            counts["DATA_GAP"] += 1
            continue
        badge, _, _ = _ratio_verdict(r)
        if "KÖP-STÖD" in badge:
            counts["KÖP-STÖD"] += 1
        elif "SÄLJ-STÖD" in badge:
            counts["SÄLJ-STÖD"] += 1
        elif "SPÄNNING" in badge:
            counts["SPÄNNING"] += 1
        else:
            counts["NEUTRAL"] += 1

    parts = []
    for label, key in [("KÖP-STÖD", "KÖP-STÖD"), ("SÄLJ-STÖD", "SÄLJ-STÖD"),
                        ("SPÄNNING", "SPÄNNING"), ("NEUTRAL", "NEUTRAL")]:
        if counts[key] > 0:
            parts.append(f"{counts[key]} {label}")
    if counts["DATA_GAP"] > 0:
        parts.append(f"{counts['DATA_GAP']} DATA_GAP")
    return ("Gummisnoddar: " + ", ".join(parts)) if parts else ""


# ── Contrarian helpers ────────────────────────────────────────────────────────

def _render_verdict_banner_contrarian(r: RegimeResult, c, ratio_summary: str = "") -> None:
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

    summary_line = (
        f'<div style="font-size:0.78rem;color:#8899aa;margin-top:6px;">{ratio_summary}</div>'
        if ratio_summary else ""
    )

    st.markdown(
        f"""<div style="border:3px solid {color};border-radius:8px;padding:22px 28px;
            background:rgba({_hex_to_rgb(color)},0.10);margin-bottom:16px;">
          <div style="font-size:0.72rem;color:#8899aa;letter-spacing:0.1em;
              text-transform:uppercase;">Deep Contrarian Signal</div>
          <div style="font-size:2.2rem;font-weight:900;color:{color};
              letter-spacing:0.06em;margin:4px 0;">{icon} {headline}</div>
          <div style="font-size:0.9rem;color:#aabbcc;margin-bottom:8px;">{info['sub']}</div>
          {conf_badge}&nbsp;{phase_badge}
          {summary_line}
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


def _render_single_ratio(ratio, idx: int) -> None:
    """Render one ratio gauge row: verdict badge, percentile bar, sparkline."""
    if ratio.status == "DATA_GAP":
        st.warning(f"DATA_GAP — {ratio.label}: {ratio.error or 'data unavailable'}")
        return

    badge_text, badge_color, explanation = _ratio_verdict(ratio)
    status_color = _STATUS_COLORS.get(ratio.status, "#607080")

    # Header row
    h1, h2, h3, h4 = st.columns([2.2, 1.2, 1.0, 1.0])
    with h1:
        st.markdown(
            f'<span style="font-weight:700;color:#e8e4dc;">{ratio.label}</span>',
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(_css_badge(badge_text, badge_color), unsafe_allow_html=True)
    with h3:
        st.markdown(
            _css_badge(f"{ratio.percentile:.0f}th pct", status_color),
            unsafe_allow_html=True,
        )
    with h4:
        z_color = "#1aaa5a" if ratio.zscore > 1.5 else ("#cc3333" if ratio.zscore < -1.5 else "#8899aa")
        st.markdown(
            f'<span style="color:{z_color};font-size:0.82rem;">{ratio.zscore:+.2f}σ</span>',
            unsafe_allow_html=True,
        )

    # Explanation line
    st.markdown(
        f'<div style="font-size:0.78rem;color:#8899aa;margin:2px 0 6px;">{explanation}</div>',
        unsafe_allow_html=True,
    )

    # Horizontal percentile gauge
    pct = ratio.percentile
    if ratio.cheap_direction == "high":
        tension_zone   = '<div style="position:absolute;left:80%;width:10%;top:0;bottom:0;background:rgba(201,168,76,0.22);"></div>'
        stretched_zone = '<div style="position:absolute;left:90%;right:0;top:0;bottom:0;background:rgba(0,229,255,0.28);"></div>'
        cheap_label, cheap_align = "CHEAP ZONE →", "right"
    else:
        tension_zone   = '<div style="position:absolute;left:10%;width:10%;top:0;bottom:0;background:rgba(201,168,76,0.22);"></div>'
        stretched_zone = '<div style="position:absolute;left:0;width:10%;top:0;bottom:0;background:rgba(0,229,255,0.28);"></div>'
        cheap_label, cheap_align = "← CHEAP ZONE", "left"

    st.markdown(
        f"""<div style="margin:4px 0 2px;">
          <div style="position:relative;background:#1a2030;border-radius:6px;height:24px;overflow:hidden;">
            {tension_zone}{stretched_zone}
            <div style="position:absolute;left:0;width:{pct:.1f}%;top:0;bottom:0;
                background:linear-gradient(to right,#2a3545,{status_color}55);pointer-events:none;"></div>
            <div style="position:absolute;left:{pct:.1f}%;top:2px;bottom:2px;width:3px;
                background:#ffffff;border-radius:2px;transform:translateX(-50%);pointer-events:none;"></div>
            <span style="position:absolute;left:4px;top:50%;transform:translateY(-50%);
                font-size:0.6rem;color:#6677aa;">0</span>
            <span style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
                font-size:0.6rem;color:#6677aa;pointer-events:none;">50</span>
            <span style="position:absolute;right:4px;top:50%;transform:translateY(-50%);
                font-size:0.6rem;color:#6677aa;">100</span>
          </div>
          <div style="font-size:0.6rem;color:#00E5FF;text-align:{cheap_align};margin-top:2px;">{cheap_label}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # 10y sparkline
    if ratio.sparkline_dates and ratio.sparkline_values:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ratio.sparkline_dates,
            y=ratio.sparkline_values,
            mode="lines",
            line=dict(color="#00E5FF", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(0,229,255,0.05)",
            name=ratio.label,
        ))
        fig.add_hline(
            y=ratio.current,
            line_dash="dot", line_color="#c9a84c", line_width=1,
            annotation_text=f"  {ratio.current:.4f}",
            annotation_font_color="#c9a84c",
            annotation_font_size=9,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0D1117",
            font=dict(color="#aabbcc", size=9),
            xaxis=dict(gridcolor="rgba(138,133,120,0.08)", title=""),
            yaxis=dict(gridcolor="rgba(138,133,120,0.08)", title=""),
            margin=dict(l=36, r=16, t=4, b=24),
            height=110,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"rb_spark_{ratio.key}_{idx}")


def _render_rubber_band_gauge(r: RegimeResult, exposure_override) -> None:
    """Render all commodity ratio gauges for the current exposure, stacked vertically."""
    # Resolve which list of ratio keys to show
    if exposure_override is None:          # Auto
        keys = r.detected_exposure or []
    elif isinstance(exposure_override, list):
        if not exposure_override:          # user chose "None" (empty list)
            return
        keys = exposure_override
    else:
        keys = []

    if not keys:
        st.caption("Commodity exposure not detected — select manually in the control above.")
        return

    if not r.commodity_ratios:
        st.caption("Commodity ratio data unavailable (module not loaded).")
        return

    st.markdown(
        '<p style="color:#c9a84c;font-weight:600;margin-bottom:8px;">Rubber Band Gauges</p>',
        unsafe_allow_html=True,
    )

    for i, key in enumerate(keys):
        ratio = r.commodity_ratios.get(key)
        if ratio is None:
            st.caption(f"No data for ratio key: {key}")
            continue
        _render_single_ratio(ratio, idx=i)
        if i < len(keys) - 1:
            st.markdown('<hr style="border-color:#2a3040;margin:8px 0;">', unsafe_allow_html=True)


# ── Custom ratio builder ─────────────────────────────────────────────────────

def _render_custom_ratio_builder() -> None:
    """Expander with two ticker inputs — compute and cache a custom rubber-band ratio."""
    with st.expander("➕ Egen ratio", expanded=False):
        cb1, cb2, cb3 = st.columns([1.2, 1.2, 0.8])
        with cb1:
            num_t = st.text_input("Täljare (t.ex. GLD)", value="GLD",
                                  key="cr_num").strip().upper()
        with cb2:
            den_t = st.text_input("Nämnare (t.ex. GDXJ)", value="GDXJ",
                                  key="cr_den").strip().upper()
        with cb3:
            period = st.selectbox("Period", ["5y", "10y", "max"], key="cr_period")

        run_custom = st.button("Beräkna", key="cr_run")
        saved: list = st.session_state.get("ar_custom_ratios", [])

        if run_custom:
            if not num_t or not den_t:
                st.error("Ange båda tickerkoderna.")
            elif not _CR_OK:
                st.error("Commodity ratio-modulen är inte tillgänglig.")
            else:
                with st.spinner(f"Hämtar {num_t} / {den_t} ({period}) …"):
                    try:
                        import numpy as np
                        import pandas as pd

                        num_s = _cr_download_close(num_t, period=period)
                        den_s = _cr_download_close(den_t, period=period)

                        if num_s.empty:
                            st.error(f"Ingen kursdata för täljaren '{num_t}'. "
                                     "Kontrollera att tickerkoden är rätt (t.ex. GDXJ, GLD, AAPL).")
                        elif den_s.empty:
                            st.error(f"Ingen kursdata för nämnaren '{den_t}'. "
                                     "Kontrollera att tickerkoden är rätt.")
                        else:
                            aligned = pd.concat({"n": num_s, "d": den_s}, axis=1).dropna()
                            if len(aligned) < 50:
                                st.error(
                                    f"För lite gemensam historik ({len(aligned)} dagar). "
                                    "Välj längre period eller andra tickerkoder."
                                )
                            else:
                                series = aligned["n"] / aligned["d"]
                                arr    = series.values.astype(float)
                                cur    = float(arr[-1])
                                pct    = _cr_percentile_of(arr, cur)
                                mean, std = float(arr.mean()), float(arr.std())
                                zscore = round((cur - mean) / std, 2) if std > 0 else 0.0

                                series_w = series.resample("W").last().dropna()
                                cr = _CRRatioResult(
                                    key=f"custom_{num_t}_{den_t}",
                                    label=f"{num_t} / {den_t}",
                                    denominator_label=den_t,
                                    cheap_direction="high",
                                    current=round(cur, 5),
                                    percentile=round(pct, 1),
                                    zscore=zscore,
                                    status=_cr_classify(pct, "high"),
                                    sparkline_dates=[str(d.date()) for d in series_w.index],
                                    sparkline_values=[round(float(v), 5) for v in series_w.values],
                                )
                                saved = [x for x in saved if x.label != cr.label]
                                saved = [cr] + saved
                                saved = saved[:5]
                                st.session_state["ar_custom_ratios"] = saved
                    except Exception as exc:
                        st.error(f"Kunde inte beräkna ratio: {exc}")

        if saved:
            st.markdown(
                '<p style="color:#c9a84c;font-weight:600;margin:14px 0 6px;">'
                'Sparade egna ratios (senaste 5)</p>',
                unsafe_allow_html=True,
            )
            for i, cr in enumerate(saved):
                _render_single_ratio(cr, idx=1000 + i)
                if i < len(saved) - 1:
                    st.markdown(
                        '<hr style="border-color:#2a3040;margin:8px 0;">',
                        unsafe_allow_html=True,
                    )


# ── Action box ────────────────────────────────────────────────────────────────

def _render_action_box(r: RegimeResult) -> None:
    """Top-level Swedish action box — what to do right now, in plain language."""
    if r.mode == "quality":
        verdict = r.quality_verdict
        passed  = r.signals_passed
        total   = sum(1 for s in r.signals if s.label != "NO DATA")
        if verdict == "BUY":
            action_key = "KÖP NU"
            color      = "#1aaa5a"
            action_txt = f"Alla {total} köpvillkor är uppfyllda. Köp nu — max 10% av portföljvärdet per aktie."
            why_txt    = "Upptrend bekräftad, rimlig värdering, gynnsamt marknadsläge och hög bolagskvalitet."
        elif verdict == "WATCH":
            action_key = "AVVAKTA"
            color      = "#607080"
            action_txt = "Lägg aktien på bevakningslista och vänta tills alla villkor är uppfyllda."
            why_txt    = f"{passed} av {total} signaler är gröna — ett villkor saknas fortfarande."
        else:
            action_key = "AVVAKTA"
            color      = "#607080"
            action_txt = "Vänta. Tillräckligt många köpvillkor är inte uppfyllda ännu."
            why_txt    = f"Bara {passed} av {total} signaler godkända — se vilka som inte klarar nedan."
    else:  # contrarian
        c = r.contrarian
        if c is None:
            return
        stage = c.stage
        if stage.startswith("ACCUMULATE"):
            action_key = "KÖP NU"
            color      = c.color
            n          = int(stage[-1])
            stop_part  = (
                f" Sätt mental stopp under 3-månadersbotten ({r.price_3m_low:.2f})."
                if r.price_3m_low else ""
            )
            if n == 1:
                action_txt = f"Köp första tredjedelen (1/3) av din planerade position nu.{stop_part}"
                why_txt    = (f"Marknaden är i {r.market_phase} — maximalt pessimism och kapitulation. "
                              "Rick Rule: det bästa köpläget finns när alla ger upp.")
            elif n == 2:
                action_txt = f"Köp andra tredjedelen (2/3) av din position nu.{stop_part}"
                why_txt    = (f"Marknaden är i {r.market_phase} — tvivel och ilska dominerar. "
                              "Sprott: pressa in mer när pessimismen håller priset nere.")
            else:
                action_txt = f"Köp sista tredjedelen (3/3) av din position nu.{stop_part}"
                why_txt    = (f"Hoppfas börjar ({r.market_phase}) — sista chansen att köpa billigt "
                              "innan trenden bekräftas.")
        elif stage == "HOLD":
            action_key = "AVVAKTA"
            color      = "#607080"
            action_txt = "Håll befintliga positioner. Köp inga nya andelar till dessa priser."
            why_txt    = (f"Trenden är bekräftad ({r.market_phase}) — ackumuleringsfönstret är stängt. "
                          "Vänta på nästa distributionssignal.")
        elif stage.startswith("DISTRIBUTE"):
            action_key = "SÄLJ / TA HEM"
            color      = "#FF6B3D"
            n          = int(stage[-1])
            if n == 1:
                action_txt = "Sälj 25–33% av din position nu. Ta hem vinst och minska risken."
                why_txt    = (f"Marknaden är i {r.market_phase} — FOMO och entusiasm driver priserna. "
                              "Sprott: sälj till glada köpare.")
            elif n == 2:
                action_txt = "Sälj ytterligare 50% av din position. Distribuera aggressivt."
                why_txt    = (f"Marknaden är i {r.market_phase} — extrem girighet. "
                              "Rick Rule: sälj när alla andra köper.")
            else:
                action_txt = "Sälj resterande position omedelbart. Bevara kapitalet."
                why_txt    = (f"Marknaden är i {r.market_phase} — trenden bryts ner. "
                              "Bevara kapitalet till nästa köpcykel.")
        else:
            action_key = "AVVAKTA"
            color      = "#607080"
            action_txt = "Ingen tydlig signal. Gör ingenting och vänta på klarare marknadsläge."
            why_txt    = f"Oklar marknadsfas ({r.market_phase}) — cykeln behöver mer tid att klargöras."

    icon = "🟢" if action_key == "KÖP NU" else ("🔴" if "SÄLJ" in action_key else "⚪")
    st.markdown(
        f"""<div style="border:3px solid {color};border-radius:10px;padding:24px 28px;
            background:rgba({_hex_to_rgb(color)},0.12);margin-bottom:20px;">
          <div style="font-size:0.7rem;color:#8899aa;letter-spacing:0.12em;
              text-transform:uppercase;margin-bottom:6px;">Vad ska jag göra?</div>
          <div style="font-size:2.6rem;font-weight:900;color:{color};
              letter-spacing:0.04em;margin-bottom:10px;">{icon}&nbsp;{action_key}</div>
          <div style="font-size:0.98rem;color:#e8e4dc;font-weight:600;
              margin-bottom:6px;">{action_txt}</div>
          <div style="font-size:0.86rem;color:#aabbcc;">{why_txt}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Mode renderers ────────────────────────────────────────────────────────────

def _render_quality_mode(r: RegimeResult) -> None:
    if r.error:
        st.error(r.error)
        return

    _render_action_box(r)

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

    # Resolve display keys (same logic as gauge)
    if exposure_override is None:
        _display_keys = r.detected_exposure or []
    elif isinstance(exposure_override, list):
        _display_keys = exposure_override
    else:
        _display_keys = []

    # 0. Action box (Swedish, beginner-friendly)
    _render_action_box(r)

    # 1. Verdict banner (with ratio summary)
    _ratio_sum = _compute_ratio_summary(r.commodity_ratios, _display_keys)
    _render_verdict_banner_contrarian(r, c, ratio_summary=_ratio_sum)

    # 2. Next trigger box
    _render_next_trigger(r, c)

    # 3. Rubber band gauge + custom builder
    _render_rubber_band_gauge(r, exposure_override)
    _render_custom_ratio_builder()

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
    exposure_override = None   # None = auto-detect
    if mode == "contrarian":
        _exp_col, _ = st.columns([2, 5])
        with _exp_col:
            _EXP_OPTIONS = ["Auto", "Gold Miners", "Junior Miners", "Silver", "Oil", "Copper", "None"]
            _EXP_TO_KEY: dict[str, object] = {
                "Auto":          None,
                "Gold Miners":   ["metal_miners", "gdxj_gdx", "gold_gdxj", "gdx_spy"],
                "Junior Miners": ["gdxj_gdx", "gold_gdxj"],
                "Silver":        ["gold_silver", "silver_juniors"],
                "Oil":           ["gold_oil"],
                "Copper":        ["copper_gold"],
                "None":          [],
            }
            _exp_sel = st.selectbox(
                "Commodity exposure",
                options=_EXP_OPTIONS,
                key="ar_exposure",
                help="Override auto-detected commodity sector for the rubber band gauges",
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
