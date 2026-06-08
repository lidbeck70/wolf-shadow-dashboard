"""
tabs/market_cycle.py
====================
Market Cycle Engine — 14-phase psychology cycle detector for any ticker.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from ui.theme import inject_css, section_title, card as _card, PALETTE as _P
from market_cycle.rules import MARKET_CYCLE_RULES, PHASE_ORDER
from market_cycle.cache import cached_market_cycle_analysis, cached_market_cycle_history

_BG   = _P["bg"]
_BG2  = _P["bg2"]
_BG3  = _P["bg3"]
_GOLD = _P["gold"]
_DIM  = _P["text_dim"]
_TEXT = _P["text"]

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0c0c12",
    plot_bgcolor="#0c0c12",
    font=dict(color=_TEXT, family="Courier New"),
    margin=dict(l=24, r=24, t=40, b=24),
)

# ── Gauge ─────────────────────────────────────────────────────────────────────

def _build_gauge(phase: str, confidence: float) -> go.Figure:
    step_size = 100.0 / len(PHASE_ORDER)
    current_idx = PHASE_ORDER.index(phase)
    needle_val = (current_idx + 0.5) * step_size

    steps = []
    for i, name in enumerate(PHASE_ORDER):
        cfg = MARKET_CYCLE_RULES[name]
        lo, hi = i * step_size, (i + 1) * step_size
        alpha = "cc" if name == phase else "28"
        steps.append({"range": [lo, hi], "color": cfg["color"] + alpha})

    cfg = MARKET_CYCLE_RULES[phase]
    label = phase.replace("_", " ")
    subtitle = cfg["description"][:72] + ("…" if len(cfg["description"]) > 72 else "")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=needle_val,
        number={"font": {"size": 1, "color": "rgba(0,0,0,0)"}},
        title={
            "text": (
                f"<b>{cfg['emoji']} {label}</b><br>"
                f"<span style='font-size:0.68em;color:{_DIM};'>{subtitle}</span>"
            ),
            "font": {"size": 20, "color": _GOLD, "family": "Courier New"},
        },
        gauge={
            "axis": {"range": [0, 100], "visible": False},
            "steps": steps,
            "bar": {
                "color": cfg["color"],
                "thickness": 0.06,
            },
            "bgcolor": "#0c0c12",
            "borderwidth": 2,
            "bordercolor": _P["border_hi"],
        },
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=360,
        margin=dict(l=30, r=30, t=100, b=10),
    )
    return fig


# ── Phase score bar chart ─────────────────────────────────────────────────────

def _build_score_chart(phase_scores: dict) -> go.Figure:
    phases = list(phase_scores.keys())
    scores = [phase_scores[p] for p in phases]
    colors = [
        MARKET_CYCLE_RULES[p]["color"] + ("ff" if scores[i] == max(scores) else "88")
        for i, p in enumerate(phases)
    ]
    labels = [p.replace("_", " ") for p in phases]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.0f}%" for s in scores],
        textposition="outside",
        textfont=dict(color=_TEXT, size=10),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=420,
        xaxis=dict(
            range=[0, 115],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            autorange="reversed",
            showgrid=False,
            tickfont=dict(size=10, color=_DIM),
        ),
        bargap=0.25,
        margin=dict(l=110, r=50, t=16, b=16),
    )
    return fig


# ── Historical timeline ───────────────────────────────────────────────────────

def _build_timeline(history: list[dict]) -> go.Figure:
    if not history:
        return None

    df = pd.DataFrame(history)
    df["label"] = df["phase"].str.replace("_", " ")
    df["color"] = df["phase"].map(lambda p: MARKET_CYCLE_RULES[p]["color"])
    df["emoji"] = df["phase"].map(lambda p: MARKET_CYCLE_RULES[p]["emoji"])

    fig = go.Figure()

    # Colored area background per phase band
    for i, phase in enumerate(PHASE_ORDER):
        cfg = MARKET_CYCLE_RULES[phase]
        fig.add_hrect(
            y0=i - 0.5, y1=i + 0.5,
            fillcolor=cfg["color"] + "18",
            line_width=0,
        )

    # Line + scatter
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["phase_index"],
        mode="lines+markers",
        line=dict(color=_GOLD, width=1.5),
        marker=dict(
            color=df["color"].tolist(),
            size=8,
            line=dict(color=_GOLD, width=1),
        ),
        text=df.apply(lambda r: f"{r['emoji']} {r['label']}<br>Conf: {r['confidence']:.0f}%", axis=1),
        hovertemplate="<b>%{text}</b><br>%{x|%Y-%m-%d}<extra></extra>",
        name="Phase",
    ))

    tickvals = list(range(len(PHASE_ORDER)))
    ticktext = [
        f"{MARKET_CYCLE_RULES[p]['emoji']} {p.replace('_', ' ')}"
        for p in PHASE_ORDER
    ]

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=460,
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=9, color=_DIM),
        ),
        yaxis=dict(
            tickvals=tickvals,
            ticktext=ticktext,
            tickfont=dict(size=9, color=_DIM),
            showgrid=True,
            gridcolor="rgba(201,168,76,0.06)",
            range=[-0.6, len(PHASE_ORDER) - 0.4],
        ),
        showlegend=False,
        margin=dict(l=160, r=24, t=24, b=40),
    )
    return fig


# ── Rules breakdown ───────────────────────────────────────────────────────────

def _render_rules_breakdown(matched_rules: dict, phase_cfg: dict) -> None:
    color = phase_cfg["color"]
    matched = matched_rules.get("matched", [])
    unmatched = matched_rules.get("unmatched", [])

    def _op_label(op: str, value) -> str:
        if op == "gt":
            return f"> {value}"
        if op == "gte":
            return f"≥ {value}"
        if op == "lt":
            return f"< {value}"
        if op == "lte":
            return f"≤ {value}"
        if op == "between":
            return f"{value[0]} — {value[1]}"
        return str(value)

    c1, c2 = st.columns(2)
    with c1:
        section_title(f"Matched ({len(matched)})", "✅")
        for cond in matched:
            _card(
                f'<span style="color:{_DIM};font-size:0.7rem;">{cond["field"]}</span>'
                f'&nbsp;<span style="color:{color};font-size:0.75rem;font-weight:700;">'
                f'{_op_label(cond["op"], cond["value"])}</span>',
                padding="8px 12px",
                margin_bottom="4px",
                accent_color=color,
            )
        if not matched:
            _card(
                f'<span style="color:{_DIM};font-size:0.72rem;">No conditions matched</span>',
                padding="8px 12px",
            )

    with c2:
        section_title(f"Unmatched ({len(unmatched)})", "❌")
        for cond in unmatched:
            actual = cond.get("actual")
            actual_str = (
                f'actual: <b>{actual:.2f}</b>' if isinstance(actual, (int, float)) else "no data"
            )
            _card(
                f'<span style="color:{_DIM};font-size:0.7rem;">{cond["field"]}</span>'
                f'&nbsp;<span style="color:{_P["text_dim"]};font-size:0.72rem;">'
                f'{_op_label(cond["op"], cond["value"])}</span>'
                f'&nbsp;&nbsp;<span style="color:{_P["amber"]};font-size:0.68rem;">'
                f'({actual_str})</span>',
                padding="8px 12px",
                margin_bottom="4px",
            )
        if not unmatched:
            _card(
                f'<span style="color:{_DIM};font-size:0.72rem;">All conditions matched</span>',
                padding="8px 12px",
            )


# ── Indicator summary cards ───────────────────────────────────────────────────

def _render_indicator_summary(indicators: dict) -> None:
    section_title("Key Indicators", "📐")

    def _fmt(val, suffix="", decimals=1):
        if val is None:
            return "N/A"
        return f"{val:.{decimals}f}{suffix}"

    items = [
        ("RSI",         _fmt(indicators.get("rsi"), "", 1),        ""),
        ("Price/MA50",  _fmt(indicators.get("price_vs_ma50"), "%"), ""),
        ("Price/MA200", _fmt(indicators.get("price_vs_ma200"), "%"), ""),
        ("Mom 30d",     _fmt(indicators.get("momentum_30"), "%"),   ""),
        ("Mom 60d",     _fmt(indicators.get("momentum_60"), "%"),   ""),
        ("Mom 90d",     _fmt(indicators.get("momentum_90"), "%"),   ""),
        ("Drawdown 90", _fmt(indicators.get("drawdown_90"), "%"),   ""),
        ("Vol/Avg20",   _fmt(indicators.get("volume_vs_avg20"), "x"), ""),
    ]

    cols = st.columns(4)
    for i, (label, value, _) in enumerate(items):
        try:
            num = float(value.replace("%", "").replace("x", "").strip())
            if "%" in value:
                color = _P["green"] if num > 0 else (_P["red"] if num < 0 else _DIM)
            else:
                color = _GOLD
        except (ValueError, AttributeError):
            color = _DIM

        with cols[i % 4]:
            _card(
                f'<div style="color:{_DIM};font-size:0.6rem;text-transform:uppercase;'
                f'letter-spacing:0.1em;margin-bottom:4px;">{label}</div>'
                f'<div style="color:{color};font-size:1.05rem;font-weight:700;">{value}</div>',
                padding="10px 14px",
                margin_bottom="6px",
            )


# ── Main entry point ──────────────────────────────────────────────────────────

def render_market_cycle_page() -> None:
    inject_css()
    section_title("Market Cycle Engine", "🔄")

    st.markdown(
        f'<p style="color:{_DIM};font-size:0.78rem;margin:-8px 0 20px;">'
        f'14-phase psychology cycle detector — identifies where the market stands '
        f'in the classic Wall Street emotional cycle using technical indicators.</p>',
        unsafe_allow_html=True,
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 2, 1])
    with ctrl_c1:
        ticker = st.text_input(
            "Ticker",
            value="SPY",
            placeholder="e.g. SPY, AAPL, OMX.ST",
            key="mc_ticker",
        ).strip().upper()
    with ctrl_c2:
        period = st.selectbox(
            "Timeframe",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            index=3,
            key="mc_period",
            format_func=lambda x: {"1mo": "1 Month", "3mo": "3 Months",
                                    "6mo": "6 Months", "1y": "1 Year", "2y": "2 Years"}[x],
        )
    with ctrl_c3:
        run = st.button("Analyse", use_container_width=True, key="mc_run")

    if not ticker:
        st.info("Enter a ticker symbol to begin.")
        return

    # ── Data loading ──────────────────────────────────────────────────────────
    with st.spinner(f"Analysing {ticker} …"):
        data = cached_market_cycle_analysis(ticker, period)

    if not data:
        st.error(
            f"Could not fetch data for **{ticker}** ({period}). "
            "Check the ticker symbol and try again."
        )
        return

    indicators = data["indicators"]
    result = data["result"]
    phase = result["phase"]
    confidence = result["confidence"]
    phase_scores = result["phase_scores"]
    matched_rules = result["matched_rules"]
    phase_cfg = MARKET_CYCLE_RULES[phase]

    # ── Header: gauge + confidence + description ───────────────────────────────
    gauge_col, meta_col = st.columns([3, 2])

    with gauge_col:
        st.plotly_chart(
            _build_gauge(phase, confidence),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with meta_col:
        st.markdown("<br>", unsafe_allow_html=True)
        phase_label = phase.replace("_", " ")
        _card(
            content=(
                f'<div style="font-size:2.2rem;text-align:center;margin-bottom:8px;">'
                f'{phase_cfg["emoji"]}</div>'
                f'<div style="color:{phase_cfg["color"]};font-size:1.1rem;font-weight:700;'
                f'text-align:center;letter-spacing:0.08em;text-transform:uppercase;'
                f'margin-bottom:10px;">{phase_label}</div>'
                f'<div style="color:{_DIM};font-size:0.72rem;text-align:center;'
                f'line-height:1.5;margin-bottom:14px;">{phase_cfg["description"]}</div>'
            ),
            border_color=phase_cfg["color"] + "55",
            accent_color=phase_cfg["color"],
            padding="18px 20px",
            margin_bottom="10px",
        )

        n_matched = len(matched_rules.get("matched", []))
        n_total = n_matched + len(matched_rules.get("unmatched", []))
        st.metric(
            label="Confidence",
            value=f"{confidence:.0f}%",
            delta=f"{n_matched}/{n_total} conditions met",
        )
        st.progress(int(confidence))

        cycle_pos = PHASE_ORDER.index(phase) + 1
        st.markdown(
            f'<p style="color:{_DIM};font-size:0.68rem;margin-top:8px;">'
            f'Cycle position: <b style="color:{_GOLD};">{cycle_pos} / {len(PHASE_ORDER)}</b>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;Ticker: <b style="color:{_GOLD};">{ticker}</b>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;Period: <b style="color:{_GOLD};">{period}</b></p>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Indicator summary ──────────────────────────────────────────────────────
    _render_indicator_summary(indicators)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Phase scores ───────────────────────────────────────────────────────────
    section_title("Phase Scores (all 14 phases)", "📊")
    st.plotly_chart(
        _build_score_chart(phase_scores),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Matched / Unmatched rules ──────────────────────────────────────────────
    section_title(f"Rule Breakdown — {phase_label}", "🔍")
    _render_rules_breakdown(matched_rules, phase_cfg)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Historical timeline ────────────────────────────────────────────────────
    section_title("Phase History (rolling window)", "📈")

    with st.spinner("Computing phase history …"):
        history = cached_market_cycle_history(ticker, period)

    if history:
        timeline_fig = _build_timeline(history)
        if timeline_fig:
            st.plotly_chart(
                timeline_fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
    else:
        _card(
            f'<div style="color:{_DIM};font-size:0.76rem;text-align:center;padding:12px 0;">'
            f'Not enough historical data to build timeline for {ticker} ({period}).</div>',
            padding="14px 16px",
        )

    # ── Phase reference legend ─────────────────────────────────────────────────
    with st.expander("Phase Reference — all 14 phases", expanded=False):
        section_title("Psychology Cycle Reference", "📖")
        for i, phase_name in enumerate(PHASE_ORDER):
            cfg = MARKET_CYCLE_RULES[phase_name]
            _card(
                content=(
                    f'<div style="display:flex;align-items:center;gap:12px;">'
                    f'  <div style="font-size:1.4rem;">{cfg["emoji"]}</div>'
                    f'  <div>'
                    f'    <div style="color:{cfg["color"]};font-weight:700;font-size:0.8rem;'
                    f'    text-transform:uppercase;letter-spacing:0.06em;">'
                    f'    {i+1}. {phase_name.replace("_", " ")}</div>'
                    f'    <div style="color:{_DIM};font-size:0.7rem;margin-top:3px;">'
                    f'    {cfg["description"]}</div>'
                    f'  </div>'
                    f'</div>'
                ),
                border_color=cfg["color"] + "44",
                accent_color=cfg["color"],
                padding="10px 14px",
                margin_bottom="6px",
            )
