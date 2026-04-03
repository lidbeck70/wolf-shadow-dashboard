"""
cagr_streamlit.py
CAGR Strategy module — Streamlit page for Wolf Panel.

Cyberpunk theme: #050510 background, #00ffff cyan, #ff00ff magenta.
Entry point: render_cagr_page()

Supports two data modes:
  • Börsdata Pro+ (10-point fundamental, 17 max total)
  • yfinance fallback (6-point fundamental, 13 max total)
"""

from __future__ import annotations

import logging
import time
import io
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .cagr_loader import (
    load_nordic_tickers,
    load_etf_tickers,
    fetch_price_data,
    fetch_fundamentals,
    fetch_fundamentals_batch_fast,
    fetch_insider_transactions,
    get_data_source,
)
from .cagr_fundamentals import score_fundamentals
from .cagr_cycle import (
    DEFAULT_CYCLE,
    SECTOR_CONFIG,
    SCORE_LABELS,
    score_label,
    score_cycle,
    score_cycle_for_sector,
)
from .cagr_technical import score_technical, get_indicator_series
from .cagr_scoring import (
    calculate_total_score,
    build_summary_stats,
    SIGNAL_COLORS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
BG       = "#050510"
BG2      = "#0a0a1e"
CYAN     = "#00ffff"
MAGENTA  = "#ff00ff"
GREEN    = "#00ff88"
YELLOW   = "#ffdd00"
RED      = "#ff3355"
TEXT     = "#e0e0ff"
DIM      = "#4a4a6a"

PLOTLY_TEMPLATE = "plotly_dark"

# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        /* ── Wolf Panel CAGR – Cyberpunk CSS ── */
        html, body, .stApp {{
            background-color: {BG};
            color: {TEXT};
            font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
        }}

        /* Headers */
        h1, h2, h3, h4 {{
            color: {CYAN};
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {BG2};
            border-right: 1px solid rgba(0,255,255,0.2);
        }}

        /* KPI cards */
        .kpi-card {{
            background: linear-gradient(135deg, {BG2}, #0d0d30);
            border: 1px solid rgba(0,255,255,0.27);
            border-radius: 8px;
            padding: 16px 20px;
            text-align: center;
            box-shadow: 0 0 16px rgba(0,255,255,0.13);
        }}
        .kpi-card .kpi-value {{
            font-size: 2.2rem;
            font-weight: 700;
            color: {CYAN};
            line-height: 1.1;
        }}
        .kpi-card .kpi-buy   {{ color: {GREEN};   }}
        .kpi-card .kpi-hold  {{ color: {YELLOW};  }}
        .kpi-card .kpi-sell  {{ color: {RED};     }}
        .kpi-card .kpi-label {{
            font-size: 0.7rem;
            color: {DIM};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 4px;
        }}

        /* Signal badges in dataframe */
        .signal-strongbuy {{ color: {CYAN};    font-weight: 700; }}
        .signal-buy       {{ color: {GREEN};   font-weight: 700; }}
        .signal-hold      {{ color: {YELLOW};  font-weight: 700; }}
        .signal-sell      {{ color: {RED};     font-weight: 700; }}
        .signal-strongsell {{ color: {MAGENTA}; font-weight: 700; }}

        /* Data source badge */
        .data-badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 4px;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .data-badge-borsdata {{
            background: rgba(0,255,136,0.15);
            color: {GREEN};
            border: 1px solid rgba(0,255,136,0.3);
        }}
        .data-badge-yfinance {{
            background: rgba(255,221,0,0.15);
            color: {YELLOW};
            border: 1px solid rgba(255,221,0,0.3);
        }}

        /* Scan button */
        .stButton > button {{
            background: linear-gradient(90deg, rgba(0,255,255,0.2), rgba(255,0,255,0.2));
            border: 1px solid {CYAN};
            color: {CYAN};
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            transition: all 0.2s;
        }}
        .stButton > button:hover {{
            background: linear-gradient(90deg, rgba(0,255,255,0.4), rgba(255,0,255,0.4));
            box-shadow: 0 0 20px rgba(0,255,255,0.33);
        }}

        /* Dividers */
        hr {{
            border-color: rgba(0,255,255,0.13);
        }}

        /* Progress bar / spinner */
        .stProgress > div > div {{
            background: linear-gradient(90deg, {CYAN}, {MAGENTA});
        }}

        /* Expanders */
        [data-testid="stExpander"] {{
            border: 1px solid rgba(0,255,255,0.13) !important;
            background-color: {BG2} !important;
        }}

        /* Selectbox / slider labels */
        .stSelectbox label, .stSlider label, .stMultiSelect label {{
            color: {CYAN} !important;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: {BG}; }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(0,255,255,0.27);
            border-radius: 3px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached data fetchers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fetch_price_data(tickers: tuple, period: str = "2y") -> dict:
    """Cache wrapper — tickers must be a hashable tuple."""
    return fetch_price_data(list(tickers), period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fetch_fundamentals(ticker: str) -> dict:
    return fetch_fundamentals(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fetch_insider(ticker: str):
    return fetch_insider_transactions(ticker)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def _run_scan(
    tickers_meta: Dict[str, dict],
    cycle_overrides: Dict[str, dict],
    period: str = "2y",
) -> List[dict]:
    """Full scan pipeline. Returns list of scored record dicts."""
    tickers = list(tickers_meta.keys())
    results: List[dict] = []

    progress = st.progress(0, text="Fetching price data...")
    price_data = _cached_fetch_price_data(tuple(tickers), period=period)
    progress.progress(15, text="Prices fetched. Loading fundamentals...")

    # Batch-fetch all fundamentals at once (23 API calls instead of 1500+)
    try:
        all_fundamentals = fetch_fundamentals_batch_fast(tuple(tickers))
    except Exception as exc:
        logger.warning("Batch fundamentals failed, falling back: %s", exc)
        all_fundamentals = {}
    progress.progress(50, text="Fundamentals loaded. Scoring...")

    for i, (ticker, meta) in enumerate(tickers_meta.items()):
        progress.progress(
            50 + int(50 * (i + 1) / max(len(tickers_meta), 1)),
            text=f"Scoring {ticker}...",
        )

        df = price_data.get(ticker, pd.DataFrame())

        # Fundamentals (from batch or per-ticker fallback)
        try:
            fund_info = all_fundamentals.get(ticker)
            if not fund_info:
                fund_info = _cached_fetch_fundamentals(ticker)
            insider_df = _cached_fetch_insider(ticker)
            fund_result = score_fundamentals(fund_info, insider_df)
        except Exception as exc:
            logger.warning("fund score failed for %s: %s", ticker, exc)
            fund_result = {"fund_score": 0, "fund_max": 6, "details": {}}

        # Sector for cycle
        sector = meta.get("sector", "Unknown")
        override = cycle_overrides.get(sector, {})

        try:
            cycle_result = score_cycle_for_sector(sector, override)
        except Exception as exc:
            logger.warning("cycle score failed for %s: %s", ticker, exc)
            cycle_result = {"cycle_score": 0, "details": {}}

        # Technical
        try:
            tech_result = score_technical(df)
        except Exception as exc:
            logger.warning("tech score failed for %s: %s", ticker, exc)
            tech_result = {"tech_score": 0, "details": {}, "sparkline_data": []}

        # Combined score
        total = calculate_total_score(fund_result, cycle_result, tech_result)

        results.append({
            "ticker":       ticker,
            "name":         meta.get("name", ticker),
            "country":      meta.get("country", meta.get("category", "ETF")),
            "sector":       sector,
            "fund_score":   total["fund_score"],
            "fund_max":     total["fund_max"],
            "cycle_score":  total["cycle_score"],
            "tech_score":   total["tech_score"],
            "total_score":  total["total_score"],
            "max_score":    total["max_score"],
            "score_pct":    total["score_pct"],
            "signal":       total["signal"],
            "signal_color": total["signal_color"],
            "sparkline":    tech_result.get("sparkline_data", []),
            "_fund_details":  fund_result.get("details", {}),
            "_cycle_details": cycle_result.get("details", {}),
            "_tech_details":  tech_result.get("details", {}),
            "tech_max":       tech_result.get("tech_max", 7),
            "_data_source":   fund_result.get("_data_source", "yfinance"),
            "_df":            df,
        })

    progress.empty()
    return results


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _kpi_card(label: str, value: str, extra_class: str = "") -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-value {extra_class}">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>'
    )


def _signal_badge(signal: str) -> str:
    cls = {
        "STRONG BUY": "signal-strongbuy",
        "BUY": "signal-buy",
        "HOLD": "signal-hold",
        "SELL": "signal-sell",
        "STRONG SELL": "signal-strongsell",
    }.get(signal, "")
    return f'<span class="{cls}">{signal}</span>'


def _data_source_badge(source: str) -> str:
    """Render a coloured badge showing the data source."""
    if "borsdata" in source.lower() or "börsdata" in source.lower():
        return '<span class="data-badge data-badge-borsdata">BÖRSDATA PRO+</span>'
    return '<span class="data-badge data-badge-yfinance">YFINANCE</span>'


def _build_sparkline_fig(prices: list) -> go.Figure:
    """Tiny inline sparkline figure."""
    if not prices:
        fig = go.Figure()
    else:
        color = CYAN
        fig = go.Figure(
            go.Scatter(
                y=prices,
                mode="lines",
                line=dict(color=color, width=1.5),
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=50,
        width=120,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _build_price_chart(df: pd.DataFrame, ticker: str) -> Optional[go.Figure]:
    """Full 1-year Plotly candlestick + EMA chart."""
    if df is None or df.empty:
        return None

    close_col = next(
        (c for c in ("Close", "Adj Close", "close") if c in df.columns), None
    )
    if close_col is None:
        return None

    df_1y = df.tail(252).copy()
    inds = get_indicator_series(df_1y)

    fig = go.Figure()

    ohlc_cols = ["Open", "High", "Low", "Close"]
    if all(c in df_1y.columns for c in ohlc_cols):
        fig.add_trace(
            go.Candlestick(
                x=df_1y.index,
                open=df_1y["Open"],
                high=df_1y["High"],
                low=df_1y["Low"],
                close=df_1y["Close"],
                name="OHLC",
                increasing_line_color=GREEN,
                decreasing_line_color=RED,
                increasing_fillcolor="rgba(0,255,136,0.33)",
                decreasing_fillcolor="rgba(255,51,85,0.33)",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df_1y.index,
                y=df_1y[close_col],
                mode="lines",
                name="Price",
                line=dict(color=CYAN, width=1.5),
            )
        )

    if inds.get("ema50"):
        fig.add_trace(
            go.Scatter(
                x=df_1y.index,
                y=inds["ema50"],
                mode="lines",
                name="EMA50",
                line=dict(color=YELLOW, width=1.2, dash="dot"),
            )
        )

    if inds.get("ema200"):
        fig.add_trace(
            go.Scatter(
                x=df_1y.index,
                y=inds["ema200"],
                mode="lines",
                name="EMA200",
                line=dict(color=MAGENTA, width=1.5),
            )
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG2,
        plot_bgcolor=BG,
        title=dict(text=f"{ticker} — 1 Year", font=dict(color=CYAN, size=14)),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,255,255,0.08)",
            color=DIM,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,255,255,0.08)", color=DIM),
        legend=dict(
            font=dict(color=TEXT, size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=40, r=20, t=40, b=30),
        height=380,
    )
    return fig


def _build_rsi_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """RSI sub-chart."""
    if df is None or df.empty:
        return None

    df_1y = df.tail(252).copy()
    inds = get_indicator_series(df_1y)

    if not inds.get("rsi"):
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_1y.index,
            y=inds["rsi"],
            mode="lines",
            name="RSI(14)",
            line=dict(color=CYAN, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0,255,255,0.08)",
        )
    )
    for level, color, label in [(70, RED, "OB"), (50, DIM, "Mid"), (30, GREEN, "OS")]:
        fig.add_hline(
            y=level,
            line=dict(color=color, dash="dash", width=1),
            annotation_text=str(level),
            annotation_font_color=color,
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG2,
        plot_bgcolor=BG,
        title=dict(text="RSI (14)", font=dict(color=CYAN, size=12)),
        xaxis=dict(showgrid=False, color=DIM),
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(0,255,255,0.06)", color=DIM),
        height=160,
        margin=dict(l=40, r=20, t=30, b=20),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Cycle sidebar controls
# ---------------------------------------------------------------------------

def _cycle_sidebar_controls(sectors: list) -> Dict[str, dict]:
    """Render per-sector cycle controls as clean sliders (0–3) with color coding."""
    overrides: Dict[str, dict] = {}
    st.sidebar.markdown(
        f"<div style='color:{CYAN};font-size:0.7rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;margin-top:1rem;'>Cycle Assessment</div>",
        unsafe_allow_html=True,
    )
    with st.sidebar.expander("Sector Cycle Scores", expanded=False):
        st.markdown(
            f"<div style='color:{DIM};font-size:0.65rem;margin-bottom:8px;'>"
            "0 = Bearish &nbsp; 1 = Neutral &nbsp; 2 = Bullish &nbsp; 3 = Strong"
            "</div>",
            unsafe_allow_html=True,
        )
        for sector in sorted(sectors):
            cfg = SECTOR_CONFIG.get(sector, SECTOR_CONFIG.get("Unknown", {}))
            default_val = cfg.get("default_score", 1)
            thesis = cfg.get("thesis", "")

            col_name, col_slider = st.columns([1, 1.5])
            with col_name:
                lbl, clr = score_label(default_val)
                st.markdown(
                    f"<div style='font-size:0.72rem;font-family:monospace;color:{CYAN};padding-top:8px;'>"
                    f"{sector}</div>",
                    unsafe_allow_html=True,
                )
            with col_slider:
                val = st.slider(
                    sector,
                    min_value=0,
                    max_value=3,
                    value=default_val,
                    key=f"cycle_{sector}",
                    label_visibility="collapsed",
                )
            # Show thesis as tooltip-style caption
            if thesis:
                lbl_now, clr_now = score_label(val)
                st.markdown(
                    f"<div style='font-size:0.6rem;color:{clr_now};margin:-10px 0 4px 0;'>"
                    f"{lbl_now} — {thesis}</div>",
                    unsafe_allow_html=True,
                )

            overrides[sector] = {"cycle_score": val}

    return overrides


# ---------------------------------------------------------------------------
# Score bar widget
# ---------------------------------------------------------------------------

def _score_bar(score: int, max_score: int, color: str = CYAN) -> str:
    pct = int(100 * score / max(max_score, 1))
    return (
        f'<div style="background:{BG};border:1px solid rgba(0,255,255,0.2);border-radius:4px;'
        f'height:8px;width:100%;overflow:hidden;">'
        f'<div style="background:{color};height:100%;width:{pct}%;'
        f'box-shadow:0 0 6px {color};"></div></div>'
    )


# ---------------------------------------------------------------------------
# Detail expander
# ---------------------------------------------------------------------------

def _render_detail_expander(rec: dict) -> None:
    ticker = rec["ticker"]
    max_score = rec.get("max_score", 13)
    fund_max = rec.get("fund_max", 6)
    ds = rec.get("_data_source", "yfinance")
    ds_badge = _data_source_badge(ds)

    with st.expander(
        f"  {ticker} — {rec['name']}  |  Total: {rec['total_score']}/{max_score}  [{rec['signal']}]",
        expanded=False,
    ):
        left, right = st.columns([1, 1])

        with left:
            st.markdown(
                f"<div style='color:{CYAN};font-size:0.75rem;text-transform:uppercase;"
                f"letter-spacing:0.1em;'>Fundamental Score: {rec['fund_score']}/{fund_max} "
                f"{ds_badge}</div>",
                unsafe_allow_html=True,
            )
            fund_rows = []
            for crit, val in rec["_fund_details"].items():
                fund_rows.append({
                    "Criterion": crit,
                    "Value": str(val.get("value", "N/A")),
                    "Threshold": val.get("threshold", "—"),
                    "Pass": "✓" if val.get("pass") else "✗",
                })
            if fund_rows:
                st.dataframe(
                    pd.DataFrame(fund_rows),
                    use_container_width=True,
                    hide_index=True,
                    height=min(380, 35 + 35 * len(fund_rows)),
                )

            # Cycle details
            st.markdown(
                f"<div style='color:{MAGENTA};font-size:0.75rem;text-transform:uppercase;"
                f"letter-spacing:0.1em;margin-top:12px;'>Cycle Score: {rec['cycle_score']}/3</div>",
                unsafe_allow_html=True,
            )
            cycle_rows = []
            for crit, val in rec["_cycle_details"].items():
                cycle_rows.append({
                    "Criterion": crit,
                    "Value": "Yes" if val.get("value") else "No",
                    "Pass": "✓" if val.get("pass") else "✗",
                })
            if cycle_rows:
                st.dataframe(
                    pd.DataFrame(cycle_rows),
                    use_container_width=True,
                    hide_index=True,
                    height=min(145, 35 + 35 * len(cycle_rows)),
                )

        with right:
            tech_max = rec.get('tech_max', 7)
            st.markdown(
                f"<div style='color:{YELLOW};font-size:0.75rem;text-transform:uppercase;"
                f"letter-spacing:0.1em;'>Technical Score: {rec['tech_score']}/{tech_max}</div>",
                unsafe_allow_html=True,
            )
            tech_rows = []
            for crit, val in rec["_tech_details"].items():
                tech_rows.append({
                    "Indicator": crit,
                    "Value": str(val.get("value", "N/A")),
                    "Pass": "✓" if val.get("pass") else "✗",
                })
            if tech_rows:
                st.dataframe(
                    pd.DataFrame(tech_rows),
                    use_container_width=True,
                    hide_index=True,
                    height=min(180, 35 + 35 * len(tech_rows)),
                )

        # Price chart
        df = rec.get("_df")
        price_fig = _build_price_chart(df, ticker)
        if price_fig:
            st.plotly_chart(price_fig, use_container_width=True)
        else:
            st.info("No price data available for chart.")

        # RSI sub-chart
        rsi_fig = _build_rsi_chart(df)
        if rsi_fig:
            st.plotly_chart(rsi_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main table
# ---------------------------------------------------------------------------

def _build_display_df(records: list) -> pd.DataFrame:
    """Build a clean display DataFrame from scored records."""
    rows = []
    for r in records:
        fund_max = r.get("fund_max", 6)
        max_score = r.get("max_score", 13)
        rows.append({
            "Ticker":     r["ticker"],
            "Name":       r["name"],
            "Country":    r["country"],
            "Sector":     r["sector"],
            f"Fund (0-{fund_max})": r["fund_score"],
            "Cyc (0-3)":  r["cycle_score"],
            "Tech (0-7)": r["tech_score"],
            f"Total (0-{max_score})": r["total_score"],
            "Score %":    f"{r.get('score_pct', 0) * 100:.0f}%",
            "Signal":     r["signal"],
        })
    return pd.DataFrame(rows)


def _style_signal_col(val: str) -> str:
    colors = {
        "STRONG BUY": CYAN, "BUY": GREEN,
        "HOLD": YELLOW, "SELL": RED, "STRONG SELL": MAGENTA,
    }
    c = colors.get(val, TEXT)
    return f"color: {c}; font-weight: 700;"


def _style_score_col(val, max_val: int = 13) -> str:
    try:
        ratio = float(val) / max(max_val, 1)
    except (TypeError, ValueError):
        return f"color: {DIM};"
    if ratio >= 0.65:
        return f"color: {GREEN};"
    if ratio >= 0.40:
        return f"color: {YELLOW};"
    return f"color: {RED};"


def _style_pct_col(val: str) -> str:
    try:
        pct = float(val.replace("%", "")) / 100
    except (TypeError, ValueError):
        return f"color: {DIM};"
    if pct >= 0.65:
        return f"color: {GREEN}; font-weight: 700;"
    if pct >= 0.40:
        return f"color: {YELLOW};"
    return f"color: {RED};"


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _to_csv_bytes(records: list) -> bytes:
    df = _build_display_df(records)
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

def _render_header() -> None:
    data_source = get_data_source()
    badge = _data_source_badge(data_source)

    st.markdown(
        f"""
        <div style='padding:20px 0 10px 0;'>
            <h1 style='margin:0;font-size:2rem;letter-spacing:0.12em;
                       background:linear-gradient(90deg,{CYAN},{MAGENTA});
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
                LONG SCREENER
            </h1>
            <div style='color:{DIM};font-size:0.7rem;letter-spacing:0.15em;margin-top:4px;'>
                NORDIC STOCKS &amp; UCITS ETFs — 20-POINT FUNDAMENTAL · 7-POINT TECHNICAL · 5-LEVEL SIGNALS
                &nbsp;&nbsp;{badge}
            </div>
        </div>
        <hr style='border-color:rgba(0,255,255,0.13);margin-bottom:20px;'/>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

def _render_kpi_row(stats: dict) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(
            _kpi_card("Scanned", str(stats["total_scanned"])),
            unsafe_allow_html=True,
        )
    with c2:
        strong_buy = stats.get("strong_buy_count", 0)
        buy = stats.get("buy_count", 0)
        st.markdown(
            _kpi_card("BUY", f"{strong_buy}+{buy}", "kpi-buy"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _kpi_card("HOLD", str(stats["hold_count"]), "kpi-hold"),
            unsafe_allow_html=True,
        )
    with c4:
        sell = stats.get("sell_count", 0)
        strong_sell = stats.get("strong_sell_count", 0)
        st.markdown(
            _kpi_card("SELL", f"{sell}+{strong_sell}", "kpi-sell"),
            unsafe_allow_html=True,
        )
    with c5:
        ds = stats.get("data_source", "yfinance")
        badge_class = "kpi-buy" if "börsdata" in ds.lower() else ""
        st.markdown(
            _kpi_card("Data Source", ds, badge_class),
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            _kpi_card("Max Score", "30" if "börsdata" in stats.get("data_source", "").lower() else "16"),
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_cagr_page() -> None:
    """Full CAGR Strategy Streamlit page."""
    _inject_css()
    _render_header()

    # ── Sidebar ────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        f"<div style='color:{CYAN};font-size:0.8rem;letter-spacing:0.1em;"
        f"text-transform:uppercase;padding:10px 0 6px 0;'>"
        f"CAGR Scanner Controls</div>",
        unsafe_allow_html=True,
    )

    market_choice = st.sidebar.selectbox(
        "Market",
        options=["All", "Nordic Stocks", "UCITS ETFs"],
        index=0,
    )

    country_choice = st.sidebar.selectbox(
        "Country",
        options=["All", "Sweden", "Norway", "Denmark", "Finland"],
        index=0,
    )

    # Score filter uses percentage (works for both scales)
    min_score_pct = st.sidebar.slider(
        "Min Score %",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
    )

    sort_by = st.sidebar.selectbox(
        "Sort By",
        options=["Score % (↓)", "Name (A-Z)", "Country", "Signal"],
        index=0,
    )

    data_period = st.sidebar.selectbox(
        "Price History Period",
        options=["6mo", "1y", "2y"],
        index=2,
    )

    all_sectors = sorted(set(DEFAULT_CYCLE.keys()))
    cycle_overrides = _cycle_sidebar_controls(all_sectors)

    st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.07);'/>", unsafe_allow_html=True)
    scan_clicked = st.sidebar.button("⟳  SCAN", use_container_width=True)

    # ── Build ticker list ──────────────────────────────────────────────────
    nordic_tickers = load_nordic_tickers()
    etf_tickers    = load_etf_tickers()

    if market_choice == "Nordic Stocks":
        tickers_meta = nordic_tickers
    elif market_choice == "UCITS ETFs":
        tickers_meta = {
            k: {**v, "sector": "ETF"} for k, v in etf_tickers.items()
        }
    else:
        tickers_meta = {
            **nordic_tickers,
            **{k: {**v, "sector": "ETF"} for k, v in etf_tickers.items()},
        }

    if country_choice != "All" and market_choice != "UCITS ETFs":
        tickers_meta = {
            k: v for k, v in tickers_meta.items()
            if v.get("country") == country_choice
        }

    # ── Session state ──────────────────────────────────────────────────────
    if "cagr_results" not in st.session_state:
        st.session_state.cagr_results = []
    if "cagr_last_scan" not in st.session_state:
        st.session_state.cagr_last_scan = 0.0

    needs_scan = (
        scan_clicked
        or not st.session_state.cagr_results
        or (time.time() - st.session_state.cagr_last_scan) > 3600
    )

    if needs_scan and tickers_meta:
        with st.spinner("Running CAGR scan..."):
            results = _run_scan(tickers_meta, cycle_overrides, period=data_period)
            st.session_state.cagr_results = results
            st.session_state.cagr_last_scan = time.time()

    all_results: List[dict] = st.session_state.cagr_results or []

    # Apply filters (percentage-based)
    filtered = [
        r for r in all_results
        if r.get("score_pct", 0) * 100 >= min_score_pct
    ]

    if country_choice != "All":
        filtered = [
            r for r in filtered
            if r.get("country") == country_choice or r.get("country") == "ETF"
        ]

    # Sorting
    if sort_by == "Score % (↓)":
        filtered = sorted(filtered, key=lambda x: x.get("score_pct", 0), reverse=True)
    elif sort_by == "Name (A-Z)":
        filtered = sorted(filtered, key=lambda x: x["name"])
    elif sort_by == "Country":
        filtered = sorted(filtered, key=lambda x: (x["country"], -x.get("score_pct", 0)))
    elif sort_by == "Signal":
        order = {"BUY": 0, "HOLD": 1, "SELL": 2}
        filtered = sorted(filtered, key=lambda x: (order.get(x["signal"], 9), -x.get("score_pct", 0)))

    # ── KPI cards ──────────────────────────────────────────────────────────
    if filtered:
        stats = build_summary_stats(filtered)
        _render_kpi_row(stats)

    # ── Export ─────────────────────────────────────────────────────────────
    if filtered:
        export_col, spacer = st.columns([2, 8])
        with export_col:
            st.download_button(
                label="⬇  Export CSV",
                data=_to_csv_bytes(filtered),
                file_name="cagr_scan.csv",
                mime="text/csv",
            )

    # ── Main table ─────────────────────────────────────────────────────────
    if not filtered:
        st.markdown(
            f"<div style='color:{DIM};text-align:center;padding:40px;'>"
            "No results match the current filters. Press SCAN to refresh data."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"<div style='color:{DIM};font-size:0.7rem;letter-spacing:0.1em;"
        f"text-transform:uppercase;margin-bottom:8px;'>"
        f"Showing {len(filtered)} instruments</div>",
        unsafe_allow_html=True,
    )

    display_df = _build_display_df(filtered)

    # Find column names dynamically
    fund_col = next((c for c in display_df.columns if c.startswith("Fund")), None)
    total_col = next((c for c in display_df.columns if c.startswith("Total")), None)

    # Extract max values from column names for styling
    fund_max = 6
    total_max = 13
    if filtered:
        fund_max = filtered[0].get("fund_max", 6)
        total_max = filtered[0].get("max_score", 13)

    style_map = {}
    if fund_col:
        style_map[fund_col] = lambda v: _style_score_col(v, fund_max)
    if total_col:
        style_map[total_col] = lambda v: _style_score_col(v, total_max)

    styled = display_df.style
    # pandas >= 2.1 renamed applymap → map
    _map = styled.map if hasattr(styled, "map") else styled.applymap
    styled = _map(_style_signal_col, subset=["Signal"])
    styled = _map(_style_pct_col, subset=["Score %"])
    styled = _map(lambda v: _style_score_col(v, 3), subset=["Cyc (0-3)"])
    styled = _map(lambda v: _style_score_col(v, 7), subset=["Tech (0-7)"])

    if fund_col:
        styled = _map(lambda v: _style_score_col(v, fund_max), subset=[fund_col])
    if total_col:
        styled = _map(lambda v: _style_score_col(v, total_max), subset=[total_col])

    styled = styled.set_properties(**{
        "background-color": BG2,
        "color": TEXT,
        "border-color": "rgba(0,255,255,0.13)",
    })

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(600, 38 + 35 * len(display_df)),
    )

    # ── Sparklines row ─────────────────────────────────────────────────────
    st.markdown(
        f"<div style='color:{CYAN};font-size:0.7rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;margin:20px 0 8px 0;'>6-Month Price Trend</div>",
        unsafe_allow_html=True,
    )

    COLS_PER_ROW = 8
    spark_records = [r for r in filtered if r.get("sparkline")]
    for row_start in range(0, len(spark_records), COLS_PER_ROW):
        row_recs = spark_records[row_start: row_start + COLS_PER_ROW]
        cols = st.columns(len(row_recs))
        for col, rec in zip(cols, row_recs):
            with col:
                spark_prices = rec["sparkline"][-30:] if len(rec["sparkline"]) >= 30 else rec["sparkline"]
                sig_color = SIGNAL_COLORS.get(rec["signal"], CYAN)
                fig = go.Figure(
                    go.Scatter(
                        y=spark_prices,
                        mode="lines",
                        line=dict(color=sig_color, width=1.5),
                        hoverinfo="skip",
                    )
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=16, b=0),
                    height=60,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    title=dict(
                        text=rec["ticker"],
                        font=dict(color=DIM, size=9),
                        x=0.5,
                        y=1.0,
                        xanchor="center",
                        yanchor="top",
                    ),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Detail expanders ───────────────────────────────────────────────────
    st.markdown(
        f"<div style='color:{CYAN};font-size:0.7rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;margin:24px 0 8px 0;border-top:1px solid rgba(0,255,255,0.13);"
        f"padding-top:16px;'>Detailed Analysis (click to expand)</div>",
        unsafe_allow_html=True,
    )

    for rec in filtered[:30]:
        _render_detail_expander(rec)

    if len(filtered) > 30:
        st.markdown(
            f"<div style='color:{DIM};font-size:0.7rem;text-align:center;padding:8px;'>"
            f"Showing detail for top 30 instruments. Use filters to narrow results."
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    st.set_page_config(
        page_title="CAGR Strategy Scanner — Wolf Panel",
        page_icon="🐺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_cagr_page()
