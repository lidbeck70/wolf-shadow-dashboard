"""
RS Momentum Backtest — Streamlit Page
Nordic Gold theme: EMA crossover strategy with RS pre-filtering across 11 US sector ETFs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .rs_backtest_engine import run_rs_sector_backtest, SECTOR_MAP

# ---------------------------------------------------------------------------
# Nordic Gold color palette
# ---------------------------------------------------------------------------
BG = "#0c0c12"
BG2 = "#14141e"
CYAN = "#c9a84c"
MAGENTA = "#8b7340"
GREEN = "#2d8a4e"
RED = "#c44545"
YELLOW = "#d4943a"
TEXT = "#e8e4dc"
DIM = "#8a8578"

PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor=BG2,
    plot_bgcolor=BG,
)

# Sector line colours (cycle for equity curves)
SECTOR_COLORS = [
    "rgba(201,168,76,1)",
    "rgba(139,115,64,1)",
    "rgba(45,138,78,1)",
    "rgba(196,69,69,1)",
    "rgba(212,148,58,1)",
    "rgba(201,168,76,0.7)",
    "rgba(212,148,58,0.7)",
    "rgba(139,115,64,0.7)",
    "rgba(45,138,78,0.7)",
    "rgba(196,69,69,0.7)",
    "rgba(45,138,78,0.5)",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

YEARS_MAP = {"1y": 1, "2y": 2, "3y": 3, "5y": 5}


def _cyberpunk_css() -> None:
    st.markdown(
        f"""
        <style>
        /* Root overrides */
        [data-testid="stAppViewContainer"] {{
            background-color: {BG};
        }}
        [data-testid="stSidebar"] {{
            background-color: {BG2};
        }}
        .stMetric {{
            background: {BG2};
            border: 1px solid {DIM};
            border-radius: 8px;
            padding: 12px;
        }}
        .stMetric label {{
            color: {DIM} !important;
            font-size: 0.72rem !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .stMetric [data-testid="metric-container"] div:nth-child(2) {{
            color: {CYAN} !important;
            font-size: 1.4rem !important;
            font-weight: 700;
        }}
        h1, h2, h3 {{
            color: {TEXT};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div style="border-left:3px solid {CYAN}; padding-left:12px; margin:24px 0 12px 0;">
            <span style="color:{CYAN}; font-size:1.15rem; font-weight:700; letter-spacing:0.04em;">
                {title}
            </span>
            {"<br><span style='color:" + DIM + "; font-size:0.8rem;'>" + subtitle + "</span>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _kpi_card(label: str, value: str, color: str = CYAN) -> None:
    st.markdown(
        f"""
        <div style="background:{BG2}; border:1px solid {DIM}; border-top:2px solid {color};
                    border-radius:8px; padding:16px; text-align:center;">
            <div style="color:{DIM}; font-size:0.7rem; text-transform:uppercase;
                        letter-spacing:0.1em; margin-bottom:6px;">{label}</div>
            <div style="color:{color}; font-size:1.5rem; font-weight:700;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _style_sector_df(df: pd.DataFrame):
    """Apply Nordic Gold colour coding to sector summary table."""

    def color_win_rate(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v > 55:
            return f"color: {GREEN}; font-weight:600"
        elif v < 45:
            return f"color: {RED}"
        return f"color: {YELLOW}"

    def color_pf(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v >= 1.5:
            return f"color: {GREEN}; font-weight:600"
        elif v < 1.0:
            return f"color: {RED}"
        return f"color: {YELLOW}"

    def color_maxdd(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v < -20:
            return f"color: {RED}; font-weight:600"
        elif v < -10:
            return f"color: {YELLOW}"
        return f"color: {GREEN}"

    def color_return(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v > 0:
            return f"color: {GREEN}"
        return f"color: {RED}"

    styler = df.style
    if "Win Rate %" in df.columns:
        styler = styler.map(color_win_rate, subset=["Win Rate %"])
    if "Profit Factor" in df.columns:
        styler = styler.map(color_pf, subset=["Profit Factor"])
    if "Max DD %" in df.columns:
        styler = styler.map(color_maxdd, subset=["Max DD %"])
    if "Total Return %" in df.columns:
        styler = styler.map(color_return, subset=["Total Return %"])
    if "Best Return" in df.columns:
        styler = styler.map(color_return, subset=["Best Return"])
    if "Worst Return" in df.columns:
        styler = styler.map(color_return, subset=["Worst Return"])

    styler = styler.set_table_styles([
        {"selector": "thead th", "props": f"background-color:{BG2}; color:{CYAN}; text-transform:uppercase; font-size:0.7rem; letter-spacing:0.08em;"},
        {"selector": "tbody tr", "props": f"background-color:{BG};"},
        {"selector": "tbody tr:hover", "props": f"background-color:{BG2};"},
        {"selector": "td", "props": f"color:{TEXT}; border-color:{DIM};"},
    ])
    return styler


# ---------------------------------------------------------------------------
# Section 2: Sector comparison chart
# ---------------------------------------------------------------------------

def _render_sector_comparison(display_df: pd.DataFrame) -> None:
    _section_header("SECTOR COMPARISON", "Win Rate vs Profit Factor by sector")

    sectors = display_df["Sector"].tolist()
    win_rates = display_df["Win Rate %"].tolist()
    profit_factors = display_df["Profit Factor"].tolist()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Win Rate (%)",
        x=sectors,
        y=win_rates,
        marker_color=f"rgba(201,168,76,0.75)",
        marker_line_color=CYAN,
        marker_line_width=1,
        text=[f"{v:.1f}%" for v in win_rates],
        textposition="outside",
        textfont=dict(color=CYAN, size=11),
    ))

    fig.add_trace(go.Bar(
        name="Profit Factor",
        x=sectors,
        y=profit_factors,
        marker_color=f"rgba(139,115,64,0.75)",
        marker_line_color=MAGENTA,
        marker_line_width=1,
        text=[f"{v:.2f}" for v in profit_factors],
        textposition="outside",
        textfont=dict(color=MAGENTA, size=11),
        yaxis="y2",
    ))

    # Reference lines
    fig.add_hline(
        y=50, line_dash="dash", line_color=f"rgba(201,168,76,0.35)",
        annotation_text="50% WR target", annotation_font_color=CYAN,
        annotation_position="top right",
    )

    fig.update_layout(
        **PLOTLY_BASE,
        barmode="group",
        yaxis=dict(
            title="Win Rate (%)",
            title_font_color=CYAN,
            tickfont_color=DIM,
            gridcolor=f"rgba(138,133,120,0.3)",
            range=[0, max(win_rates + [60]) * 1.25 if win_rates else 80],
        ),
        yaxis2=dict(
            title="Profit Factor",
            title_font_color=MAGENTA,
            tickfont_color=DIM,
            overlaying="y",
            side="right",
            gridcolor=f"rgba(139,115,64,0.1)",
            range=[0, max(profit_factors + [2]) * 1.35 if profit_factors else 3],
        ),
        xaxis=dict(tickfont_color=TEXT, gridcolor=f"rgba(138,133,120,0.15)"),
        legend=dict(font_color=TEXT, bgcolor=BG2, bordercolor=DIM),
        margin=dict(t=40, b=60, l=60, r=80),
        height=420,
    )

    # PF reference line at 1.0 on y2 — add as scatter annotation
    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(sectors) - 0.5,
        y0=1.0, y1=1.0,
        yref="y2",
        line=dict(color=f"rgba(139,115,64,0.35)", dash="dash"),
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 3: Top performers horizontal bar
# ---------------------------------------------------------------------------

def _render_top_performers(all_results: dict) -> None:
    _section_header("TOP INDIVIDUAL PERFORMERS", "Top 10 tickers by Total Return %")

    rows = []
    for sector, results in all_results.items():
        for r in results:
            rows.append({
                "Ticker": r["ticker"],
                "Sector": sector,
                "Total Return %": r["metrics"]["total_return"],
            })

    if not rows:
        st.info("No individual results available.")
        return

    perf_df = pd.DataFrame(rows).sort_values("Total Return %", ascending=False)
    top10 = perf_df.head(10).reset_index(drop=True)

    colors = [
        f"rgba(45,138,78,0.8)" if v >= 0 else f"rgba(196,69,69,0.8)"
        for v in top10["Total Return %"]
    ]

    fig = go.Figure(go.Bar(
        x=top10["Total Return %"],
        y=top10["Ticker"],
        orientation="h",
        marker_color=colors,
        marker_line_color=[
            f"rgba(45,138,78,1)" if v >= 0 else f"rgba(196,69,69,1)"
            for v in top10["Total Return %"]
        ],
        marker_line_width=1,
        text=[
            f"{row['Ticker']}  {row['Total Return %']:+.1f}%"
            for _, row in top10.iterrows()
        ],
        textposition="outside",
        textfont=dict(color=TEXT, size=11),
    ))

    max_abs = max(abs(top10["Total Return %"].max()), abs(top10["Total Return %"].min()), 10)

    fig.update_layout(
        **PLOTLY_BASE,
        xaxis=dict(
            title="Total Return (%)",
            title_font_color=DIM,
            tickfont_color=DIM,
            gridcolor=f"rgba(138,133,120,0.3)",
            range=[-max_abs * 0.1, max_abs * 1.35],
        ),
        yaxis=dict(tickfont_color=TEXT, autorange="reversed"),
        showlegend=False,
        margin=dict(t=30, b=50, l=80, r=150),
        height=380,
    )

    # Zero reference line
    fig.add_vline(x=0, line_color=f"rgba(232,228,220,0.25)", line_width=1)

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 4: Individual trades expander
# ---------------------------------------------------------------------------

def _render_trade_drilldown(all_results: dict) -> None:
    _section_header("INDIVIDUAL TRADES", "Drill-down per sector — top RS% tickers only")

    for sector, results in all_results.items():
        if not results:
            continue

        all_trades = []
        for r in results:
            for t in r["trades"]:
                all_trades.append({
                    "Ticker": r["ticker"],
                    "Entry Date": t["entry_date"],
                    "Exit Date": t["exit_date"],
                    "Entry Price": t["entry_price"],
                    "Exit Price": t["exit_price"],
                    "Return %": t["return_pct"],
                    "Duration (days)": t["duration_days"],
                })

        if not all_trades:
            continue

        trades_df = pd.DataFrame(all_trades).sort_values("Return %", ascending=False)
        n_trades = len(trades_df)
        n_tickers = len(set(trades_df["Ticker"]))

        etf = SECTOR_MAP.get(sector, {}).get("etf", "")
        label = f"{sector} ({etf}) — {n_tickers} tickers, {n_trades} trades"

        with st.expander(label, expanded=False):
            def color_trade_return(val):
                try:
                    v = float(val)
                except (ValueError, TypeError):
                    return ""
                return f"color: {GREEN}; font-weight:600" if v > 0 else f"color: {RED}"

            styled = trades_df.style.map(color_trade_return, subset=["Return %"])
            st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Section 5: Equity curves
# ---------------------------------------------------------------------------

def _render_equity_curves(all_results: dict) -> None:
    _section_header("EQUITY CURVES", "Aggregated per sector, normalised to 100")

    fig = go.Figure()
    color_idx = 0

    for sector, results in all_results.items():
        if not results:
            continue

        # Collect all equity curves and align on date index
        curve_dfs = []
        for r in results:
            if r["equity_curve"]:
                ec = pd.DataFrame(r["equity_curve"])
                ec["date"] = pd.to_datetime(ec["date"])
                ec = ec.set_index("date")["equity"]
                curve_dfs.append(ec)

        if not curve_dfs:
            continue

        # Merge on common date range, forward-fill, compute mean
        merged = pd.concat(curve_dfs, axis=1, join="outer").ffill().bfill()
        avg_curve = merged.mean(axis=1)
        # Normalise to 100
        avg_curve = avg_curve / avg_curve.iloc[0] * 100

        col = SECTOR_COLORS[color_idx % len(SECTOR_COLORS)]
        color_idx += 1

        fig.add_trace(go.Scatter(
            x=avg_curve.index,
            y=avg_curve.values,
            name=sector,
            mode="lines",
            line=dict(color=col, width=2),
            hovertemplate=f"<b>{sector}</b><br>Date: %{{x|%Y-%m-%d}}<br>Equity: %{{y:.1f}}<extra></extra>",
        ))

    # Baseline
    fig.add_hline(y=100, line_dash="dot", line_color=f"rgba(232,228,220,0.2)", line_width=1)

    fig.update_layout(
        **PLOTLY_BASE,
        xaxis=dict(
            title="Date",
            title_font_color=DIM,
            tickfont_color=DIM,
            gridcolor=f"rgba(138,133,120,0.2)",
        ),
        yaxis=dict(
            title="Equity (normalised to 100)",
            title_font_color=DIM,
            tickfont_color=DIM,
            gridcolor=f"rgba(138,133,120,0.2)",
        ),
        legend=dict(font_color=TEXT, bgcolor=BG2, bordercolor=DIM, orientation="v"),
        margin=dict(t=40, b=60, l=70, r=160),
        height=480,
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_rs_backtest_page() -> None:
    """Entry point called by the parent dashboard router."""

    _cyberpunk_css()

    # Page header
    st.markdown(
        f"""
        <div style="margin-bottom:8px;">
            <span style="color:{CYAN}; font-size:1.7rem; font-weight:800;
                         letter-spacing:0.06em; text-shadow:0 0 16px {CYAN};">
                RS MOMENTUM BACKTEST
            </span>
            <span style="color:{DIM}; font-size:0.85rem; margin-left:16px;">
                EMA(20/50/200) crossover · Top RS% pre-filter · 11 US Sector ETFs
            </span>
        </div>
        <div style="height:2px; background:linear-gradient(90deg,{CYAN},{MAGENTA},transparent);
                    margin-bottom:24px;"></div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Sidebar controls
    # ------------------------------------------------------------------
    # ── Controls in main area ────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])

    with ctrl_col1:
        period_label = st.radio(
            "Backtest Period",
            options=["1y", "2y", "3y", "5y"],
            index=2,
            horizontal=True,
            key="rs_bt_period",
        )
        years = YEARS_MAP[period_label]

    with ctrl_col2:
        top_pct = st.slider(
            "RS Top %",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            key="rs_bt_top_pct",
            help="Keep only the top N% of tickers by 6-month momentum before backtesting",
        )

    with ctrl_col3:
        st.markdown(
            f"""
            <div style='padding:12px; background:{BG2};
                        border:1px solid {DIM}; border-radius:6px;
                        font-size:0.72rem; color:{DIM}; line-height:1.6;'>
                <b style='color:{CYAN}'>Strategy:</b><br>
                BUY: EMA20 ↑ cross EMA50 AND Close > EMA200<br>
                SELL: EMA20 ↓ cross EMA50 OR Close < EMA200<br><br>
                <b style='color:{CYAN}'>RS Filter:</b><br>
                Top {top_pct}% by 6M momentum per sector<br><br>
                <b style='color:{CYAN}'>Data:</b> yfinance · TTL 1h
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#1a1a22;'>", unsafe_allow_html=True)
    run_btn = st.button(
        "▶  RUN BACKTEST",
        use_container_width=True,
        type="primary",
        key="rs_bt_run",
    )

    # ------------------------------------------------------------------
    # Run on button press
    # ------------------------------------------------------------------
    if not run_btn:
        st.markdown(
            f"""
            <div style="text-align:center; padding:80px 0; color:{DIM};">
                <div style="font-size:2.5rem; margin-bottom:16px; opacity:0.4;">◈</div>
                <div style="font-size:1rem;">Configure parameters above and press
                    <b style="color:{CYAN}">RUN BACKTEST</b>
                </div>
                <div style="font-size:0.8rem; margin-top:8px;">
                    Analysis may take 2–5 minutes for full sector coverage
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ------------------------------------------------------------------
    # Execute backtest with progress feedback
    # ------------------------------------------------------------------
    progress_bar = st.progress(0, text="Initialising sector scan…")

    try:
        progress_bar.progress(5, text="Fetching RS rankings and running backtests…")
        result_df, all_results = run_rs_sector_backtest(years=years, top_pct=float(top_pct))
        progress_bar.progress(100, text="Backtest complete.")
        progress_bar.empty()
    except Exception as exc:
        progress_bar.empty()
        st.error(f"Backtest failed: {exc}")
        return

    if result_df.empty:
        st.warning("No results returned. Check data availability or try different parameters.")
        return

    # Drop internal column before display
    display_df = result_df.drop(columns=["_results"], errors="ignore").copy()
    display_df = display_df.sort_values("Avg_TotalReturn", ascending=False).reset_index(drop=True)

    # Rename for display
    display_df = display_df.rename(columns={
        "Tickers_Tested": "# Tested",
        "Avg_WinRate": "Win Rate %",
        "Avg_ProfitFactor": "Profit Factor",
        "Avg_MaxDD": "Max DD %",
        "Avg_TotalReturn": "Total Return %",
        "Best_Ticker": "Best Ticker",
        "Best_Return": "Best Return",
        "Worst_Ticker": "Worst Ticker",
        "Worst_Return": "Worst Return",
    })

    # ------------------------------------------------------------------
    # SECTION 1: KPI cards + Sector table
    # ------------------------------------------------------------------
    _section_header("SECTOR PERFORMANCE", f"Top {top_pct}% RS tickers · {period_label} backtest")

    total_tested = int(display_df["# Tested"].sum())
    avg_wr = float(display_df["Win Rate %"].mean())
    avg_pf = float(display_df["Profit Factor"].mean())
    avg_dd = float(display_df["Max DD %"].mean())

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        _kpi_card("Instruments Tested", str(total_tested), CYAN)
    with kpi2:
        _kpi_card("Avg Win Rate", f"{avg_wr:.1f}%", GREEN if avg_wr > 55 else YELLOW)
    with kpi3:
        _kpi_card("Avg Profit Factor", f"{avg_pf:.2f}", GREEN if avg_pf >= 1.5 else YELLOW)
    with kpi4:
        _kpi_card("Avg Max Drawdown", f"{avg_dd:.1f}%", RED if avg_dd < -20 else YELLOW)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    styled = _style_sector_df(display_df)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # SECTION 2: Sector comparison chart
    # ------------------------------------------------------------------
    _render_sector_comparison(display_df)

    # ------------------------------------------------------------------
    # SECTION 3: Top performers
    # ------------------------------------------------------------------
    _render_top_performers(all_results)

    # ------------------------------------------------------------------
    # SECTION 4: Individual trades drill-down
    # ------------------------------------------------------------------
    _render_trade_drilldown(all_results)

    # ------------------------------------------------------------------
    # SECTION 5: Equity curves
    # ------------------------------------------------------------------
    _render_equity_curves(all_results)

    # Footer
    st.markdown(
        f"""
        <div style="margin-top:40px; padding-top:16px; border-top:1px solid {DIM};
                    color:{DIM}; font-size:0.7rem; text-align:center;">
            RS Momentum Backtest · EMA(20/50/200) crossover · Data: yfinance ·
            Past performance does not guarantee future results.
        </div>
        """,
        unsafe_allow_html=True,
    )
