"""
heatmap_streamlit.py
Performance Heatmap module — Streamlit page for Wolf Panel.

Cyberpunk theme: #050510 background, #00ffff cyan, #ff00ff magenta.
Entry point: render_heatmap_page()

Sections:
  1. Heatmap Grid (Plotly Treemap)
  2. Performance Table (st.dataframe with styled.map())
  3. Top/Bottom Movers (horizontal bar charts)
  4. Summary KPI Row (HTML cards)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker registry import with fallback
# ---------------------------------------------------------------------------
try:
    from cagr.cagr_loader import load_nordic_tickers, load_etf_tickers
except ImportError:
    try:
        from dashboard.cagr.cagr_loader import load_nordic_tickers, load_etf_tickers
    except ImportError:
        try:
            import sys
            import os as _os
            _parent = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
            if _parent not in sys.path:
                sys.path.insert(0, _parent)
            from cagr.cagr_loader import load_nordic_tickers, load_etf_tickers
        except ImportError:
            # Last-resort inline definitions (mirrors cagr_loader)
            def load_nordic_tickers() -> Dict[str, dict]:  # type: ignore[misc]
                return {
                    "VOLV-B.ST": {"name": "Volvo B", "country": "Sweden", "sector": "Industrials"},
                    "AZN.ST":    {"name": "AstraZeneca", "country": "Sweden", "sector": "Healthcare"},
                    "SAND.ST":   {"name": "Sandvik", "country": "Sweden", "sector": "Industrials"},
                    "SEB-A.ST":  {"name": "SEB A", "country": "Sweden", "sector": "Financials"},
                    "HEXA-B.ST": {"name": "Hexagon B", "country": "Sweden", "sector": "Technology"},
                    "ERIC-B.ST": {"name": "Ericsson B", "country": "Sweden", "sector": "Technology"},
                    "ATCO-A.ST": {"name": "Atlas Copco A", "country": "Sweden", "sector": "Industrials"},
                    "ABB.ST":    {"name": "ABB", "country": "Sweden", "sector": "Industrials"},
                    "INVE-B.ST": {"name": "Investor B", "country": "Sweden", "sector": "Financials"},
                    "ASSA-B.ST": {"name": "Assa Abloy B", "country": "Sweden", "sector": "Industrials"},
                    "ESSITY-B.ST": {"name": "Essity B", "country": "Sweden", "sector": "Consumer Staples"},
                    "HM-B.ST":   {"name": "H&M B", "country": "Sweden", "sector": "Consumer Discretionary"},
                    "SWED-A.ST": {"name": "Swedbank A", "country": "Sweden", "sector": "Financials"},
                    "SHB-A.ST":  {"name": "Handelsbanken A", "country": "Sweden", "sector": "Financials"},
                    "ALFA.ST":   {"name": "Alfa Laval", "country": "Sweden", "sector": "Industrials"},
                    "SKF-B.ST":  {"name": "SKF B", "country": "Sweden", "sector": "Industrials"},
                    "EVO.ST":    {"name": "Evolution", "country": "Sweden", "sector": "Consumer Discretionary"},
                    "NIBE-B.ST": {"name": "NIBE B", "country": "Sweden", "sector": "Industrials"},
                    "SINCH.ST":  {"name": "Sinch", "country": "Sweden", "sector": "Technology"},
                    "BOL.ST":    {"name": "Boliden", "country": "Sweden", "sector": "Materials"},
                    "EQNR.OL":   {"name": "Equinor", "country": "Norway", "sector": "Energy"},
                    "DNB.OL":    {"name": "DNB Bank", "country": "Norway", "sector": "Financials"},
                    "NHY.OL":    {"name": "Norsk Hydro", "country": "Norway", "sector": "Materials"},
                    "MOWI.OL":   {"name": "Mowi", "country": "Norway", "sector": "Consumer Staples"},
                    "AKRBP.OL":  {"name": "Aker BP", "country": "Norway", "sector": "Energy"},
                    "TEL.OL":    {"name": "Telenor", "country": "Norway", "sector": "Communication Services"},
                    "SUBC.OL":   {"name": "Subsea 7", "country": "Norway", "sector": "Energy"},
                    "SGSN.OL":   {"name": "Storebrand", "country": "Norway", "sector": "Financials"},
                    "SALM.OL":   {"name": "SalMar", "country": "Norway", "sector": "Consumer Staples"},
                    "BAKKA.OL":  {"name": "Bakkafrost", "country": "Norway", "sector": "Consumer Staples"},
                    "NOVO-B.CO": {"name": "Novo Nordisk B", "country": "Denmark", "sector": "Healthcare"},
                    "DSV.CO":    {"name": "DSV", "country": "Denmark", "sector": "Industrials"},
                    "MAERSK-B.CO": {"name": "Maersk B", "country": "Denmark", "sector": "Industrials"},
                    "CARL-B.CO": {"name": "Carlsberg B", "country": "Denmark", "sector": "Consumer Staples"},
                    "VWS.CO":    {"name": "Vestas Wind", "country": "Denmark", "sector": "Energy"},
                    "ORSTED.CO": {"name": "Orsted", "country": "Denmark", "sector": "Utilities"},
                    "NZYM-B.CO": {"name": "Novozymes B", "country": "Denmark", "sector": "Materials"},
                    "GMAB.CO":   {"name": "Genmab", "country": "Denmark", "sector": "Healthcare"},
                    "TRYG.CO":   {"name": "Tryg", "country": "Denmark", "sector": "Financials"},
                    "COLOB.CO":  {"name": "Coloplast B", "country": "Denmark", "sector": "Healthcare"},
                    "NOKIA.HE":  {"name": "Nokia", "country": "Finland", "sector": "Technology"},
                    "SAMPO.HE":  {"name": "Sampo", "country": "Finland", "sector": "Financials"},
                    "NESTE.HE":  {"name": "Neste", "country": "Finland", "sector": "Energy"},
                    "UPM.HE":    {"name": "UPM-Kymmene", "country": "Finland", "sector": "Materials"},
                    "FORTUM.HE": {"name": "Fortum", "country": "Finland", "sector": "Utilities"},
                    "KNEBV.HE":  {"name": "Kone", "country": "Finland", "sector": "Industrials"},
                    "STERV.HE":  {"name": "Stora Enso R", "country": "Finland", "sector": "Materials"},
                    "METSO.HE":  {"name": "Metso", "country": "Finland", "sector": "Industrials"},
                    "KESKOB.HE": {"name": "Kesko B", "country": "Finland", "sector": "Consumer Staples"},
                    "WRT1V.HE":  {"name": "Wartsila", "country": "Finland", "sector": "Industrials"},
                }

            def load_etf_tickers() -> Dict[str, dict]:  # type: ignore[misc]
                return {
                    "IWDA.AS":  {"name": "iShares Core MSCI World UCITS ETF", "country": "IE", "sector": "ETF", "category": "Global Equity"},
                    "IEMA.AS":  {"name": "iShares MSCI EM UCITS ETF", "country": "IE", "sector": "ETF", "category": "Emerging Markets"},
                    "CSPX.AS":  {"name": "iShares Core S&P 500 UCITS ETF", "country": "IE", "sector": "ETF", "category": "US Equity"},
                    "VWRL.AS":  {"name": "Vanguard FTSE All-World UCITS ETF", "country": "IE", "sector": "ETF", "category": "Global Equity"},
                    "VUSA.AS":  {"name": "Vanguard S&P 500 UCITS ETF", "country": "IE", "sector": "ETF", "category": "US Equity"},
                    "IQQQ.DE":  {"name": "iShares Nasdaq 100 UCITS ETF", "country": "IE", "sector": "ETF", "category": "US Technology"},
                    "BTCE.DE":  {"name": "ETC Group Physical Bitcoin ETP", "country": "DE", "sector": "ETF", "category": "Crypto"},
                }

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
BG      = "#050510"
BG2     = "#0a0a1e"
CYAN    = "#00ffff"
MAGENTA = "#ff00ff"
GREEN   = "#00ff88"
RED     = "#ff3355"
YELLOW  = "#ffdd00"
TEXT    = "#e0e0ff"
DIM     = "#4a4a6a"

PLOTLY_TEMPLATE = "plotly_dark"

# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        /* ── Wolf Panel Heatmap – Cyberpunk CSS ── */
        html, body, .stApp {{
            background-color: {BG};
            color: {TEXT};
            font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
        }}

        h1, h2, h3, h4 {{
            color: {CYAN};
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

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
            margin-bottom: 8px;
        }}
        .kpi-card .kpi-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: {CYAN};
            line-height: 1.1;
        }}
        .kpi-card .kpi-positive {{ color: {GREEN}; }}
        .kpi-card .kpi-negative {{ color: {RED};   }}
        .kpi-card .kpi-label {{
            font-size: 0.7rem;
            color: {DIM};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 4px;
        }}

        /* Section headers */
        .section-header {{
            color: {CYAN};
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            border-bottom: 1px solid rgba(0,255,255,0.2);
            padding-bottom: 6px;
            margin: 24px 0 12px 0;
        }}

        /* Sidebar labels */
        .stSelectbox label, .stSlider label, .stMultiSelect label, .stRadio label {{
            color: {CYAN} !important;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        /* Buttons */
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

        hr {{
            border-color: rgba(0,255,255,0.13);
        }}

        .stProgress > div > div {{
            background: linear-gradient(90deg, {CYAN}, {MAGENTA});
        }}

        [data-testid="stExpander"] {{
            border: 1px solid rgba(0,255,255,0.13) !important;
            background-color: {BG2} !important;
        }}

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
# Data layer
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_heatmap_data(tickers_tuple: tuple) -> pd.DataFrame:
    """
    Fetch 1 month of price data for all tickers and compute:
      - 1D return %
      - 5D return %
      - 1M return %
      - Trend state (vs EMA200)
    Returns a DataFrame with one row per ticker.
    """
    import yfinance as yf

    tickers = list(tickers_tuple)
    if not tickers:
        return pd.DataFrame()

    # Download ~1 year so EMA200 is stable; we only need 1M for returns
    try:
        raw = yf.download(
            tickers=tickers,
            period="1y",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.warning("Heatmap batch download failed: %s", exc)
        raw = None

    # Build per-ticker close series
    close_map: Dict[str, pd.Series] = {}

    if raw is not None and not raw.empty:
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    col = next(
                        (c for c in ("Close", "Adj Close") if (c, ticker) in raw.columns),
                        None,
                    )
                    if col:
                        s = raw[(col, ticker)].dropna()
                        if not s.empty:
                            close_map[ticker] = s
                except Exception:
                    pass
        else:
            # Single ticker scenario
            col = next((c for c in ("Close", "Adj Close") if c in raw.columns), None)
            if col and len(tickers) == 1:
                s = raw[col].dropna()
                if not s.empty:
                    close_map[tickers[0]] = s

    # Fall back for any missing
    missing = [t for t in tickers if t not in close_map]
    for ticker in missing:
        try:
            tk = yf.Ticker(ticker)
            df_t = tk.history(period="1y", auto_adjust=True)
            col = next((c for c in ("Close", "Adj Close") if c in df_t.columns), None)
            if col and not df_t.empty:
                close_map[ticker] = df_t[col].dropna()
        except Exception as exc:
            logger.warning("Heatmap fallback download(%s): %s", ticker, exc)

    # Compute metrics
    rows = []
    for ticker, series in close_map.items():
        if series.empty or len(series) < 2:
            continue
        try:
            last   = float(series.iloc[-1])
            prev1  = float(series.iloc[-2])  if len(series) >= 2  else last
            prev5  = float(series.iloc[-6])  if len(series) >= 6  else series.iloc[0]
            prev21 = float(series.iloc[-22]) if len(series) >= 22 else series.iloc[0]

            ret_1d = (last / prev1  - 1) * 100
            ret_5d = (last / prev5  - 1) * 100
            ret_1m = (last / prev21 - 1) * 100

            # EMA200
            ema200_series = series.ewm(span=200, adjust=False).mean()
            ema200 = float(ema200_series.iloc[-1])
            margin = 0.01 * ema200  # 1% band for "Neutral"
            if last > ema200 + margin:
                trend = "Uptrend"
            elif last < ema200 - margin:
                trend = "Downtrend"
            else:
                trend = "Neutral"

            rows.append({
                "ticker":  ticker,
                "ret_1d":  round(ret_1d, 2),
                "ret_5d":  round(ret_5d, 2),
                "ret_1m":  round(ret_1m, 2),
                "trend":   trend,
                "last_price": round(last, 4),
            })
        except Exception as exc:
            logger.debug("Heatmap metric calc(%s): %s", ticker, exc)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_full_df(
    market_choice: str,
    country_choice: str,
    sector_choice: str,
) -> pd.DataFrame:
    """
    Load ticker registries, apply filters, fetch price metrics,
    and return a merged DataFrame ready for display.
    """
    nordic  = load_nordic_tickers()
    etf     = load_etf_tickers()

    # Build combined registry with market_type label
    registry: Dict[str, dict] = {}
    for k, v in nordic.items():
        registry[k] = {**v, "market_type": "Nordic Stocks"}
    for k, v in etf.items():
        registry[k] = {**v, "market_type": "UCITS ETFs"}

    # Market filter
    if market_choice == "Nordic Stocks":
        registry = {k: v for k, v in registry.items() if v["market_type"] == "Nordic Stocks"}
    elif market_choice == "UCITS ETFs":
        registry = {k: v for k, v in registry.items() if v["market_type"] == "UCITS ETFs"}

    # Country filter (only meaningful for Nordic stocks)
    if country_choice != "All":
        registry = {k: v for k, v in registry.items() if v.get("country") == country_choice}

    # Sector filter
    if sector_choice != "All":
        registry = {k: v for k, v in registry.items() if v.get("sector") == sector_choice}

    if not registry:
        return pd.DataFrame()

    # Fetch price metrics (cached)
    tickers_tuple = tuple(sorted(registry.keys()))
    price_df = _fetch_heatmap_data(tickers_tuple)

    if price_df.empty:
        return pd.DataFrame()

    # Merge metadata
    meta_rows = []
    for ticker, meta in registry.items():
        meta_rows.append({
            "ticker":  ticker,
            "name":    meta.get("name", ticker),
            "country": meta.get("country", meta.get("category", "—")),
            "sector":  meta.get("sector", "Unknown"),
        })
    meta_df = pd.DataFrame(meta_rows)

    merged = meta_df.merge(price_df, on="ticker", how="left")
    # Drop rows where we have no data at all
    merged = merged.dropna(subset=["ret_1d", "ret_5d", "ret_1m"], how="all")
    # Fill remaining NaNs with 0 for returns
    for col in ("ret_1d", "ret_5d", "ret_1m"):
        merged[col] = merged[col].fillna(0.0)
    merged["trend"] = merged["trend"].fillna("Neutral")

    return merged


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _section_header(title: str) -> None:
    st.markdown(
        f"<div class='section-header'>{title}</div>",
        unsafe_allow_html=True,
    )


def _kpi_card(label: str, value: str, value_class: str = "") -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-value {value_class}">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Section 1 — Treemap Heatmap
# ---------------------------------------------------------------------------

def _render_treemap(df: pd.DataFrame, timeframe: str) -> None:
    _section_header("Performance Heatmap")

    col_map = {"1D": "ret_1d", "5D": "ret_5d", "1M": "ret_1m"}
    ret_col = col_map.get(timeframe, "ret_1d")

    if df.empty or ret_col not in df.columns:
        st.warning("No data available for heatmap.")
        return

    working = df.copy()
    # Clamp for color scale (±10%)
    max_abs = max(working[ret_col].abs().max(), 0.01)
    clamp = max(max_abs, 5.0)

    # Normalise to [0, 1] for colorscale
    working["_norm"] = (working[ret_col] + clamp) / (2 * clamp)
    working["_norm"] = working["_norm"].clip(0, 1)

    labels    = working["ticker"].tolist()
    parents   = [""] * len(working)
    values    = [1] * len(working)  # equal size
    norm_vals = working["_norm"].tolist()
    ret_vals  = working[ret_col].tolist()

    ret_1d_vals = working["ret_1d"].tolist()
    ret_5d_vals = working["ret_5d"].tolist()
    ret_1m_vals = working["ret_1m"].tolist()
    names_list  = working["name"].tolist()
    sectors     = working["sector"].tolist()
    countries   = working["country"].tolist()

    # Build custom color list using rgba (NEVER 8-digit hex)
    def _val_to_rgba(norm: float) -> str:
        """Interpolate through red → near-black → green."""
        if norm < 0.5:
            # red (#ff3355) → dark (#1a1a2e)
            t = norm / 0.5
            r = int(255 + t * (26  - 255))
            g = int(51  + t * (26  - 51))
            b = int(85  + t * (46  - 85))
        else:
            # dark (#1a1a2e) → green (#00ff88)
            t = (norm - 0.5) / 0.5
            r = int(26  + t * (0   - 26))
            g = int(26  + t * (255 - 26))
            b = int(46  + t * (136 - 46))
        return f"rgba({r},{g},{b},1)"

    marker_colors = [_val_to_rgba(n) for n in norm_vals]

    # Custom text inside each cell
    cell_texts = [
        f"<b>{lbl}</b><br>{ret:+.2f}%"
        for lbl, ret in zip(labels, ret_vals)
    ]

    # Hover template data
    custom_data = list(zip(
        names_list,
        sectors,
        countries,
        ret_1d_vals,
        ret_5d_vals,
        ret_1m_vals,
    ))

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            text=cell_texts,
            textinfo="text",
            customdata=custom_data,
            hovertemplate=(
                "<b>%{label}</b><br>"
                "%{customdata[0]}<br>"
                "Sector: %{customdata[1]}<br>"
                "Country: %{customdata[2]}<br>"
                "1D: %{customdata[3]:+.2f}%<br>"
                "5D: %{customdata[4]:+.2f}%<br>"
                "1M: %{customdata[5]:+.2f}%"
                "<extra></extra>"
            ),
            marker=dict(
                colors=marker_colors,
                line=dict(width=1.5, color=BG),
                pad=dict(t=4, l=4, r=4, b=4),
            ),
            textfont=dict(
                family="JetBrains Mono, Fira Code, monospace",
                size=12,
                color=TEXT,
            ),
            tiling=dict(packing="squarify", pad=2),
        )
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        height=600,
        margin=dict(l=4, r=4, t=4, b=4),
        font=dict(color=TEXT),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Section 2 — Performance Table
# ---------------------------------------------------------------------------

_TREND_COLORS = {"Uptrend": GREEN, "Downtrend": RED, "Neutral": YELLOW}


def _style_return(val: float) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return f"color: {DIM};"
    if v > 0:
        return f"color: {GREEN}; font-weight: 600;"
    if v < 0:
        return f"color: {RED}; font-weight: 600;"
    return f"color: {DIM};"


def _style_trend(val: str) -> str:
    c = _TREND_COLORS.get(str(val), DIM)
    return f"color: {c}; font-weight: 700;"


def _render_table(df: pd.DataFrame, sort_by: str) -> None:
    _section_header("Performance Table")

    if df.empty:
        st.info("No data to display.")
        return

    display = df[["ticker", "name", "country", "sector", "ret_1d", "ret_5d", "ret_1m", "trend"]].copy()
    display.columns = ["Ticker", "Name", "Country", "Sector", "1D %", "5D %", "1M %", "Trend"]

    # Apply sort
    if sort_by == "Performance":
        # sort will be applied at call site with selected timeframe col
        pass  # already handled in caller
    elif sort_by == "Name":
        display = display.sort_values("Name")
    elif sort_by == "Sector":
        display = display.sort_values(["Sector", "Name"])

    styled = display.style
    for col in ("1D %", "5D %", "1M %"):
        styled = styled.map(_style_return, subset=[col])
    styled = styled.map(_style_trend, subset=["Trend"])
    styled = styled.set_properties(**{
        "background-color": BG2,
        "color": TEXT,
        "border-color": "rgba(0,255,255,0.13)",
    })
    styled = styled.format({
        "1D %": lambda x: f"{x:+.2f}%",
        "5D %": lambda x: f"{x:+.2f}%",
        "1M %": lambda x: f"{x:+.2f}%",
    })

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(620, 38 + 35 * len(display)),
    )


# ---------------------------------------------------------------------------
# Section 3 — Top/Bottom Movers
# ---------------------------------------------------------------------------

def _render_movers(df: pd.DataFrame, timeframe: str) -> None:
    _section_header(f"Top & Bottom Movers — {timeframe}")

    col_map = {"1D": "ret_1d", "5D": "ret_5d", "1M": "ret_1m"}
    ret_col = col_map.get(timeframe, "ret_1d")

    if df.empty or ret_col not in df.columns:
        st.info("No data to display.")
        return

    sorted_df = df.sort_values(ret_col, ascending=False).dropna(subset=[ret_col])

    top10    = sorted_df.head(10)
    bottom10 = sorted_df.tail(10).sort_values(ret_col, ascending=True)

    col_left, col_right = st.columns(2)

    # ── Top movers ────────────────────────────────────────────────────────
    with col_left:
        st.markdown(
            f"<div style='color:{GREEN};font-size:0.72rem;text-transform:uppercase;"
            f"letter-spacing:0.1em;margin-bottom:6px;font-weight:700;'>Top 10 Movers</div>",
            unsafe_allow_html=True,
        )
        if top10.empty:
            st.info("No data.")
        else:
            labels_top = top10["ticker"].tolist()
            vals_top   = top10[ret_col].tolist()
            bar_colors_top = [
                f"rgba(0,255,136,{max(0.4, min(1.0, abs(v) / 10))})"
                for v in vals_top
            ]
            fig_top = go.Figure(
                go.Bar(
                    x=vals_top,
                    y=labels_top,
                    orientation="h",
                    marker=dict(
                        color=bar_colors_top,
                        line=dict(width=0),
                    ),
                    text=[f"{v:+.2f}%" for v in vals_top],
                    textposition="outside",
                    textfont=dict(color=GREEN, size=11),
                    hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
                )
            )
            fig_top.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor=BG2,
                plot_bgcolor=BG,
                height=max(300, 30 * len(top10) + 60),
                margin=dict(l=8, r=60, t=10, b=10),
                xaxis=dict(
                    color=DIM,
                    gridcolor="rgba(0,255,255,0.07)",
                    zeroline=True,
                    zerolinecolor="rgba(0,255,255,0.2)",
                    tickformat="+.1f",
                ),
                yaxis=dict(
                    color=TEXT,
                    autorange="reversed",
                    tickfont=dict(size=11),
                ),
                showlegend=False,
                font=dict(color=TEXT),
            )
            st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": False})

    # ── Bottom movers ─────────────────────────────────────────────────────
    with col_right:
        st.markdown(
            f"<div style='color:{RED};font-size:0.72rem;text-transform:uppercase;"
            f"letter-spacing:0.1em;margin-bottom:6px;font-weight:700;'>Bottom 10 Movers</div>",
            unsafe_allow_html=True,
        )
        if bottom10.empty:
            st.info("No data.")
        else:
            labels_bot = bottom10["ticker"].tolist()
            vals_bot   = bottom10[ret_col].tolist()
            bar_colors_bot = [
                f"rgba(255,51,85,{max(0.4, min(1.0, abs(v) / 10))})"
                for v in vals_bot
            ]
            fig_bot = go.Figure(
                go.Bar(
                    x=vals_bot,
                    y=labels_bot,
                    orientation="h",
                    marker=dict(
                        color=bar_colors_bot,
                        line=dict(width=0),
                    ),
                    text=[f"{v:+.2f}%" for v in vals_bot],
                    textposition="outside",
                    textfont=dict(color=RED, size=11),
                    hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
                )
            )
            fig_bot.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor=BG2,
                plot_bgcolor=BG,
                height=max(300, 30 * len(bottom10) + 60),
                margin=dict(l=8, r=60, t=10, b=10),
                xaxis=dict(
                    color=DIM,
                    gridcolor="rgba(255,51,85,0.07)",
                    zeroline=True,
                    zerolinecolor="rgba(255,51,85,0.2)",
                    tickformat="+.1f",
                ),
                yaxis=dict(
                    color=TEXT,
                    tickfont=dict(size=11),
                ),
                showlegend=False,
                font=dict(color=TEXT),
            )
            st.plotly_chart(fig_bot, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Section 4 — KPI Row
# ---------------------------------------------------------------------------

def _render_kpi_row(df: pd.DataFrame) -> None:
    _section_header("Market Summary")

    if df.empty:
        return

    avg_1d = df["ret_1d"].mean()
    avg_1m = df["ret_1m"].mean()
    uptrend_count = (df["trend"] == "Uptrend").sum()
    total_count   = len(df)

    # Best performer (1D)
    best_idx = df["ret_1d"].idxmax()
    best_ticker = df.loc[best_idx, "ticker"] if best_idx is not None else "N/A"
    best_val    = df.loc[best_idx, "ret_1d"] if best_idx is not None else 0.0

    avg_1d_class = "kpi-positive" if avg_1d >= 0 else "kpi-negative"
    avg_1m_class = "kpi-positive" if avg_1m >= 0 else "kpi-negative"
    best_class   = "kpi-positive" if best_val >= 0 else "kpi-negative"

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Avg 1D Return",     f"{avg_1d:+.2f}%",                       avg_1d_class),
        (c2, "Avg 1M Return",     f"{avg_1m:+.2f}%",                       avg_1m_class),
        (c3, "Uptrend",           f"{uptrend_count}/{total_count}",          ""),
        (c4, f"Best Performer",   f"{best_ticker} {best_val:+.2f}%",        best_class),
    ]
    for col_obj, label, value, cls in cards:
        with col_obj:
            st.markdown(_kpi_card(label, value, cls), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

def _render_header() -> None:
    st.markdown(
        f"""
        <div style='padding:20px 0 10px 0;'>
            <h1 style='margin:0;font-size:2rem;letter-spacing:0.12em;
                       background:linear-gradient(90deg,{CYAN},{MAGENTA});
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
                PERFORMANCE HEATMAP
            </h1>
            <div style='color:{DIM};font-size:0.7rem;letter-spacing:0.15em;margin-top:4px;'>
                NORDIC STOCKS &amp; UCITS ETFs — REAL-TIME PRICE PERFORMANCE
            </div>
        </div>
        <hr style='border-color:rgba(0,255,255,0.13);margin-bottom:16px;'/>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_heatmap_page() -> None:
    """Full Performance Heatmap Streamlit page."""
    _inject_css()
    _render_header()

    # ── Sidebar controls ───────────────────────────────────────────────────
    st.sidebar.markdown(
        f"<div style='color:{CYAN};font-size:0.8rem;letter-spacing:0.1em;"
        f"text-transform:uppercase;padding:10px 0 6px 0;'>"
        f"Heatmap Controls</div>",
        unsafe_allow_html=True,
    )

    market_choice = st.sidebar.selectbox(
        "Market",
        options=["All", "Nordic Stocks", "UCITS ETFs"],
        index=0,
        key="heatmap_market",
    )

    country_choice = st.sidebar.selectbox(
        "Country",
        options=["All", "Sweden", "Norway", "Denmark", "Finland"],
        index=0,
        key="heatmap_country",
    )

    # Build sector list dynamically from both registries
    nordic_meta = load_nordic_tickers()
    etf_meta    = load_etf_tickers()
    all_sectors = sorted(set(
        v.get("sector", "Unknown")
        for v in {**nordic_meta, **etf_meta}.values()
    ))
    sector_choice = st.sidebar.selectbox(
        "Sector",
        options=["All"] + all_sectors,
        index=0,
        key="heatmap_sector",
    )

    timeframe = st.sidebar.radio(
        "Timeframe",
        options=["1D", "5D", "1M"],
        index=0,
        horizontal=True,
        key="heatmap_timeframe",
    )

    sort_by = st.sidebar.selectbox(
        "Sort By",
        options=["Performance", "Name", "Sector"],
        index=0,
        key="heatmap_sortby",
    )

    st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.07);'/>", unsafe_allow_html=True)
    refresh_clicked = st.sidebar.button("⟳  REFRESH DATA", use_container_width=True)

    if refresh_clicked:
        st.cache_data.clear()

    # ── Load data ──────────────────────────────────────────────────────────
    with st.spinner("Loading market data..."):
        df = _build_full_df(market_choice, country_choice, sector_choice)

    if df.empty:
        st.markdown(
            f"<div style='color:{DIM};text-align:center;padding:60px;'>"
            "No data available for the selected filters. "
            "Try broadening your selection or refreshing data."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Apply performance sort to table (treemap always uses its own sort)
    col_map = {"1D": "ret_1d", "5D": "ret_5d", "1M": "ret_1m"}
    ret_col = col_map.get(timeframe, "ret_1d")
    if sort_by == "Performance":
        df = df.sort_values(ret_col, ascending=False)

    # ── Section 4: KPI row (top for quick summary) ────────────────────────
    _render_kpi_row(df)

    # ── Section 1: Heatmap treemap ────────────────────────────────────────
    _render_treemap(df, timeframe)

    # ── Section 2: Performance table ─────────────────────────────────────
    _render_table(df, sort_by)

    # ── Section 3: Top / Bottom movers ───────────────────────────────────
    _render_movers(df, timeframe)

    # Footer spacer
    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    st.set_page_config(
        page_title="Performance Heatmap — Wolf Panel",
        page_icon="🌡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_heatmap_page()
