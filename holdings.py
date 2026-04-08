"""
holdings.py — Portfolio Holdings Manager

Three portfolios tied to three strategies:
  - Swing Portfolio (max 5 holdings)
  - OVTLYR Portfolio (max 5 holdings)
  - Long Portfolio (max 10 holdings)

Each holding gets live signals from its strategy's scoring system.
Holdings persist in Streamlit session_state.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Cyberpunk theme
BG = "#050510"
BG2 = "#0a0a1e"
CYAN = "#00ffff"
MAGENTA = "#ff00ff"
GREEN = "#00ff88"
RED = "#ff3355"
YELLOW = "#ffdd00"
TEXT = "#e0e0ff"
DIM = "#4a4a6a"

# Portfolio config
PORTFOLIOS = {
    "swing": {"name": "Swing Portfolio", "max": 5, "color": CYAN, "strategy": "Swing"},
    "ovtlyr": {"name": "OVTLYR Portfolio", "max": 5, "color": MAGENTA, "strategy": "OVTLYR"},
    "long": {"name": "Long Portfolio", "max": 10, "color": GREEN, "strategy": "Long"},
}

# Signal imports (with fallbacks)
try:
    from ovtlyr.indicators.trend import compute_trend
    from ovtlyr.indicators.momentum import compute_momentum
    from ovtlyr.indicators.volatility import compute_volatility
    from ovtlyr.indicators.orderblocks import detect_orderblocks, classify_price_vs_ob
    from ovtlyr.signals.longterm_signals import compute_longterm_signal
    from ovtlyr.signals.swing_signals import compute_swing_signal
    _OVTLYR_OK = True
except ImportError:
    _OVTLYR_OK = False

try:
    from cagr.cagr_technical import score_technical
    from cagr.cagr_cycle import SECTOR_CONFIG, score_cycle_from_int
    from cagr.cagr_scoring import calculate_total_score
    _CAGR_OK = True
except ImportError:
    _CAGR_OK = False


# ── Persistent storage (JSON file) ────────────────────────────────────

import json
import os

_HOLDINGS_DIR = os.path.dirname(os.path.abspath(__file__))
_HOLDINGS_FILE = os.path.join(_HOLDINGS_DIR, ".holdings_data.json")


def _load_from_disk() -> dict:
    """Load all portfolios from JSON file."""
    if os.path.exists(_HOLDINGS_FILE):
        try:
            with open(_HOLDINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_to_disk(all_holdings: dict):
    """Save all portfolios to JSON file."""
    try:
        with open(_HOLDINGS_FILE, "w") as f:
            json.dump(all_holdings, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save holdings: %s", e)


def _get_holdings(portfolio_key: str) -> List[dict]:
    key = f"holdings_{portfolio_key}"
    if key not in st.session_state:
        disk_data = _load_from_disk()
        st.session_state[key] = disk_data.get(portfolio_key, [])
    return st.session_state[key]


def _set_holdings(portfolio_key: str, holdings: List[dict]):
    st.session_state[f"holdings_{portfolio_key}"] = holdings
    all_data = _load_from_disk()
    all_data[portfolio_key] = holdings
    _save_to_disk(all_data)


def _add_holding(portfolio_key: str, ticker: str, entry_price: float = 0, sector: str = "Unknown"):
    holdings = _get_holdings(portfolio_key)
    cfg = PORTFOLIOS[portfolio_key]

    if len(holdings) >= cfg["max"]:
        st.warning(f"Max {cfg['max']} innehav i {cfg['name']}. Ta bort ett först.")
        return False

    if any(h["ticker"] == ticker.upper() for h in holdings):
        st.warning(f"{ticker.upper()} finns redan i portföljen.")
        return False

    holdings.append({
        "ticker": ticker.upper(),
        "entry_price": entry_price,
        "sector": sector,
    })
    _set_holdings(portfolio_key, holdings)
    return True


def _remove_holding(portfolio_key: str, ticker: str):
    holdings = _get_holdings(portfolio_key)
    holdings = [h for h in holdings if h["ticker"] != ticker.upper()]
    _set_holdings(portfolio_key, holdings)


# ── Live data & signals ───────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_live_price(ticker: str, period: str = "6mo") -> dict:
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, auto_adjust=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        # Flatten MultiIndex columns (newer yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if df.empty:
            return {"price": 0, "change_1d": 0, "df": None}
        price = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else price
        change = (price / prev - 1) * 100
        return {"price": price, "change_1d": change, "df": df}
    except Exception:
        return {"price": 0, "change_1d": 0, "df": None}


def _compute_swing_signal(df: pd.DataFrame) -> dict:
    """Compute swing signal for a holding."""
    if not _OVTLYR_OK or df is None or df.empty or len(df) < 50:
        return {"signal": "N/A", "color": DIM}

    try:
        trend = compute_trend(df)
        momentum = compute_momentum(df)

        # Extract scalars
        def _last(val, default=0.0):
            if val is None: return default
            if isinstance(val, (pd.Series,)):
                return float(val.iloc[-1]) if len(val) > 0 else default
            return float(val) if val is not None else default

        trend_scalar = {
            "price": _last(trend.get("price", df["Close"].iloc[-1])),
            "ema50": _last(trend.get("ema50")),
            "ema200": _last(trend.get("ema200")),
            "trend_state": trend.get("trend_state", "neutral"),
            "regime_color": trend.get("regime_color", "orange"),
        }

        # Simple swing signal based on regime
        regime = trend_scalar["regime_color"]
        price = trend_scalar["price"]
        ema50 = trend_scalar["ema50"]
        ema200 = trend_scalar["ema200"]

        if regime == "green" and price > ema50:
            return {"signal": "HOLD", "color": GREEN, "detail": "Trend intakt"}
        elif regime == "green":
            return {"signal": "HOLD", "color": YELLOW, "detail": "Trend OK men under EMA50"}
        elif regime == "orange":
            return {"signal": "WATCH", "color": YELLOW, "detail": "Regime orange — bevaka"}
        else:
            return {"signal": "EXIT", "color": RED, "detail": "Regime röd — överväg stängning"}
    except Exception:
        return {"signal": "N/A", "color": DIM}


def _compute_ovtlyr_signal(df: pd.DataFrame) -> dict:
    """Compute OVTLYR signal for a holding."""
    if not _OVTLYR_OK or df is None or df.empty or len(df) < 50:
        return {"signal": "N/A", "color": DIM}

    try:
        trend = compute_trend(df)
        momentum = compute_momentum(df)
        volatility = compute_volatility(df)

        # Orderblocks
        obs = detect_orderblocks(df)
        current_price = float(df["Close"].iloc[-1])
        ob_analysis = classify_price_vs_ob(current_price, obs) if obs else {"signal_bias": "HOLD"}

        def _last(val, default=0.0):
            if val is None: return default
            if isinstance(val, (pd.Series,)):
                return float(val.iloc[-1]) if len(val) > 0 else default
            return float(val) if val is not None else default

        trend_scalar = {
            "price": _last(trend.get("price", current_price)),
            "ema10": _last(trend.get("ema10", df["Close"].ewm(span=10).mean().iloc[-1])),
            "ema20": _last(trend.get("ema20", df["Close"].ewm(span=20).mean().iloc[-1])),
            "ema50": _last(trend.get("ema50")),
            "ema200": _last(trend.get("ema200")),
            "regime_color": trend.get("regime_color", "orange"),
            "trend_state": trend.get("trend_state", "neutral"),
        }

        sentiment = {"score": 50, "label": "Neutral"}
        lt = compute_longterm_signal(trend_scalar, sentiment, volatility, ob_analysis, True)

        signal = lt.get("signal", "HOLD")
        if signal == "BUY":
            return {"signal": "HOLD ✓", "color": GREEN, "detail": f"OVTLYR NINE: {lt.get('ovtlyr_nine', 0)}"}
        elif signal == "SELL":
            return {"signal": "EXIT", "color": RED, "detail": f"OVTLYR NINE: {lt.get('ovtlyr_nine', 0)}"}
        elif signal == "REDUCE":
            return {"signal": "REDUCE", "color": MAGENTA, "detail": f"OVTLYR NINE: {lt.get('ovtlyr_nine', 0)}"}
        else:
            return {"signal": "HOLD", "color": YELLOW, "detail": f"OVTLYR NINE: {lt.get('ovtlyr_nine', 0)}"}
    except Exception as e:
        return {"signal": "N/A", "color": DIM, "detail": str(e)[:40]}


def _compute_long_signal(df: pd.DataFrame, sector: str = "Unknown") -> dict:
    """Compute long-term signal for a holding."""
    if df is None or df.empty or len(df) < 200:
        return {"signal": "N/A", "color": DIM}

    try:
        close = df["Close"].astype(float)
        price = float(close.iloc[-1])
        ema50 = float(close.ewm(span=50).mean().iloc[-1])
        ema200 = float(close.ewm(span=200).mean().iloc[-1])

        # Count days below EMA200
        ema200_series = close.ewm(span=200).mean()
        days_below = 0
        for i in range(len(close) - 1, max(len(close) - 10, 0), -1):
            if float(close.iloc[i]) < float(ema200_series.iloc[i]):
                days_below += 1
            else:
                break

        # Regime
        if price > ema200 and ema50 > ema200:
            regime = "green"
        elif price > ema200:
            regime = "orange"
        else:
            regime = "red"

        if regime == "green":
            return {"signal": "HOLD ✓", "color": GREEN, "detail": f"Pris {(price/ema200-1)*100:+.1f}% vs EMA200"}
        elif regime == "orange" and days_below < 3:
            return {"signal": "WATCH", "color": YELLOW, "detail": f"EMA50 under EMA200 — bevaka"}
        elif days_below >= 3:
            return {"signal": "REDUCE", "color": MAGENTA, "detail": f"{days_below}d under EMA200 — reducera 50%"}
        elif regime == "red":
            return {"signal": "EXIT", "color": RED, "detail": "Regime röd — sälj"}
        else:
            return {"signal": "HOLD", "color": YELLOW, "detail": "Neutral"}
    except Exception:
        return {"signal": "N/A", "color": DIM}


# ── UI Components ─────────────────────────────────────────────────────

def _holding_card(holding: dict, live: dict, signal: dict, portfolio_key: str) -> None:
    """Render a single holding card."""
    ticker = holding["ticker"]
    entry = holding.get("entry_price", 0)
    price = live.get("price", 0)
    change = live.get("change_1d", 0)
    sig = signal.get("signal", "N/A")
    sig_color = signal.get("color", DIM)
    detail = signal.get("detail", "")

    # P&L
    if entry > 0 and price > 0:
        pnl = (price / entry - 1) * 100
        pnl_color = GREEN if pnl > 0 else RED
        pnl_text = f"{pnl:+.1f}%"
    else:
        pnl_color = DIM
        pnl_text = "—"

    change_color = GREEN if change > 0 else RED

    col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 0.5])

    with col1:
        st.markdown(
            f"<div style='padding:4px 0;'>"
            f"<span style='color:{TEXT};font-size:1rem;font-weight:700;'>{ticker}</span>"
            f"<span style='color:{DIM};font-size:0.7rem;margin-left:8px;'>{holding.get('sector', '')}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"<div style='text-align:right;padding:4px 0;'>"
            f"<span style='color:{TEXT};font-size:0.9rem;'>{price:.2f}</span>"
            f" <span style='color:{change_color};font-size:0.75rem;'>({change:+.1f}%)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"<div style='text-align:center;padding:4px 0;'>"
            f"<span style='color:{pnl_color};font-size:0.95rem;font-weight:700;'>{pnl_text}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"<div style='text-align:center;padding:4px 0;'>"
            f"<span style='color:{sig_color};font-size:0.85rem;font-weight:700;'>{sig}</span>"
            f"<div style='color:{DIM};font-size:0.6rem;'>{detail}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col5:
        if st.button("✕", key=f"rm_{portfolio_key}_{ticker}", help=f"Ta bort {ticker}"):
            _remove_holding(portfolio_key, ticker)
            st.rerun()


def _render_portfolio(portfolio_key: str):
    """Render one portfolio section."""
    cfg = PORTFOLIOS[portfolio_key]
    color = cfg["color"]
    holdings = _get_holdings(portfolio_key)

    st.markdown(
        f"<div style='border-bottom:2px solid {color};padding-bottom:6px;margin-bottom:12px;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<h3 style='color:{color};margin:0;letter-spacing:0.08em;'>{cfg['name']}</h3>"
        f"<span style='color:{DIM};font-size:0.75rem;'>"
        f"{len(holdings)}/{cfg['max']} innehav · {cfg['strategy']}-strategi</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # Add holding form
    with st.expander(f"Lägg till i {cfg['name']}", expanded=len(holdings) == 0):
        ac1, ac2, ac3, ac4 = st.columns([2, 1.5, 1.5, 1])
        with ac1:
            new_ticker = st.text_input("Ticker", key=f"add_ticker_{portfolio_key}", placeholder="VOLV-B.ST").strip().upper()
        with ac2:
            new_entry = st.number_input("Entry-pris (valfritt)", value=0.0, min_value=0.0, step=0.01, key=f"add_entry_{portfolio_key}")
        with ac3:
            new_sector = st.text_input("Sektor", value="Unknown", key=f"add_sector_{portfolio_key}")
        with ac4:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("+ LÄGG TILL", key=f"add_btn_{portfolio_key}", use_container_width=True):
                if new_ticker:
                    if _add_holding(portfolio_key, new_ticker, new_entry, new_sector):
                        st.rerun()

    if not holdings:
        st.markdown(
            f"<div style='color:{DIM};text-align:center;padding:20px;font-size:0.85rem;'>"
            f"Inga innehav. Lägg till ovan.</div>",
            unsafe_allow_html=True,
        )
        return

    # Header row
    h1, h2, h3, h4, h5 = st.columns([2, 1.5, 1.5, 1.5, 0.5])
    h1.markdown(f"<span style='color:{DIM};font-size:0.7rem;'>TICKER</span>", unsafe_allow_html=True)
    h2.markdown(f"<span style='color:{DIM};font-size:0.7rem;text-align:right;display:block;'>PRIS (1D%)</span>", unsafe_allow_html=True)
    h3.markdown(f"<span style='color:{DIM};font-size:0.7rem;text-align:center;display:block;'>P&L</span>", unsafe_allow_html=True)
    h4.markdown(f"<span style='color:{DIM};font-size:0.7rem;text-align:center;display:block;'>SIGNAL</span>", unsafe_allow_html=True)
    h5.markdown(f"<span style='color:{DIM};font-size:0.7rem;'></span>", unsafe_allow_html=True)

    # Holdings with live signals
    for holding in holdings:
        ticker = holding["ticker"]
        # Long strategy needs 200+ days for EMA200 → fetch 2y
        fetch_period = "2y" if portfolio_key == "long" else "6mo"
        live = _fetch_live_price(ticker, period=fetch_period)
        df = live.get("df")

        # Compute signal based on strategy
        if portfolio_key == "swing":
            signal = _compute_swing_signal(df)
        elif portfolio_key == "ovtlyr":
            signal = _compute_ovtlyr_signal(df)
        else:  # long
            signal = _compute_long_signal(df, holding.get("sector", "Unknown"))

        _holding_card(holding, live, signal, portfolio_key)

    # Portfolio summary
    total_pnl = 0
    count = 0
    for h in holdings:
        live = _fetch_live_price(h["ticker"], period=fetch_period)
        if h.get("entry_price", 0) > 0 and live["price"] > 0:
            total_pnl += (live["price"] / h["entry_price"] - 1) * 100
            count += 1

    if count > 0:
        avg_pnl = total_pnl / count
        pnl_color = GREEN if avg_pnl > 0 else RED
        st.markdown(
            f"<div style='text-align:right;padding:8px 0;border-top:1px solid rgba(0,255,255,0.1);margin-top:8px;'>"
            f"<span style='color:{DIM};font-size:0.75rem;'>Snitt P&L: </span>"
            f"<span style='color:{pnl_color};font-size:0.9rem;font-weight:700;'>{avg_pnl:+.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Correlation Matrix ────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_returns_for_correlation(tickers: tuple, period: str = "6mo") -> pd.DataFrame:
    """Fetch daily returns for a list of tickers. Returns DataFrame of returns."""
    try:
        data = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data[["Close"]]
            close.columns = [tickers[0]]
        returns = close.pct_change().dropna()
        # Keep only last 90 trading days
        if len(returns) > 90:
            returns = returns.tail(90)
        return returns
    except Exception:
        return pd.DataFrame()


def _render_correlation_matrix(all_holdings: list):
    """
    Render a correlation heatmap for all active holdings.
    all_holdings: list of dicts with keys: ticker, strategy (swing/ovtlyr/long)
    """
    try:
        if len(all_holdings) < 2:
            st.info("Lägg till minst 2 innehav för att se korrelation.")
            return

        tickers = [h["ticker"] for h in all_holdings]
        strategy_map = {h["ticker"]: h["strategy"] for h in all_holdings}

        # Fetch returns
        returns_df = _fetch_returns_for_correlation(tuple(tickers))
        if returns_df.empty or len(returns_df.columns) < 2:
            st.warning("Korrelationsmatris: Kunde inte hämta tillräckligt med data.")
            return

        # Filter to tickers with valid data
        valid_tickers = [t for t in tickers if t in returns_df.columns]
        if len(valid_tickers) < 2:
            st.warning("Korrelationsmatris: Mindre än 2 ticker med data.")
            return

        returns_df = returns_df[valid_tickers]
        corr = returns_df.corr()

        # Color labels by strategy
        strategy_colors = {"swing": CYAN, "ovtlyr": MAGENTA, "long": GREEN}
        tick_colors = [strategy_colors.get(strategy_map.get(t, ""), DIM) for t in valid_tickers]

        # Build heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=valid_tickers,
            y=valid_tickers,
            colorscale=[
                [0.0, "rgba(0,255,136,0.9)"],
                [0.5, "rgba(255,255,255,0.9)"],
                [1.0, "rgba(255,51,85,0.9)"],
            ],
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont={"size": 11, "color": "rgba(224,224,255,0.9)"},
            hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>Korrelation: %{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title="Korr",
                titlefont=dict(color=DIM),
                tickfont=dict(color=DIM),
            ),
        ))

        fig.update_layout(
            title=dict(
                text="PORTFÖLJKORRELATION — 90d daglig avkastning",
                font=dict(color=CYAN, size=14),
            ),
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(color=TEXT),
            xaxis=dict(
                tickfont=dict(color=CYAN, size=10),
                gridcolor="rgba(0,255,255,0.1)",
            ),
            yaxis=dict(
                tickfont=dict(color=CYAN, size=10),
                gridcolor="rgba(0,255,255,0.1)",
                autorange="reversed",
            ),
            height=max(350, 60 * len(valid_tickers)),
            margin=dict(l=80, r=40, t=50, b=80),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Warnings for high correlations
        warned = set()
        for i in range(len(valid_tickers)):
            for j in range(i + 1, len(valid_tickers)):
                c = corr.iloc[i, j]
                pair = (valid_tickers[i], valid_tickers[j])
                if pair in warned:
                    continue
                if c > 0.85:
                    st.error(f"🔴 Extremt hög korrelation: {pair[0]}↔{pair[1]} ({c:.2f}) — överlappande risk")
                    warned.add(pair)
                elif c > 0.7:
                    st.warning(f"⚠️ Hög korrelation: {pair[0]}↔{pair[1]} ({c:.2f})")
                    warned.add(pair)

    except Exception as e:
        st.warning(f"Korrelationsmatris: {e}")


# ── Main render ───────────────────────────────────────────────────────

def render_holdings_page():
    """Full Holdings page with 3 strategy portfolios."""
    st.markdown(
        f"<div style='padding:16px 0 8px 0;'>"
        f"<h1 style='color:{CYAN};letter-spacing:0.12em;margin:0;'>HOLDINGS</h1>"
        f"<p style='color:{DIM};font-size:0.7rem;letter-spacing:0.08em;'>"
        f"Tre portföljer · Live signaler från respektive strategi · Lägg till/ta bort innehav</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Summary KPI row
    swing_h = _get_holdings("swing")
    ovtlyr_h = _get_holdings("ovtlyr")
    long_h = _get_holdings("long")
    total = len(swing_h) + len(ovtlyr_h) + len(long_h)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Totalt innehav", total)
    k2.metric("Swing", f"{len(swing_h)}/5")
    k3.metric("OVTLYR", f"{len(ovtlyr_h)}/5")
    k4.metric("Long", f"{len(long_h)}/10")

    st.markdown("<hr style='border-color:rgba(0,255,255,0.1);margin:12px 0;'/>", unsafe_allow_html=True)

    # Three portfolios
    _render_portfolio("swing")
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    _render_portfolio("ovtlyr")
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    _render_portfolio("long")

    # ── Correlation Matrix ────────────────────────────────────────────
    all_holdings = []
    for h in swing_h:
        all_holdings.append({"ticker": h["ticker"], "strategy": "swing"})
    for h in ovtlyr_h:
        all_holdings.append({"ticker": h["ticker"], "strategy": "ovtlyr"})
    for h in long_h:
        all_holdings.append({"ticker": h["ticker"], "strategy": "long"})

    st.markdown("<hr style='border-color:rgba(0,255,255,0.15);margin:24px 0;'/>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{CYAN};font-size:0.85rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:12px;'>PORTFÖLJKORRELATION</div>", unsafe_allow_html=True)
    _render_correlation_matrix(all_holdings)
