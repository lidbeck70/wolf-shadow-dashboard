"""
holdings.py — Portfolio Holdings Manager

Multi-strategy portfolio with regime integration.
Strategies: Quality, Deep Contrarian, Viking, Wolf, Untagged.
Persistent storage via GitHub Gist (backward compatible — existing holdings
load as Untagged if they have no 'strategy' field).
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Theme (Nordic Arc) ────────────────────────────────────────────────────────
BG      = "#0c0c12"
BG2     = "#14141e"
CYAN    = "#c9a84c"   # legacy gold (kept for backward compat)
MAGENTA = "#8b7340"
GREEN   = "#2d8a4e"
RED     = "#c44545"
YELLOW  = "#d4943a"
TEXT    = "#e8e4dc"
DIM     = "#8a8578"

# ── Strategy constants ────────────────────────────────────────────────────────
STRATEGY_TAGS = ["Quality", "Deep Contrarian", "Viking", "Wolf", "Untagged"]

STRATEGY_COLORS: dict = {
    "Quality":         "#00E5FF",   # Arc Cyan  (PALETTE["gold"])
    "Deep Contrarian": "#FF6B3D",   # Ember     (PALETTE["amber"])
    "Viking":          "#B400FF",   # Aurora Purple (PALETTE["silver"])
    "Wolf":            "#2d8a4e",   # Green
    "Untagged":        "#6B7280",   # Text dim
}

# Portfolio_key used when adding a new holding, by strategy
_STRATEGY_TO_KEY: dict = {
    "Quality":         "long",
    "Deep Contrarian": "long",
    "Viking":          "ovtlyr",
    "Wolf":            "swing",
    "Untagged":        "long",
}

# ── Portfolio config ──────────────────────────────────────────────────────────
# "long" max raised to 20 to accommodate Quality + Deep Contrarian + Untagged
PORTFOLIOS: dict = {
    "swing":  {"name": "Wolf Portfolio",   "max": 5,  "color": CYAN,    "strategy": "Wolf"},
    "ovtlyr": {"name": "Viking Portfolio", "max": 5,  "color": MAGENTA, "strategy": "Viking"},
    "long":   {"name": "Alpha Portfolio",  "max": 20, "color": GREEN,   "strategy": "Alpha"},
}

# ── Signal imports (with fallbacks) ──────────────────────────────────────────
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


# ── Persistent storage ────────────────────────────────────────────────────────
import json
import os

try:
    from gist_storage import load_holdings as _gist_load, save_holdings as _gist_save
    _HAS_GIST = True
except ImportError:
    _HAS_GIST = False

_HOLDINGS_DIR = os.path.dirname(os.path.abspath(__file__))
_HOLDINGS_FILE = os.path.join(_HOLDINGS_DIR, ".holdings_data.json")


def _load_all() -> dict:
    """Load all portfolios via gist_storage (preferred) or local file."""
    if _HAS_GIST:
        return _gist_load()
    if "holdings_data" in st.session_state:
        return st.session_state["holdings_data"]
    try:
        if os.path.exists(_HOLDINGS_FILE):
            with open(_HOLDINGS_FILE, "r") as f:
                data = json.load(f)
                data.setdefault("cash", 0)
                st.session_state["holdings_data"] = data
                return data
    except Exception:
        pass
    data = {"swing": [], "ovtlyr": [], "long": [], "cash": 0}
    st.session_state["holdings_data"] = data
    return data


def _save_all(all_holdings: dict) -> None:
    """Save all portfolios via gist_storage (preferred) or local file."""
    if _HAS_GIST:
        _gist_save(all_holdings)
    else:
        st.session_state["holdings_data"] = all_holdings
        try:
            with open(_HOLDINGS_FILE, "w") as f:
                json.dump(all_holdings, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save holdings: %s", e)


def _get_holdings(portfolio_key: str) -> List[dict]:
    key = f"holdings_{portfolio_key}"
    if key not in st.session_state:
        all_data = _load_all()
        st.session_state[key] = all_data.get(portfolio_key, [])
    return st.session_state[key]


def _set_holdings(portfolio_key: str, holdings: List[dict]) -> None:
    st.session_state[f"holdings_{portfolio_key}"] = holdings
    all_data = _load_all()
    all_data[portfolio_key] = holdings
    _save_all(all_data)


def _strategy_of(h: dict) -> str:
    """Return the strategy tag for a holding; defaults to 'Untagged'."""
    return h.get("strategy", "Untagged")


def _get_all_holdings_flat() -> List[dict]:
    """Merge all portfolio keys into a flat list, each enriched with '_portfolio_key'."""
    all_data = _load_all()
    result: List[dict] = []
    for key in ("swing", "ovtlyr", "long"):
        for h in all_data.get(key, []):
            result.append({**h, "_portfolio_key": key})
    return result


def _add_holding(
    portfolio_key: str,
    ticker: str,
    entry_price: float = 0,
    sector: str = "Unknown",
    shares: int = 0,
    strategy: str = "Untagged",
    tranches_deployed: int = 0,
) -> bool:
    holdings = _get_holdings(portfolio_key)
    cfg = PORTFOLIOS[portfolio_key]

    if len(holdings) >= cfg["max"]:
        st.warning(f"Max {cfg['max']} holdings in {cfg['name']}. Remove one first.")
        return False

    if any(h["ticker"] == ticker.upper() for h in holdings):
        st.warning(f"{ticker.upper()} already in portfolio.")
        return False

    from datetime import datetime
    entry: dict = {
        "ticker":      ticker.upper(),
        "entry_price": entry_price,
        "shares":      shares,
        "sector":      sector,
        "added":       datetime.now().strftime("%Y-%m-%d"),
        "strategy":    strategy,
    }
    if strategy == "Deep Contrarian":
        entry["tranches_deployed"] = max(0, min(3, tranches_deployed))

    holdings.append(entry)
    _set_holdings(portfolio_key, holdings)
    return True


def _remove_holding(portfolio_key: str, ticker: str) -> None:
    holdings = _get_holdings(portfolio_key)
    holdings = [h for h in holdings if h["ticker"] != ticker.upper()]
    _set_holdings(portfolio_key, holdings)


def _edit_holding_field(portfolio_key: str, ticker: str, field: str, value) -> None:
    """Update a single field on a holding in place and persist."""
    holdings = _get_holdings(portfolio_key)
    for h in holdings:
        if h["ticker"] == ticker.upper():
            h[field] = value
            break
    _set_holdings(portfolio_key, holdings)


# ── Live data & signals ───────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_live_price(ticker: str, period: str = "6mo") -> dict:
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, auto_adjust=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
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
    if not _OVTLYR_OK or df is None or df.empty or len(df) < 50:
        return {"signal": "N/A", "color": DIM}
    try:
        trend = compute_trend(df)

        def _last(val, default=0.0):
            if val is None:
                return default
            if isinstance(val, pd.Series):
                return float(val.iloc[-1]) if len(val) > 0 else default
            return float(val)

        regime = trend.get("regime_color", "orange")
        price  = _last(trend.get("price", df["Close"].iloc[-1]))
        ema50  = _last(trend.get("ema50"))

        if regime == "green" and price > ema50:
            return {"signal": "HOLD",  "color": GREEN,  "detail": "Trend intact"}
        elif regime == "green":
            return {"signal": "HOLD",  "color": YELLOW, "detail": "Below EMA50"}
        elif regime == "orange":
            return {"signal": "WATCH", "color": YELLOW, "detail": "Orange regime"}
        else:
            return {"signal": "EXIT",  "color": RED,    "detail": "Red regime"}
    except Exception:
        return {"signal": "N/A", "color": DIM}


def _compute_ovtlyr_signal(df: pd.DataFrame) -> dict:
    if not _OVTLYR_OK or df is None or df.empty or len(df) < 50:
        return {"signal": "N/A", "color": DIM}
    try:
        trend      = compute_trend(df)
        momentum   = compute_momentum(df)
        volatility = compute_volatility(df)
        obs        = detect_orderblocks(df)
        cur_price  = float(df["Close"].iloc[-1])
        ob_analysis = classify_price_vs_ob(cur_price, obs) if obs else {"signal_bias": "HOLD"}

        def _last(val, default=0.0):
            if val is None:
                return default
            if isinstance(val, pd.Series):
                return float(val.iloc[-1]) if len(val) > 0 else default
            return float(val)

        trend_scalar = {
            "price":       _last(trend.get("price", cur_price)),
            "ema10":       _last(trend.get("ema10", df["Close"].ewm(span=10).mean().iloc[-1])),
            "ema20":       _last(trend.get("ema20", df["Close"].ewm(span=20).mean().iloc[-1])),
            "ema50":       _last(trend.get("ema50")),
            "ema200":      _last(trend.get("ema200")),
            "regime_color": trend.get("regime_color", "orange"),
            "trend_state":  trend.get("trend_state", "neutral"),
        }
        sentiment = {"score": 50, "label": "Neutral"}
        lt = compute_longterm_signal(trend_scalar, sentiment, volatility, ob_analysis, True)
        signal = lt.get("signal", "HOLD")

        if signal == "BUY":
            return {"signal": "HOLD ✓", "color": GREEN,  "detail": f"NINE: {lt.get('ovtlyr_nine', 0)}"}
        elif signal == "SELL":
            return {"signal": "EXIT",   "color": RED,    "detail": f"NINE: {lt.get('ovtlyr_nine', 0)}"}
        elif signal == "REDUCE":
            return {"signal": "REDUCE", "color": MAGENTA,"detail": f"NINE: {lt.get('ovtlyr_nine', 0)}"}
        else:
            return {"signal": "HOLD",   "color": YELLOW, "detail": f"NINE: {lt.get('ovtlyr_nine', 0)}"}
    except Exception as e:
        return {"signal": "N/A", "color": DIM, "detail": str(e)[:40]}


def _compute_long_signal(df: pd.DataFrame, sector: str = "Unknown") -> dict:
    if df is None or df.empty or len(df) < 200:
        return {"signal": "N/A", "color": DIM}
    try:
        close      = df["Close"].astype(float)
        price      = float(close.iloc[-1])
        ema50      = float(close.ewm(span=50).mean().iloc[-1])
        ema200     = float(close.ewm(span=200).mean().iloc[-1])
        ema200_ser = close.ewm(span=200).mean()
        days_below = 0
        for i in range(len(close) - 1, max(len(close) - 10, 0), -1):
            if float(close.iloc[i]) < float(ema200_ser.iloc[i]):
                days_below += 1
            else:
                break
        if price > ema200 and ema50 > ema200:
            return {"signal": "HOLD ✓", "color": GREEN,  "detail": f"Price {(price/ema200-1)*100:+.1f}% vs EMA200"}
        elif price > ema200:
            return {"signal": "WATCH",  "color": YELLOW, "detail": "EMA50 below EMA200"}
        elif days_below >= 3:
            return {"signal": "REDUCE", "color": MAGENTA,"detail": f"{days_below}d below EMA200"}
        elif price < ema200:
            return {"signal": "EXIT",   "color": RED,    "detail": "Below EMA200"}
        else:
            return {"signal": "HOLD",   "color": YELLOW, "detail": "Neutral"}
    except Exception:
        return {"signal": "N/A", "color": DIM}


def _signal_for_holding(holding: dict, df) -> dict:
    """Route to the correct signal function based on portfolio_key."""
    pkey = holding.get("_portfolio_key", "long")
    if pkey == "swing":
        return _compute_swing_signal(df)
    elif pkey == "ovtlyr":
        return _compute_ovtlyr_signal(df)
    else:
        return _compute_long_signal(df, holding.get("sector", "Unknown"))


# ── Alpha Regime integration (cached 1h) ─────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _get_regime_signal(ticker: str, strategy_tag: str) -> dict:
    """
    Fetch Alpha Regime signal for a holding. Cached 1h.
    Returns type-tagged dict: quality | contrarian | n/a | unavailable | error.
    """
    if strategy_tag not in ("Quality", "Deep Contrarian"):
        return {"type": "n/a"}
    try:
        from alpha_regime.engine import run_regime_analysis
        mode = "quality" if strategy_tag == "Quality" else "contrarian"
        r = run_regime_analysis(ticker=ticker, mode=mode)
        if r.error:
            return {"type": "error", "error": r.error[:40]}
        if mode == "quality":
            available = [s for s in r.signals if s.label != "NO DATA"]
            return {
                "type":       "quality",
                "verdict":    r.quality_verdict,
                "passed":     r.signals_passed,
                "total":      len(available),
                "phase":      r.market_phase,
                "confidence": r.market_confidence,
            }
        else:
            c = r.contrarian
            return {
                "type":  "contrarian",
                "stage": c.stage if c else "HOLD",
                "label": c.label if c else "HOLD",
                "color": c.color if c else "#607080",
                "phase": r.market_phase,
            }
    except ImportError:
        return {"type": "unavailable"}
    except Exception as exc:
        logger.debug("_get_regime_signal(%s, %s): %s", ticker, strategy_tag, exc)
        return {"type": "error", "error": str(exc)[:40]}


# ── UI Components ─────────────────────────────────────────────────────────────

def _holding_card(
    holding: dict,
    live: dict,
    signal: dict,
    portfolio_key: str,
    regime_signal: Optional[dict] = None,
) -> None:
    """Render one holding row with optional regime signal chip."""
    try:
        ticker   = holding["ticker"]
        entry    = holding.get("entry_price", 0)
        shares   = holding.get("shares", 0)
        price    = live.get("price", 0)
        change   = live.get("change_1d", 0)
        sig      = signal.get("signal", "N/A")
        sig_c    = signal.get("color", DIM)
        detail   = signal.get("detail", "")
        strategy = _strategy_of(holding)
        strat_c  = STRATEGY_COLORS.get(strategy, DIM)
        tranches = holding.get("tranches_deployed", 0)

        # P&L
        if entry > 0 and price > 0:
            pnl       = (price / entry - 1) * 100
            pnl_color = GREEN if pnl > 0 else RED
            pnl_text  = f"{pnl:+.1f}%"
        else:
            pnl = 0
            pnl_color = DIM
            pnl_text  = "—"

        if shares > 0 and entry > 0 and price > 0:
            abs_pnl      = (price - entry) * shares
            abs_pnl_text = f"{abs_pnl:+,.0f}"
            pos_text     = f"{shares * price:,.0f}"
        else:
            abs_pnl_text = ""
            pos_text     = "—"

        change_c = GREEN if change > 0 else RED

        # Regime chip HTML
        regime_html = ""
        if regime_signal:
            rtype = regime_signal.get("type", "n/a")
            if rtype == "quality":
                verdict = regime_signal.get("verdict", "N/A")
                vc = {"BUY": "#1aaa5a", "WATCH": "#e8a020", "WAIT": "#607080"}.get(verdict, DIM)
                passed = regime_signal.get("passed", "?")
                total  = regime_signal.get("total", "?")
                regime_html = (
                    f'<span style="color:{vc};font-size:0.8rem;font-weight:700;">{verdict}</span>'
                    f'<div style="color:{DIM};font-size:0.6rem;">{passed}/{total} signals</div>'
                )
            elif rtype == "contrarian":
                rc    = regime_signal.get("color", DIM)
                stage = regime_signal.get("stage", "HOLD")
                short = stage.replace("ACCUMULATE_", "ACC·").replace("DISTRIBUTE_", "DIST·")
                dots  = "".join([
                    f'<span style="color:{STRATEGY_COLORS["Deep Contrarian"]}">●</span>'
                    if i < tranches else f'<span style="color:{DIM}">○</span>'
                    for i in range(3)
                ])
                regime_html = (
                    f'<span style="color:{rc};font-size:0.78rem;font-weight:700;">{short}</span>'
                    f'<div style="font-size:0.8rem;">{dots}</div>'
                )
            elif rtype == "error":
                regime_html = f'<span style="color:{DIM};font-size:0.68rem;">ERR</span>'
            elif rtype == "unavailable":
                regime_html = f'<span style="color:{DIM};font-size:0.68rem;">N/A</span>'

        # Columns: TICKER | PRICE | VALUE | P&L | SIGNAL | REGIME | REMOVE
        c1, c2, c3, c4, c5, c6, c7 = st.columns([1.8, 1.1, 0.9, 1.0, 1.2, 1.2, 0.5])

        with c1:
            shares_lbl = f" · {shares} st" if shares > 0 else ""
            st.markdown(
                f"<div style='padding:2px 0;'>"
                f"<span style='color:{TEXT};font-size:1rem;font-weight:700;'>{ticker}</span>"
                f"<span style='color:{DIM};font-size:0.7rem;margin-left:6px;'>"
                f"{holding.get('sector', '')}{shares_lbl}</span>"
                f"<br><span style='color:{strat_c};font-size:0.64rem;font-weight:700;"
                f"letter-spacing:0.04em;'>{strategy.upper()}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div style='text-align:right;padding:2px 0;'>"
                f"<span style='color:{TEXT};font-size:0.9rem;'>{price:.2f}</span>"
                f" <span style='color:{change_c};font-size:0.75rem;'>({change:+.1f}%)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"<div style='text-align:center;padding:2px 0;'>"
                f"<span style='color:{TEXT};font-size:0.85rem;'>{pos_text}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with c4:
            extra = (
                f"<div style='color:{pnl_color};font-size:0.65rem;'>{abs_pnl_text}</div>"
                if abs_pnl_text else ""
            )
            st.markdown(
                f"<div style='text-align:center;padding:2px 0;'>"
                f"<span style='color:{pnl_color};font-size:0.95rem;font-weight:700;'>"
                f"{pnl_text}</span>{extra}</div>",
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                f"<div style='text-align:center;padding:2px 0;'>"
                f"<span style='color:{sig_c};font-size:0.85rem;font-weight:700;'>{sig}</span>"
                f"<div style='color:{DIM};font-size:0.6rem;'>{detail}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with c6:
            if regime_html:
                st.markdown(
                    f"<div style='text-align:center;padding:2px 0;'>{regime_html}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<span style='color:{DIM};font-size:0.7rem;'>—</span></div>",
                    unsafe_allow_html=True,
                )
        with c7:
            if st.button("✕", key=f"rm_{portfolio_key}_{ticker}", help=f"Remove {ticker}"):
                _remove_holding(portfolio_key, ticker)
                st.rerun()

        # Inline edit expander (strategy tag + tranches)
        with st.expander(f"⚙ Edit {ticker}", expanded=False):
            e1, e2, e3 = st.columns([2, 1.5, 0.8])
            with e1:
                idx = STRATEGY_TAGS.index(strategy) if strategy in STRATEGY_TAGS else 4
                new_strat = st.selectbox(
                    "Strategy",
                    STRATEGY_TAGS,
                    index=idx,
                    key=f"edit_strat_{portfolio_key}_{ticker}",
                )
            with e2:
                show_tr = (strategy == "Deep Contrarian") or (new_strat == "Deep Contrarian")
                new_tranches = tranches
                if show_tr:
                    new_tranches = st.selectbox(
                        "Tranches deployed",
                        [0, 1, 2, 3],
                        index=min(tranches, 3),
                        format_func=lambda x: f"{x}/3",
                        key=f"edit_tr_{portfolio_key}_{ticker}",
                    )
            with e3:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                if st.button("Save", key=f"save_edit_{portfolio_key}_{ticker}"):
                    if new_strat != strategy:
                        _edit_holding_field(portfolio_key, ticker, "strategy", new_strat)
                    if new_tranches != tranches:
                        _edit_holding_field(portfolio_key, ticker, "tranches_deployed", new_tranches)
                    st.rerun()

    except Exception as e:
        st.warning(f"Error displaying {holding.get('ticker', '?')}: {e}")


def _render_rule_monitor(strategy_tag: str, strategy_holdings: list) -> None:
    """Rule compliance checks for Quality (position size) and Deep Contrarian (sector)."""
    if not strategy_holdings:
        return

    if strategy_tag == "Quality":
        count = len(strategy_holdings)
        total_q = 0.0
        pos_vals: dict = {}
        for h in strategy_holdings:
            pkey = h.get("_portfolio_key", "long")
            live = _fetch_live_price(h["ticker"], period="2y" if pkey == "long" else "6mo")
            val = live.get("price", 0) * h.get("shares", 0)
            pos_vals[h["ticker"]] = val
            total_q += val

        st.markdown(
            f"<div style='margin-top:6px;padding:5px 0;"
            f"border-top:1px solid rgba(0,229,255,0.1);'>"
            f"<span style='color:{DIM};font-size:0.68rem;letter-spacing:0.08em;'>"
            f"QUALITY RULES</span></div>",
            unsafe_allow_html=True,
        )
        rc1, rc2 = st.columns(2)
        with rc1:
            if count < 8:
                st.warning(f"⚠ {count}/10 holdings — target 8–10")
            elif count > 10:
                st.error(f"🔴 {count} holdings — max is 10")
            else:
                st.success(f"✓ {count}/10 holdings in target range")
        with rc2:
            if total_q > 0:
                oversized = [
                    f"{t} ({v / total_q * 100:.1f}%)"
                    for t, v in pos_vals.items()
                    if v / total_q * 100 > 10
                ]
                if oversized:
                    st.warning(f"⚠ >10% position: {', '.join(oversized)}")
                else:
                    st.success("✓ All positions ≤ 10%")

    elif strategy_tag == "Deep Contrarian":
        sector_vals: dict = {}
        total_c = 0.0
        tranche_counts: dict = {0: 0, 1: 0, 2: 0, 3: 0}

        for h in strategy_holdings:
            pkey = h.get("_portfolio_key", "long")
            live = _fetch_live_price(h["ticker"], period="2y" if pkey == "long" else "6mo")
            val = live.get("price", 0) * h.get("shares", 0)
            sector = h.get("sector", "Unknown")
            sector_vals[sector] = sector_vals.get(sector, 0.0) + val
            total_c += val
            t = min(h.get("tranches_deployed", 0), 3)
            tranche_counts[t] = tranche_counts[t] + 1

        st.markdown(
            f"<div style='margin-top:6px;padding:5px 0;"
            f"border-top:1px solid rgba(255,107,61,0.1);'>"
            f"<span style='color:{DIM};font-size:0.68rem;letter-spacing:0.08em;'>"
            f"CONTRARIAN RULES</span></div>",
            unsafe_allow_html=True,
        )
        rc1, rc2 = st.columns(2)
        with rc1:
            if total_c > 0:
                oversized = [
                    f"{s} ({v / total_c * 100:.1f}%)"
                    for s, v in sector_vals.items()
                    if v / total_c * 100 > 25
                ]
                if oversized:
                    st.warning(f"⚠ Sector >25%: {', '.join(oversized)}")
                else:
                    st.success("✓ All sectors ≤ 25%")
        with rc2:
            st.caption(
                f"Tranches — 0/3: {tranche_counts[0]} · "
                f"1/3: {tranche_counts[1]} · "
                f"2/3: {tranche_counts[2]} · "
                f"3/3: {tranche_counts[3]}"
            )


def _render_strategy_section(strategy_tag: str) -> None:
    """Render one strategy's holdings section: add form, cards, rule monitor."""
    all_flat = _get_all_holdings_flat()
    strategy_holdings = [h for h in all_flat if _strategy_of(h) == strategy_tag]
    color = STRATEGY_COLORS.get(strategy_tag, DIM)
    pkey_for_add = _STRATEGY_TO_KEY.get(strategy_tag, "long")

    # Section header
    st.markdown(
        f"<div style='border-bottom:2px solid {color};"
        f"padding-bottom:5px;margin-bottom:10px;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<h3 style='color:{color};margin:0;letter-spacing:0.08em;font-size:1.05rem;'>"
        f"{strategy_tag.upper()}</h3>"
        f"<span style='color:{DIM};font-size:0.73rem;'>"
        f"{len(strategy_holdings)} holding{'s' if len(strategy_holdings) != 1 else ''}"
        f"</span></div></div>",
        unsafe_allow_html=True,
    )

    # Add holding form
    with st.expander(f"Add {strategy_tag} holding", expanded=len(strategy_holdings) == 0):
        if strategy_tag == "Deep Contrarian":
            ac1, ac2, ac3, ac4, ac5, ac6 = st.columns([2, 1.2, 1, 1.2, 1.2, 0.8])
        else:
            ac1, ac2, ac3, ac4, ac5 = st.columns([2, 1.2, 1, 1.2, 0.8])

        with ac1:
            new_ticker = st.text_input(
                "Ticker", key=f"add_t_{strategy_tag}",
                placeholder="VOLV-B.ST",
            ).strip().upper()
        with ac2:
            new_entry = st.number_input(
                "Entry price", value=0.0, min_value=0.0, step=0.01,
                key=f"add_e_{strategy_tag}",
            )
        with ac3:
            new_shares = st.number_input(
                "Shares", value=0, min_value=0, step=1,
                key=f"add_s_{strategy_tag}",
            )
        with ac4:
            new_sector = st.text_input(
                "Sector", value="Unknown", key=f"add_sec_{strategy_tag}",
            )
        new_tranches = 0
        if strategy_tag == "Deep Contrarian":
            with ac5:
                new_tranches = st.selectbox(
                    "Tranches",
                    [0, 1, 2, 3],
                    format_func=lambda x: f"{x}/3",
                    key=f"add_tr_{strategy_tag}",
                )
            with ac6:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                add_btn = st.button("+ ADD", key=f"add_btn_{strategy_tag}", use_container_width=True)
        else:
            with ac5:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                add_btn = st.button("+ ADD", key=f"add_btn_{strategy_tag}", use_container_width=True)

        if add_btn and new_ticker:
            if _add_holding(
                pkey_for_add, new_ticker, new_entry, new_sector, new_shares,
                strategy=strategy_tag, tranches_deployed=new_tranches,
            ):
                st.rerun()

    if not strategy_holdings:
        st.markdown(
            f"<div style='color:{DIM};text-align:center;padding:14px;"
            f"font-size:0.85rem;'>No {strategy_tag} holdings yet.</div>",
            unsafe_allow_html=True,
        )
        return

    # Regime signal toggle (Quality + Deep Contrarian only)
    show_regime = False
    if strategy_tag in ("Quality", "Deep Contrarian"):
        show_regime = st.checkbox(
            "Show Alpha Regime signals (1h cache — may be slow on first load)",
            key=f"regime_toggle_{strategy_tag}",
            value=False,
        )

    # Column headers
    h1, h2, h3, h4, h5, h6, _ = st.columns([1.8, 1.1, 0.9, 1.0, 1.2, 1.2, 0.5])
    for col, label, align in [
        (h1, "TICKER",     "left"),
        (h2, "PRICE (1D%)", "right"),
        (h3, "VALUE",       "center"),
        (h4, "P&L",         "center"),
        (h5, "SIGNAL",      "center"),
        (h6, "REGIME",      "center"),
    ]:
        col.markdown(
            f"<span style='color:{DIM};font-size:0.68rem;"
            f"text-align:{align};display:block;'>{label}</span>",
            unsafe_allow_html=True,
        )

    # Holdings rows
    total_value  = 0.0
    total_abs_pnl = 0.0
    total_pnl    = 0.0
    pnl_count    = 0

    for holding in strategy_holdings:
        ticker     = holding["ticker"]
        pkey       = holding["_portfolio_key"]
        fetch_p    = "2y" if pkey == "long" else "6mo"
        live       = _fetch_live_price(ticker, period=fetch_p)
        df         = live.get("df")
        signal     = _signal_for_holding(holding, df)
        regime_sig = _get_regime_signal(ticker, strategy_tag) if show_regime else None

        _holding_card(holding, live, signal, pkey, regime_signal=regime_sig)

        price  = live.get("price", 0)
        shares = holding.get("shares", 0)
        entry  = holding.get("entry_price", 0)
        if shares > 0 and price > 0:
            total_value += price * shares
        if entry > 0 and price > 0:
            total_pnl += (price / entry - 1) * 100
            pnl_count += 1
            if shares > 0:
                total_abs_pnl += (price - entry) * shares

    # Section summary footer
    if pnl_count > 0:
        avg_pnl = total_pnl / pnl_count
        pnl_c   = GREEN if avg_pnl >= 0 else RED
        abs_c   = GREEN if total_abs_pnl >= 0 else RED
        val_part = (
            f"<span style='color:{DIM};font-size:0.75rem;'>Value: </span>"
            f"<span style='color:{color};font-size:0.9rem;font-weight:700;'>"
            f"{total_value:,.0f} SEK</span>  "
        ) if total_value > 0 else ""
        abs_part = (
            f"<span style='color:{DIM};font-size:0.75rem;'>P&L: </span>"
            f"<span style='color:{abs_c};font-size:0.9rem;font-weight:700;'>"
            f"{total_abs_pnl:+,.0f} SEK</span>  "
        ) if total_value > 0 else ""
        st.markdown(
            f"<div style='text-align:right;padding:5px 0;"
            f"border-top:1px solid rgba(100,100,100,0.12);margin-top:5px;'>"
            f"{val_part}{abs_part}"
            f"<span style='color:{DIM};font-size:0.75rem;'>Avg P&L: </span>"
            f"<span style='color:{pnl_c};font-size:0.9rem;font-weight:700;'>"
            f"{avg_pnl:+.1f}%</span></div>",
            unsafe_allow_html=True,
        )

    _render_rule_monitor(strategy_tag, strategy_holdings)


def _render_summary_header() -> None:
    """Top-of-page summary: total value, per-strategy allocation bars, cash input."""
    all_flat = _get_all_holdings_flat()
    all_data = _load_all()
    cash = float(all_data.get("cash", 0))

    # Compute per-strategy invested values (uses cached price data)
    strat_vals: dict = {tag: 0.0 for tag in STRATEGY_TAGS}
    total_invested = 0.0
    for h in all_flat:
        shares = h.get("shares", 0)
        if shares <= 0:
            continue
        pkey = h.get("_portfolio_key", "long")
        try:
            live = _fetch_live_price(h["ticker"], period="2y" if pkey == "long" else "6mo")
            val  = live.get("price", 0) * shares
            strat_vals[_strategy_of(h)] = strat_vals.get(_strategy_of(h), 0.0) + val
            total_invested += val
        except Exception:
            pass

    grand_total = total_invested + cash

    # Title + cash input
    title_col, cash_col = st.columns([3, 1])
    with title_col:
        st.markdown(
            f"<h1 style='color:{STRATEGY_COLORS['Quality']};letter-spacing:0.12em;"
            f"margin:0 0 2px 0;'>HOLDINGS</h1>"
            f"<p style='color:{DIM};font-size:0.7rem;letter-spacing:0.08em;margin:0;'>"
            f"Multi-strategy portfolio · Live signals · Alpha Regime integration</p>",
            unsafe_allow_html=True,
        )
    with cash_col:
        new_cash = st.number_input(
            "Cash (SEK)", value=cash, min_value=0.0, step=1000.0,
            key="holdings_cash_input",
        )
        if new_cash != cash:
            all_data["cash"] = new_cash
            _save_all(all_data)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Holdings", len(all_flat))
    k2.metric("Invested", f"{total_invested:,.0f} SEK" if total_invested > 0 else "—")
    k3.metric("Cash", f"{cash:,.0f} SEK" if cash > 0 else "—")
    k4.metric("Portfolio Total", f"{grand_total:,.0f} SEK" if grand_total > 0 else "—")

    # Allocation bars
    if grand_total > 0:
        st.markdown(
            f"<div style='margin:10px 0 3px 0;color:{DIM};"
            f"font-size:0.68rem;letter-spacing:0.08em;'>STRATEGY ALLOCATION</div>",
            unsafe_allow_html=True,
        )
        for tag in STRATEGY_TAGS:
            val   = strat_vals.get(tag, 0.0)
            count = sum(1 for h in all_flat if _strategy_of(h) == tag)
            if val == 0 and count == 0:
                continue
            pct   = val / grand_total * 100 if grand_total > 0 else 0
            color = STRATEGY_COLORS[tag]
            st.markdown(
                f"<div style='display:flex;align-items:center;margin:2px 0;gap:8px;'>"
                f"<span style='color:{color};font-size:0.68rem;font-weight:700;"
                f"width:120px;text-align:right;letter-spacing:0.03em;'>"
                f"{tag.upper()}</span>"
                f"<div style='flex:1;background:rgba(255,255,255,0.05);"
                f"border-radius:2px;height:9px;'>"
                f"<div style='width:{max(pct, 0.3):.1f}%;background:{color};opacity:0.85;"
                f"height:9px;border-radius:2px;'></div></div>"
                f"<span style='color:{DIM};font-size:0.68rem;width:80px;'>"
                f"{val:,.0f} SEK</span>"
                f"<span style='color:{color};font-size:0.68rem;width:38px;'>"
                f"{pct:.1f}%</span>"
                f"<span style='color:{DIM};font-size:0.68rem;'>({count})</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if cash > 0:
            cash_pct = cash / grand_total * 100
            st.markdown(
                f"<div style='display:flex;align-items:center;margin:2px 0;gap:8px;'>"
                f"<span style='color:{DIM};font-size:0.68rem;font-weight:700;"
                f"width:120px;text-align:right;'>CASH</span>"
                f"<div style='flex:1;background:rgba(255,255,255,0.05);"
                f"border-radius:2px;height:9px;'>"
                f"<div style='width:{max(cash_pct, 0.3):.1f}%;background:{DIM};opacity:0.7;"
                f"height:9px;border-radius:2px;'></div></div>"
                f"<span style='color:{DIM};font-size:0.68rem;width:80px;'>"
                f"{cash:,.0f} SEK</span>"
                f"<span style='color:{DIM};font-size:0.68rem;width:38px;'>"
                f"{cash_pct:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<hr style='border-color:rgba(100,100,100,0.15);margin:12px 0;'/>",
        unsafe_allow_html=True,
    )


# ── Correlation Matrix ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_returns_for_correlation(tickers: tuple, period: str = "6mo") -> pd.DataFrame:
    """Fetch daily returns for a list of tickers."""
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
        if len(returns) > 90:
            returns = returns.tail(90)
        return returns
    except Exception:
        return pd.DataFrame()


def _render_correlation_matrix(all_holdings: list) -> None:
    """Render a correlation heatmap for all active holdings."""
    try:
        if len(all_holdings) < 2:
            st.info("Add at least 2 holdings to see correlation.")
            return

        tickers      = [h["ticker"] for h in all_holdings]
        strategy_map = {h["ticker"]: h.get("strategy", h.get("_portfolio_key", "")) for h in all_holdings}

        returns_df = _fetch_returns_for_correlation(tuple(tickers))
        if returns_df.empty or len(returns_df.columns) < 2:
            st.warning("Correlation: insufficient price data.")
            return

        valid_tickers = [t for t in tickers if t in returns_df.columns]
        if len(valid_tickers) < 2:
            st.warning("Correlation: fewer than 2 tickers with data.")
            return

        corr = returns_df[valid_tickers].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=valid_tickers,
            y=valid_tickers,
            colorscale=[
                [0.0, "rgba(45,138,78,0.9)"],
                [0.5, "rgba(255,255,255,0.9)"],
                [1.0, "rgba(196,69,69,0.9)"],
            ],
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont={"size": 11, "color": "rgba(232,228,220,0.9)"},
            hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>Corr: %{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title=dict(text="Corr", font=dict(color=DIM)),
                tickfont=dict(color=DIM),
            ),
        ))
        fig.update_layout(
            title=dict(
                text="PORTFOLIO CORRELATION — 90d daily returns",
                font=dict(color=STRATEGY_COLORS["Quality"], size=13),
            ),
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT),
            xaxis=dict(tickfont=dict(color=CYAN, size=10), gridcolor="rgba(201,168,76,0.1)"),
            yaxis=dict(tickfont=dict(color=CYAN, size=10), gridcolor="rgba(201,168,76,0.1)", autorange="reversed"),
            height=max(350, 60 * len(valid_tickers)),
            margin=dict(l=80, r=40, t=50, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

        warned: set = set()
        for i in range(len(valid_tickers)):
            for j in range(i + 1, len(valid_tickers)):
                c = corr.iloc[i, j]
                pair = (valid_tickers[i], valid_tickers[j])
                if pair in warned:
                    continue
                if c > 0.85:
                    st.error(f"🔴 Extreme correlation: {pair[0]}↔{pair[1]} ({c:.2f})")
                    warned.add(pair)
                elif c > 0.70:
                    st.warning(f"⚠ High correlation: {pair[0]}↔{pair[1]} ({c:.2f})")
                    warned.add(pair)
    except Exception as e:
        st.warning(f"Correlation matrix: {e}")


# ── Main render ───────────────────────────────────────────────────────────────

def render_holdings_page() -> None:
    """Full Holdings page with strategy-grouped sections."""
    _render_summary_header()

    # Cloud storage status
    try:
        from gist_storage import get_storage_status
        status = get_storage_status()
        if status == "cloud_ok":
            st.caption("☁ Cloud storage active — holdings saved permanently")
        elif status == "local_only":
            st.caption("💾 Local storage — add GITHUB_TOKEN to secrets for cloud sync")
        else:
            st.caption(f"Storage: {status}")
    except Exception:
        pass

    # Per-strategy sections
    all_flat = _get_all_holdings_flat()
    for strategy_tag in STRATEGY_TAGS:
        has = any(_strategy_of(h) == strategy_tag for h in all_flat)
        # Always show Quality and Deep Contrarian; others only when they have holdings
        if strategy_tag in ("Quality", "Deep Contrarian") or has:
            _render_strategy_section(strategy_tag)
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # Risk dashboard
    st.markdown("<hr style='border-color:rgba(100,100,100,0.12);margin:20px 0;'/>", unsafe_allow_html=True)
    holdings_data = {
        "swing":  _get_holdings("swing"),
        "ovtlyr": _get_holdings("ovtlyr"),
        "long":   _get_holdings("long"),
    }
    try:
        from risk_dashboard import render_risk_dashboard
        render_risk_dashboard(holdings_data)
    except Exception:
        pass

    # Earnings calendar
    st.markdown("<hr style='border-color:rgba(100,100,100,0.12);margin:20px 0;'/>", unsafe_allow_html=True)
    try:
        from earnings_calendar import render_earnings_calendar
        render_earnings_calendar(holdings_data)
    except Exception:
        pass

    # Correlation matrix
    st.markdown("<hr style='border-color:rgba(100,100,100,0.12);margin:20px 0;'/>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{STRATEGY_COLORS['Quality']};font-size:0.83rem;"
        f"text-transform:uppercase;letter-spacing:0.1em;font-weight:700;"
        f"margin-bottom:10px;'>PORTFOLIO CORRELATION</div>",
        unsafe_allow_html=True,
    )
    _render_correlation_matrix(all_flat)
