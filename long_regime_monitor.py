"""
long_regime_monitor.py
Alpha Regime Monitor — Shows all parameters required for a long position.

Displays the user's 10 long-term rules as live gates with real-time data:
  1. Green regime (OVTLYR NINE / composite score)
  2. Price > EMA 200
  3. EMA 50 > EMA 200
  4. Sector green
  5. Fear & Greed < 60
  6. EMA 200 intact (not broken)
  7. Regime color (not red)
  8. Sector allocation check
  9. Position size check
  10. Drawdown classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Nordic Gold theme
BG = "#0c0c12"
BG2 = "#14141e"
CYAN = "#c9a84c"
MAGENTA = "#8b7340"
GREEN = "#2d8a4e"
RED = "#c44545"
YELLOW = "#d4943a"
TEXT = "#e8e4dc"
DIM = "#8a8578"

# Try imports
try:
    from cagr.cagr_loader import NORDIC_TICKERS, load_etf_tickers
except ImportError:
    NORDIC_TICKERS = {}
    load_etf_tickers = lambda: {}

try:
    from cagr.cagr_cycle import SECTOR_CONFIG, score_label
except ImportError:
    SECTOR_CONFIG = {}
    score_label = lambda x: ("NEUTRAL", YELLOW)

# Ticker universe for market selector
try:
    from ticker_universe import COUNTRY_REGIONS as TU_REGIONS, get_tickers_for_regions
    _TU_AVAILABLE = True
except ImportError:
    TU_REGIONS = {}
    _TU_AVAILABLE = False


# ── Data fetching ─────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_ticker_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, auto_adjust=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_spy_data() -> pd.DataFrame:
    return _fetch_ticker_data("SPY", "1y")


@st.cache_data(ttl=1800, show_spinner=False)
def _compute_fear_greed() -> dict:
    """Simple Fear & Greed from VIX + market breadth."""
    try:
        vix = yf.Ticker("^VIX").history(period="5d", auto_adjust=True)
        vix_val = float(vix["Close"].iloc[-1]) if not vix.empty else 20
        # Invert VIX to score: low VIX = greed, high VIX = fear
        fg_score = max(0, min(100, 100 - vix_val * 3))
        if fg_score > 75:
            label = "Extreme Greed"
        elif fg_score > 55:
            label = "Greed"
        elif fg_score > 45:
            label = "Neutral"
        elif fg_score > 25:
            label = "Fear"
        else:
            label = "Extreme Fear"
        return {"score": round(fg_score, 1), "label": label, "vix": round(vix_val, 1)}
    except Exception:
        return {"score": 50, "label": "Neutral", "vix": 20}


# ── Gate computation ──────────────────────────────────────────────────

def _compute_long_gates(ticker: str, sector: str) -> dict:
    """Compute all 10 long-term rule gates for a ticker."""
    df = _fetch_ticker_data(ticker)
    spy = _fetch_spy_data()
    fg = _compute_fear_greed()

    gates = []
    details = {}

    if df.empty or len(df) < 200:
        return {"gates": [], "details": {}, "error": "Insufficient data"}

    close = df["Close"].astype(float)
    price = float(close.iloc[-1])
    ema50 = float(close.ewm(span=50).mean().iloc[-1])
    ema200 = float(close.ewm(span=200).mean().iloc[-1])
    ema10 = float(close.ewm(span=10).mean().iloc[-1])
    ema20 = float(close.ewm(span=20).mean().iloc[-1])

    # EMA200 trend (rising?)
    ema200_series = close.ewm(span=200).mean()
    ema200_rising = float(ema200_series.iloc[-1]) > float(ema200_series.iloc[-20])

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])

    # ATR for position sizing
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])

    # SPY regime
    spy_regime = "green"
    if not spy.empty and len(spy) >= 50:
        spy_close = spy["Close"].astype(float)
        spy_price = float(spy_close.iloc[-1])
        spy_ema20 = float(spy_close.ewm(span=20).mean().iloc[-1])
        spy_ema50 = float(spy_close.ewm(span=50).mean().iloc[-1])
        spy_ema200 = float(spy_close.ewm(span=200).mean().iloc[-1])
        if spy_price < spy_ema200:
            spy_regime = "red"
        elif spy_price < spy_ema50:
            spy_regime = "orange"
    else:
        spy_regime = "orange"

    # Sector assessment
    cfg = SECTOR_CONFIG.get(sector, SECTOR_CONFIG.get("Unknown", {}))
    sector_score = cfg.get("default_score", 1)
    sector_green = sector_score >= 2

    # Regime color
    if price > ema200 and ema50 > ema200 and spy_regime == "green":
        regime_color = "green"
    elif price > ema200:
        regime_color = "orange"
    else:
        regime_color = "red"

    # Drawdown from peak
    peak = close.cummax()
    current_dd = float((price - peak.iloc[-1]) / peak.iloc[-1] * 100)

    details = {
        "price": price,
        "ema10": ema10,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "ema200_rising": ema200_rising,
        "rsi": rsi,
        "atr": atr,
        "spy_regime": spy_regime,
        "sector_score": sector_score,
        "sector_green": sector_green,
        "regime_color": regime_color,
        "fear_greed": fg["score"],
        "fear_greed_label": fg["label"],
        "vix": fg["vix"],
        "drawdown": current_dd,
    }

    # ── 10 Gates ──────────────────────────────────────────────────────

    # 1. Green regime
    gates.append({
        "rule": "1. Köp i grön regim",
        "passed": regime_color == "green",
        "value": regime_color.upper(),
        "detail": f"SPY: {spy_regime} | Stock: {'Bull' if price > ema200 else 'Bear'}",
    })

    # 2. Price > EMA200
    gates.append({
        "rule": "2. Pris > 200 EMA",
        "passed": price > ema200,
        "value": f"{price:.2f} vs {ema200:.2f}",
        "detail": f"{'Ovanför' if price > ema200 else 'Under'} med {abs(price - ema200):.2f} ({abs(price/ema200*100-100):.1f}%)",
    })

    # 3. EMA50 > EMA200
    gates.append({
        "rule": "3. 50 EMA > 200 EMA",
        "passed": ema50 > ema200,
        "value": f"{ema50:.2f} vs {ema200:.2f}",
        "detail": "Golden Cross" if ema50 > ema200 else "Death Cross",
    })

    # 4. Sektorn grön
    gates.append({
        "rule": "4. Sektorn grön",
        "passed": sector_green,
        "value": f"{sector} → {sector_score}/3",
        "detail": score_label(sector_score)[0],
    })

    # 5. Fear & Greed < 60
    gates.append({
        "rule": "5. Fear & Greed < 60",
        "passed": fg["score"] < 60,
        "value": f"{fg['score']:.0f} ({fg['label']})",
        "detail": f"VIX: {fg['vix']:.1f}",
    })

    # 6. EMA200 intakt (ej bruten)
    days_below_200 = 0
    for i in range(len(close) - 1, max(len(close) - 20, 0), -1):
        if float(close.iloc[i]) < float(ema200_series.iloc[i]):
            days_below_200 += 1
        else:
            break
    gates.append({
        "rule": "6. EMA200 ej bruten",
        "passed": days_below_200 < 3,
        "value": f"{days_below_200} dagar under",
        "detail": "OK" if days_below_200 < 3 else f"VARNING: {days_below_200} dagar under EMA200",
    })

    # 7. Regim ej röd
    gates.append({
        "rule": "7. Regim ej röd",
        "passed": regime_color != "red",
        "value": regime_color.upper(),
        "detail": f"EMA200 {'stigande' if ema200_rising else 'fallande'}",
    })

    # 8. Max 25% per sektor (informational)
    gates.append({
        "rule": "8. Max 25% per sektor",
        "passed": True,  # Can't check portfolio level here
        "value": "Portföljnivå",
        "detail": "Kontrollera manuellt mot din portfölj",
    })

    # 9. Max 10% per aktie (informational)
    gates.append({
        "rule": "9. Max 10% per aktie",
        "passed": True,
        "value": "Portföljnivå",
        "detail": f"Position size: Kapital × 1% / (½ ATR = {atr/2:.2f})",
    })

    # 10. Historisk nedgång — brus eller strukturell?
    dd_classification = "Brus"
    if current_dd < -20:
        dd_classification = "Betydande"
    elif current_dd < -10:
        dd_classification = "Moderat"
    gates.append({
        "rule": "10. Analysera nedgångar",
        "passed": current_dd > -15,
        "value": f"{current_dd:.1f}% från topp",
        "detail": dd_classification,
    })

    return {"gates": gates, "details": details}


# ── UI rendering ──────────────────────────────────────────────────────

def _gate_card(gate: dict) -> str:
    passed = gate["passed"]
    color = GREEN if passed else RED
    icon = "✓" if passed else "✗"
    return (
        f"<div style='background:{BG2};border-left:4px solid {color};"
        f"padding:10px 14px;margin:6px 0;border-radius:4px;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<span style='color:{color};font-size:1.1rem;font-weight:700;'>{icon}</span>"
        f"<span style='color:{TEXT};font-size:0.85rem;flex:1;margin-left:10px;'>{gate['rule']}</span>"
        f"<span style='color:{color};font-size:0.82rem;font-weight:600;'>{gate['value']}</span>"
        f"</div>"
        f"<div style='color:{DIM};font-size:0.68rem;margin-top:3px;margin-left:28px;'>{gate['detail']}</div>"
        f"</div>"
    )


def _regime_badge(color: str) -> str:
    c = {"green": GREEN, "orange": YELLOW, "red": RED}.get(color, DIM)
    label = color.upper()
    return (
        f"<div style='display:inline-block;padding:8px 24px;border-radius:6px;"
        f"background:{c};color:{BG};font-size:1.2rem;font-weight:800;"
        f"letter-spacing:0.1em;text-align:center;'>{label}</div>"
    )


def render_long_regime_monitor():
    """Full Alpha Regime Monitor page."""

    st.markdown(
        f"<h2 style='color:{GREEN};letter-spacing:0.1em;'>"
        f"ALPHA REGIME MONITOR</h2>"
        f"<p style='color:{DIM};font-size:0.7rem;letter-spacing:0.08em;'>"
        f"Alla 10 regler — live status — visar om du får ta en lång position</p>",
        unsafe_allow_html=True,
    )

    # Market selector
    try:
        if _TU_AVAILABLE and TU_REGIONS:
            region_options = list(TU_REGIONS.keys())
            selected_regions = st.multiselect(
                "Marknader",
                region_options,
                default=["Norden"],
                key="alpha_regime_markets",
            )
            st.caption(f"Vald marknad: {', '.join(selected_regions) if selected_regions else 'Ingen'}")
    except Exception:
        pass

    # Ticker selector
    all_tickers = {**NORDIC_TICKERS, **load_etf_tickers()}
    ticker_options = {
        f"{v.get('name', k)} ({k})": k
        for k, v in sorted(all_tickers.items(), key=lambda x: x[1].get("name", x[0]))
    }

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected = st.selectbox(
            "Aktie / ETF",
            list(ticker_options.keys()),
            key="long_regime_ticker",
        )
        ticker = ticker_options.get(selected, "VOLV-B.ST")
    with col2:
        sector_override = st.selectbox(
            "Sektor",
            list(SECTOR_CONFIG.keys()),
            index=list(SECTOR_CONFIG.keys()).index("Industrials") if "Industrials" in SECTOR_CONFIG else 0,
            key="long_regime_sector",
        )
    with col3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        refresh = st.button("↻ ANALYSERA", key="long_regime_refresh", use_container_width=True)

    # ── Benchmark Selector ─────────────────────────────────────────────
    _BENCHMARK_OPTIONS = {
        "SPY — S&P 500": "SPY",
        "OMX Nordic 40": "^OMXN40",
        "Brent Crude": "BZ=F",
        "Guld": "GC=F",
        "Silver": "SI=F",
    }
    selected_bm_label = st.selectbox(
        "Benchmark",
        list(_BENCHMARK_OPTIONS.keys()),
        index=0,
        key="benchmark_long",
    )
    benchmark_ticker = _BENCHMARK_OPTIONS[selected_bm_label]

    # Relative Strength indicator
    try:
        bm_data = yf.Ticker(benchmark_ticker).history(period="3mo", auto_adjust=True)
        stk_data_rs = yf.Ticker(ticker).history(period="3mo", auto_adjust=True)
        if not bm_data.empty and not stk_data_rs.empty and len(bm_data) >= 20 and len(stk_data_rs) >= 20:
            stk_ret = float(stk_data_rs["Close"].iloc[-1] / stk_data_rs["Close"].iloc[-20])
            bm_ret = float(bm_data["Close"].iloc[-1] / bm_data["Close"].iloc[-20])
            if bm_ret > 0:
                rs = stk_ret / bm_ret
                if rs > 1.05:
                    rs_icon, rs_color = "🟢", GREEN
                elif rs >= 0.95:
                    rs_icon, rs_color = "🟡", YELLOW
                else:
                    rs_icon, rs_color = "🔴", RED
                st.markdown(
                    f"<div style='padding:4px 0;'>"
                    f"<span style='font-size:0.85rem;'>{rs_icon}</span> "
                    f"<span style='color:{rs_color};font-size:0.85rem;font-weight:700;'>"
                    f"RS: {rs - 1:+.1%} vs {selected_bm_label}</span></div>",
                    unsafe_allow_html=True,
                )
    except Exception:
        pass

    # Auto-detect sector from ticker metadata
    meta = all_tickers.get(ticker, {})
    sector = meta.get("sector", sector_override)

    # Compute gates
    with st.spinner("Analyserar..."):
        result = _compute_long_gates(ticker, sector)

    if "error" in result:
        st.error(result["error"])
        return

    gates = result["gates"]
    details = result["details"]

    # ── Top row: Regime badge + key metrics ─────────────────────────
    top1, top2, top3, top4, top5 = st.columns(5)

    with top1:
        st.markdown(_regime_badge(details["regime_color"]), unsafe_allow_html=True)
        st.caption("Regim")

    with top2:
        passed_count = sum(1 for g in gates if g["passed"])
        total_color = GREEN if passed_count >= 8 else (YELLOW if passed_count >= 5 else RED)
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<span style='color:{total_color};font-size:2.2rem;font-weight:800;'>"
            f"{passed_count}/10</span></div>",
            unsafe_allow_html=True,
        )
        st.caption("Gates Passed")

    with top3:
        price = details["price"]
        ema200 = details["ema200"]
        dist = (price / ema200 - 1) * 100
        dist_color = GREEN if dist > 0 else RED
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<span style='color:{dist_color};font-size:1.5rem;font-weight:700;'>"
            f"{dist:+.1f}%</span></div>",
            unsafe_allow_html=True,
        )
        st.caption("vs EMA200")

    with top4:
        fg = details["fear_greed"]
        fg_color = GREEN if fg < 60 else RED
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<span style='color:{fg_color};font-size:1.5rem;font-weight:700;'>"
            f"{fg:.0f}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(f"F&G ({details['fear_greed_label']})")

    with top5:
        dd = details["drawdown"]
        dd_color = GREEN if dd > -10 else (YELLOW if dd > -20 else RED)
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<span style='color:{dd_color};font-size:1.5rem;font-weight:700;'>"
            f"{dd:.1f}%</span></div>",
            unsafe_allow_html=True,
        )
        st.caption("Drawdown")

    st.markdown("<hr style='border-color:rgba(201,168,76,0.1);margin:16px 0;'/>", unsafe_allow_html=True)

    # ── Verdict ─────────────────────────────────────────────────────
    all_pass = all(g["passed"] for g in gates[:7])  # First 7 are hard gates
    if all_pass:
        verdict = "ALLA GATES PASSERAR — OK ATT TA LÅNG POSITION"
        verdict_color = GREEN
    elif passed_count >= 5:
        verdict = "DELVIS — VÄNTA PÅ FLER GATES"
        verdict_color = YELLOW
    else:
        verdict = "BLOCKERAD — TA INTE LÅNG POSITION"
        verdict_color = RED

    st.markdown(
        f"<div style='text-align:center;padding:12px;margin:8px 0;"
        f"border:2px solid {verdict_color};border-radius:8px;'>"
        f"<span style='color:{verdict_color};font-size:1rem;font-weight:700;"
        f"letter-spacing:0.1em;'>{verdict}</span></div>",
        unsafe_allow_html=True,
    )

    # ── Gates list ──────────────────────────────────────────────────
    st.markdown(
        f"<div style='color:{CYAN};font-size:0.7rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;margin:16px 0 8px 0;'>DINA 10 REGLER — LIVE STATUS</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        for gate in gates[:5]:
            st.markdown(_gate_card(gate), unsafe_allow_html=True)
    with right:
        for gate in gates[5:]:
            st.markdown(_gate_card(gate), unsafe_allow_html=True)

    # ── EMA Levels table ────────────────────────────────────────────
    st.markdown(
        f"<div style='color:{CYAN};font-size:0.7rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;margin:20px 0 8px 0;'>EMA NIVÅER</div>",
        unsafe_allow_html=True,
    )

    ema_data = pd.DataFrame([
        {"Indikator": "Pris", "Värde": f"{details['price']:.2f}", "Status": "—"},
        {"Indikator": "EMA 10", "Värde": f"{details['ema10']:.2f}",
         "Status": "Ovanför" if details["price"] > details["ema10"] else "Under"},
        {"Indikator": "EMA 20", "Värde": f"{details['ema20']:.2f}",
         "Status": "Ovanför" if details["price"] > details["ema20"] else "Under"},
        {"Indikator": "EMA 50", "Värde": f"{details['ema50']:.2f}",
         "Status": "Ovanför" if details["price"] > details["ema50"] else "Under"},
        {"Indikator": "EMA 200", "Värde": f"{details['ema200']:.2f}",
         "Status": "Ovanför" if details["price"] > details["ema200"] else "Under"},
        {"Indikator": "ATR (14)", "Värde": f"{details['atr']:.2f}", "Status": "—"},
        {"Indikator": "RSI (14)", "Värde": f"{details['rsi']:.1f}",
         "Status": "OB" if details["rsi"] > 70 else ("OS" if details["rsi"] < 30 else "Neutral")},
        {"Indikator": "VIX", "Värde": f"{details['vix']:.1f}", "Status": "—"},
    ])

    def _status_color(val):
        if val == "Ovanför":
            return f"color:{GREEN}"
        elif val == "Under":
            return f"color:{RED}"
        return f"color:{DIM}"

    styled = ema_data.style
    _map = styled.map if hasattr(styled, "map") else styled.applymap
    styled = _map(_status_color, subset=["Status"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
