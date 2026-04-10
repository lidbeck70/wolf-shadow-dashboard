"""
OVTLYR main analysis page — Streamlit entry point.

Render with: render_ovtlyr_page()

Dual-path imports support both package and flat-file execution contexts.
All data fetches are cached (ttl=1800s) and wrapped in try/except.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# ------------------------------------------------------------------ #
#  Dual-path imports — works both as package and as flat module
# ------------------------------------------------------------------ #

# UI helpers
try:
    from ovtlyr.ui.colors import (
        BG, BG2, CYAN, MAGENTA, GREEN, RED, YELLOW, TEXT, DIM,
        REGIME_COLORS, SIGNAL_COLORS,
        risk_color, sentiment_color, ob_color,
        signal_badge_css, regime_badge_css,
    )
    from ovtlyr.ui.charts import (
        build_price_chart, build_sentiment_gauge,
        build_sector_pie, build_heatmap,
        build_risk_gauge, build_momentum_chart,
        build_volatility_histogram, build_oscillator_direction, build_bull_list_gauge,
    )
except ImportError:
    from ui.colors import (  # type: ignore
        BG, BG2, CYAN, MAGENTA, GREEN, RED, YELLOW, TEXT, DIM,
        REGIME_COLORS, SIGNAL_COLORS,
        risk_color, sentiment_color, ob_color,
        signal_badge_css, regime_badge_css,
    )
    from ui.charts import (  # type: ignore
        build_price_chart, build_sentiment_gauge,
        build_sector_pie, build_heatmap,
        build_risk_gauge, build_momentum_chart,
        build_volatility_histogram, build_oscillator_direction, build_bull_list_gauge,
    )

# Signal engines
try:
    from ovtlyr.signals.longterm_signals import compute_longterm_signal
    from ovtlyr.signals.swing_signals    import compute_swing_signal
except ImportError:
    from signals.longterm_signals import compute_longterm_signal  # type: ignore
    from signals.swing_signals    import compute_swing_signal     # type: ignore

# Data fetch modules — adapt to whichever data layer exists
try:
    from ovtlyr.data_fetch.borsdata import fetch_ohlcv, fetch_company_info
    _HAS_BORSDATA = True
except ImportError:
    _HAS_BORSDATA = False

# Import each indicator module separately to avoid one failure killing all
compute_trend        = None
compute_momentum     = None
compute_volatility   = None
compute_volume       = None
detect_orderblocks   = None
classify_price_vs_ob = None
compute_sentiment    = None
compute_breadth      = None
compute_volatility_histogram = None
compute_oscillator_direction = None
compute_bull_list_pct        = None
detect_patterns              = None

for _prefix in ("ovtlyr.indicators", "indicators"):
    try:
        _mod = __import__(f"{_prefix}.trend", fromlist=["compute_trend"])
        compute_trend = _mod.compute_trend
    except Exception:
        pass
    try:
        _mod = __import__(f"{_prefix}.momentum", fromlist=["compute_momentum"])
        compute_momentum = _mod.compute_momentum
    except Exception:
        pass
    try:
        _mod = __import__(f"{_prefix}.volatility", fromlist=["compute_volatility"])
        compute_volatility = _mod.compute_volatility
    except Exception:
        pass
    try:
        _mod = __import__(f"{_prefix}.volume", fromlist=["compute_volume"])
        compute_volume = _mod.compute_volume
    except Exception:
        pass
    try:
        _mod = __import__(f"{_prefix}.orderblocks", fromlist=["detect_orderblocks", "classify_price_vs_ob"])
        detect_orderblocks = _mod.detect_orderblocks
        classify_price_vs_ob = _mod.classify_price_vs_ob
    except Exception:
        pass
    try:
        _mod = __import__(f"{_prefix}.sentiment", fromlist=["compute_sentiment"])
        compute_sentiment = _mod.compute_sentiment
    except Exception:
        pass
    try:
        _mod = __import__(f"{_prefix}.breadth", fromlist=["compute_breadth"])
        compute_breadth = _mod.compute_breadth
    except Exception:
        pass
    try:
        _mod = __import__(f"{_prefix}.advanced", fromlist=["compute_volatility_histogram", "compute_oscillator_direction", "compute_bull_list_pct"])
        compute_volatility_histogram = _mod.compute_volatility_histogram
        compute_oscillator_direction = _mod.compute_oscillator_direction
        compute_bull_list_pct = _mod.compute_bull_list_pct
    except Exception:
        pass
    try:
        _mod = __import__(f"{_prefix}.candlesticks", fromlist=["detect_patterns"])
        detect_patterns = _mod.detect_patterns
    except Exception:
        pass
    # If we got at least orderblocks, stop trying other prefixes
    if detect_orderblocks is not None:
        break

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
DEFAULT_TICKER = "VOLV-B.ST"
COMMON_TICKERS = [
    "VOLV-B.ST", "ERIC-B.ST", "INVE-B.ST", "SHB-A.ST", "SEB-A.ST",
    "AZN.L", "NOVO-B.CO", "ASML.AS",
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
]

_CARD_CSS = (
    f"background:{BG2}; border-radius:6px; padding:14px 16px; "
    f"border-left:3px solid {{color}}; margin-bottom:10px;"
)


# ------------------------------------------------------------------ #
#  Cached data fetch helpers
# ------------------------------------------------------------------ #

@st.cache_data(ttl=1800, show_spinner=False, max_entries=20)
def _load_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV data. Falls back to yfinance if Borsdata unavailable."""
    if _HAS_BORSDATA:
        return fetch_ohlcv(ticker, period)

    # yfinance fallback
    try:
        import yfinance as yf
        mapping = {"6mo": "6mo", "1y": "1y", "2y": "2y"}
        df = yf.download(ticker, period=mapping.get(period, "1y"), auto_adjust=True, progress=False)
        df = df.reset_index()
        df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
        return df
    except Exception as exc:
        raise RuntimeError(f"Could not fetch OHLCV for {ticker}: {exc}") from exc


@st.cache_data(ttl=1800, show_spinner=False, max_entries=20)
def _load_sentiment() -> dict:
    if compute_sentiment is not None:
        return compute_sentiment({}, {}, {}, {})
    # Dummy fallback
    return {"score": 50, "label": "Neutral"}


@st.cache_data(ttl=1800, show_spinner=False, max_entries=20)
def _load_sector_breadth() -> dict:
    """Return sector breadth as {sector_name: {state, change, weight}} for pie/heatmap."""
    if compute_breadth is not None:
        try:
            raw = compute_breadth()
            # compute_breadth returns {total, bullish, bearish, sectors: {name: {bullish, bearish, ...}}}
            sectors_raw = raw.get("sectors", {})
            if sectors_raw and isinstance(sectors_raw, dict):
                result = {}
                for name, info in sectors_raw.items():
                    if not isinstance(info, dict):
                        continue
                    b = info.get("bullish", 0)
                    total = info.get("total", 1)
                    state = "bullish" if b > total / 2 else ("bearish" if b < total / 3 else "neutral")
                    result[name] = {"state": state, "change": 0.0, "weight": total}
                if result:
                    return result
        except Exception:
            pass
    # Dummy fallback
    return {
        "Technology":   {"state": "bullish", "change": 1.2,  "weight": 20},
        "Industrials":  {"state": "neutral",  "change": -0.3, "weight": 15},
        "Financials":   {"state": "bullish", "change": 0.8,  "weight": 18},
        "Health Care":  {"state": "neutral",  "change": 0.1,  "weight": 12},
        "Consumer":     {"state": "bearish",  "change": -1.1, "weight": 10},
        "Energy":       {"state": "bearish",  "change": -2.0, "weight": 8},
        "Materials":    {"state": "neutral",  "change": 0.4,  "weight": 7},
        "Utilities":    {"state": "neutral",  "change": -0.2, "weight": 5},
        "Real Estate":  {"state": "bearish",  "change": -0.9, "weight": 5},
    }


# ------------------------------------------------------------------ #
#  Small UI helpers
# ------------------------------------------------------------------ #

def _badge(text: str, color: str) -> str:
    """Inline HTML badge."""
    return (
        f'<span style="'
        f'background:{color}22; color:{color}; '
        f'border:1px solid {color}; border-radius:4px; '
        f'padding:3px 10px; font-weight:700; font-size:0.9rem; '
        f'letter-spacing:0.06em;">'
        f'{text}</span>'
    )


def _metric_card(title: str, value: str, subtitle: str, color: str) -> None:
    """Render a single metric card via st.markdown."""
    st.markdown(
        f'<div style="{_CARD_CSS.format(color=color)}">'
        f'<div style="color:{DIM}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;">{title}</div>'
        f'<div style="color:{color}; font-size:1.5rem; font-weight:700; margin:4px 0;">{value}</div>'
        f'<div style="color:{DIM}; font-size:0.76rem;">{subtitle}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _indicator_card(title: str, lines: list[tuple[str, str, str]], color: str) -> None:
    """
    Render a multi-line indicator card.
    lines: [(label, value, val_color), ...]
    """
    inner = "".join(
        f'<div style="display:flex;justify-content:space-between;margin:3px 0;">'
        f'<span style="color:{DIM};font-size:0.8rem;">{lbl}</span>'
        f'<span style="color:{vc};font-size:0.8rem;font-weight:600;">{val}</span>'
        f'</div>'
        for lbl, val, vc in lines
    )
    st.markdown(
        f'<div style="{_CARD_CSS.format(color=color)}">'
        f'<div style="color:{color}; font-size:0.78rem; font-weight:700; '
        f'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">{title}</div>'
        f'{inner}'
        f'</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=1800, show_spinner=False, max_entries=20)
def _fetch_etf_data_for_bull_list():
    import yfinance as yf
    etf_tickers = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLB", "XLC", "XLY", "XLP", "XLU", "XLRE"]
    etf_data = {}
    for etf in etf_tickers:
        try:
            t = yf.Ticker(etf)
            h = t.history(period="3mo", auto_adjust=True)
            if not h.empty:
                etf_data[etf] = h
        except Exception:
            pass
    return etf_data


# ------------------------------------------------------------------ #
#  Main page
# ------------------------------------------------------------------ #

def render_ovtlyr_page() -> None:
    """
    Full OVTLYR analysis page.

    Sidebar: ticker input, period selector, refresh button.
    Main layout: top row metrics, price chart + indicator cards, tabbed detail.
    """

    # ── Global styles ─────────────────────────────────────────────────
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {BG}; color: {TEXT}; }}
        .stTabs [role="tab"] {{ color: {DIM}; border-bottom: 2px solid {DIM}; }}
        .stTabs [aria-selected="true"] {{ color: {CYAN}; border-bottom: 2px solid {CYAN}; }}
        .stDataFrame {{ background: {BG2}; }}
        div[data-testid="stMetricValue"] {{ color: {CYAN}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Ticker selector (in main area, not sidebar) ─────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1.5, 1, 1])
    with ctrl1:
        ticker_input = st.text_input(
            "TICKER", value=DEFAULT_TICKER, key="ovtlyr_ticker"
        ).strip().upper()
    with ctrl2:
        quick_pick = st.selectbox(
            "QUICK PICK", ["— custom —"] + COMMON_TICKERS, key="ovtlyr_quick"
        )
    with ctrl3:
        period = st.selectbox(
            "PERIOD", ["6mo", "1y", "2y"], index=2, key="ovtlyr_period"
        )
    with ctrl4:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        refresh = st.button("↺ ANALYSERA", key="ovtlyr_refresh", use_container_width=True)
        if refresh:
            st.cache_data.clear()

    if quick_pick != "— custom —":
        ticker = quick_pick
    else:
        ticker = ticker_input if ticker_input else DEFAULT_TICKER

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
        key="benchmark_ovtlyr",
    )
    benchmark_ticker_bm = _BENCHMARK_OPTIONS[selected_bm_label]

    # Relative Strength indicator
    try:
        import yfinance as _yf_bm
        bm_data = _yf_bm.download(benchmark_ticker_bm, period="3mo", auto_adjust=True, progress=False)
        stk_data_rs = _yf_bm.download(ticker, period="3mo", auto_adjust=True, progress=False)
        if isinstance(bm_data.columns, pd.MultiIndex):
            bm_data.columns = bm_data.columns.get_level_values(0)
        if isinstance(stk_data_rs.columns, pd.MultiIndex):
            stk_data_rs.columns = stk_data_rs.columns.get_level_values(0)
        if not bm_data.empty and not stk_data_rs.empty and len(bm_data) >= 20 and len(stk_data_rs) >= 20:
            _stk_ret = float(stk_data_rs["Close"].iloc[-1] / stk_data_rs["Close"].iloc[-20])
            _bm_ret = float(bm_data["Close"].iloc[-1] / bm_data["Close"].iloc[-20])
            if _bm_ret > 0:
                _rs = _stk_ret / _bm_ret
                if _rs > 1.05:
                    _rs_icon, _rs_color = "🟢", GREEN
                elif _rs >= 0.95:
                    _rs_icon, _rs_color = "🟡", YELLOW
                else:
                    _rs_icon, _rs_color = "🔴", RED
                st.markdown(
                    f"<div style='padding:4px 0;'>"
                    f"<span style='font-size:0.85rem;'>{_rs_icon}</span> "
                    f"<span style='color:{_rs_color};font-size:0.85rem;font-weight:700;'>"
                    f"RS: {_rs - 1:+.1%} vs {selected_bm_label}</span></div>",
                    unsafe_allow_html=True,
                )
    except Exception:
        pass

    # ── Data load ─────────────────────────────────────────────────────
    with st.spinner(f"Loading {ticker}…"):
        try:
            df = _load_ohlcv(ticker, period)
        except Exception as exc:
            st.error(f"Failed to load price data for **{ticker}**: {exc}")
            return

    if df is None or len(df) < 20:
        st.warning(f"Not enough data returned for **{ticker}**. Try a different ticker or period.")
        return

    # Normalise column names
    df.columns = [str(c).strip() for c in df.columns]
    for col in ("Date", "Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            # Try case-insensitive match
            match = [c for c in df.columns if c.lower() == col.lower()]
            if match:
                df = df.rename(columns={match[0]: col})

    df["Date"]   = pd.to_datetime(df["Date"])
    df["Close"]  = pd.to_numeric(df["Close"],  errors="coerce")
    df["Open"]   = pd.to_numeric(df["Open"],   errors="coerce")
    df["High"]   = pd.to_numeric(df["High"],   errors="coerce")
    df["Low"]    = pd.to_numeric(df["Low"],    errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df = df.dropna(subset=["Close"]).reset_index(drop=True)

    # ── Compute indicators ────────────────────────────────────────────
    with st.spinner("Computing indicators…"):

        # Trend
        if compute_trend is not None:
            try:
                trend = compute_trend(df)
            except Exception:
                trend = {}
        else:
            trend = {}

        # EMA fallbacks
        close = df["Close"].astype(float)
        if "ema50" not in trend or trend["ema50"] is None:
            trend["ema50"] = close.ewm(span=50, adjust=False).mean().tolist()
        if "ema200" not in trend or trend["ema200"] is None:
            trend["ema200"] = close.ewm(span=200, adjust=False).mean().tolist()
        if "price" not in trend:
            trend["price"] = float(close.iloc[-1])

        # EMA10 / EMA20 — required for OVTLYR NINE scoring
        if "Close" in df.columns:
            trend["ema10"] = df["Close"].ewm(span=10, adjust=False).mean()
            trend["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        if "regime_color" not in trend:
            ema50_last  = trend["ema50"][-1] if isinstance(trend["ema50"], list) else trend["ema50"]
            ema200_last = trend["ema200"][-1] if isinstance(trend["ema200"], list) else trend["ema200"]
            price_last  = trend["price"]
            if price_last > ema200_last and ema50_last > ema200_last:
                trend["regime_color"] = "green"
            elif price_last < ema200_last:
                trend["regime_color"] = "red"
            else:
                trend["regime_color"] = "orange"
        if "direction" not in trend:
            trend["direction"] = (
                "bullish" if trend["regime_color"] == "green"
                else "bearish" if trend["regime_color"] == "red"
                else "neutral"
            )
        if "in_consolidation" not in trend:
            trend["in_consolidation"] = False

        # Momentum
        if compute_momentum is not None:
            try:
                momentum = compute_momentum(df)
            except Exception:
                momentum = {}
        else:
            momentum = {}

        if "rsi" not in momentum:
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, float("nan"))
            rsi_series = (100 - (100 / (1 + rs))).fillna(50)
            momentum["rsi"] = float(rsi_series.iloc[-1])
            momentum["rsi_series"] = rsi_series.tolist()
            rsi_val = momentum["rsi"]
            if rsi_val >= 70:
                momentum["ob_os_flag"] = "overbought"
            elif rsi_val <= 30:
                momentum["ob_os_flag"] = "oversold"
            else:
                momentum["ob_os_flag"] = "neutral"

        # Volatility / Risk
        if compute_volatility is not None:
            try:
                volatility = compute_volatility(df)
            except Exception:
                volatility = {}
        else:
            volatility = {}

        if "atr" not in volatility:
            high_low  = df["High"] - df["Low"]
            high_prev = (df["High"] - df["Close"].shift()).abs()
            low_prev  = (df["Low"]  - df["Close"].shift()).abs()
            tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
            volatility["atr"] = float(tr.rolling(14).mean().iloc[-1])
        if "hist_vol" not in volatility:
            volatility["hist_vol"] = float(close.pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5) * 100)
        if "risk_score" not in volatility:
            # Simple risk proxy: normalise hist_vol to 0-100 (50% vol = 100)
            volatility["risk_score"] = min(100, int(volatility["hist_vol"] * 2))

        # Volume
        if compute_volume is not None:
            try:
                volume_data = compute_volume(df)
            except Exception:
                volume_data = {}
        else:
            volume_data = {}

        if "confirms" not in volume_data:
            avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
            cur_vol = df["Volume"].iloc[-1]
            volume_data["confirms"] = bool(cur_vol > avg_vol)
            volume_data["ratio"]    = float(cur_vol / avg_vol) if avg_vol > 0 else 1.0
            volume_data["trend"]    = "rising" if volume_data["confirms"] else "flat"

        # Order blocks
        orderblocks: list[dict] = []
        ob_analysis: dict = {"signal_bias": "HOLD"}

        if detect_orderblocks is not None:
            try:
                orderblocks = detect_orderblocks(df)
            except Exception as _ob_err:
                orderblocks = []
                st.warning(f"Order block detection error: {_ob_err}")
        else:
            st.warning("Order block module not loaded — check imports.")

        if classify_price_vs_ob is not None and orderblocks:
            try:
                current_price = float(df["Close"].iloc[-1]) if len(df) > 0 else 0
                ob_analysis = classify_price_vs_ob(current_price, orderblocks)
            except Exception:
                ob_analysis = {"signal_bias": "HOLD"}

        # Detect candlestick patterns
        candle_patterns = {"bullish": [], "bearish": []}
        if detect_patterns is not None:
            try:
                candle_patterns = detect_patterns(df, lookback=5)
            except Exception:
                candle_patterns = {"bullish": [], "bearish": []}

        # Sentiment
        try:
            sentiment = _load_sentiment()
        except Exception:
            sentiment = {"score": 50, "label": "Neutral"}

        # Sector breadth
        try:
            breadth_data = _load_sector_breadth()
        except Exception:
            breadth_data = {}

        # Determine sector_green from breadth (placeholder: True if >50% bullish)
        bullish_sectors = sum(
            1 for v in breadth_data.values()
            if isinstance(v, dict) and v.get("state") == "bullish"
        )
        sector_green = bullish_sectors > len(breadth_data) / 2 if breadth_data else True

        # Advanced indicators
        vol_histogram = {}
        oscillator = {}
        bull_list = {}

        if compute_volatility_histogram is not None:
            try:
                vol_histogram = compute_volatility_histogram(df)
            except Exception:
                vol_histogram = {}

        if compute_oscillator_direction is not None:
            try:
                oscillator = compute_oscillator_direction(df)
            except Exception:
                oscillator = {}

        if compute_bull_list_pct is not None:
            try:
                etf_data = _fetch_etf_data_for_bull_list()
                if etf_data:
                    bull_list = compute_bull_list_pct(etf_data)
            except Exception:
                bull_list = {}

        # Build scalar-only trend dict for signal modules
        def _last(val, default=0.0):
            if val is None:
                return default
            if isinstance(val, list):
                return float(val[-1]) if val else default
            try:
                return float(val.iloc[-1]) if hasattr(val, 'iloc') and len(val) > 0 else float(val)
            except Exception:
                return default

        trend_scalar = {
            "price": _last(trend.get("price", close.iloc[-1] if len(close) > 0 else 0)),
            "last_close": _last(trend.get("price", close.iloc[-1] if len(close) > 0 else 0)),
            "ema10": _last(trend.get("ema10", 0)),
            "ema20": _last(trend.get("ema20", 0)),
            "ema50": _last(trend.get("ema50", 0)),
            "ema200": _last(trend.get("ema200", 0)),
            "regime_color": trend.get("regime_color", "orange"),
            "regime_prev": trend.get("regime_prev", trend.get("regime_color", "orange")),
            "trend_state": trend.get("trend_state", trend.get("direction", "neutral")),
            "direction": trend.get("direction", "neutral"),
            "price_above_200": trend.get("price_above_200", False),
            "ema50_above_200": trend.get("ema50_above_200", False),
            "in_consolidation": trend.get("in_consolidation", False),
            "pullback_to_ema": trend.get("pullback_to_ema", False),
        }

        # Compute signals with scalar trend dict
        lt_signal  = compute_longterm_signal(trend_scalar, sentiment, volatility, ob_analysis, sector_green)
        swg_signal = compute_swing_signal(trend_scalar, momentum, volume_data, ob_analysis)

    # ── Composite score ───────────────────────────────────────────────
    composite_score = int(
        (lt_signal["confidence"] * 0.6) + (swg_signal["confidence"] * 0.4)
    )

    regime_color_hex = REGIME_COLORS.get(trend.get("regime_color", "orange"), YELLOW)
    lt_color         = SIGNAL_COLORS.get(lt_signal["signal"],  YELLOW)
    swg_color        = SIGNAL_COLORS.get(swg_signal["signal"], DIM)

    # ── TOP ROW ───────────────────────────────────────────────────────
    st.markdown(
        f'<h2 style="color:{CYAN}; letter-spacing:0.12em; margin-bottom:2px;">'
        f'{ticker}</h2>',
        unsafe_allow_html=True,
    )

    top1, top2, top3, top4 = st.columns([1.2, 1, 1, 1])

    with top1:
        regime_label = trend.get("regime_color", "orange").upper()
        st.markdown(
            f'<div style="margin-top:8px;">'
            f'<div style="color:{DIM}; font-size:0.72rem; text-transform:uppercase;">Regime</div>'
            f'<div style="margin-top:6px;">{_badge(regime_label, regime_color_hex)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        price_now = trend.get("price", float(df["Close"].iloc[-1]))
        price_prev = float(df["Close"].iloc[-2]) if len(df) > 1 else price_now
        price_chg  = (price_now - price_prev) / price_prev * 100 if price_prev else 0
        chg_color  = GREEN if price_chg >= 0 else RED
        st.markdown(
            f'<div style="margin-top:10px; color:{TEXT}; font-size:1.3rem; font-weight:700;">'
            f'{price_now:,.2f}'
            f'<span style="color:{chg_color}; font-size:0.85rem; margin-left:8px;">'
            f'{"+" if price_chg>=0 else ""}{price_chg:.2f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with top2:
        st.markdown(
            f'<div style="color:{DIM}; font-size:0.72rem; text-transform:uppercase; margin-top:8px;">Score</div>',
            unsafe_allow_html=True,
        )
        score_color = risk_color(100 - composite_score)  # invert: higher score = less risk
        st.markdown(
            f'<div style="color:{score_color}; font-size:2.4rem; font-weight:700; line-height:1.1;">'
            f'{composite_score}<span style="font-size:1rem; color:{DIM}">/100</span></div>',
            unsafe_allow_html=True,
        )

    with top3:
        st.markdown(
            f'<div style="color:{DIM}; font-size:0.72rem; text-transform:uppercase; margin-top:8px;">Long-term</div>'
            f'<div style="margin-top:6px;">{_badge(lt_signal["signal"], lt_color)}</div>'
            f'<div style="color:{DIM}; font-size:0.72rem; margin-top:4px;">'
            f'Confidence: {lt_signal["confidence"]}%</div>',
            unsafe_allow_html=True,
        )

    with top4:
        entry_txt = swg_signal.get("entry_type") or ""
        st.markdown(
            f'<div style="color:{DIM}; font-size:0.72rem; text-transform:uppercase; margin-top:8px;">Swing</div>'
            f'<div style="margin-top:6px;">{_badge(swg_signal["signal"], swg_color)}</div>'
            f'<div style="color:{DIM}; font-size:0.72rem; margin-top:4px;">'
            f'{entry_txt or "—"}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # OVTLYR Gates Passed
    st.markdown(
        f"<div style='color:{CYAN};font-size:0.7rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;margin:12px 0 8px 0;'>OVTLYR ENTRY GATES</div>",
        unsafe_allow_html=True,
    )

    gates = lt_signal.get("gates", [])
    if gates:
        passed = sum(1 for g in gates if g.get("passed", False))
        total = len(gates)
        gate_color = GREEN if passed >= total * 0.7 else (YELLOW if passed >= total * 0.4 else RED)

        gate_html = f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">'
        for g in gates:
            c = GREEN if g.get("passed") else RED
            icon = "✓" if g.get("passed") else "✗"
            short_rule = g.get("rule", "")[:30]
            gate_html += (
                f'<div style="background:{BG2};border:1px solid {c};border-radius:4px;'
                f'padding:3px 8px;font-size:0.65rem;">'
                f'<span style="color:{c};font-weight:700;">{icon}</span> '
                f'<span style="color:{TEXT};">{short_rule}</span>'
                f'</div>'
            )
        gate_html += f'</div>'
        gate_html += (
            f'<div style="color:{gate_color};font-size:0.8rem;font-weight:700;">'
            f'{passed}/{total} GATES PASSED</div>'
        )
        st.markdown(gate_html, unsafe_allow_html=True)

    # ── MIDDLE ROW ────────────────────────────────────────────────────
    mid_left, mid_right = st.columns([7, 3])

    with mid_left:
        try:
            fig_price = build_price_chart(df.copy(), trend, orderblocks, volume_data)
            st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})
        except Exception as exc:
            st.error(f"Price chart error: {exc}")

    with mid_right:
        # Trend card
        def _safe_last(val, default=0.0):
            if val is None:
                return default
            if isinstance(val, list):
                return float(val[-1]) if val else default
            try:
                return float(val.iloc[-1]) if hasattr(val, 'iloc') and len(val) > 0 else float(val)
            except Exception:
                return default

        ema50_last  = _safe_last(trend.get("ema50"), 0.0)
        ema200_last = _safe_last(trend.get("ema200"), 0.0)
        direction_str = trend.get("direction", "")
        if hasattr(direction_str, 'capitalize'):
            direction_str = direction_str.capitalize()
        else:
            direction_str = str(direction_str)
        regime_str = str(trend.get("regime_color", "")).upper()

        _indicator_card(
            "Trend",
            [
                ("Direction",  direction_str, regime_color_hex),
                ("EMA 50",     f"{ema50_last:.2f}",  YELLOW),
                ("EMA 200",    f"{ema200_last:.2f}", MAGENTA),
                ("Regime",     regime_str, regime_color_hex),
            ],
            color=regime_color_hex,
        )

        # Volatility card
        v_color = risk_color(int(volatility.get("risk_score", 50)))
        _indicator_card(
            "Volatility & Risk",
            [
                ("ATR 14",    f"{volatility.get('atr', 0):.2f}",              v_color),
                ("Hist Vol",  f"{volatility.get('hist_vol', 0):.1f}%",         v_color),
                ("Risk Score", f"{int(volatility.get('risk_score', 50))}/100",  v_color),
            ],
            color=v_color,
        )

        # Sentiment card
        s_score = int(sentiment.get("score", 50))
        s_color = sentiment_color(s_score)
        _indicator_card(
            "Sentiment",
            [
                ("Fear & Greed", f"{s_score}/100",                  s_color),
                ("Label",        sentiment.get("label", "—"),        s_color),
                ("Rule 5 ok?",   "Yes" if s_score < 60 else "No ⚠",
                 GREEN if s_score < 60 else RED),
            ],
            color=s_color,
        )

        # Momentum card
        rsi_val = float(momentum.get("rsi", 50))
        rsi_color_now = RED if rsi_val >= 70 else (GREEN if rsi_val <= 30 else CYAN)
        _indicator_card(
            "Momentum",
            [
                ("RSI 14",    f"{rsi_val:.1f}",                       rsi_color_now),
                ("State",     momentum.get("ob_os_flag", "neutral").upper(), rsi_color_now),
                ("Vol ratio", f"{volume_data.get('ratio', 1.0):.2f}x",
                 GREEN if volume_data.get("confirms") else DIM),
            ],
            color=CYAN,
        )

        # Risk gauge (compact)
        try:
            fig_risk = build_risk_gauge(int(volatility.get("risk_score", 50)))
            st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            pass

    # ── Candlestick Pattern Cards (outside mid columns, full width) ──
    cs_left, cs_right = st.columns(2)

    with cs_left:
        bull_patterns = candle_patterns.get("bullish", [])
        if bull_patterns:
            patterns_html = "".join(
                f'<div style="display:flex;justify-content:space-between;margin:4px 0;">'
                f'<span style="color:{TEXT};font-size:0.85rem;">{p.visual} {p.name}</span>'
                f'<span style="color:{GREEN};font-size:0.75rem;font-weight:600;">{p.confidence}</span>'
                f'</div>'
                f'<div style="color:{DIM};font-size:0.65rem;margin-bottom:6px;">{p.description}</div>'
                for p in bull_patterns[:5]
            )
            count_text = f"{len(bull_patterns)} mönster hittade"
        else:
            patterns_html = f'<div style="color:{DIM};font-size:0.8rem;padding:8px 0;">Inga bullish mönster just nu</div>'
            count_text = "0 mönster"

        st.markdown(
            f'<div style="background:{BG2};border:2px solid rgba(0,255,136,0.3);border-radius:8px;padding:14px 16px;margin:8px 0;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
            f'<span style="color:{GREEN};font-size:0.85rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">ENTRY PATTERNS</span>'
            f'<span style="color:{DIM};font-size:0.7rem;">{count_text}</span>'
            f'</div>'
            f'{patterns_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with cs_right:
        bear_patterns = candle_patterns.get("bearish", [])
        if bear_patterns:
            patterns_html = "".join(
                f'<div style="display:flex;justify-content:space-between;margin:4px 0;">'
                f'<span style="color:{TEXT};font-size:0.85rem;">{p.visual} {p.name}</span>'
                f'<span style="color:{RED};font-size:0.75rem;font-weight:600;">{p.confidence}</span>'
                f'</div>'
                f'<div style="color:{DIM};font-size:0.65rem;margin-bottom:6px;">{p.description}</div>'
                for p in bear_patterns[:5]
            )
            count_text = f"{len(bear_patterns)} varningar"
        else:
            patterns_html = f'<div style="color:{DIM};font-size:0.8rem;padding:8px 0;">Inga bearish varningar just nu</div>'
            count_text = "0 varningar"

        st.markdown(
            f'<div style="background:{BG2};border:2px solid rgba(255,51,85,0.3);border-radius:8px;padding:14px 16px;margin:8px 0;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
            f'<span style="color:{RED};font-size:0.85rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">EXIT WARNINGS</span>'
            f'<span style="color:{DIM};font-size:0.7rem;">{count_text}</span>'
            f'</div>'
            f'{patterns_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── SL/TP Calculator ─────────────────────────────────────────────────────
    try:
        st.markdown(
            f"<div style='color:{CYAN};font-size:0.85rem;text-transform:uppercase;"
            f"letter-spacing:0.1em;margin:20px 0 10px 0;border-top:2px solid rgba(0,255,255,0.2);"
            f"padding-top:14px;font-weight:700;'>SL / TP KALKYLATOR — OVTLYR</div>",
            unsafe_allow_html=True,
        )

        if df is not None and not df.empty and len(df) >= 20:
            _close_col = None
            _high_col = None
            _low_col = None
            for c in df.columns:
                cl = str(c).lower().strip()
                if cl == "close":
                    _close_col = c
                elif cl == "high":
                    _high_col = c
                elif cl == "low":
                    _low_col = c

            if _close_col and _high_col and _low_col:
                _close = pd.to_numeric(df[_close_col], errors="coerce")
                _high = pd.to_numeric(df[_high_col], errors="coerce")
                _low = pd.to_numeric(df[_low_col], errors="coerce")
                _price = float(_close.dropna().iloc[-1])

                _tr = pd.concat([
                    _high - _low,
                    abs(_high - _close.shift(1)),
                    abs(_low - _close.shift(1)),
                ], axis=1).max(axis=1)
                _atr = float(_tr.rolling(14).mean().dropna().iloc[-1])
                _half_atr = _atr / 2
                _ema10 = float(_close.ewm(span=10).mean().iloc[-1])

                _sl = _price - _half_atr
                _sl_dist = _price - _sl
                _tp_2r = _price + _sl_dist * 2
                _tp_3r = _price + _sl_dist * 3

                sltp_c1, sltp_c2 = st.columns(2)
                with sltp_c1:
                    capital_ov = st.number_input(
                        "Kapital (SEK)", value=100000, step=10000,
                        key="sltp_cap_ovtlyr_main",
                    )
                with sltp_c2:
                    risk_ov = st.number_input(
                        "Risk %", value=5.0, min_value=0.5, max_value=10.0,
                        step=0.5, key="sltp_risk_ovtlyr_main",
                    )

                _risk_amount = capital_ov * (risk_ov / 100)
                _shares = int(_risk_amount / _sl_dist) if _sl_dist > 0 else 0
                _pos_value = _shares * _price
                _pos_pct = (_pos_value / capital_ov * 100) if capital_ov > 0 else 0

                sl_c, tp_c, ps_c = st.columns(3)
                with sl_c:
                    st.markdown(
                        f'<div style="background:{BG2};border:2px solid rgba(255,51,85,0.3);border-radius:8px;padding:12px;">'
                        f'<div style="color:{RED};font-weight:700;font-size:0.8rem;">STOP LOSS</div>'
                        f'<div style="color:{TEXT};font-size:1.1rem;font-weight:700;">{_sl:.2f}</div>'
                        f'<div style="color:{DIM};font-size:0.65rem;">½ ATR = {_half_atr:.2f}</div>'
                        f'<div style="color:{RED};font-size:0.72rem;">Risk: {_sl_dist:.2f} ({_sl_dist/_price*100:.1f}%)</div>'
                        f'<div style="color:{DIM};font-size:0.6rem;margin-top:4px;">Trail: EMA10 ({_ema10:.2f})</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with tp_c:
                    st.markdown(
                        f'<div style="background:{BG2};border:2px solid rgba(0,255,136,0.3);border-radius:8px;padding:12px;">'
                        f'<div style="color:{GREEN};font-weight:700;font-size:0.8rem;">TARGETS</div>'
                        f'<div style="color:{TEXT};font-size:0.85rem;">2R: <b style="color:{GREEN};">{_tp_2r:.2f}</b> (+{(_tp_2r/_price-1)*100:.1f}%)</div>'
                        f'<div style="color:{TEXT};font-size:0.85rem;">3R: <b style="color:{GREEN};">{_tp_3r:.2f}</b> (+{(_tp_3r/_price-1)*100:.1f}%)</div>'
                        f'<div style="color:{DIM};font-size:0.6rem;margin-top:4px;">Trailing stop — ej fast TP</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with ps_c:
                    st.markdown(
                        f'<div style="background:{BG2};border:2px solid rgba(0,255,255,0.2);border-radius:8px;padding:12px;">'
                        f'<div style="color:{CYAN};font-weight:700;font-size:0.8rem;">POSITION</div>'
                        f'<div style="color:{TEXT};font-size:0.85rem;">{_shares} aktier</div>'
                        f'<div style="color:{TEXT};font-size:0.85rem;">{_pos_value:,.0f} SEK ({_pos_pct:.1f}%)</div>'
                        f'<div style="color:{RED};font-size:0.72rem;">Risk: {_risk_amount:,.0f} SEK</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("SL/TP: OHLC-kolumner saknas i data.")
        else:
            st.info("SL/TP: Otillräcklig data för beräkning.")
    except Exception as sltp_err:
        st.warning(f"SL/TP-kalkylator kunde inte renderas: {sltp_err}")

    # ── Advanced Indicators Row ─────────────────────────────────────
    st.markdown(
        f"<div style='color:{CYAN};font-size:0.7rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;margin:16px 0 8px 0;border-top:1px solid rgba(0,255,255,0.13);"
        f"padding-top:12px;'>OVTLYR Advanced Analysis</div>",
        unsafe_allow_html=True,
    )

    adv_c1, adv_c2, adv_c3 = st.columns([1, 1, 1])

    with adv_c1:
        if vol_histogram:
            try:
                fig_vh = build_volatility_histogram(vol_histogram)
                st.plotly_chart(fig_vh, use_container_width=True, key="vol_hist")
                # Stats below
                st.markdown(
                    f"<div style='font-size:0.68rem;color:{DIM};text-align:center;'>"
                    f"Up days: {vol_histogram.get('up_pct', 50):.0f}% | "
                    f"Avg up: +{vol_histogram.get('mean_up', 0):.2f}% | "
                    f"Avg down: {vol_histogram.get('mean_down', 0):.2f}% | "
                    f"{vol_histogram.get('years_analyzed', 0):.1f}y data"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                st.info("Volatility histogram: chart error")
        else:
            st.info("Volatility histogram: insufficient data")

    with adv_c2:
        if oscillator:
            try:
                fig_osc = build_oscillator_direction(oscillator)
                st.plotly_chart(fig_osc, use_container_width=True, key="osc_dir")
                # Timing badge
                timing = oscillator.get("timing", "")
                tc = oscillator.get("timing_color", "rgba(74,74,106,0.9)")
                sig = oscillator.get("signal", "WAIT")
                st.markdown(
                    f"<div style='text-align:center;font-size:0.72rem;'>"
                    f"<span style='color:{tc};font-weight:700;'>{timing}</span>"
                    f" ({oscillator.get('days_in_direction', 0)} days) — "
                    f"RSI: {oscillator.get('rsi', 50):.0f} "
                    f"({'↑' if oscillator.get('direction') == 'Rising' else '↓'}{abs(oscillator.get('rsi_change_5d', 0)):.0f} 5d)"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                st.info("Oscillator: chart error")
        else:
            st.info("Oscillator: insufficient data")

    with adv_c3:
        if bull_list:
            try:
                fig_bl = build_bull_list_gauge(bull_list)
                st.plotly_chart(fig_bl, use_container_width=True, key="bull_list")
            except Exception:
                st.info("Bull List %: chart error")
        else:
            st.info("Bull List %: loading...")

    # ── BOTTOM TABS ───────────────────────────────────────────────────
    tab_a, tab_b, tab_c, tab_d = st.tabs(
        ["Order Blocks", "Sectors", "Drawdowns", "Signal Log"]
    )

    # ── Tab A: Order Blocks ──────────────────────────────────────────
    with tab_a:
        if orderblocks:
            ob_rows = []
            for ob in orderblocks:
                # Support both OrderBlock objects (with attributes) and plain dicts
                ob_rows.append({
                    "Type": ob.type.upper() if hasattr(ob, 'type') else str(ob.get('type', '')).upper(),
                    "Date": ob.date if hasattr(ob, 'date') else ob.get('date', ob.get('date_start', '')),
                    "High": f"{ob.high:.2f}" if hasattr(ob, 'high') else f"{ob.get('high', 0):.2f}",
                    "Low":  f"{ob.low:.2f}"  if hasattr(ob, 'low')  else f"{ob.get('low', 0):.2f}",
                    "Volume": f"{ob.volume:,.0f}" if hasattr(ob, 'volume') else f"{ob.get('volume', 0):,.0f}",
                    "Status": ob.status if hasattr(ob, 'status') else ob.get('status', ''),
                })
            ob_df = pd.DataFrame(ob_rows)

            def _ob_row_style(row):
                t = str(row["Type"]).lower()
                if "bull" in t:
                    color = "rgba(0,255,136,0.08)"
                elif "bear" in t:
                    color = "rgba(255,51,85,0.08)"
                else:
                    color = "rgba(74,74,106,0.08)"
                return [f"background-color:{color}"] * len(row)

            styled_ob = ob_df.style.apply(_ob_row_style, axis=1)
            st.dataframe(styled_ob, use_container_width=True, hide_index=True)
        else:
            st.info("No order blocks detected.")

    # ── Tab B: Sectors ───────────────────────────────────────────────
    with tab_b:
        if breadth_data:
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                try:
                    fig_pie = build_sector_pie(breadth_data)
                    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
                except Exception as exc:
                    st.error(f"Sector pie error: {exc}")
            with bcol2:
                try:
                    fig_hm = build_heatmap(breadth_data)
                    st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})
                except Exception as exc:
                    st.error(f"Heatmap error: {exc}")
        else:
            st.info("Sector breadth data unavailable.")

    # ── Tab C: Drawdowns ─────────────────────────────────────────────
    with tab_c:
        try:
            rolling_max = close.cummax()
            drawdown    = (close - rolling_max) / rolling_max * 100

            # Find drawdown episodes
            in_dd  = False
            dd_start = None
            episodes = []
            threshold = -5.0  # only show drawdowns > 5%

            for i, (dt, dd) in enumerate(zip(df["Date"], drawdown)):
                if dd < threshold and not in_dd:
                    in_dd    = True
                    dd_start = dt
                    dd_min   = dd
                elif in_dd:
                    if dd < dd_min:
                        dd_min = dd
                    if dd >= -1.0:  # recovery
                        episodes.append({
                            "Start":       str(dd_start.date()),
                            "End":         str(dt.date()),
                            "Max Drop %":  round(dd_min, 2),
                            "Classification": (
                                "Crash" if dd_min < -30
                                else "Bear Market" if dd_min < -20
                                else "Correction" if dd_min < -10
                                else "Pullback"
                            ),
                        })
                        in_dd = False

            # Ongoing drawdown
            if in_dd:
                episodes.append({
                    "Start":       str(dd_start.date()),
                    "End":         "Ongoing",
                    "Max Drop %":  round(dd_min, 2),
                    "Classification": "Ongoing",
                })

            if episodes:
                dd_df = pd.DataFrame(episodes).sort_values("Max Drop %")

                def _dd_style(val):
                    if isinstance(val, (int, float)) and val < 0:
                        intensity = min(1.0, abs(val) / 40)
                        r = int(255 * intensity)
                        return f"color: rgb({r},51,85)"
                    return ""

                styled_dd = dd_df.style.map(_dd_style, subset=["Max Drop %"])
                st.dataframe(styled_dd, use_container_width=True, hide_index=True)
            else:
                st.info("No significant drawdowns (> 5%) detected in the selected period.")
        except Exception as exc:
            st.error(f"Drawdown analysis error: {exc}")

    # ── Tab D: Signal Log ────────────────────────────────────────────
    with tab_d:
        sig_col1, sig_col2 = st.columns(2)

        with sig_col1:
            st.markdown(
                f'<div style="color:{GREEN}; font-size:0.85rem; font-weight:700; '
                f'letter-spacing:0.08em; margin-bottom:8px;">LONG-TERM SIGNAL</div>',
                unsafe_allow_html=True,
            )
            # ── OVTLYR NINE score breakdown ──
            ovtlyr_nine   = lt_signal.get("ovtlyr_nine",   lt_signal.get("confidence", 0))
            market_score  = lt_signal.get("market_score",  0)
            sector_score  = lt_signal.get("sector_score",  0)
            stock_score   = lt_signal.get("stock_score",   0)
            nine_color    = GREEN if ovtlyr_nine >= 70 else (YELLOW if ovtlyr_nine >= 40 else RED)
            st.markdown(
                f'<div style="margin-bottom:10px;">{_badge(lt_signal["signal"], lt_color)}'  
                f'  <span style="color:{nine_color}; font-size:1.0rem; font-weight:700;">'
                f'  OVTLYR NINE: {ovtlyr_nine}/100</span></div>',
                unsafe_allow_html=True,
            )
            # Layer breakdown
            st.markdown(
                f'<div style="display:flex; gap:10px; margin-bottom:10px;">'
                f'<div style="flex:1; background:rgba(0,255,255,0.06); border-left:3px solid rgba(0,255,255,0.5); '
                f'padding:8px; border-radius:4px;">'
                f'<div style="color:{DIM}; font-size:0.68rem; text-transform:uppercase;">Market (40%)</div>'
                f'<div style="color:{CYAN}; font-size:1.2rem; font-weight:700;">{market_score}</div>'
                f'</div>'
                f'<div style="flex:1; background:rgba(255,0,255,0.06); border-left:3px solid rgba(255,0,255,0.5); '
                f'padding:8px; border-radius:4px;">'
                f'<div style="color:{DIM}; font-size:0.68rem; text-transform:uppercase;">Sector (30%)</div>'
                f'<div style="color:{MAGENTA}; font-size:1.2rem; font-weight:700;">{sector_score}</div>'
                f'</div>'
                f'<div style="flex:1; background:rgba(0,255,136,0.06); border-left:3px solid rgba(0,255,136,0.5); '
                f'padding:8px; border-radius:4px;">'
                f'<div style="color:{DIM}; font-size:0.68rem; text-transform:uppercase;">Stock (30%)</div>'
                f'<div style="color:{GREEN}; font-size:1.2rem; font-weight:700;">{stock_score}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for reason in lt_signal.get("reasons", []):
                color = (
                    GREEN  if reason.startswith("✓")
                    else RED    if reason.startswith("✗")
                    else YELLOW if reason.startswith("⚠")
                    else DIM
                )
                st.markdown(
                    f'<div style="color:{color}; font-size:0.78rem; margin:2px 0;">{reason}</div>',
                    unsafe_allow_html=True,
                )

            # OVTLYR NINE gates table
            st.markdown(
                f'<div style="color:{DIM}; font-size:0.72rem; margin-top:10px; margin-bottom:4px;">'
                f'OVTLYR NINE GATES</div>',
                unsafe_allow_html=True,
            )
            gates_df = pd.DataFrame(lt_signal.get("gates", []))
            if not gates_df.empty:
                def _gate_style(row):
                    color = "rgba(0,255,136,0.08)" if row.get("passed") else "rgba(255,51,85,0.08)"
                    return [f"background-color:{color}"] * len(row)
                st.dataframe(
                    gates_df.style.apply(_gate_style, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

            # Exit triggers
            exit_triggers = lt_signal.get("exit_triggers", [])
            active_exits  = [t for t in exit_triggers if t.get("active")]
            if active_exits:
                st.markdown(
                    f'<div style="color:{RED}; font-size:0.78rem; font-weight:700; margin-top:10px;">'
                    f'ACTIVE EXIT TRIGGERS</div>',
                    unsafe_allow_html=True,
                )
                for t in active_exits:
                    st.markdown(
                        f'<div style="color:{RED}; font-size:0.75rem; margin:2px 0;">'
                        f'⚠ {t["trigger"]}</div>',
                        unsafe_allow_html=True,
                    )

        with sig_col2:
            st.markdown(
                f'<div style="color:{CYAN}; font-size:0.85rem; font-weight:700; '
                f'letter-spacing:0.08em; margin-bottom:8px;">SWING SIGNAL</div>',
                unsafe_allow_html=True,
            )
            entry_txt = swg_signal.get("entry_type") or "—"
            st.markdown(
                f'<div style="margin-bottom:10px;">{_badge(swg_signal["signal"], swg_color)}'
                f'  <span style="color:{DIM}; font-size:0.8rem;">{entry_txt} | confidence {swg_signal["confidence"]}%</span></div>',
                unsafe_allow_html=True,
            )
            for reason in swg_signal.get("reasons", []):
                color = GREEN if reason.startswith("✓") else (RED if reason.startswith("✗") else DIM)
                st.markdown(
                    f'<div style="color:{color}; font-size:0.78rem; margin:2px 0;">{reason}</div>',
                    unsafe_allow_html=True,
                )

            # Gates table
            st.markdown(f'<div style="color:{DIM}; font-size:0.72rem; margin-top:10px;">GATES</div>', unsafe_allow_html=True)
            swg_gates_df = pd.DataFrame(swg_signal.get("gates", []))
            if not swg_gates_df.empty:
                def _swg_gate_style(row):
                    color = "rgba(0,255,136,0.08)" if row.get("passed") else "rgba(255,51,85,0.08)"
                    return [f"background-color:{color}"] * len(row)
                st.dataframe(
                    swg_gates_df.style.apply(_swg_gate_style, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

        # Timestamp
        st.markdown(
            f'<div style="color:{DIM}; font-size:0.7rem; margin-top:16px;">'
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} CEST</div>',
            unsafe_allow_html=True,
        )
