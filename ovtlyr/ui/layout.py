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

try:
    from ovtlyr.indicators.trend      import compute_trend
    from ovtlyr.indicators.momentum   import compute_momentum
    from ovtlyr.indicators.volatility import compute_volatility
    from ovtlyr.indicators.volume     import compute_volume
    from ovtlyr.indicators.orderblocks import detect_orderblocks, analyse_orderblocks
    from ovtlyr.indicators.sentiment   import fetch_sentiment
    from ovtlyr.indicators.sector      import fetch_sector_breadth
except ImportError:
    try:
        from indicators.trend       import compute_trend       # type: ignore
        from indicators.momentum    import compute_momentum    # type: ignore
        from indicators.volatility  import compute_volatility  # type: ignore
        from indicators.volume      import compute_volume      # type: ignore
        from indicators.orderblocks import detect_orderblocks, analyse_orderblocks  # type: ignore
        from indicators.sentiment   import fetch_sentiment     # type: ignore
        from indicators.sector      import fetch_sector_breadth  # type: ignore
    except ImportError:
        compute_trend        = None
        compute_momentum     = None
        compute_volatility   = None
        compute_volume       = None
        detect_orderblocks   = None
        analyse_orderblocks  = None
        fetch_sentiment      = None
        fetch_sector_breadth = None

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

@st.cache_data(ttl=1800, show_spinner=False)
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


@st.cache_data(ttl=1800, show_spinner=False)
def _load_sentiment() -> dict:
    if fetch_sentiment is not None:
        return fetch_sentiment()
    # Dummy fallback
    return {"score": 50, "label": "Neutral"}


@st.cache_data(ttl=1800, show_spinner=False)
def _load_sector_breadth() -> dict:
    if fetch_sector_breadth is not None:
        return fetch_sector_breadth()
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

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f'<div style="color:{CYAN}; font-size:1.1rem; font-weight:700; '
            f'letter-spacing:0.1em; margin-bottom:12px;">⬡ OVTLYR</div>',
            unsafe_allow_html=True,
        )

        ticker_input = st.text_input(
            "Ticker", value=DEFAULT_TICKER, key="ovtlyr_ticker"
        ).strip().upper()

        quick_pick = st.selectbox(
            "Quick pick", ["— custom —"] + COMMON_TICKERS, key="ovtlyr_quick"
        )
        if quick_pick != "— custom —":
            ticker = quick_pick
        else:
            ticker = ticker_input if ticker_input else DEFAULT_TICKER

        period = st.selectbox(
            "Period", ["6mo", "1y", "2y"], index=1, key="ovtlyr_period"
        )

        st.markdown("---")
        refresh = st.button("↺ Refresh", key="ovtlyr_refresh", use_container_width=True)
        if refresh:
            st.cache_data.clear()

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
            except Exception:
                orderblocks = []

        if analyse_orderblocks is not None:
            try:
                ob_analysis = analyse_orderblocks(df, orderblocks, trend)
            except Exception:
                ob_analysis = {"signal_bias": "HOLD"}

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
            1 for v in breadth_data.values() if v.get("state") == "bullish"
        )
        sector_green = bullish_sectors > len(breadth_data) / 2 if breadth_data else True

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
        ema50_last  = trend["ema50"][-1]  if isinstance(trend["ema50"],  list) else trend["ema50"]
        ema200_last = trend["ema200"][-1] if isinstance(trend["ema200"], list) else trend["ema200"]
        _indicator_card(
            "Trend",
            [
                ("Direction",  trend.get("direction", "—").capitalize(), regime_color_hex),
                ("EMA 50",     f"{ema50_last:.2f}",  YELLOW),
                ("EMA 200",    f"{ema200_last:.2f}", MAGENTA),
                ("Regime",     trend.get("regime_color", "—").upper(), regime_color_hex),
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

    # ── BOTTOM TABS ───────────────────────────────────────────────────
    tab_a, tab_b, tab_c, tab_d = st.tabs(
        ["Order Blocks", "Sectors", "Drawdowns", "Signal Log"]
    )

    # ── Tab A: Order Blocks ──────────────────────────────────────────
    with tab_a:
        if orderblocks:
            ob_rows = []
            for ob in sorted(orderblocks, key=lambda x: x.get("date_start", ""), reverse=True):
                ob_rows.append({
                    "Type":    ob.get("type", "—").capitalize(),
                    "Date":    str(ob.get("date_start", "—")),
                    "High":    ob.get("high", 0),
                    "Low":     ob.get("low", 0),
                    "Volume":  int(ob.get("volume", 0)),
                    "Status":  ob.get("status", "—").capitalize(),
                })
            ob_df = pd.DataFrame(ob_rows)

            def _ob_row_style(row):
                if row["Type"].lower() == "bullish":
                    color = "rgba(0,255,136,0.08)"
                elif row["Type"].lower() == "bearish":
                    color = "rgba(255,51,85,0.08)"
                else:
                    color = "rgba(74,74,106,0.08)"
                return [f"background-color:{color}"] * len(row)

            styled_ob = ob_df.style.apply(_ob_row_style, axis=1)
            st.dataframe(styled_ob, use_container_width=True, hide_index=True)
        else:
            st.info("No order blocks detected. Run the analysis with a longer period or check indicator modules.")

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
            st.markdown(
                f'<div style="margin-bottom:10px;">{_badge(lt_signal["signal"], lt_color)}'
                f'  <span style="color:{DIM}; font-size:0.8rem;">confidence {lt_signal["confidence"]}%</span></div>',
                unsafe_allow_html=True,
            )
            for reason in lt_signal.get("reasons", []):
                color = GREEN if reason.startswith("✓") else (RED if reason.startswith("✗") else DIM)
                st.markdown(
                    f'<div style="color:{color}; font-size:0.78rem; margin:2px 0;">{reason}</div>',
                    unsafe_allow_html=True,
                )

            # Gates table
            st.markdown(f'<div style="color:{DIM}; font-size:0.72rem; margin-top:10px;">GATES</div>', unsafe_allow_html=True)
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
