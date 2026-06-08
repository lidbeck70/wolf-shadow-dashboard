import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

from ui.theme import section_title
from ui.charts import build_equity_chart, build_drawdown_chart, build_monthly_heatmap
from utils.presets import SECTOR_ETF_LIST, PRESET_LABELS, PRESET_PARAMS_BT, etf_from_display, resolve_preset_key
from utils.bd_api import BDClient, load_api_key

# Module-level Börsdata client — shared across all calls in this tab.
# Falls back to yfinance transparently when bd returns None.
bd = BDClient(load_api_key())

try:
    from long_trend.long_trend_streamlit import render_long_trend_page
    _LONG_TREND_AVAILABLE = True
except ImportError:
    _LONG_TREND_AVAILABLE = False

try:
    from rs_backtest.rs_backtest_streamlit import render_rs_backtest_page
    _RS_BACKTEST_AVAILABLE = True
except ImportError:
    _RS_BACKTEST_AVAILABLE = False

try:
    from backtest_engine import run_batch_backtest, run_backtest
    _BACKTEST_ENGINE_AVAILABLE = True
except ImportError:
    _BACKTEST_ENGINE_AVAILABLE = False

try:
    from ticker_universe import COUNTRY_REGIONS as _TU_REGIONS
    _TICKER_UNIVERSE_AVAILABLE = True
except ImportError:
    _TU_REGIONS = {}
    _TICKER_UNIVERSE_AVAILABLE = False


def _render_sl_tp_calculator(strategy: str = "swing"):
    """SL/TP calculator based on strategy rules."""
    _CYAN = "#c9a84c"
    _GREEN = "#2d8a4e"
    _RED = "#c44545"
    _YELLOW = "#d4943a"
    _TEXT = "#e8e4dc"
    _DIM = "#8a8578"
    _BG2 = "#14141e"

    st.markdown(
        f"<div style='color:{_CYAN};font-size:0.75rem;text-transform:uppercase;"
        f"letter-spacing:0.1em;margin:16px 0 8px 0;border-top:1px solid rgba(201,168,76,0.1);"
        f"padding-top:12px;'>SL / TP KALKYLATOR — {strategy.upper()}</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        calc_ticker = st.text_input("Ticker", value="VOLV-B.ST", key=f"sltp_ticker_{strategy}")
    with c2:
        capital = st.number_input("Kapital (SEK)", value=100000, step=10000, key=f"sltp_cap_{strategy}")
    with c3:
        risk_pct = st.number_input("Risk %", value=5.0, min_value=0.5, max_value=10.0, step=0.5, key=f"sltp_risk_{strategy}")

    if st.button("BERÄKNA", key=f"sltp_calc_{strategy}", width='stretch'):
        try:
            _ticker_clean = calc_ticker.strip()
            df = bd.get_price_history(_ticker_clean, period="3m")
            if df is None or df.empty:
                # Fallback: yfinance
                tk = yf.Ticker(_ticker_clean)
                df = tk.history(period="3mo", auto_adjust=True)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

            if df is None or df.empty or len(df) < 20:
                st.error("Kunde inte hämta data")
                return

            close = df["Close"].astype(float)
            high = df["High"].astype(float)
            low = df["Low"].astype(float)

            price = float(close.iloc[-1])

            tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            half_atr = atr / 2

            ema10 = float(close.ewm(span=10).mean().iloc[-1])
            ema20 = float(close.ewm(span=20).mean().iloc[-1])
            ema50 = float(close.ewm(span=50).mean().iloc[-1])

            kijun = float((high.rolling(26).max() + low.rolling(26).min()).iloc[-1] / 2)

            sl_atr = price - half_atr
            sl_kijun = kijun

            if strategy == "swing":
                sl = max(sl_atr, sl_kijun)
                sl_method = f"½ ATR ({sl_atr:.2f}) / Kijun ({sl_kijun:.2f}) → tightest"
                trail = f"Kijun-sen trail ({kijun:.2f}) + EMA 10 ({ema10:.2f})"
            else:
                sl = sl_atr
                sl_method = f"½ ATR = {half_atr:.2f} under entry"
                trail = f"EMA 10 trail ({ema10:.2f})"

            sl_distance = price - sl
            tp_2r = price + sl_distance * 2
            tp_3r = price + sl_distance * 3

            risk_amount = capital * (risk_pct / 100)
            shares = int(risk_amount / sl_distance) if sl_distance > 0 else 0
            position_value = shares * price
            position_pct = (position_value / capital * 100) if capital > 0 else 0

            r1, r2 = st.columns(2)

            with r1:
                st.markdown(
                    f'<div style="background:{_BG2};border:2px solid rgba(196,69,69,0.3);border-radius:8px;padding:14px;">'
                    f'<div style="color:{_RED};font-weight:700;font-size:0.85rem;margin-bottom:8px;">STOP LOSS</div>'
                    f'<div style="color:{_TEXT};font-size:1.2rem;font-weight:700;">{sl:.2f}</div>'
                    f'<div style="color:{_DIM};font-size:0.68rem;margin-top:4px;">{sl_method}</div>'
                    f'<div style="color:{_RED};font-size:0.75rem;margin-top:6px;">Risk: {sl_distance:.2f} ({sl_distance/price*100:.1f}%)</div>'
                    f'<div style="color:{_DIM};font-size:0.65rem;margin-top:8px;">Trailing: {trail}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with r2:
                st.markdown(
                    f'<div style="background:{_BG2};border:2px solid rgba(45,138,78,0.3);border-radius:8px;padding:14px;">'
                    f'<div style="color:{_GREEN};font-weight:700;font-size:0.85rem;margin-bottom:8px;">TARGETS (R:R)</div>'
                    f'<div style="color:{_TEXT};font-size:0.85rem;">2R: <span style="color:{_GREEN};font-weight:700;">{tp_2r:.2f}</span> (+{(tp_2r/price-1)*100:.1f}%)</div>'
                    f'<div style="color:{_TEXT};font-size:0.85rem;">3R: <span style="color:{_GREEN};font-weight:700;">{tp_3r:.2f}</span> (+{(tp_3r/price-1)*100:.1f}%)</div>'
                    f'<div style="color:{_DIM};font-size:0.65rem;margin-top:6px;">Obs: Trailing stop (ej fast TP) enl. strategi</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div style="background:{_BG2};border:2px solid rgba(201,168,76,0.2);border-radius:8px;padding:14px;margin-top:8px;">'
                f'<div style="color:{_CYAN};font-weight:700;font-size:0.85rem;margin-bottom:8px;">POSITION SIZING</div>'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<div><span style="color:{_DIM};font-size:0.7rem;">Aktier</span><br>'
                f'<span style="color:{_TEXT};font-size:1.1rem;font-weight:700;">{shares}</span></div>'
                f'<div><span style="color:{_DIM};font-size:0.7rem;">Position</span><br>'
                f'<span style="color:{_TEXT};font-size:1.1rem;font-weight:700;">{position_value:,.0f} SEK</span></div>'
                f'<div><span style="color:{_DIM};font-size:0.7rem;">% av kapital</span><br>'
                f'<span style="color:{_YELLOW};font-size:1.1rem;font-weight:700;">{position_pct:.1f}%</span></div>'
                f'<div><span style="color:{_DIM};font-size:0.7rem;">Risk belopp</span><br>'
                f'<span style="color:{_RED};font-size:1.1rem;font-weight:700;">{risk_amount:,.0f} SEK</span></div>'
                f'</div>'
                f'<div style="color:{_DIM};font-size:0.62rem;margin-top:8px;">'
                f'ATR(14): {atr:.2f} | ½ ATR: {half_atr:.2f} | EMA10: {ema10:.2f} | EMA20: {ema20:.2f} | Kijun: {kijun:.2f}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Fel: {e}")


def tab_backtest():
    section_title("Strategy Backtester — Historical Performance Analysis")

    col1, col2, col3, col4 = st.columns([1.2, 0.8, 1, 1.2])

    with col1:
        ticker_input = st.text_input(
            "TICKER SYMBOL",
            value="XOM",
            key="bt_ticker",
            placeholder="e.g. XOM, NVDA, HEXA-B.ST",
        ).strip().upper()

    with col2:
        years_opt = st.selectbox(
            "BACKTEST PERIOD",
            [1, 2, 3, 5],
            index=2,
            format_func=lambda x: f"{x} Year{'s' if x > 1 else ''}",
            key="bt_years",
        )

    with col3:
        sector_etf_display = st.selectbox(
            "SECTOR ETF (REGIME)",
            SECTOR_ETF_LIST,
            key="bt_sector",
        )
        sector_etf = etf_from_display(sector_etf_display)

    with col4:
        bt_preset = st.selectbox(
            "PRESET",
            PRESET_LABELS,
            key="bt_preset",
            help="Select parameter preset or Auto-detect based on ticker",
        )

    run_btn = st.button("▶ RUN BACKTEST", key="bt_run", width='content')
    st.markdown("---")

    if run_btn:
        if not ticker_input:
            st.warning("Please enter a ticker symbol.")
            return

        _bt_pkey = resolve_preset_key(bt_preset, ticker_input)
        _bt_p = PRESET_PARAMS_BT.get(_bt_pkey, PRESET_PARAMS_BT["Universal"])
        V2_CONFIG = {
            "atr_mult":       _bt_p["atr_mult"],
            "adx_threshold":  _bt_p["adx_thresh"],
            "tp1_r":          _bt_p["tp1_r"],
            "tp1_pct":        _bt_p["tp1_pct"],
            "tp2_r":          _bt_p["tp2_r"],
            "tp2_pct":        _bt_p["tp2_pct"],
            "daily_breaker":  -0.08,
            "core_exit_bars": 3,
            "trail_exit":     "kijun_ema10",
        }

        _use_module = False
        try:
            import wolf_shadow_backtest as _bt_mod
            if hasattr(_bt_mod, "CONFIG"):
                _bt_mod.CONFIG["atr_mult"]       = V2_CONFIG["atr_mult"]
                _bt_mod.CONFIG["adx_threshold"]  = V2_CONFIG["adx_threshold"]
                _bt_mod.CONFIG["tp1_rr"]         = V2_CONFIG["tp1_r"]
                _bt_mod.CONFIG["tp1_pct"]        = V2_CONFIG["tp1_pct"]
                _bt_mod.CONFIG["tp2_rr"]         = V2_CONFIG["tp2_r"]
                _bt_mod.CONFIG["tp2_pct"]        = V2_CONFIG["tp2_pct"]
                _bt_mod.CONFIG["daily_breaker"]  = V2_CONFIG["daily_breaker"]
                _bt_mod.CONFIG["core_exit_bars"] = V2_CONFIG["core_exit_bars"]
            from wolf_shadow_backtest import (
                add_indicators, calc_regime_scores,
                fetch_regime_data, Backtester, validate_criteria,
                ACCEPT_CRITERIA,
            )
            _use_module = True
        except ImportError:
            pass

        with st.spinner(f"🐺 Running backtest for {ticker_input} ({years_opt}y)..."):
            try:
                # Request a wider window than needed so EMA200 warmup bars are available.
                # The inline backtest trims to years_opt * 252 bars later anyway.
                _bd_period = {1: "2y", 2: "3y", 3: "5y", 5: "10y"}.get(years_opt, f"{years_opt + 1}y")
                df_raw = bd.get_price_history(ticker_input, period=_bd_period)

                if df_raw is None or df_raw.empty:
                    # Fallback: yfinance
                    end = datetime.now()
                    start = end - timedelta(days=years_opt * 365 + 100)
                    df_raw = yf.download(ticker_input, start=start, end=end,
                                         progress=False, auto_adjust=True)
                    if isinstance(df_raw.columns, pd.MultiIndex):
                        df_raw.columns = df_raw.columns.get_level_values(0)

                if df_raw is None or len(df_raw) < 50:
                    st.error(f"Insufficient data for {ticker_input} ({len(df_raw) if df_raw is not None else 0} bars). Need at least 50 bars.")
                    return

                if _use_module and len(df_raw) >= 250:
                    spy_df, sec_df = fetch_regime_data("SPY", sector_etf, years_opt)
                    df_proc = add_indicators(df_raw.copy())
                    df_proc = calc_regime_scores(spy_df, sec_df, df_proc)
                    df_proc = df_proc.dropna()

                    bt = Backtester(df_proc)
                    results = bt.run()

                    if results is None:
                        st.warning("No trades were generated for this period/ticker.")
                        return

                else:
                    if not _use_module:
                        st.info("⚠️ Backtester module not found — using simplified inline backtest with v2 parameters.")

                    df = df_raw.copy()
                    c = df["Close"]

                    df["ema10"]  = c.ewm(span=10,  adjust=False).mean()
                    df["ema21"]  = c.ewm(span=21,  adjust=False).mean()
                    df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
                    df["ema200"] = c.ewm(span=200, adjust=False).mean()

                    delta = c.diff()
                    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
                    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
                    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

                    tr = pd.concat([
                        df["High"] - df["Low"],
                        (df["High"] - c.shift()).abs(),
                        (df["Low"]  - c.shift()).abs(),
                    ], axis=1).max(axis=1)
                    df["atr"] = tr.ewm(com=13, adjust=False).mean()

                    df["kijun"] = (df["High"].rolling(26).max() + df["Low"].rolling(26).min()) / 2

                    df.dropna(inplace=True)
                    df = df.iloc[years_opt * -252:] if len(df) > years_opt * 252 else df

                    # Sentiment size multiplier — computed once on recent bars.
                    # At aggregate_score=50 (neutral) multiplier=1.0 (no change).
                    # Range: [0.80, 1.20] at score extremes (0 / 100).
                    _sent_size_mult = 1.0
                    try:
                        from strategies.sentiment_utils import compute_sentiment_bias
                        _wolf_plugins = ["ovtlyr_fg", "retail_flow"]
                        _sent_window  = df_raw.iloc[-60:] if len(df_raw) > 60 else df_raw
                        _sent_pre     = compute_sentiment_bias(_sent_window, _wolf_plugins)
                        if _sent_pre["available"]:
                            _sent_size_mult = max(0.80, min(1.20,
                                1.0 + (_sent_pre["aggregate_score"] - 50.0) / 250.0))
                    except Exception:
                        pass

                    capital   = 100_000.0
                    equity    = [capital]
                    trades_list = []
                    position  = None
                    atr_m     = V2_CONFIG["atr_mult"]
                    tp1_r     = V2_CONFIG["tp1_r"]
                    tp1_pct   = V2_CONFIG["tp1_pct"]
                    tp2_r     = V2_CONFIG["tp2_r"]
                    tp2_pct   = V2_CONFIG["tp2_pct"]
                    daily_brk = V2_CONFIG["daily_breaker"]
                    core_bars = V2_CONFIG["core_exit_bars"]

                    bars_below_ema50 = 0
                    daily_block      = False

                    for i in range(1, len(df)):
                        row   = df.iloc[i]
                        prev  = df.iloc[i - 1]
                        price = float(row["Close"])

                        day_ret = (price / float(prev["Close"]) - 1)
                        daily_block = day_ret <= daily_brk

                        if position is not None:
                            if price < float(row["ema50"]):
                                bars_below_ema50 += 1
                            else:
                                bars_below_ema50 = 0

                            pos = position
                            exit_price = None
                            exit_reason = ""

                            if not pos.get("tp1_hit") and price >= pos["tp1"]:
                                trim_shares = pos["shares"] * tp1_pct
                                capital += trim_shares * pos["tp1"]
                                pos["shares"] -= trim_shares
                                pos["tp1_hit"] = True

                            if not pos.get("tp2_hit") and price >= pos["tp2"]:
                                trim_shares = pos["shares"] * tp2_pct
                                capital += trim_shares * pos["tp2"]
                                pos["shares"] -= trim_shares
                                pos["tp2_hit"] = True

                            if price <= pos["sl"]:
                                exit_price  = pos["sl"]
                                exit_reason = "STOP"
                            elif bars_below_ema50 >= core_bars:
                                exit_price  = price
                                exit_reason = "EMA50_EXIT"
                            elif price < float(row["kijun"]) and price < float(row["ema10"]):
                                exit_price  = price
                                exit_reason = "TRAIL_EXIT"

                            if exit_price is not None:
                                capital += pos["shares"] * exit_price
                                pnl = capital - pos["capital_at_entry"]
                                pnl_pct = (exit_price / pos["entry"] - 1) * 100
                                trades_list.append({
                                    "entry_date":  pos["entry_date"],
                                    "exit_date":   df.index[i],
                                    "entry_price": pos["entry"],
                                    "exit_price":  exit_price,
                                    "pnl":         round(pnl, 2),
                                    "pnl_pct":     round(pnl_pct, 2),
                                    "exit_reason": exit_reason,
                                    "bars_held":   i - pos["entry_idx"],
                                })
                                position = None
                                bars_below_ema50 = 0

                        if (
                            position is None
                            and not daily_block
                            and float(row["ema10"]) > float(row["ema21"]) > float(row["ema50"]) > float(row["ema200"])
                            and 45 < float(row["rsi"]) < 70
                            and price > float(row["ema50"])
                        ):
                            atr_v = float(row["atr"])
                            sl    = price - atr_m * atr_v
                            risk  = price - sl
                            shares = (capital * 0.02 * _sent_size_mult) / risk if risk > 0 else 0
                            if shares > 0:
                                cost = shares * price
                                if cost <= capital:
                                    capital -= cost
                                    position = {
                                        "entry":            price,
                                        "entry_date":       df.index[i],
                                        "entry_idx":        i,
                                        "shares":           shares,
                                        "sl":               sl,
                                        "tp1":              price + tp1_r * risk,
                                        "tp2":              price + tp2_r * risk,
                                        "tp1_hit":          False,
                                        "tp2_hit":          False,
                                        "capital_at_entry": capital + cost,
                                    }

                        equity.append(capital + (position["shares"] * price if position else 0))

                    if position is not None:
                        final_price = float(df.iloc[-1]["Close"])
                        capital += position["shares"] * final_price
                        pnl = capital - position["capital_at_entry"]
                        trades_list.append({
                            "entry_date":  position["entry_date"],
                            "exit_date":   df.index[-1],
                            "entry_price": position["entry"],
                            "exit_price":  final_price,
                            "pnl":         round(pnl, 2),
                            "pnl_pct":     round((final_price / position["entry"] - 1) * 100, 2),
                            "exit_reason": "END",
                            "bars_held":   len(df) - position["entry_idx"],
                        })

                    trades = pd.DataFrame(trades_list)
                    eq_arr = np.array(equity)
                    eq_idx = df.index[:len(eq_arr)]
                    if len(eq_arr) > len(eq_idx):
                        eq_arr = eq_arr[:len(eq_idx)]
                    eq_df  = pd.DataFrame({"equity": eq_arr}, index=eq_idx)

                    peak  = eq_df["equity"].cummax()
                    dd_ser = (eq_df["equity"] - peak) / peak

                    total_ret  = (eq_arr[-1] / 100_000 - 1) * 100
                    daily_ret  = eq_df["equity"].pct_change().dropna()
                    returns    = daily_ret

                    if len(trades) > 0:
                        wins     = trades[trades["pnl"] > 0]
                        losses   = trades[trades["pnl"] <= 0]
                        winrate  = len(wins) / len(trades) * 100
                        avg_win  = wins["pnl"].mean() if len(wins) > 0 else 0
                        avg_loss = losses["pnl"].mean() if len(losses) > 0 else -1
                        pf       = abs(wins["pnl"].sum() / losses["pnl"].sum()) if losses["pnl"].sum() != 0 else 99
                        avg_rr   = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                        avg_bars = trades["bars_held"].mean() if "bars_held" in trades.columns else 0
                    else:
                        winrate = pf = avg_rr = avg_bars = 0

                    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
                    neg    = daily_ret[daily_ret < 0]
                    sortino = float(daily_ret.mean() / neg.std() * np.sqrt(252)) if len(neg) > 1 and neg.std() > 0 else 0
                    max_dd = float(dd_ser.min() * 100)
                    cagr   = ((eq_arr[-1] / 100_000) ** (252 / max(len(eq_arr), 1)) - 1) * 100
                    calmar = abs(cagr / max_dd) if max_dd < 0 else 0

                    metrics = {
                        "Total Return %":  round(total_ret, 2),
                        "Sharpe Ratio":    round(sharpe,  2),
                        "Profit Factor":   round(pf,      2),
                        "Winrate %":       round(winrate, 1),
                        "Max Drawdown %":  round(max_dd,  2),
                        "Calmar Ratio":    round(calmar,  2),
                        "CAGR %":          round(cagr,    2),
                        "Sortino Ratio":   round(sortino, 2),
                        "Avg R:R":         round(avg_rr,  2),
                        "Total Trades":    len(trades),
                        "Avg Bars Held":   round(avg_bars, 1),
                        "Final Equity":    round(float(eq_arr[-1]), 0),
                    }

                    results = {
                        "metrics":  metrics,
                        "equity":   eq_df,
                        "drawdown": dd_ser,
                        "trades":   trades,
                        "returns":  returns,
                    }

            except Exception as e:
                st.error(f"Backtest error: {e}")
                import traceback; st.code(traceback.format_exc())
                return

        metrics = results["metrics"]
        eq_df   = results["equity"]
        dd_ser  = results["drawdown"]
        trades  = results["trades"]
        returns = results["returns"]

        # ── Alerts ───────────────────────────────────────────────────────────
        try:
            from alerts.engine import send_alert as _send_alert
            from strategies.registry import STRATEGIES as _STRATS
            from strategies.sentiment_utils import compute_sentiment_bias as _csb

            _strat     = _STRATS.get("Wolf") or _STRATS.get("wolf") or {}
            _alerts_on = _strat.get("alerts_enabled", False)
            _channels  = _strat.get("alert_channels", ["discord"])
            _plugins   = _strat.get("sentiment_plugins", ["ovtlyr_fg", "retail_flow"])

            if _alerts_on:
                # Entry / exit alerts — only for trades in the last 5 calendar days
                if not trades.empty and "exit_date" in trades.columns:
                    _last_date = pd.to_datetime(df_raw.index[-1])
                    _cutoff    = _last_date - pd.Timedelta(days=5)
                    _recent    = trades[pd.to_datetime(trades["exit_date"]) >= _cutoff]
                    for _, _tr in _recent.iterrows():
                        _reason  = str(_tr.get("exit_reason", "EXIT"))
                        _pnl_pct = float(_tr.get("pnl_pct", 0.0))
                        _send_alert(
                            f"Wolf x Shadow EXIT — {ticker_input} | {_reason} | P&L: {_pnl_pct:+.1f}%",
                            _channels,
                            metadata={
                                "ticker": ticker_input,
                                "signal": _reason,
                                "title":  f"Wolf — {ticker_input}",
                                "color":  0x2D8A4E if _pnl_pct > 0 else 0xC44545,
                            },
                        )

                # Sentiment extreme alert
                _sent_w = df_raw.iloc[-60:] if len(df_raw) > 60 else df_raw
                _sent_r = _csb(_sent_w, _plugins)
                if _sent_r["available"]:
                    for _pkey, _psig in _sent_r["plugins"].items():
                        _sc = float(_psig.get("score", 50.0))
                        if _sc > 90 or _sc < 10:
                            _direction = "EXTREME GREED" if _sc > 90 else "EXTREME FEAR"
                            _send_alert(
                                f"Sentiment extreme — {ticker_input} [{_pkey}] {_sc:.0f} ({_direction})",
                                _channels,
                                metadata={
                                    "ticker": ticker_input,
                                    "signal": "sentiment_extreme",
                                    "plugin": _pkey,
                                    "score":  _sc,
                                    "title":  f"Sentiment Extreme — {ticker_input}",
                                    "color":  0xC9A84C,
                                },
                            )

                # Regime shift alert (EMA stack state vs. previous run)
                _c    = df_raw["Close"].astype(float)
                _e10  = float(_c.ewm(span=10,  adjust=False).mean().iloc[-1])
                _e21  = float(_c.ewm(span=21,  adjust=False).mean().iloc[-1])
                _e50  = float(_c.ewm(span=50,  adjust=False).mean().iloc[-1])
                _e200 = float(_c.ewm(span=200, adjust=False).mean().iloc[-1])
                _bull_now  = bool(_e10 > _e21 > _e50 > _e200)
                _rkey      = f"bt_regime_{ticker_input}"
                _bull_prev = st.session_state.get(_rkey)
                st.session_state[_rkey] = _bull_now

                if _bull_prev is not None and _bull_prev != _bull_now:
                    _shift = "BULL → BEAR" if _bull_prev else "BEAR → BULL"
                    _send_alert(
                        f"Regime shift — {ticker_input} {_shift} (EMA stack)",
                        _channels,
                        metadata={
                            "ticker": ticker_input,
                            "signal": "regime_shift",
                            "shift":  _shift,
                            "title":  f"Regime Shift — {ticker_input}",
                            "color":  0xC44545 if _bull_prev else 0x2D8A4E,
                        },
                    )
        except Exception:
            pass  # alerts are optional — never block the UI

        section_title("Performance Metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)

        ret_val = metrics["Total Return %"]
        sharpe  = metrics["Sharpe Ratio"]
        pf      = metrics["Profit Factor"]
        winrate = metrics["Winrate %"]
        maxdd   = metrics["Max Drawdown %"]
        calmar  = metrics["Calmar Ratio"]

        m1.metric("TOTAL RETURN",  f"{ret_val:+.1f}%",  delta="▲ PROFIT" if ret_val > 0 else "▼ LOSS")
        m2.metric("SHARPE RATIO",  f"{sharpe:.2f}",     delta="✓ PASS" if sharpe >= 1.5 else "✗ FAIL")
        m3.metric("PROFIT FACTOR", f"{pf:.2f}",         delta="✓ PASS" if pf >= 1.5 else "✗ FAIL")
        m4.metric("WIN RATE",      f"{winrate:.1f}%",   delta="✓ PASS" if winrate >= 45 else "✗ FAIL")
        m5.metric("MAX DRAWDOWN",  f"{maxdd:.1f}%",     delta="✓ PASS" if maxdd >= -15 else "✗ FAIL")
        m6.metric("CALMAR RATIO",  f"{calmar:.2f}",     delta="✓ PASS" if calmar >= 1.0 else "✗ FAIL")

        x1, x2, x3, x4, x5, x6 = st.columns(6)
        x1.metric("CAGR",          f"{metrics['CAGR %']:.1f}%")
        x2.metric("SORTINO",       f"{metrics['Sortino Ratio']:.2f}")
        x3.metric("AVG R:R",       f"{metrics['Avg R:R']:.2f}")
        x4.metric("TOTAL TRADES",  f"{metrics['Total Trades']}")
        x5.metric("AVG BARS HELD", f"{metrics['Avg Bars Held']:.1f}")
        x6.metric("FINAL EQUITY",  f"${metrics['Final Equity']:,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)

        section_title("Equity & Drawdown")
        st.plotly_chart(build_equity_chart(eq_df, ticker_input),
                        width='stretch', config={"displayModeBar": False})
        st.plotly_chart(build_drawdown_chart(dd_ser),
                        width='stretch', config={"displayModeBar": False})

        section_title("Monthly Returns")
        if len(returns) > 20:
            st.plotly_chart(build_monthly_heatmap(returns, eq_df),
                            width='stretch', config={"displayModeBar": False})
        else:
            st.info("Insufficient data for monthly heatmap.")

        section_title("Accept Criteria Validation")

        if not _use_module or "validate_criteria" not in dir():
            _thresholds = [
                ("Total Return %",  metrics.get("Total Return %", 0),  ">= 20%",  lambda v: v >= 20),
                ("Sharpe Ratio",    metrics.get("Sharpe Ratio",   0),  ">= 1.5",  lambda v: v >= 1.5),
                ("Profit Factor",   metrics.get("Profit Factor",  0),  ">= 1.5",  lambda v: v >= 1.5),
                ("Winrate %",       metrics.get("Winrate %",      0),  ">= 45%",  lambda v: v >= 45),
                ("Max Drawdown %",  metrics.get("Max Drawdown %", 0),  ">= -15%", lambda v: v >= -15),
                ("Calmar Ratio",    metrics.get("Calmar Ratio",   0),  ">= 1.0",  lambda v: v >= 1.0),
            ]
            criteria_df = pd.DataFrame([
                {"Metric": m, "Value": round(float(v), 2), "Threshold": t, "Status": "PASS" if fn(v) else "FAIL"}
                for m, v, t, fn in _thresholds
            ])
        else:
            criteria_df = validate_criteria(metrics)

        passed_count = len(criteria_df[criteria_df["Status"] == "PASS"])
        total_count = len(criteria_df)
        st.write(f"**Result: {passed_count}/{total_count} PASSED**")

        def color_criteria(row):
            if row["Status"] == "PASS":
                return ["background-color: rgba(0,180,80,0.15); color: #00cc66"] * len(row)
            return ["background-color: rgba(200,50,80,0.15); color: #ff6666"] * len(row)

        st.dataframe(
            criteria_df.style.apply(color_criteria, axis=1),
            width='stretch',
            hide_index=True,
            height=280,
        )

        if not trades.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            section_title(f"Trade Log ({len(trades)} trades)")

            trade_display = trades.copy()
            if "entry_date" in trade_display.columns:
                trade_display["entry_date"] = pd.to_datetime(trade_display["entry_date"]).dt.strftime("%Y-%m-%d")
            if "exit_date" in trade_display.columns:
                trade_display["exit_date"] = pd.to_datetime(trade_display["exit_date"]).dt.strftime("%Y-%m-%d")

            for col in ["entry_price", "exit_price", "pnl"]:
                if col in trade_display.columns:
                    trade_display[col] = trade_display[col].round(2)
            if "pnl_pct" in trade_display.columns:
                trade_display["pnl_pct"] = trade_display["pnl_pct"].round(2)

            def style_trades(row):
                if "pnl" in row.index:
                    if row["pnl"] > 0:
                        return ["background-color: rgba(45,138,78,0.07); color: #2d8a4e"] * len(row)
                    else:
                        return ["background-color: rgba(196,69,69,0.07); color: #ff6688"] * len(row)
                return [""] * len(row)

            st.dataframe(
                trade_display.style.apply(style_trades, axis=1)
                .set_properties(**{
                    "font-family": "Courier New, monospace",
                    "font-size": "11px",
                }),
                width='stretch',
                hide_index=True,
                height=350,
            )

        # ── Current Sentiment ────────────────────────────────────────────────
        try:
            from strategies.sentiment_utils import compute_sentiment_bias
            from sentiment.registry import SENTIMENT_PLUGINS

            # Show all three plugins in the display regardless of strategy
            _SENT_PLUGINS = ["ovtlyr_fg", "retail_flow", "options_flow"]
            _sent_df = df_raw.iloc[-60:] if len(df_raw) > 60 else df_raw
            _sent = compute_sentiment_bias(_sent_df, _SENT_PLUGINS)
            # Re-use the multiplier computed earlier for the size annotation
            _disp_mult = _sent_size_mult if "_sent_size_mult" in dir() else 1.0

            if _sent["available"]:
                st.markdown("<br>", unsafe_allow_html=True)
                section_title(f"Current Sentiment — {ticker_input}")

                _CYAN = "#c9a84c"
                _GREEN = "#2d8a4e"
                _RED = "#c44545"
                _DIM = "#8a8578"
                _BG2 = "#14141e"

                def _bias_color(bias: str) -> str:
                    return _GREEN if bias == "bullish" else (_RED if bias == "bearish" else _DIM)

                def _score_bar(score: float, color: str) -> str:
                    pct = int(score)
                    return (
                        f'<div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px;margin-top:4px;">'
                        f'<div style="width:{pct}%;background:{color};height:6px;border-radius:4px;"></div>'
                        f'</div>'
                    )

                _agg = _sent["aggregate_score"]
                _agg_bias = _sent["aggregate_bias"]
                _agg_color = _bias_color(_agg_bias)

                st.markdown(
                    f'<div style="background:{_BG2};border:1px solid rgba(201,168,76,0.2);'
                    f'border-radius:8px;padding:12px 16px;margin-bottom:12px;">'
                    f'<span style="color:{_DIM};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;">Aggregate</span>'
                    f'<span style="float:right;color:{_agg_color};font-weight:700;">{_agg_bias.upper()}</span>'
                    f'<div style="color:{_agg_color};font-size:1.5rem;font-weight:700;margin-top:2px;">{_agg:.1f}</div>'
                    f'{_score_bar(_agg, _agg_color)}'
                    f'<div style="color:{_DIM};font-size:0.65rem;margin-top:4px;">'
                    f'Confidence: {_sent["confidence"]:.1%}'
                    f'&nbsp;&nbsp;|&nbsp;&nbsp;Size mult: <span style="color:{_agg_color};font-weight:600;">'
                    f'{_disp_mult:.2f}×</span> (Wolf strategy)</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                _plugin_cols = st.columns(len(_SENT_PLUGINS))
                for _col, _pkey in zip(_plugin_cols, _SENT_PLUGINS):
                    _psig = _sent["plugins"].get(_pkey)
                    _pmeta = SENTIMENT_PLUGINS.get(_pkey, {})
                    if _psig is None:
                        continue
                    _ps = _psig.get("score", 50.0)
                    _pb = _psig.get("bias", "neutral")
                    _pc = _bias_color(_pb)
                    _pname = _pmeta.get("name", _pkey)
                    _plabel = _psig.get("label", "")
                    with _col:
                        st.markdown(
                            f'<div style="background:{_BG2};border:1px solid rgba(255,255,255,0.07);'
                            f'border-radius:8px;padding:10px 12px;">'
                            f'<div style="color:{_DIM};font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;">{_pname}</div>'
                            f'<div style="color:{_pc};font-size:1.2rem;font-weight:700;margin-top:2px;">{_ps:.0f}</div>'
                            f'{_score_bar(_ps, _pc)}'
                            f'<div style="color:{_pc};font-size:0.68rem;margin-top:4px;">{_plabel}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                _comps_all = {}
                for _pkey in _SENT_PLUGINS:
                    _psig = _sent["plugins"].get(_pkey)
                    if _psig and _psig.get("components"):
                        for _k, _v in _psig["components"].items():
                            _comps_all[f"{_pkey}/{_k}"] = _v
                if _comps_all:
                    with st.expander("Component breakdown", expanded=False):
                        _comp_rows = [{"Component": k, "Score": round(v, 1)} for k, v in _comps_all.items()]
                        st.dataframe(pd.DataFrame(_comp_rows), hide_index=True, width='stretch')

        except Exception:
            pass  # sentiment display is optional — never break backtest

    else:
        st.markdown("""
        <div style="
            text-align:center;
            padding:80px 20px;
            border:1px dashed rgba(139,115,64,0.15);
            border-radius:8px;
            margin-top:20px;
        ">
            <div style="font-size:48px;margin-bottom:16px;">📊</div>
            <div style="color:rgba(139,115,64,0.5);letter-spacing:4px;font-size:14px;margin-bottom:8px;">
                AWAITING COMMAND
            </div>
            <div style="color:rgba(139,115,64,0.3);font-size:11px;letter-spacing:2px;">
                ENTER TICKER · SELECT PERIOD · RUN BACKTEST
            </div>
        </div>
        """, unsafe_allow_html=True)


def _render_ovtlyr_backtest_ui():
    """OVTLYR backtest with Test Top N integration."""
    st.markdown(
        "<h2 style='color:#c9a84c;letter-spacing:0.1em;'>"
        "VIKING BACKTEST</h2>"
        "<p style='color:#8a8578;font-size:0.7rem;'>EMA 10/20 crossover + ADX filter + Volume confirmation</p>",
        unsafe_allow_html=True,
    )

    auto_run = st.session_state.pop("auto_run_backtest", False)
    auto_tickers = st.session_state.get("test_topn_tickers", [])
    auto_mode = st.session_state.get("test_topn_mode", "ovtlyr")

    col1, col2 = st.columns(2)
    with col1:
        ticker_input = st.text_input(
            "Tickers (comma-separated)",
            value=", ".join(auto_tickers) if auto_tickers else "VOLV-B.ST, EQNR.OL, BOL.ST",
            key="ovtlyr_bt_tickers",
        )
    with col2:
        years = st.selectbox("Period", [1, 2, 3, 5], index=2, key="ovtlyr_bt_years")

    run_bt = st.button("▶  RUN BACKTEST", key="ovtlyr_bt_run", width='stretch')

    if run_bt or auto_run:
        if not _BACKTEST_ENGINE_AVAILABLE:
            st.error("Backtest engine not found.")
            return

        tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
        if not tickers:
            st.warning("Enter at least one ticker.")
            return

        with st.spinner(f"🐺 Backtesting {len(tickers)} tickers ({years}y)..."):
            summary = run_batch_backtest(tickers, years, auto_mode)

        if summary.empty:
            st.warning("No backtest results. Check tickers or period.")
            return

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tickers Tested", len(summary))
        k2.metric("Avg Return", f"{summary['Total Return %'].mean():.1f}%")
        k3.metric("Avg Win Rate", f"{summary['Win Rate %'].mean():.1f}%")
        k4.metric("Avg Max DD", f"{summary['Max DD %'].mean():.1f}%")

        def _ret_color(val):
            try:
                v = float(val)
                return "color:#2d8a4e" if v > 0 else "color:#c44545"
            except (TypeError, ValueError):
                return ""

        styled = summary.style
        _map = styled.map if hasattr(styled, "map") else styled.applymap
        for col in ["Total Return %", "CAGR %", "Max DD %", "Avg Return %"]:
            if col in summary.columns:
                styled = _map(_ret_color, subset=[col])
        st.dataframe(styled, width='stretch', hide_index=True)

        with st.expander("Individual Trades", expanded=False):
            for ticker in tickers[:5]:
                bt = run_backtest(ticker, years, auto_mode)
                trades = bt.get("trades", [])
                if trades:
                    st.markdown(f"**{ticker}** — {len(trades)} trades")
                    st.dataframe(pd.DataFrame(trades), width='stretch', hide_index=True)


def tab_backtest_consolidated():
    """Unified Backtest tab with dropdown: Wolf / Alpha / Viking / RS Sector."""
    from ui.css import tab_not_found

    mode = st.selectbox(
        "BACKTEST MODE",
        ["Wolf", "Alpha", "Viking", "RS Sector"],
        key="backtest_mode_select",
    )

    if mode == "Wolf":
        tab_backtest()

    elif mode == "Alpha":
        try:
            if _TICKER_UNIVERSE_AVAILABLE and _TU_REGIONS:
                region_options = list(_TU_REGIONS.keys())
                st.multiselect(
                    "Marknader",
                    region_options,
                    default=["Norden"],
                    key="alpha_backtest_markets",
                )
        except Exception:
            pass
        if _LONG_TREND_AVAILABLE:
            render_long_trend_page()
        else:
            tab_not_found("Long-Term Trend", "long_trend")

    elif mode == "RS Sector":
        if _RS_BACKTEST_AVAILABLE:
            render_rs_backtest_page()
        else:
            tab_not_found("RS Backtest", "rs_backtest")

    elif mode == "Viking":
        _render_ovtlyr_backtest_ui()
