import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime

from ui.theme import section_title
from ui.charts import build_gauge
from utils.presets import SECTOR_ETF_LIST, etf_from_display


def tab_regime():
    section_title("Real-Time Regime Monitor — Live Market Intelligence")

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1.5])
    with ctrl1:
        watch_ticker = st.text_input("STOCK TICKER", value="XOM", key="reg_ticker").strip().upper()
    with ctrl2:
        watch_sector_display = st.selectbox(
            "SECTOR ETF",
            SECTOR_ETF_LIST,
            key="reg_sector",
        )
        watch_sector = etf_from_display(watch_sector_display)
    with ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_btn = st.button("🔄 REFRESH REGIME", key="reg_refresh", width='stretch')

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
        key="benchmark_swing",
    )
    benchmark_ticker = _BENCHMARK_OPTIONS[selected_bm_label]

    try:
        bm_data = yf.download(benchmark_ticker, period="3mo", auto_adjust=True, progress=False)
        stk_data = yf.download(watch_ticker, period="3mo", auto_adjust=True, progress=False)
        if isinstance(bm_data.columns, pd.MultiIndex):
            bm_data.columns = bm_data.columns.get_level_values(0)
        if isinstance(stk_data.columns, pd.MultiIndex):
            stk_data.columns = stk_data.columns.get_level_values(0)
        if not bm_data.empty and not stk_data.empty and len(bm_data) >= 20 and len(stk_data) >= 20:
            stk_ret = float(stk_data["Close"].iloc[-1] / stk_data["Close"].iloc[-20])
            bm_ret = float(bm_data["Close"].iloc[-1] / bm_data["Close"].iloc[-20])
            if bm_ret > 0:
                rs = stk_ret / bm_ret
                if rs > 1.05:
                    rs_icon, rs_color = "🟢", "#2d8a4e"
                elif rs >= 0.95:
                    rs_icon, rs_color = "🟡", "#d4943a"
                else:
                    rs_icon, rs_color = "🔴", "#c44545"
                st.markdown(
                    f"<div style='padding:4px 0;'>"
                    f"<span style='font-size:0.85rem;'>{rs_icon}</span> "
                    f"<span style='color:{rs_color};font-size:0.85rem;font-weight:700;'>"
                    f"RS: {rs - 1:+.1%} vs {selected_bm_label}</span></div>",
                    unsafe_allow_html=True,
                )
    except Exception:
        pass

    st.markdown("---")

    if "regime_data" not in st.session_state:
        st.session_state.regime_data = None
    if "regime_ticker" not in st.session_state:
        st.session_state.regime_ticker = None

    should_load = (
        refresh_btn
        or st.session_state.regime_data is None
        or st.session_state.regime_ticker != watch_ticker
    )

    if should_load:
        with st.spinner(f"📡 Fetching live regime data for {watch_ticker}..."):
            try:
                def _ema(series, span):
                    return series.ewm(span=span, adjust=False).mean()

                def _rsi(series, period=14):
                    delta = series.diff()
                    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
                    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
                    rs = gain / loss.replace(0, np.nan)
                    return 100 - (100 / (1 + rs))

                def _atr(df, period=14):
                    high, low, close = df["High"], df["Low"], df["Close"]
                    tr = pd.concat([
                        high - low,
                        (high - close.shift()).abs(),
                        (low  - close.shift()).abs(),
                    ], axis=1).max(axis=1)
                    return tr.ewm(com=period - 1, adjust=False).mean()

                def _adx(df, period=14):
                    high, low, close = df["High"], df["Low"], df["Close"]
                    plus_dm  = high.diff()
                    minus_dm = -low.diff()
                    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
                    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
                    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
                    atr_v    = tr.ewm(span=period, adjust=False).mean()
                    plus_di  = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_v)
                    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_v)
                    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
                    return dx.ewm(span=period, adjust=False).mean()

                def _ichimoku(df):
                    hi9  = df["High"].rolling(9).max()
                    lo9  = df["Low"].rolling(9).min()
                    hi26 = df["High"].rolling(26).max()
                    lo26 = df["Low"].rolling(26).min()
                    hi52 = df["High"].rolling(52).max()
                    lo52 = df["Low"].rolling(52).min()
                    tenkan  = (hi9  + lo9)  / 2
                    kijun   = (hi26 + lo26) / 2
                    span_a  = ((tenkan + kijun) / 2).shift(26)
                    span_b  = ((hi52 + lo52) / 2).shift(26)
                    chikou  = df["Close"].shift(-26)
                    return tenkan, kijun, span_a, span_b, chikou

                def _fetch(ticker, period="1y"):
                    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    return df if len(df) >= 50 else None

                def _score_market(spy_df):
                    if spy_df is None or len(spy_df) < 60:
                        return 0
                    c = spy_df["Close"]
                    score = 0
                    e20  = _ema(c, 20).iloc[-1]
                    e50  = _ema(c, 50).iloc[-1]
                    e200 = _ema(c, 200).iloc[-1]
                    last = c.iloc[-1]
                    rsi  = _rsi(c).iloc[-1]
                    if last > e200:  score += 10
                    if e50  > e200:  score += 8
                    if last > e50:   score += 7
                    if rsi  > 50:    score += 5
                    return min(score, 30)

                def _score_sector(sec_df):
                    if sec_df is None or len(sec_df) < 60:
                        return 0
                    c = sec_df["Close"]
                    score = 0
                    e20  = _ema(c, 20).iloc[-1]
                    e50  = _ema(c, 50).iloc[-1]
                    e200 = _ema(c, 200).iloc[-1]
                    last = c.iloc[-1]
                    rsi  = _rsi(c).iloc[-1]
                    if last > e200:  score += 10
                    if e50  > e200:  score += 8
                    if last > e50:   score += 7
                    if rsi  > 50:    score += 5
                    return min(score, 30)

                def _score_stock(stk_df):
                    if stk_df is None or len(stk_df) < 60:
                        return None
                    c = stk_df["Close"]
                    e10  = _ema(c, 10)
                    e21  = _ema(c, 21)
                    e50  = _ema(c, 50)
                    e200 = _ema(c, 200)
                    rsi_s = _rsi(c)
                    atr_s = _atr(stk_df)
                    tenkan, kijun, span_a, span_b, chikou = _ichimoku(stk_df)

                    adx_s    = _adx(stk_df)
                    last     = c.iloc[-1]
                    rsi_val  = float(rsi_s.iloc[-1])
                    atr_val  = float(atr_s.iloc[-1])
                    adx_val  = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0.0
                    e10_last = float(e10.iloc[-1])
                    e21_last = float(e21.iloc[-1])
                    e50_last = float(e50.iloc[-1])
                    e200_last= float(e200.iloc[-1])

                    stk_score = 0
                    if last > e200_last:                          stk_score += 10
                    if e50_last > e200_last:                      stk_score += 8
                    if last > e50_last:                           stk_score += 8
                    ema_stack = (e10_last > e21_last > e50_last > e200_last)
                    if ema_stack:                                  stk_score += 10
                    ema_trend = last > e21_last
                    if ema_trend:                                  stk_score += 7
                    if 40 < rsi_val < 75:                         stk_score += 7
                    stk_score = min(stk_score, 50)

                    ichi_score = 0
                    sa = float(span_a.iloc[-1]) if not pd.isna(span_a.iloc[-1]) else 0
                    sb = float(span_b.iloc[-1]) if not pd.isna(span_b.iloc[-1]) else 0
                    tk = float(tenkan.iloc[-1]) if not pd.isna(tenkan.iloc[-1]) else 0
                    kj = float(kijun.iloc[-1])  if not pd.isna(kijun.iloc[-1])  else 0
                    cloud_top = max(sa, sb)
                    cloud_bot = min(sa, sb)
                    if last > cloud_top:   ichi_score += 6
                    elif last > cloud_bot: ichi_score += 2
                    if tk > kj:            ichi_score += 5
                    if sa > sb:            ichi_score += 4
                    ichi_score = min(ichi_score, 15)

                    sl_dist   = 1.5 * atr_val
                    entry_zone = last
                    sl_level  = last - sl_dist
                    risk      = sl_dist
                    tp1       = last + 2.0 * risk
                    tp2       = last + 3.0 * risk
                    has_entry = ema_stack and ema_trend and 45 < rsi_val < 70

                    return {
                        "stock_score": stk_score,
                        "ichi_score":  ichi_score,
                        "close":       float(last),
                        "rsi":         rsi_val,
                        "atr":         atr_val,
                        "adx":         round(adx_val, 1),
                        "ema_stack":   ema_stack,
                        "ema_trend":   ema_trend,
                        "has_entry":   has_entry,
                        "entry_zone":  entry_zone,
                        "sl_level":    sl_level,
                        "tp1_2R":      tp1,
                        "tp2_3R":      tp2,
                        "ema10":       e10_last,
                        "ema21":       e21_last,
                        "ema50":       e50_last,
                        "ema200":      e200_last,
                        "kijun":       kj,
                    }

                spy_df   = _fetch("SPY",        period="1y")
                sec_df   = _fetch(watch_sector, period="1y")
                stock_df = _fetch(watch_ticker, period="1y")

                mkt_score  = _score_market(spy_df)
                sec_score  = _score_sector(sec_df)
                stk_result = _score_stock(stock_df)

                stk_score  = stk_result["stock_score"] if stk_result else 0
                ichi_score = stk_result["ichi_score"]   if stk_result else 0
                total      = mkt_score + sec_score + stk_score + ichi_score

                close_price = stk_result["close"]      if stk_result else 0
                rsi_val     = stk_result["rsi"]        if stk_result else 0
                ema_stack   = stk_result["ema_stack"]  if stk_result else False
                ema_trend   = stk_result["ema_trend"]  if stk_result else False
                has_entry   = stk_result["has_entry"]  if stk_result else False
                entry_zone  = stk_result["entry_zone"] if stk_result else 0
                sl_level    = stk_result["sl_level"]   if stk_result else 0
                tp1         = stk_result["tp1_2R"]     if stk_result else 0
                tp2         = stk_result["tp2_3R"]     if stk_result else 0
                atr_val     = stk_result["atr"]        if stk_result else 0
                adx_val_reg = stk_result["adx"]        if stk_result else 0

                ema10_val  = stk_result["ema10"]  if stk_result else 0
                ema21_val  = stk_result["ema21"]  if stk_result else 0
                ema50_val  = stk_result["ema50"]  if stk_result else 0
                ema200_val = stk_result["ema200"] if stk_result else 0
                kijun_val  = stk_result["kijun"]  if stk_result else 0

                st.session_state.regime_data = {
                    "mkt_score":   mkt_score,
                    "sec_score":   sec_score,
                    "stk_score":   stk_score,
                    "ichi_score":  ichi_score,
                    "total":       total,
                    "close":       close_price,
                    "rsi":         rsi_val,
                    "adx":         adx_val_reg,
                    "ema_stack":   ema_stack,
                    "ema_trend":   ema_trend,
                    "has_entry":   has_entry,
                    "entry_zone":  entry_zone,
                    "sl_level":    sl_level,
                    "tp1":         tp1,
                    "tp2":         tp2,
                    "atr":         atr_val,
                    "ema10":       ema10_val,
                    "ema21":       ema21_val,
                    "ema50":       ema50_val,
                    "ema200":      ema200_val,
                    "kijun":       kijun_val,
                    "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state.regime_stock_df = stock_df
                st.session_state.regime_ticker = watch_ticker

            except Exception as e:
                st.error(f"Regime fetch error: {e}")
                import traceback; st.code(traceback.format_exc())
                return

    data = st.session_state.regime_data
    if data is None:
        return

    total = data["total"]
    if total >= 85:
        regime_label = "BULL REGIME"
        regime_color = "#2d8a4e"
    elif total >= 65:
        regime_label = "MODERATE BULL"
        regime_color = "#d4943a"
    elif total >= 45:
        regime_label = "NEUTRAL"
        regime_color = "#d4943a"
    else:
        regime_label = "BEAR / AVOID"
        regime_color = "#c44545"

    score_col, gauge_col = st.columns([1, 3])

    with score_col:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(201,168,76,0.06) 0%, rgba(139,115,64,0.03) 100%);
            border: 1px solid {regime_color};
            border-radius: 12px;
            padding: 32px 16px;
            text-align: center;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute; top: 0; left: 0; right: 0; height: 3px;
                background: linear-gradient(90deg, #c9a84c, #8b7340);
            "></div>
            <div style="font-size:11px;letter-spacing:4px;color:rgba(201,168,76,0.5);margin-bottom:8px;">
                REGIME SCORE
            </div>
            <div style="
                font-size: 80px;
                font-weight: 900;
                line-height: 1;
                color: {regime_color};
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                font-family: 'Courier New', monospace;
            ">{total}</div>
            <div style="font-size:11px;letter-spacing:3px;color:rgba(201,168,76,0.4);margin-top:4px;">
                / 125 MAX
            </div>
            <div style="margin-top:16px;">
                <span style="
                    background: {regime_color}1a;
                    border: 1px solid {regime_color};
                    border-radius: 20px;
                    color: {regime_color};
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 3px;
                    padding: 5px 16px;
                    text-shadow: none;
                ">{regime_label}</span>
            </div>
            <div style="
                margin-top: 20px;
                font-size: 10px;
                color: rgba(201,168,76,0.3);
                letter-spacing: 2px;
            ">
                {watch_ticker} &nbsp;·&nbsp; {data['timestamp']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with gauge_col:
        g1, g2, g3, g4 = st.columns(4)
        g1.plotly_chart(build_gauge(data["mkt_score"],  30, "MARKET (SPY)", color_cyan=True),
                        width='stretch', config={"displayModeBar": False})
        g2.plotly_chart(build_gauge(data["sec_score"],  30, f"SECTOR ({watch_sector})", color_cyan=False),
                        width='stretch', config={"displayModeBar": False})
        g3.plotly_chart(build_gauge(data["stk_score"],  50, f"STOCK ({watch_ticker})", color_cyan=True),
                        width='stretch', config={"displayModeBar": False})
        g4.plotly_chart(build_gauge(data["ichi_score"], 15, "ICHIMOKU", color_cyan=False),
                        width='stretch', config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    section_title("System Status")

    s1, s2, s3, s4 = st.columns(4)

    def status_pill(label, active):
        if active:
            return f'<div class="status-active">{label}</div>'
        return f'<div class="status-inactive">{label}</div>'

    with s1:
        st.markdown("**EMA STACK**")
        st.markdown(status_pill("EMA STACK FULL" if data["ema_stack"] else "EMA STACK OFF",
                                data["ema_stack"]), unsafe_allow_html=True)

    with s2:
        st.markdown("**EMA TREND**")
        st.markdown(status_pill("TREND ACTIVE" if data["ema_trend"] else "NO TREND",
                                data["ema_trend"]), unsafe_allow_html=True)

    with s3:
        st.markdown("**ENTRY SIGNAL**")
        st.markdown(status_pill("ENTRY LIVE" if data["has_entry"] else "NO ENTRY",
                                data["has_entry"]), unsafe_allow_html=True)

    with s4:
        st.markdown("**REGIME GATE**")
        gate_ok = total >= 40
        st.markdown(status_pill("GATE OPEN" if gate_ok else "GATE CLOSED", gate_ok),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    section_title(f"Live Levels — {watch_ticker}")

    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("CLOSE PRICE",  f"{data['close']:.2f}")
    p2.metric("RSI (14)",     f"{data['rsi']:.1f}",
              delta="Overbought" if data['rsi'] > 70 else ("Oversold" if data['rsi'] < 30 else "Neutral"))
    _adx_disp = data.get("adx", 0)
    _adx_thresh_disp = 19
    p3.metric("ADX (14)",     f"{_adx_disp:.1f}",
              delta="Trending" if _adx_disp >= _adx_thresh_disp else "Weak trend")
    p4.metric("ATR (14)",     f"{data['atr']:.2f}")
    p5.metric("ENTRY ZONE",   f"{data['entry_zone']:.2f}")
    p6.metric("STOP LOSS",    f"{data['sl_level']:.2f}")

    t1, t2 = st.columns(2)
    t1.metric("TP1 (2R)", f"{data['tp1']:.2f}",
              delta=f"+{((data['tp1']/data['close'])-1)*100:.1f}%" if data['close'] > 0 else None)
    t2.metric("TP2 (3R)", f"{data['tp2']:.2f}",
              delta=f"+{((data['tp2']/data['close'])-1)*100:.1f}%" if data['close'] > 0 else None)

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("Score Breakdown")

    breakdown = [
        ("Market Regime", "SPY",         data["mkt_score"],  30),
        ("Sector ETF",    watch_sector,  data["sec_score"],  30),
        ("Stock Score",   watch_ticker,  data["stk_score"],  50),
        ("Ichimoku",      "Cloud/TK/CK", data["ichi_score"], 15),
    ]

    bd_df = pd.DataFrame([
        {"Layer": label, "Instrument": sub, "Score": score, "Max": max_s, "Pct": round(score / max_s * 100)}
        for label, sub, score, max_s in breakdown
    ])
    bd_df.loc[len(bd_df)] = {"Layer": "TOTAL", "Instrument": "", "Score": total, "Max": 125, "Pct": round(total / 125 * 100)}
    bd_df["Score Display"] = bd_df.apply(lambda r: f"{int(r['Score'])}/{int(r['Max'])}", axis=1)
    bd_df["Fill %"] = bd_df["Pct"].astype(str) + "%"

    display_df = bd_df[["Layer", "Instrument", "Score Display", "Fill %"]].copy()
    pct_values = bd_df["Pct"].tolist()

    def color_bd(row):
        idx = row.name
        pct = pct_values[idx] if idx < len(pct_values) else 0
        if pct >= 67:
            return ["background-color: rgba(0,180,80,0.15); color: #00cc66"] * len(row)
        elif pct >= 40:
            return ["background-color: rgba(200,200,0,0.1); color: #cccc00"] * len(row)
        return ["background-color: rgba(200,50,80,0.1); color: #ff6666"] * len(row)

    st.dataframe(
        display_df.style.apply(color_bd, axis=1),
        width='stretch',
        hide_index=True,
        height=230,
    )

    # ── WOLF ENTRY CHECKLIST ──────────────────────────────────────────────────
    try:
        _chk_close  = data["close"]
        _chk_rsi    = data["rsi"]
        _chk_atr    = data["atr"]
        _chk_ema10  = data.get("ema10", 0)
        _chk_ema21  = data.get("ema21", 0)
        _chk_ema50  = data.get("ema50", 0)
        _chk_ema200 = data.get("ema200", 0)
        _chk_kijun  = data.get("kijun", 0)
        _chk_stack  = data["ema_stack"]

        if _chk_stack:
            _trend_dir, _trend_clr = "BULL", "#2d8a4e"
        elif _chk_close > _chk_ema50:
            _trend_dir, _trend_clr = "NEUTRAL-BULL", "#d4943a"
        elif _chk_close > _chk_ema200:
            _trend_dir, _trend_clr = "NEUTRAL", "#d4943a"
        else:
            _trend_dir, _trend_clr = "BEAR", "#c44545"

        _stack_icon = "✓" if _chk_stack else "✗"

        if _chk_rsi > 70:
            _rsi_state = "OVERBOUGHT"
        elif _chk_rsi < 30:
            _rsi_state = "OVERSOLD"
        elif 40 <= _chk_rsi <= 60:
            _rsi_state = "NEUTRAL"
        else:
            _rsi_state = "ACTIVE"

        _stk_df = st.session_state.get("regime_stock_df")
        _vol_ratio = 0.0
        if _stk_df is not None and len(_stk_df) >= 20:
            _vol_avg = _stk_df["Volume"].rolling(20).mean().iloc[-1]
            if _vol_avg > 0:
                _vol_ratio = float(_stk_df["Volume"].iloc[-1] / _vol_avg)

        _hist_vol = 0.0
        if _stk_df is not None and len(_stk_df) >= 21:
            _log_ret = np.log(_stk_df["Close"] / _stk_df["Close"].shift(1)).dropna()
            if len(_log_ret) >= 20:
                _hist_vol = float(_log_ret.tail(20).std() * np.sqrt(252) * 100)

        _atr_ratio = 1.0
        if _stk_df is not None and len(_stk_df) >= 64:
            _high, _low, _cls = _stk_df["High"], _stk_df["Low"], _stk_df["Close"]
            _tr = pd.concat([_high - _low, (_high - _cls.shift()).abs(), (_low - _cls.shift()).abs()], axis=1).max(axis=1)
            _atr_series = _tr.ewm(com=13, adjust=False).mean()
            _atr_avg_50 = _atr_series.tail(50).mean()
            if _atr_avg_50 > 0:
                _atr_ratio = float(_atr_series.iloc[-1] / _atr_avg_50)

        _bull_obs, _bear_obs = [], []
        if _stk_df is not None and len(_stk_df) >= 50:
            _ob_atr = _chk_atr if _chk_atr > 0 else 1.0
            _ob_recent = _stk_df.tail(100) if len(_stk_df) >= 100 else _stk_df
            for _obi in range(2, len(_ob_recent)):
                _ob_c = _ob_recent.iloc[_obi]
                _ob_p = _ob_recent.iloc[_obi - 1]
                _ob_body = abs(float(_ob_c["Close"]) - float(_ob_c["Open"]))
                if (float(_ob_c["Close"]) > float(_ob_c["Open"])
                        and _ob_body > _ob_atr * 1.5
                        and float(_ob_p["Close"]) < float(_ob_p["Open"])):
                    _bull_obs.append(float(_ob_c["Low"]))
                if (float(_ob_c["Close"]) < float(_ob_c["Open"])
                        and _ob_body > _ob_atr * 1.5
                        and float(_ob_p["Close"]) > float(_ob_p["Open"])):
                    _bear_obs.append(float(_ob_c["High"]))
            _bull_obs = _bull_obs[-5:]
            _bear_obs = _bear_obs[-5:]

        _nearest_bull_ob = min(_bull_obs, key=lambda x: abs(x - _chk_close)) if _bull_obs else None
        _nearest_bear_ob = min(_bear_obs, key=lambda x: abs(x - _chk_close)) if _bear_obs else None
        _has_nearby_ob = False
        if _nearest_bull_ob and abs(_nearest_bull_ob - _chk_close) / _chk_close < 0.03:
            _has_nearby_ob = True
        if _nearest_bear_ob and abs(_nearest_bear_ob - _chk_close) / _chk_close < 0.03:
            _has_nearby_ob = True

        _candle_patterns = []
        if _stk_df is not None and len(_stk_df) >= 3:
            _cc = _stk_df.iloc[-1]
            _cp = _stk_df.iloc[-2]
            _cbody = float(_cc["Close"]) - float(_cc["Open"])
            _crange = float(_cc["High"]) - float(_cc["Low"])
            if _crange > 0:
                _lower_wick = min(float(_cc["Open"]), float(_cc["Close"])) - float(_cc["Low"])
                if _lower_wick > abs(_cbody) * 2 and _cbody > 0:
                    _candle_patterns.append("Hammer (Bull)")
                if (float(_cp["Close"]) < float(_cp["Open"])
                        and float(_cc["Close"]) > float(_cc["Open"])
                        and float(_cc["Close"]) > float(_cp["Open"])
                        and float(_cc["Open"]) < float(_cp["Close"])):
                    _candle_patterns.append("Bullish Engulfing")
                if (float(_cp["Close"]) > float(_cp["Open"])
                        and float(_cc["Close"]) < float(_cc["Open"])
                        and float(_cc["Close"]) < float(_cp["Open"])
                        and float(_cc["Open"]) > float(_cp["Close"])):
                    _candle_patterns.append("Bearish Engulfing")
                if abs(_cbody) < _crange * 0.1:
                    _candle_patterns.append("Doji")

        _is_consolidating = (35 < _chk_rsi < 65) and (_atr_ratio < 0.7)
        _ema10_dist = abs(_chk_close - _chk_ema10) / _chk_close * 100 if _chk_close > 0 else 99
        _ema21_dist = abs(_chk_close - _chk_ema21) / _chk_close * 100 if _chk_close > 0 else 99
        _is_pullback = (min(_ema10_dist, _ema21_dist) < 2.0) and (_chk_close > _chk_ema50)
        _has_candle = len(_candle_patterns) > 0
        _pattern_name = ", ".join(_candle_patterns) if _candle_patterns else "Inget mönster"
        _half_atr = _chk_atr * 0.5
        _sl_pct = (_half_atr / _chk_close * 100) if _chk_close > 0 else 0
        _rr_ratio = 0.0
        if _nearest_bear_ob and _half_atr > 0:
            _target_dist = abs(_nearest_bear_ob - _chk_close)
            _rr_ratio = _target_dist / _half_atr

        _risk_score = min(100, int((_hist_vol / 50) * 100)) if _hist_vol > 0 else 50

        st.markdown(
            "<div style='color:#8b7340;font-size:0.7rem;text-transform:uppercase;"
            "letter-spacing:0.1em;margin:20px 0 8px 0;border-top:1px solid rgba(139,115,64,0.15);"
            "padding-top:12px;'>WOLF ENTRY CHECKLIST</div>",
            unsafe_allow_html=True,
        )

        _ec1, _ec2, _ec3 = st.columns(3)

        def _checklist_card(title, lines, border_color="#c9a84c"):
            content = "".join(
                f"<div style='font-size:0.72rem;color:#e8e4dc;padding:1px 0;'>{ln}</div>"
                for ln in lines
            )
            return (
                f"<div style='background:#14141e;border:1px solid {border_color};"
                f"border-radius:6px;padding:10px 12px;'>"
                f"<div style='color:{border_color};font-size:0.65rem;font-weight:700;"
                f"letter-spacing:0.08em;margin-bottom:6px;'>{title}</div>"
                f"{content}</div>"
            )

        with _ec1:
            st.markdown(_checklist_card("TREND", [
                f"Direction: <span style='color:{_trend_clr};font-weight:700;'>{_trend_dir}</span>",
                f"EMA Stack: <span style='color:{('#2d8a4e' if _chk_stack else '#c44545')};'>{_stack_icon}</span>",
                f"EMA10: <b>{_chk_ema10:.2f}</b>",
                f"EMA20: <b>{_chk_ema21:.2f}</b>",
                f"EMA50: <b>{_chk_ema50:.2f}</b>",
                f"EMA200: <b>{_chk_ema200:.2f}</b>",
            ]), unsafe_allow_html=True)

        with _ec2:
            st.markdown(_checklist_card("VOLATILITET", [
                f"ATR 14: <b>{_chk_atr:.2f}</b>",
                f"Hist Vol: <b>{_hist_vol:.0f}%</b>",
                f"ATR ratio: <b>{_atr_ratio:.2f}x</b>",
                f"Risk: <b>{_risk_score}/100</b>",
            ], border_color="#8b7340"), unsafe_allow_html=True)

        with _ec3:
            _vol_clr = "#2d8a4e" if _vol_ratio >= 1.0 else "#c44545"
            st.markdown(_checklist_card("MOMENTUM", [
                f"RSI: <b>{_chk_rsi:.1f}</b> ({_rsi_state})",
                f"Vol ratio: <span style='color:{_vol_clr};font-weight:700;'>{_vol_ratio:.2f}x</span>",
                f"ADX: <b>{data.get('adx', 0):.1f}</b>",
            ]), unsafe_allow_html=True)

        _ec4, _ec5 = st.columns(2)

        with _ec4:
            _cpat_lines = []
            if _candle_patterns:
                for _cp_name in _candle_patterns:
                    _cp_icon = "⬆" if "Bull" in _cp_name or "Hammer" in _cp_name else "⬇"
                    _cpat_lines.append(f"{_cp_icon} {_cp_name}")
            else:
                _cpat_lines.append("<span style='color:#8a8578;'>Inga mönster detekterade</span>")
            st.markdown(_checklist_card("CANDLESTICK", _cpat_lines, border_color="#d4943a"), unsafe_allow_html=True)

        with _ec5:
            _ob_lines = [
                f"Bullish OBs: <b>{len(_bull_obs)}</b>"
                + (f" (närmaste: {_nearest_bull_ob:.2f})" if _nearest_bull_ob else ""),
                f"Bearish OBs: <b>{len(_bear_obs)}</b>"
                + (f" (närmaste: {_nearest_bear_ob:.2f})" if _nearest_bear_ob else ""),
            ]
            if _has_nearby_ob:
                _ob_lines.append("<span style='color:#2d8a4e;font-weight:700;'>OB inom 3% av pris ✓</span>")
            else:
                _ob_lines.append("<span style='color:#8a8578;'>Ingen OB inom 3%</span>")
            st.markdown(_checklist_card("ORDER BLOCKS", _ob_lines, border_color="#c9a84c"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    except Exception:
        pass

    # ── Wolf Trading Gates (11 rules) ────────────────────────────────────────
    st.markdown(
        "<div style='color:#c9a84c;font-size:0.7rem;text-transform:uppercase;"
        "letter-spacing:0.1em;margin:20px 0 8px 0;border-top:1px solid rgba(201,168,76,0.1);"
        "padding-top:12px;'>SWING TRADING GATES — 11 REGLER</div>",
        unsafe_allow_html=True,
    )

    swing_gates = []
    try:
        _g_rsi    = data["rsi"]
        _g_atr    = data["atr"]
        _g_close  = data["close"]
        _g_ema10  = data.get("ema10", 0)
        _g_ema21  = data.get("ema21", 0)
        _g_kijun  = data.get("kijun", 0)
        _g_consol = (35 < _g_rsi < 65)
        _g_atr_low = False
        try:
            _g_atr_low = _atr_ratio < 0.7  # noqa: F821
        except NameError:
            pass
        _g_is_consol = _g_consol and _g_atr_low

        _g_nearby_ob = False
        try:
            _g_nearby_ob = _has_nearby_ob  # noqa: F821
        except NameError:
            pass

        _g_nearest_ob_val = 0.0
        try:
            if _nearest_bull_ob:  # noqa: F821
                _g_nearest_ob_val = _nearest_bull_ob
            elif _nearest_bear_ob:  # noqa: F821
                _g_nearest_ob_val = _nearest_bear_ob
        except NameError:
            pass

        _g_ema10_dist = abs(_g_close - _g_ema10) / _g_close * 100 if _g_close > 0 else 99
        _g_ema21_dist = abs(_g_close - _g_ema21) / _g_close * 100 if _g_close > 0 else 99
        _g_is_pullback = (min(_g_ema10_dist, _g_ema21_dist) < 2.0) and (_g_close > data.get("ema50", 0))

        _g_has_candle = False
        _g_pattern_name = "Inget mönster"
        try:
            _g_has_candle = _has_candle  # noqa: F821
            _g_pattern_name = _pattern_name  # noqa: F821
        except NameError:
            pass

        _g_vol_ratio = 0.0
        try:
            _g_vol_ratio = _vol_ratio  # noqa: F821
        except NameError:
            pass

        _g_half_atr = _g_atr * 0.5
        _g_sl_pct = (_g_half_atr / _g_close * 100) if _g_close > 0 else 0

        _g_rr = 0.0
        try:
            _g_rr = _rr_ratio  # noqa: F821
        except NameError:
            pass

        swing_gates = [
            {"rule": "1. Trendriktning",       "passed": total >= 50,          "value": f"Score: {total}/125"},
            {"rule": "2. Ej konsolidering",     "passed": not _g_is_consol,     "value": f"RSI: {_g_rsi:.0f}"},
            {"rule": "3. Key level (OB)",        "passed": _g_nearby_ob,         "value": f"Närmaste OB: {_g_nearest_ob_val:.2f}" if _g_nearby_ob else "Ingen OB nära"},
            {"rule": "4. Pullback entry",        "passed": _g_is_pullback,       "value": f"Avstånd EMA10: {_g_ema10_dist:.1f}%"},
            {"rule": "5. Candle trigger",        "passed": _g_has_candle,        "value": _g_pattern_name},
            {"rule": "6. Volymbekräftelse",      "passed": _g_vol_ratio >= 1.0,  "value": f"Vol ratio: {_g_vol_ratio:.1f}x"},
            {"rule": "7. R:R ≥ 1:2",             "passed": _g_rr >= 2.0,         "value": f"R:R 1:{_g_rr:.1f}"},
            {"rule": "8. Max 1% risk",           "passed": True,                 "value": f"SL dist: {_g_half_atr:.2f} ({_g_sl_pct:.1f}%)"},
            {"rule": "9. SL → BE efter HH",     "passed": True,                 "value": "Post-entry regel"},
            {"rule": "10. Max 2 förluster/dag", "passed": True,                 "value": "Disciplin"},
            {"rule": "11. Kijun trail + ½ATR", "passed": True,                 "value": f"Kijun: {_g_kijun:.2f}, EMA10: {_g_ema10:.2f}"},
        ]
    except Exception:
        swing_gates = [
            {"rule": "1. Trendriktning",       "passed": total >= 50,  "value": f"Score: {total}/125"},
            {"rule": "2. Ej konsolidering",     "passed": True,         "value": "Data ej tillgänglig"},
            {"rule": "3. Key level (OB)",        "passed": True,         "value": "Data ej tillgänglig"},
            {"rule": "4. Pullback entry",        "passed": True,         "value": "Data ej tillgänglig"},
            {"rule": "5. Candle trigger",        "passed": True,         "value": "Data ej tillgänglig"},
            {"rule": "6. Volymbekräftelse",      "passed": True,         "value": "Data ej tillgänglig"},
            {"rule": "7. R:R ≥ 1:2",             "passed": True,         "value": "Data ej tillgänglig"},
            {"rule": "8. Max 1% risk",           "passed": True,         "value": "Position sizing"},
            {"rule": "9. SL → BE efter HH",     "passed": True,         "value": "Post-entry"},
            {"rule": "10. Max 2 förluster/dag", "passed": True,         "value": "Disciplin"},
            {"rule": "11. Kijun trail + ½ATR", "passed": True,         "value": "Exit-regel"},
        ]

    if swing_gates:
        passed = sum(1 for g in swing_gates if g["passed"])
        total_gates = len(swing_gates)
        gc = "#2d8a4e" if passed >= 8 else ("#d4943a" if passed >= 5 else "#c44545")

        gate_html = '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">'
        for g in swing_gates:
            c = "#2d8a4e" if g["passed"] else "#c44545"
            icon = "✓" if g["passed"] else "✗"
            gate_html += (
                f'<div style="background:#14141e;border:1px solid {c};border-radius:4px;'
                f'padding:3px 8px;font-size:0.65rem;">'
                f'<span style="color:{c};font-weight:700;">{icon}</span> '
                f'<span style="color:#e8e4dc;">{g["rule"]}</span> '
                f'<span style="color:#8a8578;">({g["value"]})</span>'
                f'</div>'
            )
        gate_html += '</div>'
        gate_html += f'<div style="color:{gc};font-size:0.8rem;font-weight:700;">{passed}/{total_gates} GATES</div>'
        st.markdown(gate_html, unsafe_allow_html=True)

    # ── Inline SL/TP Calculator ──────────────────────────────────────────────
    try:
        _sltp_df = st.session_state.get("regime_stock_df")
        if _sltp_df is not None and not _sltp_df.empty and len(_sltp_df) >= 20:
            _CYAN_SL = "#c9a84c"
            _GREEN_SL = "#2d8a4e"
            _RED_SL = "#c44545"
            _TEXT_SL = "#e8e4dc"
            _DIM_SL = "#8a8578"
            _BG2_SL = "#14141e"

            st.markdown(
                f"<div style='color:{_CYAN_SL};font-size:0.85rem;text-transform:uppercase;"
                f"letter-spacing:0.1em;margin:20px 0 10px 0;border-top:2px solid rgba(201,168,76,0.2);"
                f"padding-top:14px;font-weight:700;'>SL / TP KALKYLATOR — WOLF</div>",
                unsafe_allow_html=True,
            )

            _close_sl = pd.to_numeric(_sltp_df["Close"], errors="coerce")
            _high_sl = pd.to_numeric(_sltp_df["High"], errors="coerce")
            _low_sl = pd.to_numeric(_sltp_df["Low"], errors="coerce")
            _price_sl = float(_close_sl.dropna().iloc[-1])

            _tr_sl = pd.concat([
                _high_sl - _low_sl,
                abs(_high_sl - _close_sl.shift(1)),
                abs(_low_sl - _close_sl.shift(1)),
            ], axis=1).max(axis=1)
            _atr_sl = float(_tr_sl.rolling(14).mean().dropna().iloc[-1])
            _half_atr_sl = _atr_sl / 2
            _ema10_sl = float(_close_sl.ewm(span=10).mean().iloc[-1])
            _kijun_sl = float((_high_sl.rolling(26).max() + _low_sl.rolling(26).min()).iloc[-1] / 2)

            _sl_atr = _price_sl - _half_atr_sl
            _sl_val = max(_sl_atr, _kijun_sl)
            _sl_dist_sl = _price_sl - _sl_val
            _tp_2r_sl = _price_sl + _sl_dist_sl * 2
            _tp_3r_sl = _price_sl + _sl_dist_sl * 3

            _sltp_c1, _sltp_c2 = st.columns(2)
            with _sltp_c1:
                _cap_wolf = st.number_input(
                    "Kapital (SEK)", value=100000, step=10000,
                    key="sltp_cap_wolf_inline",
                )
            with _sltp_c2:
                _risk_wolf = st.number_input(
                    "Risk %", value=5.0, min_value=0.5, max_value=10.0,
                    step=0.5, key="sltp_risk_wolf_inline",
                )

            _risk_amt_sl = _cap_wolf * (_risk_wolf / 100)
            _shares_sl = int(_risk_amt_sl / _sl_dist_sl) if _sl_dist_sl > 0 else 0
            _pos_val_sl = _shares_sl * _price_sl
            _pos_pct_sl = (_pos_val_sl / _cap_wolf * 100) if _cap_wolf > 0 else 0

            _sc1, _sc2, _sc3 = st.columns(3)
            with _sc1:
                st.markdown(
                    f'<div style="background:{_BG2_SL};border:2px solid rgba(196,69,69,0.3);border-radius:8px;padding:12px;">'
                    f'<div style="color:{_RED_SL};font-weight:700;font-size:0.8rem;">STOP LOSS</div>'
                    f'<div style="color:{_TEXT_SL};font-size:1.1rem;font-weight:700;">{_sl_val:.2f}</div>'
                    f'<div style="color:{_DIM_SL};font-size:0.65rem;">½ ATR = {_half_atr_sl:.2f}</div>'
                    f'<div style="color:{_RED_SL};font-size:0.72rem;">Risk: {_sl_dist_sl:.2f} ({_sl_dist_sl/_price_sl*100:.1f}%)</div>'
                    f'<div style="color:{_DIM_SL};font-size:0.6rem;margin-top:4px;">Trail: Kijun ({_kijun_sl:.2f}) / EMA10 ({_ema10_sl:.2f})</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with _sc2:
                st.markdown(
                    f'<div style="background:{_BG2_SL};border:2px solid rgba(45,138,78,0.3);border-radius:8px;padding:12px;">'
                    f'<div style="color:{_GREEN_SL};font-weight:700;font-size:0.8rem;">TARGETS</div>'
                    f'<div style="color:{_TEXT_SL};font-size:0.85rem;">2R: <b style="color:{_GREEN_SL};">{_tp_2r_sl:.2f}</b> (+{(_tp_2r_sl/_price_sl-1)*100:.1f}%)</div>'
                    f'<div style="color:{_TEXT_SL};font-size:0.85rem;">3R: <b style="color:{_GREEN_SL};">{_tp_3r_sl:.2f}</b> (+{(_tp_3r_sl/_price_sl-1)*100:.1f}%)</div>'
                    f'<div style="color:{_DIM_SL};font-size:0.6rem;margin-top:4px;">Trailing stop — ej fast TP</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with _sc3:
                st.markdown(
                    f'<div style="background:{_BG2_SL};border:2px solid rgba(201,168,76,0.2);border-radius:8px;padding:12px;">'
                    f'<div style="color:{_CYAN_SL};font-weight:700;font-size:0.8rem;">POSITION</div>'
                    f'<div style="color:{_TEXT_SL};font-size:0.85rem;">{_shares_sl} aktier</div>'
                    f'<div style="color:{_TEXT_SL};font-size:0.85rem;">{_pos_val_sl:,.0f} SEK ({_pos_pct_sl:.1f}%)</div>'
                    f'<div style="color:{_RED_SL};font-size:0.72rem;">Risk: {_risk_amt_sl:,.0f} SEK</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    except Exception:
        pass
