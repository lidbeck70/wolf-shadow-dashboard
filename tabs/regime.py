import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime

import wolf_regime_score as wrs
from ui.theme import section_title
from ui.charts import build_gauge
from utils.presets import SECTOR_ETF_LIST, etf_from_display

# Poängen, etiketten, checklistan, de elva grindarna och SL/TP räknas i
# wolf_regime_score.py (ren modul, samma formler) så att Thors analysblock
# visar exakt samma siffror som den här sidan. Wolfs riktiga stoppavstånd och
# risk (strategies/wolf.py) läses därifrån.
_WOLF_ATR_MULT = wrs.ATR_MULT
_WOLF_RISK_PCT = wrs.RISK_PCT


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
        refresh_btn = st.button("📡 Uppdatera", key="reg_refresh", width='stretch')

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
        from market_prices import ohlcv as _mp_ohlcv          # cachad 6 h — kördes förr vid varje klick i hela appen
        bm_data = _mp_ohlcv(benchmark_ticker, "3mo")
        stk_data = _mp_ohlcv(watch_ticker, "3mo")
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
                def _fetch(ticker, period="1y"):
                    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    return df if len(df) >= 50 else None

                spy_df   = _fetch("SPY",        period="1y")
                sec_df   = _fetch(watch_sector, period="1y")
                stock_df = _fetch(watch_ticker, period="1y")

                st.session_state.regime_data = wrs.regime(
                    spy_df, sec_df, stock_df,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
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
    regime_label, regime_color = wrs.label(total)

    score_col, gauge_col = st.columns([1, 3])

    with score_col:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(0,229,255,0.06) 0%, rgba(139,115,64,0.03) 100%);
            border: 1px solid {regime_color};
            border-radius: 12px;
            padding: 32px 16px;
            text-align: center;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute; top: 0; left: 0; right: 0; height: 3px;
                background: linear-gradient(90deg, #00E5FF, #00A8BF);
            "></div>
            <div style="font-size:11px;letter-spacing:4px;color:rgba(0,229,255,0.5);margin-bottom:8px;">
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
            <div style="font-size:11px;letter-spacing:3px;color:rgba(0,229,255,0.4);margin-top:4px;">
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
                color: rgba(0,229,255,0.3);
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
        gate_ok = total >= wrs.GATE_MIN
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
    _chk = None
    try:
        _stk_df = st.session_state.get("regime_stock_df")
        _chk = wrs.checklist(data, _stk_df, _WOLF_ATR_MULT)

        _chk_rsi    = data["rsi"]
        _chk_atr    = data["atr"]
        _chk_ema10  = data.get("ema10", 0)
        _chk_ema21  = data.get("ema21", 0)
        _chk_ema50  = data.get("ema50", 0)
        _chk_ema200 = data.get("ema200", 0)
        _chk_stack  = data["ema_stack"]

        _trend_dir, _trend_clr = _chk["trend_dir"], _chk["trend_color"]
        _stack_icon = "✓" if _chk_stack else "✗"
        _rsi_state = _chk["rsi_state"]
        _vol_ratio, _hist_vol, _atr_ratio = _chk["vol_ratio"], _chk["hist_vol"], _chk["atr_ratio"]
        _bull_obs, _bear_obs = _chk["bull_obs"], _chk["bear_obs"]
        _nearest_bull_ob, _nearest_bear_ob = _chk["nearest_bull_ob"], _chk["nearest_bear_ob"]
        _has_nearby_ob = _chk["has_nearby_ob"]
        _candle_patterns = _chk["candle_patterns"]
        _risk_score = _chk["risk_score"]

        st.markdown(
            "<div style='color:#00A8BF;font-size:0.7rem;text-transform:uppercase;"
            "letter-spacing:0.1em;margin:20px 0 8px 0;border-top:1px solid rgba(139,115,64,0.15);"
            "padding-top:12px;'>WOLF ENTRY CHECKLIST</div>",
            unsafe_allow_html=True,
        )

        _ec1, _ec2, _ec3 = st.columns(3)

        def _checklist_card(title, lines, border_color="#00E5FF"):
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
            ], border_color="#00A8BF"), unsafe_allow_html=True)

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
            st.markdown(_checklist_card("ORDER BLOCKS", _ob_lines, border_color="#00E5FF"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    except Exception:
        pass

    # ── Wolf Trading Gates (11 rules) ────────────────────────────────────────
    st.markdown(
        "<div style='color:#00E5FF;font-size:0.7rem;text-transform:uppercase;"
        "letter-spacing:0.1em;margin:20px 0 8px 0;border-top:1px solid rgba(0,229,255,0.1);"
        "padding-top:12px;'>SWING TRADING GATES — 11 REGLER</div>",
        unsafe_allow_html=True,
    )

    # Grind 8–11 är process- och disciplinregler och visas som MANUELLA; saknas
    # checklistans data är grind 2–7 INTE godkända (ett datafel är inget ja).
    try:
        if _chk is None:
            raise ValueError("checklistan kunde inte räknas")
        swing_gates = wrs.gates(data, _chk, _WOLF_ATR_MULT, _WOLF_RISK_PCT)
    except Exception as _gate_exc:
        swing_gates = wrs.gates_unavailable(data, _gate_exc, _WOLF_ATR_MULT, _WOLF_RISK_PCT)

    if swing_gates:
        _gs = wrs.gate_summary(swing_gates)
        passed, total_gates, manual, gc = _gs["passed"], _gs["measurable"], _gs["manual"], _gs["color"]

        gate_html = '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">'
        for g in swing_gates:
            if g["passed"] is None:
                c, icon = "#8a8578", "◌"
            else:
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
        gate_html += (f'<div style="color:{gc};font-size:0.8rem;font-weight:700;">{passed}/{total_gates} MÄTBARA GRINDAR'
                      f' <span style="color:#8a8578;font-weight:400;">· {manual} manuella (◌) bedömer du själv</span></div>')
        st.markdown(gate_html, unsafe_allow_html=True)

    # ── Inline SL/TP Calculator ──────────────────────────────────────────────
    try:
        _sltp_df = st.session_state.get("regime_stock_df")
        if _sltp_df is not None and not _sltp_df.empty and len(_sltp_df) >= 20:
            _CYAN_SL = "#00E5FF"
            _GREEN_SL = "#2d8a4e"
            _RED_SL = "#c44545"
            _TEXT_SL = "#e8e4dc"
            _DIM_SL = "#8a8578"
            _BG2_SL = "#14141e"

            st.markdown(
                f"<div style='color:{_CYAN_SL};font-size:0.85rem;text-transform:uppercase;"
                f"letter-spacing:0.1em;margin:20px 0 10px 0;border-top:2px solid rgba(0,229,255,0.2);"
                f"padding-top:14px;font-weight:700;'>SL / TP KALKYLATOR — WOLF</div>",
                unsafe_allow_html=True,
            )

            _sltp_c1, _sltp_c2 = st.columns(2)
            with _sltp_c1:
                _cap_wolf = st.number_input(
                    "Kapital (SEK)", value=100000, step=10000,
                    key="sltp_cap_wolf_inline",
                )
            with _sltp_c2:
                _risk_wolf = st.number_input(
                    "Risk %", value=float(_WOLF_RISK_PCT), min_value=0.5,
                    max_value=10.0, step=0.5, key="sltp_risk_wolf_inline",
                    help=f"Wolf riskerar {_WOLF_RISK_PCT:g} % per affär "
                         f"(strategies/wolf.py: risk_pct).",
                )

            # 🌩️ Marknadsrisk HÖG → halverad position (market_risk_gate)
            _mr_factor = 1.0
            try:
                import market_risk_gate as _mg
                _mr_ticker = st.session_state.get("regime_ticker") or st.session_state.get("reg_ticker") or ""
                _mr = _mg.for_ticker(_mr_ticker)
                _mr_factor = _mg.size_factor(_mr)
                if _mr_factor < 1:
                    st.warning(f"🌩️ {_mg.describe(_mr)} — Wolf-positionen halveras "
                               f"({int(_mr_factor * 100)} % av normal storlek).")
            except Exception:
                _mr_factor = 1.0

            _p = wrs.sltp(_sltp_df, _cap_wolf, _risk_wolf, _WOLF_ATR_MULT, _mr_factor)
            _price_sl, _atr_stop_sl = _p["price"], _p["atr_stop"]
            _ema10_sl, _kijun_sl = _p["ema10"], _p["kijun"]
            _sl_val, _sl_dist_sl = _p["sl"], _p["sl_dist"]
            _tp_2r_sl, _tp_3r_sl = _p["tp_2r"], _p["tp_3r"]
            _risk_amt_sl, _shares_sl = _p["risk_amt"], _p["shares"]
            _pos_val_sl, _pos_pct_sl = _p["pos_val"], _p["pos_pct"]

            _sc1, _sc2, _sc3 = st.columns(3)
            with _sc1:
                st.markdown(
                    f'<div style="background:{_BG2_SL};border:2px solid rgba(196,69,69,0.3);border-radius:8px;padding:12px;">'
                    f'<div style="color:{_RED_SL};font-weight:700;font-size:0.8rem;">STOP LOSS</div>'
                    f'<div style="color:{_TEXT_SL};font-size:1.1rem;font-weight:700;">{_sl_val:.2f}</div>'
                    f'<div style="color:{_DIM_SL};font-size:0.65rem;">{_WOLF_ATR_MULT:g} × ATR = {_atr_stop_sl:.2f}</div>'
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
                    f'<div style="background:{_BG2_SL};border:2px solid rgba(0,229,255,0.2);border-radius:8px;padding:12px;">'
                    f'<div style="color:{_CYAN_SL};font-weight:700;font-size:0.8rem;">POSITION</div>'
                    f'<div style="color:{_TEXT_SL};font-size:0.85rem;">{_shares_sl} aktier</div>'
                    f'<div style="color:{_TEXT_SL};font-size:0.85rem;">{_pos_val_sl:,.0f} SEK ({_pos_pct_sl:.1f}%)</div>'
                    f'<div style="color:{_RED_SL};font-size:0.72rem;">Risk: {_risk_amt_sl:,.0f} SEK</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    except Exception:
        pass
