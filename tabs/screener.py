import streamlit as st

from ui.theme import inject_css, section_title
from utils.bd_api import BDClient, load_api_key

# Module-level Börsdata client.
# Used for metadata and sector enrichment; underlying screener modules
# (wolf_shadow_screener, screener_ovtlyr) use their own data pipelines.
bd = BDClient(load_api_key())

try:
    from cagr.cagr_streamlit import render_cagr_page
    _CAGR_AVAILABLE = True
except ImportError:
    _CAGR_AVAILABLE = False

try:
    from screener_ovtlyr import run_ovtlyr_screener
    _OVTLYR_SCREENER_AVAILABLE = True
except ImportError:
    _OVTLYR_SCREENER_AVAILABLE = False

try:
    from ticker_universe import COUNTRY_REGIONS as _TU_REGIONS, get_tickers_for_regions
    _TICKER_UNIVERSE_AVAILABLE = True
except ImportError:
    _TU_REGIONS = {}
    _TICKER_UNIVERSE_AVAILABLE = False


def tab_screener():
    inject_css()
    section_title("Market Scanner — 4-Layer Regime Scoring")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 0.6])

    with col1:
        market_opt = st.selectbox(
            "SELECT MARKET",
            ["All", "Commodity", "S&P 500", "US Mid Cap", "US Small Cap",
             "Stockholm", "Oslo", "Copenhagen", "Helsinki",
             "Europe", "Canada", "Junior Miners"],
            key="screener_market",
        )

    with col2:
        min_score = st.slider(
            "MINIMUM REGIME SCORE",
            min_value=0, max_value=125,
            value=50,
            key="screener_min_score",
        )

    with col3:
        from utils.presets import PRESET_LABELS
        screener_preset = st.selectbox(
            "PRESET",
            PRESET_LABELS,
            key="screener_preset",
            help="Select parameter preset or Auto-detect based on ticker",
        )

    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("⚡ SCAN", key="screener_run", width='stretch')

    try:
        if _TICKER_UNIVERSE_AVAILABLE and _TU_REGIONS:
            region_options = list(_TU_REGIONS.keys())
            selected_regions_wolf = st.multiselect(
                "Marknader",
                region_options,
                default=["Norden"],
                key="wolf_screener_markets",
            )
            universe_tickers_wolf = get_tickers_for_regions(selected_regions_wolf)
            st.caption(f"{len(universe_tickers_wolf)} aktier i universumet")
            intl_tickers = [t for t in universe_tickers_wolf if not any(t.endswith(s) for s in [".ST", ".OL", ".CO", ".HE"])]
        else:
            intl_tickers = []
    except Exception:
        intl_tickers = []

    st.markdown("---")

    lcol, rcol = st.columns([3, 1])
    with rcol:
        st.markdown("""
        <div style="border:1px solid rgba(201,168,76,0.15);border-radius:6px;padding:12px;font-size:11px;letter-spacing:1px;">
            <div style="color:rgba(201,168,76,0.5);letter-spacing:3px;margin-bottom:8px;">SCORE LEGEND</div>
            <div><span class="score-green">■</span> &nbsp;STRONG &nbsp;&ge; 70</div>
            <div><span class="score-yellow">■</span> &nbsp;MODERATE &nbsp;&ge; 50</div>
            <div><span class="score-red">■</span> &nbsp;WEAK &nbsp;&lt; 50</div>
            <div style="margin-top:8px;"><span class="entry-yes">✦ YES</span> &nbsp;= Entry Signal</div>
        </div>
        """, unsafe_allow_html=True)

    if run_btn:
        try:
            from wolf_shadow_screener import run_screener, MARKETS, SECTOR_MAP

            if intl_tickers:
                MARKETS["intl_expanded"] = {t: t.split(".")[0] for t in intl_tickers}

            market_map = {
                "All": None,
                "Commodity": ["commodity"],
                "S&P 500": ["sp500"],
                "US Mid Cap": ["us_midcap"],
                "US Small Cap": ["us_smallcap"],
                "Stockholm": ["stockholm"],
                "Oslo": ["oslo"],
                "Copenhagen": ["copenhagen"],
                "Helsinki": ["helsinki"],
                "Europe": ["europe"],
                "Canada": ["canada"],
                "Junior Miners": ["junior_miners"],
            }
            selected_markets = market_map[market_opt]
            if intl_tickers and selected_markets is not None:
                selected_markets = selected_markets + ["intl_expanded"]

            # Collect every ticker that run_screener will need so we can
            # pre-fetch them all in a single batch call before scanning.
            _scan_keys = selected_markets if selected_markets is not None else list(MARKETS.keys())
            _stock_tickers = [
                t for key in _scan_keys for t in MARKETS.get(key, {}).keys()
            ]
            _regime_tickers = ["SPY"] + list(SECTOR_MAP.values())  # SPY + 11 sector ETFs
            _all_fetch = list(dict.fromkeys(_regime_tickers + _stock_tickers))

            with st.spinner("📡 Fetching price data from Börsdata..."):
                pre_fetched = bd.get_price_history_batch(_all_fetch, period="1y")

            with st.spinner("🐺 WOLF IS HUNTING... scanning markets..."):
                df_results = run_screener(
                    markets=selected_markets,
                    min_score=min_score,
                    pre_fetched=pre_fetched,
                )

        except Exception as e:
            st.error(f"Screener error: {e}")
            return

        if df_results is None or df_results.empty:
            st.warning("No stocks matched the criteria. Try lowering the minimum score.")
            return

        # Enrich results: fill missing names using Börsdata metadata.
        if "Name" in df_results.columns and "Ticker" in df_results.columns:
            _missing_name = df_results["Name"].isna() | (df_results["Name"] == "")
            if _missing_name.any():
                for idx in df_results.index[_missing_name]:
                    _meta = bd.get_metadata(df_results.at[idx, "Ticker"])
                    if _meta and _meta.get("name"):
                        df_results.at[idx, "Name"] = _meta["name"]

        # Enrich results: fill missing sector info using Börsdata sectors.
        if "Sector" in df_results.columns and "Ticker" in df_results.columns:
            _bd_sectors = {s["id"]: s["name"] for s in (bd.get_sectors() or [])}
            _missing_sector = df_results["Sector"].isna() | (df_results["Sector"] == "")
            if _missing_sector.any() and _bd_sectors:
                for idx in df_results.index[_missing_sector]:
                    _meta = bd.get_metadata(df_results.at[idx, "Ticker"])
                    if _meta and _meta.get("sector_id") in _bd_sectors:
                        df_results.at[idx, "Sector"] = _bd_sectors[_meta["sector_id"]]

        # ── Sentiment scores (Wolf strategy plugins) ─────────────────────────
        _sent_col_map: list = []  # [(plugin_key, column_label), ...]
        try:
            from strategies.sentiment_utils import compute_sentiment_bias
            from strategies.registry import STRATEGIES
            from sentiment.registry import SENTIMENT_PLUGINS as _ALL_SENT

            _wolf_strat   = STRATEGIES.get("Wolf") or STRATEGIES.get("wolf") or {}
            _wolf_plugins = _wolf_strat.get("sentiment_plugins", ["ovtlyr_fg", "retail_flow"])

            # Build human-readable column labels from plugin metadata
            for _pkey in _wolf_plugins:
                _pmeta = _ALL_SENT.get(_pkey, {})
                _label = _pmeta.get("name", _pkey)[:8]  # keep column width sane
                _sent_col_map.append((_pkey, _label))
                df_results[_label] = 50

            for _idx, _row in df_results.iterrows():
                _tick = _row["Ticker"]
                _pdf  = pre_fetched.get(_tick)
                if _pdf is not None and len(_pdf) >= 15:
                    _sbias = compute_sentiment_bias(_pdf, _wolf_plugins)
                    for _pkey, _col_label in _sent_col_map:
                        _sig = _sbias["plugins"].get(_pkey)
                        if _sig:
                            df_results.at[_idx, _col_label] = int(round(_sig["score"]))
        except Exception:
            pass  # sentiment is optional — never block the scan

        # ── Alerts ───────────────────────────────────────────────────────────
        try:
            from alerts.engine import send_alert as _send_alert
            from strategies.registry import STRATEGIES as _STRATS

            _strat     = _STRATS.get("Wolf") or _STRATS.get("wolf") or {}
            _alerts_on = _strat.get("alerts_enabled", False)
            _channels  = _strat.get("alert_channels", ["discord"])

            if _alerts_on:
                # Entry signal alerts
                for _, _row in df_results[df_results["Entry Signal"] == "YES"].iterrows():
                    _t  = _row["Ticker"]
                    _sc = int(_row.get("Total Score", 0))
                    _rs = float(_row.get("RSI", 0))
                    _send_alert(
                        f"Wolf x Shadow ENTRY — {_t} | Score: {_sc}/125 | RSI: {_rs:.1f}",
                        _channels,
                        metadata={
                            "ticker": _t,
                            "signal": "entry",
                            "score":  _sc,
                            "title":  f"Entry Signal — {_t}",
                            "color":  0x2D8A4E,
                        },
                    )

                # Sentiment extreme alerts (from already-computed columns)
                for _pkey, _col_label in _sent_col_map:
                    if _col_label not in df_results.columns:
                        continue
                    for _, _row in df_results.iterrows():
                        _sc = int(_row.get(_col_label, 50))
                        if _sc > 90 or _sc < 10:
                            _t = _row["Ticker"]
                            _direction = "EXTREME GREED" if _sc > 90 else "EXTREME FEAR"
                            _send_alert(
                                f"Sentiment extreme — {_t} [{_pkey}] {_sc} ({_direction})",
                                _channels,
                                metadata={
                                    "ticker": _t,
                                    "signal": "sentiment_extreme",
                                    "plugin": _pkey,
                                    "score":  _sc,
                                    "title":  f"Sentiment Extreme — {_t}",
                                    "color":  0xC9A84C,
                                },
                            )

                # Regime shift alerts — compare Total Score category to previous scan
                def _regime_cat(score: int) -> str:
                    if score >= 70:  return "bull"
                    if score <  50:  return "bear"
                    return "neutral"

                _prev_regime = st.session_state.get("screener_prev_regime", {})
                _curr_regime: dict = {}
                for _, _row in df_results.iterrows():
                    _t   = _row["Ticker"]
                    _cat = _regime_cat(int(_row.get("Total Score", 50)))
                    _curr_regime[_t] = _cat
                    _prev_cat = _prev_regime.get(_t)
                    if _prev_cat is not None and _prev_cat != _cat:
                        if {_prev_cat, _cat} >= {"bull", "bear"}:  # only direct bull↔bear flips
                            _shift = f"{_prev_cat.upper()} → {_cat.upper()}"
                            _send_alert(
                                f"Regime shift — {_t} {_shift} (score: {int(_row['Total Score'])})",
                                _channels,
                                metadata={
                                    "ticker": _t,
                                    "signal": "regime_shift",
                                    "shift":  _shift,
                                    "title":  f"Regime Shift — {_t}",
                                    "color":  0xC44545 if _cat == "bear" else 0x2D8A4E,
                                },
                            )
                st.session_state["screener_prev_regime"] = _curr_regime
        except Exception:
            pass  # alerts are optional — never block the scan

        n_total   = len(df_results)
        n_strong  = len(df_results[df_results["Total Score"] >= 70])
        n_entry   = len(df_results[df_results["Entry Signal"] == "YES"])
        top_score = int(df_results["Total Score"].max())
        top_tick  = df_results.iloc[0]["Ticker"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("STOCKS SCANNED", n_total)
        k2.metric("STRONG REGIME (≥70)", n_strong,
                  delta=f"{n_strong/n_total*100:.0f}% of results" if n_total else None)
        k3.metric("ENTRY SIGNALS", n_entry,
                  delta="🐺 Active" if n_entry > 0 else "None found")
        k4.metric("TOP SCORE", f"{top_score} — {top_tick}")

        st.markdown("<br>", unsafe_allow_html=True)

        _adx_cols  = ["ADX"] if "ADX" in df_results.columns else []
        _preset_cols = ["Preset"] if "Preset" in df_results.columns else []
        _sent_labels = [lbl for _, lbl in _sent_col_map if lbl in df_results.columns]
        display_cols = (
            ["Ticker", "Name"] + _preset_cols +
            ["Total Score", "Market(30)", "Sector(30)",
             "Stock(50)", "Ichi(15)", "EMA Stack", "Entry Signal", "RSI"]
            + _adx_cols + _sent_labels + ["Close"]
        )
        df_display = df_results[display_cols].copy()

        def style_screener_row(row):
            total = row["Total Score"]
            is_entry = row["Entry Signal"] == "YES"
            if is_entry:
                bg = "background-color: rgba(45,138,78,0.12)"
            elif total >= 70:
                bg = "background-color: rgba(45,138,78,0.06)"
            elif total >= 50:
                bg = "background-color: rgba(212,148,58,0.05)"
            else:
                bg = "background-color: rgba(196,69,69,0.04)"
            return [bg] * len(row)

        def style_score_cell(val):
            if val >= 70:
                return "color: #2d8a4e; font-weight: bold"
            elif val >= 50:
                return "color: #d4943a; font-weight: bold"
            else:
                return "color: #c44545; font-weight: bold"

        def style_entry_cell(val):
            if val == "YES":
                return "color: #2d8a4e; font-weight: bold"
            return "color: rgba(232,228,220,0.3)"

        _styler = df_display.style
        _map_fn = _styler.map if hasattr(_styler, "map") else _styler.applymap
        styled_df = (
            _styler
            .apply(style_screener_row, axis=1)
            .format({"RSI": "{:.1f}", "Close": "{:.2f}"})
        )
        styled_df = _map_fn(style_score_cell, subset=["Total Score"])
        styled_df = _map_fn(style_entry_cell, subset=["Entry Signal"])

        st.dataframe(
            styled_df,
            width='stretch',
            hide_index=True,
            height=min(600, 40 + len(df_display) * 35),
            column_config={
                "Total Score": st.column_config.NumberColumn("TOTAL SCORE", help="Composite regime score (max 125)", format="%d"),
                "Ticker":       st.column_config.TextColumn("TICKER"),
                "Name":         st.column_config.TextColumn("NAME"),
                "Market(30)":   st.column_config.NumberColumn("MKT(30)",  format="%d"),
                "Sector(30)":   st.column_config.NumberColumn("SEC(30)",  format="%d"),
                "Stock(50)":    st.column_config.NumberColumn("STK(50)",  format="%d"),
                "Ichi(15)":     st.column_config.NumberColumn("ICHI(15)", format="%d"),
                "EMA Stack":    st.column_config.TextColumn("EMA STACK"),
                "Entry Signal": st.column_config.TextColumn("ENTRY"),
                "RSI":          st.column_config.NumberColumn("RSI",    format="%.1f"),
                "ADX":          st.column_config.NumberColumn("ADX",    format="%.1f"),
                "Preset":       st.column_config.TextColumn("PRESET"),
                "Close":        st.column_config.NumberColumn("CLOSE",  format="%.2f"),
                **{
                    lbl: st.column_config.NumberColumn(
                        lbl.upper(), format="%d",
                        help=f"Sentiment score 0-100 ({lbl})",
                    )
                    for _, lbl in _sent_col_map
                },
            },
        )

        entries = df_results[df_results["Entry Signal"] == "YES"]
        if not entries.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            section_title(f"Active Entry Signals ({len(entries)})")
            entry_display = entries[[
                "Ticker", "Name", "Total Score", "Entry Zone",
                "SL (1.5 ATR)", "TP1 (2R)", "TP2 (3R)", "RSI", "ATR"
            ]].copy()
            st.dataframe(
                entry_display.style
                .set_properties(**{
                    "background-color": "rgba(45,138,78,0.08)",
                    "color": "#2d8a4e",
                    "font-family": "Courier New, monospace",
                    "font-size": "12px",
                })
                .set_table_styles([{
                    "selector": "thead th",
                    "props": [
                        ("background-color", "#10101a"),
                        ("color", "rgba(201,168,76,0.6)"),
                        ("font-size", "10px"),
                        ("letter-spacing", "2px"),
                    ]
                }]),
                width='stretch',
                hide_index=True,
            )
    else:
        st.markdown("""
        <div style="
            text-align:center;
            padding:80px 20px;
            border:1px dashed rgba(201,168,76,0.15);
            border-radius:8px;
            margin-top:20px;
        ">
            <div style="font-size:48px;margin-bottom:16px;">🐺</div>
            <div style="color:rgba(201,168,76,0.5);letter-spacing:4px;font-size:14px;margin-bottom:8px;">
                READY TO HUNT
            </div>
            <div style="color:rgba(201,168,76,0.3);font-size:11px;letter-spacing:2px;">
                SELECT MARKET · SET MIN SCORE · CLICK SCAN
            </div>
        </div>
        """, unsafe_allow_html=True)


def _render_ovtlyr_screener_ui():
    """Viking Screener with z-score weighted scoring."""
    try:
        st.markdown(
            "<h2 style='color:#c9a84c;letter-spacing:0.1em;'>"
            "VIKING SCREENER</h2>"
            "<p style='color:#8a8578;font-size:0.7rem;'>Z-score normalized · "
            "Weighted composite · Trend 30% + Momentum 25% + Vol 15% + Volume 15% + ADX 15%</p>",
            unsafe_allow_html=True,
        )

        if _TICKER_UNIVERSE_AVAILABLE and _TU_REGIONS:
            region_options = list(_TU_REGIONS.keys())
            selected_regions = st.multiselect(
                "Marknader",
                region_options,
                default=["Norden"],
                key="ovtlyr_screener_markets",
            )
            universe_tickers = get_tickers_for_regions(selected_regions)
            st.caption(f"{len(universe_tickers)} aktier i universumet")
        else:
            universe_tickers = None
            selected_regions = []

        col1, col2 = st.columns(2)
        with col1:
            min_vol = st.number_input("Min Avg Volume", value=100_000, step=50_000, key="ovtlyr_minvol")
        with col2:
            top_n = st.number_input("Top N for Test", value=10, min_value=3, max_value=50, key="ovtlyr_topn")

        col_scan, col_test = st.columns(2)
        with col_scan:
            scan_clicked = st.button("↻  SCAN", key="ovtlyr_scan", width='stretch')
        with col_test:
            test_clicked = st.button("⚡ TEST TOP N", key="ovtlyr_test_topn", width='stretch')

        if scan_clicked or test_clicked:
            if not _OVTLYR_SCREENER_AVAILABLE:
                st.error("Viking Screener module not found.")
                return

            if universe_tickers is not None and not universe_tickers:
                st.warning("Välj minst en marknad.")
                return

            with st.spinner("🐺 Scanning universe..."):
                if universe_tickers is not None:
                    results = run_ovtlyr_screener(
                        universe="custom",
                        min_volume=min_vol,
                        ticker_list=tuple(universe_tickers),
                    )
                else:
                    results = run_ovtlyr_screener("Nordic", min_vol)

            if results.empty:
                st.warning("No results. Try a different universe or lower the volume filter.")
                return

            st.session_state["ovtlyr_results"] = results

            total = len(results)
            strong_buy = len(results[results["Signal"] == "STRONG BUY"])
            buy = len(results[results["Signal"] == "BUY"])
            hold = len(results[results["Signal"] == "HOLD"])
            sell = len(results[results["Signal"] == "SELL"])

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Scanned", total)
            k2.metric("STRONG BUY", strong_buy)
            k3.metric("BUY", buy)
            k4.metric("HOLD", hold)
            k5.metric("SELL", sell)

            def _signal_color(val):
                colors = {"STRONG BUY": "color:#c9a84c", "BUY": "color:#2d8a4e",
                          "HOLD": "color:#d4943a", "SELL": "color:#c44545"}
                return colors.get(val, "")

            styled = results.style
            _map = styled.map if hasattr(styled, "map") else styled.applymap
            styled = _map(_signal_color, subset=["Signal"])
            st.dataframe(styled, width='stretch', hide_index=True,
                         height=min(600, 38 + 35 * len(results)))

            if test_clicked:
                top_tickers = results.head(int(top_n))["Ticker"].tolist()
                st.session_state["test_topn_tickers"] = top_tickers
                st.session_state["test_topn_mode"] = "ovtlyr"
                st.session_state["auto_run_backtest"] = True
                st.success(f"Top {len(top_tickers)} tickers queued for backtest: {', '.join(top_tickers)}")
                st.info("→ Switch to the BACKTEST tab to see results.")

        elif "ovtlyr_results" in st.session_state:
            results = st.session_state["ovtlyr_results"]
            st.dataframe(results, width='stretch', hide_index=True,
                         height=min(600, 38 + 35 * len(results)))
    except Exception as e:
        st.error(f"Viking Screener error: {e}")


def tab_screener_consolidated():
    """Unified Screener tab with dropdown: Wolf / Alpha / Viking."""
    from ui.css import tab_not_found

    mode = st.selectbox(
        "SCREENER MODE",
        ["Wolf Screener", "Alpha Screener", "Viking Screener"],
        key="screener_mode_select",
    )

    if mode == "Wolf Screener":
        tab_screener()

    elif mode == "Alpha Screener":
        if _CAGR_AVAILABLE:
            render_cagr_page()
        else:
            tab_not_found("Alpha Screener", "cagr")

    elif mode == "Viking Screener":
        _render_ovtlyr_screener_ui()
