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
    from ui.theme import PALETTE as _P
    inject_css()

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="border-top:2px solid {_P["gold"]};padding-top:14px;margin-bottom:4px">'
        f'  <span style="font-family:\'Courier New\',monospace;font-size:18px;font-weight:900;'
        f'  letter-spacing:0.2em;text-transform:uppercase;color:{_P["gold"]}">🐺 WOLF SCREENER</span>'
        f'  <span style="font-family:\'Courier New\',monospace;font-size:10px;letter-spacing:0.25em;'
        f'  text-transform:uppercase;color:{_P["gold_muted"]};margin-left:16px">'
        f'  4-LAYER REGIME SCORING</span>'
        f'</div>'
        f'<p style="font-family:\'Courier New\',monospace;font-size:10px;letter-spacing:0.12em;'
        f'color:{_P["text_dim"]};margin:4px 0 20px 0">Market · Sector · Stock · Ichimoku</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── ZONE 1 — CONTROL BAR ────────────────────────────────────────────────
    section_title("CONTROL PANEL", "⚙")

    c1, c2, c3 = st.columns([1.5, 1.5, 1])
    with c1:
        market_opt = st.selectbox(
            "SELECT MARKET",
            ["All", "Commodity", "S&P 500", "US Mid Cap", "US Small Cap",
             "Stockholm", "Oslo", "Copenhagen", "Helsinki",
             "Europe", "Canada", "Junior Miners"],
            key="screener_market",
        )
    with c2:
        min_score = st.slider(
            "MIN REGIME SCORE",
            min_value=0, max_value=125,
            value=50,
            key="screener_min_score",
        )
    with c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("⚡ SCAN", key="screener_run", type="primary", width='stretch')

    # Nordic market universe selector (always visible below control bar)
    try:
        if _TICKER_UNIVERSE_AVAILABLE and _TU_REGIONS:
            region_options = list(_TU_REGIONS.keys())
            selected_regions_wolf = st.multiselect(
                "NORDIC MARKETS",
                region_options,
                default=["Norden"],
                key="wolf_screener_markets",
            )
            universe_tickers_wolf = get_tickers_for_regions(selected_regions_wolf)
            st.caption(f"{len(universe_tickers_wolf)} tickers in universe")
            intl_tickers = [
                t for t in universe_tickers_wolf
                if not any(t.endswith(s) for s in [".ST", ".OL", ".CO", ".HE"])
            ]
        else:
            intl_tickers = []
    except Exception:
        intl_tickers = []

    with st.expander("Advanced"):
        from utils.presets import PRESET_LABELS
        screener_preset = st.selectbox(
            "PRESET",
            PRESET_LABELS,
            key="screener_preset",
            help="Select parameter preset or Auto-detect based on ticker",
        )
        st.markdown(
            f'<div style="font-family:\'Courier New\',monospace;font-size:11px;'
            f'letter-spacing:1px;padding:10px;border:1px solid {_P["border"]};'
            f'border-radius:6px;margin-top:8px">'
            f'<div style="color:{_P["gold_muted"]};letter-spacing:3px;font-size:9px;'
            f'margin-bottom:8px">SCORE LEGEND</div>'
            f'<div style="color:{_P["green"]}">■ STRONG ≥ 70</div>'
            f'<div style="color:{_P["amber"]}">■ MODERATE ≥ 50</div>'
            f'<div style="color:{_P["red"]}">■ WEAK &lt; 50</div>'
            f'<div style="margin-top:6px;color:{_P["gold"]}">✦ YES = Entry Signal</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

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

            _scan_keys = selected_markets if selected_markets is not None else list(MARKETS.keys())
            _stock_tickers = [
                t for key in _scan_keys for t in MARKETS.get(key, {}).keys()
            ]
            _regime_tickers = ["SPY"] + list(SECTOR_MAP.values())
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

        # Enrich: missing names via Börsdata metadata
        if "Name" in df_results.columns and "Ticker" in df_results.columns:
            _missing_name = df_results["Name"].isna() | (df_results["Name"] == "")
            if _missing_name.any():
                for idx in df_results.index[_missing_name]:
                    _meta = bd.get_metadata(df_results.at[idx, "Ticker"])
                    if _meta and _meta.get("name"):
                        df_results.at[idx, "Name"] = _meta["name"]

        # Enrich: missing sector via Börsdata sectors
        if "Sector" in df_results.columns and "Ticker" in df_results.columns:
            _bd_sectors = {s["id"]: s["name"] for s in (bd.get_sectors() or [])}
            _missing_sector = df_results["Sector"].isna() | (df_results["Sector"] == "")
            if _missing_sector.any() and _bd_sectors:
                for idx in df_results.index[_missing_sector]:
                    _meta = bd.get_metadata(df_results.at[idx, "Ticker"])
                    if _meta and _meta.get("sector_id") in _bd_sectors:
                        df_results.at[idx, "Sector"] = _bd_sectors[_meta["sector_id"]]

        # ── Sentiment scores (Wolf strategy plugins) ─────────────────────────
        _sent_col_map: list = []
        try:
            from strategies.sentiment_utils import compute_sentiment_bias
            from strategies.registry import STRATEGIES
            from sentiment.registry import SENTIMENT_PLUGINS as _ALL_SENT

            _wolf_strat   = STRATEGIES.get("Wolf") or STRATEGIES.get("wolf") or {}
            _wolf_plugins = _wolf_strat.get("sentiment_plugins", ["ovtlyr_fg", "retail_flow"])

            for _pkey in _wolf_plugins:
                _pmeta = _ALL_SENT.get(_pkey, {})
                _label = _pmeta.get("name", _pkey)[:8]
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

        # ── Alerts ────────────────────────────────────────────────────────────
        try:
            from alerts.engine import send_alert as _send_alert
            from strategies.registry import STRATEGIES as _STRATS

            _strat     = _STRATS.get("Wolf") or _STRATS.get("wolf") or {}
            _alerts_on = _strat.get("alerts_enabled", False)
            _channels  = _strat.get("alert_channels", ["discord"])

            if _alerts_on:
                for _, _row in df_results[df_results["Entry Signal"] == "YES"].iterrows():
                    _t  = _row["Ticker"]
                    _sc = int(_row.get("Total Score", 0))
                    _rs = float(_row.get("RSI", 0))
                    _send_alert(
                        f"Wolf x Shadow ENTRY — {_t} | Score: {_sc}/125 | RSI: {_rs:.1f}",
                        _channels,
                        metadata={
                            "ticker": _t, "signal": "entry", "score": _sc,
                            "title": f"Entry Signal — {_t}", "color": 0x2D8A4E,
                        },
                    )

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
                                    "ticker": _t, "signal": "sentiment_extreme",
                                    "plugin": _pkey, "score": _sc,
                                    "title": f"Sentiment Extreme — {_t}",
                                    "color": 0xC9A84C,
                                },
                            )

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
                        if {_prev_cat, _cat} >= {"bull", "bear"}:
                            _shift = f"{_prev_cat.upper()} → {_cat.upper()}"
                            _send_alert(
                                f"Regime shift — {_t} {_shift} (score: {int(_row['Total Score'])})",
                                _channels,
                                metadata={
                                    "ticker": _t, "signal": "regime_shift",
                                    "shift": _shift,
                                    "title": f"Regime Shift — {_t}",
                                    "color": 0xC44545 if _cat == "bear" else 0x2D8A4E,
                                },
                            )
                st.session_state["screener_prev_regime"] = _curr_regime
        except Exception:
            pass  # alerts are optional — never block the scan

        # ── ZONE 2 — RESULTS ────────────────────────────────────────────────
        section_title("RESULTS", "📊")

        n_total   = len(df_results)
        n_strong  = len(df_results[df_results["Total Score"] >= 70])
        n_entry   = len(df_results[df_results["Entry Signal"] == "YES"])
        top_score = int(df_results["Total Score"].max())
        top_tick  = df_results.iloc[0]["Ticker"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("RESULTS", n_total)
        k2.metric("STRONG REGIME ≥70", n_strong,
                  delta=f"{n_strong/n_total*100:.0f}% of results" if n_total else None)
        k3.metric("ENTRY SIGNALS", n_entry,
                  delta="🐺 Active" if n_entry > 0 else "None found")
        k4.metric("TOP SCORE", f"{top_score} — {top_tick}")

        st.markdown("<br>", unsafe_allow_html=True)

        _adx_cols    = ["ADX"] if "ADX" in df_results.columns else []
        _preset_cols = ["Preset"] if "Preset" in df_results.columns else []
        _sent_labels = [lbl for _, lbl in _sent_col_map if lbl in df_results.columns]
        _rank_cols   = (
            ["Rank_Composite", "RS_score"]
            if "Rank_Composite" in df_results.columns else []
        )
        _want_cols = (
            ["Ticker", "Name"] + _preset_cols +
            ["Total Score", "Market(30)", "Sector(30)",
             "Stock(50)", "Ichi(15)", "EMA Stack", "Entry Signal", "RSI"]
            + _adx_cols + _sent_labels + _rank_cols + ["Close"]
        )
        df_display = df_results[[c for c in _want_cols if c in df_results.columns]].copy()

        _top15_idx = set(df_display.index[:15])

        def _style_row(row):
            total    = row.get("Total Score", 0)
            is_entry = row.get("Entry Signal") == "YES"
            is_top15 = row.name in _top15_idx
            if is_entry:
                bg = f"background-color:{_P['green']}1e"
            elif is_top15 and total >= 70:
                bg = f"background-color:{_P['green']}12"
            elif total >= 70:
                bg = f"background-color:{_P['green']}0a"
            elif total >= 50:
                bg = f"background-color:{_P['amber']}0d"
            else:
                bg = f"background-color:{_P['red']}0a"
            border = f";border-left:3px solid {_P['gold']}" if is_top15 else ""
            return [bg + border] * len(row)

        def _style_score(val):
            if val >= 70:   return f"color:{_P['green']};font-weight:bold"
            elif val >= 50: return f"color:{_P['amber']};font-weight:bold"
            else:           return f"color:{_P['red']};font-weight:bold"

        def _style_entry(val):
            if val == "YES": return f"color:{_P['green']};font-weight:bold"
            return f"color:{_P['text_dim']}"

        _styler = df_display.style
        _map_fn = _styler.map if hasattr(_styler, "map") else _styler.applymap
        styled_df = (
            _styler
            .apply(_style_row, axis=1)
            .format({"RSI": "{:.1f}", "Close": "{:.2f}"})
        )
        styled_df = _map_fn(_style_score, subset=["Total Score"])
        styled_df = _map_fn(_style_entry, subset=["Entry Signal"])

        _col_config = {
            "Total Score":    st.column_config.NumberColumn("TOTAL SCORE", format="%d"),
            "Ticker":         st.column_config.TextColumn("TICKER"),
            "Name":           st.column_config.TextColumn("NAME"),
            "Market(30)":     st.column_config.NumberColumn("MKT(30)",   format="%d"),
            "Sector(30)":     st.column_config.NumberColumn("SEC(30)",   format="%d"),
            "Stock(50)":      st.column_config.NumberColumn("STK(50)",   format="%d"),
            "Ichi(15)":       st.column_config.NumberColumn("ICHI(15)",  format="%d"),
            "EMA Stack":      st.column_config.TextColumn("EMA STACK"),
            "Entry Signal":   st.column_config.TextColumn("ENTRY"),
            "RSI":            st.column_config.NumberColumn("RSI",       format="%.1f"),
            "ADX":            st.column_config.NumberColumn("ADX",       format="%.1f"),
            "Preset":         st.column_config.TextColumn("PRESET"),
            "Close":          st.column_config.NumberColumn("CLOSE",     format="%.2f"),
            "Rank_Composite": st.column_config.NumberColumn("RANK SCORE", format="%.1f"),
            "RS_score":       st.column_config.NumberColumn("RS",        format="%.1f"),
            **{
                lbl: st.column_config.NumberColumn(lbl.upper(), format="%d")
                for _, lbl in _sent_col_map
            },
        }

        st.dataframe(
            styled_df,
            width='stretch',
            hide_index=True,
            height=min(600, 40 + len(df_display) * 35),
            column_config=_col_config,
        )

        # ── ZONE 3 — ENTRY SIGNALS ──────────────────────────────────────────
        entries = df_results[df_results["Entry Signal"] == "YES"]
        if not entries.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            section_title(f"ENTRY SIGNALS ({len(entries)})", "✦")
            _entry_want = [
                "Ticker", "Name", "Total Score", "Entry Zone",
                "SL (1.5 ATR)", "TP1 (2R)", "TP2 (3R)", "RSI", "ATR",
            ]
            entry_display = entries[
                [c for c in _entry_want if c in entries.columns]
            ].copy()
            st.dataframe(
                entry_display.style
                .set_properties(**{
                    "background-color": f"{_P['green']}14",
                    "color":            _P["green"],
                    "font-family":      "Courier New, monospace",
                    "font-size":        "12px",
                })
                .set_table_styles([{
                    "selector": "thead th",
                    "props": [
                        ("background-color", _P["surface"]),
                        ("color",            _P["gold_muted"]),
                        ("font-size",        "10px"),
                        ("letter-spacing",   "2px"),
                    ],
                }]),
                width='stretch',
                hide_index=True,
            )
    else:
        st.markdown(
            f'<div style="text-align:center;padding:80px 20px;'
            f'border:1px dashed {_P["border"]};border-radius:8px;margin-top:20px">'
            f'  <div style="font-size:48px;margin-bottom:16px;">🐺</div>'
            f'  <div style="color:{_P["gold_muted"]};letter-spacing:4px;font-size:14px;'
            f'  margin-bottom:8px">READY TO HUNT</div>'
            f'  <div style="color:{_P["border"]};font-size:11px;letter-spacing:2px">'
            f'  SELECT MARKET · SET MIN SCORE · CLICK SCAN</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_ovtlyr_screener_ui():
    """Viking Screener with z-score weighted scoring."""
    from ui.theme import PALETTE as _P
    try:
        # ── Header ──────────────────────────────────────────────────────────
        st.markdown(
            f'<div style="border-top:2px solid {_P["gold"]};padding-top:14px;margin-bottom:4px">'
            f'  <span style="font-family:\'Courier New\',monospace;font-size:18px;font-weight:900;'
            f'  letter-spacing:0.2em;text-transform:uppercase;color:{_P["gold"]}">⚡ VIKING SCREENER</span>'
            f'  <span style="font-family:\'Courier New\',monospace;font-size:10px;letter-spacing:0.25em;'
            f'  text-transform:uppercase;color:{_P["gold_muted"]};margin-left:16px">'
            f'  Z-SCORE NORMALIZED</span>'
            f'</div>'
            f'<p style="font-family:\'Courier New\',monospace;font-size:10px;letter-spacing:0.12em;'
            f'color:{_P["text_dim"]};margin:4px 0 20px 0">'
            f'Trend 30% · Momentum 25% · Volatility 15% · Volume 15% · ADX 15%</p>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── ZONE 1 — CONTROL BAR ────────────────────────────────────────────
        section_title("CONTROL PANEL", "⚙")

        try:
            if _TICKER_UNIVERSE_AVAILABLE and _TU_REGIONS:
                region_options = list(_TU_REGIONS.keys())
                selected_regions = st.multiselect(
                    "MARKETS",
                    region_options,
                    default=["Norden"],
                    key="ovtlyr_screener_markets",
                )
                universe_tickers = get_tickers_for_regions(selected_regions)
                st.caption(f"{len(universe_tickers)} tickers in universe")
            else:
                universe_tickers = None
                selected_regions = []
        except Exception:
            universe_tickers = None
            selected_regions = []

        c1, c2 = st.columns([3, 1])
        with c1:
            min_vol = st.number_input(
                "MIN AVG DAILY VOLUME",
                value=100_000, step=50_000,
                key="ovtlyr_minvol",
            )
        with c2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            scan_clicked = st.button("↻  SCAN", key="ovtlyr_scan", type="primary", width='stretch')

        # test_clicked defined here so it's always in scope regardless of expander state
        test_clicked = False
        top_n = 10
        with st.expander("Advanced"):
            top_n = st.number_input(
                "Top N for Backtest",
                value=10, min_value=3, max_value=50,
                key="ovtlyr_topn",
            )
            test_clicked = st.button("⚡ TEST TOP N", key="ovtlyr_test_topn", width='stretch')

        st.divider()

        # ── Scan execution ───────────────────────────────────────────────────
        if scan_clicked or test_clicked:
            if not _OVTLYR_SCREENER_AVAILABLE:
                st.error("Viking Screener module not found.")
                return

            if universe_tickers is not None and not universe_tickers:
                st.warning("Select at least one market.")
                return

            with st.spinner("⚡ Scanning universe..."):
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

        elif "ovtlyr_results" in st.session_state:
            results = st.session_state["ovtlyr_results"]
        else:
            st.markdown(
                f'<div style="text-align:center;padding:80px 20px;'
                f'border:1px dashed {_P["border"]};border-radius:8px;margin-top:20px">'
                f'  <div style="font-size:48px;margin-bottom:16px;">⚡</div>'
                f'  <div style="color:{_P["gold_muted"]};letter-spacing:4px;font-size:14px;'
                f'  margin-bottom:8px">READY TO SCAN</div>'
                f'  <div style="color:{_P["border"]};font-size:11px;letter-spacing:2px">'
                f'  SELECT MARKETS · SET VOLUME THRESHOLD · CLICK SCAN</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            return

        # ── ZONE 2 — RESULTS ────────────────────────────────────────────────
        section_title("RESULTS", "📊")

        total      = len(results)
        strong_buy = len(results[results["Signal"] == "STRONG BUY"])
        buy_count  = len(results[results["Signal"] == "BUY"])
        top_comp   = (
            float(results["Rank_Composite"].max())
            if "Rank_Composite" in results.columns
            else float(results["Composite"].max())
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("SCANNED", total)
        k2.metric("STRONG BUY", strong_buy,
                  delta=f"{strong_buy/total*100:.0f}% of results" if total else None)
        k3.metric("BUY", buy_count)
        k4.metric("TOP COMPOSITE", f"{top_comp:.1f}")

        st.markdown("<br>", unsafe_allow_html=True)

        _top15_idx = set(results.index[:15])

        def _signal_color(val):
            colors = {
                "STRONG BUY": f"color:{_P['gold']};font-weight:bold",
                "BUY":        f"color:{_P['green']};font-weight:bold",
                "HOLD":       f"color:{_P['amber']};font-weight:bold",
                "SELL":       f"color:{_P['red']};font-weight:bold",
            }
            return colors.get(val, "")

        def _style_top15(row):
            if row.name in _top15_idx:
                return [
                    f"background-color:{_P['gold_faint']};border-left:3px solid {_P['gold']}"
                ] * len(row)
            return [""] * len(row)

        styled = results.style
        _map = styled.map if hasattr(styled, "map") else styled.applymap
        styled = styled.apply(_style_top15, axis=1)
        styled = _map(_signal_color, subset=["Signal"])
        st.dataframe(styled, width='stretch', hide_index=True,
                     height=min(600, 38 + 35 * len(results)))

        # Test Top N
        if test_clicked:
            top_tickers = results.head(int(top_n))["Ticker"].tolist()
            st.session_state["test_topn_tickers"] = top_tickers
            st.session_state["test_topn_mode"] = "ovtlyr"
            st.session_state["auto_run_backtest"] = True
            st.success(f"Top {len(top_tickers)} tickers queued for backtest: {', '.join(top_tickers)}")
            st.info("→ Switch to the BACKTEST tab to see results.")

        # ── ZONE 3 — TOP 15 DETAIL ──────────────────────────────────────────
        top15 = results.head(15)
        if not top15.empty:
            section_title("TOP 15 DETAIL", "🎯")
            for rank_i, (_, row) in enumerate(top15.iterrows(), start=1):
                ticker  = row.get("Ticker", "?")
                signal  = row.get("Signal", "?")
                comp    = float(row.get("Composite", 0))
                rank_c  = float(row.get("Rank_Composite", comp))
                v9      = row.get("V9", "—")
                oc      = row.get("OC", "—")

                with st.expander(
                    f"#{rank_i:02d}  {ticker}  —  {signal}"
                    f"  │  Composite {comp:.1f}  │  Rank Score {rank_c:.1f}",
                    expanded=False,
                ):
                    dc1, dc2, dc3 = st.columns(3)
                    dc1.metric("COMPOSITE",  f"{comp:.1f}")
                    dc2.metric("RANK SCORE", f"{rank_c:.1f}")
                    dc3.metric("SIGNAL",     signal)

                    sub_cols = st.columns(5)
                    for i, (lbl, key) in enumerate([
                        ("TREND",     "trend"),
                        ("MOMENTUM",  "momentum"),
                        ("VOLATILITY","volatility"),
                        ("VOLUME",    "volume"),
                        ("ADX",       "adx"),
                    ]):
                        sub_cols[i].metric(lbl, f"{float(row.get(key, 0)):.1f}")

                    st.markdown(
                        f'<div style="font-family:\'Courier New\',monospace;font-size:10px;'
                        f'color:{_P["text_dim"]};margin-top:8px">'
                        f'Viking Nine: <b style="color:{_P["gold"]}">{v9}</b>'
                        f'&nbsp;·&nbsp;OC Status: <b style="color:{_P["gold"]}">{oc}</b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

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
