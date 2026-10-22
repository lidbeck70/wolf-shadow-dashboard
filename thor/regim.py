"""
thor/regim.py — Regim-, Teknik- och Regler-flikarna i Thors analysblock.

Regim: strategins eget regimbeslut för en aktie, i den layout du redan känner —
  Viking Nine   marknadsbannern, OVTLYR Nine-kortet, Viking Execution + Risk Engine, grafen, exitmotorn
  Wolf          REGIME SCORE n/125 med etiketten, de fyra mätarna, nivåerna, checklistan, elva grindar, SL/TP
  Momentum      Swing-domen (KÖP-KANDIDAT/BEVAKA/HÅLL/…) och de sex entry-reglerna
  BERSERK       temats drivarkort och aktiens setup (KÖP/BEVAKA, stopp, position)
  Alpha         VAD SKA JAG GÖRA? och signalkorten (Quality) eller ACCUMULATE-steget och gummibanden (Contrarian)
  Long Screener signalen, fundamenta 0–20, cykel 0–3 (Valhall gron), teknik 0–7, köpgrindar och säljtriggers
  EMBER         komplexregimen och trendgrindarna
Samma motorer som de gamla sidorna (ovtlyr_nine, viking_execution, wolf_regime_score, swing_verdict, berserk.live,
alpha_regime.engine, cagr, ember.regime) — inga regler ändras. Teknik = tech_analysis för aktien. Regler = hela
playbooken ur strategy_rules.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from thor import cards
from thor import registry as reg
from thor.ticker import Ticker
from ui.tokens import AMBER, CYAN, DIM, GREEN, GREY, RED, TEXT

TA_PERIODS = {"6 mån": "6mo", "1 år": "1y", "2 år": "2y"}


def _section(st, title: str) -> None:
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;"
                f"margin:12px 0 6px;'>{title}</div>", unsafe_allow_html=True)


def _frame(df):
    """Dagsdata utan tidszon, eller None."""
    if df is None or len(df) == 0:
        return None
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def _default_getter():
    from market_prices import ohlcv
    return ohlcv


# ── Regim ───────────────────────────────────────────────────────────────────
def render(st, ident: Ticker, strategy: str, data: dict, ohlcv: Optional[Callable] = None, cands: Optional[list] = None) -> None:
    """Rita strategins regimlayout för aktien. ohlcv(ticker, period) → OHLCV (market_prices som standard)."""
    from ui.components import note
    s = reg.get(strategy)
    layout = (s.regime if s else "") or "sheet"
    fn = LAYOUTS.get(layout, _sheet)
    try:
        fn(st, ident, strategy, data, ohlcv or _default_getter(), cands or [])
    except Exception as exc:
        note(f"Regimen för {reg.label(strategy)} kunde inte ritas för {ident.yf}: {exc}")


def bars_caption(nine) -> str:
    """Vilka stängningar motorn räknar på — aktien och marknaden kan sluta olika dagar (olika börstider), och
    kortet ovanför (schemats rad) kan bygga på en annan stängning än motorn här."""
    def stamp(layer):
        return next((f.timestamp for f in nine.layer(layer) if f.timestamp), None)
    s, m = stamp("stock"), stamp("market")
    parts = [f"aktien t.o.m. {s}" if s else "", f"marknaden ({nine.market_label}) t.o.m. {m}" if m else ""]
    return ("Stängda candles: " + " · ".join(p for p in parts if p) + ". Dagens stapel räknas först när börsen "
            "har stängt — kortet ovan bygger på körningens stängning, motorn här på den senaste.")


def _viking_nine(st, ident, strategy, data, ohlcv, cands) -> None:
    import ovtlyr_nine as on
    import viking_execution as vx
    import viking_screen as vs
    from ovtlyr.ui.execution_card import render_execution
    from ovtlyr.ui.exit_card import render_exit_section
    from ovtlyr.ui.nine_card import render_nine_card
    from ovtlyr.ui.viking_nine_regime import price_chart
    from ovtlyr.ui.viking_screens import _earnings, _market_banner, _risk_banner, _risk_for, _sector
    if strategy == "viking":
        from ui.components import note
        note("Motorn nedan är Viking Nine (OVTLYR Nine + Viking Execution) — Viking har ingen egen regimmotor per aktie. "
             "Vikings eget verdikt står i kortet ovan: screenerns BUY (över EMA200, ADX ≥ 20, composite ≥ 60) och dess "
             "egna nio faktorer, som inte är OVTLYR Nine. Viking Nines regel 7–8/9 = WAIT och rapportspärren gäller "
             "motorn nedan, inte screenerns siffra.")
    from ui.components import note
    df = _frame(ohlcv(ident.yf, on.PERIOD))
    if df is None or len(df) < on.MIN_BARS:
        note(f"DATA UNAVAILABLE — ingen kurshistorik för {ident.yf} (kräver {on.MIN_BARS} dagar).")
        return
    _market_banner((ident.market,))
    dfc = on.closed_only(df, ident.yf)                 # Nine och order blocks på stängda staplar (playbooken)
    ob = vs._ob_analysis(dfc)
    nine = on.evaluate(ident.yf, stock_df=dfc, ob_analysis=ob, sector_getter=_sector)
    _section(st, "OVTLYR NINE — SETUP")
    st.caption(bars_caption(nine))
    render_nine_card(nine)
    _section(st, "VIKING EXECUTION · RISK ENGINE · FINAL DECISION")
    k1, k2 = st.columns(2)
    capital = k1.number_input("Kapital (SEK)", min_value=0.0, value=100000.0, step=10000.0, key="thor_capital")
    max_pos = k2.number_input("Max position (% av kapitalet)", min_value=5.0, max_value=100.0,
                              value=float(vx.MAX_POSITION_PCT), step=5.0, key="thor_max_pos")
    try:
        from trade_journal import load_journal
        trades = load_journal()
    except Exception:
        trades = []
    earnings = _earnings(ident.yf)
    _risk_banner((ident.market,))
    decision = vx.evaluate_entry(ident.yf, df, nine=nine, capital=capital, ob_analysis=ob, earnings_date=earnings,
                                 earnings_known=earnings is not None, trades=trades, max_position_pct=max_pos,
                                 market_risk=_risk_for(ident.yf))
    try:
        st.plotly_chart(price_chart(ident.yf, df, decision), use_container_width=True, config={"displayModeBar": False},
                        key="thor_vn_chart")
    except Exception as exc:                                     # pragma: no cover - grafen är inte regeln
        note(f"Grafen kunde inte ritas: {exc}")
    render_execution(decision)
    render_exit_section(ident.yf, df, nine, ob, earnings)


def _wolf(st, ident, strategy, data, ohlcv, cands) -> None:
    import wolf_regime_score as wrs
    from ui.charts import build_gauge
    from ui.components import note
    sector = ident.sector_etf or "SPY"
    spy, sec, stk = _frame(ohlcv("SPY", "1y")), _frame(ohlcv(sector, "1y")), _frame(ohlcv(ident.yf, "1y"))
    if stk is None or len(stk) < 60:
        note(f"DATA UNAVAILABLE — Wolf kräver 60 dagars kurshistorik för {ident.yf}.")
        return
    d = wrs.regime(spy, sec, stk, timestamp=str(stk.index[-1].date()))
    lbl, col = wrs.label(d["total"])
    st.markdown(cards.wolf_score_card(d["total"], lbl, col, ident.yf, d["timestamp"]), unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    g1.plotly_chart(build_gauge(d["mkt_score"], 30, "MARKET (SPY)", color_cyan=True), use_container_width=True,
                    config={"displayModeBar": False}, key="thor_wolf_g1")
    g2.plotly_chart(build_gauge(d["sec_score"], 30, f"SECTOR ({sector})", color_cyan=False), use_container_width=True,
                    config={"displayModeBar": False}, key="thor_wolf_g2")
    g3, g4 = st.columns(2)
    g3.plotly_chart(build_gauge(d["stk_score"], 50, f"STOCK ({ident.yf})", color_cyan=True), use_container_width=True,
                    config={"displayModeBar": False}, key="thor_wolf_g3")
    g4.plotly_chart(build_gauge(d["ichi_score"], 15, "ICHIMOKU", color_cyan=False), use_container_width=True,
                    config={"displayModeBar": False}, key="thor_wolf_g4")
    pills = [("EMA STACK FULL" if d["ema_stack"] else "EMA STACK OFF", d["ema_stack"]),
             ("TREND ACTIVE" if d["ema_trend"] else "NO TREND", d["ema_trend"]),
             ("ENTRY LIVE" if d["has_entry"] else "NO ENTRY", d["has_entry"]),
             ("GATE OPEN" if d["total"] >= wrs.GATE_MIN else "GATE CLOSED", d["total"] >= wrs.GATE_MIN)]
    st.markdown("<div style='display:flex;flex-wrap:wrap;gap:4px;margin:6px 0;'>"
                + "".join(cards.chip(t, GREEN if ok else RED, filled=bool(ok)) for t, ok in pills) + "</div>",
                unsafe_allow_html=True)
    rsi_note = "överköpt" if d["rsi"] > 70 else "översåld" if d["rsi"] < 30 else "neutral"
    levels = [("Kurs", f"{d['close']:.2f}", TEXT, ""), ("RSI 14", f"{d['rsi']:.1f}", CYAN, rsi_note),
              ("ADX 14", f"{d['adx']:.1f}", CYAN, "trend" if d["adx"] >= 19 else "svag trend"),
              ("ATR 14", f"{d['atr']:.2f}", AMBER, ""), ("Entry", f"{d['entry_zone']:.2f}", TEXT, ""),
              ("Stopp 1,5 × ATR", f"{d['sl_level']:.2f}", RED, "sidans nivå"),
              ("TP1 2R", f"{d['tp1']:.2f}", GREEN, f"{(d['tp1'] / d['close'] - 1) * 100:+.1f} %" if d["close"] else ""),
              ("TP2 3R", f"{d['tp2']:.2f}", GREEN, f"{(d['tp2'] / d['close'] - 1) * 100:+.1f} %" if d["close"] else "")]
    st.markdown(cards.kpi_row(levels), unsafe_allow_html=True)
    try:
        chk = wrs.checklist(d, stk)
        gs = wrs.gates(d, chk)
        _section(st, "WOLF ENTRY CHECKLIST")
        st.markdown(cards.kpi_row([("Trend", chk["trend_dir"], chk["trend_color"], f"EMA-stack {'✓' if chk['stack'] else '✗'}"),
                                   ("Volatilitet", f"ATR ratio {chk['atr_ratio']:.2f}x", AMBER, f"hist vol {chk['hist_vol']:.0f} % · risk {chk['risk_score']}/100"),
                                   ("Momentum", f"RSI {d['rsi']:.1f} · {chk['rsi_state']}", CYAN, f"vol ratio {chk['vol_ratio']:.2f}x · ADX {d['adx']:.1f}"),
                                   ("Candlestick", chk["pattern_name"], AMBER, ""),
                                   ("Order blocks", f"{len(chk['bull_obs'])} bull · {len(chk['bear_obs'])} bear", CYAN,
                                    "OB inom 3 % av pris ✓" if chk["has_nearby_ob"] else "ingen OB inom 3 %")]),
                    unsafe_allow_html=True)
    except Exception as exc:
        gs = wrs.gates_unavailable(d, exc)
    _section(st, "SWING TRADING GATES — 11 REGLER")
    st.markdown(cards.gate_chips(gs, wrs.gate_summary(gs)), unsafe_allow_html=True)
    _section(st, "SL / TP KALKYLATOR — WOLF")
    k1, k2 = st.columns(2)
    capital = k1.number_input("Kapital (SEK)", min_value=0.0, value=100000.0, step=10000.0, key="thor_capital")
    risk_pct = k2.number_input("Risk %", value=float(wrs.RISK_PCT), min_value=0.5, max_value=10.0, step=0.5, key="thor_wolf_risk",
                               help=f"Wolf riskerar {wrs.RISK_PCT:g} % per affär (strategies/wolf.py: risk_pct).")
    factor, risk_text = 1.0, ""
    try:
        import market_risk_gate as mg
        mr = mg.for_ticker(ident.yf)
        factor = mg.size_factor(mr)
        if factor < 1:
            risk_text = f"🌩️ {mg.describe(mr)} — Wolf-positionen halveras."
    except Exception:
        pass
    p = wrs.sltp(stk, capital, risk_pct, size_factor=factor)
    if p:
        st.markdown(cards.kpi_row([("Stop loss", f"{p['sl']:.2f}", RED, f"{wrs.ATR_MULT:g} × ATR = {p['atr_stop']:.2f} · risk {p['sl_dist'] / p['price'] * 100:.1f} %"),
                                   ("Mål 2R / 3R", f"{p['tp_2r']:.2f} / {p['tp_3r']:.2f}", GREEN, "trailing stop — ej fast TP"),
                                   ("Position", f"{p['shares']} aktier", CYAN, f"{p['pos_val']:,.0f} SEK ({p['pos_pct']:.1f} %) · risk {p['risk_amt']:,.0f} SEK"),
                                   ("Trail", f"Kijun {p['kijun']:.2f}", AMBER, f"EMA10 {p['ema10']:.2f}")]), unsafe_allow_html=True)
    if risk_text:
        note(risk_text)
    note("Samma formler som ODINS VISDOM → Marknad → Arc Regime → Wolf Regime (wolf_regime_score.py). Entry kräver EMA-stack, trend "
         "och RSI 45–70; grindarna 8–11 bedömer du själv.")


def _momentum(st, ident, strategy, data, ohlcv, cands) -> None:
    import swing_verdict as sv
    from ui.components import note
    screener, regime, swing_data = {}, {}, {}
    try:
        import wolf_screener_ui
        screener = wolf_screener_ui._get_data() or {}
    except Exception:
        pass
    try:
        import wolf_regime_ui
        regime = wolf_regime_ui._get_data() or {}
    except Exception:
        pass
    try:
        import swing
        swing_data = swing.rules_data()
    except Exception:
        pass
    v = sv.verdict(ident.yf, screener, regime, swing_data)
    col = {sv.BUY: GREEN, sv.HOLD: GREEN, sv.PARTIAL: AMBER, sv.WATCH: AMBER, sv.SELL: RED, sv.ABSTAIN: RED}.get(v["verdict"], GREY)
    reasons = v.get("reasons") or []
    st.markdown(cards.verdict_card(v["verdict"] + (" · INNEHAV" if v.get("held") else ""), reasons[0] if reasons else "",
                                   " · ".join(reasons[1:3]), col, label="SWING-DOMEN"), unsafe_allow_html=True)
    row = sv.screener_row(ident.yf, screener) or {}
    if row:
        st.markdown(cards.kpi_row([("Rank", f"{row.get('rank')}/40", CYAN, f"regim {v.get('light') or '—'}"),
                                   ("Kurs", f"{row.get('price')}", TEXT, f"stopp −10 % = {float(row.get('price') or 0) * 0.9:.2f}"),
                                   ("Momentum 3 m", f"{row.get('mom3')}", AMBER, f"6 m {row.get('mom6')}"),
                                   ("RSI", f"{row.get('rsi')}", CYAN, f"MA20 {row.get('dist_ma20')} · MA50 {row.get('dist_ma50')}"),
                                   ("Setup", "A (pullback)" if row.get("setupA") else "B? (nära 52v)" if row.get("nearHigh") else "ingen",
                                    GREEN if row.get("setupA") else AMBER, "")]), unsafe_allow_html=True)
    _section(st, "MOMENTUM — ENTRY-REGLERNA")
    st.markdown(cards.check_rows(sv.rule_checks(ident.yf, screener, regime, swing_data)), unsafe_allow_html=True)
    note("Domen och reglerna är swing_verdict — samma som Swing Regime-flikens tickerkoll och Copiloten. "
         "Listan kommer ur wolf_screener.json (Swing Screener, Börsdata), regimljuset ur wolf_regime.json.")


def _berserk(st, ident, strategy, data, ohlcv, cands) -> None:
    from berserk import backtest as bt
    from berserk import live
    from berserk import universe as uv
    from berserk.regime_ui import theme_card
    from ui.components import note
    theme = ident.theme or uv.theme_of(ident.yf) or None
    if not theme:
        note(f"{ident.yf} har inget råvarutema — BERSERK kräver en drivare (råvaran). Inte i BERSERK-universumet.")
        return
    drivers = live.load_drivers([theme], getter=ohlcv)
    rows = [r for r in live.theme_states(drivers) if r["theme"] == theme]
    if rows:
        st.markdown(theme_card(rows[0]), unsafe_allow_html=True)
    stock = _frame(ohlcv(ident.yf, live.STOCK_PERIOD))
    region = uv.region_of(ident.yf)
    try:
        markets = live.load_markets([region], getter=ohlcv)
    except Exception:
        markets = {}
    cfg = bt.live_config()
    commodity = None
    try:
        commodity = bt.gate_series(cfg.gate_kind, lambda s: ohlcv(s, live.STOCK_PERIOD))
    except Exception:
        commodity = None
    sym, drv = drivers.get(theme, (None, None))
    mkt = markets.get(uv.REGION_INDEX.get(region, "SPY")) or markets.get("SPY")
    row = live.evaluate(ident.yf, stock, drv, mkt, cfg=cfg, driver_symbol=sym, commodity=commodity)
    if row.get("error"):
        note(f"{ident.yf}: {row['error']}")
        return
    status = row.get("status") or ""
    col = GREEN if status == live.KOP else AMBER if status == live.BEVAKA else GREY
    why = row.get("why") or []
    st.markdown(cards.verdict_card(status or "INGEN SETUP", why[0] if why else "ingen setup i dag",
                                   "; ".join(why[1:4]), col, label="BERSERK — AKTIEN"), unsafe_allow_html=True)
    st.markdown(cards.kpi_row([("Setup", row.get("setup") or "—", CYAN, f"drivare {sym or '—'}"),
                               ("Kurs", f"{row.get('close')}", TEXT, row.get("date") or ""),
                               ("Stopp", f"{row.get('stop') or '—'}", RED, f"ATR {row.get('atr') or '—'}"),
                               ("Position", f"{row.get('position_pct') or '—'} %", GREEN, f"{row.get('shares') or '—'} aktier · risk {row.get('risk_pct') or '—'} %"),
                               ("RSI(2) · rvol", f"{row.get('rsi2')} · {row.get('rvol')}", AMBER,
                                f"divergens {row.get('divergence')} pe · dd252 {row.get('dd252')} %")]), unsafe_allow_html=True)
    note("Samma motor som 🪓 BERSERK Regime och skannern (berserk.live.evaluate, live_config): S1 divergens, S2 vändning, "
         "råvarugrinden och marknadsgrinden. Temakortet är drivarens läge.")


def _alpha(st, ident, strategy, data, ohlcv, cands) -> None:
    from alpha_regime import ui as aui
    from alpha_regime.engine import run_regime_analysis
    mode = "contrarian" if strategy == "deep" else "quality"
    key = f"thor_ar_{ident.yf}_{mode}"
    rerun = st.button("↺ Analysera igen", key="thor_ar_rerun")
    r = st.session_state.get(key)
    if r is None or rerun:
        with st.spinner(f"Alpha Regime analyserar {ident.yf} mot {ident.home_index} …"):
            r = run_regime_analysis(ticker=ident.yf, mode=mode, market_ticker=ident.home_index)
            st.session_state[key] = r
    if mode == "quality":
        aui._render_quality_mode(r)
    else:
        aui._render_contrarian_mode(r, None)


def _long_screener(st, ident, strategy, data, ohlcv, cands) -> None:
    from cagr.cagr_cycle import score_cycle_from_int
    from cagr.cagr_fundamentals import score_fundamentals
    from cagr.cagr_scoring import SIGNAL_COLORS, calculate_total_score
    from cagr.cagr_technical import score_technical
    from ui.components import note
    from valhall import cycle as vcy
    from valhall import long_screener as ls
    df = _frame(ohlcv(ident.yf, "2y"))
    if df is None or len(df) < 200:
        note(f"DATA UNAVAILABLE — Long Screener kräver EMA200 (200 dagar) för {ident.yf}.")
        return
    fund_info = {}
    try:
        from cagr.cagr_loader import fetch_fundamentals_batch_fast
        fund_info = (fetch_fundamentals_batch_fast((ident.yf,)) or {}).get(ident.yf) or {}
    except Exception as exc:
        note(f"Fundamenta kunde inte hämtas ({exc}) — poängen räknas utan.")
    segs = ((data.get("board") or {}).get("board") or {}).get("segments") or []
    status = next(((r.get("nordic") or {}).get("status", vcy.NODATA) for r in segs if r.get("key") == ident.segment), vcy.NODATA)
    market = vcy.NODATA

    def _close(sym):
        d = ohlcv(sym, "2y")
        return d["Close"] if d is not None and len(d) else None

    try:
        market = ls.market_regime(ls.index_series(ident.home_index, _close))
    except Exception:
        pass
    cyc = ls.cycle_score(status, market, "gron")
    fund = score_fundamentals(fund_info or {}, None)
    tech = score_technical(pd.DataFrame({"Close": df["Close"].astype(float).tail(ls.TECH_BARS)}))
    total = calculate_total_score(fund, score_cycle_from_int(cyc), tech)
    sig = total["signal"]
    col = SIGNAL_COLORS.get(sig, GREY)
    pct = total["score_pct"] * 100
    st.markdown(cards.verdict_card(sig, f"{pct:.0f} % · {total['total_score']}/{total['max_score']} poäng",
                                   f"fundamenta {total['fund_score']}/{total['fund_max']} · cykel {cyc}/3 (Valhall gron: sektorn "
                                   f"{status}, marknaden {market}) · teknik {total['tech_score']}/7", col,
                                   label="LONG SCREENER"), unsafe_allow_html=True)
    gates = {g.get("rule", "?"): ("PASS" if g.get("pass") else "FAIL", g.get("reason", "")) for g in
             (total.get("gates_result") or {}).get("gates") or []}
    if gates:
        _section(st, "KÖPGRINDARNA")
        st.markdown(cards.check_rows(gates), unsafe_allow_html=True)
    sells = (total.get("sell_triggers") or {}).get("triggers") or []
    if sells:
        _section(st, "SÄLJTRIGGERS")
        st.markdown(cards.check_rows({t.get("rule", "?"): ("FAIL" if t.get("triggered") else "PASS", t.get("reason", ""))
                                      for t in sells}), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(cards.detail_rows("FUNDAMENTA", fund.get("details") or {}, CYAN, f"{total['fund_score']}/{total['fund_max']}"),
                unsafe_allow_html=True)
    c2.markdown(cards.detail_rows("TEKNIK", tech.get("details") or {}, AMBER, f"{total['tech_score']}/7"), unsafe_allow_html=True)
    note("Screenerns egen poäng (cagr: fundamenta via Börsdata, teknik, grindar) med Valhalls cykel gron — samma som "
         "Valhall-kartans Long Screener-rad. Utan cykel (alltid 3) skulle poängen bli "
         f"{calculate_total_score(fund, score_cycle_from_int(3), tech)['score_pct'] * 100:.0f} %.")


def _ember(st, ident, strategy, data, ohlcv, cands) -> None:
    from ember.regime import _render_ticker_analysis, compute_all_complex_regimes
    _render_ticker_analysis(ident.yf, compute_all_complex_regimes())


def _sheet(st, ident, strategy, data, ohlcv, cands) -> None:
    from ui.components import note
    note(f"{reg.label(strategy)} har ingen regimmotor per aktie — regimen är råvarucykeln (fliken Råvara) och arkets "
         f"egen bedömning (fliken Arken). Dagens rad i listan står i verdiktkortet ovan.")


def _royalty(st, ident, strategy, data, ohlcv, cands) -> None:
    """Royalty-motorns rad för aktien ur royalty_scan.json (ingen motor körs i panelen): verdiktet med orsaken,
    composite och pelarna med vikterna, playbookens grindar och SMA200-vakten, värdering och kurs, noterna."""
    from royalty import engine as re_
    from ui.components import note
    blob = (data or {}).get("royalty") or {}
    row = next((r for r in blob.get("rows") or [] if str(r.get("ticker") or "").upper() == ident.yf.upper()), None)
    if row is None:
        note(f"Royalty-motorn har ingen rad för {ident.yf} — bolaget är inte i royaltyuniversumet (listan plus håvens "
             f"royalty/streaming-träffar) eller så har Royalty-steget i scheduled-scan inte körts än.")
        return
    v, why = str(row.get("verdict") or "—"), [str(x) for x in (row.get("why") or [])]
    col = GREEN if v == re_.BUY else AMBER if v == re_.WATCH else RED
    st.markdown(cards.verdict_card(v, why[0] if why else "grindarna, vakten och rangen håller", "; ".join(why[1:4]),
                                   col, label="ROYALTY — AKTIEN"), unsafe_allow_html=True)
    p, w = row.get("pillars") or {}, blob.get("weights") or re_.WEIGHTS
    thr = blob.get("thresholds") or {"buy": re_.BUY_MIN, "watch": re_.WATCH_MIN}

    def num(x, suf="", nd=0):
        return "—" if x is None else f"{float(x):.{nd}f}{suf}"
    st.markdown(cards.kpi_row([
        ("Composite", num(row.get("composite")), CYAN, f"BUY ≥ {thr['buy']:.0f} · WATCH ≥ {thr['watch']:.0f}"),
        ("Hat", num(p.get("hate")), AMBER, f"vikt {w.get('hate', 0):.2f} · SMA200, 52v, cykel, volym"),
        ("Värde", num(p.get("value")), GREEN, f"vikt {w.get('value', 0):.2f} · mot egen historik"),
        ("Kvalitet", num(p.get("quality")), TEXT, f"vikt {w.get('quality', 0):.2f} · styrkepoängen"),
        ("Katalysator", num(p.get("catalyst")), DIM, f"vikt {w.get('catalyst', 0):.2f} · vändning, regim")]),
        unsafe_allow_html=True)
    _section(st, "GRINDARNA (PLAYBOOKEN) OCH VAKTEN")
    k, fails, gd = row.get("kpi") or {}, [str(x).lower() for x in ((row.get("gates") or {}).get("fails") or [])], row.get("guard") or {}

    def gate_col(prefix):
        return RED if any(f.startswith(prefix) for f in fails) else GREEN
    st.markdown(cards.kpi_row([
        ("Bruttomarginal", num(k.get("gross_margin"), " %"), gate_col("bruttomarginal"), "> 70 %"),
        ("EBIT-marginal", num(k.get("ebit_margin"), " %"), gate_col("ebit-marginal"), "> 40 %"),
        ("Skuld/EBITDA", num(k.get("nd_ebitda"), "", 1), gate_col("skuld/ebitda"), "< 1,5"),
        ("FCF", num(k.get("fcf")), TEXT, "info · väger in i kvalitet (M i bolagets valuta)"),
        ("Mot SMA200", num(gd.get("above_sma200_pct"), " %"), GREEN if gd.get("ok") else RED,
         f"vakten: högst +{float(blob.get('max_above_sma200_pct') or re_.MAX_ABOVE_SMA200_PCT):.0f} %")]),
        unsafe_allow_html=True)
    _section(st, "VÄRDERING OCH KURS")
    pr = row.get("price") or {}
    st.markdown(cards.kpi_row([
        ("EV/EBITDA", num(k.get("ev_ebitda"), "", 1), TEXT, "12–25 är normalt för royalty"),
        ("P/FCF", num(k.get("p_fcf"), "", 1), TEXT, "mot egen historik"),
        ("Från 52v-lägsta", num(pr.get("from_low_52w_pct"), " %"), AMBER, f"kurs {num(pr.get('close'), '', 2)}"),
        ("Regim", str(row.get("regime") or "—"), DIM, "OVTLYR-färg"),
        ("Börsvärde", num(k.get("mcap_musd")), DIM, "MUSD")]), unsafe_allow_html=True)
    for n in row.get("notes") or []:
        note(n)
    note("Samma pelare som Deep Contrarian (contrarian_alpha hate/value/strength/catalyst) på royaltyuniversumet; "
         "grindarna är Royalty-playbookens. Raden räknas i Royalty-steget i scheduled-scan — kör det i GitHub Actions "
         "för en färsk rad.")


LAYOUTS = {"viking_nine": _viking_nine, "viking": _viking_nine, "wolf": _wolf, "momentum": _momentum, "berserk": _berserk,
           "ember": _ember, "alpha_quality": _alpha, "alpha_contrarian": _alpha, "long_screener": _long_screener,
           "sheet": _sheet, "insider": _sheet, "royalty": _royalty}


# ── Teknik ──────────────────────────────────────────────────────────────────
def render_technical(st, ident: Ticker, loader: Optional[Callable] = None) -> None:
    """Teknisk analys för aktien — samma vy som 🕯️ Teknisk analys (tech_analysis._render)."""
    import tech_analysis as ta
    from ui.components import note
    period = st.pills("Period", list(TA_PERIODS), default="1 år", selection_mode="single", key="thor_ta_period",
                      label_visibility="collapsed") if hasattr(st, "pills") else "1 år"
    df, src = (loader or ta.load_ohlcv)(ident.yf, TA_PERIODS.get(period or "1 år", "1y"))
    if df is None:
        note(f"DATA UNAVAILABLE — ingen kurshistorik för {ident.yf} (Yahoo och Börsdata).")
        return
    ta._render(ident.yf, df, src)


# ── Regler ──────────────────────────────────────────────────────────────────
def render_rules(st, strategy: str) -> None:
    """Hela playbooken: riskmodellen, entry/exit/mindset-reglerna, steg för steg, fusklappen och fallgroparna."""
    from ovtlyr.ui.rules_page import _level_badge, _risk_box, _rule_row, _support_badge
    from strategy_rules import PLAYBOOKS
    from ui.components import note
    s = reg.get(strategy)
    pb = PLAYBOOKS.get((s.playbook if s else "") or "")
    if pb is None:
        note(f"{reg.label(strategy)} har ingen playbook i strategy_rules än — den skrivs i fas 7.")
        return
    st.markdown(f"<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:4px 0;'>"
                f"<b style='color:{pb.color};font-size:1.05rem;'>{pb.name}</b>{_level_badge(pb.level)}{_support_badge(pb)}"
                f"<span style='color:{DIM};font-size:0.74rem;'>{pb.horizon} · {pb.universe}</span></div>"
                f"<div style='color:{TEXT};font-size:0.82rem;line-height:1.5;margin:4px 0 8px;'>{pb.idea}</div>",
                unsafe_allow_html=True)
    _risk_box(pb)
    if pb.workflow:
        _section(st, "SÅ GÖR DU — STEG FÖR STEG")
        st.markdown("".join(f"<div style='display:flex;gap:10px;padding:3px 0;font-size:0.82rem;color:{TEXT};'>"
                            f"<span style='flex:none;width:22px;height:22px;border-radius:50%;border:1px solid {pb.color};"
                            f"color:{pb.color};text-align:center;line-height:22px;font-size:0.72rem;'>{i}</span><span>{step}</span></div>"
                            for i, step in enumerate(pb.workflow, 1)), unsafe_allow_html=True)
    for title, rules, col in (("ENTRY", pb.entry, pb.color), ("EXIT", pb.exit, RED), ("MINDSET", pb.mindset, AMBER)):
        if rules:
            _section(st, title)
            st.markdown("".join(_rule_row(r, col) for r in rules), unsafe_allow_html=True)
    if pb.cheatsheet:
        _section(st, "FUSKLAPP")
        st.markdown("".join(f"<div style='display:flex;gap:12px;font-size:0.8rem;padding:2px 0;'><span style='color:{DIM};"
                            f"min-width:140px;'>{k}</span><span style='color:{TEXT};'>{v}</span></div>" for k, v in pb.cheatsheet),
                    unsafe_allow_html=True)
    if pb.pitfalls:
        _section(st, "FALLGROPAR")
        st.markdown("".join(f"<div style='font-size:0.8rem;color:{TEXT};padding:2px 0;'><span style='color:{RED};'>✗</span> {p}</div>"
                            for p in pb.pitfalls), unsafe_allow_html=True)
    if pb.support_note or pb.note:
        note(" ".join(x for x in (pb.support_note, pb.note) if x))
    note(f"Källa: {pb.source}. Reglerna ändras inte här — de bor i strategy_rules.py och visas i RULES → Regler & Guider.")
