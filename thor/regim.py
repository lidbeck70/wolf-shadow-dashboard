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
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, GREY, RED, TEXT

TA_PERIODS = {"6 mån": "6mo", "1 år": "1y", "2 år": "2y"}


def _capital_default() -> float:
    """Kapitalfältets standardvärde: ditt kapital ur ⚙️ Inställningar (beslut 5), annars 100 000 som förut."""
    from ui import settings as us
    return us.capital_or(100000.0)


def _section(st, title: str) -> None:
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;"
                f"margin:12px 0 6px;'>{title}</div>", unsafe_allow_html=True)


def _closed(df, sym: str):
    """Bara stängda staplar, som skannrarna räknar (ovtlyr_nine.closed_only) — Yahoos halvdagsstapel under
    börsens öppettid ger annars ett annat svar i Thor än i listan (granskning 2)."""
    df = _frame(df)
    if df is None:
        return None
    try:
        import ovtlyr_nine as on
        return on.closed_only(df, sym)
    except Exception:
        return df


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


_LAYER = {"market": "Market", "sector": "Sector", "stock": "Stock"}


def first_block(nine, decision) -> tuple:
    """(text, färg) för den första grinden som stoppar en entry, i spelordningen: OVTLYR Nine (marknad → sektor →
    aktie), sedan Viking Executions krav, sist de hårda spärrarna (förlustgränsen). Allt grönt → GOLDEN TICKET."""
    import ovtlyr_nine as on
    import viking_execution as vx
    for f in nine.factors:
        if not f.passed:
            fail = f.status == on.FAIL
            return (f"Nine: {_LAYER[f.layer]} {f.label} ({'FAIL' if fail else f.status}) — {f.detail}"[:180],
                    RED if fail else DIM)
    for c in decision.missing:
        return (f"Execution: {c.label}" + (f" ({c.flag})" if c.flag else "") + f" — {c.detail}")[:180], AMBER
    if decision.status != vx.GO and decision.reasons:
        return str(decision.reasons[0])[:180], RED
    return "inget — alla nio faktorer och alla Execution-filter är gröna", GOLD


def _levels(decision) -> str:
    p = decision.position
    parts = [f"entry {decision.entry:,.2f}" if decision.entry is not None else "",
             f"stopp {p.stop:,.2f}" if p is not None else "", f"{p.shares} aktier" if p is not None else "",
             "R fri väg" if decision.rr is None and decision.resistance is None else
             f"R {decision.rr:.1f}" if decision.rr is not None else ""]
    return " · ".join(x for x in parts if x)


def _switch_strategy(strategy: str) -> None:
    """Knappen i Viking-layouten: byt 'Bedöm som' till Viking Nine för samma aktie (callback, före omritningen)."""
    import streamlit as st
    st.session_state["thor_strategy"] = reg.label(strategy)
    st.session_state["_thor_strategy_set"] = True


def _viking(st, ident, strategy, data, ohlcv, cands) -> None:
    """Viking är screenern — ingen motor per aktie. Dagens screenerrad och en knapp till Viking Nine, i stället för
    att rita Viking Nines motor under Vikings verdikt (det såg ut som en regelkonflikt)."""
    from ui.components import note
    c = next((x for x in cands if x.strategy == "viking"), None)
    if c is not None:
        st.markdown(cards.candidate_card(c), unsafe_allow_html=True)
    else:
        note(f"Viking har inte {ident.yf} i dagens screenerlista — raden kommer ur schemats Arc-körning (Norden och USA).")
    note("Viking är screenern: BUY när kursen är över EMA200, ADX ≥ 20 och composite-poängen ≥ 60 mot resten av "
         "universumet; 'screener n/9' är dess egna nio faktorer, inte OVTLYR Nine. Entryn för en swing tas med "
         "Viking Nine (OVTLYR Nine + Viking Execution) — samma aktie, en knapp bort.")
    st.button("⚔️ Viking Nine för samma aktie →", key="thor_viking_to_nine", on_click=_switch_strategy,
              args=("viking_nine",))


def _viking_nine(st, ident, strategy, data, ohlcv, cands) -> None:
    """En beslutsrad, en kompakt 9-ruta och en Execution-rad — orsakerna, riskmotorn och de fullständiga korten
    i en expander. Samma motorer och regler som förut; bara ordningen och upprepningen är ändrad."""
    import ovtlyr_nine as on
    import viking_execution as vx
    import viking_screen as vs
    from ovtlyr.ui.execution_card import render_execution
    from ovtlyr.ui.exit_card import render_exit_section
    from ovtlyr.ui.nine_card import render_nine_card
    from ovtlyr.ui.viking_nine_regime import price_chart
    from ovtlyr.ui.viking_screens import _earnings, _risk_banner, _risk_for, _sector
    from ui.components import note
    df = _frame(ohlcv(ident.yf, on.PERIOD))
    if df is None or len(df) < on.MIN_BARS:
        note(f"DATA UNAVAILABLE — ingen kurshistorik för {ident.yf} (kräver {on.MIN_BARS} dagar).")
        return
    head = st.container()                  # beslutsraden, 9-rutan och Execution-raden fylls i efter uträkningen
    with st.expander("Kapital och positionstak", expanded=False):
        k1, k2 = st.columns(2)
        capital = k1.number_input("Kapital (SEK)", min_value=0.0, value=_capital_default(), step=10000.0, key="thor_capital")
        max_pos = k2.number_input("Max position (% av kapitalet)", min_value=5.0, max_value=100.0,
                                  value=float(vx.MAX_POSITION_PCT), step=5.0, key="thor_max_pos")
    dfc = on.closed_only(df, ident.yf)                 # Nine och order blocks på stängda staplar (playbooken)
    ob = vs._ob_analysis(dfc)
    nine = on.evaluate(ident.yf, stock_df=dfc, ob_analysis=ob, sector_getter=_sector)
    try:
        from trade_journal import load_journal
        trades = load_journal()
    except Exception:
        trades = []
    earnings = _earnings(ident.yf)
    decision = vx.evaluate_entry(ident.yf, df, nine=nine, capital=capital, ob_analysis=ob, earnings_date=earnings,
                                 earnings_known=earnings is not None, trades=trades, max_position_pct=max_pos,
                                 market_risk=_risk_for(ident.yf))
    block, bcol = first_block(nine, decision)
    with head:
        st.markdown(cards.vn_decision_row(decision.status, f"{nine.passed}/{on.NINE_TOTAL}",
                                          f"{decision.execution_passed}/{decision.execution_total}", block, bcol,
                                          _levels(decision), bars_caption(nine)), unsafe_allow_html=True)
        st.markdown(cards.vn_nine_compact(nine), unsafe_allow_html=True)
        st.markdown(cards.vn_execution_row(decision), unsafe_allow_html=True)
    _risk_banner((ident.market,))
    with st.expander("Varför — varje faktor, varje filter och riskmotorn", expanded=False):
        _section(st, "OVTLYR NINE — SETUP")
        render_nine_card(nine)
        _section(st, "VIKING EXECUTION · RISK ENGINE · FINAL DECISION")
        render_execution(decision)
    try:
        st.plotly_chart(price_chart(ident.yf, df, decision), use_container_width=True, config={"displayModeBar": False},
                        key="thor_vn_chart")
    except Exception as exc:                                     # pragma: no cover - grafen är inte regeln
        note(f"Grafen kunde inte ritas: {exc}")
    render_exit_section(ident.yf, df, nine, ob, earnings)


def _wolf(st, ident, strategy, data, ohlcv, cands) -> None:
    import wolf_regime_score as wrs
    from ui.charts import build_gauge
    from ui.components import note
    sector = ident.sector_etf or "SPY"
    spy, sec, stk = _closed(ohlcv("SPY", "1y"), "SPY"), _closed(ohlcv(sector, "1y"), sector), \
        _closed(ohlcv(ident.yf, "1y"), ident.yf)
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
    capital = k1.number_input("Kapital (SEK)", min_value=0.0, value=_capital_default(), step=10000.0, key="thor_capital")
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
    note("Poängen, pillen och kalkylatorn ovan är Wolf Regime-sidans formler (wolf_regime_score.py) — inte listans "
         "125-poängsskala (wolf_shadow_screener) och inte strategins köpregel.")
    _wolf_rule_row(st, ident, ohlcv, cands)


def wolf_rule(df) -> Optional[dict]:
    """Strategins egen köpregel (strategies/wolf.py entry_fn) — den som listan och papperskontot använder:
    EMA10 > 21 > 50 > 200, kurs > EMA50, RSI 45–70, ADX ≥ 19; stopp = entry − 2,5 × ATR, TP1 2,6R."""
    from strategies.wolf import entry_fn
    if df is None or len(df) < 210:
        return None
    e = entry_fn(df)
    gates = e.get("gates") or []
    return {"signal": e.get("signal"), "entry": e.get("entry_price"), "stop": e.get("stop_loss"),
            "tp1": e.get("tp1_price"), "passed": sum(1 for g in gates if g.get("passed")), "total": len(gates)}


def _wolf_rule_row(st, ident, ohlcv, cands) -> None:
    """Listans siffror bredvid sidans (granskning 2, W1/W2): strategins köpregel på 2 års data och listans poäng."""
    from ui.components import note
    _section(st, "STRATEGINS KÖPREGEL — SAMMA SOM LISTAN OCH PAPPERET")
    try:
        r = wolf_rule(_closed(ohlcv(ident.yf, "2y"), ident.yf))
    except Exception as exc:
        r, err = None, str(exc)
    else:
        err = "kräver 210 dagars historik"
    c = next((x for x in cands or [] if x.strategy == "wolf"), None)
    listed = f" · listan: {c.score_text}" if c is not None and c.score_text else ""
    if not r:
        note(f"Köpregeln gick inte att räkna ({err}){listed}.")
        return
    buy = r["signal"] == "BUY"
    items = [("Köpregel", "ENTRY" if buy else "INGEN ENTRY", GREEN if buy else RED, f"{r['passed']}/{r['total']} grindar")]
    if buy:
        items += [("Stopp", f"{r['stop']:.2f}", RED, "entry − 2,5 × ATR"), ("TP1", f"{r['tp1']:.2f}", GREEN, "2,6R")]
    st.markdown(cards.kpi_row(items), unsafe_allow_html=True)
    if listed:
        note(f"Listans regimpoäng{listed} räknas med en annan motor (4 lager, 125 poäng) än REGIME SCORE ovan.")


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
    if not screener:
        note("wolf_screener.json kunde inte läsas (Gisten/nätet) — momentumlistan saknas i den här bedömningen; 📡 Uppdatera försöker igen.")
    if not regime:
        note("wolf_regime.json kunde inte läsas — regimljuset saknas i den här bedömningen.")
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
    theme = uv.theme_of(ident.yf) or ident.theme or None       # skannerns tema först — samma drivare som listan (B1)
    if not theme:
        note(f"{ident.yf} har inget råvarutema — BERSERK kräver en drivare (råvaran). Inte i BERSERK-universumet.")
        return
    drivers = live.load_drivers([theme], getter=ohlcv)
    rows = [r for r in live.theme_states(drivers) if r["theme"] == theme]
    if rows:
        st.markdown(theme_card(rows[0]), unsafe_allow_html=True)
    stock = _closed(ohlcv(ident.yf, live.STOCK_PERIOD), ident.yf)
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
    mkt = markets.get(uv.REGION_INDEX.get(region, "SPY"))      # aldrig `or` på en Series: sanningsvärdet är tvetydigt
    if mkt is None:
        mkt = markets.get("SPY")
    row = live.evaluate(ident.yf, stock, drv, mkt, capital=_capital_default(), cfg=cfg, driver_symbol=sym,
                        commodity=commodity)                       # ditt kapital, inte alltid 100 000
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


def _no_fundamentals(fund: dict) -> bool:
    """Sant när hämtningen gav tomt: inga nyckeltal alls, bara N/A/None, eller källan 'none'."""
    if not fund or fund.get("_data_source") == "none":
        return True
    vals = [v for k, v in fund.items() if not str(k).startswith("_")]
    return not vals or all(v in (None, "", "N/A") for v in vals)


def kap_badge_for(data: dict, yf: str) -> bool:
    """Om Quality-listan märkte bolaget ★ KAP SCREENED — samma flagga som skannern gav Alpha-motorn
    (alpha_regime.quality_scan), så att Thors verdikt inte skiljer sig från listans (granskning 2, A1)."""
    try:
        from alpha_regime.screener_link import to_yf_ticker
    except Exception:
        return False
    def same(raw) -> bool:
        t = to_yf_ticker(str(raw or "")).upper()             # 'INVE B' → 'INVE-B' (Börsdata-form utan suffix)
        return t == yf.upper() or ("." not in t and t == yf.upper().split(".")[0])

    ca = (data or {}).get("ca_quality") or {}
    for r in ca.get("results") or []:
        if isinstance(r, dict) and r.get("kap_badge") and same(r.get("ticker")):
            return True
    for raw, sig in (ca.get("quality_signals") or {}).items():      # skanningens grindar bär märkningen också
        gate = (((sig or {}).get("gates") or {}).get("Kvalitet") or {}) if isinstance(sig, dict) else {}
        if gate.get("label") == "KAP SCREENED" and same(raw):
            return True
    return False


def _alpha(st, ident, strategy, data, ohlcv, cands) -> None:
    from alpha_regime import ui as aui
    from alpha_regime.engine import run_regime_analysis
    mode = "contrarian" if strategy == "deep" else "quality"
    key = f"thor_ar_{ident.yf}_{mode}"
    rerun = st.button("↺ Analysera igen", key="thor_ar_rerun")
    r = st.session_state.get(key)
    if r is None or rerun or getattr(r, "error", None):         # ett fel körs om automatiskt, cachas inte
        with st.spinner(f"Alpha Regime analyserar {ident.yf} mot {ident.home_index} …"):
            r = run_regime_analysis(ticker=ident.yf, mode=mode, market_ticker=ident.home_index,
                                    kap_badge=kap_badge_for(data, ident.yf) if mode == "quality" else False)
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
    if _no_fundamentals(fund_info):
        note("Fundamenta saknas (Börsdata/Yahoo svarade inte) — kriterierna räknas som 0/6 tills nästa försök, "
             "inte för att bolaget saknar nyckeltal.")
        try:
            fetch_fundamentals_batch_fast.clear()                 # cacha inte ett tomt svar i en timme
        except Exception:
            pass
    from thor.dossier import valhall_tag
    blob = data.get("board") or {}
    inner = blob.get("board") if isinstance(blob.get("board"), dict) else blob
    tag = valhall_tag(inner, ident) if inner else {}                # temat ur sin egen segmentrad (T8)
    # Nordiska bolag bedöms som i Valhalls Long Screener-rad (sektorns nordiska status); utländska på temat om det
    # finns på kartan, annars globalt (SPDR) — inte på Nordens sektorstatus som förut (NEM ≠ nordiskt Material).
    g = "n" if ident.nordic else ("t" if ident.theme and tag.get("t_status") not in (None, vcy.NODATA) else "g")
    status = tag.get(f"{g}_status") or vcy.NODATA
    view = {"n": "sektorn", "t": "temat", "g": "globalt"}[g]
    market = vcy.NODATA

    def _close(sym):
        d = ohlcv(sym, "2y")
        return d["Close"] if d is not None and len(d) else None

    # Hemmamarknadens läge: skannerns sparade värde först (samma som listraden), annars räknat här — med
    # Börsdata, så att Oslos OBX inte blir DATA SAKNAS när Yahoo saknar historik (granskning 2, L1).
    saved = (((blob.get("long") or {}) if isinstance(blob, dict) else {}).get("markets") or {}).get(ident.home_index)
    if saved and saved != vcy.NODATA:
        market = saved
    else:
        try:
            from thor.ui import _api
            market = ls.market_regime(ls.index_series(ident.home_index, _close, _api()))
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
                                   f"fundamenta {total['fund_score']}/{total['fund_max']} · cykel {cyc}/3 (Valhall gron: {view} "
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


LAYOUTS = {"viking_nine": _viking_nine, "viking": _viking, "wolf": _wolf, "momentum": _momentum, "berserk": _berserk,
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
    note(f"Källa: {pb.source}. Reglerna ändras inte här — de bor i strategy_rules.py och visas i MER → Regler & Guider.")
