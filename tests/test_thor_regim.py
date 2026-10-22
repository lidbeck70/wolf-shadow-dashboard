"""
Thor fas 4 — thor/regim: Regim-fliken i varje strategilayout (Viking Nine, Wolf, Momentum, BERSERK, Alpha,
Long Screener, EMBER, arken), felvägen, Teknik-fliken och Regler-fliken under AppTest. Motorerna körs på
syntetiska kurser; nätverkskällorna (Börsdata, FRED, screenerblobbarna, Alpha Regime) är fejkade.
"""
import os
import re
import sys
import zlib

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_candidates as fx  # noqa: E402
import wolf_regime_score as wrs  # noqa: E402
from test_wolf_regime_score import _walk  # noqa: E402
from thor import dossier as td  # noqa: E402
from thor import regim  # noqa: E402
from thor import registry as reg  # noqa: E402
from thor import ui as tui  # noqa: E402

TODAY = pd.Timestamp.today().normalize()


def _prices(ticker, period="1y"):
    """Deterministisk kursram per ticker som slutar i dag (Viking Nine räknar annars STALE DATA)."""
    return _walk(zlib.crc32(ticker.encode()) % 1000, 0.0006, end=TODAY)


@pytest.fixture
def engines(monkeypatch):
    import alpha_regime.engine as ae
    import alpha_regime.ui as aui
    import cagr.cagr_loader as cl
    import ember.regime as er
    import market_prices
    import market_risk_gate
    import ovtlyr_nine as on
    import storage
    import streamlit as st
    import swing
    import wolf_regime_ui
    import wolf_screener_ui
    from berserk import live
    from ovtlyr.ui import viking_screens
    monkeypatch.setattr(market_prices, "ohlcv", _prices)
    monkeypatch.setattr(tui, "_prices", _prices)
    monkeypatch.setattr(tui, "_ohlcv", lambda t: _prices(t))
    monkeypatch.setattr(market_risk_gate, "for_ticker", lambda t, evaluator=None: None)
    for name in ("_sector", "_earnings", "_risk_for"):
        monkeypatch.setattr(viking_screens, name, lambda t: None)
    monkeypatch.setattr(on, "nordic_market", lambda *a, **k: {"close": _prices("OMXS30")["Close"], "source": "test",
                                                              "breadth": None})
    monkeypatch.setattr(wolf_screener_ui, "_get_data", lambda force=False: fx.MOMENTUM)
    monkeypatch.setattr(wolf_regime_ui, "_get_data", lambda force=False: {"regime": "GRÖN", "generated": "2026-10-14T04:17"})
    monkeypatch.setattr(swing, "rules_data", lambda: {"positions": []})
    monkeypatch.setattr(live, "load_markets", lambda regions, getter=None, nordic_provider=None: {"SPY": _prices("SPY")["Close"]})
    monkeypatch.setattr(ae, "run_regime_analysis", lambda ticker, mode="quality", market_ticker="SPY", **k:
                        {"ticker": ticker, "mode": mode, "market": market_ticker})
    monkeypatch.setattr(aui, "_render_quality_mode", lambda r: st.markdown(f"QUALITY RENDERED {r['ticker']} mot {r['market']}"))
    monkeypatch.setattr(aui, "_render_contrarian_mode", lambda r, exposure_override=None: st.markdown(f"CONTRARIAN RENDERED {r['ticker']}"))
    monkeypatch.setattr(er, "compute_all_complex_regimes", lambda: {})
    monkeypatch.setattr(er, "_render_ticker_analysis", lambda t, regs: st.markdown(f"EMBER RENDERED {t}"))
    monkeypatch.setattr(cl, "fetch_fundamentals_batch_fast", lambda tickers, _version="": {t: {} for t in tickers})
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None: st.session_state.setdefault(name, default))


def _script():
    import streamlit as st
    from thor import dossier as d
    from thor import regim as r
    from thor import ui as t
    strategy, ticker, what = (st.session_state[k] for k in ("_strategy", "_ticker", "_what"))
    ident = d.identify(ticker)
    if what == "regim":
        r.render(st, ident, strategy, {"board": st.session_state.get("_board") or {},
                                       "royalty": st.session_state.get("_royalty") or {}}, t._prices, [])
    elif what == "teknik":
        r.render_technical(st, ident)
    elif what == "teknik_nodata":
        r.render_technical(st, ident, loader=lambda tk, p: (None, ""))
    else:
        r.render_rules(st, strategy)


def _run(strategy, ticker, what="regim", royalty=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_script, default_timeout=120)
    at.session_state["_strategy"], at.session_state["_ticker"], at.session_state["_what"] = strategy, ticker, what
    at.session_state["_board"] = fx.BOARD
    at.session_state["_royalty"] = royalty or {}
    at.run()
    assert not at.exception, at.exception
    return at, " ".join(m.value for m in at.markdown)


# ── Registret och layouterna hänger ihop ─────────────────────────────────────
def test_every_regime_layout_in_the_registry_has_a_renderer():
    assert {s.regime for s in reg.STRATEGIES.values() if s.regime} <= set(regim.LAYOUTS)
    assert regim.LAYOUTS["viking"] is regim._viking and regim.LAYOUTS["insider"] is regim.LAYOUTS["sheet"]
    assert list(regim.TA_PERIODS) == ["6 mån", "1 år", "2 år"]


# ── Regim per layout ─────────────────────────────────────────────────────────
def test_wolf_layout_shows_the_module_numbers(engines):
    at, text = _run("wolf", "BOL.ST")
    ident = td.identify("BOL.ST")
    want = wrs.regime(_prices("SPY"), _prices(ident.sector_etf or "SPY"), _prices("BOL.ST"))
    lbl, _c = wrs.label(want["total"])
    assert "REGIME SCORE" in text and f">{want['total']}</div>" in text and lbl in text and "/ 125 MAX" in text
    assert len(at.get("plotly_chart")) == 4                                               # fyra mätare
    assert ("GATE OPEN" if want["total"] >= wrs.GATE_MIN else "GATE CLOSED") in text
    assert "WOLF ENTRY CHECKLIST" in text and "SWING TRADING GATES" in text and "MÄTBARA GRINDAR" in text
    chk = wrs.checklist(want, _prices("BOL.ST"))
    gs = wrs.gate_summary(wrs.gates(want, chk))
    assert f"{gs['passed']}/{gs['measurable']} MÄTBARA GRINDAR" in text and "4 manuella" in text
    assert f"ATR ratio {chk['atr_ratio']:.2f}x" in text and f"RSI {want['rsi']:.1f}" in text
    p = wrs.sltp(_prices("BOL.ST"), 100000, wrs.RISK_PCT)
    assert "SL / TP KALKYLATOR — WOLF" in text and f"{p['sl']:.2f}" in text and f"{p['shares']} aktier" in text
    assert at.number_input(key="thor_capital").value == 100000.0 and at.number_input(key="thor_wolf_risk").value == wrs.RISK_PCT
    assert "wolf_regime_score.py" in text


def test_wolf_layout_needs_sixty_bars(engines, monkeypatch):
    monkeypatch.setattr(tui, "_prices", lambda t, p="1y": _walk(1, n=30, end=TODAY))
    _at, text = _run("wolf", "BOL.ST")
    assert "DATA UNAVAILABLE" in text and "60 dagars" in text and "REGIME SCORE" not in text


def test_momentum_layout_is_the_swing_verdict(engines):
    at, text = _run("momentum", "ERIC-B.ST")
    assert "SWING-DOMEN" in text and "KÖP-KANDIDAT" in text and "Rank 1 av 20" in text
    assert ">1/40<" in text and "regim GRÖN" in text and "A (pullback)" in text and "= 72.00" in text   # stopp −10 %
    assert "MOMENTUM — ENTRY-REGLERNA" in text and "marknadsfilter" in text and "swing_verdict" in text


def test_berserk_layout_runs_the_live_engine_on_the_theme(engines):
    at, text = _run("berserk", "BOL.ST")
    assert "Koppar" in text and "BERSERK — AKTIEN" in text and "RSI(2)" in text and "drivare" in text
    assert "berserk.live.evaluate" in text
    _at, text = _run("berserk", "ERIC-B.ST")                                             # inget råvarutema
    assert "har inget råvarutema" in text and "BERSERK — AKTIEN" not in text


def test_alpha_layouts_run_quality_and_contrarian_against_the_home_index(engines):
    at, text = _run("quality", "BOL.ST")
    assert "QUALITY RENDERED BOL.ST mot ^OMX" in text and at.button(key="thor_ar_rerun")
    _at, text = _run("deep", "NEM")
    assert "CONTRARIAN RENDERED NEM" in text


def test_long_screener_layout_scores_with_the_valhall_cycle(engines):
    _at, text = _run("long", "BOL.ST")
    assert "LONG SCREENER" in text and "poäng" in text and "cykel" in text and "Valhall gron: sektorn RÖD" in text
    assert "KÖPGRINDARNA" in text and "FUNDAMENTA" in text and "TEKNIK" in text and "Utan cykel" in text


def test_ember_and_sheet_layouts(engines):
    _at, text = _run("ember", "NEM")
    assert "EMBER RENDERED NEM" in text
    _at, text = _run("rule", "X")
    assert "Rick Rule har ingen regimmotor per aktie" in text


def test_royalty_layout_reads_the_engines_row(engines):
    _at, text = _run("royalty", "FNV.TO", royalty=fx.ROYALTY)
    assert "ROYALTY — AKTIEN" in text and "BUY CANDIDATE" in text and "71" in text and "GRINDARNA" in text
    assert "Bruttomarginal" in text and "93 %" in text and "-8 %" in text and "12–25 är normalt" in text
    assert "Royalty C-arket" in text and "Deep Contrarian" in text
    _at, text = _run("royalty", "LEV.TO", royalty=fx.ROYALTY)
    assert "PASS" in text and "skuld/EBITDA 3.1" in text
    _at, text = _run("royalty", "ERIC-B.ST", royalty=fx.ROYALTY)                      # inte i universumet
    assert "har ingen rad för ERIC-B.ST" in text and "ROYALTY — AKTIEN" not in text
    _at, text = _run("royalty", "FNV.TO")                                               # bloben saknas
    assert "har ingen rad" in text


def test_viking_nine_layout_shows_the_whole_decision_chain(engines):
    at, text = _run("viking_nine", "NEM")
    for part in ("MARKET SPY", "OVTLYR NINE", "VIKING EXECUTION", "RISK ENGINE", "VIKING EXIT ENGINE",
                 "Stoppar:", "Stängda candles: aktien t.o.m.", "TOTAL"):
        assert part in text, part
    assert len(at.get("plotly_chart")) == 1 and at.number_input(key="thor_capital").value == 100000.0
    labels = [e.label for e in at.expander]
    assert "Kapital och positionstak" in labels and any(lbl.startswith("Varför") for lbl in labels)
    at2, text = _run("viking", "NEM")
    assert "Viking är screenern" in text and "OVTLYR NINE" not in text               # ingen motor under Viking
    assert "⚔️ Viking Nine för samma aktie →" in [b.label for b in at2.button]


def test_a_failing_layout_becomes_a_note(engines, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("motorn sprack")
    monkeypatch.setitem(regim.LAYOUTS, "wolf", boom)
    _at, text = _run("wolf", "BOL.ST")
    assert "Regimen för Wolf kunde inte ritas för BOL.ST: motorn sprack" in text


# ── Teknik ──────────────────────────────────────────────────────────────────
def test_technical_tab_is_the_tech_analysis_view(engines):
    at, text = _run("quality", "BOL.ST", "teknik")
    for part in ("<b>BOL.ST</b>", "ORDER BLOCKS", "ENTRY PATTERNS", "RISK · MOMENTUM", "RSI 14", "AVANCERAT"):
        assert part in text, part
    assert len(at.get("plotly_chart")) >= 5
    assert "nanx" not in text                                                            # volume_ratio-nyckeln
    atr = re.search(r"ATR 14 <b style='float:right;'>([\d.]+)</b>", text)
    assert atr and float(atr.group(1)) > 0                                               # atr14-nyckeln
    _at, text = _run("quality", "BOL.ST", "teknik_nodata")
    assert "DATA UNAVAILABLE" in text


# ── Regler ──────────────────────────────────────────────────────────────────
def test_rules_tab_is_the_whole_playbook(engines):
    from strategy_rules import PLAYBOOKS
    _at, text = _run("viking_nine", "BOL.ST", "regler")
    for part in (PLAYBOOKS[reg.get("viking_nine").playbook].name, "RISKMODELL", "SÅ GÖR DU — STEG FÖR STEG", "ENTRY", "EXIT", "MINDSET",
                 "FUSKLAPP", "FALLGROPAR", "strategy_rules.py"):
        assert part in text, part
    _at, text = _run("wolf", "BOL.ST", "regler")
    assert PLAYBOOKS["wolf"].name in text
    _at, text = _run("hav", "BOL.ST", "regler")
    assert "har ingen playbook" in text
