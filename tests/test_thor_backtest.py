"""
Thor fas 6 — thor/backtest: de fyra per-aktie-motorerna (Viking Nine, BERSERK, Momentum, Wolf) på syntetiska
kurser genom samma omslag som fliken använder, det normaliserade resultatet, grafen med markörer, och fliken
under AppTest (Kör-knappen, nyckeltal, affärstabell, förbehåll; strategi utan motor; fel). Inget nätverk.
"""
import math
import os
import sys
import zlib

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_wolf_regime_score import _walk  # noqa: E402
from thor import backtest as tb  # noqa: E402
from thor import dossier as td  # noqa: E402
from thor import ui as tui  # noqa: E402

TODAY = pd.Timestamp("2026-10-15")
BARS = {"1y": 260, "2y": 520, "3y": 780, "4y": 1040, "5y": 1300, "6y": 1560, "7y": 1820, "8y": 2080, "9y": 2340,
        "10y": 2600, "11y": 2860, "12y": 3120, "13y": 3380, "max": 3380}


def _prices(ticker, period="1y"):
    """Deterministisk kursram per ticker och period, slutar på TODAY."""
    return _walk(zlib.crc32(ticker.encode()) % 1000, 0.0006, n=BARS.get(period, 1300), end=TODAY)


@pytest.fixture
def engines(monkeypatch):
    import market_prices
    import ovtlyr_nine as on
    monkeypatch.setattr(market_prices, "ohlcv", _prices)
    monkeypatch.setattr(tui, "_prices", _prices)
    monkeypatch.setattr(on, "nordic_market", lambda *a, **k: None)            # ingen Börsdata i testet


# ── Motorerna ────────────────────────────────────────────────────────────────
def _check(r: tb.Result, strategy: str, fill: str):
    assert r.error == "" and r.strategy == strategy and r.fill == fill and r.years == 3
    assert r.period[0] < r.period[1] and r.close is not None and len(r.close) > 200
    assert r.metrics and r.metrics[0][0] == "Affärer"
    for t in r.trades:
        assert set(t) >= {"signal", "entry_date", "entry", "exit_date", "exit", "result", "reason", "days", "open"}
        assert t["open"] or (t["exit_date"] and t["result"] != "öppen")
    assert tb.chart(r, "X") is not None and tb.trade_table(r.trades) is not None


def test_viking_nine_runs_in_fast_mode_against_spy(engines):
    r = tb.run("viking_nine", td.identify("NEM"), 3, _prices, today=TODAY)
    _check(r, "viking_nine", tb.FILL_NEXT_OPEN)
    assert any(m[0] == "Signaler" and "marknad SPY" in m[3] for m in r.metrics)
    assert any("Snabbläge" in w for w in r.warnings) and r.notes
    rn = tb.run("viking_nine", td.identify("BOL.ST"), 3, _prices, today=TODAY)      # nordisk utan Börsdata
    assert rn.error == "" and any("OMXS30" in w and "SPY" in w for w in rn.warnings)
    assert tb.run("viking", td.identify("BOL.ST"), 3, _prices, today=TODAY).error   # Viking ≠ Viking Nine (granskning 2)


def test_berserk_runs_the_scanner_rules_on_the_theme(engines):
    r = tb.run("berserk", td.identify("BOL.ST"), 3, _prices, today=TODAY)
    _check(r, "berserk", tb.FILL_NEXT_OPEN)
    assert any(m[0] == "Signaler" and "S1" in m[3] and "marknadsspärr" in m[3] for m in r.metrics)
    assert any("HG=F" in w and "koppar" in w for w in r.warnings)
    assert all(t["result"].endswith(" R") or t["result"] == "öppen" for t in r.trades)
    bad = tb.run("berserk", td.identify("ERIC-B.ST"), 3, _prices, today=TODAY)       # inte i BERSERK-universumet
    assert "BERSERK" in bad.error or "DATA" in bad.error


def test_momentum_runs_on_a_one_stock_panel_with_caveats(engines):
    r = tb.run("momentum", td.identify("ERIC-B.ST"), 3, _prices, today=TODAY)
    _check(r, "momentum", tb.FILL_MOMENTUM)
    assert any("En-aktieportfölj" in w for w in r.warnings) and r.notes
    assert all(t["result"].endswith(" %") or t["result"] == "öppen" for t in r.trades)
    for t in r.trades:
        assert t["stop"] == pytest.approx(round(t["entry"] * 0.9, 2)) and " · " in t["signal"]
    short = tb.run("momentum", td.identify("ERIC-B.ST"), 3, lambda t, p="1y": _walk(1, n=100, end=TODAY), today=TODAY)
    assert "DATA UNAVAILABLE" in short.error


def test_wolf_wraps_the_shadow_backtester_with_a_copied_config(engines):
    import wolf_shadow_backtest as wsb
    before = dict(wsb.CONFIG)
    r = tb.run("wolf", td.identify("NEM"), 3, _prices, today=TODAY)
    _check(r, "wolf", tb.FILL_CLOSE)
    assert wsb.CONFIG == before                                                        # configen rörs inte
    assert any(m[0] == "Avkastning" and "CAGR" in m[3] for m in r.metrics) and any(m[0] == "Sharpe" for m in r.metrics)
    assert any("XLB" in w for w in r.warnings) and r.notes
    assert all(t["signal"] in ("CORE", "TRIM") and t["stop"] is None for t in r.trades)


def test_dispatcher_handles_missing_engine_and_failures(engines):
    r = tb.run("quality", td.identify("BOL.ST"), 5, _prices)
    assert r.error == "ingen backtestmotor per aktie" and r.label == "Quality Contrarian"

    def boom(t, p="1y"):
        raise RuntimeError("nätet nere")
    r = tb.run("wolf", td.identify("NEM"), 5, boom)
    assert r.error == "RuntimeError: nätet nere" and r.metrics == [] and tb.chart(r, "NEM") is None


# ── Ren logik ────────────────────────────────────────────────────────────────
def test_metric_rows_and_chart_markers():
    assert tb.metrics_r({})[0][1] == "0" and tb.metrics_pct({"trades": 0, "open": 1})[0][3] == "1 öppen"
    m = {"trades": 4, "win_rate": 50.0, "avg_r": 0.5, "median_r": 0.2, "profit_factor": math.inf, "max_drawdown_r": 1.2,
         "avg_winner": 2.0, "avg_loser": -1.0, "expectancy": 0.5, "max_consecutive_losses": 2, "avg_holding_days": 9.0,
         "total_r": 2.0}
    rows = tb.metrics_r(m)
    assert [r[0] for r in rows] == ["Affärer", "Förväntan", "Profit factor", "Max DD", "Summa"]
    assert rows[1][1] == "+0.50 R" and rows[2][1] == "∞" and rows[4][1] == "+2.0 R"
    assert tb.metrics_pct({"trades": 2, "win_rate": 50.0, "total_return_pct": -3.0, "cagr_pct": None, "profit_factor": 0.8,
                           "max_dd_pct": 4.0, "avg_days": 7.0, "half_sold_share": 0.0})[1][3] == "CAGR —"
    idx = pd.bdate_range("2026-01-01", periods=120)
    res = tb.Result("wolf", "Wolf", 3, period=("2026-02-01", "2026-06-01"), close=pd.Series(range(120), index=idx, dtype=float),
                    trades=[{"signal": "x", "entry_date": "2026-02-10", "entry": 30.0, "stop": 27.0, "exit_date": "2026-03-10",
                             "exit": 40.0, "result": "+2.00 R", "reason": "trail", "days": 20, "open": False},
                            {"signal": "y", "entry_date": "2026-04-01", "entry": 60.0, "stop": 55.0, "exit_date": "",
                             "exit": None, "result": "öppen", "reason": "", "days": 5, "open": True}])
    fig = tb.chart(res, "X")
    names = [tr.name for tr in fig.data]
    assert names == ["X", "entry", "exit vinst"] and len(fig.layout.shapes) == 2                # två stoppar
    assert list(tb.trade_table(res.trades).columns) == ["Signal", "In", "Entry", "Stopp", "Ut", "Exit", "Resultat", "Dagar", "Skäl"]
    assert tb.trade_table(res.trades)["Ut"].tolist() == ["2026-03-10", "öppen"]


# ── Fliken ───────────────────────────────────────────────────────────────────
def _script():
    import pandas as pd
    import streamlit as st
    from thor import backtest as b
    from thor import dossier as d
    ident = d.identify(st.session_state["_ticker"])

    def runner(strategy, ident, years, prices, today=None):
        if st.session_state["_mode"] == "fail":
            return b.Result(strategy, "Wolf", years, error="RuntimeError: nätet nere")
        idx = pd.bdate_range(end="2026-10-15", periods=300)
        return b.Result(strategy, "Wolf", years, b.FILL_CLOSE, ("2026-01-02", "2026-10-15"),
                        metrics=[("Affärer", "3", "#00E5FF", "träff 67 %"), ("Avkastning", "+12.0 %", "#2d8a4e", "CAGR +9 %")],
                        trades=[{"signal": "CORE", "entry_date": "2026-03-02", "entry": 100.0, "stop": None, "exit_date": "2026-04-01",
                                 "exit": 112.0, "result": "+12.0 %", "reason": "TP1", "days": 21, "open": False}],
                        notes=("Pine-spegling",), warnings=["Sektor-ETF XLB"], close=pd.Series(range(300), index=idx, dtype=float))
    b.render(st, ident, st.session_state["_strategy"], lambda t, p="1y": None, runner=runner)


def _run(ticker, strategy, mode="ok", click=True):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_script, default_timeout=120)
    at.session_state["_ticker"], at.session_state["_strategy"], at.session_state["_mode"] = ticker, strategy, mode
    at.run()
    assert not at.exception, at.exception
    if click:
        at.button(key="thor_bt_run").click().run()
        assert not at.exception, at.exception
    return at, " ".join(m.value for m in at.markdown)


def test_tab_runs_on_the_button_and_shows_metrics_chart_table_and_notes():
    at, text = _run("NEM", "wolf", click=False)
    assert "Tryck Kör" in text and "▶ Kör Wolf på NEM · 5 år" in [b.label for b in at.button] and not at.get("plotly_chart")
    at, text = _run("NEM", "wolf")
    for part in ("Pine-spegling", "2026-01-02 → 2026-10-15", ">3<", "träff 67 %", "+12.0 %", "Sektor-ETF XLB"):
        assert part in text, part
    assert len(at.get("plotly_chart")) == 1 and len(at.dataframe) == 1 and "Förbehåll och regler" in [e.label for e in at.expander]
    assert at.session_state["thor_bt_NEM_wolf_5"].strategy == "wolf"                        # körningen sparas


def test_tab_without_engine_and_with_a_failed_run():
    _at, text = _run("BOL.ST", "quality", click=False)
    assert "har ingen backtestmotor per aktie" in text and "Valhalls backtester" in text
    _at, text = _run("NEM", "wolf", mode="fail")
    assert "Wolf på NEM: RuntimeError: nätet nere" in text
