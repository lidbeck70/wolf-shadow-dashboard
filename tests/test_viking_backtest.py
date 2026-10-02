"""
Viking PR 6 — backtest av Viking Nine i R, utan look-ahead. Syntetiska
kurser — inget nätverk.
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import ovtlyr_nine as on  # noqa: E402
import viking_backtest as vb  # noqa: E402

IDX = pd.bdate_range(end="2026-09-30", periods=900)


def _walk(drift, vol=0.012, seed=0):
    g = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(g.normal(drift, vol, len(IDX))))
    o = c * np.exp(g.normal(0, 0.004, len(IDX)))
    h = np.maximum(c, o) * (1 + abs(g.normal(0, 0.006, len(IDX))))
    lo = np.minimum(c, o) * (1 - abs(g.normal(0, 0.006, len(IDX))))
    v = 1e6 * np.exp(g.normal(0, 0.3, len(IDX)))
    return pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c, "Volume": v}, index=IDX)


DATA = {"SPY": _walk(0.0005, seed=10), **{t: _walk(0.0004, seed=20 + i) for i, t in enumerate(on.SECTOR_ETFS.values())},
        **{f"S{i}": _walk(0.0008, 0.02, seed=100 + i) for i in range(4)}}


def _run(**cfg):
    return vb.run([f"S{i}" for i in range(4)], getter=lambda t, p: DATA.get(t), sector_getter=lambda t: "Technology",
                  cfg=vb.Config(**cfg), today=pd.Timestamp("2026-09-30"))


# ── Look-ahead ───────────────────────────────────────────────────────────────
def test_factor_series_are_causal():
    """Serierna en dag får inte ändras av framtida data: räkna på hela serien och
    på serien kapad vid dagen — samma värde."""
    stock, spy, sec = DATA["S0"], DATA["SPY"], DATA["XLK"]
    breadth = on.breadth_series({t: DATA[t]["Close"] for t in on.SECTOR_ETFS.values()})
    full = vb.factor_frame(stock, spy, sec, breadth)
    xfull = vb.execution_frame(stock)
    for i in (300, 500, 700):
        d = IDX[i]
        cut = vb.factor_frame(stock.iloc[:i + 1], spy[spy.index <= d], sec[sec.index <= d], breadth[breadth.index <= d])
        assert full.iloc[i].equals(cut.iloc[-1]), d
        xcut = vb.execution_frame(stock.iloc[:i + 1])
        assert xfull.iloc[i].equals(xcut.iloc[-1]), d


def test_fear_greed_series_equals_the_screener_value():
    from screener_ovtlyr import _score_fear_greed
    s = on.fear_greed_series(DATA["S1"])
    for i in (100, 400, 899):
        assert s.iloc[i] == _score_fear_greed(DATA["S1"].iloc[:i + 1])


def test_entry_is_next_open_and_never_before_the_signal():
    res = _run(min_nine=8)
    assert res["trades"], "syntetiska data ska ge affärer vid Nine ≥ 8"
    for t in res["trades"]:
        assert t.entry_date > t.signal_date
        i = IDX.get_loc(pd.Timestamp(t.entry_date))
        assert t.entry == pytest.approx(DATA[t.ticker]["Open"].iloc[i], rel=1e-6)
        assert t.exit_date >= t.entry_date and t.stop < t.entry and t.nine >= 8
        assert t.r == pytest.approx((t.exit - t.entry) / t.risk, abs=1e-3)


def test_signals_only_inside_the_period():
    res = _run(min_nine=8, years=1)
    start = pd.Timestamp("2026-09-30") - pd.DateOffset(years=1)
    assert all(pd.Timestamp(t.signal_date) >= start for t in res["trades"])


def test_stricter_nine_gives_fewer_or_equal_trades():
    loose, strict = _run(min_nine=7), _run(min_nine=9)
    assert len(strict["trades"]) <= len(_run(min_nine=8)["trades"]) <= len(loose["trades"])


# ── Nyckeltalen ──────────────────────────────────────────────────────────────
def _t(r, day, days=3, open_=False):
    t = vb.Trade("X", day, day, 100, 99, 1, 9, exit_date=day, exit=100 + r, exit_reason="test", r=r, days=days)
    t.open = open_
    return t


def test_metrics_in_r_including_the_spec_example():
    trades = [_t(2.8, f"2026-01-{d:02d}") for d in range(1, 43)] + \
             [_t(-1.0, f"2026-03-{d:02d}") for d in range(1, 29)] + \
             [_t(-1.0, f"2026-04-{d:02d}") for d in range(1, 31)]
    m = vb.metrics(trades)                                     # 42 vinster à 2,8R, 58 förluster à −1R
    assert m["trades"] == 100 and m["win_rate"] == 42.0
    assert m["expectancy"] == pytest.approx(0.42 * 2.8 - 0.58 * 1, abs=1e-3)       # +0,596R
    assert m["avg_winner"] == 2.8 and m["avg_loser"] == -1.0
    assert m["profit_factor"] == pytest.approx(42 * 2.8 / 58, abs=0.01)
    assert m["max_consecutive_losses"] == 58 and m["max_drawdown_r"] == 58.0
    assert m["avg_holding_days"] == 3 and m["median_r"] == -1.0


def test_metrics_edge_cases():
    assert vb.metrics([]) == {"trades": 0}
    assert vb.metrics([_t(1.0, "2026-01-02", open_=True)]) == {"trades": 0}         # öppna räknas inte
    assert math.isinf(vb.metrics([_t(1.0, "2026-01-02")])["profit_factor"])


def test_missing_data_is_reported():
    res = vb.run(["NOPE"], getter=lambda t, p: DATA.get(t), sector_getter=lambda t: None,
                 cfg=vb.Config(), today=pd.Timestamp("2026-09-30"))
    assert res["per_ticker"][0]["error"] == "DATA UNAVAILABLE" and res["metrics"] == {"trades": 0}
    assert any("look" in n or "nästa" in n for n in res["notes"])


# ── Sidan ────────────────────────────────────────────────────────────────────
def test_result_view_renders(monkeypatch):
    from streamlit.testing.v1 import AppTest
    res = _run(min_nine=8)
    monkeypatch.setenv("VNB_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["VNB_TEST_ROOT"])
        import streamlit as _st
        from ovtlyr.ui.viking_nine_backtest import render_viking_nine_backtest
        render_viking_nine_backtest()
        _ = _st

    at = AppTest.from_function(app, default_timeout=90)
    at.session_state["vnb_result"] = res
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "EXPECTANCY" in html and "PROFIT FACTOR" in html and "MAX DRAWDOWN" in html and "WIN RATE" in html
    assert "EXITORSAKER" in html and "nästa" in html


def test_backtest_mode_is_listed():
    src = open(os.path.join(ROOT, "tabs", "backtest.py"), encoding="utf-8").read()
    assert '"⚔️ Viking Nine"' in src and "render_viking_nine_backtest" in src
