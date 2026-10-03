"""
Viking Nine: EMA10-trailing gäller först när stoppen flyttats till breakeven —
i den riktiga exitmotorn (viking_exit) och som förval i backtestet.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import viking_backtest as vb  # noqa: E402
import viking_exit as vex  # noqa: E402
from tests.test_viking_exit import _by, _df, _exit  # noqa: E402


def _under_ema10():
    df = _df()
    return _df(last=float(df["Close"].iloc[-1]) - 4)


def test_ema10_waits_for_breakeven():
    d = _exit(_under_ema10(), be_moved=False)
    trail = _by(d)["trail"]
    assert not trail.active and trail.status == vex.CLEAR
    assert "väntar på breakeven" in trail.detail and "EMA10" in trail.detail
    assert d.trailing_stop is not None and d.status == vex.HOLD


def test_ema10_exits_after_breakeven():
    d = _exit(_under_ema10(), be_moved=True)
    assert _by(d)["trail"].active and d.status == vex.EXIT
    assert any("Trailing stop (EMA10)" in r for r in d.reasons)


def test_ema10_uses_computed_breakeven_when_not_given():
    d = _exit(_under_ema10())                     # ny högre topp efter entry → armerad
    assert d.breakeven_armed and _by(d)["trail"].active


def test_initial_stop_still_applies_before_breakeven():
    df = _df()
    ed = df.index[-30]
    crash = _df(last=float(df["Close"].iloc[-30]) - 10)
    d = _exit(crash, entry_date=ed, entry=float(df["Close"].iloc[-30]), be_moved=False)
    assert _by(d)["stop"].active and not _by(d)["trail"].active and d.status == vex.EXIT


def test_backtest_default_matches_live_engine():
    first, (rules, after_be) = next(iter(vb.EXIT_PRESETS.items()))
    assert first == "EMA10 först efter breakeven" and after_be and rules == vb.ALL_EXITS
    assert vb.Config().trail_after_be is True
    assert vb.EXIT_PRESETS["Alla regler"] == (vb.ALL_EXITS, False)


def test_backtest_no_ema10_exit_before_breakeven():
    n = 250
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0015, 0.015, n)))
    idx = pd.bdate_range(end="2026-09-30", periods=n)
    stock = pd.DataFrame({"Open": close * 0.998, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                          "Volume": 1e6}, index=idx)
    spy = pd.DataFrame({"Close": np.linspace(100, 150, n)}, index=idx)
    breadth = pd.Series(80.0, index=idx)
    run = lambda after: vb.backtest_ticker("X", stock, spy, stock.copy(), breadth,  # noqa: E731
                                           vb.Config(min_nine=7, exit_rules=("trail",), trail_after_be=after,
                                                     require_volume=False))
    res, early = run(True), run(False)
    assert [t.exit_reason for t in res["trades"]] != [t.exit_reason for t in early["trades"]]
    h = stock["High"].values
    checked = 0
    for t in res["trades"]:
        if t.exit_reason != "trailing EMA10":
            continue
        i = idx.get_loc(pd.Timestamp(t.signal_date))
        pre = h[max(0, i + 1 - vex.BE_LOOKBACK):i + 2].max()
        rule_day = idx.get_loc(pd.Timestamp(t.exit_date)) - 1           # stängningsregel → exit nästa öppning
        assert h[i + 2:rule_day].max() > pre                            # breakeven nådd innan EMA10 fick stänga
        checked += 1
    assert checked
