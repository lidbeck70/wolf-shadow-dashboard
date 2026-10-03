"""
Viking PR 4 — EXIT ENGINE: varje exitregel ur strategy_rules.py (Viking)
som en egen kontroll. Syntetiska kurser — inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import viking_exit as vex  # noqa: E402
import viking_execution as vx  # noqa: E402

END = "2026-09-30"
TODAY = pd.Timestamp(END)


def _df(n=120, step=0.5, last=None, last_open=None):
    close = 100 + step * np.arange(n, dtype=float)
    idx = pd.bdate_range(end=END, periods=n)
    df = pd.DataFrame({"Open": close - 0.2, "High": close + 0.5, "Low": close - 0.5, "Close": close,
                       "Volume": 1e6}, index=idx)
    if last is not None:
        df.iloc[-1, df.columns.get_loc("Close")] = last
        df.iloc[-1, df.columns.get_loc("Low")] = min(last, df["Low"].iloc[-1])
    if last_open is not None:
        df.iloc[-1, df.columns.get_loc("Open")] = last_open
        df.iloc[-1, df.columns.get_loc("High")] = max(last_open, df["High"].iloc[-1])
    return df


class _Nine:
    def __init__(self, spy="PASS", sector="PASS", breadth="PASS", signal="PASS"):
        self.f = {"market.signal": SimpleNamespace(status=spy, detail=f"SPY {spy}"),
                  "sector.breadth": SimpleNamespace(status=sector, detail=""),
                  "market.breadth": SimpleNamespace(status=breadth, detail=""),
                  "stock.signal": SimpleNamespace(status=signal, detail=f"stock {signal}")}

    def get(self, key):
        return self.f.get(key)


def _exit(df=None, entry=None, entry_date=None, **kw):
    df = _df() if df is None else df
    entry_date = entry_date or df.index[-30]
    entry = entry if entry is not None else float(df.loc[entry_date, "Close"])
    args = dict(nine=_Nine(), earnings_date=pd.Timestamp("2026-12-01"), today=TODAY)
    args.update(kw)
    return vex.evaluate_exit("NVDA", df, entry, entry_date, **args)


def _by(d):
    return {t.key: t for t in d.triggers}


def test_hold_while_trend_runs():
    d = _exit()
    assert d.status == vex.HOLD and not d.active
    assert d.breakeven_armed and d.current_stop == d.entry          # ny högre topp → stopp till breakeven
    assert d.initial_stop < d.entry and d.trailing_stop < d.price and d.r_now > 0


def test_hard_market_exit_closes_all():
    d = _exit(nine=_Nine(spy="FAIL"))
    assert d.status == vex.CLOSE_ALL and "stäng alla positioner" in d.reasons[0]
    spy = _df(step=-0.5)
    d = _exit(nine=None, spy_df=spy)
    assert _by(d)["market"].active and d.status == vex.CLOSE_ALL


def test_initial_stop_is_entry_minus_atr_at_entry():
    df = _df()
    ed = df.index[-30]
    atr_e = float(vx.atr(df)[df.index <= ed].iloc[-1])
    d = _exit(df, entry_date=ed, be_moved=False)
    assert d.initial_stop == pytest.approx(d.entry - 1.5 * atr_e, abs=0.01) and d.current_stop == d.initial_stop
    crash = _df(last=float(df["Close"].iloc[-30]) - 10)
    d = _exit(crash, entry_date=ed, entry=float(df["Close"].iloc[-30]), be_moved=False)
    assert _by(d)["stop"].active and d.status == vex.EXIT


def test_trailing_stop_ema10():
    df = _df()
    d = _exit(_df(last=float(df["Close"].iloc[-1]) - 4), be_moved=True)
    assert _by(d)["trail"].active and "EMA10" in _by(d)["trail"].detail


def test_breakeven_exit_needs_a_moved_stop():
    df = _df()
    below_prev_low = float(df["Low"].iloc[-2]) - 0.1
    d = _exit(_df(last=below_prev_low), be_moved=True)
    assert _by(d)["be"].active and "gårdagens low" in _by(d)["be"].detail
    d = _exit(_df(last=below_prev_low), be_moved=False)
    assert not _by(d)["be"].active and "inte flyttad" in _by(d)["be"].detail


def test_gap_and_crap():
    df = _df()
    prev_high, prev_close = float(df["High"].iloc[-2]), float(df["Close"].iloc[-2])
    d = _exit(_df(last=prev_close - 0.2, last_open=prev_high + 1.0))
    assert _by(d)["gap"].active and "över gårdagens high" in _by(d)["gap"].detail
    d = _exit(_df(last=prev_close + 2, last_open=prev_high + 1.0))
    assert not _by(d)["gap"].active


def test_bearish_block_sector_breadth_and_stock_signal():
    df = _df()
    price = float(df["Close"].iloc[-1])
    d = _exit(ob_analysis={"nearest_bearish_ob": {"low": price - 1, "high": price + 3}})
    assert _by(d)["block"].active and "inne i blocket" in _by(d)["block"].detail
    d = _exit(ob_analysis={"nearest_bearish_ob": {"low": price + 5, "high": price + 8}})
    assert not _by(d)["block"].active
    assert _by(_exit(nine=_Nine(sector="FAIL", breadth="FAIL")))["breadth"].active
    assert not _by(_exit(nine=_Nine(sector="FAIL")))["breadth"].active           # båda krävs
    assert _by(_exit(nine=_Nine(signal="FAIL")))["signal"].active
    gap = _exit(nine=_Nine(sector="DATA UNAVAILABLE"))
    assert _by(gap)["breadth"].status == vex.UNAVAILABLE and gap.status == vex.HOLD   # saknad data utlöser inget


def test_fear_greed_targets():
    assert vex.fg_target(30) == 63 and vex.fg_target(49.9) == 63
    assert vex.fg_target(60) == 70 and vex.fg_target(80) == 85 and vex.fg_target(None) is None
    d = _exit()
    assert d.fg_entry is not None and d.fg_target == vex.fg_target(d.fg_entry)
    assert "WOLF APPROXIMATION" in _by(d)["fg"].detail


def test_earnings_before_report():
    d = _exit(earnings_date=pd.Timestamp("2026-10-05"))
    assert _by(d)["earnings"].active and "EXIT / REDUCE BEFORE EARNINGS" in _by(d)["earnings"].detail
    assert _by(_exit(earnings_date=None))["earnings"].status == vex.UNAVAILABLE


def test_missing_data():
    d = vex.evaluate_exit("X", _df(n=10), 100, END)
    assert d.status == vex.HOLD and "DATA UNAVAILABLE" in d.reasons[0]


def test_exit_card():
    from ovtlyr.ui.exit_card import exit_html
    html = exit_html(_exit())
    assert ">HOLD<" in html and "Initial SL" in html and "Current SL" in html and "(breakeven)" in html
    assert "Trailing (EMA10)" in html and "EXITREGLER" in html and "Gap & crap" in html
    html = exit_html(_exit(nine=_Nine(spy="FAIL")))
    assert "CLOSE ALL" in html
