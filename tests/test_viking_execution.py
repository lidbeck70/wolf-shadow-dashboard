"""
Viking PR 3 — VIKING EXECUTION, RISK ENGINE och evaluate_entry().
Specens acceptanstester 1–7 plus earnings, SPY-signal, stängd candle och
exponering. Syntetiska kurser — inget nätverk.
"""
import math
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import viking_execution as vx  # noqa: E402

END = "2026-09-30"
AFTER_CLOSE = datetime(2026, 9, 30, 22, 0, tzinfo=timezone.utc)      # 18:00 New York
INTRADAY = datetime(2026, 9, 30, 17, 0, tzinfo=timezone.utc)         # 13:00 New York
FAR = pd.Timestamp("2026-11-15")


def _df(n=260, spike=None, last_up=1.5):
    """Stigande trappa (+1, +1, −0,5); sista dagen grön och starkare. spike = (dagar sedan, high)."""
    steps = np.tile([1.0, 1.0, -0.5], n // 3 + 1)[:n] * 0.5
    steps[-2], steps[-1] = -0.25, last_up
    close = 100 + np.cumsum(steps)
    idx = pd.bdate_range(end=END, periods=n)
    df = pd.DataFrame({"Open": close - 0.3 * np.sign(steps), "High": close + 0.6, "Low": close - 0.6,
                       "Close": close, "Volume": 1e6}, index=idx)
    df.iloc[-1, df.columns.get_loc("Open")] = close[-1] - 1.0
    if spike:
        ago, high = spike
        df.iloc[-ago, df.columns.get_loc("High")] = high
    return df


class _Nine:
    def __init__(self, passed=9, spy="PASS"):
        self.passed, self._spy = passed, spy

    def get(self, key):
        return SimpleNamespace(status=self._spy) if key == "market.signal" else None


def _eval(df=None, nine=None, **kw):
    args = dict(nine=nine or _Nine(), capital=100_000, earnings_date=FAR, earnings_known=True, trades=[],
                now=AFTER_CLOSE)
    args.update(kw)
    return vx.evaluate_entry("NVDA", _df() if df is None else df, **args)


def _stop_distance(df):
    return float(vx.atr(df).iloc[-1]) * vx.ATR_STOP_MULT


# ── Acceptanstester ur specen ────────────────────────────────────────────────
def test_1_golden_ticket():
    d = _eval()
    assert d.status == vx.GO, d.reasons
    by = {c.key: c for c in d.checks}
    assert all(by[k].passed for k in ("momentum", "trend", "candle", "chase", "rr", "earnings"))
    assert by["volume"].required is False                       # visas, krävs inte förrän backtestat
    assert d.position.risk_pct <= vx.MAX_RISK_PCT and d.as_dict()["ovtlyr_nine"] == "9/9"


def test_2_nine_full_but_execution_waits():
    d = _eval(now=INTRADAY)                                      # dagens candle inte stängd
    assert d.status == vx.WAIT and vx.WAIT_CLOSE in d.flags
    assert any("Entry candle" in r for r in d.reasons)


def test_3_nine_six_is_no_trade():
    d = _eval(nine=_Nine(passed=6))
    assert d.status == vx.NO_TRADE and "6/9" in d.reasons[0]
    assert _eval(nine=_Nine(passed=8)).status == vx.WAIT          # DEVELOPING


def test_4_position_size_example():
    p = vx.size_position(100_000, 318.40, 6.29, max_position_pct=100)
    assert p.risk_budget == 1500 and p.stop_distance == pytest.approx(9.435)
    assert p.shares == 158 == math.floor(1500 / 9.435)          # 158,98 → nedåt, risken går aldrig över 1,5 %
    assert p.stop == pytest.approx(308.97, abs=0.01)
    assert p.risk_amount == pytest.approx(158 * 9.435, abs=0.01) and p.risk_pct <= 1.5
    assert p.position_value == pytest.approx(158 * 318.40) and p.exposure_pct == pytest.approx(50.3, abs=0.1)
    assert p.capped_by_exposure is False
    capped = vx.size_position(100_000, 318.40, 6.29)              # standardtak 25 %
    assert capped.shares == 78 and capped.capped_by_exposure and capped.shares_by_risk == 158
    assert capped.exposure_pct <= 25 and capped.risk_pct < 1.5


def test_5_two_losses_today_disable_trading():
    trades = [{"exit_date": END, "pnl_pct": -2.0}, {"exit_date": END, "r_multiple": -1.0},
              {"exit_date": "2026-09-29", "pnl_pct": -3.0}, {"exit_date": END, "pnl_pct": 4.0}]
    assert vx.daily_losses(trades, pd.Timestamp(END)) == 2
    d = _eval(trades=trades)
    assert d.status == vx.NO_TRADE and vx.DAILY_LIMIT in d.flags and "Trading disabled" in d.reasons[0]
    assert _eval(trades=trades[:1]).status == vx.GO


def test_6_no_chase():
    df = _df()
    d = _eval(df, current_price=float(df["Close"].iloc[-1]) * 1.03)
    assert vx.NO_CHASE in d.flags and d.status == vx.WAIT
    assert "+3.00 %" in next(c for c in d.checks if c.key == "chase").detail


def test_7_insufficient_rr():
    base = _df()
    entry, risk = float(base["Close"].iloc[-1]), _stop_distance(base)
    d = _eval(_df(spike=(100, entry + 1.0 * risk)))
    assert vx.INSUFFICIENT_RR in d.flags and d.status == vx.WAIT and d.rr == pytest.approx(1.0, abs=0.05)
    ok = _eval(_df(spike=(100, entry + 3.0 * risk)))
    assert ok.status == vx.GO and ok.rr == pytest.approx(3.0, abs=0.05)
    assert _eval().rr is None and "fri väg" in next(c for c in _eval().checks if c.key == "rr").detail


# ── Övrigt ───────────────────────────────────────────────────────────────────
def test_earnings_and_market_signal():
    soon = _eval(earnings_date=pd.Timestamp("2026-10-05"))
    assert soon.status == vx.NO_TRADE and vx.EARNINGS_RISK in soon.flags
    unknown = _eval(earnings_date=None, earnings_known=False)
    assert unknown.status == vx.WAIT and any("rapportdatum okänt" in r for r in unknown.reasons)
    past = _eval(earnings_date=pd.Timestamp("2026-07-20"))
    assert past.status == vx.GO
    spy = _eval(nine=_Nine(spy="FAIL"))
    assert spy.status == vx.NO_TRADE and any("SPY under EMA20" in r for r in spy.reasons)


def test_momentum_and_trend_failures():
    df = _df()
    df.iloc[-1, df.columns.get_loc("Close")] = float(df["Close"].iloc[-2]) - 3.0     # stark nedgång i dag
    d = _eval(df)
    by = {c.key: c for c in d.checks}
    assert not by["momentum"].passed and not by["candle"].passed and d.status == vx.WAIT
    assert "FAILED" in by["candle"].detail


def test_candle_close_per_exchange():
    bar = pd.Timestamp("2026-09-30")
    assert not vx.candle_closed("NVDA", bar, INTRADAY) and vx.candle_closed("NVDA", bar, AFTER_CLOSE)
    sthlm_open = datetime(2026, 9, 30, 14, 0, tzinfo=timezone.utc)                 # 16:00 Stockholm
    sthlm_closed = datetime(2026, 9, 30, 16, 0, tzinfo=timezone.utc)               # 18:00 Stockholm
    assert not vx.candle_closed("VOLV-B.ST", bar, sthlm_open) and vx.candle_closed("VOLV-B.ST", bar, sthlm_closed)
    assert vx.candle_closed("NVDA", pd.Timestamp("2026-09-29"), INTRADAY)          # gårdagens stapel är stängd


def test_intraday_uses_the_last_closed_candle_as_trigger():
    d = _eval(now=INTRADAY)
    assert d.trigger_date == "2026-09-29"


def test_engine_constants_come_from_the_viking_strategy():
    from strategies.viking import DEFAULT_PARAMS as P
    assert vx.MAX_RISK_PCT == P["risk_pct"] * 100 == 1.5 and vx.ATR_STOP_MULT == P["atr_stop_mult"] == 1.5
    assert vx.RSI_MIN == 50 and vx.RELATIVE_VOLUME_MIN == 1.2 and vx.MAX_CHASE_PCT == 2.0
    assert vx.MINIMUM_RR == 2.0 and vx.MAX_DAILY_LOSSES == 2 and vx.EARNINGS_BUFFER_DAYS == 5


def test_watchlist_categories():
    assert vx.watchlist_category(9, vx.GO) == "GOLDEN TICKET"
    assert vx.watchlist_category(9, vx.WAIT) == "READY"
    assert vx.watchlist_category(8) == "DEVELOPING" and vx.watchlist_category(7) == "DEVELOPING"
    assert vx.watchlist_category(6) == "REJECTED" and vx.watchlist_category(None) == "REJECTED"


def test_too_little_history_is_no_trade():
    d = vx.evaluate_entry("NVDA", _df(n=40), nine=_Nine(), now=AFTER_CLOSE)
    assert d.status == vx.NO_TRADE and "DATA UNAVAILABLE" in d.reasons[0]


def test_cards_render_status_checks_and_risk():
    from ovtlyr.ui.execution_card import decision_html, risk_html
    html = decision_html(_eval())
    assert "GOLDEN TICKET" in html and "OVTLYR Nine <b>9/9</b>" in html and "Viking Execution <b>" in html
    assert "Momentum" in html and "No chase" in html and "(info)" in html
    wait = decision_html(_eval(now=INTRADAY))
    assert ">WAIT<" in wait and "WAIT FOR CLOSE" in wait and "Missing / väntar på" in wait
    risk = risk_html(_eval())
    assert "RISK ENGINE" in risk and "Riskbudget" in risk and "1.5 % = 1,500" in risk and "Exponering" in risk
