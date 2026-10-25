"""Kostnadsmodellen (backtest_costs.py): nivå per marknad, R efter kostnad, motorerna orörda."""
import pytest

import backtest_costs as bc
import viking_backtest as vb


def _t(ticker, r=1.0, reason="mål", open_=False):
    # entry 100, stopp 96 → stoppavstånd 4 %; 40 bp = 0,1 R, 90 bp = 0,225 R
    return vb.Trade(ticker, "2020-01-02", "2020-01-03", 100.0, 96.0, 4.0, 9, exit_date="2020-02-01",
                    exit=None if r is None else 100.0 + 4.0 * r, exit_reason=reason, r=r, open=open_)


def test_market_and_level():
    assert bc.market_of("VOLV-B.ST") == "norden" and bc.market_of("EQNR.OL") == "norden"
    assert bc.market_of("FCX") == "utland" and bc.market_of("CCJ.TO") == "utland"
    assert bc.bps_for("NOVO-B.CO") == 40 and bc.bps_for("XOM") == 90


def test_net_r_per_market_and_stop_slippage():
    a, b, c = bc.net_trades([_t("ABB.ST"), _t("XOM"), _t("ABB.ST", r=-1.0, reason="stopp")])
    assert a.r == pytest.approx(1.0 - 0.10)
    assert b.r == pytest.approx(1.0 - 0.225)
    assert c.r == pytest.approx(-1.0 - (40 + 25) / 10_000 / 0.04, abs=1e-3)       # R avrundas till tre decimaler


def test_originals_are_untouched_and_open_trades_skip():
    t = _t("ABB.ST")
    o = _t("ABB.ST", r=None, open_=True)
    bc.net_trades([t, o])
    assert t.r == 1.0
    assert bc.net_trades([o])[0] is o


def test_compare_counts_closed_trades_only():
    res = bc.compare([_t("ABB.ST"), _t("XOM", r=-1.0, reason="stopp"), _t("ABB.ST", r=None, open_=True)])
    assert res["n"] == 2 and res["exp"] == 0.0
    assert res["exp_net"] < res["exp"] and res["sum_r_net"] < res["sum_r"]
    assert res["cost_r"] == pytest.approx((0.10 + (90 + 25) / 10_000 / 0.04) / 2, abs=1e-3)
    assert bc.compare([])["n"] == 0
