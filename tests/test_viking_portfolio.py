"""
Viking Nine-backtestets portföljläge: samma affärer genom ett konto med
Vikings gränser (1,5 % risk, max 25 % per position, max 100 % investerat,
max två förluster per dag). Syntetiska affärer — inget nätverk.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import viking_backtest as vb  # noqa: E402
import viking_portfolio as vp  # noqa: E402

D = [str(d.date()) for d in pd.bdate_range("2026-01-05", periods=40)]


def _t(tk, sig, exit_, r=1.0, risk=3.75, nine=9, mom=0.1, open_=False):
    """Signal på D[sig], entry D[sig+1], exit D[exit_]. risk 3,75 på 100 → stopp 3,75 % → 25 % (taket)."""
    return vb.Trade(tk, D[sig], D[sig + 1], 100.0, 100.0 - risk, risk, nine, exit_date=D[exit_], exit=100 + r * risk,
                    exit_reason="öppen" if open_ else "test", r=r, days=exit_ - sig - 1, open=open_, mom63=mom)


def test_position_size_uses_risk_and_the_position_cap():
    assert vp.position_pct(_t("A", 0, 3, risk=3.75)) == 25.0                       # 1,5/3,75 = 40 % → tak 25
    assert vp.position_pct(_t("A", 0, 3, risk=10.0)) == pytest.approx(15.0)        # 1,5/10 = 15 %
    assert vp.position_pct(_t("A", 0, 3, risk=0.0)) == 0.0


def test_no_more_than_fully_invested():
    trades = [_t(f"T{k}", 0, 10) for k in range(5)] + [_t("LATE", 11, 15)]
    p = vp.simulate(trades)
    assert p["taken"] == 5 and p["skipped_full"] == 1 and p["max_open"] == 4      # 4 × 25 % = 100 %
    assert "LATE" in [r["trade"].ticker for r in p["rows"]]                       # platsen frigjord efter exit


def test_slot_is_not_reused_on_the_exit_day():
    p = vp.simulate([_t(f"T{k}", 0, 5) for k in range(4)] + [_t("SAME", 4, 8)])
    assert p["skipped_full"] == 1                                                 # exit D5 = entry D5 → fullt


def test_priority_highest_nine_then_momentum():
    trades = [_t("LOW", 0, 5, nine=8, mom=0.9), _t("A", 0, 5, mom=0.05), _t("B", 0, 5, mom=0.3),
              _t("C", 0, 5, mom=0.2), _t("D", 0, 5, mom=0.1)]
    p = vp.simulate(trades)
    assert sorted(r["trade"].ticker for r in p["rows"]) == ["A", "B", "C", "D"]   # Nine 8 åker ut först
    p = vp.simulate([_t(f"M{k}", 0, 5, mom=k / 10) for k in range(5)])
    assert "M0" not in [r["trade"].ticker for r in p["rows"]]                     # svagast momentum åker ut


def test_two_losses_stop_new_entries_after_that_day():
    trades = [_t("L1", 0, 4, r=-1.0, risk=10), _t("L2", 0, 4, r=-1.0, risk=10),
              _t("BLOCK", 4, 8, risk=10), _t("NEXT", 5, 9, risk=10)]
    p = vp.simulate(trades)
    names = [r["trade"].ticker for r in p["rows"]]
    assert "BLOCK" not in names and "NEXT" in names and p["skipped_losses"] == 1
    one = vp.simulate([_t("L1", 0, 4, r=-1.0, risk=10), _t("OK", 4, 8, risk=10)])
    assert one["skipped_losses"] == 0                                             # en förlust räcker inte


def test_skipped_losses_do_not_count():
    trades = [_t(f"T{k}", 0, 10) for k in range(4)] + [_t("X1", 0, 4, r=-1), _t("X2", 0, 4, r=-1),
                                                        _t("AFTER", 4, 6, nine=9, risk=3.75)]
    p = vp.simulate(trades)
    assert p["skipped_losses"] == 0                                               # X1/X2 togs aldrig


def test_returns_compound_on_the_account():
    p = vp.simulate([_t("A", 0, 3, r=2.0, risk=10), _t("B", 4, 8, r=-1.0, risk=10)], years=1)
    a, b = (r["return_pct"] for r in p["rows"])
    assert a == pytest.approx(3.0) and b == pytest.approx(-1.5)                   # R × 1,5 % risk
    assert p["return_pct"] == pytest.approx(round((1.03 * 0.985 - 1) * 100, 1))
    assert p["max_dd_pct"] == pytest.approx(1.5) and p["cagr_pct"] == pytest.approx(p["return_pct"], abs=0.1)
    capped = vp.simulate([_t("C", 0, 3, r=1.0, risk=3.75)])
    assert capped["rows"][0]["risk_pct"] == pytest.approx(0.94, abs=0.01)         # 25 % × 3,75 % — under 1,5 %


def test_open_trades_hold_a_slot_but_are_not_counted():
    p = vp.simulate([_t(f"O{k}", 0, 39, open_=True) for k in range(4)] + [_t("NEW", 10, 12)])
    assert p["skipped_full"] == 1 and p["metrics"] == {"trades": 0} and p["curve"] == []


def test_run_and_page_show_the_portfolio(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from tests.test_viking_backtest import DATA, ROOT
    res = vb.run([f"S{i}" for i in range(4)], getter=lambda t, p: DATA.get(t), sector_getter=lambda t: "Technology",
                 cfg=vb.Config(min_nine=8, years=3), today=pd.Timestamp("2026-09-30"))
    assert all(t.mom63 is not None for t in res["trades"])
    from ovtlyr.ui.viking_nine_backtest import comparison_rows, portfolio_of
    p = portfolio_of(res)
    assert 0 < p["taken"] <= p["candidates"] == len(res["trades"])
    row = comparison_rows({"X": res})[0]
    assert row["Portfölj affärer"] == p["taken"] and row["Portfölj DD %"] == p["max_dd_pct"]
    monkeypatch.setenv("VNB_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["VNB_TEST_ROOT"])
        from ovtlyr.ui.viking_nine_backtest import render_viking_nine_backtest
        render_viking_nine_backtest()

    res.pop("portfolio")                                                          # som ett äldre sessionsresultat
    at = AppTest.from_function(app, default_timeout=90)
    at.session_state["vnb_result"] = res
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "PORTFÖLJ — ETT KONTO" in html and "TAGNA AFFÄRER" in html and "AVKASTNING" in html
    assert "max 25 % per position" in html
