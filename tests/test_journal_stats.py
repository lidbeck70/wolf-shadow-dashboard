"""
Tests for journal_stats.py — statistikbladet ur tradingjournal_swing.xlsx.

  L: =IF(OR(I="",F="",E=""),"",(I-F)*E-IF(K="",0,K))
  M: =IF(OR(I="",F=""),"",(I-F)/F)
  N: =IF(OR(H="",B=""),"",H-B)
  O: =IF(OR(M="",G="",F="",F=G),"",M/((F-G)/F))
  payoff: =AVERAGEIF(L,">0")/ABS(AVERAGEIF(L,"<0"))
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import journal_stats as js


def _t(pnl_amount=None, pnl_pct=None, **kw) -> dict:
    t = {"exit_price": 1, "pnl_amount": pnl_amount, "pnl_pct": pnl_pct}
    t.update(kw)
    return t


# ── Per affär ────────────────────────────────────────────────────────────────
def test_pnl_amount_subtracts_fees():
    assert js.pnl_amount(100, 120, 10) == 200
    assert js.pnl_amount(100, 120, 10, 39) == 161
    assert js.pnl_amount(100, 120, 10, None) == 200      # tomt courtage = 0
    assert js.pnl_amount(100, None, 10) is None


def test_pnl_pct():
    assert js.pnl_pct(100, 120) == 20.0
    assert js.pnl_pct(100, 90) == -10.0
    assert js.pnl_pct(0, 90) is None


def test_risk_pct_is_the_distance_to_the_stop():
    assert js.risk_pct(100, 90) == 10.0
    assert js.risk_pct(100, 100) is None      # arket: F=G ger tom cell
    assert js.risk_pct(100, None) is None
    assert js.risk_pct(0, 90) is None


def test_r_multiple_from_the_stop_you_actually_set():
    """A +20 % win on a −10 % stop is 2R, not something you type in."""
    assert js.r_multiple(100, 90, exit_price=120) == 2.0
    assert js.r_multiple(100, 90, exit_price=90) == -1.0     # stoppad = −1R
    assert js.r_multiple(100, 95, exit_price=120) == 4.0     # tightare stop
    assert js.r_multiple(100, 90, result_pct=20.0) == 2.0


def test_r_multiple_is_blank_without_a_stop():
    """The panel used to let you type R with no stop recorded at all."""
    assert js.r_multiple(100, None, exit_price=120) is None
    assert js.r_multiple(100, 100, exit_price=120) is None
    assert js.r_multiple(100, 90) is None                    # varken kurs/%


def test_holding_days_accepts_dates_and_iso_strings():
    assert js.holding_days("2026-01-01", "2026-02-01") == 31
    assert js.holding_days(date(2026, 1, 1), date(2026, 1, 15)) == 14
    assert js.holding_days("2026-01-01T09:00:00", "2026-01-08") == 7
    assert js.holding_days("", "2026-01-08") is None
    assert js.holding_days("inte ett datum", "2026-01-08") is None


# ── Över alla affärer ────────────────────────────────────────────────────────
def _book() -> list:
    return [_t(300, 15.0, setup="A", sell_rule=js.SELL_PARTIAL, r_multiple=1.5),
            _t(200, 10.0, setup="A", sell_rule=js.SELL_MA50, r_multiple=1.0),
            _t(-100, -10.0, setup="B", sell_rule=js.SELL_STOP, r_multiple=-1.0),
            _t(-100, -10.0, setup="B", sell_rule=js.SELL_STOP, r_multiple=-1.0)]


def test_win_rate_and_payoff():
    b = _book()
    assert js.win_rate(b) == 50.0
    # snittvinst 250, snittförlust −100 -> 2,5
    assert js.payoff_ratio(b) == 2.5
    assert js.PAYOFF_TARGET == 2.0


def test_payoff_is_blank_until_both_sides_exist():
    """No losses yet is not an infinite payoff — the sheet shows "-"."""
    assert js.payoff_ratio([_t(300, 15.0), _t(100, 5.0)]) is None
    assert js.payoff_ratio([_t(-100, -5.0)]) is None
    assert js.payoff_ratio([]) is None
    assert js.win_rate([]) is None


def test_averages():
    b = _book()
    assert js.average(b, "r_multiple") == 0.125
    assert js.average(b, "pnl_pct") == 1.25
    assert js.average(b, "saknas") is None


def test_setup_breakdown_compares_a_against_b():
    out = js.setup_breakdown(_book())
    assert out["A"]["count"] == 2 and out["A"]["avg_pct"] == 12.5
    assert out["B"]["count"] == 2 and out["B"]["avg_pct"] == -10.0


def test_exit_breakdown_counts_every_rule():
    out = js.exit_breakdown(_book())
    assert out[js.SELL_STOP] == 2        # många stoppar = slagig eller sena köp
    assert out[js.SELL_MA50] == 1
    assert out[js.SELL_PARTIAL] == 1
    assert out[js.SELL_RANK] == 0
    assert set(out) == set(js.SELL_RULES)


def test_unknown_sell_rules_are_ignored_not_crashed_on():
    out = js.exit_breakdown([_t(100, 5.0, sell_rule="Regime Change"),
                             _t(100, 5.0)])
    assert sum(out.values()) == 0


def test_open_trades_are_excluded_from_every_statistic():
    """The sheet counts on a filled sell price — an open position is not a result."""
    book = _book() + [{"ticker": "OPEN"}]
    assert js.summary(book)["closed"] == 4
    assert js.win_rate(book) == 50.0


def test_the_twenty_trade_warning():
    assert js.MIN_TRADES == 20
    assert not js.enough_trades(_book())
    assert js.enough_trades([_t(100, 5.0)] * 20)


def test_the_warning_threshold_brackets_the_guides_range():
    """Masterguiden says 15–20; routines holds the lower bound, this the upper."""
    import routines
    assert routines.MIN_TRADES_FOR_STATS == 15
    assert js.MIN_TRADES == 20
    assert routines.MIN_TRADES_FOR_STATS < js.MIN_TRADES


def test_summary_is_safe_on_an_empty_book():
    s = js.summary([])
    assert s["closed"] == 0 and s["enough"] is False
    assert s["win_rate"] is None and s["payoff"] is None
    assert s["total"] == 0
    assert s["exits"][js.SELL_STOP] == 0
    assert js.summary(None)["closed"] == 0


def test_sell_rules_match_the_momentum_playbook():
    """1 = MA50, 2 = −10 % stop, 3 = ur topp 40, plus delvinsten vid +20 %."""
    import reference
    row = reference.sell_rule("momentum").rule
    assert "MA50" in js.SELL_RULE_LABEL[js.SELL_MA50] and "MA50" in row
    assert "−10 %" in js.SELL_RULE_LABEL[js.SELL_STOP] and "−10 %" in row
    assert "topp 40" in js.SELL_RULE_LABEL[js.SELL_RANK] and "topp 40" in row
    assert "+20 %" in js.SELL_RULE_LABEL[js.SELL_PARTIAL] and "+20 %" in row
