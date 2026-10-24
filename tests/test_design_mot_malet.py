"""Designförslaget PR 7: Mot målet — golv 20 / mål 35, papper och journal sida vid sida, inga slutsatser under 20."""
from datetime import date

import journal_stats as js
from portfolio import mot_malet as mm
from ui import nav


def _t(entry, exit_, pnl, strategy="viking", **kw):
    return {"entry_date": entry, "exit_date": exit_, "pnl_amount": pnl, "pnl_pct": pnl / 100, "exit_price": 1.0,
            "strategy": strategy, **kw}


def test_zones_against_the_goal():
    assert mm.GOAL_FLOOR == 20 and mm.GOAL_TARGET == 35
    assert mm.zone(40)[0] == "MÅLET" and mm.zone(25)[0] == "GOLVET" and mm.zone(10)[0] == "UNDER"
    assert mm.zone(-3)[0] == "MINUS" and mm.zone(None)[0] == "Saknas"


def test_cagr_runs_to_today_not_to_last_sale():
    trades = [_t("2025-01-01", "2025-02-01", 7500)]
    one_year = js.realized_cagr(trades, 75000, until=date(2026, 1, 1))
    two_years = js.realized_cagr(trades, 75000, until=date(2027, 1, 1))
    assert round(one_year) == 10 and two_years < one_year


def test_cagr_needs_time_and_closed_trades():
    assert js.realized_cagr([], 75000) is None
    assert js.realized_cagr([_t("2026-10-01", "2026-10-02", 100)], 75000, until=date(2026, 10, 10)) is None


def test_strategy_rows_join_paper_and_journal_via_the_registry():
    paper = {"by_strategy": [{"strategy": "viking", "closed": 25, "closed_win": 40.0, "avg_r": 0.3, "payoff": 2.1,
                              "unit_cagr": 12.0}]}
    rows = mm.strategy_rows(paper, [_t("2026-01-01", "2026-02-01", 100, "viking_nine")])
    vn = next(r for r in rows if r["Strategi"] == "Viking Nine")
    assert vn["Papper stängda"] == "25/20" and vn["Papper årstakt %"] == 12.0
    assert vn["Journal stängda"] == "1/20" and vn["Journal snitt-R"] is None


def test_no_conclusion_under_twenty():
    paper = {"by_strategy": [{"strategy": "wolf_arc", "closed": 5, "closed_win": 80.0, "avg_r": 2.0, "unit_cagr": 90.0}]}
    r = next(r for r in mm.strategy_rows(paper, []) if r["Strategi"] == "Wolf")
    assert r["Papper vinst %"] is None and r["Papper årstakt %"] is None and r["Slutsats"] == "för få affärer"


def test_plan_followed_share():
    assert mm.plan_followed([{"followed_plan": True}, {"followed_plan": False}, {}]) == 50.0
    assert mm.plan_followed([{}]) is None


def test_today_line():
    line = mm.today_line(None, {"by_strategy": []}, [])
    assert line == "Papper: för få stängda affärer · journalen: för få affärer (0/20)"


def test_goal_page_in_tree():
    assert nav.resolve(nav.target_slug("goal")) == ["portfolio", "🎯 Mot målet"]
    assert nav.resolve(nav.target_slug("backtests")) == ["valhall", "Backtester"]
