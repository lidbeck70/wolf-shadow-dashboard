"""Besluten 2, 3 och 4: journalen färgas mot 20/35, årstakten räknas till i dag, inga 0R utan stopp, plan följd."""
import pathlib
from datetime import date

import pandas as pd

import journal_stats as js
from portfolio import mot_malet as mm

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_journal_colors_follow_the_goal():
    import trade_journal as tj
    assert tj._cagr_color(36) == tj._GREEN and tj._cagr_color(21) == tj._CYAN
    assert tj._cagr_color(16) == tj._GOLD and tj._cagr_color(-1) == tj._EMBER and tj._cagr_color(None) == tj._TEXT_DIM


def test_journal_cagr_uses_realized_cagr_to_today():
    import trade_journal as tj
    df = pd.DataFrame([{"entry_date": "2025-01-01", "exit_date": "2025-02-01", "pnl_amount": 7500.0, "pnl_pct": 5.0,
                        "exit_price": 10.0}])
    assert tj._cagr_of(df) == js.realized_cagr(df.to_dict("records"), tj._start_capital())


def test_nat_dates_are_ignored():
    rows = [{"entry_date": pd.NaT, "pnl_amount": 10.0}, {"entry_date": "2025-01-01", "pnl_amount": 7500.0}]
    assert js.realized_cagr(rows, 75000, until=date(2026, 1, 1)) is not None


def test_no_stop_and_no_r_is_empty_not_zero():
    src = (ROOT / "trade_journal.py").read_text(encoding="utf-8")
    assert "or (r_multiple if r_multiple else None)" in src and "round(_r, 2) if _r is not None else None" in src


def test_plan_followed_reads_ja_nej_and_bools():
    trades = [{"followed_plan": "Ja"}, {"followed_plan": "Nej"}, {"followed_plan": True}, {"followed_plan": None}, {}]
    assert mm.plan_followed(trades) == round(2 / 3 * 100, 1)


def test_journal_form_has_the_plan_field():
    src = (ROOT / "trade_journal.py").read_text(encoding="utf-8")
    assert '"Följde affären planen?", ["", "Ja", "Nej"]' in src and '"followed_plan":      followed_plan or None' in src
