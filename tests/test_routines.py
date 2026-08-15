"""
Tests for routines.py — Årshjulet (Masterguiden Del 6).

The table, the ritual and the journal thresholds are asserted against the
guide. The calendar anchors are this module's own construction (the guide
names the cadences but never fixes the dates), so those tests pin the
*behaviour the guide describes* rather than a date it states.
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import routines as rt


# ── Årshjulet ────────────────────────────────────────────────────────────────
def test_the_five_rows_of_the_wheel_match_the_guide():
    assert [r.when for r in rt.ROUTINES] == [
        "Varje söndag",
        "Första helgen i månaden",
        "Kvartalsvis (fast datum)",
        "Årligen",
        "Vid nytt kapital",
    ]
    assert [r.minutes for r in rt.ROUTINES] == [
        "45–60 min", "1–2 h", "2–3 h", "En kväll", "5 min"]


def test_every_routine_has_steps_and_tools():
    for r in rt.ROUTINES:
        assert r.steps, r.key
        assert r.tools, r.key
        assert r.title, r.key


def test_panel_paths_point_at_tabs_that_exist():
    """A routine that sends you to a tab we do not have is worse than none."""
    known = {"REGIME", "SCREENING", "GRANSKNING", "PORTFOLIO", "INTELLIGENCE",
             "RULES", "STRATEGIES", "HOME", "ALERTS"}
    for r in rt.ROUTINES:
        for s in r.steps:
            if not s.panel:
                continue
            assert s.panel.split(" → ")[0] in known, f"{r.key}: {s.panel}"


def test_new_capital_is_the_only_event_driven_routine():
    ev = [r.key for r in rt.ROUTINES if r.cadence == rt.ON_EVENT]
    assert ev == ["new_capital"]
    assert rt.next_due("new_capital", date(2026, 8, 14)) is None
    assert rt.days_until("new_capital", date(2026, 8, 14)) is None


# ── Kvartalsritualen ─────────────────────────────────────────────────────────
def test_quarterly_ritual_is_four_steps_that_sum_inside_the_stated_window():
    assert [s.number for s in rt.QUARTERLY_RITUAL] == [1, 2, 3, 4]
    assert [s.title for s in rt.QUARTERLY_RITUAL] == [
        "Siffrorna", "Beteendet", "Strategihälsan", "Ramarna"]
    assert [s.minutes for s in rt.QUARTERLY_RITUAL] == [20, 30, 20, 10]
    total = sum(s.minutes for s in rt.QUARTERLY_RITUAL)
    assert rt.RITUAL_TOTAL_MIN <= total <= rt.RITUAL_TOTAL_MAX


def test_the_behaviour_rules_the_ritual_exists_to_enforce():
    body = " ".join(s.body for s in rt.QUARTERLY_RITUAL)
    assert "ETT kvartal" in body            # one bad quarter changes nothing
    assert "4+ kvartal" in body             # four does
    assert "aldrig mitt i ett kvartal" in body


# ── Journalen ────────────────────────────────────────────────────────────────
def test_journal_thresholds_are_the_guides():
    assert rt.MIN_TRADES_FOR_STATS == 15
    joined = " ".join(rt.JOURNAL_RULES)
    assert "samma dag" in joined
    assert "R-multipel" in joined
    assert "15–20" in joined


def test_backtest_rules_keep_the_curve_fitting_warning():
    joined = " ".join(rt.BACKTEST_RULES)
    assert "platta berg" in joined
    assert "kurvanpassning" in joined
    assert "survivorship" in joined


def test_onboarding_is_the_ten_week_order():
    assert [p for p, _ in rt.ONBOARDING] == ["Vecka 1", "Vecka 2–9", "Vecka 10"]


# ── Kalenderlogiken ──────────────────────────────────────────────────────────
def test_weekly_lands_on_sundays_only():
    # 2026-08-16 is a Sunday, 2026-08-17 a Monday.
    assert rt.due_on("weekly", date(2026, 8, 16))
    assert not rt.due_on("weekly", date(2026, 8, 17))
    assert rt.days_until("weekly", date(2026, 8, 14)) == 2


def test_monthly_lands_on_the_first_weekend():
    assert rt.due_on("monthly", date(2026, 8, 1))    # Saturday the 1st
    assert rt.due_on("monthly", date(2026, 8, 2))    # Sunday the 2nd
    assert not rt.due_on("monthly", date(2026, 8, 8))   # second Saturday
    assert not rt.due_on("monthly", date(2026, 8, 3))   # first Monday


def test_quarterly_only_in_quarter_months():
    assert rt.QUARTER_MONTHS == (1, 4, 7, 10)
    assert rt.due_on("quarterly", date(2026, 10, 3))    # first Sat of October
    assert not rt.due_on("quarterly", date(2026, 8, 1))  # August is not one
    # A quarterly date is always also a monthly date — you never do the
    # portfolio review without the rotation being due too.
    for d in (date(2026, 1, 3), date(2026, 4, 4), date(2026, 7, 4),
              date(2026, 10, 3)):
        assert rt.due_on("quarterly", d)
        assert rt.due_on("monthly", d)


def test_yearly_is_january_only():
    assert rt.due_on("yearly", date(2027, 1, 2))
    assert not rt.due_on("yearly", date(2026, 4, 4))
    # And a yearly date is a quarterly date.
    assert rt.due_on("quarterly", date(2027, 1, 2))


def test_next_due_counts_today_and_never_looks_backwards():
    sunday = date(2026, 8, 16)
    assert rt.next_due("weekly", sunday) == sunday
    assert rt.next_due("weekly", date(2026, 8, 17)) == date(2026, 8, 23)
    for key in ("weekly", "monthly", "quarterly", "yearly"):
        nxt = rt.next_due(key, date(2026, 8, 14))
        assert nxt is not None and nxt >= date(2026, 8, 14), key


def test_status_flags_the_week_ahead():
    assert rt.status("weekly", date(2026, 8, 16)) == rt.DUE
    assert rt.status("weekly", date(2026, 8, 14)) == rt.SOON
    assert rt.status("yearly", date(2026, 8, 14)) == rt.VILANDE
    assert rt.status("new_capital", date(2026, 8, 14)) == rt.VILANDE
    assert rt.SOON_DAYS == 7


def test_agenda_covers_every_routine_and_sorts_soonest_first():
    rows = rt.agenda(date(2026, 8, 14))
    assert len(rows) == len(rt.ROUTINES)
    left = [r["days_until"] for r in rows if r["days_until"] is not None]
    assert left == sorted(left)
    assert rows[-1]["routine"].key == "new_capital"   # event-driven last


def test_due_today_returns_what_to_actually_do():
    # First Sunday of October 2026 = 4 Oct: weekly + monthly + quarterly.
    keys = [r.key for r in rt.due_today(date(2026, 10, 4))]
    assert keys == ["weekly", "monthly", "quarterly"]
    # A random Tuesday: nothing.
    assert rt.due_today(date(2026, 8, 18)) == []


def test_unknown_key_is_never_due():
    assert not rt.due_on("nonsense", date(2026, 8, 16))
    assert rt.next_due("nonsense", date(2026, 8, 16)) is None
    assert rt.status("nonsense", date(2026, 8, 16)) == rt.VILANDE


# ── Svenska datum ────────────────────────────────────────────────────────────
def test_dates_render_in_swedish_regardless_of_server_locale():
    d = date(2026, 10, 4)
    assert rt.fmt_date(d) == "4 oktober"
    assert rt.fmt_date(d, with_year=True) == "4 oktober 2026"
    assert rt.fmt_weekday(d) == "söndag 4 oktober"
    assert len(rt.MONTHS_SV) == 12 and len(rt.WEEKDAYS_SV) == 7
