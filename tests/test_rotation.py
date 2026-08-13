"""
Tests for rotation.py — Råvarurotationen (Masterguiden Del 3).

The master table is asserted against the guide. The priority formula is this
module's own construction (the guide never publishes it), so those tests pin
the *behaviour the guide describes* rather than a number it states.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rotation as r


# ── Master-tabellen ──────────────────────────────────────────────────────────
def test_master_table_matches_the_guide():
    names = [c.name for c in r.COMMODITIES]
    assert names == ["Guld", "Silver", "Platina", "Palladium", "Uran", "Olja",
                     "Gas", "Kol", "Koppar", "Zink", "Järnmalm", "Litium",
                     "Royalty"]
    assert len(r.COMMODITIES) == r.DOCUMENTED_COUNT


def test_every_commodity_has_an_engine_and_a_buy_signal():
    for c in r.COMMODITIES:
        assert c.engine, c.name
        assert c.buy_signal, c.name


def test_gold_and_royalty_are_the_anchors():
    """They stay put regardless of grade — gold rises in risk aversion."""
    assert [c.name for c in r.anchors()] == ["Guld", "Royalty"]


def test_the_guide_count_discrepancy_is_recorded_not_invented():
    """The text says 14, the table lists 13. Flagged, not padded."""
    assert r.GUIDE_CLAIMS == 14
    assert r.DOCUMENTED_COUNT == 13
    assert len(r.COMMODITIES) == r.DOCUMENTED_COUNT


def test_known_buy_signals_are_verbatim():
    by = {c.key: c.buy_signal for c in r.COMMODITIES}
    assert "85–90" in by["silver"]            # guld/silver-kvot
    assert "$80–90/lb" in by["uran"]
    assert "20 %" in by["kol"]                # FCF-yield
    assert "$4,5/lb" in by["koppar"]


# ── Statuslogiken ────────────────────────────────────────────────────────────
def test_broken_case_can_never_be_agera():
    """The guide: capital goes to the most hated with INTACT cases."""
    st, why = r.status(5, r.TIMING_YES, case_intact=False)
    assert st == r.VILA
    assert "brutet" in why
    assert r.priority(5, r.TIMING_YES, case_intact=False) == 0.0


def test_agera_requires_hate_and_some_timing():
    assert r.status(5, r.TIMING_YES, True)[0] == r.AGERA
    assert r.status(4, r.TIMING_PARTLY, True)[0] == r.AGERA
    # hated but no timing yet -> wait for the signal
    assert r.status(5, r.TIMING_NO, True)[0] == r.BEVAKA
    # timing but not hated enough
    assert r.status(2, r.TIMING_YES, True)[0] == r.BEVAKA


def test_vila_when_neither_hated_nor_timed():
    assert r.status(1, r.TIMING_NO, True)[0] == r.VILA
    assert r.status(0, r.TIMING_NO, True)[0] == r.VILA


def test_priority_orders_by_hate_then_timing():
    assert r.priority(5, r.TIMING_YES, True) > r.priority(5, r.TIMING_PARTLY, True)
    assert r.priority(5, r.TIMING_NO, True) > r.priority(4, r.TIMING_NO, True)
    assert r.priority(3, r.TIMING_YES, True) == r.priority(5, r.TIMING_NO, True)


def test_priority_clamps_out_of_range_hate():
    assert r.priority(99, r.TIMING_NO, True) == 5.0
    assert r.priority(-3, r.TIMING_NO, True) == 1.0
    assert r.priority(None, r.TIMING_NO, True) == 0.0
    assert r.priority("abc", r.TIMING_NO, True) == 0.0


# ── Kapitalallokeringen ──────────────────────────────────────────────────────
def _grades() -> dict:
    return {
        "uran":   {"hat": 5, "timing": r.TIMING_YES, "case_intact": True},
        "kol":    {"hat": 5, "timing": r.TIMING_PARTLY, "case_intact": True},
        "litium": {"hat": 5, "timing": r.TIMING_NO, "case_intact": True},
        "olja":   {"hat": 4, "timing": r.TIMING_YES, "case_intact": False},
        "koppar": {"hat": 3, "timing": r.TIMING_YES, "case_intact": True},
    }


def test_capital_goes_to_the_most_hated_with_intact_cases():
    targets = [t["commodity"].name for t in r.capital_targets(_grades())]
    assert targets == ["Uran", "Kol"]
    assert "Olja" not in targets          # broken case, despite hat 4 + timing Ja
    assert "Litium" not in targets        # hated but no timing


def test_capital_is_limited_to_two_or_three_slots():
    assert r.CAPITAL_SLOTS == 3
    g = {c.key: {"hat": 5, "timing": r.TIMING_YES, "case_intact": True}
         for c in r.COMMODITIES}
    assert len(r.capital_targets(g)) == 3
    assert len(r.capital_targets(g, slots=2)) == 2


def test_ranking_covers_every_commodity_and_sorts_by_priority():
    rows = r.ranked(_grades())
    assert len(rows) == len(r.COMMODITIES)
    prios = [row["priority"] for row in rows]
    assert prios == sorted(prios, reverse=True)


def test_ranking_survives_empty_and_malformed_grades():
    assert len(r.ranked({})) == len(r.COMMODITIES)
    assert all(row["status"] == r.VILA for row in r.ranked({}))
    bad = {"uran": {"hat": "x", "timing": "???", "case_intact": None}}
    rows = r.ranked(bad)
    assert len(rows) == len(r.COMMODITIES)


def test_no_agera_is_a_valid_state():
    """Nothing hated enough is a legitimate outcome, not a bug."""
    quiet = {c.key: {"hat": 1, "timing": r.TIMING_NO, "case_intact": True}
             for c in r.COMMODITIES}
    assert r.capital_targets(quiet) == []
