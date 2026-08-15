"""
Tests for rotation.py — Råvarurotationen med Triple Signal (Masterguiden 4.0).

The master table is asserted against the guide. The Triple Signal thresholds
are the 4.0 spec's (13–15 AGERA, 10–12 Bevaka, <= 9 Vila), unlike the 3.x
priority formula they replaced, which was this module's own construction.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rotation as r


def _g(hatred=1, fundamentals=1, catalyst=1, case_intact=True, **kw) -> dict:
    g = {"hatred": hatred, "fundamentals": fundamentals, "catalyst": catalyst,
         "case_intact": case_intact}
    g.update(kw)
    return g


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


# ── Triple Signal ────────────────────────────────────────────────────────────
def test_the_three_axes_and_their_range():
    assert [k for k, _l, _h in r.SIGNALS] == ["hatred", "fundamentals",
                                              "catalyst"]
    assert (r.SIGNAL_MIN, r.SIGNAL_MAX) == (1, 5)
    assert (r.SUM_MIN, r.SUM_MAX) == (3, 15)


def test_signal_sum():
    assert r.signal_sum(_g(5, 5, 5)) == 15
    assert r.signal_sum(_g(1, 1, 1)) == 3
    assert r.signal_sum(_g(5, 4, 4)) == 13


def test_axes_clamp_and_default_to_one_not_zero():
    """An ungraded axis is the weakest reading, not a missing one."""
    assert r.signal_sum({}) == 3
    assert r.signal_sum(_g(99, 99, 99)) == 15
    assert r.signal_sum(_g(-4, 0, None)) == 3
    assert r.signal_sum(_g("abc", "", "x")) == 3


def test_status_thresholds_are_the_specs():
    assert r.status(_g(5, 5, 5))[0] == r.AGERA          # 15
    assert r.status(_g(5, 4, 4))[0] == r.AGERA          # 13
    assert r.status(_g(4, 4, 4))[0] == r.BEVAKA         # 12
    assert r.status(_g(4, 3, 3))[0] == r.BEVAKA         # 10
    assert r.status(_g(3, 3, 3))[0] == r.VILA           # 9
    assert r.status(_g(1, 1, 1))[0] == r.VILA           # 3
    assert (r.AGERA_MIN, r.BEVAKA_MIN) == (13, 10)


def test_broken_case_can_never_be_agera():
    """The hard gate sits before the sum, not inside it."""
    st, why = r.status(_g(5, 5, 5, case_intact=False))
    assert st == r.VILA
    assert "brutet" in why
    assert r.priority(_g(5, 5, 5, case_intact=False)) == 0.0
    assert r.priority(_g(5, 5, 5)) == 15.0


def test_a_case_defaults_to_intact_when_unset():
    assert r.status({"hatred": 5, "fundamentals": 5, "catalyst": 5})[0] == r.AGERA


# ── Varningsbadges ───────────────────────────────────────────────────────────
def test_high_hatred_without_a_case_is_a_value_trap():
    """The whole reason for splitting hatred from fundamentals."""
    assert r.warnings(_g(5, 2, 3)) == [r.WARN_VALUE_TRAP]
    assert r.warnings(_g(4, 1, 1)) == [r.WARN_VALUE_TRAP]
    assert r.warnings(_g(3, 2, 3)) == []          # hat under 4
    assert r.warnings(_g(5, 3, 3)) == []          # fundamenta över 2


def test_strong_case_without_hatred_is_not_a_contrarian_buy():
    assert r.warnings(_g(2, 5, 3)) == [r.WARN_NOT_CONTRARIAN]
    assert r.warnings(_g(1, 4, 1)) == [r.WARN_NOT_CONTRARIAN]
    assert r.warnings(_g(3, 5, 3)) == []          # hat över 2


def test_the_two_warnings_are_mutually_exclusive():
    """They are opposite failure modes — both at once would be a bug."""
    for h in range(1, 6):
        for f in range(1, 6):
            assert len(r.warnings(_g(h, f, 3))) <= 1, (h, f)


def test_a_warning_does_not_block_the_status():
    """It informs the judgement; it does not overrule the sum."""
    trap = _g(5, 2, 5)                     # summa 12
    assert r.warnings(trap) == [r.WARN_VALUE_TRAP]
    assert r.status(trap)[0] == r.BEVAKA


# ── Migreringen från 3.x ─────────────────────────────────────────────────────
def test_migration_maps_hat_and_timing_onto_the_new_axes():
    old = {"hat": 5, "timing": r.TIMING_YES, "case_intact": True}
    new = r.migrate_grade(old)
    assert new["hatred"] == 5
    assert new["catalyst"] == 3            # Ja -> 3
    assert new["fundamentals"] == r.LEGACY_FUNDAMENTALS_DEFAULT == 3
    assert new["case_intact"] is True
    assert new["migrated"] is True


def test_migration_maps_every_timing_value():
    assert r.migrate_grade({"hat": 3, "timing": r.TIMING_YES})["catalyst"] == 3
    assert r.migrate_grade({"hat": 3, "timing": r.TIMING_PARTLY})["catalyst"] == 2
    assert r.migrate_grade({"hat": 3, "timing": r.TIMING_NO})["catalyst"] == 1
    assert r.migrate_grade({"hat": 3, "timing": "skräp"})["catalyst"] == 1


def test_migration_is_idempotent_and_leaves_new_grades_alone():
    new = _g(5, 4, 3)
    assert r.migrate_grade(new) == new
    assert "migrated" not in r.migrate_grade(new)
    once = r.migrate_grade({"hat": 4, "timing": r.TIMING_PARTLY})
    assert r.migrate_grade(once) == once


def test_migration_survives_empty_and_unknown_rows():
    assert r.migrate_grade({}) == {}
    assert r.migrate_grade(None) == {}
    assert r.migrate_grades({}) == {}
    assert r.migrate_grades(None) == {}


def test_a_migrated_row_is_flagged_so_the_placeholder_is_visible():
    """fundamentals=3 is a placeholder — the old model never asked."""
    rows = r.ranked({"uran": {"hat": 5, "timing": r.TIMING_YES}})
    uran = [x for x in rows if x["commodity"].key == "uran"][0]
    assert uran["migrated"] is True
    assert uran["sum"] == 5 + 3 + 3 == 11
    assert uran["status"] == r.BEVAKA


# ── Kapitalallokeringen ──────────────────────────────────────────────────────
def _grades() -> dict:
    return {
        "uran":   _g(5, 5, 5),                       # 15 AGERA
        "kol":    _g(5, 5, 3),                       # 13 AGERA
        "litium": _g(5, 5, 5, case_intact=False),    # brutet -> Vila
        "olja":   _g(4, 4, 4),                       # 12 Bevaka
        "koppar": _g(5, 4, 5),                       # 14 AGERA
    }


def test_capital_goes_to_the_highest_scoring_with_intact_cases():
    targets = [t["commodity"].name for t in r.capital_targets(_grades())]
    assert targets == ["Uran", "Koppar", "Kol"]
    assert "Litium" not in targets        # brutet case trots 15
    assert "Olja" not in targets          # bara Bevaka


def test_capital_is_limited_to_two_or_three_slots():
    assert r.CAPITAL_SLOTS == 3
    g = {c.key: _g(5, 5, 5) for c in r.COMMODITIES}
    assert len(r.capital_targets(g)) == 3
    assert len(r.capital_targets(g, slots=2)) == 2


def test_ranking_covers_every_commodity_and_sorts_by_priority():
    rows = r.ranked(_grades())
    assert len(rows) == len(r.COMMODITIES)
    prios = [row["priority"] for row in rows]
    assert prios == sorted(prios, reverse=True)


def test_ranking_exposes_every_axis_for_the_ui_and_the_export():
    row = r.ranked({"uran": _g(5, 4, 3, screener_hits=42)})[0]
    assert row["commodity"].key == "uran"
    assert (row["hatred"], row["fundamentals"], row["catalyst"]) == (5, 4, 3)
    assert row["sum"] == 12
    assert row["screener_hits"] == 42


def test_ranking_survives_empty_and_malformed_grades():
    assert len(r.ranked({})) == len(r.COMMODITIES)
    assert all(row["status"] == r.VILA for row in r.ranked({}))
    bad = {"uran": {"hatred": "x", "fundamentals": None, "case_intact": None}}
    rows = r.ranked(bad)
    assert len(rows) == len(r.COMMODITIES)
    assert all(row["sum"] >= r.SUM_MIN for row in rows)


def test_no_agera_is_a_valid_state():
    """Nothing hated enough is a legitimate outcome, not a bug."""
    quiet = {c.key: _g(1, 1, 1) for c in r.COMMODITIES}
    assert r.capital_targets(quiet) == []


def test_the_hatred_checklist_is_five_statements():
    """It is how the hatred axis is set — it did not become obsolete."""
    assert len(r.HATRED_CHECKLIST) == 5
    joined = " ".join(t for _k, t in r.HATRED_CHECKLIST).lower()
    for word in ("incitament", "utbudet", "kapitalet", "screener", "media"):
        assert word in joined
    assert len({k for k, _t in r.HATRED_CHECKLIST}) == 5      # unika nycklar


def test_hatred_is_counted_not_judged():
    """Guiden 4.0: "Antal Ja ger poängen"."""
    keys = [k for k, _t in r.HATRED_CHECKLIST]
    assert r.hatred_from_checklist({k: True for k in keys}) == 5
    assert r.hatred_from_checklist({keys[0]: True, keys[1]: True}) == 2
    assert r.hatred_from_checklist({keys[0]: True}) == 1


def test_an_unticked_checklist_is_the_bottom_of_the_scale_not_zero():
    """The axes are 1–5, so no ticks is 1 — there is no zero reading."""
    assert r.hatred_from_checklist({}) == r.SIGNAL_MIN == 1
    assert r.hatred_from_checklist(None) == 1
    assert r.hatred_from_checklist({"finns_inte": True}) == 1
