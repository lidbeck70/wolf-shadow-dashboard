"""
Tester för levels.py — de räknade entry- och exitnivåerna.

Det som testas hårdast är gränsfallen där en nivå skulle kunna se rimlig ut
utan att vara det: en stop ovanför entry, en R:R som blir 0 i stället för
"går inte att räkna", och riktningen på targeten.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import levels as lv


class _Snap:
    """Minimal ögonblicksbild — bara fälten levels läser."""
    def __init__(self, atr14=None, swing_low_20=None, swing_high_20=None,
                 ema50=None):
        self.atr14 = atr14
        self.swing_low_20 = swing_low_20
        self.swing_high_20 = swing_high_20
        self.ema50 = ema50


# ── Grundräkningen ───────────────────────────────────────────────────────────
def test_risk_and_rr():
    assert lv.risk_pct(100, 90) == 10.0
    assert lv.rr(100, 90, 130) == 3.0
    assert lv.rr(100, 90, 120) == 2.0


def test_rr_is_none_when_it_cannot_be_computed():
    """None, inte 0 — noll läses som 'dåligt R:R', inte som 'ofullständigt'."""
    assert lv.rr(100, 100, 130) is None      # stop på entry
    assert lv.rr(100, 90, 0) is None         # target saknas
    assert lv.rr(0, 90, 130) is None
    assert lv.rr(None, None, None) is None


def test_target_for_rr_follows_the_direction_of_the_stop():
    assert lv.target_for_rr(100, 90, 2.0) == 120.0     # lång
    assert lv.target_for_rr(100, 110, 2.0) == 80.0     # kort
    assert lv.target_for_rr(100, 100, 2.0) is None


def test_target_for_rr_is_the_inverse_of_rr():
    t = lv.target_for_rr(100, 92, 2.5)
    assert lv.rr(100, 92, t) == 2.5


# ── Stoppkandidaterna ────────────────────────────────────────────────────────
def test_without_a_snapshot_only_the_fixed_percent_survives():
    out = lv.stop_candidates(100, None, fixed_pct=10)
    assert [s.name for s in out] == [lv.PCT_STOP]
    assert out[0].price == 90.0
    # och utan ens den finns ingenting att hitta på
    assert lv.stop_candidates(100, None) == []


def test_all_four_levels_when_the_data_is_there():
    snap = _Snap(atr14=3.0, swing_low_20=94.0, ema50=96.0)
    out = lv.stop_candidates(100, snap, fixed_pct=10)
    assert len(out) == 4
    # dyraste risk först
    assert [round(s.risk_pct, 1) for s in out] == sorted(
        [round(s.risk_pct, 1) for s in out], reverse=True)
    assert out[0].name == lv.PCT_STOP and out[0].risk_pct == 10.0


def test_the_atr_multiple_comes_from_the_strategy():
    """Viking 1,5× och Wolf 2,5× ska ge olika nivåer, inte samma default."""
    snap = _Snap(atr14=4.0)
    viking = lv.stop_candidates(100, snap, atr_mult=1.5)[0]
    wolf = lv.stop_candidates(100, snap, atr_mult=2.5)[0]
    assert viking.price == 94.0 and wolf.price == 90.0
    assert "1.5×" in viking.name and "2.5×" in wolf.name
    # utan multipel gäller modulens standard
    assert lv.stop_candidates(100, snap)[0].price == 100 - lv.ATR_MULT * 4.0


def test_a_stop_above_entry_is_not_a_stop():
    """EMA50 över kursen får inte bli en 'stop' som ligger över entry."""
    snap = _Snap(ema50=120.0, swing_low_20=115.0)
    assert lv.stop_candidates(100, snap) == []


def test_each_level_carries_the_targets_it_demands():
    snap = _Snap(atr14=2.0)
    s = lv.stop_candidates(100, snap, atr_mult=2.0)[0]
    assert s.price == 96.0
    assert s.target_for_min_rr == 108.0        # 2:1 på 4 kr risk
    assert s.target_for_pref_rr == 112.0       # 3:1


# ── Bedömningen ──────────────────────────────────────────────────────────────
def test_a_thin_rr_says_what_target_would_fix_it():
    a = lv.assess(100, 90, 110)
    assert a.rr == 1.0 and a.meets_min is False
    assert any("120" in n for n in a.notes)


def test_meeting_the_minimum_but_not_the_preferred_is_said_out_loud():
    a = lv.assess(100, 90, 120)
    assert a.meets_min is True and a.meets_preferred is False
    assert any("3" in n for n in a.notes)


def test_a_clean_setup_has_no_notes():
    a = lv.assess(100, 90, 130)
    assert a.meets_preferred is True
    assert a.notes == ()


def test_exactly_two_to_one_passes():
    """Flyttalsgränsen igen — 2,0 ska räknas som godkänt."""
    a = lv.assess(100, 92.5, 115)
    assert a.rr == 2.0
    assert a.meets_min is True


def test_a_stop_inside_the_daily_noise_is_flagged():
    """Den vanligaste dyra nybörjarmissen: stoppen sitter inom ATR."""
    snap = _Snap(atr14=5.0)
    a = lv.assess(100, 97, 130, snap)          # 3 kr = 0,6 ATR
    assert any("ATR" in n and "dagsbrus" in n for n in a.notes)


def test_a_stop_above_the_swing_low_is_flagged():
    snap = _Snap(atr14=1.0, swing_low_20=90.0)
    a = lv.assess(100, 95, 130, snap)
    assert any("swing-low" in n for n in a.notes)


def test_a_target_under_the_recent_high_is_flagged():
    snap = _Snap(atr14=1.0, swing_high_20=140.0)
    a = lv.assess(100, 90, 120, snap)
    assert any("swing" in n.lower() or "högsta" in n for n in a.notes)


def test_assess_survives_missing_input():
    a = lv.assess(0, 0, 0)
    assert a.rr is None and a.meets_min is False
    assert a.notes            # och säger varför
