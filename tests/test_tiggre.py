"""
Tests for tiggre.py — the Lobo sheet as a panel tab.

The numbers are checked against Masterguiden's own worked examples, so the panel
can't quietly disagree with the guide it implements.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tiggre as t


# ── The guide's worked examples ──────────────────────────────────────────────
def test_masterguide_example_upside_and_pnav():
    """MCap $200M, NAV $650M -> 0,31× NAV -> uppsida +160 %."""
    assert round(t.p_nav(200, 650), 2) == 0.31
    assert round(t.upside_pct(200, 650)) == 160


def test_masterguide_example_un_ratios():
    """160/40 = 4:1 -> godkänt. Ofinansierat 85/50 = 1,7:1 -> vänta."""
    assert round(t.un_ratio(160, -40), 1) == 4.0
    assert round(t.un_ratio(85, -50), 1) == 1.7
    # The gate itself
    assert t.un_ratio(160, -40) >= t.UN_MIN
    assert t.un_ratio(85, -50) < t.UN_MIN


def test_downside_sign_does_not_matter():
    """The downside is an estimate; -40 and 40 mean the same magnitude."""
    assert t.un_ratio(160, -40) == t.un_ratio(160, 40)


def test_upside_uses_the_08_nav_target():
    # Priced at NAV already -> upside to 0.8x is negative
    assert t.upside_pct(650, 650) < 0
    # Deep in the buy zone (0.2x) -> large upside
    assert t.upside_pct(130, 650) > 200


# ── Guards ───────────────────────────────────────────────────────────────────
def test_math_survives_missing_and_zero_input():
    for bad in (None, 0, "", "abc"):
        assert t.upside_pct(bad, 650) is None
        assert t.upside_pct(200, bad) is None
        assert t.p_nav(200, bad) is None
    assert t.un_ratio(None, -40) is None
    assert t.un_ratio(160, None) is None
    assert t.un_ratio(160, 0) is None          # no division by zero


def test_factor_score_clamps_to_zero_two():
    full = {k: 2 for k, _l, _h in t.FACTORS}
    assert t.factor_score(full) == 10
    assert t.factor_score({}) == 0
    # out-of-range values are clamped, not trusted
    assert t.factor_score({k: 9 for k, _l, _h in t.FACTORS}) == 10
    assert t.factor_score({k: -5 for k, _l, _h in t.FACTORS}) == 0
    assert t.factor_score({k: None for k, _l, _h in t.FACTORS}) == 0


def test_screen_hits_counts_the_three_phrases():
    assert t.screen_hits({}) == 0
    assert t.screen_hits({"fs": True, "permits": True}) == 2
    assert t.screen_hits({"fs": True, "permits": True, "funded": True}) == 3


# ── The hard gates ───────────────────────────────────────────────────────────
def _good_candidate() -> dict:
    return {
        "screen": {"fs": True, "permits": True, "funded": True},
        "mcap": 200, "nav": 650, "downside": -40,
        "factors": {k: 2 for k, _l, _h in t.FACTORS},
        "catalysts": [{"name": "Finansieringsbesked", "date": "2026-03"},
                      {"name": "Byggstart", "date": "2026-09"}],
    }


def test_all_gates_pass_for_a_complete_case():
    gates = t.buy_gates(_good_candidate())
    assert len(gates) == 4
    assert all(passed for _label, passed, _detail in gates), gates


def test_each_gate_can_fail_on_its_own():
    # too few phrases
    c = _good_candidate(); c["screen"] = {"fs": True}
    assert not t.buy_gates(c)[0][1]
    # U/N below 3
    c = _good_candidate(); c["downside"] = -200
    assert not t.buy_gates(c)[1][1]
    # score below 8
    c = _good_candidate(); c["factors"] = {"stadium": 1}
    assert not t.buy_gates(c)[2][1]
    # only one catalyst
    c = _good_candidate(); c["catalysts"] = [{"name": "FID", "date": "2026-03"}]
    assert not t.buy_gates(c)[3][1]


def test_catalyst_must_be_named_and_dated():
    """An unnamed or undated catalyst is a hope, not a speculation."""
    c = _good_candidate()
    c["catalysts"] = [{"name": "Byggstart", "date": ""},
                      {"name": "", "date": "2026-09"}]
    assert not t.buy_gates(c)[3][1]


# ── Free ride ────────────────────────────────────────────────────────────────
def test_free_ride_triggers_at_plus_100_percent():
    assert t.free_ride_reached(10, 20) is True
    assert t.free_ride_reached(10, 25) is True
    assert t.free_ride_reached(10, 19.99) is False
    assert t.free_ride_reached(0, 20) is False       # no entry, no signal


def test_equity_at_risk_is_zero_after_the_free_ride():
    # before selling half: the stake is still at risk
    assert t.equity_at_risk(10, 20, 100, half_sold=False) == 1000
    # after selling half at +100 %: stake recovered, house money left
    assert t.equity_at_risk(10, 20, 100, half_sold=True) == 0.0
    # ticking the box below +100 % must NOT zero the risk
    assert t.equity_at_risk(10, 15, 100, half_sold=True) == 1000


# ── Requirements match the playbook ──────────────────────────────────────────
def test_thresholds_match_the_documented_playbook():
    import strategy_rules as sr

    pb = sr.PLAYBOOKS["tiggre"]
    joined = " ".join(r.text + r.explanation for r in pb.entry + pb.exit)
    assert f"≥ {t.UN_MIN:g}" in joined or f">= {t.UN_MIN:g}" in joined
    assert f"≥ {t.SCORE_MIN}" in joined
    assert f"{t.NAV_TARGET:g}× NAV".replace(".", ",") in joined
    assert f"+{t.FREE_RIDE_PCT:g} %" in joined
    assert "4–6 bolag" in pb.risk.max_positions
    assert t.MAX_POSITIONS == 6
