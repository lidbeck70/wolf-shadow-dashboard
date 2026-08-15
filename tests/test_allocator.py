"""
Tests for allocator.py — Portföljallokeraren (Masterguiden Del 2).

This is the layer that turns every strategy's "max X %" from text into a number
the panel checks, so the figures are asserted against the guide directly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import allocator as a


def _at_target() -> dict:
    """Portfolio sitting exactly on the model weights (SEK, 1M total)."""
    return {s.key: s.target * 10_000 for s in a.SLEEVES}


# ── Fördelningsmodellen ──────────────────────────────────────────────────────
def test_targets_sum_to_one_hundred():
    assert sum(s.target for s in a.SLEEVES) == 100


def test_every_target_sits_inside_its_own_range():
    for s in a.SLEEVES:
        assert s.lo <= s.target <= s.hi, s.name


def test_sleeve_pct_and_status_at_target():
    pcts = a.sleeve_pct(_at_target())
    for s in a.SLEEVES:
        assert round(pcts[s.key], 1) == s.target
        assert a.sleeve_status(s.key, pcts[s.key])[0] == "inom ram"


def test_status_flags_outside_the_range():
    assert a.sleeve_status("swing", 5)[0] == "under ram"      # ram 10–30
    assert a.sleeve_status("swing", 35)[0] == "över ram"
    assert a.sleeve_status("swing", 20)[0] == "inom ram"


def test_sleeve_pct_survives_empty_and_bad_input():
    assert all(v == 0.0 for v in a.sleeve_pct({}).values())
    bad = {s.key: None for s in a.SLEEVES}
    assert all(v == 0.0 for v in a.sleeve_pct(bad).values())
    assert all(v == 0.0 for v in a.sleeve_pct({"swing": "abc"}).values())


# ── Råvarutaket ──────────────────────────────────────────────────────────────
def test_commodity_cap_is_55_and_target_weights_fit_under_it():
    assert a.COMMODITY_CAP == 55.0
    # royalty 20 + producenter 15 + optionalitet 7 + durrett 8 = 50
    assert round(a.commodity_exposure(_at_target())) == 50
    assert not a.commodity_breach(_at_target())


def test_commodity_cap_breaches_when_the_sleeves_grow():
    v = _at_target()
    v["royalty"] *= 3
    assert a.commodity_breach(v)


def test_only_the_four_commodity_sleeves_count():
    commodity = {s.key for s in a.SLEEVES if s.commodity}
    assert commodity == {"royalty", "producenter", "optionalitet", "durrett"}


# ── Kassaregeln ──────────────────────────────────────────────────────────────
def test_cash_rule_thresholds():
    assert a.cash_rule(3)[0] == "låg"            # < 5 %
    assert a.cash_rule(10)[0] == "ok"
    assert a.cash_rule(30)[0] == "hög"           # > 25 %
    # the "more than one quarter" wording only appears once it has persisted
    assert "sänk medvetet ribban" in a.cash_rule(30, quarters_high=1)[1]
    assert "sänk medvetet ribban" not in a.cash_rule(30, quarters_high=0)[1]


# ── Strömbrytaren ────────────────────────────────────────────────────────────
def test_breaker_levels_match_the_guide():
    assert a.breaker_state(0)[0] == "NORMAL"
    assert a.breaker_state(9.9)[0] == "NORMAL"
    assert a.breaker_state(10)[0] == "SKÄRPT"
    assert a.breaker_state(19.9)[0] == "SKÄRPT"
    assert a.breaker_state(20)[0] == "HALVERAD RISK"
    assert a.breaker_state(60)[0] == "HALVERAD RISK"
    assert a.breaker_state(None)[0] == "OKÄND"


def test_breaker_protects_cash_and_the_core():
    """At the deepest level the guide protects the stabiliser, not the tail."""
    action = a.breaker_state(25)[2]
    assert "Royaltykärna" in action and "kassa" in action
    assert "Halvera" in action


def test_exact_twenty_percent_fall_reaches_the_deepest_level():
    """Binary floating point makes an exact 20 % fall compute as 19.9999…,
    which would leave the portfolio one level too calm."""
    assert a.drawdown_pct(100, 80) == 20.0
    assert a.breaker_state(a.drawdown_pct(100, 80))[0] == "HALVERAD RISK"
    assert a.breaker_state(a.drawdown_pct(1_000_000, 800_000))[0] == "HALVERAD RISK"


def test_drawdown_maths():
    assert a.drawdown_pct(100, 80) == 20.0
    assert a.drawdown_pct(100, 100) == 0.0
    assert a.drawdown_pct(100, 120) == 0.0       # above peak is not a drawdown
    assert a.drawdown_pct(0, 50) is None
    assert a.drawdown_pct(None, 50) is None


# ── Nytt kapital ─────────────────────────────────────────────────────────────
def test_new_capital_goes_to_the_sleeve_furthest_below_target():
    v = _at_target()
    v["swing"] = 0                                # 20 % target, now 0
    assert a.next_capital_target(v).key == "swing"


def test_new_capital_never_targets_cash():
    v = {s.key: 0.0 for s in a.SLEEVES}
    v["kassa"] = 1_000_000                        # everything in cash
    assert a.next_capital_target(v).key != "kassa"


def test_new_capital_returns_none_when_everything_is_at_or_above_target():
    v = {s.key: 0.0 for s in a.SLEEVES}
    v["royalty"] = 1_000_000
    # every other sleeve is 0 % and below target, so there IS a target
    assert a.next_capital_target(v) is not None


# ── Positionstak ─────────────────────────────────────────────────────────────
def test_position_caps_match_the_guide():
    caps = {s.key: s.position_cap for s in a.SLEEVES}
    assert caps["royalty"] == 10.0
    assert caps["swing"] == 6.0
    assert caps["producenter"] == 4.0      # Rule — the cap wins over 5–10 %
    assert caps["insider"] == 4.0
    assert caps["durrett"] == 3.0
    assert caps["optionalitet"] == 4.0     # Tiggre 2–4 %, Sprott 1,5 %
    assert caps["kassa"] is None


def test_position_breach_detected():
    breaches = a.position_breaches(
        [{"ticker": "X", "sleeve": "durrett", "value": 50_000}], 1_000_000)
    assert len(breaches) == 1
    assert breaches[0]["pct"] == 5.0 and breaches[0]["cap"] == 3.0


def test_position_within_cap_is_not_flagged():
    assert a.position_breaches(
        [{"ticker": "X", "sleeve": "durrett", "value": 20_000}], 1_000_000) == []


def test_royalty_core_may_drift_to_12_percent():
    """The guide's explicit exception: the core is trimmed only past 12 %."""
    assert a.ROYALTY_DRIFT_CAP == 12.0
    at_11 = a.position_breaches(
        [{"ticker": "R", "sleeve": "royalty", "value": 110_000}], 1_000_000)
    assert at_11 == []
    at_13 = a.position_breaches(
        [{"ticker": "R", "sleeve": "royalty", "value": 130_000}], 1_000_000)
    assert len(at_13) == 1


def test_breaches_survive_missing_values():
    assert a.position_breaches([{"ticker": "X"}], 1_000_000) == []
    assert a.position_breaches([{"ticker": "X", "sleeve": "durrett"}], 0) == []


# ── Mot migrationsspecens varningsband ───────────────────────────────────────
# Specen §1: "flagga 'FÖR STOR — trimma' om andel > tak (varning vid > 90 % av
# tak)" och "Råvarutak ... <= 55 % av total (varning från 50 %)".
def test_position_state_warns_before_the_cap_is_broken():
    assert a.position_state(3.5, 4.0) == a.POS_OK      # 87,5 % av taket
    assert a.position_state(3.6, 4.0) == a.POS_NEAR    # 90 % — förvarning
    assert a.position_state(4.0, 4.0) == a.POS_NEAR    # exakt på taket
    assert a.position_state(4.1, 4.0) == a.POS_OVER
    assert a.POSITION_WARN_FRAC == 0.9


def test_position_state_handles_missing_caps():
    assert a.position_state(50.0, None) == a.POS_OK    # kassan har inget tak
    assert a.position_state(50.0, 0) == a.POS_OK
    assert a.position_state(None, 4.0) == a.POS_OK


def test_commodity_state_warns_from_fifty_percent():
    def vals(commodity_pct):
        """Portfölj där råvarubenen tillsammans utgör commodity_pct."""
        return {"royalty": commodity_pct, "producenter": 0, "optionalitet": 0,
                "durrett": 0, "swing": 100 - commodity_pct, "insider": 0,
                "kassa": 0}

    assert a.commodity_state(vals(49))[0] == a.POS_OK
    assert a.commodity_state(vals(50))[0] == a.POS_NEAR
    assert a.commodity_state(vals(55))[0] == a.POS_NEAR   # taket ej brutet ännu
    assert a.commodity_state(vals(56))[0] == a.POS_OVER
    assert a.COMMODITY_WARN == 50.0


def test_commodity_state_agrees_with_the_breach_flag():
    """The warning band must never contradict the hard flag."""
    for pct in (0, 30, 49, 50, 55, 56, 80, 100):
        vals = {"royalty": pct, "producenter": 0, "optionalitet": 0,
                "durrett": 0, "swing": 100 - pct, "insider": 0, "kassa": 0}
        over = a.commodity_state(vals)[0] == a.POS_OVER
        assert over == a.commodity_breach(vals), pct


def test_an_exactly_full_commodity_budget_is_not_a_breach():
    """The cap is "<= 55 %". Binary floating point made exactly 55 % break it."""
    vals = {"royalty": 55, "producenter": 0, "optionalitet": 0, "durrett": 0,
            "swing": 45, "insider": 0, "kassa": 0}
    assert a.commodity_exposure(vals) == 55.0
    assert not a.commodity_breach(vals)
    # And split across all four commodity sleeves, which is the realistic case.
    split = {"royalty": 20, "producenter": 15, "optionalitet": 7, "durrett": 13,
             "swing": 45, "insider": 0, "kassa": 0}
    assert a.commodity_exposure(split) == 55.0
    assert not a.commodity_breach(split)


# ── Positionsregeln, två nivåer (Masterguiden 4.0) ───────────────────────────
# "NORMAL POSITION anges inom strategidelen, HÅRT TAK mot hela portföljen."
def test_the_seven_position_rules_match_the_guide():
    got = {r.key: (r.normal_lo, r.normal_hi, r.hard_cap) for r in a.POSITION_RULES}
    assert got == {
        "royalty1": (5, 10, 10.0),
        "rule":     (5, 10, 4.0),
        "sprott":   (10, 20, 1.5),
        "tiggre":   (15, 30, 4.0),
        "durrett":  (10, 20, 3.0),
        "swing":    (15, 30, 6.0),
        "insider":  (10, 20, 4.0),
    }


def test_sprott_and_tiggre_share_a_sleeve_but_not_a_cap():
    """The defect this rule fixes: one sleeve cap let a Sprott lot run to 4 %."""
    sprott = a.RULE_BY_KEY["sprott"]
    tiggre = a.RULE_BY_KEY["tiggre"]
    assert sprott.sleeve == tiggre.sleeve == "optionalitet"
    assert sprott.hard_cap == 1.5 and tiggre.hard_cap == 4.0


def test_a_sprott_lot_over_one_and_a_half_percent_now_breaches():
    positions = [{"ticker": "SPR", "rule": "sprott", "sleeve": "optionalitet",
                  "value": 3.0}]
    breaches = a.position_breaches(positions, total=100.0)
    assert [b["ticker"] for b in breaches] == ["SPR"]
    assert breaches[0]["cap"] == 1.5
    assert breaches[0]["rule"] == "Sprott"
    # samma storlek som Tiggre är inom taket
    positions[0]["rule"] = "tiggre"
    assert a.position_breaches(positions, total=100.0) == []


def test_the_rule_is_derived_from_the_sleeve_where_it_is_unambiguous():
    """Old positions store only a sleeve; six of seven can be resolved."""
    for sleeve, expected in (("royalty", "royalty1"), ("producenter", "rule"),
                             ("durrett", "durrett"), ("swing", "swing"),
                             ("insider", "insider")):
        assert a.position_rule({"sleeve": sleeve}).key == expected


def test_an_old_optionality_position_is_reported_not_guessed():
    """Guessing would pick between 1,5 % and 4 % on the user's behalf."""
    old = {"ticker": "OLD", "sleeve": "optionalitet", "value": 3.0}
    assert a.position_rule(old) is None
    assert a.unresolved_positions([old]) == [old]
    assert a.AMBIGUOUS_SLEEVES == ("optionalitet",)
    # ...och den flaggas inte som brott mot ett tak den inte har
    assert a.position_breaches([old], total=100.0) == []


def test_resolved_positions_are_not_reported_as_unresolved():
    assert a.unresolved_positions([{"sleeve": "swing"}]) == []
    assert a.unresolved_positions([{"rule": "sprott",
                                    "sleeve": "optionalitet"}]) == []
    assert a.unresolved_positions([]) == []
    assert a.unresolved_positions(None) == []


def test_normal_pct_measures_against_the_sleeve_not_the_portfolio():
    """20 % of a 7 % sleeve is 1,4 % of the total — different questions."""
    assert a.normal_pct(20.0, 100.0) == 20.0
    assert a.normal_pct(1.4, 7.0) == 20.0
    assert a.normal_pct(5, 0) is None
    assert a.normal_pct(None, 100) is None


def test_normal_state_bands():
    sprott = a.RULE_BY_KEY["sprott"]
    assert a.normal_state(5.0, sprott)[0] == "under normal"
    assert a.normal_state(10.0, sprott)[0] == "normal"
    assert a.normal_state(20.0, sprott)[0] == "normal"
    assert a.normal_state(25.0, sprott)[0] == "över normal"
    assert a.normal_state(None, sprott)[0] == "okänd"
    assert a.normal_state(15.0, None)[0] == "okänd"


def test_normal_and_cap_are_independent():
    """A position can be normal inside its sleeve and still break the cap."""
    # Tiggre 30 % av en optionalitetsdel som är 12 % av portföljen = 3,6 %
    assert a.normal_state(30.0, a.RULE_BY_KEY["tiggre"])[0] == "normal"
    assert a.position_breaches(
        [{"ticker": "T", "rule": "tiggre", "value": 4.5}], total=100.0)
