"""
Tests for controls.py — DS, AQS, CSM och proportionalitetsregeln
(tilläggsspecen till Masterguiden 4.0, punkt B–D och E).

DS är riskpoäng — lägre är bättre — vilket är motsatt riktning mot varje annan
poängmodell i panelen. Flera tester finns just för att fånga en förväxling.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import controls as c


# ═════════════════════════════════════════════════════════════════════════════
#  DS
# ═════════════════════════════════════════════════════════════════════════════
def _ds(**kw) -> dict:
    row = {f.key: 0 for f in c.DS_FIELDS}
    row.update(kw)
    return row


def test_ds_has_the_five_fields_from_the_spec():
    assert [f.key for f in c.DS_FIELDS] == [
        "ds_runway", "ds_capex", "ds_warrants", "ds_aktier_3ar", "ds_historik"]
    assert c.DS_MAX == 10
    for f in c.DS_FIELDS:
        assert f.zero and f.one and f.two, f.key


def test_ds_totals_and_clamps():
    assert c.ds_total(_ds()) == 0
    assert c.ds_total(_ds(**{f.key: 2 for f in c.DS_FIELDS})) == 10
    assert c.ds_total(_ds(ds_runway=9, ds_capex=-3)) == 2


def test_ds_is_blank_until_something_is_assessed():
    """Unassessed dilution is not low dilution."""
    assert c.ds_total({}) is None
    assert c.ds_total(None) is None
    assert c.ds_band(None) is None
    # ...men en rad som faktiskt fått nollor är bedömd
    assert c.ds_total({"ds_runway": 0}) == 0
    assert c.ds_band(0) == c.DS_LOW


def test_ds_bands_at_the_specs_boundaries():
    assert c.ds_band(0) == c.DS_LOW
    assert c.ds_band(2) == c.DS_LOW
    assert c.ds_band(3) == c.DS_OK
    assert c.ds_band(5) == c.DS_OK
    assert c.ds_band(6) == c.DS_HIGH
    assert c.ds_band(7) == c.DS_HIGH
    assert c.ds_band(8) == c.DS_EXTREME
    assert c.ds_band(10) == c.DS_EXTREME


def test_ds_is_a_risk_score_so_low_is_good():
    """The direction is opposite to every other model in the panel."""
    clean = _ds()
    dirty = _ds(**{f.key: 2 for f in c.DS_FIELDS})
    assert c.ds_total(clean) < c.ds_total(dirty)
    assert c.ds_band(c.ds_total(clean)) == c.DS_LOW
    assert c.ds_band(c.ds_total(dirty)) == c.DS_EXTREME


def test_the_runway_suggestion_is_inverted_against_the_scoring_model():
    """scoring.runway_points: >2 år = 2 poäng. Here: >2 år = 0 risk."""
    import scoring
    assert c.ds_runway_suggestion(3.0) == 0
    assert scoring.runway_points(3.0) == 2
    assert c.ds_runway_suggestion(1.5) == 1
    assert c.ds_runway_suggestion(0.5) == 2
    assert scoring.runway_points(0.5) == 0
    assert c.ds_runway_suggestion(None) is None


def test_high_ds_blocks_the_buy_from_six():
    assert not c.ds_blocks_buy(_ds(ds_runway=2, ds_capex=2, ds_warrants=1))  # 5
    assert c.ds_blocks_buy(_ds(ds_runway=2, ds_capex=2, ds_warrants=2))      # 6
    assert c.DS_BLOCK_MIN == 6


def test_a_dated_financing_catalyst_unlocks_it():
    row = _ds(**{f.key: 2 for f in c.DS_FIELDS})
    assert c.ds_blocks_buy(row)
    row["fin_catalyst_text"] = "Riktad emission klar"
    assert c.ds_blocks_buy(row), "vad utan när är inget besked"
    row["fin_catalyst_date"] = "2026-11"
    assert not c.ds_blocks_buy(row)
    assert c.ds_note(row) == c.DS_BLOCK_TEXT


def test_whitespace_is_not_a_financing_catalyst():
    row = _ds(**{f.key: 2 for f in c.DS_FIELDS})
    row["fin_catalyst_text"] = "   "
    row["fin_catalyst_date"] = "  "
    assert not c.has_financing_catalyst(row)
    assert c.ds_blocks_buy(row)


def test_an_unassessed_ds_does_not_block_here():
    """It is caught by the scorecard's gap rule instead — see köpgrinden."""
    assert not c.ds_blocks_buy({})
    assert c.ds_note({}) is None


# ═════════════════════════════════════════════════════════════════════════════
#  AQS
# ═════════════════════════════════════════════════════════════════════════════
def test_aqs_has_eight_fields_scored_zero_to_two():
    assert len(c.AQS_FIELDS) == 8
    assert c.AQS_MAX == 16
    assert c.aqs_total({f.key: 2 for f in c.AQS_FIELDS}) == 16


def test_aqs_bands_at_the_specs_boundaries():
    assert c.aqs_band(16) == c.AQS_HIGH
    assert c.aqs_band(13) == c.AQS_HIGH
    assert c.aqs_band(12) == c.AQS_OK
    assert c.aqs_band(10) == c.AQS_OK
    assert c.aqs_band(9) == c.AQS_DISCOUNT
    assert c.aqs_band(7) == c.AQS_DISCOUNT
    assert c.aqs_band(6) == c.AQS_PASS
    assert c.aqs_band(0) == c.AQS_PASS
    assert c.aqs_band(None) is None


def test_aqs_is_blank_until_assessed():
    assert c.aqs_total({}) is None
    assert c.aqs_total(None) is None
    assert c.aqs_total({"aqs_kostnad": 0}) == 0


def test_the_ten_year_life_scores_two():
    """The spec singles this one out: >10 år = 2."""
    life = [f for f in c.AQS_FIELDS if f.key == "aqs_livslangd"][0]
    assert "> 10 år" in life.two


# Ingen dubbelinmatning — jurisdiktion och management finns redan.
def test_producer_checkboxes_map_onto_the_aqs_scale():
    """0/1 -> 0/2, per the spec's explicit instruction."""
    assert c.aqs_prefill_from_producer({"jurisdiktion": True, "insyn": True}) == {
        "aqs_jurisdiktion": 2, "aqs_management": 2}
    assert c.aqs_prefill_from_producer({"jurisdiktion": False, "insyn": False}) == {
        "aqs_jurisdiktion": 0, "aqs_management": 0}
    assert c.aqs_prefill_from_producer({}) == {}
    assert c.aqs_prefill_from_producer(None) == {}


def test_tiggre_factors_carry_across_unchanged():
    """Lobo already scores these 0-2, so nothing is rescaled."""
    assert c.aqs_prefill_from_tiggre({"jurisdiktion": 2, "manniskor": 1}) == {
        "aqs_jurisdiktion": 2, "aqs_management": 1}
    assert c.aqs_prefill_from_tiggre({"jurisdiktion": 0}) == {
        "aqs_jurisdiktion": 0}
    assert c.aqs_prefill_from_tiggre({}) == {}


def test_prefilled_fields_are_named_so_the_ui_can_mark_them():
    assert set(c.AQS_PREFILLED) == {"aqs_jurisdiktion", "aqs_management"}
    keys = {f.key for f in c.AQS_FIELDS}
    assert set(c.AQS_PREFILLED) <= keys


# ═════════════════════════════════════════════════════════════════════════════
#  CSM
# ═════════════════════════════════════════════════════════════════════════════
def test_three_scenarios_by_default_five_for_core_holdings():
    assert c.scenarios() == ("Bear", "Base", "Bull")
    assert len(c.scenarios(is_core=True)) == 5
    assert c.scenarios(is_core=True)[0] == c.DEEP_BEAR


def test_a_developer_needing_capital_in_bear_does_not_survive_it():
    bear = {"nav_musd": 400, "financing_need": 120}
    assert c.bear_survival_suggestion(c.DEVELOPER, bear) is False
    assert c.bear_survival_suggestion(c.DEVELOPER, bear, secured_cash=True) is True
    assert c.bear_survival_suggestion(c.DEVELOPER, {"financing_need": 0}) is True


def test_a_producer_survives_bear_on_non_negative_cash_flow():
    assert c.bear_survival_suggestion(c.PRODUCER, {"fcf_musd": 5}) is True
    assert c.bear_survival_suggestion(c.PRODUCER, {"fcf_musd": 0}) is True
    assert c.bear_survival_suggestion(c.PRODUCER, {"fcf_musd": -5}) is False
    assert c.bear_survival_suggestion(c.PRODUCER, {}) is None


def test_the_red_flag_follows_the_suggestion_unless_overridden():
    matrix = {c.BEAR: {"financing_need": 100}}
    assert c.csm_red_flag(c.DEVELOPER, matrix)
    matrix["bear_survival"] = True          # du vet något om kassan
    assert not c.csm_red_flag(c.DEVELOPER, matrix)
    matrix["bear_survival"] = False
    assert c.csm_red_flag(c.DEVELOPER, matrix)


def test_no_red_flag_when_bear_is_simply_unfilled():
    assert not c.csm_red_flag(c.DEVELOPER, {})
    assert not c.csm_red_flag(c.PRODUCER, {})


def test_leverage_ratio_separates_cheap_from_cheap_with_leverage():
    matrix = {c.BEAR: {"fcf_musd": 10}, c.BULL: {"fcf_musd": 80}}
    assert c.leverage_ratio(matrix) == 8.0
    flat = {c.BEAR: {"fcf_musd": 50}, c.BULL: {"fcf_musd": 60}}
    assert c.leverage_ratio(flat) == 1.2


def test_leverage_is_undefined_when_bear_cash_flow_is_not_positive():
    """Dividing by zero is not infinite leverage — it is a company that dies."""
    assert c.leverage_ratio({c.BEAR: {"fcf_musd": 0},
                             c.BULL: {"fcf_musd": 80}}) is None
    assert c.leverage_ratio({c.BEAR: {"fcf_musd": -10},
                             c.BULL: {"fcf_musd": 80}}) is None
    assert c.leverage_ratio({}) is None


def test_csm_completeness_requires_every_scenario():
    m = {s: {"price": 100} for s in c.SCENARIOS_3}
    assert c.csm_complete(m)
    assert not c.csm_complete(m, is_core=True)      # fem krävs då
    del m[c.BULL]
    assert not c.csm_complete(m)
    assert not c.csm_complete({})


# ═════════════════════════════════════════════════════════════════════════════
#  Proportionalitetsregeln
# ═════════════════════════════════════════════════════════════════════════════
def test_a_position_above_two_percent_requires_everything():
    req = c.required_sections(3.0, "tiggre")
    assert req == {c.SEC_STRATEGY, c.SEC_DS, c.SEC_AQS, c.SEC_CSM}


def test_one_to_two_percent_requires_only_ds_and_the_strategy_score():
    for pct in (1.0, 1.5, 2.0):
        assert c.required_sections(pct, "sprott") == {c.SEC_STRATEGY, c.SEC_DS}
    # exakt 2 % är inte "över 2 %"
    assert c.SEC_AQS not in c.required_sections(2.0, "sprott")
    assert c.SEC_AQS in c.required_sections(2.01, "sprott")


def test_swing_and_insider_never_need_asset_or_commodity_work():
    for strat in ("swing", "momentum", "insider"):
        assert c.required_sections(10.0, strat) == {c.SEC_STRATEGY}
        assert c.SEC_AQS not in c.required_sections(10.0, strat)
        assert c.SEC_CSM not in c.required_sections(10.0, strat)


def test_swing_shows_ds_only_when_the_dilution_flag_is_ticked():
    assert c.required_sections(5.0, "swing", dilution_risk=True) == {
        c.SEC_STRATEGY, c.SEC_DS}
    assert c.SEC_DS not in c.required_sections(5.0, "swing")


def test_strategy_matching_is_case_and_space_insensitive():
    assert c.required_sections(9.0, " Swing ") == {c.SEC_STRATEGY}
    assert c.required_sections(9.0, "INSIDER") == {c.SEC_STRATEGY}


def test_section_required_helper_agrees_with_the_set():
    assert c.section_required(c.SEC_CSM, 5.0, "durrett")
    assert not c.section_required(c.SEC_CSM, 1.0, "durrett")
    assert not c.section_required(c.SEC_CSM, 5.0, "swing")


def test_missing_position_size_falls_back_to_the_light_requirement():
    """Unknown size should not silently demand the full workup."""
    assert c.required_sections(None, "sprott") == {c.SEC_STRATEGY, c.SEC_DS}
    assert c.required_sections("", "sprott") == {c.SEC_STRATEGY, c.SEC_DS}
