"""
Tests for the PR3 resource-composite v1 (contrarian_alpha/resource_scoring.py).

Covers stage weights, missing-data flags/neutral scoring, jurisdiction/commodity
mapping, and confirms the composite is deterministic and bounded. Also verifies
the engine still leaves resource fields at their Nordic defaults.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrarian_alpha.resource_scoring import (
    compute_resource_composite,
    score_commodity,
    score_jurisdiction,
    score_survival,
    score_dilution,
    get_stage_weights,
    ResourceScore,
    _STAGE_WEIGHTS,
    _NEUTRAL,
)


class TestStageWeights:
    def test_all_profiles_sum_to_one(self):
        for name, w in _STAGE_WEIGHTS.items():
            assert abs(sum(w.values()) - 1.0) < 1e-9, f"{name} sums to {sum(w.values())}"

    def test_all_profiles_share_same_factor_keys(self):
        keys = set(_STAGE_WEIGHTS["_default"].keys())
        for name, w in _STAGE_WEIGHTS.items():
            assert set(w.keys()) == keys, f"{name} has mismatched factor keys"

    def test_explorer_weights_survival_over_quality(self):
        _, w = get_stage_weights("explorer")
        assert w["survival"] > w["quality"]
        assert w["catalyst"] > w["value"]

    def test_producer_weights_quality_over_survival(self):
        _, w = get_stage_weights("producer")
        assert w["quality"] > w["survival"]

    def test_unknown_stage_falls_back_to_default(self):
        key, w = get_stage_weights("banana")
        assert key == "_default"
        assert w == _STAGE_WEIGHTS["_default"]

    def test_empty_stage_falls_back_to_default(self):
        key, _ = get_stage_weights("")
        assert key == "_default"


class TestCommodityMapping:
    def test_strategic_commodity_scores_high_and_flags(self):
        score, flags = score_commodity("uranium")
        assert score >= 90
        assert "STRATEGIC_COMMODITY" in flags

    def test_gold_is_recognised(self):
        score, flags = score_commodity("gold")
        assert score == 85.0
        assert "COMMODITY_UNKNOWN" not in flags

    def test_unknown_commodity_neutral_and_flagged(self):
        score, flags = score_commodity("unobtainium")
        assert score == 55.0
        assert "COMMODITY_UNKNOWN" in flags

    def test_empty_commodity_unknown(self):
        score, flags = score_commodity("")
        assert "COMMODITY_UNKNOWN" in flags

    def test_strategic_secondary_nudges_up_but_capped(self):
        base, _ = score_commodity("gold")
        boosted, _ = score_commodity("gold", "copper")
        assert boosted > base
        assert boosted <= 98.0


class TestJurisdictionMapping:
    def test_canada_high_baseline(self):
        score, conf, flags = score_jurisdiction("CA")
        assert score == 85.0
        assert conf == 1.0
        assert flags == []

    def test_us_high_baseline(self):
        score, _, _ = score_jurisdiction("US")
        assert score == 85.0

    def test_region_field_takes_priority(self):
        score, conf, _ = score_jurisdiction("CA", jurisdiction="Nevada")
        assert score == 90.0
        assert conf == 1.0

    def test_high_risk_region_flagged(self):
        score, _, flags = score_jurisdiction("", jurisdiction="Venezuela")
        assert score < 50
        assert "HIGH_RISK_JURISDICTION" in flags

    def test_exchange_implies_country(self):
        score, conf, flags = score_jurisdiction("", exchange="TSXV")
        assert score == 85.0
        assert conf < 1.0
        assert "JURISDICTION_FROM_EXCHANGE" in flags

    def test_unknown_jurisdiction_neutral_low_conf(self):
        score, conf, flags = score_jurisdiction("", exchange="", jurisdiction="")
        assert score == _NEUTRAL
        assert conf < 0.5
        assert "JURISDICTION_UNKNOWN" in flags


class TestSurvival:
    def test_missing_data_flags_and_neutral(self):
        score, conf, flags = score_survival({}, "explorer")
        assert "SURVIVAL_DATA_MISSING" in flags
        assert conf < 0.5
        assert score == 45.0  # pre-revenue conservative neutral

    def test_missing_data_producer_higher_neutral(self):
        score, _, flags = score_survival({}, "producer")
        assert score == 60.0
        assert "SURVIVAL_DATA_MISSING" in flags

    def test_long_runway_scores_high(self):
        score, conf, flags = score_survival(
            {"cash_musd": "100", "quarterly_burn_musd": "5"}, "explorer",
        )
        assert score >= 90          # 20 quarters runway
        assert conf == 1.0
        assert "SURVIVAL_DATA_MISSING" not in flags

    def test_critical_runway_flagged(self):
        score, _, flags = score_survival(
            {"cash_musd": "2", "quarterly_burn_musd": "5"}, "explorer",
        )
        assert score <= 20
        assert "CRITICAL_RUNWAY" in flags

    def test_debt_reduces_net_cash(self):
        no_debt, _, _ = score_survival(
            {"cash_musd": "40", "quarterly_burn_musd": "5"}, "explorer")
        with_debt, _, flags = score_survival(
            {"cash_musd": "40", "quarterly_burn_musd": "5", "debt_musd": "45"}, "explorer")
        assert with_debt < no_debt
        assert "NET_DEBT_POSITION" in flags

    def test_zero_burn_treated_as_missing(self):
        _, conf, flags = score_survival(
            {"cash_musd": "40", "quarterly_burn_musd": "0"}, "explorer")
        assert "SURVIVAL_DATA_MISSING" in flags
        assert conf < 0.5


class TestDilution:
    def test_missing_data_explorer_conservative(self):
        score, conf, flags = score_dilution({}, "explorer")
        assert "DILUTION_DATA_MISSING" in flags
        assert score == 40.0
        assert conf < 0.5

    def test_missing_data_producer_favorable(self):
        score, _, flags = score_dilution({}, "producer")
        assert score == 60.0
        assert "DILUTION_DATA_MISSING" in flags

    def test_buyback_scores_high(self):
        score, conf, _ = score_dilution({"shares_yoy_growth_pct": "-2"}, "producer")
        assert score >= 90
        assert conf == 1.0

    def test_severe_dilution_flagged(self):
        score, _, flags = score_dilution({"shares_yoy_growth_pct": "45"}, "explorer")
        assert score <= 15
        assert "SEVERE_DILUTION" in flags


class TestComposite:
    def test_composite_bounded_and_deterministic(self):
        rs = compute_resource_composite(
            stage="explorer", meta={}, country="CA", exchange="TSXV",
            primary_commodity="uranium", hate_score=70, catalyst_score=60,
        )
        assert isinstance(rs, ResourceScore)
        assert 0.0 <= rs.resource_composite <= 100.0
        # Same inputs → same output (deterministic).
        rs2 = compute_resource_composite(
            stage="explorer", meta={}, country="CA", exchange="TSXV",
            primary_commodity="uranium", hate_score=70, catalyst_score=60,
        )
        assert rs.resource_composite == rs2.resource_composite

    def test_missing_quality_value_neutral_not_penalised(self):
        rs = compute_resource_composite(
            stage="producer", meta={}, country="US",
            primary_commodity="gold", hate_score=50, catalyst_score=50,
            quality_score=None, value_score=None,
        )
        assert "QUALITY_DATA_MISSING" in rs.flags
        assert "VALUE_DATA_MISSING" in rs.flags

    def test_full_data_beats_missing_data_confidence(self):
        rich = compute_resource_composite(
            stage="explorer",
            meta={"cash_musd": "100", "quarterly_burn_musd": "5",
                  "shares_yoy_growth_pct": "3", "jurisdiction": "Nevada"},
            country="US", primary_commodity="uranium",
            hate_score=60, catalyst_score=60,
        )
        sparse = compute_resource_composite(
            stage="explorer", meta={}, country="", primary_commodity="",
            hate_score=60, catalyst_score=60,
        )
        assert rich.resource_confidence > sparse.resource_confidence
        assert "LOW_DATA_CONFIDENCE" in sparse.flags

    def test_stage_profile_recorded(self):
        rs = compute_resource_composite(
            stage="royalty", meta={}, country="CA", primary_commodity="gold",
        )
        assert rs.stage_profile == "royalty"

    def test_never_raises_on_none_meta(self):
        rs = compute_resource_composite(stage="", meta=None)
        assert isinstance(rs, ResourceScore)
        assert 0.0 <= rs.resource_composite <= 100.0


class TestNordicUnchanged:
    def test_result_defaults_have_zero_resource_composite(self):
        from contrarian_alpha.engine import ContrairianAlphaResult
        r = ContrairianAlphaResult(
            ticker="ABB", ins_id=1, name="ABB", market="SE",
            sector="Industrials", branch="Electrical", composite_score=0.0,
        )
        assert r.resource_composite == 0.0
        assert r.survival_score == 0.0
        assert r.resource_confidence == 0.0
        assert r.resource_stage_profile == ""
