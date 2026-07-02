"""
Tests for the PR2 stage-aware resource guardrails.

Verifies that _apply_resource_stage_guardrails relaxes mature-company balance
sheet / ROIC gates for pre-revenue juniors and data-sparse US/CA rows, while
keeping real weakness eliminating for mature stages — and that Nordic behavior
is untouched (empty resource fields, helper never engaged).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrarian_alpha.engine import (
    _apply_resource_stage_guardrails,
    ContrairianAlphaResult,
    PipelineConfig,
    _RESOURCE_PRE_REVENUE_STAGES,
    _RESOURCE_MATURE_STAGES,
)

# Balance-sheet failure strings exactly as engine._run_single_ticker builds them.
_ALL_BS_FAIL = ["FCF ≤ 0", "EBITDA margin ≤ 0%", "Equity ≤ 0"]


class TestPreRevenueStages:
    def test_explorer_missing_data_survives(self):
        # ins_id=None US/CA row → all fundamentals missing → all gates fail.
        gates = {"fcf_positive": False, "ebitda_margin_positive": False,
                 "equity_positive": False, "debt_equity_low": False}
        kept, drop_roic, mode, flags = _apply_resource_stage_guardrails(
            "explorer", fund_dict={}, gates=gates, roic=None,
            bs_failures=list(_ALL_BS_FAIL),
        )
        assert kept == []                 # nothing eliminates it
        assert drop_roic is True
        assert mode == "RELAXED"
        assert "PRE_REVENUE" in flags
        assert "NO_FCF_EXPECTED" in flags
        assert "ROIC_NOT_APPLICABLE" in flags
        assert "STAGE_AWARE_GATE_RELAXED" in flags

    def test_explorer_negative_fcf_present_still_survives(self):
        # Even with real negative FCF/EBITDA, a pre-revenue miner is not killed.
        gates = {"fcf_positive": False, "ebitda_margin_positive": False,
                 "equity_positive": True, "debt_equity_low": True}
        kept, drop_roic, mode, _ = _apply_resource_stage_guardrails(
            "explorer", fund_dict={"fcf": -50.0, "ebitda_margin": -12.0, "equity": 100.0},
            gates=gates, roic=None, bs_failures=["FCF ≤ 0", "EBITDA margin ≤ 0%"],
        )
        assert kept == []
        assert mode == "RELAXED"

    def test_developer_is_pre_revenue(self):
        gates = {"fcf_positive": False, "ebitda_margin_positive": False,
                 "equity_positive": False, "debt_equity_low": True}
        kept, drop_roic, mode, flags = _apply_resource_stage_guardrails(
            "developer", fund_dict={}, gates=gates, roic=None,
            bs_failures=list(_ALL_BS_FAIL),
        )
        assert kept == []
        assert mode == "RELAXED"
        assert "PRE_REVENUE" in flags

    def test_explorer_high_debt_still_gated(self):
        # Over-levered junior: D/E is the one gate kept even for explorers.
        gates = {"fcf_positive": False, "ebitda_margin_positive": False,
                 "equity_positive": True, "debt_equity_low": False}
        kept, _, mode, _ = _apply_resource_stage_guardrails(
            "explorer", fund_dict={"equity": 10.0},
            gates=gates, roic=None,
            bs_failures=["FCF ≤ 0", "EBITDA margin ≤ 0%", "D/E ≥ 0.6"],
        )
        assert kept == ["D/E ≥ 0.6"]
        assert mode == "RELAXED"


class TestMatureStages:
    def test_producer_missing_data_not_eliminated(self):
        # US/CA producer with no Börsdata snapshot → suppress on missing data.
        gates = {"fcf_positive": False, "ebitda_margin_positive": False,
                 "equity_positive": False, "debt_equity_low": True}
        kept, drop_roic, mode, flags = _apply_resource_stage_guardrails(
            "producer", fund_dict={}, gates=gates, roic=None,
            bs_failures=list(_ALL_BS_FAIL),
        )
        assert kept == []                 # missing-data does not eliminate
        assert drop_roic is True          # ROIC missing → gate dropped
        assert mode == "MATURE"
        assert "FCF_DATA_MISSING" in flags
        assert "ROIC_DATA_MISSING" in flags

    def test_producer_present_negative_fcf_eliminated(self):
        # Real, present weakness for a mature producer still eliminates.
        gates = {"fcf_positive": False, "ebitda_margin_positive": True,
                 "equity_positive": True, "debt_equity_low": True}
        kept, _, mode, _ = _apply_resource_stage_guardrails(
            "producer", fund_dict={"fcf": -20.0, "ebitda_margin": 15.0, "equity": 500.0},
            gates=gates, roic=None, bs_failures=["FCF ≤ 0"],
        )
        assert kept == ["FCF ≤ 0"]        # not suppressed — data was present
        assert mode == "MATURE"

    def test_producer_present_roic_below_gate_not_dropped(self):
        gates = {"fcf_positive": True, "ebitda_margin_positive": True,
                 "equity_positive": True, "debt_equity_low": True}
        _, drop_roic, _, _ = _apply_resource_stage_guardrails(
            "producer", fund_dict={"fcf": 10.0, "ebitda_margin": 20.0, "equity": 500.0},
            gates=gates, roic=8.0, bs_failures=[],
        )
        assert drop_roic is False         # real ROIC present → normal gate applies

    def test_royalty_missing_fundamentals_survives(self):
        gates = {"fcf_positive": False, "ebitda_margin_positive": False,
                 "equity_positive": False, "debt_equity_low": True}
        kept, _, mode, flags = _apply_resource_stage_guardrails(
            "royalty", fund_dict={}, gates=gates, roic=None,
            bs_failures=list(_ALL_BS_FAIL),
        )
        assert kept == []
        assert mode == "MATURE"
        assert "STAGE_AWARE_MISSING_DATA_RELAXED" in flags


class TestNordicUnchanged:
    def test_result_defaults_have_empty_resource_fields(self):
        r = ContrairianAlphaResult(
            ticker="ABB", ins_id=1, name="ABB", market="SE",
            sector="Industrials", branch="Electrical", composite_score=0.0,
        )
        assert r.stage == ""
        assert r.primary_commodity == ""
        assert r.resource_flags == []
        assert r.resource_gate_mode == ""

    def test_default_universe_is_nordic(self):
        assert PipelineConfig().universe == "nordic"

    def test_stage_vocabularies_are_disjoint_and_complete(self):
        from contrarian_alpha.universe_static import VALID_STAGES
        assert _RESOURCE_PRE_REVENUE_STAGES & _RESOURCE_MATURE_STAGES == set()
        # Every declared CSV stage is classified by the guardrail.
        assert VALID_STAGES == (_RESOURCE_PRE_REVENUE_STAGES | _RESOURCE_MATURE_STAGES)
