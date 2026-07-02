"""
Tests for the PR5 existing-source overlay
(contrarian_alpha/existing_source_enrichment.py).

The overlay is additive context built from data the pipeline already fetched
from EXISTING sources. It must:
  * make no network calls (all inputs are passed in — these tests never touch
    external services),
  * flag missing data rather than fabricate numbers,
  * keep its score separate from resource_composite,
  * leave Nordic result defaults blank.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrarian_alpha.existing_source_enrichment import (
    LIQUIDITY_THRESHOLD_USD,
    enrich_resource_candidate,
)


class TestDrawdown:
    def test_deep_drawdown_flagged(self):
        ov = enrich_resource_candidate(close=40.0, high_52w=100.0)
        assert ov.drawdown_52w_pct == -60.0
        assert "DEEP_52W_DRAWDOWN" in ov.existing_source_flags

    def test_shallow_drawdown_not_flagged(self):
        ov = enrich_resource_candidate(close=95.0, high_52w=100.0)
        assert ov.drawdown_52w_pct == -5.0
        assert "DEEP_52W_DRAWDOWN" not in ov.existing_source_flags

    def test_missing_price_flags_drawdown(self):
        ov = enrich_resource_candidate(close=0.0, high_52w=0.0)
        assert ov.drawdown_52w_pct is None
        assert "DRAWDOWN_DATA_MISSING" in ov.existing_source_flags


class TestLiquidity:
    def test_low_liquidity_flagged(self):
        # 1000 shares * $1 = $1k daily turnover, well below threshold.
        ov = enrich_resource_candidate(close=1.0, high_52w=2.0, avg_volume_20d=1_000)
        assert ov.liquidity_flag == "LOW"
        assert "LOW_LIQUIDITY" in ov.existing_source_flags

    def test_ok_liquidity(self):
        vol = (LIQUIDITY_THRESHOLD_USD * 10)  # $10x threshold at $1 close
        ov = enrich_resource_candidate(close=1.0, high_52w=2.0, avg_volume_20d=vol)
        assert ov.liquidity_flag == "OK"
        assert "LOW_LIQUIDITY" not in ov.existing_source_flags

    def test_missing_volume_flags_liquidity(self):
        ov = enrich_resource_candidate(close=10.0, high_52w=20.0, avg_volume_20d=None)
        assert ov.liquidity_flag == "UNKNOWN"
        assert "LIQUIDITY_DATA_MISSING" in ov.existing_source_flags


class TestMarketCap:
    def test_estimated_from_shares_out(self):
        # 100M shares * $5 = $500M → small cap, flagged as an estimate.
        ov = enrich_resource_candidate(
            close=5.0, high_52w=10.0, meta={"shares_out_m": "100"}
        )
        assert ov.market_cap_bucket == "small"
        assert "MARKET_CAP_ESTIMATED" in ov.existing_source_flags

    def test_nano_cap_flagged(self):
        ov = enrich_resource_candidate(
            close=0.10, high_52w=1.0, meta={"shares_out_m": "50"}
        )  # 50M * $0.10 = $5M → nano
        assert ov.market_cap_bucket == "nano"
        assert "NANO_CAP" in ov.existing_source_flags

    def test_real_market_cap_preferred(self):
        ov = enrich_resource_candidate(
            close=5.0, high_52w=10.0,
            meta={"shares_out_m": "100"}, market_cap_usd=15_000_000_000.0,
        )
        assert ov.market_cap_bucket == "large"
        assert "MARKET_CAP_ESTIMATED" not in ov.existing_source_flags

    def test_missing_market_cap_flagged(self):
        ov = enrich_resource_candidate(close=5.0, high_52w=10.0, meta={})
        assert ov.market_cap_bucket == "unknown"
        assert "MARKET_CAP_DATA_MISSING" in ov.existing_source_flags


class TestShortInterest:
    def test_high_short_interest(self):
        ov = enrich_resource_candidate(short_data={"short_float_pct": 22.0})
        assert ov.short_interest_flag == "HIGH"
        assert "HIGH_SHORT_INTEREST" in ov.existing_source_flags

    def test_elevated_short_interest(self):
        ov = enrich_resource_candidate(short_data={"short_float_pct": 10.0})
        assert ov.short_interest_flag == "ELEVATED"

    def test_normal_short_interest(self):
        ov = enrich_resource_candidate(short_data={"short_float_pct": 2.0})
        assert ov.short_interest_flag == "NORMAL"

    def test_missing_short_data(self):
        ov = enrich_resource_candidate(short_data=None)
        assert ov.short_interest_flag == "UNKNOWN"
        assert "SHORT_INTEREST_DATA_MISSING" in ov.existing_source_flags


class TestAnalystRevisions:
    def test_net_downgrades(self):
        ov = enrich_resource_candidate(
            analyst_data={"downgrades_90d": 4, "upgrades_90d": 1}
        )
        assert ov.analyst_revision_flag == "NET_DOWNGRADES"
        assert "ANALYST_NET_DOWNGRADES" in ov.existing_source_flags

    def test_net_upgrades(self):
        ov = enrich_resource_candidate(
            analyst_data={"downgrades_90d": 0, "upgrades_90d": 3}
        )
        assert ov.analyst_revision_flag == "NET_UPGRADES"

    def test_missing_analyst_data(self):
        ov = enrich_resource_candidate(analyst_data=None)
        assert ov.analyst_revision_flag == "UNKNOWN"
        assert "ANALYST_DATA_MISSING" in ov.existing_source_flags


class TestPlaceholders:
    def test_commodity_rs_placeholder(self):
        ov = enrich_resource_candidate()
        assert ov.commodity_relative_strength is None
        assert "COMMODITY_RS_NOT_AVAILABLE" in ov.existing_source_flags

    def test_sentiment_and_macro_placeholders(self):
        ov = enrich_resource_candidate()
        assert ov.sentiment_attention_flag == "NOT_WIRED"
        assert ov.macro_context_flag == "NOT_WIRED"
        assert "SENTIMENT_NOT_WIRED" in ov.existing_source_flags
        assert "MACRO_CONTEXT_NOT_WIRED" in ov.existing_source_flags


class TestOverlayScore:
    def test_empty_row_is_neutral_no_signal(self):
        ov = enrich_resource_candidate()
        assert ov.resource_overlay_score == 50.0
        assert "OVERLAY_NO_SIGNAL" in ov.existing_source_flags

    def test_contrarian_signals_lift_score(self):
        ov = enrich_resource_candidate(
            close=40.0, high_52w=100.0, avg_volume_20d=LIQUIDITY_THRESHOLD_USD * 10,
            short_data={"short_float_pct": 20.0},
            analyst_data={"downgrades_90d": 5, "upgrades_90d": 0},
        )
        # deep drawdown (+12), high short (+8), net downgrades (+6) → 76
        assert ov.resource_overlay_score > 50.0
        assert "OVERLAY_NO_SIGNAL" not in ov.existing_source_flags

    def test_low_liquidity_penalises_score(self):
        low = enrich_resource_candidate(close=1.0, high_52w=2.0, avg_volume_20d=1_000)
        assert low.resource_overlay_score < 50.0

    def test_score_bounded_0_100(self):
        ov = enrich_resource_candidate(
            close=1.0, high_52w=100.0, avg_volume_20d=1_000,
            short_data={"short_float_pct": 90.0},
            analyst_data={"downgrades_90d": 10, "upgrades_90d": 0},
        )
        assert 0.0 <= ov.resource_overlay_score <= 100.0

    def test_missing_data_is_not_penalised(self):
        # No signals at all stays exactly neutral (no fabricated penalty).
        ov = enrich_resource_candidate(meta={})
        assert ov.resource_overlay_score == 50.0


class TestNoNetworkAndNordicUnchanged:
    def test_pure_function_handles_garbage_gracefully(self):
        ov = enrich_resource_candidate(
            close="oops", high_52w=None,  # type: ignore[arg-type]
            avg_volume_20d="n/a", meta={"shares_out_m": "not-a-number"},
            short_data={"short_float_pct": "bad"},
        )
        assert ov.short_interest_flag == "UNKNOWN"
        assert ov.market_cap_bucket == "unknown"

    def test_nordic_result_defaults_blank(self):
        from contrarian_alpha.engine import ContrairianAlphaResult
        r = ContrairianAlphaResult(
            ticker="ABB", ins_id=1, name="ABB", market="SE",
            sector="Industrials", branch="Electrical", composite_score=0.0,
        )
        assert r.resource_overlay_score == 0.0
        assert r.market_cap_bucket == ""
        assert r.liquidity_flag == ""
        assert r.drawdown_52w_pct is None
        assert r.existing_source_flags == []
