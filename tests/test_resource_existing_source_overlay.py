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
    classify_commodity_rs,
    classify_macro_context,
    classify_sentiment_attention,
    compute_commodity_rs,
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


class TestNotAvailableDefaults:
    def test_commodity_rs_not_available_without_series(self):
        ov = enrich_resource_candidate()
        assert ov.commodity_relative_strength is None
        assert ov.commodity_rs_flag == "COMMODITY_RS_NOT_AVAILABLE"
        assert "COMMODITY_RS_NOT_AVAILABLE" in ov.existing_source_flags

    def test_sentiment_and_macro_not_available_without_inputs(self):
        ov = enrich_resource_candidate()
        assert ov.sentiment_attention_flag == "SENTIMENT_NOT_AVAILABLE"
        assert ov.macro_context_flag == "MACRO_CONTEXT_NOT_AVAILABLE"
        assert "SENTIMENT_NOT_AVAILABLE" in ov.existing_source_flags
        assert "MACRO_CONTEXT_NOT_AVAILABLE" in ov.existing_source_flags


class TestCommodityRS:
    def test_compute_rs_outperforming(self):
        # candidate +20% over 3 bars, proxy +10% → RS ≈ +10pp.
        cand = [100.0, 110.0, 120.0]
        proxy = [100.0, 105.0, 110.0]
        rs = compute_commodity_rs(cand, proxy, window=2)
        assert rs == 10.0
        assert classify_commodity_rs(rs) == "OUTPERFORMING_PROXY"

    def test_compute_rs_lagging(self):
        cand = [100.0, 100.0, 100.0]   # flat
        proxy = [100.0, 110.0, 120.0]  # +20%
        rs = compute_commodity_rs(cand, proxy, window=2)
        assert rs == -20.0
        assert classify_commodity_rs(rs) == "LAGGING_PROXY"

    def test_compute_rs_neutral(self):
        cand = [100.0, 101.0, 102.0]
        proxy = [100.0, 100.5, 101.0]
        rs = compute_commodity_rs(cand, proxy, window=2)
        assert classify_commodity_rs(rs) == "RS_NEUTRAL"

    def test_rs_none_when_series_too_short(self):
        assert compute_commodity_rs([100.0], [100.0, 101.0, 102.0], window=2) is None
        assert compute_commodity_rs(None, None, window=2) is None
        assert classify_commodity_rs(None) == "COMMODITY_RS_NOT_AVAILABLE"

    def test_rs_wired_through_enrich(self):
        cand = [100.0, 110.0, 130.0]
        proxy = [100.0, 100.0, 100.0]
        ov = enrich_resource_candidate(
            candidate_closes=cand, proxy_closes=proxy, rs_window=2
        )
        assert ov.commodity_relative_strength == 30.0
        assert ov.commodity_rs_flag == "OUTPERFORMING_PROXY"
        assert "OUTPERFORMING_PROXY" in ov.existing_source_flags


class TestSentimentAttention:
    def test_explicit_label_passthrough(self):
        assert classify_sentiment_attention({"attention": "hype_risk"}) == "HYPE_RISK"

    def test_hype_from_high_score(self):
        assert classify_sentiment_attention({"composite_score": 85}) == "HYPE_RISK"

    def test_low_attention_from_low_score(self):
        assert classify_sentiment_attention({"composite_score": 10}) == "LOW_ATTENTION"

    def test_normal_from_mid_score(self):
        assert classify_sentiment_attention({"score": 50}) == "NORMAL_ATTENTION"

    def test_low_attention_from_message_count(self):
        assert classify_sentiment_attention({"message_count": 2}) == "LOW_ATTENTION"

    def test_missing_is_not_available(self):
        assert classify_sentiment_attention(None) == "SENTIMENT_NOT_AVAILABLE"
        assert classify_sentiment_attention({}) == "SENTIMENT_NOT_AVAILABLE"

    def test_wired_through_enrich(self):
        ov = enrich_resource_candidate(sentiment={"composite_score": 90})
        assert ov.sentiment_attention_flag == "HYPE_RISK"


class TestMacroContext:
    def test_tailwind_from_steepening(self):
        assert classify_macro_context({"t10y2y_change_4w": 0.15}) == "COMMODITY_MACRO_TAILWIND"

    def test_headwind_from_inversion(self):
        assert classify_macro_context({"t10y2y_change_4w": -0.30}) == "COMMODITY_MACRO_HEADWIND"

    def test_neutral_from_flat(self):
        assert classify_macro_context({"t10y2y_change_4w": 0.0}) == "MACRO_NEUTRAL"

    def test_ember_regime_mapping(self):
        assert classify_macro_context({"regime": "GREEN"}) == "COMMODITY_MACRO_TAILWIND"
        assert classify_macro_context({"regime": "RED"}) == "COMMODITY_MACRO_HEADWIND"
        assert classify_macro_context({"regime": "AMBER"}) == "MACRO_NEUTRAL"

    def test_missing_is_not_available(self):
        assert classify_macro_context(None) == "MACRO_CONTEXT_NOT_AVAILABLE"
        assert classify_macro_context({}) == "MACRO_CONTEXT_NOT_AVAILABLE"

    def test_wired_through_enrich(self):
        ov = enrich_resource_candidate(macro={"t10y2y_change_4w": 0.20})
        assert ov.macro_context_flag == "COMMODITY_MACRO_TAILWIND"


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
        assert r.commodity_rs_flag == ""
        assert r.sentiment_attention_flag == ""
        assert r.macro_context_flag == ""
        assert r.existing_source_flags == []
