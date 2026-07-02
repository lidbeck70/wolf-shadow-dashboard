"""
Tests for the PR4 resource enrichment layer: data-freshness flags, the
data-quality label, commodity proxy metadata, the expanded CSV schema, and the
standalone CSV validator (scripts/validate_resource_universe.py).

All additions are transparency-only: they must never change the deterministic
resource_composite math from PR3, and Nordic behavior stays untouched.
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrarian_alpha.resource_scoring import (
    commodity_proxy,
    compute_resource_composite,
    score_data_freshness,
)
from scripts.validate_resource_universe import validate_resource_universe

_TODAY = datetime.date(2026, 7, 2)


class TestCommodityProxy:
    def test_uranium_maps_to_urnm_ura(self):
        assert commodity_proxy("uranium") == "URNM,URA"

    def test_underscored_oil_gas_spelling(self):
        assert commodity_proxy("oil_gas") == "XLE,USO"

    def test_gold_and_silver(self):
        assert commodity_proxy("gold") == "GLD,GDX"
        assert commodity_proxy("silver") == "SLV,SIL"

    def test_unknown_returns_empty(self):
        assert commodity_proxy("unobtainium") == ""

    def test_falls_back_to_secondary(self):
        assert commodity_proxy("unobtainium", "copper") == "COPX,CPER"


class TestDataFreshness:
    def test_missing_date_flagged(self):
        as_of, flags = score_data_freshness({}, today=_TODAY)
        assert as_of == ""
        assert flags == ["DATA_AS_OF_MISSING"]

    def test_invalid_date_flagged(self):
        as_of, flags = score_data_freshness({"data_as_of": "07/02/2026"}, today=_TODAY)
        assert "DATA_AS_OF_INVALID" in flags
        assert as_of == "07/02/2026"

    def test_stale_date_flagged(self):
        as_of, flags = score_data_freshness({"data_as_of": "2024-01-01"}, today=_TODAY)
        assert "DATA_AS_OF_STALE" in flags
        assert as_of == "2024-01-01"

    def test_fresh_date_no_flag(self):
        as_of, flags = score_data_freshness({"data_as_of": "2026-05-01"}, today=_TODAY)
        assert flags == []
        assert as_of == "2026-05-01"


class TestDataQualityLabel:
    def test_sparse_row_is_low_quality(self):
        rs = compute_resource_composite(
            stage="explorer", meta={}, country="", primary_commodity="",
            hate_score=50, catalyst_score=50, today=_TODAY,
        )
        assert rs.resource_data_quality == "LOW"
        assert "DATA_AS_OF_MISSING" in rs.flags

    def test_rich_fresh_row_is_high_quality(self):
        rs = compute_resource_composite(
            stage="explorer",
            meta={"cash_musd": "100", "quarterly_burn_musd": "5",
                  "shares_yoy_growth_pct": "3", "jurisdiction": "Nevada",
                  "data_as_of": "2026-05-01"},
            country="US", primary_commodity="uranium",
            hate_score=60, catalyst_score=60,
            quality_score=70, value_score=70, today=_TODAY,
        )
        assert rs.resource_data_quality == "HIGH"
        assert rs.commodity_proxy == "URNM,URA"
        assert rs.data_as_of == "2026-05-01"

    def test_stale_date_caps_quality_below_high(self):
        rs = compute_resource_composite(
            stage="explorer",
            meta={"cash_musd": "100", "quarterly_burn_musd": "5",
                  "shares_yoy_growth_pct": "3", "jurisdiction": "Nevada",
                  "data_as_of": "2023-01-01"},
            country="US", primary_commodity="uranium",
            hate_score=60, catalyst_score=60,
            quality_score=70, value_score=70, today=_TODAY,
        )
        assert rs.resource_data_quality != "HIGH"
        assert "DATA_AS_OF_STALE" in rs.flags


class TestCompositeMathUnchanged:
    def test_freshness_does_not_alter_composite(self):
        base = dict(
            stage="explorer",
            meta={"cash_musd": "100", "quarterly_burn_musd": "5",
                  "shares_yoy_growth_pct": "3", "jurisdiction": "Nevada"},
            country="US", primary_commodity="uranium",
            hate_score=60, catalyst_score=60,
        )
        fresh = compute_resource_composite(
            **{**base, "meta": {**base["meta"], "data_as_of": "2026-05-01"}},
            today=_TODAY)
        stale = compute_resource_composite(
            **{**base, "meta": {**base["meta"], "data_as_of": "2020-01-01"}},
            today=_TODAY)
        assert fresh.resource_composite == stale.resource_composite


class TestCsvSchema:
    def test_new_columns_present_and_read(self):
        from contrarian_alpha.universe_static import load_resource_universe
        recs = load_resource_universe()
        by_ticker = {r.ticker: r for r in recs}
        # A curated enriched row carries jurisdiction/project_region.
        fcu = by_ticker["FCU.TO"]
        assert fcu.jurisdiction == "Saskatchewan"
        assert fcu.project_region == "Athabasca Basin"
        # Financials remain blank + flagged (no fabrication).
        assert fcu.cash_musd == ""
        assert fcu.resource_notes == "needs_validation"

    def test_metadata_exposes_new_fields(self):
        from contrarian_alpha.universe_static import load_resource_universe
        meta = load_resource_universe()[0].to_metadata()
        for key in ("project_region", "resource_notes", "data_source", "data_as_of"):
            assert key in meta


class TestValidator:
    def test_seed_csv_passes(self):
        res = validate_resource_universe()
        assert res.ok, f"unexpected errors: {res.errors}"
        assert res.row_count >= 50

    def test_completeness_summary_populated(self):
        res = validate_resource_universe()
        assert res.completeness["jurisdiction"] >= 1
        assert res.completeness["cash_musd"] == 0  # blank by design in PR4

    def test_duplicate_ticker_detected(self, tmp_path):
        p = tmp_path / "dup.csv"
        p.write_text(
            "ticker,yf_ticker,name,country,stage,primary_commodity\n"
            "AAA,AAA,A,US,producer,gold\n"
            "AAA,AAA,A,US,producer,gold\n")
        res = validate_resource_universe(p)
        assert not res.ok
        assert any("duplicate" in e for e in res.errors)

    def test_invalid_stage_detected(self, tmp_path):
        p = tmp_path / "stage.csv"
        p.write_text(
            "ticker,yf_ticker,name,country,stage,primary_commodity\n"
            "AAA,AAA,A,US,banana,gold\n")
        res = validate_resource_universe(p)
        assert not res.ok
        assert any("invalid stage" in e for e in res.errors)

    def test_bad_data_as_of_detected(self, tmp_path):
        p = tmp_path / "date.csv"
        p.write_text(
            "ticker,yf_ticker,name,country,stage,primary_commodity,data_as_of\n"
            "AAA,AAA,A,US,producer,gold,not-a-date\n")
        res = validate_resource_universe(p)
        assert not res.ok
        assert any("data_as_of" in e for e in res.errors)

    def test_empty_primary_commodity_detected(self, tmp_path):
        p = tmp_path / "comm.csv"
        p.write_text(
            "ticker,yf_ticker,name,country,stage,primary_commodity\n"
            "AAA,AAA,A,US,producer,\n")
        res = validate_resource_universe(p)
        assert not res.ok
        assert any("primary_commodity" in e for e in res.errors)


class TestNordicUnchanged:
    def test_result_defaults_have_blank_enrichment(self):
        from contrarian_alpha.engine import ContrairianAlphaResult
        r = ContrairianAlphaResult(
            ticker="ABB", ins_id=1, name="ABB", market="SE",
            sector="Industrials", branch="Electrical", composite_score=0.0,
        )
        assert r.resource_data_quality == ""
        assert r.commodity_proxy == ""
        assert r.resource_data_as_of == ""
