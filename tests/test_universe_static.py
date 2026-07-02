"""
Tests for the static US/CA resource universe (PR1 foundation).

Verifies the loader, CSV integrity, and that engine._build_universe wires the
resource branch without touching Nordic behavior or scoring.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestUniverseStaticLoader:
    def test_import(self):
        from contrarian_alpha.universe_static import (
            load_resource_universe, ResourceRecord, VALID_STAGES, UNIVERSE_KEY,
        )
        assert callable(load_resource_universe)
        assert UNIVERSE_KEY == "us_ca_resource"

    def test_seed_csv_loads(self):
        from contrarian_alpha.universe_static import load_resource_universe
        recs = load_resource_universe()
        assert 50 <= len(recs) <= 80, f"seed size out of expected range: {len(recs)}"

    def test_stages_are_conservative(self):
        from contrarian_alpha.universe_static import (
            load_resource_universe, VALID_STAGES,
        )
        for r in load_resource_universe():
            assert r.stage in VALID_STAGES, f"{r.ticker} has invalid stage {r.stage}"

    def test_no_duplicate_tickers(self):
        from contrarian_alpha.universe_static import load_resource_universe
        tickers = [r.yf_ticker for r in load_resource_universe()]
        assert len(tickers) == len(set(tickers))

    def test_metadata_shape(self):
        from contrarian_alpha.universe_static import load_resource_universe
        meta = load_resource_universe()[0].to_metadata()
        for key in ("stage", "primary_commodity", "exchange", "country", "yf_ticker"):
            assert key in meta

    def test_missing_file_raises(self):
        from contrarian_alpha.universe_static import load_resource_universe
        with pytest.raises(FileNotFoundError):
            load_resource_universe("/nonexistent/does_not_exist.csv")

    def test_missing_columns_raises(self, tmp_path):
        from contrarian_alpha.universe_static import load_resource_universe
        bad = tmp_path / "bad.csv"
        bad.write_text("ticker,foo\nAAA,bar\n")
        with pytest.raises(ValueError):
            load_resource_universe(bad)


class TestEngineWiring:
    def test_config_default_universe_is_nordic(self):
        from contrarian_alpha.engine import PipelineConfig
        assert PipelineConfig().universe == "nordic"

    def test_build_universe_resource_branch(self):
        from contrarian_alpha.engine import PipelineConfig, _build_universe
        cfg = PipelineConfig(universe="us_ca_resource")
        universe = _build_universe(cfg, None)
        assert len(universe) >= 50
        entry = universe[0]
        assert entry["ins_id"] is None
        assert entry["inst_info"]["stage"]
        assert entry["inst_info"]["resource_meta"]["yf_ticker"] == entry["ticker"]

    def test_nordic_branch_empty_without_api(self):
        from contrarian_alpha.engine import PipelineConfig, _build_universe
        # No API + no manual tickers → empty Nordic universe (unchanged behavior).
        assert _build_universe(PipelineConfig(), None) == []

    def test_resource_branch_honours_manual_tickers(self):
        from contrarian_alpha.engine import PipelineConfig, _build_universe
        cfg = PipelineConfig(universe="us_ca_resource", manual_tickers=["ZZZZ"])
        tickers = {u["ticker"] for u in _build_universe(cfg, None)}
        assert "ZZZZ" in tickers
