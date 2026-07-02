"""
Tests for the resource-universe necessity fix.

Regression cover for the "67 scanned, 0 results" bug: the static US/CA resource
universe carries no GICS codes, so the necessity gate fell through to a flaky
per-ticker yfinance sector lookup that usually returned nothing → FALLBACK_SCORE
(40) < threshold (60) → the whole curated universe was eliminated at NECESSITY.

The fix seeds the necessity sector name from the curated commodity metadata and,
for the resource universe only, converts a below-threshold necessity into a flag
instead of a hard elimination. Nordic behavior must stay identical.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrarian_alpha.engine import _run_single_ticker, PipelineConfig
from contrarian_alpha.necessity import get_necessity_score


def _resource_inst(name, stage, commodity):
    return {
        "name": name,
        "marketId": 12,
        "instrumentType": 1,
        "stage": stage,
        "primary_commodity": commodity,
        "resource_meta": {"stage": stage, "primary_commodity": commodity},
    }


def _run(ticker, stage, commodity, mode="quality"):
    cfg = PipelineConfig(universe="us_ca_resource", mode=mode)
    # ins_id=None + empty fund_snap + no price_df == the offline, data-sparse
    # US/CA row the live pipeline builds (no network touched here).
    return _run_single_ticker(
        ticker, None, _resource_inst(ticker, stage, commodity),
        {}, None, "", "", cfg, None,
    )


class TestCommodityNecessity:
    def test_known_commodities_pass_necessity_without_network(self):
        # Commodity metadata alone must lift necessity above the 60 gate — no
        # yfinance sector call, no fundamentals, no price data required.
        for ticker, stage, commodity in [
            ("CCJ", "producer", "uranium"),
            ("NEM", "producer", "gold"),
            ("XOM", "producer", "oil_gas"),
            ("VALE", "producer", "iron_ore"),
            ("MP", "producer", "rare_earth"),
            ("FCX", "producer", "copper"),
        ]:
            r = _run(ticker, stage, commodity)
            assert not r.eliminated, f"{ticker} wrongly eliminated: {r.elimination_reason}"
            assert r.necessity_score >= 60, f"{ticker} necessity {r.necessity_score}"

    def test_underscore_commodities_are_normalised(self):
        # CSV uses oil_gas / iron_ore / rare_earth; necessity map uses spaces.
        assert get_necessity_score(sector_name="oil gas").score >= 60
        assert get_necessity_score(sector_name="iron ore").score >= 60
        assert get_necessity_score(sector_name="rare earth").score >= 60


class TestWatchlistFallback:
    def test_missing_commodity_metadata_flags_not_eliminates(self):
        # Unrecognised / missing commodity → survives as a flagged watchlist row.
        r = _run("JUNK", "explorer", "")
        assert not r.eliminated
        assert "NECESSITY_BELOW_THRESHOLD" in r.all_flags
        assert "NECESSITY_DATA_MISSING" in r.all_flags

    def test_unrecognised_commodity_uses_distinct_flag(self):
        r = _run("WEIRD", "producer", "unobtainium")
        assert not r.eliminated
        assert "LOW_NECESSITY_UNRECOGNISED" in r.all_flags

    def test_survivor_gets_resource_composite(self):
        # A surviving resource row should carry the PR3 composite for the watchlist.
        r = _run("CCJ", "producer", "uranium")
        assert r.resource_composite > 0.0


class TestNonEmptyUniverse:
    def test_small_resource_universe_yields_watchlist_rows(self):
        # The core regression: a handful of data-sparse resource rows must produce
        # survivors, never an all-eliminated (0-result) list.
        rows = [
            _run("CCJ", "producer", "uranium"),
            _run("NEM", "producer", "gold"),
            _run("XOM", "producer", "oil_gas"),
            _run("JUNK", "explorer", ""),
        ]
        survivors = [r for r in rows if not r.eliminated]
        assert len(survivors) == len(rows)


class TestNordicUnchanged:
    def test_nordic_low_necessity_still_hard_eliminates(self):
        cfg = PipelineConfig(universe="nordic", mode="quality")
        inst = {"name": "SomeSaaS", "marketId": 1, "instrumentType": 1, "sectorId": 45}
        r = _run_single_ticker("SAAS.ST", 123, inst, {}, None, "", "", cfg, None)
        assert r.eliminated
        assert r.elimination_stage == "NECESSITY"
        assert "NECESSITY_BELOW_THRESHOLD" not in r.all_flags

    def test_nordic_high_necessity_passes_gate(self):
        # A Nordic energy name (GICS 10) clears necessity exactly as before.
        cfg = PipelineConfig(universe="nordic", mode="quality")
        inst = {"name": "Equinor", "marketId": 4, "instrumentType": 1, "sectorId": 10}
        r = _run_single_ticker("EQNR.OL", 456, inst, {}, None, "", "", cfg, None)
        # Passes necessity; may still be eliminated later, but NOT at NECESSITY.
        assert r.elimination_stage != "NECESSITY"
