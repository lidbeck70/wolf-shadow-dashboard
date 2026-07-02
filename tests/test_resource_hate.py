"""
Tests for the resource-universe HATE-gate fix.

Regression cover for the follow-up "67 scanned, 0 results — all killed at HATE"
bug (screenshot after PR #12): the necessity fix let US/CA resource rows past the
NECESSITY gate, but in deep_contrarian mode the HATE filter then hard-eliminated
every row because these tickers are absent from Börsdata and carry no sentiment /
analyst / short / price context, so their Hat Score defaults to ~0 ("Hat 0.0 < 45
(not sufficiently hated/neglected) [deep_contrarian mode]").

The fix converts a below-threshold Hat into a transparent flag for the
us_ca_resource universe only, so the curated watchlist survives and can be ranked
by resource_composite / the overlay. Nordic deep_contrarian behavior must stay
identical (a below-threshold Hat still hard-eliminates at HATE).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrarian_alpha.engine import _run_single_ticker, PipelineConfig


def _resource_inst(name, stage, commodity):
    return {
        "name": name,
        "marketId": 12,
        "instrumentType": 1,
        "stage": stage,
        "primary_commodity": commodity,
        "resource_meta": {"stage": stage, "primary_commodity": commodity},
    }


def _run_resource(ticker, stage, commodity, mode="deep_contrarian"):
    cfg = PipelineConfig(universe="us_ca_resource", mode=mode)
    # ins_id=None + empty fund_snap + no price_df == the offline, data-sparse
    # US/CA row the live pipeline builds (no network touched here). With no price
    # data the Hat Score is 0.0 — exactly the screenshot case.
    return _run_single_ticker(
        ticker, None, _resource_inst(ticker, stage, commodity),
        {}, None, "", "", cfg, None,
    )


class TestResourceHateNotEliminated:
    def test_missing_hate_data_flags_not_eliminates(self):
        # The core regression: a data-sparse resource row (Hat 0.0) in
        # deep_contrarian mode must survive as a flagged watchlist row.
        for ticker, stage, commodity in [
            ("NEM",  "producer", "gold"),
            ("GOLD", "producer", "gold"),
            ("CCJ",  "producer", "uranium"),
            ("FCX",  "producer", "copper"),
        ]:
            r = _run_resource(ticker, stage, commodity)
            assert not r.eliminated, (
                f"{ticker} wrongly eliminated at {r.elimination_stage}: "
                f"{r.elimination_reason}"
            )
            assert r.elimination_stage != "HATE"
            assert "HATE_BELOW_THRESHOLD" in r.all_flags
            assert "HATE_DATA_MISSING" in r.all_flags

    def test_survivor_gets_resource_composite(self):
        # A row that survives the HATE gate must reach PR3 resource scoring so the
        # watchlist can rank it despite the missing hate signal.
        r = _run_resource("CCJ", "producer", "uranium")
        assert r.resource_composite > 0.0

    def test_no_hate_elimination_stage_in_small_universe(self):
        # A handful of data-sparse deep_contrarian resource rows must produce
        # survivors, never an all-eliminated (0-result) list killed at HATE.
        rows = [
            _run_resource("NEM",  "producer", "gold"),
            _run_resource("GOLD", "producer", "gold"),
            _run_resource("CCJ",  "producer", "uranium"),
        ]
        assert all(not r.eliminated for r in rows)
        assert all(r.elimination_stage != "HATE" for r in rows)


class TestNordicHateUnchanged:
    def test_nordic_low_hate_still_hard_eliminates(self):
        # Nordic energy name (GICS 10) clears necessity, but with no price/sentiment
        # data the Hat Score stays below 45 → must still hard-eliminate at HATE in
        # deep_contrarian mode, with no resource watchlist flags.
        cfg = PipelineConfig(universe="nordic", mode="deep_contrarian")
        inst = {"name": "Equinor", "marketId": 4, "instrumentType": 1, "sectorId": 10}
        r = _run_single_ticker("EQNR.OL", 456, inst, {}, None, "", "", cfg, None)
        assert r.eliminated
        assert r.elimination_stage == "HATE"
        assert "HATE_BELOW_THRESHOLD" not in r.all_flags
        assert "HATE_DATA_MISSING" not in r.all_flags
