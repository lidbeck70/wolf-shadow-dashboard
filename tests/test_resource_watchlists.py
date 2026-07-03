"""
Tests for contrarian_alpha.resource_watchlists.build_resource_watchlists (PR15).

Pure helper — no Streamlit / pandas — so it is importable and testable without
the UI's heavy deps. Covers each watchlist category, missing-column/None
tolerance, empty input, and that non-resource-shaped rows produce no false
positives in the flag-based views.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

from contrarian_alpha.resource_watchlists import (  # noqa: E402
    build_resource_watchlists,
    WATCHLIST_ORDER,
    WATCHLIST_META,
    WATCHLIST_COLUMNS,
    DEEP_DRAWDOWN_PCT,
    HIGH_COMPOSITE,
)


def _row(**kw):
    """A ContrairianAlphaResult-like object; unset fields fall back via getattr."""
    return SimpleNamespace(**kw)


class TestStructure:
    def test_returns_all_views_in_order(self):
        views = build_resource_watchlists([])
        assert list(views.keys()) == WATCHLIST_ORDER

    def test_meta_covers_all_views(self):
        for key in WATCHLIST_ORDER:
            assert key in WATCHLIST_META
            assert WATCHLIST_META[key]["title"]
            assert WATCHLIST_META[key]["caption"]

    def test_empty_input(self):
        for rows in build_resource_watchlists([]).values():
            assert rows == []

    def test_none_input(self):
        for rows in build_resource_watchlists(None).values():
            assert rows == []


class TestTopComposite:
    def test_sorted_descending(self):
        rows = [
            _row(ticker="A", resource_composite=40.0),
            _row(ticker="B", resource_composite=80.0),
            _row(ticker="C", resource_composite=60.0),
        ]
        out = build_resource_watchlists(rows)["top_composite"]
        assert [r["ticker"] for r in out] == ["B", "C", "A"]

    def test_missing_composite_sorts_last(self):
        rows = [
            _row(ticker="A", resource_composite=None),
            _row(ticker="B", resource_composite=50.0),
        ]
        out = build_resource_watchlists(rows)["top_composite"]
        assert out[0]["ticker"] == "B"
        assert out[-1]["ticker"] == "A"

    def test_includes_all_rows(self):
        rows = [_row(ticker="A"), _row(ticker="B")]
        assert len(build_resource_watchlists(rows)["top_composite"]) == 2


class TestDeepDrawdownStrongRs:
    def test_matches_deep_dd_and_strong_rs(self):
        rows = [
            _row(ticker="A", drawdown_52w_pct=-55.0,
                 commodity_rs_flag="OUTPERFORMING_PROXY"),
            _row(ticker="B", drawdown_52w_pct=-45.0,
                 commodity_rs_flag="RS_NEUTRAL"),
        ]
        out = build_resource_watchlists(rows)["deep_drawdown_strong_rs"]
        assert {r["ticker"] for r in out} == {"A", "B"}

    def test_shallow_drawdown_excluded(self):
        rows = [_row(ticker="A", drawdown_52w_pct=-10.0,
                     commodity_rs_flag="OUTPERFORMING_PROXY")]
        assert build_resource_watchlists(rows)["deep_drawdown_strong_rs"] == []

    def test_lagging_rs_excluded(self):
        rows = [_row(ticker="A", drawdown_52w_pct=-60.0,
                     commodity_rs_flag="LAGGING_PROXY")]
        assert build_resource_watchlists(rows)["deep_drawdown_strong_rs"] == []

    def test_missing_drawdown_excluded(self):
        rows = [_row(ticker="A", commodity_rs_flag="OUTPERFORMING_PROXY")]
        assert build_resource_watchlists(rows)["deep_drawdown_strong_rs"] == []

    def test_threshold_boundary_inclusive(self):
        rows = [_row(ticker="A", drawdown_52w_pct=DEEP_DRAWDOWN_PCT,
                     commodity_rs_flag="RS_NEUTRAL")]
        assert len(build_resource_watchlists(rows)["deep_drawdown_strong_rs"]) == 1


class TestOutperformingProxy:
    def test_matches_only_outperforming(self):
        rows = [
            _row(ticker="A", commodity_rs_flag="OUTPERFORMING_PROXY"),
            _row(ticker="B", commodity_rs_flag="RS_NEUTRAL"),
            _row(ticker="C", commodity_rs_flag=""),
        ]
        out = build_resource_watchlists(rows)["outperforming_proxy"]
        assert [r["ticker"] for r in out] == ["A"]


class TestHighScoreLowQuality:
    def test_high_score_low_dq(self):
        rows = [_row(ticker="A", resource_composite=75.0,
                     resource_data_quality="LOW")]
        assert len(build_resource_watchlists(rows)["high_score_low_quality"]) == 1

    def test_high_score_missing_data_flag(self):
        rows = [_row(ticker="A", resource_composite=80.0,
                     resource_data_quality="HIGH",
                     resource_flags=["SURVIVAL_DATA_MISSING"])]
        assert len(build_resource_watchlists(rows)["high_score_low_quality"]) == 1

    def test_high_score_clean_data_excluded(self):
        rows = [_row(ticker="A", resource_composite=80.0,
                     resource_data_quality="HIGH")]
        assert build_resource_watchlists(rows)["high_score_low_quality"] == []

    def test_low_score_excluded(self):
        rows = [_row(ticker="A", resource_composite=HIGH_COMPOSITE - 1,
                     resource_data_quality="LOW")]
        assert build_resource_watchlists(rows)["high_score_low_quality"] == []


class TestNeedsValidation:
    def test_matches_various_tokens(self):
        rows = [
            _row(ticker="A", resource_flags=["DATA_AS_OF_MISSING"]),
            _row(ticker="B", existing_source_flags=["LOW_DATA_CONFIDENCE"]),
            _row(ticker="C", resource_notes="needs_validation"),
            _row(ticker="D", all_flags=["DILUTION_DATA_MISSING"]),
        ]
        out = build_resource_watchlists(rows)["needs_validation"]
        assert {r["ticker"] for r in out} == {"A", "B", "C", "D"}

    def test_clean_row_excluded(self):
        rows = [_row(ticker="A", resource_flags=["STRATEGIC_COMMODITY"])]
        assert build_resource_watchlists(rows)["needs_validation"] == []


class TestLowLiquidity:
    def test_low_and_thin_matched(self):
        rows = [
            _row(ticker="A", liquidity_flag="LOW"),
            _row(ticker="B", liquidity_flag="THIN"),
            _row(ticker="C", liquidity_flag="OK"),
            _row(ticker="D", liquidity_flag=""),
        ]
        out = build_resource_watchlists(rows)["low_liquidity"]
        assert {r["ticker"] for r in out} == {"A", "B"}


class TestMissingColumnsAndDicts:
    def test_bare_rows_do_not_crash(self):
        # Rows with essentially no resource fields (Nordic-shaped) must not
        # raise and must not appear in any flag-based view.
        rows = [_row(ticker="NORDIC1"), _row(ticker="NORDIC2")]
        views = build_resource_watchlists(rows)
        assert len(views["top_composite"]) == 2  # composite view keeps all
        for key in WATCHLIST_ORDER:
            if key == "top_composite":
                continue
            assert views[key] == []

    def test_accepts_dict_rows(self):
        rows = [{"ticker": "A", "resource_composite": 90.0,
                 "commodity_rs_flag": "OUTPERFORMING_PROXY"}]
        views = build_resource_watchlists(rows)
        assert views["top_composite"][0]["ticker"] == "A"
        assert len(views["outperforming_proxy"]) == 1

    def test_nan_composite_tolerated(self):
        rows = [_row(ticker="A", resource_composite=float("nan"))]
        out = build_resource_watchlists(rows)["top_composite"]
        assert out[0]["ticker"] == "A"
        assert out[0]["resource_composite"] is None

    def test_row_dict_has_display_columns(self):
        rows = [_row(ticker="A", resource_composite=50.0)]
        row = build_resource_watchlists(rows)["top_composite"][0]
        for col in WATCHLIST_COLUMNS:
            assert col in row
