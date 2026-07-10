"""
Tests for the Contrarian Alpha score-breakdown chart data prep
(contrarian_alpha.ui.build_breakdown_data / _finite).

Regression guard for the US/CA resource crash: detail cards must never feed
None/NaN/non-numeric sub-scores into Plotly. The pure helper coerces every
value to a finite float and guarantees equal-length arrays, so the chart can
always be rendered (or cleanly skipped when there is no data).

`streamlit` is stubbed so the pure helper is importable even where the UI's
heavy deps are not installed (the helper itself uses neither streamlit nor
plotly).
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub streamlit before importing the UI module (import-time `import streamlit`).
sys.modules.setdefault("streamlit", MagicMock())

from contrarian_alpha.ui import build_breakdown_data, _finite  # noqa: E402


def _row(**kw):
    """A result-like object; unspecified score fields fall back via getattr."""
    return SimpleNamespace(**kw)


class TestFinite:
    def test_passes_through_finite_float(self):
        assert _finite(42.5) == 42.5

    def test_int_becomes_float(self):
        assert _finite(7) == 7.0

    def test_none_uses_default(self):
        assert _finite(None) == 0.0
        assert _finite(None, default=-1.0) == -1.0

    def test_nan_uses_default(self):
        assert _finite(float("nan")) == 0.0

    def test_inf_uses_default(self):
        assert _finite(float("inf")) == 0.0
        assert _finite(float("-inf")) == 0.0

    def test_non_numeric_string_uses_default(self):
        assert _finite("not-a-number") == 0.0

    def test_numeric_string_is_coerced(self):
        assert _finite("12.5") == 12.5


def _full_row(**kw):
    """A row with every pillar present (defaults to a mid score) so array-length
    and index-based assertions have all six pillars to work with."""
    base = dict(
        necessity_score=50.0, hat_score=50.0, quality_score=50.0,
        value_score=50.0, catalyst_score=50.0, viking_bonus_raw=50.0,
        composite_score=50.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestBuildBreakdownData:
    def test_arrays_are_equal_length(self):
        # A full row keeps all six pillars; arrays must stay equal-length.
        d = build_breakdown_data(_full_row(), "quality")
        n = len(d["labels"])
        assert n == 6
        assert len(d["scores_raw"]) == n
        assert len(d["contributions"]) == n
        assert len(d["max_contribs"]) == n

    def test_missing_pillars_are_skipped(self):
        # Only necessity/hate present -> exactly those two pillars are kept,
        # arrays stay aligned, and the chart still has data.
        r = _row(necessity_score=55.0, hat_score=45.0)
        d = build_breakdown_data(r, "quality")
        assert d["labels"] == ["Necessity", "Hat"]
        n = len(d["labels"])
        assert len(d["scores_raw"]) == n
        assert len(d["contributions"]) == n
        assert len(d["max_contribs"]) == n
        assert d["has_data"] is True

    def test_empty_row_has_no_pillars(self):
        # Nothing present -> nothing drawn (no misleading zero bars).
        d = build_breakdown_data(_row(), "quality")
        assert d["labels"] == []
        assert d["contributions"] == []
        assert d["has_data"] is False

    def test_normal_nordic_row_contributions(self):
        r = _row(
            necessity_score=80.0, hat_score=60.0, quality_score=70.0,
            value_score=50.0, catalyst_score=40.0, viking_bonus_raw=100.0,
            composite_score=64.0,
        )
        d = build_breakdown_data(r, "quality")
        # quality weight 0.30 in quality mode -> 70 * 0.30 = 21.0
        assert d["contributions"][2] == 21.0
        # viking 100 * 0.05 = 5.0
        assert d["contributions"][5] == 5.0
        assert d["total"] == 64.0
        assert d["has_data"] is True

    def test_present_zero_pillar_is_kept(self):
        # A pillar that is genuinely 0.0 (present, not missing) is still drawn,
        # while absent pillars are skipped.
        r = _row(necessity_score=55.0, hat_score=45.0, quality_score=0.0)
        d = build_breakdown_data(r, "quality")
        assert d["labels"] == ["Necessity", "Hat", "Quality"]
        assert d["contributions"][2] == 0.0  # quality present but zero -> kept
        assert d["has_data"] is True

    def test_none_and_nan_are_coerced(self):
        r = _row(
            necessity_score=None, hat_score=float("nan"),
            quality_score="x", value_score=None,
            catalyst_score=float("inf"), viking_bonus_raw=None,
            composite_score=float("nan"),
        )
        d = build_breakdown_data(r, "quality")
        assert all(isinstance(c, float) for c in d["contributions"])
        assert all(c == 0.0 for c in d["contributions"])
        assert d["total"] == 0.0
        assert d["has_data"] is False

    def test_all_zero_row_has_no_data(self):
        r = _row(
            necessity_score=0.0, hat_score=0.0, quality_score=0.0,
            value_score=0.0, catalyst_score=0.0, viking_bonus_raw=0.0,
            composite_score=0.0,
        )
        d = build_breakdown_data(r, "quality")
        assert d["has_data"] is False

    def test_deep_contrarian_weights_differ(self):
        # Full row so pillar indices are stable across both modes.
        r = _full_row(hat_score=100.0, quality_score=100.0)
        q = build_breakdown_data(r, "quality")
        deep = build_breakdown_data(r, "deep_contrarian")
        # Hat weight: 0.20 (quality) vs 0.30 (deep_contrarian)
        assert q["contributions"][1] == 20.0
        assert deep["contributions"][1] == 30.0
        # Quality weight: 0.30 (quality) vs 0.20 (deep_contrarian)
        assert q["contributions"][2] == 30.0
        assert deep["contributions"][2] == 20.0

    def test_max_contribs_are_finite_and_positive(self):
        d = build_breakdown_data(_full_row(), "quality")
        assert all(m > 0 for m in d["max_contribs"])
        # Viking max is a flat 5.0p
        assert d["max_contribs"][5] == 5.0
