"""
REGIME → Råvaror → 🥇🥈 Guld/Silver: kvoten, historiken, revalveringen och
referenskvoterna. Specens testfall 1–13 plus datalagret och sidan.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from gold_silver import config as gc  # noqa: E402
from gold_silver import data as gd  # noqa: E402
from gold_silver import engine as ge  # noqa: E402


# ── 1–7: kvoten och silverpriset per kvot ────────────────────────────────────
def test_ratio_calculation():
    assert ge.ratio(4000, 50) == 80.0 and ge.ratio(3300, 50) == 66.0


@pytest.mark.parametrize("r,silver", [(66, 75.76), (50, 100.0), (40, 125.0), (30, 166.67), (20, 250.0),
                                      (19, 263.16)])
def test_implied_silver_per_ratio(r, silver):
    assert round(ge.implied_silver(5000, r), 2) == silver


def test_formula_is_written_out():
    assert ge.formula(5000, 40) == "5,000 / 40 = 125.00"


# ── 8–9: guld och silver ändras ──────────────────────────────────────────────
def test_gold_and_silver_changes_move_the_table():
    a = {r.ratio: r for r in ge.revaluation_table(5000, 75.0, (50, 40))}
    b = {r.ratio: r for r in ge.revaluation_table(6000, 75.0, (50, 40))}
    assert a[40].silver == 125.0 and b[40].silver == 150.0                     # guld upp → silver per kvot upp
    c = {r.ratio: r for r in ge.revaluation_table(5000, 100.0, (40,))}
    assert a[40].change_pct == 66.7 and c[40].change_pct == 25.0                # silver upp → mindre uppsida


def test_revaluation_starts_with_today_and_marks_the_reference():
    rows = ge.revaluation_table(4950, 75.0)                                     # kvot 66
    assert rows[0].is_current and rows[0].ratio == 66.0 and rows[0].change_pct == 0.0
    ref = next(r for r in rows if r.ratio == 19.0)
    assert ref.is_reference and ref.silver == 260.53 and ref.formula == "4,950 / 19 = 260.53"
    assert [r.ratio for r in rows] == [66.0, 60.0, 50.0, 40.0, 30.0, 25.0, 20.0, 19.0]


# ── 10–13: saknat och ogiltigt ───────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, 0, -5, "", "abc", float("nan")])
def test_missing_zero_or_invalid_prices_give_none(bad):
    assert ge.ratio(bad, 50) is None and ge.ratio(5000, bad) is None
    assert ge.implied_silver(bad, 40) is None
    assert ge.geological_gap(bad, 50) is None


@pytest.mark.parametrize("bad_ratio", [0, -19, None, "x"])
def test_negative_or_invalid_ratio(bad_ratio):
    assert ge.implied_silver(5000, bad_ratio) is None
    assert ge.formula(5000, bad_ratio) == "ogiltigt guldpris eller kvot"
    assert all(r.ratio > 0 for r in ge.revaluation_table(5000, 75.0, (bad_ratio, 40)))
    assert ge.geological_gap(5000, 75.0, bad_ratio) is None


def test_revaluation_without_silver_still_gives_prices_but_no_change():
    rows = ge.revaluation_table(5000, None, (40,))
    assert rows[0].silver == 125.0 and rows[0].change is None and rows[0].change_pct is None


def test_history_unavailable():
    assert ge.period_stats(pd.Series(dtype=float), "1 år", 1) is None
    assert ge.all_periods(pd.Series(dtype=float)) == []


# ── historik ────────────────────────────────────────────────────────────────
def _series(years=12, start=80.0, end=60.0):
    idx = pd.bdate_range(end="2026-09-30", periods=252 * years)
    return pd.Series([start + (end - start) * i / (len(idx) - 1) for i in range(len(idx))], index=idx)


def test_period_stats_and_only_complete_periods():
    s = _series(12)
    stats = ge.all_periods(s)
    assert [x.label for x in stats] == ["1 år", "5 år", "10 år", "Max"]           # 20 år saknas → visas inte
    ten = next(x for x in stats if x.years == 10)
    assert ten.current == 60.0 and ten.min == 60.0 and ten.percentile == 0 and ten.p10 < ten.median < ten.p90
    assert ten.start.startswith("2016")


def test_position_is_neutral_and_guide_zones():
    assert ge.position(66, 64) == "Nära historisk median"
    assert ge.position(90, 70) == "Över historisk median" and ge.position(50, 70) == "Under historisk median"
    assert "ackumuleringszon" in ge.guide_zone(88) and "sencykliska" in ge.guide_zone(45)
    assert ge.guide_zone(66).startswith("Mellan") and ge.guide_zone(None) is None


def test_geological_gap_language():
    g = ge.geological_gap(4950, 75.0)
    assert (g.current, g.reference, g.difference, g.pct_of_current) == (66.0, 19.0, 47.0, 71.2)
    assert g.implied_silver == 260.53 and g.upside_pct == 247.4


def test_reference_ratios_are_separate_and_never_invented():
    assert round(ge.production_ratio(), 1) == 7.6                                # 25 000 t / 3 300 t
    assert ge.above_ground_ratio() is None                                       # silversidan saknas → Partial data
    assert gc.REFERENCES["geological"]["kind"] == "ESTIMATE"
    for ref in gc.REFERENCES.values():
        assert ref["source"] and ref["kind"] in ("ACTUAL", "ESTIMATE")


def test_matrix():
    rows = dict(ge.matrix())
    assert list(rows) == [4000.0, 5000.0, 6000.0, 7000.0, 8000.0]
    assert rows[5000.0][40.0] == 125.0 and rows[8000.0][19.0] == 421.05


def test_ratio_series_aligns_and_drops_bad_days():
    idx = pd.bdate_range("2026-01-01", periods=4)
    g = pd.Series([4000.0, 4100.0, 0.0, 4200.0], index=idx)
    s = pd.Series([50.0, 0.0, 60.0, 70.0], index=idx)
    r = ge.ratio_series(g, s)
    assert list(r.values) == [80.0, 60.0] and len(ge.ratio_series(None, s)) == 0


# ── datalagret ───────────────────────────────────────────────────────────────
def test_fetch_carries_provenance_and_reuses_the_getter():
    calls = []
    idx = pd.bdate_range("2025-01-01", periods=300)
    series = {"GC=F": pd.Series([4000.0] * 300, index=idx), "SI=F": pd.Series([50.0] * 300, index=idx)}

    def getter(t, p):
        calls.append((t, p))
        return series[t]
    d = gd.fetch(getter)
    assert calls == [("GC=F", "max"), ("SI=F", "max")] and d["error"] is None
    assert d["gold"]["value"] == 4000.0 and d["gold"]["kind"] == "ACTUAL" and "GC=F" in d["gold"]["source"]
    assert float(d["series"].iloc[-1]) == 80.0
    empty = gd.fetch(lambda t, p: pd.Series(dtype=float))
    assert empty["gold"] is None and "skriv in priserna själv" in empty["error"]


# ── sidan ────────────────────────────────────────────────────────────────────
def test_page_renders_cards_history_and_engine(monkeypatch):
    from streamlit.testing.v1 import AppTest
    s = _series(22, 90.0, 66.0)
    fake = {"gold": {"value": 4950.0, "source": "Yahoo Finance GC=F", "date": "2026-09-30", "kind": "ACTUAL"},
            "silver": {"value": 75.0, "source": "Yahoo Finance SI=F", "date": "2026-09-30", "kind": "ACTUAL"},
            "series": s, "error": None}
    monkeypatch.setattr(gd, "fetch", lambda getter=None: fake)
    monkeypatch.setenv("GS_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["GS_TEST_ROOT"])
        from gold_silver.ui import render_gold_silver_page
        render_gold_silver_page()

    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "GULD / SILVER" in html and ">66.0<" in html and "~19" in html
    assert "not a market equilibrium value" in html and "HISTORISK MEDIAN" in html
    assert "20 år" in html and "Nära historisk median" in html          # 66 mot 20-års-median
    assert "REFERENCE SCENARIO" in html and "4,950 / 40 = 123.75" in html
    assert "DATA UNAVAILABLE" not in html and "7.6 : 1" in html and "Partial data" in html
    assert "fair value" in html and "vad som händer om marknadskvoten ändras" in html
    assert any("VECKOVIS" in c.proto.spec for c in at.get("plotly_chart"))
    # egna värden: silver 100 → kvot 49.5
    at.number_input(key="gs_silver").set_value(100.0).run()
    html = " ".join(m.value for m in at.markdown)
    assert ">49.5<" in html and "egen inmatning · ASSUMPTION" in html
