"""
Temabrädan cachas bara hel: ett tema utan data (Yahoo strulade) töms ur cachen så nästa anrop försöker igen —
annars fick EMBER "cykel okänd" (DATA_GAP, högst BEVAKA) för temats bolag i 12 timmar. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blindspot import theme_board as tb  # noqa: E402


def _patch(monkeypatch, fail_first: set):
    calls = {"n": 0}

    def compute(spec):
        calls["n"] += 1
        if spec.key in fail_first and calls["n"] <= len(tb._THEMES):
            return tb.ThemeResult(key=spec.key, label=spec.label, necessity=spec.necessity, error="ingen kurs")
        return tb.ThemeResult(key=spec.key, label=spec.label, necessity=spec.necessity, cykel_label="TIDIG")

    monkeypatch.setattr(tb, "_compute_theme", compute)
    getattr(tb._build_board_cached, "clear", lambda: None)()
    return calls


def test_a_partial_board_is_not_cached(monkeypatch):
    calls = _patch(monkeypatch, {"koppar"})
    first = {r.key: r for r in tb.build_theme_board()}
    assert first["koppar"].error and first["koppar"].cykel_label == "DATA_GAP"
    second = {r.key: r for r in tb.build_theme_board()}
    assert calls["n"] == 2 * len(tb._THEMES) and second["koppar"].cykel_label == "TIDIG"   # försökte igen


def test_a_full_board_is_cached(monkeypatch):
    calls = _patch(monkeypatch, set())
    tb.build_theme_board()
    tb.build_theme_board()
    assert calls["n"] == len(tb._THEMES)
    getattr(tb._build_board_cached, "clear", lambda: None)()
