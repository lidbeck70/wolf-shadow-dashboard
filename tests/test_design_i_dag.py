"""Designförslaget PR 8: 🏠 I DAG — vändningen överst, sedan kapital, kvalar, storlek, sälj och mål. Bara sparat."""
import pathlib

import pandas as pd

from tabs import home_today as ht
from thor import candidates as tc

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOARD = {"turn_plan": {"window_days": 5, "window": [
    {"level": "komplex", "key": "basmetaller", "label": "Basmetaller", "kind": "VÄNDER",
     "confirmed": True, "confirmed_since": "2026-10-22", "deep": [{"ticker": "BOL.ST"}, {"ticker": "LUMI.ST"}],
     "long": [{"ticker": "SSAB-B.ST"}], "etf": {"ticker": "COPX.L", "exposure": "koppargruvor", "verified": False}}]},
    "turns": [{"confirmed": True}, {"confirmed": False}, {"confirmed": False}]}


def test_turn_card_counts_days_from_confirmation():
    c = ht.turn_cards(BOARD, today=pd.Timestamp("2026-10-23"))[0]
    assert ht.turn_headline(c) == "⚡ VÄNDNING — Basmetaller · dag 1 av 5"
    c0 = ht.turn_cards(BOARD, today=pd.Timestamp("2026-10-22"))[0]
    assert ht.turn_headline(c0) == "⚡ VÄNDNING — Basmetaller · bekräftad i dag"
    assert c["deep"] == ["BOL.ST", "LUMI.ST"] and c["long"] == ["SSAB-B.ST"] and c["etf"]["ticker"] == "COPX.L"


def test_no_turn_shows_how_many_are_on_the_way():
    assert ht.turn_cards({}) == []
    assert ht.no_turn_line(BOARD) == "Ingen vändning i köpfönstret · 2 på väg"
    assert ht.no_turn_line({}) == "Ingen vändning i köpfönstret"


def _c(t, cls, score, strat="viking_nine", entry=100.0, stop=92.0):
    return tc.Candidate(t, t, strat, "x", cls, score=score, entry=entry, stop=stop)


def test_qualifiers_use_thor_order_and_top_five():
    rows = [_c(f"B{i}", tc.KOP, 50 + i) for i in range(7)] + [_c("W", tc.BEVAKA, 99)]
    n_buy, n_watch, top = ht.qualifiers(rows)
    assert (n_buy, n_watch) == (7, 1)
    assert [c.ticker for c in top] == ["B6", "B5", "B4", "B3", "B2"]


def test_size_line_uses_the_size_row():
    line = ht.size_line(_c("ABC.ST", tc.KOP, 80), 100000)
    assert line.startswith("ABC.ST · Viking Nine · köp 100.00 · stopp 92.00 · mål 3R 124.00")
    assert "nivåer saknas" in ht.size_line(_c("X", tc.KOP, 1, entry=None), 100000)


def test_holdings_lines_from_saved_verdicts():
    state = {"swing": {"verdicts": {"EKTA-B.ST": {"verdict": "SELL", "reason": "stängning under MA50"},
                                    "AAA.ST": {"verdict": "HOLD", "reason": ""},
                                    "BBB.ST": {"verdict": "PARTIAL", "reason": "+20 %"}}}}
    assert ht.holdings_lines(state) == [("EKTA-B.ST", "SÄLJ", "stängning under MA50"), ("BBB.ST", "MINSKA", "+20 %")]
    assert ht.all_hold_line({"swing": {"verdicts": {"A": {}, "B": {}}}}, 2) == "Alla 2 Momentum-positioner: HÅLL"


def test_load_never_fails_and_reads_only_saved_data():
    d = ht.load(loader=lambda name, default=None: None)
    assert d["board"] is None and "rows" in d["thor"]
    src = (ROOT / "tabs" / "home_today.py").read_text(encoding="utf-8")
    assert "evaluate(" not in src and "ohlcv" not in src and "requests" not in src
