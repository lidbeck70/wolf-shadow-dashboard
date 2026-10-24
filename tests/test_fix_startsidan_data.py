"""Granskning 2 (data/motorer): storleken i rätt valuta, startsidan ärlig om saknad data, Norden-ljuset, köpfönstret."""
import pandas as pd

import positions
from odin import laget as lg
from tabs import home_today as ht
from thor import candidates as tc
from thor import storlek as sk


def test_size_converts_the_listing_currency():
    ccy, fx = sk.fx_for("NOKIA.HE")
    assert ccy == "EUR" and fx == positions.FX_TO_SEK["EUR"]
    sz = sk.size(10.0, 9.0, 100000, risk=2.0, cap=10.0, fx=fx, currency=ccy)
    assert sz["risk_shares"] * 1.0 * fx <= sz["risk_amt"]                       # verklig risk ≤ budget
    assert sz["cap_shares"] * 10.0 * fx <= sz["cap_amt"]
    assert " EUR" in sk.row_text("NOKIA.HE", "Wolf", sz)
    assert sk.size(10.0, 9.0, 100000, risk=2.0)["risk_shares"] == 2000          # kronor som förut


def test_gul_regime_halves_and_own_target_is_kept():
    sz = sk.halved(sk.size(100, 90, 100000, risk=2.0, cap=10.0))
    assert sz["risk_shares"] == 100 and sz["cap_shares"] == 50 and "halverat" in sz["formula"]
    assert sk.position_args("X", "X", "ember", sz, 10, target=58.0)["target"] == 58.0
    assert sk.position_args("X", "X", "ember", sz, 10)["target"] == sz["target_3r"]


def test_nordic_light_reads_the_saved_board_shape():
    saved = {"board": {"segments": [{"key": "a", "nordic": {"status": "GRÖN"}}, {"key": "b", "nordic": {"status": "RÖD"}},
                                    {"key": "c", "nordic": {"status": "GRÖN"}}]}}
    assert lg.nordic_light(saved) == ("2/3 GRÖN", "GRÖN")


def test_closed_turn_windows_are_not_shown():
    board = {"turn_plan": {"window_days": 5, "window": [
        {"label": "Koppar", "kind": "VÄNDER", "confirmed": True, "confirmed_since": "2026-10-01"},
        {"label": "Guld", "kind": "VÄNDER", "confirmed": True, "confirmed_since": "2026-10-20"}]}}
    cards = ht.turn_cards(board, today=pd.Timestamp("2026-10-23"))
    assert [c["label"] for c in cards] == ["Guld"] and cards[0]["day"] == 3


def test_stale_rows_are_marked():
    c = tc.Candidate("ERIC-B.ST", "Ericsson", "momentum", "rad 1", tc.KOP, stale=True, asof="2026-10-01")
    assert ht.stale_tag(c) == " · GAMMAL (körd 2026-10-01)"


def test_hold_line_is_honest():
    assert ht.all_hold_line(None, 7).startswith("Säljläget saknas")
    state = {"swing": {"verdicts": {"ERIC-B.ST": {"verdict": "HOLD"}}}}
    line = ht.all_hold_line(state, 7)
    assert line.startswith("Alla 1 Momentum-positioner: HÅLL") and "övriga 6" in line


def test_reload_rules():
    now = pd.Timestamp("2026-10-24 09:00")
    assert not ht.needs_reload("2026-10-24 08:45", now)
    assert ht.needs_reload("2026-10-24 08:00", now) and ht.needs_reload("2026-10-23 23:59", now)
    assert not ht.needs_reload(None, now)
