"""
Automatisering runda 2, beslut 30 — Tiggre-positionerna får rutan Antal aktier.

_promote lägger positionen med antal 0, och Kapital i risk visar då 'Antal saknas'. Rutan Antal aktier under
positionens tal sparas bara när du klickar Spara antal (positions.put); att öppna sidan eller skriva i rutan sparar
ingenting.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_automatisering_batch11 import ASOF, _held, _metric, _tiggre_stores, page   # noqa: E402,F401

_BLOB = {"rows": {"tiggre:t1": {"ticker": "LOB.TO", "price": 1.5, "currency": "CAD", "asof": ASOF}}}


def _shares_box(at):
    hits = [n for n in at.number_input if n.key.startswith("tg_p_shares_t1_")]
    assert len(hits) == 1, [n.key for n in at.number_input]
    return hits[0]


def test_the_shares_box_saves_only_on_click(page):
    at = page("Tiggre", _BLOB, _tiggre_stores(held=[_held()]))
    at.run()
    assert not at.exception, at.exception
    assert _metric(at, "Kapital i risk") == "Antal saknas"
    assert at.button(key="tg_p_shares_save_t1").disabled                      # inget att spara
    _shares_box(at).set_value(2000.0).run()
    assert not at.exception, at.exception
    assert not page.saves                                                      # att skriva sparar inget
    assert at.session_state["holdings"]["tiggre"][0]["shares"] == 0
    assert "Antalet är inte sparat — klicka Spara antal." in " ".join(c.value for c in at.caption)
    assert _metric(at, "Kapital i risk") == "Antal saknas"                    # räknas på det sparade
    at.button(key="tg_p_shares_save_t1").click().run()
    assert not at.exception, at.exception
    row = at.session_state["holdings"]["tiggre"][0]
    assert row["shares"] == 2000 and row["extras"]["current"] == 1.1 and row["entry_price"] == 1.0
    assert [n for n, _d in page.saves] == ["holdings"]
    assert _metric(at, "Kapital i risk") == "2 000 CAD"
    assert _shares_box(at).value == 2000.0 and at.button(key="tg_p_shares_save_t1").disabled


def test_the_shares_box_shows_the_saved_number(page):
    at = page("Tiggre", _BLOB, _tiggre_stores(held=[_held(shares=1500)]))
    at.run()
    assert not at.exception, at.exception
    assert _shares_box(at).value == 1500.0
    assert not page.saves
