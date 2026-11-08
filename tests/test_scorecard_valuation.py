"""
Scorecard: Värdering föreslås ur Durrett-arket — samma tal och gränser som arket räknar med.

Förslaget visas av sig självt under rutan (MCap/framtida vinst mot köpregeln under 10×, MCap/uns mot $100/$50, och
Börsdatas EV/EBITDA när sifferuppdateringen har den). Det skrivs in först med "Använd": att öppna Scorecardet skriver
ingenting, och inget sparas utan 💾. Ingen grind- eller poängregel är ändrad.
"""
import copy
import json
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scorecard  # noqa: E402
import storage    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = "gold|AU"
DURRETT = {"id": "d1", "ticker": "AU", "name": "Gold", "commodity": "Guld", "date": "2026-09-01",
           "factors": {}, "mcap": 300.0, "moz": 4.0, "profit": 55.0}
STORES = {
    "producers": {"producers": [], "royalty": []},
    "scoring": {"sprott": [dict(DURRETT, id="s1", factors={})], "durrett": [DURRETT]},
    "insider": {"signals": []},
    "tiggre": {"candidates": [], "positions": [], "closed": [], "parked": []},
    "holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0},
    "scorecard": {"cards": {}},
    "confidence": {"companies": {}, "commodity_overrides": {}},
}
TEXT = "MCap/framtida vinst 5,5× (under köpregelns 10×) · MCap/uns $75 (billigt) · EV/EBITDA 4,2× (Börsdata)"


def _entry(**row):
    return {"rows": {"durrett": dict(DURRETT, **row)}}


# ── Den rena funktionen ─────────────────────────────────────────────────────
def test_the_suggestion_uses_the_sheets_own_numbers_and_limits():
    assert scorecard.valuation_suggestion(_entry(), "durrett", {"ev_ebitda": 4.2}) == TEXT
    assert scorecard.valuation_suggestion(_entry(profit=20.0, moz=1.5), "durrett") == (
        "MCap/framtida vinst 15,0× (över köpregelns 10×) · MCap/uns $200 (över $100)")
    assert scorecard.valuation_suggestion(_entry(moz=10.0), "durrett").endswith("MCap/uns $30 (mycket billigt)")


def test_only_what_can_be_computed_is_shown():
    assert scorecard.valuation_suggestion(_entry(moz=None), "durrett") == (
        "MCap/framtida vinst 5,5× (under köpregelns 10×)")
    assert scorecard.valuation_suggestion(_entry(profit=None, moz=None), "durrett", {"ev_ebitda": 4.2}) == (
        "EV/EBITDA 4,2× (Börsdata)")
    assert scorecard.valuation_suggestion(_entry(mcap=None), "durrett") is None
    # Negativa tal är luckor som i arket — aldrig ett tal under 10× som ser ut som ett köp
    assert scorecard.valuation_suggestion(_entry(profit=-5.0, moz=None), "durrett") is None
    assert scorecard.valuation_suggestion(_entry(moz=None, profit=None), "durrett", {"ev_ebitda": -3.0}) is None


def test_other_strategies_get_no_suggestion():
    e = {"rows": {"durrett": DURRETT, "sprott": DURRETT}}
    assert scorecard.valuation_suggestion(e, "sprott") is None
    assert scorecard.valuation_suggestion({"rows": {}}, "durrett") is None
    assert scorecard.valuation_suggestion(None, "durrett") is None


# ── Kortet (AppTest) ────────────────────────────────────────────────────────
def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


@pytest.fixture
def page(monkeypatch):
    """Scorecardet ur en fejkad lagring — inget GitHub, ingen Gist; sifferuppdateringen styrs per test."""
    stores = copy.deepcopy(STORES)
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    import gist_storage
    import refresh_ui
    import screens_ui
    monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
    monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)

    def run(blob=None):
        from streamlit.testing.v1 import AppTest
        monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob or {})

        def app(root):
            import sys as _s
            _s.path.insert(0, root)
            import scorecard as sc
            sc.render_scorecard_page()
        at = AppTest.from_function(app, args=(ROOT,), default_timeout=90)
        at.run()
        assert not at.exception, at.exception
        return at
    return run


def _hint(at) -> str:
    return " ".join(c.value for c in at.caption if "Ur Durrett-arket" in c.value)


def test_the_card_shows_the_valuation_by_itself_and_writes_nothing(page):
    at = page({"rows": {"scoring:d1": {"ev_ebitda": 4.2, "asof": "2026-11-07"}}})
    at.selectbox(key=f"sc_strat_{KEY}").set_value("durrett").run()
    assert f"🤖 Ur Durrett-arket: {TEXT} — förslag, inte inskrivet." in _hint(at)
    assert at.text_input(key=f"sc_val_{KEY}").value == ""
    # Att byta strategi i kortet är ett eget val — men förslaget i sig skriver ingenting
    assert at.session_state["scorecard"]["cards"][KEY].get("valuation", "") == ""
    for other in ("scoring", "producers", "tiggre", "insider"):
        assert _same(at.session_state[other], STORES[other]), other


def test_opening_with_durrett_first_writes_no_card(page):
    """Durrett som enda/förvalda strategi: förslaget syns direkt och lagringen är orörd."""
    at = page()
    at.session_state["scoring"] = dict(copy.deepcopy(STORES["scoring"]), sprott=[])
    at.run()
    assert not at.exception, at.exception
    assert "MCap/framtida vinst 5,5×" in _hint(at)
    assert _same(at.session_state["scorecard"], STORES["scorecard"])


def test_use_fills_the_box_and_the_hint_goes_away(page):
    at = page({"rows": {"scoring:d1": {"ev_ebitda": 4.2}}})
    at.selectbox(key=f"sc_strat_{KEY}").set_value("durrett").run()
    at.button(key=f"sc_useval_{KEY}").click().run()
    assert not at.exception, at.exception
    assert at.text_input(key=f"sc_val_{KEY}").value == TEXT
    card = at.session_state["scorecard"]["cards"][KEY]
    assert card["valuation"] == TEXT and card["strategy"] == "durrett"
    assert not _hint(at)


def test_the_hint_stays_beside_an_own_text_and_sprott_gets_none(page):
    at = page()
    at.selectbox(key=f"sc_strat_{KEY}").set_value("durrett").run()
    at.text_input(key=f"sc_val_{KEY}").input("Egen bedömning").run()
    assert at.session_state["scorecard"]["cards"][KEY]["valuation"] == "Egen bedömning"
    assert "MCap/framtida vinst 5,5×" in _hint(at)                    # förslaget skriver inte över ens egen text
    at.selectbox(key=f"sc_strat_{KEY}").set_value("sprott").run()
    assert not _hint(at)
