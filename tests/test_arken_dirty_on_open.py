"""
Arken (PR A2): att öppna Tiggre, Insider eller Scorecard ändrar inget i det sparade.

Förut skrev första ritningen in värden ingen valt: Tiggres faktorer blev 0 (och fyllde i AQS jurisdiktion/management
med 0), CSM fick en tom matris, Insider fick insiders=0 och Scorecardet skapade ett kort med position 0,0 per
kandidat. Varje flik stod då som "osparade ändringar" direkt. Inga grind- eller poängformler är ändrade — testet
nedan låser att ett orört Scorecard-kort ger samma grind som de gamla automatiska standardvärdena.
"""
import copy
import json
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import insider    # noqa: E402
import scorecard  # noqa: E402
import storage    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raderna som arken själva skapar dem (Lägg till / Ny signal), inga tal ifyllda.
STORES = {
    "producers": {"producers": [{"id": "p1", "ticker": "BOL.ST", "name": "Boliden", "commodity": "Koppar",
                                 "date": "2026-09-01"}],
                  "royalty": [{"id": "r1", "ticker": "FNV", "name": "Franco", "level": 1, "date": "2026-09-01"}]},
    "scoring": {"sprott": [{"id": "s1", "ticker": "JR", "name": "Junior", "commodity": "Guld",
                            "date": "2026-09-01", "factors": {}}],
                "durrett": [{"id": "d1", "ticker": "AU", "name": "Gold", "commodity": "Guld",
                             "date": "2026-09-01", "factors": {}}]},
    "insider": {"signals": [{"id": "i1", "ticker": "EKTA-B.ST", "name": "Elekta", "found": "2026-09-01",
                             "role": insider.ROLE_OTHER, "gate": insider.GATE_BLANK, "trigger": ""}]},
    "tiggre": {"candidates": [{"id": "c1", "ticker": "LOB", "name": "Lobo", "added": "2026-09-01", "screen": {},
                               "downside": -40.0, "factors": {}, "catalysts": []}],
               "positions": [], "closed": [], "parked": []},
    "holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0},
    "scorecard": {"cards": {}},
    "confidence": {"companies": {}, "commodity_overrides": {}},
}
SHEETS = (("tiggre", "render_tiggre_page", "tiggre"),
          ("insider", "render_insider_page", "insider"),
          ("scorecard", "render_scorecard_page", "scorecard"))


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


@pytest.fixture
def sheets(monkeypatch):
    """Arken ur en fejkad lagring — inget GitHub, ingen Gist, ingen sifferuppdatering."""
    stores = copy.deepcopy(STORES)
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    import gist_storage
    import refresh_ui
    import screens_ui
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
    monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)
    monkeypatch.setenv("A2_TEST_ROOT", ROOT)

    def run(module: str, fn: str):
        from streamlit.testing.v1 import AppTest
        monkeypatch.setenv("A2_MOD", module)
        monkeypatch.setenv("A2_FN", fn)

        def app():
            import importlib
            import os as _o
            import sys as _s
            _s.path.insert(0, _o.environ["A2_TEST_ROOT"])
            getattr(importlib.import_module(_o.environ["A2_MOD"]), _o.environ["A2_FN"])()

        at = AppTest.from_function(app, default_timeout=90)
        at.run()
        assert not at.exception, (module, at.exception)
        return at
    return run


@pytest.mark.parametrize("module,fn,store", SHEETS)
def test_opening_a_sheet_leaves_the_store_as_it_was(sheets, module, fn, store):
    at = sheets(module, fn)
    assert _same(at.session_state[store], STORES[store]), at.session_state[store]
    at.run()                                                            # en omritning till ändrar inget heller
    assert _same(at.session_state[store], STORES[store]), at.session_state[store]
    if module == "scorecard":                                           # läser de andra arken, skriver inget i dem
        for other in ("tiggre", "insider", "producers", "scoring"):
            assert _same(at.session_state[other], STORES[other]), other


def test_tiggre_factors_start_unassessed_and_clearing_removes_the_answer(sheets):
    at = sheets("tiggre", "render_tiggre_page")
    assert at.selectbox(key="tg_f_c1_stadium").value == "–"
    assert at.number_input(key="tg_mcap_c1").value is None and at.number_input(key="tg_nav_c1").value is None
    assert "aqs_jurisdiktion" not in at.session_state["tiggre"]["candidates"][0]     # AQS är fortfarande ej bedömd
    at.selectbox(key="tg_f_c1_jurisdiktion").select(2).run()
    cand = at.session_state["tiggre"]["candidates"][0]
    assert cand["factors"] == {"jurisdiktion": 2}
    at.selectbox(key="tg_f_c1_jurisdiktion").select("–").run()
    assert at.session_state["tiggre"]["candidates"][0]["factors"] == {}


def test_csm_is_written_into_the_row_once_something_is_filled_in(sheets):
    """Matrisen skapas inte vid öppning — men det man fyller i hamnar i raden som förut."""
    at = sheets("tiggre", "render_tiggre_page")
    assert "csm" not in at.session_state["tiggre"]["candidates"][0]
    at.number_input(key="csm_c1_Bear_nav").set_value(120.0).run()
    csm = at.session_state["tiggre"]["candidates"][0]["csm"]
    assert csm == {"Bear": {"price": None, "nav_musd": 120.0, "financing_need": None}}
    at.selectbox(key="csm_c1_bearsurv").select("Nej").run()
    assert at.session_state["tiggre"]["candidates"][0]["csm"]["bear_survival"] is False


def test_tiggre_new_candidate_has_no_zero_mcap_or_nav(sheets):
    at = sheets("tiggre", "render_tiggre_page")
    at.text_input(key="tg_add_ticker").input("NYA").run()
    at.button(key="tg_add_btn").click().run()
    assert not at.exception, at.exception
    new = at.session_state["tiggre"]["candidates"][-1]
    assert new["ticker"] == "NYA" and "mcap" not in new and "nav" not in new
    assert new["factors"] == {} and new["downside"] == -40.0


def test_insider_count_starts_blank_and_a_typed_count_is_stored(sheets):
    at = sheets("insider", "render_insider_page")
    assert at.number_input(key="ins_n_i1").value is None
    at.number_input(key="ins_n_i1").set_value(3).run()
    assert at.session_state["insider"]["signals"][0]["insiders"] == 3


def test_scorecard_writes_a_card_only_when_something_is_chosen(sheets):
    at = sheets("scorecard", "render_scorecard_page")
    key = "lobo|LOB"
    assert at.number_input(key=f"sc_post_{key}").value is None
    at.number_input(key=f"sc_post_{key}").set_value(3.0).run()
    cards = at.session_state["scorecard"]["cards"]
    assert list(cards) == [key]                                         # bara kortet som ändrades
    assert cards[key]["position_pct_total"] == 3.0 and cards[key]["strategy"] == "tiggre"
    assert cards[key]["position_pct_strategy"] is None


def test_an_untouched_scorecard_card_gates_exactly_like_the_old_defaults():
    """Grinden är oförändrad: ett orört kort räknas som det gamla automatiska {strategi 1, position 0,0}."""
    entries = scorecard.collect({k: v for k, v in {
        "tiggre": STORES["tiggre"]["candidates"], "insider": STORES["insider"]["signals"],
        "producenter": STORES["producers"]["producers"], "royalty": STORES["producers"]["royalty"],
        "sprott": STORES["scoring"]["sprott"], "durrett": STORES["scoring"]["durrett"]}.items()})
    assert entries
    for e in entries:
        old = {"strategy": e["strategies"][0], "position_pct_total": 0.0, "position_pct_strategy": 0.0}
        assert scorecard.gate_state(scorecard._shown({}, e), e) == scorecard.gate_state(old, e)
        stale = {"strategy": "finns-inte"}
        assert scorecard._shown(stale, e)["strategy"] == e["strategies"][0] and stale == {"strategy": "finns-inte"}
