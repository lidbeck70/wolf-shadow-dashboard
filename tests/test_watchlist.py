"""
👀 Övervakningen — lagret (watchlist.py): formen överlever trasig data, ett par (aktie, strategi) bevakas en gång,
bara inmatningar sparas, ett sparfel sägs rakt ut, och läget läses ur Thors rader (status_of). Inget nätverk.
"""
import copy
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import storage  # noqa: E402
import test_thor_candidates as fx  # noqa: E402
import watchlist as wl  # noqa: E402
from thor import candidates as tc  # noqa: E402

ROWS = tc.build(fx.BLOBS, today="2026-10-15")


@pytest.fixture
def store(monkeypatch):
    """Lagret mot ett tomt sessionsminne, utan nätverk: (state, saves) — sessionen och varje sparat innehåll."""
    state: dict = {}
    saves: list = []
    monkeypatch.setattr(streamlit, "session_state", state)
    monkeypatch.setattr(storage, "session_load",
                        lambda name, default=None, legacy_file=None: state.setdefault(name, copy.deepcopy(default)))

    def _save(name):
        saves.append(copy.deepcopy(state[name]))
        return storage.SaveResult(name, "abc1234def", "https://x", "2026-10-29T10:00:00Z")

    monkeypatch.setattr(storage, "save_session", _save)
    monkeypatch.setattr(wl, "_today", lambda: "2026-10-29")
    return state, saves


def test_normalize_survives_broken_shapes_and_keeps_one_row_per_pair():
    assert wl.normalize(None) == {"items": []} and wl.normalize([1, 2]) == {"items": []}
    assert wl.normalize({"items": "trasig"}) == {"items": []}
    data = wl.normalize({"items": [
        {"ticker": "ekta b.st", "strategy": "wolf", "id": "a1", "levels": {"entry": "12,5", "stop": 0}, "extra": 1},
        {"ticker": "EKTA-B.ST", "strategy": "wolf", "id": "a2"},                     # samma par: bara den första
        {"ticker": "EKTA-B.ST", "strategy": "momentum", "id": "a3", "at_add": "trasig", "source": "okänd"},
        {"ticker": "", "strategy": "wolf"}, {"ticker": "X.ST"}, "skräp"]})
    items = data["items"]
    assert [(i["id"], i["ticker"], i["strategy"]) for i in items] == \
        [("a1", "EKTA-B.ST", "wolf"), ("a3", "EKTA-B.ST", "momentum")]
    first = items[0]
    assert first["levels"] == {"entry": None, "stop": None, "target": None} and first["extra"] == 1   # okänt behålls
    assert first["at_add"] == {"status": "", "class": "", "entry": None, "stop": None, "target": None, "asof": ""}
    assert first["alert"] is True and first["source"] == "manual" and items[1]["source"] == "manual"
    assert wl.normalize(copy.deepcopy(data)) == data                                     # idempotent: inget blir 'osparat'


def test_add_dedupes_on_ticker_and_strategy_and_saves_only_inputs(store):
    state, saves = store
    c = next(x for x in ROWS if (x.strategy, x.ticker) == ("berserk", "BOL.ST"))
    ok, msg = wl.add("bol.st", "berserk", name="Boliden", at_add=wl.snapshot(c))
    assert ok and msg == "BOL.ST lagd i Övervakning (BERSERK)."
    item = state[wl.STORE]["items"][0]
    assert item["ticker"] == "BOL.ST" and item["strategy"] == "berserk" and item["added"] == "2026-10-29"
    assert item["source"] == "thor" and item["name"] == "Boliden"
    assert item["at_add"] == {"status": c.status, "class": tc.KOP, "entry": 300.0, "stop": 280.0, "target": None,
                              "asof": "2026-10-14"}
    assert not {"last_status", "last_class", "last_checked", "status_class"} & set(item)    # inget läge lagras
    assert len(saves) == 1
    again = wl.add("BOL.ST", "berserk")
    assert again == (False, "BOL.ST bevakas redan med BERSERK (sedan 2026-10-29).") and len(saves) == 1
    ok2, _ = wl.add("BOL.ST", "wolf", at_add=wl.snapshot(None))                          # annan strategi = nytt par
    assert ok2 and len(state[wl.STORE]["items"]) == 2 and state[wl.STORE]["items"][1]["at_add"]["class"] == ""
    assert wl.find("bol.st", "wolf")["strategy"] == "wolf" and wl.find("BOL.ST", "ember") is None
    assert wl.keys() == {("BOL.ST", "berserk"), ("BOL.ST", "wolf")}
    assert wl.add("", "wolf") == (False, "Ticker saknas.") and wl.add("X.ST", "påhittad")[0] is False


def test_remove_and_own_levels(store):
    state, saves = store
    wl.add("NOPE.ST", "wolf", source="manual")
    iid = state[wl.STORE]["items"][0]["id"]
    ok, msg = wl.set_levels(iid, entry=10, stop="9", target=None, note="vänta på rapporten")
    assert ok and msg == "NOPE.ST: egna nivåer sparade."
    item = state[wl.STORE]["items"][0]
    assert item["levels"] == {"entry": 10.0, "stop": 9.0, "target": None} and item["note"] == "vänta på rapporten"
    assert wl.set_levels("finns-inte", 1)[0] is False
    assert wl.remove(iid) == (True, "NOPE.ST borttagen ur Övervakning.") and state[wl.STORE]["items"] == []
    assert wl.remove(iid)[0] is False and len(saves) == 3


def test_a_failed_save_keeps_the_row_and_says_so(store, monkeypatch):
    state, _saves = store

    def _boom(name):
        raise storage.StorageError("Läsningen av watchlist misslyckades — sparar inte")

    monkeypatch.setattr(storage, "save_session", _boom)
    ok, msg = wl.add("BOL.ST", "berserk")
    assert ok and "men INTE sparad" in msg and "Läsningen av watchlist misslyckades" in msg
    assert state[wl.STORE]["items"][0]["ticker"] == "BOL.ST"                        # raden finns i sessionen
    assert state[storage._meta_key(wl.STORE)]["error"].startswith("Läsningen")        # sparraden visar felet


def test_status_of_reads_thors_row_and_falls_back_to_own_levels():
    s = wl.status_of({"ticker": "hexa-b.st", "strategy": "viking_nine"}, ROWS)
    assert s["in_list"] and s["status_class"] == tc.KOP and s["status"] == "GOLDEN TICKET" and s["score_text"] == "8/9 · 86"
    assert (s["entry"], s["stop"], s["own"], s["asof"], s["stale"]) == (110.0, 104.0, False, "2026-10-14", False)
    assert s["candidate"].ticker == "HEXA-B.ST"
    miss = wl.status_of({"ticker": "NOPE.ST", "strategy": "wolf", "levels": {"entry": 10.0, "stop": 9.0}}, ROWS)
    assert not miss["in_list"] and miss["status"] == wl.NOT_IN_RUN and miss["candidate"] is None
    assert (miss["entry"], miss["stop"], miss["own"]) == (10.0, 9.0, True)
    # rätt strategi: BOL.ST finns både som BERSERK KÖP och Quality WAIT
    assert wl.status_of({"ticker": "BOL.ST", "strategy": "quality"}, ROWS)["status"] == "WAIT 2/4"
    # Momentum-raden utan nivåer från strategin? Den har entry (kursen) och stopp −10 % — strategins nivåer gäller
    abb = wl.status_of({"ticker": "ABB.ST", "strategy": "momentum", "levels": {"entry": 1.0}}, ROWS)
    assert abb["entry"] == 500.0 and abb["stop"] == 450.0 and not abb["own"]
    # med suffix krävs samma notering: EQNR.OL får inte en bar EQNR-rad
    bare = [tc.Candidate("EQNR", "Equinor ADR", "wolf", "MODERATE", tc.BEVAKA)]
    assert not wl.status_of({"ticker": "EQNR.OL", "strategy": "wolf"}, bare)["in_list"]
    stale = wl.status_of({"ticker": "HEXA-B.ST", "strategy": "viking_nine"}, tc.build(fx.BLOBS, today="2026-10-30"))
    assert stale["stale"]


def test_distance_pct_and_snapshot_of_nothing():
    assert round(wl.distance_pct(105, 100), 1) == 5.0 and round(wl.distance_pct(90, 100), 1) == -10.0
    assert wl.distance_pct(None, 100) is None and wl.distance_pct(100, 0) is None and wl.distance_pct("x", 1) is None
    assert wl.snapshot(None) == {"status": "", "class": "", "entry": None, "stop": None, "target": None, "asof": ""}


def test_the_store_is_plain_not_private():
    import private_data
    assert wl.STORE not in private_data.PRIVATE_STORES and storage.path_for(wl.STORE) == "data/watchlist.json"
