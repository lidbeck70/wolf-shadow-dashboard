"""
👀 Övervakningen — knappen i Thors analysblock: aktien med strategin i 'Bedöm som' läggs i bevakningen en gång per
par, med en toast; bevakas paret redan blir knappen en länk till PORTFÖLJ → 👀 Övervakning. Knappen sitter mellan
verdiktet och Storlek (aldrig i _storlek, som hoppar ut utan entry/stopp) och fungerar för rader utan signal
(Största i segmentet). Inget nätverk: lagret går mot ett repo i minnet.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_ui as tu  # noqa: E402
from test_watch_ui import fake_repo  # noqa: E402
from thor import candidates as tc  # noqa: E402
from thor import ui as tui  # noqa: E402


def _app(monkeypatch, repo_files=None):
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    repo = fake_repo(monkeypatch, repo_files or {})
    return AppTest.from_function(tu._script), repo


def test_the_button_sits_between_the_verdict_and_storlek():
    src = inspect.getsource(tui._analysis)
    assert src.index("cards.verdict_for(") < src.index("_watch_button(st, ident, strategy, c)") < src.index("_storlek(st,")
    assert "_watch_button" not in inspect.getsource(tui._storlek)


def test_thor_button_adds_once_per_strategy_and_links_to_the_tab(monkeypatch):
    at, repo = _app(monkeypatch)
    at.session_state["thor_ticker"] = "bol.st"
    at.run(timeout=90)
    assert not at.exception
    btn = at.button(key="thor_watch_add")
    assert btn.label == "👀 Lägg i Övervakning (BERSERK)"
    btn.click().run(timeout=90)
    assert not at.exception
    items = repo["watchlist"]["items"]
    assert [(i["ticker"], i["strategy"], i["source"], i["at_add"]["class"]) for i in items] == \
        [("BOL.ST", "berserk", "thor", tc.KOP)]
    assert items[0]["at_add"]["entry"] == 300.0 and items[0]["at_add"]["stop"] == 280.0
    assert any(t.value == "BOL.ST lagd i Övervakning (BERSERK)." for t in at.toast)
    open_btn = at.button(key="thor_watch_open")                                   # dubblett: knappen blir en länk
    assert open_btn.label.startswith("👀 Bevakas sedan ") and open_btn.label.endswith("(BERSERK) — öppna Övervakning →")
    assert not [b for b in at.button if b.key == "thor_watch_add"]
    at.selectbox(key="thor_strategy").select("Wolf").run(timeout=90)            # annan strategi = nytt par
    assert at.button(key="thor_watch_add").label == "👀 Lägg i Övervakning (Wolf)"
    at.button(key="thor_watch_add").click().run(timeout=90)
    assert [(i["ticker"], i["strategy"], i["at_add"]["class"]) for i in repo["watchlist"]["items"]] == \
        [("BOL.ST", "berserk", tc.KOP), ("BOL.ST", "wolf", "")]                    # Wolf har ingen rad för BOL.ST
    at.button(key="thor_watch_open").click().run(timeout=90)
    assert at.session_state["nav_goto"] == "portfolio/overvakning"


def test_a_watched_pair_from_an_earlier_session_shows_the_link(monkeypatch):
    at, _repo = _app(monkeypatch, {"watchlist": {"items": [
        {"id": "w1", "ticker": "BOL.ST", "strategy": "berserk", "added": "2026-10-20"}]}})
    at.session_state["thor_ticker"] = "BOL.ST"
    at.run(timeout=90)
    assert not at.exception
    assert at.button(key="thor_watch_open").label == "👀 Bevakas sedan 2026-10-20 (BERSERK) — öppna Övervakning →"


def test_the_biggest_in_segment_row_without_a_signal_can_be_watched(monkeypatch):
    """Största i segmentet: ingen rad för strategin (c = None) — knappen fungerar ändå, utan ögonblicksbild."""
    at, repo = _app(monkeypatch)
    at.run(timeout=60)
    at.selectbox(key="thor_segment").select("tema:palladium").run(timeout=60)
    next(b for b in at.button if b.key == "thor_open_storst_SBSW").click().run(timeout=120)
    assert not at.exception
    assert "INGEN SIGNAL" in " ".join(m.value for m in at.markdown)
    at.button(key="thor_watch_add").click().run(timeout=90)
    assert not at.exception
    (item,) = repo["watchlist"]["items"]
    assert (item["ticker"], item["strategy"]) == ("SBSW", "berserk")
    assert item["at_add"] == {"status": "", "class": "", "entry": None, "stop": None, "target": None, "asof": ""}


def test_an_unsaved_add_is_said_plainly(monkeypatch):
    """Utan läsbart lager (läsningen fallerade) vägrar storage att spara — raden finns i sessionen och toasten
    säger att den INTE är sparad."""
    import storage
    at, _repo = _app(monkeypatch)

    def _fail(name, default=None):
        raise storage.StorageError("Ingen [github]-sektion")

    monkeypatch.setattr(storage, "load_json", _fail)
    at.session_state["thor_ticker"] = "bol.st"
    at.run(timeout=90)
    at.button(key="thor_watch_add").click().run(timeout=90)
    assert not at.exception
    assert any("men INTE sparad" in t.value for t in at.toast)
    assert [i["ticker"] for i in at.session_state["watchlist"]["items"]] == ["BOL.ST"]
