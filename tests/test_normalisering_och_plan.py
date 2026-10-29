"""
F9 · Att öppna en flik är ingen osparad ändring. Lagren fyller i saknade nycklar (ett nytt larmben, Wolf
Asymmetrys strategi-taggar) efter läsningen; förr gjorde det sessionen "osparad" direkt — sparraden sa
'⚠️ Du har osparade ändringar' och sidfoten 'Osparade ändringar i: alerts/confidence' hela sessionen utan att du
rört något. storage.session_normalize formar datan och räknar den fortfarande som ren; en riktig ändring syns, och
inget skrivs vid ritning. Samma sak för Copilotens journalhistorik, och fiat-reglerna i MER → Larm visar det
alert_rules använder (rules: [] = inga).

F10 · Mot målets 'Plan följd' räknar Copilotens stängda poster (data/copilot_journal.json) tillsammans med Trade
Journal — en affär som står i båda räknas en gång.

Inget nätverk: lagret går mot ett repo i minnet (test_watch_ui.fake_repo) eller en fejkad session_load.
"""
import copy
import datetime
import json
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import durrett_cases as dcs  # noqa: E402
import storage  # noqa: E402
import test_watch_ui as twu  # noqa: E402
from confidence import store as cs  # noqa: E402
from portfolio import mot_malet as mm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNSAVED = "Du har osparade ändringar"
FOOTER = "Osparade ändringar i"


def _text(at) -> str:
    return " ".join(str(m.value) for m in at.markdown)


# ── storage.session_normalize ────────────────────────────────────────────────
@pytest.fixture
def state(monkeypatch):
    s: dict = {}
    monkeypatch.setattr(streamlit, "session_state", s)
    return s


def _fill(d):
    return storage.fill_defaults(d, {"a": 1, "nytt": {"på": True}})


def test_normalization_alone_keeps_a_clean_store_clean(state, monkeypatch):
    monkeypatch.setattr(storage, "load_json", lambda name, default=None: {"a": 5})
    data = storage.session_normalize("rotation", storage.session_load("rotation", {}), _fill)
    assert data == {"a": 5, "nytt": {"på": True}} and state["rotation"] is data      # befintligt värde rörs inte
    assert not storage.is_dirty("rotation") and storage.dirty_stores() == []
    assert storage.session_normalize("rotation", storage.session_load("rotation", {}), _fill) is data
    assert not storage.is_dirty("rotation")                                          # idempotent
    data["a"] = 6                                                                    # din ändring
    storage.session_normalize("rotation", storage.session_load("rotation", {}), _fill)
    assert storage.is_dirty("rotation") and storage.dirty_stores() == ["rotation"]


def test_an_edited_store_stays_dirty_after_normalization(state, monkeypatch):
    monkeypatch.setattr(storage, "load_json", lambda name, default=None: {"a": 5})
    data = storage.session_load("rotation", {})
    data["a"] = 7                                                                    # ändrat före normaliseringen
    storage.session_normalize("rotation", data, _fill)
    assert storage.is_dirty("rotation")


def test_migrated_data_stays_unsaved(state, monkeypatch):
    """Gist-migrerad data ligger bara i Gisten — den ska fortfarande be om ett sparklick."""
    monkeypatch.setattr(storage, "load_json", lambda name, default=None: None)
    monkeypatch.setattr(storage, "legacy_gist", lambda filename: {"a": 2})
    data = storage.session_load("rotation", {}, legacy_file="rotation_data.json")
    storage.session_normalize("rotation", data, _fill)
    assert storage.is_dirty("rotation")


def test_another_page_loading_first_without_normalization_is_also_caught(state, monkeypatch):
    """Thor och Underlaget läser lagren utan normalisering; när fliken sedan formar datan är den fortfarande ren."""
    monkeypatch.setattr(storage, "load_json", lambda name, default=None: "trasig")
    storage.session_load("rotation", {})                                             # en annan sida läste först
    data = storage.session_normalize("rotation", storage.session_load("rotation", {}), _fill)
    assert data == {"a": 1, "nytt": {"på": True}} and state["rotation"] is data      # annat än dict → standardformen
    assert not storage.is_dirty("rotation")


def test_fill_defaults_copies_the_defaults():
    defaults = {"lista": [], "ben": {"enabled": True}}
    a, b = storage.fill_defaults({}, defaults), storage.fill_defaults(None, defaults)
    a["lista"].append(1)
    a["ben"]["enabled"] = False
    assert b == defaults == {"lista": [], "ben": {"enabled": True}}


def test_watchlist_uses_the_shared_helper():
    src = open(os.path.join(ROOT, "watchlist.py"), encoding="utf-8").read()
    assert "storage.session_normalize(STORE," in src and "_clean_key" not in src


@pytest.mark.parametrize("module, missing", [
    ("scorecard", "cards"), ("producers", "royalty"), ("allocator", "positions"), ("rotation", "history"),
    ("insider", "signals"), ("scoring", "durrett"), ("swing", "closed"), ("tiggre", "parked")])
def test_the_sheets_fill_missing_keys_without_going_dirty(state, monkeypatch, module, missing):
    import importlib
    mod = importlib.import_module(module)
    stored = {k: copy.deepcopy(v) for k, v in mod._default().items() if k != missing}
    monkeypatch.setattr(storage, "load_json", lambda name, default=None: copy.deepcopy(stored))
    data = mod._load()
    assert missing in data and not storage.is_dirty(mod.STORE)


def test_positions_and_the_copilot_journal_too(state, monkeypatch):
    import positions
    from tabs import copilot
    stored = {"holdings": {"long": [{"id": "h1", "ticker": "abb.st", "strategy": "Quality", "entry_price": 300,
                                     "shares": 10}]},
              "copilot_journal": {}}
    monkeypatch.setattr(storage, "load_json", lambda name, default=None: copy.deepcopy(stored.get(name, default)))
    monkeypatch.setattr(copilot, "_legacy_entries", lambda: [])
    assert positions.load()["long"][0]["ticker"] == "ABB.ST" and not storage.is_dirty(positions.STORE)
    assert copilot._load_journal() == [] and not storage.is_dirty(copilot.STORE)


# ── MER → Larm ───────────────────────────────────────────────────────────────
STORED_ALERTS = {"swing": {"enabled": True, "channels": ["discord"]},
                 "wolf": {"enabled": True, "channels": ["discord"]}}       # ribban saknas, liksom de nya benen


def _larm_page():
    import storage_ui
    from tabs.alerts import tab_alerts
    tab_alerts()                                       # hela MER → Larm, larmloggen ur en fejkad Gist
    storage_ui.footer()


def test_opening_the_alert_page_is_not_an_unsaved_change(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import gist_storage
    monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)
    repo = twu.fake_repo(monkeypatch, {"alerts": STORED_ALERTS})
    at = AppTest.from_function(_larm_page, default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.toggle(key="sched_watch_on").value is True                       # benet finns, påslaget
    assert at.number_input(key="sched_wolf_min_score").value == 80
    assert UNSAVED not in _text(at) and FOOTER not in _text(at) and "Inga osparade ändringar" in _text(at)
    at.run()                                                                   # och nästa körning likadant
    assert UNSAVED not in _text(at) and FOOTER not in _text(at)
    assert repo["alerts"] == STORED_ALERTS                                     # inget skrevs vid ritningen
    # en riktig ändring syns — i sparraden och i sidfoten
    at.toggle(key="sched_watch_on").set_value(False).run()
    assert not at.exception, at.exception
    assert f"{FOOTER}: alerts" in _text(at)                                   # sidfoten direkt
    at.run()
    assert UNSAVED in _text(at) and f"{FOOTER}: alerts" in _text(at)          # sparraden (överst) nästa körning
    assert at.session_state["alerts"]["watch"] == {"enabled": False, "channels": ["discord"]}
    assert repo["alerts"] == STORED_ALERTS                                     # sparas först med 💾


def test_a_changed_threshold_and_fiat_rules_still_count(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import gist_storage
    monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)
    twu.fake_repo(monkeypatch, {"alerts": STORED_ALERTS})
    at = AppTest.from_function(_larm_page, default_timeout=60).run()
    at.number_input(key="sched_wolf_min_score").set_value(90).run()
    assert f"{FOOTER}: alerts" in _text(at) and at.session_state["alerts"]["wolf"]["min_score"] == 90
    rules = at.multiselect(key="sched_fiat_rules")
    assert len(rules.value) == 6                                               # saknad lista = alla regler
    rules.set_value(rules.value[:2]).run()
    assert at.session_state["alerts"]["fiat"]["rules"] == rules.value[:2]


def test_fiat_rules_all_switched_off_show_as_off(monkeypatch):
    """rules: [] = inga regler, som i alert_rules.fiat_alerts. Förr visade listan då alla sex som aktiva medan
    larmen var tysta."""
    from streamlit.testing.v1 import AppTest
    import gist_storage
    monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)
    stored = dict(STORED_ALERTS, fiat={"enabled": True, "channels": ["discord"], "rules": []})
    repo = twu.fake_repo(monkeypatch, {"alerts": stored})
    at = AppTest.from_function(_larm_page, default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.multiselect(key="sched_fiat_rules").value == []
    assert UNSAVED not in _text(at) and FOOTER not in _text(at)
    at.multiselect(key="sched_fiat_rules").select("m2_high").run()             # du slår på en regel
    assert at.session_state["alerts"]["fiat"]["rules"] == ["m2_high"] and f"{FOOTER}: alerts" in _text(at)
    assert repo["alerts"] == stored                                           # sparas först med 💾


# ── Copilotens journalhistorik ───────────────────────────────────────────────
STORED_COPILOT = {"entries": [
    {"ticker": "VOLV-B.ST", "date": "2026-09-10", "strategy": "ovtlyr", "entry": 250.0, "stop": 240.0,
     "target": 280.0},                                                         # öppen, inga säljfält alls
    {"ticker": "ABB.ST", "date": "2026-09-01", "entry": 300.0, "stop": 290.0, "exit_price": 330.0,
     "shares": 0, "sell_rule": "okänd"},                                       # stängd utan säljdatum
    {"ticker": "SAND.ST", "date": "2026-09-02", "entry": 240.0, "stop": 230.0, "exit_price": 250.0,
     "exit_date": "2026-09-20T15:30:00", "followed_plan": True}]}              # tidsstämpel, gammalt svar


def _history_page():
    import storage_ui
    from tabs import copilot
    copilot._render_journal_history()
    storage_ui.footer()


def test_opening_the_copilot_journal_history_is_not_an_unsaved_change(monkeypatch):
    """Fälten fyllde förr i saknade nycklar (exit_price None, sell_rule '' …) vid varje ritning — då blev
    copilot_journal osparad utan att du rört något. Nu skrivs posten bara när du ändrat ett fält."""
    from streamlit.testing.v1 import AppTest
    repo = twu.fake_repo(monkeypatch, {"copilot_journal": STORED_COPILOT})
    at = AppTest.from_function(_history_page, default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="cj_plan_0").value == "" and at.number_input(key="cj_exit_1").value == 330.0
    assert FOOTER not in _text(at)
    at.run()                                                                   # och nästa körning likadant
    assert FOOTER not in _text(at)
    assert at.session_state["copilot_journal"] == STORED_COPILOT              # inget fylldes i, inget gammalt svar föll
    assert repo["copilot_journal"] == STORED_COPILOT
    # en riktig ändring syns, och posten får sina fält
    at.selectbox(key="cj_plan_0").set_value("Ja").run()
    assert not at.exception, at.exception
    assert f"{FOOTER}: copilot_journal" in _text(at)
    e = at.session_state["copilot_journal"]["entries"][0]
    assert e["followed_plan"] == "Ja" and e["exit_price"] is None and e["exit_date"] is None
    at.number_input(key="cj_exit_0").set_value(270.0).run()                    # stänger affären i dag
    e = at.session_state["copilot_journal"]["entries"][0]
    assert e["exit_price"] == 270.0 and e["exit_date"] == str(datetime.date.today()) and e["followed_plan"] == "Ja"
    assert repo["copilot_journal"] == STORED_COPILOT                           # sparas först med 💾


# ── 🐺 Wolf Asymmetry ────────────────────────────────────────────────────────
def _asym_page():
    from asymmetry.ui import render_asymmetry_page
    import storage_ui
    render_asymmetry_page()
    storage_ui.footer()


def _asym_app(monkeypatch, mode=None):
    from streamlit.testing.v1 import AppTest
    import commodity_prices
    import positions
    import refresh_ui
    conf = cs.default()
    cs.put(conf, dcs.gold_producer())
    stored = {"companies": conf["companies"]}                                  # varken overrides eller strategier
    repo = twu.fake_repo(monkeypatch, {"confidence": stored})
    monkeypatch.setattr(positions, "open_positions", lambda strategy=None, bucket=None: [])
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(commodity_prices, "spot", lambda name: None)
    at = AppTest.from_function(_asym_page, default_timeout=60)
    if mode:
        at.session_state["asym_mode"] = mode
    return at.run(), repo, stored


@pytest.mark.parametrize("mode", [None, "Analys", "Ark"])
def test_opening_wolf_asymmetry_is_not_an_unsaved_change(monkeypatch, mode):
    at, repo, stored = _asym_app(monkeypatch, mode)
    assert not at.exception, at.exception
    assert at.session_state["confidence"]["strategies"] == {}                 # formen är på plats i sessionen
    assert UNSAVED not in _text(at) and FOOTER not in _text(at) and "Inga osparade ändringar" in _text(at)
    assert repo["confidence"] == json.loads(json.dumps(stored))               # inget skrevs vid ritningen


def test_a_real_edit_in_wolf_asymmetry_still_shows(monkeypatch):
    at, repo, stored = _asym_app(monkeypatch, "Ark")
    at.selectbox(key="asym_strat_GPR").set_value("Ember").run()
    at.button(key="FormSubmitter:asym_ident_GPR-Uppdatera").click().run()
    assert not at.exception, at.exception
    assert at.session_state["confidence"]["strategies"]["GPR"] == "Ember"
    assert UNSAVED in _text(at) and f"{FOOTER}: confidence" in _text(at)
    assert "strategies" not in repo["confidence"]                             # sparas först med 💾


# ── F10 · Plan följd ─────────────────────────────────────────────────────────
JOURNAL = [{"ticker": "ABB.ST", "entry_date": "2026-09-01", "followed_plan": "Ja"},
           {"ticker": "SAND.ST", "entry_date": "2026-09-03", "position_id": "h7"},              # obesvarad
           {"ticker": "HEXA-B.ST", "entry_date": "2026-09-05", "followed_plan": "Nej"}]
COPILOT = [{"ticker": "VOLV-B.ST", "date": "2026-09-10", "exit_price": 280.0, "followed_plan": "Nej"},
           {"ticker": "ERIC-B.ST", "date": "2026-09-12", "followed_plan": "Ja"},                # öppen: räknas inte
           {"ticker": "abb.st", "date": "2026-09-01", "exit_price": 330.0, "followed_plan": "Nej"},  # samma aktie+dag
           {"ticker": "SAND.ST", "date": "2026-09-02", "holdings_id": "h7", "exit_price": 250.0,
            "followed_plan": "Ja"},                                     # samma position: journalen saknar svar
           {"ticker": "BOL.ST", "date": "2026-09-20", "exit_price": 400.0, "followed_plan": ""}]  # obesvarad


def test_plan_followed_counts_closed_copilot_entries_once():
    # journalen: Ja (ABB), Nej (HEXA) · Copiloten: Nej (VOLV), Ja (SAND, journalens rad är obesvarad)
    assert mm.plan_followed(JOURNAL, COPILOT) == 50.0
    assert mm.plan_followed([], COPILOT) == round(1 / 3 * 100, 1)             # ABB räknas här, ingen journalrad
    assert mm.plan_followed(JOURNAL) == 50.0 and mm.plan_followed(JOURNAL, None) == 50.0
    assert mm.plan_followed([], [COPILOT[1]]) is None                          # en öppen post är ingen affär
    assert mm.plan_followed([{"followed_plan": ["skräp"]}], [{"exit_price": 1, "followed_plan": {}}]) is None


def test_journal_answer_wins_over_the_copilots_for_the_same_trade():
    journal = [{"ticker": "SAND.ST", "entry_date": "2026-09-02T00:00:00", "position_id": "h7", "followed_plan": "Nej"}]
    copilot = [{"ticker": "SAND.ST", "date": "2026-09-02", "holdings_id": "h7", "exit_price": 250.0,
                "followed_plan": "Ja"}]
    assert mm.plan_followed(journal, copilot) == 0.0


def test_copilot_entries_read_the_session_store(monkeypatch, state):
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        state.setdefault(name, {"entries": copy.deepcopy(COPILOT)}))
    assert mm.copilot_entries() == COPILOT
    state["copilot_journal"] = {"entries": "trasig"}
    assert mm.copilot_entries() == []

    def _boom(*a, **k):
        raise RuntimeError("ingen lagring")
    monkeypatch.setattr(storage, "session_load", _boom)
    assert mm.copilot_entries() == []


def test_mot_malet_shows_plan_followed_with_the_copilots_answers(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import trade_journal as tj
    seed = {"settings": {"capital": 300000.0}, "copilot_journal": {"entries": COPILOT}}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(seed.get(name, default))))
    monkeypatch.setattr(tj, "journal_unreadable", lambda: None)

    def app():
        from portfolio import mot_malet
        mot_malet.render_mot_malet_page()

    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["goal_data"] = {"trades": [], "paper": None}
    at.run()
    assert not at.exception, at.exception
    text = _text(at)                                   # Copiloten ensam: Nej (VOLV), Nej (ABB), Ja (SAND)
    assert "Plan följd" in text and "33 %" in text and "Copilotens stängda poster" in text
    assert "samma affär en gång" in text
