"""
Panelens lagring: en misslyckad läsning skrivs aldrig över, testdata i repot läses aldrig som riktig data,
och journalen sparar inte en lista byggd på en läsning som fallerade.

F1  storage.save_session vägrar när sessionens läsning misslyckades (klistrigt, töms inte av mark_saved).
F2  legacy_gist läser bara Gisten — aldrig den lokala ".<fil>" — och ett läsfel blir ett läsfel.
F3  trade_journal läser via gist_storage.read_blob; 'unreadable' spärrar sparandet för sessionen.
F13 kvittot efter Log Trade överlever omritningen; Manage Trades raderar på id.
"""
import copy
import json
import os
import pathlib
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gist_storage    # noqa: E402
import positions       # noqa: E402
import storage         # noqa: E402
import trade_journal as tj  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

REGISTER = {"long": [{"id": "q1", "ticker": "BOL.ST", "strategy": "Quality", "entry_price": 310.0, "shares": 40}],
            "closed": [{"id": "c1", "ticker": "NEM", "strategy": "Quality", "entry_price": 40.0, "exit_price": 44.0}],
            "cash": 15000}


@pytest.fixture
def session(monkeypatch):
    state: dict = {}
    monkeypatch.setattr(streamlit, "session_state", state)
    return state


@pytest.fixture
def github(monkeypatch):
    """Ett fejkat GitHub: {namn: innehåll} på "disken", läsningar som kan fallera och varje skrivning."""
    disk = {"holdings": copy.deepcopy(REGISTER), "swing": {"positions": [{"id": "s1", "ticker": "VOLV-B.ST"}]}}
    fail: set = set()
    writes: list = []

    def load_json(name, default=None):
        if name in fail:
            raise storage.StorageError("GitHub svarade 502.")
        return copy.deepcopy(disk.get(name, default))

    def save_json(name, data, sha=None):
        writes.append(name)
        disk[name] = copy.deepcopy(data)
        return storage.SaveResult(name, "abc1234def", "https://x", "2026-10-29T10:00:00+00:00")

    monkeypatch.setattr(storage, "load_json", load_json)
    monkeypatch.setattr(storage, "save_json", save_json)
    monkeypatch.setattr(storage, "legacy_gist", lambda filename: None)
    return disk, fail, writes


# ── F1: en misslyckad läsning skrivs aldrig över ─────────────────────────────
def test_a_failed_load_refuses_every_save_and_the_store_is_untouched(session, github):
    disk, fail, writes = github
    fail.add("holdings")
    before = copy.deepcopy(disk["holdings"])
    positions.load()                                                    # läsningen fallerar → tom default
    assert storage.load_failed("holdings") and positions.all_positions() == []

    positions.set_cash(25000)                                           # en ändring sparas direkt …
    ok, msg = positions.add("CCJ", "Quality")
    assert writes == [] and disk["holdings"] == before                  # … men inget skrevs över
    assert not ok and "INTE sparad" in msg and "sparar inte över" in msg
    assert "Läsningen av holdings misslyckades" in storage.load_error("holdings")

    with pytest.raises(storage.StorageError, match="sparar inte över det som ligger i GitHub"):
        storage.save_session("holdings")                                # 💾 Spara vägrar också
    storage.mark_saved("holdings")                                      # en "lyckad" markering tömmer inte spärren
    assert storage.load_failed("holdings")
    with pytest.raises(storage.StorageError):
        storage.save_session("holdings")
    assert writes == [] and disk["holdings"] == before


def test_a_failed_save_after_a_good_load_is_not_sticky(session, github, monkeypatch):
    disk, _fail, writes = github
    positions.load()
    real_save = storage.save_json

    def flaky(name, data, sha=None):
        raise storage.StorageError("GitHub svarade 503.")

    monkeypatch.setattr(storage, "save_json", flaky)
    positions.set_cash(1)
    assert "503" in storage.load_error("holdings") and not storage.load_failed("holdings")
    monkeypatch.setattr(storage, "save_json", real_save)
    positions.set_cash(2)                                               # GitHub tillbaka → sparas som vanligt
    assert writes == ["holdings"] and disk["holdings"]["cash"] == 2.0
    assert storage.load_error("holdings") is None


def test_swing_and_tiggre_keep_their_lists_while_the_register_cannot_be_saved(session, github):
    disk, fail, writes = github
    disk["tiggre"] = {"positions": [{"id": "p1", "ticker": "AU"}]}
    fail.add("holdings")
    import swing
    import tiggre
    for _ in range(2):                                                  # även på nästa omritning
        assert swing._load()["positions"][0]["id"] == "s1"
        assert tiggre._load()["positions"][0]["id"] == "p1"
    assert writes == [] and disk["swing"]["positions"][0]["id"] == "s1"


def test_the_save_bar_keeps_the_load_error_and_the_button_disabled(github):
    from streamlit.testing.v1 import AppTest
    _disk, fail, writes = github
    fail.add("holdings")

    def app():
        import storage as s
        import storage_ui
        s.session_load("holdings", {})
        storage_ui.save_bar("holdings", "Holdings")

    at = AppTest.from_function(app, default_timeout=30).run()
    assert not at.exception
    assert any("Kunde inte läsa sparad data för Holdings" in e.value for e in at.error)
    assert next(b for b in at.button if b.label == "💾 Spara").disabled
    assert writes == []


def test_holdings_cash_edit_after_a_failed_load_writes_nothing(github, monkeypatch):
    """AppTest som i fyndet: läsfel → ändra Cash → ingen sparning, felet och spärren står kvar."""
    from streamlit.testing.v1 import AppTest
    import holdings
    disk, fail, writes = github
    fail.add("holdings")
    before = copy.deepcopy(disk["holdings"])
    monkeypatch.setattr(holdings, "_fetch_live_price", lambda ticker, period="6mo": {"price": 0, "change_1d": 0,
                                                                                     "df": None})
    monkeypatch.setattr(holdings, "_get_regime_signal", lambda t, s: None)
    monkeypatch.setenv("F1_TEST_ROOT", str(ROOT))

    def app():
        import importlib
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["F1_TEST_ROOT"])
        importlib.import_module("holdings").render_holdings_page()

    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    at.number_input(key="holdings_cash_input").set_value(25000.0).run()
    at.run()
    assert not at.exception, at.exception
    assert writes == [] and disk["holdings"] == before
    assert any("Spara INTE" in e.value for e in at.error)
    assert next(b for b in at.button if b.label == "💾 Spara").disabled


# ── F2: testdata i repot läses aldrig som riktig data ────────────────────────
def test_legacy_gist_reads_only_the_gist(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".scorecard_data.json").write_text(json.dumps({"cards": {"alfa|AAA": {}}}), encoding="utf-8")
    monkeypatch.setattr(gist_storage, "read_blob", lambda f: ("ok", {"cards": {"riktig|X": {}}}))
    assert storage.legacy_gist("scorecard_data.json") == {"cards": {"riktig|X": {}}}
    monkeypatch.setattr(gist_storage, "read_blob", lambda f: ("missing", None))
    assert storage.legacy_gist("scorecard_data.json") is None              # inte den lokala filen
    monkeypatch.setattr(gist_storage, "read_blob", lambda f: ("unreadable", "HTTP 403"))
    with pytest.raises(storage.StorageError, match="HTTP 403"):
        storage.legacy_gist("scorecard_data.json")


def test_an_unreadable_gist_is_a_load_error_not_fixture_data(session, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".scorecard_data.json").write_text(json.dumps({"cards": {"alfa|AAA": {}}}), encoding="utf-8")
    monkeypatch.setattr(storage, "load_json", lambda name, default=None: None)
    monkeypatch.setattr(gist_storage, "read_blob", lambda f: ("unreadable", "HTTP 403"))
    data = storage.session_load("scorecard", {"cards": {}}, legacy_file="scorecard_data.json")
    assert data == {"cards": {}} and "HTTP 403" in storage.load_failed("scorecard")
    with pytest.raises(storage.StorageError):                           # 💾 Spara kan inte skugga Gisten
        storage.save_session("scorecard")


def test_no_fixture_dotfiles_are_tracked():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/.*_data.json" in ignore and "/.ca_auto_scan_date" in ignore


# ── F3: journalen läses med token och sparar aldrig efter ett läsfel ─────────
class _Resp:
    def __init__(self, code):
        self.status_code = code


def test_an_unreadable_journal_blocks_saving_and_is_not_cached(session, monkeypatch):
    patches = []
    monkeypatch.setattr(tj, "_get_github_token", lambda: "tok")
    monkeypatch.setattr(tj.requests, "patch", lambda *a, **k: patches.append(k) or _Resp(200))
    monkeypatch.setattr(gist_storage, "read_blob", lambda f: ("unreadable", "HTTP 403"))
    with open(tj.LOCAL_FALLBACK, "w") as f:                             # den gamla reserven ska inte läsas
        json.dump([{"id": "fejk", "ticker": "BOL.ST"}], f)
    assert tj.load_journal() == [] and tj.journal_unreadable() == "HTTP 403"
    assert "journal_trades" not in session                              # nästa omritning försöker igen
    assert tj.save_journal([{"id": "ny"}]) is False and patches == []
    assert json.load(open(tj.LOCAL_FALLBACK))[0]["id"] == "fejk"        # lokala filen orörd

    monkeypatch.setattr(gist_storage, "read_blob", lambda f: ("ok", [{"id": "t1"}]))
    assert tj.load_journal() == [{"id": "t1"}] and tj.journal_unreadable() is None
    assert tj.save_journal([{"id": "t1"}, {"id": "t2"}]) is True and len(patches) == 1


def test_a_missing_journal_is_a_first_run_and_never_the_local_file(session, monkeypatch):
    monkeypatch.setattr(tj, "_get_github_token", lambda: "tok")
    monkeypatch.setattr(gist_storage, "read_blob", lambda f: ("missing", None))
    with open(tj.LOCAL_FALLBACK, "w") as f:
        json.dump([{"id": "fejk"}], f)
    assert tj.load_journal() == []
    session.clear()
    monkeypatch.setattr(gist_storage, "read_blob", lambda f: ("ok", {"inte": "en lista"}))
    assert tj.load_journal() == [] and "lista" in tj.journal_unreadable()


def test_local_journal_without_token_never_reaches_the_gist(session, monkeypatch):
    patches = []
    monkeypatch.setattr(tj.requests, "patch", lambda *a, **k: patches.append(k) or _Resp(200))
    monkeypatch.setattr(gist_storage, "read_blob", lambda f: ("missing", None))
    with open(tj.LOCAL_FALLBACK, "w") as f:
        json.dump([{"id": "lokal"}], f)
    assert tj.load_journal() == [{"id": "lokal"}]                       # lokal körning: den lokala filen
    monkeypatch.setattr(tj, "_get_github_token", lambda: "tok")         # token dyker upp mitt i sessionen
    assert tj.save_journal([{"id": "lokal"}, {"id": "ny"}]) is False and patches == []
    assert [t["id"] for t in json.load(open(tj.LOCAL_FALLBACK))] == ["lokal", "ny"]


def test_journal_uses_the_gist_storage_token(monkeypatch):
    monkeypatch.undo()                                                  # conftest stänger av token — släpp den här
    monkeypatch.setattr(gist_storage, "_get_github_token", lambda: "tok-ur-miljon")
    assert tj._get_github_token() == "tok-ur-miljon"


def test_a_close_during_a_journal_outage_writes_nothing(session, monkeypatch):
    import journal_bridge
    patches = []
    monkeypatch.setattr(tj, "_get_github_token", lambda: "tok")
    monkeypatch.setattr(tj.requests, "patch", lambda *a, **k: patches.append(k) or _Resp(200))
    monkeypatch.setattr(gist_storage, "read_blob", lambda f: ("unreadable", "Timeout"))
    closed = {"id": "p9", "ticker": "SAND.ST", "strategy": "Quality", "entry_price": 200.0, "exit_price": 220.0,
              "shares": 10, "entry_date": "2026-09-01", "exit_date": "2026-10-28"}
    assert journal_bridge.record_close(closed)["ticker"] == "SAND.ST"
    assert patches == [] and "journal_trades" not in session


# ── F13: kvittot överlever omritningen, radering på id ───────────────────────
def _journal_app():
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.environ["F13_TEST_ROOT"])
    import trade_journal
    trade_journal.render_trade_journal_page()


TWINS = [{"id": "a", "ticker": "BOL.ST", "strategy": "contrarian", "exit_date": "2026-09-23", "entry_date": "2026-09-01",
          "pnl_pct": 10.0, "pnl_amount": 100.0, "entry_price": 10.0, "exit_price": 11.0, "shares": 10, "note": "först"},
         {"id": "b", "ticker": "BOL.ST", "strategy": "contrarian", "exit_date": "2026-09-23", "entry_date": "2026-09-01",
          "pnl_pct": 10.0, "pnl_amount": 100.0, "entry_price": 10.0, "exit_price": 11.0, "shares": 10, "note": "andra"}]


def test_manage_trades_deletes_by_id_and_the_receipt_survives_the_rerun(monkeypatch):
    from streamlit.testing.v1 import AppTest
    patches = []
    monkeypatch.setattr(tj, "_get_github_token", lambda: "tok")
    monkeypatch.setattr(tj.requests, "patch", lambda *a, **k: patches.append(k["json"]) or _Resp(200))
    monkeypatch.setattr(gist_storage, "read_blob", lambda f: ("ok", copy.deepcopy(TWINS)))
    monkeypatch.setenv("F13_TEST_ROOT", str(ROOT))
    at = AppTest.from_function(_journal_app, default_timeout=60).run()
    assert not at.exception, at.exception
    box = next(s for s in at.selectbox if s.label == "Select trade to delete")
    assert len(box.options) == 2 and box.options[0] == box.options[1]   # samma etikett, två affärer
    box.select_index(1).run()                                           # den andra — som webbläsaren skickar den
    next(b for b in at.button if b.label == "Delete Selected Trade").click().run()
    assert not at.exception, at.exception
    left = at.session_state["journal_trades"]
    assert [t["id"] for t in left] == ["a"]                             # den valda, inte den första
    saved = json.loads(patches[-1]["files"][tj.JOURNAL_FILENAME]["content"])
    assert [t["id"] for t in saved] == ["a"]
    assert any("BOL.ST borttagen" in s.value and "sparat i Gisten" in s.value for s in at.success)


def test_log_trade_receipt_shows_after_the_rerun_and_says_where_it_went(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(gist_storage, "read_blob", lambda f: ("ok", []))
    monkeypatch.setenv("F13_TEST_ROOT", str(ROOT))
    at = AppTest.from_function(_journal_app, default_timeout=60).run()
    next(t for t in at.text_input if t.label == "Ticker").input("eqnr.ol")
    next(b for b in at.button if b.label == "Log Trade").click().run()
    assert not at.exception, at.exception
    assert [t["ticker"] for t in at.session_state["journal_trades"]] == ["EQNR.OL"]
    assert any("EQNR.OL loggad" in w.value and "bara lokalt" in w.value for w in at.warning)


def test_the_journal_page_shows_the_outage_and_locks_the_form(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(gist_storage, "read_blob", lambda f: ("unreadable", "HTTP 502"))
    monkeypatch.setenv("F13_TEST_ROOT", str(ROOT))
    at = AppTest.from_function(_journal_app, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("Journalen gick inte att läsa ur Gisten (HTTP 502)" in e.value for e in at.error)
    assert next(b for b in at.button if b.label == "Log Trade").disabled
