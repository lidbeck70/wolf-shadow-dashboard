"""
PR 10 av panelgenomgången — Swing och Tiggre läser sina positioner ur
registret (positions.py, Holdings).

Arken behåller sin form (entry/current/date/half_sold …) via en vy över
registret; en sparad "positions"-lista i data/swing.json eller
data/tiggre.json flyttas in en gång och töms. Jobben (alert_scan,
sheets_refresh) läser data/holdings.json, och scorecard, review_link,
copilot, wolf_regime_ui och confidence-prefill går samma väg.
"""
import copy
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import positions   # noqa: E402
import storage     # noqa: E402


def _src(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


SWING_ROW = {"id": "s1", "ticker": "VOLV-B.ST", "entry": 250.0, "current": 262.0, "setup": "A",
             "date": "2026-09-01", "belowMA50": False, "outOfRank": False, "halfTaken": True}
TIGGRE_ROW = {"id": "p1", "ticker": "AU", "name": "Au Corp", "entry": 10.0, "current": 12.0,
              "shares": 100, "mcap": 300.0, "nav": 500.0, "date": "2026-08-20",
              "half_sold": False, "triggers": {"key_person": False},
              "catalysts": [{"id": "c1", "name": "FS", "date": "2026-11", "status": "Väntar"}]}


@pytest.fixture
def reg(monkeypatch):
    """Tomt register + arken med en kvarglömd positionslista var; inget nätverk."""
    state: dict = {}
    saves: list = []
    seed = {"swing": {"market": {"aboveMA200": True}, "checklist": {"week": "", "done": []},
                      "watchlist": [], "positions": [copy.deepcopy(SWING_ROW)], "closed": []},
            "tiggre": {"candidates": [{"id": "c9", "ticker": "LOB", "name": "Lobo", "nav": 100.0,
                                       "mcap": 20.0, "screen": {}, "factors": {}, "catalysts": []}],
                       "positions": [copy.deepcopy(TIGGRE_ROW)], "closed": [], "parked": []}}
    monkeypatch.setattr(streamlit, "session_state", state)
    monkeypatch.setattr(storage, "session_load",
                        lambda name, default=None, legacy_file=None:
                        state.setdefault(name, copy.deepcopy(seed.get(name, default))))

    def _save(name):
        saves.append((name, copy.deepcopy(state[name])))
        return storage.SaveResult(name, "abc1234def", "https://x", "2026-09-23T10:00:00Z")

    monkeypatch.setattr(storage, "save_session", _save)
    return state, saves


# ── Vyn ──────────────────────────────────────────────────────────────────────
def test_view_round_trips_the_sheets_own_fields():
    reg_row = positions.from_view(TIGGRE_ROW, "Tiggre", "tiggre")
    assert reg_row["id"] == "p1" and reg_row["entry_price"] == 10.0
    assert reg_row["entry_date"] == "2026-08-20" and reg_row["shares"] == 100
    assert reg_row["extras"]["nav"] == 500.0 and reg_row["extras"]["catalysts"][0]["name"] == "FS"
    back = positions.to_view(reg_row)
    for k, v in TIGGRE_ROW.items():
        assert back[k] == v, k
    assert back["strategy"] == "Tiggre"
    # rå lagring utan session (jobben): bara rätt strategi kommer med
    data = {"tiggre": [reg_row], "momentum": [positions.from_view(SWING_ROW, "Momentum", "swing")]}
    assert [r["id"] for r in positions.view_rows_from(data, "Tiggre")] == ["p1"]
    assert positions.view_rows_from(data, "Momentum")[0]["halfTaken"] is True
    assert positions.view_rows_from(None, "Tiggre") == []


def test_caps_agree_with_the_sheets():
    import swing, tiggre
    assert positions.BUCKETS["momentum"]["max"] == swing.MAX_POSITIONS == 8
    assert positions.BUCKETS["tiggre"]["max"] == tiggre.MAX_POSITIONS == 6
    assert positions.bucket_for("Momentum") == "momentum" and positions.bucket_for("Tiggre") == "tiggre"


# ── Engångsflytten ───────────────────────────────────────────────────────────
def test_swing_and_tiggre_move_their_lists_into_the_register_once(reg):
    state, saves = reg
    import swing, tiggre
    sd = swing._load()
    assert sd["positions"] == []                                 # tömd i arket
    assert [p["id"] for p in swing.open_positions()] == ["s1"]   # finns i registret
    assert swing.open_positions()[0]["halfTaken"] is True
    td = tiggre._load()
    assert td["positions"] == [] and td["candidates"][0]["ticker"] == "LOB"
    assert tiggre.open_positions()[0]["catalysts"][0]["name"] == "FS"
    assert positions.find_id("p1")["_bucket"] == "tiggre"
    # registret sparades, och arkens tömning skrevs en gång
    assert [n for n, _ in saves] == ["holdings", "swing", "holdings", "tiggre"]
    # idempotent: en andra flytt av samma id lägger inte till något
    assert positions.migrate_rows("Tiggre", [TIGGRE_ROW], "tiggre") == 0
    assert len(positions.open_positions("Tiggre")) == 1


def test_sheet_edits_write_through_to_the_register(reg):
    state, saves = reg
    import swing
    swing._load()
    p = swing.open_positions()[0]
    p["current"], p["belowMA50"] = 300.0, True
    positions.put("Momentum", p, "swing")
    again = swing.open_positions()[0]
    assert again["current"] == 300.0 and again["belowMA50"] is True and again["id"] == "s1"
    # rules_data: samma dict swing_verdict/alert_rules alltid tagit
    import swing_verdict as sv
    rd = swing.rules_data()
    assert sv.held_position("VOLV-B.ST", rd)["entry"] == 250.0
    assert sv.position_count(rd) == 1
    # stängning: bort ur registret, in i closed med resultat
    closed = positions.close(row_id="s1", exit_price=300.0, reason="under MA50")
    assert closed["result_pct"] == 20.0 and swing.open_positions() == []
    assert state["holdings"]["closed"][0]["exit_reason"] == "under MA50"


# ── Jobben och de andra läsarna ──────────────────────────────────────────────
def test_sheets_refresh_reads_tiggre_positions_from_holdings():
    import sheets_refresh as sr
    reg_row = positions.from_view(TIGGRE_ROW, "Tiggre", "tiggre")
    sheets = {"tiggre": {"candidates": [], "positions": []},
              "holdings": {"tiggre": [reg_row], "swing": [], "long": []}}
    refs = sr.collect_rows(sheets)
    keys = {r["key"] for r in refs}
    assert "tiggre:p1" in keys                                   # nyckeln oförändrad
    pos = next(r for r in refs if r["key"] == "tiggre:p1")
    assert pos["bucket"] == "positions" and pos["row"]["entry"] == 10.0
    # händelserna räknar på vyn: −40 % → omvärdera
    kinds = {e["kind"] for e in sr.build_events(sheets, {"tiggre:p1": {"price": 5.5, "mcap_musd": 50.0}})}
    assert "tiggre_drawdown" in kinds
    assert "holdings" in sr.SHEET_FILES


def test_alert_scan_and_the_other_readers_go_through_the_register():
    src = _src("alert_scan.py")
    assert 'HOLDINGS_PATH = "data/holdings.json"' in src and 'view_rows_from(' in src
    assert 'm.open_positions()' in _src("scorecard.py")
    assert '_swing.rules_data()' in _src("tabs/copilot.py")
    assert '_swing.rules_data()' in _src("wolf_regime_ui.py")
    assert '_positions.view_rows("Tiggre")' in _src("confidence/ui.py")
    assert 'data["positions"].append' not in _src("swing.py")
    assert 'data["positions"].append' not in _src("tiggre.py")
    # review_link: en Tiggre-position i registret hittas i granskningen
    import review_link as rl
    stores = {"tiggre": {"candidates": [], "positions": []},
              "holdings": {"tiggre": [positions.from_view({"ticker": "LOB", "id": "x1"},
                                                          "Tiggre", "tiggre")]}}
    assert rl.find_row("tiggre", "LOB", stores)["ticker"] == "LOB"
    assert "holdings" in rl.STORE_DEFAULTS


# ── Sidorna ritas ur registret ───────────────────────────────────────────────
def test_swing_and_tiggre_pages_render_their_positions_from_the_register(reg, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    import refresh_ui, screens_ui
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
    monkeypatch.setenv("PR10_TEST_ROOT", ROOT)

    def app():
        import os as _o, sys as _s, importlib
        _s.path.insert(0, _o.environ["PR10_TEST_ROOT"])
        importlib.import_module(_o.environ["PR10_MOD"]).__dict__[_o.environ["PR10_FN"]]()

    state, _saves = reg
    for module, fn, key in (("swing", "render_swing_page", "swing_entry_s1"),
                            ("tiggre", "render_tiggre_page", "tg_p_entry_p1")):
        monkeypatch.setenv("PR10_MOD", module)
        monkeypatch.setenv("PR10_FN", fn)
        at = AppTest.from_function(app, default_timeout=60)
        at.run()
        assert not at.exception, (module, at.exception)
        assert any(n.key == key for n in at.number_input), f"{module}: positionen ritas inte"
