"""
PR 9 av panelgenomgången — Holdings som enda positionsregister, steg 1.

positions.py är registret: data/holdings.json via storage.py, Gisten som
engångskälla, ett radschema, direkt-sparande. holdings.py ritar det.
Steg 2 (Swing/Tiggre) och 3 (allokeraren/copiloten/importen) läser härifrån.
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


# Så här såg Gisten ut: "added" i stället för entry_date, null-tal, inga id:n.
LEGACY = {
    "swing": [{"ticker": "volv-b.st", "entry_price": 250.0, "shares": None,
               "sector": "Industrials", "added": "2026-08-01", "strategy": "Wolf"}],
    "ovtlyr": [],
    "long": [{"ticker": "BOL.ST", "entry_price": "310", "shares": 40.0, "sector": "Materials",
              "added": "2026-06-10", "strategy": "Deep Contrarian", "tranches_deployed": 7},
             {"ticker": "EQNR.OL", "entry_price": 280, "shares": 20, "added": "2026-05-02"}],
    "cash": "15000",
}


@pytest.fixture
def reg(monkeypatch):
    """Registret mot ett tomt sessionsminne, utan nätverk. Returnerar
    (state, saves): sessionen och varje sparat innehåll i ordning."""
    state: dict = {}
    saves: list = []
    seed = {"holdings": copy.deepcopy(LEGACY)}
    monkeypatch.setattr(streamlit, "session_state", state)
    monkeypatch.setattr(storage, "session_load",
                        lambda name, default=None, legacy_file=None:
                        state.setdefault(name, copy.deepcopy(seed.get(name, default))))

    def _save(name):
        saves.append(copy.deepcopy(state[name]))
        return storage.SaveResult(name, "abc1234def", "https://x", "2026-09-23T10:00:00Z")

    monkeypatch.setattr(storage, "save_session", _save)
    return state, saves


# ── Migrering och radschema ──────────────────────────────────────────────────
def test_legacy_gist_rows_load_into_the_schema_without_losing_anything(reg):
    rows = positions.all_positions()
    by = {r["ticker"]: r for r in rows}
    assert set(by) == {"VOLV-B.ST", "BOL.ST", "EQNR.OL"}          # versaler
    v = by["VOLV-B.ST"]
    assert v["entry_date"] == "2026-08-01" and "added" not in v   # added → entry_date
    assert v["shares"] == 0 and v["_bucket"] == "swing"           # null → 0, inte krasch
    assert v["id"] and v["source"] == "holdings" and v["stop"] is None
    b = by["BOL.ST"]
    assert b["entry_price"] == 310.0 and b["shares"] == 40        # "310" → tal, 40.0 → 40
    assert b["extras"]["tranches_deployed"] == 3                  # 7 klipps till 0–3
    assert by["EQNR.OL"]["strategy"] == "Untagged"                # utan tagg → Untagged
    assert positions.cash() == 15000.0
    assert positions.load()["closed"] == []
    # normaliseringen är idempotent — samma id:n vid nästa läsning
    assert [r["id"] for r in positions.all_positions()] == [r["id"] for r in rows]


def test_default_store_has_every_bucket_and_no_save_happens_on_read(reg):
    state, saves = reg
    state.clear()
    positions.load()
    assert set(state["holdings"]) == set(positions.BUCKET_KEYS) | {"closed", "cash"}
    assert saves == []


# ── Skrivning ────────────────────────────────────────────────────────────────
def test_add_puts_the_row_in_the_strategys_bucket_and_saves_at_once(reg):
    state, saves = reg
    ok, msg = positions.add("nem", "Quality", entry_price=45.5, shares=10, sector="Materials")
    assert ok and "Alpha Portfolio" in msg
    row = positions.find("NEM")
    assert row["_bucket"] == "long" and row["entry_date"] and row["entry_price"] == 45.5
    assert len(saves) == 1 and saves[0]["long"][-1]["ticker"] == "NEM"   # direkt-sparat
    assert positions.tickers("Quality") == {"NEM"}


def test_add_refuses_duplicates_caps_and_unknown_strategies(reg):
    assert positions.add("BOL.ST", "Deep Contrarian")[0] is False       # finns redan i long
    assert positions.add("", "Wolf") == (False, "Ticker saknas.")
    assert positions.add("X", "Okänd")[0] is False
    for i in range(4):                                                  # swing: 1 + 4 = 5 = taket
        assert positions.add(f"W{i}", "Wolf")[0]
    ok, msg = positions.add("W9", "Wolf")
    assert not ok and "Max 5" in msg
    assert positions.counts()["swing"] == (5, 5)


def test_update_changes_fields_and_moves_the_row_when_the_strategy_changes(reg):
    state, saves = reg
    assert positions.update("EQNR.OL", strategy="Wolf", stop=250, tranches_deployed=2)
    row = positions.find("EQNR.OL")
    assert row["_bucket"] == "swing" and row["strategy"] == "Wolf" and row["stop"] == 250.0
    assert row["extras"]["tranches_deployed"] == 2                      # okänt fält → extras
    assert positions.find("EQNR.OL", "long") is None
    assert positions.update("FINNS-INTE", stop=1) is False
    assert len(saves) == 1


def test_remove_close_and_cash(reg):
    state, saves = reg
    closed = positions.close("BOL.ST", exit_price=341, reason="EMA200")
    assert closed["exit_price"] == 341.0 and closed["result_pct"] == 10.0
    assert closed["exit_reason"] == "EMA200" and closed["exit_date"]
    assert positions.find("BOL.ST") is None
    c0 = state["holdings"]["closed"][0]
    assert c0["ticker"] == "BOL.ST" and c0["exit_reason"] == "EMA200"   # exit-fälten överlever omläsning
    assert c0["result_pct"] == 10.0 and "exit_reason" not in c0["extras"]
    assert positions.close("BOL.ST") is None                             # redan stängd
    assert positions.remove("VOLV-B.ST") and positions.find("VOLV-B.ST") is None
    assert positions.remove("VOLV-B.ST") is False
    positions.set_cash("2000")
    assert positions.cash() == 2000.0
    assert len(saves) == 3


def test_a_failed_save_is_reported_not_swallowed_and_the_session_keeps_the_change(reg, monkeypatch):
    state, _ = reg

    def _boom(name):
        raise storage.StorageError("403 — token saknar behörighet")

    monkeypatch.setattr(storage, "save_session", _boom)
    ok, _msg = positions.add("CCJ", "Quality")
    assert ok and positions.find("CCJ") is not None                      # ändringen finns kvar
    assert "403" in storage.meta("holdings")["error"]                    # och felet syns i sparraden


# ── holdings.py ritar registret ──────────────────────────────────────────────
def test_holdings_page_reads_and_writes_only_through_positions(reg, monkeypatch):
    src = _src("holdings.py")
    assert "gist_storage" not in src and "_holdings_all_cache" not in src
    assert "storage_ui.save_bar(STORE" in src and "import positions" in src
    import holdings
    flat = holdings._get_all_holdings_flat()
    by = {h["ticker"]: h for h in flat}
    assert by["BOL.ST"]["_portfolio_key"] == "long" and by["BOL.ST"]["tranches_deployed"] == 3
    assert holdings.PORTFOLIOS["long"]["max"] == positions.BUCKETS["long"]["max"]
    warned = []
    monkeypatch.setattr(holdings.st, "warning", lambda m: warned.append(m))
    assert holdings._add_holding("long", "BOL.ST", strategy="Quality") is False
    assert warned and "finns redan" in warned[0]
    assert holdings._add_holding("long", "NEM", 40, "Materials", 5, "Deep Contrarian", 2)
    assert positions.find("NEM")["extras"]["tranches_deployed"] == 2
    holdings._edit_holding_field("long", "NEM", "tranches_deployed", 3)
    assert positions.find("NEM")["extras"]["tranches_deployed"] == 3
    holdings._remove_holding("long", "NEM")
    assert positions.find("NEM") is None
    assert set(holdings._get_holdings("long")[0]) >= {"ticker", "entry_price", "shares", "sector"}


def test_holdings_page_renders_from_the_register(monkeypatch):
    """AppTest-rök: sidan ritas ur registret utan Gist och utan nätverk."""
    from streamlit.testing.v1 import AppTest
    import holdings
    seed = copy.deepcopy(LEGACY)
    monkeypatch.setattr(storage, "session_load",
                        lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(seed) if name == "holdings" else default))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    monkeypatch.setattr(holdings, "_fetch_live_price", lambda ticker, period="6mo": {"price": 0, "change_1d": 0, "df": None})
    monkeypatch.setattr(holdings, "_get_regime_signal", lambda t, s: None)
    monkeypatch.setenv("PR9_TEST_ROOT", ROOT)

    def app():
        import os as _o, sys as _s, importlib
        _s.path.insert(0, _o.environ["PR9_TEST_ROOT"])
        importlib.import_module("holdings").render_holdings_page()

    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["holdings"]["long"][0]["ticker"] == "BOL.ST"
    assert any(b.label == "💾 Spara" for b in at.button)                  # sparraden finns
    assert any("BOL.ST" in m.value for m in at.markdown)
