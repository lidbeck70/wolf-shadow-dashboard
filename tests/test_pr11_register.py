"""
PR 11 av panelgenomgången — Holdings som enda positionsregister, steg 3.

Allokeraren härleder positionerna ur registret (antal × kurs i SEK) i
stället för handskrivna belopp, copiloten loggar köp till Holdings och
stänger dem där, mäklarimportens öppna positioner går in i Holdings, och
en stängd position skriver sin journalrad (journal_bridge).
"""
import copy
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import positions        # noqa: E402
import storage          # noqa: E402
import journal_bridge   # noqa: E402


def _src(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture
def reg(monkeypatch):
    state: dict = {}
    saves: list = []
    seed = {"holdings": {
        "momentum": [{"id": "m1", "ticker": "VOLV-B.ST", "strategy": "Momentum", "entry_price": 250.0,
                      "shares": 100, "stop": 225.0, "entry_date": "2026-08-01",
                      "extras": {"current": 260.0}}],
        "long": [{"id": "q1", "ticker": "FCX", "strategy": "Quality", "entry_price": 40.0, "shares": 50},
                 {"id": "u1", "ticker": "EQNR.OL", "strategy": "Untagged", "entry_price": 280.0, "shares": 0}],
        "tiggre": [{"id": "t1", "ticker": "AU.TO", "strategy": "Tiggre", "entry_price": 10.0, "shares": 1000}],
        "cash": 50000}}
    monkeypatch.setattr(streamlit, "session_state", state)
    monkeypatch.setattr(storage, "session_load",
                        lambda name, default=None, legacy_file=None:
                        state.setdefault(name, copy.deepcopy(seed.get(name, default))))

    def _save(name):
        saves.append(name)
        return storage.SaveResult(name, "abc1234def", "https://x", "2026-09-23T10:00:00Z")

    monkeypatch.setattr(storage, "save_session", _save)
    # journalen (Gisten) fejkas: inget nätverk
    import trade_journal as tj
    journal: list = []
    monkeypatch.setattr(tj, "load_journal", lambda: journal)
    monkeypatch.setattr(tj, "save_journal", lambda trades: journal.__init__(trades) or True)
    return state, saves, journal


# ── Värdering ────────────────────────────────────────────────────────────────
def test_valuation_takes_the_freshest_price_and_converts_to_sek():
    row = {"id": "m1", "ticker": "VOLV-B.ST", "shares": 100, "entry_price": 250.0,
           "extras": {"current": 260.0}}
    fresh = {"holdings:m1": {"price": 270.0, "currency": "SEK", "asof": "2026-09-22T10:00"}}
    v = positions.valuation(row, fresh)
    assert (v["price"], v["value_sek"], v["source"], v["asof"]) == (270.0, 27000.0, "sifferuppdateringen", "2026-09-22")
    v = positions.valuation(row, {})                              # arkets "Kurs nu"
    assert v["value_sek"] == 26000.0 and v["source"] == "kurs nu i arket"
    v = positions.valuation({"id": "q1", "ticker": "FCX", "shares": 50, "entry_price": 40.0}, {})
    assert v["currency"] == "USD" and v["value_sek"] == round(40 * 50 * positions.FX_TO_SEK["USD"], 0)
    assert v["source"] == "inköpskursen"
    # utan antal: inget värde, inte noll
    assert positions.valuation({"id": "u1", "ticker": "EQNR.OL", "shares": 0, "entry_price": 280.0}, {})["value_sek"] is None
    # London i pence
    v = positions.valuation({"id": "l1", "ticker": "RIO.L", "shares": 10, "entry_price": 5000.0}, {})
    assert v["value_sek"] == round(50.0 * 10 * positions.FX_TO_SEK["GBP"], 0)
    # tiggre-nyckeln duger också
    assert positions.valuation({"id": "t1", "ticker": "AU.TO", "shares": 1000, "entry_price": 10.0},
                               {"tiggre:t1": {"price": 12.0, "currency": "CAD"}})["price"] == 12.0


def test_fx_table_matches_sheets_refresh():
    import sheets_refresh as sr
    for ccy, sek in positions.FX_TO_SEK.items():
        assert abs(sek - sr.FX_TO_USD[ccy] / sr.FX_TO_USD["SEK"]) < 0.01, ccy


# ── Allokeraren ──────────────────────────────────────────────────────────────
def test_allocator_derives_its_positions_from_the_register(reg):
    import allocator as a
    rows = {r["ticker"]: r for r in a.derived_positions({})}
    assert rows["VOLV-B.ST"]["rule"] == "swing" and rows["VOLV-B.ST"]["sleeve"] == "swing"
    assert rows["VOLV-B.ST"]["value"] == 26000.0
    assert rows["FCX"]["rule"] == "quality" and rows["FCX"]["sleeve"] == "langsiktigt"
    assert rows["AU.TO"]["rule"] == "tiggre" and rows["AU.TO"]["sleeve"] == "optionalitet"
    assert rows["EQNR.OL"]["rule"] is None and rows["EQNR.OL"]["value"] is None   # antal saknas
    measurable = [r for r in a.derived_positions({}) if r["value"] is not None]
    # taken räknas på de härledda raderna som förut
    br = {b["ticker"]: b for b in a.position_breaches(measurable, total=100000.0)}
    assert set(br) == {"VOLV-B.ST", "FCX", "AU.TO"}                 # 26 % > 6, 21 % > 10, 77 % > 4
    assert br["AU.TO"]["cap"] == 4.0 and br["VOLV-B.ST"]["cap"] == 6.0
    assert a.position_breaches(measurable, total=10_000_000.0) == []   # 0,3 / 0,2 / 0,8 %
    sums = a.sleeve_sums(a.derived_positions({}), positions.cash())
    assert sums["swing"] == 26000.0 and sums["kassa"] == 50000.0
    assert sums["optionalitet"] == round(10 * 1000 * positions.FX_TO_SEK["CAD"], 0)
    src = _src("allocator.py")
    assert 'data["positions"].append' not in src and "Fyll i ur registret" in src
    assert "Rensa handskrivna" in src


def test_allocator_page_renders_from_the_register_and_fills_the_sleeves(reg, monkeypatch):
    from streamlit.testing.v1 import AppTest
    state, _saves, _j = reg
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    import refresh_ui
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setenv("PR11_TEST_ROOT", ROOT)

    def app():
        import os as _o, sys as _s, importlib
        _s.path.insert(0, _o.environ["PR11_TEST_ROOT"])
        importlib.import_module("allocator").render_allocator_page()

    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    fill = [b for b in at.button if b.label == "Fyll i ur registret"]
    assert fill, "knappen saknas"
    fill[0].click().run()
    assert not at.exception, at.exception
    vals = state["allocator"]["values"]
    assert vals["swing"] == 26000.0 and vals["kassa"] == 50000.0
    assert any("VOLV-B.ST" in m.value for m in at.markdown)          # raden ritas ur registret


# ── Journalraden ─────────────────────────────────────────────────────────────
def test_closing_a_position_writes_the_trade_journal_row_once(reg):
    state, _saves, journal = reg
    closed = positions.close(row_id="m1", exit_price=275.0, exit_date="2026-09-20", reason="under MA50")
    assert closed["result_pct"] == 10.0
    assert len(journal) == 1
    t = journal[0]
    assert t["ticker"] == "VOLV-B.ST" and t["strategy"] == "momentum"
    assert t["entry_price"] == 250.0 and t["exit_price"] == 275.0 and t["shares"] == 100
    assert t["pnl_pct"] == 10.0 and t["pnl_amount"] == 2500.0
    assert t["r_multiple"] == 1.0                                    # 10 % vinst / 10 % risk
    assert t["holding_days"] == 50 and t["exit_reason"] == "under_ma50"
    assert t["position_id"] == "m1" and t["source"] == "holdings"
    # samma stängning en gång till journalförs inte
    assert journal_bridge.record_close(closed) is None and len(journal) == 1
    # utan säljkurs finns inget resultat — ingen rad
    assert journal_bridge.trade_from_closed({"entry_price": 10, "exit_price": None}) is None


# ── Copiloten och importen ───────────────────────────────────────────────────
def test_playbook_keys_map_to_register_tags_both_ways():
    from strategy_rules import PLAYBOOKS
    for key in PLAYBOOKS:
        tag = positions.tag_for_playbook(key)
        assert tag in positions.STRATEGIES, key
        assert positions.TAG_PLAYBOOK[tag] == key
    assert positions.tag_for_playbook("finns-inte") == "Untagged"
    import allocator as a
    for tag, rule in positions.ALLOCATOR_RULE.items():
        assert rule is None or rule in a.RULE_BY_KEY, tag


def test_copilot_logs_to_holdings_and_closes_there():
    src = _src("tabs/copilot.py")
    assert "positions.tag_for_playbook(log_strat)" in src
    assert '"holdings_id": holdings_id' in src
    assert 'positions.close(row_id=e["holdings_id"]' in src
    assert "lägg in som öppen position i Holdings" in src


def test_broker_import_open_positions_go_into_holdings(reg):
    import trade_journal as tj
    from journal_import.fifo import OpenPosition
    rows = [OpenPosition("ISK", "SE0001", "SAND.ST", "Sandvik", "2026-05-02", 40.0, 210.5, "SEK"),
            OpenPosition("ISK", "US0002", "FCX", "Freeport", "2026-01-10", 5.0, 38.0, "USD"),   # finns redan
            OpenPosition("ISK", "US0003", None, "Okänd", "2026-01-10", 5.0, 1.0, "USD")]      # ingen ticker
    added, skipped = tj.import_open_positions(rows)
    assert added == ["SAND.ST"] and skipped == ["FCX", "US0003"]
    row = positions.find("SAND.ST")
    assert row["strategy"] == "Untagged" and row["shares"] == 40 and row["entry_price"] == 210.5
    assert row["entry_date"] == "2026-05-02" and row["source"] == "import" and "ISK" in row["notes"]
    assert "Lägg till i Holdings" in _src("trade_journal.py")


def test_refresh_job_prices_the_register_rows_without_doubling_tiggre():
    import sheets_refresh as sr
    reg_t = positions.from_view({"id": "t1", "ticker": "AU.TO", "entry": 10.0}, "Tiggre", "tiggre")
    sheets = {"tiggre": {"candidates": [], "positions": []},
              "holdings": {"tiggre": [reg_t],
                           "momentum": [{"id": "m1", "ticker": "VOLV-B.ST", "strategy": "Momentum"}],
                           "long": [{"id": "q1", "ticker": "FCX", "strategy": "Quality"}]}}
    keys = [r["key"] for r in sr.collect_rows(sheets)]
    assert "holdings:m1" in keys and "holdings:q1" in keys
    assert keys.count("tiggre:t1") == 1 and "holdings:t1" not in keys
