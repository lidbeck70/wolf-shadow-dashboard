"""
Wolf Asymmetry, steg E: fyll arket utan handskrift — registret, Durrett-arket,
Börsdata-förslag ur sifferuppdateringen, prefill och extraktorn — och
asymmetry-arket som femte ark i sheets_refresh.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import durrett_cases as dcs  # noqa: E402
from asymmetry import fetch  # noqa: E402
from asymmetry import store as ast  # noqa: E402
from confidence import store as cs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOB = {"generated": "2026-09-25T06:00", "rows": {
    "asymmetry:GPR": {"ticker": "GPR", "ins_id": 105, "price": 6.4, "asof": "2026-09-25", "currency": "USD",
                      "mcap_musd": 1600.0, "ev_musd": 1520.0, "ev_ebitda": 4.0, "nd_ebitda": -0.3, "pe": 8.0,
                      "revenue_musd": 600.0, "fcf_musd": 110.0, "fx_to_usd": 1.0,
                      "fx_table": "sheets_refresh.FX_TO_USD (fast tabell)"},
    "confidence:GPR": {"ticker": "GPR", "ins_id": 105, "price": 9.9, "asof": "2026-09-25", "currency": "USD",
                       "mcap_musd": 2500.0}}}


# ── rena funktioner ──────────────────────────────────────────────────────────
def test_register_candidates_skip_existing_and_duplicates():
    d = ast.default()
    ast.put(d, dcs.gold_producer(), "Viking")
    rows = [{"ticker": "gpr", "name": "x", "strategy": "Viking"},
            {"ticker": "NYX", "name": "Nyx Gold", "strategy": "Momentum"},
            {"ticker": "NYX", "name": "dubblett", "strategy": "Wolf"},
            {"ticker": "ZZZ", "name": "Zed", "strategy": "Untagged"},
            {"ticker": "", "name": "tom"}]
    assert fetch.register_candidates(rows, d) == [("NYX", "Nyx Gold", "Momentum"), ("ZZZ", "Zed", ast.NO_STRATEGY)]
    c = fetch.add_from_register(d, "nyx", "Nyx Gold", "Momentum")
    assert c.ticker == "NYX" and c.fields == {} and ast.strategy(d, "NYX") == "Momentum"
    assert fetch.register_candidates(rows, d) == [("ZZZ", "Zed", ast.NO_STRATEGY)]


def test_copy_from_durrett_is_independent():
    conf, d = cs.default(), ast.default()
    cs.put(conf, dcs.gold_producer())
    cs.put(conf, dcs.copper_developer())
    ast.put(d, dcs.copper_developer(), "Ember")
    assert fetch.durrett_candidates(conf, d) == [("GPR", "Test Gold Producer", "producer")]
    assert fetch.durrett_candidates(None, d) == []
    c = fetch.copy_from_durrett(conf, d, "GPR")
    assert c.num("aisc") == 1450 and ast.strategy(d, "GPR") == "Durrett"
    c.set("aisc", dcs.dp(1700, **dcs._FS))
    ast.put(d, c)
    assert ast.get(d, "GPR").num("aisc") == 1700 and cs.get(conf, "GPR").num("aisc") == 1450
    assert fetch.copy_from_durrett(conf, d, "SAKNAS") is None


def test_refresh_proposals_read_the_asymmetry_row_not_durretts():
    c = dcs.gold_producer()
    props = {k: p for k, p, _cur in fetch.refresh_proposals(BLOB, c)}
    assert props["market_cap_musd"].value == 1600.0 and props["share_price"].value == 6.4
    assert "sifferuppdatering 2026-09-25" in props["market_cap_musd"].source
    assert fetch.refresh_row(BLOB, "gpr")["mcap_musd"] == 1600.0
    assert fetch.refresh_proposals({"rows": {"confidence:GPR": BLOB["rows"]["confidence:GPR"]}}, c) == []
    assert fetch.refresh_proposals(None, c) == []
    # Durrett-arkets egen läsning är oförändrad
    from engines.durrett import refresh as dr
    assert dr.refresh_row(BLOB, "GPR")["price"] == 9.9


def test_sheets_refresh_collects_the_asymmetry_sheet():
    import sheets_refresh as sr
    assert sr.SHEET_FILES["asymmetry"] == "data/asymmetry.json" and sr._BUCKETS["asymmetry"] == ("companies",)
    d = ast.default()
    ast.put(d, dcs.gold_producer(), "Viking")
    rows = sr.collect_rows({"asymmetry": d, "confidence": cs.default()})
    assert [(r["sheet"], r["key"], r["ticker"]) for r in rows] == [("asymmetry", "asymmetry:GPR", "GPR")]


# ── fliken ───────────────────────────────────────────────────────────────────
def _app(monkeypatch, asym: dict, conf: dict, register_rows: list, blob: dict):
    import streamlit as st
    import storage
    import positions
    import refresh_ui

    stores = {"asymmetry": asym, "confidence": conf, "producers": {}, "tiggre": {}}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, stores.get(name, default)))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    monkeypatch.setattr(positions, "open_positions", lambda strategy=None, bucket=None: register_rows)
    monkeypatch.setattr(positions, "view_rows", lambda strategy=None: [])
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob)
    monkeypatch.setenv("ASYM_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["ASYM_TEST_ROOT"])
        from asymmetry.ui import render_asymmetry_page
        render_asymmetry_page()

    return app


def test_tab_imports_from_register_and_durrett_and_applies_borsdata(monkeypatch):
    from streamlit.testing.v1 import AppTest

    conf = cs.default()
    cs.put(conf, dcs.gold_producer())
    reg = [{"ticker": "NYX", "name": "Nyx Gold", "strategy": "Momentum"}]
    app = _app(monkeypatch, ast.default(), conf, reg, BLOB)

    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    labels = " ".join(e.label for e in at.expander)
    assert "Hämta från registret" in labels and "Hämta från Durrett-arket" in labels
    # registret → skal med strategi-tagg
    at.button(key="asym_reg_NYX").click().run()
    assert not at.exception, at.exception
    s = at.session_state["asymmetry"]
    assert s["companies"]["NYX"]["name"] == "Nyx Gold" and s["strategies"]["NYX"] == "Momentum"
    assert s["companies"]["NYX"]["fields"] == {}
    # Durrett-arket → oberoende kopia
    at.button(key="asym_dur_GPR").click().run()
    assert not at.exception, at.exception
    s = at.session_state["asymmetry"]
    assert s["companies"]["GPR"]["fields"]["aisc"]["value"] == 1450 and s["strategies"]["GPR"] == "Durrett"
    assert at.session_state["confidence"]["companies"]["GPR"]["fields"]["aisc"]["value"] == 1450
    at.run()                                                                        # ren körning efter st.rerun
    assert not any(b.key in ("asym_reg_NYX", "asym_dur_GPR") for b in at.button)   # inte längre kandidater
    text = " ".join(c.value for c in at.caption)
    assert "Inga öppna positioner" in text and "Inget i Durrett-arket" in text

    # Ark för GPR: Börsdata-förslag ur asymmetry-raden, Använd skriver till asymmetry
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["asymmetry"] = s
    at.session_state["asym_pick"] = "GPR"
    at.session_state["asym_sub"] = "Ark"
    at.run()
    assert not at.exception, at.exception
    assert any("Börsdata 2026-09-25" in e.label for e in at.expander)
    at.button(key="asym_rf_GPR_market_cap_musd").click().run()
    assert not at.exception, at.exception
    f = at.session_state["asymmetry"]["companies"]["GPR"]["fields"]
    assert f["market_cap_musd"]["value"] == 1600.0 and "Börsdata" in f["market_cap_musd"]["source"]
    assert at.session_state["confidence"]["companies"]["GPR"]["fields"]["market_cap_musd"]["value"] == 1500
    at.button(key="asym_rf_all_GPR").click().run()
    assert not at.exception, at.exception
    f = at.session_state["asymmetry"]["companies"]["GPR"]["fields"]
    assert f["share_price"]["value"] == 6.4
    # prefill och extraktor ritas (extraktorn avstängd utan nyckel), Durrett-arket orört
    labels = " ".join(e.label for e in at.expander)
    assert "Förslag ur granskningsarken" in labels and "Läs ur presentationen" in labels
    assert at.session_state["confidence"]["companies"]["GPR"]["fields"]["share_price"]["value"] == 6.0


def test_tab_without_register_rows_or_blob(monkeypatch):
    from streamlit.testing.v1 import AppTest

    asym = ast.default()
    ast.put(asym, dcs.copper_developer(), "Ember")
    at = AppTest.from_function(_app(monkeypatch, asym, cs.default(), [], {}), default_timeout=60)
    at.session_state["asym_sub"] = "Ark"
    at.run()
    assert not at.exception, at.exception
    text = " ".join(c.value for c in at.caption)
    assert "Inga öppna positioner" in text and "Inget i Durrett-arket" in text
    assert "inga tal än" in text
