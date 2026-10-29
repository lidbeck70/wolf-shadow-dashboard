"""
review/chatgpt_ui — "📋 Underlag till ChatGPT" i arkens kort, i Scorecardet och i Thors Arken-flik (AppTest).

Expandern finns i varje kort med Excel- och CSV-knapp och prompten i ett kodblock (kopieringsknappen), arket är
oförändrat efter ritningen (underlaget skriver ingenting), utan openpyxl finns bara CSV, och i Thor följer arkens
dom med när Regim-fliken redan fällt den. Inget nätverk: lagringen, sifferuppdateringen och Gisten är fejkade.
"""
import copy
import json
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import controls as ctl  # noqa: E402
import insider  # noqa: E402
import storage  # noqa: E402
from review import chatgpt_ui  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS0 = {f.key: 0 for f in ctl.DS_FIELDS}
STORES = {
    "producers": {"producers": [{"id": "p1", "ticker": "BOL.ST", "name": "Boliden", "commodity": "Koppar",
                                 "date": "2026-09-01", "price": 4.1, **DS0}],
                  "royalty": [{"id": "r1", "ticker": "FNV", "name": "Franco", "level": 1, "date": "2026-09-01"}]},
    "scoring": {"sprott": [{"id": "s1", "ticker": "JR", "name": "Junior", "commodity": "Guld",
                            "date": "2026-09-01", "factors": {"balans": 2}}],
                "durrett": []},
    "insider": {"signals": [{"id": "i1", "ticker": "EKTA-B.ST", "name": "Elekta", "found": "2026-09-01",
                             "role": insider.ROLE_OTHER, "gate": insider.GATE_BLANK, "trigger": ""}]},
    "tiggre": {"candidates": [{"id": "c1", "ticker": "LOB", "name": "Lobo", "added": "2026-09-01", "screen": {},
                               "downside": -40.0, "factors": {}, "catalysts": []}],
               "positions": [], "closed": [], "parked": []},
    "holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0},
    "scorecard": {"cards": {"boliden|BOL.ST": {"strategy": "producenter", "position_pct_total": 1.5}}},
    "confidence": {"companies": {}, "commodity_overrides": {}},
}
PAGES = (("producers", "render_producers_page", {"sheet": "Rick Rule"}, "producers", "rv_csv_rule_p1", "BOL.ST"),
         ("producers", "render_producers_page", {"sheet": "Royalty C"}, "producers", "rv_csv_royalty_r1", "FNV"),
         ("scoring", "render_scoring_page", {}, "scoring", "rv_csv_sprott_s1", "JR"),
         ("tiggre", "render_tiggre_page", {}, "tiggre", "rv_csv_tiggre_c1", "LOB"),
         ("insider", "render_insider_page", {}, "insider", "rv_csv_insider_i1", "EKTA-B.ST"),
         ("scorecard", "render_scorecard_page", {}, "scorecard", "rv_csv_scorecard_boliden|BOL.ST", "BOL.ST"))


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


@pytest.fixture
def page(monkeypatch):
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
    monkeypatch.setenv("RV_TEST_ROOT", ROOT)

    def run(module: str, fn: str, kw: dict):
        from streamlit.testing.v1 import AppTest
        monkeypatch.setenv("RV_MOD", module)
        monkeypatch.setenv("RV_FN", fn)
        monkeypatch.setenv("RV_KW", json.dumps(kw))

        def app():
            import importlib
            import json as _j
            import os as _o
            import sys as _s
            _s.path.insert(0, _o.environ["RV_TEST_ROOT"])
            getattr(importlib.import_module(_o.environ["RV_MOD"]), _o.environ["RV_FN"])(**_j.loads(_o.environ["RV_KW"]))

        at = AppTest.from_function(app, default_timeout=90)
        at.run()
        assert not at.exception, (module, at.exception)
        return at
    return run


@pytest.mark.parametrize("module,fn,kw,store,csv_key,ticker", PAGES)
def test_every_card_has_the_export_and_writes_nothing(page, module, fn, kw, store, csv_key, ticker):
    at = page(module, fn, kw)
    assert chatgpt_ui.TITLE in [e.label for e in at.expander]
    downloads = {d.key: d.proto.label for d in at.get("download_button")}
    assert downloads.get(csv_key) == "⬇ CSV", downloads
    assert downloads.get(csv_key.replace("rv_csv_", "rv_xlsx_")) == "⬇ Excel (.xlsx)", downloads
    prompts = [c.value for c in at.code if c.value.startswith("Underlag för köpbeslut")]
    assert prompts and any(f"({ticker})" in p for p in prompts), prompts
    assert all("\"Bara du\" rör du aldrig" in p for p in prompts)
    assert not any("kunde inte byggas" in c.value for c in at.caption)
    assert _same(at.session_state[store], STORES[store])               # underlaget skriver ingenting
    for other in ("producers", "scoring", "tiggre", "insider", "scorecard"):
        if other in at.session_state:
            assert _same(at.session_state[other], STORES[other]), other


def test_scorecard_export_covers_the_sheet_and_the_card(page):
    at = page("scorecard", "render_scorecard_page", {})
    p = next(c.value for c in at.code if "(BOL.ST)" in c.value)
    assert "Ark: Rick Rule, Scorecard" in p
    assert "[rule:unit_cost]" in p and "Vald säljregel" in p and "[beslutas av mig — fyll inte i]" in p
    caps = " ".join(c.value for c in at.caption)
    assert "fält saknas för köpbeslut" in caps and "fält ifyllda" in caps


def test_without_openpyxl_only_csv_is_offered(page, monkeypatch):
    from review import export as ex
    monkeypatch.setattr(ex, "xlsx_available", lambda: False)
    at = page("insider", "render_insider_page", {})
    keys = [d.key for d in at.get("download_button")]
    assert "rv_csv_insider_i1" in keys and "rv_xlsx_insider_i1" not in keys
    assert any("Excel kräver paketet openpyxl" in c.value for c in at.caption)


def test_durrett_sheet_exports_every_field_for_the_stage(monkeypatch):
    import durrett_cases as dcs
    import refresh_ui
    import streamlit as st
    from confidence import store as cs
    from streamlit.testing.v1 import AppTest

    data = cs.default()
    cs.put(data, dcs.gold_producer())
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, {"confidence": data}.get(name, default)))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    import screens_ui
    monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
    monkeypatch.setenv("RV_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["RV_TEST_ROOT"])
        from engines.durrett.ui import render_durrett_page
        render_durrett_page()

    at = AppTest.from_function(app, default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    assert chatgpt_ui.TITLE in [e.label for e in at.expander]
    p = next(c.value for c in at.code if c.value.startswith("Underlag för köpbeslut"))
    assert "(GPR)" in p and "Ark: Durrett 10-steg" in p and "confidence:fields." in p
    assert {d.key for d in at.get("download_button")} >= {"rv_csv_durrett_GPR", "rv_xlsx_durrett_GPR"}


THOR_STORES = {
    "producers": {"producers": [{"id": "p1", "ticker": "BOL", "name": "Boliden", "commodity": "Koppar",
                                 "price": 9000.0, "unit_cost": 6000.0, "mine_life": 12.0}], "royalty": []},
    "scorecard": {"cards": {"boliden|BOL": {"strategy": "producenter", "position_pct_total": 1.5}}},
}


def _thor_script():
    import streamlit as st
    from thor import arken as a
    from thor import dossier as d
    from thor import ui as t
    ident = d.identify("BOL.ST")
    sheets = d.sheets_for("BOL.ST", st.session_state["_stores"])
    a.render(st, {}, ident, sheets, [], t.SHEET_LINK, t._jump, strategy=st.session_state.get("_strategy", "rule"))


def _thor(monkeypatch, strategy="rule", ctx=None):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    stores = copy.deepcopy(THOR_STORES)
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
    at = AppTest.from_function(_thor_script, default_timeout=120)
    at.session_state["_stores"], at.session_state["_strategy"] = stores, strategy
    if ctx is not None:
        from thor import dom
        at.session_state[dom.CTX_STATE] = ctx
    at.run()
    assert not at.exception, at.exception
    return at


def test_thor_arken_tab_has_the_export_for_the_strategy_sheet(monkeypatch):
    at = _thor(monkeypatch)
    assert chatgpt_ui.TITLE in [e.label for e in at.expander]
    p = next(c.value for c in at.code if c.value.startswith("Underlag för köpbeslut"))
    assert "(BOL.ST)" in p and "· strategi Rick Rule" in p and "Ark: Rick Rule, Scorecard" in p
    assert "Arkens dom just nu" not in p                         # Regim-fliken har inte fällt domen än
    from thor import arken
    assert arken.export_sheets({"reviews": {}}, "tiggre") == ["tiggre"]          # tom mall för strategins ark
    assert arken.export_sheets({"reviews": {}}, "viking_nine") == []             # inget ark, inget underlag
    assert arken.export_sheets({"reviews": {"insider": {}}, "confidence": {"x": 1}, "scorecard": {"k": {}}},
                               "rule") == ["rule", "insider", "confidence", "scorecard"]


def test_thor_export_carries_the_dom_when_the_regim_tab_has_judged(monkeypatch):
    at = _thor(monkeypatch, ctx={"ticker": "BOL.ST", "ctx": {}})
    p = next(c.value for c in at.code if c.value.startswith("Underlag för köpbeslut"))
    assert "Arkens dom just nu (Thor)" in p and "Arkens dom:" in p and "Strategi: Rick Rule" in p
    other = _thor(monkeypatch, ctx={"ticker": "NEM", "ctx": {}})               # domen gällde en annan aktie
    p2 = next(c.value for c in other.code if c.value.startswith("Underlag för köpbeslut"))
    assert "Arkens dom just nu" not in p2


def _dom_script():
    import streamlit as st
    from thor import dom
    from thor import dossier as d
    ident = d.identify("BOL.ST")
    dom.render(st, {}, ident, "rule", d.sheets_for("BOL.ST", st.session_state["_stores"]), [], ctx={"errors": {}})


def test_the_regim_tab_remembers_the_cycle_data_it_judged_with(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from thor import dom
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, default))
    at = AppTest.from_function(_dom_script, default_timeout=120)
    at.session_state["_stores"] = copy.deepcopy(THOR_STORES)
    at.run()
    assert not at.exception, at.exception
    assert at.session_state[dom.CTX_STATE] == {"ticker": "BOL.ST", "ctx": {"errors": {}}}
