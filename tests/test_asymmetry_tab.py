"""
Wolf Asymmetry, steg G: en sida, ett ark.

Fliken läser och skriver samma ark som Durrett (data/confidence.json) med en
strategi-tagg per bolag. Två lägen: Analys (KPI-rad, verdikt, thesis
killers, diagram, hopfällda steg) och Ark (nytt bolag, registret, Börsdata,
extraktor, fält, råvaror, signaler). Det gamla data/asymmetry.json flyttas
in en gång.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import durrett_cases as dcs  # noqa: E402
from asymmetry import fetch  # noqa: E402
from asymmetry import store as ast  # noqa: E402
from confidence import store as cs  # noqa: E402
from ui import nav  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = "🐺 Wolf Asymmetry"
BLOB = {"generated": "2026-09-25T06:00", "rows": {
    "confidence:GPR": {"ticker": "GPR", "ins_id": 105, "price": 6.4, "asof": "2026-09-25", "currency": "USD",
                       "mcap_musd": 1600.0, "ev_musd": 1520.0, "ev_ebitda": 4.0, "nd_ebitda": -0.3, "pe": 8.0,
                       "revenue_musd": 600.0, "fcf_musd": 110.0, "fx_to_usd": 1.0}}}


def _src(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


# ── navigation ───────────────────────────────────────────────────────────────
def test_tab_is_wired_under_review_after_durrett():
    review = nav.options("review")
    assert review.index(TAB) == review.index("🧭 Durrett") + 1
    assert nav.resolve("granskning/wolf-asymmetry") == ["review", TAB]
    assert f'sub == "{TAB}"' in _src("wolf_panel.py")
    assert f"GRANSKNING → {TAB}" in _src(os.path.join("ovtlyr", "ui", "rules_page.py"))
    import sheets_refresh as sr
    assert "asymmetry" not in sr.SHEET_FILES and "asymmetry" not in sr._BUCKETS   # ett ark, inte två


# ── lagret: samma ark som Durrett + strategi-tagg ────────────────────────────
def test_store_is_the_shared_confidence_store_with_a_strategy_tag():
    assert ast.STORE == cs.STORE == "confidence"
    d = ast.normalize(cs.default())
    cs.put(d, dcs.gold_producer())
    cs.put(d, dcs.copper_developer())
    ast.set_strategy(d, "gpr", "Viking")
    assert ast.strategy(d, "GPR") == "Viking" and ast.strategy(d, "CDV") == ast.NO_STRATEGY
    assert ast.tickers_for(d, "Viking") == ["GPR"] and ast.tickers_for(d, "Alla") == ["GPR", "CDV"]
    ast.set_strategy(d, "GPR", ast.NO_STRATEGY)
    assert d["strategies"] == {}
    assert cs.normalize(d)["strategies"] == {}           # Durretts normalize behåller taggarna
    tags = ast.strategy_tags()
    assert tags[0] == ast.NO_STRATEGY and "Durrett" in tags and "Untagged" not in tags


def test_merge_legacy_moves_companies_once_without_overwriting():
    d = ast.normalize(cs.default())
    cs.put(d, dcs.gold_producer())
    legacy = {"companies": {"GPR": {"ticker": "GPR", "name": "gammal kopia"},
                            "CDV": dcs.copper_developer().as_dict()},
              "strategies": {"GPR": "Viking", "CDV": "Ember"},
              "commodity_overrides": {"copper": {"supply_balance_pct": {"value": -2}}}}
    assert ast.merge_legacy(d, legacy) == ["CDV"]
    assert cs.get(d, "GPR").name == "Test Gold Producer"          # befintligt bolag rörs inte
    assert ast.strategy(d, "CDV") == "Ember" and ast.strategy(d, "GPR") == "Viking"
    assert d["commodity_overrides"]["copper"]["supply_balance_pct"]["value"] == -2
    assert ast.merge_legacy(d, legacy) == [] and ast.merge_legacy(d, None) == []


def test_register_candidates_and_add():
    d = ast.normalize(cs.default())
    cs.put(d, dcs.gold_producer())
    rows = [{"ticker": "gpr", "name": "x", "strategy": "Viking"},
            {"ticker": "NYX", "name": "Nyx Gold", "strategy": "Momentum"},
            {"ticker": "NYX", "name": "dubblett", "strategy": "Wolf"},
            {"ticker": "ZZZ", "name": "Zed", "strategy": "Untagged"}]
    assert fetch.register_candidates(rows, d) == [("NYX", "Nyx Gold", "Momentum"), ("ZZZ", "Zed", ast.NO_STRATEGY)]
    c = fetch.add_from_register(d, "nyx", "Nyx Gold", "Momentum")
    assert c.ticker == "NYX" and c.fields == {} and ast.strategy(d, "NYX") == "Momentum"
    assert "NYX" in d["companies"]
    props = {k: p for k, p, _c in fetch.refresh_proposals(BLOB, dcs.gold_producer())}
    assert props["market_cap_musd"].value == 1600.0


# ── fliken ───────────────────────────────────────────────────────────────────
def _app(monkeypatch, conf: dict, legacy=None, register_rows=None, blob=None):
    import streamlit as st
    import storage
    import positions
    import refresh_ui

    stores = {"confidence": conf, "asymmetry": legacy, "producers": {}, "tiggre": {}}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, stores.get(name) if stores.get(name) is not None else default))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    monkeypatch.setattr(positions, "open_positions", lambda strategy=None, bucket=None: register_rows or [])
    monkeypatch.setattr(positions, "view_rows", lambda strategy=None: [])
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob or {})
    monkeypatch.setenv("ASYM_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["ASYM_TEST_ROOT"])
        from asymmetry.ui import render_asymmetry_page
        render_asymmetry_page()

    return app


def _text(at) -> str:
    return (" ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)
            + " " + " ".join(e.label for e in at.expander))


def _conf(*makers, tags=None):
    d = ast.normalize(cs.default())
    for mk in makers:
        cs.put(d, mk())
    for t, s in (tags or {}).items():
        ast.set_strategy(d, t, s)
    return d


def test_analysis_is_one_page(monkeypatch):
    from streamlit.testing.v1 import AppTest

    app = _app(monkeypatch, _conf(dcs.gold_producer, dcs.copper_developer, dcs.missing_everything,
                                  tags={"GPR": "Viking", "CDV": "Ember"}))
    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert at.radio(key="asym_mode").value == "Analys"
    vals = {m.label: m.value for m in at.metric}
    assert list(vals) == ["Commodity Leverage", "Margin of Safety", "Break-even-marginal", "Confidence",
                          "Asymmetri", "Justerad uppsida"]
    assert vals["Commodity Leverage"] == "6/10" and vals["Margin of Safety"] == "7/8"
    assert vals["Asymmetri"] == "1.3×" and float(vals["Confidence"]) > 60
    assert len(at.get("plotly_chart")) == 5                     # grid, MoS, scenarier, matris + justerad (i stegen)
    labels = [e.label for e in at.expander]
    assert labels[0].startswith("Varför?") and labels[1].startswith("Confidence-caset") and labels[2].startswith("Tabeller")
    assert len(labels) == 3                                    # inga fler flikar, inga fler expanders
    text = _text(at)
    assert "Råvarupris" in text                                 # thesis killer ur Confidence-caset
    assert "Viking" in text and at.selectbox(key="asym_pick").options[0] == "GPR · Viking"
    assert at.selectbox(key="asym_strategy_filter").options == ["Alla", "Ember", "Viking"]

    # strategi-filter smalnar listan
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["asym_strategy_filter"] = "Ember"
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="asym_pick").options == ["CDV · Ember"]
    assert {m.label: m.value for m in at.metric}["Margin of Safety"] == "10/10"

    # tomt bolag: DATA_MISSING, inga diagram, ingen krasch
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["asym_pick"] = "NUL"
    at.run()
    assert not at.exception, at.exception
    vals = {m.label: m.value for m in at.metric}
    assert vals["Commodity Leverage"] == "DATA_MISSING" and vals["Justerad uppsida"] == "DATA_MISSING"
    assert vals["Asymmetri"] == "DATA_MISSING"
    assert len(at.get("plotly_chart")) == 0 and "Diagrammet kan inte ritas" in _text(at)
    assert "DATA_MISSING:" in _text(at) and "fyll i under Ark" in _text(at)


def test_empty_sheet_opens_ark_and_new_company_writes_to_shared_store(monkeypatch):
    from streamlit.testing.v1 import AppTest

    app = _app(monkeypatch, _conf(), register_rows=[{"ticker": "NYX", "name": "Nyx Gold", "strategy": "Momentum"}])
    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert at.radio(key="asym_mode").value == "Ark" and not at.metric
    assert any("Inga bolag i arket" in i.value for i in at.info)
    # registret → skal med strategi-tagg i det gemensamma arket
    at.button(key="asym_reg_NYX").click().run()
    assert not at.exception, at.exception
    s = at.session_state["confidence"]
    assert s["companies"]["NYX"]["name"] == "Nyx Gold" and s["strategies"]["NYX"] == "Momentum"
    # nytt bolag via formuläret
    at.text_input(key="asym_new_ticker").set_value("abc").run()
    at.selectbox(key="asym_new_strategy").set_value("Viking").run()
    at.button(key="FormSubmitter:asym_new-Lägg till").click().run()
    assert not at.exception, at.exception
    s = at.session_state["confidence"]
    assert "ABC" in s["companies"] and s["strategies"]["ABC"] == "Viking"
    assert "asymmetry" not in s                                 # inget andra lager


def test_ark_applies_borsdata_and_fields_on_shared_store(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from confidence import config as cfg

    app = _app(monkeypatch, _conf(dcs.gold_producer, tags={"GPR": "Viking"}), blob=BLOB)
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["asym_mode"] = "Ark"
    at.run()
    assert not at.exception, at.exception
    labels = " ".join(e.label for e in at.expander)
    for needle in ("Nytt bolag", "Hämta från registret", "Identitet och strategi", "Börsdata 2026-09-25",
                   "Förslag ur granskningsarken", "Läs ur presentationen", "Råvaran", "Signaler till Why Now"):
        assert needle in labels, needle
    at.button(key="asym_rf_GPR_market_cap_musd").click().run()
    assert not at.exception, at.exception
    f = at.session_state["confidence"]["companies"]["GPR"]["fields"]
    assert f["market_cap_musd"]["value"] == 1600.0 and "Börsdata" in f["market_cap_musd"]["source"]
    # ett fält i arket
    at.text_input(key="cf_GPR_aisc_v").set_value("1400").run()
    pillar = next(x.pillar for x in cfg.fields_for("producer") if x.key == "aisc")
    at.button(key=f"FormSubmitter:conf_form_GPR_{pillar}-Uppdatera").click().run()
    assert not at.exception, at.exception
    assert at.session_state["confidence"]["companies"]["GPR"]["fields"]["aisc"]["value"] == 1400.0
    # strategi-taggen
    at.selectbox(key="asym_strat_GPR").set_value("Ember").run()
    at.button(key="FormSubmitter:asym_ident_GPR-Uppdatera").click().run()
    assert not at.exception, at.exception
    assert at.session_state["confidence"]["strategies"]["GPR"] == "Ember"


def test_legacy_asymmetry_store_is_merged_once(monkeypatch):
    from streamlit.testing.v1 import AppTest

    legacy = {"companies": {"CDV": dcs.copper_developer().as_dict()}, "strategies": {"CDV": "Ember"},
              "commodity_overrides": {}}
    at = AppTest.from_function(_app(monkeypatch, _conf(dcs.gold_producer), legacy=legacy), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("Flyttade CDV" in i.value for i in at.info)
    s = at.session_state["confidence"]
    assert "CDV" in s["companies"] and s["strategies"]["CDV"] == "Ember"
    assert at.session_state["asym_migrated"] is True
    at.run()
    assert not any("Flyttade" in i.value for i in at.info)     # bara en gång per session
