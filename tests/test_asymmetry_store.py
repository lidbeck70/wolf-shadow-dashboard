"""
Wolf Asymmetry, steg D: eget ark (data/asymmetry.json), strategi-tagg och
Ark-vyn. Fliken läser inte Durrett-arket, och det den skriver hamnar bara
i sitt eget lager.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import durrett_cases as dcs  # noqa: E402
from asymmetry import store as ast  # noqa: E402
from confidence import store as cs  # noqa: E402
from confidence.data.models import CompanyInput  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── lagret ───────────────────────────────────────────────────────────────────
def test_store_shape_and_strategy_tags():
    d = ast.default()
    assert set(d) == {"companies", "commodity_overrides", "strategies"} and ast.STORE == "asymmetry"
    assert ast.STORE != cs.STORE
    assert ast.normalize({"companies": {"X": {}}})["strategies"] == {}
    assert ast.normalize(None) == ast.default()
    tags = ast.strategy_tags()
    assert tags[0] == ast.NO_STRATEGY and "Viking" in tags and "Durrett" in tags and "Untagged" not in tags


def test_put_get_strategy_filter_remove():
    d = ast.default()
    ast.put(d, dcs.gold_producer(), "Viking")
    ast.put(d, dcs.copper_developer(), "Ember")
    ast.put(d, dcs.lithium_explorer())
    assert list(ast.companies(d)) == ["GPR", "CDV", "LEX"]
    assert ast.strategy(d, "GPR") == "Viking" and ast.strategy(d, "lex") == ast.NO_STRATEGY
    assert ast.tickers_for(d, "Viking") == ["GPR"] and ast.tickers_for(d, "Alla") == ["GPR", "CDV", "LEX"]
    assert ast.tickers_for(d, ast.NO_STRATEGY) == ["LEX"]
    ast.set_strategy(d, "GPR", ast.NO_STRATEGY)
    assert "GPR" not in d["strategies"]
    ast.remove(d, "CDV")
    assert "CDV" not in d["companies"] and "CDV" not in d["strategies"]
    assert ast.get(d, "GPR").num("aisc") == 1450         # samma CompanyInput-form som confidence.store


# ── fliken: eget lager, Durrett-arket orört ──────────────────────────────────
def _app(monkeypatch, asym: dict, conf: dict):
    import streamlit as st
    import storage

    stores = {"asymmetry": asym, "confidence": conf}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, stores.get(name, default)))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    monkeypatch.setenv("ASYM_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["ASYM_TEST_ROOT"])
        from asymmetry.ui import render_asymmetry_page
        render_asymmetry_page()

    return app


def test_tab_ignores_the_durrett_sheet(monkeypatch):
    from streamlit.testing.v1 import AppTest

    conf = cs.default()
    cs.put(conf, dcs.gold_producer())                    # finns bara i Durrett-arket
    at = AppTest.from_function(_app(monkeypatch, ast.default(), conf), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("Nytt bolag" in i.value for i in at.info) and not at.metric
    assert "GPR" not in at.session_state["asymmetry"]["companies"]


def test_new_company_strategy_and_fields_write_only_to_own_store(monkeypatch):
    from streamlit.testing.v1 import AppTest

    app = _app(monkeypatch, ast.default(), cs.default())
    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    at.text_input(key="asym_new_ticker").set_value("nyx").run()
    at.text_input(key="asym_new_name").set_value("Test Nyx Gold").run()
    at.selectbox(key="asym_new_stage").set_value("producer").run()
    at.selectbox(key="asym_new_strategy").set_value("Viking").run()
    at.button(key="FormSubmitter:asym_new-Lägg till").click().run()
    assert not at.exception, at.exception
    store = at.session_state["asymmetry"]
    assert store["companies"]["NYX"]["name"] == "Test Nyx Gold" and store["companies"]["NYX"]["stage"] == "producer"
    assert store["strategies"]["NYX"] == "Viking"
    assert at.session_state["confidence"]["companies"] == {}          # Durrett-arket orört
    assert at.session_state["asym_sub"] == "Ark"
    # arket: fält med källa skrivs till asymmetry-lagret
    assert any("Wolf Asymmetrys eget ark" in c.value for c in at.caption)
    at.text_input(key="cf_NYX_aisc_v").set_value("1400").run()
    at.text_input(key="cf_NYX_aisc_s").set_value("MD&A Q2").run()
    from confidence import config as cfg
    pillar = next(f.pillar for f in cfg.fields_for("producer") if f.key == "aisc")
    at.button(key=f"FormSubmitter:conf_form_NYX_{pillar}-Uppdatera").click().run()
    assert not at.exception, at.exception
    fields = at.session_state["asymmetry"]["companies"]["NYX"]["fields"]
    assert fields["aisc"]["value"] == 1400.0 and fields["aisc"]["source"] == "MD&A Q2"
    assert at.session_state["confidence"]["companies"] == {}
    # strategi-taggen ändras i Ark och styr filtret
    at.selectbox(key="asym_strat_NYX").set_value("Ember").run()
    at.button(key="FormSubmitter:asym_ident_NYX-Uppdatera").click().run()
    assert not at.exception, at.exception
    assert at.session_state["asymmetry"]["strategies"]["NYX"] == "Ember"
    assert "Ember" in at.selectbox(key="asym_strategy_filter").options


def test_strategy_filter_narrows_the_company_list(monkeypatch):
    from streamlit.testing.v1 import AppTest

    asym = ast.default()
    ast.put(asym, dcs.gold_producer(), "Viking")
    ast.put(asym, dcs.copper_developer(), "Ember")
    at = AppTest.from_function(_app(monkeypatch, asym, cs.default()), default_timeout=60)
    at.session_state["asym_strategy_filter"] = "Ember"
    at.run()
    assert not at.exception, at.exception
    assert [o.split(" · ")[0] for o in at.selectbox(key="asym_pick").options] == ["CDV"]
    assert at.selectbox(key="asym_pick").options == ["CDV · Ember"]        # taggen syns i listan
    assert next(m for m in at.metric if m.label == "Margin of Safety").value == "10/10"
