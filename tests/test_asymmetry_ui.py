"""
Wolf Asymmetry, steg B: fliken 🐺 Wolf Asymmetry under GRANSKNING.

Navigationen känner fliken, FLIKGUIDEN har en rad, och sidan ritas utan
nätverk för en producent, en utvecklare och ett tomt bolag (DATA_MISSING
utan krasch) i alla sex undervyer.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import durrett_cases as dcs  # noqa: E402
from ui import nav  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = "🐺 Wolf Asymmetry"


def _src(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


# ── navigation ───────────────────────────────────────────────────────────────
def test_tab_is_wired_under_review():
    review = nav.options("review")
    assert TAB in review
    assert review.index(TAB) == review.index("🧭 Durrett & Confidence") + 1
    assert TAB not in nav.options("screening")
    assert nav.slugify(TAB) == "wolf-asymmetry"
    assert nav.resolve("granskning/wolf-asymmetry") == ["review", TAB]
    assert any(t == TAB for _k, _h, items in nav.HOME_ZONES if _k == "review" for t, _d in items)
    src = _src("wolf_panel.py")
    assert "from asymmetry.ui import render_asymmetry_page" in src
    assert f'sub == "{TAB}"' in src
    assert f"GRANSKNING → {TAB}" in _src(os.path.join("ovtlyr", "ui", "rules_page.py"))


# ── sidan ────────────────────────────────────────────────────────────────────
def _app_factory(monkeypatch, companies):
    import streamlit as st
    import storage
    from confidence import store as cs

    data = cs.default()
    for mk in companies:
        cs.put(data, mk())
    stores = {"confidence": data}
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


def _text(at) -> str:
    return (" ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)
            + " " + " ".join(e.label for e in at.expander))


def test_page_renders_every_subview_without_network(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from asymmetry.ui import SUBS

    app = _app_factory(monkeypatch, (dcs.gold_producer, dcs.copper_developer, dcs.missing_everything))
    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert "WOLF ASYMMETRY" in _text(at)
    labels = {m.label for m in at.metric}
    assert {"Commodity Leverage", "Margin of Safety", "Break-even-marginal", "Confidence",
            "Uppsida (Base)", "Justerad uppsida"} <= labels
    lev = next(m for m in at.metric if m.label == "Commodity Leverage")
    assert lev.value == "6/10"
    assert next(m for m in at.metric if m.label == "Margin of Safety").value == "7/8"
    assert len(at.dataframe) >= 1                       # prisgriden

    for sub in SUBS:
        at = AppTest.from_function(app, default_timeout=60)
        at.session_state["asym_sub"] = sub
        at.run()
        assert not at.exception, (sub, at.exception)
        text = _text(at)
        if sub == "Varför?":
            assert "Commodity Leverage 6/10" in text and "tabell:" in text
        elif sub == "Scenarier":
            assert "Asymmetri" in " ".join(m.label for m in at.metric)
            assert any("×" in e.label and "vad krävs" in e.label for e in at.expander)
        elif sub == "Stressmatris":
            assert len(at.dataframe) == 2 and "Producent" in text
        elif sub == "Thesis killers":
            assert "Råvarupris" in text
        elif sub == "Data":
            assert "Antaganden" in text and "Confidence" in text


def test_developer_matrix_and_missing_company(monkeypatch):
    from streamlit.testing.v1 import AppTest

    app = _app_factory(monkeypatch, (dcs.copper_developer, dcs.missing_everything))
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["asym_sub"] = "Stressmatris"
    at.run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "Producent" not in text and "Värsta rutan" in text
    assert next(m for m in at.metric if m.label == "Margin of Safety").value == "10/10"

    # tomt bolag: allt DATA_MISSING, inget påhittat, ingen krasch i någon vy
    from asymmetry.ui import SUBS
    for sub in SUBS:
        at = AppTest.from_function(app, default_timeout=60)
        at.session_state["asym_pick"] = "NUL"
        at.session_state["asym_sub"] = sub
        at.run()
        assert not at.exception, (sub, at.exception)
        vals = {m.label: m.value for m in at.metric}
        assert vals["Commodity Leverage"] == "DATA_MISSING"
        assert vals["Margin of Safety"] == "DATA_MISSING"
        assert vals["Uppsida (Base)"] == "DATA_MISSING" and vals["Justerad uppsida"] == "DATA_MISSING"
        assert "Saknade fält" in _text(at)


def test_empty_store_points_to_durrett(monkeypatch):
    from streamlit.testing.v1 import AppTest

    app = _app_factory(monkeypatch, ())
    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("Durrett & Confidence" in i.value for i in at.info)
    assert not at.metric
