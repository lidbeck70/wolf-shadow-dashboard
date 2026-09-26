"""
Wolf Asymmetry, steg F: Confidence-caset inbyggt (Confidence · Råvaror ·
Signaler) på Wolf Asymmetrys ark. Under GRANSKNING heter Durrett-fliken
🧭 Durrett och har ingen Confidence-undervy längre. Gamla djuplänkar
landar på Durrett.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import durrett_cases as dcs  # noqa: E402
from asymmetry import store as ast  # noqa: E402
from confidence import store as cs  # noqa: E402
from ui import nav  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


# ── navigation ───────────────────────────────────────────────────────────────
def test_durrett_tab_is_standalone_and_confidence_lives_in_wolf_asymmetry():
    assert "🧭 Durrett" in nav.options("review") and "🧭 Durrett & Confidence" not in nav.options("review")
    assert "review/🧭 Durrett" not in nav.SUBS and "review/🧭 Durrett & Confidence" not in nav.STATE_KEY
    assert "Confidence-case" not in nav.leaves() and "Durrett 10-steg" not in nav.leaves()
    assert nav.resolve("review/durrett") == ["review", "🧭 Durrett"]
    assert nav.resolve("granskning/durrett-confidence") == ["review", "🧭 Durrett"]
    assert nav.resolve("granskning/durrett-confidence/durrett-10-steg") == ["review", "🧭 Durrett"]
    src = _src("wolf_panel.py")
    assert 'sub == "🧭 Durrett"' in src and "render_confidence_page" not in src
    guide = _src(os.path.join("ovtlyr", "ui", "rules_page.py"))
    assert "🧭 Durrett & Confidence" not in guide and "GRANSKNING → 🧭 Durrett\"" in guide
    from asymmetry.ui import SUBS
    assert ("Confidence", "Råvaror", "Signaler") == tuple(s for s in SUBS if s in ("Confidence", "Råvaror", "Signaler"))


# ── fliken ───────────────────────────────────────────────────────────────────
def _app(monkeypatch, asym: dict):
    import streamlit as st
    import storage
    import positions
    import refresh_ui

    stores = {"asymmetry": asym, "confidence": cs.default(), "producers": {}, "tiggre": {}}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, stores.get(name, default)))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    monkeypatch.setattr(positions, "open_positions", lambda strategy=None, bucket=None: [])
    monkeypatch.setattr(positions, "view_rows", lambda strategy=None: [])
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setenv("ASYM_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["ASYM_TEST_ROOT"])
        from asymmetry.ui import render_asymmetry_page
        render_asymmetry_page()

    return app


def test_confidence_view_renders_case_and_confidence_on_own_store(monkeypatch):
    from streamlit.testing.v1 import AppTest

    asym = ast.default()
    ast.put(asym, dcs.gold_producer(), "Viking")
    ast.put(asym, dcs.missing_everything())
    app = _app(monkeypatch, asym)
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["asym_sub"] = "Confidence"
    at.run()
    assert not at.exception, at.exception
    vals = {m.label: m.value for m in at.metric}
    assert {"Case Score", "Confidence", "Why Now", "Regional knapphet", "Time-to-money", "Asymmetri"} <= set(vals)
    assert float(vals["Confidence"]) > 60                       # GPR: väl belagt
    text = " ".join(m.value for m in at.markdown)
    assert "Thesis Killer" in text and "Scenarier" in text
    labels = " ".join(e.label for e in at.expander)
    assert "Investment Card" in labels
    assert "Commodity Leverage" not in vals                     # egna KPI:er, inte asymmetri-raden
    # tomt bolag: DATA_MISSING, ingen krasch
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["asym_pick"] = "NUL"
    at.session_state["asym_sub"] = "Confidence"
    at.run()
    assert not at.exception, at.exception
    vals = {m.label: m.value for m in at.metric}
    assert float(vals["Confidence"]) < 10


def test_commodity_overrides_write_to_own_store(monkeypatch):
    from streamlit.testing.v1 import AppTest

    asym = ast.default()
    ast.put(asym, dcs.gold_producer(), "Viking")
    app = _app(monkeypatch, asym)
    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["asym_sub"] = "Råvaror"
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="conf_com").value == "gold"
    key = next(t.key for t in at.text_input if t.key.startswith("cc_gold_") and t.key.endswith("_v"))
    field = key[len("cc_gold_"):-2]
    at.text_input(key=key).set_value("12").run()
    at.button(key=next(b.key for b in at.button if b.key.startswith("FormSubmitter:") and "cc_" in b.key
                       or b.key.startswith("FormSubmitter:conf_com"))).click().run()
    assert not at.exception, at.exception
    ov = at.session_state["asymmetry"]["commodity_overrides"]
    assert ov["gold"][field]["value"] == 12.0
    assert at.session_state["confidence"]["commodity_overrides"] == {}          # Durrett-arket orört


def test_signals_view_renders(monkeypatch):
    from streamlit.testing.v1 import AppTest

    asym = ast.default()
    ast.put(asym, dcs.copper_developer(), "Ember")
    at = AppTest.from_function(_app(monkeypatch, asym), default_timeout=60)
    at.session_state["asym_sub"] = "Signaler"
    at.run()
    assert not at.exception, at.exception
    assert any(b.key == "conf_sig_rot" for b in at.button)
