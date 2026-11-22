"""LIDSKJALV PR 18: Morgonrundan (ui/routine.py) — bara navigering. Stegen är mål i nav.TARGETS, rundan nollställs
en ny dag, pillret och foten har sina nycklar och etiketter, och de andra sidorna är rena tills rundan startats
från I DAG."""
from datetime import datetime

import pytest

from ui import nav, routine

DAY, NEXT_DAY = "2026-11-21", "2026-11-22"


def test_every_step_is_a_target_in_spec_order():
    assert routine.STEPS == (("home", "I DAG"), ("laget", "Läget"), ("thor", "Thor"), ("scorecard", "🎯 Scorecard"),
                             ("holdings", "Positioner"))
    for key, _label in routine.STEPS:
        assert key in nav.TARGETS, key
    assert [nav.target_slug(k) for k in routine.KEYS] == ["home", "odin/laget", "thor", "review/scorecard",
                                                          "portfolio/positioner"]
    assert [routine.next_step(k) for k in routine.KEYS] == ["laget", "thor", "scorecard", "holdings", "home"]


def test_mark_starts_from_i_dag_and_resets_on_a_new_date():
    state = {}
    assert routine.mark("home", state, DAY) == {"date": DAY, "seen": ["home"]}
    routine.mark("laget", state, DAY)
    routine.mark("laget", state, DAY)                                   # samma steg två gånger: en gång
    assert state["morning"] == {"date": DAY, "seen": ["home", "laget"]}
    assert routine.active(state, DAY) and not routine.active(state, NEXT_DAY)
    routine.mark("thor", state, NEXT_DAY)                               # ny dag: gårdagens runda glöms
    assert state["morning"] == {"date": NEXT_DAY, "seen": ["thor"]} and not routine.active(state, NEXT_DAY)
    routine.mark("home", state, NEXT_DAY)
    routine.mark("scorecard", state, NEXT_DAY)
    routine.mark("home", state, NEXT_DAY)                               # I DAG igen: rundan börjar om
    assert state["morning"] == {"date": NEXT_DAY, "seen": ["home"]}
    with pytest.raises(ValueError):
        routine.mark("odin", state, DAY)
    assert set(state) == {"morning"}                                    # ingen data, ingen regel — bara stegen


def test_mark_uses_the_swedish_date(monkeypatch):
    from tabs import home_today as ht
    monkeypatch.setattr(ht, "now_se", lambda: datetime(2026, 11, 22, 0, 30))   # servern på UTC: fortfarande den 21:a
    state = {}
    routine.mark("home", state)
    assert state["morning"]["date"] == "2026-11-22" and routine.active(state)


def test_other_pages_are_clean_until_the_round_is_started_and_after_it_is_done():
    state = {}
    assert routine.shows("home", state, DAY)
    assert not any(routine.shows(k, state, DAY) for k in routine.KEYS[1:])
    routine.mark("laget", state, DAY)                                   # utan start från I DAG: fortfarande rent
    assert not routine.shows("thor", state, DAY)
    routine.mark("home", state, DAY)
    assert all(routine.shows(k, state, DAY) for k in routine.KEYS)
    assert not routine.shows("thor", state, NEXT_DAY)                   # i morgon: ny runda
    for k in routine.KEYS[1:]:
        routine.mark(k, state, DAY)
    assert not routine.active(state, DAY) and not routine.shows("laget", state, DAY)
    assert routine.shows("home", state, DAY)


def test_advance_marks_the_step_and_opens_the_next_page(monkeypatch):
    monkeypatch.setattr(routine, "_today", lambda: DAY)
    state = {}
    routine.advance("home", state)
    assert state["nav_goto"] == "odin/laget" and state["morning"] == {"date": DAY, "seen": ["home"]}
    routine.advance("laget", state)
    assert state["nav_goto"] == "thor" and "thor_ticker" not in state
    routine.advance("scorecard", state)
    assert state["nav_goto"] == "portfolio/positioner"
    routine.advance("holdings", state)
    assert state["nav_goto"] == "home" and not routine.active(state, DAY)


def test_labels_and_dots():
    assert [routine.next_label(k) for k in routine.KEYS] == [
        "Nästa: Läget →", "Nästa: Thor →", "Nästa: 🎯 Scorecard →", "Nästa: Positioner →",
        "✓ Morgonrundan klar · Till I DAG ↑"]
    dots = routine.dots_html("thor")
    assert dots.count("●") == 3 and dots.count("○") == 2 and "3/5" in dots
    assert 'aria-label="Morgonrundan: steg 3 av 5"' in dots
    assert routine.dots_html("home").count("●") == 1


# ── Knapparna i en körning ───────────────────────────────────────────────────
def _app(step):
    from ui import routine as r
    r.pill(step)
    r.footer(step)


def _run(step, state=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_app, args=(step,), default_timeout=30)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    return at.run()


def test_pill_and_footer_keys_labels_and_help(monkeypatch):
    monkeypatch.setattr(routine, "_today", lambda: DAY)
    at = _run("home")
    assert not at.exception, at.exception
    buttons = {b.key: b for b in at.button}
    assert set(buttons) == {"morning_top_home", "morning_next_home"}
    assert buttons["morning_top_home"].label == "1/5 ›"
    assert buttons["morning_top_home"].help == "Morgonrundan — nästa: Läget"
    assert buttons["morning_next_home"].label == "Nästa: Läget →"
    assert buttons["morning_top_home"].proto.type == "tertiary" and buttons["morning_next_home"].proto.type == "primary"
    assert "morning" not in at.session_state                            # inget skrivs när sidan öppnas
    styles = [m.value for m in at.markdown if "<style>" in m.value]
    assert len(styles) == 1 and "morning_top_" in styles[0]            # rundans CSS en gång, fast två knappar
    at.button(key="morning_next_home").click().run()
    assert at.session_state["nav_goto"] == "odin/laget"
    assert at.session_state["morning"] == {"date": DAY, "seen": ["home"]}


def test_other_steps_render_nothing_until_started(monkeypatch):
    monkeypatch.setattr(routine, "_today", lambda: DAY)
    for step in routine.KEYS[1:]:
        at = _run(step)
        assert not at.exception, at.exception
        assert not list(at.button) and not [m for m in at.markdown if "wolf-dots" in m.value], step
    started = {"morning": {"date": DAY, "seen": ["home"]}}
    at = _run("laget", started)
    assert {b.key: b.label for b in at.button} == {"morning_top_laget": "2/5 ›", "morning_next_laget": "Nästa: Thor →"}
    at = _run("holdings", {"morning": {"date": DAY, "seen": ["home", "laget", "thor", "scorecard"]}})
    assert {b.key: b.label for b in at.button} == {"morning_top_holdings": "5/5 ›",
                                                   "morning_next_holdings": "✓ Morgonrundan klar · Till I DAG ↑"}
    at.button(key="morning_next_holdings").click().run()
    assert at.session_state["nav_goto"] == "home" and not list(at.button)          # klar: Positioner är ren igen
    at = _run("thor", {"morning": {"date": "2026-11-20", "seen": ["home"]}})         # gårdagens runda: ren
    assert not list(at.button)


def test_i_dag_wires_the_pill_in_the_head_row_and_the_footer(monkeypatch):
    import routines
    from tabs import home_today as ht
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(ht, "now_se", lambda: datetime(2026, 11, 21, 8, 0))
    monkeypatch.setattr(ht, "load", lambda loader=None: (_ for _ in ()).throw(AssertionError("laddade")))
    monkeypatch.setattr(routines, "due_today", lambda day: [])

    def app():
        from tabs import home
        home.tab_home()

    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["home_today"] = {"board": {}, "thor": {"rows": []}, "alert_state": {},
                                      "loaded": "2026-11-21 07:55"}
    at.session_state["settings"] = {"capital": 100000.0}
    at.run()
    assert not at.exception, at.exception
    keys = [b.key for b in at.button]
    assert keys.index("home_refresh") < keys.index("morning_top_home") < keys.index("morning_next_home")
    assert at.button(key="morning_top_home").label == "1/5 ›"
    assert "morning" not in at.session_state
    at.button(key="morning_top_home").click().run()
    assert at.session_state["nav_goto"] == "odin/laget" and at.session_state["morning"]["seen"] == ["home"]
    src = open(routine.__file__, encoding="utf-8").read()
    assert "load(" not in src and "save" not in src.lower() and "storage" not in src     # ingen data i rundan


# ── LIDSKJALV PR 19: rundans steg på sidorna — Läget, Thor (listan), 🎯 Scorecard, Positioner (upplåst) ──────────
import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTED = {"morning": {"date": DAY, "seen": ["home"]}}


def _round_keys(at) -> dict:
    return {b.key: b.label for b in at.button if str(b.key).startswith("morning_")}


def _page(script, state=None, timeout=120):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(script, default_timeout=timeout)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    return at.run()


def _laget():
    from odin import laget
    laget.render_laget_page()


def _thor():
    from thor import ui as t
    t.render_thor_page()


def _scorecard():
    import scorecard
    scorecard.render_scorecard_page()


def _positioner():
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.environ["MORNING_TEST_ROOT"])
    import wolf_panel
    wolf_panel._tab_portfolio()


def _patch_laget(monkeypatch):
    import wolf_regime_ui
    from odin import laget
    monkeypatch.setattr(laget, "load", lambda: {"alloc": None, "regime": None, "risk": None, "board": None})
    monkeypatch.setattr(wolf_regime_ui, "_load_wolf", lambda name: {"regime": "GUL", "generated": "2026-11-20T18:05",
                                                                    "rules": ["Halv storlek."]})


def _patch_store(monkeypatch):
    import storage
    import streamlit as st
    monkeypatch.setattr(storage, "session_load",
                        lambda name, default=None, legacy_file=None: st.session_state.setdefault(name, default))


def _patch_positioner(monkeypatch):
    import copy

    import holdings
    import refresh_ui
    import storage
    seed = {"holdings": {"long": [{"id": "q1", "ticker": "BOL.ST", "strategy": "Quality", "entry_price": 70.0,
                                   "shares": 10}], "cash": 1000}}
    monkeypatch.setattr(storage, "load_json", lambda name, default=None: copy.deepcopy(seed.get(name, default)))
    monkeypatch.setattr(storage, "legacy_gist", lambda *a, **k: None)
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(holdings, "_fetch_live_price", lambda t, period="6mo": {"price": 0, "change_1d": 0, "df": None})
    monkeypatch.setattr(holdings, "_get_regime_signal", lambda t, s: None)
    monkeypatch.setenv("MORNING_TEST_ROOT", ROOT)


def _assert_round(at, step, shown):
    assert not at.exception, at.exception
    keys = _round_keys(at)
    if not shown:
        assert keys == {}, (step, keys)
        assert not [m for m in at.markdown if "wolf-dots" in m.value], step
        return
    i = routine.index(step) + 1
    assert keys == {f"morning_top_{step}": f"{i}/5 ›", f"morning_next_{step}": routine.next_label(step)}, keys
    order = [b.key for b in at.button]
    assert order.index(f"morning_top_{step}") < order.index(f"morning_next_{step}")   # pillret överst, foten sist
    assert order[-1] == f"morning_next_{step}" or step == "holdings", order


def test_laget_has_the_pill_in_the_head_row_and_the_footer_only_during_the_round(monkeypatch):
    monkeypatch.setattr(routine, "_today", lambda: DAY)
    _patch_laget(monkeypatch)
    _assert_round(_page(_laget), "laget", False)
    at = _page(_laget, STARTED)
    _assert_round(at, "laget", True)
    keys = [b.key for b in at.button]
    assert keys[:2] == ["laget_refresh", "morning_top_laget"]          # huvudraden: 📡 och pillret
    assert keys[-1] == "morning_next_laget"
    assert at.session_state["morning"] == STARTED["morning"]          # inget skrivs när sidan öppnas
    at.button(key="morning_top_laget").click().run()
    assert at.session_state["nav_goto"] == "thor" and at.session_state["morning"]["seen"] == ["home", "laget"]


def test_thor_shows_the_round_in_list_mode_only(monkeypatch):
    import test_thor_ui as tu
    monkeypatch.setattr(routine, "_today", lambda: DAY)
    tu._patch(monkeypatch)
    _assert_round(_page(_thor), "thor", False)
    at = _page(_thor, STARTED)
    _assert_round(at, "thor", True)
    assert "32 KANDIDATER" in " ".join(m.value for m in at.markdown)  # listan som förut
    at = _page(_thor, {**STARTED, "thor_ticker": "bol.st"})           # analysen: inte rundans steg
    _assert_round(at, "thor", False)
    assert "VAD SKA JAG GÖRA?" in " ".join(m.value for m in at.markdown)


def test_scorecard_shows_the_round_with_and_without_candidates(monkeypatch):
    import test_thor_scorecard_entry as tse
    monkeypatch.setattr(routine, "_today", lambda: DAY)
    _patch_store(monkeypatch)
    _assert_round(_page(_scorecard), "scorecard", False)
    _assert_round(_page(_scorecard, STARTED), "scorecard", True)                     # inga kandidater: foten ändå
    at = _page(_scorecard, {**STARTED, **tse._stores()})
    _assert_round(at, "scorecard", True)
    assert {m.label: m.value for m in at.metric}["Klara för köp"] == "1"            # sidan som förut
    at.button(key="morning_next_scorecard").click().run()
    assert at.session_state["nav_goto"] == "portfolio/positioner"


def test_positioner_shows_the_pill_only_after_unlock_and_the_closing_footer_always(monkeypatch):
    """Låst: bara foten ('✓ Morgonrundan klar') under låskortet — navigering, inga innehav — så rundan kan avslutas;
    annars stod pillret och foten kvar på Läget, Thor och Scorecard resten av dagen."""
    from portfolio import private_lock as pl
    monkeypatch.setattr(routine, "_today", lambda: DAY)
    _patch_positioner(monkeypatch)
    late = {"morning": {"date": DAY, "seen": ["home", "laget", "thor", "scorecard"]}, "sub_portfolio": "Positioner"}
    at = _page(_positioner, late)                                                    # låst: bara foten
    assert not at.exception, at.exception
    assert _round_keys(at) == {"morning_next_holdings": routine.DONE}
    assert any("låsta" in m.value for m in at.markdown)
    at.button(key="morning_next_holdings").click().run()
    assert at.session_state["nav_goto"] == "home"
    assert not routine.active({"morning": at.session_state["morning"]}, DAY)        # rundan är klar
    at = _page(_positioner, {**late, pl.STATE: True})                                # upplåst: pillret och foten
    _assert_round(at, "holdings", True)
    assert at.button(key="morning_next_holdings").label == "✓ Morgonrundan klar · Till I DAG ↑"
    at = _page(_positioner, {"sub_portfolio": "Positioner", pl.STATE: True})        # upplåst utan runda: rent
    _assert_round(at, "holdings", False)


def _chain():
    """Panelens väg mellan rundans sidor: nav_goto → sidans steg (som wolf_panel._apply_deep_link) → sidans pill."""
    import streamlit as st
    from ui import nav, routine as r
    slug = st.session_state.pop("nav_goto", None) or st.session_state.get("_at", "home")
    st.session_state["_at"] = slug
    step = {nav.target_slug(k): k for k in r.KEYS}[slug]
    st.session_state["_steps"] = st.session_state.get("_steps", []) + [step]
    r.pill(step)
    r.footer(step)


def test_four_taps_from_i_dag_reach_positioner(monkeypatch):
    """§8.11: från I DAG når pillret Positioner på fyra tryck — Läget, Thor, 🎯 Scorecard, Positioner."""
    monkeypatch.setattr(routine, "_today", lambda: DAY)
    at = _page(_chain)
    taps = 0
    for step in routine.KEYS[:-1]:
        assert at.session_state["_steps"][-1] == step
        at.button(key=f"morning_top_{step}").click().run()
        taps += 1
    assert taps == 4 and at.session_state["_steps"][-1] == "holdings" and at.session_state["_at"] == "portfolio/positioner"
    assert at.session_state["morning"]["seen"] == ["home", "laget", "thor", "scorecard"]
    at.button(key="morning_next_holdings").click().run()                             # klar: tillbaka till I DAG
    assert at.session_state["_at"] == "home" and not routine.active({"morning": at.session_state["morning"]}, DAY)


def test_step_pages_wire_the_round_in_the_source():
    import pathlib
    root = pathlib.Path(ROOT)
    for path, step in (("odin/laget.py", "laget"), ("thor/ui.py", "thor"), ("scorecard.py", "scorecard"),
                       ("holdings.py", "holdings")):
        src = (root / path).read_text(encoding="utf-8")
        assert f'routine.pill("{step}")' in src, path
    panel = (root / "wolf_panel.py").read_text(encoding="utf-8")
    gate = panel.split("if gate():", 1)[1].split("elif sub ==", 1)[0]
    assert 'routine.footer("holdings")' in gate                       # Positionerns fot efter sidan (och låskortet)
