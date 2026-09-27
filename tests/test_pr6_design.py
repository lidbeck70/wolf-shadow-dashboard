"""
PR 6 av panelgenomgången — design.

Ett navigationsträd (ui/nav.py) som panelen, FLIKGUIDEN och Home ritar ur.
Ett komponentkit (ui/tokens.py, ui/components.py) i stället för sex
rubrikmönster och fem identiska _badge-kopior. Global CSS som inte längre
skriver över sidornas rubrikfärger, primärknappar och felmeddelanden.
Home som läser det som faktiskt finns sparat. Tomt = saknas i arken.
"""
import os
import re
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ui import nav, tokens, components   # noqa: E402


def _src(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


# ── Navigationsträdet ────────────────────────────────────────────────────────
def test_nav_tree_is_consistent():
    keys = {k for k, _l in nav.TOP}
    for path in nav.SUBS:
        head, *rest = path.split("/")
        assert head in keys, path
        # varje mellannivå i en sökväg är ett alternativ på nivån ovanför
        for i, part in enumerate(rest):
            parent = "/".join([head] + rest[:i])
            assert part in nav.SUBS.get(parent, []), f"{path}: '{part}' finns inte under '{parent}'"
    for opts in nav.SUBS.values():
        assert len(opts) == len(set(opts))
    assert "🧭 Durrett" in nav.options("review") and "🐺 Wolf Asymmetry" in nav.options("review")
    assert "🐺 Durrett" not in nav.leaves() and "🧭 Confidence score" not in nav.leaves()
    assert nav.options("regime") == ["Marknad", "Råvaror"]
    assert "Market Cycle" in nav.options("regime/Marknad")
    assert "Market Cycle" not in nav.options("screening")
    assert nav.options("screening/Contrarian Alpha")[0] == "Screener"     # inte "Contrarian Alpha → Contrarian Alpha"
    assert "REGIME → Råvaror → 🌍 EMBER Regime" in nav.paths()


def test_panel_renders_its_radios_from_the_tree():
    src = _src("wolf_panel.py")
    assert 'st.radio(\n                "",' not in src and 'st.radio("",' not in src
    assert "nav.options(path)" in src and "nav.TOP" in src
    assert "_render_copilot_stub" not in src and "tab_screener_consolidated" not in src
    assert "user-scalable=no" not in src


def test_home_zones_cover_review_and_only_name_real_tabs():
    zone_keys = [k for k, _t, _c in nav.HOME_ZONES]
    assert "review" in zone_keys and zone_keys[0] == "regime"     # guidens ordning
    leaves = nav.leaves()
    for _k, _t, cards in nav.HOME_ZONES:
        for name, _desc in cards:
            for part in name.split(" · "):
                assert part in leaves, f"Home-kortet '{part}' är ingen flik"


def test_flikguide_paths_exist_in_the_tree_and_cover_every_top_tab():
    from ovtlyr.ui.rules_page import _PANEL_GUIDE
    leaves = nav.leaves()
    tops = {label.split(" ", 1)[1] for _k, label in nav.TOP}
    heads = set()
    for tab, _r, _u in _PANEL_GUIDE:
        parts = [p.strip() for p in tab.split("→")]
        assert parts[0] in tops, parts[0]
        heads.add(parts[0])
        for p in parts[1:]:
            assert p in leaves, f"FLIKGUIDE: '{p}' i '{tab}' är ingen flik"
    assert {"REGIME", "SCREENING", "GRANSKNING", "INTELLIGENCE", "PORTFOLIO",
            "ALERTS", "RULES", "COPILOT"} <= heads
    assert any("🧭 Durrett" in t for t, _r, _u in _PANEL_GUIDE)
    assert any("Wolf Asymmetry" in t for t, _r, _u in _PANEL_GUIDE)
    assert any("Allokering" in t for t, _r, _u in _PANEL_GUIDE)


def test_no_stale_navigation_text_remains():
    for path in (("ovtlyr", "ui", "rules_page.py"), ("strategy_rules.py"), ("ember", "ui.py")):
        src = _src(*path) if isinstance(path, tuple) else _src(path)
        assert "SIGNALS →" not in src, path
        assert "Arc Regime → 🌍 EMBER" not in src, path
        assert "SCREENING → Market Cycle" not in src, path


# ── Tokens, komponenter, CSS ─────────────────────────────────────────────────
def test_tokens_keep_error_and_warning_apart():
    from ui.theme import PALETTE
    assert tokens.RED != tokens.AMBER and PALETTE["red"] != PALETTE["amber"]
    assert tokens.regime_color("GRÖN") == tokens.GREEN
    assert tokens.regime_color("RÖD") == tokens.RED
    assert tokens.regime_color(None) == tokens.GREY


def test_components_render_expected_html():
    assert ">köp<" in components.badge("köp", tokens.GREEN) and tokens.GREEN in components.badge("x", tokens.GREEN)
    box = components.verdict_box("8 / 10", "Kärninnehav", "Full position.", tokens.GREEN)
    assert "8 / 10" in box and "Kärninnehav" in box and "Full position." in box
    assert "<b>" not in components.badge("<b>x</b>", tokens.RED)      # escapat
    k = components.kpi("Swing-regim", "GRÖN", tokens.GREEN, "2026-09-22")
    assert "SWING-REGIM" in k.upper() and "2026-09-22" in k


def test_global_css_no_longer_overrides_page_styling():
    css = _src("ui", "css.py")
    assert 'h1, h2, h3 {\n    color: #00E5FF;' in css              # ingen !important
    assert '.stButton > button[kind="secondary"] {' in css        # primärknappar lämnas
    assert ".stAlert {" not in css                                 # fel/varning/OK behåller färg


def test_review_sheets_share_one_header_and_one_badge():
    for path in ("producers.py", "scoring.py", "insider.py", "scorecard.py",
                 os.path.join("confidence", "ui.py"), os.path.join("engines", "durrett", "ui.py")):
        src = _src(path)
        assert "page_header(" in src, path
        assert "<h2 style='color:{CYAN};letter-spacing:0.12em;margin:0;'>" not in src, path
    for path in (os.path.join("confidence", "ui.py"), os.path.join("engines", "durrett", "ui.py")):
        assert "def _badge(" not in _src(path), path


def test_deletes_sit_behind_a_confirmation():
    for path in ("producers.py", "scoring.py", "insider.py",
                 os.path.join("confidence", "ui.py"), os.path.join("engines", "durrett", "ui.py")):
        src = _src(path)
        assert "confirm_delete(" in src, path
        assert not re.search(r'st\.button\("(?:🗑 )?Ta bort(?! sista)', src), path


# ── Home ─────────────────────────────────────────────────────────────────────
def test_home_speaks_swedish_and_reads_the_regime():
    from datetime import datetime
    from tabs import home
    assert home.swedish_date(datetime(2026, 9, 23)) == "onsdag 23 september 2026"
    assert "RÖD" in home._next_step({"swing": {"label": "RÖD"}}) and "inga nya köp" in home._next_step({"swing": {"label": "RÖD"}})
    assert "GUL" in home._next_step({"swing": {"label": "GUL"}})
    assert "Swing Screener" in home._next_step({"swing": {"label": "GRÖN"}})
    assert "wolf_data.py" in home._next_step({})
    src = _src("tabs", "home.py")
    assert "wolf_regime_status" not in src and "ALERT_LOG" not in src


# ── Tomt = saknas ────────────────────────────────────────────────────────────
def test_number_inputs_start_blank_when_nothing_is_stored():
    for path in ("producers.py", "scoring.py", "insider.py", "tiggre.py", "allocator.py",
                 "lukacs_ui.py", "controls_ui.py"):
        src = _src(path)
        assert not re.search(r"value=float\((?:_num|fv\.num)\([^\n]*\), 0\.0\) or 0\.0\)", src), path
        assert not re.search(r"storage\.differs\([^\n]*, 0\.0\)", src), path
    assert "Saknad data ger 0" not in _src("confidence", "ui.py")
    import storage
    assert storage.differs(None, None) is False
    assert storage.differs(0.0, None) is True             # ett inskrivet 0 är ett svar
    assert storage.differs(None, 0.0) is True
    assert storage.differs(0.0, None, 0.0) is False       # gamla widgets: som förut


def test_review_sheets_render_with_blank_fields(monkeypatch):
    """Arken ritas med rader där talen saknas — inga TypeError på None."""
    from streamlit.testing.v1 import AppTest
    import storage
    stores = {
        "producers": {"producers": [{"id": "p1", "ticker": "BOL.ST", "name": "Boliden",
                                     "commodity": "Koppar", "date": "2026-09-01", "factors": {}}],
                      "royalty": [{"id": "r1", "ticker": "FNV", "name": "Franco", "date": "2026-09-01"}]},
        "scoring": {"sprott": [{"id": "s1", "ticker": "JR", "name": "Junior", "commodity": "Guld",
                                "date": "2026-09-01", "factors": {}}],
                    "durrett": [{"id": "d1", "ticker": "AU", "name": "Gold", "commodity": "Guld",
                                 "date": "2026-09-01", "factors": {"balans": 2}}]},
        "insider": {"signals": [{"id": "i1", "ticker": "EKTA-B.ST", "name": "Elekta",
                                 "date": "2026-09-01"}]},
        "tiggre": {"candidates": [{"id": "c1", "ticker": "LOB", "name": "Lobo", "added": "2026-09-01",
                                   "screen": {}, "factors": {}, "catalysts": []}],
                   "positions": [{"id": "p1", "ticker": "AU", "name": "Au", "entry": 10.0,
                                  "current": 12.0, "shares": 100, "date": "2026-09-01",
                                  "half_sold": False, "triggers": {}, "catalysts": []}],
                   "closed": [], "parked": []},
        "allocator": {"values": {}, "positions": [], "peak": None, "current": None,
                      "quarters_cash_high": 0, "updated": ""},
        "scorecard": {"cards": {}},
    }
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, stores.get(name, default)))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    import refresh_ui, screens_ui
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
    monkeypatch.setenv("PR6_TEST_ROOT", ROOT)

    # AppTest kör funktionens KÄLLKOD som ett skript — inga closures, så
    # modul/funktion/argument går via miljövariabler.
    def app():
        import os as _o, sys as _s, importlib, json as _j
        _s.path.insert(0, _o.environ["PR6_TEST_ROOT"])
        mod = importlib.import_module(_o.environ["PR6_MOD"])
        getattr(mod, _o.environ["PR6_FN"])(**_j.loads(_o.environ["PR6_KW"]))

    import json
    for module, fn, kw in (("producers", "render_producers_page", {"sheet": "Rick Rule"}),
                           ("producers", "render_producers_page", {"sheet": "Royalty C"}),
                           ("scoring", "render_scoring_page", {}),
                           ("insider", "render_insider_page", {}),
                           ("tiggre", "render_tiggre_page", {}),
                           ("allocator", "render_allocator_page", {})):
        monkeypatch.setenv("PR6_MOD", module)
        monkeypatch.setenv("PR6_FN", fn)
        monkeypatch.setenv("PR6_KW", json.dumps(kw))
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        at = AppTest.from_function(app, default_timeout=60)
        at.run()
        assert not at.exception, (module, kw, at.exception)
        # inget nummerfält visar 0,0 för ett tomt lagrat värde
        blanks = [n for n in at.number_input if n.value is None]
        assert blanks, f"{module}: inga tomma nummerfält — tomt visas som 0"
