"""
PR 8 av panelgenomgången — djuplänkar.

?p=regime/marknad/arc-regime/wolf-regime öppnar fliken direkt, adressfältet
följer fliken så URL:en alltid går att kopiera, Home-korten öppnar sin flik
och FLIKGUIDEN visar länken per rad. Ingen st.navigation — den skulle ha
gjort om hela fliklayouten; det här ligger ovanpå st.tabs + radioknapparna
som panelen redan ritar ur ui/nav.py.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ui import nav   # noqa: E402


def _src(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def _all_segments() -> list:
    """Alla fullständiga sökvägar som segment [toppnyckel, etikett, ...]."""
    out = []

    def _walk(path, segs):
        opts = nav.SUBS.get(path)
        if not opts:
            out.append(segs)
            return
        for o in opts:
            _walk(f"{path}/{o}", segs + [o])

    for key, _l in nav.TOP:
        _walk(key, [key])
    return out


# ── slug ↔ segment ───────────────────────────────────────────────────────────
def test_slugs_are_ascii_and_round_trip_for_every_path():
    for segs in _all_segments():
        s = nav.slug(segs)
        assert re.fullmatch(r"[a-z0-9/-]+", s), s
        assert nav.resolve(s) == segs, (s, segs)
    assert nav.slug(["regime", "Marknad", "Arc Regime", "Wolf Regime"]) == "regime/marknad/arc-regime/wolf-regime"
    assert nav.slugify("🧭 Durrett & Confidence") == "durrett-confidence"
    assert nav.slugify("Råvaror") == "ravaror" and nav.slugify("Odin's Blindspot") == "odins-blindspot"


def test_resolve_accepts_the_swedish_top_label_and_truncates_unknown_tails():
    assert nav.resolve("granskning/tiggre") == nav.resolve("review/tiggre") == ["review", "Tiggre"]
    assert nav.resolve("regime/marknad/finns-inte") == ["regime", "Marknad"]
    assert nav.resolve("okand-flik/x") is None and nav.resolve("") is None
    assert nav.resolve("/portfolio/") == ["portfolio"]


def test_seed_and_current_round_trip_through_widget_keys():
    state = {}
    nav.seed(state, ["regime", "Marknad", "Arc Regime", "Viking Regime"])
    assert state[nav.TABS_KEY] == nav.tab_label("regime")
    assert state["sub_regime_group"] == "Marknad" and state["sub_regime"] == "Arc Regime"
    assert state["sub_regime_arc"] == "Viking Regime"
    assert nav.current(state) == ["regime", "Marknad", "Arc Regime", "Viking Regime"]
    # tom session → första fliken, första alternativen
    assert nav.current({}) == ["home"]
    assert nav.current({nav.TABS_KEY: nav.tab_label("screening")}) == ["screening", "Arc Screener", "Wolf"]


def test_every_branch_in_the_tree_has_a_widget_key_and_the_panel_uses_them():
    assert set(nav.STATE_KEY) == set(nav.SUBS)
    src = _src("wolf_panel.py")
    assert 'key="sub_' not in src                                   # inga handskrivna nycklar kvar
    for path in re.findall(r'_sub\("([^"]+)"\)', src):
        assert path in nav.STATE_KEY, path
    assert "_apply_deep_link()" in src and "_sync_url()" in src
    rules = _src("ovtlyr/ui/rules_page.py")
    assert '_nav.STATE_KEY["rules/Regler & Guider"]' in rules
    assert 'key="rules_sub"' not in rules


def test_flikguide_rows_all_resolve_to_a_link():
    from ovtlyr.ui.rules_page import _PANEL_GUIDE
    for tab, _r, _u in _PANEL_GUIDE:
        assert nav.link_for(tab).startswith("?p="), tab
    assert nav.link_for("REGIME → Marknad → Swing Regime") == "?p=regime/marknad/swing-regime"
    assert nav.link_for("REGIME → Finns inte") == ""
    assert "LÄNK" in _src("ovtlyr/ui/rules_page.py")


def test_home_cards_open_their_tab_via_nav_goto():
    src = _src("tabs/home.py")
    assert 'st.session_state["nav_goto"] = target' in src and "Öppna →" in src
    # varje korts mål finns i trädet
    for key, _t, cards in nav.HOME_ZONES:
        for name, _d in cards:
            first = name.split(" · ")[0]
            segs = [key, first] if first in nav.options(key) else [key]
            assert nav.resolve(nav.slug(segs)) == segs


# ── Panelen: ?p= öppnar fliken, URL:en följer med, och länken drar inte tillbaka
def _p(at) -> str:
    """AppTest ger tillbaka skrivna query_params som lista."""
    v = at.query_params["p"]
    return v[0] if isinstance(v, list) else v


def _script():
    import streamlit as st
    from ui import nav
    import wolf_panel as wp
    wp._apply_deep_link()
    tabs = st.tabs([nav.tab_label(k) for k, _l in nav.TOP], on_change="rerun", key=nav.TABS_KEY)
    with tabs[3]:
        g = st.radio("g", nav.options("regime"), key=nav.STATE_KEY["regime"], horizontal=True)
        if g == "Marknad":
            s = st.radio("s", nav.options("regime/Marknad"), key=nav.STATE_KEY["regime/Marknad"])
            if s == "Arc Regime":
                st.radio("i", nav.options("regime/Marknad/Arc Regime"),
                         key=nav.STATE_KEY["regime/Marknad/Arc Regime"])
    with tabs[5]:
        st.radio("p", nav.options("portfolio"), key=nav.STATE_KEY["portfolio"])
    wp._sync_url()


def test_query_param_opens_the_tab_and_the_url_follows_the_user():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_script)
    at.query_params["p"] = "regime/marknad/arc-regime/viking-regime"
    at.run()
    assert not at.exception
    assert at.session_state[nav.TABS_KEY] == nav.tab_label("regime")
    assert at.radio(key="sub_regime_arc").value == "Viking Regime"
    assert _p(at) == "regime/marknad/arc-regime/viking-regime"

    # användaren klickar vidare — länken ska INTE dra tillbaka
    at.radio(key="sub_regime_arc").set_value("Wolf Regime").run()
    assert at.radio(key="sub_regime_arc").value == "Wolf Regime"
    assert _p(at) == "regime/marknad/arc-regime/wolf-regime"

    # Home-kortet: nav_goto byter toppflik i samma session
    at.session_state["nav_goto"] = "portfolio/holdings"
    at.run()
    assert at.session_state[nav.TABS_KEY] == nav.tab_label("portfolio")
    assert at.radio(key="sub_portfolio").value == "Holdings"
    assert _p(at) == "portfolio/holdings"


def test_unknown_link_leaves_the_panel_on_home():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_script)
    at.query_params["p"] = "finns-inte"
    at.run()
    assert not at.exception
    assert _p(at) == "home"
