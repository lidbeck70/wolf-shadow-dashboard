"""
LIDSKJALV PR 13 — ramen: ett stilblock, varumärkesraden, tyst topp (docs/design_lidskjalv.md §3.5.1, §3.5.2, §7).

  · ETT <style> per körning, i den dolda behållaren wolf_shell, med PWA-taggarna — även när en gammal sida
    anropar inject_css() en gång till
  · skalets regler ligger EFTER CYBERPUNK_CSS i samma block; huvudet döljs på telefonen och är genomskinligt
    med flytande ⋮ på datorn; toppblocket har glipa 0
  · varumärkesraden: triangelögat (_SVG_ICON) och NORDIC, ingen base64; pulsen läser bara sessionen och anropar
    aldrig en laddare; osparat-chipet ur storage.dirty_stores(); låset bara när det är konfigurerat; en ruta
  · sidfoten har årets år och "▲ Till toppen" till raden; Inter Tight är borta; toolbarMode = "viewer"
  · pulsen följer sidans egen kapitalrad (I DAG, Läget), annars den data som laddades senast
  · efter flikraden får elementen tillbaka sin 1rem — lagringsraden skrivs inte ovanpå sig själv
"""
import os
import re
import sys
import tomllib

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ui import shell, theme              # noqa: E402
from ui.css import CYBERPUNK_CSS         # noqa: E402


def _src(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _rules(css: str) -> dict:
    """Selektor → deklarationer (utan kommentarer och @media-huvuden)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out = {}
    for sels, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        decls = {}
        for d in body.split(";"):
            if ":" in d:
                k, v = d.split(":", 1)
                decls[k.strip()] = v.strip().lower()
        for s in sels.split(","):
            out.setdefault(" ".join(s.split()), {}).update(decls)
    return out


def _media(css: str, query: str) -> dict:
    """Reglerna i alla @media-block med exakt den frågan (t.ex. '(max-width: 640px)')."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    blocks = re.findall(r"@media " + re.escape(query) + r"\s*\{((?:[^{}]*\{[^{}]*\})*[^{}]*)\}", css)
    return _rules("\n".join(blocks))


SHELL = shell.shell_css()
GLOBAL = _rules(re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", re.sub(r"/\*.*?\*/", "", SHELL, flags=re.S)))


# ── Ett stilblock per körning ────────────────────────────────────────────────
def _frame():
    import streamlit as st

    from ui import theme
    from ui.css import inject_css
    inject_css()
    inject_css()                         # en gammal sida (tabs/screener) anropar igen: inget nytt block
    bar = theme.render_header()
    st.markdown("sidan")
    theme.render_footer()
    theme.fill_bar(bar)


def _run(state=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_frame, default_timeout=30)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    at.run()
    assert not at.exception
    return at


def test_one_style_block_per_run_inside_the_hidden_shell():
    at = _run()
    styled = [m.value for m in at.markdown if "<style" in m.value]
    assert len(styled) == 1 and styled[0].count("<style") == 1 and styled[0].count("</style>") == 1
    block = styled[0]
    assert block.startswith("<style>") and "apple-mobile-web-app-capable" in block     # PWA-taggarna inbakade
    assert 'rel="apple-touch-icon"' in block and "user-scalable=no" not in block
    shells = [b for b in at.main.children.values() if getattr(b.proto, "id", "").endswith(f"-{shell.SHELL_KEY}")]
    assert len(shells) == 1
    inner = [c.value for c in shells[0].children.values()]
    assert inner == [block]                                                         # stilen ligger i skalet


def test_shell_css_follows_cyberpunk_css_in_the_same_block():
    block = shell.style_block()
    assert block.count("<style") == 1 and block.count("</style>") == 1
    assert block.index("@import") < block.index(".stTabs [role=\"tablist\"]") < block.index("LIDSKJALV · ramen")
    assert block.count("@import") == 1 and block[len("<style>\n"):].startswith("@import")    # importen först
    assert block.index("</style>") < block.index("<meta")


def test_shell_and_page_style_containers_are_hidden():
    assert GLOBAL[f".st-key-{shell.SHELL_KEY}"]["display"] == "none"
    assert GLOBAL[f'[class*="st-key-{shell.PAGE_KEY_PREFIX}"]']["display"] == "none"


def _page_style_script():
    from ui.shell import page_style
    page_style("<style>.x{color:red}</style>", "demo")


def test_page_style_writes_one_style_in_its_own_hidden_container():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_page_style_script).run()
    assert not at.exception
    (blk,) = list(at.main.children.values())
    assert blk.proto.id.endswith("-pcss_demo")
    (md,) = [c.value for c in blk.children.values()]
    assert md.count("<style") == 1 and ".x{color:red}" in md


# ── Tyst topp och ramen ──────────────────────────────────────────────────────
def test_header_is_hidden_on_the_phone_and_floats_on_desktop():
    phone = _media(SHELL, "(max-width: 640px)")
    assert phone['[data-testid="stHeader"]']["display"] == "none"
    assert phone['[data-testid="stMainBlockContainer"]']["padding"] == ".25rem 1rem 5rem"
    desk = _media(SHELL, "(min-width: 641px)")
    head = desk['[data-testid="stHeader"]']
    assert head["background"] == "transparent" and head["height"] == "0" and head["min-height"] == "0"
    assert head["pointer-events"] == "none"
    tool = desk['[data-testid="stToolbar"]']
    assert tool["position"] == "fixed" and tool["right"] == "12px"
    assert desk['[data-testid="stToolbar"] button']["pointer-events"] == "auto"     # ⋮ går att trycka på
    # Streamlit Clouds knappar (Share/Fork/stjärna/GitHub) skulle täcka datumet och pulsen — bara ⋮ ryms
    assert desk['[data-testid="stToolbarActions"]']["display"] == "none"
    assert '[data-testid="stToolbarActions"]' not in GLOBAL                         # bara på datorn
    # Körindikatorn finns (viewer) och ritas i radens färg — den täcker datumet rent under en körning
    status = desk['[data-testid="stStatusWidget"]']
    assert status["background"] == shell.BG.lower()
    assert "display" not in status and "visibility" not in status                    # aldrig dold: strålen läser den
    wide = _media(SHELL, "(min-width: 1024px)")
    assert wide['[data-testid="stMainBlockContainer"]']["max-width"] == "calc(1280px + 64px)"


def test_top_level_block_has_no_gap_and_the_bar_sits_on_the_tabs():
    assert GLOBAL['[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]']["gap"] == "0"
    # Markdown-rutans -1rem skulle annars dra flikraden 16 px in i märkesraden
    assert GLOBAL['[data-testid="stMarkdownContainer"]:has(> .wolf-bar)']["margin-bottom"] == "0"


def test_elements_after_the_tabs_get_their_spacing_back():
    """Glipa 0 gäller bara ovanför flikraden. Varje markdown-ruta har margin-bottom -1rem (Streamlit 1.59), som
    glipan förut tog ut — utan den skrevs lagringsradens hr, "Senast sparad" och "Osparade ändringar" ovanpå
    varandra. Elementen efter flikraden får tillbaka 1rem, så avstånden blir desamma som före PR 13."""
    from ui import nav
    assert nav.TABS_KEY == "main_tabs"
    after = GLOBAL[f'.st-key-{nav.TABS_KEY} ~ [data-testid="stElementContainer"]']
    assert after["margin-top"] == "1rem"
    css = re.sub(r"/\*.*?\*/", "", SHELL, flags=re.S)
    assert css.index("{ gap: 0; }") < css.index(f".st-key-{nav.TABS_KEY} ~")       # samma ordning som i ramen
    src = _src("wolf_panel.py")
    main = src[src.index("def main():"):]
    # Lagringsraden och sidfoten ligger i toppblocket efter flikraden — det är dem regeln når
    assert main.index('key="main_tabs"') < main.index("storage_ui.footer()") < main.index("render_footer()")


def test_config_hides_the_toolbar_and_keeps_theme_and_browser():
    with open(os.path.join(ROOT, ".streamlit", "config.toml"), "rb") as fh:
        cfg = tomllib.load(fh)
    # viewer: ingen Deploy och inga utvecklarval, men ⋮ och körindikatorn (stStatusWidget) finns kvar i DOM:en —
    # minimal ritar ingen verktygsrad alls, och strålen i PR 14 läser körindikatorn
    assert cfg["client"] == {"toolbarMode": "viewer"}
    # temat: mörk bas — hela [theme] (panelens färger, PR 16) låses ordagrant i test_css_readability
    assert cfg["theme"]["base"] == "dark" and cfg["browser"] == {"gatherUsageStats": False}
    from streamlit import config
    assert config.get_option("client.toolbarMode") in ("minimal", "auto", "developer", "viewer")


# ── Varumärkesraden ──────────────────────────────────────────────────────────
def test_bar_has_the_eye_and_wordmark_and_no_base64():
    html = theme.bar_html(today="lör 21 nov")
    assert '<polygon points="24,4 44,44 4,44"' in html and 'width="24" height="24"' in html
    assert "NORDIC" in html and "<b>ARC</b>" in html and 'id="nas-top"' in html
    assert "base64" not in html and "data:image" not in html
    assert "lör 21 nov" in html and "wolf-snack" not in html and "wolf-puls" not in html


def test_bar_sizes_44_on_the_phone_and_52_on_desktop():
    assert GLOBAL[".wolf-bar"]["height"] == "44px"
    assert _media(SHELL, "(min-width: 1024px)")[".wolf-bar"]["height"] == "52px"
    assert GLOBAL[".wolf-bar::before"]["height"] == "2px"                     # auroralinjen överst
    assert _media(SHELL, "(max-width: 400px)")[".wolf-bar:has(.wolf-dirty) .wolf-date"]["display"] == "none"


def test_date_label_is_swedish_and_short():
    from datetime import datetime
    assert theme._today_label(datetime(2026, 11, 21, 9, 0)) == "lör 21 nov"
    assert theme._today_label(datetime(2026, 5, 4)) == "mån 4 maj"


@pytest.fixture
def no_loaders(monkeypatch):
    """Varje laddare som pulsen skulle kunna frestas att anropa smäller."""
    def boom(*_a, **_k):
        raise AssertionError("pulsen får aldrig ladda något")
    import gist_storage
    import storage
    from odin import laget
    from tabs import home_today
    monkeypatch.setattr(laget, "load", boom)
    monkeypatch.setattr(home_today, "load", boom)
    monkeypatch.setattr(gist_storage, "load_blob", boom)
    monkeypatch.setattr(gist_storage, "load_wolf_json", boom)
    monkeypatch.setattr(storage, "load_json", boom)


def test_puls_reads_only_the_session_and_never_a_loader(no_loaders):
    from odin import laget
    assert theme.puls_items({}) == [] and theme._puls_html([]) == ""             # ingen data: ingen puls
    regime = {"regime": "GUL", "generated": "2026-11-20T18:05"}
    risk = {"levels": {"SPY": {"label": "SPY", "level": "LÅG"}, "OMXS30": {"label": "OMXS30", "level": "FÖRHÖJD"}}}
    state = {"home_today": {"alloc": {"peak": 100, "current": 94}, "regime": regime, "risk": risk, "board": None}}
    items = theme.puls_items(state)
    assert items == laget.capital_row({"peak": 100, "current": 94}, regime, risk, None)
    html = theme._puls_html(items)
    assert html.count("<span style=") == 4                                        # fyra ljus, aldrig ihopslagna
    for _t, _w, _s, col in items:
        assert f"background:{col};" in html
    assert 'role="img"' in html and f'aria-label="{laget.row_text(items)}"' in html     # orden var för sig
    assert "Swing GUL" in html and "SPY LÅG · OMXS30 FÖRHÖJD" in html
    # Utan minne och utan sida: I DAGs data (har en färskhetsregel) före Lägets
    state["laget_data"] = {"alloc": None, "regime": {"regime": "RÖD"}, "risk": None, "board": None}
    assert _swing(theme.puls_items(state)) == "GUL"
    assert _swing(theme.puls_items({"laget_data": state["laget_data"]})) == "RÖD"     # bara Lägets: den
    assert theme.puls_items({"home_today": {"alloc": ["trasig"]}}) == []            # trasig data: ingen puls, inget fel


def _swing(items):
    return [w for _t, w, _s, _c in items][1]


def _data(word):
    return {"alloc": None, "regime": {"regime": word}, "risk": None, "board": None}


def test_puls_follows_the_data_loaded_last(no_loaders):
    """Läget laddas en gång och läses sedan aldrig om av sig själv; I DAG läser om efter RELOAD_MIN minuter eller
    en ny dag. Pulsen får inte frysa på Lägets gamla ljus när I DAG har nyare (granskningen av PR 13)."""
    state = {"laget_data": _data("GRÖN")}
    seen = theme.mark_loads(state)
    assert _swing(theme.puls_items(state, seen)) == "GRÖN"
    state["home_today"] = _data("RÖD")                               # I DAG läser om (📡 eller efter 30 min)
    seen = theme.mark_loads(state)
    assert _swing(theme.puls_items(state, seen)) == "RÖD"
    assert theme.mark_loads(state) == seen                           # ingen ny load(): samma ordning, inget skrivs om
    state["laget_data"] = _data("GUL")                               # 📡 på Läget
    seen = theme.mark_loads(state)
    assert _swing(theme.puls_items(state, seen)) == "GUL"
    state["home_today"] = _data("RÖD")                               # I DAG igen
    assert _swing(theme.puls_items(state, theme.mark_loads(state))) == "RÖD"
    # Minnet ligger bara i sessionen, under en egen nyckel som inte är ett lager
    assert set(state) == {"laget_data", "home_today", theme.PULS_SEEN}
    assert not theme.PULS_SEEN.startswith("_storage_")
    del state["home_today"]                                          # borta ur sessionen: glöms
    assert set(theme.mark_loads(state)) == {"laget_data"}
    assert _swing(theme.puls_items(state, theme.mark_loads(state))) == "GUL"


def test_puls_never_contradicts_the_capital_row_on_the_page_shown(no_loaders):
    """På I DAG och Läget är kapitalraden pulsens legend — där läser pulsen sidans egen data, hur färsk den
    andra än är. På alla andra sidor den som laddades senast."""
    from ui import nav
    state = {"home_today": _data("RÖD")}
    theme.mark_loads(state)
    state["laget_data"] = _data("GRÖN")                              # Läget laddat senare
    seen = theme.mark_loads(state)
    assert _swing(theme.puls_items(state, seen)) == "GRÖN"
    assert theme.page_source({nav.TABS_KEY: nav.tab_label("home")}) == "home_today"
    assert theme.page_source({nav.TABS_KEY: nav.tab_label("odin"), nav.STATE_KEY["odin"]: "Läget"}) == "laget_data"
    assert theme.page_source({nav.TABS_KEY: nav.tab_label("thor")}) is None
    assert theme.page_source({nav.TABS_KEY: nav.tab_label("odin"), nav.STATE_KEY["odin"]: "Kartan"}) is None
    assert _swing(theme.puls_items(state, seen, prefer="home_today")) == "RÖD"     # I DAG: dess egen rad
    assert _swing(theme.puls_items(state, seen, prefer="laget_data")) == "GRÖN"    # Läget: dess egen rad
    assert _swing(theme.puls_items({"laget_data": _data("GUL")}, {}, prefer="home_today")) == "GUL"  # saknas: annan


def _reload_frame():
    """Samma ordning som main(): raden, sidan läser om I DAG (som home._today_data), raden fylls om sist."""
    import streamlit as st

    from ui import theme
    bar = theme.render_header()
    if st.session_state.pop("_reload", False):
        st.session_state["home_today"] = {"alloc": None, "regime": {"regime": "RÖD"}, "risk": None, "board": None}
    theme.fill_bar(bar)


def _bar_words(at):
    assert not at.exception
    bars = [m.value for m in at.markdown if 'class="wolf-bar"' in m.value]
    assert len(bars) == 1
    return bars[0]


def test_bar_in_a_run_follows_a_reload_after_laget(no_loaders):
    from streamlit.testing.v1 import AppTest

    from ui import nav
    at = AppTest.from_function(_reload_frame, default_timeout=30)
    at.session_state[nav.TABS_KEY] = nav.tab_label("thor")           # en sida utan egen kapitalrad
    at.session_state["home_today"] = _data("GUL")
    at.run()
    assert "Swing GUL" in _bar_words(at)
    at.session_state["laget_data"] = _data("GRÖN")                   # Läget laddat (eller 📡 där): nyast
    at.run()
    assert "Swing GRÖN" in _bar_words(at)
    at.session_state["_reload"] = True                               # I DAG läser om efter 30 min
    at.run()
    words = _bar_words(at)
    assert "Swing RÖD" in words and "Swing GRÖN" not in words
    at.run()                                                         # ingen ny laddning: ligger kvar
    assert "Swing RÖD" in _bar_words(at)
    at.session_state[nav.TABS_KEY] = nav.tab_label("odin")           # Läget visas: dess egen (äldre) rad
    at.session_state[nav.STATE_KEY["odin"]] = "Läget"
    at.run()
    assert "Swing GRÖN" in _bar_words(at)


def test_bar_in_a_run_never_loads_and_shows_the_puls_from_the_session(no_loaders):
    at = _run({"home_today": {"alloc": None, "regime": {"regime": "GRÖN"}, "risk": None, "board": None}})
    bars = [m.value for m in at.markdown if 'class="wolf-bar"' in m.value]
    assert len(bars) == 1 and "wolf-puls" in bars[0] and "Swing GRÖN" in bars[0]


def test_dirty_chip_lock_and_a_single_snackbar():
    html = theme.bar_html(dirty=["scorecard", "holdings"], lock="🔓", today="lör 21 nov", snack=True)
    assert "● 2 osparade" in html and 'title="Osparade ändringar i: scorecard, holdings"' in html
    assert html.count('class="wolf-snack"') == 1 and "Osparade ändringar · Spara ↑" in html
    assert 'class="wolf-lock"' in html and "🔓" in html
    one = theme.bar_html(dirty=["swing"], snack=True)
    assert "● 1 osparad<" in one and one.count("wolf-snack") == 1
    assert "wolf-snack" not in theme.bar_html(dirty=[], snack=True)                 # bara när något är osparat
    assert "&lt;x&gt;" in theme.bar_html(dirty=["<x>"])                              # lagrets namn escapas


def test_snackbar_is_phone_only_and_lifts_the_toasts():
    assert GLOBAL[".wolf-snack"]["display"] == "none"
    phone = _media(SHELL, "(max-width: 640px)")
    snack = phone[".wolf-snack"]
    assert snack["display"] == "flex" and snack["position"] == "fixed" and snack["height"] == "44px"
    assert phone['body:has(.wolf-snack) [data-testid="stToastContainer"]']["transform"] == "translatey(-56px)"


def test_dirty_store_in_the_session_shows_the_chip_and_one_snackbar():
    at = _run({"swing": {"a": 1}, "_storage_clean_swing": "annat fingeravtryck"})
    bars = [m.value for m in at.markdown if 'class="wolf-bar"' in m.value]
    assert len(bars) == 1 and "● 1 osparad" in bars[0] and bars[0].count("wolf-snack") == 1


def test_lock_glyph_only_when_the_lock_is_configured(monkeypatch):
    from portfolio import private_lock
    monkeypatch.setattr(private_lock, "configured", lambda: False)
    assert theme._lock_glyph() == ""
    monkeypatch.setattr(private_lock, "configured", lambda: True)
    monkeypatch.setattr(private_lock, "unlocked", lambda state=None: False)
    assert theme._lock_glyph() == "🔒"
    monkeypatch.setattr(private_lock, "unlocked", lambda state=None: True)
    assert theme._lock_glyph() == "🔓"


# ── Sidfoten och typsnitten ──────────────────────────────────────────────────
def test_footer_year_is_dynamic_and_links_to_the_top(monkeypatch):
    from datetime import datetime as real

    class _Clock(real):
        @classmethod
        def now(cls, tz=None):
            return real(2031, 1, 2)
    monkeypatch.setattr(theme, "datetime", _Clock)
    html = theme.footer_html()
    assert "Nordic Arc Systems © 2031 · See What the Market Can't." in html
    assert '<a class="wolf-top" href="#nas-top">▲ Till toppen</a>' in html
    assert "2025" not in _src("ui", "theme.py")
    top = GLOBAL["a.wolf-top"]
    assert top["min-width"] == "44px" and top["min-height"] == "44px"


def test_inter_tight_is_gone_and_one_font_import_remains():
    assert "Inter Tight" not in CYBERPUNK_CSS and "Inter+Tight" not in CYBERPUNK_CSS
    assert "Inter Tight" not in _src("ui", "theme.py") and "Inter Tight" not in SHELL
    (imp,) = re.findall(r"@import url\('([^']+)'\)", shell.style_block())
    assert "Space+Grotesk:wght@400;600;700;800;900" in imp and "JetBrains+Mono:wght@400;500;600" in imp


def test_banner_stays_cached_for_the_legacy_banner():
    assert callable(theme._banner_b64)
    assert "_banner_b64" in _src("ui", "css.py")                                 # css.wolf_banner läser den


# ── wolf_panel kopplar ihop ramen ────────────────────────────────────────────
def test_panel_draws_the_bar_and_refills_it_last():
    src = _src("wolf_panel.py")
    main = src[src.index("def main():"):]
    assert "bar = render_header()" in main and 'st.tabs(tab_labels, on_change="rerun", key="main_tabs")' in main
    assert main.index("storage_ui.footer()") < main.index("fill_bar(bar)") < main.index("_sync_url()")
    assert main.index("inject_css()") < main.index("bar = render_header()") < main.index("st.tabs(")
    assert "apple-mobile-web-app-capable" not in src                             # taggarna ligger i skalet
    assert "PWA_METAS" in main[:main.index("return")]                              # och på inloggningen tills PR 17


def test_inject_css_delegates_to_the_shell():
    src = _src("ui", "css.py")
    body = src[src.index("def inject_css():"):src.index("def wolf_banner")]
    assert "render_shell()" in body and "st.markdown(CYBERPUNK_CSS" not in body
