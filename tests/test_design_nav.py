"""
LIDSKJALV PR 14 — navigeringen: sex flikar, klistrad rad, chip-rader, rundgång (docs/design_lidskjalv.md §3.5.3–§3.5.6,
§3.5.22, §7).

  · telefonens flikceller genereras ur nav.TOP, i samma ordning, med Safari-reserven (vanlig content) FÖRE alt-texten
  · flikraden klistrar på alla bredder; datorns rad är dagens (inga nya regler på själva fliken utanför telefonen)
  · strålen sveper bara medan Streamlit räknar och står still med prefers-reduced-motion
  · varje nav.STATE_KEY-värde har sin räls: L2 (chip) utan '/' i sökvägen, L3 (understruken) med; radioringen döljs
    med prototypens exakta nodselektor
  · Thors pills-rader rullar i en rad; sidornas inre flikar är 44 px, radbryts på telefonen och träffar aldrig
    toppflikarna
  · nav-skriptet ligger i skalet, med rutten inbakad, och dess kropp innehåller aldrig '<'
  · inget st.markdown("---") direkt efter _sub(; brödsmulan vid djup 3 ur nav.current
  · värmekartans CSS går genom page_style och når bara sidans egna element
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ui import nav, shell, tokens         # noqa: E402

CYAN = tokens.CYAN.lower()


def _src(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _nocomments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _rules(css: str) -> dict:
    """Selektor → deklarationer (utan kommentarer, utan !important; @-huvuden ignoreras)."""
    out = {}
    for sels, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _nocomments(css)):
        decls = {}
        for d in body.split(";"):
            if ":" in d:
                k, v = d.split(":", 1)
                decls[k.strip()] = v.replace("!important", "").strip().lower()
        for s in sels.split(","):
            out.setdefault(" ".join(s.split()), {}).update(decls)
    return out


_MEDIA = r"@media {q}\s*\{{((?:[^{{}}]*\{{[^{{}}]*\}})*[^{{}}]*)\}}"


def _media_blocks(css: str, query: str) -> str:
    return "\n".join(re.findall(_MEDIA.format(q=re.escape(query)), _nocomments(css)))


def _media(css: str, query: str) -> dict:
    return _rules(_media_blocks(css, query))


def _global(css: str) -> dict:
    """Bara reglerna utanför @media och @keyframes."""
    css = _nocomments(css)
    css = re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)
    css = re.sub(r"@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)
    return _rules(css)


NAV = shell.nav_css()
GLOBAL = _global(NAV)
PHONE = _media(NAV, "(max-width: 640px)")
WIDE = _media(NAV, "(min-width: 1024px)")


# ── L1: flikraden ────────────────────────────────────────────────────────────
_LABEL = re.compile(r':nth-child\((\d+)\) p::(before|after) \{ ([^}]*)\}')


def test_tab_labels_are_generated_from_nav_top_in_order():
    phone = _media_blocks(NAV, "(max-width: 640px)")
    found = {}
    for i, part, body in _LABEL.findall(phone):
        found.setdefault(int(i), {})[part] = body
    assert sorted(found) == list(range(1, len(nav.TOP) + 1))
    for i, (_key, label) in enumerate(nav.TOP, start=1):
        emo, word = label.split(" ", 1)
        assert f'content: "{emo}";' in found[i]["before"], (i, label)
        assert f'content: "{word}";' in found[i]["after"], (i, label)
    # samma ordning som i strängen — fliken på plats i får etiketten på plats i
    words = [w for _i, part, body in _LABEL.findall(phone) if part == "after"
             for w in re.findall(r'content: "([^"]+)";', body)]
    assert words == [label.split(" ", 1)[1] for _k, label in nav.TOP]
    # bara toppflikarna: barnkombinatorerna gör att sidornas inre flikar aldrig får etiketterna
    assert shell.TAB == '.st-key-main_tabs > div > [role="tablist"] > [role="tab"]'
    for sel in PHONE:
        if "nth-child" in sel:
            assert sel.startswith(shell.TAB + ":nth-child("), sel


def test_each_tab_has_the_plain_content_before_the_alt_one():
    phone = _media_blocks(NAV, "(max-width: 640px)")
    bodies = _LABEL.findall(phone)
    assert len(bodies) == 2 * len(nav.TOP)
    for i, part, body in bodies:
        plain = re.search(r'content: ("[^"]*");', body)
        assert plain, (i, part, body)
        alt = f'content: {plain.group(1)} / "";'
        assert alt in body, (i, part, body)
        assert body.index(plain.group(0)) < body.index(alt), (i, part)     # Safari före 17.4 läser den första


def test_phone_tabs_are_six_cells_with_emoji_over_word():
    tab = PHONE[shell.TAB]
    assert tab["height"] == "56px" and tab["flex"] == "1 1 auto" and tab["min-width"] == "52px"
    assert PHONE[shell.TABLIST]["overflow-x"] == "visible" and PHONE[shell.TABLIST]["gap"] == "0"
    p = PHONE[f"{shell.TAB} p"]
    assert p["font-size"] == "0" and p["text-align"] == "center"           # etiketten finns kvar för skärmläsare
    before, after = PHONE[f"{shell.TAB} p::before"], PHONE[f"{shell.TAB} p::after"]
    assert before["display"] == after["display"] == "block"
    assert before["font-size"] == "20px" and before["filter"] == "grayscale(1)"        # grå emoji när inaktiv
    assert after["font"].startswith("600 12px/14px") and after["white-space"] == "nowrap"  # 12 px-golvet
    sel = f'{shell.TAB}[aria-selected="true"]'
    assert PHONE[f"{sel} p::before"]["filter"] == "none"
    assert PHONE[f"{sel} p::after"]["color"] == CYAN
    assert _media(NAV, "(max-width: 359px)")[shell.TAB]["min-width"] == "0"   # 320 px: alla sex ryms ändå


def test_tablist_sticks_at_every_width_and_desktop_keeps_todays_row():
    sticky = GLOBAL[shell.TABLIST]
    assert sticky["position"] == "sticky" and sticky["top"] == "0" and sticky["z-index"] == "50"
    assert sticky["background"] == shell.APP_BG.lower()
    assert sticky["box-shadow"].startswith("inset 0 -2px 0 rgba(0,229,255,0.2)")  # dagens baslinje ligger kvar
    # Datorn: inga regler på själva fliken utanför telefonens @media — CYBERPUNK_CSS:s rad gäller oförändrad
    assert not [s for s in GLOBAL if s.startswith(shell.TAB)]
    for query in ("(min-width: 641px) and (max-width: 1023px)", "(min-width: 1024px)"):
        assert not [s for s in _media(NAV, query) if shell.TABLIST in s]
    # call-raden i wolf_panel är orörd
    assert 'st.tabs(tab_labels, on_change="rerun", key="main_tabs")' in _src("wolf_panel.py")


def test_scan_beam_runs_only_while_streamlit_runs_and_stops_for_reduced_motion():
    beam = GLOBAL[f"{shell.TABLIST}::after"]
    assert beam["height"] == "2px" and beam["opacity"] == "0" and beam["position"] == "absolute"
    assert "linear-gradient(90deg, transparent, #00e5ff, #b400ff, transparent)" in beam["background"]
    # :has(a, b) har ett kommatecken inuti — läs regeln direkt i stället för via _rules
    running = re.findall(r"body:has\(([^)]*)\) " + re.escape(shell.TABLIST) + r"::after \{([^}]*)\}",
                         _nocomments(NAV))
    assert len(running) == 1
    icons, body = running[0]
    assert icons == ('[data-testid="stStatusWidgetRunningIcon"], '
                     '[data-testid="stStatusWidgetRunningManIcon"]')
    on = _rules(f"x {{{body}}}")["x"]
    assert on["opacity"] == "1" and on["animation"] == "wolf-scan 1.2s linear infinite"
    assert "@keyframes wolf-scan" in NAV
    still = _media(NAV, "(prefers-reduced-motion: reduce)")[f"{shell.TABLIST}::after"]
    assert still["animation"] == "none"                                     # en stilla linje i stället


# ── L2/L3: rälsarna ur nav.STATE_KEY ─────────────────────────────────────────
def test_every_state_key_value_has_a_rail_split_by_depth():
    l2, l3 = shell.rail_keys()
    assert sorted(l2 + l3) == sorted(nav.STATE_KEY.values()) and not set(l2) & set(l3)
    for path, key in nav.STATE_KEY.items():
        opt = GLOBAL.get(f".st-key-{key} {shell.OPTION}")
        assert opt, key
        assert opt["min-height"] == "44px", key                             # en tumme
        if path.count("/") == 0:
            assert key in l2 and opt["border-radius"] == "999px", key       # L2: chip
            assert GLOBAL[f".st-key-{key} {shell.OPTION}[data-selected]"]["border-color"] == CYAN
            assert GLOBAL[f".st-key-{key} {shell.OPTION}[data-selected] p"]["color"] == CYAN
            grp = GLOBAL[f".st-key-{key} {shell.GROUP}"]
            assert grp["flex-wrap"] == "nowrap" and grp["overflow-x"] == "auto" and "mask-image" in grp
        else:
            assert key in l3 and opt["background"] == "transparent", key    # L3: understruken textrad
            assert GLOBAL[f".st-key-{key} {shell.OPTION}[data-selected]"]["box-shadow"] == f"inset 0 -2px 0 {CYAN}"
            assert GLOBAL[f".st-key-{key} {shell.OPTION}[data-selected] p"]["color"] == tokens.TEXT
        assert WIDE[f".st-key-{key} {shell.GROUP}"]["flex-wrap"] == "wrap", key   # datorn radbryter
        assert f".st-key-{key} {shell.GROUP}" in shell.rail_selector()           # skriptet centrerar den
    assert nav.STATE_KEY["odin"] in l2 and nav.STATE_KEY["more/Motorsidor"] in l3
    assert nav.STATE_KEY["more/Regler & Guider"] in l3                           # rules_sub täcks av generatorn


def test_rail_groups_contain_their_hidden_radio_inputs():
    """Varje räls är det omslutande blocket för radioknapparnas dolda (position: absolute) span — annars klipps de
    inte av rälsens overflow, stMain blir bredare än skärmen och hela sidan sveps i sidled på telefonen."""
    for key in nav.STATE_KEY.values():
        grp = GLOBAL[f".st-key-{key} {shell.GROUP}"]
        assert grp["overflow-x"] == "auto" and grp["position"] == "relative", key
    # villkorslöst: ingen bredd (641–1023 px rullar också) sätter tillbaka den
    for query in ("(max-width: 640px)", "(min-width: 641px) and (max-width: 1023px)", "(min-width: 1024px)"):
        for sel, decls in _media(NAV, query).items():
            if shell.GROUP in sel:
                assert "position" not in decls, (query, sel)


def test_radio_circle_is_hidden_with_the_prototypes_exact_node_selector():
    assert shell.RADIO_DOT == '[data-testid="stRadioOption"] > div > div > div:first-child'
    for key in nav.STATE_KEY.values():
        assert GLOBAL[f".st-key-{key} {shell.RADIO_DOT}"]["display"] == "none", key
    assert shell.OPTION == '[role="radiogroup"] > label[data-testid="stRadioOption"]'


def test_thor_pills_rows_scroll_in_one_line():
    for key in ("thor_filters", "thor_markets"):
        row = GLOBAL[f'.st-key-{key} [role="toolbar"]']
        assert row["flex-wrap"] == "nowrap" and row["overflow-x"] == "auto" and "mask-image" in row
        # hela bredden (Streamlit sätter max-width: fit-content) — annars tonas sista pillret ut fast raden ryms
        assert row["width"] == row["max-width"] == "100%"
        assert GLOBAL[f'.st-key-{key} [role="toolbar"] > button']["flex"] == "0 0 auto"


# ── L4: sidornas inre flikar ─────────────────────────────────────────────────
def test_inner_tabs_are_44px_wrap_on_the_phone_and_never_hit_the_main_tabs():
    assert shell.INNER_LIST.startswith(".stTabs:not(.st-key-main_tabs) > div > ")
    tab = GLOBAL[shell.INNER_TAB]
    assert tab["height"] == "44px" and tab["font-size"] == "14px" and tab["background-color"] == "transparent"
    sel = GLOBAL[f'{shell.INNER_TAB}[aria-selected="true"]']
    assert sel["box-shadow"] == f"inset 0 -2px 0 {CYAN}" and sel["color"] == tokens.TEXT
    assert GLOBAL[f"{shell.INNER_TAB} .react-aria-SelectionIndicator"]["display"] == "none"
    assert PHONE[shell.INNER_LIST]["flex-wrap"] == "wrap"
    # färgerna: cyan, text, MUTED, appens bakgrund, genomskinligt och cyan-toner
    allowed = {CYAN, tokens.TEXT, shell.MUTED.lower(), shell.APP_BG.lower(), "transparent"}
    for rules in (GLOBAL, PHONE):
        for s, decls in rules.items():
            if not s.startswith(shell.INNER_LIST):
                continue
            for val in decls.values():
                for col in re.findall(r"#[0-9a-f]{6}|rgba?\([^)]*\)|transparent", val):
                    assert col in allowed or col.startswith("rgba(0,229,255,"), (s, col)


# ── Nav-skriptet ─────────────────────────────────────────────────────────────
def test_script_body_has_no_less_than_sign_even_with_a_hostile_ticker():
    html = shell.nav_js('thor|<script>alert("x")</script>')
    assert html.startswith("<script>") and html.endswith("</script>")
    body = html[len("<script>"):-len("</script>")]
    assert "<" not in body
    assert "\\u003cscript" in body                                       # tickern kodad, inte borttagen
    assert "<" not in shell._NAV_JS
    for key in nav.STATE_KEY.values():
        assert f".st-key-{key} [role=\\\"radiogroup\\\"]" in body           # rälsarnas selektorer inbakade
    for part in ("scrollTo({top: 0", "data-jump", ".wolf-snack", '[class*=\\"st-key-save_\\"]', "a.wolf-top",
                 "__wolfNav", "MutationObserver"):
        assert part.replace('\\"', '"') in body or part in body, part


def test_route_key_is_the_slug_plus_thors_ticker():
    assert shell.route_key({}) == "home|"
    odin = {nav.TABS_KEY: nav.tab_label("odin"), "sub_odin": "Råvaror", "sub_odin_ravaror": "🥇 Guld"}
    assert shell.route_key(odin) == "odin/ravaror/guld|"
    thor = {nav.TABS_KEY: nav.tab_label("thor")}
    assert shell.route_key(thor, {"t": "VOLV-B.ST"}) == "thor|VOLV-B.ST"        # länken innan Thor läst den
    assert shell.route_key({**thor, "thor_ticker": "NEM"}, {"t": "VOLV-B.ST"}) == "thor|NEM"
    assert shell.route_key({**thor, "thor_ticker": ""}, {"t": "VOLV-B.ST"}) == "thor|"     # stängd analys
    assert shell.route_key({**odin, "thor_ticker": "NEM"}) == "odin/ravaror/guld|"         # bara Thor


def _shell_app():
    import streamlit as st

    import wolf_panel as wp
    from ui.css import inject_css
    from ui.shell import nav_script
    shell = inject_css()
    inject_css()                                       # en gammal sida anropar igen: inget nytt skal
    wp._apply_deep_link()
    nav_script(shell)
    st.markdown("sidan")


def test_nav_script_sits_in_the_hidden_shell_with_the_deep_linked_route():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_shell_app, default_timeout=60)
    at.query_params["p"] = "odin/ravaror/guld"
    at.run()
    assert not at.exception
    shells = [b for b in at.main.children.values() if getattr(b.proto, "id", "").endswith(f"-{shell.SHELL_KEY}")]
    assert len(shells) == 1
    kids = list(shells[0].children.values())
    assert len(kids) == 2 and "<style>" in kids[0].value                       # stilen, sedan skriptet
    html = kids[1].proto
    assert html.unsafe_allow_javascript and 'route = "odin/ravaror/guld|"' in html.body


def test_main_writes_the_script_after_the_deep_link_and_before_the_tabs():
    src = _src("wolf_panel.py")
    main = src[src.index("def main():"):]
    assert main.index("shell = inject_css()") < main.index("_apply_deep_link()") < main.index("nav_script(shell)")
    assert main.index("nav_script(shell)") < main.index('st.tabs(tab_labels, on_change="rerun", key="main_tabs")')
    body = _src("ui", "shell.py")
    assert "st.html(nav_js(" in body and "unsafe_allow_javascript=True" in body


# ── wolf_panel: inga avdelare efter rälsen, brödsmulan vid djup 3 ────────────
def test_no_divider_follows_a_sub_or_inner_call():
    lines = [ln.strip() for ln in _src("wolf_panel.py").splitlines()]
    calls = [i for i, ln in enumerate(lines) if re.search(r"\b_(sub|inner)\(", ln) and not ln.startswith("def ")]
    assert len(calls) >= 8
    for i in calls:
        nxt = next(ln for ln in lines[i + 1:] if ln)
        assert "---" not in nxt, lines[i]
    # och ingen st.markdown("---") inne i _sub själv
    src = _src("wolf_panel.py")
    sub = src[src.index("def _sub("):src.index("def _crumb(")]
    assert "---" not in sub


def _crumb_app():
    import streamlit as st

    import wolf_panel as wp
    st.session_state.setdefault("_crumb_path", "odin/Råvaror")
    wp._sub("odin")
    wp._sub(st.session_state["_crumb_path"])


def test_crumb_sits_above_the_l3_rail_and_omits_the_leaf():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_crumb_app, default_timeout=60)
    at.session_state[nav.TABS_KEY] = nav.tab_label("odin")
    at.session_state["sub_odin"] = "Råvaror"
    at.session_state["sub_odin_ravaror"] = "🥇 Guld"
    at.run()
    assert not at.exception
    crumbs = [m.value for m in at.markdown if "wolf-crumb" in m.value]
    assert crumbs == ['<div class="wolf-crumb">ODIN › Råvaror</div>']          # lövet visar raden själv
    kinds = [type(e).__name__ for e in at.main.children.values()]
    assert kinds == ["Radio", "Markdown", "Radio"]                             # L2, brödsmulan, L3
    at.session_state["_crumb_path"] = "odin/Makro"
    at.session_state["sub_odin"] = "Makro"
    at.run()
    assert [m.value for m in at.markdown if "wolf-crumb" in m.value] == ['<div class="wolf-crumb">ODIN › Makro</div>']


def test_crumb_is_drawn_for_regler_and_guider_and_never_at_depth_two():
    src = _src("wolf_panel.py")
    more = src[src.index("def _tab_more("):src.index("def main():")]
    assert more.index('_crumb("more/Regler & Guider")') < more.index("render_rules_page")
    crumb = src[src.index("def _crumb("):src.index("def _apply_deep_link(")]
    assert "nav.current(st.session_state)" in crumb and "html.escape(" in crumb
    assert GLOBAL[".wolf-crumb"]["height"] == "20px" and "12px/20px" in GLOBAL[".wolf-crumb"]["font"]


# ── Värmekartan genom page_style ─────────────────────────────────────────────
def _heatmap_css_app():
    from heatmap import heatmap_streamlit as hm
    hm._inject_css()


def test_heatmap_css_goes_through_page_style_and_only_reaches_the_page():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_heatmap_css_app, default_timeout=60).run()
    assert not at.exception
    (blk,) = list(at.main.children.values())
    assert blk.proto.id.endswith("-pcss_heatmap")
    (md,) = [c.value for c in blk.children.values()]
    assert md.count("<style") == 1
    scope = shell.page_scope("heatmap")
    css = _nocomments(md.replace("<style>", "").replace("</style>", ""))
    sels = [" ".join(s.split()) for group in re.findall(r"([^{}]+)\{", css) for s in group.split(",")]
    # :is(...) innehåller själv ett kommatecken — sätt ihop selektorerna igen innan de kontrolleras
    joined = ",".join(sels).replace(scope.replace(", ", ","), "SCOPE").replace(scope, "SCOPE")
    for sel in joined.split(","):
        assert sel.strip().startswith("SCOPE "), sel
    assert ".stRadio label" in md and "st.markdown(\n        f\"\"\"\n        <style>" not in _src("heatmap", "heatmap_streamlit.py")


def test_page_style_wrappers_take_no_space():
    hidden = GLOBAL['[data-testid="stLayoutWrapper"]:has(> [class*="st-key-pcss_"])']
    assert hidden["display"] == "none"
    assert GLOBAL[f'[data-testid="stLayoutWrapper"]:has(> .st-key-{shell.SHELL_KEY})']["display"] == "none"
    assert shell.page_scope("x") == ':is([data-testid="stLayoutWrapper"]:has(> .st-key-pcss_x), .st-key-pcss_x) ~ *'
