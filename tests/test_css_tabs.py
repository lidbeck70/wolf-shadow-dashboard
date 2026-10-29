"""
Tydligare flikar — "Ja, gör tydligare flikar".

Streamlit 1.59 ritar st.tabs med React Aria: div[role=tablist] >
div[role=tab][data-testid=stTab][aria-selected] med en markdown-etikett och
en .react-aria-SelectionIndicator-list. Panelens gamla flikregler valde på
BaseWebs [data-baseweb="tab"] / "tab-list" och matchade ingenting, så
flikarna fick Streamlits standard (14px, röd list under den aktiva).
Här kontrolleras att reglerna väljer på den riktiga DOM:en, att aktiv flik
får panelens cyan, att texten är större och läsbar, att raden rullar i sidled
på telefonen och att döda selektorer inte kommit tillbaka.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ui import tokens                    # noqa: E402
from ui.css import CYBERPUNK_CSS         # noqa: E402

APP_BG = "#05070A"                       # .stApp
GRID_BG = "#161A20"                      # rutnätets linje rgba(26,31,37,0.8) över APP_BG — sämsta fallet

LIST = '.stTabs [role="tablist"]'
TAB = f'{LIST} > [role="tab"]'
ACTIVE = f'{TAB}[aria-selected="true"]'
INDICATOR = ".react-aria-SelectionIndicator"


def _src(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _strip_media(css: str) -> str:
    """CSS utan @media-block (bara de globala reglerna)."""
    return re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)


def _rules(css: str) -> dict:
    """Selektor → deklarationer (utan kommentarer, utan !important)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out = {}
    for sels, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        decls = {}
        for d in body.split(";"):
            if ":" in d:
                k, v = d.split(":", 1)
                decls[k.strip()] = v.replace("!important", "").strip().lower()
        for s in sels.split(","):
            out.setdefault(" ".join(s.split()), {}).update(decls)
    return out


def _media(width: int) -> dict:
    blocks = re.findall(rf"@media \(max-width: {width}px\) \{{(.*?)\n\}}", CYBERPUNK_CSS, flags=re.S)
    return _rules("\n".join(blocks))


def _rgb(c: str):
    c = c.strip().lower()
    if c.startswith("#"):
        h = c.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)), 1.0
    p = [float(x) for x in re.findall(r"[\d.]+", c)]
    return tuple(int(x) for x in p[:3]), (p[3] if len(p) > 3 else 1.0)


def _over(top: str, bottom: str) -> str:
    """Halvgenomskinlig färg lagd över en ogenomskinlig — som webbläsaren ritar den."""
    (tr, a), br = _rgb(top), _rgb(bottom)[0]
    return "#" + "".join(f"{round(t * a + b * (1 - a)):02x}" for t, b in zip(tr, br))


def _contrast(fg: str, bg: str) -> float:
    def lum(h):
        c = [x / 255 for x in _rgb(h)[0]]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    a, b = sorted((lum(fg), lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def _specificity(sel: str) -> tuple:
    sel = re.sub(r"::[\w-]+", "", sel)
    ids = sel.count("#")
    classes = len(re.findall(r"\.[\w-]+|\[[^\]]+\]|:(?!:)[\w-]+", sel))
    types = len(re.findall(r"(?:^|[\s>+~])([a-zA-Z][\w-]*)", sel))
    return ids, classes, types


RULES = _rules(_strip_media(CYBERPUNK_CSS))
PHONE = _media(768)
CODE = re.sub(r"/\*.*?\*/", "", CYBERPUNK_CSS, flags=re.S)    # bara regler, inte kommentarer


# ── Döda selektorer borta ────────────────────────────────────────────────────
def test_dead_selectors_are_gone():
    # BaseWeb-flikar och "column" finns inte i Streamlit 1.59 (stTab / stColumn)
    for dead in ('[data-baseweb="tab"]', '[data-baseweb="tab-list"]',
                 '[data-baseweb="tab-highlight"]', '[data-baseweb="tab-border"]',
                 '[data-testid="column"]', ".stSelectbox > div > div:focus"):
        assert dead not in CODE, dead
    # oscoped [aria-selected] under .stTabs träffade även annat än flikar
    assert '.stTabs [aria-selected="true"]' not in RULES


# ── Raden: rullar i sidled, radbryts aldrig ─────────────────────────────────
def test_tab_row_scrolls_horizontally_instead_of_wrapping():
    row = RULES[LIST]
    assert row["overflow-x"] == "auto"
    assert row["flex-wrap"] == "nowrap"
    tab = RULES[TAB]
    assert tab["flex-shrink"] == "0" and tab["white-space"] == "nowrap"
    assert "background-color" in RULES[f"{LIST}::after"]            # baslinjen i panelens ton


# ── Större text och tryckytor ────────────────────────────────────────────────
def test_tabs_have_larger_text_and_44px_tap_targets():
    tab = RULES[TAB]
    assert tab["height"] == "44px"                                   # Apple HIG
    assert int(tab["font-size"].rstrip("px")) > 14                   # Streamlits standard är 14px
    assert int(tab["font-weight"]) >= 600
    phone = PHONE[TAB]
    assert int(phone["font-size"].rstrip("px")) > 14
    assert "height" not in phone                                     # 44px gäller även på telefonen
    for btn in ('[data-testid="stTabsScrollLeft"]', '[data-testid="stTabsScrollRight"]'):
        assert RULES[f".stTabs {btn}"]["height"] == tab["height"]


def test_label_inherits_size_weight_and_font_from_the_tab():
    # markdown-rutan sätter 14px Source Sans och p vikt 400 — båda måste ärva
    for sel in (f'{TAB} [data-testid="stMarkdownContainer"]', f"{TAB} p"):
        r = RULES[sel]
        for prop in ("font-size", "font-weight", "font-family", "color"):
            assert r[prop] == "inherit", (sel, prop)


# ── Aktiv flik i panelens cyan ──────────────────────────────────────────────
def test_active_tab_is_marked_in_the_panel_accent():
    act = RULES[ACTIVE]
    assert act["color"] == tokens.CYAN.lower()
    assert act["background-color"].startswith("rgba(0,229,255,")
    assert int(act["font-weight"]) > int(RULES[TAB]["font-weight"])
    ind = RULES[f"{ACTIVE} {INDICATOR}"]
    assert ind["background-color"] == tokens.CYAN.lower()            # inte Streamlits röda primärfärg
    assert RULES[f"{TAB} {INDICATOR}"]["height"] == "3px"
    hover = RULES[f"{TAB}:hover"]
    assert hover["color"] == tokens.CYAN.lower()


def test_tab_text_contrast_is_at_least_4_5_to_1():
    inactive_fg = RULES[TAB]["color"]
    assert inactive_fg == tokens.TEXT
    active_fg, active_bg = RULES[ACTIVE]["color"], RULES[ACTIVE]["background-color"]
    hover_fg, hover_bg = RULES[f"{TAB}:hover"]["color"], RULES[f"{TAB}:hover"]["background-color"]
    for base in (APP_BG, GRID_BG):
        assert _contrast(inactive_fg, base) >= 4.5, base
        assert _contrast(active_fg, _over(active_bg, base)) >= 4.5, base
        assert _contrast(hover_fg, _over(hover_bg, base)) >= 4.5, base


def test_tab_colours_come_from_the_existing_palette():
    allowed = {tokens.CYAN.lower(), tokens.TEXT, APP_BG.lower(), "transparent"}
    tab_rules = {s: d for s, d in RULES.items() if s.startswith(".stTabs")}
    tab_rules.update({s: d for s, d in PHONE.items() if s.startswith(".stTabs")})
    for sel, decls in tab_rules.items():
        for prop, val in decls.items():
            for col in re.findall(r"#[0-9a-f]{6}|rgba?\([^)]*\)|transparent", val):
                ok = col in allowed or col.startswith("rgba(0,229,255,") or col.startswith("rgba(5,7,10,")
                assert ok, (sel, prop, col)


# ── Sidornas egna flikregler tar inte över raden ────────────────────────────
def test_global_tab_rules_outrank_page_level_tab_rules():
    # Viking-sidan (ovtlyr/ui/layout.py, får inte ändras) injicerar
    # ".stTabs [role=tab]" och ".stTabs [aria-selected=true]" — (0,2,0).
    viking = _src("ovtlyr", "ui", "layout.py")
    page_sels = re.findall(r"^\s*(\.stTabs \[[^{]+?)\s*\{\{", viking, flags=re.M)
    assert page_sels, "Viking-sidans flikregler hittades inte"
    for sel in page_sels:
        assert _specificity(TAB) > _specificity(sel), sel
        assert _specificity(ACTIVE) > _specificity(sel), sel
    assert RULES[TAB]["border"] == "none"                            # sidans grå understrykning tas bort


# ── Resten av panelens CSS orörd ────────────────────────────────────────────
def test_readability_block_and_page_overrides_are_kept():
    css = _src("ui", "css.py")
    assert "/* ── Läsbar text i fält, rullgardiner och utfällbara rutor ───── */" in css
    assert '.stButton > button[kind="secondary"] {' in css
    assert 'h1, h2, h3 {\n    color: #00E5FF;' in css
    assert ".stAlert {" not in css
