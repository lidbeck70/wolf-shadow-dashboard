"""
Läsbara fält — "texten i fallout-rutorna syns inte".

Panelens CSS målar fälten mörka (#1A1F25) men Streamlit valde ljust eller
mörkt tema efter telefonen. I ljust tema fick valt värde, platshållare och
utfällbara rutors rubriker #31333F mot mörk botten (1,2–1,6:1). Två lager:
.streamlit/config.toml tvingar mörkt tema, och ui/css.py sätter själv
textfärgen överallt där den målar en mörk yta.
"""
import os
import re
import sys
import tomllib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ui import tokens                    # noqa: E402
from ui.css import CYBERPUNK_CSS         # noqa: E402

FIELD_BG = "#1A1F25"                     # fältens yta i ui/css.py
LIST_BG = "#0D1117"                      # den utfällda listan
APP_BG = "#05070A"                       # .stApp


def _src(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


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


def _contrast(fg: str, bg: str) -> float:
    def lum(h):
        h = h.lstrip("#")
        c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    a, b = sorted((lum(fg), lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


RULES = _rules(CYBERPUNK_CSS)
FIELDS = (".stSelectbox input", ".stTextInput input", ".stNumberInput input",
          ".stDateInput input", ".stTextArea textarea")


# ── Mörkt tema alltid ────────────────────────────────────────────────────────
def test_streamlit_config_forces_the_dark_base_theme():
    with open(os.path.join(ROOT, ".streamlit", "config.toml"), "rb") as fh:
        cfg = tomllib.load(fh)
    assert cfg == {"theme": {"base": "dark"}}          # bara base: knappar/piller som förut
    from streamlit import config
    for key in cfg["theme"]:
        config.get_option(f"theme.{key}")              # Streamlit känner nyckeln
    ignored = [ln.strip() for ln in _src(".gitignore").splitlines()]
    assert ".streamlit/" not in ignored and ".streamlit" not in ignored
    assert ".streamlit/secrets.toml" in ignored        # hemligheterna förblir utanför git


# ── Fältens värde, platshållare och låsta fält ───────────────────────────────
def test_field_values_are_cyan_including_ios_text_fill():
    for sel in FIELDS + (".stMultiSelect input",):
        r = RULES[sel]
        assert r["color"] == tokens.CYAN.lower(), sel
        assert r["-webkit-text-fill-color"] == tokens.CYAN.lower(), sel
        assert r["opacity"] == "1", sel
    assert _contrast(tokens.CYAN, FIELD_BG) >= 7


def test_placeholders_and_disabled_fields_stay_readable():
    for sel in FIELDS:
        ph = RULES[f"{sel}::placeholder"]
        assert ph["color"] == ph["-webkit-text-fill-color"] == "#a8a397", sel
        dis = RULES[f"{sel}:disabled"]
        assert dis["color"] == dis["-webkit-text-fill-color"] == tokens.DIM.lower(), sel
    assert _contrast("#a8a397", FIELD_BG) > _contrast(tokens.DIM, FIELD_BG) >= 4.5


def test_selectbox_focus_uses_react_aria_not_dead_focus_rule():
    assert ".stSelectbox > div > div:focus" not in CYBERPUNK_CSS     # matchar inget i 1.59
    assert RULES['.stSelectbox [role="group"][data-focus-within]']["border-color"] == tokens.CYAN.lower()
    assert RULES['.stSelectbox [role="group"] button']["color"] == tokens.CYAN.lower()


# ── Den utfällda listan ──────────────────────────────────────────────────────
def test_open_dropdown_list_is_dark_with_light_options():
    dd = '[data-testid="stSelectboxVirtualDropdown"]'
    assert RULES[dd]["background-color"] == LIST_BG.lower()
    assert RULES[f'{dd} [role="option"]']["color"] == tokens.TEXT
    assert RULES[f'{dd} [role="option"][aria-selected="true"]']["color"] == tokens.CYAN.lower()
    assert f'{dd} [role="option"][data-focused] [data-item-hl]' in RULES
    assert _contrast(tokens.TEXT, LIST_BG) >= 7


# ── Utfällbara rutor, etiketter, radio ───────────────────────────────────────
def test_expander_header_has_own_background_and_text_colour():
    sm = '[data-testid="stExpander"] summary'
    assert RULES[sm]["background-color"] == "transparent"            # ingen ljus list i ljust tema
    assert "background-color" in RULES['[data-testid="stExpander"] details[open] > summary']
    assert RULES[sm]["color"] == RULES[f"{sm} p"]["color"] == tokens.TEXT
    assert RULES[f"{sm} p"]["font-weight"] == "600"
    assert RULES[f'{sm} [data-testid="stIconMaterial"]']["color"] == tokens.TEXT
    assert _contrast(tokens.TEXT, APP_BG) >= 7


def test_labels_and_options_get_text_colour_but_pages_keep_their_own():
    assert RULES['[data-testid="stWidgetLabel"]']["color"] == tokens.TEXT
    assert RULES['.stRadio [role="radiogroup"] label p']["color"] == tokens.TEXT
    assert RULES[".stCheckbox label p"]["color"] == tokens.TEXT
    # bara på label-elementet (0,1,0): heatmaps/cagrs ".stSelectbox label" (0,1,1)
    # vinner, och p ärver den — en regel på "stWidgetLabel p" skulle ta över
    assert '[data-testid="stWidgetLabel"] p' not in RULES


def test_phone_fields_use_16px():
    blocks = re.findall(r"@media \(max-width: 768px\) \{(.*?)\n\}", CYBERPUNK_CSS, flags=re.S)
    phone = _rules("\n".join(blocks))
    for sel in FIELDS:
        assert phone[sel]["font-size"] == "16px", sel


def test_alerts_keep_their_own_colours():
    css = _src("ui", "css.py")
    assert ".stAlert {" not in css                     # fel/varning/OK går att skilja åt
    assert 'h1, h2, h3 {\n    color: #00E5FF;' in css
