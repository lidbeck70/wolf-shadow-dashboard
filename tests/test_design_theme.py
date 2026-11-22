"""
LIDSKJALV PR 16 — cyan och bläck (docs/design_lidskjalv.md §3.1, §3.2, §3.5.16, §7 "Notes for PR 16", beslut 4 och 5).

  · [theme] i .streamlit/config.toml är panelens egna färger ur ui/tokens: cyan primärfärg (kryssrutor, reglage,
    skjutreglage, radioknappar, flikar, fokusringar och primärknappar slutar vara Streamlit-röda), APP_BG som
    bakgrund, röd/orange/grön/lila som panelens trafikljus och bläcktonerna som temats textfärger
  · primärknappen har MÖRK text på cyan (vit vore 1,5:1); avstängd primär syns
  · undantagen väljs på knapparnas egna nycklar: sparradens osparat-knapp (save_<lager>) är bärnsten, bekräfta
    radering (confirm_delete:s <nyckel>__confirm och Övervakningens watch_del_<id>) är röd — "Confirm Import" är
    ingen radering och förblir cyan
  · ink() ger bläcktonerna; de och MUTED klarar 4,5:1 mot BG_ALT (§8 punkt 9)
  · inget Streamlit-rött (#ff4b4b / 255,75,75) i stilblocket
"""
import colorsys
import os
import re
import sys
import tomllib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ui import shell, tokens  # noqa: E402


def _src(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _rules(css: str) -> dict:
    """Selektor → deklarationer (utan kommentarer, @media/@keyframes och !important)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"@(?:media|keyframes)[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)
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


def _lum(h: str) -> float:
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast(fg: str, bg: str) -> float:
    a, b = sorted((_lum(fg), _lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def _darken(h: str, amount: float) -> str:
    """polished.darken (Streamlits primärknapp vid hovring/fokus): HSL-ljusheten minus amount."""
    r, g, b = (int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(hh, max(0.0, ll - amount), ss)
    return "#" + "".join(f"{round(x * 255):02x}" for x in (r, g, b))


def _theme() -> dict:
    with open(os.path.join(ROOT, ".streamlit", "config.toml"), "rb") as fh:
        return tomllib.load(fh)["theme"]


PARTS = shell.parts_css()
RULES = _rules(PARTS)
PRIMARY = '.stButton > button[kind="primary"]'


# ── Temat är panelens färger ─────────────────────────────────────────────────
def test_theme_colours_are_the_panel_tokens():
    th = {k: str(v).lower() for k, v in _theme().items()}
    assert th["base"] == "dark"
    assert th["primaryColor"] == tokens.ACCENT.lower()
    assert th["backgroundColor"] == tokens.APP_BG.lower()
    assert (th["redColor"], th["orangeColor"], th["greenColor"], th["violetColor"]) == tuple(
        c.lower() for c in (tokens.RED, tokens.AMBER, tokens.GREEN, tokens.PURPLE))
    assert (th["redTextColor"], th["greenTextColor"], th["violetTextColor"]) == tuple(
        c.lower() for c in (tokens.RED_INK, tokens.GREEN_INK, tokens.PURPLE_INK))


def test_config_keeps_browser_and_client_untouched():
    src = _src(".streamlit", "config.toml")
    assert "[browser]\n# Av: Streamlits användningsstatistik skickar sidans adress" in src
    assert "gatherUsageStats = false" in src and 'toolbarMode = "viewer"' in src


# ── Primärknappen ────────────────────────────────────────────────────────────
def test_primary_button_text_is_dark_on_cyan():
    assert RULES[f"{PRIMARY}:not(:disabled)"]["color"] == tokens.APP_BG.lower()
    assert RULES['button[kind="primary"]:not(:disabled)']["color"] == tokens.APP_BG.lower()
    assert RULES['button[kind="primaryFormSubmit"]:not(:disabled)']["color"] == tokens.APP_BG.lower()
    assert _contrast(tokens.APP_BG, tokens.ACCENT) >= 7                  # 13,1:1
    assert _contrast("#ffffff", tokens.ACCENT) < 2                       # varför: Streamlits vita text, 1,5:1
    assert _contrast(tokens.APP_BG, _darken(tokens.ACCENT, .15)) >= 4.5  # Streamlits mörkare hovring/fokus
    shape = RULES[PRIMARY]
    assert shape["min-height"] == "44px" and shape["border-radius"] == "10px"
    assert shape["text-transform"] == "uppercase" and shape["font-size"] == "14px"


def test_disabled_primary_stays_readable():
    # Streamlits avstängda primär är grå text på genomskinligt — mörk text där vore osynlig, därför :not(:disabled)
    dis = RULES[f"{PRIMARY}:disabled"]
    assert dis["color"] == tokens.ACCENT.lower() and dis["background"] == "rgba(0,229,255,.08)"
    assert "color" not in RULES.get(PRIMARY, {})


# ── Undantagen: osparat och bekräfta radering ────────────────────────────────
def test_dirty_save_is_amber_and_keyed_on_the_save_bar():
    dirty = RULES['[class*="st-key-save_"] button[kind="primary"]:not(:disabled)']
    assert dirty["background"] == tokens.AMBER.lower() and dirty["border-color"] == tokens.AMBER.lower()
    assert dirty["color"] == tokens.APP_BG.lower()
    assert _contrast(tokens.APP_BG, tokens.AMBER) >= 7
    assert _contrast(tokens.APP_BG, shell.AMBER_HOVER) >= 4.5
    # samma nyckel som sparraden skriver, och samma selektor som osparat-rutan scrollar till
    src = _src("storage_ui.py")
    assert 'key = key or f"save_{name}"' in src and 'type="primary" if dirty else "secondary"' in src
    assert "⚠️ Du har osparade ändringar." in src and "AMBER" in src
    assert shell.SAVE_DIRTY in shell._NAV_JS


def test_confirm_delete_is_red_and_keyed_on_the_delete_buttons():
    for sel in shell.CONFIRM_DEL:
        red = RULES[f"{sel}:not(:disabled)"]
        assert red["background"] == tokens.RED.lower() and red["color"] == "#ffffff", sel
    assert _contrast("#ffffff", tokens.RED) >= 4.5 and _contrast("#ffffff", shell.RED_HOVER) >= 4.5
    assert 'key=f"{key}__confirm", type="primary"' in _src("ui", "components.py")
    assert 'key=f"watch_del_{it[\'id\']}", type="primary"' in _src("portfolio", "watch_ui.py")
    # "Confirm Import" (import_confirm_btn) är ingen radering — ett bredare [class*="confirm"] hade gjort den röd
    assert '[class*="confirm"]' not in PARTS


def test_exceptions_come_after_the_cyan_primary():
    # lika specifika regler: undantagen vinner på ordningen
    at = PARTS.index(f"{PRIMARY}:not(:disabled)")
    assert at < PARTS.index('[class*="st-key-save_"] button[kind="primary"]:not(:disabled)')
    assert at < PARTS.index('[class*="__confirm"] button[kind="primary"]:not(:disabled)')


def test_widgets_filled_with_cyan_keep_readable_text():
    dark = tokens.APP_BG.lower()
    assert RULES['[data-testid="stNumberInputContainer"] button:hover:enabled']["color"] == dark   # − och +
    assert RULES['.stMultiSelect [data-baseweb="tag"]']["color"] == dark                           # valda etiketter
    assert RULES['.stMultiSelect [data-baseweb="tag"] *']["color"] == dark                         # och deras ×
    assert RULES['[data-baseweb="calendar"] [role="gridcell"][aria-label^="Selected"]']["color"] == dark
    # skjutreglagets värde: CYBERPUNK_CSS:s cyan ruta bort, annars cyan text på cyan
    assert RULES['.stSlider [data-testid="stSliderThumbValue"]']["background"] == "none"


# ── Bläcket ──────────────────────────────────────────────────────────────────
def test_ink_tones_and_muted_are_readable_on_bg_alt():
    for col in (tokens.GREEN_INK, tokens.RED_INK, tokens.PURPLE_INK, tokens.MUTED):
        assert _contrast(col, tokens.BG_ALT) >= 4.5, col
        assert _contrast(col, tokens.APP_BG) >= 4.5, col
    for base, tone in ((tokens.GREEN, tokens.GREEN_INK), (tokens.RED, tokens.RED_INK),
                       (tokens.PURPLE, tokens.PURPLE_INK)):
        assert tokens.ink(base) == tone
        assert _contrast(base, tokens.BG_ALT) < 4.5 <= _contrast(tone, tokens.BG_ALT)   # därför bläcket


# ── Inget Streamlit-rött ─────────────────────────────────────────────────────
def test_no_streamlit_red_in_the_style_block():
    block = re.sub(r"\s", "", shell.style_block().lower())
    assert "ff4b4b" not in block and "255,75,75" not in block
