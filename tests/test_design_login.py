"""
LIDSKJALV PR 17 — inloggningen och låset (docs/design_lidskjalv.md §4.1, §4.9, §7, beslut 14).

  · skalet ritas före inloggningen: ETT skal-<style> på inloggningssidan (bakgrund, typsnitt, PWA-taggarna), men
    varken märkesrad, nav-skript eller flikar; bryggans skript står kvar
  · inloggningen är ett formulär: Enter/"Gå" skickar (formulärets knapp), fel kod ger "Fel kod — försök igen.",
    rätt kod släpper in och kakan skrivs — samma lösenord, kaka och brygga som förut
  · fältet är 16 px (ingen iOS-zoom) och 52 px; auth:s egen <style> har bara .arc-login-*, ingen @import och ingen 900
  · låset är ett formulär i ett kort: samma nycklar, etiketter, meddelanden och spärr; Enter låser upp
"""
import os
import re
import sys

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth                                   # noqa: E402
from portfolio import private_lock as pl      # noqa: E402
from ui import shell                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHELL_MARK = "LIDSKJALV · ramen"              # står bara i skalets stilblock


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


def _without_media(css: str) -> str:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)


LOGIN = shell.login_css()
GLOBAL = _rules(_without_media(LOGIN))
ROOT_EL = '[data-testid="stTextInputRootElement"]'


# ── Inloggningen: formuläret ─────────────────────────────────────────────────
_GATE = """
import auth
import streamlit as st
if auth.render_login_gate():
    st.markdown("INNE")
"""


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setattr(auth, "_password", lambda: "hemligt")
    monkeypatch.setattr(auth, "_cookie_token", lambda: "")
    return AppTest.from_string(_GATE, default_timeout=30)


def _inne(at):
    return any(m.value == "INNE" for m in at.markdown)


def _blocks(node):
    for child in getattr(node, "children", {}).values():
        yield child
        yield from _blocks(child)


def _form(at, form_id):
    return [b for b in _blocks(at.main)
            if type(getattr(b, "proto", None)).__name__ == "Block" and b.proto.WhichOneof("type") == "form"
            and b.proto.form.form_id == form_id]


def test_login_is_one_form_that_enter_submits(gate):
    at = gate.run()
    assert not at.exception
    (form,) = _form(at, "login_form")
    assert form.proto.form.enter_to_submit is True and form.proto.form.border is False
    field, btn = at.text_input(key="login_pw"), at.button(key="login_btn")
    assert field.proto.form_id == btn.proto.form_id == "login_form"           # Enter/"Gå" skickar formuläret
    assert btn.proto.is_form_submitter and btn.label == "LOGGA IN" and btn.proto.type == "primary"
    assert field.label == "Åtkomstkod" and field.placeholder == "Ange åtkomstkod"
    assert field.proto.type == field.proto.PASSWORD and field.proto.label_visibility.value == 0     # synlig
    assert field.proto.autocomplete == "current-password"                     # ingen "starkt lösenord"-ruta på iOS


def test_enter_submit_logs_in_and_writes_the_cookie(gate, monkeypatch):
    seen = []
    monkeypatch.setattr(auth, "_set_cookie", lambda tok: seen.append(tok))
    at = gate.run()
    at.text_input(key="login_pw").input("hemligt")
    at.button(key="login_btn").click().run()                                    # formulärets knapp = Enter
    assert not at.exception
    assert _inne(at) and at.session_state["authenticated"] is True
    assert len(seen) == 1 and auth.token_valid(seen[0], "hemligt")             # kakan och localStorage som förut


def test_wrong_code_shows_the_swedish_message(gate):
    at = gate.run()
    at.text_input(key="login_pw").input("fel")
    at.button(key="login_btn").click().run()
    assert not at.exception and not _inne(at) and at.session_state["authenticated"] is False
    assert [e.value for e in at.error] == ["Fel kod — försök igen."]
    assert auth.WRONG_CODE == "Fel kod — försök igen."
    texts = " ".join(m.value for m in at.markdown)
    assert f"Inloggningen sparas i {auth.REMEMBER_DAYS} dagar på den här enheten." in texts
    assert "Access denied" not in texts and "ACCESS SYSTEM" not in _src("auth.py")


def test_login_page_keeps_the_bridge_and_the_fallback(gate):
    at = gate.run()
    assert any("wolf_bridge" in str(getattr(h, "proto", "")) for h in at.get("html"))
    src = _src("auth.py")
    assert '"wolf2026"' in src and "def check" not in src                      # huvudlösenordets reserv orörd
    assert auth.REMEMBER_DAYS == 30 and auth.QUERY_KEY == "wolf_t" and auth.COOKIE_NAME == "wolf_auth"


# ── Inloggningen: utseendet ──────────────────────────────────────────────────
def test_login_field_is_16px_and_52px():
    inp = GLOBAL[f".st-key-login_pw {ROOT_EL} input"]
    assert inp["font-size"] == "16px !important"                               # iOS zoomar in under 16 px
    assert "jetbrains mono" in inp["font-family"]
    assert GLOBAL[f".st-key-login_pw {ROOT_EL}"]["height"] == "52px"
    assert GLOBAL[".st-key-login_btn .stFormSubmitButton button"]["min-height"] == "52px"
    label = GLOBAL['.st-key-login_pw [data-testid="stWidgetLabel"] p']
    assert "13px" in label["font"] and label["color"] == "#a8a397 !important"  # MUTED, synlig
    eye = GLOBAL[f".st-key-login_pw {ROOT_EL} button"]
    assert eye["min-width"] == "44px" and eye["align-self"] == "stretch"      # ögat är en tryckyta


def test_login_column_is_a_card_on_desktop_and_the_header_is_hidden():
    card = GLOBAL[".st-key-login_card"]
    assert card["max-width"] == "360px" and "auto" in card["margin"]
    desk = re.findall(r"@media \(min-width: 641px\) \{([^@]*?)\n\}", LOGIN)
    desk_rules = _rules("\n".join(desk))
    c = desk_rules[".st-key-login_card"]
    assert c["background"] == "#0c0c12" and c["border-radius"] == "16px" and c["padding"] == "28px"
    assert c["border"] == "1px solid rgba(0,229,255,.15)"
    assert GLOBAL['body:has(.st-key-login_card) [data-testid="stHeader"]']["display"] == "none"   # alla bredder
    assert auth.LOGIN_KEY == "login_card"


def test_auth_style_has_only_arc_login_rules():
    css = re.sub(r"/\*.*?\*/", "", auth.LOGIN_CSS, flags=re.S)
    assert "@import" not in css and "900" not in css and ".stApp" not in css and "!important" not in css
    sels = [s.strip() for chunk in re.findall(r"([^{}]+)\{[^{}]*\}", re.sub(r"@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}",
                                                                              "", re.sub(r"@media[^{]*\{", "", css)))
            for s in chunk.split(",")]
    assert sels and all(s.startswith(".arc-login-") for s in sels), sels
    assert "@keyframes arc-login-breathe" in css and "4s" in css                # andningen
    assert re.search(r"prefers-reduced-motion: reduce\)\s*\{\s*\.arc-login-logo \{ animation: none; \}", css)
    assert ".arc-login-logo { width: 180px; height: 180px; }" in css            # 148 px telefon, 180 px dator
    assert "width: 148px; height: 148px" in css


def test_logo_without_duplicate_wordmark_and_tagline_once():
    html = auth._brand_html()
    assert "<img" in html and "arc-login-crop" in html and "NORDIC" not in html   # konsten bär ordmärket
    assert html.count("See What the Market Can't.") == 1 and html.count("<style>") == 1


# ── Inloggningen i panelen: skalet först ─────────────────────────────────────
def _panel():
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.environ["LOGIN_TEST_ROOT"])
    import wolf_panel
    wolf_panel.main()


def test_shell_style_rendered_once_on_the_login_page(monkeypatch):
    monkeypatch.setattr(auth, "_password", lambda: "hemligt")
    monkeypatch.setattr(auth, "_cookie_token", lambda: "")
    monkeypatch.setenv("LOGIN_TEST_ROOT", ROOT)
    at = AppTest.from_function(_panel, default_timeout=120).run()
    assert not at.exception, at.exception
    md = [m.value for m in at.markdown]
    shells = [v for v in md if SHELL_MARK in v]
    assert len(shells) == 1 and shells[0].count("apple-mobile-web-app-capable") == 1       # PWA-taggarna i skalet
    assert sum(v.count("apple-mobile-web-app-capable") for v in md) == 1                   # ingen egen PWA-markdown
    assert sum(v.count("<style") for v in md) == 2                                         # skalet + .arc-login-*
    assert at.button(key="login_btn") and at.text_input(key="login_pw")
    htmls = [str(getattr(h, "proto", "")) for h in at.get("html")]
    assert any("wolf_bridge" in h for h in htmls)                                          # bryggan kvar
    assert not any("__wolfNav" in h for h in htmls)                                        # inget nav-skript
    assert not any('class="wolf-bar"' in v for v in md)                                    # ingen märkesrad
    assert not at.tabs                                                                     # inga flikar


def test_main_draws_the_shell_before_the_gate():
    src = _src("wolf_panel.py")
    main = src[src.index("def main():"):]
    assert main.index("shell = inject_css()") < main.index("render_login_gate()") < main.index("bar = render_header()")
    assert "st.markdown(PWA_METAS" not in src and "import PWA_METAS" not in src and ", PWA_METAS" not in src
    assert "apple-mobile-web-app-capable" not in src                                       # taggarna bara i skalet


# ── Låset (§4.9) ─────────────────────────────────────────────────────────────
def _lock_app():
    import streamlit as st

    from portfolio.private_lock import gate
    key = st.session_state.get("_lock_key", "holdings_pw")
    if gate(key=key):
        st.markdown("UPPLÅST")


def _lock(key="holdings_pw"):
    at = AppTest.from_function(_lock_app, default_timeout=30)
    at.secrets[pl.SECRET] = "rätt-lösen"
    at.session_state["_lock_key"] = key
    return at.run()


@pytest.mark.parametrize("key", ["holdings_pw", "holdings_pw_alloc"])
def test_lock_is_a_form_in_a_card_with_the_same_keys(key):
    at = _lock(key)
    assert not at.exception
    (form,) = _form(at, f"{key}_form")
    assert form.proto.form.border is False and form.proto.form.enter_to_submit is True
    field, btn = at.text_input(key=key), at.button(key=f"{key}_btn")
    assert field.label == "Lösenord för innehaven" and btn.label == "Lås upp"
    assert field.proto.form_id == btn.proto.form_id == f"{key}_form" and btn.proto.is_form_submitter
    cards = [b for b in _blocks(at.main) if getattr(b.proto, "id", "").endswith(f"-{pl.CARD_PREFIX}{key}")]
    assert len(cards) == 1 and any("Dina innehav är låsta" in m.value for m in at.markdown)


def test_lock_submit_unlocks_and_lock_button_stays():
    at = _lock()
    at.text_input(key="holdings_pw").input("rätt-lösen")
    at.button(key="holdings_pw_btn").click().run()                              # Enter/"Gå" = formulärets knapp
    assert not at.exception and at.session_state[pl.STATE] is True
    assert any(m.value == "UPPLÅST" for m in at.markdown)
    assert at.button(key="holdings_pw_lock").label == "🔒 Lås innehaven"


def test_lock_wrong_password_and_rate_limit_messages_unchanged():
    at = _lock()
    at.text_input(key="holdings_pw").input("fel")
    at.button(key="holdings_pw_btn").click().run()
    assert [e.value for e in at.error] == ["Fel lösenord."] and pl.STATE not in at.session_state
    for _ in range(pl.MAX_FAILS - 1):
        at.button(key="holdings_pw_btn").click().run()
    assert [e.value for e in at.error] == [f"För många försök — vänta {pl.BLOCK_S} sekunder."]
    assert pl.STATE not in at.session_state and at.session_state[pl.UNTIL] > 0         # spärren som förut


def test_lock_field_is_16px_and_52px_in_a_card():
    card = GLOBAL[shell.LOCK_CARD]
    assert card["border-radius"] == "12px" and "auto" in card["margin"] and card["max-width"] == "420px"
    lock_field = f'{shell.LOCK_CARD} .stTextInput {ROOT_EL}'
    assert GLOBAL[lock_field]["height"] == "52px"
    assert GLOBAL[f"{lock_field} input"]["font-size"] == "16px !important"
    assert GLOBAL[f"{shell.LOCK_CARD} .stFormSubmitButton button"]["min-height"] == "52px"
    assert f'[class*="st-key-{pl.CARD_PREFIX}"]' == shell.LOCK_CARD                   # samma prefix i båda filerna


def test_login_css_is_part_of_the_one_shell_block():
    block = shell.style_block()
    assert block.count("<style") == 1 and LOGIN.strip() in block
    assert block.index("LIDSKJALV · byggstenarna") < block.index("LIDSKJALV · inloggningen")   # efter knapparna
