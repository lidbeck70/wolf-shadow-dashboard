"""
LIDSKJALV PR 15 — byggstenarna: tokens, rubrik, ruta, kort, statuschip, beslutsremsa, knappar (docs/design_lidskjalv.md
§3.1–§3.5.19, §7).

  · tokens: bara tillägg (APP_BG, NOTE flyttad och omexporterad, MUTED, bläcktonerna, skalorna); ink() ger
    bläcktonerna sedan PR 16 (beslut 5); tint() ger rgba(r,g,b,a); skalets tillfälliga konstanter kommer ur ui/tokens
  · tile: sign > 0 grönt, < 0 rött, annars rutans färg; ' · ' staplas; Nordens 'n/m GRÖN' visas som 'n/m gröna'
    utan att laget.row_text eller ljuset ändras; saknas-ord blir det grå Saknas-chipet
  · all text som byggstenarna tar emot escapas
  · verdict_strip visar bara orden den får — den härleder inget eget
  · chip: huvudordet ur ui/vocab.pair i chiptabellens stil, källans ord grått bredvid
  · tap_card behåller nyckel, etikett och callback (AppTest); page_header har samma signatur, vänster som standard
  · sparraden är EN horisontell rad med samma nycklar och samma logik
  · knapparna: :disabled ser avstängd ut, :active trycks in; sekundärselektorn står kvar i ui/css.py
  · typsnittsgolvet 12 px i byggstenarna och i thor/cards
"""
import inspect
import os
import re
import sys
from html import unescape

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from odin import laget  # noqa: E402
from ui import components as cp  # noqa: E402
from ui import shell, tokens, vocab  # noqa: E402

EVIL = '<b>x</b>"\'><script>alert(1)</script>&'


def _src(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _text(html: str) -> str:
    """Den synliga texten: taggarna bort, entiteterna tillbaka."""
    return unescape(re.sub(r"<[^>]+>", "", html))


def _sizes(html: str) -> list:
    """Alla font-size i px (rem räknas om, clamp() läses på sitt golv)."""
    out = []
    for v, unit in re.findall(r"font-size:\s*(?:clamp\()?([0-9.]+)(px|rem)", html):
        out.append(float(v) * (16 if unit == "rem" else 1))
    return out


def _capture(monkeypatch) -> list:
    """st.markdown i komponentmodulen → en lista (för byggstenarna som ritar själva)."""
    seen = []
    monkeypatch.setattr(cp.st, "markdown", lambda body, **kw: seen.append(body))
    return seen


# ── Tokens ───────────────────────────────────────────────────────────────────
def test_tokens_are_additions_with_the_spec_values():
    t = tokens
    assert (t.APP_BG, t.BG, t.BG_CARD, t.BG_ALT, t.BORDER) == ("#05070A", "#0c0c12", "#14141e", "#1a1f25", "#2a2a38")
    assert t.BORDER_FAINT == "rgba(255,255,255,0.06)"
    assert (t.TEXT, t.NOTE, t.MUTED, t.DIM) == ("#e8e4dc", "#c4bfb3", "#a8a397", "#8a8578")
    assert (t.GREEN_INK, t.RED_INK, t.PURPLE_INK) == ("#45b36b", "#e36464", "#c35cff")
    # de gamla värdena står kvar
    assert (t.GREEN, t.AMBER, t.RED, t.ACCENT, t.CYAN, t.GOLD, t.EMBER, t.PURPLE, t.GREY) == (
        "#2d8a4e", "#d4943a", "#c44545", "#00E5FF", "#00E5FF", "#c9a84c", "#FF6B3D", "#B400FF", "#6b7280")
    assert t.TAP == 44 and t.FIELD_H == 48 and t.FS_MIN == 12 and t.FS_LABEL == 12 and t.FS_INPUT == 16
    assert (t.SP_1, t.SP_2, t.SP_3, t.SP_4, t.SP_5, t.SP_6, t.SP_8) == (4, 8, 12, 16, 20, 24, 32)
    assert (t.R_CHIP, t.R_TILE, t.R_BUTTON, t.R_CARD, t.R_HERO, t.R_POPOVER, t.R_PILL) == (6, 10, 10, 12, 14, 16, 999)
    assert t.FS_TITLE == "clamp(20px, 5.6vw, 26px)"


def test_note_moved_to_tokens_and_is_re_exported_and_the_shell_reads_tokens():
    assert cp.NOTE is tokens.NOTE == "#c4bfb3"
    assert "NOTE = " not in _src("ui", "components.py")               # flyttad, inte kopierad
    assert shell.MUTED is tokens.MUTED and shell.APP_BG is tokens.APP_BG and shell.NOTE is tokens.NOTE
    src = _src("ui", "shell.py")
    assert not re.search(r'^(MUTED|APP_BG|CARD_BG) = "#', src, re.M)  # inga tillfälliga konstanter kvar


def test_ink_returns_the_ink_tones_since_pr16():
    assert tokens.INK_ON is True                                       # beslut 5 godkänt (PR 16)
    assert tokens.ink(tokens.GREEN) == tokens.GREEN_INK == "#45b36b"
    assert tokens.ink(tokens.RED) == tokens.RED_INK == "#e36464"
    assert tokens.ink(tokens.PURPLE) == tokens.PURPLE_INK == "#c35cff"
    for col in (tokens.AMBER, tokens.ACCENT, tokens.GOLD, tokens.EMBER, tokens.GREY, "#123456"):
        assert tokens.ink(col) == col                                  # alla andra oförändrade


def test_ink_switch_is_one_flag(monkeypatch):
    assert tokens.ink(tokens.GREEN.upper()) == tokens.GREEN_INK        # skiftläget spelar ingen roll
    monkeypatch.setattr(tokens, "INK_ON", False)                       # avslaget: basfärgen överallt
    for col in (tokens.GREEN, tokens.RED, tokens.PURPLE, tokens.AMBER, tokens.ACCENT):
        assert tokens.ink(col) == col


def test_tint_format():
    assert tokens.tint(tokens.AMBER, .14) == "rgba(212,148,58,0.14)"
    assert tokens.tint("#00E5FF", 0.5) == "rgba(0,229,255,0.5)"
    assert tokens.tint("#fff", 1) == "rgba(255,255,255,1)"
    assert tokens.tint(tokens.RED, 2) == "rgba(196,69,69,1)" and tokens.tint(tokens.RED, -1) == "rgba(196,69,69,0)"
    assert tokens.tint("rgba(1,2,3,.5)", .2) == "rgba(1,2,3,.5)" and tokens.tint("#xyzxyz", .2) == "#xyzxyz"


# ── Rutan ────────────────────────────────────────────────────────────────────
def test_tile_sign_colours_only_follow_the_sign():
    up = cp.tile("1 mån", "+3.2 %", tokens.ACCENT, sign=3.2)
    down = cp.tile("1 mån", "-1.0 %", tokens.ACCENT, sign=-1)
    flat = cp.tile("1 mån", "0.0 %", tokens.ACCENT, sign=0)
    price = cp.tile("Kurs", "245.10", tokens.TEXT)                  # pris: ingen teckenfärg
    assert f"color:{tokens.GREEN}" in up and tokens.RED not in up
    assert f"color:{tokens.RED}" in down and tokens.GREEN not in down
    assert f"color:{tokens.ACCENT}" in flat and tokens.GREEN not in flat and tokens.RED not in flat
    assert f"color:{tokens.TEXT}" in price and tokens.GREEN not in price and tokens.RED not in price
    assert f"border-left:3px solid {tokens.GREEN}" in up                     # vänsterlinjen i värdets färg
    assert cp.tile("x", "1", tokens.GOLD, sign=None).count(tokens.GOLD) == 2
    assert cp.tile("x", "1", tokens.GOLD, sign="inte ett tal").count(tokens.GOLD) == 2


def test_tile_number_is_mono_word_is_grotesk_and_long_words_shrink():
    assert "JetBrains Mono" in cp.tile("Risk", "1 500 kr")
    word = cp.tile("Allokering", "NORMAL", tokens.GREEN, kind="word")
    assert "Space Grotesk" in word and "font-size:20px" in word
    assert "font-size:16px" in cp.tile("Läge", "HALVERAD RISK NU", tokens.RED, kind="word")   # över 14 tecken
    assert "JetBrains Mono" in cp.tile("Swing", "GUL", tokens.AMBER, "2026-11-20", kind="word").split("GUL")[-1]


def test_tile_stacks_parts_coloured_by_laget_color():
    html = cp.tile("Marknadsrisk", "SPY LÅG · OMXS30 FÖRHÖJD", tokens.AMBER, kind="word")
    lines = re.findall(r'<div style="color:([^;]+);[^"]*font-size:15px[^"]*">([^<]+)</div>', html)
    # 15 px: bläcktonen av laget._color (beslut 5, PR 16) — LÅG är grön → GREEN_INK
    assert lines == [(tokens.ink(laget._color("LÅG")), "SPY LÅG"), (tokens.ink(laget._color("FÖRHÖJD")), "OMXS30 FÖRHÖJD")]


def test_norden_display_rule_leaves_row_text_and_the_light_unchanged():
    board = {"segments": [{"key": k, "nordic": {"status": s}} for k, s in
                          zip("abcdefghijk", ["GRÖN"] * 4 + ["RÖD"] * 7)]}
    items = laget.capital_row(None, None, None, board)
    before = [tuple(i) for i in items]
    text_before = laget.row_text(items)
    norden = items[-1]
    assert norden[1] == "4/11 GRÖN" and norden[3] == laget._color("GUL")      # ljuset: GUL (minoritet gröna)
    html = cp.light_grid(items)
    assert "4/11" in html and ">gröna<" in html and "4/11 GRÖN" not in html
    tile = html.split("Norden")[1]
    assert f"color:{norden[3]}" in tile                                       # rutans egen färg, ingen ny
    assert [tuple(i) for i in items] == before                                # posterna orörda
    assert laget.row_text(items) == text_before and "Norden 4/11 GRÖN" in laget.row_text(items)


def test_tile_missing_value_is_the_grey_saknas_chip():
    for value in (None, "Saknas", "DATA_GAP", "DATA UNAVAILABLE"):
        html = cp.tile("Swing", value, tokens.GREEN, "wolf_regime.json saknas", kind="word")
        assert ">Saknas<" in html and "dashed" in html and 'title="wolf_regime.json saknas"' in html, value
        assert f"border-left:3px solid {tokens.GREY}" in html and "None" not in _text(html)


def test_none_nan_and_data_gap_show_saknas_never_the_raw_word():
    """§3.5.12: 'DATA GAP' och None är Saknas — tile och chip skriver aldrig ut 'None', 'nan' eller 'DATA GAP'."""
    import pandas as pd
    for value in (None, "None", "nan", float("nan"), pd.NA, "", "DATA GAP", "No data available", "N/A"):
        t = cp.tile("Swing", value, tokens.GREEN, kind="word")
        c = cp.chip(value)
        for html in (t, c):
            assert ">Saknas<" in html and "dashed" in html, repr(value)
            assert not re.search(r"\b(None|nan|NaN|DATA GAP|No data available)\b", _text(html)), repr(value)
    assert ">Saknas<" not in cp.chip("KÖP") and ">Saknas<" not in cp.tile("Pris", "271.40")


def test_light_grid_has_one_tile_per_light_in_order():
    items = laget.capital_row({"peak": 100, "current": 99}, {"regime": "GUL", "generated": "2026-11-20"}, None, None)
    html = cp.light_grid(items)
    assert html.count('class="wolf-tile"') == 4 and 'class="wolf-grid"' in html
    pos = [html.index(t) for t, *_ in items]
    assert pos == sorted(pos)


def test_kpi_strip_takes_kpi_row_items():
    html = cp.kpi_strip([("Kurs", "245.10", tokens.TEXT, "2026-11-20"), ("1 mån", "+3.2 %", tokens.GREEN, ""),
                         ("Läge", "GRÖN", tokens.GREEN)])
    assert html.count("flex:0 0 132px") == 3 and "overflow-x:auto" in html and 'class="wolf-kstrip"' in html
    assert "min-height:64px" in html


# ── Escapning ────────────────────────────────────────────────────────────────
def _no_markup(html: str):
    assert "<b>" not in html and "<script" not in html and "&lt;b&gt;" in html, html
    assert '"\'><script' not in html


def test_every_text_input_is_escaped(monkeypatch):
    E = EVIL
    for html in (
        cp.section(E), cp.section("x", count=E), cp.section("x", count=[(E, tokens.GREEN)]),
        cp.verdict_strip(E, [(E, tokens.GREEN)]), cp.verdict_strip("k", [(E, tokens.GREEN)], E, E),
        cp.tile(E, E), cp.tile("x", "1", sub=E), cp.tile(E, E + " · " + E, kind="word"), cp.tile(E, E, kind="word"),
        cp.light_grid([(E, E, E, tokens.GREEN)]), cp.kpi_strip([(E, E, tokens.GREEN, E)]),
        cp.chip(E), cp.chip(E, vocab.CHECK), cp.chip("KÖP · " + E), cp.source_chip(E),
        cp.action_row(E, tokens.GREEN, E, E, E, jump=E), cp.card(E, "rad"),
    ):
        _no_markup(html)
    assert 'title="' + E.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;") \
        .replace("'", "&#x27;") + '"' in cp.missing(E)
    assert "<script" not in cp.chip("DATA_GAP", why=E)
    seen = _capture(monkeypatch)
    cp.page_header(E, E)
    cp.empty_state(E)
    for html in seen:
        assert "<b>x" not in html and "<script" not in html, html


def test_head_row_escapes_meta_and_subtitle(monkeypatch):
    seen = _capture(monkeypatch)

    class Box:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    made = {}
    monkeypatch.setattr(cp.st, "container", lambda **kw: made.update(kw) or Box())
    with cp.head_row("laget", subtitle=EVIL, meta=EVIL):
        cp.page_header("👁 Läget")
    assert made == {"horizontal": True, "vertical_alignment": "center", "key": "head_laget"}
    assert len(seen) == 2 and "👁 LÄGET" in seen[0]
    assert "<script" not in seen[1] and "&lt;b&gt;" in seen[1] and "JetBrains Mono" in seen[1]


# ── Beslutsremsan ────────────────────────────────────────────────────────────
def test_verdict_strip_shows_only_the_words_it_is_given():
    words = [("KÖP", tokens.GREEN), ("GUL", tokens.AMBER), ("2 SÄLJ", tokens.RED)]
    html = cp.verdict_strip("Dagens besked · antal i dag", words, "Ingen vändning i köpfönstret", "hämtad 16:52")
    assert _text(html) == ("Dagens besked · antal i dag" + "KÖP · GUL · 2 SÄLJ" + "Ingen vändning i köpfönstret"
                           + "hämtad 16:52")
    for w, c in words:
        assert f'<span style="color:{c};">{w}</span>' in html
    # inget ord ur ordlistan som inte skickades in
    for main in (vocab.BEVAKA, vocab.HALL, vocab.VAKSAM, vocab.MINSKA, vocab.NEJ, vocab.SAKNAS):
        assert main not in _text(html)
    one = cp.verdict_strip("Swing", [("RÖD", tokens.RED)], rule=tokens.RED)
    assert _text(one) == "SwingRÖD" and f"border-left:4px solid {tokens.RED}" in one
    assert f"border-left:4px solid {tokens.ACCENT}" in cp.verdict_strip("k", words)
    assert _text(cp.verdict_strip("k", [])) == "k"


# ── Statuschip ───────────────────────────────────────────────────────────────
def test_chip_follows_the_vocab_table():
    kop = cp.chip("GOLDEN TICKET")
    assert ">KÖP<" in kop and f"background:{tokens.GREEN}" in kop and f"color:{tokens.APP_BG}" in kop
    assert ">GOLDEN TICKET<" in kop and f"background:{tokens.BG_CARD}" in kop        # källans ord, grått
    sell = cp.chip("EXIT")
    assert ">SÄLJ<" in sell and f"background:{tokens.RED}" in sell and "color:#ffffff" in sell and ">EXIT<" in sell
    reduce = cp.chip("REDUCE")
    assert ">MINSKA<" in reduce and f"1.5px solid {tokens.AMBER}" in reduce
    assert ">BEVAKA<" in cp.chip("WATCH") and ">● VAKSAM<" in cp.chip("WATCH", vocab.HOLDING)
    hold = cp.chip("HOLD")
    assert ">HÅLL ✓<" in hold and f"color:{tokens.ink(tokens.GREEN)}" in hold
    nej = cp.chip("PASS")
    assert ">NEJ<" in nej and f"color:{tokens.MUTED}" in nej and ">PASS<" in nej
    assert "line-through" in cp.chip("SPÄRRAD") and "line-through" not in nej
    ok = cp.chip("PASS", vocab.CHECK)
    assert ">✓ OK<" in ok and ">NEJ<" not in ok
    assert ">✗ NEJ<" in cp.chip("FAIL", vocab.CHECK) and tokens.RED in cp.chip("FAIL", vocab.CHECK)
    assert ">◌ KOLLA<" in cp.chip("MANUAL", vocab.CHECK) and ">◌<" not in cp.chip("◌", vocab.CHECK)
    gap = cp.chip("DATA_GAP")
    assert ">Saknas<" in gap and "dashed" in gap and 'title="DATA_GAP"' in gap
    assert ">KÖP<" in cp.chip("KÖP") and cp.chip("KÖP").count("wolf-chip") == 1      # inget källchip när lika
    neutral = cp.chip("Neutral")
    assert ">Neutral<" in neutral and f"1px solid {tokens.GREY}" in neutral          # omappat: som det är
    assert ">GOLDEN TICKET<" not in cp.chip("GOLDEN TICKET", source=False)


def test_chips_are_22px_and_12px():
    for html in (cp.chip("KÖP"), cp.missing(), cp.source_chip("X"), cp.chip("FAIL", vocab.CHECK)):
        assert "height:22px" in html and "font-size:12px" in html


def test_action_row_uses_the_chip_style_and_jump():
    row = cp.action_row("SÄLJ", tokens.RED, "VOLV-B.ST", "under EMA50", "12:40", chips=[cp.source_chip("EXIT")],
                        jump="pos-volv-b")
    assert 'data-jump="pos-volv-b"' in row and "min-height:56px" in row and "84px minmax(0,1fr) auto" in row
    assert f"background:{tokens.RED}" in row and ">EXIT<" in row and ">12:40<" in row
    other = cp.action_row("EGET ORD", tokens.EMBER, "X")
    assert f"1px solid {tokens.EMBER}" in other and "data-jump" not in other


# ── Rubriken, kort, tomt, saknas, skelett ────────────────────────────────────
def test_page_header_keeps_its_signature_and_is_left_by_default(monkeypatch):
    sig = inspect.signature(cp.page_header)
    assert list(sig.parameters) == ["title", "subtitle", "accent", "align"]
    assert sig.parameters["align"].default == "left" and sig.parameters["accent"].default == tokens.GOLD
    seen = _capture(monkeypatch)
    cp.page_header("👀 Övervakning", "Aktier nära köpläge")
    (html,) = seen
    assert "<h2" in html and "👀 ÖVERVAKNING" in html and "text-align:left" in html
    assert "word-break:keep-all" in html and "text-wrap:balance" in html and "clamp(20px, 5.6vw, 26px)" in html
    assert f"color:{tokens.NOTE}" in html and "max-width:60ch" in html and "font-size:13px" in html
    cp.page_header("x", align="center")
    assert "text-align:center" in seen[-1]


def test_restyled_components_have_no_text_under_12px(monkeypatch):
    htmls = [cp.section("x", count="2"), cp.verdict_strip("k", [("KÖP", tokens.GREEN)], "s", "m"),
             cp.tile("x", "1", sub="2026-11-20"), cp.tile("x", "SPY LÅG · OMX HÖG", kind="word"),
             cp.light_grid(laget.capital_row(None, None, None, None)), cp.kpi_strip([("a", "1")]),
             cp.chip("GOLDEN TICKET"), cp.missing("x"), cp.action_row("KÖP", tokens.GREEN, "t", "d", "m"),
             cp.card("t", "b"), cp.kpi("Swing-regim", "GRÖN", tokens.GREEN, "2026-09-22"),
             cp.big_card("T", "1", "s", tokens.GREEN), cp.badge("x", tokens.GREEN), cp.verdict_box("1", "x", "n")]
    seen = _capture(monkeypatch)
    cp.page_header("t", "s")
    cp.empty_state("tom")
    for html in htmls + seen:
        sizes = _sizes(html)
        assert sizes and min(sizes) >= 12, (min(sizes or [0]), html[:120])


def test_thor_cards_have_a_12px_floor():
    sizes = _sizes(_src("thor", "cards.py"))
    assert sizes and min(sizes) >= 12


def test_card_empty_state_and_skeleton(monkeypatch):
    c = cp.card("Thor", "Alla kandidater", tokens.GOLD)
    assert f"background:{tokens.BG_CARD}" in c and "border-radius:12px" in c and f"3px solid {tokens.GOLD}" in c
    assert "border-left" not in cp.card("Thor", "x", "")
    seen = _capture(monkeypatch)
    assert cp.empty_state("Inget bevakas än") is False
    assert "dashed" in seen[-1] and "text-align:center" in seen[-1] and f"color:{tokens.MUTED}" in seen[-1]
    sk = cp.skeleton_html(3, 96)
    assert sk.count('class="wolf-skel"') == 3 and "height:96px" in sk and 'role="status"' in sk


# ── Tryckkortet (AppTest) ────────────────────────────────────────────────────
def _tap_app():
    import streamlit as st

    from ui.components import tap_card

    def go(target, ticker=""):
        st.session_state["went"] = (target, ticker)

    tap_card("thor_open_VOLV", "<div>VOLV-B.ST · KÖP</div>", "Öppna →", on_click=go, args=("thor",),
             kwargs={"ticker": "VOLV-B.ST"})


def test_tap_card_keeps_key_label_and_callback():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_tap_app).run()
    assert not at.exception
    btn = at.button(key="thor_open_VOLV")
    assert btn.label == "Öppna →"
    (box,) = list(at.main.children.values())
    assert box.proto.id.endswith("-tapc_thor_open_VOLV")
    assert "VOLV-B.ST · KÖP" in at.markdown[0].value
    btn.click().run()
    assert not at.exception and at.session_state["went"] == ("thor", "VOLV-B.ST")


# ── Sparraden (AppTest) ──────────────────────────────────────────────────────
def _save_app():
    import storage_ui
    storage_ui.save_bar("holdings", "Holdings")


def test_save_bar_is_one_horizontal_row_with_the_same_keys(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import storage
    monkeypatch.setattr(storage, "load_failed", lambda name: None)
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: {"when": "2026-11-21T16:40:00", "sha": "abc1234"})
    at = AppTest.from_function(_save_app).run()
    assert not at.exception
    (row,) = list(at.main.children.values())
    assert row.proto.id.endswith("-save_holdings_row")
    fc = row.proto.flex_container
    assert fc.direction == type(fc).Direction.Value("HORIZONTAL") and fc.align == type(fc).Align.Value("ALIGN_CENTER")
    btn = at.button(key="save_holdings")
    assert btn.label == "💾 Spara" and not btn.disabled
    text = " ".join(m.value for m in at.markdown)
    assert "Inga osparade ändringar" in text and "16:40" in text and "UTC" in text
    src = _src("storage_ui.py")
    assert 'st.container(horizontal=True, vertical_alignment="center", key=f"{key}_row")' in src
    assert "st.columns" not in src.split("def save_bar")[1].split("def footer")[0]


def test_save_bar_dirty_still_turns_the_button_primary(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import storage
    monkeypatch.setattr(storage, "load_failed", lambda name: None)
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: True)
    at = AppTest.from_function(_save_app).run()
    assert not at.exception
    assert at.button(key="save_holdings").proto.type == "primary"
    assert "Du har osparade ändringar" in " ".join(m.value for m in at.markdown)


# ── Knappar och skalets regler ───────────────────────────────────────────────
def _rules(css: str) -> dict:
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


PARTS = shell.parts_css()
RULES = _rules(re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "",
                      re.sub(r"@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", PARTS)))


def test_buttons_disabled_and_active_states():
    sec = '.stButton > button[kind="secondary"]'
    assert '.stButton > button[kind="secondary"] {' in _src("ui", "css.py")       # test_pr6: selektorn står kvar
    assert RULES[sec]["min-height"] == "44px" and RULES[sec]["border-radius"] == "10px"
    assert RULES[sec]["font-size"] == "14px" and RULES[sec]["letter-spacing"] == ".06em"
    dis = RULES[".stButton > button:disabled"]
    assert dis["opacity"] == ".45" and dis["cursor"] == "not-allowed" and dis["box-shadow"] == "none"
    assert RULES[f"{sec}:disabled"]["border-color"] == "rgba(0,229,255,.25)"
    assert RULES[".stButton > button:active"]["transform"] == "scale(.98)"
    for kind in ('[data-testid="stDownloadButton"] button', '[data-testid="stFormSubmitButton"] button'):
        assert RULES[f"{kind}:disabled"]["opacity"] == ".45"
    # skalet ligger EFTER CYBERPUNK_CSS: samma selektor vinner på ordningen
    block = shell.style_block()
    assert block.index("LIDSKJALV · byggstenarna") > block.index('.stButton > button[kind="secondary"] {')
    # primärknappen sedan beslut 4 (PR 16): mörk text på temats cyan — tests/test_design_theme.py
    assert '.stButton > button[kind="primary"]:not(:disabled)' in RULES


def test_tap_card_head_row_links_and_kpis_css():
    tap = RULES['[class*="st-key-tapc_"] > [data-testid="stElementContainer"]:has(.stButton)']
    assert tap["position"] == "absolute" and tap["inset"] == "0"
    assert RULES['[class*="st-key-tapc_"] .stButton button']["opacity"] == "0"
    assert RULES['[class*="st-key-tapc_"]::after']["content"] == '"›"'
    assert RULES['[class*="st-key-head_"]']["flex-wrap"] == "wrap"
    title = RULES['[class*="st-key-head_"] > [data-testid="stElementContainer"]:has(.wolf-head)']
    assert title["flex"] == "1 1 auto" and title["min-width"] == "min(180px, 100%)"
    assert RULES['[class*="st-key-tapc_"] [data-testid="stMarkdownContainer"]']["margin-bottom"] == "0"
    link = RULES['[class*="st-key-links_"] .stButton > button[kind="tertiary"]']
    assert link["min-height"] == "48px" and link["width"] == "100%"
    assert 'button[kind="tertiary"]' not in PARTS.replace('[class*="st-key-links_"] .stButton > button[kind="tertiary"]',
                                                          "")    # aldrig en global omstyrning av tertiära knappar
    kpis = '[class*="st-key-kpis_"]'
    assert RULES[f'{kpis} [data-testid="stMetricValue"]']["font-size"] == "20px"
    assert RULES[f'{kpis} [data-testid="stMetricLabel"]']["font-size"] == "12px"
    assert RULES[f'{kpis} [data-testid="stMetric"]::before']["content"] == "none"
    assert ".wolf-grid" in RULES and "@keyframes wolf-shimmer" in PARTS
    assert re.search(r"prefers-reduced-motion: reduce\)\s*\{\s*\.wolf-skel \{ animation: none; \}", PARTS)
    assert shell.parts_css() in shell.shell_css()


def test_section_title_wrappers_share_the_new_label():
    from ui import theme
    seen = []
    import streamlit as st
    orig = st.markdown
    try:
        st.markdown = lambda body, **kw: seen.append(body)
        theme.section_title("Läget just nu", "📡")
    finally:
        st.markdown = orig
    assert seen and "wolf-sec" in seen[0] and "Läget just nu" in seen[0] and f"color:{tokens.MUTED}" in seen[0]
    assert RULES[".section-title"]["color"] == tokens.MUTED


def test_sparkline_is_inline_svg_of_the_same_series():
    from ui.charts import sparkline_svg
    svg = sparkline_svg([1, 3, 2], "#2d8a4e", width=100, height=20)
    assert svg.startswith("<svg") and "polyline" in svg and 'stroke="#2d8a4e"' in svg
    pts = re.search(r'points="([^"]+)"', svg).group(1).split()
    assert len(pts) == 3 and pts[1].endswith(",2.0")                     # högsta värdet överst
    assert sparkline_svg([1]) == "" and sparkline_svg([]) == "" and sparkline_svg(None) == ""
    # Ett tomt värde i serien hoppas över — linjen ritas ändå, oavsett typ
    import pandas as pd
    for series in ([1, None, 2, 3], pd.Series([1, None, 2, 3], dtype=object),
                   pd.Series([1, pd.NA, 2, 3], dtype="Float64"), [1, float("nan"), 2, 3]):
        svg = sparkline_svg(series)
        assert "polyline" in svg and len(re.search(r'points="([^"]+)"', svg).group(1).split()) == 3, series


def test_kpis_metric_values_wrap_instead_of_clipping_on_the_phone():
    css = shell.shell_css()
    phone = css[css.index("Telefon: rutan är 114"):]
    phone = phone[:phone.index("Ljusrutnätet")]
    assert "@media (max-width: 640px)" in phone and '[class*="st-key-kpis_"] [data-testid="stMetricValue"]' in phone
    assert "white-space: normal !important" in phone and "text-overflow: clip !important" in phone
    assert "font-size: 16px !important" in phone


@pytest.mark.parametrize("path", ["ui/components.py", "ui/tokens.py", "storage_ui.py"])
def test_no_model_names_or_secrets_in_the_parts(path):
    src = _src(*path.split("/"))
    assert "wolf2026" not in src and "claude" not in src.lower() and "gpt" not in src.lower()


# ── LIDSKJALV PR 20: thor.cards.storlek_grid (§3.5.14) ───────────────────────
def test_storlek_grid_shows_exactly_the_sz_values_and_the_row_text():
    from thor import cards
    from thor import storlek as sk
    sz = sk.size(100.0, 92.0, 100000, risk=1.5, cap=6.0)
    row = sk.row_text("ABC.ST", "🛡 Viking", sz)
    html = cards.storlek_grid(sz, 1.5, row)
    text = _text(html)
    labels = [m.upper() for m in re.findall(r"text-transform:uppercase;overflow-wrap:anywhere;\">([^<]+)<", html)]
    assert labels == ["KÖP", "STOPP", "MÅL 3R", "RISKSTORLEK", "RISK", "TAK"]          # 2 × 3 i specens ordning
    values = re.findall(r"font-variant-numeric:tabular-nums;margin-top:4px;overflow-wrap:anywhere;\">([^<]+)<", html)
    assert values == [f"{sz['entry']:.2f}", f"{sz['stop']:.2f}", f"{sz['target_3r']:.2f}", str(sz["risk_shares"]),
                      "1 500", str(sz["cap_shares"])] == ["100.00", "92.00", "124.00", "187", "1 500", "60"]
    assert "6 000 kr" in text and "1.5 %" in text                                      # taket i kronor, risk-%
    assert row in text and html.count(row) == 1                                        # raden att kopiera, ordagrant
    assert "user-select:all" in html and f"color:{tokens.MUTED};font-size:12px" in html
    assert min(_sizes(html)) >= 12 and "None" not in text
    half = sk.halved(sz)                                                               # GUL regim: Thors egen halvering
    h = cards.storlek_grid(half, 1.5, sk.row_text("ABC.ST", "x", half))
    assert ">93<" in h and ">30<" in h and "halverat" in _text(h)                     # talen ur dicten, ingen ny räkning
    bare = sk.size(100.0, 92.0, 100000)                                                # ingen risk-% och inget tak
    b = cards.storlek_grid(bare, None)
    assert _text(b).count("Saknas") == 3 and "None" not in _text(b) and "thor-rowtext" not in b
    src = inspect.getsource(cards.storlek_grid)
    assert not re.search(r"(?<![a-z_])(sz\[[^\]]+\]|v|x)\s*[-+*/]\s*[\d(]", src)        # ingen aritmetik på talen
