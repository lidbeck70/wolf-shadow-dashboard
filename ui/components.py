"""
ui/components.py — byggstenarna (LIDSKJALV §3.5.7–§3.5.19): EN uppsättning för hela panelen.

Sex rubrikmönster, fem identiska _badge-kopior, sju "verdiktrutor" med samma CSS och tolv raderaknappar utan
bekräftelse — var och en kopierad mellan filer. Här finns EN version av varje. De som returnerar HTML (str) går
att sätta ihop; de som ritar widgets (page_header, head_row, tap_card, link_rows, skeleton, empty_state,
confirm_delete) skriver själva.

Reglerna för alla:
  · all text escapas — en ticker, ett felmeddelande eller ett namn kan aldrig bli HTML
  · inget ord härleds: remsor, rutor och chip visar exakt de ord och färger anroparen skickar (ordlistan
    ui/vocab väljer bara huvudordet för källans ord, och källans ord står kvar som grått chip)
  · utseendet bor i stilattributen; det som kräver mediafrågor, :has() eller animering (rutnätets kolumner,
    tryckkortets osynliga knapp, huvudraden, skelettet) står i ui/shell.parts_css()
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from html import escape
from typing import Optional

import streamlit as st

from ui import vocab
from ui.tokens import (ACCENT, AMBER, APP_BG, BG_ALT, BG_CARD, BORDER, BORDER_CARD, BORDER_FAINT, DIM, FONT_MONO,
                       FONT_UI, FS_TITLE, GOLD, GREEN, GREY, MUTED, NOTE, RED, TEXT, ink)

__all__ = ["NOTE", "page_header", "head_row", "section", "verdict_strip", "tile", "light_grid", "kpi_strip",
           "tap_card", "chip", "source_chip", "missing", "action_row", "link_rows", "skeleton", "skeleton_html", "empty_state",
           "card", "badge", "verdict_box", "kpi", "note", "big_card", "confirm_delete"]


def _t(x) -> str:
    """Text i ett element: < > & escapas (citattecken behöver inte — texten står kvar ordagrant i markdownen)."""
    return escape("" if x is None else str(x), quote=False)


def _a(x) -> str:
    """Värde i ett attribut: också citattecken escapas."""
    return escape("" if x is None else str(x), quote=True)


# ── Sidhuvudet (§3.5.7) ──────────────────────────────────────────────────────
def page_header(title: str, subtitle: str = "", accent: str = GOLD,
                align: str = "left") -> None:
    """Sidhuvudet: rubriken ÄR flikens namn (så navigationen och sidan säger samma sak), en rad i NOTE under.
    EN st.markdown; rubriken bryts aldrig mitt i ett ord (RÅVARUHÄVSTÅNG ryms på en rad i 20 px).
    Ritas direkt under sparraden — eller i head_row() när sidan har knappar bredvid rubriken."""
    centre = align == "center"
    st.markdown(
        f'<div class="wolf-head" style="text-align:{_a(align)};padding:8px 0 4px;min-width:0;">'
        f'<h2 style="color:{_a(accent)};font-family:{FONT_UI};font-size:{FS_TITLE};font-weight:700;line-height:1.15;'
        f'letter-spacing:.06em;margin:0;padding:0;word-break:keep-all;overflow-wrap:normal;hyphens:manual;'
        f'text-wrap:balance;">{escape(title).upper()}</h2>'
        + (f'<p style="color:{NOTE};font-family:{FONT_UI};font-size:13px;line-height:1.45;max-width:60ch;'
           f'margin:4px {"auto" if centre else "0"} 0;">{_t(subtitle)}</p>' if subtitle else "")
        + "</div>", unsafe_allow_html=True)


@contextmanager
def head_row(key: str, subtitle: str = "", meta: str = ""):
    """Huvudraden: rubriken och sidans knappar (📡 Uppdatera …) på en rad, som radbryts högerställt när de inte
    ryms. Sidan anropar page_header(titel) INNE i blocket och skickar underraden hit:

        with head_row("laget", subtitle="…", meta="hämtad 16:52"):
            page_header("👁 Läget")
            st.button("📡 Uppdatera", key="laget_refresh")

    Efter blocket skrivs meta (mono 12 px MUTED) och underraden med note()."""
    box = st.container(horizontal=True, vertical_alignment="center", key=f"head_{key}")
    with box:
        yield box
    if meta or subtitle:
        note((f'<span style="display:block;font-family:{FONT_MONO};font-size:12px;line-height:1.3;color:{MUTED};">'
              f'{_t(meta)}</span>' if meta else "") + _t(subtitle))


# ── Sektionsrubriken (§3.5.8) ────────────────────────────────────────────────
def section(label: str, color: str = MUTED, count="") -> str:
    """Liten versal sektionsrubrik (12 px, MUTED) med en tunn cyan linje under. count = en räkning till höger i
    mono: text, eller [(text, färg), …] där varje del står i sin egen färg ('2 KÖP · 5 BEVAKA')."""
    if isinstance(count, (list, tuple)):
        right = f'<span style="color:{MUTED};"> · </span>'.join(
            f'<span style="color:{_a(ink(c))};">{_t(t)}</span>' for t, c in count)
    else:
        right = _t(count)
    return (f'<div class="wolf-sec" style="display:flex;align-items:baseline;justify-content:space-between;gap:8px;'
            f'margin:20px 0 8px;padding-bottom:6px;border-bottom:1px solid rgba(0,229,255,.15);">'
            f'<span style="color:{_a(ink(color))};font-family:{FONT_UI};font-size:12px;font-weight:600;'
            f'line-height:1.2;letter-spacing:.12em;text-transform:uppercase;min-width:0;">{_t(label)}</span>'
            + (f'<span style="color:{MUTED};font-family:{FONT_MONO};font-size:12px;line-height:1.2;'
               f'white-space:nowrap;">{right}</span>' if right else "")
            + "</div>")


# ── Beslutsremsan (§3.5.9) ───────────────────────────────────────────────────
def verdict_strip(kicker: str, words, sentence: str = "", meta: str = "", rule: str = ACCENT) -> str:
    """Beslutsremsan överst på en sida: varje källas EGET ord i sin egen färg, sida vid sida — aldrig ihopslagna.
    words = [(text, färg), …] — ord eller räkningar som sidan redan räknat fram; remsan härleder inget eget ord.
    rule = vänsterlinjen: ACCENT när flera ord delar remsan, ordets färg när det bara finns ett."""
    sep = f'<span style="color:{MUTED};font-weight:400;"> · </span>'
    ws = sep.join(f'<span style="color:{_a(c)};">{_t(w)}</span>'
                  for w, c in ((x if isinstance(x, (list, tuple)) else (x, TEXT)) for x in (words or ())))
    return (f'<div class="wolf-verdict-strip" style="background:{BG_ALT};border:1px solid {BORDER_CARD};'
            f'border-left:4px solid {_a(rule)};border-radius:12px;padding:12px 14px;margin:4px 0 8px;min-width:0;">'
            + (f'<div style="color:{MUTED};font-family:{FONT_UI};font-size:12px;font-weight:600;line-height:1.2;'
               f'letter-spacing:.12em;text-transform:uppercase;">{_t(kicker)}</div>' if kicker else "")
            + (f'<div style="font-family:{FONT_UI};font-size:24px;font-weight:800;line-height:1.15;'
               f'letter-spacing:.02em;margin-top:4px;overflow-wrap:anywhere;">{ws}</div>' if ws else "")
            + (f'<div style="color:{TEXT};font-size:15px;line-height:1.45;margin-top:6px;">{_t(sentence)}</div>'
               if sentence else "")
            + (f'<div style="color:{MUTED};font-family:{FONT_MONO};font-size:12px;line-height:1.3;margin-top:4px;">'
               f'{_t(meta)}</div>' if meta else "")
            + "</div>")


# ── Rutan, ljusrutnätet och nyckeltalsremsan (§3.5.10) ───────────────────────
_NORDEN = re.compile(r"^(\d+/\d+) GRÖN$")                  # Nordens ljus: "4/11 GRÖN" (laget.nordic_light)
_NUMLIKE = re.compile(r"^[\s\d.,:;+\-−–%/()]+$")           # datum, tid och tal skrivs i mono
_LABEL = (f"color:{MUTED};font-family:{FONT_UI};font-size:12px;font-weight:600;line-height:1.2;letter-spacing:.1em;"
          "text-transform:uppercase;overflow-wrap:anywhere;")


def _part_color(part: str, fallback: str) -> str:
    """Färgen för en del av ett staplat värde: laget._color på sista ordet (ordlistan som redan finns)."""
    try:
        from odin.laget import _color
        c = _color((part.split() or [""])[-1])
    except Exception:
        return fallback
    return fallback if c == DIM else c


def _sign_color(sign, color: str) -> str:
    try:
        s = float(sign)
    except (TypeError, ValueError):
        return color
    return GREEN if s > 0 else RED if s < 0 else color


def tile(label: str, value, color: str = ACCENT, sub: str = "", sign=None, kind: str = "num") -> str:
    """En ruta: etikett, värde, en dämpad rad. kind = 'num' (mono) eller 'word' (Space Grotesk, 16 px över 14 tecken).
    sign > 0 ger GREEN och < 0 RED — bara för förändringar (P/L, 1 mån), aldrig för priser; None behåller color.
    Ett värde med ' · ' staplas en del per rad, var och en i laget._color för sitt sista ord. Nordens 'n/m GRÖN'
    visas som 'n/m gröna' i rutans egen färg (bara visningen — ljuset och laget.row_text är oförändrade).
    None eller ett saknas-ord ('Saknas', 'DATA_GAP' …) blir det grå Saknas-chipet med underraden som förklaring."""
    col = _sign_color(sign, color)
    text = "" if value is None else str(value).strip()
    absent = _absent(value)
    rule = GREY if absent else col
    if absent:
        body = f'<div style="margin-top:6px;">{missing(sub)}</div>'
    elif " · " in text:
        parts = [p.strip() for p in text.split(" · ") if p.strip()]
        body = "".join(f'<div style="color:{_a(ink(_part_color(p, col)))};font-family:{FONT_UI};font-size:15px;'
                       f'font-weight:700;line-height:1.25;overflow-wrap:anywhere;">{_t(p)}</div>' for p in parts)
        body = f'<div style="margin-top:4px;">{body}</div>'
    elif _NORDEN.match(text):
        count = _NORDEN.match(text).group(1)
        body = (f'<div style="margin-top:4px;line-height:1.2;"><span style="color:{_a(col)};font-family:{FONT_MONO};'
                f'font-size:20px;font-weight:600;font-variant-numeric:tabular-nums;">{_t(count)}</span>'
                f'<span style="color:{MUTED};font-family:{FONT_UI};font-size:12px;margin-left:6px;">gröna</span></div>')
    elif kind == "word":
        size = 16 if len(text) > 14 else 20
        body = (f'<div style="color:{_a(ink(col) if size < 19 else col)};font-family:{FONT_UI};font-size:{size}px;'
                f'font-weight:700;line-height:1.2;margin-top:4px;overflow-wrap:anywhere;">{_t(text)}</div>')
    else:
        body = (f'<div style="color:{_a(col)};font-family:{FONT_MONO};font-size:20px;font-weight:600;line-height:1.2;'
                f'font-variant-numeric:tabular-nums;margin-top:4px;overflow-wrap:anywhere;">{_t(text)}</div>')
    sub_html = ""
    if sub and not absent:
        font = FONT_MONO if _NUMLIKE.match(str(sub)) else FONT_UI
        sub_html = (f'<div style="color:{MUTED};font-family:{font};font-size:12px;line-height:1.3;margin-top:2px;'
                    f'overflow-wrap:anywhere;">{_t(sub)}</div>')
    return (f'<div class="wolf-tile" style="background:{BG_ALT};border-radius:10px;border-left:3px solid {_a(rule)};'
            f'padding:10px 12px;min-height:64px;box-sizing:border-box;min-width:0;">'
            f'<div style="{_LABEL}">{_t(label)}</div>{body}{sub_html}</div>')


def light_grid(items) -> str:
    """Kapitalradens ljus som rutor — laget.capital_row:s [(rubrik, ord, under, färg)] oförändrade. Två kolumner
    upp till 1023 px, fyra från 1024 px (kolumnerna i ui/shell.parts_css). Ljusen slås aldrig ihop."""
    tiles = "".join(tile(t, w, c, s, kind="word") for t, w, s, c in (items or ()))
    return f'<div class="wolf-grid" style="display:grid;gap:8px;min-width:0;">{tiles}</div>'


def kpi_strip(items) -> str:
    """En rad rutor à 132 × 64 px som sveps i sidled (Thors läge, aktiens nyckeltal).
    items = [(etikett, värde, färg, underrad[, sign]), …] — samma ordning som thor.cards.kpi_row."""
    cells = []
    for it in items or ():
        label, value, *rest = it
        color = rest[0] if len(rest) > 0 else ACCENT
        sub = rest[1] if len(rest) > 1 else ""
        sign = rest[2] if len(rest) > 2 else None
        kind = "num" if _NUMLIKE.match(str(value if value is not None else "x")) else "word"
        cells.append(f'<div style="flex:0 0 132px;min-width:0;">{tile(label, value, color, sub, sign, kind)}</div>')
    return (f'<div class="wolf-kstrip" style="display:flex;flex-wrap:nowrap;gap:8px;overflow-x:auto;'
            f'overscroll-behavior-x:contain;scrollbar-width:none;min-width:0;">{"".join(cells)}</div>')


# ── Tryckkortet (§3.5.11) ────────────────────────────────────────────────────
def tap_card(key: str, html: str, label: str, on_click=None, args=(), kwargs=None, help: Optional[str] = None) -> bool:
    """Hela kortet är knappen: kortets HTML och en osynlig st.button ovanpå (samma nyckel, etikett och callback
    som förut — 'Öppna →', thor_open_*, home_buy_*). Chevronen › kommer från CSS:en, så kortets text är
    oförändrad. Ett tryckkort har inga andra interaktiva element. Returnerar knappens värde."""
    with st.container(key=f"tapc_{key}"):
        st.markdown(html, unsafe_allow_html=True)
        return st.button(label, key=key, on_click=on_click, args=args, kwargs=kwargs, help=help, width="stretch")


# ── Statuschip (§3.5.12) ─────────────────────────────────────────────────────
_CHIP = (f"display:inline-flex;align-items:center;height:22px;box-sizing:border-box;padding:0 8px;border-radius:6px;"
         f"font-family:{FONT_UI};font-size:12px;font-weight:700;line-height:1;letter-spacing:.06em;"
         f"text-transform:uppercase;white-space:nowrap;vertical-align:middle;")
MAIN_COLOR = {vocab.KOP: GREEN, vocab.SALJ: RED, vocab.MINSKA: AMBER, vocab.BEVAKA: AMBER, vocab.VAKSAM: AMBER,
              vocab.HALL: GREEN, vocab.NEJ: GREY}
CHECK_COLOR = {vocab.OK: GREEN, vocab.KOLLA: AMBER, vocab.NEJ: RED}
CHECK_MARK = {vocab.OK: "✓", vocab.KOLLA: "◌", vocab.NEJ: "✗"}


def _pill(text: str, fg: str, border: str, bg: str = "transparent", extra: str = "", title: str = "") -> str:
    tip = f' title="{_a(title)}"' if title else ""
    return (f'<span class="wolf-chip"{tip} style="{_CHIP}color:{_a(fg)};background:{_a(bg)};border:{border};{extra}">'
            f'{_t(text)}</span>')


def _styled(word: str, color: str, strike: bool = False) -> str:
    """Huvudordet i chiptabellens stil: KÖP och SÄLJ fyllda, resten konturer; VAKSAM med ●, HÅLL med ✓,
    NEJ i MUTED text (SPÄRRAD behåller överstrykningen)."""
    line = "text-decoration:line-through;" if strike else ""
    if word == vocab.KOP:
        return _pill(word, APP_BG, f"1px solid {_a(color)}", color, line)
    if word == vocab.SALJ:
        return _pill(word, "#ffffff", f"1px solid {_a(color)}", color, line)
    text = {vocab.VAKSAM: "● VAKSAM", vocab.HALL: "HÅLL ✓"}.get(word, word)
    width = "1.5px" if word == vocab.MINSKA else "1px"
    fg = MUTED if word == vocab.NEJ else ink(color)
    return _pill(text, fg, f"{width} solid {_a(color)}", "transparent", line)


def source_chip(word: str) -> str:
    """Källans eget ord (GOLDEN TICKET, EXIT, PASS) — grått, färgas aldrig om."""
    return _pill(word, MUTED, f"1px solid {BORDER}", BG_CARD)


# Tomt, None, NaN och källornas egna "data saknas"-varianter som ordlistan inte har (§3.5.12: 'DATA GAP', 'None')
# visas som Saknas — bara visningen; ui/vocab.py och lagrade värden är oförändrade
_MISSING_WORDS = frozenset({"", "NONE", "NAN", "N/A", "NA", "<NA>", "NAT", "DATA GAP", "NO DATA AVAILABLE"})


def _absent(value, context: str = vocab.VERDICT) -> bool:
    """True när värdet ska visas som Saknas-chipet: None, NaN, ett tomt/None-ord eller ordlistans saknas-ord."""
    if value is None:
        return True
    try:
        if value != value:                                  # float('nan'); pd.NA ger TypeError → faller igenom
            return True
    except (TypeError, ValueError):
        pass
    text = str(value)                                       # str först: bool(pd.NA) kastar i vocab._norm
    return vocab._norm(text) in _MISSING_WORDS or vocab.main(text, context) == vocab.SAKNAS


def missing(why: str = "") -> str:
    """Saknade data har ETT utseende: ett streckat grått 'Saknas'-chip med orsaken i tooltipen — aldrig en tyst
    0, 50, en röd 0.0 eller 'None'."""
    return _pill("Saknas", MUTED, f"1px dashed {GREY}", "transparent", "text-transform:none;letter-spacing:.02em;",
                 why)


def chip(word, context: str = vocab.VERDICT, why: str = "", source: bool = True) -> str:
    """Statuschipet för källans ord: ordlistans huvudord i sin färg (ui/vocab.pair) och källans ord som grått
    chip bredvid när det skiljer sig — [KÖP] [GOLDEN TICKET]. context = vocab.VERDICT, CHECK eller HOLDING
    (WATCH → VAKSAM). Ett ord ordlistan inte känner (NEUTRAL) visas som det är, grått. Saknas-ord blir
    missing(why). source=False döljer källans chip."""
    if _absent(word, context):
        return missing(why or ("" if word is None or vocab._norm(str(word)) in _MISSING_WORDS else str(word)))
    main, src = vocab.pair(word, context)
    mapped = vocab.main(word, context)
    if mapped is None:
        return _pill(main or "—", MUTED, f"1px solid {GREY}")
    if context == vocab.CHECK:
        col = CHECK_COLOR.get(mapped, GREY)
        head = _pill(f"{CHECK_MARK.get(mapped, '')} {mapped}".strip(), ink(col), f"1px solid {_a(col)}")
        src = "" if src in CHECK_MARK.values() else src
    else:
        strike = vocab._norm(src or word) == "SPÄRRAD"
        head = _styled(mapped, MAIN_COLOR.get(mapped, GREY), strike)
    return head + (" " + source_chip(src) if source and src else "")


# ── Listrader (§3.5.13) ──────────────────────────────────────────────────────
def action_row(word: str, colour: str, title: str, detail: str = "", meta: str = "", chips=(), jump: str = "") -> str:
    """En åtgärdsrad, minst 56 px: [ordet som chip 84 px][rubrik + detalj][meta]. word och colour är sidans egna
    (KÖP, SÄLJ, VAKSAM …); ett huvudord får chiptabellens stil. chips = färdiga chip-HTML (chip(), missing())
    efter rubriken. jump = id för kortet som nav-skriptet rullar till (data-jump, t.ex. 'pos-volv-b')."""
    main = vocab._norm(word)
    known = {vocab._norm(w): w for w in MAIN_COLOR}
    tag = _styled(known[main], colour) if main in known else _pill(word, ink(colour), f"1px solid {_a(colour)}")
    extra = "".join(f" {c}" for c in (chips or ()) if c)
    attr = f' data-jump="{_a(jump)}"' if jump else ""
    # 84 px, inte specens 76: "● VAKSAM" mäter 81 px i 12 px — samma kolumn i varje rad håller rubrikerna i linje
    return (f'<div class="wolf-row"{attr} style="display:grid;grid-template-columns:84px minmax(0,1fr) auto;'
            f'align-items:center;gap:8px 10px;min-height:56px;padding:8px 0;box-sizing:border-box;'
            f'border-bottom:1px solid {BORDER_FAINT};">'
            f'<div style="min-width:0;overflow:hidden;">{tag}</div>'
            f'<div style="min-width:0;"><div style="color:{TEXT};font-family:{FONT_UI};font-size:16px;font-weight:700;'
            f'line-height:1.3;overflow-wrap:anywhere;">{_t(title)}{extra}</div>'
            + (f'<div style="color:{MUTED};font-size:13px;line-height:1.4;margin-top:2px;">{_t(detail)}</div>'
               if detail else "")
            + "</div>"
            + f'<div style="color:{MUTED};font-family:{FONT_MONO};font-size:13px;white-space:nowrap;">{_t(meta)}</div>'
            + "</div>")


def link_rows(key: str):
    """En behållare (st-key-links_<key>) där tertiära knappar blir länkrader: hela bredden, 48 px, cyan › till
    höger, en tunn linje mellan. Bara här — aldrig en global omstyrning av tertiära knappar.

        with link_rows("home"):
            st.button("Hela listan i Thor", key="home_thor", type="tertiary", on_click=go, args=("thor",))"""
    return st.container(key=f"links_{key}")


# ── Laddar, tomt, saknas (§3.5.19) ───────────────────────────────────────────
def skeleton_html(n: int = 3, h: int = 96) -> str:
    """n skimrande kort à h px (animeringen i ui/shell.parts_css — står still med prefers-reduced-motion)."""
    one = (f'<div class="wolf-skel" style="height:{int(h)}px;border-radius:12px;background-color:{BG_CARD};">'
           f'</div>')
    return (f'<div class="wolf-skels" role="status" aria-label="Laddar" style="display:flex;flex-direction:column;'
            f'gap:8px;">{one * max(1, int(n))}</div>')


def skeleton(n: int = 3, h: int = 96):
    """Skelettet i en st.empty() medan data laddas första gången — anroparen tömmer platsen (slot.empty())
    när laddningen är klar. Visa det bara när datan inte redan ligger i sessionen."""
    slot = st.empty()
    slot.markdown(skeleton_html(n, h), unsafe_allow_html=True)
    return slot


def empty_state(text: str, button: str = "", key: Optional[str] = None, on_click=None, args=()) -> bool:
    """Tom lista, sagt vänligt och centrerat i en streckad ruta — med högst en knapp (button = etiketten).
    Returnerar knappens värde (False utan knapp)."""
    st.markdown(
        f'<div class="wolf-empty" style="background:{BG_CARD};border:1px dashed {BORDER};border-radius:12px;'
        f'padding:16px;margin:4px 0 8px;color:{MUTED};font-size:14px;line-height:1.45;text-align:center;">'
        f'{_t(text)}</div>',
        unsafe_allow_html=True)
    if button:
        return st.button(button, key=key, on_click=on_click, args=args)
    return False


def card(title: str, body: str, color: str = ACCENT) -> str:
    """Navigations-/infokort: rubrik (16 px) och en rad text (13 px NOTE; body är HTML) med valfri 3 px
    accentlinje (color='' utan linje)."""
    rule = f"border-left:3px solid {_a(color)};" if color else ""
    return (f'<div class="wolf-card" style="background:{BG_CARD};border:1px solid {BORDER_FAINT};{rule}'
            f'border-radius:12px;padding:12px 14px;min-width:0;">'
            f'<div style="font-family:{FONT_UI};font-size:16px;font-weight:700;line-height:1.3;color:{TEXT};'
            f'margin-bottom:4px;">{escape(title)}</div>'
            f'<div style="font-size:13px;color:{NOTE};line-height:1.45;">{body}</div></div>')


# ── Äldre byggstenar (behåller API och text; typsnittsgolv 12 px) ────────────
def badge(text: str, color: str) -> str:
    """Färgad pill: bedömning, status, band."""
    return (f"<span style='background:{color}22;color:{color};border:1px solid {color};"
            f"border-radius:4px;padding:2px 8px;font-size:0.78rem;font-weight:700;'>"
            f"{escape(str(text))}</span>")


def verdict_box(score: str, label: str, note: str = "", color: str = DIM) -> str:
    """Rutan som sammanfattar en rad: poäng, bedömning, en förklarande rad."""
    return (f"<div style='border:1px solid {color}55;background:{color}0d;"
            f"border-radius:8px;padding:10px 14px;margin:10px 0;'>"
            f"<span style='color:{color};font-weight:700;font-size:1.05rem;'>{escape(str(score))}</span>"
            f"<span style='color:{color};font-weight:700;margin-left:12px;'>{escape(str(label))}</span>"
            + (f"<div style='color:{TEXT};font-size:0.8rem;margin-top:3px;'>{note}</div>" if note else "")
            + "</div>")


def kpi(label: str, value: str, color: str = ACCENT, caption: str = "") -> str:
    """Nyckeltalskort: etikett, stort värde, valfri bildtext."""
    return (f"<div style='background:{BG_ALT};border:1px solid {BORDER_FAINT};"
            f"border-left:3px solid {color};border-radius:10px;padding:12px 14px;'>"
            f"<div style='font-size:12px;letter-spacing:.1em;text-transform:uppercase;"
            f"color:{MUTED};margin-bottom:6px;'>{escape(label)}</div>"
            f"<div style='font-size:16px;font-weight:700;color:{color};'>{escape(str(value))}</div>"
            + (f"<div style='font-size:12px;color:{MUTED};margin-top:4px;'>{caption}</div>" if caption else "")
            + "</div>")


def note(text: str) -> None:
    """Förklaringsrad som syns på mobilen (ersätter st.caption) — NOTE, ljusare än st.caption."""
    st.markdown(f"<div class='wolf-note' style='color:{NOTE};font-size:0.76rem;line-height:1.45;"
                f"margin:2px 0 6px;'>{text}</div>", unsafe_allow_html=True)


def big_card(title: str, big: str, sub: str, color: str) -> str:
    """Stort siffrkort (Snabbkollens och Guld/Silver-flikens toppkort)."""
    return (f"<div style='border:1px solid {color}55;background:{color}0d;border-radius:10px;padding:12px 14px;"
            f"text-align:center;'><div style='font-size:12px;letter-spacing:.1em;color:{MUTED};'>{title}</div>"
            f"<div style='font-size:2rem;font-weight:900;color:{color};font-family:Courier New;'>{big}</div>"
            f"<div style='font-size:12px;color:{TEXT};margin-top:2px;'>{sub}</div></div>")


def confirm_delete(label: str, key: str, help: Optional[str] = None) -> bool:
    """Radera bakom en bekräftelse: en 🗑-popover med den riktiga knappen inuti.

    Returnerar True när användaren bekräftat. Ett klick på fel rad ska inte
    kosta ett bolag med tio ifyllda fält."""
    with st.popover(f"🗑 {label}", help=help or "Öppnar en bekräftelse — inget raderas direkt."):
        st.caption("Raderingen går inte att ångra i sessionen.")
        return st.button(f"Ja, {label.lower()}", key=f"{key}__confirm", type="primary")
