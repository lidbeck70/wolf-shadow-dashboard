"""
ui/theme.py
===========
Global branding and UI helpers for Nordic Arc Systems.

This module is the single source of truth for colours, typography, and
reusable layout primitives.  Import from here instead of ui.css so that
a palette change in PALETTE propagates everywhere automatically.

Public API
----------
    PALETTE          — colour dict
    inject_css()     — panelens stilblock via skalet (ui/shell.render_shell)
    render_header()  — varumärkesraden; returnerar platsen som fill_bar() fyller om sist i main()
    fill_bar(slot)   — fyller om raden (datum, osparat, låset, pulsen ur sessionen)
    render_footer()  — sidfoten med årets år och "▲ Till toppen"
    section_title()  — sektionsrubriken (omslag kring ui/components.section)
    card()           — styled content card rendered via st.markdown
"""

from __future__ import annotations

import os
import base64
from datetime import datetime
from html import escape
from typing import Optional

import streamlit as st

# Re-export the CSS injector and legacy helpers so callers can use
# `from ui.theme import inject_css, section_title` without touching ui.css.
from ui.css import inject_css, color_score, color_entry, tab_not_found  # noqa: F401

# ---------------------------------------------------------------------------
# Palette — single source of truth
# ---------------------------------------------------------------------------

PALETTE: dict = {
    # Backgrounds
    "bg":           "#05070A",
    "bg2":          "#1A1F25",
    "bg3":          "#0D1117",
    "surface":      "#0D1117",

    # Arc Cyan (replaces gold as primary accent)
    "gold":         "#00E5FF",
    "gold_dim":     "#00A8BF",
    "gold_muted":   "rgba(0,229,255,0.40)",
    "gold_faint":   "rgba(0,229,255,0.10)",

    # Aurora Purple (replaces silver)
    "silver":       "#B400FF",
    "ice_blue":     "#8A00CC",
    "ice_faint":    "rgba(180,0,255,0.12)",

    # Semantik (ui/tokens.py är källan). "red" och "amber" var samma orange,
    # så fel och varning gick inte att skilja åt.
    "green":        "#2d8a4e",
    "red":          "#c44545",
    "amber":        "#d4943a",
    "ember":        "#FF6B3D",
    "text":         "#E8EDF2",
    "text_dim":     "#6B7280",

    # Borders
    "border":       "rgba(0,229,255,0.15)",
    "border_hi":    "rgba(0,229,255,0.35)",
    "border_ice":   "rgba(180,0,255,0.20)",
}

_P = PALETTE  # shorthand used internally

# ---------------------------------------------------------------------------
# section_title — re-implemented here so it is the canonical version
# ---------------------------------------------------------------------------

def section_title(text: str, icon: str = "") -> None:
    """Sektionsrubriken — tunt omslag kring ui/components.section (12 px versaler i MUTED, LIDSKJALV §3.5.8)."""
    from ui.components import section
    st.markdown(section(f"{icon}  {text}" if icon else str(text)), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# card — styled content block
# ---------------------------------------------------------------------------

def card(
    content: str,
    title: Optional[str] = None,
    border_color: str = "",
    accent_color: Optional[str] = None,
    bg: str = "",
    padding: str = "16px 18px",
    margin_bottom: str = "12px",
) -> None:
    """Render an HTML content block styled as a card."""
    _bc = border_color or _P["border"]
    _bg = bg or _P["bg2"]
    _left = (
        f"border-left:3px solid {accent_color};"
        if accent_color else ""
    )
    _title_html = (
        f'<div style="color:{_P["gold"]};font-size:0.72rem;text-transform:uppercase;'
        f'letter-spacing:0.12em;margin-bottom:8px;font-weight:700;">{title}</div>'
        if title else ""
    )
    st.markdown(
        f'<div style="background:{_bg};border:1px solid {_bc};{_left}'
        f'border-radius:8px;padding:{padding};margin-bottom:{margin_bottom};">'
        f'{_title_html}{content}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# SVG icon — triangle with eye (Arc Cyan)
# ---------------------------------------------------------------------------

_SVG_ICON = (
    '<svg width="48" height="48" viewBox="0 0 48 48" fill="none" '
    'xmlns="http://www.w3.org/2000/svg">'
    '<polygon points="24,4 44,44 4,44" fill="none" stroke="#00E5FF" stroke-width="2"/>'
    '<ellipse cx="24" cy="32" rx="6" ry="4" fill="none" stroke="#00E5FF" stroke-width="1.5"/>'
    '<circle cx="24" cy="32" r="2" fill="#00E5FF"/>'
    '</svg>'
)

# Bannern ritas inte längre på sidorna (varumärkesraden, PR 13) men finns kvar: css.wolf_banner läser den.
_BANNER_B64: dict = {}          # läses och kodas EN gång per process, inte per omritning


def _banner_b64() -> str:
    """Base64 för assets/banner.jpg (165 KB JPEG). Tidigare lästes och
    kodades en 2 MB PNG vid varje rerun — ~2,7 MB extra per klick."""
    if "b64" not in _BANNER_B64:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "banner.jpg")
        try:
            with open(path, "rb") as bf:
                _BANNER_B64["b64"] = base64.b64encode(bf.read()).decode()
        except OSError:
            _BANNER_B64["b64"] = ""
    return _BANNER_B64["b64"]


# ---------------------------------------------------------------------------
# render_header — varumärkesraden (LIDSKJALV §3.5.2)
# ---------------------------------------------------------------------------
# 44 px på telefonen, 52 px på datorn, i stället för bannern på 260 px (inloggningen behåller sin logga).
# render_header() lämnar en st.empty()-plats som fylls direkt; fill_bar(plats) fyller om den sist i main(),
# när sidan räknat klart — så att osparat-chipet och pulsen stämmer. Utseendet står i ui/shell.shell_css().
# Raden ritas med st.markdown: inline-SVG överlever där, inte i st.html.

_WEEKDAYS = ("mån", "tis", "ons", "tor", "fre", "lör", "sön")
_MONTHS = ("jan", "feb", "mar", "apr", "maj", "jun", "jul", "aug", "sep", "okt", "nov", "dec")

# Pulsen läser BARA det som redan ligger i sessionen — I DAGs data och Lägets. Ingen laddare anropas här; utan
# data i sessionen ritas ingen puls. Vilken av dem (puls_source):
#   1. sidans egen när I DAG eller Läget visas — raden får aldrig säga emot kapitalraden under sig (dess legend)
#   2. annars den som laddades senast — en ny dict under nyckeln är en ny load() (I DAG läser om sig själv efter
#      RELOAD_MIN minuter eller en ny dag, Läget på 📡); minnet ligger i PULS_SEEN, bara i sessionen
#   3. annars I DAGs, som har en färskhetsregel, före Lägets
PULS_KEYS = ("home_today", "laget_data")
PULS_PAGE = {("home",): "home_today", ("odin", "Läget"): "laget_data"}
PULS_SEEN = "_wolf_puls_seen"


def _icon(size: int) -> str:
    return _SVG_ICON.replace('width="48" height="48"', f'width="{size}" height="{size}"')


def _today_label(now: Optional[datetime] = None) -> str:
    """'lör 21 nov' — svensk tid, samma klocka som I DAG."""
    if now is None:
        try:
            from tabs.home_today import now_se
            now = now_se()
        except Exception:
            now = datetime.now()
    return f"{_WEEKDAYS[now.weekday()]} {now.day} {_MONTHS[now.month - 1]}"


def _session_dict(state, key: str):
    try:
        d = state.get(key)
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def mark_loads(state) -> dict:
    """{nyckel: ordningsnummer} för pulsens källor — högst = senast laddad. laget.py och home.py lägger en NY dict
    under sin nyckel vid varje load(), så ett annat objekt än sist betyder en omläsning. Minnet (objektet och
    numret) ligger under PULS_SEEN i sessionen: inget sparas, inget laddas."""
    try:
        seen = dict(state.get(PULS_SEEN) or {})
    except Exception:
        seen = {}
    n = max([v[1] for v in seen.values() if isinstance(v, tuple)] or [0])
    changed = False
    for key in reversed(PULS_KEYS):                 # laddas båda på en gång vinner den som står först i PULS_KEYS
        d = _session_dict(state, key)
        old = seen.get(key)
        if d is None:
            if old is not None:
                seen.pop(key)
                changed = True
        elif not isinstance(old, tuple) or old[0] is not d:
            n += 1
            seen[key] = (d, n)
            changed = True
    if changed:
        try:
            state[PULS_SEEN] = seen
        except Exception:
            pass
    return {k: v[1] for k, v in seen.items() if isinstance(v, tuple)}


def page_source(state) -> Optional[str]:
    """Sessionsnyckeln vars kapitalrad sidan som visas ritar (I DAG, Läget) — annars None."""
    try:
        from ui import nav
        segs = list(nav.current(state))
    except Exception:
        return None
    for prefix, key in PULS_PAGE.items():
        if tuple(segs[:len(prefix)]) == prefix:
            return key
    return None


def puls_source(state, seen=None, prefer=None) -> Optional[str]:
    """Vilken nyckel pulsen läser: prefer (sidans egen) om den finns, annars den senast laddade enligt seen
    (mark_loads), annars PULS_KEYS ordning. None när ingen av dem ligger i sessionen."""
    present = [k for k in PULS_KEYS if _session_dict(state, k) is not None]
    if not present:
        return None
    if prefer in present:
        return prefer
    seen = seen or {}
    return max(present, key=lambda k: (seen.get(k, 0), -PULS_KEYS.index(k)))


def puls_items(state, seen=None, prefer=None) -> list:
    """Kapitalradens fyra ljus [(rubrik, ord, under, färg)] ur sessionens data — [] när inget är laddat.
    Bara laget.capital_row på det som redan finns (puls_source väljer); aldrig ett fel, aldrig en laddning."""
    key = puls_source(state, seen, prefer)
    if key is None:
        return []
    d = _session_dict(state, key)
    try:
        from odin import laget
        return laget.capital_row(d.get("alloc"), d.get("regime"), d.get("risk"), d.get("board"))
    except Exception:
        return []


def _puls_html(items: list) -> str:
    """Fyra prickar à 8 px, var och en i sitt eget ljus färg — de slås aldrig ihop. Orden står i aria-label."""
    if not items:
        return ""
    from odin import laget
    words = escape(laget.row_text(items), quote=True)
    dots = "".join(f'<span style="background:{escape(str(c), quote=True)};"></span>' for _t, _w, _s, c in items)
    return f'<span class="wolf-puls" role="img" aria-label="{words}" title="{words}">{dots}</span>'


def bar_html(items=(), dirty=(), lock: str = "", today: str = "", snack: bool = False) -> str:
    """Varumärkesradens HTML. items = puls_items(), dirty = storage.dirty_stores(),
    lock = '🔒'/'🔓' ('' när låset inte är konfigurerat). snack lägger osparat-rutan (telefon) efter raden."""
    right = ""
    if today:
        right += f'<span class="wolf-date">{escape(today)}</span>'
    if dirty:
        n = len(dirty)
        names = escape("Osparade ändringar i: " + ", ".join(str(d) for d in dirty), quote=True)
        right += f'<span class="wolf-dirty" title="{names}">● {n} {"osparad" if n == 1 else "osparade"}</span>'
    if lock:
        tip = "Innehaven är upplåsta" if lock == "🔓" else "Innehaven är låsta"
        right += f'<span class="wolf-lock" title="{tip}">{escape(lock)}</span>'
    right += _puls_html(list(items))
    html = (
        '<div class="wolf-bar" id="nas-top">'
        f'<div class="wolf-brand">{_icon(24)}'
        '<span class="wolf-word">NORDIC <b>ARC</b><span class="wolf-wide"> SYSTEMS</span></span>'
        "<span class=\"wolf-tag wolf-wide\">See What the Market Can't.</span></div>"
        f'<div class="wolf-meta">{right}</div>'
        '</div>'
    )
    if snack and dirty:                                     # en ruta, hur många lager som än är osparade
        html += '<div class="wolf-snack" role="status">⚠️ Osparade ändringar · Spara ↑</div>'
    return html


def _lock_glyph() -> str:
    try:
        from portfolio import private_lock
        if not private_lock.configured():
            return ""
        return "🔓" if private_lock.unlocked() else "🔒"
    except Exception:
        return ""


def _dirty() -> list:
    try:
        import storage
        return list(storage.dirty_stores())
    except Exception:
        return []


def fill_bar(slot) -> None:
    """Fyller (om) varumärkesradens plats med det sessionen vet just nu. Läser bara — det enda den skriver är
    pulsens minne i sessionen (mark_loads); inget sparas och inget laddas."""
    try:
        dirty = _dirty()
        state = st.session_state
        items = puls_items(state, mark_loads(state), page_source(state))
        html = bar_html(items, dirty, _lock_glyph(), _today_label(), snack=bool(dirty))
    except Exception:
        html = bar_html()                                   # märket utan meta hellre än en trasig sida
    slot.markdown(html, unsafe_allow_html=True)


def render_header():
    """Varumärkesraden överst på varje sida. Returnerar platsen (st.empty) som fill_bar() fyller om sist."""
    slot = st.empty()
    fill_bar(slot)
    return slot


# ---------------------------------------------------------------------------
# render_footer — sidfoten (LIDSKJALV §3.5.21)
# ---------------------------------------------------------------------------

def footer_html(year: Optional[int] = None) -> str:
    """En rad i MUTED med årets år, och "▲ Till toppen" (ankare till märkesraden — fungerar utan skript)."""
    year = year or datetime.now().year
    return (
        '<div class="wolf-foot">'
        f"<div class=\"wolf-foot-line\">Nordic Arc Systems © {year} · See What the Market Can't.</div>"
        '<a class="wolf-top" href="#nas-top">▲ Till toppen</a>'
        '</div>'
    )


def render_footer() -> None:
    """Sidfoten längst ned (efter lagringsraden)."""
    st.markdown(footer_html(), unsafe_allow_html=True)
