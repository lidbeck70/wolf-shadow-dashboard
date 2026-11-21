"""
ui/shell.py — LIDSKJALVS ram (designspecen §3.5.1): ETT stilblock per körning.

render_shell() skriver en dold st.container(key="wolf_shell") med en enda st.markdown: <style> med panelens
CYBERPUNK_CSS och skalets egna regler EFTER den (lika specifika regler vinner då på ordningen), plus PWA-taggarna.
Stilar gäller även i en display:none-behållare, så skalet tar ingen plats och lämnar ingen glipa.

shell_css() är ramen:
  · de dolda stilbehållarna (wolf_shell och page_style:s pcss_<sida>)
  · Streamlits huvud: dolt på telefonen (≤640 px), genomskinligt med flytande ⋮ på datorn (värdens knappar dolda)
  · ramens marginaler, 1280 px innehållsbredd och glipa 0 i toppblocket (märkesraden ligger an mot flikraden;
    efter flikraden får elementen tillbaka sin 1rem)
  · varumärkesraden, sidfoten med "▲ Till toppen" och osparat-rutan (telefon, högst en)
Flikraden, rälsarna, strålen och nav-skriptet kommer i PR 14 — ingen regel här rör flikar, radio eller sidornas
knappar.
"""
from __future__ import annotations

import re

import streamlit as st

from ui.components import NOTE
from ui.nav import TABS_KEY
from ui.tokens import ACCENT, AMBER, BG, PURPLE, TEXT

SHELL_KEY = "wolf_shell"
PAGE_KEY_PREFIX = "pcss_"

MUTED = "#a8a397"         # etiketter och meta, 8:1 mot #05070A (flyttar till ui/tokens.MUTED i PR 15)

# Hemskärmsikonen och statusraden på iPad/iPhone — förut en egen markdown överst i wolf_panel.py
PWA_METAS = (
    '<meta name="apple-mobile-web-app-capable" content="yes">\n'
    '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">\n'
    '<meta name="apple-mobile-web-app-title" content="Nordic Arc">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    '<meta name="theme-color" content="#05070A">\n'
    '<link rel="apple-touch-icon" href="https://em-content.zobj.net/source/apple/391/trident_1f531.png">\n'
)

_TAG = re.compile(r"</?style[^>]*>", re.I)


def _bare(css: str) -> str:
    """CSS utan egna <style>-taggar (CYBERPUNK_CSS och sidornas block har dem)."""
    return _TAG.sub("", css or "").strip()


def shell_css() -> str:
    """Ramens regler som EN sträng — läggs efter CYBERPUNK_CSS i samma <style>."""
    return f"""
/* ── LIDSKJALV · ramen (PR 13) ───────────────────────────────── */
/* Stilbehållarna syns aldrig — stilarna i dem gäller ändå */
.st-key-{SHELL_KEY}, [class*="st-key-{PAGE_KEY_PREFIX}"] {{ display: none; }}

/* Toppblocket: ingen glipa mellan dolda block, märkesraden och flikraden */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {{ gap: 0; }}
/* …men efter flikraden samma 1rem som förut: varje markdown-ruta drar sig 1rem uppåt, och utan glipan skrevs
   lagringsraden (hr, "Senast sparad", "Osparade ändringar") och sidfoten ovanpå varandra */
.st-key-{TABS_KEY} ~ [data-testid="stElementContainer"] {{ margin-top: 1rem; }}

/* Telefon: Streamlits huvud bort (⋮ skulle täcka MER), smal ram */
@media (max-width: 640px) {{
    [data-testid="stHeader"] {{ display: none; }}
    [data-testid="stMainBlockContainer"] {{ padding: .25rem 1rem 5rem; }}
}}
/* Dator: huvudet genomskinligt och utan höjd, ⋮ flyter mitt i märkesradens högra 48 px. Verktygsradens
   osynliga ruta släpper igenom klick och hovring — bara dess knappar tar emot dem. */
@media (min-width: 641px) {{
    [data-testid="stHeader"] {{ background: transparent; height: 0; min-height: 0; pointer-events: none; }}
    [data-testid="stToolbar"] {{ position: fixed; top: 12px; right: 12px; left: auto; width: auto;
        height: auto; pointer-events: none; }}
    [data-testid="stToolbar"] button, [data-testid="stToolbar"] a {{ pointer-events: auto; }}
    /* Värdens knappar (Streamlit Cloud: Share/Fork/stjärna/GitHub) ritas i verktygsraden även i viewer-läge och
       skulle täcka datumet och pulsen — bara ⋮ ryms i radens högra reserv */
    [data-testid="stToolbarActions"] {{ display: none; }}
    /* Körindikatorn (Running/Stop, Connecting) står till vänster om ⋮ medan en körning pågår — i radens egen
       färg, så den täcker datumet och pulsen rent i stället för att skrivas ovanpå dem */
    [data-testid="stStatusWidget"] {{ background: {BG}; box-shadow: -12px 0 12px {BG}; border-radius: 8px; }}
    [data-testid="stMainBlockContainer"] {{ padding: .25rem 24px 5rem; }}
}}
@media (min-width: 1024px) {{
    [data-testid="stToolbar"] {{ top: 16px; right: max(12px, calc((100vw - 1344px) / 2 + 12px)); }}
    [data-testid="stMainBlockContainer"] {{ padding: .25rem 32px 5rem; max-width: calc(1280px + 64px);
        margin: 0 auto; }}
}}

/* ── Varumärkesraden (§3.5.2) ── */
/* Markdown-rutan drar sig 1rem uppåt (för p-marginalen); med glipa 0 skulle flikraden då täcka raden */
[data-testid="stMarkdownContainer"]:has(> .wolf-bar) {{ margin-bottom: 0; }}
.wolf-bar {{ position: relative; display: flex; align-items: center; gap: 8px; height: 44px;
    margin: 0 -1rem; padding: 0 1rem; box-sizing: border-box; overflow: hidden;
    background: {BG}; box-shadow: inset 0 -1px 0 rgba(0,229,255,.15); }}
.wolf-bar::before {{ content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, {ACCENT}, {PURPLE}); }}
.wolf-brand {{ display: flex; align-items: center; gap: 8px; min-width: 0; flex: 1 1 auto;
    white-space: nowrap; overflow: hidden; }}
.wolf-brand svg {{ width: 24px; height: 24px; flex: 0 0 24px; filter: drop-shadow(0 0 6px rgba(0,229,255,.5)); }}
.wolf-word {{ font: 700 13px/1 'Space Grotesk', sans-serif; letter-spacing: .22em; color: {TEXT}; }}
.wolf-word b {{ font-weight: 700; color: {ACCENT}; text-shadow: 0 0 12px rgba(0,229,255,.6); }}
.wolf-wide {{ display: none; }}
.wolf-tag {{ font: italic 400 12px/1.2 'Space Grotesk', sans-serif; color: {NOTE}; letter-spacing: .02em;
    margin-left: 4px; }}
.wolf-meta {{ display: flex; align-items: center; gap: 8px; flex: 0 0 auto; margin-left: auto; }}
.wolf-date {{ font: 500 12px/1 'JetBrains Mono', monospace; color: {MUTED}; white-space: nowrap; }}
.wolf-dirty {{ font: 600 12px/1 'Space Grotesk', sans-serif; color: {AMBER}; background: rgba(212,148,58,.14);
    border-radius: 999px; padding: 5px 8px; white-space: nowrap; }}
.wolf-lock {{ font-size: 14px; line-height: 1; }}
.wolf-puls {{ display: inline-flex; align-items: center; gap: 4px; }}
.wolf-puls span {{ display: block; width: 8px; height: 8px; border-radius: 50%; }}
@media (max-width: 400px) {{
    .wolf-bar:has(.wolf-dirty) .wolf-date {{ display: none; }}
}}
@media (min-width: 641px) {{
    .wolf-bar {{ margin: 0 -24px; padding: 0 72px 0 24px; }}
}}
@media (min-width: 1024px) {{
    .wolf-bar {{ height: 52px; margin: 0 -32px; padding: 0 80px 0 32px; }}
    .wolf-wide {{ display: inline; }}
}}

/* ── Sidfoten (§3.5.21) ── */
.wolf-foot {{ display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;
    margin-top: 32px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,.06); }}
.wolf-foot-line {{ font: 400 12px/1.4 'Space Grotesk', sans-serif; color: {MUTED}; }}
a.wolf-top {{ display: inline-flex; align-items: center; justify-content: center; min-width: 44px;
    min-height: 44px; padding: 0 12px; border-radius: 10px; color: {ACCENT}; text-decoration: none;
    font: 600 13px/1 'Space Grotesk', sans-serif; letter-spacing: .04em; }}
a.wolf-top:hover, a.wolf-top:focus-visible {{ background: rgba(0,229,255,.08); color: {ACCENT}; }}

/* ── Osparat-rutan (§3.5.17): bara telefon, och bara en — den ritas i märkesraden ── */
.wolf-snack {{ display: none; }}
@media (max-width: 640px) {{
    .wolf-snack {{ display: flex; align-items: center; justify-content: center; position: fixed; z-index: 60;
        left: 16px; right: 16px; bottom: calc(12px + env(safe-area-inset-bottom)); height: 44px;
        box-sizing: border-box; border-radius: 12px; border: 1px solid {AMBER}; color: {AMBER};
        background: linear-gradient(rgba(212,148,58,.16), rgba(212,148,58,.16)), rgba(12,12,18,.92);
        -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
        font: 600 14px/1 'Space Grotesk', sans-serif; }}
    body:has(.wolf-snack) [data-testid="stMainBlockContainer"] {{ padding-bottom: calc(5rem + 72px); }}
    body:has(.wolf-snack) [data-testid="stToastContainer"] {{ transform: translateY(-56px); }}
}}
"""


def style_block() -> str:
    """Hela stilblocket: CYBERPUNK_CSS först, skalet efter, PWA-taggarna sist."""
    from ui.css import CYBERPUNK_CSS
    return f"<style>\n{_bare(CYBERPUNK_CSS)}\n{shell_css()}\n</style>\n{PWA_METAS}"


def render_shell() -> None:
    """Skriver skalet en gång per körning. Anropas det igen (en gammal sida med egen inject_css()) händer inget:
    nyckeln finns redan i körningen."""
    from streamlit.errors import StreamlitDuplicateElementKey
    try:
        box = st.container(key=SHELL_KEY)
    except StreamlitDuplicateElementKey:
        return
    with box:
        st.markdown(style_block(), unsafe_allow_html=True)


def page_style(css: str, key: str) -> None:
    """Sidans egen CSS i en dold behållare (st-key-pcss_<key>) — ett <style>, ingen 16 px-glipa i sidan."""
    with st.container(key=f"{PAGE_KEY_PREFIX}{key}"):
        st.markdown(f"<style>\n{_bare(css)}\n</style>", unsafe_allow_html=True)
