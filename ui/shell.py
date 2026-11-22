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
nav_css() är navigeringen (PR 14, §3.5.3–§3.5.6) — genererad ur ui/nav.py, inga handskrivna fliknamn eller nycklar:
  · L1 flikraden: klistrar överst på alla bredder; på telefonen sex celler med emoji över ord (etiketterna ur
    nav.TOP, med Safari-reserven först), på datorn dagens rad oförändrad; strålen medan Streamlit räknar
  · L2 chip-rad och L3 understruken rad ur nav.STATE_KEY efter djup; Thors pills-rader rullar i sidled
  · L4 sidornas inre flikar: 44 px, radbryts på telefonen; brödsmulan ovanför L3-raden
nav_js() är nav-skriptet: centrerar valt chip, till toppen vid ny sida och vid tryck på aktiv flik / "▲ Till
toppen", data-jump och osparat-rutan. Det ritas i skalet med st.html (nav_script) och innehåller aldrig '<'.
Inga widgetar, nycklar eller etiketter ändras — allt här är CSS och ett skript.
"""
from __future__ import annotations

import json
import re

import streamlit as st

from ui import nav
from ui.components import NOTE
from ui.nav import TABS_KEY
from ui.tokens import ACCENT, AMBER, BG, PURPLE, TEXT

SHELL_KEY = "wolf_shell"
PAGE_KEY_PREFIX = "pcss_"

MUTED = "#a8a397"         # etiketter och meta, 8:1 mot #05070A (flyttar till ui/tokens.MUTED i PR 15)
APP_BG = "#05070A"        # .stApp — den klistrade flikradens bakgrund (ui/tokens.APP_BG i PR 15)
CARD_BG = "#14141e"       # chipens yta (ui/tokens.BG_CARD)

# ── Navigeringens selektorer (Streamlit 1.59, React Aria) ────────────────────
# Toppflikarna: div.stTabs.st-key-main_tabs > div > div[role=tablist] > div[role=tab] > stMarkdownContainer > p.
# Barnkombinatorerna gör att sidornas inre flikar (som ligger INNE i toppflikarnas paneler) aldrig träffas.
TABLIST = f'.st-key-{TABS_KEY} > div > [role="tablist"]'
TAB = f'{TABLIST} > [role="tab"]'
TAB_PANEL = f'.st-key-{TABS_KEY} > div > [data-testid="stTabPanel"]'
INNER_LIST = f'.stTabs:not(.st-key-{TABS_KEY}) > div > [role="tablist"]'
INNER_TAB = f'{INNER_LIST} > [role="tab"]'
# Radioknappen: label[stRadioOption] > span(input) + div > div > (div > div = ringen) + stMarkdownContainer.
GROUP = '[role="radiogroup"]'
OPTION = '[role="radiogroup"] > label[data-testid="stRadioOption"]'
RADIO_DOT = '[data-testid="stRadioOption"] > div > div > div:first-child'   # ordagrant ur prototypen (proto.py)
PILLS = ("thor_filters", "thor_markets")                                   # Thors st.pills-rader
FADE = "linear-gradient(90deg, #000 calc(100% - 28px), transparent)"      # tona ut vid högerkanten

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
    .wolf-snack {{ cursor: pointer; -webkit-tap-highlight-color: transparent; }}
}}
{nav_css()}"""


# ── Navigeringen (PR 14) ─────────────────────────────────────────────────────
def _q(text: str) -> str:
    """En CSS-sträng inom citattecken."""
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _each(keys, suffix: str = "") -> str:
    """'.st-key-a <suffix>, .st-key-b <suffix>' — en selektor per widgetnyckel."""
    return ",\n".join(f".st-key-{k} {suffix}".rstrip() for k in keys)


def rail_keys() -> tuple:
    """(L2, L3): radionycklarna ur nav.STATE_KEY efter djup — inget '/' i sökvägen ger L2 (chip-raden), ett eller
    fler ger L3 (den understrukna raden). En ny gren i trädet får sin rad utan att någon rör CSS:en."""
    l2 = [k for path, k in nav.STATE_KEY.items() if path.count("/") == 0]
    l3 = [k for path, k in nav.STATE_KEY.items() if path.count("/") >= 1]
    return l2, l3


def rail_selector() -> str:
    """Alla rälsars radiogrupper — nav-skriptet centrerar det valda chipet i var och en."""
    l2, l3 = rail_keys()
    return ", ".join(f".st-key-{k} {GROUP}" for k in l2 + l3)


def tab_label_css() -> str:
    """Telefonens etiketter ur nav.TOP: emojin i ::before, ordet i ::after. Först en vanlig content (Safari före
    17.4 förstår inte alt-texten och kastar hela deklarationen), sedan samma med '/ ""' så att skärmläsaren bara
    läser den riktiga etiketten (p, storlek 0) och inte allt två gånger."""
    out = []
    for i, (_key, label) in enumerate(nav.TOP, start=1):
        emo, word = label.split(" ", 1)
        t = f"{TAB}:nth-child({i}) p"
        out.append(f"{t}::before {{ content: {_q(emo)}; content: {_q(emo)} / \"\"; }}")
        out.append(f"{t}::after {{ content: {_q(word)}; content: {_q(word)} / \"\"; }}")
    return "\n".join(out)


def nav_css() -> str:
    """Flikraden, rälsarna, de inre flikarna och brödsmulan som EN sträng (§3.5.3–§3.5.6)."""
    l2, l3 = rail_keys()
    rails = l2 + l3
    pills = ",\n".join(f'.st-key-{k} [role="toolbar"]' for k in PILLS)
    return f"""
/* ── LIDSKJALV · navigeringen (PR 14) ────────────────────────── */
/* L1 flikraden klistrar överst på alla bredder — regeln sitter på själva tablist, vars förälder är lika hög som
   flikarnas innehåll. Dagens baslinje ligger kvar; en hårlinje under skiljer raden från innehållet som rullar. */
{TABLIST} {{ position: sticky; top: 0; z-index: 50; background: {APP_BG};
    box-shadow: inset 0 -2px 0 rgba(0,229,255,0.2), 0 1px 0 rgba(0,229,255,.15); }}
/* Strålen: en tunn cyan svepning i radens nederkant medan Streamlit räknar (körindikatorn finns i DOM:en) */
{TABLIST}::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px; opacity: 0;
    pointer-events: none; background: linear-gradient(90deg, transparent, {ACCENT}, {PURPLE}, transparent);
    background-size: 50% 100%; background-repeat: no-repeat; background-position: -100% 0; }}
body:has([data-testid="stStatusWidgetRunningIcon"], [data-testid="stStatusWidgetRunningManIcon"]) {TABLIST}::after {{
    opacity: 1; animation: wolf-scan 1.2s linear infinite; }}
@keyframes wolf-scan {{ from {{ background-position: -100% 0; }} to {{ background-position: 200% 0; }} }}
@media (prefers-reduced-motion: reduce) {{
    {TABLIST}::after {{ animation: none !important; background-size: 100% 100%; background-position: 0 0; }}
}}

/* Telefon (≤640 px): sex celler, emoji över ord, alla synliga — ingen rullning i sidled */
@media (max-width: 640px) {{
    {TABLIST} {{ gap: 0; overflow-x: visible; margin: 0 -1rem; padding: 0 1rem; }}
    {TAB} {{ flex: 1 1 auto; min-width: 52px; height: 56px; padding: 0 3px; border-radius: 0;
        -webkit-tap-highlight-color: transparent; }}
    {TAB} p {{ font-size: 0 !important; line-height: 0; text-align: center; }}
    {TAB} p::before {{ display: block; font-size: 20px; line-height: 22px; filter: grayscale(1); opacity: .7; }}
    {TAB} p::after {{ display: block; font: 600 12px/14px 'Space Grotesk', sans-serif; letter-spacing: .01em;
        text-transform: uppercase; margin-top: 3px; white-space: nowrap; color: rgba(232,228,220,.7); }}
    {TAB}[aria-selected="true"] p::before {{ filter: none; opacity: 1; }}
    {TAB}[aria-selected="true"] p::after {{ color: {ACCENT}; font-weight: 700; }}
    {TAB}:active {{ background-color: rgba(0,229,255,.22); }}
    {tab_label_css()}
    /* Panelens 16 px-luft blir 8 px; en chip-rad ligger an mot flikraden */
    {TAB_PANEL} {{ padding-top: 8px; }}
    {TAB_PANEL}:has({", ".join(f".st-key-{k}" for k in l2)}) {{ padding-top: 0; }}
}}
/* Smal telefon (320 px): cellerna får krympa under 52 px så att alla sex ryms */
@media (max-width: 359px) {{
    {TAB} {{ min-width: 0; }}
}}

/* Rälsarna fyller raden (radions behållare är annars bara så bred som innehållet) */
{_each(rails)} {{ width: 100%; max-width: 100%; }}
/* Gruppen är de dolda radioknapparnas omslutande block: React Arias VisuallyHidden-span är position: absolute, och
   utan en positionerad förälder under stMain klipps de inte av rälsens overflow — chipen som rullat ut till höger
   gjorde då stMain bredare än skärmen och hela sidan gick att svepa i sidled (alla bredder, även 641–1023 px) */
{_each(rails, GROUP)} {{ position: relative; }}
/* L2 chip-raden (sökväg utan '/' i nav.STATE_KEY): samma st.radio, sveps med tummen, valt chip i cyan */
{_each(l2, GROUP)} {{ display: flex; flex-wrap: nowrap; gap: 8px; overflow-x: auto; overscroll-behavior-x: contain;
    scrollbar-width: none; scroll-snap-type: x proximity; margin: 0 -1rem; padding: 4px 1rem 0;
    -webkit-mask-image: {FADE}; mask-image: {FADE}; }}
{_each(l2, OPTION)} {{ flex: 0 0 auto; display: flex; align-items: center; box-sizing: border-box; min-height: 44px;
    margin: 0; padding: 0 16px; border-radius: 999px; border: 1px solid rgba(0,229,255,.25); background: {CARD_BG};
    scroll-snap-align: center; cursor: pointer; -webkit-tap-highlight-color: transparent;
    transition: transform .12s ease; }}
{_each(l2, OPTION + " p")} {{ font: 600 14px/1 'Space Grotesk', sans-serif; color: {TEXT} !important;
    white-space: nowrap; margin: 0; }}
{_each(l2, OPTION + "[data-selected]")} {{ background: rgba(0,229,255,.14); border-color: {ACCENT}; }}
{_each(l2, OPTION + "[data-selected] p")} {{ color: {ACCENT} !important; }}
{_each(l2, OPTION + ":active")} {{ transform: scale(.97); }}

/* L3 den understrukna raden (sökväg med '/'): textrad med cyan streck under det valda */
{_each(l3, GROUP)} {{ display: flex; flex-wrap: nowrap; gap: 0; overflow-x: auto; overscroll-behavior-x: contain;
    scrollbar-width: none; margin: 0 -1rem; padding: 0 1rem; box-shadow: inset 0 -1px 0 rgba(255,255,255,.08);
    -webkit-mask-image: {FADE}; mask-image: {FADE}; }}
{_each(l3, OPTION)} {{ flex: 0 0 auto; display: flex; align-items: center; box-sizing: border-box; min-height: 44px;
    margin: 0; padding: 0 12px; border: 0; border-radius: 0; background: transparent; cursor: pointer;
    box-shadow: inset 0 -1px 0 rgba(255,255,255,.08); -webkit-tap-highlight-color: transparent; }}
{_each(l3, OPTION + " p")} {{ font: 500 14px/1 'Space Grotesk', sans-serif; color: {MUTED} !important;
    white-space: nowrap; margin: 0; }}
{_each(l3, OPTION + "[data-selected]")} {{ box-shadow: inset 0 -2px 0 {ACCENT}; }}
{_each(l3, OPTION + "[data-selected] p")} {{ color: {TEXT} !important; font-weight: 700; }}

/* Radioringen bort i båda raderna, och ingen rullningslist */
{_each(rails, RADIO_DOT)} {{ display: none; }}
{_each(rails, GROUP + "::-webkit-scrollbar")} {{ display: none; }}

/* Thors pills-rader (Filter, Marknad): en rad som rullar och tonar ut, i stället för att radbrytas */
{_each(PILLS)},
{_each(PILLS, '[data-testid="stButtonGroup"]')} {{ width: 100%; max-width: 100%; }}
{pills} {{ width: 100%; max-width: 100%; flex-wrap: nowrap; overflow-x: auto; overscroll-behavior-x: contain; scrollbar-width: none;
    -webkit-mask-image: {FADE}; mask-image: {FADE}; }}
{",".join(f'.st-key-{k} [role="toolbar"] > button' for k in PILLS)} {{ flex: 0 0 auto; }}
{",".join(f'.st-key-{k} [role="toolbar"]::-webkit-scrollbar' for k in PILLS)} {{ display: none; }}

@media (min-width: 641px) and (max-width: 1023px) {{
    {_each(rails, GROUP)} {{ margin: 0 -24px; padding-left: 24px; padding-right: 24px; }}
}}
/* Dator: rälsarna radbryts i stället för att rulla */
@media (min-width: 1024px) {{
    {_each(rails, GROUP)} {{ flex-wrap: wrap; overflow-x: visible; margin: 0; padding-left: 0; padding-right: 0;
        -webkit-mask-image: none; mask-image: none; }}
}}

/* L4 sidornas inre flikar (thor_tabs, odin_gold_tabs, …): 44 px, understrykning i stället för indikatorn */
{INNER_LIST} {{ gap: 0; }}
{INNER_TAB} {{ height: 44px; padding: 0 12px; border-radius: 0; background-color: transparent; color: {MUTED};
    font-family: 'Space Grotesk', sans-serif; font-size: 14px; font-weight: 600; }}
{INNER_TAB}:hover {{ color: {ACCENT}; background-color: rgba(0,229,255,.06); }}
{INNER_TAB}[aria-selected="true"] {{ color: {TEXT}; font-weight: 700; background-color: transparent;
    box-shadow: inset 0 -2px 0 {ACCENT}; }}
{INNER_TAB}[data-focus-visible] {{ box-shadow: inset 0 0 0 2px rgba(0,229,255,.5); }}
{INNER_TAB} .react-aria-SelectionIndicator {{ display: none; }}
@media (max-width: 640px) {{
    {INNER_LIST} {{ flex-wrap: wrap; overflow-x: visible; }}
    {INNER_TAB} {{ flex: 1 1 auto; }}
}}

/* Stilbehållarnas omslag (stLayoutWrapper) tar inte heller plats — i flikarnas paneler (glipa 16 px) gav ett tomt
   omslag annars en glipa för mycket där page_style ritades */
[data-testid="stLayoutWrapper"]:has(> .st-key-{SHELL_KEY}),
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-{PAGE_KEY_PREFIX}"]) {{ display: none; }}

/* Brödsmulan (§3.5.6): rälsens rubrik vid djup 3, an mot L2-raden ovanför och L3-raden under */
[data-testid="stElementContainer"]:has(.wolf-crumb) {{ margin-top: -1rem; }}
.wolf-crumb {{ height: 20px; font: 500 12px/20px 'JetBrains Mono', monospace; color: {MUTED};
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
"""


def route_key(state, params=None) -> str:
    """Ruttens nyckel för nav-skriptet: 'nav-slug|thor-ticker'. Ändras den har användaren bytt sida — då öppnar
    sidan överst. Tickern hör bara till Thor (analysen är en egen sida); innan Thor har läst ?t= gäller länken."""
    segs = nav.current(state)
    ticker = ""
    if segs[0] == "thor":
        try:
            ticker = state["thor_ticker"] if "thor_ticker" in state else (params or {}).get("t")
        except Exception:
            ticker = ""
    return f"{nav.slug(segs)}|{str(ticker or '').strip()}"


# Skriptets kropp — får aldrig innehålla '<' (DOMPurify kastar ett skript med '<' följt av en bokstav).
_NAV_JS = """(function () {
  var d = document, w = window, route = __ROUTE__, RAILS = __RAILS__;
  var TOP = '.st-key-main_tabs > div > [role="tablist"] > [role="tab"][aria-selected="true"], a.wolf-top';
  var still = w.matchMedia && w.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var B = still ? 'auto' : 'smooth';
  function sc() { return d.querySelector('[data-testid="stMain"]') || d.scrollingElement; }
  function centre(g) {
    var s = g.querySelector('[data-selected]'); if (!s || g.__wolfSel === s) return;
    var a = g.getBoundingClientRect(), b = s.getBoundingClientRect(); if (!a.width) return;
    var first = !g.__wolfSel; g.__wolfSel = s;
    var off = (b.left - a.left) - (a.width - b.width) / 2;
    if (Math.abs(off) > 4) g.scrollBy({left: off, behavior: first ? 'auto' : B});
  }
  function pass() { d.querySelectorAll(RAILS).forEach(centre); }
  var N = w.__wolfNav;
  if (!N) {
    N = w.__wolfNav = {route: null, busy: false};
    new MutationObserver(function () {
      if (N.busy) return; N.busy = true;
      w.requestAnimationFrame(function () { N.busy = false; pass(); });
    }).observe(d.body, {subtree: true, childList: true, attributes: true, attributeFilter: ['data-selected']});
    d.addEventListener('click', function (e) {
      var t = e.target; if (!t || !t.closest) return;
      if (t.closest(TOP)) { e.preventDefault(); sc().scrollTo({top: 0, behavior: B}); return; }
      var j = t.closest('[data-jump]');
      if (j) { var el = d.getElementById(j.getAttribute('data-jump'));
        if (el) el.scrollIntoView({block: 'start', behavior: B}); }
      if (t.closest('.wolf-snack')) {
        var s = d.querySelector('[class*="st-key-save_"] button[kind="primary"]');
        if (s) s.scrollIntoView({block: 'center', behavior: B}); }
    }, true);
  }
  if (N.route !== route) { if (N.route !== null) sc().scrollTo({top: 0}); N.route = route; }
  pass();
})();"""


def _js_str(value: str) -> str:
    """En JS-sträng utan '<' (tickern är fritext)."""
    return json.dumps(str(value)).replace("<", "\\u003c")


def nav_js(route: str) -> str:
    """Nav-skriptet med rutten och rälsarnas selektorer inbakade. Rutten står i texten, så elementet ritas om
    (och skriptet körs igen) bara när sidan byts."""
    body = _NAV_JS.replace("__ROUTE__", _js_str(route)).replace("__RAILS__", _js_str(rail_selector()))
    return f"<script>{body}</script>"


def style_block() -> str:
    """Hela stilblocket: CYBERPUNK_CSS först, skalet efter, PWA-taggarna sist."""
    from ui.css import CYBERPUNK_CSS
    return f"<style>\n{_bare(CYBERPUNK_CSS)}\n{shell_css()}\n</style>\n{PWA_METAS}"


def render_shell():
    """Skriver skalet en gång per körning och returnerar dess dolda behållare (nav_script skriver i den). Anropas
    det igen (en gammal sida med egen inject_css()) händer inget och None returneras: nyckeln finns redan."""
    from streamlit.errors import StreamlitDuplicateElementKey
    try:
        box = st.container(key=SHELL_KEY)
    except StreamlitDuplicateElementKey:
        return None
    with box:
        st.markdown(style_block(), unsafe_allow_html=True)
    return box


def nav_script(box) -> None:
    """Nav-skriptet i skalets dolda behållare — anropas efter djuplänken, så att rutten är sidan som ritas."""
    if box is None:
        return
    with box:
        st.html(nav_js(route_key(st.session_state, st.query_params)), unsafe_allow_javascript=True)


def page_scope(key: str) -> str:
    """Selektorprefix för sidans egna element: allt som ritas EFTER page_style(css, key) i samma block (behållaren
    ligger i en stLayoutWrapper). Ramens rälsar ovanför och sidfoten utanför fliken träffas aldrig."""
    box = f".st-key-{PAGE_KEY_PREFIX}{key}"
    return f':is([data-testid="stLayoutWrapper"]:has(> {box}), {box}) ~ *'


def page_style(css: str, key: str) -> None:
    """Sidans egen CSS i en dold behållare (st-key-pcss_<key>) — ett <style>, ingen 16 px-glipa i sidan."""
    with st.container(key=f"{PAGE_KEY_PREFIX}{key}"):
        st.markdown(f"<style>\n{_bare(css)}\n</style>", unsafe_allow_html=True)
