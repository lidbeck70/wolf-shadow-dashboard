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
parts_css() är byggstenarna (PR 15, §3.5.7–§3.5.19) — det ui/components inte kan skriva i ett stilattribut:
  · knapparna: sekundär 44 px med radie 10, :active scale(.98), :disabled som ser avstängd ut, kompakta i huvud-
    och sparraden; ladda ned- och formulärknappar som sekundära. Primärknappen (PR 16, beslut 4) har temats cyan
    med mörk text och samma form; undantagen: osparat är bärnsten, "Ja, ta bort" är röd
  · huvudraden (head_<sida>), tryckkortets osynliga knapp (tapc_<nyckel>), länkraderna (links_<nyckel>),
    st.metric som rutor (kpis_<sida>), ljusrutnätets kolumner, nyckeltalsremsan och skelettets skimmer
nav_js() är nav-skriptet: centrerar valt chip, till toppen vid ny sida och vid tryck på aktiv flik / "▲ Till
toppen", data-jump och osparat-rutan. Det ritas i skalet med st.html (nav_script) och innehåller aldrig '<'.
Inga widgetar, nycklar eller etiketter ändras — allt här är CSS och ett skript.
"""
from __future__ import annotations

import json
import re

import streamlit as st

from ui import nav
from ui.nav import TABS_KEY
from ui.tokens import ACCENT, AMBER, APP_BG, BG, BG_ALT, BG_CARD, BORDER_FAINT, MUTED, NOTE, PURPLE, RED, TEXT, tint

SHELL_KEY = "wolf_shell"
PAGE_KEY_PREFIX = "pcss_"

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
{nav_css()}
{parts_css()}
{login_css()}"""


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
    margin: 0; padding: 0 16px; border-radius: 999px; border: 1px solid rgba(0,229,255,.25); background: {BG_CARD};
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


# ── Byggstenarna (PR 15) ─────────────────────────────────────────────────────
# Sekundärknappen: selektorn står ordagrant i ui/css.py (test_pr6) — här samma selektor senare i blocket, så att
# lika specifika regler vinner på ordningen och CYBERPUNK_CSS lämnas orörd (test_css_tabs läser bara den).
BTN_SECONDARY = '.stButton > button[kind="secondary"]'
BTN_ALL = ('.stButton > button', '[data-testid="stDownloadButton"] button', '[data-testid="stFormSubmitButton"] button')
BTN_AS_SECONDARY = (BTN_SECONDARY, '[data-testid="stDownloadButton"] button[kind="secondary"]',
                    '[data-testid="stFormSubmitButton"] button[kind="secondaryFormSubmit"]')
# Primärknappen (PR 16, beslut 4): temats cyan — och två undantag, valda på knapparnas egna nycklar (inga etiketter)
BTN_PRIMARY = ('.stButton > button[kind="primary"]', '[data-testid="stDownloadButton"] button[kind="primary"]',
               '[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]')
SAVE_DIRTY = '[class*="st-key-save_"] button[kind="primary"]'         # storage_ui.save_bar: primär = osparat
CONFIRM_DEL = ('[class*="__confirm"] button[kind="primary"]',           # components.confirm_delete: <nyckel>__confirm
               '[class*="st-key-watch_del_"] button[kind="primary"]')   # Övervakningens "Ja, ta bort …" (watch_ui)
PRIMARY_TEXT = APP_BG     # mörk text på cyan (13,1:1) och bärnsten (7,8:1) — vit text på cyan är 1,5:1
AMBER_HOVER, RED_HOVER = "#c0842c", "#b03c3c"   # mörkare vid hovring: mörk text 6,3:1, vit text 5,9:1
HEAD = '[class*="st-key-head_"]'          # components.head_row
TAPC = '[class*="st-key-tapc_"]'          # components.tap_card
LINKS = '[class*="st-key-links_"]'        # components.link_rows
KPIS = '[class*="st-key-kpis_"]'          # st.metric som rutor (sidornas kpis_<sida>-behållare)
SAVE_ROW = '[class*="st-key-save_"][class*="_row"]'    # storage_ui.save_bar: sparraden på en rad
# Knappar vars etikett tar knappens storlek (Streamlit sätter 14 px på etikettens markdown-ruta)
BTN_INHERIT = BTN_AS_SECONDARY + BTN_PRIMARY + (f"{HEAD} .stButton > button", f"{SAVE_ROW} .stButton > button")


def _sel(sels, suffix: str = "") -> str:
    return ",\n".join(f"{s}{suffix}" for s in sels)


def parts_css() -> str:
    """Byggstenarnas regler (§3.5.7–§3.5.19) som EN sträng: knapparna, huvudraden, tryckkortet, länkraderna,
    st.metric-rutorna, ljusrutnätet, nyckeltalsremsan och skelettet. Utseendet i övrigt bor i stilattributen
    som ui/components skriver; här står bara det som kräver mediafrågor, :has(), pseudoelement eller animering."""
    return f"""
/* ── LIDSKJALV · byggstenarna (PR 15) ────────────────────────── */
/* Knappar (§3.5.16): sekundär = 44 px, radie 10, cyan kant och svag cyan yta, 14 px versaler .06em. Ladda ned- och
   formulärknappar ser likadana ut när de inte är primära. Primärknappen: se nedan (PR 16). */
{_sel(BTN_AS_SECONDARY)} {{ min-height: 44px; padding: 0 16px; border-radius: 10px; border: 1px solid {ACCENT};
    background: rgba(0,229,255,.08); color: {ACCENT}; font-family: 'Space Grotesk', sans-serif; font-size: 14px;
    font-weight: 600; letter-spacing: .06em; text-transform: uppercase; box-shadow: none; text-shadow: none;
    transition: transform .12s ease, background-color .12s ease; }}
{_sel(BTN_INHERIT, ' [data-testid="stMarkdownContainer"]')},
{_sel(BTN_INHERIT, " p")} {{ font-size: inherit; font-weight: inherit; letter-spacing: inherit; }}
{_sel(BTN_AS_SECONDARY, ":hover")} {{ background: rgba(0,229,255,.16); border-color: {ACCENT}; color: {ACCENT};
    box-shadow: none; text-shadow: none; }}
{_sel(BTN_AS_SECONDARY, ":focus-visible")} {{ box-shadow: 0 0 0 2px rgba(0,229,255,.5); }}
/* Primärknappen (beslut 4, PR 16): fyllningen är temats cyan (primaryColor), formen sekundärens och texten MÖRK —
   Streamlits vita text på cyan är 1,5:1. Streamlit mörkar fyllningen vid hovring och fokus; texten står kvar. */
button[kind="primary"]:not(:disabled), button[kind="primaryFormSubmit"]:not(:disabled) {{ color: {PRIMARY_TEXT}; }}
{_sel(BTN_PRIMARY)} {{ min-height: 44px; padding: 0 16px; border-radius: 10px; font-family: 'Space Grotesk', sans-serif;
    font-size: 14px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; text-shadow: none;
    transition: transform .12s ease, background-color .12s ease; }}
{_sel(BTN_PRIMARY, ":not(:disabled)")} {{ color: {PRIMARY_TEXT}; }}
/* Avstängd primär som avstängd sekundär (Streamlits egen är grå text på genomskinligt — osynlig vid opacitet .45) */
{_sel(BTN_PRIMARY, ":disabled")},
{_sel(BTN_PRIMARY, ":disabled:hover")} {{ border-color: rgba(0,229,255,.25); background: rgba(0,229,255,.08);
    color: {ACCENT}; }}
/* Osparat (§3.5.16–17): sparradens primärknapp i bärnsten, som "⚠️ Du har osparade ändringar." bredvid — osparat ser
   aldrig ut som "Lägg till" */
{SAVE_DIRTY}:not(:disabled) {{ background: {AMBER}; border-color: {AMBER}; color: {PRIMARY_TEXT}; }}
{SAVE_DIRTY}:not(:disabled):hover {{ background: {AMBER_HOVER}; border-color: {AMBER_HOVER}; }}
{SAVE_DIRTY}:focus-visible {{ box-shadow: 0 0 0 2px {tint(AMBER, .5)}; }}
/* Bekräfta radering (§3.5.18): "Ja, ta bort" är röd med vit text (4,9:1) */
{_sel(CONFIRM_DEL, ":not(:disabled)")} {{ background: {RED}; border-color: {RED}; color: #ffffff; }}
{_sel(CONFIRM_DEL, ":not(:disabled):hover")} {{ background: {RED_HOVER}; border-color: {RED_HOVER}; }}
{_sel(CONFIRM_DEL, ":focus-visible")} {{ box-shadow: 0 0 0 2px {tint(RED, .5)}; }}
/* Widgetar som Streamlit fyller med primärfärgen (PR 16) — där cyan text eller vit text på cyan inte syns:
   talfältets − och + (CYBERPUNK_CSS:s cyan ikon på cyan vid hovring), flervalets valda etiketter och datumväljarens
   valda dag (vit text, 1,5:1) och skjutreglagets värde (Streamlit skriver det i primärfärgen — på CYBERPUNK_CSS:s
   cyan ruta försvann det; rutan bort, värdet står i cyan ovanför tummen som i Streamlit) */
[data-testid="stNumberInputContainer"] button:hover:enabled,
[data-testid="stNumberInputContainer"] button:focus:enabled {{ color: {PRIMARY_TEXT} !important; }}
.stMultiSelect [data-baseweb="tag"], .stMultiSelect [data-baseweb="tag"] * {{ color: {PRIMARY_TEXT}; }}
[data-baseweb="calendar"] [role="gridcell"][aria-label^="Selected"] {{ color: {PRIMARY_TEXT}; }}
.stSlider [data-testid="stSliderThumbValue"],
.stSlider [data-testid="stSliderTickBar"] [data-testid="stMarkdownContainer"] {{ background: none !important; }}
/* Tryck: 120 ms scale(.98) */
{_sel(BTN_ALL, ":active")} {{ transform: scale(.98); }}
/* Avstängd ser avstängd ut (förut såg den aktiv ut): halv opacitet, svag kant, ingen glöd */
{_sel(BTN_ALL, ":disabled")} {{ opacity: .45; cursor: not-allowed; box-shadow: none; transform: none; }}
{_sel(BTN_AS_SECONDARY, ":disabled")},
{_sel(BTN_AS_SECONDARY, ":disabled:hover")} {{ border-color: rgba(0,229,255,.25); background: rgba(0,229,255,.08);
    color: {ACCENT}; box-shadow: none; }}
/* Kompakta knappar i huvudraden och sparraden: 44 px, 13 px */
{HEAD} .stButton > button, {SAVE_ROW} .stButton > button {{ min-height: 44px; padding: 0 12px; font-size: 13px;
    letter-spacing: .04em; }}

/* Huvudraden (§3.5.7): rubriken tar raden, knapparna till höger — och bryts högerställda till nästa rad när de
   inte ryms bredvid. Rubrikens bas är dess egen bredd (minst 180 px): då flyttar knappen ned i stället för att
   rubriken bryts mellan emojin och ordet ("🥇" ovanför "RÅVARUHÄVSTÅNG" på 390 px) */
{HEAD} {{ flex-wrap: wrap; gap: 8px; row-gap: 4px; justify-content: flex-end; }}
{HEAD} > [data-testid="stElementContainer"]:has(.wolf-head) {{ flex: 1 1 auto; min-width: min(180px, 100%);
    width: auto; }}
{HEAD} > :not(:has(.wolf-head)) {{ flex: 0 0 auto; width: auto; }}

/* Sparraden (§3.5.17): knappen och raden bredvid på en rad */
{SAVE_ROW} {{ flex-wrap: wrap; gap: 8px 12px; }}
{SAVE_ROW} > [data-testid="stElementContainer"] {{ width: auto; min-width: 0; }}
/* Markdown-rutans -1rem (för p-marginalen) gör texten 1 px hög i en rad — då centreras den under knappens mitt */
{HEAD} [data-testid="stMarkdownContainer"], {SAVE_ROW} [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}

/* Sektionsrubriken (§3.5.8) — ui/css.section_title ritar class="section-title": samma utseende som section() */
.section-title {{ font: 600 12px/1.2 'Space Grotesk', sans-serif; letter-spacing: .12em; color: {MUTED};
    border-bottom: 1px solid rgba(0,229,255,.15); padding-bottom: 6px; margin: 20px 0 8px; }}

/* Tryckkortet (§3.5.11): hela kortet är knappen — en osynlig st.button ovanpå, chevronen › från CSS:en */
{TAPC} {{ position: relative; gap: 0; }}
{TAPC} > [data-testid="stElementContainer"]:has(.stButton) {{ position: absolute; inset: 0; z-index: 2;
    width: auto !important; margin: 0; }}
{TAPC} .stButton, {TAPC} .stButton button {{ width: 100%; height: 100%; min-height: 0; opacity: 0; padding: 0;
    margin: 0; cursor: pointer; }}
{TAPC}:has(button:focus-visible) {{ outline: 2px solid {ACCENT}; outline-offset: 2px; border-radius: 12px; }}
{TAPC}::after {{ content: "›"; position: absolute; right: 14px; top: 50%; transform: translateY(-50%);
    color: {ACCENT}; font: 700 22px/1 'Space Grotesk', sans-serif; pointer-events: none; z-index: 3; }}
/* Markdown-rutans -1rem skulle låta kortets sista 16 px hänga utanför behållaren — och utanför knappen */
{TAPC} [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
{TAPC} [data-testid="stMarkdownContainer"] > div {{ padding-right: 36px !important; transition: transform .12s ease; }}
{TAPC}:active [data-testid="stMarkdownContainer"] > div {{ transform: scale(.99); background: {BG_ALT} !important; }}

/* Länkrader (§3.5.13): tertiära knappar i links_<nyckel> — hela bredden, 48 px, cyan › till höger */
{LINKS} {{ gap: 0; }}
{LINKS} > [data-testid="stElementContainer"], {LINKS} .stButton {{ width: 100% !important; }}
{LINKS} .stButton > button[kind="tertiary"] {{ position: relative; width: 100%; min-height: 48px;
    justify-content: flex-start; text-align: left; padding: 0 32px 0 2px; border-radius: 0;
    border-bottom: 1px solid {BORDER_FAINT}; color: {TEXT}; font: 600 15px/1.3 'Space Grotesk', sans-serif; }}
{LINKS} .stButton > button[kind="tertiary"] > div {{ justify-content: flex-start; }}
{LINKS} .stButton > button[kind="tertiary"] p {{ font: inherit; text-align: left; }}
{LINKS} .stButton > button[kind="tertiary"]::after {{ content: "›"; position: absolute; right: 8px; top: 50%;
    transform: translateY(-50%); color: {ACCENT}; font: 700 20px/1 'Space Grotesk', sans-serif; }}
{LINKS} .stButton > button[kind="tertiary"]:hover {{ color: {ACCENT}; background: rgba(0,229,255,.06); }}

/* st.metric som rutor (§3.5.10): kpis_<sida> är en horisontell behållare — 2 × 2 på telefonen, 4 i rad från
   1024 px; etikett 12 px MUTED, värde 20 px mono i TEXT (inte 28 px cyan, ingen ::before-linje) */
{KPIS} {{ flex-wrap: wrap; gap: 8px; }}
{KPIS} > [data-testid="stElementContainer"] {{ flex: 1 1 calc(50% - 4px); min-width: 0; width: auto !important; }}
{KPIS} [data-testid="stMetric"] {{ background: {BG_ALT} !important; border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 10px !important; padding: 10px 12px !important; min-height: 64px; box-sizing: border-box; }}
{KPIS} [data-testid="stMetric"]::before {{ content: none !important; display: none !important; }}
{KPIS} [data-testid="stMetricLabel"], {KPIS} [data-testid="stMetricLabel"] p {{ color: {MUTED} !important;
    font-family: 'Space Grotesk', sans-serif !important; font-size: 12px !important; font-weight: 600 !important;
    letter-spacing: .1em !important; text-transform: uppercase !important; }}
{KPIS} [data-testid="stMetricValue"], {KPIS} [data-testid="stMetricValue"] > div {{ color: {TEXT} !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 20px !important; font-weight: 600 !important;
    text-shadow: none !important; }}
/* Telefon: rutan är 114–149 px bred — Streamlit klipper värdet med '…' ("1,120,000 S…"). 16 px, och värde och
   etikett får radbrytas (SEK hamnar då på raden under) i stället för att klippas */
@media (max-width: 640px) {{
    {KPIS} [data-testid="stMetricValue"], {KPIS} [data-testid="stMetricValue"] * {{
        font-size: 16px !important; white-space: normal !important; overflow: visible !important;
        text-overflow: clip !important; line-height: 1.25 !important; }}
    {KPIS} [data-testid="stMetricLabel"], {KPIS} [data-testid="stMetricLabel"] * {{
        white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }}
}}

/* Ljusrutnätet (§3.5.10): två kolumner, fyra från 1024 px */
.wolf-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
@media (min-width: 1024px) {{
    {KPIS} > [data-testid="stElementContainer"] {{ flex-basis: calc(25% - 6px); }}
    .wolf-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
}}
/* Nyckeltalsremsan: en rad rutor som sveps, tonar ut vid högerkanten */
.wolf-kstrip {{ -webkit-mask-image: {FADE}; mask-image: {FADE}; }}
.wolf-kstrip::-webkit-scrollbar {{ display: none; }}

/* Skelettet (§3.5.19): skimrar 1.2 s medan data laddas första gången; står still med reducerad rörelse */
.wolf-skel {{ background-image: linear-gradient(90deg, {BG_CARD} 0%, {BG_ALT} 50%, {BG_CARD} 100%);
    background-size: 200% 100%; animation: wolf-shimmer 1.2s linear infinite; }}
@keyframes wolf-shimmer {{ from {{ background-position: 100% 0; }} to {{ background-position: -100% 0; }} }}
@media (prefers-reduced-motion: reduce) {{
    .wolf-skel {{ animation: none; }}
    {_sel(BTN_ALL, ":active")}, {TAPC}:active [data-testid="stMarkdownContainer"] > div {{ transform: none; }}
}}
"""


# ── Inloggningen och låset (PR 17, §4.1, §4.9) ───────────────────────────────
# Valda på widgetarnas egna nycklar (auth.py, portfolio/private_lock.py) — inga etiketter, inga nya widgetar.
LOGIN_CARD = ".st-key-login_card"         # auth.LOGIN_KEY: kolumnen, ett kort på datorn
LOCK_CARD = '[class*="st-key-lock_"]'     # private_lock.gate: lock_<nyckel> (holdings_pw, holdings_pw_alloc)
PW_FIELDS = (".st-key-login_pw", f"{LOCK_CARD} .stTextInput")
PW_BUTTONS = (".st-key-login_btn .stFormSubmitButton", f"{LOCK_CARD} .stFormSubmitButton")   # (0,2,1): slår 44 px
FIELD_LOGIN_H = 52                        # §3.3: fält 48, inloggningen och låset 52


def login_css() -> str:
    """Inloggningssidans och låskortets regler: kolumnen (kortet på datorn), fältet 52 px med 16 px JetBrains Mono
    (iOS zoomar in på fält under 16 px), knappen 52 px, ögat i fältet som 44 px tryckyta och inget Streamlit-huvud
    på inloggningen. Loggan, devisen och raden om kakan har egna klasser i auth.py (.arc-login-*)."""
    root = '[data-testid="stTextInputRootElement"]'
    return f"""
/* ── LIDSKJALV · inloggningen och låset (PR 17) ───────────────── */
/* Inget Streamlit-huvud på inloggningen, på någon bredd: ⋮ och körindikatorn skulle ligga över loggan */
body:has({LOGIN_CARD}) [data-testid="stHeader"] {{ display: none; }}
/* Kolumnen: centrerad, högst 360 px; på telefonen utan ram — på datorn ett kort */
{LOGIN_CARD} {{ max-width: 360px; width: 100%; margin: 44px auto 0; gap: 16px; box-sizing: border-box; }}
{LOGIN_CARD} .stMarkdown [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
{LOGIN_CARD} > [data-testid="stElementContainer"]:first-child {{ margin-bottom: 8px; }}
@media (min-width: 641px) {{
    {LOGIN_CARD} {{ margin-top: 64px; padding: 28px; background: {BG}; border: 1px solid rgba(0,229,255,.15);
        border-radius: 16px; }}
}}
/* Låskortet (§4.9): centrerat kort med rubriken, fältet och knappen */
{LOCK_CARD} {{ max-width: 420px; width: 100%; margin: 8px auto 0; padding: 20px 16px; gap: 12px;
    box-sizing: border-box; background: {BG_CARD}; border: 1px solid rgba(255,255,255,.08); border-radius: 12px; }}
{LOCK_CARD} h4 {{ text-align: center; padding: 0; font: 700 18px/1.3 'Space Grotesk', sans-serif; color: {TEXT}; }}
{LOCK_CARD} [data-testid="stHeaderActionElements"] {{ display: none; }}   /* ankarlänken drog rubriken ur mitten */
{LOCK_CARD} .stMarkdown [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
@media (min-width: 641px) {{
    {LOCK_CARD} {{ margin-top: 24px; padding: 24px; }}
}}
/* Formulären: tätare än Streamlits 16 px mellan etikett/fält och knapp */
{LOGIN_CARD} [data-testid="stForm"] [data-testid="stVerticalBlock"],
{LOCK_CARD} [data-testid="stForm"] [data-testid="stVerticalBlock"] {{ gap: 12px; }}
/* Etiketten: synlig, 13 px MUTED */
{_sel(PW_FIELDS, ' [data-testid="stWidgetLabel"]')},
{_sel(PW_FIELDS, ' [data-testid="stWidgetLabel"] p')} {{ color: {MUTED} !important;
    font: 500 13px/1.3 'Space Grotesk', sans-serif; letter-spacing: .02em; }}
/* Fältet: 52 px, radie 10, en kant (CYBERPUNK_CSS ger annars inputen en egen ram innanför), cyan vid fokus */
{_sel(PW_FIELDS, " " + root)} {{ height: {FIELD_LOGIN_H}px; border-radius: 10px !important; overflow: hidden;
    border: 1px solid rgba(0,229,255,.25) !important; background-color: {BG_ALT} !important; }}
{_sel(PW_FIELDS, " " + root + ":focus-within")} {{ border-color: {ACCENT} !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,.25); }}
{_sel(PW_FIELDS, " " + root + " input")},
{_sel(PW_FIELDS, " " + root + " input:focus")} {{ height: 100%; padding: 0 14px; border: 0 !important;
    border-radius: 0 !important; background: transparent !important; box-shadow: none !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 16px !important; letter-spacing: .04em; }}
{_sel(PW_FIELDS, " " + root + " input::placeholder")} {{ font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0; }}
/* Ögat (visa koden): hela fältets höjd och 44 px bred — tryckytan */
{_sel(PW_FIELDS, " " + root + " button")} {{ align-self: stretch; height: auto; min-width: 44px; width: 44px;
    margin: 0; padding: 0; justify-content: center; color: {MUTED}; }}
{_sel(PW_FIELDS, " " + root + " button:hover")} {{ color: {ACCENT}; background: rgba(0,229,255,.08); }}
/* "Press Enter to submit form" — engelska; Gå/Enter skickar ändå */
{_sel(PW_FIELDS, ' [data-testid="InputInstructions"]')} {{ display: none; }}
/* Knapparna: 52 px; LOGGA IN med .12em spärrning och 700 */
{_sel(PW_BUTTONS, " button")} {{ min-height: {FIELD_LOGIN_H}px; }}
{_sel(PW_BUTTONS, ' button [data-testid="stMarkdownContainer"]')},
{_sel(PW_BUTTONS, " button p")} {{ font-family: inherit; }}   /* etiketten i knappens Space Grotesk */
.st-key-login_btn button[kind="primaryFormSubmit"] {{ font-weight: 700; letter-spacing: .12em; }}
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
