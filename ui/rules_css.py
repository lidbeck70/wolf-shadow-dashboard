"""
ui/rules_css.py — Regler & Guiders sid-CSS (LIDSKJALV §4.10, PR 23).

wolf_panel ritar ovtlyr/ui/rules_page i st.container(key="nav_rules_page") och lägger page_css() via
ui.shell.page_style(…, "rules") först i den. Varje regel börjar med .st-key-nav_rules_page, så inget når andra sidor
(Thors regelflik ritar samma _risk_box/_rule_row utanför behållaren och lämnas orörd). Bara utseende: texterna,
widgetarna, nycklarna och etiketterna i rules_page är desamma. Chip-raderna (rules_pb_*, sg_pick, rules_ref_view) och
den dolda linjen efter rules_sub ligger i skalet (ui/shell.nav_css), som rälsarna.

  · sidrubrikerna (h2): guld, vänsterställda, clamp(20px, 5.6vw, 26px), aldrig brutna mitt i ett ord
  · sektionsrubrikerna (_section_header_html): 20 px guld, aldrig brutna mitt i ett ord
  · breda tabeller (Flikguide, Snabbreferens) rullar i sin egen ruta — stMain rullar aldrig i sidled — och varje
    tabell som rullar visar det på telefonen (Flikguiden tonar ut, de smalare får en cyan kant bara när de rullar)
  · 12 px-golvet för sidans inline-storlekar under .75rem, och kodtexten i bildtexterna
  · RISKMODELL som rutnät 38/62 (etikett/värde) på telefonen
  · tryckytor: expandrar, sökfältet i ordlistan (48 px, 16 px text — iOS zoomar annars)
"""
from __future__ import annotations

from ui.tokens import ACCENT, APP_BG, GOLD, MUTED, NOTE, RED_INK

PAGE_KEY = "nav_rules_page"
SCOPE = f".st-key-{PAGE_KEY}"
# Varje inline-storlek under 12 px (.75rem) som rules_page skriver — tests/test_design_nav läser källan och kräver
# att listan täcker dem alla. Webbläsaren skriver om stilattributet med mellanslag ("font-size: 0.7rem"); båda
# stavningarna står med.
SMALL = ("0.55rem", "0.6rem", "0.62rem", "0.65rem", "0.66rem", "0.68rem", "0.7rem", "0.72rem", "0.74rem",
         "9px", "10px", "11px")
FADE = "linear-gradient(90deg, #000 calc(100% - 28px), transparent)"
H2_BOX = '[data-testid="stHeadingWithActionElements"]'
RISK_LABEL = 'span[style*="min-width: 140px"]'       # _risk_box: etiketten (Risk per affär …) i varje rad
SECTION = f'div[style*="border-bottom: 2px solid"] > {H2_BOX} > h3'     # _section_header_html
TABLE_BOX = '[data-testid="stMarkdownContainer"]:has(> table)'
NARROW_TABLE = f'{TABLE_BOX}:not(:has(> table th:nth-child(4)))'   # 2–3 spalter: Snabbreferens, Riskdoktrinen, larmen
EDGE_PX = 10                                                        # kantens bredd = täckets bredd


def _edge(side: str, deg: int) -> str:
    """Glöden vid en kant: 2 px cyan linje (ACCENT) som tonar ut över 10 px — fast i rutans kant (scroll)."""
    return (f"linear-gradient({deg}deg, rgba(0, 229, 255, .75) 0 2px, rgba(0, 229, 255, .25) 2px, "
            f"rgba(0, 229, 255, 0)) {side} center / {EDGE_PX}px 100% no-repeat scroll")


def _cover(side: str) -> str:
    """Täcket i sidans bakgrund: följer innehållet (local) och täcker glöden där tabellen tar slut."""
    return f"linear-gradient({APP_BG}, {APP_BG}) {side} center / {EDGE_PX}px 100% no-repeat local"


# Kanten som bara syns när tabellen är bredare än sin ruta (rullande skuggor, ingen JS): täckena ligger överst och
# följer innehållet, glöderna under dem sitter fast i rutans kanter. Ryms tabellen täcker de båda glöderna; rullar
# den syns glöden på den sida där det finns mer, och försvinner när du nått änden. Täcket är lika brett som glöden,
# så hela kanten syns redan när tabellen är 10 px för bred (Screeners på 375: 11 px).
EDGE = ", ".join((_cover("left"), _cover("right"), _edge("left", 90), _edge("right", 270)))


def floor_selector() -> str:
    """Alla element i sidan med en inline-storlek under 12 px, i båda stavningarna."""
    return ",\n".join(f'{SCOPE} [style*="font-size:{v}"], {SCOPE} [style*="font-size: {v}"]' for v in SMALL)


def page_css() -> str:
    """Sidans regler som EN sträng, alla under .st-key-nav_rules_page."""
    return f"""
/* ── LIDSKJALV · Regler & Guider (PR 23, §4.10) ───────────────── */
/* Sidrubrikerna: rules_page ritar h2 centrerat i cyan/magenta 36 px — här som page_header: guld, vänster, clamp,
   aldrig mitt i ett ord (HANDELSREGLER bröts i två rader på 390 px). Ingressen 13 px NOTE under rubriken. */
{SCOPE} div[style*="text-align: center"]:has(> {H2_BOX} > h2) {{ text-align: left !important;
    padding: 8px 0 4px !important; margin: 0 !important; }}
{SCOPE} h2 {{ color: {GOLD} !important; font-family: 'Space Grotesk', sans-serif !important;
    font-size: clamp(20px, 5.6vw, 26px) !important; font-weight: 700 !important; line-height: 1.15 !important;
    letter-spacing: .06em !important; text-align: left; margin: 0 !important; padding: 0 !important;
    word-break: keep-all !important; overflow-wrap: normal !important; hyphens: manual; text-wrap: balance; }}
{SCOPE} div:has(> {H2_BOX} > h2) > p {{ color: {NOTE} !important; font-size: 13px !important;
    line-height: 1.45 !important; max-width: 60ch; margin: 4px 0 0 !important; text-align: left; }}
{SCOPE} div:has(> {H2_BOX} > h2) > p span[style*="color: rgb(196, 69, 69)"] {{ color: {RED_INK} !important; }}

/* Sektionsrubrikerna: 20 px guld, hela ord; underraden MUTED */
{SCOPE} {SECTION} {{ color: {GOLD} !important; font-family: 'Space Grotesk', sans-serif !important;
    font-size: 20px !important; font-weight: 700 !important; line-height: 1.25 !important;
    letter-spacing: .04em !important; padding: 0 !important; }}
{SCOPE} div[style*="border-bottom: 2px solid"] > span {{ color: {MUTED} !important; }}
{SCOPE} h2, {SCOPE} h3 {{ word-break: keep-all !important; overflow-wrap: normal !important; hyphens: manual; }}

/* Breda tabeller (Flikguide 852 px på 390): tabellen rullar i sin egen ruta, stMain aldrig. Varje tabell som
   rullar har ett tecken på telefonen (§8.8): fyrkolumnstabellen (Flikguide, alltid bredare än skärmen) tonar ut vid
   högerkanten; de smalare får en cyan kant bara när de faktiskt är bredare än rutan (Screeners på 320–375,
   Datakällor på 320) — en tabell som ryms får ingen. */
{SCOPE} {TABLE_BOX} {{ overflow-x: auto; max-width: 100%;
    overscroll-behavior-x: contain; }}
{SCOPE} [data-testid="stMarkdownContainer"] > table {{ min-width: 100%; }}
@media (max-width: 640px) {{
    {SCOPE} [data-testid="stMarkdownContainer"]:has(> table th:nth-child(4)) {{
        -webkit-mask-image: {FADE}; mask-image: {FADE}; }}
    {SCOPE} {NARROW_TABLE} {{ background: {EDGE}; }}
    /* 6 px sidluft i stället för 8 i de smalare tabellerna: Screeners (366 px i en 358 px ruta) ryms då på 390 */
    {SCOPE} [data-testid="stMarkdownContainer"] > table:not(:has(th:nth-child(4))) th,
    {SCOPE} [data-testid="stMarkdownContainer"] > table:not(:has(th:nth-child(4))) td {{
        padding-left: 6px !important; padding-right: 6px !important; }}
    /* Flikguiden: utan golv krymper varje spalt till sitt längsta ord (rader upp till 301 px höga). Användningen
       får en skärmbredd, så en rad läses utan att svepa fram och tillbaka */
    {SCOPE} [data-testid="stMarkdownContainer"] > table:has(th:nth-child(4)) td:nth-child(1) {{
        white-space: normal !important; min-width: 120px; }}
    {SCOPE} [data-testid="stMarkdownContainer"] > table:has(th:nth-child(4)) td:nth-child(2) {{ min-width: 140px; }}
    {SCOPE} [data-testid="stMarkdownContainer"] > table:has(th:nth-child(4)) td:nth-child(3) {{
        min-width: min(320px, calc(100vw - 48px)); }}
}}

/* 12 px-golvet (§2.6): sidans inline-storlekar under .75rem, bildtexterna 13 px och deras kod 12 px */
{floor_selector()} {{ font-size: 12px !important; }}
{SCOPE} [data-testid="stCaptionContainer"], {SCOPE} [data-testid="stCaptionContainer"] p {{ font-size: 13px; }}
{SCOPE} [data-testid="stCaptionContainer"] code {{ font-size: 12px; }}

/* RISKMODELL på telefonen: etikett 38 % och värde 62 % i ett rutnät — etikettens min-width 140 px tryckte
   värdet till en smal spalt */
@media (max-width: 640px) {{
    {SCOPE} div:has(> {RISK_LABEL}) {{ display: grid !important;
        grid-template-columns: minmax(0, 38fr) minmax(0, 62fr); gap: 2px 12px !important; padding: 4px 0 !important; }}
    {SCOPE} {RISK_LABEL} {{ min-width: 0 !important; }}
}}

/* Chipen (HÅRD REGEL, nivå, stöd, rutinens status) bryts aldrig mitt i — hela chipet flyttar till nästa rad */
{SCOPE} span[style*="border-radius: 10px"][style*="font-size: 0.6rem"],
{SCOPE} span[style*="border-radius: 3px"][style*="font-size: 0.55rem"] {{ display: inline-block; white-space: nowrap;
    line-height: 1.4; }}
{SCOPE} div:has(> {H2_BOX} > h2) > p span {{ white-space: nowrap; }}
/* Linjen under strategiguidens chip-rad: raden bär sin egen luft, som efter rules_sub */
{SCOPE} .st-key-sg_pick + [data-testid="stElementContainer"]:has(hr) {{ display: none; }}

/* Fusklappens ruta HÅRDA REGLER står nu ovanför rutnätet (beslut 12): 12 px luft under den */
{SCOPE} div[style*="border: 1px solid rgb(196, 69, 69)"][style*="border-radius: 8px"] {{ margin-bottom: 12px; }}

/* Hårda regler (beslut 12): pekaren i listan hoppar till kortet överst */
{SCOPE} .rules-pointer:hover {{ background: rgba(0,229,255,.06) !important; }}
{SCOPE} .rules-pointer:active {{ transform: scale(.99); }}

/* Tryckytor (§2.4): expandrar 44 px, ordlistans sökfält 48 px med 16 px text */
{SCOPE} summary {{ min-height: 44px; box-sizing: border-box; }}
{SCOPE} .st-key-rules_gloss_q [data-testid="stTextInputRootElement"] {{ height: 48px; border-radius: 10px; }}
{SCOPE} .st-key-rules_gloss_q input {{ height: 100%; font-size: 16px !important; }}
{SCOPE} .st-key-rules_gloss_q [data-testid="stTextInputRootElement"]:focus-within {{ border-color: {ACCENT}; }}
"""
