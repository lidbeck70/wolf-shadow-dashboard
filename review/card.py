"""
review/card.py — kortens stöd i arken: raden först i kortet ('12/34 fält ifyllda' och "Saknas för köpbeslut") och
fokusväljaren överst (review.focus).

Kortets rubrik är densamma som förr: en expander som byter etikett får tillbaka sitt läge från koden (stängd
utanför fokusläget), så en räknare i rubriken fällde ihop kortet vid nästan varje ändring. Räknaren står därför i
kortet. Raden ritas i en platshållare som skapas först i kortet (slot) och fylls sist (fill) — då har kortets
widgets skrivit den här körningens värden. De platta arken körs inte om efter en ändring, så en rad som räknades
före widgetarna låg ett steg efter.

Allt härleds ur fältkatalogen (review.catalog.progress) och arkens egna grindar (review.missing.missing_for) — ingen
regel räknas här. Inget får fälla ett kort: ett fel ger ingen rad och ingen väljare.

LIDSKJALV PR 21 (§4.7–§4.8): domen överst i kortet. fill(..., verdict_html=…) lägger arkets egen dom (Scorecardets
mätare och KLAR FÖR KÖP / INGEN AFFÄR, Rick Rules poäng och bedömning, Royaltys signal) först i SAMMA markdown som
räknaren och "Saknas för köpbeslut" — ritad sist, med körningens värden. verdict_box bygger rutan av orden kortet redan
har; CSS (style) är arkens egna regler: korthuvuden 48 px, kryssrader 48 px, Scorecardets två positionsfält bredvid
varandra på telefonen och 12 px-golvet.
"""

from __future__ import annotations

from html import escape
from typing import Iterable, Optional

import streamlit as st

from ui.tokens import ACCENT, FONT_MONO, FONT_UI, NOTE, TEXT, ink, tint

MISSING_COLOR, DONE_COLOR = "#c44545", "#8a8578"
MAX_LABELS = 8

# Arkens sidor och kort (nycklarna): sidbehållarna i scorecard.py och producers.py, korten är nycklade expandrar
PAGES = (".st-key-sc_page", ".st-key-pr_page")
CARDS = ('[class*="st-key-sc_card_"]', '[class*="st-key-pr_card_"]', '[class*="st-key-ro_card_"]')
HEADS = (".st-key-head_scorecard", ".st-key-head_rule", ".st-key-head_royalty", ".st-key-head_producers")
_SMALL = ("0.55rem", "0.6rem", "0.62rem", "0.65rem", "0.66rem", "0.68rem", "0.7rem", "0.72rem", "0.74rem", "0.75rem",
          "9px", "10px", "11px")


def _each(sels, suffix: str = "") -> str:
    return ", ".join(f"{s}{suffix}" for s in sels)


_P, _C = f":is({', '.join(PAGES)})", f":is({', '.join(CARDS)})"
_TIP = (f'{_P} .stButton > div > button[kind="secondary"]',
        f'{_P} .stButton [data-testid="stTooltipHoverTarget"] > button[kind="secondary"]')
_TIP_BTN = ", ".join(_TIP)
_TIP_BTN_IN = ", ".join(f'{t} [data-testid="stMarkdownContainer"], {t} p' for t in _TIP)
_FLOOR = ",\n".join(f'{_P} [style*="font-size:{v}"], {_P} [style*="font-size: {v}"]' for v in _SMALL)
_SUMMARY = ' > [data-testid="stExpander"] > details > summary'   # nyckeln sitter på stLayoutWrapper runt stExpander
CSS = f"""
/* 8 px mellan blocken och korten (§3.4), som Thor */
{_each(PAGES)} {{ gap: 8px; }}
/* Underraden och Rick Rules inledning: Streamlits −16 px under markdown fick texten att nudda remsan respektive
   poängnoten — −8 px ger ~8 px luft (§4.7: underraden slutar 298, remsan börjar 306) */
{_P} [data-testid="stMarkdownContainer"]:has(> .wolf-note),
{_P} [data-testid="stMarkdownContainer"]:has(> .arken-intro) {{ margin-bottom: -8px; }}
/* Huvudraden: rubriken är kort — ⬇ CSV och rundans pill ryms bredvid (kompakta 44 px, 13 px som 📡) */
{_each(HEADS, ' > [data-testid="stElementContainer"]:has(.wolf-head)')} {{ min-width: 0; }}
{_each(HEADS, ' [data-testid="stDownloadButton"] button')} {{ min-height: 44px; padding: 0 12px; font-size: 13px;
    letter-spacing: .04em; }}
/* Bildtexterna 13 px NOTE (§3.3) — poängnoten "Marginal ≥ 40 % ger 2 p …" står kvar, läsbar */
{_P} [data-testid="stCaptionContainer"], {_P} [data-testid="stCaptionContainer"] p {{ font-size: 13px;
    color: {NOTE}; }}
/* Remsan: ingen egen marginal (sidans 8 px räcker); orden bryts efter mittpunkten, aldrig inne i ett ord
   ("1 KLARA FÖR KÖP ·" / "7 kandidater", inte "· 7" / "kandidater") */
{_P} .wolf-verdict-strip {{ margin: 0 !important; }}
{_P} [data-testid="stMarkdownContainer"]:has(> .wolf-verdict-strip) {{ margin-bottom: 0; }}
{_P} .wolf-verdict-strip > div:nth-child(2) {{ overflow-wrap: normal !important; }}
{_P} .wolf-verdict-strip > div:nth-child(2) > span:nth-child(odd) {{ white-space: nowrap; }}
/* Scorecardets två st.metric (test_thor_scorecard_entry läser "Klara för köp") som ett kompakt par under remsan:
   etikett och värde på en rad */
.st-key-kpis_scorecard [data-testid="stMetric"] {{ min-height: 0 !important; padding: 6px 12px !important; }}
.st-key-kpis_scorecard [data-testid="stMetric"] > div {{ display: flex; align-items: baseline;
    justify-content: space-between; gap: 8px; }}
.st-key-kpis_scorecard [data-testid="stMetricValue"], .st-key-kpis_scorecard [data-testid="stMetricValue"] * {{
    font-size: 16px !important; line-height: 1.25 !important; }}
/* Datorn: rutorna är ~636 px breda — värdet direkt efter etiketten, inte 500 px bort */
@media (min-width: 641px) {{
    .st-key-kpis_scorecard [data-testid="stMetric"] > div {{ justify-content: flex-start; gap: 12px; }}
}}
/* Fokusraden (review.focus, samma widgetar): väljaren och "Visa alla" på en rad även på telefonen */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-arken_focus_"]) {{
    flex-wrap: nowrap; align-items: flex-end; gap: 8px; }}
[data-testid="stColumn"]:has(> * [class*="st-key-arken_focus_"]:not([class*="_all"])),
[data-testid="stColumn"]:has(> [class*="st-key-arken_focus_"]:not([class*="_all"])) {{
    flex: 1 1 auto !important; min-width: 0 !important; width: auto !important; }}
[data-testid="stColumn"]:has([class*="st-key-arken_focus_"][class*="_all"]) {{
    flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }}
[class*="st-key-arken_focus_"] .react-aria-ComboBox > div {{ min-height: 44px; }}
/* I korten 8 px mellan raderna (Streamlits 16 px fick 48 px-kryssraderna att se toppställda ut) */
{_C} [data-testid="stExpanderDetails"] > [data-testid="stVerticalBlock"] {{ gap: 8px; }}
/* Korthuvudena 48 px (§4.7); expandrar, kryssrutor och knappar i korten är tryckytor på minst 44 px */
{_C} summary, {_P} summary {{ min-height: 44px; box-sizing: border-box; }}
{_each(CARDS, _SUMMARY)} {{ min-height: 48px; }}
{_C} [data-testid="stCheckbox"] label {{ min-height: 44px; align-items: center; box-sizing: border-box; }}
{_C} [class*="st-key-sc_g_"] [data-testid="stCheckbox"] label {{ min-height: 48px; width: 100%; }}
/* Hjälpikonen i en kryssrad: 44 × 44 inom radens 44/48 px */
{_C} [data-testid="stCheckbox"] [data-testid="stTooltipHoverTarget"] button {{ min-width: 44px; min-height: 44px;
    display: inline-flex; align-items: center; justify-content: center; }}
{_P} [data-testid="stFileUploader"] button {{ min-height: 44px; }}
/* Hjälpikonen i en fält- eller nyckeltalsetikett: 44 × 44 tryckyta utan att etikettraden växer (negativ marginal) */
{_P} :is([data-testid="stWidgetLabel"], [data-testid="stMetricLabel"]) [data-testid="stTooltipHoverTarget"] button {{
    min-width: 44px; min-height: 44px; margin: -14px; display: inline-flex; align-items: center;
    justify-content: center; }}
{_P} [data-testid="stNumberInputContainer"] {{ min-height: 44px; }}
{_P} [data-testid="stNumberInputContainer"] button {{ min-width: 44px; min-height: 44px; }}
{_P} .react-aria-ComboBox > div {{ min-height: 44px; }}
{_P} .react-aria-ComboBox button {{ min-width: 44px; min-height: 44px; }}
{_P} [data-testid="stPopoverButton"] {{ min-height: 44px; }}
.st-key-arken_fiat summary {{ min-height: 44px; box-sizing: border-box; }}
/* Knappar med tooltip (help=: "Hämta nu", "Hämta EV nu") ligger ett omslag längre ned än skalets
   '.stButton > button' — samma sekundärutseende och 44 px (som Thor, PR 20) */
{_TIP_BTN} {{ min-height: 44px; padding: 0 16px; border-radius: 10px; border: 1px solid {ACCENT};
    background: rgba(0,229,255,.08); color: {ACCENT}; font-family: {FONT_UI}; font-size: 14px; font-weight: 600;
    letter-spacing: .06em; text-transform: uppercase; }}
{_TIP_BTN_IN} {{ font: inherit; letter-spacing: inherit; }}
/* Toppen i kortet: domen och (när kortet är klart) "Registrera beslut" direkt under */
[class*="st-key-sc_top_"] {{ gap: 8px; }}
[class*="st-key-sc_top_"] [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
{_C} [data-testid="stMarkdownContainer"]:has(.arken-verdict) {{ margin-bottom: 0; }}
@media (max-width: 640px) {{
    /* Telefonen: remsans ord i 20 px — "1 KLARA FÖR KÖP · 7 kandidater" ryms då på en rad (362 px i 24 px) */
    {_P} .wolf-verdict-strip > div:nth-child(2) {{ font-size: 20px !important; }}
    /* Telefonen: Strategi över hela raden, de två positionsfälten bredvid varandra */
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-sc_post_"]) {{
        flex-wrap: wrap; gap: 8px 12px; align-items: flex-end; }}
    [data-testid="stColumn"]:has([class*="st-key-sc_post_"]), [data-testid="stColumn"]:has([class*="st-key-sc_poss_"]) {{
        flex: 1 1 calc(50% - 6px) !important; min-width: calc(50% - 6px) !important; width: calc(50% - 6px) !important; }}
    /* Rick Rule och Royalty: talfälten står kvar ett per rad — två i bredd lämnade ~65 px åt talet bredvid de
       44 px stora − och + ("6000." i stället för 6000.00) */
}}
/* 12 px-golvet (§8.5) för inline-storlekar under 12 px i det som ritas på arkens sidor */
{_FLOOR} {{ font-size: 12px !important; }}
/* Fokusmarkeringen på korthuvudet */
{_C} summary:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 2px; }}
"""


def style() -> None:
    """Arkens CSS i en dold stilbehållare (pcss_arken), en gång per körning. Ett fel får aldrig fälla sidan."""
    try:
        from streamlit.errors import StreamlitDuplicateElementKey

        from ui.shell import page_style
    except Exception:                                   # pragma: no cover
        return
    try:
        page_style(CSS, "arken")
    except StreamlitDuplicateElementKey:
        pass
    except Exception:                                   # pragma: no cover
        pass


def verdict_box(word: str, color: str, why: str = "", lead: str = "", lines=(), size: int = 20,
                text_color: Optional[str] = None) -> str:
    """Domen som ruta överst i kortet — verdiktrutans anatomi (§3.5.9) i kortstorlek: poängen (lead, mono) och ordet i
    arkets färg, skälet i 13 px och luckorna som rader. Visar bara de ord kortet redan har; allt escapas. Text under
    19 px tar färgens bläckton (ink); text_color ersätter ordets färg (NEJ-ord i MUTED, kanten grå)."""
    fg = text_color or (color if size >= 19 else ink(color))
    head = ((f'<span style="font-family:{FONT_MONO};font-weight:600;margin-right:10px;">{escape(str(lead))}</span>'
             if lead else "") + escape(str(word)))
    return (f'<div class="arken-verdict" style="background:{tint(color, .06)};border:1px solid {tint(color, .4)};'
            f'border-left:4px solid {color};border-radius:12px;padding:10px 14px;margin:6px 0 8px;">'
            f'<div style="color:{fg};font-family:{FONT_UI};font-size:{int(size)}px;font-weight:800;line-height:1.25;'
            f'letter-spacing:.02em;">{head}</div>'
            + (f'<div style="color:{TEXT};font-size:13px;line-height:1.45;margin-top:4px;">{escape(str(why))}</div>'
               if why else "")
            + "".join(f'<div style="color:{TEXT};font-size:13px;line-height:1.45;margin-top:2px;">• {escape(str(m))}'
                      f'</div>' for m in lines or ())
            + "</div>")


def count_text(sheet: str, row, stage: Optional[str] = None, position_pct=None) -> str:
    """'12/34 fält ifyllda' — samma fält som underlaget till ChatGPT räknar. Tom om något går fel."""
    try:
        from review import catalog as cat
        done, total, _missing = cat.progress(sheet, row, stage, position_pct)
    except Exception:
        return ""
    return f"{done}/{total} fält ifyllda" if total else ""


def missing_text(items: list) -> str:
    """'A · B · C … +2' — fälten i ordning, varje etikett en gång."""
    labels = list(dict.fromkeys(it.label for it in items if getattr(it, "label", "")))
    head = " · ".join(labels[:MAX_LABELS])
    return head + (f" … +{len(labels) - MAX_LABELS}" if len(labels) > MAX_LABELS else "")


def slot():
    """Platshållaren först i kortet (st.empty) — fill ritar raden där sist i kortet. None om något går fel."""
    try:
        return st.empty()
    except Exception:
        return None


def fill(place, sheet: str, row, entry: Optional[dict] = None, stage: Optional[str] = None,
         position_pct=None, verdict_html: Optional[str] = None) -> list:
    """Raden först i kortet, ritad i platshållaren från slot: räknaren och det som saknas för att grindarna ska gå
    att pröva (dämpat rött, skälen i hovringstexten) — eller att inget saknas. Anropas sist i kortet, så att båda
    läser den här körningens värden. Returnerar luckorna (tom lista när allt finns, eller när något gick fel).

    verdict_html = arkets dom (färdig HTML, t.ex. verdict_box) — står först i SAMMA markdown som räknaren, så att
    kortets första markdown fortfarande är raden med 'x/y fält ifyllda' och "Saknas för köpbeslut"."""
    if place is None:
        return []
    count = count_text(sheet, row, stage, position_pct)
    try:
        from review import missing as mis
        items = mis.missing_for(sheet, row, entry=entry, stage=stage, position_pct=position_pct)
    except Exception:
        items = None
    html = verdict_html or ""
    html += (f"<div style='color:{DONE_COLOR};font-size:0.78rem;margin:0 0 2px;'>{escape(count)}</div>"
             if count else "")
    if items == []:
        html += (f"<div style='color:{DONE_COLOR};font-size:0.78rem;margin:0 0 6px;'>✓ Inget saknas för "
                 f"köpbeslut — grindarna går att pröva.</div>")
    elif items:
        why = escape(" · ".join(dict.fromkeys(f"{it.label}: {it.reason}" for it in items))[:900], quote=True)
        html += (f"<div title=\"{why}\" style='color:{MISSING_COLOR};opacity:0.85;font-size:0.8rem;margin:0 0 6px;'>"
                 f"Saknas för köpbeslut: {escape(missing_text(items))}</div>")
    if html:
        try:
            place.markdown(html, unsafe_allow_html=True)
        except Exception:
            pass
    return items or []


def focus(sheet: str, tickers: Iterable) -> Optional[str]:
    """Fokusväljaren (review.focus) — None, och ingen väljare, om något går fel."""
    try:
        from review import focus as fc
        return fc.render_focus_bar(sheet, list(tickers))
    except Exception:
        return None


def same_ticker(ticker, focused: Optional[str]) -> bool:
    """Hör raden till bolaget i fokus? Utan fokus: alla rader."""
    if not focused:
        return True
    return str(ticker or "").strip().upper() == focused
