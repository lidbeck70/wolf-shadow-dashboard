"""
ui/tables.py — tabellerna på telefonen (LIDSKJALV §3.5.15, PR 24).

  1. table(df, key)                  EN st.dataframe i st.container(key="tbl_<key>"): första spalten fastnålad,
                                     saknade värden som "—" (aldrig "None"), utan index, hela bredden
  2. table(df, key, primary_cols)    från sju spalter: bara huvudspalterna, och st.toggle("Visa alla kolumner",
                                     key="tbl_<key>_all"), i huvudspalternas ordning. Urvalet är bara visningen
                                     (column_order) — tabellens data är hela df i samma ordning, så det testerna och
                                     nedladdningen läser är oförändrat; reglaget på visar alla i tabellens ordning
  3. cards_html(rows, title)         rad-per-sak som styr ett beslut (Inställningars låsta regler): ett kort per rad
                                     ur samma rader, varje värde ordagrant och helt; en spalt på telefonen, 2–3 på datorn

CSS nedan (högerkanten tonar ut på telefonen medan det finns mer att svepa, reglaget 44 px) läggs i skalets
stilblock via ui/shell.parts_css(), så sidorna får inga nya element. Inga regler, inga nya värden: tabellerna och
korten visar raderna de får.
"""
from __future__ import annotations

from html import escape
from typing import Callable, Optional

from ui.tokens import BG_CARD, BORDER_CARD, FONT_UI, MUTED, TEXT

SUBSET_FROM = 7                      # §3.5.15: från sju spalter visas huvudspalterna först
TOGGLE = "Visa alla kolumner"
MISSING = "—"                        # saknat värde i en tabell (Streamlits eget är "None")
KEY_PREFIX = "tbl_"
BOX = f'[class*="st-key-{KEY_PREFIX}"]'
FADE = "linear-gradient(90deg, #000 calc(100% - 28px), transparent)"      # som skalets FADE
CARD_MIN = 320                       # kortens minsta bredd: en spalt på telefonen, 2–3 på datorn


def visible(columns, primary_cols=None) -> Optional[list]:
    """Spalterna som visas när reglaget är av: första spalten, sedan huvudspalterna som finns i primary_cols ordning
    (beslutet först — det som syns på telefonens första skärm). None = alla i tabellens ordning (färre än
    SUBSET_FROM spalter, inga huvudspalter, eller urvalet är redan alla)."""
    cols = [str(c) for c in columns]
    if not primary_cols or len(cols) < SUBSET_FROM:
        return None
    have = set(cols[1:])
    keep = cols[:1] + [c for c in dict.fromkeys(str(p) for p in primary_cols) if c in have]
    return keep if len(keep) < len(cols) else None


def column_config(columns) -> dict:
    """Första spalten fastnålad — tickern eller strategin står kvar när resten sveps i sidled."""
    import streamlit as st
    cols = [str(c) for c in columns]
    return {cols[0]: st.column_config.Column(pinned=True)} if cols else {}


def table(df, key: str, primary_cols=None, style: Optional[Callable] = None) -> None:
    """Ritar EN st.dataframe i st.container(key='tbl_<key>'). primary_cols = huvudspalterna (från sju spalter visas
    bara de, med reglaget 'Visa alla kolumner' ovanför); style = df → Styler (t.ex. lägets färg). Datan i tabellen är
    alltid hela df — urvalet är column_order, bara visningen."""
    import streamlit as st
    order = visible(df.columns, primary_cols)
    with st.container(key=f"{KEY_PREFIX}{key}"):
        if order is not None and st.toggle(TOGGLE, key=f"{KEY_PREFIX}{key}_all", value=False):
            order = None
        st.dataframe(style(df) if style else df, hide_index=True, width="stretch", column_order=order,
                     column_config=column_config(df.columns), placeholder=MISSING)


# ── Kortlistan (§3.5.15 regel 3) ─────────────────────────────────────────────
def _t(x) -> str:
    """Text i ett element, escapad — ett värde kan aldrig bli HTML. None blir '—'."""
    return escape(MISSING if x is None else str(x), quote=False)


def card_html(row: dict, title: str) -> str:
    """Ett kort: row[title] som rubrik (16 px), sedan varje annan nyckel som etikett (12 px MUTED versaler) och dess
    värde ordagrant under (14 px TEXT, radbryts — aldrig avkortat). Kortet som komponenternas (§3.5.11)."""
    rows = "".join(
        f'<div style="min-width:0;"><div style="color:{MUTED};font-family:{FONT_UI};font-size:12px;font-weight:600;'
        f'line-height:1.25;letter-spacing:.1em;text-transform:uppercase;">{_t(k)}</div>'
        f'<div style="color:{TEXT};font-size:14px;line-height:1.45;margin-top:2px;overflow-wrap:anywhere;">{_t(v)}</div>'
        f'</div>'
        for k, v in row.items() if k != title)
    return (f'<div class="wolf-rule-card" style="background:{BG_CARD};border:1px solid {BORDER_CARD};border-radius:12px;'
            f'padding:12px 14px;min-width:0;box-sizing:border-box;">'
            f'<div style="color:{TEXT};font-family:{FONT_UI};font-size:16px;font-weight:700;line-height:1.3;'
            f'margin-bottom:8px;">{_t(row.get(title))}</div>'
            f'<div style="display:grid;gap:6px;">{rows}</div></div>')


def cards_html(rows, title: str) -> str:
    """Kortlistan som EN HTML-sträng (ritas med st.markdown): ett kort per rad i radernas ordning, i ett rutnät som
    fyller raden — en spalt på telefonen, 2–3 på datorn (auto-fill, minst CARD_MIN px per kort). Ingen tom rad i
    strängen, så markdown rör aldrig texten."""
    cards = "".join(card_html(r, title) for r in rows or ())
    return (f'<div class="wolf-rule-cards" style="display:grid;grid-template-columns:repeat(auto-fill,'
            f'minmax(min({CARD_MIN}px,100%),1fr));gap:8px;margin:4px 0 8px;min-width:0;">{cards}</div>')


# ── CSS (i skalets stilblock, ui/shell.parts_css) ────────────────────────────
GRID = f'{BOX} [data-testid="stDataFrameResizable"]'      # tabellens ruta (canvasen ritas här i, under rullaren)
SCROLLER = f"{BOX} .dvn-scroller"                          # glide-tabellens egen rullningsruta (ovanpå canvasen)
TIMELINE = "--wolf-tbl"


def _edge(size: str) -> str:
    return (f"-webkit-mask-image: {FADE}; mask-image: {FADE}; -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat; "
            f"-webkit-mask-size: {size} 100%; mask-size: {size} 100%;")


CSS = f"""
/* ── LIDSKJALV · tabellerna (PR 24, §3.5.15) ───────────────────── */
/* tbl_<nyckel>: en tabell och ev. reglaget. Telefonen: tabellens högerkant tonar ut — fler spalter att svepa fram
   (första spalten är fastnålad). Verktygsraden ovanför tabellen tonas inte.
   Reserven (webbläsare utan rullningsstyrd animering, och med reducerad rörelse): en fast toning. */
@media (max-width: 640px) {{
    {GRID} {{ -webkit-mask-image: {FADE}; mask-image: {FADE}; }}
}}
/* Med rullningsstyrd animering följer toningen rullningen: den syns bara när det finns mer till höger, glider ut de
   sista 8 % och är borta vid änden — sista spaltens text tonas aldrig. En tabell som ryms har ingen tidslinje och
   ingen toning. Ingen tid, ingen rörelse av sig själv: bara rullningens läge. */
@supports (animation-timeline: scroll()) and (timeline-scope: --x) {{
    @media (max-width: 640px) and (prefers-reduced-motion: no-preference) {{
        {GRID} {{ -webkit-mask-image: none; mask-image: none; timeline-scope: {TIMELINE};
            animation: wolf-tbl-edge linear both; animation-timeline: {TIMELINE}; }}
        {SCROLLER} {{ scroll-timeline: {TIMELINE} x; }}
    }}
}}
@keyframes wolf-tbl-edge {{ from, 92% {{ {_edge("100%")} }} to {{ {_edge("calc(100% + 28px)")} }} }}
/* Reglaget "Visa alla kolumner": 44 px tryckyta */
{BOX} [data-testid="stCheckbox"] label {{ min-height: 44px; align-items: center; }}
/* Pekskärm (ingen hovring): Streamlit visar tabellens verktygsrad hela tiden, 42 px ovanför tabellen. Med reglaget
   ligger den till höger om det; utan reglage låg den över rubriken ("standardbolag" på 390) — då en egen rad. */
@media (hover: none) {{
    {BOX}:not(:has([data-testid="stCheckbox"])) {{ padding-top: 32px; }}
}}
"""
