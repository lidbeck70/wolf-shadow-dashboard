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
"""

from __future__ import annotations

from html import escape
from typing import Iterable, Optional

import streamlit as st

MISSING_COLOR, DONE_COLOR = "#c44545", "#8a8578"
MAX_LABELS = 8


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
         position_pct=None) -> list:
    """Raden först i kortet, ritad i platshållaren från slot: räknaren och det som saknas för att grindarna ska gå
    att pröva (dämpat rött, skälen i hovringstexten) — eller att inget saknas. Anropas sist i kortet, så att båda
    läser den här körningens värden. Returnerar luckorna (tom lista när allt finns, eller när något gick fel)."""
    if place is None:
        return []
    count = count_text(sheet, row, stage, position_pct)
    try:
        from review import missing as mis
        items = mis.missing_for(sheet, row, entry=entry, stage=stage, position_pct=position_pct)
    except Exception:
        items = None
    html = (f"<div style='color:{DONE_COLOR};font-size:0.78rem;margin:0 0 2px;'>{escape(count)}</div>"
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
