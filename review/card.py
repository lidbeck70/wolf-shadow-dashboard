"""
review/card.py — kortens stöd i arken: '· x/y ifyllda' i rubriken, raden "Saknas för köpbeslut" först i kortet
och fokusväljaren överst (review.focus).

Allt härleds ur fältkatalogen (review.catalog.progress) och arkens egna grindar (review.missing.missing_for) — ingen
regel räknas här. Inget får fälla ett kort: ett fel ger en tom rubriksvans, ingen rad och ingen väljare.
"""

from __future__ import annotations

from html import escape
from typing import Iterable, Optional

import streamlit as st

MISSING_COLOR, DONE_COLOR = "#c44545", "#8a8578"
MAX_LABELS = 8


def suffix(sheet: str, row, stage: Optional[str] = None, position_pct=None) -> str:
    """' · 12/34 ifyllda' för kortets rubrik — samma fält som underlaget till ChatGPT räknar."""
    try:
        from review import catalog as cat
        done, total, _missing = cat.progress(sheet, row, stage, position_pct)
    except Exception:
        return ""
    return f" · {done}/{total} ifyllda" if total else ""


def missing_text(items: list) -> str:
    """'A · B · C … +2' — fälten i ordning, varje etikett en gång."""
    labels = list(dict.fromkeys(it.label for it in items if getattr(it, "label", "")))
    head = " · ".join(labels[:MAX_LABELS])
    return head + (f" … +{len(labels) - MAX_LABELS}" if len(labels) > MAX_LABELS else "")


def render_missing(sheet: str, row, entry: Optional[dict] = None, stage: Optional[str] = None,
                   position_pct=None) -> list:
    """Den dämpat röda raden först i kortet: det som saknas för att grindarna ska gå att pröva. Skälen står i
    hovringstexten. Returnerar luckorna (tom lista när allt finns, eller när något gick fel)."""
    try:
        from review import missing as mis
        items = mis.missing_for(sheet, row, entry=entry, stage=stage, position_pct=position_pct)
    except Exception:
        return []
    if not items:
        st.markdown(f"<div style='color:{DONE_COLOR};font-size:0.78rem;margin:0 0 6px;'>✓ Inget saknas för "
                    f"köpbeslut — grindarna går att pröva.</div>", unsafe_allow_html=True)
        return items
    why = escape(" · ".join(dict.fromkeys(f"{it.label}: {it.reason}" for it in items))[:900], quote=True)
    st.markdown(f"<div title=\"{why}\" style='color:{MISSING_COLOR};opacity:0.85;font-size:0.8rem;margin:0 0 6px;'>"
                f"Saknas för köpbeslut: {escape(missing_text(items))}</div>", unsafe_allow_html=True)
    return items


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
