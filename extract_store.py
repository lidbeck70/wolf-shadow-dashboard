"""
extract_store.py — ett dokument, ett utdrag, förslag i alla ark.

AI-läsningen av en presentation gjordes förut per ark med arkets egen
fältlista: samma PDF fick laddas upp i Rick Rule, Poängmodellen, Tiggre
och Durrett-arket var för sig. Nu ställs alla arks fält i ett anrop
(extract_prompt.ALL_SHEETS) och svaret sparas här per ticker, i sessionen.
Varje arks extraktor läser sina förslag härifrån — och visar var utdraget
kom ifrån. Sessionen räcker: dokumentet är en engångsläsning, och det som
används skrivs in i arket med källa.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import streamlit as st

PREFIX = "xt_doc:"


def _key(ticker: str) -> str:
    return PREFIX + str(ticker or "").strip().upper()


def save(ticker: str, parsed: dict, doc: str = "", sheet: str = "", model: str = "",
         pages: int = 0, chars: int = 0) -> dict:
    """Sparar utdraget för tickern (ersätter ett äldre)."""
    entry = {"parsed": parsed, "doc": doc or "Presentation", "sheet": sheet, "model": model,
             "pages": pages, "chars": chars,
             "when": datetime.now().strftime("%Y-%m-%d %H:%M")}
    st.session_state[_key(ticker)] = entry
    return entry


def get(ticker: str) -> Optional[dict]:
    e = st.session_state.get(_key(ticker))
    return e if isinstance(e, dict) and isinstance(e.get("parsed"), dict) else None


def forget(ticker: str) -> None:
    st.session_state.pop(_key(ticker), None)


def describe(entry: dict, here: str = "") -> str:
    """'Utdrag ur DFS 2025 (2026-09-23 10:12, via Tiggre)'."""
    via = entry.get("sheet") or ""
    where = "" if not via or via == here else f", via {via}"
    return f"Utdrag ur {entry.get('doc') or 'dokumentet'} ({entry.get('when', '?')}{where})"
