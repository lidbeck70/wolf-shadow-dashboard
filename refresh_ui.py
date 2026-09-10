"""
refresh_ui.py — "🤖 Börsdata: 123,4 · Använd" bredvid ett arkfält.

Läser sheets_refresh.json (sheets_refresh.py i GitHub Actions) och visar
jobbets färska tal som FÖRSLAG. Knappen skriver in värdet i raden och i
widgetens session-state via on_click-callback (det enda tillfället då
Streamlit tillåter att en instansierad widgets state ändras) — inget
skrivs annars, och 💾 Spara sköter persistensen som vanligt.
"""
from __future__ import annotations

from typing import Callable, Optional

import streamlit as st

DIM = "#8a8578"


@st.cache_data(ttl=600, show_spinner=False)
def load_refresh() -> dict:
    try:
        from gist_storage import load_blob
        d = load_blob("sheets_refresh.json", None)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def suggestion(blob: dict, sheet: str, row: dict, source_field: str):
    """(värde, asof, valuta) ur bloben för raden, eller None."""
    s = ((blob or {}).get("rows") or {}).get(f"{sheet}:{row.get('id')}")
    if not s or s.get(source_field) is None:
        return None
    try:
        return float(s[source_field]), str(s.get("asof") or "")[:10], s.get("currency") or ""
    except (TypeError, ValueError):
        return None


def _apply(row: dict, field: str, value: float, widget_key: Optional[str],
           on_apply: Optional[Callable]) -> None:
    row[field] = value
    if widget_key:
        st.session_state[widget_key] = value
    if on_apply:
        on_apply()


def suggest(sheet: str, row: dict, field: str, source_field: str, label: str,
            widget_key: Optional[str] = None, on_apply: Optional[Callable] = None,
            fmt: str = "{:g}", with_currency: bool = False) -> None:
    """Rita förslaget om det finns och skiljer sig från fältets värde."""
    sug = suggestion(load_refresh(), sheet, row, source_field)
    if sug is None:
        return
    val, asof, ccy = sug
    cur = row.get(field)
    try:
        if cur is not None and abs(float(cur) - val) < 1e-6:
            return
    except (TypeError, ValueError):
        pass
    unit = f" {ccy}" if with_currency and ccy else ""
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"<span style='color:{DIM};font-size:0.78rem;'>🤖 Börsdata {asof}: "
                f"{label} {fmt.format(val)}{unit}</span>", unsafe_allow_html=True)
    c2.button("Använd", key=f"rf_{sheet}_{row.get('id')}_{field}",
              on_click=_apply, args=(row, field, val, widget_key, on_apply))
