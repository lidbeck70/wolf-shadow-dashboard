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


def suggestion(blob: dict, sheet: str, row: dict, source_field: str,
               row_id=None, unit_field: Optional[str] = None):
    """(värde, asof, valuta/enhet) ur bloben för raden, eller None.

    row_id när raden som skrivs är en underdict (Lukacs FV ligger i
    row["fv"]) — nyckeln är fortfarande arkets rad. unit_field pekar på ett
    enhetsfält i bloben (commodity_unit) i stället för valutan."""
    s = ((blob or {}).get("rows") or {}).get(f"{sheet}:{row_id or row.get('id')}")
    if not s or s.get(source_field) is None:
        return None
    try:
        unit = s.get(unit_field) if unit_field else s.get("currency")
        return float(s[source_field]), str(s.get("asof") or "")[:10], unit or ""
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
            fmt: str = "{:g}", with_currency: bool = False, row_id=None,
            unit_field: Optional[str] = None) -> None:
    """Rita förslaget om det finns och skiljer sig från fältets värde."""
    sug = suggestion(load_refresh(), sheet, row, source_field, row_id=row_id,
                     unit_field=unit_field)
    if sug is None:
        return
    val, asof, ccy = sug
    with_currency = with_currency or bool(unit_field)
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
    c2.button("Använd", key=f"rf_{sheet}_{row_id or row.get('id')}_{field}",
              on_click=_apply, args=(row, field, val, widget_key, on_apply))
