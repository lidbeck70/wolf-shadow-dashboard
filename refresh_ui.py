"""
refresh_ui.py — "🤖 Börsdata: 123,4 · Använd" bredvid ett arkfält.

Läser sheets_refresh.json (sheets_refresh.py i GitHub Actions) och visar
jobbets färska tal som FÖRSLAG. Knappen skriver in värdet i raden och i
widgetens session-state via on_click-callback (det enda tillfället då
Streamlit tillåter att en instansierad widgets state ändras) — inget
skrivs annars, och 💾 Spara sköter persistensen som vanligt.
"""
from __future__ import annotations

import html
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Callable, Optional

import streamlit as st

DIM = "#8a8578"

# Bolagsformer och ord som inte skiljer bolag åt: "Boliden AB" liknar "Boliden", men "Gold" gör inte
# "Gold Fields" till "Gold Resource".
_LEGAL = frozenset("ab publ asa as oyj abp a s inc corp corporation co company ltd limited plc sa se nv ag "
                   "the group holding holdings class ser series".split())
_GENERIC = frozenset("gold silver mining mines minerals metals resources royalty royalties energy energi "
                     "international global capital".split())
NAME_RATIO_MIN = 0.8        # teckenlikhet mellan de hopskrivna namnen som räknas som samma namn


_NORDIC = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE"})     # NFKD delar inte ø/æ ('Møller' = 'Moller')


def _name_parts(name) -> list:
    s = unicodedata.normalize("NFKD", str(name or "").translate(_NORDIC))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def _name_tokens(name) -> list:
    return [t for t in _name_parts(name) if len(t) > 1 and t not in _LEGAL]


def name_resembles(own, bd_name) -> bool:
    """Liknar radens eget namn Börsdatas? Bolagsformer, accenter och skiljetecken räknas inte ('Var Energi' =
    'Vår Energi ASA'); ett namn som ryms i det andra ('Kinross' i 'Kinross Gold') räcker, liksom ett
    gemensamt ord som inte är branschord ('Gold', 'Mining'). True när något av namnen saknas — inget att jämföra."""
    a, b = _name_tokens(own), _name_tokens(bd_name)
    if not a or not b:
        return True
    ja, jb = "".join(a), "".join(b)
    if ja in jb or jb in ja:
        return True
    if {t for t in a if len(t) >= 3 and t not in _GENERIC} & {t for t in b if len(t) >= 3 and t not in _GENERIC}:
        return True
    return SequenceMatcher(None, ja, jb).ratio() >= NAME_RATIO_MIN


def matched_note(blob: dict, sheet: str, row: dict, row_id=None) -> str:
    """' (matchat: Genovis AB)' när sifferuppdateringens bolag inte liknar radens eget namn, eller raden saknar
    namn — så syns det när en manuell rad fått fel bolag. Tom sträng annars, och utan Börsdatas namn."""
    s = ((blob or {}).get("rows") or {}).get(f"{sheet}:{row_id or (row or {}).get('id')}") or {}
    bd = str(s.get("bd_name") or "").strip()
    if not bd:
        return ""
    own = str((row or {}).get("name") or "").strip()
    if _name_tokens(own):
        if name_resembles(own, bd):
            return ""
    elif own:
        # bara enbokstavsdelar ('H&M B', Börsdatas eget namn): jämför bokstäverna hopskrivna, bolagsformer borta
        ca, cb = ("".join(t for t in _name_parts(n) if t not in _LEGAL) for n in (own, bd))
        if ca and (ca in cb or cb in ca):
            return ""
    return f" (matchat: {html.escape(bd).replace('$', '&#36;')})"


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
            unit_field: Optional[str] = None, lo: Optional[float] = None) -> None:
    """Rita förslaget om det finns och skiljer sig från fältets värde. lo = fältets golv: ett förslag under det
    (Börsdatas negativa EV/EBITDA vid negativt EBITDA) erbjuds inte — rutan skulle inte kunna visa det."""
    blob = load_refresh()
    sug = suggestion(blob, sheet, row, source_field, row_id=row_id,
                     unit_field=unit_field)
    if sug is None or (lo is not None and sug[0] < lo):
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
                f"{label} {fmt.format(val)}{unit}{matched_note(blob, sheet, row, row_id)}</span>",
                unsafe_allow_html=True)
    c2.button("Använd", key=f"rf_{sheet}_{row_id or row.get('id')}_{field}",
              on_click=_apply, args=(row, field, val, widget_key, on_apply))
