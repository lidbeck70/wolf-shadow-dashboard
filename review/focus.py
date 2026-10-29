"""
review/focus.py — fokusläget i arken: ett bolag i taget.

Thors arkknappar sätter fokus på aktien (FOCUS via bryggan thor.ui._jump), så arket öppnar på bolaget i stället för
en lista med stängda kort. Väljaren överst byter bolag och "Visa alla" tar tillbaka listan. Samma fokus följer med
mellan arken tills du släpper det; ett ark där bolaget inte finns visar alla. Bara visning — inget i arken skrivs.
"""

from __future__ import annotations

from typing import Iterable, Optional

import streamlit as st

FOCUS = "arken_focus"            # tickern i fokus
FORMS = "arken_focus_forms"      # {ticker, forms}: arkformerna Thors dossier läste bolaget på (registrets karta)
ALL = "Alla bolag"


def _forms(ticker) -> set:
    """Arkformerna (thor.dossier.sheet_forms, som Thor och underlaget): 'BOL.ST' når 'BOL', men en utländsk
    notering (EQX.TO) bara sin egen form — en bar 'EQX' är den amerikanska."""
    t = str(ticker or "").strip().upper()
    if not t:
        return set()
    try:
        from thor.dossier import sheet_forms
        out = {f.upper() for f in sheet_forms(t) if f}
    except Exception:
        out = set()
    return out | {t}


def matches(ticker, focus, forms: Optional[Iterable] = None) -> bool:
    """Samma aktie i någon av tickerns former ('BOL.ST' och 'BOL'). forms = arkformerna Thor läste fokus på:
    då räknas bara de — med registrets karta är arkets 'TEL' TE Connectivity, inte Telenor TEL.OL."""
    t = str(ticker or "").strip().upper()
    if forms:
        return bool(t) and t in {str(f).strip().upper() for f in forms}
    a, b = _forms(t), _forms(focus)
    return bool(a and b and a & b)


def focus_ticker() -> str:
    return str(st.session_state.get(FOCUS) or "").strip().upper()


def focus_forms() -> Optional[list]:
    """Arkformerna Thor skickade med fokus — None när fokus sattes någon annanstans eller gäller en annan ticker."""
    v = st.session_state.get(FORMS)
    foc = focus_ticker()
    if isinstance(v, dict) and foc and str(v.get("ticker") or "").strip().upper() == foc:
        return list(v.get("forms") or []) or None
    return None


def set_focus(ticker) -> None:
    st.session_state[FOCUS] = str(ticker or "").strip().upper()
    st.session_state.pop(FORMS, None)


def clear_focus() -> None:
    st.session_state[FOCUS] = ""
    st.session_state.pop(FORMS, None)


def _picked(widget_key: str) -> None:
    v = st.session_state.get(widget_key)
    if v in (None, ALL):
        clear_focus()
    else:
        set_focus(v)


def pick(tickers: Iterable, focus: str, forms: Optional[Iterable] = None) -> Optional[str]:
    """Arkets egen form av tickern i fokus, eller None när fokus saknas eller bolaget inte finns i arket.
    forms = arkformerna Thor läste fokus på (focus_forms)."""
    if not focus:
        return None
    return next((t for t in tickers if matches(t, focus, forms)), None)


def render_focus_bar(sheet: str, tickers: Iterable) -> Optional[str]:
    """Väljaren överst i arket (två bolag eller fler). Returnerar tickern i fokus i arkets form, eller None = visa
    alla. Väljarens läge sätts före widgeten, så den följer fokus som Thor eller ett annat ark satt."""
    opts = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t or "").strip()))
    foc = focus_ticker()
    hit = pick(opts, foc, focus_forms())
    if len(opts) < 2:
        return hit
    wkey = f"{FOCUS}_{sheet}"
    want = hit or ALL
    if st.session_state.get(wkey) != want:
        st.session_state[wkey] = want
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    c1.selectbox("Fokus — ett bolag i taget", [ALL] + opts, key=wkey, on_change=_picked, args=(wkey,),
                 help="Visar bara bolaget du väljer, med kortet öppet. Fokus följer med till de andra arken.")
    c2.button("Visa alla", key=f"{wkey}_all", on_click=clear_focus, disabled=hit is None, width="stretch")
    if foc and hit is None:
        st.caption(f"{foc} finns inte i arket — visar alla bolag.")
    return hit
