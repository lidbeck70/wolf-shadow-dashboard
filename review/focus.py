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
ALL = "Alla bolag"


def _forms(ticker) -> set:
    t = str(ticker or "").strip().upper()
    if not t:
        return set()
    try:
        from thor.ticker import forms
        out = {f.upper() for f in forms(t) if f}
    except Exception:
        out = set()
    return out | {t}


def matches(ticker, focus) -> bool:
    """Samma aktie i någon av tickerns former ('BOL.ST' och 'BOL')."""
    a, b = _forms(ticker), _forms(focus)
    return bool(a and b and a & b)


def focus_ticker() -> str:
    return str(st.session_state.get(FOCUS) or "").strip().upper()


def set_focus(ticker) -> None:
    st.session_state[FOCUS] = str(ticker or "").strip().upper()


def clear_focus() -> None:
    st.session_state[FOCUS] = ""


def _picked(widget_key: str) -> None:
    v = st.session_state.get(widget_key)
    if v in (None, ALL):
        clear_focus()
    else:
        set_focus(v)


def pick(tickers: Iterable, focus: str) -> Optional[str]:
    """Arkets egen form av tickern i fokus, eller None när fokus saknas eller bolaget inte finns i arket."""
    if not focus:
        return None
    return next((t for t in tickers if matches(t, focus)), None)


def render_focus_bar(sheet: str, tickers: Iterable) -> Optional[str]:
    """Väljaren överst i arket (två bolag eller fler). Returnerar tickern i fokus i arkets form, eller None = visa
    alla. Väljarens läge sätts före widgeten, så den följer fokus som Thor eller ett annat ark satt."""
    opts = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t or "").strip()))
    foc = focus_ticker()
    hit = pick(opts, foc)
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
