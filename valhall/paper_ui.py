"""
valhall/paper_ui.py — 📒 PORTFÖLJ → 📊 Papperskonton (K8): alla papperskonton på ett ställe.

Sidan räknar ingenting — den läser vad Valhall Scan (valhall_paper.json, valhall_board.json["paper"],
valhall_paper_history.json) och BERSERK Scan (berserk_paper.json) sparat i Gisten:
  Sammanställningen  per strategi, med uppdelning på segment, komplex, status/cykel/fas och vändning vid köp
  Utvecklingen       snitt-R och enhetskurvans årstakt per strategi över tid (historiken)
  Positioner         öppna positioner och order, stängda affärer och händelser — granska varje pappersaffär
  BERSERK            BERSERK:s eget kassakonto (fliken som tidigare inte var monterad)
Riktiga order läggs alltid manuellt.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from ui.tokens import CYAN, DIM

_STATE = "paper_page_data"
BLOBS = {"board": "valhall_board.json", "state": "valhall_paper.json", "history": "valhall_paper_history.json"}


def load(loader: Optional[Callable] = None) -> dict:
    """Blobbarna ur Gisten — None för det som inte går att läsa, aldrig ett fel."""
    if loader is None:
        from gist_storage import load_blob as loader
    out = {}
    for k, name in BLOBS.items():
        try:
            out[k] = loader(name, None)
        except Exception:
            out[k] = None
    out["loaded"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return out


def _label(key: str) -> str:
    from valhall.ui import STRATEGY_LABEL
    return STRATEGY_LABEL.get(key, key)


def positions_table(state: Optional[dict]) -> pd.DataFrame:
    """Öppna positioner och väntande order."""
    rows = []
    for kind, xs in (("position", (state or {}).get("positions") or []), ("order", (state or {}).get("orders") or [])):
        for x in xs:
            entry, last = x.get("entry"), x.get("last_close")
            turn = x.get("turn") or {}
            rows.append({"Läge": kind, "Strategi": _label(x.get("strategy")), "Ticker": x.get("ticker"),
                         "Namn": x.get("name"), "Komplex": x.get("complex") or "—", "Segment": x.get("segment"),
                         "Signal": x.get("signal_date"), "Köpt": x.get("entry_date"), "Entry": entry,
                         "Stopp": x.get("init_stop"), "Senast": last,
                         "Avk. %": round((last / entry - 1) * 100, 1) if entry and last else None,
                         "Vändning": f"{turn.get('kind')} dag {turn.get('days')}" + (" ✓" if turn.get("confirmed") else "")
                         if turn.get("kind") else "—", "Varför": x.get("why")})
    return pd.DataFrame(rows)


def closed_table(state: Optional[dict], n: int = 200) -> pd.DataFrame:
    """De senaste stängda affärerna, nyast först."""
    rows = []
    for x in reversed(((state or {}).get("closed") or [])[-n:]):
        entry, ex = x.get("entry"), x.get("exit")
        stop = x.get("init_stop")
        r = (ex - entry) / (entry - stop) if entry and ex and stop and entry > stop else None
        rows.append({"Strategi": _label(x.get("strategy")), "Ticker": x.get("ticker"), "Komplex": x.get("complex") or "—",
                     "Köpt": x.get("entry_date"), "Såld": x.get("exit_date"), "Entry": entry, "Exit": ex,
                     "Avk. %": round((ex / entry - 1) * 100, 1) if entry and ex else None,
                     "R": round(r, 2) if r is not None else None, "Orsak": x.get("reason")})
    return pd.DataFrame(rows)


def events_table(state: Optional[dict], n: int = 100) -> pd.DataFrame:
    return pd.DataFrame([{"Datum": e.get("date"), "Händelse": e.get("kind"), "Strategi": _label(e.get("strategy")),
                          "Ticker": e.get("ticker"), "Text": e.get("text")}
                         for e in reversed(((state or {}).get("events") or [])[-n:])])


def history_frame(history: Optional[list], metric: str = "avg_r") -> pd.DataFrame:
    """Datum × strategi för ett mått ur historiken (tom ram utan historik)."""
    rows = {}
    for h in history or []:
        if not isinstance(h, dict) or not h.get("date"):
            continue
        rows[h["date"]] = {_label(k): v.get(metric) for k, v in (h.get("by_strategy") or {}).items()}
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def render_paper_page(loader: Optional[Callable] = None) -> None:
    import streamlit as st
    from ui.components import note, page_header
    from valhall.ui import render_paper
    page_header("📊 Papperskonton", "Varje strategi handlas på papper så att vi efter en tid kan se vilken som "
                "fungerar bäst — per komplex, per fas och dag för dag efter cykelvändningen.")
    c1, c2 = st.columns([1, 3])
    if c1.button("📡 Hämta senaste körningen", key="paper_refresh") or _STATE not in st.session_state:
        with st.spinner("Läser papperskontona ur Gisten …"):
            st.session_state[_STATE] = load(loader)
    data = st.session_state[_STATE]
    state = data.get("state") or {}
    c2.markdown(f"<div style='color:{DIM};font-size:0.78rem;padding-top:8px;'>Senaste körning "
                f"{state.get('last_run') or '—'} · hämtad {data.get('loaded')}</div>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Sammanställning", "Utveckling", "Affärer", "BERSERK-kontot"])
    with t1:
        render_paper(((data.get("board") or {}).get("paper")))
    with t2:
        metric = st.radio("Mått", ["Snitt R", "Årstakt % (enhet)", "Payoff", "Vinst % (stängda)"], horizontal=True,
                          key="paper_metric")
        key = {"Snitt R": "avg_r", "Årstakt % (enhet)": "unit_cagr", "Payoff": "payoff", "Vinst % (stängda)": "closed_win"}[metric]
        df = history_frame(data.get("history"), key)
        if df.empty:
            note("Historiken börjar med första Valhall-körningen efter den här uppdateringen — en punkt per vardag.")
        else:
            st.line_chart(df)
    with t3:
        st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;'>ÖPPNA POSITIONER OCH ORDER</div>",
                    unsafe_allow_html=True)
        pos = positions_table(state)
        st.dataframe(pos, hide_index=True, use_container_width=True) if len(pos) else note("Inga öppna positioner.")
        st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;'>STÄNGDA AFFÄRER</div>",
                    unsafe_allow_html=True)
        cl = closed_table(state)
        st.dataframe(cl, hide_index=True, use_container_width=True) if len(cl) else note("Inga stängda affärer än.")
        with st.expander("Händelser (senaste 100)"):
            ev = events_table(state)
            st.dataframe(ev, hide_index=True, use_container_width=True) if len(ev) else note("Inga händelser.")
    with t4:
        try:
            from berserk.screen_ui import render_paper_tab
            render_paper_tab()
        except Exception as exc:
            note(f"BERSERK-kontot kunde inte visas: {exc}")
    note("En enhet per signal, ingen kassa (BERSERK har ett eget kassakonto). Riktiga order läggs alltid manuellt.")
