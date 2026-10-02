"""
ovtlyr/ui/exit_card.py — VIKING EXIT ENGINE på Viking Regime.

Fyll i entry (eller låt en öppen Viking-position i Holdings fylla i den):
kortet visar status (HOLD / EXIT / CLOSE ALL), initial stopp, nuvarande
stopp och trailing-stopp (EMA10), R just nu och varje exitregel med sin
uträkning.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

import viking_exit as vex
from ui.components import note
from ui.tokens import AMBER, BG_CARD, BORDER, CYAN, DIM, GREEN, RED, TEXT

_STATUS_COLOR = {vex.HOLD: GREEN, vex.EXIT: RED, vex.CLOSE_ALL: RED}


def _mark(t: vex.Trigger) -> tuple:
    if t.status == vex.ACTIVE:
        return "✗", RED
    if t.status == vex.CLEAR:
        return "✓", GREEN
    return "?", DIM


def exit_html(d: vex.ExitDecision) -> str:
    col = _STATUS_COLOR.get(d.status, AMBER)

    def _v(x, fmt="{:,.2f}"):
        return "—" if x is None else fmt.format(x)
    stops = [("Initial SL", _v(d.initial_stop)), ("Current SL", _v(d.current_stop)
                                                   + (" (breakeven)" if d.breakeven_armed else "")),
             ("Trailing (EMA10)", _v(d.trailing_stop)), ("Kurs", _v(d.price)),
             ("R nu", _v(d.r_now, "{:+.2f}R")),
             ("F&G-target", "—" if d.fg_target is None else f"{d.fg_target:g} (entry {d.fg_entry:g})")]
    grid = "".join(f"<div style='flex:1 1 110px;'><div style='color:{DIM};font-size:0.68rem;'>{k}</div>"
                   f"<div style='color:{TEXT};font-weight:600;'>{v}</div></div>" for k, v in stops)
    rows = "".join(
        f"<div style='font-size:0.8rem;color:{TEXT};line-height:1.55;'><span style='color:{c};font-weight:700;'>"
        f"{m}</span> <b>{t.label}</b> <span style='color:{DIM};font-size:0.72rem;'>— {t.detail}</span></div>"
        for t in d.triggers for m, c in [_mark(t)])
    why = "".join(f"<li>{r}</li>" for r in d.reasons)
    return (f"<div style='border:2px solid {col};border-radius:8px;padding:10px 12px;margin:8px 0;"
            f"background:{BG_CARD};'><div style='display:flex;flex-wrap:wrap;gap:14px;align-items:baseline;'>"
            f"<div style='color:{col};font-size:1.2rem;font-weight:800;letter-spacing:0.1em;'>{d.status}</div>"
            f"<div style='color:{DIM};'>entry {d.entry:,.2f} · {d.entry_date}</div></div>"
            f"<ul style='color:{TEXT};font-size:0.78rem;margin:4px 0 6px 18px;padding:0;'>{why}</ul>"
            f"<div style='display:flex;flex-wrap:wrap;gap:10px;border-top:1px solid {BORDER};padding-top:6px;'>"
            f"{grid}</div><div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:8px;'>"
            f"EXITREGLER</div>{rows}</div>")


def _open_viking_position(ticker: str):
    try:
        import positions
        row = positions.find(ticker)
        return row if row and row.get("strategy") == "Viking" and row.get("entry_price") else None
    except Exception:
        return None


def render_exit_section(ticker: str, frame, nine, ob_analysis: dict, earnings_date=None) -> None:
    st.markdown(f"<div style='color:{CYAN};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;"
                f"margin:14px 0 6px 0;'>VIKING EXIT ENGINE — öppen position</div>", unsafe_allow_html=True)
    pos = _open_viking_position(ticker)
    default_entry = float(pos["entry_price"]) if pos else 0.0
    try:
        default_date = pd.Timestamp(pos["entry_date"]).date() if pos and pos.get("entry_date") else date.today()
    except Exception:
        default_date = date.today()
    c1, c2, c3 = st.columns(3)
    entry = c1.number_input("Entrypris", min_value=0.0, value=default_entry, step=0.5, key=f"vxe_entry_{ticker}")
    entry_date = c2.date_input("Entrydatum", value=default_date, key=f"vxe_date_{ticker}")
    be = c3.selectbox("Stopp flyttad till breakeven?", ["Auto (ny högre topp)", "Ja", "Nej"],
                      key=f"vxe_be_{ticker}")
    if pos:
        note(f"Ifyllt från din öppna Viking-position i Holdings ({pos['shares']} st à {pos['entry_price']:g}).")
    if not entry or frame is None:
        note("Fyll i entrypris och entrydatum för en öppen Viking-position så räknas varje exitregel.")
        return
    be_moved = None if be.startswith("Auto") else be == "Ja"
    d = vex.evaluate_exit(ticker, frame, entry, entry_date, nine=nine, ob_analysis=ob_analysis,
                          earnings_date=earnings_date, be_moved=be_moved)
    st.markdown(exit_html(d), unsafe_allow_html=True)
    note("HARD MARKET EXIT (SPY under EMA20) stänger alla positioner. Övriga regler stänger den här. "
         "Breakeven: stoppen flyttas till entry efter en ny högre topp; därefter är stängning under "
         "gårdagens low exit. F&G-exit: 0–50 vid entry → exit vid 63, 50–75 → +10, 75+ → +5.")
