"""
ovtlyr/ui/execution_card.py — VIKING EXECUTION · RISK ENGINE · FINAL DECISION
på Viking Regime, under OVTLYR Nine-kortet.

Beslutskortet (GOLDEN TICKET / WAIT / NO TRADE), exekveringsfiltren med
✓ / ✗ / ?, och riskmotorn: kapital, risk, ATR, stopp, antal aktier, risk i
kronor och exponering — risk och exponering visas var för sig.
"""

from __future__ import annotations

import streamlit as st

import viking_execution as vx
from ui.components import note
from ui.tokens import AMBER, BG_CARD, BORDER, CYAN, DIM, GOLD, GREEN, RED, TEXT

_STATUS_COLOR = {vx.GO: GOLD, vx.WAIT: AMBER, vx.NO_TRADE: RED}


def _mark(c: vx.Check) -> tuple:
    if c.status == vx.PASS:
        return "✓", GREEN
    if c.status == vx.FAIL:
        return "✗", RED if c.required else AMBER
    return "?", DIM


def decision_html(d: vx.EntryDecision) -> str:
    col = _STATUS_COLOR.get(d.status, DIM)
    nine = "—" if d.nine_passed is None else f"{d.nine_passed}/{d.nine_total}"
    rows = "".join(
        f"<div style='font-size:0.8rem;color:{TEXT};line-height:1.55;'>"
        f"<span style='color:{c};font-weight:700;'>{m}</span> <b>{ch.label}</b>"
        + (f" <span style='color:{AMBER};font-size:0.7rem;'>{ch.flag}</span>" if ch.flag else "")
        + ("" if ch.required else f" <span style='color:{DIM};font-size:0.68rem;'>(info)</span>")
        + f"<div style='color:{DIM};font-size:0.7rem;margin-left:16px;'>{ch.detail}</div></div>"
        for ch in d.checks for m, c in [_mark(ch)])
    why = "".join(f"<li>{r}</li>" for r in d.reasons)
    label = {vx.WAIT: "Missing / väntar på", vx.NO_TRADE: "Failed"}.get(d.status, "Varför")
    return (
        f"<div style='border:2px solid {col};border-radius:8px;padding:10px 12px;margin:8px 0;'>"
        f"<div style='display:flex;flex-wrap:wrap;gap:14px;align-items:baseline;'>"
        f"<div style='color:{col};font-size:1.3rem;font-weight:800;letter-spacing:0.1em;'>{d.status}</div>"
        f"<div style='color:{TEXT};'>OVTLYR Nine <b>{nine}</b></div>"
        f"<div style='color:{TEXT};'>Viking Execution <b>{d.execution_passed}/{d.execution_total}</b></div>"
        + "".join(f"<div style='color:{AMBER};font-size:0.75rem;border:1px solid {AMBER};border-radius:4px;"
                  f"padding:0 6px;'>{f}</div>" for f in d.flags)
        + f"</div><div style='color:{DIM};font-size:0.75rem;margin-top:6px;'>{label}:</div>"
        f"<ul style='color:{TEXT};font-size:0.78rem;margin:2px 0 6px 18px;padding:0;'>{why}</ul>"
        f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:6px;'>VIKING EXECUTION</div>"
        f"<div style='display:flex;flex-direction:column;gap:2px;'>{rows}</div></div>")


def risk_html(d: vx.EntryDecision) -> str:
    p = d.position
    if p is None:
        return f"<div style='color:{DIM};'>Riskmotorn kräver kapital, entry och ATR &gt; 0.</div>"
    cells = [
        ("Kapital", f"{p.capital:,.0f}"), ("Riskbudget", f"{p.max_risk_pct:g} % = {p.risk_budget:,.0f}"),
        ("ATR14", f"{p.atr:,.2f}"), ("Stoppavstånd", f"{vx.ATR_STOP_MULT:g} × ATR = {p.stop_distance:,.2f}"),
        ("Entry", f"{p.entry:,.2f}"), ("Stopp", f"{p.stop:,.2f}"),
        ("Aktier", f"{p.shares}" + (f" (risken tillåter {p.shares_by_risk})" if p.capped_by_exposure else "")),
        ("Positionsvärde", f"{p.position_value:,.0f}"),
        ("Exponering", f"{p.exposure_pct:g} % (tak {p.max_position_pct:g} %)"),
        ("Risk i kronor", f"{p.risk_amount:,.0f} ({p.risk_pct:g} %)"),
        ("Motstånd", "—" if d.resistance is None else f"{d.resistance:,.2f}"),
        ("R/R", "fri väg" if d.rr is None and d.resistance is None else "—" if d.rr is None else f"{d.rr:.2f}R"),
    ]
    grid = "".join(f"<div style='flex:1 1 130px;'><div style='color:{DIM};font-size:0.68rem;'>{k}</div>"
                   f"<div style='color:{TEXT};font-size:0.9rem;font-weight:600;'>{v}</div></div>" for k, v in cells)
    return (f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;padding:10px 12px;'>"
            f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-bottom:6px;'>RISK ENGINE</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:10px;'>{grid}</div></div>")


def render_execution(d: vx.EntryDecision) -> None:
    st.markdown(decision_html(d), unsafe_allow_html=True)
    st.markdown(risk_html(d), unsafe_allow_html=True)
    p = d.position
    if p is not None and p.capped_by_exposure:
        note(f"Begränsad av exponering: risken på {p.max_risk_pct:g} % hade tillåtit {p.shares_by_risk} aktier "
             f"({p.shares_by_risk * p.entry:,.0f}), men en position får vara högst {p.max_position_pct:g} % av "
             f"kapitalet. Risk = avståndet till stoppen × antal aktier; exponering = positionens värde.")
    note("GOLDEN TICKET kräver OVTLYR Nine 9/9 och att alla krav i Viking Execution är klara: momentum, "
         f"trendstruktur, stängd grön triggercandle, högst {vx.MAX_CHASE_PCT:g} % över triggern, R/R ≥ "
         f"{vx.MINIMUM_RR:g}, ingen rapport inom {vx.EARNINGS_BUFFER_DAYS} handelsdagar och färre än "
         f"{vx.MAX_DAILY_LOSSES} förlustaffärer i dag. Volym visas men krävs inte förrän den är backtestad. "
         "Appen förutsäger ingenting — den räknar villkor.")
