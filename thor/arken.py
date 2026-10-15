"""
thor/arken.py — Arken-fliken i Thors analysblock: vad arken säger om bolaget, bara läsning.

  Granskningarna   Rick Rule / Royalty / Poängmodell / Tiggre / Insider → PASS/MANUAL/FAIL och underlaget
  Motorerna        Confidence-raden, Durrett-motorn (kvalitet, risk, uppsida, röda flaggor, katalysatorer) och Wolf
                   Asymmetry (hävstång, säkerhetsmarginal, break-even) — räknade ur Confidence/Durrett-arket
  Scorecardet      köpgrindens sju kryss, luckorna och READY/NO TRADE
  Insiderklustret  dagens rad ur insider_scan (poäng, status, insiders, roll, kurs, stopp)
  Registret        öppna och stängda positioner, journalens affärer och Copilotens loggade anteckningar
"Finns inte i arket → lägg in" med knappar till arken. Ingenting skrivs; samma data som arken själva visar.
"""

from __future__ import annotations

from datetime import date
from html import escape
from typing import Callable, Optional

import pandas as pd

from thor import cards
from thor import dossier as td
from thor import registry as reg
from thor.ticker import Ticker
from ui.tokens import AMBER, CYAN, DIM, GREEN, GREY, RED, TEXT


def _title(st, text: str) -> None:
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:10px;'>{escape(text)}</div>",
                unsafe_allow_html=True)


def _block(name: str, fn: Callable, note: Callable) -> None:
    try:
        fn()
    except Exception as exc:
        note(f"{name} kunde inte läsas: {exc}")


def insider_cluster(blob: Optional[dict], ticker: str) -> Optional[dict]:
    """Dagens insiderrad för aktien ur insider_scan.json (någon av tickerns former), eller None."""
    forms = set(td.forms(ticker))
    for c in (blob or {}).get("clusters") or []:
        if isinstance(c, dict) and str(c.get("ticker") or "").strip().upper() in forms:
            return c
    return None


def company_for(store: Optional[dict], ticker: str):
    """CompanyInput ur Confidence/Durrett-arket för någon av tickerns former, eller None."""
    from confidence import store as cs
    data = cs.normalize(dict(store or {}))
    for f in td.forms(ticker):
        c = cs.get(data, f)
        if c is not None:
            return c
    return None


def render(st, data: dict, ident: Ticker, sheets: dict, cands: list, links: dict, jump: Callable) -> None:
    from ui.components import note
    if not sheets.get("found"):
        note(f"{ident.yf} finns inte i något ark än. Ordningen är screener → granskning → köp: lägg in bolaget i "
             f"rätt ark först.")
    _block("Granskningarna", lambda: _reviews(st, sheets), note)
    if sheets.get("confidence"):
        _title(st, "CONFIDENCE · DURRETT · WOLF ASYMMETRY")
        _block("Motorerna", lambda: _engines(st, ident, sheets, note), note)
    if sheets.get("scorecard"):
        _title(st, "SCORECARDET")
        _block("Scorecardet", lambda: _scorecard(st, ident, sheets), note)
    cluster = insider_cluster((data or {}).get("insider"), ident.yf)
    if cluster:
        _title(st, "INSIDERKLUSTRET")
        _block("Insiderklustret", lambda: _insider(st, cluster), note)
    _block("Registret", lambda: _register(st, sheets, note), note)
    cols = st.columns(3)
    for i, (key, path) in enumerate(links.items()):
        if key in (sheets.get("reviews") or {}) or i < 3:
            cols[i % 3].button(f"{reg.label(key) if reg.get(key) else key.title()} →", key=f"thor_sheet_{key}",
                               on_click=jump, args=(path, None, ident.yf), width="stretch")


def _reviews(st, sheets: dict) -> None:
    for key, rev in (sheets.get("reviews") or {}).items():
        st.markdown(cards.review_card(key, rev), unsafe_allow_html=True)
        details = []
        try:
            import review_link
            details = review_link.detail_lines(key, rev.get("row") or {})
        except Exception:
            details = []
        if details:
            with st.expander(f"Underlaget i {rev.get('sheet') or key}", expanded=False):
                for line in details:
                    st.caption(line.strip())


def _engines(st, ident: Ticker, sheets: dict, note) -> None:
    conf = sheets.get("confidence") or {}
    st.markdown(cards.chip(f"{conf.get('name') or ident.yf} · {conf.get('commodity') or '—'} · {conf.get('stage') or ''}".strip(" ·"),
                           AMBER, filled=True), unsafe_allow_html=True)
    try:
        import scorecard
        line = scorecard.engine_summary(ident.yf)
    except Exception:
        line = None
    if line:
        st.markdown(f"<div style='color:{TEXT};font-size:0.8rem;margin:4px 0;'>{escape(line)}</div>", unsafe_allow_html=True)
    company = company_for(sheets.get("confidence_store"), ident.yf)
    if company is None:
        note("Bolaget finns i arket men raden gick inte att läsa som CompanyInput.")
        return
    items = []
    try:
        from engines.durrett.engine import analyze as durrett_analyze
        d = durrett_analyze(company, today=date.today())
        q = getattr(d.quality_score, "value", None)
        items.append(("Durrett kvalitet", f"{q:.0f}/100" if q is not None else "—",
                      GREY if q is None else GREEN if q >= 70 else AMBER if q >= 50 else RED,
                      f"risk {d.risk_level}" + (f" · data {d.data_confidence:.0f} %" if d.data_confidence is not None else "")))
        up = d.upside_multiple
        items.append(("Uppsida", f"{up:.1f}×" if up is not None else "—", GREEN if up and up >= 3 else AMBER if up else GREY,
                      " · ".join(x for x in (f"FV {d.fair_value_per_share:,.2f}" if d.fair_value_per_share is not None else "",
                                             f"{d.potential_upside_pct:+.0f} %" if d.potential_upside_pct is not None else "") if x)))
        flags = [f"{getattr(f, 'flag', f)} ({getattr(f, 'severity', '')})".replace(" ()", "") for f in d.red_flags or []]
        cats = [f"{getattr(c, 'name', c)}" + (f" · {getattr(c, 'expected', '')}" if getattr(c, "expected", "") else "") for c in d.catalysts or []]
    except Exception as exc:
        flags, cats = [], []
        items.append(("Durrett", "—", GREY, f"motorn: {exc}"))
    try:
        import asymmetry
        a = asymmetry.analyze(company)
        lev, saf, be = a.leverage, a.safety, a.break_even
        items.append(("Hävstång", f"{lev.score}/{lev.max}" if lev.score is not None else "—", AMBER, str(lev.metric or "")))
        items.append(("Säkerhetsmarginal", f"{saf.total:.0f}/{saf.max:.0f}" if saf.total is not None else "—", CYAN,
                      f"mätbart {saf.measurable_max:.0f}" if saf.measurable_max is not None else ""))
        items.append(("Break-even", f"{be.margin_pct:+.0f} %" if be.margin_pct is not None else "—",
                      GREEN if be.margin_pct is not None and be.margin_pct >= 30 else AMBER, str(be.band or "")))
        if a.base_upside_pct is not None:
            items.append(("Basuppsida", f"{a.base_upside_pct:+.0f} %", GREEN if a.base_upside_pct > 0 else RED,
                          f"justerad {a.adjusted_upside_pct:+.0f} %" if a.adjusted_upside_pct is not None else ""))
    except Exception as exc:
        items.append(("Wolf Asymmetry", "—", GREY, f"motorn: {exc}"))
    st.markdown(cards.kpi_row(items), unsafe_allow_html=True)
    if flags:
        st.markdown("".join(f"<div style='font-size:0.78rem;color:{TEXT};'><span style='color:{RED};'>⚑</span> {escape(f)}</div>"
                            for f in flags[:8]), unsafe_allow_html=True)
    if cats:
        st.markdown("".join(f"<div style='font-size:0.78rem;color:{TEXT};'><span style='color:{GREEN};'>▸</span> {escape(c)}</div>"
                            for c in cats[:8]), unsafe_allow_html=True)


def _scorecard(st, ident: Ticker, sheets: dict) -> None:
    from scorecard import GATE_BY_KEY, gate_state
    for k, card in (sheets.get("scorecard") or {}).items():
        g = gate_state(card, {"ticker": ident.yf})
        word = "READY" if g.get("ready") else "NO TRADE"
        st.markdown(cards.chip(f"Scorecard: {word} · {card.get('strategy') or ''}".strip(" ·"), GREEN if g.get("ready") else AMBER,
                               filled=True), unsafe_allow_html=True)
        checks = {getattr(GATE_BY_KEY.get(key), "label", key): ("PASS" if ok else "FAIL", "") for key, ok in (g.get("checks") or {}).items()}
        if checks:
            st.markdown(cards.check_rows(checks), unsafe_allow_html=True)
        gaps = list(g.get("gaps") or []) + list(g.get("valuation_gaps") or []) + list(g.get("position_gaps") or [])
        if gaps:
            st.markdown("".join(f"<div style='font-size:0.76rem;color:{DIM};'>· {escape(str(x))}</div>" for x in gaps[:8]),
                        unsafe_allow_html=True)


def _insider(st, c: dict) -> None:
    score = c.get("score")
    status = str(c.get("status") or "")
    col = GREEN if status.upper().startswith("KÖP") and not c.get("chase") else AMBER
    items = [("Insider", f"{float(score):g}/10" if score is not None else "—", col, status),
             ("Insiders", f"{c.get('insiders') or '—'}", CYAN, str(c.get("role") or "")),
             ("Kurs nu", f"{c.get('price_now')}" if c.get("price_now") is not None else "—", TEXT,
              f"stopp {c.get('stop')}" if c.get("stop") is not None else ""),
             ("Klustersnitt", f"{c.get('cluster_avg')}" if c.get("cluster_avg") is not None else "—", AMBER,
              "passa — jagar kursen" if c.get("chase") else "")]
    st.markdown(cards.kpi_row(items), unsafe_allow_html=True)


def _register(st, sheets: dict, note) -> None:
    if sheets.get("positions"):
        _title(st, "POSITIONER")
        st.dataframe(pd.DataFrame([{"Hink": p.get("_bucket"), "Strategi": p.get("strategy"), "Antal": p.get("shares"),
                                    "Entry": p.get("entry_price"), "Datum": p.get("entry_date"), "Stopp": p.get("stop"),
                                    "Mål": p.get("target")} for p in sheets["positions"]]),
                     hide_index=True, use_container_width=True)
    if sheets.get("closed"):
        _title(st, "STÄNGDA POSITIONER")
        st.dataframe(pd.DataFrame([{"Strategi": p.get("strategy"), "Antal": p.get("shares"), "Entry": p.get("entry_price"),
                                    "In": p.get("entry_date"), "Exit": p.get("exit_price"), "Ut": p.get("exit_date"),
                                    "Skäl": p.get("exit_reason")} for p in sheets["closed"]]),
                     hide_index=True, use_container_width=True)
    if sheets.get("journal"):
        _title(st, "JOURNALEN")
        st.dataframe(pd.DataFrame([{"Strategi": t.get("strategy"), "In": t.get("entry_date"), "Ut": t.get("exit_date"),
                                    "Resultat %": t.get("pnl_pct"), "R": t.get("r_multiple"), "Skäl": t.get("exit_reason")}
                                   for t in sheets["journal"]]), hide_index=True, use_container_width=True)
    if sheets.get("copilot"):
        _title(st, "COPILOTENS ANTECKNINGAR")
        for e in sheets["copilot"][-5:]:
            parts = (e.get("date"), e.get("strategy"), f"entry {e.get('entry')}" if e.get("entry") else "",
                     f"stopp {e.get('stop')}" if e.get("stop") else "", f"mål {e.get('target')}" if e.get("target") else "")
            head = " · ".join(str(x) for x in parts if x)
            st.markdown(f"<div style='font-size:0.78rem;color:{TEXT};'><span style='color:{DIM};'>{escape(head)}</span>"
                        + (f" — {escape(str(e.get('note')))}" if e.get("note") else "") + "</div>", unsafe_allow_html=True)
