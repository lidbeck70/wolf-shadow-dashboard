"""
portfolio/mot_malet.py — 💼 PORTFÖLJ → Mot målet (designförslaget PR 7, avsnitt 7): hur det går mot CAGR-målet.

Fem delar:
  1. Mätaren         realiserad årstakt ur journalen mot golvet 20 % och målet 35 % (beslut 2)
  2. Per strategi    papperskontona (valhall/paper.py, samma siffror som 📊 Papperskonton) bredvid journalen
                     — stängda affärer x/20, vinstandel, snitt-R, payoff och enhetskurvans årstakt
  3. Plan följd      andelen affärer enligt plan (fältet followed_plan, beslut 4) — Trade Journal och
                     Copilotens stängda poster; en affär som står i båda räknas en gång
  4. Historiskt      länk till Valhalls backtester
  5. Länk            till 📊 Papperskonton

Under 20 stängda affärer står "för få affärer" i stället för en slutsats. Sidan räknar inga nya regler; den
läser journalen, Copilotens journal (bara för Plan följd) och papperet. Strategi, journal och papper kopplas via
Thors register (playbook och paper).
"""
from __future__ import annotations

from typing import Optional

GOAL_FLOOR, GOAL_TARGET = 20.0, 35.0       # beslut 2: golv 20 %, mål 35 % per år
MIN_CLOSED = 20                            # journal_stats.MIN_TRADES och valhall.paper.MIN_CLOSED


def zone(cagr: Optional[float]) -> tuple:
    """(ord, färgnyckel) för årstakten mot målet: MÅLET · GOLVET · UNDER · MINUS · Saknas."""
    if cagr is None:
        return "Saknas", "dim"
    if cagr >= GOAL_TARGET:
        return "MÅLET", "green"
    if cagr >= GOAL_FLOOR:
        return "GOLVET", "cyan"
    if cagr >= 0:
        return "UNDER", "gold"
    return "MINUS", "red"


def strategy_rows(paper: Optional[dict], trades: Optional[list]) -> list:
    """En rad per Thor-strategi med papperets och journalens mått sida vid sida. Strategier utan någon affär hoppas."""
    import journal_stats as js
    from thor import registry as reg
    groups = paper.get("by_strategy") if isinstance(paper, dict) else None
    by_paper = {g.get("strategy"): g for g in (groups if isinstance(groups, list) else []) if isinstance(g, dict)}
    rows = []
    for key, s in reg.STRATEGIES.items():
        p = by_paper.get(s.paper) if s.paper else None
        j = [t for t in trades or [] if isinstance(t, dict)
             and str(t.get("strategy") or "").lower() in {s.playbook.lower(), key}] if s.playbook else []
        js_sum = js.summary(j) if j else None
        if not p and not js_sum:
            continue
        pc = int((p or {}).get("closed") or 0)
        jc = int((js_sum or {}).get("closed") or 0)
        rows.append({
            "Strategi": s.label,
            "Papper stängda": f"{pc}/{MIN_CLOSED}" if p else "—",
            "Papper vinst %": (p or {}).get("closed_win") if pc >= MIN_CLOSED else None,
            "Papper snitt-R": (p or {}).get("avg_r") if pc >= MIN_CLOSED else None,
            "Papper payoff": (p or {}).get("payoff") if pc >= MIN_CLOSED else None,
            "Papper årstakt %": (p or {}).get("unit_cagr") if pc >= MIN_CLOSED else None,
            "Journal stängda": f"{jc}/{MIN_CLOSED}" if js_sum else "—",
            "Journal vinst %": round(js_sum["win_rate"], 1) if js_sum and js_sum.get("enough") and js_sum.get("win_rate") is not None else None,
            "Journal snitt-R": round(js_sum["avg_r"], 2) if js_sum and js_sum.get("enough") and js_sum.get("avg_r") is not None else None,
            "Slutsats": "för få affärer" if max(pc, jc) < MIN_CLOSED else "räcker för slutsats",
        })
    return rows


def _plan(t: dict) -> Optional[bool]:
    """followed_plan som True/False, eller None när fältet inte är ifyllt. "Ja"/"Nej" och True/False läses."""
    m = {"JA": True, "NEJ": False, True: True, False: False}
    v = t.get("followed_plan")
    try:
        v = m.get(v.strip().upper() if isinstance(v, str) else v)
    except TypeError:                      # en lista eller dict i fältet är inget svar
        return None
    return v if isinstance(v, bool) else None


def _same_trade(t: dict, id_key: str, day_key: str) -> set:
    """Nycklarna en affär känns igen på i båda journalerna: positionen i registret (journalens position_id =
    Copilotens holdings_id) och aktie + köpdag (journalens entry_date = Copilotens date)."""
    keys = set()
    if t.get(id_key):
        keys.add(("pos", str(t[id_key])))
    ticker, day = str(t.get("ticker") or "").strip().upper(), str(t.get(day_key) or "")[:10]
    if ticker and day:
        keys.add(("dag", ticker, day))
    return keys


def plan_followed(trades: Optional[list], copilot: Optional[list] = None) -> Optional[float]:
    """Andelen affärer med followed_plan = ja, av dem där fältet är ifyllt. None utan ifyllda.

    Läser Trade Journal (trades) och Copilotens STÄNGDA poster (copilot, data/copilot_journal.json). Båda sparar
    "Ja"/"Nej"; True/False läses också. En affär som står i båda — samma position, eller samma aktie och köpdag —
    räknas en gång: journalens svar gäller, och Copilotens bara när journalens rad saknar svar."""
    vals, answered = [], set()
    for t in trades or []:
        if isinstance(t, dict) and _plan(t) is not None:
            vals.append(_plan(t))
            answered |= _same_trade(t, "position_id", "entry_date")
    for e in copilot or []:
        if not isinstance(e, dict) or not e.get("exit_price") or _plan(e) is None:
            continue                       # öppen post, eller frågan obesvarad
        if _same_trade(e, "holdings_id", "date") & answered:
            continue                       # samma affär har redan ett svar i journalen
        vals.append(_plan(e))
    return round(sum(vals) / len(vals) * 100, 1) if vals else None


def copilot_entries() -> list:
    """Copilotens journalposter ur sessionen (laddas en gång per session, som på Copilot-sidan). Tom lista när
    lagret inte går att läsa — läsfelet visas på Copilot-sidan, och här räknas då bara Trade Journal."""
    try:
        import storage
        data = storage.session_load("copilot_journal", {"entries": []})
    except Exception:
        return []
    entries = data.get("entries") if isinstance(data, dict) else None
    return list(entries) if isinstance(entries, list) else []


def today_line(cagr: Optional[float], paper: Optional[dict], trades: Optional[list]) -> str:
    """Startsidans rad: 'Papper +24 % · journalen: för få affärer (4/20)'."""
    import journal_stats as js
    groups = paper.get("by_strategy") if isinstance(paper, dict) else None
    best = [g.get("unit_cagr") for g in (groups if isinstance(groups, list) else [])
            if isinstance(g, dict) and g.get("unit_cagr") is not None and int(g.get("closed") or 0) >= MIN_CLOSED]
    pap = f"Papper bästa {max(best):+.0f} %" if best else "Papper: för få stängda affärer"
    n = len(js._closed(trades or []))
    jr = f"journalen {cagr:+.1f} %" if cagr is not None and n >= MIN_CLOSED else f"journalen: för få affärer ({n}/{MIN_CLOSED})"
    return f"{pap} · {jr}"


def load() -> dict:
    out = {}
    try:
        from gist_storage import load_blob
        out["paper"] = (load_blob("valhall_board.json", None) or {}).get("paper")
    except Exception:
        out["paper"] = None
    try:
        from trade_journal import load_journal
        out["trades"] = load_journal()
    except Exception:
        out["trades"] = []
    return out


def render_mot_malet_page() -> None:
    import pandas as pd
    import streamlit as st

    import journal_stats as js
    from thor import cards
    from ui.components import note, page_header
    from ui.links import go
    from ui.tokens import CYAN, DIM, GOLD, GREEN, RED
    page_header("🎯 Mot målet", f"Golv {GOAL_FLOOR:.0f} % och mål {GOAL_TARGET:.0f} % per år — ur journalen och "
                "papperskontona. Under 20 stängda affärer dras ingen slutsats.")
    from trade_journal import journal_unreadable, start_capital
    if st.button("📡 Uppdatera", key="goal_refresh") or "goal_data" not in st.session_state or journal_unreadable():
        st.session_state["goal_data"] = load()
    d = st.session_state["goal_data"]
    blocked = journal_unreadable()
    if blocked:                            # tom journal här betyder "gick inte att läsa", inte "inga affärer"
        st.error(f"Journalen gick inte att läsa ur Gisten ({blocked}) — journalens siffror nedan saknas. "
                 f"Ladda om sidan om en stund.")
    trades = d.get("trades") or []
    cagr = js.realized_cagr(trades, start_capital())       # journalens startkapital (beslut 5), som Journal-sidan
    word, ck = zone(cagr)
    col = {"green": GREEN, "cyan": CYAN, "gold": GOLD, "red": RED, "dim": DIM}[ck]
    n = len(js._closed(trades))
    plan = plan_followed(trades, copilot_entries())        # sessionens Copilot-journal: dina svar syns direkt
    st.markdown(cards.kpi_row([
        ("Journalens årstakt", f"{cagr:+.1f} %" if cagr is not None else "—", col,
         f"{word} · {n}/{MIN_CLOSED} stängda" + (" — för få affärer" if n < MIN_CLOSED else "")),
        ("Golv / mål", f"{GOAL_FLOOR:.0f} % / {GOAL_TARGET:.0f} %", CYAN, "ditt mål, ingen strategiregel"),
        ("Plan följd", f"{plan:.0f} %" if plan is not None else "—", CYAN,
         "andel affärer enligt plan — Trade Journal och Copilotens stängda poster, samma affär en gång")]),
        unsafe_allow_html=True)
    pct = max(0.0, min(50.0, cagr or 0.0)) / 50 * 100
    st.markdown(f"<div style='position:relative;height:12px;background:#1a1f25;border-radius:6px;margin:6px 0 2px;'>"
                f"<div style='width:{pct:.0f}%;height:12px;background:{col};border-radius:6px;'></div>"
                f"<div style='position:absolute;left:{GOAL_FLOOR / 50 * 100:.0f}%;top:-3px;height:18px;border-left:2px solid {CYAN};'></div>"
                f"<div style='position:absolute;left:{GOAL_TARGET / 50 * 100:.0f}%;top:-3px;height:18px;border-left:2px solid {GREEN};'></div>"
                f"</div><div style='color:{DIM};font-size:0.7rem;'>0 % · golv {GOAL_FLOOR:.0f} · mål {GOAL_TARGET:.0f} · 50 %</div>",
                unsafe_allow_html=True)
    note("Journalens årstakt räknas till i dag på startkapitalet och bara på stängda affärer. Öppna positioner räknas "
         "inte in — kontovärdet över tid finns inte i panelen.")
    st.markdown("#### Per strategi — papper och journal")
    rows = strategy_rows(d.get("paper"), trades)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        note("Inga affärer än — varken på papper eller i journalen.")
    st.button("📊 Papperskonton →", key="goal_paper", on_click=go, args=("paper",), width="stretch")
    st.markdown("#### Vad reglerna gett historiskt")
    if st.toggle("Visa Valhalls backtester (2008–idag)", key="goal_backtests"):
        try:
            from valhall.ui import render_valhall_page
            render_valhall_page("Backtester")
        except Exception as exc:
            note(f"Backtesterna kunde inte visas: {exc}")
