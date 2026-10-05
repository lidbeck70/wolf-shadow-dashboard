"""
ovtlyr/ui/viking_robustness_ui.py — ROBUSTHET under Viking Nine-backtestets resultat.

Försöker fälla resultatet (viking_robustness): konfidensintervall för
expectancy, hur mycket de fem bästa affärerna bär, strategin mot köp och
behåll av indexet, kostnadskänslighet, Monte Carlo på de faktiska affärerna
och var kanten kommer ifrån. Räknas en gång per körning och sparas i resultatet.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import viking_backtest as vb
import viking_portfolio as vp
import viking_robustness as rb
from ui.components import big_card, note
from ui.tokens import AMBER, BG, CYAN, DIM, GOLD, GREEN, RED, TEXT

_LISTS = (("Norden 50", set(vb.NORDIC_50)), ("Norden OOS 100", set(vb.NORDIC_OOS_100)), ("USA 25", set(vb.US_25)))


def list_of(ticker: str) -> str:
    """Vilken fast lista tickern hör till — 'Norden OOS 100' är aktier reglerna aldrig sett."""
    return next((name for name, members in _LISTS if ticker in members), "Egna")


def _years(res: dict):
    return (res.get("period") or {}).get("years") or getattr(res.get("config"), "years", None)


def robustness_of(res: dict, portfolio: dict, pc: vp.PortfolioConfig = None) -> dict:
    """Allt robusthetsmaterial för en körning (räknas en gång). pc = strategins portföljregler — samma som
    portföljkortet, så att kostnadstabellens portföljkolumner räknas med rätt spärrar (förval: Vikings)."""
    pc = pc or vp.PortfolioConfig()
    if (res.get("robustness") or {}).get("pc") != repr(pc):
        trades = res.get("trades") or []
        years = _years(res)
        res["robustness"] = {"pc": repr(pc),
            "ci": rb.expectancy_ci(trades), "conc": rb.concentration(trades),
            "bench": rb.strategy_vs_benchmarks(portfolio, res.get("benchmarks") or {}),
            "costs": rb.cost_table(trades, years, pc=pc), "be_cost": rb.breakeven_cost_bps(trades),
            "mc": rb.monte_carlo(portfolio.get("rows") or [], years),
            "breakdown": rb.breakdown(trades),
            "lists": rb.group(trades, lambda t: list_of(t.ticker)),
        }
    return res["robustness"]


STICKY = f"position:sticky;left:0;z-index:1;background:{BG};"     # radnamnet syns när tabellen scrollas i mobilen


def _table(rows: list, first: str) -> str:
    if not rows:
        return ""
    head = "".join(f"<th style='{STICKY if k == first else ''}'>{k}</th>" for k in rows[0])
    body = "".join("<tr>" + "".join(
        f"<td style='{'text-align:left;' + STICKY if k == first else ''}'>"
        f"{'—' if v is None else (f'{v:g}' if isinstance(v, float) else v)}</td>" for k, v in r.items()) + "</tr>"
        for r in rows)
    return (f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.74rem;color:{TEXT};"
            f"text-align:right;'><tr style='color:{DIM};'>{head}</tr>{body}</table></div>")


def _title(text: str, color=CYAN) -> None:
    st.markdown(f"<div style='color:{color};font-size:0.72rem;letter-spacing:0.1em;margin-top:12px;'>{text}</div>",
                unsafe_allow_html=True)


def render_robustness(res: dict, portfolio: dict, pc: vp.PortfolioConfig = None, lists: bool = True) -> None:
    """lists=False döljer Viking-listorna (Norden 50 / OOS 100) för strategier med eget universum."""
    if not (res.get("trades") or []):
        return
    r = robustness_of(res, portfolio, pc)
    _title("ROBUSTHET — FÖRSÖK FÄLLA RESULTATET", GOLD)
    ci, conc, be = r["ci"], r["conc"], r["be_cost"]
    cards = []
    if ci:
        cards.append(("EXPECTANCY 95 %-INTERVALL", f"{ci['low']:+.2f} … {ci['high']:+.2f}R",
                      f"{ci['n']} affärer · snitt {ci['mean']:+.2f}R · standardfel ±{ci['se']:.2f}R · "
                      f"P(≤ 0) {ci['p_le_zero']:g} %", GREEN if ci["low"] > 0 else AMBER if ci["mean"] > 0 else RED))
    if conc:
        share = conc["top_share_of_gross"]
        cards.append((f"DE {rb.TOP_N} BÄSTA AFFÄRERNA", f"{conc['top_r']:+.1f}R",
                      f"{'—' if share is None else f'{share:g}'} % av alla vinster · utan dem {conc['total_without_top']:+.1f}R "
                      f"(snitt {conc['expectancy_without_top'] if conc['expectancy_without_top'] is not None else '—'}R)",
                      AMBER if (share or 0) > 50 else CYAN))
    if be is not None:
        cards.append(("KOSTNAD DÅ KANTEN ÄR SLUT", f"≈ {be:g} bp", "tur och retur per affär (courtage + spread + "
                      "slippage) där expectancy blir 0 — svensk nätmäklare ≈ 10–30 bp", GREEN if be >= 50 else AMBER
                      if be >= 20 else RED))
    if cards:
        cols = st.columns(len(cards))
        for k, (t, big, sub, col) in enumerate(cards):
            cols[k].markdown(big_card(t, big, sub, col), unsafe_allow_html=True)

    if r["bench"]:
        _title("MOT KÖP OCH BEHÅLL (SAMMA PERIOD)")
        st.markdown(_table(r["bench"], "Vad"), unsafe_allow_html=True)
        note("Indexen är prisindex utan utdelning (≈ 2–4 % per år lägre än avkastningsindex). Strategin står "
             "ofta i kontanter — jämför CAGR/DD, inte bara CAGR.")

    lists = [x for x in r["lists"] if x["Grupp"] != "Egna"] if lists else []
    if lists:
        _title("PER LISTA — NORDEN OOS 100 ÄR AKTIER REGLERNA ALDRIG SETT")
        st.markdown(_table(sorted(lists, key=lambda x: x["Grupp"]), "Grupp"), unsafe_allow_html=True)

    _title("KOSTNADER — NÄR FÖRSVINNER KANTEN?")
    st.markdown(_table(r["costs"], "Kostnad bp"), unsafe_allow_html=True)
    note(f"Kostnad i R = kostnad i % / stoppavstånd i %. Gap-slippage = extra {rb.GAP_SLIPPAGE_BPS} bp på stoppar "
         "(gap och snabba fall). Snittinnehavet är kort, så kostnaden per affär väger tungt.")

    if r["mc"]:
        _title(f"MONTE CARLO — {rb.MC_SIMS:,} SIMULERINGAR PÅ PORTFÖLJENS AFFÄRER".replace(",", " "))
        st.markdown(_table(r["mc"], "Metod"), unsafe_allow_html=True)
        note("Bootstrap drar affärer med återläggning; block-bootstrap drar fem affärer i följd så att dåliga "
             "perioder hänger ihop; slumpad ordning visar drawdown-risken med exakt samma affärer. Drawdown räknas "
             "på stängda affärer — den dagsvärderade är högre, så se percentilerna som en undre gräns. Planera "
             "kapitalet efter DD p95, inte efter historisk max DD.")

    bd = r["breakdown"]
    if bd:
        _title("VAR KOMMER KANTEN IFRÅN?")
        for k, (title, rows) in enumerate(bd.items()):
            with st.expander(f"{title} ({len(rows)} grupper)", expanded=k == 0):
                st.markdown(_table(rows, "Grupp"), unsafe_allow_html=True)
        note("Mått på signaldagen: ATR % = ATR14 / kurs · CLV = var i dagens spann aktien stängde (1 = på högsta) · "
             "RS63 = aktiens 63-dagarsavkastning minus indexets · gap = nästa dags öppning mot stängningen i ATR. "
             "En grupp med få affärer är brus — leta efter mönster som håller i flera perioder och listor.")
