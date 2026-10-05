"""
berserk/ui.py — PORTFOLIO → Backtest → 🪓 BERSERK.

Kör berserk.backtest över råvaruaktier och råvaru-ETF:er och visar nyckeltalen
i R per setup, portföljen (max 8 positioner, 20 % per position, 2 per tema,
4 per komplex, 6 % öppen risk) med dagsvärderad drawdown, uppdelning per tema
och komplex, vilka drivare som användes och robusthetsverktygen från Viking
Nine (kostnader, Monte Carlo, kantanalys, köp och behåll).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import viking_backtest as vb
import viking_portfolio as vp
import viking_robustness as rb
import viking_screen as vs
from berserk import backtest as bt
from berserk import signals as sg
from berserk import themes as th
from berserk import universe as uv
from ovtlyr.ui.viking_robustness_ui import STICKY
from ui.components import big_card, note, page_header
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, RED, TEXT

_RES, _TICKERS = "bz_result", "bz_tickers"
_OWN = "Egen period"
_GLOBAL = "Utanför Norden (USA, Kanada, London, Australien)"
_ALL = "Allt (Norden + utanför Norden + ETF:er)"
LISTS = {**uv.LISTS, _GLOBAL: tuple(uv.GLOBAL), _ALL: tuple(uv.NORDIC) + tuple(uv.GLOBAL) + tuple(uv.ETFS)}
ALL_SETUPS = "Alla tre"


def _apply_list() -> None:
    st.session_state[_TICKERS] = ", ".join(LISTS.get(st.session_state.get("bz_list"), uv.LISTS["Norden"]))


def variants(setups: list, compare: bool) -> list:
    """[(namn, setups)] — vald kombination, eller varje setup för sig plus alla tre."""
    if not compare:
        return [(" + ".join(s.split(" ", 1)[0] for s in setups) or ALL_SETUPS, tuple(setups))]
    return [(s, (s,)) for s in sg.SETUPS] + [(ALL_SETUPS, sg.SETUPS)]


def portfolio_of(res: dict) -> dict:
    if "portfolio" not in res:
        res["portfolio"] = vp.simulate(res.get("trades") or [], years=(res.get("period") or {}).get("years"),
                                       pc=bt.portfolio_config())
    return res["portfolio"]


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


def comparison_rows(runs: dict) -> list:
    rows = []
    for name, res in runs.items():
        m, p = res["metrics"], portfolio_of(res)
        rows.append({"Körning": name, "Affärer": m.get("trades", 0), "Win rate %": m.get("win_rate"),
                     "Expectancy R": m.get("expectancy"), "Profit factor": m.get("profit_factor"),
                     "Summa R": m.get("total_r"), "Snittinnehav d": m.get("avg_holding_days"),
                     "Portfölj affärer": p["taken"], "Portfölj avk. %": p["return_pct"],
                     "Portfölj CAGR %": p.get("cagr_pct"), "Portfölj DD %": p["max_dd_pct"]})
    return rows


def render_berserk_backtest() -> None:
    page_header("🪓 BERSERK — backtest", "Contrarian swing i hela råvarumarknaden: divergens mot råvaran, "
                                         "cykelvändning och panik-snapback. Inga data som inte var kända vid entry.")
    if _TICKERS not in st.session_state:
        _apply_list()
    st.selectbox("Lista", list(LISTS), key="bz_list", on_change=_apply_list,
                 help="Producentbolag kopplade till sin råvara, och råvaru-ETF:er. Fältet går att ändra.")
    with st.form("bz_form", clear_on_submit=False):
        raw = st.text_area("Tickers", key=_TICKERS, height=90)
        c1, c2, c3 = st.columns(3)
        years = c1.selectbox("Period (år)", [3, 5, 10, 15, _OWN], index=1, key="bz_years")
        this_year = pd.Timestamp.today().year
        start_year = c2.number_input("Från år (egen period)", 2000, this_year, 2008, 1, key="bz_start_year")
        end_year = c3.number_input("Till år (egen period)", 2000, this_year, 2020, 1, key="bz_end_year")
        s1, s2 = st.columns([3, 2])
        setups = s1.multiselect("Setups", list(sg.SETUPS), default=list(sg.SETUPS), key="bz_setups")
        compare = s2.checkbox("Jämför setups", value=True, key="bz_compare",
                              help="Kör varje setup för sig och alla tre tillsammans på samma data.")
        g1, g2 = st.columns(2)
        min_turn = g1.number_input("Minsta omsättning (milj/dag)", 0.0, 100.0, bt.MIN_TURNOVER_M, 0.5,
                                   key="bz_min_turnover", help="Snitt 20 dagar i lokal valuta (USA: USD).")
        gate = g2.checkbox("Marknaden över SMA200", value=True, key="bz_market_gate",
                           help="Regionens index: OMXS30, SPY, TSX, FTSE eller ASX 200. Av = köp även i "
                                "björnmarknad.")
        go_ = st.form_submit_button("🪓 Kör backtest")
    tickers = vs.parse_tickers(raw, limit=400)
    if go_ and tickers and setups:
        own = years == _OWN
        out = {}
        runs = variants(setups, compare)
        bar = st.progress(0.0, text="Hämtar kurser och drivare …")
        for k, (name, sets) in enumerate(runs):
            cfg = bt.Config(setups=sets, years=5 if own else int(years), start_year=int(start_year) if own else None,
                            end_year=int(max(end_year, start_year)) if own else None,
                            min_turnover_m=float(min_turn), market_gate=bool(gate))
            res = bt.run(tickers, cfg=cfg, progress=lambda i, n, t, k=k, name=name: bar.progress(
                (k + i / n) / len(runs), text=f"{name}: {t} ({i}/{n})"))
            portfolio_of(res)
            out[name] = res
        bar.empty()
        st.session_state[_RES] = {"runs": out, "selected": runs[-1][0]}
    state = st.session_state.get(_RES)
    if not state:
        note("Välj lista och tryck 🪓 Kör backtest. Tips: 'Egen period' 2008–2020 testar reglerna på år de "
             "aldrig sett, med råvarukraschen 2014–2016.")
        return
    runs = state["runs"]
    if len(runs) > 1:
        _title("JÄMFÖRELSE AV SETUPS")
        st.markdown(_table(comparison_rows(runs), "Körning"), unsafe_allow_html=True)
        names = list(runs)
        shown = st.selectbox("Visa detaljer för", names, index=names.index(state["selected"])
                             if state["selected"] in names else 0, key="bz_show")
    else:
        shown = next(iter(runs))
    render_result(runs[shown])


def render_result(res: dict) -> None:
    from ovtlyr.ui.viking_nine_backtest import render_portfolio
    from ovtlyr.ui.viking_robustness_ui import render_robustness
    m, per = res["metrics"], res.get("period") or {}
    st.markdown(f"<div style='color:{DIM};font-size:0.78rem;'>{len(res['per_ticker'])} tickers · "
                f"{per.get('start', '—')} – {per.get('end', '—')} · setups: "
                f"{', '.join(res['config'].setups)} · {res.get('thin', 0)} signaler för tunn omsättning · "
                f"{res.get('market_blocked', 0)} spärrade av marknaden</div>", unsafe_allow_html=True)
    if not m.get("trades"):
        note("Inga stängda affärer under perioden.")
        return
    exp = m["expectancy"]
    cards = [("AFFÄRER", f"{m['trades']}", f"snittinnehav {m['avg_holding_days']:g} handelsdagar", CYAN),
             ("WIN RATE", f"{m['win_rate']:g} %", f"flest förluster i rad: {m['max_consecutive_losses']}", CYAN),
             ("EXPECTANCY", f"{exp:+.2f}R", f"PF {m['profit_factor'] if m['profit_factor'] is not None else '—'} · "
                                            f"summa {m['total_r']:+.1f}R", GREEN if exp > 0 else RED)]
    cols = st.columns(3)
    for k, (t, big, sub, col) in enumerate(cards):
        cols[k].markdown(big_card(t, big, sub, col), unsafe_allow_html=True)
    trades = res["trades"]
    for title, key in (("PER SETUP", lambda t: t.features.get("setup")),
                       ("PER KOMPLEX", lambda t: th.COMPLEXES.get(t.features.get("complex"), "—")),
                       ("PER TEMA", lambda t: th.label(t.features.get("theme"))),
                       ("PRODUCENT ELLER ETF", lambda t: t.features.get("kind")),
                       ("PER REGION", lambda t: "ETF" if t.features.get("kind") == "etf" else uv.region_of(t.ticker)),
                       ("EXITORSAK", lambda t: t.exit_reason)):
        rows = sorted(rb.group(trades, key), key=lambda r: -r["Summa R"])
        if rows:
            _title(title)
            st.markdown(_table(rows, "Grupp"), unsafe_allow_html=True)
    p = portfolio_of(res)
    render_portfolio(p)
    note(f"BERSERK-portföljen: max 8 positioner, 20 % per position, 2 per tema, 4 per komplex, 6 % öppen risk. "
         f"Hoppade över: {p.get('skipped_heat', 0)} värmetak · {p.get('skipped_group', 0)} komplex · "
         f"{p.get('skipped_sector', 0)} tema.")
    render_robustness(res, p, pc=bt.portfolio_config(), lists=False)
    with st.expander("Drivare som användes"):
        st.markdown(_table([{"Tema": th.label(t), "Komplex": th.COMPLEXES.get(th.complex_of(t), ""),
                             "Drivare": s or "ingen — bara S3"} for t, s in sorted(res.get("drivers", {}).items())],
                           "Tema"), unsafe_allow_html=True)
    with st.expander(f"Alla affärer ({len(trades)})"):
        st.dataframe(pd.DataFrame([{
            "Ticker": t.ticker, "Setup": t.features.get("setup"), "Tema": th.label(t.features.get("theme")),
            "Entry": t.entry_date, "Pris in": round(t.entry, 2), "Exit": t.exit_date, "Pris ut": round(t.exit, 2),
            "Orsak": t.exit_reason, "R": t.r, "Dagar": t.days} for t in trades]), hide_index=True, width="stretch")
    with st.expander("Per ticker"):
        st.dataframe(pd.DataFrame([{
            "Ticker": r["ticker"], "Region": uv.region_of(r["ticker"]), "Tema": th.label(r.get("theme") or ""),
            "Affärer": len(r["trades"]),
            "Summa R": round(sum(t.r for t in r["trades"] if t.r is not None and not t.open), 2),
            **{s.split(" ", 1)[0] + " signaler": r["signals"].get(s, 0) for s in sg.SETUPS},
            "Tunn": r.get("thin", 0), "Marknad": r.get("market_blocked", 0), "Data": r.get("error") or "ok"}
            for r in res["per_ticker"]]), hide_index=True, width="stretch")
    note(" ".join(res.get("notes") or ()))
