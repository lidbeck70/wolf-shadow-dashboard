"""
berserk/ui.py — PORTFOLIO → Backtest → 🪓 BERSERK.

Kör berserk.backtest över råvaruaktier och råvaru-ETF:er och visar nyckeltalen
i R per setup, portföljen (max 8 positioner, 20 % per position, 2 per tema,
4 per komplex, 6 % öppen risk) med dagsvärderad drawdown, uppdelning per tema
och komplex, vilka drivare som användes och robusthetsverktygen från Viking
Nine (kostnader, Monte Carlo, kantanalys, köp och behåll).
"""

from __future__ import annotations

import time

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
_GLOBAL = "Utanför Norden (USA, Kanada, London)"
_ALL = "Allt (Norden + utanför Norden + ETF:er)"
LISTS = {**uv.LISTS, _GLOBAL: tuple(uv.GLOBAL), _ALL: tuple(uv.NORDIC) + tuple(uv.GLOBAL) + tuple(uv.ETFS)}
ALL_SETUPS = "Alla tre"
M_SETUPS, M_PORTFOLIO, M_NONE = "Setups var för sig", "Portföljvarianter", "Ingen jämförelse"
MODES = (M_SETUPS, M_PORTFOLIO, M_NONE)
BASE = "Nuvarande regel"
_LIVE = {"drop_s3": True, "commodity_gate": True, "max_gap_atr": 1.0}          # = backtest.live_config()
_SAT = {**_LIVE, "satellite_risk": 0.5, "core_first": True}                   # kärna Norden + satellit
_SAT_PC = {"max_heat": 8.0, "max_sat": 3}
# (namn, Config-ändringar, portfolio_config-ändringar) — provkörningar; den gällande regeln är den första
PORTFOLIO_VARIANTS = (
    (BASE, _LIVE, {"max_heat": 8.0}),
    ("Kärna + satellit (DBC)", _SAT, _SAT_PC),
    ("Kärna + satellit (koppar/guld)", {**_SAT, "gate_kind": bt.GATE_CU_AU}, _SAT_PC),
    ("Kärna + satellit (DBC) + S1 ej i TOPP", {**_SAT, "s1_top_block": True}, _SAT_PC),
    ("Kärna + satellit (koppar/guld) + S1 ej i TOPP", {**_SAT, "gate_kind": bt.GATE_CU_AU, "s1_top_block": True},
     _SAT_PC),
    ("Bara Norden (referens)", {"drop_s3": True, "max_gap_atr": 1.0, "regions": ("Norden",)}, {"max_heat": 8.0}),
)


def _apply_list() -> None:
    st.session_state[_TICKERS] = ", ".join(LISTS.get(st.session_state.get("bz_list"), uv.LISTS["Norden"]))


def variants(setups: list, compare: bool) -> list:
    """[(namn, setups)] — vald kombination, eller varje setup för sig plus alla tre."""
    if not compare:
        return [(" + ".join(s.split(" ", 1)[0] for s in setups) or ALL_SETUPS, tuple(setups))]
    return [(s, (s,)) for s in sg.SETUPS] + [(ALL_SETUPS, sg.SETUPS)]


def plan_runs(mode: str, setups: list) -> list:
    """[(namn, Config-argument, portfolio_config-argument)] för vald jämförelse."""
    if mode == M_PORTFOLIO:
        base = tuple(setups)
        out = []
        for name, cfg_kw, pc_kw in PORTFOLIO_VARIANTS:
            kw = dict(cfg_kw)
            kw.pop("drop_s3", None)
            if ("s3_regions" in kw or "max_s3" in pc_kw) and sg.S3 not in base:
                continue                                             # varianten handlar om S3 — finns inte valt
            sets = tuple(s for s in base if s != sg.S3) if cfg_kw.get("drop_s3") else base
            out.append((name, {**kw, "setups": sets}, dict(pc_kw)))
        return out
    return [(name, {"setups": sets}, {}) for name, sets in variants(setups, mode == M_SETUPS)]


def stable_getter(base=None, tries: int = 3, wait: float = 2.0, sleep=time.sleep):
    """Hämtare som minns varje (ticker, period) och försöker igen när Yahoo svarar tomt (begränsning).
    Delas av alla varianter i en jämförelse — samma kursdata i varje körning."""
    if base is None:
        from market_prices import ohlcv as base
    memo: dict = {}

    def get(ticker, period):
        key = (ticker, period)
        if key not in memo:
            df = None
            for k in range(tries):
                try:
                    df = base(ticker, period)
                except Exception:
                    df = None
                if df is not None and len(df):
                    break
                if k < tries - 1:
                    sleep(wait * (k + 1))
            memo[key] = df if df is not None and len(df) else None
        return memo[key]
    return get


def missing_of(res: dict) -> list:
    """Tickers i universumet som saknade kursdata i körningen."""
    return [p["ticker"] for p in res.get("per_ticker") or [] if p.get("error") == "DATA UNAVAILABLE"]


def pc_of(res: dict) -> vp.PortfolioConfig:
    return bt.portfolio_config(**(res.get("pc_kw") or {}))


def portfolio_of(res: dict) -> dict:
    if "portfolio" not in res:
        res["portfolio"] = vp.simulate(res.get("trades") or [], years=(res.get("period") or {}).get("years"),
                                       pc=pc_of(res))
    return res["portfolio"]


def mc_cost_row(res: dict) -> dict:
    """Monte Carlo-raden 'Bootstrap + kostnad' (den försiktigaste) — räknas en gång per körning."""
    if "mc_cost" not in res:
        rows = rb.monte_carlo(portfolio_of(res).get("rows") or [], (res.get("period") or {}).get("years"))
        res["mc_cost"] = next((r for r in rows if r["Metod"].startswith("Bootstrap +")), {})
    return res["mc_cost"]


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
        m, p, mc = res["metrics"], portfolio_of(res), mc_cost_row(res)
        rows.append({"Körning": name, "Utan data": len(missing_of(res)), "Affärer": m.get("trades", 0), "Win rate %": m.get("win_rate"),
                     "Expectancy R": m.get("expectancy"), "Profit factor": m.get("profit_factor"),
                     "Summa R": m.get("total_r"), "Snittinnehav d": m.get("avg_holding_days"),
                     "Portfölj affärer": p["taken"], "Portfölj avk. %": p["return_pct"],
                     "Portfölj CAGR %": p.get("cagr_pct"), "Portfölj DD %": p["max_dd_pct"],
                     "Tagna %": round(p["taken"] / max(1, m.get("trades", 0) or 1) * 100, 0),
                     "Portfölj exp. R": (p.get("metrics") or {}).get("expectancy"),
                     "MC CAGR p5 %": mc.get("CAGR p5 %"), "MC CAGR p50 %": mc.get("CAGR p50 %"),
                     "MC DD p95 %": mc.get("DD p95 %")})
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
        mode = s2.radio("Jämförelse", MODES, index=0, key="bz_mode",
                        help="Setups var för sig: S1, S2, S3 och alla tre. Portföljvarianter: den gällande "
                             "regeln mot kärna + satellit (Norden full risk; övriga regioner och ETF:er halv "
                             "risk, högst 3 platser, bara när råvarugrinden — DBC eller koppar/guld över "
                             "SMA200 — är på) med och utan Blindspot-TOPP för S1, och bara Norden.")
        g1, g2 = st.columns(2)
        min_turn = g1.number_input("Minsta omsättning (milj/dag)", 0.0, 100.0, bt.MIN_TURNOVER_M, 0.5,
                                   key="bz_min_turnover", help="Snitt 20 dagar i lokal valuta (USA: USD).")
        gate = g2.checkbox("Marknaden över SMA200", value=True, key="bz_market_gate",
                           help="Regionens index: OMXS30, SPY, TSX eller FTSE. Av = köp även i "
                                "björnmarknad.")
        go_ = st.form_submit_button("🪓 Kör backtest")
    tickers = vs.parse_tickers(raw, limit=400)
    if go_ and tickers and setups:
        own = years == _OWN
        out = {}
        runs = plan_runs(mode, setups)
        keys = list(dict.fromkeys(repr(sorted(cfg_kw.items())) for _n, cfg_kw, _p in runs))
        done = {}                                                    # en backtest per unik signaluppsättning
        getter = stable_getter()                                     # samma kursdata i alla varianter
        frames: dict = {}                                            # signalerna räknas en gång per ticker
        bar = st.progress(0.0, text="Hämtar kurser och drivare …")
        for name, cfg_kw, pc_kw in runs:
            key = repr(sorted(cfg_kw.items()))
            if key not in done:
                k = keys.index(key)
                cfg = bt.Config(**cfg_kw, years=5 if own else int(years),
                                start_year=int(start_year) if own else None,
                                end_year=int(max(end_year, start_year)) if own else None,
                                min_turnover_m=float(min_turn), market_gate=bool(gate))
                done[key] = bt.run(tickers, getter=getter, cfg=cfg, frame_cache=frames,
                                   progress=lambda i, n, t, k=k, name=name: bar.progress(
                                       (k + i / n) / len(keys), text=f"{name}: {t} ({i}/{n})"))
            res = {kk: v for kk, v in done[key].items() if kk not in ("portfolio", "robustness", "mc_cost")}
            res["pc_kw"] = pc_kw
            portfolio_of(res)
            out[name] = res
        bar.empty()
        st.session_state[_RES] = {"runs": out, "selected": runs[0][0] if mode == M_PORTFOLIO else runs[-1][0]}
    state = st.session_state.get(_RES)
    if not state:
        note("Välj lista och tryck 🪓 Kör backtest. Tips: 'Egen period' 2008–2020 testar reglerna på år de "
             "aldrig sett, med råvarukraschen 2014–2016.")
        return
    runs = state["runs"]
    if len(runs) > 1:
        _title("JÄMFÖRELSE AV PORTFÖLJVARIANTER" if BASE in runs else "JÄMFÖRELSE AV SETUPS")
        st.markdown(_table(comparison_rows(runs), "Körning"), unsafe_allow_html=True)
        note("Tagna % = andel av signalerna som portföljen hann ta. Portfölj exp. R = expectancy på just de "
             "affärerna — ligger den långt under alla signalers är det urvalet som kostar. MC = Monte Carlo "
             "(bootstrap + kostnad 0–30 bp): CAGR p5/p50 och DD p95 på stängda affärer — välj det som håller "
             "både här och 2008–2020, inte det som ser bäst ut i en period.")
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
    missing = missing_of(res)
    if missing:
        st.warning(f"{len(missing)} tickers saknade kursdata (Yahoo svarade tomt även efter omförsök) — resultatet "
                   f"bygger på färre bolag. Kör om om en stund. Saknas: {', '.join(missing[:20])}"
                   f"{' …' if len(missing) > 20 else ''}")
    st.markdown(f"<div style='color:{DIM};font-size:0.78rem;'>{len(res['per_ticker'])} tickers · "
                f"{per.get('start', '—')} – {per.get('end', '—')} · setups: "
                f"{', '.join(res['config'].setups)} · {res.get('thin', 0)} signaler för tunn omsättning · "
                f"{res.get('market_blocked', 0)} spärrade av marknaden · {res.get('data_blocked', 0)} spärrade av "
                f"datavakten · {res.get('commodity_blocked', 0)} av råvarugrinden · {res.get('gap_skipped', 0)} "
                f"gap &gt; 1 ATR · {res.get('top_blocked', 0)} S1 i TOPP</div>", unsafe_allow_html=True)
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
    p, pc = portfolio_of(res), pc_of(res)
    render_portfolio(p)
    s3cap, satcap = dict(pc.group_caps).get("s3"), dict(pc.group_caps).get("satellite")
    note(f"BERSERK-portföljen: max {pc.max_positions} positioner, {pc.max_position_pct:g} % per position, "
         f"{pc.sector_cap} per tema, {dict(pc.group_caps).get('complex')} per komplex, {pc.max_heat_pct:g} % öppen "
         f"risk{f', max {s3cap} S3 samtidigt' if s3cap else ''}"
         f"{f', max {satcap} satellitpositioner (utanför Norden, halv risk)' if satcap else ''}. Hoppade över: {p.get('skipped_heat', 0)} värmetak · "
         f"{p.get('skipped_group', 0)} komplex/S3-tak · {p.get('skipped_sector', 0)} tema.")
    render_robustness(res, p, pc=pc, lists=False)
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
