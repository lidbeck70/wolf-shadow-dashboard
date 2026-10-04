"""
ovtlyr/ui/viking_nine_backtest.py — PORTFOLIO → Backtest → ⚔️ Viking Nine.

Kör viking_backtest över en lista tickers (förval: senaste ⚔️ Viking Nine-
skanningen) och visar nyckeltalen i R: antal affärer, win rate, snitt- och
median-R, profit factor, max drawdown, snittvinnare/-förlorare, expectancy,
flest förluster i rad och snittinnehav — plus R-kurvan, exitorsakerna och
varje affär. Utan look-ahead; begränsningarna står under resultatet.

PORTFÖLJ kör samma affärer genom ett konto (viking_portfolio): 1,5 % risk,
max 25 % per position, max 100 % investerat, max två förluster per dag.

Marknadsriskspärren (🌩️ Marknadsrisk) är på som live (per marknad: OMXS30
från FÖRHÖJD, SPY vid HÖG) och kan jämföras mot FÖRHÖJD / bara HÖG / av. Risknivåerna delar cache med fliken.

OVTLYR GOLDEN TICKET (aktier): reglerna ur OVTLYR:s Golden Ticket-material
kan slås på var för sig (viking_backtest.OVTLYR_RULES + viking_portfolio.
PORTFOLIO_RULES) eller jämföras en och en mot live-reglerna. Live ändras inte.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import viking_backtest as vb
import viking_execution as vx
import viking_portfolio as vp
import viking_screen as vs
from ovtlyr.ui.viking_screens import _ROWS, _sector
from ui.charts import PLOTLY_LAYOUT
from ui.components import big_card, note, page_header
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, RED, TEXT

_RES = "vnb_result"
_CUSTOM = "Egna"
_TICKERS = "vnb_tickers"
_SCAN = "Senaste ⚔️ Viking Nine-skanningen"
_COUNTRY = ((".ST", "Sverige"), (".OL", "Norge"), (".CO", "Danmark"), (".HE", "Finland"))


def _apply_list() -> None:
    """Tickerlistan → fältet (callback innan sidan ritas om)."""
    name = st.session_state.get("vnb_list")
    st.session_state[_TICKERS] = (", ".join(vb.TICKER_LISTS[name]) if name in vb.TICKER_LISTS
                                  else _default_tickers())


def _default_tickers() -> str:
    data = st.session_state.get(_ROWS) or {}
    rows = [r for r in vs.results(data.get("rows") or []) if r.get("nine") is not None]
    return ", ".join(r["ticker"] for r in rows[:20])


def _risk_points(market: str):
    """Marknadsriskens poäng per dag — samma beräkning och sessionscache som 🌩️ Marknadsrisk."""
    from market_risk_ui import _load
    r = _load(market)
    return None if r.error else r.history


def portfolio_of(res: dict) -> dict:
    """Portföljläget för en körning (räknas en gång; äldre sessionsresultat räknas här)."""
    if "portfolio" not in res:
        res["portfolio"] = vp.simulate(res.get("trades") or [], years=getattr(res.get("config"), "years", None))
    return res["portfolio"]


def _fmt(v, f="{:+.2f}R") -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    return f.format(v)


def curve_chart(m: dict) -> go.Figure:
    pts = m.get("curve") or []
    fig = go.Figure(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines",
                               line=dict(color=CYAN, width=1.6), name="Summa R"))
    fig.add_hline(y=0, line=dict(color=DIM, width=1, dash="dot"))
    layout = dict(PLOTLY_LAYOUT)
    layout.update(height=300, showlegend=False, title=dict(text="R-KURVA (STÄNGDA AFFÄRER)",
                                                           font=dict(size=12, color=CYAN)),
                  yaxis=dict(title="R", gridcolor="rgba(255,255,255,0.05)"))
    fig.update_layout(**layout)
    return fig


def render_viking_nine_backtest() -> None:
    page_header("⚔️ Viking Nine — backtest", "OVTLYR Nine + Viking Execution + exitmotorn bakåt i tiden, "
                                            "rapporterat i R. Inga data som inte var kända vid entry.")
    if _TICKERS not in st.session_state:
        _apply_list()
    st.selectbox("Tickerlista", [_SCAN] + list(vb.TICKER_LISTS), key="vnb_list", on_change=_apply_list,
                 help="Fyller fältet nedan. Samma fasta lista varje gång ger rättvisa jämförelser; "
                      "listan går att ändra i fältet innan du kör.")
    with st.form("vnb_form", clear_on_submit=False):
        raw = st.text_area("Tickers", key=_TICKERS, height=90, placeholder="t.ex. NVDA, MSFT, VOLV-B.ST")
        c1, c2, c3 = st.columns(3)
        years = c1.selectbox("Period (år)", [1, 2, 3, 5], index=2, key="vnb_years")
        min_nine = c2.selectbox("Minsta Nine för entry", [9, 8, 7], index=0, key="vnb_min_nine",
                                help="9 = systemets regel (GOLDEN TICKET). 8/7 visar vad en lösare regel hade gett.")
        need_vol = c3.checkbox("Kräv relativ volym", value=vx.VOLUME_REQUIRED, key="vnb_vol",
                               help=f"Relativ volym ≥ {vx.RELATIVE_VOLUME_MIN:g}× på signaldagen. På = samma regel "
                                    f"som live.")
        e1, e2 = st.columns([3, 2])
        preset = e1.selectbox("Exitregler", list(vb.EXIT_PRESETS) + [_CUSTOM], key="vnb_exit_preset",
                              help="Stopp och breakeven-stopp gäller alltid.")
        compare = e2.checkbox("Jämför alla förval", value=False, key="vnb_compare",
                              help="Kör samma tickers med varje förval och visar nyckeltalen sida vid sida.")
        custom = st.multiselect("Egna exitregler (när 'Egna' är valt)", list(vb.EXIT_RULES),
                                default=list(vb.ALL_EXITS), format_func=lambda k: vb.EXIT_RULES[k],
                                key="vnb_custom_rules")
        custom_after_be = st.checkbox("EMA10 först efter breakeven (egna)", value=False, key="vnb_custom_be")
        r1, r2 = st.columns([3, 2])
        gate = r1.selectbox("Marknadsriskspärr", list(vb.RISK_GATES), key="vnb_risk_gate",
                            help="Signaldagar där 🌩️ Marknadsrisk (SPY, OMXS30 för nordiska) låg på spärrad nivå "
                                 "ger ingen entry. Per marknad = samma regel som live (OMXS30 från FÖRHÖJD, SPY vid HÖG).")
        compare_risk = r2.checkbox("Jämför spärrar", value=False, key="vnb_compare_risk",
                                   help="Kör samma tickers med varje spärr och visar nyckeltalen sida vid sida.")
        n1, n2 = st.columns([3, 2])
        entry = n1.selectbox("Entry", list(vb.ENTRY_MODES), key="vnb_entry",
                             help=f"Efter pullback = entry bara om låget någon av de senaste {vb.PULLBACK_DAYS} dagarna "
                                  f"nådde EMA20 + {vb.PULLBACK_ATR:g} × ATR. En hypotes att testa — live använder "
                                  f"'Som live'.")
        compare_entry = n2.checkbox("Jämför entry", value=False, key="vnb_compare_entry",
                                    help="Kör samma tickers med båda entrysätten.")
        b1, b2 = st.columns([3, 2])
        breadth_rule = b1.selectbox("Marknadsbredd", list(vb.BREADTH_RULES), key="vnb_breadth",
                                    help="OVTLYR = live: bredden över sin EMA10, under 25 bara efter uppvändning, "
                                         "över 75 och nedvänd = inga nya affärer. Gamla = ≥ 50 % och stigande.")
        one_sector = b2.checkbox("En aktie per sektor", value=True, key="vnb_one_sector",
                                 help="Portföljläget: högst en öppen position per sektor — som live (SEKTOR UPPTAGEN).")
        st.markdown(f"<div style='color:{GOLD};font-size:0.72rem;letter-spacing:0.1em;margin-top:6px;'>OVTLYR GOLDEN "
                    f"TICKET — AKTIEREGLER ATT TESTA</div>", unsafe_allow_html=True)
        oc = st.columns(2)
        ovt = {k: oc[n % 2].checkbox(label, value=False, key=f"vnb_ovt_{k}", help=why)
               for n, (k, (label, why)) in enumerate(OVT_ALL.items())}
        compare_ovt = st.checkbox("Jämför OVTLYR-reglerna en och en", value=False, key="vnb_compare_ovt",
                                  help="Kör live-reglerna, varje OVTLYR-regel för sig och alla tillsammans — med "
                                       "exitförval, spärr och entry som ovan. De andra jämförelserna hoppas då över.")
        go_ = st.form_submit_button("⚔️ Kör backtest")
    tickers = vs.parse_tickers(raw)
    if go_ and tickers:
        base = dict(min_nine=int(min_nine), require_volume=bool(need_vol), years=int(years))
        chosen = [k for k, on_ in ovt.items() if on_]
        if compare_ovt:
            runs = [(name, preset, gate, entry, keys) for name, keys in ovtlyr_variants(chosen)]
        else:
            runs = [(name, ex, gt, en, chosen) for name, ex, gt, en in
                    run_names(preset, gate, compare, compare_risk, entry, compare_entry)]
        out, cache = {}, {}
        bar = st.progress(0.0, text="Hämtar marknadsdata …")
        for k, (name, ex, gt, en, keys) in enumerate(runs):
            rules, after_be = ((tuple(custom), bool(custom_after_be)) if ex == _CUSTOM
                               else vb.EXIT_PRESETS[ex])
            bt = {kk: True for kk in keys if kk in vb.OVTLYR_RULES}
            pc = vp.PortfolioConfig(one_per_sector=bool(one_sector),
                                    **{kk: True for kk in keys if kk in vp.PORTFOLIO_RULES})
            cfg = vb.Config(exit_rules=rules, trail_after_be=after_be, risk_gate=vb.RISK_GATES[gt],
                            pullback=vb.ENTRY_MODES[en], ovt_breadth=vb.BREADTH_RULES[breadth_rule], **base, **bt)
            ck = (ex, gt, en, cfg.ovt_breadth, tuple(sorted(bt)))
            if ck not in cache:                                # portföljregler återanvänder samma affärer
                cache[ck] = vb.run(tickers, sector_getter=_sector, cfg=cfg, risk_provider=_risk_points,
                                   progress=lambda i, n, t, k=k, name=name: bar.progress(
                                       (k + i / n) / len(runs), text=f"{name}: {t} ({i}/{n})"))
            res = dict(cache[ck])
            res.pop("portfolio", None)
            res["labels"] = {"exit": ex, "gate": gt, "entry": en, "ovtlyr": list(keys)}
            res["portfolio"] = vp.simulate(res.get("trades") or [], years=cfg.years, pc=pc)
            out[name] = res
        bar.empty()
        selected = next((r[0] for r in runs if r[1] == preset and r[2] == gate and r[3] == entry), runs[0][0])
        st.session_state[_RES] = {"selected": selected, "runs": out}
    state = st.session_state.get(_RES)
    if not state:
        note("Välj tickers och tryck ⚔️ Kör backtest. Tips: kör ⚔️ Viking Nine-skanningen först så fylls "
             "kandidaterna i här.")
        return
    if "runs" not in state:                                    # äldre sessionsformat
        state = {"selected": "Alla regler", "runs": {"Alla regler": state}}
    if len(state["runs"]) > 1:
        render_comparison(state["runs"])
        shown = st.selectbox("Visa detaljer för", list(state["runs"]), key="vnb_show",
                             index=list(state["runs"]).index(state["selected"]) if state["selected"] in state["runs"]
                             else 0)
    else:
        shown = next(iter(state["runs"]))
    render_result(state["runs"][shown], shown)


OVT_ALL = {**vb.OVTLYR_RULES, **vp.PORTFOLIO_RULES}
LIVE = "Som live"
ALL_OVT = "Alla OVTLYR-regler"


def ovtlyr_variants(chosen: list = None) -> list:
    """[(namn, regelnycklar)]: live, varje OVTLYR-regel för sig och alla tillsammans.
    chosen = ikryssade regler — tomt = alla regler jämförs."""
    keys = [k for k in OVT_ALL if not chosen or k in chosen]
    return [(LIVE, [])] + [(OVT_ALL[k][0], [k]) for k in keys] + ([(ALL_OVT, keys)] if len(keys) > 1 else [])


def run_names(preset: str, gate: str, compare: bool, compare_risk: bool, entry: str = None,
              compare_entry: bool = False) -> list:
    """[(namn, exitförval, spärr, entry)] för körningarna. Namnet visar bara det som jämförs."""
    entry = entry or next(iter(vb.ENTRY_MODES))
    exits = list(vb.EXIT_PRESETS) if compare else [preset]
    gates = list(vb.RISK_GATES) if compare_risk else [gate]
    entries = list(vb.ENTRY_MODES) if compare_entry else [entry]
    out = []
    for ex in exits:
        for gt in gates:
            for en in entries:
                parts = ([ex] if len(exits) > 1 else []) + ([f"spärr {gt}"] if len(gates) > 1 else []) \
                    + ([f"entry {en}"] if len(entries) > 1 else [])
                name = " · ".join(parts) if parts else ex
                out.append((name[0].upper() + name[1:], ex, gt, en))
    return out


def comparison_rows(runs: dict) -> list:
    rows = []
    for name, res in runs.items():
        m = res["metrics"]
        rows.append({"Körning": name, "Affärer": m.get("trades", 0), "Spärrade": res.get("risk_blocked", 0),
                     "Win rate %": m.get("win_rate"),
                     "Expectancy R": m.get("expectancy"), "Profit factor": m.get("profit_factor"),
                     "Summa R": m.get("total_r"), "Max DD R": m.get("max_drawdown_r"),
                     "Snittvinnare R": m.get("avg_winner"), "Snittinnehav d": m.get("avg_holding_days")})
        p = portfolio_of(res)
        rows[-1].update({"Portfölj affärer": p["taken"], "Portfölj avk. %": p["return_pct"],
                         "Portfölj CAGR %": p.get("cagr_pct"), "Portfölj DD %": p["max_dd_pct"]})
    return rows


def render_comparison(runs: dict) -> None:
    rows = comparison_rows(runs)
    best = max((r for r in rows if r["Expectancy R"] is not None), key=lambda r: r["Expectancy R"], default=None)
    lowest = {k: min((r[k] for r in rows if r.get(k) is not None), default=None) for k in ("Max DD R", "Portfölj DD %")}
    top = {k: max((r[k] for r in rows if r.get(k) is not None), default=None)
           for k in ("Portfölj avk. %", "Portfölj CAGR %")}
    head = "".join(f"<th>{k}</th>" for k in rows[0])
    body = "".join(
        "<tr>" + "".join(
            f"<td style='{'text-align:left;' if k == 'Körning' else ''}"
            f"{'color:' + GREEN + ';font-weight:700;' if best is r and k in ('Körning', 'Expectancy R') else ''}"
            f"{'color:' + GREEN + ';font-weight:700;' if k in lowest and v is not None and v == lowest[k] else ''}"
            f"{'color:' + GREEN + ';font-weight:700;' if k in top and v is not None and v == top[k] else ''}'>"
            f"{'—' if v is None else _fmt(v, '{:.2f}') if isinstance(v, float) else v}</td>" for k, v in r.items())
        + "</tr>" for r in rows)
    ovt = any((res.get("labels") or {}).get("ovtlyr") for res in runs.values())
    title = "OVTLYR-REGLERNA MOT LIVE" if ovt else "EXITREGLER OCH RISKSPÄRR"
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:8px;'>JÄMFÖRELSE AV "
                f"{title}</div><div style='overflow-x:auto;'><table style='width:100%;font-size:0.76rem;"
                f"color:{TEXT};text-align:right;'><tr style='color:{DIM};'>{head}</tr>{body}</table></div>",
                unsafe_allow_html=True)
    note("Samma tickers — bara reglerna i körningens namn skiljer. Grönt = högst expectancy, lägst max "
         "drawdown och högst portföljavkastning/CAGR. En OVTLYR-regel är värd att ta live först om den "
         "förbättrar både Norden och USA. ½ ATR-stoppet räknar R på 2 × ATR — jämför det på portföljens "
         "avkastning och drawdown, inte på R. Portfölj = samma affärer genom ett konto (1,5 % risk, max 25 % "
         "per position, max 100 % investerat, max två förluster per dag). "
         "Spärrade = signaler som stoppades av marknadsrisken. Få affärer ger brusiga tal; jämför helst "
         "på 50+ affärer.")


def render_result(res: dict, name: str = "") -> None:
    m, cfg = res["metrics"], res["config"]
    rules = ", ".join(vb.EXIT_RULES[k] for k in cfg.exit_rules) or "bara stopp"
    gate = getattr(cfg, "risk_gate", ())
    st.markdown(f"<div style='color:{DIM};font-size:0.78rem;'>{len(res['per_ticker'])} tickers · {cfg.years} år · "
                f"entry vid Nine ≥ {cfg.min_nine}/9 · volymkrav {'på' if cfg.require_volume else 'av'}<br>"
                f"Exit: <b style='color:{TEXT};'>{name or 'Alla regler'}</b> — stopp, breakeven-stopp, {rules}"
                f"{' · EMA10 först efter breakeven' if cfg.trail_after_be else ''}<br>"
                f"Entry: <b style='color:{TEXT};'>{'efter pullback till EMA20' if getattr(cfg, 'pullback', False) else 'som live (stark dag)'}</b>"
                f"{' · ' + str(res.get('no_pullback', 0)) + ' signaler utan pullback' if getattr(cfg, 'pullback', False) else ''}"
                f"<br>Marknadsbredd: <b style='color:{TEXT};'>"
                f"{'OVTLYR (som live)' if getattr(cfg, 'ovt_breadth', False) else 'gamla: ≥ 50 % och stigande'}</b>"
                f"<br>{risk_line(res, gate)}{ovtlyr_line(res)}</div>",
                unsafe_allow_html=True)
    if not m.get("trades"):
        note("Inga stängda affärer under perioden. Prova fler tickers, längre period eller lägre minsta Nine "
             "för att se hur ofta setupen uppstår.")
    else:
        exp = m["expectancy"]
        cards = [
            ("AFFÄRER", f"{m['trades']}", f"snittinnehav {m['avg_holding_days']:g} handelsdagar", CYAN),
            ("WIN RATE", f"{m['win_rate']:g} %", f"flest förluster i rad: {m['max_consecutive_losses']}", CYAN),
            ("EXPECTANCY", _fmt(exp), f"{m['win_rate'] / 100:.2f} × {m['avg_winner']:.2f} − "
                                       f"{1 - m['win_rate'] / 100:.2f} × {abs(m['avg_loser']):.2f}",
             GREEN if exp > 0 else RED),
            ("PROFIT FACTOR", _fmt(m["profit_factor"], "{:.2f}"), "vinster i R / förluster i R",
             GREEN if (m["profit_factor"] or 0) > 1 else RED),
            ("SNITT / MEDIAN", f"{_fmt(m['avg_r'])} / {_fmt(m['median_r'])}", f"summa {_fmt(m['total_r'])}", GOLD),
            ("MAX DRAWDOWN", _fmt(-m["max_drawdown_r"]), f"snittvinnare {_fmt(m['avg_winner'])} · snittförlorare "
                                                         f"{_fmt(m['avg_loser'])}", AMBER),
        ]
        cols = st.columns(3)
        for k, (title, big, sub, col) in enumerate(cards):
            cols[k % 3].markdown(big_card(title, big, sub, col), unsafe_allow_html=True)
        try:
            st.plotly_chart(curve_chart(m), use_container_width=True, config={"displayModeBar": False},
                            key="vnb_curve")
        except Exception:
            pass
        trades = [t for t in res["trades"] if not t.open]
        by = pd.DataFrame([{"Exit": t.exit_reason, "R": t.r} for t in trades]).groupby("Exit")["R"]
        rows = "".join(f"<tr><td style='text-align:left;'>{k}</td><td>{int(v.count())}</td>"
                       f"<td>{v.mean():+.2f}R</td></tr>" for k, v in by)
        st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:8px;'>EXITORSAKER"
                    f"</div><table style='font-size:0.78rem;color:{TEXT};text-align:right;'><tr style='color:{DIM};'>"
                    f"<th style='text-align:left;'>Exit</th><th>Antal</th><th>Snitt</th></tr>{rows}</table>",
                    unsafe_allow_html=True)
        render_groups(res["trades"])
        render_portfolio(portfolio_of(res))
        with st.expander(f"Alla affärer ({len(res['trades'])})"):
            st.dataframe(pd.DataFrame([{
                "Ticker": t.ticker, "Signal": t.signal_date, "Entry": t.entry_date, "Pris in": round(t.entry, 2),
                "Stopp": round(t.stop, 2), "Exit": t.exit_date, "Pris ut": round(t.exit, 2), "Orsak": t.exit_reason,
                "R": t.r, "Dagar": t.days, "Nine": f"{t.nine}/9"} for t in res["trades"]]),
                hide_index=True, width="stretch")
    with st.expander("Per ticker (sämst först)"):
        st.dataframe(pd.DataFrame(ticker_rows(res)), hide_index=True, width="stretch")
    note("Expectancy = win rate × snittvinnare − förlustandel × snittförlorare, i R. " + " ".join(res["notes"])
         + " Marknadsriskens poäng räknas om bakåt med dagens modell; OMXS30-bredden bygger på dagens Large "
           "Cap-lista (överlevnadsbias).")


def country_of(ticker: str) -> str:
    t = str(ticker or "").upper()
    return next((name for suf, name in _COUNTRY if t.endswith(suf)), "USA/övriga")


def group_rows(trades: list, key) -> list:
    """Nyckeltal i R per grupp (stängda affärer), sämst summa först."""
    groups = {}
    for t in trades:
        if not t.open and t.r is not None:
            groups.setdefault(key(t), []).append(t)
    rows = []
    for g, ts in groups.items():
        m = vb.metrics(ts)
        rows.append({"Grupp": g, "Affärer": m["trades"], "Win rate %": m["win_rate"], "Expectancy R": m["expectancy"],
                     "Summa R": m["total_r"], "Max DD R": m["max_drawdown_r"]})
    return sorted(rows, key=lambda r: r["Summa R"])


def ticker_rows(res: dict) -> list:
    """Per ticker: signaler, filter och resultat i R — sämst summa först."""
    stats = {r["Grupp"]: r for r in group_rows(res.get("trades") or [], lambda t: t.ticker)}
    rows = []
    for p in res["per_ticker"]:
        s = stats.get(p["ticker"], {})
        rows.append({"Ticker": p["ticker"], "Land": country_of(p["ticker"]), "Summa R": s.get("Summa R"),
                     "Expectancy R": s.get("Expectancy R"), "Win rate %": s.get("Win rate %"),
                     "Affärer": len(p["trades"]), "Signaler": p["signals"], "Spärrade": p.get("risk_blocked", 0),
                     "Utan pullback": p.get("no_pullback", 0), "Illikvid": p.get("illiquid", 0),
                     "No chase": p["no_chase"], "R/R < 2": p["low_rr"],
                     "Marknad": p.get("market") or "—", "Sektor-ETF": p.get("sector_etf") or "—",
                     "Data": p.get("error") or "ok"})
    return sorted(rows, key=lambda r: (r["Summa R"] is None, r["Summa R"] if r["Summa R"] is not None else 0))


def render_groups(trades: list) -> None:
    rows = group_rows(trades, lambda t: country_of(t.ticker))
    if not rows:
        return
    body = "".join(
        f"<tr><td style='text-align:left;'>{r['Grupp']}</td><td>{r['Affärer']}</td><td>{r['Win rate %']:g} %</td>"
        f"<td style='color:{GREEN if r['Expectancy R'] > 0 else RED};'>{r['Expectancy R']:+.2f}R</td>"
        f"<td>{r['Summa R']:+.1f}R</td><td>{r['Max DD R']:.1f}R</td></tr>" for r in rows)
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:8px;'>PER LAND</div>"
                f"<table style='font-size:0.78rem;color:{TEXT};text-align:right;'><tr style='color:{DIM};'>"
                f"<th style='text-align:left;'>Land</th><th>Affärer</th><th>Win rate</th><th>Expectancy</th>"
                f"<th>Summa</th><th>Max DD</th></tr>{body}</table>", unsafe_allow_html=True)


def equity_chart(p: dict) -> go.Figure:
    pts = p.get("curve") or []
    fig = go.Figure(go.Scatter(x=[d for d, _v in pts], y=[v for _d, v in pts], mode="lines",
                               line=dict(color=GOLD, width=1.6), name="Konto %"))
    fig.add_hline(y=0, line=dict(color=DIM, width=1, dash="dot"))
    layout = dict(PLOTLY_LAYOUT)
    layout.update(height=280, showlegend=False, title=dict(text="PORTFÖLJ — KONTOT I % (STÄNGDA AFFÄRER)",
                                                           font=dict(size=12, color=GOLD)),
                  yaxis=dict(title="%", gridcolor="rgba(255,255,255,0.05)"))
    fig.update_layout(**layout)
    return fig


def render_portfolio(p: dict) -> None:
    """PORTFÖLJ — samma affärer genom ett konto med Vikings gränser."""
    m = p["metrics"]
    st.markdown(f"<div style='color:{GOLD};font-size:0.72rem;letter-spacing:0.1em;margin-top:14px;'>PORTFÖLJ — "
                f"ETT KONTO</div>", unsafe_allow_html=True)
    if not m.get("trades"):
        note("Inga stängda affärer i portföljläget. " + p["note"])
        return
    cagr = "—" if p["cagr_pct"] is None else f"{p['cagr_pct']:+g} % per år"
    exp = m["expectancy"]
    cards = [
        ("TAGNA AFFÄRER", f"{p['taken']} av {p['candidates']}",
         f"hoppade över: {p['skipped_full']} fullt · {p['skipped_losses']} förlustspärr"
         f"{' · ' + str(p['skipped_sector']) + ' sektor' if p.get('skipped_sector') else ''} · max {p['max_open']} "
         f"öppna", CYAN),
        ("AVKASTNING", f"{p['return_pct']:+g} %", f"{cagr} · snittposition {p['avg_position_pct']:g} %", GOLD),
        ("MAX DRAWDOWN", f"−{p['max_dd_pct']:g} %", f"i R: {_fmt(-m['max_drawdown_r'])} · snittrisk "
                                                     f"{p['avg_risk_pct']:g} % per affär", AMBER),
        ("EXPECTANCY", _fmt(exp), f"summa {_fmt(m['total_r'])} · win rate {m['win_rate']:g} % · PF "
                                  f"{_fmt(m['profit_factor'], '{:.2f}')}", GREEN if exp > 0 else RED),
    ]
    cols = st.columns(2)
    for k, (title, big, sub, col) in enumerate(cards):
        cols[k % 2].markdown(big_card(title, big, sub, col), unsafe_allow_html=True)
    try:
        st.plotly_chart(equity_chart(p), use_container_width=True, config={"displayModeBar": False},
                        key="vnb_equity")
    except Exception:
        pass
    with st.expander(f"Portföljens affärer ({p['taken']})"):
        st.dataframe(pd.DataFrame([{
            "Ticker": r["trade"].ticker, "Entry": r["trade"].entry_date, "Exit": r["trade"].exit_date,
            "Orsak": r["trade"].exit_reason, "Position %": r["position_pct"], "Risk %": r["risk_pct"],
            "R": r["trade"].r, "Konto %": r["return_pct"]} for r in p["rows"]]), hide_index=True, width="stretch")
    note(p["note"] + " Avkastningen räknas med ränta på ränta på stängda affärer — utan courtage, skatt och "
                     "valuta.")


def ovtlyr_line(res: dict) -> str:
    """En rad om påslagna OVTLYR-regler (tom om inga)."""
    keys = ((res.get("labels") or {}).get("ovtlyr")
            or getattr(res.get("config"), "ovtlyr_rules", lambda: [])())
    if not keys:
        return ""
    extra = []
    if "liquidity" in keys:
        extra.append(f"{res.get('illiquid', 0)} signaler för illikvida")
    if "history" in keys:
        extra.append(f"{res.get('neg_history', 0)} affärer bortfiltrerade (negativ historik)")
    return (f"<br>OVTLYR: <b style='color:{GOLD};'>{', '.join(OVT_ALL[k][0] for k in keys if k in OVT_ALL)}</b>"
            + (" · " + " · ".join(extra) if extra else ""))


def risk_line(res: dict, gate) -> str:
    """En rad om marknadsriskspärren: nivåer, spärrade signaler och hur ofta varje marknad låg spärrad."""
    if not gate:
        return "Marknadsriskspärr: av"
    parts = []
    for m, info in (res.get("risk") or {}).items():
        pct = info.get("blocked_pct")
        parts.append(f"{m}: {info['status']}" if pct is None else f"{m} spärrad {pct:g} % av dagarna")
    rule = (" · ".join(f"{m}: {' eller '.join(lv) or 'av'}" for m, lv in gate.items()) if isinstance(gate, dict)
            else " eller ".join(gate))
    return (f"Marknadsriskspärr: <b style='color:{TEXT};'>{rule}</b> stoppar entry · "
            f"{res.get('risk_blocked', 0)} signaler spärrade" + (" · " + " · ".join(parts) if parts else ""))
