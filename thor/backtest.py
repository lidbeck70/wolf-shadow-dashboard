"""
thor/backtest.py — Backtest-fliken i Thors analysblock: strategin på aktien med en knapp.

Per-aktie-motorer finns för fyra strategier — samma motorer som skannrarna och Valhalls papperskonton:
  Viking Nine     viking_backtest.run: OVTLYR Nine + Viking Execution + exitmotorn, i R. Snabbläge utan
                  marknadsriskspärr; nordiska aktier mot OMXS30 och svensk bredd (Börsdata) när den finns.
  BERSERK         berserk.backtest.run med live_config — skannerns regler, i R.
  Momentum Swing  swing_backtest på en en-aktiepanel: portföljreglerna (ranking, regim, −10 % stopp, halva vid
                  +20 %), men rankingen jämför bara aktien med sig själv — läs med förbehållen.
  Wolf            wolf_shadow_backtest.Backtester (Pine-speglingen: Core/Trim, fyra regimlager, delexits) med en
                  kopierad config — inte PORTFOLIO → Backtest-sidan, som skriver i delad config och larmar.
Resultatet normaliseras (Result): nyckeltal, affärslista och kursgraf med entry/exit-markörer och stoppen,
märkt med fyllnadsregeln. Strategier utan per-aktie-motor (Long Screener, Alpha, EMBER, arken) säger det
rakt ut och pekar på Valhalls aggregerade backtester. Inga regler ändras.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from thor import cards
from thor import registry as reg
from thor.ticker import Ticker
from ui.tokens import AMBER, CYAN, DIM, GREEN, GREY, RED, TEXT

PERIODS = {"3 år": 3, "5 år": 5, "10 år": 10}
DEFAULT_PERIOD = "5 år"
FILL_NEXT_OPEN = "signal på stängning · entry nästa öppning · stopp intradag"
FILL_MOMENTUM = "ranking fredag stängning · köp nästa öppning · stopp −10 % intradag"
FILL_CLOSE = "Pine-spegling: beslut och fyllnad på stängningen"
WOLF_NOTES = ("Wolf x Shadow Backtester: Core/Trim-positioner, fyra regimlager (SPY, sektor-ETF, aktien, Ichimoku), "
              "delexits vid TP1/TP2 och dags-/veckobrytare — samma config som Pine-strategin, kopierad för körningen.",
              "Affärsraderna är delstängningar (CORE/TRIM) — en position kan ge flera rader.",
              "Courtage 0,05 % och slippage 0,1 % per transaktion.")
MOMENTUM_NOTES = ("Veckorutinen: ranking varje veckas sista handelsdag (50 % 3-mån + 50 % 6-mån), kval pris > MA200, "
                  "3-mån > 0, 6-mån ≥ +10 %; köp nästa öppning; stopp −10 % intradag; stängning under MA50 → nästa "
                  "öppning; +20 % → halva säljs och stoppen flyttas till entry.",
                  "Regimen (GRÖN/GUL/RÖD) läses ur OMXSPI mot MA200 och panelens bredd — här är bredden aktiens egen.",
                  "Courtage 0,1 % per transaktion.")


@dataclass
class Result:
    strategy: str
    label: str
    years: int
    fill: str = ""
    period: tuple = ("", "")
    metrics: list = field(default_factory=list)      # [(etikett, värde, färg, underrad)] — cards.kpi_row
    trades: list = field(default_factory=list)       # [{signal, entry_date, entry, stop, exit_date, exit, result, reason, days, open}]
    notes: tuple = ()
    warnings: list = field(default_factory=list)
    close: Optional[pd.Series] = None
    error: str = ""


# ── Hjälpare ────────────────────────────────────────────────────────────────
def _frame(df):
    if df is None or len(df) == 0:
        return None
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def _close(df) -> Optional[pd.Series]:
    df = _frame(df)
    return df["Close"].astype(float).dropna() if df is not None and "Close" in df else None


def _day(x) -> str:
    return "" if x is None else str(x)[:10]


def _num(x) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _pf_text(pf) -> str:
    if pf is None:
        return "—"
    return "∞" if pf == math.inf else f"{pf:.2f}"


def metrics_r(m: dict) -> list:
    """Nyckeltalen i R (viking_backtest.metrics) som kpi-rader."""
    if not m or not m.get("trades"):
        return [("Affärer", "0", GREY, "inga stängda affärer i perioden")]
    exp, pf, total = m.get("expectancy") or 0.0, m.get("profit_factor"), m.get("total_r") or 0.0
    pf_col = GREY if pf is None else GREEN if pf == math.inf or pf >= 1.5 else AMBER if pf >= 1.0 else RED
    return [("Affärer", str(m["trades"]), CYAN, f"träff {m.get('win_rate', 0):.0f} % · snitt {m.get('avg_holding_days', 0):.0f} dagar"),
            ("Förväntan", f"{exp:+.2f} R", GREEN if exp > 0 else RED, f"snitt {m.get('avg_r', 0):+.2f} R · median {m.get('median_r', 0):+.2f} R"),
            ("Profit factor", _pf_text(pf), pf_col, f"vinst {m.get('avg_winner', 0):+.2f} R · förlust {m.get('avg_loser', 0):+.2f} R"),
            ("Max DD", f"{m.get('max_drawdown_r', 0):.1f} R", RED, f"{m.get('max_consecutive_losses', 0)} förluster i rad"),
            ("Summa", f"{total:+.1f} R", GREEN if total > 0 else RED, f"{m['trades']} stängda affärer")]


def metrics_pct(m: dict) -> list:
    """Nyckeltalen i procent (swing_backtest.metrics) som kpi-rader."""
    if not m or not m.get("trades"):
        return [("Affärer", "0", GREY, f"{(m or {}).get('open', 0)} öppen" if (m or {}).get("open") else "inga stängda affärer i perioden")]
    pf, cagr, dd = m.get("profit_factor"), m.get("cagr_pct"), m.get("max_dd_pct")
    pf_col = GREY if pf is None else GREEN if pf == math.inf or pf >= 1.5 else AMBER if pf >= 1.0 else RED
    return [("Affärer", str(m["trades"]), CYAN, f"träff {m.get('win_rate', 0):.0f} % · snitt {m.get('avg_days', 0):.0f} dagar"),
            ("Avkastning", f"{m.get('total_return_pct', 0):+.1f} %", GREEN if (m.get("total_return_pct") or 0) > 0 else RED,
             f"CAGR {cagr:+.1f} %" if cagr is not None else "CAGR —"),
            ("Profit factor", _pf_text(pf), pf_col, f"vinst {m.get('avg_win_pct', 0):+.1f} % · förlust {m.get('avg_loss_pct', 0):+.1f} %"),
            ("Max DD", f"{dd:.1f} %" if dd is not None else "—", RED, f"exponering {m.get('exposure_pct') or 0:.0f} %"),
            ("Halva sålda", f"{m.get('half_sold_share', 0):.0f} %", AMBER, "andel affärer som nådde +20 %")]


def rows_r(trades: list, signal_of: Callable) -> list:
    """viking_backtest.Trade (Viking Nine och BERSERK) → affärsrader."""
    out = []
    for t in trades:
        out.append({"signal": signal_of(t), "entry_date": _day(t.entry_date), "entry": t.entry, "stop": t.stop,
                    "exit_date": _day(t.exit_date), "exit": t.exit,
                    "result": "öppen" if t.open or t.r is None else f"{t.r:+.2f} R", "r": t.r,
                    "reason": t.exit_reason or "", "days": t.days, "open": bool(t.open)})
    return out


# ── Motorerna ───────────────────────────────────────────────────────────────
def run_viking(ident: Ticker, years: int, prices: Callable, today=None) -> Result:
    import ovtlyr_nine as on
    import viking_backtest as vb
    cfg = vb.Config(years=years, risk_gate=())
    res = vb.run([ident.yf], getter=prices, cfg=cfg, today=today)
    per = (res.get("per_ticker") or [{}])[0]
    p = res.get("period") or {}
    out = Result("viking_nine", "Viking Nine", years, FILL_NEXT_OPEN, (p.get("start", ""), p.get("end", "")),
                 notes=tuple(res.get("notes") or ()))
    if per.get("error"):
        out.error = str(per["error"])
        return out
    trades = per.get("trades") or []
    out.metrics = metrics_r(vb.metrics(trades)) + [
        ("Signaler", str(per.get("signals", 0)), DIM,
         f"ej jagade {per.get('no_chase', 0)} · låg R:R {per.get('low_rr', 0)} · marknad {per.get('market') or '—'}")]
    out.trades = rows_r(trades, lambda t: f"Nine {t.nine}/9")
    if ident.nordic and per.get("market") != on.NORDIC_LABEL:
        out.warnings.append("OMXS30 och den svenska bredden saknas (Börsdata) — aktien testades mot SPY och USA:s bredd.")
    out.warnings.append("Snabbläge: marknadsriskspärren (🌩️) är av och sektorn är dagens.")
    out.close = _close(prices(ident.yf, f"{years + 1}y"))
    return out


def run_berserk(ident: Ticker, years: int, prices: Callable, today=None) -> Result:
    import viking_backtest as vb
    from berserk import backtest as bb
    cfg = bb.live_config(years=years)
    res = bb.run([ident.yf], getter=prices, cfg=cfg, today=today)
    per = (res.get("per_ticker") or [{}])[0]
    p = res.get("period") or {}
    out = Result("berserk", "BERSERK", years, FILL_NEXT_OPEN, (p.get("start", ""), p.get("end", "")),
                 notes=tuple(res.get("notes") or ()))
    if per.get("error"):
        out.error = str(per["error"])
        return out
    trades = per.get("trades") or []
    sig = per.get("signals") or {}
    out.metrics = metrics_r(vb.metrics(trades)) + [
        ("Signaler", str(sum(sig.values())), DIM,
         " · ".join(f"{k.split()[0]} {v}" for k, v in sig.items()) + f" · marknadsspärr {per.get('market_blocked', 0)}")]
    out.trades = rows_r(trades, lambda t: str((t.features or {}).get("setup") or ""))
    drv = (res.get("drivers") or {}).get(per.get("theme")) or per.get("driver") or "—"
    out.warnings.append(f"Drivare {drv} (temat {per.get('theme') or '—'}) · skannerns regler (live_config).")
    out.close = _close(prices(ident.yf, f"{years + 1}y"))
    return out


def run_momentum(ident: Ticker, years: int, prices: Callable, today=None) -> Result:
    import swing_backtest as sb
    out = Result("momentum", "Momentum Swing", years, FILL_MOMENTUM, notes=MOMENTUM_NOTES)
    df = _frame(prices(ident.yf, f"{years + 2}y"))
    if df is None or len(df) < sb.WARMUP_BARS:
        out.error = f"DATA UNAVAILABLE — Momentum kräver {sb.WARMUP_BARS} dagars historik (MA200 och 52 veckor)."
        return out
    idx = _close(prices("^OMXSPI", f"{years + 2}y"))
    pn = sb.panels({ident.yf: df}, idx)
    res = sb.run(pn, idx, cfg=sb.Config(years=years), today=today)
    end = pd.Timestamp(today) if today is not None else df.index[-1]
    out.period = (_day(end - pd.DateOffset(years=years)), _day(end))
    out.metrics = metrics_pct(res.get("metrics") or {})
    out.trades = [{"signal": f"{t.setup} · {t.regime}", "entry_date": _day(t.entry_date), "entry": t.entry,
                   "stop": round(float(t.entry) * (1 - sb.Config().stop_pct / 100), 2), "exit_date": _day(t.exit_date),
                   "exit": t.exit, "result": "öppen" if t.open or t.ret_pct is None else f"{t.ret_pct:+.1f} %",
                   "r": None, "reason": t.reason or "", "days": t.days, "open": bool(t.open)}
                  for t in res.get("trades") or []]
    out.warnings.append("En-aktieportfölj: rankingen jämför bara aktien med sig själv (rad 1 så fort kvalkraven är "
                        "uppfyllda) och bredden är aktiens egen — läs som 'vad reglerna hade gjort med den här aktien ensam'.")
    if idx is None:
        out.warnings.append("OMXSPI saknas — regimen blir OKÄND och regimfiltret släpper inte igenom några köp.")
    out.close = df["Close"].astype(float)
    return out


def run_wolf(ident: Ticker, years: int, prices: Callable, today=None) -> Result:
    import wolf_shadow_backtest as wsb
    cfg = dict(wsb.CONFIG)
    out = Result("wolf", "Wolf", years, FILL_CLOSE, notes=WOLF_NOTES)
    df = _frame(prices(ident.yf, f"{years + 1}y"))
    if df is None or len(df) < 250 or "Volume" not in df:
        out.error = "DATA UNAVAILABLE — Wolf kräver 250 dagars kurshistorik med volym."
        return out
    spy, sec = _frame(prices("SPY", f"{years + 1}y")), _frame(prices(ident.sector_etf or "SPY", f"{years + 1}y"))
    if spy is None or sec is None:
        out.error = "DATA UNAVAILABLE — SPY eller sektor-ETF:en saknas (regimlagren)."
        return out
    d = wsb.calc_regime_scores(spy, sec, wsb.add_indicators(df, cfg)).dropna()
    if len(d) < 201:
        out.error = "DATA UNAVAILABLE — för kort historik efter indikatorernas uppvärmning."
        return out
    out.period = (_day(d.index[0]), _day(d.index[-1]))
    results = wsb.Backtester(d, cfg=cfg).run()
    out.close = d["Close"].astype(float)
    out.warnings.append(f"Sektor-ETF {ident.sector_etf or 'SPY'} · regimlagren räknas på SPY, sektorn och aktien.")
    if not results:
        out.metrics = [("Affärer", "0", GREY, "inga affärer i perioden")]
        return out
    m = results.get("metrics") or {}
    ret, pf, wr = _num(m.get("Total Return %")) or 0.0, _num(m.get("Profit Factor")), _num(m.get("Winrate %")) or 0.0
    pf_col = GREY if pf is None else GREEN if pf >= 1.5 else AMBER if pf >= 1.0 else RED
    out.metrics = [("Affärer", str(int(_num(m.get("Total Trades")) or 0)), CYAN, f"träff {wr:.0f} % · snitt {_num(m.get('Avg Bars Held')) or 0:.0f} dagar"),
                   ("Avkastning", f"{ret:+.1f} %", GREEN if ret > 0 else RED, f"CAGR {_num(m.get('CAGR %')) or 0:+.1f} %"),
                   ("Profit factor", _pf_text(pf), pf_col, f"snitt R:R {_num(m.get('Avg R:R')) or 0:.2f}"),
                   ("Max DD", f"{_num(m.get('Max Drawdown %')) or 0:.1f} %", RED, f"Calmar {_num(m.get('Calmar Ratio')) or 0:.2f}"),
                   ("Sharpe", f"{_num(m.get('Sharpe Ratio')) or 0:.2f}", AMBER, f"Sortino {_num(m.get('Sortino Ratio')) or 0:.2f}")]
    tdf = results.get("trades")
    if tdf is not None and len(tdf):
        for row in tdf.to_dict("records"):
            pct = _num(row.get("pnl_pct"))
            out.trades.append({"signal": str(row.get("type") or ""), "entry_date": _day(row.get("entry_date")),
                               "entry": row.get("entry_price"), "stop": None, "exit_date": _day(row.get("exit_date")),
                               "exit": row.get("exit_price"), "result": f"{pct:+.1f} %" if pct is not None else "—",
                               "r": None, "reason": str(row.get("reason") or ""), "days": int(_num(row.get("bars_held")) or 0),
                               "open": False})
    return out


ENGINES = {"viking_nine": run_viking, "viking": run_viking, "berserk": run_berserk, "momentum": run_momentum, "wolf": run_wolf}


def run(strategy: str, ident: Ticker, years: int, prices: Callable, today=None) -> Result:
    """Kör strategins motor på aktien — ett fel blir Result.error, aldrig ett undantag på sidan."""
    fn = ENGINES.get(strategy)
    if fn is None:
        return Result(strategy, reg.label(strategy), years, error="ingen backtestmotor per aktie")
    try:
        return fn(ident, years, prices, today)
    except Exception as exc:
        return Result(strategy, reg.label(strategy), years, error=f"{type(exc).__name__}: {exc}")


# ── Grafen och fliken ───────────────────────────────────────────────────────
def chart(res: Result, ticker: str):
    """Stängningskursen med entry (▲), exit (▼, grön vinst/röd förlust) och stoppen som streckad linje."""
    if res.close is None or not len(res.close):
        return None
    import plotly.graph_objects as go
    c = res.close
    start = pd.Timestamp(res.period[0]) if res.period and res.period[0] else None
    if start is not None:
        c = c[c.index >= start - pd.DateOffset(days=30)]
    fig = go.Figure(go.Scatter(x=c.index, y=c.values, name=ticker, line=dict(color=CYAN, width=1.4), hovertemplate="%{y:.2f}"))
    ent = [(pd.Timestamp(t["entry_date"]), t["entry"]) for t in res.trades if t.get("entry_date") and _num(t.get("entry")) is not None]
    if ent:
        fig.add_trace(go.Scatter(x=[d for d, _p in ent], y=[p for _d, p in ent], mode="markers", name="entry",
                                 marker=dict(symbol="triangle-up", size=10, color=GREEN)))
    for good in (True, False):
        ex = [(pd.Timestamp(t["exit_date"]), t["exit"]) for t in res.trades
              if t.get("exit_date") and _num(t.get("exit")) is not None and (str(t.get("result", "")).startswith("+") == good)]
        if ex:
            fig.add_trace(go.Scatter(x=[d for d, _p in ex], y=[p for _d, p in ex], mode="markers",
                                     name="exit vinst" if good else "exit förlust",
                                     marker=dict(symbol="triangle-down", size=10, color=GREEN if good else RED)))
    for t in res.trades:
        if _num(t.get("stop")) is None or not t.get("entry_date"):
            continue
        x0 = pd.Timestamp(t["entry_date"])
        x1 = pd.Timestamp(t["exit_date"]) if t.get("exit_date") else c.index[-1]
        fig.add_shape(type="line", x0=x0, x1=x1, y0=t["stop"], y1=t["stop"], line=dict(color=RED, width=1, dash="dot"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11), showlegend=True,
                      legend=dict(orientation="h", y=1.08, x=0), xaxis=dict(showgrid=False),
                      yaxis=dict(gridcolor="rgba(255,255,255,0.06)"), hovermode="x unified")
    return fig


def trade_table(trades: list) -> pd.DataFrame:
    return pd.DataFrame([{"Signal": t.get("signal"), "In": t.get("entry_date"), "Entry": t.get("entry"), "Stopp": t.get("stop"),
                          "Ut": t.get("exit_date") or ("öppen" if t.get("open") else ""), "Exit": t.get("exit"),
                          "Resultat": t.get("result"), "Dagar": t.get("days"), "Skäl": t.get("reason")} for t in trades])


def render(st, ident: Ticker, strategy: str, prices: Callable, runner: Optional[Callable] = None) -> bool:
    """Rita fliken. True när strategin har en motor (annars skriver vi varför och sidan lägger bryggorna)."""
    from ui.components import note
    label = reg.label(strategy)
    if strategy not in ENGINES:
        note(f"{label} har ingen backtestmotor per aktie — signalen bygger på fundamenta, cykel eller arket, inte på "
             f"en regel som går att spela upp på en kurs. Valhalls backtester visar strategins aggregerade beslut "
             f"(segment och teman, 2008–i dag).")
        return False
    if hasattr(st, "pills"):
        period = st.pills("Period", list(PERIODS), default=DEFAULT_PERIOD, selection_mode="single", key="thor_bt_period",
                          label_visibility="collapsed") or DEFAULT_PERIOD
    else:                                                       # pragma: no cover - äldre Streamlit
        period = st.selectbox("Period", list(PERIODS), index=1, key="thor_bt_period")
    years = PERIODS[period]
    key = f"thor_bt_{ident.yf}_{strategy}_{years}"
    if st.button(f"▶ Kör {label} på {ident.yf} · {period}", key="thor_bt_run", width="stretch"):
        with st.spinner(f"{label} på {ident.yf}, {period} …"):
            st.session_state[key] = (runner or run)(strategy, ident, years, prices)
    res = st.session_state.get(key)
    if res is None:
        note(f"Tryck Kör, så spelas {label}s regler upp på {ident.yf} de senaste {period}en — nyckeltal, affärer och graf. "
             f"Körningen sparas i sessionen.")
        return True
    chips = [cards.chip(res.fill, AMBER)]
    if res.period and res.period[0]:
        chips.append(cards.chip(f"{res.period[0]} → {res.period[1]}", DIM))
    st.markdown("<div style='display:flex;flex-wrap:wrap;gap:4px;margin:4px 0;'>" + "".join(chips) + "</div>", unsafe_allow_html=True)
    if res.error:
        note(f"{label} på {ident.yf}: {res.error}")
        return True
    st.markdown(cards.kpi_row(res.metrics), unsafe_allow_html=True)
    fig = chart(res, ident.yf)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="thor_bt_chart")
    if res.trades:
        st.dataframe(trade_table(res.trades), hide_index=True, use_container_width=True)
    else:
        note("Inga affärer i perioden — reglerna gav ingen entry på den här aktien.")
    for w in res.warnings:
        note(w)
    if res.notes:
        with st.expander("Förbehåll och regler", expanded=False):
            for n in res.notes:
                st.caption(n)
    return True
