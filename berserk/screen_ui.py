"""
berserk/screen_ui.py — SCREENING → Arc Screener → 🪓 BERSERK.

Skannar universumet på senaste stängda dag med backtestets regler:
  KÖP      en setup triggade idag och grindarna (omsättning, marknad) är gröna
           — köp på nästa öppning; planen visar stopp, position och antal aktier
  BEVAKA   setupen är nästan klar (allt utom triggern), eller en trigger som
           stoppas av en grind
Spärrar mot dina innehav (Holdings): ÄGS REDAN, TEMA FULLT (2), KOMPLEX FULLT (4),
MAX 8 POSITIONER. KÖP och BEVAKA loggas i signalloggen (en rad per ticker och dag).
PAPPERSKONTO visar den automatiska körningen (berserk_scan.py, vardagar efter
USA:s stängning): papperskurvan, öppna positioner med stopp, order, avslutade
affärer, händelserna som larmats till Discord och senaste automatiska skanningen.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

import storage
import viking_screen as vs
from berserk import backtest as bt
from berserk import live
from berserk import signals as sg
from berserk import themes as th
from berserk import universe as uv
from berserk import paper
from ovtlyr.ui.viking_robustness_ui import STICKY
from ui.components import kpi, note, page_header
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, GREY, RED, TEXT

_ROWS = "bz_scan"
_AUTO = "bz_auto"
SCAN_BLOB, PAPER_BLOB = "berserk_scan.json", "berserk_paper.json"
LOG_STORE = "berserk_signals"
LOG_MAX = 2000
STATUS_COLOR = {live.KOP: GREEN, live.BEVAKA: AMBER, live.INGET: GREY}
SETUP_COLOR = {sg.S1: CYAN, sg.S2: GOLD, sg.S3: GREEN}


def log_entries(rows: list, now: datetime = None) -> list:
    now = now or datetime.now()
    out = []
    for r in rows:
        if r.get("status") not in (live.KOP, live.BEVAKA) or not r.get("setup"):
            continue
        out.append({"timestamp": now.strftime("%Y-%m-%d %H:%M"), "date": now.strftime("%Y-%m-%d"),
                    "ticker": r["ticker"], "status": r["status"], "setup": r["setup"], "theme": r.get("label"),
                    "region": r.get("region"), "close": r.get("close"), "stop": r.get("stop"),
                    "position_pct": r.get("position_pct"), "why": "; ".join(r.get("why") or [])[:300],
                    "flags": ", ".join(r.get("flags") or [])})
    return out


def append_log(log: list, rows: list, now: datetime = None) -> tuple:
    """(ny logg, antal nya/ändrade) — en rad per ticker och dag, senaste vinner."""
    log = list(log or [])
    n = 0
    for e in log_entries(rows, now):
        idx = next((i for i, x in enumerate(log) if x.get("date") == e["date"] and x.get("ticker") == e["ticker"]),
                   None)
        if idx is None:
            log.append(e)
            n += 1
        elif {k: v for k, v in log[idx].items() if k != "timestamp"} != {k: v for k, v in e.items()
                                                                          if k != "timestamp"}:
            log[idx] = e
            n += 1
    return log[-LOG_MAX:], n


def _num(v) -> str:
    return "—" if v is None else f"{v:,.2f}"


def _table(rows: list) -> str:
    head = ("<tr style='color:%s;'><th style='text-align:left;" + STICKY + "'>Ticker</th><th>Status</th><th>Setup</th>"
            "<th style='text-align:left;'>Tema</th><th>Region</th><th>Stängning</th><th>Stopp</th><th>Position</th>"
            "<th>Antal</th><th style='text-align:left;'>Varför</th></tr>") % DIM
    body = []
    for r in rows:
        if r.get("error"):
            body.append(f"<tr><td style='text-align:left;{STICKY}'>{r['ticker']}</td><td colspan='9' style='color:{DIM};"
                        f"text-align:left;'>{r['error']}</td></tr>")
            continue
        sc = STATUS_COLOR.get(r["status"], GREY)
        setup = r.get("setup") or ""
        flags = "".join(f"<br><span style='color:{RED};font-size:0.66rem;'>{f}</span>" for f in r.get("flags") or [])
        etf = f" <span style='color:{DIM};'>ETF</span>" if r.get("kind") == "etf" else ""
        body.append(
            f"<tr><td style='text-align:left;{STICKY}'><b>{r['ticker']}</b>{etf}</td>"
            f"<td style='color:{sc};font-weight:700;'>{r['status'] or '—'}{flags}</td>"
            f"<td style='color:{SETUP_COLOR.get(setup, DIM)};'>{setup.split(' ', 1)[0] if setup else '—'}</td>"
            f"<td style='text-align:left;'>{r.get('label', '')}</td><td>{r.get('region', '')}</td>"
            f"<td>{_num(r.get('close'))}</td><td>{_num(r.get('stop'))}</td>"
            f"<td>{'—' if r.get('position_pct') is None else str(r['position_pct']) + ' %'}</td>"
            f"<td>{'—' if r.get('shares') is None else r['shares']}</td>"
            f"<td style='text-align:left;color:{DIM};'>{'; '.join(r.get('why') or []) or '—'}</td></tr>")
    return (f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.76rem;color:{TEXT};"
            f"text-align:right;'>{head}{''.join(body)}</table></div>")


def render_berserk_screen_page() -> None:
    page_header("🪓 BERSERK — skanner", "Dagens signaler på senaste stängda dag: köp på nästa öppning, med "
                                        "stopp och storlek enligt backtestets regler.")
    lists = list(uv.LISTS)
    with st.form("bz_scan_form"):
        chosen = st.multiselect("Universum", lists, default=lists, key="bz_scan_lists")
        c1, c2 = st.columns(2)
        capital = c1.number_input("Kapital", min_value=0.0, value=100000.0, step=10000.0, key="bz_scan_capital")
        min_turn = c2.number_input("Minsta omsättning (milj/dag)", 0.0, 100.0, bt.MIN_TURNOVER_M, 0.5,
                                   key="bz_scan_turnover")
        extra = st.text_input("Egna tickers ur universumet (valfritt)", key="bz_scan_extra")
        go = st.form_submit_button("🪓 SKANNA")
    log = storage.session_load(LOG_STORE, [])
    if go:
        tickers = [t for name in chosen for t in uv.LISTS[name]] + vs.parse_tickers(extra, limit=400)
        tickers = list(dict.fromkeys(tickers))
        if not tickers:
            st.warning("Välj minst ett universum.")
        else:
            bar = st.progress(0.0, text="Hämtar drivare och index …")
            res = live.scan(tickers, capital=float(capital), cfg=bt.Config(min_turnover_m=float(min_turn)),
                            progress=lambda i, n, t: bar.progress(i / n, text=f"{t} ({i}/{n})"))
            bar.empty()
            live.portfolio_flags(res["rows"], live.held_tickers())
            st.session_state[_ROWS] = res
            new_log, n = append_log(log, res["rows"])
            if n:
                st.session_state[LOG_STORE] = new_log
                log = new_log
    res = st.session_state.get(_ROWS)
    t1, t2, t3 = st.tabs(["SIGNALER", "SIGNALLOGG", "PAPPERSKONTO"])
    with t1:
        if not res:
            note("Välj universum och tryck 🪓 SKANNA. Kör efter stängning — signalerna bedöms på stängd dag och "
                 "köps på nästa öppning.")
        else:
            render_results(res)
    with t2:
        render_log(log)
    with t3:
        render_paper_tab()


def render_results(res: dict) -> None:
    rows = live.sort_rows(res["rows"])
    counts = {s: sum(1 for r in rows if r.get("status") == s) for s in (live.KOP, live.BEVAKA)}
    st.markdown(f"<div style='color:{DIM};font-size:0.78rem;'>Skannad {res['when']} · {len(rows)} tickers · "
                f"<b style='color:{GREEN};'>{counts[live.KOP]} KÖP</b> · <b style='color:{AMBER};'>"
                f"{counts[live.BEVAKA]} BEVAKA</b></div>", unsafe_allow_html=True)
    show_all = st.checkbox("Visa alla tickers (även utan signal)", value=False, key="bz_scan_all")
    shown = rows if show_all else [r for r in rows if r.get("status") in (live.KOP, live.BEVAKA)]
    if shown:
        st.markdown(_table(shown), unsafe_allow_html=True)
    else:
        note("Inga KÖP eller BEVAKA just nu.")
    missing = [r["ticker"] for r in rows if r.get("error")]
    if missing:
        note(f"Utan data eller okända: {', '.join(missing[:30])}{' …' if len(missing) > 30 else ''}")
    note("Entry = nästa dags öppning (stängningen visas som ungefärligt pris). Stopp: S1/S2 2 × ATR, S3 3 × ATR. "
         "Position = risk per setup (S1/S2 1,25 %, S3 1 %) / stoppavstånd, max 20 %. Spärrarna räknas mot "
         "Holdings: max 2 per tema, 4 per komplex och 8 positioner. Exit enligt setupens regler — "
         "papperskontot förvaltar dem automatiskt (fliken PAPPERSKONTO).")


def render_log(log: list) -> None:
    if not log:
        note("Signalloggen är tom. KÖP och BEVAKA loggas när skannern körs (en rad per ticker och dag).")
        return
    df = pd.DataFrame(list(reversed(log[-300:])))
    st.dataframe(df, hide_index=True, width="stretch")
    c1, c2 = st.columns([1, 3])
    if storage.is_dirty(LOG_STORE):
        if c1.button("💾 Spara signalloggen", key="bz_save_log"):
            try:
                storage.save_session(LOG_STORE)
                c2.success("Sparad.")
            except Exception as exc:
                c2.error(f"Kunde inte spara: {exc}")
    note(f"{len(log)} rader (högst {LOG_MAX}). Lagras i {storage.path_for(LOG_STORE)}.")


# ── Papperskontot ───────────────────────────────────────────────────────────
def load_auto() -> dict:
    """Den schemalagda körningens resultat ur Gisten."""
    from gist_storage import load_blob
    return {"scan": load_blob(SCAN_BLOB, None), "paper": load_blob(PAPER_BLOB, None),
            "loaded": datetime.now().strftime("%Y-%m-%d %H:%M")}


def _df(rows: list, cols: dict) -> pd.DataFrame:
    return pd.DataFrame([{label: r.get(k) for k, label in cols.items()} for r in rows])


def render_paper_tab() -> None:
    c1, c2 = st.columns([1, 3])
    if c1.button("📡 Hämta senaste körningen", key="bz_auto_refresh") or _AUTO not in st.session_state:
        try:
            st.session_state[_AUTO] = load_auto()
        except Exception as exc:
            st.session_state[_AUTO] = {"scan": None, "paper": None, "loaded": "—", "error": str(exc)}
    auto = st.session_state[_AUTO]
    c2.markdown(f"<div style='color:{DIM};font-size:0.78rem;padding-top:8px;'>Hämtad {auto.get('loaded')}</div>",
                unsafe_allow_html=True)
    render_paper(auto.get("paper"), auto.get("scan"))


def render_paper(state, scan) -> None:
    if not state:
        note("Papperskontot har inte startat ännu. Det sköts av arbetsflödet BERSERK Scan (vardagar 23:35 svensk "
             "sommartid / 22:35 vintertid, efter USA:s stängning) — kör det manuellt i GitHub Actions för att "
             "starta direkt. Riktiga order läggs alltid manuellt.")
    else:
        s = paper.summary(state)
        cols = st.columns(5)
        cards = [("Papperskonto", f"{s['equity']:.1f}", GREEN if s["return_pct"] >= 0 else RED,
                  f"{s['return_pct']:+.1f} % sedan {s['since']}"),
                 ("Max drawdown", f"{s['max_dd_pct']:.1f} %", AMBER, "dagsvärderad"),
                 ("Affärer", str(s["trades"]), CYAN,
                  "—" if s["win_rate"] is None else f"{s['win_rate']:.0f} % vinnare"),
                 ("Snitt R", "—" if s["avg_r"] is None else f"{s['avg_r']:+.2f}", GOLD, f"totalt {s['total_r']:+.1f} R"),
                 ("Öppna", f"{s['open']} / 8", TEXT, f"värme {s['heat']:.1f} % · {s['orders']} order")]
        for col, (label, val, color, cap) in zip(cols, cards):
            col.markdown(kpi(label, val, color, cap), unsafe_allow_html=True)
        curve = state.get("curve") or []
        if len(curve) >= 2:
            st.line_chart(pd.DataFrame(curve).set_index("date")[["equity"]], height=220)
        _section("ÖPPNA POSITIONER")
        pos = state.get("positions") or []
        if pos:
            for p in pos:
                p["_r"] = round((p["last_close"] - p["entry"]) / (p["entry"] - p["init_stop"]), 2)
                p["_note"] = f"SÄLJ PÅ ÖPPNING ({p['pending']})" if p.get("pending") else ""
            st.dataframe(_df(pos, {"ticker": "Ticker", "setup": "Setup", "theme": "Tema", "entry_date": "Köpt",
                                   "entry": "Entry", "cur_stop": "Stopp", "last_close": "Senast", "_r": "R",
                                   "_note": "Åtgärd"}), hide_index=True, width="stretch")
        else:
            note("Inga öppna positioner.")
        orders = state.get("orders") or []
        if orders:
            _section("ORDER — KÖP PÅ NÄSTA ÖPPNING")
            st.dataframe(_df(orders, {"ticker": "Ticker", "setup": "Setup", "theme": "Tema",
                                      "signal_date": "Signal", "close": "Stängning", "stop": "Stopp ≈",
                                      "position_pct": "Position %"}), hide_index=True, width="stretch")
        closed = state.get("closed") or []
        if closed:
            _section("AVSLUTADE AFFÄRER")
            st.dataframe(_df(list(reversed(closed[-200:])),
                             {"ticker": "Ticker", "setup": "Setup", "entry_date": "Köpt", "exit_date": "Såld",
                              "entry": "Entry", "exit": "Exit", "r": "R", "result_pct": "Resultat %",
                              "reason": "Skäl", "days": "Dagar"}), hide_index=True, width="stretch")
        events = state.get("events") or []
        if events:
            _section("HÄNDELSER (LARMADE TILL DISCORD)")
            st.markdown("<div style='font-size:0.76rem;line-height:1.6;'>" + "<br>".join(
                f"<span style='color:{DIM};'>{e['date']}</span> {paper.ICON.get(e['kind'], '•')} "
                f"<b style='color:{TEXT};'>{e['kind']}</b> {e['ticker']} — "
                f"<span style='color:{DIM};'>{e['text']}</span>" for e in reversed(events[-40:])) + "</div>",
                unsafe_allow_html=True)
    if scan:
        _section("SENASTE AUTOMATISKA SKANNINGEN")
        cnt = scan.get("counts") or {}
        st.markdown(f"<div style='color:{DIM};font-size:0.78rem;'>Skannad {scan.get('when')} · "
                    f"{scan.get('tickers')} tickers · <b style='color:{GREEN};'>{cnt.get(live.KOP, 0)} KÖP</b> · "
                    f"<b style='color:{AMBER};'>{cnt.get(live.BEVAKA, 0)} BEVAKA</b></div>", unsafe_allow_html=True)
        if scan.get("rows"):
            st.markdown(_table(scan["rows"]), unsafe_allow_html=True)
    note("Papperskontot följer backtestets regler exakt: fyllning på nästa öppning, stopp intradag, "
         "stängningsregler säljer på nästa öppning, max 8 positioner, 2 per tema, 4 per komplex, 6 % värme, "
         "ingen belåning. Kontot räknas i procent av start = 100 och i lokal valuta (som backtestet). Jämför "
         "snitt-R och drawdown med backtestet efter 1–2 månader — det är underlaget för riskskalningen (PR 4). "
         "Riktiga order läggs manuellt.")


def _section(title: str) -> None:
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:12px;'>{title}</div>",
                unsafe_allow_html=True)
