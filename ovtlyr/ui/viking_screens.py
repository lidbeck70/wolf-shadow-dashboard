"""
ovtlyr/ui/viking_screens.py — SCREENING → Arc Screener → ⚔️ Viking Nine.

Kör OVTLYR Nine + Viking Execution över en lista tickers och visar
OVTLYR SCREEN, VIKING MOMENTUM SCREEN, bevakningslistan och signalloggen.
Screening ≠ automatisk entry.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

import storage
import viking_execution as vx
import viking_screen as vs
from ui.components import note, page_header
from ui.tokens import AMBER, CYAN, DIM, GOLD, GREEN, RED, TEXT

_ROWS = "vn_screen_rows"
_CAT_COLOR = {"GOLDEN TICKET": GOLD, "READY": GREEN, "DEVELOPING": AMBER, "REJECTED": RED}


def _default_universe() -> str:
    res = st.session_state.get("ovtlyr_results")
    try:
        if res is not None and len(res) and "Ticker" in res:
            return ", ".join(list(res["Ticker"])[:30])
    except Exception:
        pass
    return ""


def _earnings(ticker: str):
    try:
        from earnings_calendar import _fetch_earnings_date
        return (_fetch_earnings_date(ticker) or {}).get("date")
    except Exception:
        return None


def _sector(ticker: str):
    try:
        from ovtlyr.ui.layout import _sector_of
        return _sector_of(ticker)
    except Exception:
        import ovtlyr_nine as on
        return on._sector_default(ticker)


def _table(rows: list, extra=None) -> str:
    head = ("<tr style='color:%s;'><th style='text-align:left;'>Ticker</th><th>Nine</th><th>Viktat</th>"
            "<th>M</th><th>S</th><th>St</th><th>Viking</th><th>Beslut</th><th>Kategori</th><th>Entry</th>"
            "<th>Stopp</th><th>R/R</th>%s</tr>") % (DIM, "<th style='text-align:left;'>Saknas</th>" if extra else "")
    body = []
    for r in rows:
        n, d = r["nine"], r["decision"]
        if n is None or d is None:
            body.append(f"<tr><td style='text-align:left;'>{r['ticker']}</td>"
                        f"<td colspan='11' style='color:{DIM};text-align:left;'>{r['error']}</td></tr>")
            continue
        p = d.position
        cc = _CAT_COLOR.get(r["category"], TEXT)
        rr = "fri väg" if d.rr is None and d.resistance is None else "—" if d.rr is None else f"{d.rr:.2f}R"
        body.append(
            f"<tr><td style='text-align:left;'><b>{r['ticker']}</b></td><td>{n.passed}/9</td><td>{n.weighted:g}</td>"
            f"<td>{n.layer_passed('market')}/3</td><td>{n.layer_passed('sector')}/2</td>"
            f"<td>{n.layer_passed('stock')}/4</td><td>{d.execution_passed}/{d.execution_total}</td>"
            f"<td>{d.status}</td><td style='color:{cc};font-weight:700;'>{r['category']}</td>"
            f"<td>{'—' if d.entry is None else f'{d.entry:,.2f}'}</td>"
            f"<td>{'—' if p is None else f'{p.stop:,.2f}'}</td><td>{rr}</td>"
            + (f"<td style='text-align:left;color:{DIM};'>{extra(r)}</td>" if extra else "") + "</tr>")
    return (f"<div style='overflow-x:auto;'><table style='width:100%;font-size:0.76rem;color:{TEXT};"
            f"text-align:right;'>{head}{''.join(body)}</table></div>")


def render_viking_nine_page() -> None:
    page_header("⚔️ Viking Nine", "OVTLYR Nine + Viking Execution över en lista tickers — screening, "
                                  "bevakningslista och signallogg. Screening ≠ automatisk entry.")
    with st.form("vn_form", clear_on_submit=False):
        raw = st.text_area("Tickers (komma eller radbrytning, högst %d)" % vs.MAX_TICKERS, _default_universe(),
                           key="vn_tickers", height=80, placeholder="t.ex. NVDA, MSFT, VOLV-B.ST, EQNR.OL")
        c1, c2 = st.columns(2)
        capital = c1.number_input("Kapital (SEK)", min_value=0.0, value=100000.0, step=10000.0, key="vn_capital")
        need_vol = c2.checkbox("Kräv relativ volym ≥ %.2f× i momentum-screenen" % vx.RELATIVE_VOLUME_MIN,
                               value=False, key="vn_need_vol")
        go = st.form_submit_button("⚔️ Kör screen")
    note("Förvalda tickers = de 30 första från senaste Viking-screenern (Arc Screener → Viking), om du kört den. "
         "Varje ticker hämtar ett års dagsdata och sektor; SPY och sektor-ETF:erna delas.")
    tickers = vs.parse_tickers(raw)
    log = storage.session_load(vs.LOG_STORE, [])
    if go and tickers:
        bar = st.progress(0.0, text="Startar …")
        try:
            from trade_journal import load_journal
            trades = load_journal()
        except Exception:
            trades = []
        rows = vs.run(tickers, progress=lambda i, n, t: bar.progress(i / n, text=f"{t} ({i}/{n})"),
                      sector_getter=_sector, earnings_getter=_earnings, capital=capital, trades=trades)
        bar.empty()
        st.session_state[_ROWS] = {"rows": rows, "when": datetime.now().strftime("%Y-%m-%d %H:%M")}
        new_log, n = vs.append_log(log, rows)
        if n:
            st.session_state[vs.LOG_STORE] = new_log
            log = new_log
    data = st.session_state.get(_ROWS)
    if not data:
        note("Skriv tickers och tryck ⚔️ Kör screen.")
        rows = []
    else:
        rows = data["rows"]
        st.markdown(f"<div style='color:{DIM};font-size:0.75rem;'>Senaste körning {data['when']} · "
                    f"{len(rows)} tickers</div>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["OVTLYR SCREEN", "VIKING MOMENTUM SCREEN", "BEVAKNINGSLISTA", "SIGNALLOGG"])
    with t1:
        if rows:
            st.markdown(_table(vs.ovtlyr_screen(rows)), unsafe_allow_html=True)
            note("Alla tickers sorterade på OVTLYR Nine (antal godkända, sedan viktad poäng). M/S/St = "
                 "Market/Sector/Stock. Allt är WOLF APPROXIMATION — riktiga priser, panelens definitioner.")
    with t2:
        if rows:
            hits = vs.momentum_screen(rows, need_vol)
            if hits:
                st.markdown(_table(hits), unsafe_allow_html=True)
            else:
                note("Ingen ticker klarar momentum-screenen just nu.")
            with st.expander("Varför föll de andra?"):
                rest = [r for r in vs.ovtlyr_screen(rows) if r not in hits]
                st.markdown(_table(rest, extra=lambda r: ", ".join(vs.momentum_pass(r, need_vol)[1])),
                            unsafe_allow_html=True)
            note(f"Krav: Nine ≥ {vs.MOMENTUM_MIN_NINE}/9 · kurs > EMA10 > EMA20 > EMA50 · RSI > {vx.RSI_MIN:g} och "
                 f"stigande · inget bearish block nära · R/R ≥ {vx.MINIMUM_RR:g}"
                 + (" · relativ volym" if need_vol else "") + ". Hittar kandidater — ingen automatisk entry.")
    with t3:
        if rows:
            wl = vs.watchlist(rows)
            for cat in vs.CATEGORY_ORDER:
                items = wl.get(cat, [])
                st.markdown(f"<div style='color:{_CAT_COLOR[cat]};font-weight:700;letter-spacing:0.08em;"
                            f"margin-top:10px;'>{cat} <span style='color:{DIM};font-weight:400;'>({len(items)})"
                            f"</span></div>", unsafe_allow_html=True)
                if items:
                    st.markdown(_table(items), unsafe_allow_html=True)
            note("GOLDEN TICKET = 9/9 + Viking Execution klar · READY = 9/9 men entryn väntar · DEVELOPING = 7–8/9 "
                 "· REJECTED = ≤ 6/9. Kategorin säger hur många villkor som är uppfyllda — inte aktiens kvalitet.")
    with t4:
        _render_log(log)


def _render_log(log: list) -> None:
    if not log:
        note("Signalloggen är tom. GOLDEN TICKET och READY loggas automatiskt när screenen körs "
             "(en rad per ticker och dag).")
        return
    cols = ("timestamp", "ticker", "ovtlyr_nine", "market", "sector", "stock", "viking", "entry", "stop", "atr",
            "shares", "rr", "action", "reason")
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{'—' if e.get(c) is None else e.get(c)}</td>" for c in cols) + "</tr>"
                   for e in reversed(log[-200:]))
    st.markdown(f"<div style='overflow-x:auto;max-height:420px;'><table style='width:100%;font-size:0.72rem;"
                f"color:{TEXT};'><tr style='color:{DIM};'>{head}</tr>{body}</table></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3])
    if storage.is_dirty(vs.LOG_STORE):
        if c1.button("💾 Spara signalloggen", key="vn_save_log"):
            try:
                storage.save_session(vs.LOG_STORE)
                c2.success("Sparad.")
            except Exception as exc:
                c2.error(f"Kunde inte spara: {exc}")
        else:
            c2.markdown(f"<span style='color:{AMBER};font-size:0.8rem;'>Osparade rader i loggen.</span>",
                        unsafe_allow_html=True)
    note(f"{len(log)} rader (högst {vs.LOG_MAX}). Varje rad har tidsstämpel och triggerdatum — det som var "
         "känt när signalen loggades. Sparas i repots datalager.")
    st.markdown(f"<div style='color:{CYAN};font-size:0.7rem;'>Lagras i {storage.path_for(vs.LOG_STORE)}</div>",
                unsafe_allow_html=True)
