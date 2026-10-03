"""
ovtlyr/ui/viking_screens.py — SCREENING → Arc Screener → ⚔️ Viking Nine.

Skannar ett helt universum (samma marknader som Viking-screenern) i två steg
och visar EN resultatlista (kategori, Nine, momentumkrav — med filter för
momentumkandidater) och signalloggen. Marknadslagret (SPY + bredd) är gemensamt och visas överst.
Screening ≠ automatisk entry — beslutet tas i REGIME → Viking Regime.
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


def _regions() -> tuple:
    """(alla marknader, funktion marknader → tickers) ur ticker_universe, som Viking-screenern."""
    try:
        from ticker_universe import COUNTRY_REGIONS, get_tickers_for_regions
        return list(COUNTRY_REGIONS), get_tickers_for_regions
    except Exception:
        return [], None


def _earnings(ticker: str):
    try:
        from earnings_calendar import _fetch_earnings_date
        return (_fetch_earnings_date(ticker) or {}).get("date")
    except Exception:
        return None


def _risk_for(ticker: str):
    try:
        import market_risk_gate as mg
        return mg.for_ticker(ticker)
    except Exception:
        return None


def _sector(ticker: str):
    """Yahoos sektor (reserv — Börsdata prövas först i ovtlyr_nine.resolve_sector).
    Bara lyckade svar cachas länge; ett strypt Yahoo-svar provas igen efter tio minuter."""
    import ovtlyr_nine as on
    return on._sector_default(ticker)


def _table(rows: list, extra=None) -> str:
    head = ("<tr style='color:%s;'><th style='text-align:left;'>Ticker</th><th>Nine</th><th>Viktat</th>"
            "<th>M</th><th>S</th><th>St</th><th>Viking</th><th>Beslut</th><th>Kategori</th><th>Entry</th>"
            "<th>Stopp</th><th>R/R</th>%s</tr>") % (DIM, "<th style='text-align:left;'>Momentum</th>" if extra else "")
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


def _risk_banner(markets) -> None:
    """🌩️ Marknadsrisk för marknaderna — spärrar nya Viking Nine-entries per marknad."""
    try:
        import market_risk_gate as mg
    except ImportError:
        return
    parts = []
    for m in markets:
        r = mg.current(m)
        if r is None:
            parts.append(f"<span style='color:{DIM};'>{m}: okänd</span>")
            continue
        col = RED if r["level"] == mg.HIGH else AMBER if r["level"] == mg.ELEVATED else GREEN
        parts.append(f"<span style='color:{col};font-weight:700;'>{r['label']}: {r['level']}</span>"
                     f" <span style='color:{DIM};'>({r['points']}/{r['possible']})</span>")
    blocked = [m for m in markets if mg.blocks_viking_entry(mg.current(m))]
    st.markdown(f"<div style='font-size:0.8rem;margin:2px 0 6px;'>🌩️ Marknadsrisk: {' · '.join(parts)}"
                + (f"<div style='color:{RED};font-size:0.75rem;'>Inga nya Viking Nine-entries: {', '.join(blocked)} "
                   f"— spärr {mg.viking_rule_text()} (REGIME → 🌩️ Marknadsrisk).</div>" if blocked else "") + "</div>", unsafe_allow_html=True)


def _market_banner(markets=("SPY", "OMXS30")) -> None:
    """Marknadslagret i OVTLYR Nine — SPY för amerikanska aktier, OMXS30 för nordiska."""
    import ovtlyr_nine as on
    blocks = []
    for m in markets:
        try:
            facts = on.market_layer(m)
        except Exception as exc:
            note(f"Marknadslagret ({m}) kunde inte räknas: {exc}")
            continue
        n = sum(f.passed for f in facts)
        col = GREEN if n == 3 else AMBER if n else RED
        marks = " · ".join(f"<span style='color:{GREEN if f.passed else RED if f.status == 'FAIL' else DIM};'>"
                           f"{'✓' if f.passed else '✗' if f.status == 'FAIL' else '?'} {f.label}</span>"
                           for f in facts)
        blocks.append((m, n, col, marks))
    if not blocks:
        return
    rows = "".join(f"<div><b style='color:{col};'>MARKET {m} {n}/3</b> <span style='color:{TEXT};'>{marks}</span>"
                   f"</div>" for m, n, col, marks in blocks)
    worst = min(n for _m, n, _c, _k in blocks)
    col = GREEN if worst == 3 else AMBER if worst else RED
    st.markdown(f"<div style='border:1px solid {col};border-radius:6px;padding:6px 10px;margin:6px 0;'>{rows}"
                + ("" if worst == 3 else f"<div style='color:{DIM};font-size:0.75rem;'>En marknad som inte är 3/3 "
                   f"gör att ingen aktie på den kan nå 9/9 i dag. Nordiska aktier mäts mot OMXS30, övriga mot "
                   f"SPY.</div>") + "</div>", unsafe_allow_html=True)


def render_viking_nine_page() -> None:
    page_header("⚔️ Viking Nine", "OVTLYR Nine + Viking Execution över hela universumet — hittar setups. "
                                  "Beslutet tas i REGIME → Viking Regime. Screening ≠ automatisk entry.")
    _market_banner()
    _risk_banner(("SPY", "OMXS30"))
    regions, tickers_for = _regions()
    with st.form("vn_form", clear_on_submit=False):
        if regions:
            markets = st.multiselect("MARKETS", regions, default=[r for r in ("Norden",) if r in regions],
                                     key="vn_markets")
        else:
            markets = []
        c1, c2 = st.columns(2)
        n_cand = c1.number_input("Kandidater till steg 2", min_value=5, max_value=vs.CANDIDATES_CAP,
                                 value=vs.MAX_CANDIDATES, step=5, key="vn_candidates")
        capital = c2.number_input("Kapital (SEK)", min_value=0.0, value=100000.0, step=10000.0, key="vn_capital")
        extra = st.text_input("Egna tickers (läggs till universumet, valfritt)", key="vn_extra",
                              placeholder="t.ex. NVDA, MSFT")
        go = st.form_submit_button("⚔️ SCAN")
    note("Steg 1 skannar hela universumet på stängningskurser: aktiens Trend (EMA10 > EMA20, kurs > EMA50) och "
         "Signal (kurs ≥ EMA20) — utan dem kan Nine aldrig bli 9/9. Klara aktier rangordnas på avkastning "
         "tre månader. Steg 2 ger de bästa kandidaterna full OVTLYR Nine och Viking Execution.")
    log = storage.session_load(vs.LOG_STORE, [])
    if go:
        universe = list(tickers_for(markets)) if (tickers_for and markets) else []
        universe += vs.parse_tickers(extra)
        if not universe:
            st.warning("Välj minst en marknad eller skriv egna tickers.")
        else:
            bar = st.progress(0.0, text=f"Steg 1: hämtar {len(universe)} tickers …")
            try:
                from trade_journal import load_journal
                trades = load_journal()
            except Exception:
                trades = []
            res = vs.scan(universe, max_candidates=n_cand,
                          progress=lambda i, n, t: bar.progress(i / n, text=f"Steg 2: {t} ({i}/{n})"),
                          sector_getter=_sector, earnings_getter=_earnings, capital=capital, trades=trades,
                          risk_getter=_risk_for)
            bar.empty()
            st.session_state[_ROWS] = {"rows": res["rows"], "funnel": res["funnel"],
                                       "when": datetime.now().strftime("%Y-%m-%d %H:%M")}
            new_log, n = vs.append_log(log, res["rows"])
            if n:
                st.session_state[vs.LOG_STORE] = new_log
                log = new_log
    data = st.session_state.get(_ROWS)
    if not data:
        note("Välj marknader och tryck ⚔️ SCAN.")
        rows = []
    else:
        rows = data["rows"]
        f = data.get("funnel") or {}
        st.markdown(f"<div style='color:{DIM};font-size:0.78rem;'>Senaste skanning {data['when']} · universum "
                    f"<b style='color:{TEXT};'>{f.get('universe', len(rows))}</b> → med kursdata "
                    f"<b style='color:{TEXT};'>{f.get('data', '—')}</b> → steg 1 (trend + signal) "
                    f"<b style='color:{TEXT};'>{f.get('passed', '—')}</b> → steg 2 analyserade "
                    f"<b style='color:{TEXT};'>{f.get('candidates', len(rows))}</b></div>", unsafe_allow_html=True)

    t1, t2 = st.tabs(["RESULTAT", "SIGNALLOGG"])
    with t1:
        if rows:
            _render_results(rows)
    with t2:
        _render_log(log)


def _render_results(rows: list) -> None:
    counts = {c: sum(1 for r in rows if r["category"] == c and r["nine"] is not None) for c in vs.CATEGORY_ORDER}
    st.markdown("<div style='display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 8px;'>" + "".join(
        f"<div style='border:1px solid {_CAT_COLOR[c]};border-radius:4px;padding:2px 8px;color:{_CAT_COLOR[c]};"
        f"font-weight:700;font-size:0.78rem;'>{c} {n}</div>" for c, n in counts.items()) + "</div>",
        unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    only = c1.checkbox("Bara momentumkandidater", value=False, key="vn_momentum_only")
    need_vol = c2.checkbox("Kräv relativ volym ≥ %.2f×" % vx.RELATIVE_VOLUME_MIN, value=False, key="vn_need_vol")
    shown = vs.results(rows, momentum_only=only, require_volume=need_vol)

    def _why(r):
        ok, why = vs.momentum_pass(r, need_vol)
        return "✓ momentumkandidat" if ok else ", ".join(why)
    if shown:
        st.markdown(_table(shown, extra=_why), unsafe_allow_html=True)
    else:
        note("Ingen ticker klarar momentum-kraven just nu.")
    note("Sorterad på kategori och sedan OVTLYR Nine. GOLDEN TICKET = 9/9 + Viking Execution klar · READY = 9/9 men "
         "entryn väntar · DEVELOPING = 7–8/9 · REJECTED = ≤ 6/9 — hur många villkor som är uppfyllda, inte aktiens "
         f"kvalitet. Momentumkandidat = Nine ≥ {vs.MOMENTUM_MIN_NINE}/9 · kurs > EMA10 > EMA20 > EMA50 · RSI > "
         f"{vx.RSI_MIN:g} och stigande · inget bearish block nära · R/R ≥ {vx.MINIMUM_RR:g}"
         + (" · relativ volym" if need_vol else "") + ". M/S/St = Market/Sector/Stock. WOLF APPROXIMATION.")


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
