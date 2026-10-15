"""
thor/ui.py — 🔨 THOR: läget, tickersök, rullgardinen, listan och analysblocket på EN sida.

Flödet är screening → tryck → analys → köp, utan att byta flik: tryck "Öppna →" på en rad (eller skriv en
ticker) så öppnas analysblocket överst med aktien och strategin förifyllda, och listan fälls ihop under
det. Sidan räknar inga signaler: listan läses ur körningarna (thor.candidates) och analysen sätts ihop av
motorerna som redan finns (thor.dossier). Regim per strategi, Teknik, Råvara, Backtest och Regler fylls
på i fas 4–6 — tills dess har flikarna bryggknappar till de gamla sidorna med aktien förifylld.

Designen följer Valhall: sidans data i st.session_state[_STATE] med en 📡-knapp, kort som rena HTML-
funktioner (thor/cards.py), bara den öppna sidofliken ritas (st.tabs med on_change="rerun").
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from thor import candidates as tc
from thor import cards
from thor import dossier as td
from thor import registry as reg
from ui.tokens import AMBER, CYAN, DIM, GREEN, GREY, regime_color

_STATE, _CACHE = "thor_data", "thor_cache"
PAGE = 20                          # rader per "Visa fler"
CACHE_TTL_S = 3600                 # identitet och ögonblicksbild per aktie
ALL = "Alla strategier"
FAMILY_ALL = {f"{lbl} (alla)": key for key, lbl in reg.FAMILIES}
FILTERS = ("bara KÖP", "bara Norden", "bara nya")
TABS = ("Regim", "Teknik", "Cykel", "Råvara", "Arken", "Backtest", "Regler")
STORES = ("producers", "scoring", "tiggre", "insider", "holdings", "confidence", "scorecard", "copilot_journal")
STORE_DEFAULT = {"confidence": {"companies": {}, "commodity_overrides": {}, "strategies": {}},
                 "scorecard": {"cards": {}}, "copilot_journal": {"entries": []}}

# Bryggorna till de gamla sidorna tills fas 4–6: regimlayout → (djuplänk, sidans namn, widget att förifylla).
# Wolf Regime och Viking Regime sätter egna startvärden i sina fält — där navigerar vi bara.
BRIDGE = {"viking_nine": ("regime/marknad/arc-regime/viking-nine-regime", "Viking Nine Regime", "vnr_ticker"),
          "viking": ("regime/marknad/arc-regime/viking-regime", "Viking Regime", None),
          "wolf": ("regime/marknad/arc-regime/wolf-regime", "Wolf Regime", None),
          "momentum": ("regime/marknad/swing-regime", "Swing Regime", "swreg_ticker"),
          "berserk": ("regime/ravaror/berserk-regime", "BERSERK Regime", None),
          "ember": ("regime/ravaror/ember-regime", "EMBER Regime", None),
          "alpha_quality": ("regime/marknad/alpha-regime/quality-contrarian", "Alpha Regime", "ar_ticker"),
          "alpha_contrarian": ("regime/marknad/alpha-regime/quality-contrarian", "Alpha Regime", "ar_ticker"),
          "long_screener": ("regime/marknad/alpha-regime/long-trend", "Long Trend", "long_regime_ticker_input"),
          "insider": ("review/insider", "Insiderbevakaren", None), "sheet": ("review", "Granskningsarken", None)}
TEKNIK = ("intel/teknisk-analys", "Teknisk analys", "ta_ticker")
COPILOT = ("copilot", "Copiloten", "copilot_ticker")
BACKTEST = ("portfolio/backtest", "Backtest", None)
SHEET_LINK = {"rule": "review/rick-rule", "royalty": "review/royalty-c", "sprott": "review/poangmodell",
              "durrett": "review/poangmodell", "tiggre": "review/tiggre", "insider": "review/insider"}


# ── Data ────────────────────────────────────────────────────────────────────
def load(loader=None) -> dict:
    """Alla källor ur Gisten plus swingregimen — None för det som inte går att läsa, aldrig ett fel."""
    if loader is None:
        from gist_storage import load_blob as loader
    out = tc.load_blobs(loader)
    try:
        out["regime"] = loader("wolf_regime.json", None)
    except Exception:
        out["regime"] = None
    out["loaded"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return out


def _ohlcv(ticker: str):
    """Ett års dagsdata (market_prices, cachad sex timmar). None när det inte går."""
    try:
        from market_prices import ohlcv
        df = ohlcv(ticker, "1y")
        return df if df is not None and len(df) else None
    except Exception:
        return None


def _phase(ticker: str) -> tuple:
    """Market Cycle Engine (cachad en timme): ({phase, confidence}, fel)."""
    import cycle as cy
    return cy.market_phase(ticker)


def _api():
    """Börsdata-klienten när nyckeln finns, annars None (identiteten klarar sig utan)."""
    try:
        from borsdata_api import get_api
        api = get_api()
        return api if getattr(api, "is_configured", False) else None
    except Exception:
        return None


def _table():
    """markets-tabellen om den redan är laddad eller finns i filen — aldrig ett nytt API-anrop."""
    try:
        import markets
        return markets.current()
    except Exception:
        return None


def _stores() -> dict:
    """Arken ur sessionen (samma objekt som granskningsflikarna skriver till) och journalen — bara läsning."""
    import review_link
    import storage
    out: dict = {}
    for name in STORES:
        try:
            out[name] = storage.session_load(name, review_link.STORE_DEFAULTS.get(name) or STORE_DEFAULT.get(name) or {})
        except Exception:
            out[name] = None
    try:
        from trade_journal import load_journal
        out["journal"] = load_journal() or []
    except Exception:
        out["journal"] = []
    return out


def lage_items(data: dict, rows: list) -> list:
    """Lägets fyra kort: swingregimen, Norden och globalt på Valhall-kartan, kandidaterna. Ren funktion."""
    regime = data.get("regime") or {}
    segs = ((data.get("board") or {}).get("board") or {}).get("segments") or []
    n_green = sum(1 for r in segs if (r.get("nordic") or {}).get("status") == "GRÖN")
    g_green = sum(1 for r in segs if (r.get("global") or {}).get("status") == "GRÖN")
    s = tc.summary(rows)
    asof = max((v.get("asof") or "" for v in s["by_strategy"].values()), default="")
    sw = str(regime.get("regime") or "OKÄND").upper()

    def col(n):
        return GREEN if segs and n > len(segs) / 2 else AMBER if n else GREY

    return [("Swing-regim", sw, regime_color(sw), str(regime.get("generated") or "")[:10]),
            ("Norden", f"{n_green}/{len(segs)} GRÖN" if segs else "—", col(n_green), "Valhall-kartan"),
            ("Globalt", f"{g_green}/{len(segs)} GRÖN" if segs else "—", col(g_green), "SPDR mot S&P 500"),
            ("Kandidater", f"{s['n']} · {s['buy']} KÖP", CYAN, f"körd {asof}" if asof else "ingen körning")]


def pick_to_filter(pick: str) -> tuple:
    """Rullgardinens val → (familj, strategi)."""
    if pick in FAMILY_ALL:
        return FAMILY_ALL[pick], None
    for s in reg.STRATEGIES.values():
        if s.label == pick:
            return None, s.key
    return None, None


def shown_rows(rows: list, pick: str, filters) -> list:
    family, strategy = pick_to_filter(pick)
    flt = set(filters or [])
    out = tc.filter_rows(rows, family=family, strategy=strategy, only_buy="bara KÖP" in flt,
                         nordic_only="bara Norden" in flt)
    return [c for c in out if c.new] if "bara nya" in flt else out


# ── Tillstånd (callbacks körs före omritningen — då får widgetnycklarna sättas) ──
def _goto(ticker: str, strategy: Optional[str] = None) -> None:
    import streamlit as st
    st.session_state["thor_ticker"] = ticker
    if strategy:
        st.session_state["thor_strategy"] = reg.label(strategy)
        st.session_state["_thor_strategy_set"] = True


def _close() -> None:
    import streamlit as st
    st.session_state["thor_ticker"] = ""


def _more() -> None:
    import streamlit as st
    st.session_state["thor_show"] = int(st.session_state.get("thor_show") or PAGE) + PAGE


def _jump(path: str, key: Optional[str], ticker: str) -> None:
    """Bryggan: förifyll den gamla sidans fält och be panelen öppna fliken (Home-kortens nav_goto)."""
    import streamlit as st
    if key:
        st.session_state[key] = ticker
    st.session_state["nav_goto"] = path


def _apply_links(st) -> None:
    """?t=BOL.ST i adressfältet öppnar aktien — en gång per värde."""
    t = str(st.query_params.get("t") or "").strip()
    if t and st.session_state.get("_thor_t") != t:
        st.session_state["_thor_t"] = t
        st.session_state["thor_ticker"] = t
    if "thor_show" not in st.session_state:
        st.session_state["thor_show"] = PAGE


# ── Sidan ───────────────────────────────────────────────────────────────────
def render_thor_page() -> None:
    import streamlit as st
    from ui.components import note, page_header
    page_header("🔨 Thor", "Alla strategiers kandidater på ett ställe — tryck på en aktie, så kommer analysen "
                "innan köp. Inga nya regler: varje rad bär strategins eget verdikt.")
    _apply_links(st)
    c1, c2 = st.columns([1, 3])
    if c1.button("📡 Uppdatera", key="thor_refresh") or _STATE not in st.session_state:
        with st.spinner("Läser körningarna ur Gisten …"):
            data = load()
            data["rows"] = tc.build(data)
            st.session_state[_STATE] = data
            st.session_state.pop(_CACHE, None)
    data = st.session_state[_STATE]
    rows = data.get("rows") or []
    c2.markdown(f"<div style='color:{DIM};font-size:0.78rem;padding-top:8px;'>hämtad {data.get('loaded')} · "
                f"{len(rows)} rader</div>", unsafe_allow_html=True)
    st.markdown(cards.kpi_row(lage_items(data, rows)), unsafe_allow_html=True)
    ticker = tc.yf(st.text_input("Ticker", key="thor_ticker", label_visibility="collapsed",
                                 placeholder="Skriv en ticker — BOL.ST, ERIC B, NEM — eller tryck Öppna på en rad"))
    if ticker:
        _analysis(st, data, rows, ticker)
        with st.expander(f"◀ Kandidater ({len(rows)})", expanded=False):
            _list(st, rows)
    else:
        _list(st, rows)
    note("Thor-rank 0–100 är bara en sorteringsskala mellan strategier — strategins egen siffra står alltid "
         "bredvid. KÖP/BEVAKA/HÅLL/SPÄRRAD är strategins eget verdikt, översatt rakt av. Listan läses ur de "
         "schemalagda körningarna; kör dem i GitHub Actions för färska rader. Riktiga order läggs alltid manuellt.")


def _list(st, rows: list) -> None:
    pick = st.selectbox("Screening", [ALL] + list(FAMILY_ALL) + [s.label for s in reg.STRATEGIES.values()],
                        key="thor_pick", label_visibility="collapsed")
    if hasattr(st, "pills"):
        flt = st.pills("Filter", list(FILTERS), selection_mode="multi", key="thor_filters", label_visibility="collapsed")
    else:                                                       # pragma: no cover - äldre Streamlit
        flt = st.multiselect("Filter", list(FILTERS), key="thor_filters", label_visibility="collapsed")
    shown = shown_rows(rows, pick, flt)
    s = tc.summary(shown)
    st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin:6px 0 2px;'>"
                f"{s['n']} KANDIDATER · {s['buy']} KÖP · {pick.upper()}</div>", unsafe_allow_html=True)
    if not shown:
        from ui.components import empty_state
        empty_state("Inga rader med de filtren. Kör skanningarna i GitHub Actions eller lätta på filtren.")
        return
    n = int(st.session_state.get("thor_show") or PAGE)
    for c in shown[:n]:
        a, b = st.columns([5, 1])
        a.markdown(cards.candidate_card(c), unsafe_allow_html=True)
        b.button("Öppna →", key=f"thor_open_{c.strategy}_{c.ticker}", on_click=_goto, args=(c.ticker, c.strategy),
                 width="stretch")
    if len(shown) > n:
        st.button(f"Visa fler ({len(shown) - n} till)", key="thor_more", on_click=_more, width="stretch")


def _dossier(st, ticker: str, stores: dict) -> dict:
    """Identiteten och ögonblicksbilden, cachade en timme per aktie (arken läses varje gång — de ändras)."""
    cache = st.session_state.setdefault(_CACHE, {})
    hit = cache.get(ticker)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    with st.spinner(f"Hämtar {ticker} …"):
        ident = td.identify(ticker, _api(), stores, _table())
        snap, err = td.snapshot(_ohlcv(ident.yf), ident.yf)
    out = {"ident": ident, "snapshot": snap, "snapshot_error": err}
    cache[ticker] = (time.time(), out)
    return out


def _analysis(st, data: dict, rows: list, ticker: str) -> None:
    from ui.components import note
    st.button("◀ Tillbaka till listan", key="thor_back", on_click=_close)
    stores = _stores()
    d = _dossier(st, ticker, stores)
    ident = d["ident"]
    cands = td.candidates_for(rows, ident.yf)
    if not st.session_state.get("_thor_strategy_set") and st.session_state.get("_thor_last") != ident.yf:
        st.session_state["thor_strategy"] = reg.label(td.default_strategy(ident, cands))
    st.session_state["_thor_last"] = ident.yf
    st.session_state.pop("_thor_strategy_set", None)
    sheets = td.sheets_for(ident.yf, stores)
    st.markdown(cards.header_card(ident.as_dict(), cands, held=bool(sheets["positions"])), unsafe_allow_html=True)
    if d["snapshot"]:
        st.markdown(cards.snapshot_row(d["snapshot"]), unsafe_allow_html=True)
    else:
        note(f"Ingen kursdata för {ident.yf}: {d.get('snapshot_error') or 'okänt fel'}")
    labels = [s.label for s in reg.STRATEGIES.values()]
    picked = st.selectbox("Bedöm som", labels, key="thor_strategy")
    strategy = next((s.key for s in reg.STRATEGIES.values() if s.label == picked), "viking_nine")
    c = next((x for x in cands if x.strategy == strategy), None)
    st.markdown(cards.verdict_for(c, strategy), unsafe_allow_html=True)
    labels = [x for x in TABS if x != "Råvara" or ident.is_commodity]
    tabs = st.tabs(labels, on_change="rerun", key="thor_tabs")
    render = {"Regim": _tab_regim, "Teknik": _tab_teknik, "Cykel": _tab_cykel, "Råvara": _tab_ravara,
              "Arken": _tab_arken, "Backtest": _tab_backtest, "Regler": _tab_regler}
    for lbl, tab in zip(labels, tabs):
        if getattr(tab, "open", None) is False:
            continue
        with tab:
            render[lbl](st, data, ident, strategy, sheets, cands)


def _bridge(st, label: str, target: tuple, ticker: str, key: str) -> None:
    path, name, widget = target
    st.button(f"{label} {name} →", key=key, on_click=_jump, args=(path, widget, ticker))


# ── Sidoflikarna ────────────────────────────────────────────────────────────
def _tab_regim(st, data, ident, strategy, sheets, cands) -> None:
    from ui.components import note
    s = reg.get(strategy)
    note(f"Regimen per strategi — Viking Nine-, Wolf- och Alpha-layouterna — kommer i fas 4. Tills dess: "
         f"den gamla sidan med {ident.yf} förifylld.")
    target = BRIDGE.get((s.regime if s else "") or "", BRIDGE["sheet"])
    _bridge(st, "Öppna i", target, ident.yf, "thor_bridge_regim")
    _bridge(st, "Öppna i", COPILOT, ident.yf, "thor_bridge_copilot")


def _tab_teknik(st, data, ident, strategy, sheets, cands) -> None:
    from ui.components import note
    note("Teknisk analys för aktien — pris, EMA, order blocks, mönster, risk — flyttar in här i fas 4.")
    _bridge(st, "Öppna i", TEKNIK, ident.yf, "thor_bridge_teknik")


def _tab_cykel(st, data, ident, strategy, sheets, cands) -> None:
    from ui.components import note
    cyc = td.cycle(ident, strategy, data.get("board"), _phase)
    st.markdown(cards.gate_card(cyc["gate"], cyc["playbook"]), unsafe_allow_html=True)
    own = cyc.get("phase") or {}
    st.markdown(cards.phase_strip(own.get("phase"), cyc["phases"]["buy"], cyc["phases"]["hold"], cyc["phases"]["sell"]),
                unsafe_allow_html=True)
    chips = []
    bench = cyc.get("benchmark") or {}
    if (bench.get("phase") or {}).get("phase"):
        chips.append(cards.chip(f"{bench['index']}: {bench['phase']['phase'].replace('_', ' ')} "
                                f"{bench['phase'].get('confidence', 0):.0f} %", CYAN))
    seg = cyc.get("segment") or {}
    if seg.get("status"):
        parts = (f"Valhall {seg.get('view')}", seg.get("status"), seg.get("phase"), seg.get("cykel"))
        chips.append(cards.chip(" · ".join(x for x in parts if x), regime_color(seg.get("status"))))
    if chips:
        st.markdown("<div style='display:flex;flex-wrap:wrap;gap:4px;margin:4px 0;'>" + "".join(chips) + "</div>",
                    unsafe_allow_html=True)
    if own.get("phase"):
        try:
            from tabs.market_cycle import _build_gauge
            st.plotly_chart(_build_gauge(own["phase"], float(own.get("confidence") or 0)), use_container_width=True,
                            key="thor_cycle_gauge")
        except Exception as exc:                                 # pragma: no cover - ritningen är inte regeln
            note(f"Mätaren kunde inte ritas: {exc}")
    elif cyc.get("error"):
        note(f"Aktiens fas kunde inte läsas: {cyc['error']}")
    note("Aktiens egen psykologifas (Market Cycle Engine, 14 faser) och hemmamarknadens. Grinden läser playbookens "
         "köp-, håll- och säljfaser ordagrant — grön = köpfas, gul = håll eller fasen nämns inte, röd = säljfas.")


def _tab_ravara(st, data, ident, strategy, sheets, cands) -> None:
    from ui.components import note
    st.markdown(cards.chip(f"{ident.theme_label or ident.theme} · källa: {ident.theme_source}", AMBER),
                unsafe_allow_html=True)
    note("Råvarucykeln (BERSERK, Valhall, Blindspot, rotationsbetyget), hävstången mot råvaran, debasement och "
         "kvoterna för just det här temat kommer i fas 5.")
    _bridge(st, "Öppna i", BRIDGE["berserk"], ident.yf, "thor_bridge_berserk")


def _tab_arken(st, data, ident, strategy, sheets, cands) -> None:
    from ui.components import note
    if not sheets["found"]:
        note(f"{ident.yf} finns inte i något ark än. Ordningen är screener → granskning → köp: lägg in bolaget i "
             f"rätt ark först.")
    for key, rev in sheets["reviews"].items():
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
    conf = sheets.get("confidence")
    if conf:
        st.markdown(cards.chip(f"Durrett/Wolf Asymmetry: {conf.get('name') or ident.yf} · {conf.get('commodity') or '—'} · "
                               f"{conf.get('stage') or ''}".strip(" ·"), AMBER), unsafe_allow_html=True)
    if sheets.get("scorecard"):
        try:
            from scorecard import gate_state
            for k, card in sheets["scorecard"].items():
                g = gate_state(card, {"ticker": ident.yf})
                label = f"Scorecard: {'READY' if g.get('ready') else 'NO TRADE'} · {card.get('strategy') or ''}"
                st.markdown(cards.chip(label, GREEN if g.get("ready") else AMBER), unsafe_allow_html=True)
        except Exception:
            pass
    if sheets["positions"]:
        st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:8px;'>POSITIONER</div>",
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Hink": p.get("_bucket"), "Strategi": p.get("strategy"), "Antal": p.get("shares"),
                                    "Entry": p.get("entry_price"), "Datum": p.get("entry_date"), "Stopp": p.get("stop"),
                                    "Mål": p.get("target")} for p in sheets["positions"]]),
                     hide_index=True, use_container_width=True)
    if sheets["journal"]:
        st.markdown(f"<div style='color:{CYAN};font-size:0.72rem;letter-spacing:0.1em;margin-top:8px;'>JOURNALEN</div>",
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Strategi": t.get("strategy"), "In": t.get("entry_date"), "Ut": t.get("exit_date"),
                                    "Resultat %": t.get("pnl_pct"), "R": t.get("r_multiple"), "Skäl": t.get("exit_reason")}
                                   for t in sheets["journal"]]), hide_index=True, use_container_width=True)
    cols = st.columns(3)
    for i, (key, path) in enumerate(SHEET_LINK.items()):
        if key in sheets["reviews"] or i < 3:
            cols[i % 3].button(f"{reg.label(key) if reg.get(key) else key.title()} →", key=f"thor_sheet_{key}",
                               on_click=_jump, args=(path, None, ident.yf), width="stretch")


def _tab_backtest(st, data, ident, strategy, sheets, cands) -> None:
    from ui.components import note
    note("Strategin på aktien med en knapp — nyckeltal, affärer och kursgraf med entry/exit — kommer i fas 6 "
         "(Viking Nine, BERSERK, Momentum och Wolf). Tills dess finns PORTFOLIO → Backtest.")
    _bridge(st, "Öppna", BACKTEST, ident.yf, "thor_bridge_backtest")


def _tab_regler(st, data, ident, strategy, sheets, cands) -> None:
    from ui.components import note
    s = reg.get(strategy)
    if s and s.playbook:
        try:
            from ovtlyr.ui.rules_page import render_inline_rules
            render_inline_rules(s.playbook)
            return
        except Exception as exc:                                 # pragma: no cover
            note(f"Reglerna kunde inte ritas: {exc}")
    note(f"{reg.label(strategy)} har ingen playbook i strategy_rules än — den skrivs i fas 7.")
