"""
thor/ui.py — 🔨 THOR: läget, tickersök, rullgardinen, listan och analysblocket på EN sida.

Flödet är screening → tryck → analys → köp, utan att byta flik: tryck "Öppna →" på en rad (eller skriv en
ticker) så öppnas analysblocket överst med aktien och strategin förifyllda, och listan fälls ihop under
det. Sidan räknar inga signaler: listan läses ur körningarna (thor.candidates) och analysen sätts ihop av
motorerna som redan finns (thor.dossier). Regim per strategi, Teknik och Regler ritas av thor.regim, Cykel av
thor.cykel, Råvara av thor.ravara, Asymmetri av thor.asymmetri (Wolf Asymmetrys Snabbkoll med riktpris,
kurshävstång och tillväxtscenarier), Arken av thor.arken och Backtest av thor.backtest — samma motorer som de
gamla sidorna, som har bryggknappar kvar tills städningen.

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
TABS = ("Regim", "Teknik", "Cykel", "Råvara", "Asymmetri", "Arken", "Backtest", "Regler")
STORES = ("producers", "scoring", "tiggre", "insider", "holdings", "confidence", "scorecard", "copilot_journal",
          "rotation")
STORE_DEFAULT = {"confidence": {"companies": {}, "commodity_overrides": {}, "strategies": {}},
                 "scorecard": {"cards": {}}, "copilot_journal": {"entries": []},
                 "rotation": {"month": "", "grades": {}, "history": []}}

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
COPILOT = ("copilot", "Copiloten", "copilot_ticker")
ASYMMETRY = ("review/wolf-asymmetry", "Wolf Asymmetry", "asym_quick_ticker")
BACKTEST = ("portfolio/backtest", "Backtest", None)
VALHALL_BT = ("valhall/backtester", "Valhalls backtester", None)
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


def _prices(ticker: str, period: str = "1y"):
    """Dagsdata för en period (market_prices, cachad sex timmar). None när det inte går."""
    try:
        from market_prices import ohlcv
        df = ohlcv(ticker, period)
        return df if df is not None and len(df) else None
    except Exception:
        return None


def _ohlcv(ticker: str):
    """Ett års dagsdata — ögonblicksbilden i sidhuvudet."""
    return _prices(ticker, "1y")


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
            errors: dict = {}
            data["rows"] = tc.build(data, errors=errors)
            data["errors"] = errors
            st.session_state[_STATE] = data
            st.session_state.pop(_CACHE, None)
    data = st.session_state[_STATE]
    rows = data.get("rows") or []
    c2.markdown(f"<div style='color:{DIM};font-size:0.78rem;padding-top:8px;'>hämtad {data.get('loaded')} · "
                f"{len(rows)} rader</div>", unsafe_allow_html=True)
    st.markdown(cards.kpi_row(lage_items(data, rows)), unsafe_allow_html=True)
    if data.get("errors"):
        note("Kunde inte läsa " + ", ".join(f"{k} ({v})" for k, v in data["errors"].items())
             + " — de raderna saknas i listan tills filen är rättad eller körningen körd om.")
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
              "Asymmetri": _tab_asymmetri, "Arken": _tab_arken, "Backtest": _tab_backtest, "Regler": _tab_regler}
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
    """Strategins eget regimbeslut för aktien (thor.regim) — och knappar till den gamla sidan och Copiloten."""
    from thor import regim
    regim.render(st, ident, strategy, data, _prices, cands)
    s = reg.get(strategy)
    target = BRIDGE.get((s.regime if s else "") or "", BRIDGE["sheet"])
    b1, b2 = st.columns(2)
    with b1:
        _bridge(st, "Öppna i", target, ident.yf, "thor_bridge_regim")
    with b2:
        _bridge(st, "Öppna i", COPILOT, ident.yf, "thor_bridge_copilot")


def _tab_teknik(st, data, ident, strategy, sheets, cands) -> None:
    """Teknisk analys för aktien — samma vy som 🕯️ Teknisk analys."""
    from thor import regim
    regim.render_technical(st, ident)


def _tab_cykel(st, data, ident, strategy, sheets, cands) -> None:
    """Aktiens psykologifas med regelutfall och tidslinje, hemmamarknaden, Valhall-läget och grinden (thor.cykel)."""
    from thor import cykel
    cykel.render(st, data, ident, strategy, _phase)


def _tab_ravara(st, data, ident, strategy, sheets, cands) -> None:
    """Råvaran bakom aktien: cykeln, hävstången, debasement och kvoterna (thor.ravara) — och BERSERK-sidan."""
    from thor import ravara
    ravara.render(st, ident, data, sheets.get("rotation"), _prices)
    _bridge(st, "Öppna i", BRIDGE["berserk"], ident.yf, "thor_bridge_berserk")


def _tab_asymmetri(st, data, ident, strategy, sheets, cands) -> None:
    """Wolf Asymmetrys Snabbkoll för aktien, för alla strategier, med riktpris, kurshävstång och tillväxtscenarier
    (thor.asymmetri) — och knappen till Wolf Asymmetry-sidan."""
    from thor import asymmetri
    asymmetri.render(st, ident, strategy, _api())
    _bridge(st, "Öppna i", ASYMMETRY, ident.yf, "thor_bridge_asym")


def _tab_arken(st, data, ident, strategy, sheets, cands) -> None:
    """Vad arken säger om bolaget — granskningarna, motorerna, scorecardet, insiderklustret, registret (thor.arken)."""
    from thor import arken
    arken.render(st, data, ident, sheets, cands, SHEET_LINK, _jump)


def _tab_backtest(st, data, ident, strategy, sheets, cands) -> None:
    """Strategin på aktien med en knapp (thor.backtest); utan per-aktie-motor: Valhalls backtester och gamla sidan."""
    from thor import backtest
    if not backtest.render(st, ident, strategy, _prices):
        _bridge(st, "Öppna", VALHALL_BT, ident.yf, "thor_bridge_valhall_bt")
    _bridge(st, "Öppna", BACKTEST, ident.yf, "thor_bridge_backtest")


def _tab_regler(st, data, ident, strategy, sheets, cands) -> None:
    """Hela playbooken för strategin — regler och walkthrough, sist på sidan."""
    from thor import regim
    regim.render_rules(st, strategy)
