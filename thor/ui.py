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
funktioner (thor/cards.py), bara den öppna sidofliken ritas (st.tabs med on_change="rerun"). Med ett segment,
tema eller komplex valt står segmentets största bolag utan signal under listan (thor.segment_top) — bara där.
Under verdiktet lägger 👀 aktien med strategin i Övervakningen (watchlist.py, PORTFÖLJ → 👀 Övervakning).
"""

from __future__ import annotations

import time
from html import escape
from typing import Optional

import pandas as pd

from thor import candidates as tc
from thor import cards
from thor import dossier as td
from thor import live
from thor import registry as reg
from ui.tokens import AMBER, CYAN, GREEN, GREY, MUTED, regime_color
from universe import registry as ur

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
# Universum-registret i Thor: marknadsfiltret (registrets etiketter) och medlemmarna per marknad i sessionen.
MARKET_LABELS = {m.label: m.key for m in ur.MARKETS}
_UNIVERSE = "thor_universe"

# Bryggorna till sidorna i Odins visdom och Arken: regimlayout → (djuplänk, sidans namn, widget att förifylla).
# Ett värde satt i session_state före widgeten vinner över value= — så fälten förifylls även där sidan har ett
# eget startvärde (Wolf Regime XOM, Viking Regime, EMBER Regime). BERSERK Regime har inget tickerfält.
BRIDGE = {"viking_nine": ("more/motorsidor/viking-nine-regime", "Viking Nine Regime", "vnr_ticker"),
          "viking": ("more/motorsidor/viking-regime", "Viking Regime", "ovtlyr_ticker"),
          "wolf": ("more/motorsidor/wolf-regime", "Wolf Regime", "reg_ticker"),
          "momentum": ("odin/laget", "Läget (Swing Regime)", "swreg_ticker"),
          "berserk": ("odin/ravaror/berserk-regime", "BERSERK Regime", None),
          "ember": ("odin/ravaror/ember", "🔥 EMBER", "ember_regime_ticker_search"),
          "alpha_quality": ("more/motorsidor/quality-contrarian", "Alpha Regime", "ar_ticker"),
          "alpha_contrarian": ("more/motorsidor/quality-contrarian", "Alpha Regime", "ar_ticker"),
          "long_screener": ("more/motorsidor/long-trend", "Long Trend", "long_regime_ticker_input"),
          "insider": ("review/insider", "Insiderbevakaren", None), "sheet": ("review", "Granskningsarken", None),
          "royalty": ("review/royalty-c", "Royalty C", None)}
COPILOT = ("more/copilot", "Copiloten", "copilot_ticker")
ASYMMETRY = ("review/wolf-asymmetry", "Wolf Asymmetry", "asym_quick_ticker")
BACKTEST = ("more/egen-backtest", "Backtest", None)
VALHALL_BT = ("portfolio/mot-malet", "Valhalls backtester", None)
# Arkknapparna. Durrett-granskningen som Thor läser bor i Poängmodellen (review_link.SHEET) — dit går "Durrett →";
# 🧭 Durrett (10-stegsarket) är en egen knapp i Arken-fliken (thor.arken.DURRETT_10).
SHEET_LINK = {"rule": "review/rick-rule", "royalty": "review/royalty-c", "sprott": "review/poangmodell",
              "durrett": "review/poangmodell", "tiggre": "review/tiggre", "insider": "review/insider"}
# Extra fält som bryggan sätter på målsidan (granskning 2, D10/D11): Alpha-sidans läge och Copilotens strategi.
BRIDGE_EXTRA = {"alpha_quality": {"ar_mode": "quality"}, "alpha_contrarian": {"ar_mode": "contrarian"}}


# ── Data ────────────────────────────────────────────────────────────────────
def load(loader=None) -> dict:
    """Alla källor ur Gisten plus swingregimen — None för det som inte går att läsa, aldrig ett fel."""
    if loader is None:
        from gist_storage import load_blob as loader
    out = tc.load_blobs(loader)
    try:                                                    # filer vars läsning fallerade — FEL, inte SAKNAS (F7)
        import gist_storage as gs
        out["unreadable"] = sorted(k for k, name in tc.BLOBS.items()
                                   if name in gs.UNREADABLE_NAMES and out.get(k) is None)
    except Exception:
        out["unreadable"] = []
    try:
        out["regime"] = loader("wolf_regime.json", None)
    except Exception:
        out["regime"] = None
    out["universe"] = ur.load_index(loader)                 # registrets index: statistik och Nordens tickerkarta
    try:                                                    # registrets sektorer → segment, och namn, för rader utan eget
        reg_ = ur.load(None, loader)
        out["segment_map"] = tc.segment_map_from_register(reg_)
        out["names"] = {str(t).upper(): getattr(m, "name", "") for t, m in (getattr(reg_, "members", None) or {}).items()
                        if getattr(m, "name", "")}
        out["sheet_map"] = reg_.sheet_map() if reg_ is not None else {}   # arkens 'BOL' → 'BOL.ST' (T4)
    except Exception:
        out["segment_map"], out["names"], out["sheet_map"] = {}, {}, {}
    try:                                                    # segmentens största med kassaflöde (Universum-registret)
        from universe import segment_top as ust
        out["segment_top"] = loader(ust.BLOB, None)
    except Exception:
        out["segment_top"] = None
    out["loaded"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return out


def rebuild(data: dict) -> dict:
    """Listans rader ur källorna i data (samma bygge efter 📡 Uppdatera och efter ⚡ Skanna nu)."""
    errors: dict = {}
    bd_map = (data.get("universe") or {}).get("bd_map") or None   # 'EQNR' → 'EQNR.OL', inte '.ST'
    data["rows"] = tc.build(data, errors=errors, momentum_map=bd_map, segment_map=data.get("segment_map"),
                            names=data.get("names"), sheet_map=data.get("sheet_map"))
    data["errors"] = errors
    return data


def freshness_rows(rows: list, data: dict) -> list:
    """Körningarnas läge per strategi som tabellrader — och hur strategin körs (live eller bara i schemat)."""
    out = []
    for key, label, n, asof, state in live.freshness(rows, data):
        how = f"schemat — {live.HEAVY[key]}" if key in live.HEAVY else "⚡ Skanna nu" if live.can_run(key) else "schemat"
        acct = (reg.get(key).paper if reg.get(key) else "") or ""
        out.append({"Strategi": label, "Läge": state, "Rader": n, "Körd": asof or "—", "Körs": how,
                    "Papper": "PORTFÖLJ → 📊 Papperskonton" if acct else "—"})
    return out


def universe_member(st, ticker: str, loader=None):
    """Registrets medlem för tickern — marknadens fil läses ur Gisten första gången och ligger kvar i sessionen."""
    key = ur.market_of(ticker)
    if not key:
        return None
    store = st.session_state.setdefault(_UNIVERSE, {})
    if key not in store:
        try:
            u = ur.load([key], loader)
        except Exception:
            u = None
        if not u:                                         # misslyckad läsning sparas inte — nästa tryck försöker igen
            return None
        store[key] = u.members
    return store[key].get(str(ticker or "").strip().upper())


def fill_from_universe(ident, member) -> bool:
    """Namn, valuta och råvarutema ur registret när Börsdata Norden inte kände tickern — med temats segment och
    komplex när de saknas, som identify gör (annars hittar cykeln inget, granskning 3, T8). True om något fylldes."""
    if member is None:
        return False
    filled = False
    if not ident.name and member.name:
        ident.name, filled = member.name, True
    if not ident.currency and member.currency:
        ident.currency, filled = member.currency, True
    if not ident.theme and member.theme:
        from berserk import themes as bth
        from valhall import segments as sgm
        try:
            ident.theme_label = bth.label(member.theme)
        except Exception:
            ident.theme_label = member.theme
        ident.theme, ident.theme_source, filled = member.theme, "universum-registret", True
        if not ident.segment:
            ident.segment = sgm.segment_of_theme(member.theme)
            ident.segment_label = sgm.get(ident.segment).label if ident.segment and sgm.get(ident.segment) else ""
        if not ident.complex:
            try:
                ident.complex = bth.complex_of(member.theme)
            except Exception:
                ident.complex = None
    return filled


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
    regime = data.get("regime") if isinstance(data.get("regime"), dict) else {}
    segs = tc.segment_context(data.get("board")).get("segments") or {}      # tål en trasig karta (tom i stället för krasch)
    n_green = sum(1 for r in segs.values() if r.get("status") == "GRÖN")
    g_green = sum(1 for r in segs.values() if r.get("global_status") == "GRÖN")
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


SEGMENT_ALL = "alla"                         # segmentrullgardinens "Alla segment"


THEME_PREFIX, COMPLEX_PREFIX = "tema:", "komplex:"   # rullgardinens nycklar för kartans teman och komplex


def _with_status(text: str, status) -> str:
    return f"{text} · {status}" if status else text


def segment_choices(board: Optional[dict] = None) -> list:
    """[(nyckel, text)] för segmentrullgardinen: "Alla segment" först, sedan Valhalls segment i kartans ordning med
    sina teman under ("Energi › Uran") och sist råvarukomplexen — med kartans status när den finns ("Hälsovård ·
    GRÖN"), så du kan söka köp bara där det är grönt."""
    from berserk import themes as bth
    from ember.config import COMPLEX_LABEL
    from valhall import segments as sgm
    ctx = tc.segment_context(board)
    status, tstatus = ctx.get("segments") or {}, ctx.get("themes") or {}
    out = [(SEGMENT_ALL, "Alla segment")]
    for s in sgm.SEGMENTS:
        out.append((s.key, _with_status(s.label, (status.get(s.key) or {}).get("status"))))
        for th in s.themes:
            out.append((THEME_PREFIX + th, _with_status(f"{s.label} › {bth.label(th)}", (tstatus.get(th) or {}).get("status"))))
    inner = (board or {}).get("board") if isinstance(board, dict) else None
    cxs = inner.get("complexes") if isinstance(inner, dict) and isinstance(inner.get("complexes"), list) else []
    cstatus = {c.get("key"): (c.get("state") or {}).get("status") for c in cxs if isinstance(c, dict)}
    for key, lbl in COMPLEX_LABEL.items():
        out.append((COMPLEX_PREFIX + key, _with_status(f"Komplex: {lbl.capitalize()}", cstatus.get(key))))
    return out


def _segment_themes(segment: Optional[str]) -> Optional[set]:
    """Temana bakom ett tema- eller komplexval, annars None (vanligt segment)."""
    s = str(segment or "")
    if s.startswith(THEME_PREFIX):
        return {s[len(THEME_PREFIX):]}
    if s.startswith(COMPLEX_PREFIX):
        from ember.config import THEME_TO_COMPLEX
        return {th for th, cx in THEME_TO_COMPLEX.items() if cx == s[len(COMPLEX_PREFIX):]}
    return None


def shown_rows(rows: list, pick: str, filters, markets=None, segment: Optional[str] = None) -> list:
    """Listans filter: rullgardinen, chipsen, marknaderna (registrets etiketter; tomt = alla) och segmentet
    (Valhall-kartans nyckel; raden måste ha segmentet, eller ett tema som hör till det). Ett tema- eller komplexval
    visar rader med temat — eller bolag som står under temat i temalistan (ADM: vete, majs, soja, socker). Rader
    utan eget tema (Wolf, håvarna …) matchas på bolagets tema i BERSERK-, EMBER- och temalistorna (NXE: uran) —
    radens tema och kort ändras inte (granskning 3, T5)."""
    from thor.ticker import theme_for
    from universe import theme_members as tm
    family, strategy = pick_to_filter(pick)
    flt = set(filters or [])
    themes = _segment_themes(segment)
    seg = None if segment in (None, SEGMENT_ALL) or themes is not None else segment
    out = tc.filter_rows(rows, family=family, strategy=strategy, only_buy="bara KÖP" in flt,
                         nordic_only="bara Norden" in flt, segment=seg)
    if themes is not None:
        listed = {t for th in themes for t in tm.members(th)}
        out = [c for c in out if c.theme in themes or c.ticker.upper() in listed
               or (not c.theme and theme_for(c.ticker)[0] in themes)]
    keys = {MARKET_LABELS.get(m, m) for m in (markets or [])}
    if keys:
        out = [c for c in out if ur.market_of(c.ticker) in keys]
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


def _jump(path: str, key: Optional[str], ticker: str, extra: Optional[dict] = None) -> None:
    """Bryggan: förifyll den gamla sidans fält (och extra fält som läge/strategi) och be panelen öppna fliken."""
    import streamlit as st
    if key:
        st.session_state[key] = ticker
    for k, v in (extra or {}).items():
        st.session_state[k] = v
    st.session_state["nav_goto"] = path


def _apply_links(st) -> None:
    """?t=BOL.ST i adressfältet öppnar aktien och ?s=wolf väljer strategin (larmlänkarna) — en gång per värde."""
    t = str(st.query_params.get("t") or "").strip()
    s = str(st.query_params.get("s") or "").strip()
    if t and st.session_state.get("_thor_t") != (t, s):
        st.session_state["_thor_t"] = (t, s)
        st.session_state["thor_ticker"] = t
        if reg.get(s):
            st.session_state["thor_strategy"] = reg.label(s)
            st.session_state["_thor_strategy_set"] = True
    if "thor_show" not in st.session_state:
        st.session_state["thor_show"] = PAGE


# ── Sidan ───────────────────────────────────────────────────────────────────
SUBTITLE = ("Alla strategiers kandidater på ett ställe — tryck på en aktie, så kommer analysen innan köp. Inga nya "
            "regler: varje rad bär strategins eget verdikt.")
FOOTNOTE = ("Thor-rank 0–100 är bara en sorteringsskala mellan strategier — strategins egen siffra står alltid "
            "bredvid. KÖP/BEVAKA/HÅLL/SPÄRRAD är strategins eget verdikt, översatt rakt av. Listan läses ur de "
            "schemalagda körningarna; kör dem i GitHub Actions för färska rader. Riktiga order läggs alltid manuellt.")
PLACEHOLDER = "Skriv en ticker — BOL.ST, ERIC B, NEM — eller tryck på en rad"
_RELOAD = "_thor_reload"                    # 📡 Uppdatera tryckt: läs om innan sidan ritas (knappen kan stå under analysen)
_STALE = ("GAMMAL", "SAKNAS", "FEL")

# Sidans egna regler (LIDSKJALV §4.5–§4.6) i en dold stilbehållare, valda på sidans egna nycklar: sökraden (fältet
# 48 px med 16 px text — iOS zoomar under 16 px — och ⚙ Filter bredvid), pillsraden med 🌍 Marknad, raderna som
# tryckkort (två kolumner från 1280 px), navraden och "Bedöm som" i analysen, analysens två kolumner från 1024 px
# (staplade under) och 12 px-golvet för det som ritas inne i Thor (st.caption, flikarnas inline-storlekar).
_ROOT = '.st-key-thor_ticker [data-testid="stTextInputRootElement"]'
_SMALL = ("0.55rem", "0.6rem", "0.62rem", "0.63rem", "0.64rem", "0.65rem", "0.66rem", "0.67rem", "0.68rem", "0.69rem",
          "0.7rem", "0.72rem", "0.74rem", "0.7em", "9px", "10px", "11px")
_FLOOR = ",\n".join(f'.st-key-thor_page [style*="font-size:{v}"], .st-key-thor_page [style*="font-size: {v}"]'
                    for v in _SMALL)
_TIP = ('.st-key-thor_page .stButton > div > button[kind="secondary"]',
        '.st-key-thor_page .stButton [data-testid="stTooltipHoverTarget"] > button[kind="secondary"]')
_TIP_BTN = ", ".join(_TIP)
_TIP_BTN_IN = ", ".join(f'{t} [data-testid="stMarkdownContainer"], {t} p' for t in _TIP)
_CSS = f"""
.st-key-thor_page {{ gap: 8px; }}
.st-key-thor_page > [data-testid="stElementContainer"] [data-testid="stMarkdownContainer"],
.st-key-thor_rows [data-testid="stMarkdownContainer"], .st-key-thor_toprows [data-testid="stMarkdownContainer"],
.st-key-thor_left [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
.st-key-thor_page summary {{ min-height: 44px; box-sizing: border-box; }}
.st-key-thor_page [data-testid="stCaptionContainer"], .st-key-thor_page [data-testid="stCaptionContainer"] p {{
    font-size: 13px; }}
/* Huvudraden: "🔨 THOR" är kort — 📡, ⓘ och rundans pill ryms bredvid; hämtad-raden ligger an under den */
.st-key-head_thor > [data-testid="stElementContainer"]:has(.wolf-head) {{ min-width: 0; }}
:is([data-testid="stLayoutWrapper"]:has(> .st-key-head_thor), .st-key-head_thor) + [data-testid="stElementContainer"] {{
    margin-top: -4px; }}
.st-key-thor_about button, .st-key-thor_about_an button {{ min-height: 44px; min-width: 44px; padding: 0 12px; }}
:is(.st-key-thor_about, .st-key-thor_about_an) [data-testid="stPopoverButton"] > div > div + div {{ display: none; }}
/* Sökraden och navraden: fältet tar raden, ⚙ Filter (eller ◀) bredvid */
.st-key-thor_search, .st-key-thor_nav, .st-key-thor_quick {{ flex-wrap: nowrap; gap: 8px; align-items: center; }}
.st-key-thor_search > .st-key-thor_ticker, .st-key-thor_nav > .st-key-thor_ticker {{ flex: 1 1 auto; min-width: 0;
    width: auto; }}
{_ROOT} {{ height: 48px; border-radius: 10px; overflow: hidden; }}
{_ROOT} input {{ height: 100%; font-size: 16px !important; font-family: 'JetBrains Mono', monospace !important;
    text-overflow: ellipsis; }}
{_ROOT} input::placeholder {{ font-family: 'Space Grotesk', sans-serif; }}
.st-key-thor_filter_pop button {{ min-height: 48px; }}
/* Pillsraden rullar i sidled (skalets regel för thor_filters); 🌍 Marknad står kvar till höger */
.st-key-thor_quick > .st-key-thor_filters, .st-key-thor_quick > .st-key-thor_filters .stButtonGroup {{
    flex: 0 1 auto; min-width: 0; width: auto; }}
.st-key-thor_quick [data-testid="stButtonGroup"] button, .st-key-thor_market_pop button {{ min-height: 44px; }}
/* Popoverns innehåll ritas i en portal utanför sidan — utanför .st-key-thor_page och triggerns nyckel — så det väljs
   på widgetarnas egna nycklar, som följer med dit: marknadspillsen, Screening- och Segmentfälten med sina ⌄ och
   Segments hjälpikon som 44 px tryckytor (§8.7) */
.st-key-thor_markets [data-testid="stButtonGroup"] button {{ min-height: 44px; }}
.st-key-thor_pick .react-aria-ComboBox > div, .st-key-thor_segment .react-aria-ComboBox > div {{ min-height: 44px; }}
.st-key-thor_pick .react-aria-ComboBox button, .st-key-thor_segment .react-aria-ComboBox button {{ min-width: 44px;
    min-height: 44px; }}
.st-key-thor_segment [data-testid="stTooltipHoverTarget"] button {{ min-width: 44px; min-height: 44px;
    display: inline-flex; align-items: center; justify-content: center; }}
/* I popovern bryts marknaderna på två rader i stället för att svepas (Thor-avvikelse från §3.5.4; pillsraden
   thor_filters på sidan sveps som förut) — alla sex syns utan svep */
.st-key-thor_markets [role="toolbar"] {{ flex-wrap: wrap !important; overflow-x: visible !important;
    -webkit-mask-image: none !important; mask-image: none !important; row-gap: 8px; }}
.st-key-thor_scan {{ flex-wrap: wrap; gap: 8px 12px; align-items: center; }}
/* Raderna: 8 px mellan korten (kortets egen marginal bort); två kolumner från 1280 px */
.st-key-thor_rows, .st-key-thor_toprows {{ gap: 8px; }}
.st-key-thor_rows [class*="st-key-tapc_"] [data-testid="stMarkdownContainer"] > div,
.st-key-thor_toprows [class*="st-key-tapc_"] [data-testid="stMarkdownContainer"] > div {{ margin: 0 !important; }}
@media (min-width: 1280px) {{
    .st-key-thor_rows, .st-key-thor_toprows {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
        align-items: stretch; }}
    .st-key-thor_rows > *, .st-key-thor_toprows > * {{ min-width: 0; width: auto !important; }}
    /* korten i samma rad lika höga */
    :is(.st-key-thor_rows, .st-key-thor_toprows) [class*="st-key-tapc_"] {{ height: 100%; }}
    :is(.st-key-thor_rows, .st-key-thor_toprows) [class*="st-key-tapc_"] > [data-testid="stElementContainer"]:first-child {{
        flex: 1 1 auto; }}
    :is(.st-key-thor_rows, .st-key-thor_toprows) [class*="st-key-tapc_"] :is(.stMarkdown, .stMarkdown > div,
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] > div) {{ height: 100%;
        box-sizing: border-box; }}
}}
/* Analysen: ◀ tillbaka och fältet på en rad, "Bedöm som" med etiketten till vänster */
.st-key-thor_back button {{ min-height: 48px; padding: 0 8px; color: {CYAN}; font-size: 14px; }}
.st-key-thor_strategy .stSelectbox {{ display: flex; flex-direction: row; align-items: center; gap: 12px; }}
.st-key-thor_strategy [data-testid="stWidgetLabel"] {{ flex: 0 0 auto; margin: 0; min-height: 0; }}
.st-key-thor_strategy [data-testid="stWidgetLabel"] p {{ font-size: 13px; color: {MUTED}; white-space: nowrap; }}
.st-key-thor_strategy .stSelectbox > div {{ flex: 1 1 auto; min-width: 0; }}
.st-key-thor_left, .st-key-thor_right {{ gap: 12px; }}
:is(.thor-lage, .st-key-thor_right) .wolf-kstrip > div > .wolf-tile {{ height: 100%; }}
.st-key-thor_actions {{ gap: 8px; }}
.st-key-thor_left .wolf-sec {{ margin-top: 8px !important; }}
/* Knappar med tooltip (help=) ligger ett omslag längre ned än skalets '.stButton > button' — samma sekundärutseende
   (på datorn i tooltipens hovringsmål, på telefonen i ett eget omslag) */
{_TIP_BTN} {{ min-height: 44px;
    padding: 0 16px; border-radius: 10px; border: 1px solid {CYAN}; background: rgba(0,229,255,.08); color: {CYAN};
    font-family: 'Space Grotesk', sans-serif; font-size: 14px; font-weight: 600; letter-spacing: .06em;
    text-transform: uppercase; }}
{_TIP_BTN_IN} {{ font: inherit; letter-spacing: inherit; }}
/* Fält och stegknappar i sidan och flikarna: 48 px "Bedöm som", talfältens − och + som 44 px tryckytor */
.st-key-thor_strategy .react-aria-ComboBox > div {{ min-height: 48px; }}
.st-key-thor_page .react-aria-ComboBox > div {{ min-height: 44px; }}
.st-key-thor_page .react-aria-ComboBox button {{ min-width: 44px; min-height: 44px; }}
.st-key-thor_page [data-testid="stNumberInputContainer"] {{ min-height: 44px; }}
.st-key-thor_page [data-testid="stNumberInputContainer"] button {{ min-width: 44px; min-height: 44px; }}
/* Storleksrutorna på telefonen: tre à ~114 px — tätare luft och spärrning, så att den längsta etiketten ryms */
@media (max-width: 640px) {{
    .thor-storlek .wolf-tile {{ padding: 10px 8px 10px 10px !important; }}
    .thor-storlek .wolf-tile > div:first-child {{ letter-spacing: .04em !important; }}
    /* talet bryts aldrig mitt i ("110.0" / "0") — det krymper i stället på smala telefoner */
    .thor-storlek .wolf-tile > div:nth-child(2) {{ white-space: nowrap; overflow-wrap: normal !important;
        font-size: clamp(14px, 5.1vw, 20px) !important; }}
    /* Lägets remsa på telefonen: underraden på en rad (hela texten står i remsans aria-label) och kandidaterna
       "32 · 13 KÖP" på en rad — samma värden, lägre remsa */
    .thor-lage .wolf-tile {{ padding: 8px 12px !important; }}
    .thor-lage .wolf-tile > div:nth-child(3) {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        overflow-wrap: normal !important; }}
    .thor-lage .wolf-tile > div:nth-child(2) > div {{ display: inline; }}
    .thor-lage .wolf-tile > div:nth-child(2) > div + div::before {{ content: " · "; color: {MUTED}; }}
}}
@media (max-width: 359px) {{
    .thor-storlek .wolf-tile > div:first-child {{ letter-spacing: 0 !important; }}
}}
/* 12 px-golvet (§8.5) för det flikarna ritar med egna inline-storlekar under 12 px (bara inom Thor) */
{_FLOOR} {{ font-size: 12px !important; }}
/* Beslut till vänster, bevis till höger från 1024 px; under det staplade i den ordningen (st.columns står annars
   sida vid sida redan från 640 px) */
@media (max-width: 1023px) {{
    .st-key-thor_an [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-thor_left) {{
        flex-direction: column; flex-wrap: nowrap; gap: 12px; }}
    .st-key-thor_an [data-testid="stColumn"]:has(.st-key-thor_left),
    .st-key-thor_an [data-testid="stColumn"]:has(.st-key-thor_right) {{ width: 100% !important;
        min-width: 100% !important; flex: 1 1 auto !important; }}
}}
"""
# Etiketten i bärnsten när något är gammalt: skalets expanderfärg är !important (ui/css.py), så regeln måste vara
# skarpare och också !important
_CSS_STALE = (f'.st-key-thor_fresh [data-testid="stExpander"] summary p, '
              f'.st-key-thor_fresh [data-testid="stExpander"] summary [data-testid="stIconMaterial"] '
              f'{{ color: {AMBER} !important; }}')


def _refresh() -> None:
    """📡 Uppdatera (callback): körningarna läses om överst i nästa körning — före listan och analysen."""
    import streamlit as st
    st.session_state[_RELOAD] = True


def _reload(st) -> dict:
    """Thors data ur sessionen — läst ur Gisten första gången eller efter 📡 Uppdatera."""
    if st.session_state.pop(_RELOAD, False) or _STATE not in st.session_state:
        with st.spinner("Läser körningarna ur Gisten …"):
            st.session_state[_STATE] = rebuild(load())
            st.session_state.pop(_CACHE, None)
            for k in ("_wolf_screener_cache", "_wolf_regime_cache"):       # sidornas sessionscacher läses om också
                st.session_state.pop(k, None)
    return st.session_state[_STATE]


def _ticker_field(st) -> str:
    """Tickerfältet — samma nyckel och samma argument i listan och i analysens navrad, så värdet följer med."""
    return st.text_input("Ticker", key="thor_ticker", label_visibility="collapsed", placeholder=PLACEHOLDER)


def _refresh_button(st) -> None:
    st.button("📡 Uppdatera", key="thor_refresh", on_click=_refresh)


def _about(st, key: str) -> None:
    """ⓘ Om listan: sidans underrad och fotnot, ordagrant, i en popover (beslut 9)."""
    from ui.components import note, section
    with st.popover("ⓘ", help="Om listan", key=key):
        st.markdown(section("Om listan"), unsafe_allow_html=True)
        note(SUBTITLE)
        note(FOOTNOTE)


def render_thor_page() -> None:
    """🔨 Thor (LIDSKJALV §4.5–§4.6). Listläget: huvudraden (titel · 📡 · ⓘ · rundans pill) och hämtad-raden, lägets
    remsa, EN rad om körningar som är gamla, saknas eller fallerade, sökraden med ⚙ Filter, pillsraden med 🌍
    Marknad, raderna som tryckkort — och under listan Körningar per strategi, Universum-registret och rundans fot.
    Analysläget (en ticker vald): navraden (◀ och fältet) först, sedan analysen; listans ram står under den.
    Ritordningen är ny, logiken densamma: inga regler, nycklar eller texter ändrade."""
    import streamlit as st
    from ui.shell import page_style
    _apply_links(st)
    with st.container(key="thor_page"):
        data = _reload(st)
        rows = data.get("rows") or []
        fresh = freshness_rows(rows, data)
        old = [r["Strategi"] for r in fresh if r["Läge"] in _STALE]
        page_style(_CSS + (_CSS_STALE if old else ""), "thor")
        # Läget väljs ur sessionen innan fältet ritas (samma värde som st.text_input ger i den här körningen)
        if tc.yf(st.session_state.get("thor_ticker") or ""):
            _analysis_page(st, data, rows, fresh, old)
        else:
            _list_page(st, data, rows, fresh, old)


def _list_page(st, data: dict, rows: list, fresh: list, old: list) -> None:
    """Listläget. Morgonrundans pill (§3.5.20) står i huvudraden och foten sist — bara under rundan (routine)."""
    from ui import routine
    from ui.components import head_row, page_header
    with head_row("thor", meta=f"hämtad {data.get('loaded')} · {len(rows)} rader"):
        page_header("🔨 Thor")
        _refresh_button(st)
        _about(st, "thor_about")
        routine.pill("thor")
    st.markdown(cards.lage_strip(lage_items(data, rows)), unsafe_allow_html=True)
    _problem_line(st, data, old)
    _list(st, rows, data.get("board"), data, field=True)
    _market_notes(data)
    _fresh_box(st, data, fresh, old)
    _universe_expander(st, data)
    routine.footer("thor")


def _analysis_page(st, data: dict, rows: list, fresh: list, old: list) -> None:
    """Analysläget: navraden, analysen (beslut till vänster, bevis till höger från 1024 px) och sedan listans ram —
    ◀ Kandidater, läget med 📡 och körningarna, Universum-registret. Ingen pill och ingen fot: analysen är inte
    rundans steg."""
    from ui.components import section
    with st.container(horizontal=True, vertical_alignment="center", key="thor_nav"):
        st.button("◀ Tillbaka till listan", key="thor_back", on_click=_close, type="tertiary")
        ticker = tc.yf(_ticker_field(st))
    if ticker:
        _analysis(st, data, rows, ticker)
    with st.expander(f"◀ Kandidater ({len(rows)})", expanded=False):
        _list(st, rows, data.get("board"), data)
    st.markdown(section("Listans läge", count=f"hämtad {data.get('loaded')} · {len(rows)} rader"),
                unsafe_allow_html=True)
    st.markdown(cards.lage_strip(lage_items(data, rows)), unsafe_allow_html=True)
    _problem_line(st, data, old)
    _market_notes(data)
    _fresh_box(st, data, fresh, old)
    with st.container(horizontal=True, vertical_alignment="center", key="thor_lage_actions"):
        _refresh_button(st)
        _about(st, "thor_about_an")
    _universe_expander(st, data)


def _problem_line(st, data: dict, old: list) -> None:
    """EN bärnstensrad när körningar är gamla, saknas eller fallerade (§4.5). Gistens läsfel står här — en gång på
    sidan; felen i detalj står under Körningar per strategi."""
    gist = (data.get("errors") or {}).get("gist")
    if not (old or gist or data.get("errors") or live.blob_errors(data)):
        return
    n = len(old)
    if gist:
        text = f"⚠ {gist}"
    elif n:
        text = ("⚠ 1 körning gammal/saknas/fel" if n == 1 else f"⚠ {n} körningar gamla/saknas/fel") \
            + " — se Körningar per strategi nedan"
    else:
        text = "⚠ Läsfel i körningarna — se Körningar per strategi nedan"
    st.markdown(f"<div class='thor-warn' role='status' style='color:{AMBER};font-size:13px;font-weight:600;"
                f"line-height:1.4;'>{escape(text, quote=False)}</div>", unsafe_allow_html=True)


def _market_notes(data: dict) -> None:
    """Viking Nines marknadsläge (inte ett fel) synligt under listan — en gång på sidan, inte inne i felrutan."""
    from ui.components import note
    for line in tc.vn_market_notes(data.get("viking_nine")):
        note(line)


def _fresh_box(st, data: dict, fresh: list, old: list) -> None:
    """Körningar per strategi (under listan): felen i detalj — filer som inte gick att läsa, körningar som
    fallerade med strategierna som chip — och tabellen med läget per strategi."""
    from ui.components import note
    errors = {k: v for k, v in (data.get("errors") or {}).items() if k != "gist"}   # Gistens fel: raden ovan
    failed = live.blob_errors(data)
    with st.container(key="thor_fresh"):
        with st.expander("Körningar per strategi" + (f" · {len(old)} gamla/saknas/fel" if old else " · alla färska"),
                         expanded=False):
            if errors:
                note("Kunde inte läsa " + escape(", ".join(f"{k} ({v})" for k, v in errors.items()), quote=False)
                     + " — de raderna saknas i listan tills filen är rättad eller körningen körd om.")
            if failed:
                groups: dict = {}
                for k, (err, gen) in failed.items():
                    groups.setdefault((err, gen), []).append(reg.label(k))
                for (err, gen), names in groups.items():
                    note("Körningen fallerade" + (f" ({escape(gen, quote=False)})" if gen else "") + f": {escape(err, quote=False)} "
                         + " ".join(cards.chip(n_, GREY) for n_ in names))
                note("De raderna saknas tills körningen lyckas igen (schemat eller ⚡ Skanna nu).")
            st.dataframe(pd.DataFrame(fresh), hide_index=True, use_container_width=True)
            note("LIVE = körd med ⚡ Skanna nu i den här sessionen (📡 Uppdatera läser Gisten igen). GAMMAL = äldre än "
                 "strategins gräns. De tunga skanningarna körs bara i schemat (GitHub Actions).")


def _universe_expander(st, data: dict) -> None:
    with st.expander("Universum-registret", expanded=False):
        _universe_box(st, data.get("universe"))


def _scan_now(st, data: dict, strategy: str) -> None:
    """⚡ Skanna nu för en vald strategi: samma headless-skanning som schemat, raderna byts i sessionen."""
    if strategy in live.HEAVY:
        asof = (tc.summary(data.get("rows") or [])["by_strategy"].get(strategy) or {}).get("asof")
        st.caption(f"{live.HEAVY[strategy]} körs i schemat (GitHub Actions) — senast {asof or 'aldrig'}.")
        return
    if not live.can_run(strategy):
        return
    stamp = (data.get("live") or {}).get(strategy)
    part = live.PARTIAL.get(strategy)
    with st.container(horizontal=True, vertical_alignment="center", key="thor_scan"):
        clicked = st.button("⚡ Skanna nu", key=f"thor_scan_{strategy}")
        if stamp:
            st.caption(f"skannad live {stamp}" + (f" — bara {part}; arkets och kartans rader behåller sitt datum" if part else ""))
        else:
            st.caption("Skannar strategin nu med schemats förval — Gisten rörs inte."
                       + (f" Bara {part} skannas; övriga rader kommer från kartan/arken." if part else ""))
    if clicked:
        with st.spinner(f"Skannar {reg.label(strategy)} …"):
            try:
                new = rebuild(live.run(strategy, data))
            except live.LiveScanError as exc:
                st.error(f"Skanningen fallerade: {exc}")
                return
            except Exception as exc:
                st.error(f"Skanningen fallerade: {exc} — Gistens rader behålls.")
                return
        st.session_state[_STATE] = new
        st.session_state.pop(_CACHE, None)
        st.rerun()


def _filter_count(st) -> int:
    """Hur många av Screening och Segment som inte står på förvalet — talet på ⚙ Filter-knappen."""
    return int((st.session_state.get("thor_pick") or ALL) != ALL) + \
        int((st.session_state.get("thor_segment") or SEGMENT_ALL) != SEGMENT_ALL)


def _list(st, rows: list, board: Optional[dict] = None, data: Optional[dict] = None, field: bool = False) -> None:
    """Listan: sökraden (tickerfältet när field=True, och ⚙ Filter med Screening-valet och segmentet), filterchipsen
    och 🌍 Marknad, de riktiga raderna (sidvis) som tryckkort — och under dem segmentets största bolag utan signal
    (thor.segment_top) när ett segment, tema eller komplex är valt."""
    from thor import segment_top as stp
    from ui.components import tap_card
    n_flt = _filter_count(st)
    choices = dict(segment_choices(board))
    with st.container(horizontal=True, vertical_alignment="center", key="thor_search"):
        if field:
            _ticker_field(st)
        with st.popover(f"⚙ Filter · {n_flt}" if n_flt else "⚙ Filter", key="thor_filter_pop"):
            pick = st.selectbox("Screening", [ALL] + list(FAMILY_ALL) + [s.label for s in reg.STRATEGIES.values()],
                                key="thor_pick")
            seg = st.selectbox("Segment", list(choices), format_func=lambda k: choices.get(k, str(k)),
                               key="thor_segment",
                               help="Valhall-kartans segment, teman och råvarukomplex med status — sök köp bara där "
                                    "det är grönt")
    n_mkt = len(st.session_state.get("thor_markets") or [])
    with st.container(horizontal=True, vertical_alignment="center", key="thor_quick"):
        if hasattr(st, "pills"):
            flt = st.pills("Filter", list(FILTERS), selection_mode="multi", key="thor_filters",
                           label_visibility="collapsed")
            with st.popover(f"🌍 Marknad · {n_mkt}" if n_mkt else "🌍 Marknad", key="thor_market_pop"):
                mkts = st.pills("Marknad", list(MARKET_LABELS), selection_mode="multi", key="thor_markets",
                                label_visibility="collapsed", help="Universum-registrets marknader — inget valt = alla")
        else:                                                       # pragma: no cover - äldre Streamlit
            flt = st.multiselect("Filter", list(FILTERS), key="thor_filters", label_visibility="collapsed")
            mkts = st.multiselect("Marknad", list(MARKET_LABELS), key="thor_markets", label_visibility="collapsed")
    strategy = pick_to_filter(pick)[1]
    if strategy and data is not None:
        _scan_now(st, data, strategy)
    shown = shown_rows(rows, pick, flt, mkts, seg)
    d = data or {}
    unreadable = "gist" in (d.get("errors") or {})                  # ingen körning gick att läsa: inget att säga om signaler
    extra, extra_note = ([], "") if unreadable else stp.top_rows(seg, rows, pick, flt, mkts, board, d.get("segment_top"),
                                                                 d.get("names"), shown)
    s = tc.summary(shown)
    where = f" · {escape(choices.get(seg, str(seg)).upper())}" if seg and seg != SEGMENT_ALL else ""
    more = f" · {len(extra)} STÖRSTA UTAN SIGNAL" if extra else ""
    st.markdown(f"<div class='thor-count' style='color:{CYAN};font-size:12px;font-weight:600;letter-spacing:0.1em;"
                f"line-height:1.3;'>{s['n']} KANDIDATER · {s['buy']} KÖP · {pick.upper()}{where}{more}</div>",
                unsafe_allow_html=True)
    if not shown:
        from ui.components import empty_state
        fails = {k: v for k, v in live.blob_errors(data).items() if pick == ALL or reg.label(k) == pick}
        if unreadable:                       # Gistens fel står en gång på sidan (raden om körningarna) — här bara varför
            empty_state("Listan är tom för att körningarna inte gick att läsa — inte för att körningar saknas. "
                        "Tryck 📡 Uppdatera om en stund.")
        elif fails:                          # felet i detalj står under Körningar per strategi
            empty_state("Körningen fallerade för " + ", ".join(reg.label(k) for k in fails)
                        + " — felet står under Körningar per strategi nedan. Kör den igen när felet är rättat.")
        elif extra:
            empty_state("Ingen kandidatrad med de filtren i dag — segmentets största bolag utan signal står nedan.")
        else:
            empty_state("Inga rader med de filtren. Kör skanningarna i GitHub Actions eller lätta på filtren.")
        if not (extra or extra_note):
            return
    else:
        n = int(st.session_state.get("thor_show") or PAGE)
        with st.container(key="thor_rows"):              # hela kortet är knappen (§3.5.11) — nyckel, etikett och callback som förut
            for c in shown[:n]:
                tap_card(f"thor_open_{c.strategy}_{c.ticker}", cards.candidate_card(c), "Öppna →", on_click=_goto,
                         args=(c.ticker, c.strategy))
        if len(shown) > n:
            st.button(f"Visa fler ({len(shown) - n} till)", key="thor_more", on_click=_more, width="stretch")
    _top_section(st, extra, extra_note, stp.title(seg, pick), strategy, d)


def _top_section(st, extra: list, note_txt: str, title: str, strategy: Optional[str], data: dict) -> None:
    """Avsnittet under listan: segmentets största bolag utan rad i Screening-valet (thor.segment_top) — INGEN
    SIGNAL i grått, egna Öppna-knappar (tryckkort). Raderna finns bara här, aldrig i data["rows"]. Öppna → skickar
    strategin när valet är en strategi, annars väljer analysblocket (td.default_strategy)."""
    from ui.components import note, tap_card
    if extra:
        st.markdown(f"<div class='thor-toptitle' style='color:{MUTED};font-size:12px;font-weight:600;"
                    f"letter-spacing:0.1em;line-height:1.3;margin-top:8px;'>{escape(title)}</div>",
                    unsafe_allow_html=True)
    if note_txt:
        note(escape(note_txt))
    failed = sorted(set(data.get("errors") or {}) | set(live.blob_errors(data)))
    if extra and failed:                     # bara namnen — felet i detalj står under Körningar per strategi
        note("Saknas i listan: " + " ".join(cards.chip(reg.label(k), GREY) for k in failed)
             + " — där kan det finnas signaler som inte syns här.")
    if extra:
        with st.container(key="thor_toprows"):
            for c in extra:
                tap_card(f"thor_open_{tc.TOP}_{c.ticker}", cards.candidate_card(c), "Öppna →", on_click=_goto,
                         args=(c.ticker, strategy))


def _universe_box(st, index: Optional[dict]) -> None:
    """Registrets läge: bolag per marknad, förvalet per strategi och hur det byggs."""
    from ui.components import note
    if not index:
        note("Universum-registret har inte byggts än — kör Universum-registret i GitHub Actions (måndagar automatiskt). "
             "Tills dess läser Thor strategiernas egna listor som förut.")
    else:
        st.markdown(f"<div style='color:{MUTED};font-size:12px;line-height:1.4;'>byggt "
                    f"{escape(str(index.get('generated') or '—')[:16])} · "
                    f"{index.get('members') or 0} bolag · källor: "
                    + " · ".join(f"{escape(str(k))} {v}" for k, v in sorted((index.get("stats") or {}).items(),
                                                                            key=lambda kv: -kv[1]))
                    + "</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(ur.summary_rows(index)), hide_index=True, use_container_width=True)
        for n_ in index.get("notes") or []:
            note(str(n_))
    st.markdown(f"<div style='color:{CYAN};font-size:12px;font-weight:600;letter-spacing:0.1em;margin-top:6px;'>"
                f"FÖRVALET PER STRATEGI</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(ur.strategy_rows()), hide_index=True, use_container_width=True)
    note("Ett universum för alla strategier: Norden som förval, USA och Kanada för råvarustrategierna (BERSERK, EMBER, "
         "Rick Rule, Durrett, Tiggre, Håvarna) — alla marknader går att välja i marknadsfiltret ovan. Registret byggs ur "
         "Börsdata (nordiskt + globalt) och panelens egna listor; skannrarna läser det i egna steg.")


def _dossier(st, ticker: str, stores: dict, rows: Optional[list] = None) -> dict:
    """Identiteten och ögonblicksbilden, cachade en timme per aktie (arken läses varje gång — de ändras)."""
    cache = st.session_state.setdefault(_CACHE, {})
    hit = cache.get(ticker)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    with st.spinner(f"Hämtar {ticker} …"):
        bare = str(ticker or "").strip().upper()
        keep_bare = "." not in bare and " " not in bare and any(str(c.ticker).upper() == bare for c in rows or [])
        ident = td.identify(ticker, _api(), stores, _table(), keep_bare=keep_bare)
        if not ident.name or not ident.theme:
            try:
                fill_from_universe(ident, universe_member(st, ident.yf))   # registret som reserv för okända tickers
            except Exception:
                pass
        df = _ohlcv(ident.yf)
        try:                                                  # stängda staplar som skannrarna (granskning 2, F16)
            import ovtlyr_nine as on
            df = on.closed_only(df, ident.yf) if df is not None and len(df) else df
        except Exception:
            pass
        snap, err = td.snapshot(df, ident.yf)
    out = {"ident": ident, "snapshot": snap, "snapshot_error": err}
    # Ett misslyckat uppslag (ingen kurs, eller Börsdata svarade inte) cachas inte — nästa tryck försöker igen (C1)
    failed = snap is None or (_api() is not None and getattr(ident, "ins_id", None) is None and ident.nordic)
    if failed:
        cache.pop(ticker, None)
    else:
        cache[ticker] = (time.time(), out)
    return out


def _analysis(st, data: dict, rows: list, ticker: str) -> None:
    """Analysblocket (§4.6): beslutet till vänster — huvudet, Bedöm som, verdiktet, STORLEK och knapparna — och bevisen
    till höger — nyckeltalen och sidoflikarna. Två kolumner från 1024 px, staplade i den ordningen under."""
    from ui.components import kpi_strip, missing
    stores = _stores()
    d = _dossier(st, ticker, stores, rows)
    ident = d["ident"]
    cands = td.candidates_for(rows, ident.yf)
    exact = td.bare_rows(cands, ident.yf)
    if exact:                                                          # 'TEL' i listan är TE Connectivity, inte TEL.OL (T1)
        cands = exact
    elif "." not in ident.yf and cands and "." in cands[0].ticker:    # 'ERIC B' utan Börsdata → listan vet att det är ERIC-B.ST
        d = _dossier(st, cands[0].ticker, stores, rows)
        ident = d["ident"]
        cands = td.candidates_for(rows, ident.yf)
    if not st.session_state.get("_thor_strategy_set") and st.session_state.get("_thor_last") != ident.yf:
        st.session_state["thor_strategy"] = reg.label(td.default_strategy(ident, cands))
    st.session_state["_thor_last"] = ident.yf
    st.session_state.pop("_thor_strategy_set", None)
    sheets = td.sheets_for(ident.yf, stores, data.get("sheet_map"))
    with st.container(key="thor_an"):
        left_col, right_col = st.columns([5, 7])
    left, right = left_col.container(key="thor_left"), right_col.container(key="thor_right")
    with left:
        st.markdown(cards.header_card(ident.as_dict(), cands, held=bool(sheets["positions"]), board=data.get("board")),
                    unsafe_allow_html=True)
        labels = [s.label for s in reg.STRATEGIES.values()]
        picked = st.selectbox("Bedöm som", labels, key="thor_strategy")
        strategy = next((s.key for s in reg.STRATEGIES.values() if s.label == picked), "viking_nine")
        c = next((x for x in cands if x.strategy == strategy), None)
        st.markdown(cards.verdict_for(c, strategy), unsafe_allow_html=True)
        # STORLEK ritas i size_box ovanför knapparna; 👀 står först i knappraden och ritas alltid (även utan entry/stopp)
        size_box, act_box = st.container(), st.container(key="thor_actions")
        with act_box:
            _watch_button(st, ident, strategy, c)
        with size_box:
            _storlek(st, ident, strategy, c, act_box)
    with right:
        if d["snapshot"]:
            st.markdown(kpi_strip(cards.snapshot_items(d["snapshot"])), unsafe_allow_html=True)
        else:
            err = str(d.get("snapshot_error") or "okänt fel")
            said = f"ingen kursdata för {ident.yf}"
            why = "" if said.lower() in err.lower() else f": {err}"        # felet ordagrant, men inte två gånger
            st.markdown(f"<div class='thor-nosnap' style='display:flex;flex-wrap:wrap;align-items:center;gap:6px;"
                        f"color:{MUTED};font-size:13px;line-height:1.4;'>{missing(err)}<span>— "
                        f"{escape(said + why, quote=False)}</span></div>", unsafe_allow_html=True)
        labels = [x for x in TABS if x != "Råvara" or ident.is_commodity]
        tabs = st.tabs(labels, on_change="rerun", key="thor_tabs")
        render = {"Regim": _tab_regim, "Teknik": _tab_teknik, "Cykel": _tab_cykel, "Råvara": _tab_ravara,
                  "Asymmetri": _tab_asymmetri, "Arken": _tab_arken, "Backtest": _tab_backtest, "Regler": _tab_regler}
        for lbl, tab in zip(labels, tabs):
            if getattr(tab, "open", None) is False:
                continue
            with tab:
                render[lbl](st, data, ident, strategy, sheets, cands)


def _watch_add(ticker: str, name: str, strategy: str, at_add: dict) -> None:
    """Callback: paret till Övervakningen och en toast — ett sparfel sägs rakt ut, som Logga köp."""
    import streamlit as st
    import watchlist as wl
    _ok, msg = wl.add(ticker, strategy, name=name, source="thor", at_add=at_add)
    st.toast(msg, icon="👀")


def _watch_button(st, ident, strategy: str, c) -> None:
    """👀 Lägg i Övervakning: aktien med strategin i 'Bedöm som' (watchlist.py) — en gång per par. Bevakas paret
    redan blir knappen en länk dit. c = strategins rad eller None (ingen rad, Största i segmentet)."""
    import watchlist as wl
    from ui.links import go
    try:
        hit = wl.find(ident.yf, strategy)
    except Exception:                                      # ett trasigt lager får aldrig fälla analysen
        hit = None
    label = reg.label(strategy)
    if hit:
        since = f" sedan {hit['added']}" if hit.get("added") else ""
        st.button(f"👀 Bevakas{since} ({label}) — öppna Övervakning →", key="thor_watch_open", on_click=go,
                  args=("watch",))
        return
    st.button(f"👀 Lägg i Övervakning ({label})", key="thor_watch_add", on_click=_watch_add,
              args=(ident.yf, ident.name or "", strategy, wl.snapshot(c)),
              help="Bevaka aktien med strategin — läget följs under PORTFÖLJ → 👀 Övervakning")


def _storlek(st, ident, strategy: str, c, actions=None) -> None:
    """Storleken (thor/storlek.py, §3.5.14): riskstorlek och allokerarens tak sida vid sida som rutor, raden att
    kopiera under — och i knappraden (actions) 'Logga köp' som lägger positionen i registret. Arkstrategierna går via
    🎯 Scorecard (köpgrinden). Utan färsk KÖP-rad visas storleken bara för planering och 'Logga köp' är avstängt —
    strategins eget verdikt gäller (granskning 3, T2)."""
    from thor import storlek as sk
    from ui import settings as us
    from ui.components import note, section
    from ui.links import go
    if c is None or not c.entry or not c.stop:
        return
    capital = us.capital() or 100000.0
    ccy, fx = sk.fx_for(c.ticker)
    risk = sk.risk_pct(strategy)
    sz = sk.size(c.entry, c.stop, capital, risk, sk.cap_pct(strategy), fx=fx, currency=ccy)
    if not sz:
        return
    if "GUL regim" in (c.status or ""):
        sz = sk.halved(sz)
    st.markdown(section("STORLEK") + cards.storlek_grid(sz, risk, sk.row_text(ident.yf, reg.label(strategy), sz)),
                unsafe_allow_html=True)
    buy_ok = c.status_class == tc.KOP and not c.stale
    if c.status_class != tc.KOP:
        note(f"Ingen köpsignal ({escape(cards.main_word(c))}: {escape(c.status or '')}) — storleken är bara för planering.")
    elif c.stale:
        note(f"Gammal körning (körd {c.asof}, äldre än {tc.STALE_DAYS} dagar) — storleken är bara för planering. "
             "Tryck 📡 Uppdatera och kontrollera entry och stopp först.")
    if not us.capital():
        note("Kapital 100 000 kr (standard) — fyll i ditt eget under MER → ⚙️ Inställningar.")
    with actions if actions is not None else st.container():
        if strategy in sk.SHEET_STRATEGIES:
            st.button("Köpgrinden: öppna 🎯 Scorecard →", key="thor_to_scorecard", on_click=go, args=("scorecard",))
            return
        with st.expander("📝 Logga köp", expanded=False):
            default = sz["risk_shares"] if sz["risk_shares"] is not None else (sz["cap_shares"] or 0)
            shares = st.number_input("Antal aktier", min_value=0, value=int(default), step=1,
                                     key=f"thor_buy_shares_{strategy}_{ident.yf}")   # nytt fält per aktie och strategi (D2)
            note("Riskstorleken och taket står ovan — du väljer antalet. Positionen hamnar i PORTFÖLJ → Positioner med "
                 "stopp och mål; journalraden skrivs när du stänger den.")
            if not buy_ok:
                note(f"Logga köp öppnas när {escape(reg.label(strategy))} säger KÖP i en färsk körning.")
            if st.button("Logga köp", key="thor_log_buy", disabled=not shares or not buy_ok):
                import positions
                ok, msg = positions.add(**sk.position_args(ident.yf, ident.name or "", strategy, sz, shares, c.target))
                (st.success if ok and positions.UNSAVED not in msg else st.warning)(msg)
                st.toast(msg, icon="📝")                        # inget tyst (§2.8): kvittot syns även när rutan är långt ned


def _bridge(st, label: str, target: tuple, ticker: str, key: str, extra: Optional[dict] = None) -> None:
    path, name, widget = target
    st.button(f"{label} {name} →", key=key, on_click=_jump, args=(path, widget, ticker, extra))


# ── Sidoflikarna ────────────────────────────────────────────────────────────
def _tab_regim(st, data, ident, strategy, sheets, cands) -> None:
    """Strategins eget regimbeslut för aktien (thor.regim) — för arkstrategierna Arkens dom (thor.dom) — och
    knappar till den gamla sidan och Copiloten."""
    s = reg.get(strategy)
    if ((s.regime if s else "") or "sheet") in ("sheet", "insider"):
        from thor import dom
        dom.render(st, data, ident, strategy, sheets, cands, _prices)
    else:
        from thor import regim
        regim.render(st, ident, strategy, data, _prices, cands)
    regime_key = (s.regime if s else "") or ""
    target = BRIDGE.get(regime_key, BRIDGE["sheet"])
    b1, b2 = st.columns(2)
    with b1:
        _bridge(st, "Öppna i", target, ident.yf, "thor_bridge_regim", BRIDGE_EXTRA.get(regime_key))
    with b2:
        import strategy_rules as _sr
        pb = s.playbook if s and s.playbook in _sr.PLAYBOOKS else None
        _bridge(st, "Öppna i", COPILOT, ident.yf, "thor_bridge_copilot", {"copilot_strategy": pb} if pb else None)


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
    arken.render(st, data, ident, sheets, cands, SHEET_LINK, _jump, strategy=strategy)


def _tab_backtest(st, data, ident, strategy, sheets, cands) -> None:
    """Strategin på aktien med en knapp (thor.backtest); utan per-aktie-motor: Valhalls backtester och gamla sidan."""
    from thor import backtest
    if not backtest.render(st, ident, strategy, _prices):
        _bridge(st, "Öppna", VALHALL_BT, ident.yf, "thor_bridge_valhall_bt", {"goal_backtests": True})
    _bridge(st, "Öppna", BACKTEST, ident.yf, "thor_bridge_backtest")


def _tab_regler(st, data, ident, strategy, sheets, cands) -> None:
    """Hela playbooken för strategin — regler och walkthrough, sist på sidan."""
    from thor import regim
    regim.render_rules(st, strategy)
