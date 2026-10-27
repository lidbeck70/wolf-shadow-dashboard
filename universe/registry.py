"""
universe/registry.py — universum-registret: ETT tickeruniversum för alla strategier.

Panelen har åtta olika universum med var sitt ordförråd (Norden/USA/Kanada hette olika saker i BERSERK, EMBER,
Contrarian Alpha, ticker_universe, Wolf-screenern …) och bara markets.py som delad tabell. Här finns EN lista:

  MARKETS      Norden (förval) · USA · Kanada — och London, Australien som extra för de håvar som redan
               skannar dem. Varje marknad har land, Yahoo-suffix och marknadsgrindens index.
  Member       en ticker i Yahoo-form med namn, marknad, Börsdata-id (insId, marknad, sektor, bransch),
               valuta, råvarutema (BERSERK-vokabulär), slag (aktie/ETF) och källorna den kommer ur.
  build()      Börsdata /instruments (Norden) och /instruments/global (USA, Kanada … — kräver Pro+ global;
               utan licens blir listan tom och USA/Kanada kommer bara ur listorna) + panelens egna listor
               (BERSERK, ticker_universe, Contrarian Alphas US/CA-CSV, Viking Nine-backtestet, Wolf-screenern,
               heatmapen, EMBER, Long Screener). Bara aktielistor (markets.is_stock); OTC hoppas över.
  for_strategy()  förvalet per strategi: råvarustrategierna (BERSERK, EMBER, Rick Rule, Durrett, Tiggre,
               Håvarna) Norden + USA + Kanada, övriga Norden — alla marknader går att välja.
  bloben       universe.json (index: statistik, Nordens Börsdata-form → Yahoo-karta) och universe_<marknad>.json
               per marknad (medlemmarna som kompakta rader) — Gistens filer har ett tak på ≈ 1 MB.

Registret ÄNDRAR inga skanningar: skannrarna läser det i egna steg. Thor läser det för marknadsfiltret,
momentumradernas suffix och som reserv för tickers Börsdata Norden inte känner. Ren Python; Börsdata-klienten,
tabellen och listorna skickas in eller importeras lat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

NORDEN, USA, KANADA, LONDON, AUSTRALIEN = "norden", "usa", "kanada", "london", "australien"
STOCK, ETF = "stock", "etf"
EXCLUDE_MARKET_IDS = (34,)             # OTC — går inte att handla hos Nordnet; EMBER och Contrarian hoppar den
INDEX_BLOB = "universe.json"
MEMBER_BLOB = "universe_{key}.json"
COLUMNS = ("ticker", "name", "market", "country", "bd_ticker", "ins_id", "market_id", "sector_id", "branch_id",
           "instrument_type", "currency", "theme", "kind", "sources")
# Contrarian Alphas US/CA-CSV (primary_commodity) → BERSERK-tema
CSV_THEME = {"gold": "guld", "silver": "silver", "copper": "koppar", "uranium": "uran", "oil_gas": "olja",
             "natural_gas": "naturgas", "coal": "kol", "iron_ore": "jarnmalm", "lithium": "litium",
             "rare_earth": "sallsynta", "steel": "stal", "aluminum": "aluminium", "aluminium": "aluminium",
             "nickel": "zink_nickel", "zinc": "zink_nickel", "platinum": "platina", "palladium": "palladium"}


@dataclass(frozen=True)
class Market:
    key: str
    label: str
    countries: tuple           # landsnamn (gemener) som Börsdata ger dem — samma som markets.COUNTRY_SUFFIX
    suffixes: tuple            # Yahoo-suffix ('' = bar ticker)
    index: str                 # marknadsgrindens index
    default: bool = False      # förval
    source: str = ""


MARKETS: tuple = (
    Market(NORDEN, "Norden", ("sverige", "sweden", "norge", "norway", "danmark", "denmark", "finland", "island",
                              "iceland"), (".ST", ".OL", ".CO", ".HE", ".IC"), "OMXS30", True,
           "Börsdata /instruments"),
    Market(USA, "USA", ("usa", "united states", "förenta staterna", "amerika"), ("",), "SPY", False,
           "Börsdata /instruments/global (Pro+ global) · annars panelens listor"),
    Market(KANADA, "Kanada", ("kanada", "canada"), (".TO", ".V", ".CN", ".NE"), "^GSPTSE", False,
           "Börsdata /instruments/global (Pro+ global) · annars panelens listor"),
    Market(LONDON, "London", ("england", "storbritannien", "united kingdom", "uk"), (".L",), "^FTSE", False,
           "extra — håvarna (Rule/Sprott/Durrett/Tiggre) och BERSERK skannar London redan"),
    Market(AUSTRALIEN, "Australien", ("australien", "australia"), (".AX",), "^AXJO", False,
           "extra — håvarna räknar med ASX (går inte att handla hos Nordnet)"),
)
BY_KEY: dict = {m.key: m for m in MARKETS}
DEFAULT: tuple = tuple(m.key for m in MARKETS if m.default)
COMMODITY_MARKETS: tuple = (NORDEN, USA, KANADA)
COMMODITY_FAMILIES = ("råvara", "håv")


def label(key: str) -> str:
    m = BY_KEY.get(key)
    return m.label if m else str(key)


def market_of(ticker) -> str:
    """Marknadsnyckeln ur Yahoo-suffixet; en bar ticker räknas som USA, ett okänt suffix (.DE, .PA …) som ''."""
    t = str(ticker or "").strip().upper()
    if not t:
        return ""
    for m in MARKETS:
        if any(s and t.endswith(s) for s in m.suffixes):
            return m.key
    return USA if "." not in t else ""


def market_for_country(country) -> Optional[str]:
    c = str(country or "").strip().lower()
    return next((m.key for m in MARKETS if c in m.countries), None) if c else None


def markets_for(strategy: str) -> tuple:
    """Strategins förval: råvarustrategierna Norden + USA + Kanada, övriga Norden (thor.registry-nycklar)."""
    try:
        from thor import registry as reg
        s = reg.get(strategy)
        family = s.family if s else ""
    except Exception:
        family = ""
    return COMMODITY_MARKETS if family in COMMODITY_FAMILIES else DEFAULT


# ── Medlemmarna ─────────────────────────────────────────────────────────────
@dataclass
class Member:
    ticker: str
    name: str = ""
    market: str = ""
    country: str = ""
    bd_ticker: str = ""            # Börsdatas form ('ERIC B')
    ins_id: Optional[int] = None
    market_id: Optional[int] = None
    sector_id: Optional[int] = None
    branch_id: Optional[int] = None
    instrument_type: Optional[int] = None
    currency: str = ""
    theme: str = ""                # BERSERK-vokabulär (koppar, guld, olja …)
    kind: str = STOCK
    sources: tuple = ()

    def row(self) -> list:
        return [getattr(self, c) if c != "sources" else list(self.sources) for c in COLUMNS]

    @classmethod
    def from_row(cls, row: list) -> "Member":
        d = dict(zip(COLUMNS, row))
        d["sources"] = tuple(d.get("sources") or ())
        return cls(**d)


@dataclass
class Universe:
    members: dict = field(default_factory=dict)    # {ticker: Member}
    generated: str = ""
    stats: dict = field(default_factory=dict)       # {källa: antal}
    notes: list = field(default_factory=list)

    def get(self, ticker) -> Optional[Member]:
        return self.members.get(str(ticker or "").strip().upper())

    def counts(self) -> dict:
        out = {m.key: 0 for m in MARKETS}
        for mem in self.members.values():
            out[mem.market] = out.get(mem.market, 0) + 1
        return out

    def tickers(self, markets: Optional[Iterable[str]] = None, themes: Optional[Iterable[str]] = None,
                kinds: Optional[Iterable[str]] = None, commodity_only: bool = False) -> list:
        ms = set(markets) if markets else None
        ts = set(themes) if themes else None
        ks = set(kinds) if kinds else None
        return sorted(t for t, m in self.members.items()
                      if (ms is None or m.market in ms) and (ts is None or m.theme in ts)
                      and (ks is None or m.kind in ks) and (not commodity_only or m.theme))

    def for_strategy(self, strategy: str, commodity_only: Optional[bool] = None) -> list:
        """Strategins universum: förvalda marknader; råvarustrategierna bara bolag med råvarutema."""
        ms = markets_for(strategy)
        if commodity_only is None:
            commodity_only = ms == COMMODITY_MARKETS
        return self.tickers(ms, commodity_only=commodity_only)

    def bd_map(self, markets: Optional[Iterable[str]] = None) -> dict:
        """Börsdatas form → Yahoo-form ('ERIC B' → 'ERIC-B.ST') — momentumlistan skriver Börsdata-formen."""
        ms = set(markets) if markets else None
        out: dict = {}
        for t, m in sorted(self.members.items()):
            if m.bd_ticker and (ms is None or m.market in ms):
                out.setdefault(m.bd_ticker.upper(), t)
        return out


# ── Byggarna ────────────────────────────────────────────────────────────────
def _int(v) -> Optional[int]:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _theme(ticker: str, branch_id=None) -> str:
    try:
        from thor.ticker import theme_for
        return theme_for(ticker, None, branch_id)[0] or ""
    except Exception:
        return ""


def from_instruments(instruments: Iterable[dict], table: Optional[dict] = None, source: str = "Börsdata",
                     exclude_market_ids: Iterable[int] = EXCLUDE_MARKET_IDS) -> list:
    """Börsdata-instrument → medlemmar: bara aktielistor (markets.is_stock), marknad ur landet i tabellen,
    Yahoo-form via markets.to_yf, tema ur panelens kartor och Börsdata-branschen."""
    import markets as mk
    table = table if table is not None else mk.current()
    skip = set(exclude_market_ids or ())
    out = []
    for inst in instruments or []:
        if not isinstance(inst, dict) or not inst.get("ticker"):
            continue
        mid = _int(inst.get("marketId"))
        if mid is None or mid in skip or not mk.is_stock(mid, table):
            continue
        m = mk.get(mid, table)
        key = market_for_country(m.country if m else "")
        if key is None:
            continue
        yf = mk.to_yf(inst["ticker"], mid, table)
        out.append(Member(yf, str(inst.get("name") or ""), key, m.country if m else "",
                          str(inst["ticker"]).strip().upper(), _int(inst.get("insId")), mid,
                          _int(inst.get("sectorId")), _int(inst.get("branchId")), _int(inst.get("instrumentType")),
                          str(inst.get("stockPriceCurrency") or ""), _theme(yf, inst.get("branchId")), STOCK,
                          (source,)))
    return out


def _alive(ticker) -> str:
    try:
        from dead_tickers import alive
        hit = alive([ticker])
        return hit[0] if hit else ""
    except Exception:
        return str(ticker or "").strip().upper()


def _list_members(tickers, source: str, themes: Optional[dict] = None, kind: str = STOCK, names: Optional[dict] = None,
                  market: Optional[str] = None) -> list:
    out = []
    for raw in tickers or []:
        t = _alive(raw)
        key = market or market_of(t)
        if not t or not key:
            continue
        nm = (names or {}).get(raw) or (names or {}).get(t) or ""
        if isinstance(nm, dict):
            nm = nm.get("name") or ""
        out.append(Member(t, str(nm), key, theme=str((themes or {}).get(raw) or (themes or {}).get(t) or ""),
                          kind=kind, sources=(source,)))
    return out


def from_lists() -> list:
    """Panelens egna listor — varje källa för sig, en som saknas hoppas över."""
    out: list = []

    def add(name: str, fn: Callable) -> None:
        try:
            out.extend(fn())
        except Exception:
            pass

    def berserk():
        from berserk import universe as bu
        mem = []
        for region in (bu.NORDIC, bu.US, bu.CANADA, bu.LONDON):
            mem += _list_members(list(region), "BERSERK", region)
        return mem + _list_members(list(bu.ETFS), "BERSERK", bu.ETFS, ETF, market=USA)

    def ticker_universe():
        import ticker_universe as tu
        mem = []
        for lst in tu.FALLBACK_TICKERS.values():
            mem += _list_members(lst, "ticker_universe")
        etfs = set(tu.US_ETFS_COMMODITY)
        for m in mem:
            if m.ticker in etfs:
                m.kind = ETF
        return mem

    def contrarian_csv():
        from contrarian_alpha.universe_static import load_resource_universe
        recs = load_resource_universe()
        names = {r.yf_ticker: r.name for r in recs}
        themes = {r.yf_ticker: CSV_THEME.get(str(r.primary_commodity or "").lower(), "") for r in recs}
        return _list_members([r.yf_ticker for r in recs], "Contrarian Alpha US/CA", themes, names=names)

    def viking():
        import viking_backtest as vbt
        lists = list(dict.fromkeys(vbt.NORDIC_50 + vbt.NORDIC_OOS_100 + vbt.US_25))
        return _list_members(lists, "Viking Nine-backtestet")

    def wolf():
        import wolf_shadow_screener as ws
        mem = []
        for lst in ws.MARKETS.values():
            mem += _list_members(list(lst), "Wolf-screenern", names=lst if isinstance(lst, dict) else None)
        return mem

    def heatmap():
        from heatmap import heatmap_streamlit as hm
        return (_list_members(list(hm.US_TICKERS), "Heatmap", names=hm.US_TICKERS)
                + _list_members(list(hm.CANADA_TICKERS), "Heatmap", names=hm.CANADA_TICKERS))

    def ember():
        from ember import config as ec
        from ember import universe as eu
        themes = dict(getattr(ec, "TICKER_THEME_MAP", {}) or {})
        return (_list_members(list(ec.EMBER_ETF_UNIVERSE), "EMBER", themes, ETF, market=USA)
                + _list_members(list(ec.EMBER_STOCK_UNIVERSE), "EMBER", themes)
                + _list_members(list(eu.US_INTL_CURATED), "EMBER", themes))

    def cagr():
        from cagr import cagr_loader as cl
        return (_list_members(list(cl.NORDIC_TICKERS), "Long Screener", names=cl.NORDIC_TICKERS)
                + _list_members(list(cl.ETF_TICKERS), "Long Screener", kind=ETF, names=cl.ETF_TICKERS))

    def theme_lists():
        from universe import theme_members as tm
        return _list_members(tm.tickers(), "Temalistor", {t: tm.theme_of(t) for t in tm.tickers()})

    for name, fn in (("berserk", berserk), ("ticker_universe", ticker_universe), ("csv", contrarian_csv),
                     ("viking", viking), ("wolf", wolf), ("heatmap", heatmap), ("ember", ember), ("cagr", cagr),
                     ("teman", theme_lists)):
        add(name, fn)
    return out


def merge(*groups: Iterable[Member]) -> Universe:
    """En medlem per ticker: den första källan (Börsdata) äger fälten, de andra fyller luckor och lägger till källa."""
    u = Universe(generated=datetime.now(tz=timezone.utc).isoformat())
    for group in groups:
        for m in group or []:
            t = str(m.ticker or "").strip().upper()
            if not t:
                continue
            have = u.members.get(t)
            if have is None:
                u.members[t] = Member(t, m.name, m.market, m.country, m.bd_ticker, m.ins_id, m.market_id, m.sector_id,
                                      m.branch_id, m.instrument_type, m.currency, m.theme, m.kind, tuple(m.sources))
                continue
            for f in ("name", "market", "country", "bd_ticker", "ins_id", "market_id", "sector_id", "branch_id",
                      "instrument_type", "currency", "theme"):
                if not getattr(have, f) and getattr(m, f):
                    setattr(have, f, getattr(m, f))
            if m.kind == ETF and have.kind != ETF and not have.ins_id:
                have.kind = ETF
            have.sources = tuple(dict.fromkeys(have.sources + tuple(m.sources)))
    for m in u.members.values():
        for s in m.sources:
            u.stats[s] = u.stats.get(s, 0) + 1
    return u


def build(api=None, table: Optional[dict] = None, nordic: Optional[list] = None, global_: Optional[list] = None,
          include_lists: bool = True) -> Universe:
    """Hela registret: Börsdata (nordiskt + globalt) och listorna. api = Börsdata-klient (valfri); nordic/global_
    = instrumentlistor (testbart utan nätverk)."""
    import markets as mk
    notes = []
    if api is not None:
        table = mk.load(api=api) if table is None else table
        if nordic is None:
            try:
                nordic = api.get_instruments() or []
            except Exception as exc:
                nordic = []
                notes.append(f"Börsdata /instruments: {exc}")
        if global_ is None:
            try:
                global_ = api.get_global_instruments_list() or []
            except Exception as exc:
                global_ = []
                notes.append(f"Börsdata /instruments/global: {exc}")
    table = table if table is not None else mk.current()
    groups = [from_instruments(nordic or [], table, "Börsdata"),
              from_instruments(global_ or [], table, "Börsdata global")]
    if not global_:
        notes.append("Börsdata global tom (ingen Pro+ global-licens?) — USA/Kanada bara ur panelens listor")
    if include_lists:
        groups.append(from_lists())
    u = merge(*groups)
    u.notes = notes
    return u


# ── Bloben ──────────────────────────────────────────────────────────────────
def to_index(u: Universe) -> dict:
    """Indexfilen: statistik, marknaderna och Nordens Börsdata-form → Yahoo-karta (liten)."""
    return {"generated": u.generated, "stats": u.stats, "notes": u.notes, "counts": u.counts(),
            "markets": [{"key": m.key, "label": m.label, "default": m.default, "index": m.index} for m in MARKETS],
            "bd_map": u.bd_map((NORDEN,)), "members": len(u.members)}


def to_rows(u: Universe, key: str) -> dict:
    """Medlemmarna på en marknad som kompakta rader (kolumnerna i COLUMNS)."""
    rows = [m.row() for t, m in sorted(u.members.items()) if m.market == key]
    return {"generated": u.generated, "market": key, "columns": list(COLUMNS), "rows": rows}


def from_rows(blobs: Iterable[dict], index: Optional[dict] = None) -> Universe:
    u = Universe(generated=str((index or {}).get("generated") or ""), stats=dict((index or {}).get("stats") or {}),
                 notes=list((index or {}).get("notes") or []))
    for b in blobs or []:
        cols = list((b or {}).get("columns") or COLUMNS)
        for row in (b or {}).get("rows") or []:
            d = dict(zip(cols, row))
            m = Member(**{c: d.get(c) for c in COLUMNS if c in d})
            m.sources = tuple(m.sources or ())
            if m.ticker:
                u.members[m.ticker] = m
        if not u.generated:
            u.generated = str((b or {}).get("generated") or "")
    return u


def save(u: Universe, saver: Optional[Callable] = None) -> dict:
    """Skriver indexet och en fil per marknad till Gisten. {fil: ok}."""
    if saver is None:
        from gist_storage import save_blob as saver
    out = {INDEX_BLOB: bool(saver(INDEX_BLOB, to_index(u)))}
    for m in MARKETS:
        name = MEMBER_BLOB.format(key=m.key)
        out[name] = bool(saver(name, to_rows(u, m.key)))
    return out


def default_loader() -> Callable:
    """Registrets förvalda läsare: Gisten (gist_storage.load_blob — går att läsa utan token). Testerna byter ut
    den (tests/conftest.py) så att ingen test läser det levande registret av misstag."""
    from gist_storage import load_blob
    return load_blob


def load_index(loader: Optional[Callable] = None) -> Optional[dict]:
    if loader is None:
        loader = default_loader()
    try:
        return loader(INDEX_BLOB, None)
    except Exception:
        return None


def load(markets: Optional[Iterable[str]] = None, loader: Optional[Callable] = None) -> Optional[Universe]:
    """Registret ur Gisten (indexet och de marknader som begärs — alla som förval). None när inget sparats."""
    if loader is None:
        loader = default_loader()
    index = load_index(loader)
    blobs = []
    for key in (markets or [m.key for m in MARKETS]):
        try:
            b = loader(MEMBER_BLOB.format(key=key), None)
        except Exception:
            b = None
        if b:
            blobs.append(b)
    if not blobs and not index:
        return None
    return from_rows(blobs, index)


def summary_rows(index: Optional[dict]) -> list:
    """Rader för panelen: marknad, antal, förval, index — ur indexfilen."""
    idx = index or {}
    counts = idx.get("counts") or {}
    return [{"Marknad": m.label, "Bolag": counts.get(m.key, 0), "Förval": "ja" if m.default else "",
             "Grind": m.index, "Källa": m.source} for m in MARKETS]


def strategy_rows() -> list:
    """Förvalet per strategi (thor.registry) — vilka marknader strategin får kandidater ur."""
    from thor import registry as reg
    return [{"Strategi": s.label, "Familj": reg.FAMILY_LABEL.get(s.family, s.family),
             "Marknader": " · ".join(label(k) for k in markets_for(s.key))} for s in reg.STRATEGIES.values()]
