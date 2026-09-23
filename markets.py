"""
markets.py — Börsdatas marknader: EN tabell i stället för sex.

Sex moduler hade var sin handskriven kopia av "marknads-id → land/Yahoo-
suffix/index" (ticker_universe, borsdata_api ×2, screens_scan, insider_scan,
ember/universe, contrarian_alpha/engine), och de sa olika saker om id 4/5/6
(Sverige eller NO/FI/DK?), 9 (.OL eller .ST?) och 14–16 (Danmark eller
tillväxtlistor?). Ingen av dem läste Börsdatas eget svar från /markets.

Här finns EN tabell, med tre källor i prioritetsordning:

  1. Börsdata själva — from_api(api.get_markets(), api.get_countries()).
     Skanningsjobben (screens_scan, insider_scan) och motorerna som har ett
     API-objekt använder den, och screens_scan skriver den till bloben så
     panelen får samma tabell utan egen licens.
  2. config/markets.json — skriven av scripts/sync_markets.py.
  3. FALLBACK — tabellen ur ticker_universe.py, den enda av de sex som
     uppgav sig vara avläst ur /instruments och /instruments/global.
     Osäkerheten mellan de gamla kopiorna löses av källa 1 och 2, inte av
     att gissa här.

Suffixet härleds ur LANDET (och börsnamnet där ett land har flera listor:
Toronto/TSX Venture/CSE), inte ur marknads-id. Så får en ny lista rätt
suffix utan att någon uppdaterar en tabell. Australien (.AX) finns med —
masterguidens Sprott/Durrett/Tiggre/Royalty räknar med ASX.

to_yf() gör mellanslag → bindestreck ("SKF A" → "SKF-A.ST") på ett ställe.
Ren Python: ingen Streamlit, inga nätverksanrop utan ett api-objekt.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "markets.json")

STOCK, INDEX, OTHER = "stock", "index", "other"


@dataclass(frozen=True)
class Market:
    id: int
    name: str
    country_id: Optional[int]
    country: str            # landets namn som Börsdata ger det ("Sverige")
    suffix: str             # Yahoo-suffix (".ST"); "" för USA
    kind: str = STOCK       # stock | index | other (valuta, råvaror)
    exchange: str = ""


# ── Land → Yahoo-suffix (svenska och engelska namn; /countries ger svenska) ──
COUNTRY_SUFFIX = {
    "sverige": ".ST", "sweden": ".ST",
    "norge": ".OL", "norway": ".OL",
    "danmark": ".CO", "denmark": ".CO",
    "finland": ".HE",
    "usa": "", "united states": "", "förenta staterna": "", "amerika": "",
    "kanada": ".TO", "canada": ".TO",
    "england": ".L", "storbritannien": ".L", "united kingdom": ".L", "uk": ".L",
    "tyskland": ".DE", "germany": ".DE",
    "frankrike": ".PA", "france": ".PA",
    "spanien": ".MC", "spain": ".MC",
    "portugal": ".LS",
    "italien": ".MI", "italy": ".MI",
    "schweiz": ".SW", "switzerland": ".SW",
    "belgien": ".BR", "belgium": ".BR",
    "nederländerna": ".AS", "netherlands": ".AS", "holland": ".AS",
    "österrike": ".VI", "austria": ".VI",
    "polen": ".WA", "poland": ".WA",
    "estland": ".TL", "estonia": ".TL",
    "lettland": ".RG", "latvia": ".RG",
    "litauen": ".VS", "lithuania": ".VS",
    "australien": ".AX", "australia": ".AX",
    "irland": ".IR", "ireland": ".IR",
}

# Börsnamn som avgör suffixet inom ett land: Kanada har tre listor med olika
# suffix, USA:s tre saknar suffix.
EXCHANGE_SUFFIX = (
    ("tsx venture", ".V"), ("tsxv", ".V"), ("venture", ".V"),
    ("cse", ".CN"), ("canadian securities", ".CN"), ("neo", ".NE"),
)

NORDIC_COUNTRIES = frozenset({"sverige", "sweden", "norge", "norway",
                              "danmark", "denmark", "finland"})

_INDEX_HINTS = ("index", "indices", "indexes")
_OTHER_HINTS = ("forex", "valuta", "currency", "nymex", "commodit", "råvar",
                "crypto", "krypto")


def suffix_for(country: str, exchange: str = "", market_name: str = "") -> str:
    """Yahoo-suffix ur land + börs. Okänt land → '' (samma som USA — men
    okänt, så to_yf lämnar tickern orörd)."""
    text = f"{exchange or ''} {market_name or ''}".lower()
    for hint, suf in EXCHANGE_SUFFIX:
        if hint in text:
            return suf
    return COUNTRY_SUFFIX.get((country or "").strip().lower(), "")


def _kind(name: str, exchange: str, is_index) -> str:
    text = f"{name or ''} {exchange or ''}".lower()
    if is_index or any(h in text for h in _INDEX_HINTS):
        return INDEX
    if any(h in text for h in _OTHER_HINTS):
        return OTHER
    return STOCK


def from_api(markets: Iterable[dict], countries: Iterable[dict]) -> dict:
    """{id: Market} ur Börsdatas /markets och /countries."""
    cmap = {}
    for c in countries or []:
        try:
            cmap[int(c.get("id"))] = str(c.get("name") or "")
        except (TypeError, ValueError):
            continue
    out = {}
    for m in markets or []:
        try:
            mid = int(m.get("id"))
        except (TypeError, ValueError):
            continue
        cid = m.get("countryId")
        try:
            cid = int(cid) if cid is not None else None
        except (TypeError, ValueError):
            cid = None
        country = cmap.get(cid, "") if cid is not None else ""
        name = str(m.get("name") or "")
        exch = str(m.get("exchangeName") or "")
        out[mid] = Market(mid, name, cid, country,
                          suffix_for(country, exch, name),
                          _kind(name, exch, m.get("isIndex")), exch)
    return out


# ── FALLBACK — ticker_universe.py:s avlästa tabell ───────────────────────────
def _fb(mid, name, cid, country, kind=STOCK, exchange=""):
    return Market(mid, name, cid, country, suffix_for(country, exchange, name), kind, exchange)


FALLBACK: dict = {m.id: m for m in (
    # Sverige (landId 1). Listnamnen för 4–6 är inte avlästa — bara att de
    # är svenska aktielistor. Id 18 syns i /instruments utan känt namn.
    _fb(1, "Large Cap", 1, "Sverige"), _fb(2, "Mid Cap", 1, "Sverige"),
    _fb(3, "Small Cap", 1, "Sverige"), _fb(4, "Sverige · lista 4", 1, "Sverige"),
    _fb(5, "Sverige · lista 5", 1, "Sverige"), _fb(6, "Sverige · lista 6", 1, "Sverige"),
    _fb(18, "Sverige · lista 18", 1, "Sverige"),
    _fb(7, "Sverige · index", 1, "Sverige", INDEX), _fb(8, "Sverige · index", 1, "Sverige", INDEX),
    # Norge (2)
    _fb(9, "Oslo Børs", 2, "Norge"), _fb(10, "Norge · lista 10", 2, "Norge"),
    _fb(11, "Norge · lista 11", 2, "Norge"), _fb(12, "Norge · lista 12", 2, "Norge"),
    _fb(27, "Norge · lista 27", 2, "Norge"), _fb(78, "Norge · lista 78", 2, "Norge"),
    _fb(13, "Norge · index", 2, "Norge", INDEX),
    # Danmark (3)
    _fb(14, "Köpenhamn", 3, "Danmark"), _fb(15, "Danmark · lista 15", 3, "Danmark"),
    _fb(16, "Danmark · lista 16", 3, "Danmark"), _fb(17, "Danmark · lista 17", 3, "Danmark"),
    _fb(30, "Danmark · lista 30", 3, "Danmark"),
    _fb(19, "Danmark · index", 3, "Danmark", INDEX),
    # Finland (4)
    _fb(20, "Helsingfors", 4, "Finland"), _fb(21, "Finland · lista 21", 4, "Finland"),
    _fb(22, "Finland · lista 22", 4, "Finland"), _fb(23, "Finland · lista 23", 4, "Finland"),
    _fb(48, "Finland · lista 48", 4, "Finland"),
    _fb(28, "Index", None, "", INDEX), _fb(31, "Index", None, "", INDEX),
    # Globalt (/instruments/global)
    _fb(32, "NYSE", 5, "USA", exchange="NYSE"), _fb(33, "Nasdaq", 5, "USA", exchange="Nasdaq"),
    _fb(34, "OTC", 5, "USA", exchange="OTC"),
    _fb(35, "Toronto", 6, "Kanada", exchange="TSX"),
    _fb(36, "TSX Venture", 6, "Kanada", exchange="TSX Venture"),
    _fb(37, "CSE", 6, "Kanada", exchange="CSE"),
    _fb(38, "London", 7, "England"), _fb(39, "Xetra", 8, "Tyskland"),
    _fb(40, "Paris", 9, "Frankrike"), _fb(41, "Madrid", 10, "Spanien"),
    _fb(42, "Lissabon", 11, "Portugal"), _fb(43, "Milano", 12, "Italien"),
    _fb(44, "Schweiz", 13, "Schweiz"), _fb(45, "Bryssel", 14, "Belgien"),
    _fb(46, "Amsterdam", 15, "Nederländerna"), _fb(50, "Warszawa", 17, "Polen"),
    _fb(52, "Tallinn", 19, "Estland"), _fb(53, "Riga", 20, "Lettland"),
    _fb(54, "Vilnius", 21, "Litauen"),
    _fb(76, "Forex", None, "", OTHER), _fb(77, "Nymex", None, "", OTHER),
)}


# ── Laddning ─────────────────────────────────────────────────────────────────
_LIVE: Optional[dict] = None          # senast lyckade tabell ur API/blob/fil


def serialize(table: dict) -> list:
    return [asdict(m) for m in sorted(table.values(), key=lambda m: m.id)]


def deserialize(rows) -> dict:
    out = {}
    for r in rows or []:
        try:
            m = Market(int(r["id"]), str(r.get("name") or ""), r.get("country_id"),
                       str(r.get("country") or ""), str(r.get("suffix") or ""),
                       str(r.get("kind") or STOCK), str(r.get("exchange") or ""))
        except (KeyError, TypeError, ValueError):
            continue
        out[m.id] = m
    return out


def load(api=None, blob: Optional[dict] = None, path: str = CONFIG_PATH) -> dict:
    """Tabellen, bästa tillgängliga källa först: blob → API → fil → FALLBACK.

    Ett api-objekt utan get_markets (t.ex. ett testfejk) eller ett anrop som
    felar faller tyst vidare. Resultatet från API/blob/fil minns för
    processen (current()).
    """
    global _LIVE
    rows = (blob or {}).get("markets") if isinstance(blob, dict) else None
    if rows:
        t = deserialize(rows)
        if t:
            _LIVE = _merge(t)
            return _LIVE
    if api is not None:
        try:
            t = from_api(api.get_markets(), api.get_countries())
            if t:
                _LIVE = _merge(t)
                return _LIVE
        except Exception:
            pass
    try:
        with open(path, encoding="utf-8") as fh:
            t = deserialize(json.load(fh).get("markets"))
            if t:
                _LIVE = _merge(t)
                return _LIVE
    except (OSError, ValueError, AttributeError):
        pass
    return _LIVE or FALLBACK


def _merge(live: dict) -> dict:
    """Levande tabell ovanpå FALLBACK: det Börsdata säger vinner, och id som
    Börsdata inte nämner (t.ex. globala listor utan global licens) behåller
    den avlästa raden i stället för att försvinna."""
    return {**FALLBACK, **live}


def reset() -> None:
    """Glöm den levande tabellen (tester)."""
    global _LIVE
    _LIVE = None


def current() -> dict:
    """Tabellen panelen räknar med just nu (utan nätverk)."""
    return _LIVE or load()


# ── Uppslag ──────────────────────────────────────────────────────────────────
def _t(table: Optional[dict]) -> dict:
    return table if table is not None else current()


def get(mid, table: Optional[dict] = None) -> Optional[Market]:
    try:
        return _t(table).get(int(mid))
    except (TypeError, ValueError):
        return None


def suffix(mid, table: Optional[dict] = None) -> str:
    m = get(mid, table)
    return m.suffix if m else ""


def is_stock(mid, table: Optional[dict] = None) -> bool:
    """Handlad aktielista. Okänt id → False: index och valutor får inte
    smyga in i ett aktieuniversum bara för att tabellen saknar dem."""
    m = get(mid, table)
    return bool(m and m.kind == STOCK)


def to_yf(ticker, mid=None, table: Optional[dict] = None) -> str:
    """Börsdata-ticker → Yahoo-ticker: 'SKF A' + Large Cap → 'SKF-A.ST'."""
    t = str(ticker or "").strip().upper().replace(" ", "-")
    return t + (suffix(mid, table) if mid is not None else "")


def stock_ids(table: Optional[dict] = None, countries: Optional[Iterable[str]] = None) -> set:
    """Marknads-id för aktielistor, valfritt begränsat till länder (namn)."""
    want = {c.lower() for c in countries} if countries else None
    return {m.id for m in _t(table).values()
            if m.kind == STOCK and (want is None or m.country.lower() in want)}


def nordic_stock_ids(table: Optional[dict] = None) -> set:
    return stock_ids(table, NORDIC_COUNTRIES)


def is_nordic(mid, table: Optional[dict] = None) -> bool:
    m = get(mid, table)
    return bool(m and m.country.lower() in NORDIC_COUNTRIES)


def country_ids(countries: Iterable[dict], hints: Iterable[str]) -> set:
    """Lands-id ur /countries för de landsnamn som innehåller någon av
    ledtrådarna ('australi' träffar både Australien och Australia)."""
    hs = tuple(h.lower() for h in hints)
    out = set()
    for c in countries or []:
        name = str(c.get("name") or "").lower()
        if any(h in name for h in hs):
            out.add(c.get("id"))
    return out


def name(mid, table: Optional[dict] = None) -> str:
    m = get(mid, table)
    if not m:
        return str(mid)
    return f"{m.country} · {m.name}" if m.country else m.name
