"""
positions.py — panelens ENDA positionsregister.

Holdings-fliken ägde sina rader i en Gist (holdings_init.json) som inget
annat ark läste, medan Swing, Tiggre, allokeraren och copiloten höll egna
positionslistor med egna fältnamn. Det här är registret alla ska läsa ur:
data/holdings.json via storage.py (samma väg som de andra arken), med Gisten
som engångskälla vid första laddningen.

Lagringens form är oförändrad så att gammal data laddar rakt av:
    {"swing": [...], "ovtlyr": [...], "long": [...], "closed": [...], "cash": 0}

Radschema (normaliseras vid varje läsning, äldre rader saknar fält):
    id            stabil nyckel (sätts om den saknas)
    ticker        Yahoo-form, versaler
    name          bolagsnamn ("" om okänt)
    strategy      Quality · Deep Contrarian · Viking · Wolf · Untagged
    entry_price   inköpskurs (0 = okänd)
    shares        antal (0 = okänt)
    entry_date    YYYY-MM-DD (hette "added" i Gisten)
    stop, target  kurser eller None
    sector        text
    notes         text
    source        vem som skrev raden: "holdings", senare "swing"/"tiggre"/…
    extras        strategispecifika fält (tranches_deployed för Deep Contrarian)

Sparandet sker direkt vid varje ändring (som Holdings alltid gjort) men via
storage.save_session, så att sidfoten och sparraden visar samma läge som för
de andra arken. Ett misslyckat sparande hamnar i storage.meta() i stället för
att svälja tyst.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import List, Optional

import streamlit as st

import storage

logger = logging.getLogger(__name__)

STORE = "holdings"                    # data/holdings.json
LEGACY_FILE = "holdings_init.json"    # Gisten — läses bara när repofilen saknas

# Hinkarna och deras tak. Nycklarna är lagringens och får inte byta namn.
# momentum (Swing-flikens 6–8) och tiggre (Lobo-arkets 4–6) kom in i steg 2;
# taken är de flikarna hade.
BUCKETS: dict = {
    "swing":    {"name": "Wolf Portfolio",     "max": 5},
    "ovtlyr":   {"name": "Viking Portfolio",   "max": 5},
    "long":     {"name": "Alpha Portfolio",    "max": 20},
    "momentum": {"name": "Momentum Swing",     "max": 8},
    "tiggre":   {"name": "Tiggre (Lobo-arket)", "max": 6},
}
BUCKET_KEYS: tuple = tuple(BUCKETS)

STRATEGIES: tuple = ("Quality", "Deep Contrarian", "Viking", "Wolf", "Untagged",
                     "Momentum", "Tiggre",
                     # masterguidens övriga (copiloten loggar dem hit sedan steg 3)
                     "Alpha", "Ember", "Sprott", "Durrett", "Rule", "Royalty", "Insider")
STRATEGY_BUCKET: dict = {
    "Quality":         "long",
    "Deep Contrarian": "long",
    "Viking":          "ovtlyr",
    "Wolf":            "swing",
    "Untagged":        "long",
    "Momentum":        "momentum",
    "Tiggre":          "tiggre",
    "Alpha":           "long",
    "Ember":           "swing",
    "Sprott":          "long",
    "Durrett":         "long",
    "Rule":            "long",
    "Royalty":         "long",
    "Insider":         "long",
}

# Strategitagg ↔ playbook-nyckel (strategy_rules.PLAYBOOKS, copiloten,
# journalen) och ↔ allokerarens positionsregel (allocator.RULE_BY_KEY).
PLAYBOOK_TAG: dict = {
    "momentum": "Momentum", "wolf": "Wolf", "viking": "Viking", "ember": "Ember",
    "alpha": "Alpha", "quality": "Quality", "contrarian": "Deep Contrarian",
    "rule": "Rule", "sprott": "Sprott", "tiggre": "Tiggre", "durrett": "Durrett",
    "royalty": "Royalty", "insider": "Insider",
}
TAG_PLAYBOOK: dict = {v: k for k, v in PLAYBOOK_TAG.items()}
ALLOCATOR_RULE: dict = {**{tag: key for key, tag in PLAYBOOK_TAG.items()},
                        "Momentum": "swing", "Royalty": "royalty1", "Untagged": None}


def tag_for_playbook(key: str) -> str:
    return PLAYBOOK_TAG.get(str(key or "").strip().lower(), "Untagged")


# ── Värdering i SEK (allokeraren) ────────────────────────────────────────────
# Samma kurser som sheets_refresh.FX_TO_USD, uttryckta i SEK per enhet;
# testet låser att tabellerna stämmer överens. Grova tal — allokeraren
# mäter procent av portföljen, inte ören.
FX_TO_SEK: dict = {"SEK": 1.0, "NOK": 1.0, "DKK": 1.526, "EUR": 11.368, "USD": 10.526,
                   "CAD": 7.684, "AUD": 6.947, "GBP": 13.368, "CHF": 11.895, "PLN": 2.632}
_SUFFIX_CCY: tuple = ((".ST", "SEK"), (".OL", "NOK"), (".CO", "DKK"), (".HE", "EUR"),
                      (".TO", "CAD"), (".V", "CAD"), (".CN", "CAD"), (".AX", "AUD"),
                      (".L", "GBP"), (".SW", "CHF"), (".DE", "EUR"), (".PA", "EUR"),
                      (".AS", "EUR"), (".MI", "EUR"), (".MC", "EUR"), (".WA", "PLN"))


def currency_for(ticker: str) -> str:
    t = str(ticker or "").upper()
    for suf, ccy in _SUFFIX_CCY:
        if t.endswith(suf):
            return ccy
    return "USD"


def valuation(row: dict, fresh: Optional[dict] = None) -> dict:
    """Positionens värde i SEK: antal × kurs. Kursen tas i ordning ur
    sifferuppdateringen (holdings:<id> eller tiggre:<id>), arkets "Kurs nu"
    (extras.current) och sist inköpskursen. Utan antal blir värdet None —
    allokeraren kan inte mäta en position den inte vet storleken på."""
    fresh = fresh or {}
    rid = row.get("id")
    f = fresh.get(f"holdings:{rid}") or fresh.get(f"tiggre:{rid}") or {}
    price, ccy, source = _opt(f.get("price")), f.get("currency"), "sifferuppdateringen"
    if price is None:
        price = _opt((row.get("extras") or {}).get("current"))
        source = "kurs nu i arket"
    if price is None:
        price = _opt(row.get("entry_price")) or None
        source = "inköpskursen"
    ccy = str(ccy or currency_for(row.get("ticker"))).upper()
    # London noterar i pence
    unit = 0.01 if (ccy in ("GBP", "GBX") and str(row.get("ticker", "")).upper().endswith(".L")) else 1.0
    fx = FX_TO_SEK.get("GBP" if ccy == "GBX" else ccy, FX_TO_SEK["USD"])
    shares = _num(row.get("shares"))
    value = round(price * unit * fx * shares, 0) if (price and shares > 0) else None
    return {"price": price, "currency": ccy, "source": source, "shares": shares,
            "value_sek": value, "asof": str(f.get("asof") or "")[:10]}

FIELDS: tuple = ("id", "ticker", "name", "strategy", "entry_price", "shares",
                 "entry_date", "stop", "target", "sector", "notes", "source", "extras")
# Bara på rader i closed — behålls på toppnivå, inte i extras.
EXIT_FIELDS: tuple = ("exit_price", "exit_date", "exit_reason", "result_pct")


# ── Hjälpare ─────────────────────────────────────────────────────────────────
def _num(value, default: float = 0.0) -> float:
    """Tal ur lagrad data: None/""/text/NaN → default. Holdings kraschade
    förut på `shares > 0` när Gisten hade null."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return f


def _opt(value) -> Optional[float]:
    """Frivilligt tal: tomt förblir None, inte 0."""
    if value is None or value == "":
        return None
    f = _num(value, float("nan"))
    return None if f != f else f


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def bucket_for(strategy: str) -> str:
    return STRATEGY_BUCKET.get(strategy, "long")


def normalize_row(raw: dict, bucket: str = "") -> dict:
    """En lagrad rad → radschemat. Okända nycklar behålls under extras så
    att inget som någon skrivit försvinner vid nästa sparning."""
    raw = dict(raw or {})
    extras = dict(raw.pop("extras", None) or {})
    row = {
        "id":          str(raw.pop("id", "") or uuid.uuid4().hex[:8]),
        "ticker":      str(raw.pop("ticker", "") or "").strip().upper(),
        "name":        str(raw.pop("name", "") or ""),
        "strategy":    raw.pop("strategy", None) or "Untagged",
        "entry_price": _num(raw.pop("entry_price", 0)),
        "shares":      _num(raw.pop("shares", 0)),
        "entry_date":  str(raw.pop("entry_date", None) or raw.pop("added", None) or ""),
        "stop":        _opt(raw.pop("stop", None)),
        "target":      _opt(raw.pop("target", None)),
        "sector":      str(raw.pop("sector", None) or "Unknown"),
        "notes":       str(raw.pop("notes", "") or ""),
        "source":      str(raw.pop("source", None) or "holdings"),
    }
    raw.pop("added", None)
    for k in EXIT_FIELDS:
        if k in raw:
            row[k] = raw.pop(k)
    if row["strategy"] not in STRATEGIES:
        row["strategy"] = "Untagged"
    if row["shares"] == int(row["shares"]):
        row["shares"] = int(row["shares"])
    if "tranches_deployed" in raw:
        extras["tranches_deployed"] = raw.pop("tranches_deployed", 0)
    if "tranches_deployed" in extras:
        extras["tranches_deployed"] = max(0, min(3, int(_num(extras["tranches_deployed"]))))
    # allt annat gammalt följer med i extras i stället för att tappas
    for k, v in raw.items():
        if not k.startswith("_"):
            extras.setdefault(k, v)
    row["extras"] = extras
    if bucket:
        row["_bucket"] = bucket
    return row


def _default() -> dict:
    return {**{k: [] for k in BUCKET_KEYS}, "closed": [], "cash": 0}


def _normalize(data) -> dict:
    if not isinstance(data, dict):
        data = _default()
    for k in BUCKET_KEYS + ("closed",):
        rows = data.get(k)
        data[k] = [normalize_row(r) for r in rows if isinstance(r, dict)] \
            if isinstance(rows, list) else []
    data["cash"] = _num(data.get("cash", 0))
    return data


# ── Läsning ──────────────────────────────────────────────────────────────────
def load() -> dict:
    """Registret, laddat EN gång per session (repot först, annars Gisten).
    Raderna normaliseras på plats så att sessionen alltid har radschemat."""
    data = storage.session_load(STORE, _default(), legacy_file=LEGACY_FILE)
    norm = _normalize(data)
    if norm is not data:
        st.session_state[STORE] = norm
    return norm


def all_positions() -> List[dict]:
    """Alla öppna positioner, platt, med _bucket på varje rad."""
    data = load()
    out = []
    for bucket in BUCKET_KEYS:
        for r in data[bucket]:
            out.append(dict(r, _bucket=bucket))
    return out


def open_positions(strategy: Optional[str] = None,
                   bucket: Optional[str] = None) -> List[dict]:
    """Öppna positioner, valfritt per strategi och/eller hink."""
    rows = all_positions()
    if strategy is not None:
        rows = [r for r in rows if r["strategy"] == strategy]
    if bucket is not None:
        rows = [r for r in rows if r["_bucket"] == bucket]
    return rows


def find(ticker: str, bucket: Optional[str] = None) -> Optional[dict]:
    t = str(ticker or "").strip().upper()
    for r in all_positions():
        if r["ticker"] == t and (bucket is None or r["_bucket"] == bucket):
            return r
    return None


def find_id(row_id) -> Optional[dict]:
    for r in all_positions():
        if r["id"] == str(row_id):
            return r
    return None


def tickers(strategy: Optional[str] = None) -> set:
    return {r["ticker"] for r in open_positions(strategy)}


def cash() -> float:
    return _num(load().get("cash", 0))


def by_bucket() -> dict:
    """{hink: [rader]} — formen risk_dashboard och earnings_calendar tar."""
    data = load()
    return {k: list(data[k]) for k in BUCKET_KEYS}


# ── Skrivning ────────────────────────────────────────────────────────────────
def save() -> Optional[storage.SaveResult]:
    """Sparar registret direkt. Fel loggas och läggs i storage.meta() så
    sparraden visar dem — sessionen behåller ändringen oavsett."""
    try:
        return storage.save_session(STORE)
    except storage.StorageError as exc:
        logger.warning("holdings save failed: %s", exc)
        st.session_state.setdefault(storage._meta_key(STORE), {})["error"] = str(exc)
        return None


def _commit(data: dict) -> None:
    st.session_state[STORE] = data
    save()


def add(ticker: str, strategy: str = "Untagged", entry_price=0, shares=0,
        sector: str = "Unknown", name: str = "", entry_date: str = "",
        stop=None, target=None, notes: str = "", source: str = "holdings",
        extras: Optional[dict] = None, bucket: Optional[str] = None) -> tuple:
    """Lägger till en position. Returnerar (ok, meddelande) — UI:t visar
    meddelandet, registret ropar inte på Streamlit själv."""
    t = str(ticker or "").strip().upper()
    if not t:
        return False, "Ticker saknas."
    if strategy not in STRATEGIES:
        return False, f"Okänd strategi: {strategy}"
    bucket = bucket or bucket_for(strategy)
    data = load()
    rows = data[bucket]
    cap = BUCKETS[bucket]["max"]
    if len(rows) >= cap:
        return False, f"Max {cap} positioner i {BUCKETS[bucket]['name']} — ta bort en först."
    if any(r["ticker"] == t for r in rows):
        return False, f"{t} finns redan i {BUCKETS[bucket]['name']}."
    row = normalize_row({
        "ticker": t, "name": name, "strategy": strategy,
        "entry_price": entry_price, "shares": shares, "sector": sector,
        "entry_date": entry_date or _today(), "stop": stop, "target": target,
        "notes": notes, "source": source, "extras": dict(extras or {}),
    })
    rows.append(row)
    _commit(data)
    return True, f"{t} tillagd i {BUCKETS[bucket]['name']}."


def update(ticker: str, bucket: Optional[str] = None, **fields) -> bool:
    """Ändrar fält på en öppen position. Okända fält hamnar i extras.
    Byter strategin hink flyttas raden dit (om det finns plats)."""
    row = find(ticker, bucket)
    if row is None:
        return False
    data = load()
    src_bucket = row["_bucket"]
    rows = data[src_bucket]
    live = next(r for r in rows if r["id"] == row["id"])
    for k, v in fields.items():
        if k in FIELDS and k not in ("id", "extras"):
            live[k] = v
        else:
            live.setdefault("extras", {})[k] = v
    live.update(normalize_row(live))
    live.pop("_bucket", None)
    new_bucket = bucket_for(live["strategy"])
    if new_bucket != src_bucket:
        if len(data[new_bucket]) >= BUCKETS[new_bucket]["max"]:
            return False
        rows.remove(live)
        data[new_bucket].append(live)
    _commit(data)
    return True


def remove(ticker: str, bucket: Optional[str] = None) -> bool:
    row = find(ticker, bucket)
    if row is None:
        return False
    data = load()
    data[row["_bucket"]] = [r for r in data[row["_bucket"]] if r["id"] != row["id"]]
    _commit(data)
    return True


def close(ticker: str = "", exit_price=None, exit_date: str = "", reason: str = "",
          bucket: Optional[str] = None, row_id=None) -> Optional[dict]:
    """Stänger en position (på ticker, eller på row_id från arkens vy): raden
    flyttas till closed med exit-fält och resultat i procent. Returnerar den
    stängda raden (journalen får den i ett senare steg)."""
    row = find_id(row_id) if row_id is not None else find(ticker, bucket)
    if row is None:
        return None
    data = load()
    data[row["_bucket"]] = [r for r in data[row["_bucket"]] if r["id"] != row["id"]]
    closed = dict(row)
    closed.pop("_bucket", None)
    xp = _opt(exit_price)
    closed["exit_price"] = xp
    closed["exit_date"] = exit_date or _today()
    closed["exit_reason"] = reason
    closed["result_pct"] = (round((xp / closed["entry_price"] - 1) * 100, 2)
                            if xp and closed["entry_price"] > 0 else None)
    data["closed"].append(closed)
    _commit(data)
    # Journalraden — en stängd position ÄR en affär. Bäst-ansträngning:
    # journalen bor i Gisten och får inte fälla stängningen.
    try:
        import journal_bridge
        journal_bridge.record_close(closed)
    except Exception as exc:                        # pragma: no cover
        logger.warning("journalraden kunde inte skrivas: %s", exc)
    return closed


def set_cash(value) -> None:
    data = load()
    data["cash"] = _num(value)
    _commit(data)


def counts() -> dict:
    """{hink: (antal, tak)} för rubriker och grindar."""
    data = load()
    return {k: (len(data[k]), BUCKETS[k]["max"]) for k in BUCKET_KEYS}


# ── Arkens vy: samma rad i den form Swing och Tiggre alltid ritat ────────────
# Swing- och Tiggre-flikarna (och deras läsare: swing_verdict, alert_scan,
# sheets_refresh, scorecard, review_link, confidence/prefill) tar rader med
# entry/date/current/half_sold osv. Vyn är den formen; registret är källan.
# Fälten i VIEW_MAP byter namn åt båda hållen, resten bor i extras.
VIEW_MAP: dict = {"entry": "entry_price", "date": "entry_date"}


def to_view(row: dict) -> dict:
    """Registerrad → arkrad: {id, ticker, name, shares, entry, date, stop,
    target, strategy, **extras}."""
    out = {"id": row.get("id"), "ticker": row.get("ticker"), "name": row.get("name", ""),
           "shares": row.get("shares", 0), "stop": row.get("stop"),
           "target": row.get("target"), "strategy": row.get("strategy")}
    for view_key, reg_key in VIEW_MAP.items():
        out[view_key] = row.get(reg_key)
    for k, v in (row.get("extras") or {}).items():
        out.setdefault(k, v)
    return out


def from_view(view_row: dict, strategy: str, source: str) -> dict:
    """Arkrad → registerrad (normaliserad). Okända fält → extras."""
    raw = dict(view_row or {})
    reg = {"id": raw.pop("id", None), "ticker": raw.pop("ticker", ""),
           "name": raw.pop("name", ""), "shares": raw.pop("shares", 0),
           "stop": raw.pop("stop", None), "target": raw.pop("target", None),
           "strategy": strategy, "source": source,
           "sector": raw.pop("sector", None), "notes": raw.pop("notes", "")}
    for view_key, reg_key in VIEW_MAP.items():
        reg[reg_key] = raw.pop(view_key, None)
    raw.pop("strategy", None)
    raw.pop("source", None)
    extras = dict(raw.pop("extras", None) or {})
    for k, v in raw.items():
        if not k.startswith("_"):
            extras[k] = v
    reg["extras"] = extras
    return normalize_row(reg)


def view_rows_from(data, strategy: str) -> List[dict]:
    """Vyn ur en redan laddad lagring (jobben läser data/holdings.json utan
    session). Tolerant mot None/fel form."""
    if not isinstance(data, dict):
        return []
    out = []
    for bucket in BUCKET_KEYS:
        for r in data.get(bucket) or []:
            if isinstance(r, dict) and (r.get("strategy") or "Untagged") == strategy:
                out.append(to_view(normalize_row(r)))
    return out


def view_rows(strategy: str) -> List[dict]:
    """Strategins öppna positioner i arkets form, ur sessionen."""
    return [to_view(r) for r in open_positions(strategy)]


def put(strategy: str, view_row: dict, source: str) -> dict:
    """Upsert av en arkrad (matchas på id, annars ticker inom strategin).
    Inga tak här — arket håller sitt eget (Swing 8, Tiggre 6)."""
    reg = from_view(view_row, strategy, source)
    data = load()
    bucket = bucket_for(strategy)
    rows = data[bucket]
    for i, r in enumerate(rows):
        if r["id"] == reg["id"] or (not view_row.get("id") and r["ticker"] == reg["ticker"]):
            reg["id"] = r["id"]
            reg["source"] = r.get("source") or source
            rows[i] = reg
            break
    else:
        rows.append(reg)
    _commit(data)
    return to_view(reg)


def drop(strategy: str, row_id: str) -> bool:
    """Tar bort en rad på id (arkets "Stäng" utan att skriva closed —
    arket har sin egen historik med sina egna fält)."""
    data = load()
    bucket = bucket_for(strategy)
    before = len(data[bucket])
    data[bucket] = [r for r in data[bucket] if r["id"] != row_id]
    if len(data[bucket]) == before:
        return False
    _commit(data)
    return True


def migrate_rows(strategy: str, rows: List[dict], source: str) -> int:
    """Engångsflytt av ett arks egen positionslista in i registret. Rader
    med ett id som redan finns hoppas över (idempotent); returnerar antalet
    som flyttades. En sparning för hela flytten."""
    data = load()
    bucket = bucket_for(strategy)
    have = {r["id"] for r in data[bucket]}
    moved = 0
    for vr in rows or []:
        if not isinstance(vr, dict) or not vr.get("ticker"):
            continue
        reg = from_view(vr, strategy, source)
        if reg["id"] in have:
            continue
        data[bucket].append(reg)
        have.add(reg["id"])
        moved += 1
    if moved:
        _commit(data)
    return moved
