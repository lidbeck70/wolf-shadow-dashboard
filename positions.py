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
BUCKETS: dict = {
    "swing":  {"name": "Wolf Portfolio",   "max": 5},
    "ovtlyr": {"name": "Viking Portfolio", "max": 5},
    "long":   {"name": "Alpha Portfolio",  "max": 20},
}
BUCKET_KEYS: tuple = tuple(BUCKETS)

STRATEGIES: tuple = ("Quality", "Deep Contrarian", "Viking", "Wolf", "Untagged")
STRATEGY_BUCKET: dict = {
    "Quality":         "long",
    "Deep Contrarian": "long",
    "Viking":          "ovtlyr",
    "Wolf":            "swing",
    "Untagged":        "long",
}

FIELDS: tuple = ("id", "ticker", "name", "strategy", "entry_price", "shares",
                 "entry_date", "stop", "target", "sector", "notes", "source", "extras")


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


def close(ticker: str, exit_price=None, exit_date: str = "", reason: str = "",
          bucket: Optional[str] = None) -> Optional[dict]:
    """Stänger en position: raden flyttas till closed med exit-fält och
    resultat i procent. Returnerar den stängda raden (journalen får den i
    ett senare steg)."""
    row = find(ticker, bucket)
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
    return closed


def set_cash(value) -> None:
    data = load()
    data["cash"] = _num(value)
    _commit(data)


def counts() -> dict:
    """{hink: (antal, tak)} för rubriker och grindar."""
    data = load()
    return {k: (len(data[k]), BUCKETS[k]["max"]) for k in BUCKET_KEYS}
