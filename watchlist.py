"""
watchlist.py — 👀 Övervakningen: aktier nära köpläge, per strategi (data/watchlist.json).

Ett par = (aktien i Thors Yahoo-form, strategin ur thor.registry). Samma aktie kan bevakas under två strategier,
men varje par bara en gång. Lagret följer positions.py: laddas EN gång per session (storage.session_load), varje
ändring sparas direkt (storage.save_session) och ett misslyckat sparande hamnar i storage.meta() — aldrig tyst.

LAGRAS: bara inmatningar — aktien, strategin, när och varifrån, egna nivåer, en anteckning och en ögonblicksbild
av Thors rad när paret lades till (at_add). Läget räknas aldrig här och skrivs aldrig vid ritning: status_of läser
det ur Thors rader — samma verdikt som listan. Filen är inte krypterad (ägarens beslut): den säger bara vilka
aktier som bevakas.

Form:
    {"items": [{"id", "ticker": "BOL.ST", "name", "strategy": "berserk", "added": "YYYY-MM-DD",
                "source": "thor" | "manual" | "auto", "note", "levels": {"entry", "stop", "target"},
                "at_add": {"status", "class", "entry", "stop", "target", "asof"}, "alert": true}]}
"""

from __future__ import annotations

import uuid
from typing import Optional

STORE = "watchlist"                        # data/watchlist.json
SOURCES = ("thor", "manual", "auto")       # Thor-knappen, för hand, 👀 under Nära köp nu
LEVELS = ("entry", "stop", "target")
SNAPSHOT = ("status", "class", "entry", "stop", "target", "asof")
NOT_IN_RUN = "inte i dagens körning"


def _default() -> dict:
    return {"items": []}


def _today() -> str:
    """Dagens datum i svensk tid (servern går på UTC)."""
    from tabs.home_today import now_se
    return now_se().strftime("%Y-%m-%d")


def _yf(ticker) -> str:
    """Thors Yahoo-form: 'ekta b.st' → 'EKTA-B.ST' — samma nyckel som listans rader."""
    from thor import candidates as tc
    return tc.yf(ticker)


def _num(v) -> Optional[float]:
    """Frivilligt tal: tomt, 0 och skräp blir None."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and x > 0 else None


def _snap(raw) -> dict:
    """Ögonblicksbildens form: texterna som text, nivåerna som tal eller None."""
    snap = raw if isinstance(raw, dict) else {}
    return {k: _num(snap.get(k)) if k in LEVELS else str(snap.get(k) or "") for k in SNAPSHOT}


def _item(raw: dict) -> dict:
    """En lagrad rad → radschemat. Saknade eller trasiga fält får standardvärden; okända nycklar behålls."""
    i = dict(raw)
    i["id"] = str(i.get("id") or uuid.uuid4().hex[:10])
    i["ticker"] = _yf(i.get("ticker"))
    i["strategy"] = str(i.get("strategy") or "")
    for k in ("name", "added", "note"):
        i[k] = str(i.get(k) or "")
    i["source"] = i.get("source") if i.get("source") in SOURCES else "manual"
    lv = i.get("levels") if isinstance(i.get("levels"), dict) else {}
    i["levels"] = {k: _num(lv.get(k)) for k in LEVELS}
    i["at_add"] = _snap(i.get("at_add"))
    i["alert"] = bool(i.get("alert", True))
    return i


def normalize(data) -> dict:
    """Lagrets form oavsett vad som lästes. Rader utan aktie eller strategi faller bort, ett par som står
    två gånger behålls en gång (den första), och ett id som redan används får ett nytt (knapparnas nycklar och
    remove/set_levels går på id)."""
    if not isinstance(data, dict):
        data = _default()
    raw = data.get("items") if isinstance(data.get("items"), list) else []
    seen, ids, items = set(), set(), []
    for r in raw:
        if not isinstance(r, dict) or not _yf(r.get("ticker")) or not r.get("strategy"):
            continue
        i = _item(r)
        if (i["ticker"], i["strategy"]) in seen:
            continue
        while i["id"] in ids:                                 # handskriven eller hopslagen fil: samma id två gånger
            i["id"] = uuid.uuid4().hex[:10]
        seen.add((i["ticker"], i["strategy"]))
        ids.add(i["id"])
        items.append(i)
    data["items"] = items
    return data


# ── Läsning ─────────────────────────────────────────────────────────────────
def load() -> dict:
    """Listan, laddad EN gång per session och normaliserad på plats (sessionen äger den sedan). Ändrade bara
    normaliseringen en ren lista (id, versaler, standardfält) räknas den fortfarande som sparad — sparraden ska
    inte säga 'osparade ändringar' när du inte ändrat något."""
    import streamlit as st
    import storage
    data = storage.session_load(STORE, _default())
    clean = not storage.is_dirty(STORE)
    norm = normalize(data)
    if norm is not data:
        st.session_state[STORE] = norm
    if clean and storage.is_dirty(STORE):
        st.session_state[storage._clean_key(STORE)] = storage._fingerprint(norm)
    return norm


def find(ticker, strategy: str, data: Optional[dict] = None) -> Optional[dict]:
    """Raden för paret (aktie, strategi), eller None."""
    t = _yf(ticker)
    data = load() if data is None else data
    return next((i for i in data.get("items") or [] if i["ticker"] == t and i["strategy"] == strategy), None)


def keys(data: Optional[dict] = None) -> set:
    """{(ticker, strategi)} som bevakas — för 👀-knapparna under Nära köp nu."""
    data = load() if data is None else data
    return {(i["ticker"], i["strategy"]) for i in data.get("items") or []}


def snapshot(c) -> dict:
    """Ögonblicksbilden vid tillägget ur Thors rad — tom när aktien saknar rad för strategin (c = None)."""
    return _snap({"status": getattr(c, "status", ""), "class": getattr(c, "status_class", ""),
                  "entry": getattr(c, "entry", None), "stop": getattr(c, "stop", None),
                  "target": getattr(c, "target", None), "asof": getattr(c, "asof", "")})


# ── Skrivning ───────────────────────────────────────────────────────────────
def _save() -> Optional[str]:
    """Sparar listan direkt. None = sparat; annars felet (det läggs också i storage.meta, så sparraden visar det).
    Misslyckades läsningen vägrar storage att spara — då skrivs inget över i GitHub."""
    import streamlit as st
    import storage
    try:
        storage.save_session(STORE)
        return None
    except storage.StorageError as exc:
        st.session_state.setdefault(storage._meta_key(STORE), {})["error"] = str(exc)
        return str(exc)


def _done(text: str, err: Optional[str]) -> str:
    from positions import UNSAVED
    return f"{text}, {UNSAVED}: {err}" if err else f"{text}."


def add(ticker, strategy: str, name: str = "", source: str = "thor", at_add: Optional[dict] = None,
        levels: Optional[dict] = None, note: str = "") -> tuple:
    """Lägger paret i Övervakningen. (ok, meddelande) — ett par bevakas en gång. ok även när sparningen
    misslyckades (raden finns i sessionen); texten säger det då, som positions.add."""
    from thor import registry as reg
    t = _yf(ticker)
    if not t:
        return False, "Ticker saknas."
    if reg.get(strategy) is None:
        return False, f"Okänd strategi: {strategy}"
    data = load()
    hit = find(t, strategy, data)
    if hit:
        return False, f"{t} bevakas redan med {reg.label(strategy)} (sedan {hit['added'] or 'tidigare'})."
    data["items"].append(_item({"ticker": t, "name": name, "strategy": strategy, "added": _today(),
                                "source": source, "note": note, "levels": levels or {}, "at_add": at_add or {},
                                "alert": True}))
    return True, _done(f"{t} lagd i Övervakning ({reg.label(strategy)})", _save())


def remove(item_id: str) -> tuple:
    """Tar bort raden. (ok, meddelande)."""
    data = load()
    hit = next((i for i in data["items"] if i["id"] == str(item_id)), None)
    if hit is None:
        return False, "Raden finns inte längre."
    data["items"] = [i for i in data["items"] if i is not hit]
    return True, _done(f"{hit['ticker']} borttagen ur Övervakning", _save())


def set_levels(item_id: str, entry=None, stop=None, target=None, note: Optional[str] = None) -> tuple:
    """Egna nivåer (och anteckningen) på en rad — dina inmatningar, inte strategins. (ok, meddelande)."""
    data = load()
    hit = next((i for i in data["items"] if i["id"] == str(item_id)), None)
    if hit is None:
        return False, "Raden finns inte längre."
    hit["levels"] = {"entry": _num(entry), "stop": _num(stop), "target": _num(target)}
    if note is not None:
        hit["note"] = str(note)
    return True, _done(f"{hit['ticker']}: egna nivåer sparade", _save())


# ── Läget ur Thors rader (rena funktioner) ──────────────────────────────────
def status_of(item: dict, rows: list) -> dict:
    """Dagens rad för paret ur Thors lista — eller 'inte i dagens körning'. Entry, stopp och mål kommer ur
    strategins rad; saknas de där gäller dina egna nivåer (own = True)."""
    from thor import dossier as td
    t = _yf(item.get("ticker"))
    cands = td.candidates_for(rows, t)
    exact = [c for c in cands if str(c.ticker).upper() == t]           # 'NEM' före 'NEM.TO'
    c = next((x for x in (exact or cands) if x.strategy == item.get("strategy")), None)
    own = item.get("levels") if isinstance(item.get("levels"), dict) else {}
    lv = {k: getattr(c, k, None) for k in LEVELS}
    from_own = all(v is None for v in lv.values()) and any(own.get(k) is not None for k in LEVELS)
    if from_own:
        lv = {k: own.get(k) for k in LEVELS}
    base = {"candidate": c, "own": from_own, **lv}
    if c is None:
        return {**base, "in_list": False, "status_class": "", "status": NOT_IN_RUN, "score_text": "", "asof": "",
                "stale": False, "why": ""}
    return {**base, "in_list": True, "status_class": c.status_class, "status": c.status, "score_text": c.score_text,
            "asof": c.asof, "stale": bool(c.stale), "why": c.why}


def distance_pct(price, level) -> Optional[float]:
    """Kursen mot en nivå i procent (+ = över nivån). None när något saknas."""
    p, lv = _num(price), _num(level)
    return None if p is None or lv is None else (p / lv - 1.0) * 100.0
