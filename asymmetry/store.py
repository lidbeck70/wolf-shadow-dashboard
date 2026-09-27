"""
asymmetry/store.py — Wolf Asymmetry läser och skriver SAMMA ark som Durrett
(data/confidence.json). Det enda som är eget är strategi-taggen per bolag:

{"companies": {...}, "commodity_overrides": {...},
 "strategies": {TICKER: "Viking"}}          # vilken strategi bolaget kompletterar

Ett bolag, ett register, två läsningar (Durretts 10 steg, Wolf Asymmetry).
Rena funktioner; persistensen sköts av UI:t (💾 Spara).
"""

from __future__ import annotations

from typing import Optional

from confidence import store as cs

STORE = cs.STORE              # data/confidence.json — gemensamt
LEGACY_STORE = "asymmetry"    # data/asymmetry.json från steg D–F; flyttas in en gång
NO_STRATEGY = "—"


def strategy_tags() -> tuple:
    """Strategierna ur registret (positions.STRATEGIES) plus 'ingen'."""
    from positions import STRATEGIES
    return (NO_STRATEGY,) + tuple(s for s in STRATEGIES if s != "Untagged")


def normalize(data: Optional[dict]) -> dict:
    data = cs.normalize(data)
    if not isinstance(data.get("strategies"), dict):
        data["strategies"] = {}
    return data


def strategy(data: dict, ticker: str) -> str:
    return (data.get("strategies") or {}).get(cs._key(ticker)) or NO_STRATEGY


def set_strategy(data: dict, ticker: str, strategy: str) -> None:
    normalize(data)
    if strategy and strategy != NO_STRATEGY:
        data["strategies"][cs._key(ticker)] = strategy
    else:
        data["strategies"].pop(cs._key(ticker), None)


def tickers_for(data: dict, strategy: Optional[str] = None) -> list:
    """Tickers i lagrad ordning, filtrerade på strategi (None/'Alla' = alla)."""
    out = list(cs.companies(data))
    if strategy in (None, "Alla"):
        return out
    return [t for t in out if globals()["strategy"](data, t) == strategy]


def merge_legacy(data: dict, legacy: Optional[dict]) -> list:
    """Engångsflytt: bolag och taggar ur det gamla data/asymmetry.json in i
    det gemensamma arket. Bolag som redan finns rörs inte. Returnerar
    tickers som flyttades."""
    normalize(data)
    if not isinstance(legacy, dict):
        return []
    moved = []
    for t, d in (legacy.get("companies") or {}).items():
        k = cs._key(t)
        if k in data["companies"] or not isinstance(d, dict):
            continue
        data["companies"][k] = d
        moved.append(k)
    for t, s in (legacy.get("strategies") or {}).items():
        k = cs._key(t)
        if k in data["companies"] and k not in data["strategies"] and s:
            data["strategies"][k] = s
    for ck, ov in (legacy.get("commodity_overrides") or {}).items():
        data["commodity_overrides"].setdefault(ck, ov)
    return moved
