"""
asymmetry/store.py — formen på data/asymmetry.json: Wolf Asymmetrys eget ark.

{"companies": {TICKER: CompanyInput.as_dict()},
 "commodity_overrides": {råvarunyckel: {...}},
 "strategies": {TICKER: "Viking"}}          # vilken strategi bolaget kompletterar

Samma bolagsform som confidence.store så alla motorer (asymmetry,
confidence-score, scenarier) fungerar oförändrade — men ett eget lager,
oberoende av Durrett-arket. Rena funktioner; persistensen sköts av UI:t.
"""

from __future__ import annotations

from typing import Optional

from confidence import store as cs
from confidence.data.models import CompanyInput

STORE = "asymmetry"           # data/asymmetry.json
NO_STRATEGY = "—"


def strategy_tags() -> tuple:
    """Strategierna ur registret (positions.STRATEGIES) plus 'ingen'."""
    from positions import STRATEGIES
    return (NO_STRATEGY,) + tuple(s for s in STRATEGIES if s != "Untagged")


def default() -> dict:
    d = cs.default()
    d["strategies"] = {}
    return d


def normalize(data: Optional[dict]) -> dict:
    data = cs.normalize(data)
    if not isinstance(data.get("strategies"), dict):
        data["strategies"] = {}
    return data


companies = cs.companies
get = cs.get
overrides = cs.overrides
set_override = cs.set_override


def put(data: dict, company: CompanyInput, strategy: Optional[str] = None) -> None:
    normalize(data)
    cs.put(data, company)
    if strategy is not None:
        set_strategy(data, company.ticker, strategy)


def remove(data: dict, ticker: str) -> None:
    cs.remove(data, ticker)
    (data.get("strategies") or {}).pop(cs._key(ticker), None)


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
    out = list(companies(data))
    if strategy in (None, "Alla"):
        return out
    return [t for t in out if globals()["strategy"](data, t) == strategy]
