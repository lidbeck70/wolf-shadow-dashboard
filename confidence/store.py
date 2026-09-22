"""
confidence/store.py — formen på data/confidence.json (bara INMATNINGAR).

{"companies": {TICKER: CompanyInput.as_dict()},
 "commodity_overrides": {råvarunyckel: {supply_balance_pct: {...}, ...}}}

Beräknade resultat lagras inte — de räknas om ur inmatningarna. Rena
funktioner; persistensen (storage.session_load / 💾 Spara) sköts av UI:t.
"""

from __future__ import annotations

from typing import Optional

from confidence.data.models import CompanyInput

STORE = "confidence"          # data/confidence.json


def default() -> dict:
    return {"companies": {}, "commodity_overrides": {}}


def normalize(data: Optional[dict]) -> dict:
    data = data if isinstance(data, dict) else {}
    for k, v in default().items():
        if not isinstance(data.get(k), dict):
            data[k] = v
    return data


def companies(data: dict) -> dict:
    """{ticker: CompanyInput} i lagrad ordning."""
    out = {}
    for t, d in (data.get("companies") or {}).items():
        try:
            out[t] = CompanyInput.from_dict(d)
        except (TypeError, ValueError):
            continue
    return out


def get(data: dict, ticker: str) -> Optional[CompanyInput]:
    d = (data.get("companies") or {}).get(_key(ticker))
    return CompanyInput.from_dict(d) if d else None


def put(data: dict, company: CompanyInput) -> None:
    normalize(data)
    company.ticker = _key(company.ticker)
    data["companies"][company.ticker] = company.as_dict()


def remove(data: dict, ticker: str) -> None:
    (data.get("companies") or {}).pop(_key(ticker), None)


def overrides(data: dict) -> dict:
    return data.get("commodity_overrides") or {}


def set_override(data: dict, commodity_key: str, field: str, point: Optional[dict]) -> None:
    normalize(data)
    o = data["commodity_overrides"].setdefault(commodity_key, {})
    if point is None:
        o.pop(field, None)
    else:
        o[field] = point


def _key(ticker: str) -> str:
    return (ticker or "").strip().upper()
