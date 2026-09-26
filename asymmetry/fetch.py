"""
asymmetry/fetch.py — fyll Wolf Asymmetrys ark utan handskrift. Rena funktioner.

  från registret      öppna positioner (positions.py) → bolagsskal med ticker,
                      namn och strategi-tagg
  från Durrett-arket  kopiera ett bolag ur data/confidence.json (oberoende kopia)
  ur Börsdata         sifferuppdateringens rad 'asymmetry:<TICKER>' → förslag,
                      samma översättning som Durrett-arket (engines/durrett/refresh)
"""

from __future__ import annotations

from typing import Optional

from confidence import store as cs
from confidence.data.models import CompanyInput
from engines.durrett import refresh as dr

from asymmetry import store as ast

SHEET = ast.STORE                     # nyckeln i sheets_refresh.json: "asymmetry:<TICKER>"


# ── registret ────────────────────────────────────────────────────────────────
def register_candidates(rows: list, data: dict) -> list:
    """[(ticker, namn, strategi)] ur registrets rader som inte redan finns i arket.
    En ticker en gång, första raden vinner."""
    have = set(ast.companies(data))
    seen, out = set(), []
    for r in rows or []:
        t = str(r.get("ticker") or "").strip().upper()
        if not t or t in have or t in seen:
            continue
        seen.add(t)
        strat = str(r.get("strategy") or "")
        out.append((t, str(r.get("name") or ""), strat if strat in ast.strategy_tags() else ast.NO_STRATEGY))
    return out


def company_from_register(ticker: str, name: str) -> CompanyInput:
    """Skal: identitet fylls i Ark (råvara, stage, land). Inga tal gissas."""
    return CompanyInput(ticker=ticker.strip().upper(), name=name)


def add_from_register(data: dict, ticker: str, name: str, strategy: str) -> CompanyInput:
    c = company_from_register(ticker, name)
    ast.put(data, c, strategy)
    return c


# ── Durrett-arket ────────────────────────────────────────────────────────────
def durrett_candidates(conf: Optional[dict], data: dict) -> list:
    """[(ticker, namn, stage)] ur Durrett-arket som inte finns i Wolf Asymmetry."""
    have = set(ast.companies(data))
    return [(t, c.name, c.stage) for t, c in cs.companies(conf or {}).items() if t not in have]


def copy_from_durrett(conf: dict, data: dict, ticker: str, strategy: str = "Durrett") -> Optional[CompanyInput]:
    """Oberoende kopia: ändringar i det ena arket rör inte det andra."""
    c = cs.get(conf, ticker)
    if c is None:
        return None
    copy = CompanyInput.from_dict(c.as_dict())
    ast.put(data, copy, strategy)
    return copy


# ── Börsdata (sifferuppdateringen) ───────────────────────────────────────────
def refresh_row(blob: Optional[dict], ticker: str) -> Optional[dict]:
    return dr.refresh_row(blob, ticker, SHEET)


def refresh_proposals(blob: Optional[dict], company: CompanyInput) -> list:
    """[(fältnyckel, Datapoint, nuvarande värde | None)] — bara avvikande tal."""
    return dr.proposals(blob, company, SHEET)
