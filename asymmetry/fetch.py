"""
asymmetry/fetch.py — fyll arket utan handskrift. Rena funktioner.

  från registret   öppna positioner (positions.py) → bolagsskal med ticker,
                   namn och strategi-tagg
  ur Börsdata      sifferuppdateringens rad 'confidence:<TICKER>' → förslag
                   (engines/durrett/refresh, samma ark)
"""

from __future__ import annotations

from typing import Optional

from confidence import store as cs
from confidence.data.models import CompanyInput
from engines.durrett import refresh as dr

from asymmetry import store as ast


# ── registret ────────────────────────────────────────────────────────────────
def register_candidates(rows: list, data: dict) -> list:
    """[(ticker, namn, strategi)] ur registrets rader som inte redan finns i arket.
    En ticker en gång, första raden vinner."""
    have = set(cs.companies(data))
    seen, out = set(), []
    for r in rows or []:
        t = str(r.get("ticker") or "").strip().upper()
        if not t or t in have or t in seen:
            continue
        seen.add(t)
        strat = str(r.get("strategy") or "")
        out.append((t, str(r.get("name") or ""), strat if strat in ast.strategy_tags() else ast.NO_STRATEGY))
    return out


def add_from_register(data: dict, ticker: str, name: str, strategy: str) -> CompanyInput:
    """Skal: identitet fylls i Ark (råvara, stage, land). Inga tal gissas."""
    c = CompanyInput(ticker=ticker.strip().upper(), name=name)
    cs.put(data, c)
    ast.set_strategy(data, c.ticker, strategy)
    return c


# ── Börsdata (sifferuppdateringen) ───────────────────────────────────────────
def refresh_row(blob: Optional[dict], ticker: str) -> Optional[dict]:
    return dr.refresh_row(blob, ticker)


def refresh_proposals(blob: Optional[dict], company: CompanyInput) -> list:
    """[(fältnyckel, Datapoint, nuvarande värde | None)] — bara avvikande tal."""
    return dr.proposals(blob, company)
