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


def borsdata_now(company: CompanyInput, api=None) -> tuple:
    """Hämta bolagets rad från Börsdata direkt (samma rad som det nattliga
    jobbet) och översätt till förslag. Returnerar (blob, förslag, meddelande);
    blob är None när nyckel saknas eller Börsdata inte känner tickern."""
    import os
    from datetime import datetime, timezone

    import sheets_refresh as sr

    if api is None:
        try:
            from borsdata_api import BorsdataAPI
            api = BorsdataAPI()
        except Exception as exc:                          # pragma: no cover
            return None, [], f"Börsdata-nyckel saknas eller klienten kunde inte startas ({exc})."
    try:
        row = sr.fetch_row(api, company.ticker, company.ins_id)
    except Exception as exc:
        return None, [], f"Börsdata svarade inte: {exc}"
    if row is None:
        return None, [], (f"Börsdata känner inte {company.ticker}. Ange Börsdata-id (ins_id) under Identitet, "
                          "eller skriv talen själv.")
    blob = {"generated": datetime.now(tz=timezone.utc).isoformat(), "rows": {f"{sr.ref_key('confidence', {'id': company.ticker})}": row}}
    props = dr.proposals(blob, company)
    return blob, props, (f"{len(props)} tal ur Börsdata" + (f" (kurs {row['price']:g} {row.get('currency') or ''})"
                                                            if row.get("price") is not None else ""))
