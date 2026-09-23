"""
engines/durrett/refresh.py — sifferuppdateringens förslag för Durrett-arket.

sheets_refresh.py (GitHub Actions) läser data/confidence.json, hämtar
färska tal ur Börsdata och skriver dem till Gisten (sheets_refresh.json)
under nyckeln "confidence:<TICKER>". Här översätts en sådan rad till
förslag {fältnyckel: Datapoint} — bara sådant som skiljer sig från det
bolaget redan har. Inget skrivs utan Använd. Rena funktioner.
"""

from __future__ import annotations

from typing import Optional

from confidence.data.models import CompanyInput
from confidence.data.provenance import Datapoint, dp

SHEET = "confidence"
# (fält i bloben, fält i bolaget, enhet, kind)
_MAP = (("mcap_musd", "market_cap_musd", "MUSD", "ACTUAL"), ("price", "share_price", "valuta/aktie", "ACTUAL"),
        ("ev_musd", "enterprise_value_musd", "MUSD", "ACTUAL"), ("ev_ebitda", "ev_ebitda", "×", "ACTUAL"),
        ("nd_ebitda", "net_debt_ebitda", "×", "ACTUAL"), ("pe", "pe", "×", "ACTUAL"),
        ("revenue_musd", "revenue_musd", "MUSD", "ACTUAL"), ("fcf_musd", "free_cash_flow_musd", "MUSD", "ACTUAL"),
        ("ocf_musd", "operating_cash_flow_musd", "MUSD", "ACTUAL"), ("rs_rank", "rs_rank", "", "ACTUAL"),
        ("ebitda_margin", "ebitda_margin_pct", "%", "ACTUAL"), ("fx_to_usd", "fx_to_usd", "USD per enhet", "ASSUMPTION"),
        # rapportfälten (PR 15): kassa, skuld, aktiehistorik, ROIC, FCF-yield, EV/EBIT
        ("cash_musd", "cash_musd", "MUSD", "ACTUAL"), ("debt_musd", "debt_musd", "MUSD", "ESTIMATE"),
        ("shares_now_m", "basic_shares_m", "M", "ACTUAL"), ("shares_1y_ago_m", "shares_1y_ago_m", "M", "ACTUAL"),
        ("shares_3y_ago_m", "shares_3y_ago_m", "M", "ACTUAL"), ("shares_5y_ago_m", "shares_5y_ago_m", "M", "ACTUAL"),
        ("roic_pct", "roic_pct", "%", "ACTUAL"), ("fcf_yield_pct", "fcf_yield_pct", "%", "ACTUAL"),
        ("ev_ebit", "ev_ebit", "×", "ACTUAL"))
_NOTES = {"debt_musd": "bruttoskuld ≈ nettoskuld + kassa ur senaste rapporten",
          "shares_now_m": "numberOfShares ur senaste årsrapporten (miljoner)"}
_PCT_FIELDS = {"ebitda_margin": 100.0}          # snapshoten ger decimaltal → %


def refresh_row(blob: Optional[dict], ticker: str) -> Optional[dict]:
    return ((blob or {}).get("rows") or {}).get(f"{SHEET}:{(ticker or '').strip().upper()}")


def proposals(blob: Optional[dict], company: CompanyInput) -> list:
    """[(fältnyckel, Datapoint, nuvarande värde | None)] — bara avvikande tal."""
    s = refresh_row(blob, company.ticker)
    if not s:
        return []
    asof = str(s.get("asof") or (blob or {}).get("generated") or "")[:10] or None
    src = f"Börsdata (sifferuppdatering {asof or '?'})"
    out = []
    ccy = str(s.get("currency") or "").upper()
    for bkey, fkey, unit, kind in _MAP:
        v = s.get(bkey)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if bkey in _PCT_FIELDS:
            v = v * _PCT_FIELDS[bkey]
        cur = company.num(fkey) if company.has(fkey) else None
        if cur is not None and abs(cur - v) < 1e-6:
            continue
        note = _NOTES.get(bkey, "")
        source = src
        if bkey == "fx_to_usd":
            source = str(s.get("fx_table") or "sheets_refresh FX-tabell")
            note = f"{ccy} → USD ur jobbets fasta tabell — kontrollera mot dagskurs"
        elif bkey == "price":
            unit = f"{ccy}/aktie" if ccy else unit
        out.append((fkey, dp(round(v, 4), kind=kind, source=source, source_type="secondary", pub_date=asof,
                             unit=unit, note=note), cur))
    if ccy and (company.get("market_currency") is None or str(company.get("market_currency").value) != ccy) \
            and ccy in ("USD", "CAD", "AUD", "GBP", "EUR", "SEK", "NOK"):
        out.append(("market_currency", dp(ccy, kind="ACTUAL", source=src, source_type="secondary", pub_date=asof),
                    str(company.get("market_currency").value) if company.get("market_currency") else None))
    return out


__all__ = ["proposals", "refresh_row", "Datapoint"]
