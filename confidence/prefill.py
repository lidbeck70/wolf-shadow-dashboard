"""
confidence/prefill.py — förslag ur det som redan finns i panelen.

Granskningsarken (Rick Rule/Royalty C, Tiggre) och kontrollerna (AQS, DS)
innehåller bedömningar som Confidence score annars skulle fråga om igen.
Här översätts de till Datapoints med källa "Granskningsarket …" och kind
ESTIMATE — som FÖRSLAG. Inget skrivs in utan att användaren trycker Använd.
Rena funktioner på de sparade dict:arna (data/producers.json, data/tiggre.json).
"""

from __future__ import annotations

from typing import Optional

from confidence.data.models import CompanyInput
from confidence.data.provenance import Datapoint, dp

_SRC_AQS = "Granskningsarket Rick Rule — AQS (controls)"
_SRC_DS = "Granskningsarket Rick Rule — DS (controls)"
_SRC_RULE = "Granskningsarket Rick Rule"
_SRC_TIGGRE = "Granskningsarket Tiggre"

# AQS 0/1/2 → Resource & Project Quality-delpoäng (config.RESOURCE_SUB max)
_AQS_MAP = {
    "aqs_kostnad": ("res_grade", {0: 0, 1: 1, 2: 2}),
    "aqs_livslangd": ("res_mine_life", {0: 0, 1: 0, 2: 1}),
    "aqs_metallurgi": ("res_metallurgy", {0: 0, 1: 0, 2: 1}),
    "aqs_infrastruktur": ("res_infrastructure", {0: 0, 1: 2, 2: 4}),
    "aqs_expansion": ("res_expansion", {0: 0, 1: 1, 2: 3}),
    "aqs_management": ("mgmt_track_record", {0: 0, 1: 0, 2: 1}),
}


def _num(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _find(rows: list, ticker: str) -> Optional[dict]:
    t = (ticker or "").strip().upper()
    for r in rows or []:
        if str(r.get("ticker") or "").strip().upper() == t:
            return r
    return None


def from_producers(company: CompanyInput, producers_data: Optional[dict]) -> dict:
    """Rick Rule-/Royalty C-raden med samma ticker → {fältnyckel: Datapoint}."""
    out: dict = {}
    if not producers_data:
        return out
    row = _find(producers_data.get("producers") or [], company.ticker) or \
        _find(producers_data.get("royalty") or [], company.ticker)
    if not row:
        return out
    when = str(row.get("date") or "") or None
    for akey, (fkey, table) in _AQS_MAP.items():
        v = _num(row.get(akey))
        if v is not None:
            out[fkey] = dp(table.get(int(v), 0), kind="ESTIMATE", source=_SRC_AQS, source_type="mixed",
                           pub_date=when, note=f"{akey} = {int(v)}")
    try:
        import controls as ctl
        ds = ctl.ds_total(row)
    except Exception:                                   # pragma: no cover
        ds = None
    if ds is not None:
        out["dilution_score"] = dp(ds, kind="ESTIMATE", source=_SRC_DS, source_type="mixed", pub_date=when)
    for rkey, fkey, unit in (("unit_cost", "aisc", "USD/enhet"), ("mine_life", "mine_life_years", "år"),
                             ("price", "commodity_price", "USD/enhet")):
        v = _num(row.get(rkey))
        if v is not None:
            out[fkey] = dp(v, kind="ESTIMATE", source=_SRC_RULE, source_type="mixed", pub_date=when, unit=unit)
    return out


def from_tiggre(company: CompanyInput, tiggre_data: Optional[dict]) -> dict:
    """Tiggre-kandidaten/positionen med samma ticker → {fältnyckel: Datapoint}
    plus "maturity" som textförslag när FS är klar."""
    out: dict = {}
    if not tiggre_data:
        return out
    cand = _find(tiggre_data.get("candidates") or [], company.ticker) or \
        _find(tiggre_data.get("positions") or [], company.ticker)
    if not cand:
        return out
    nav = _num(cand.get("nav"))
    if nav is not None:
        out["nav_musd"] = dp(nav, kind="ESTIMATE", source=_SRC_TIGGRE, source_type="mixed", unit="MUSD")
        if company.stage in ("explorer", "developer"):
            out["npv_musd"] = dp(nav, kind="ESTIMATE", source=_SRC_TIGGRE, source_type="mixed", unit="MUSD",
                                 note="NAV = NPV efter skatt i Tiggre-arket")
    mcap = _num(cand.get("mcap"))
    if mcap is not None:
        out["market_cap_musd"] = dp(mcap, kind="ESTIMATE", source=_SRC_TIGGRE, source_type="mixed", unit="MUSD")
    screen = cand.get("screen") or {}
    for skey, fkey in (("permits", "permits_granted"), ("funded", "financing_committed")):
        if skey in screen and screen.get(skey) is not None:
            out[fkey] = dp(bool(screen.get(skey)), kind="ESTIMATE", source=_SRC_TIGGRE, source_type="mixed")
    if screen.get("fs"):
        out["maturity"] = "dfs"
    return out


def proposals(company: CompanyInput, producers_data: Optional[dict] = None,
              tiggre_data: Optional[dict] = None) -> list:
    """[(fältnyckel, Datapoint|str, redan_satt)] — bara sådant som skiljer sig från bolaget."""
    merged: dict = {}
    merged.update(from_producers(company, producers_data))
    merged.update(from_tiggre(company, tiggre_data))
    out = []
    for key, point in merged.items():
        if key == "maturity":
            if company.maturity != point:
                out.append((key, point, False))
            continue
        current = company.get(key)
        same = current is not None and not current.missing and _same(current, point)
        if not same:
            out.append((key, point, current is not None and not current.missing))
    return out


def _same(a: Datapoint, b: Datapoint) -> bool:
    try:
        return float(a.value) == float(b.value)
    except (TypeError, ValueError):
        return str(a.value) == str(b.value)
