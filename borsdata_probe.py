#!/usr/bin/env python3
"""
borsdata_probe.py — engångsdiagnostik mot Börsdata (körs i Actions där nyckeln finns).

Skriver ut det som behövs för att rätta Contrarian Alpha-screenern på fakta
i stället för gissningar:
  1. Börsdatas sektor- och branschlista (id → namn) + antal nordiska bolag
     per bransch — underlaget för necessity-namnkartan.
  2. KPI-metadata (id → namn) för de KPI:er motorn använder — bekräftar att
     rätt id går till rätt fält.
  3. Råa screenervärden för referensbolag (G5EN, QAIR, BOL, EQNR) — så
     enheterna (procent vs kvot, MSEK vs per aktie) kan fastställas mot
     kända siffror.

Skriver ingenting, ändrar ingenting. Exit 0 alltid.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

REF_TICKERS = ["G5EN", "QAIR", "BOL", "EQNR"]
PROBE_KPIS = {
    "ebitda_margin": 32, "operating_margin": 29, "gross_margin": 28,
    "fcf_margin": 31, "roic": 37, "roe": 33, "debt_to_equity": 40,
    "equity_ratio": 39, "net_debt_ebitda": 42, "ev_ebitda": 11, "pe": 2,
    "p_fcf": 76, "fcf_m": 63, "revenue_m": 53, "ebitda_m": 54,
    "total_equity_m": 58, "total_assets_m": 57, "market_cap": 50,
    "net_debt_m": 60, "short_selling": 207,
}


def main() -> int:
    if not os.environ.get("BORSDATA_API_KEY"):
        print("BORSDATA_API_KEY saknas — inget att göra.")
        return 0
    from borsdata_api import BorsdataAPI, ALL_NORDIC_MARKETS
    api = BorsdataAPI()

    print("=" * 72)
    print("SEKTORER (id → namn)")
    print("=" * 72)
    sectors = {s["id"]: s.get("name", "") for s in api.get_sectors()}
    for sid, name in sorted(sectors.items()):
        print(f"  {sid:>4}  {name}")

    instruments = api.get_instruments()
    nordic = [i for i in instruments
              if i.get("marketId") in ALL_NORDIC_MARKETS
              and i.get("instrumentType", 1) in (1, None)]
    per_branch = Counter(i.get("branchId") for i in nordic)
    branch_sector: dict = {}
    for i in nordic:
        branch_sector.setdefault(i.get("branchId"), i.get("sectorId"))

    print("\n" + "=" * 72)
    print(f"BRANSCHER (id → namn · sektor · antal nordiska bolag av {len(nordic)})")
    print("=" * 72)
    for b in sorted(api.get_branches(), key=lambda x: x["id"]):
        bid = b["id"]
        sname = sectors.get(b.get("sectorId", branch_sector.get(bid)), "?")
        print(f"  {bid:>4}  {b.get('name', ''):40} sektor={sname:22} n={per_branch.get(bid, 0)}")

    print("\n" + "=" * 72)
    print("KPI-METADATA (id → namn) för motorns KPI:er")
    print("=" * 72)
    try:
        meta = api._get("/instruments/kpis/metadata")
        by_id = {m.get("kpiId"): m for m in meta.get("kpiHistoryMetadatas", [])}
        for key, kid in PROBE_KPIS.items():
            m = by_id.get(kid, {})
            print(f"  {kid:>4}  {key:18} → {m.get('nameSv') or m.get('nameEn', '?')}"
                  f"  [{m.get('format', '')}]")
    except Exception as e:
        print(f"  metadata misslyckades: {e}")

    print("\n" + "=" * 72)
    print("REFERENSBOLAG — råa screenervärden (last/latest)")
    print("=" * 72)
    refs = {}
    for i in instruments:
        t = str(i.get("ticker", "")).upper()
        if t in REF_TICKERS:
            refs[t] = i
    for t, i in refs.items():
        print(f"\n  {t}: {i.get('name')}  insId={i.get('insId')}  "
              f"sectorId={i.get('sectorId')} ({sectors.get(i.get('sectorId'), '?')})  "
              f"branchId={i.get('branchId')}  marketId={i.get('marketId')}")
    ids = {i.get("insId"): t for t, i in refs.items()}
    for key, kid in PROBE_KPIS.items():
        try:
            vals = api.get_kpi_screener(kid, "last", "latest")
        except Exception as e:
            print(f"  {key:18} (id {kid}): FEL {e}")
            continue
        row = {ids[v.get('i')]: v.get("n") for v in vals if v.get("i") in ids}
        cells = "  ".join(f"{t}={row.get(t)}" for t in REF_TICKERS)
        print(f"  {key:18} (id {kid:>3}): {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
