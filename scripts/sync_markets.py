#!/usr/bin/env python3
"""
sync_markets.py — hämta Börsdatas /markets och /countries och skriv
config/markets.json, tabellen markets.py läser före sin inbyggda FALLBACK.

    BORSDATA_API_KEY=... python scripts/sync_markets.py [--print]

Kör den en gång med din nyckel så är tvisten mellan de gamla handskrivna
kopiorna avgjord av Börsdata själva. --print visar tabellen utan att skriva.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import markets  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", dest="show")
    parser.add_argument("--out", default=markets.CONFIG_PATH)
    args = parser.parse_args(argv)

    key = os.environ.get("BORSDATA_API_KEY", "") or os.environ.get("BD_API_KEY", "")
    if not key:
        print("BORSDATA_API_KEY saknas.", file=sys.stderr)
        return 1
    from borsdata_api import BorsdataAPI
    api = BorsdataAPI(api_key=key)
    raw_markets, raw_countries = api.get_markets(), api.get_countries()
    table = markets.from_api(raw_markets, raw_countries)
    if not table:
        print("Tomt svar från /markets.", file=sys.stderr)
        return 1

    for m in sorted(table.values(), key=lambda m: m.id):
        print(f"{m.id:>3}  {m.country:<14} {m.name:<28} {m.exchange:<20} "
              f"{m.suffix or '(inget)':<8} {m.kind}")
    diff = [m.id for m in table.values()
            if m.id in markets.FALLBACK and markets.FALLBACK[m.id].suffix != m.suffix]
    if diff:
        print(f"\nSuffix skiljer sig från FALLBACK för id: {sorted(diff)}")
    if args.show:
        return 0
    payload = {"generated": __import__("datetime").datetime.utcnow().isoformat(),
               "markets": markets.serialize(table),
               "countries": [{"id": c.get("id"), "name": c.get("name")} for c in raw_countries]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"\nSkrev {len(table)} marknader till {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
