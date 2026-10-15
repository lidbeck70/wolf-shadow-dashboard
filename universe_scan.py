#!/usr/bin/env python3
"""
universe_scan.py — bygger universum-registret (universe/registry.py) och sparar det i Gisten.

  1. Börsdata /instruments (Norden) och /instruments/global (USA, Kanada, London, Australien — Pro+ global;
     utan licens blir listan tom och de marknaderna kommer bara ur panelens listor).
  2. Panelens egna listor: BERSERK, ticker_universe, Contrarian Alphas US/CA-CSV, Viking Nine-backtestet,
     Wolf-screenern, heatmapen, EMBER, Long Screener.
  3. universe.json (index) och universe_<marknad>.json (medlemmarna) i Gisten.

Loggen visar antal per marknad och källa, och Börsdatas instrumentType-fördelning per lista (0 eller 1 = aktie?
— panelens moduler säger olika; här sparas fältet som det är). Körs måndagar och manuellt (universe-scan.yml).
Env: BORSDATA_API_KEY, GITHUB_TOKEN. Ändrar inga skanningar.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter

from universe import registry as ur

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("universe_scan")


def run(api=None, save=None, no_save: bool = False, nordic=None, global_=None, table=None) -> dict:
    if api is None and nordic is None:
        from borsdata_api import get_api
        api = get_api()
        if not getattr(api, "is_configured", False):
            log.warning("BORSDATA_API_KEY saknas — registret byggs bara ur panelens listor.")
            api = None
    if api is not None and nordic is None:
        try:
            nordic = api.get_instruments() or []
        except Exception as exc:
            log.warning("Börsdata /instruments: %s", exc)
            nordic = []
        try:
            global_ = api.get_global_instruments_list() or []
        except Exception as exc:
            log.warning("Börsdata /instruments/global: %s", exc)
            global_ = []
        for name, lst in (("nordisk", nordic), ("global", global_)):
            types = Counter(i.get("instrumentType") for i in lst if isinstance(i, dict))
            log.info("Börsdata %s: %d instrument · instrumentType %s", name, len(lst), dict(types.most_common()))
    u = ur.build(api=api, table=table, nordic=nordic or [], global_=global_ or [])
    counts = u.counts()
    log.info("Registret: %d bolag · %s", len(u.members),
             " · ".join(f"{ur.label(k)} {n}" for k, n in counts.items()))
    log.info("Källor: %s", " · ".join(f"{k} {n}" for k, n in sorted(u.stats.items(), key=lambda kv: -kv[1])))
    themed = sum(1 for m in u.members.values() if m.theme)
    log.info("Med råvarutema: %d · ETF:er: %d", themed, sum(1 for m in u.members.values() if m.kind == ur.ETF))
    for note in u.notes:
        log.warning("%s", note)
    out = {"generated": u.generated, "members": len(u.members), "counts": counts, "stats": u.stats, "notes": u.notes,
           "saved": {}}
    if not no_save:
        out["saved"] = ur.save(u, save)
        for name, ok in out["saved"].items():
            log.info("%s: %s", name, "sparad" if ok else "KUNDE INTE SPARAS")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    run(no_save=args.no_save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
