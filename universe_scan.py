#!/usr/bin/env python3
"""
universe_scan.py — bygger universum-registret (universe/registry.py) och sparar det i Gisten.

  1. Börsdata /instruments (Norden) och /instruments/global (USA, Kanada, London, Australien — Pro+ global;
     utan licens blir listan tom och de marknaderna kommer bara ur panelens listor).
  2. Panelens egna listor: BERSERK, ticker_universe, Contrarian Alphas US/CA-CSV, Viking Nine-backtestet,
     Wolf-screenern, heatmapen, EMBER, Long Screener.
  3. universe.json (index) och universe_<marknad>.json (medlemmarna) i Gisten.
  4. segment_top.json: segmentens största bolag med kassaflöde per marknad (universe/segment_top.py, Börsdata
     KPI 50/62) — Thors segmentval visar dem även utan signal. Ett fel där stoppar aldrig registret.

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


def run(api=None, save=None, no_save: bool = False, nordic=None, global_=None, table=None,
        fx_getter=None) -> dict:
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
    if api is not None and table is None:                   # en tabell för registret och segmentlistan
        import markets as mk
        table = mk.load(api=api)
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
    out["saved"].update(segment_top(api, nordic or [], global_ or [], table, save, no_save, fx_getter))
    return out


def segment_top(api, nordic: list, global_: list, table, save=None, no_save: bool = False, fx_getter=None) -> dict:
    """Segmentens största bolag med kassaflöde (universe/segment_top.py) efter registret — ett fel här loggas
    och stoppar aldrig registrets sparning. {fil: ok} när bloben sparades eller försökte sparas."""
    from universe import segment_top as ust
    if api is None or not (nordic or global_):
        log.info("Segmentens största: hoppas över (ingen Börsdata-nyckel eller inga instrument).")
        return {}
    try:
        payload = ust.build(api, nordic, global_, table, fx_getter)
    except Exception as exc:
        log.warning("Segmentens största gick inte att räkna: %s — registret är sparat ändå.", exc)
        return {}
    for line in ust.describe(payload):
        log.info("Störst: %s", line)
    if no_save:
        return {}
    try:
        if save is None:
            from gist_storage import save_blob as save
        ok = bool(save(ust.BLOB, payload))
    except Exception as exc:
        log.warning("%s: %s", ust.BLOB, exc)
        ok = False
    log.info("%s: %s", ust.BLOB, "sparad" if ok else "KUNDE INTE SPARAS")
    return {ust.BLOB: ok}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    try:                                                    # växelkurserna för segmentlistan (annars reservtalen)
        from market_prices import close
        fx_getter = lambda sym: close(sym, "1mo")  # noqa: E731
    except Exception:
        fx_getter = None
    run(no_save=args.no_save, fx_getter=fx_getter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
