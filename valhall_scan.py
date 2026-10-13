#!/usr/bin/env python3
"""
valhall_scan.py — Valhall headless: cykelkartan och swingkandidaterna, sparade i Gisten.

Körs vardagar efter nordisk stängning (valhall-scan.yml). Panelen läser inte resultatet än —
den kopplas in när hela Valhall är klart.

  1. Nordiska sektorindex (Börsdata-sektorer, likviditetsfilter) och hela Norden.
  2. Kartan: elva segment (global = SPDR mot S&P 500, Norden = sektorindex mot hela Norden) och
     råvaruteman, med status GRÖN/GUL/RÖD och cykelstart (valhall.board).
  3. Swing: två bästa Viking Nine-affärer per grönt nordiskt segment och grönt råvarutema
     (valhall.swing), plus nischsignaler ur BERSERK och Insider Norden.

Gisten: valhall_board.json. Env: BORSDATA_API_KEY, GITHUB_TOKEN (gist-scope).
Flaggor: --dry-run (räkna, spara inget).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("valhall_scan")

BLOB = "valhall_board.json"


def _json_safe(board: dict) -> dict:
    """Kartan utan pandas-objekt (sparas som JSON)."""
    return {"when": board.get("when"), "segments": board.get("segments")}


def run(api=None, data_getter=None, getter=None, load=None, save=None, today=None, dry_run=False,
        scan_fn=None, nordic_market=None) -> dict:
    from valhall import board as vb
    from valhall import nordic_index as ni
    from valhall import swing
    if load is None or save is None:
        from gist_storage import load_blob, save_blob
        load, save = load or load_blob, save or save_blob
    if api is None:
        from borsdata_api import get_api
        api = get_api()
    if data_getter is None:
        from market_prices import closes_volumes as data_getter
    if getter is None:
        from market_prices import ohlcv as getter
    import markets as _markets
    table = _markets.load(api=api)
    nordic_ids = _markets.nordic_stock_ids(table)
    instruments = api.get_instruments()
    to_yf = lambda t, m: _markets.to_yf(t, m, table)  # noqa: E731
    universe = swing.nordic_universe(instruments, nordic_ids, to_yf)
    mem = ni.members(instruments, nordic_ids, to_yf)
    ccy = {t: v["currency"] for t, v in universe.items()}
    keep: dict = {}
    indices = ni.build_indices(mem, ccy, data_getter, keep=keep)
    log.info("Norden: %d aktier, %d med kurs", len(universe), len(keep))
    if nordic_market is None:
        try:
            import ovtlyr_nine as on
            nordic_market = (on.nordic_market() or {}).get("close")
        except Exception:
            nordic_market = None
    board = vb.build(getter=getter, nordic=indices, nordic_market=nordic_market, today=today)
    sw = swing.candidates(board, universe, keep, scan_fn=scan_fn, getter=getter, today=today)
    nich = swing.niche(universe, load("berserk_scan.json", None), load("insider_scan.json", None))
    payload = {"generated": datetime.now(tz=timezone.utc).isoformat(), "board": _json_safe(board),
               "cycle_starts": vb.cycle_starts(board), "swing": sw, "niche": nich,
               "nordic": {k: {kk: v.get(kk) for kk in ("members", "with_data", "liquid")} for k, v in indices.items()}}
    green = [r["label"] for r in board["segments"] if (r.get("nordic") or {}).get("status") == "GRÖN"]
    log.info("Gröna nordiska segment: %s · likvida aktier: %s", ", ".join(green) or "inga", sw.get("liquid"))
    if dry_run:
        import json
        for k, v in sw["segments"].items():
            log.info("SWING %-16s %-11s skannade %2d  %s", k, v["status"], v["scanned"],
                     [(p["ticker"], p["category"], p["ovtlyr_nine"], p["status"]) for p in v["picks"]])
        for k, v in sw["themes"].items():
            if v["status"] == "GRÖN":
                log.info("TEMA  %-16s %-11s skannade %2d  %s", k, v["status"], v["scanned"],
                         [(p["ticker"], p["category"], p["ovtlyr_nine"], p["status"]) for p in v["picks"]])
        log.info("NISCH %s", {k: (len(v["berserk"]), len(v["insider"])) for k, v in nich.items()})
        log.info("NORDEN %s", json.dumps(payload["nordic"]))
        return {"payload": payload, "saved": False}
    saved = bool(save(BLOB, payload))
    if not saved:
        log.error("Kunde inte spara %s — kontrollera GITHUB_TOKEN (gist-scope).", BLOB)
    return {"payload": payload, "saved": saved}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
