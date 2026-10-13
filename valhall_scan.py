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
  4. Köp (valhall.picks): två kvalitetsköp per segment (Quality Contrarian) och per råvara (Rick Rule),
     fem hävstångsköp per råvara (Durrett för guld/silver med 15-BAGGER-regeln, Tiggre för övriga).
     Källor: Quality Contrarian-körningarna och håvarna (Gisten) och dina ark på panel-data.

Gisten: valhall_board.json. Env: BORSDATA_API_KEY, GITHUB_TOKEN (gist-scope), ALERT_REPO_TOKEN (läser
arken på panel-data). --dry-run skriver en sammanfattning i loggen.
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


def _load_sources(load) -> dict:
    """Quality Contrarian (båda lägena), håvarna och arken — tomt när en källa inte går att läsa."""
    out = {"ca": [], "screens": None, "sheets": {}}
    try:
        from contrarian_alpha.cache import load_screener_results
        out["ca"] = [b for b in (load_screener_results("quality"), load_screener_results("deep_contrarian")) if b]
    except Exception as exc:
        log.warning("Quality Contrarian gick inte att läsa: %s", exc)
    out["screens"] = load("screens.json", None)
    try:
        from sheets_refresh import load_sheets
        out["sheets"] = load_sheets()
    except Exception as exc:
        log.warning("Arken på panel-data gick inte att läsa: %s", exc)
    return out


def summary(payload: dict) -> list:
    """Läsbar sammanfattning (loggen vid --dry-run)."""
    lines = []
    sw, buys = payload["swing"], payload["buys"]
    for r in payload["board"]["segments"]:
        k = r["key"]
        lines.append(f"{r['label']:16s} global {r['global']['status']:11s} Norden "
                     f"{(r.get('nordic') or {}).get('status', '—'):11s} swing "
                     f"{[(p['ticker'], p['category']) for p in sw['segments'][k]['picks']]} kvalitet "
                     f"{[p['ticker'] for p in buys['segments'][k]['picks']]}")
        for t in r["themes"]:
            q = buys["themes"][t["theme"]]["quality"]["picks"]
            lv = buys["themes"][t["theme"]]["leverage"]
            ts = sw["themes"].get(t["theme"], {})
            if q or lv["picks"] or ts.get("picks"):
                lines.append(f"    {t['label']:24s} {t['state']['status']:11s} kvalitet {[p['ticker'] for p in q]} "
                             f"hävstång ({lv['strategy']}) {[(p['ticker'], p.get('bagger')) for p in lv['picks']]} "
                             f"swing {[p['ticker'] for p in ts.get('picks', [])]}")
    lines.append(f"Källor: {payload['sources']} · likvida: {sw.get('liquid')}")
    return lines


def run(api=None, data_getter=None, getter=None, load=None, save=None, today=None, dry_run=False,
        scan_fn=None, nordic_market=None, sources=None, analyze=None) -> dict:
    from valhall import board as vb
    from valhall import nordic_index as ni
    from valhall import picks as vp
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
    src = sources if sources is not None else _load_sources(load)
    buys = vp.build(src.get("ca") or [], src.get("screens"), src.get("sheets") or {}, analyze=analyze)
    payload = {"generated": datetime.now(tz=timezone.utc).isoformat(), "board": _json_safe(board),
               "cycle_starts": vb.cycle_starts(board), "swing": sw, "niche": nich, "buys": buys,
               "sources": {"ca_runs": len(src.get("ca") or []), "screens": bool(src.get("screens")),
                           "sheets": sorted(k for k, v in (src.get("sheets") or {}).items() if v)},
               "nordic": {k: {kk: v.get(kk) for kk in ("members", "with_data", "liquid")} for k, v in indices.items()}}
    green = [r["label"] for r in board["segments"] if (r.get("nordic") or {}).get("status") == "GRÖN"]
    log.info("Gröna nordiska segment: %s · likvida aktier: %s", ", ".join(green) or "inga", sw.get("liquid"))
    if dry_run:
        for line in summary(payload):
            log.info("%s", line)
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
