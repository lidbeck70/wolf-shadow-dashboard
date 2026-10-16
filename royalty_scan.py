#!/usr/bin/env python3
"""
royalty_scan.py — headless körning av Royalty: playbookens grindar och Contrarian Alpha-rangen på
royaltyuniversumet (config/universes/royalty.csv plus Royalty-håvens royalty/streaming-träffar ur screens.json).

Körs i scheduled-scan-workflowen efter håvarna (så screens.json är färsk). Resultatet skrivs till Gisten som
royalty_scan.json: {"generated", "universe_source", "rows": [...], "stats", "errors", "weights", "gates"}.

Data: kurser ur Yahoo (market_prices.ohlcv, 2 år), nyckeltal ur Börsdatas KPI-screener (nordisk eller global
beroende på registrets marknad) med håvens värden som reserv, EV/EBITDA- och P/FCF-historik ur Börsdata för
värdepelaren. Utan BORSDATA_API_KEY räknas grindarna på håvens nyckeltal och värdepelaren blir 0 (loggas).

Env: BORSDATA_API_KEY, GITHUB_TOKEN (gist-scope). Flaggor: --dry-run (räkna, skriv inget till Gisten).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from royalty import engine as eng
from royalty import universe as ru

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("royalty_scan")

BLOB_NAME = "royalty_scan.json"
SCREENS_BLOB = "screens.json"
PRICE_PERIOD = "2y"
HISTORY_KPIS = ("ev_ebitda", "p_fcf")


def companies(loader=None, register=None) -> tuple:
    """(bolagen, källtext): listan + håvens träffar, med Börsdata-id ur universum-registret."""
    if loader is None:
        from gist_storage import load_blob as loader
    screens = None
    try:
        screens = loader(SCREENS_BLOB, None)
    except Exception as exc:
        log.warning("screens.json gick inte att läsa (%s) — bara listan.", exc)
    if register is None:
        try:
            from universe import registry as ur
            register = ur.load(None, loader)
        except Exception as exc:
            log.warning("universum-registret gick inte att läsa (%s) — utan Börsdata-id från det.", exc)
            register = None
    rows = ru.build(None, screens, register)
    src = "listan" + (" + Royalty-håven" if screens else "") + (" + registret" if register is not None else "")
    return rows, src


def data_getter(api, rows: list, history=None):
    """Nyckeltalen per bolag: Börsdatas snapshot (nordisk/global efter marknad) med håvens värden som reserv,
    plus EV/EBITDA- och P/FCF-historik för värdepelaren. history(ins_id, kpi_nyckel) → lista (nyast först)."""
    snaps: dict = {}
    if api is not None:
        by_scope: dict = {"nordic": [], "global": []}
        for r in rows:
            if r.get("ins_id") is not None:
                by_scope["nordic" if r.get("market") == "norden" else "global"].append(int(r["ins_id"]))
        for scope, ids in by_scope.items():
            if not ids:
                continue
            try:
                got = (api.get_fundamentals_snapshot_fast(ids, scope="global") if scope == "global"
                       else api.get_fundamentals_snapshot_fast(ids))
                snaps.update(got or {})
            except Exception as exc:
                log.warning("Börsdata-snapshot (%s) föll: %s", scope, exc)
    if history is None:
        def history(ins_id, key):
            if api is None:
                return []
            from borsdata_api import KPI
            from contrarian_alpha.engine import _fetch_kpi_history
            return _fetch_kpi_history(ins_id, KPI[key], api)

    def get(company: dict) -> tuple:
        snap = snaps.get(company.get("ins_id")) or {}
        kpi = eng.merge_kpis(eng.kpis_from_snapshot(snap), eng.kpis_from_screen(company.get("m")))
        if snap:
            from contrarian_alpha.engine import _build_fundamentals_dict
            fund = _build_fundamentals_dict(snap)
        else:
            fund = eng.fund_from_kpis(kpi)
        value: dict = {}
        for k in HISTORY_KPIS:
            if kpi.get(k) is not None:
                value[k] = kpi[k]
        if company.get("ins_id") is not None:
            for k in HISTORY_KPIS:
                try:
                    hist = history(company["ins_id"], k)
                except Exception as exc:
                    log.debug("historik %s %s: %s", company["ticker"], k, exc)
                    hist = []
                if hist:
                    value[f"{k}_history"] = [float(x) for x in hist]
        return kpi, fund, value
    return get


def run(api=None, loader=None, getter=None, save=None, today=None, dry_run: bool = False, register=None) -> dict:
    """Hela körningen med utbytbara delar (tester): api = Börsdata, loader(namn, fallback), getter(ticker) → OHLCV,
    save(namn, data). Returnerar {"payload", "saved"}."""
    if loader is None:
        from gist_storage import load_blob as loader
    if getter is None:
        from market_prices import ohlcv

        def getter(ticker):
            return ohlcv(ticker, PRICE_PERIOD)
    rows, src = companies(loader, register)
    log.info("Royalty-universumet: %s (%s)", ru.describe(rows), src)
    if api is None:
        log.warning("Ingen Börsdata-nyckel — grindarna räknas på håvens nyckeltal, värdepelaren blir 0.")
    payload = eng.run(rows, getter, data_getter(api, rows), today=today)
    payload["universe_source"] = src
    payload["generated"] = eng.stamp(today)
    st = payload["stats"]
    log.info("Royalty: %d bolag · %d BUY CANDIDATE · %d WATCH · %d PASS · %d fel", st["n"], st["buy"], st["watch"],
             st["pass"], len(payload["errors"]))
    for r in payload["rows"][:15]:
        log.info("  %-8s %-28s %-13s comp %5.1f  hat %3.0f värde %3.0f kval %3.0f kat %3.0f  %s", r["ticker"],
                 r["name"][:28], r["verdict"], r["composite"], r["pillars"]["hate"], r["pillars"]["value"],
                 r["pillars"]["quality"], r["pillars"]["catalyst"], "; ".join(r["why"])[:60])
    for t, e in payload["errors"].items():
        log.warning("  %s: %s", t, e)
    if dry_run:
        log.info("[DRY-RUN] sparar inte till Gisten.")
        return {"payload": payload, "saved": False}
    if save is None:
        from gist_storage import save_blob as save
    saved = bool(save(BLOB_NAME, payload))
    if saved:
        log.info("Sparat till Gisten som %s.", BLOB_NAME)
    else:
        log.error("Kunde inte spara %s till Gisten — kontrollera GITHUB_TOKEN (gist-scope).", BLOB_NAME)
    return {"payload": payload, "saved": saved}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    api = None
    key = os.environ.get("BORSDATA_API_KEY", "") or os.environ.get("BD_API_KEY", "")
    if key:
        from borsdata_api import BorsdataAPI
        api = BorsdataAPI(api_key=key)
    else:
        log.warning("BORSDATA_API_KEY saknas.")
    run(api=api, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
