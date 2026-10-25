#!/usr/bin/env python3
"""
sector_rotation_backtest.py — historiskt test av sektorrotationen (docs/forskning/sektorrotation_spec.md, v1).

Hämtar SPY och de 11 SPDR-fonderna (market_prices, hela historiken), kör odin.rotation_backtest och skriver
utfallet i loggen och i Gisten (sector_rotation_backtest.json) om GITHUB_TOKEN finns. Ändrar inga regler.
Körs manuellt i Actions (sector-rotation-backtest.yml).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sector_rotation_backtest")

BLOB = "sector_rotation_backtest.json"


def load(getter) -> tuple:
    """(dagskurser per fond på SPY:s kalender, SPY, {fond: segmentnamn})."""
    from valhall import segments as sgm
    from valhall_event_study import _close
    spy = _close(getter("SPY", "max"))
    cols, labels = {}, {}
    for s in sgm.SEGMENTS:
        c = _close(getter(s.spdr, "max"))
        if c is None:
            log.warning("%s saknas", s.spdr)
            continue
        cols[s.spdr] = c
        labels[s.spdr] = s.label
    daily = pd.DataFrame(cols).reindex(spy.index).ffill(limit=5)
    return daily, spy, labels


def run(getter=None, save=None, no_save=False, n_boot=None) -> dict:
    from odin import rotation_backtest as rb
    if getter is None:
        from market_prices import ohlcv as getter
    daily, spy, labels = load(getter)
    log.info("Fonder: %s · SPY från %s", ", ".join(daily.columns), spy.index[0].date())
    res = rb.run_backtest(daily, spy, n_boot=n_boot or rb.N_BOOT)
    for r in res["latest"]:
        r["label"] = labels.get(r["ticker"], r["ticker"])
    res["generated"] = datetime.now(tz=timezone.utc).isoformat()
    for line in rb.report_lines(res):
        log.info("%s", line)
    if not no_save:
        if save is None:
            from gist_storage import save_blob as save
        res["saved"] = bool(save(BLOB, res))
    return res


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    run(no_save=args.no_save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
