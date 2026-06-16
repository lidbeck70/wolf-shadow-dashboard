#!/usr/bin/env python3
"""
scheduled_scan.py
Standalone scanner run by GitHub Actions on weekdays (08/12/18 Europe/Stockholm).
Runs all three slow screeners and publishes results to their Gists, so the
panel can read pre-computed results instantly instead of scanning live.

Requires env vars:
  BORSDATA_API_KEY  - Borsdata Pro+ key
  GITHUB_TOKEN      - PAT with 'gist' scope (writes results to Gists)
  CA_GIST_ID        - Gist id for Contrarian Alpha (defaults to known id)
  EMBER_GIST_ID     - Gist id for EMBER results

Exits 0 even if individual screeners fail (logged), so one failure does not
abort the others.
"""
from __future__ import annotations
import os
import sys
import time
import logging
import traceback

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("scheduled_scan")


def _run_contrarian(mode: str) -> None:
    from contrarian_alpha.engine import PipelineConfig, run_pipeline
    from contrarian_alpha.cache import save_screener_results
    from borsdata_api import ALL_NORDIC_MARKETS
    t0 = time.time()
    log.info("Contrarian Alpha [%s] starting...", mode)
    cfg = PipelineConfig(mode=mode, market_ids=list(ALL_NORDIC_MARKETS), top_n=40)
    res = run_pipeline(cfg)
    ok = save_screener_results(res, mode=mode)
    log.info("Contrarian Alpha [%s] done: %d ranked in %.0fs, gist_saved=%s",
             mode, res.composite_ranked, time.time() - t0, ok)


def _run_ember() -> None:
    from ember.engine import run_ember_scan
    from ember.cache import save_ember_results
    t0 = time.time()
    log.info("EMBER starting...")
    res = run_ember_scan(universe_source="both", use_prefilter=True)
    ok = save_ember_results(res)
    log.info("EMBER done: %d eligible, %d near-miss in %.0fs, gist_saved=%s",
             len(res.eligible), len(res.near_misses), time.time() - t0, ok)


def main() -> int:
    if not os.environ.get("BORSDATA_API_KEY"):
        log.error("BORSDATA_API_KEY missing - aborting.")
        return 1

    tasks = [
        ("Contrarian Quality",        lambda: _run_contrarian("quality")),
        ("Contrarian Deep",           lambda: _run_contrarian("deep_contrarian")),
        ("EMBER",                     _run_ember),
    ]
    failures = 0
    for name, fn in tasks:
        try:
            fn()
        except Exception:
            failures += 1
            log.error("%s FAILED:\n%s", name, traceback.format_exc())

    log.info("Scheduled scan complete. %d/%d screeners failed.", failures, len(tasks))
    return 0   # always succeed so one failure does not abort the workflow


if __name__ == "__main__":
    sys.exit(main())
