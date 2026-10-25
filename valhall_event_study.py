#!/usr/bin/env python3
"""
valhall_event_study.py — eventtest nivå 1 (docs/forskning/vandning_eventtest_spec.md, version 1).

  1. Serierna: de 11 SPDR-fonderna (cykelläget mot SPY, som i kartan), råvarutemanas drivare
     (berserk.backtest.pick_driver från 2008, samma som fas 3; en drivarsymbol räknas en gång) och de sex
     komplexkorgarna (beskrivande). Nordiska index och producentkorgar är uteslutna (dagens universum).
  2. Vändningsdagarna ur valhall.cycle (oförändrad) och trendregeln som jämförelse.
  3. valhall.event_study.run_study → hypoteserna H1–H3 mot de låsta kriterierna.

Körs manuellt i Actions (valhall-event-study.yml). Resultatet i loggen och i Gisten (valhall_event_study.json)
om GITHUB_TOKEN finns. Ändrar inga regler.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("valhall_event_study")

BLOB = "valhall_event_study.json"
START_YEAR = 2008                      # drivarvalet som i fas 3 (pick_driver)


def _close(df):
    if df is None or len(df) == 0 or "Close" not in df:
        return None
    s = df["Close"].astype(float).dropna()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s if len(s) else None


def load_series(getter) -> list:
    """[{name, group, close, events: {signal: [datum]}}] för segment, råvaror och komplex."""
    from berserk import backtest as bbt
    from berserk import themes as bth
    from valhall import board as vb
    from valhall import complexes as vcx
    from valhall import cycle
    from valhall import event_study as es
    from valhall import segments as sgm
    spy = _close(getter("SPY", "max"))
    out = []
    for s in sgm.SEGMENTS:
        df = getter(s.spdr, "max")
        c = _close(df)
        if c is None:
            log.warning("%s saknas", s.spdr)
            continue
        fr = cycle.cycle_frame(c, df["Volume"] if "Volume" in df else None, spy, s.necessity, cykel_base=spy)
        out.append({"name": s.spdr, "group": es.SEGMENT, "close": c,
                    "events": {es.TURN: es.segment_events(fr), es.TREND: es.trend_events(c)}})
    cache: dict = {}
    drivers: dict = {}

    def driver(theme):
        for sym in bth.drivers(theme):
            if sym not in cache:
                cache[sym] = _close(getter(sym, "max"))
        sym, d = bbt.pick_driver(theme, f"{START_YEAR}-01-01", cache)
        if d is not None:
            drivers[theme] = (sym, d)
        return sym, d

    seen = set()
    for theme in dict.fromkeys(t for s in sgm.SEGMENTS for t in s.themes):
        sym, d = driver(theme)
        if d is None or sym in seen:
            continue
        seen.add(sym)
        fr = cycle.cycle_frame(d, None, spy, vb.THEME_NECESSITY.get(theme, 50))
        out.append({"name": f"{theme}:{sym}", "group": es.RAVARA, "close": d,
                    "events": {es.TURN: es.ravara_events(fr), es.TREND: es.trend_events(d)}})
    for k in vcx.KEYS:
        for theme in vcx.themes_of(k):
            if theme not in drivers:
                driver(theme)
        _syms, b = vcx.basket(k, drivers)
        if b is None or len(b) == 0:
            log.warning("komplex %s saknar drivare", k)
            continue
        fr = cycle.cycle_frame(b, None, spy, vcx.NECESSITY[k])
        out.append({"name": f"komplex:{k}", "group": es.KOMPLEX, "close": b,
                    "events": {es.TURN: es.ravara_events(fr), es.TREND: es.trend_events(b)}})
    log.info("Serier: %s", ", ".join(f"{g} {sum(1 for x in out if x['group'] == g)}"
                                     for g in (es.SEGMENT, es.RAVARA, es.KOMPLEX)))
    return out


def run(getter=None, save=None, no_save=False, n_perm=None) -> dict:
    from valhall import event_study as es
    if getter is None:
        from market_prices import ohlcv as getter
    series = load_series(getter)
    result = es.run_study(series, n_perm=n_perm or es.N_PERM)
    result["generated"] = datetime.now(tz=timezone.utc).isoformat()
    result["series"] = [{"name": s["name"], "group": s["group"], "first": str(s["close"].index[0].date()),
                         "last": str(s["close"].index[-1].date())} for s in series]
    for line in es.report_lines(result):
        log.info("%s", line)
    if not no_save:
        if save is None:
            from gist_storage import save_blob as save
        result["saved"] = bool(save(BLOB, result))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--n-perm", type=int, default=None)
    args = parser.parse_args(argv)
    run(no_save=args.no_save, n_perm=args.n_perm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
