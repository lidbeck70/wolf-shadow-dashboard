#!/usr/bin/env python3
"""
valhall_fas4c.py — Valhall fas 4c: Long Screener punkt för punkt 2008–idag (valhall.long_backtest).

  1. Samma kurser, nordiska sektorindex och likviditetsfilter som fas 4b (valhall_fas3.frames).
  2. Börsdatas årshistorik (20 år) för screenerns nyckeltal + omsättning, EV/EBIT, avkastning på kapital
     och ROIC — ett år känt fyra månader efter räkenskapsårets slut.
  3. Screenerns egen poäng varje månadsskifte i tre cykelvarianter (Valhall / screenerns förval / ingen),
     framåtavkastningen för köpen per variant och Valhall-status, och screenerns portfölj (10 innehav,
     10/7 %, 25 % per sektor, sälj på SELL). Som jämförelse: Contrarian Alphas ROIC-grind på samma rader.

Körs manuellt i Actions (valhall-longscreener.yml). Resultatet i loggen och i Gisten
(valhall_fas4c.json) om GITHUB_TOKEN finns. Env: BORSDATA_API_KEY. Ändrar inga regler.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("valhall_fas4c")

BLOB = "valhall_fas4c.json"
START = "2008-01-01"
KPI_YEARS = 20
INDICES = ("^OMX", "OBX.OL", "^OMXC25", "^OMXH25")


def _fmt_buys(rows: list) -> list:
    lines = ["== LONG SCREENER — KÖPEN (STRONG BUY/BUY) MOT HELA NORDEN"]
    for r in rows:
        if r.get("n"):
            grp = " · ".join(str(r[k]) for k in ("variant", "status") if k in r)
            lines.append(f"  {grp:34s} {r['period']:10s} n={r['n']:6d} | 12m {r.get('x12')} (median {r.get('x12_med')}, "
                         f"träff {r.get('hit12')} %) · 24m {r.get('x24')} (median {r.get('x24_med')}) · 36m "
                         f"{r.get('x36')} (median {r.get('x36_med')}, träff {r.get('hit36')} %)")
    return lines


def _fmt_port(rows: list) -> list:
    lines = ["== LONG SCREENER — PORTFÖLJEN (10 innehav, STRONG BUY 10 % / BUY 7 %, 25 % per sektor, sälj på SELL)"]
    for r in rows:
        if r.get("months"):
            lines.append(f"  {r['variant']:8s} {r['period']:10s} mån={r['months']:3d} CAGR={r['cagr']:+5.1f} % "
                         f"DD={r['dd']:+6.1f} % snitt {r['avg_n']:4.1f} innehav · köp {r['buys']:4d} · "
                         f"Norden {r['bench_cagr']} %")
    return lines


def run(api=None, getter=None, data_getter=None, kpi_getter=None, today=None, save=None, no_save=False) -> dict:
    import valhall_fas3 as fas3
    import valhall_fas4b as fas4b
    from valhall import long_backtest as lb
    from valhall import longterm as lt
    if api is None:
        from borsdata_api import get_api
        api = get_api()
    if getter is None:
        from market_prices import ohlcv as getter
    if data_getter is None:
        from market_prices import closes_volumes as data_getter
    if kpi_getter is None:
        def kpi_getter(ids, kpi_id):
            return api.get_kpi_history_batch(ids, kpi_id, "year", "mean", max_count=KPI_YEARS)
    keep: dict = {}
    fr = fas3.frames(api, getter, data_getter, keep=keep)
    bench = fr["bench"]["nordic"]
    universe, closes, turnover = fas4b.stock_inputs(api, fr, keep)
    closes = {t: lt.clean(c) for t, c in closes.items()}
    by_ins = {}
    for t in closes:
        if universe[t]["ins_id"] is not None:
            by_ins.setdefault(universe[t]["ins_id"], []).append(t)
    ids = sorted(by_ins)
    hist: dict = {}
    for field, kid in lb.kpi_ids().items():
        got = kpi_getter(ids, kid) if ids else {}
        vals = [float(v["v"]) for vs in got.values() for v in vs or [] if v.get("v") is not None]
        log.info("KPI %-20s id %3d: %4d bolag · median %s", field, kid, sum(1 for v in got.values() if v),
                 round(float(np.median(vals)), 2) if vals else "—")
        for iid, vs in got.items():
            for t in by_ins.get(iid, []):
                hist.setdefault(t, {})[field] = vs
    from valhall import long_screener as ls
    index_closes = {sym: ls.index_series(sym, lambda x: fas4b._close(getter(x, "max")), api) for sym in INDICES}
    log.info("Index: %s", {k: (f"{v.index[0].date()}–{v.index[-1].date()}" if v is not None and len(v) else "saknas")
                           for k, v in index_closes.items()})
    missing = ["insider"] + [f for f in lb.OPTIONAL_HISTORY if not any(hist.get(t, {}).get(f) for t in hist)]
    fund_max = lb.FUND_MAX - len(missing)
    log.info("Fundamentapoäng räknas mot %d (saknar historik: %s)", fund_max, ", ".join(missing))
    dates = lt.month_ends(bench.index, START) if bench is not None else []
    log.info("Aktier: %d med kurs, %d med nyckeltal · %d månadsskiften", len(closes), len(hist), len(dates))

    def progress(i, n, k):
        if i % 12 == 0 or i == n:
            log.info("  %d/%d månader · %d aktier senast", i, n, k)

    rows = lb.signals(closes, turnover, universe, hist, fr["nordic"], index_closes, dates, progress=progress,
                      fund_max=fund_max)
    lb.add_returns(rows, closes, bench)
    buys = lb.buy_table(rows)
    buys += lt.summarize([dict(r, variant="ROIC > 15 % (Contrarian Alpha-grinden)") for r in rows if r["q"]],
                         ("variant",))
    result = {"generated": datetime.now(tz=timezone.utc).isoformat(), "buys": buys,
              "portfolios": lb.portfolios(rows, bench),
              "fund_max": fund_max, "missing": missing,
              "counts": {"rows": len(rows), "stocks": len(closes), "with_kpis": len(hist), "months": len(dates),
                         **{f"buys_{v}": sum(1 for r in rows if r[f"sig_{v}"] in ("STRONG BUY", "BUY"))
                            for v in lb.VARIANTS}}}
    log.info("Rader: %s", result["counts"])
    for line in _fmt_buys(buys) + _fmt_port(result["portfolios"]):
        log.info("%s", line)
    if not no_save:
        if save is None:
            from gist_storage import save_blob as save
        result["saved"] = bool(save(BLOB, result))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    run(no_save=args.no_save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
