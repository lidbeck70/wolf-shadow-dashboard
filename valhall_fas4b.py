#!/usr/bin/env python3
"""
valhall_fas4b.py — Valhall fas 4b: hjälper cykeln de långsiktiga köpen? (2008–idag)

  1. Samma cykelserier som fas 3a (valhall_fas3.frames): nordiska sektorindex, SPDR-fonderna och
     råvarutemanas drivare — och aktiernas kurser (likviditetsfiltret dag för dag).
  2. Segmentnivå: varje månadsskifte, segmentets läge och avkastningen 12/24/36 mån framåt mot
     marknaden (råvaror absolut), samt "sälj på RÖD" (valhall.longterm.segment_rows).
  3. Aktienivå: likvida nordiska aktier som klarar Quality Contrarians ROIC-grind punkt för punkt
     (Börsdata KPI 37, årsrapporten känd fyra månader efter årets slut), grupperade på segmentets läge,
     och kvartalsportföljer per filter (valhall.longterm.stock_rows / portfolios).

Körs manuellt i Actions (valhall-longterm.yml). Resultatet i loggen och i Gisten (valhall_fas4b.json)
om GITHUB_TOKEN finns. Env: BORSDATA_API_KEY. Ändrar inga regler.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("valhall_fas4b")

BLOB = "valhall_fas4b.json"
START = "2008-01-01"
ROIC_KPI = 37
ROIC_YEARS = 20                    # Börsdatas standard är 10 år — då börjar aktieraderna först 2017


def _close(df):
    if df is None or len(df) == 0 or "Close" not in df:
        return None
    s = df["Close"].astype(float).dropna()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s


def _fmt_seg(rows: list, title: str) -> list:
    lines = [f"== {title}"]
    for r in rows:
        if r["period"] == "alla" or r["n"] >= 12:
            lines.append(f"  {' · '.join(str(r[k]) for k in r if k not in _NUM and k != 'period'):34s} "
                         f"{r['period']:10s} n={r['n']:5d} | 12m {r.get('x12')} (median {r.get('x12_med')}, träff "
                         f"{r.get('hit12')} %) · 24m {r.get('x24')} (median {r.get('x24_med')}) · 36m {r.get('x36')} "
                         f"(median {r.get('x36_med')}, träff {r.get('hit36')} %) | sälj på RÖD 24m {r.get('xred24')}")
    return lines


_NUM = {"n"} | {f"{p}{h}{s}" for p in ("x", "r", "hit", "xred", "n") for h in (12, 24, 36) for s in ("", "_med")}


def _fmt_port(rows: list) -> list:
    lines = ["== AKTIER — KVARTALSPORTFÖLJER (likavikt, alla som klarar filtret)"]
    for r in rows:
        if r.get("quarters"):
            lines.append(f"  {r['filter']:34s} {r['period']:10s} kvartal={r['quarters']:3d} CAGR={r['cagr']:+5.1f} % "
                         f"DD={r['dd']:+6.1f} % snitt {r['avg_n']:5.1f} aktier · Norden {r['bench_cagr']} %")
    return lines


def stock_inputs(api, fr: dict, keep: dict) -> tuple:
    """(universe {ticker: {segment, ins_id}}, closes, turnover i kr) för de nordiska aktierna med kurs."""
    import markets as _markets
    from valhall import nordic_index as ni
    table = _markets.load(api=api)
    nordic_ids = _markets.nordic_stock_ids(table)
    ins = {}
    for i in api.get_instruments():
        if i.get("marketId") in nordic_ids and i.get("ticker") and i.get("insId") is not None:
            ins.setdefault(_markets.to_yf(i["ticker"], i["marketId"], table), i["insId"])
    universe = {t: {"segment": v["segment"], "ins_id": ins.get(t)} for t, v in fr["universe"].items()}
    closes, turnover = {}, {}
    for t, df in keep.items():
        c = _close(df)
        if c is None or t not in universe:
            continue
        closes[t] = c
        if "Volume" in df:
            v = df["Volume"].astype(float).reindex(c.index)
            turnover[t] = c * v * ni.FX_TO_SEK.get(fr["universe"][t]["currency"], 1.0)
    return universe, closes, turnover


def run(api=None, getter=None, data_getter=None, roic_getter=None, today=None, save=None, no_save=False) -> dict:
    import valhall_fas3 as fas3
    from valhall import longterm as lt
    if api is None:
        from borsdata_api import get_api
        api = get_api()
    if getter is None:
        from market_prices import ohlcv as getter
    if data_getter is None:
        from market_prices import closes_volumes as data_getter
    if roic_getter is None:
        def roic_getter(ids):
            return api.get_kpi_history_batch(ids, ROIC_KPI, "year", "mean", max_count=ROIC_YEARS)
    keep: dict = {}
    fr = fas3.frames(api, getter, data_getter, keep=keep)
    bench_n, bench_g = fr["bench"]["nordic"], fr["bench"]["global"]

    seg = (lt.segment_rows(fr["nordic"], bench_n, START, "Norden")
           + lt.segment_rows(fr["global"], bench_g, START, "globalt")
           + lt.segment_rows(fr["themes"], None, START, "tema"))
    log.info("Segmentnivå: %d månadsrader", len(seg))

    universe, closes, turnover = stock_inputs(api, fr, keep)
    ids = sorted({universe[t]["ins_id"] for t in closes if universe[t]["ins_id"] is not None})
    roic = roic_getter(ids) if ids else {}
    years = [int(v["y"]) for vals in roic.values() for v in vals or [] if v.get("y") is not None]
    log.info("Aktier: %d med kurs, %d med ROIC-historik (år %s–%s)", len(closes), sum(1 for i in ids if roic.get(i)),
             min(years) if years else "—", max(years) if years else "—")
    stocks = lt.stock_rows(closes, turnover, universe, roic, fr["nordic"], bench_n, START)
    log.info("Aktienivå: %d kvartalsrader (minst en grind)", len(stocks))

    quality = [r for r in stocks if r["q"]]
    result = {
        "generated": datetime.now(tz=timezone.utc).isoformat(),
        "segments": {"by_status": lt.summarize(seg, ("where", "status")),
                     "by_cykel": lt.summarize(seg, ("where", "cykel")),
                     "by_phase": lt.summarize(seg, ("where", "phase")),
                     "cycle_start": lt.summarize(seg, ("where", "cycle_start")),
                     "by_segment_status": lt.summarize(seg, ("where", "key", "status"))},
        "stocks": {"filters": lt.filter_table(stocks), "portfolios": lt.portfolios(stocks, bench_n),
                   "quality_by_segment_status": lt.summarize(quality, ("segment", "status")),
                   "quality_by_cykel": lt.summarize(quality, ("cykel",))},
        "counts": {"segment_rows": len(seg), "stock_rows": len(stocks), "stocks": len(closes),
                   "with_roic": sum(1 for i in ids if roic.get(i))},
    }
    s, k = result["segments"], result["stocks"]
    for line in (_fmt_seg(s["by_status"], "SEGMENT — STATUS PÅ KÖPDAGEN (mot marknaden; teman absolut)")
                 + _fmt_seg(s["by_cykel"], "SEGMENT — CYKELLÄGE PÅ KÖPDAGEN")
                 + _fmt_seg(s["by_phase"], "SEGMENT — FAS PÅ KÖPDAGEN")
                 + _fmt_seg(s["cycle_start"], "SEGMENT — CYKELSTART (BAISSE · VÄNDER)")
                 + _fmt_seg(k["filters"], "AKTIER — FILTER (mot hela Norden)")
                 + _fmt_seg(k["quality_by_cykel"], "AKTIER — KVALITET PER CYKELLÄGE")
                 + _fmt_seg(k["quality_by_segment_status"], "AKTIER — KVALITET PER SEGMENT OCH STATUS")
                 + _fmt_port(k["portfolios"])):
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
