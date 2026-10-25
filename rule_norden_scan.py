#!/usr/bin/env python3
"""
rule_norden_scan.py — dagens Rule-köp i Norden till papperskontot rule_norden.

Samma urval och regler som testet (rule_kop_study.universe, valhall.rule_study): Nordens 30 största per sektor,
hatperiod mot samma urvals index, Contrarian Alphas katalysator, Deep Contrarians ROIC-grind och balansräkningens
grind. Sparar rule_norden_signals.json i Gisten: signaler (katalysatordag de senaste fem handelsdagarna) och en
bevakningslista (hatade med godkända grindar, ingen katalysator än). Valhall Scan lägger signalerna i papperet.
Körs vardagsmorgnar (rule-norden-scan.yml). Ändrar inga regler. Env: BORSDATA_API_KEY.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rule_norden_scan")

BLOB = "rule_norden_signals.json"


def run(api=None, save=None, no_save=False, today=None) -> dict:
    import markets as _markets
    import rule_kop_study as rks
    from valhall import cycle
    from valhall import hate_study as hs
    from valhall import rule_study as rs
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    if api is None:
        from borsdata_api import get_api
        api = get_api()
    u = rks.universe(api, today)
    keys = ("info", "segment_of", "sel", "closes", "vols", "market", "kp", "table")
    info, seg_of, sel, closes, vols, market, kp, table = (u[k] for k in keys)
    now = sorted({i for per in sel.values() for i in per.get(today.year, [])})
    signals, watch = [], []
    funnel = {"hatade nu": 0, "kvalitet OK": 0, "balans OK": 0, "båda OK": 0, "katalysator redan i perioden": 0,
              "nyckeltal saknas": 0}
    for iid in now:
        fr = cycle.cycle_frame(closes[iid], vols.get(iid), market, 50, cykel_base=market)
        eps = hs.episodes(hs.hated(fr))
        hist = {k: kp[k].get(iid) for k in rs.KPIS}
        res = rs.live(closes[iid], market, eps, hist, today)
        if eps and eps[-1]["censored"]:                          # tratten: varför blir det (inte) signal?
            funnel["hatade nu"] += 1
            g = rs.gates(hist, today)
            funnel["kvalitet OK"] += bool(g["quality"])
            funnel["balans OK"] += bool(g["balance"])
            funnel["båda OK"] += bool(g["quality"] and g["balance"])
            funnel["nyckeltal saknas"] += g["quality"] is None or g["balance"] is None
            funnel["katalysator redan i perioden"] += rs.catalyst_day(rs.catalyst(closes[iid]), eps[-1]) is not None
        i = info[iid]
        base = {"ticker": _markets.to_yf(i.get("ticker"), i.get("marketId"), table), "name": i.get("name"),
                "segment": seg_of.get(iid)}
        if res["signal"]:
            signals.append({**base, **res["signal"]})
        if res["watch"]:
            watch.append({**base, **res["watch"]})
    last = market.index[-1] if len(market) else today
    out = {"generated": datetime.now(tz=timezone.utc).isoformat(), "date": str(pd.Timestamp(last).date()),
           "universe": len(now), "funnel": funnel, "signals": signals,
           "watch": sorted(watch, key=lambda r: r["hate_since"])}
    log.info("Tratten: %s", " · ".join(f"{k} {v}" for k, v in funnel.items()))
    log.info("Rule-köp Norden %s: %d bolag i urvalet · %d signaler · %d på bevakning", out["date"], len(now),
             len(signals), len(watch))
    for r in signals:
        log.info("  SIGNAL %s (%s) katalysator %s · hatad sedan %s", r["ticker"], r["segment"], r["day"], r["hate_since"])
    for r in watch:
        log.info("  bevaka %s (%s) hatad sedan %s", r["ticker"], r["segment"], r["hate_since"])
    if not no_save:
        if save is None:
            from gist_storage import save_blob as save
        out["saved"] = bool(save(BLOB, out))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    run(no_save=args.no_save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
