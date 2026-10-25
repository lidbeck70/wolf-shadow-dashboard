#!/usr/bin/env python3
"""
rule_kop_study.py — Rule-köp i Norden (docs/forskning/rule_kop_spec.md, version 1).

Nordens 30 största bolag per sektor (Börsdata, föregående års börsvärde): hatperioder, Contrarian Alphas
katalysator, Deep Contrarians kvalitetsgrind och balansräkningens grind på årssiffror med fördröjning.
K1 Rule-köp mot marknaden, K2 Rule-köp mot "utan kvalitet", K3 katalysatorn mot hatets start.
Körs manuellt i Actions (rule-kop-study.yml); resultatet i loggen och i Gisten (rule_kop.json). Ändrar inga
regler. Env: BORSDATA_API_KEY.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rule_kop_study")

BLOB = "rule_kop.json"
START_YEAR = 2006


def universe(api, today) -> dict:
    """Urvalet som testet och papperskontot delar: Nordens 30 största per sektor och år (föregående års
    börsvärde), kurser, marknadsindexet av samma urval och nyckeltalshistoriken."""
    import markets as _markets
    from hate_duration_study import _ohlc
    from valhall import hate_study as hs
    from valhall import rule_study as rs
    from valhall import segments as sgm
    table = _markets.load(api=api)
    nordic_ids = _markets.nordic_stock_ids(table)
    info = {i["insId"]: i for i in api.get_instruments() if i.get("marketId") in nordic_ids and i.get("insId")}
    segment_of = {iid: sgm.segment_for(i.get("sectorId"), i.get("branchId")) for iid, i in info.items()}
    ids = [iid for iid, seg in segment_of.items() if seg]
    raw = api.get_kpi_history_batch(ids, 50, "year", "mean", max_count=20)
    mcap = {iid: {int(v["y"]): float(v["v"]) for v in vals if v.get("y") is not None and v.get("v") is not None}
            for iid, vals in raw.items()}
    years = list(range(START_YEAR, pd.Timestamp(today).year + 1))
    cand = hs.top_by_year(mcap, info, segment_of, years, n=rs.TOP_N + 6)
    union = sorted({i for per in cand.values() for ys in per.values() for i in ys})
    closes, vols = {}, {}
    for iid in union:
        c, v = _ohlc(api, iid)
        if c is not None and len(c) > 300:
            closes[iid], vols[iid] = c, v
    have = set(closes)
    sel = {seg: hs.ranks(per, have, 0, rs.TOP_N) for seg, per in cand.items()}
    allm = {y: [i for per in sel.values() for i in per.get(y, [])] for y in years}
    members = sorted({i for ys in allm.values() for i in ys})
    market = hs.universe_index({i: closes[i] for i in members}, allm)
    kp = {k: api.get_kpi_history_batch(members, kid, "year", "mean", max_count=20) for k, kid in rs.KPIS.items()}
    return {"table": table, "info": info, "segment_of": segment_of, "sel": sel, "members": members,
            "closes": closes, "vols": vols, "market": market, "kp": kp, "years": years}


def study(api, today, rng, n_boot=None) -> dict:
    from valhall import cycle
    from valhall import hate_study as hs
    from valhall import rule_study as rs
    n_boot = n_boot or hs.N_BOOT
    u = universe(api, today)
    keys = ("info", "segment_of", "sel", "closes", "vols", "market", "kp")
    info, segment_of, sel, closes, vols, market, kp = (u[k] for k in keys)
    members = u["members"]
    roic_now = [vals[-1]["v"] for vals in kp["roic"].values() if vals and vals[-1].get("v") is not None]
    de_now = [vals[-1]["v"] for vals in kp["de"].values() if vals and vals[-1].get("v") is not None]
    log.info("Kontroll av enheter: ROIC median %.1f (förväntat %%) · skuldsättningsgrad median %.2f (förväntat kvot)",
             float(np.median(roic_now)) if roic_now else float("nan"), float(np.median(de_now)) if de_now else float("nan"))
    rows = []
    for iid in members:
        fr = cycle.cycle_frame(closes[iid], vols.get(iid), market, 50, cykel_base=market)
        seg = segment_of.get(iid)
        eps = [e for e in hs.episodes(hs.hated(fr))
               if iid in sel.get(seg, {}).get(int(e["start"].year), [])]
        hist = {k: kp[k].get(iid) for k in rs.KPIS}
        rows += [{**r, "ticker": str(info[iid].get("ticker")), "sector": seg}
                 for r in rs.event_rows(closes[iid], market, eps, hist)]
    log.info("Norden topp %d: %d bolag, %d händelser med katalysator och kända grindar", rs.TOP_N, len(members), len(rows))
    return {"companies": len(members), **rs.summary(rows, rng, n_boot), "events": rows}


def report_lines(res: dict) -> list:
    lines = ["RULE-KÖP I NORDEN — spec version 1 (docs/forskning/rule_kop_spec.md)"]
    if "K1" not in res:
        return lines + [f"  fel: {res.get('error')}"]
    r, u = res["rule"], res["utan"]
    lines.append(f"  Rule-köp (hat+kvalitet+balans+katalysator) n={r['n']} år={r['years']} rel 2 år snitt {r['mean']} · "
                 f"median {r['median']} · träff {r['hit']} %")
    lines.append(f"  Utan kvalitet/balans                      n={u['n']} år={u['years']} rel 2 år snitt {u['mean']} · "
                 f"median {u['median']} · träff {u['hit']} % (föll på kvalitet {res['fail_quality']}, balans {res['fail_balance']})")
    lines.append(f"  K1 Rule-köp mot marknaden 95 % {res['K1']['ci']} → {'GODKÄND' if res['K1']['pass'] else 'EJ GODKÄND'}")
    lines.append(f"  K2 Rule − utan {res['K2']['diff']} %-enheter, 95 % {res['K2']['ci']} → "
                 f"{'GODKÄND' if res['K2']['pass'] else 'EJ GODKÄND'}")
    lines.append(f"  K3 katalysatorn − hatets start (beskrivande): n={res['K3']['n']} snitt {res['K3']['mean']} · "
                 f"median {res['K3']['median']}")
    return lines


def run(api=None, save=None, no_save=False, today=None, n_boot=None) -> dict:
    from valhall import hate_study as hs
    rng = np.random.default_rng(hs.SEED)
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    res = {"spec_version": 1, "spec": "docs/forskning/rule_kop_spec.md",
           "generated": datetime.now(tz=timezone.utc).isoformat()}
    try:
        if api is None:
            from borsdata_api import get_api
            api = get_api()
        ok = getattr(api, "is_configured", True) if api is not None else False
        if callable(ok) and not isinstance(ok, bool):
            ok = ok()
        res.update(study(api, today, rng, n_boot) if ok else {"error": "Börsdata saknas"})
    except Exception as exc:
        log.exception("Studien fallerade: %s", exc)
        res["error"] = str(exc)
    for line in report_lines(res):
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
