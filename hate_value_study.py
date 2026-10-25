#!/usr/bin/env python3
"""
hate_value_study.py — hatat och billigt (docs/forskning/hatat_och_billigt_spec.md, version 1).

Prövar fyndet från "Hatets varaktighet" på bolag den studien aldrig rörde: rank 11–30 per sektor (Börsdata,
föregående års börsvärde). Topp 10 räknas om bredvid som "redan sett". V1: billigt − dyrt bland de hatade;
V2: hatat och billigt mot marknaden. Körs manuellt i Actions (hate-value-study.yml); resultatet i loggen och i
Gisten (hate_value.json). Ändrar inga regler. Env: BORSDATA_API_KEY.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("hate_value_study")

BLOB = "hate_value.json"
SPEC = "docs/forskning/hatat_och_billigt_spec.md"
START_YEAR = 2006


def study(api, today, rng, n_boot=None) -> dict:
    import markets as _markets
    from hate_duration_study import _ohlc
    from valhall import cycle
    from valhall import hate_study as hs
    from valhall import segments as sgm
    n_boot = n_boot or hs.N_BOOT
    table = _markets.load(api=api)
    nordic_ids = _markets.nordic_stock_ids(table)
    info = {i["insId"]: i for i in api.get_instruments() if i.get("marketId") in nordic_ids and i.get("insId")}
    segment_of = {iid: sgm.segment_for(i.get("sectorId"), i.get("branchId")) for iid, i in info.items()}
    ids = [iid for iid, seg in segment_of.items() if seg]
    raw = api.get_kpi_history_batch(ids, 50, "year", "mean", max_count=20)
    mcap = {iid: {int(v["y"]): float(v["v"]) for v in vals if v.get("y") is not None and v.get("v") is not None}
            for iid, vals in raw.items()}
    years = list(range(START_YEAR, pd.Timestamp(today).year + 1))
    cand = hs.top_by_year(mcap, info, segment_of, years, n=hs.MID_RANKS[1] + 6)       # reserv om kurser saknas
    union = sorted({i for per in cand.values() for ys in per.values() for i in ys})
    closes, vols = {}, {}
    for iid in union:
        c, v = _ohlc(api, iid)
        if c is not None and len(c) > 300:
            closes[iid], vols[iid] = c, v
    have = set(closes)
    ev = api.get_kpi_history_batch(sorted(have), 10, "year", "mean", max_count=20)
    pb = api.get_kpi_history_batch(sorted(have), 4, "year", "mean", max_count=20)
    out = {"companies": len(closes)}
    for name, (lo, hi) in (("rank 11–30 (prövning)", hs.MID_RANKS), ("topp 10 (redan sett)", (0, hs.TOP_N))):
        sel = {seg: hs.ranks(per, have, lo, hi) for seg, per in cand.items()}
        allm = {y: [i for per in sel.values() for i in per.get(y, [])] for y in years}
        members = sorted({i for ys in allm.values() for i in ys})
        market = hs.universe_index({i: closes[i] for i in members}, allm)
        rows = []
        for iid in members:
            fr = cycle.cycle_frame(closes[iid], vols.get(iid), market, 50, cykel_base=market)
            eps = hs.episodes(hs.hated(fr))
            seg = segment_of.get(iid)
            mine = []
            for e in eps:
                if e["start"] >= pd.Timestamp(hs.START) and iid in sel.get(seg, {}).get(int(e["start"].year), []):
                    e["undervalued"] = hs.undervalued_at(ev.get(iid), pb.get(iid), e["start"])
                    mine.append(e)
            rows += hs.start_rows(closes[iid], market, mine)
        out[name] = {"members": len(members), **hs.value_summary(rows, rng, n_boot)}
        log.info("%s: %d bolag, %d perioder", name, len(members), len(rows))
    return out


def report_lines(res: dict) -> list:
    lines = [f"HATAT OCH BILLIGT — spec version 1 ({SPEC})"]
    for name, v in res.items():
        if not isinstance(v, dict) or "V1" not in v:
            continue
        c, d = v["cheap"], v["dear"]
        lines.append(f"== {name.upper()} · {v['members']} bolag · utan värdering {v['no_value']}")
        lines.append(f"  hatad och undervärderad   n={c['n']:>4} år={c['years']:>2} rel 2 år snitt {c['mean']} · median {c['median']}")
        lines.append(f"  hatad men inte undervärd. n={d['n']:>4} år={d['years']:>2} rel 2 år snitt {d['mean']} · median {d['median']}")
        lines.append(f"  V1 billigt − dyrt {v['V1']['diff']} %-enheter, 95 % {v['V1']['ci']} → "
                     f"{'GODKÄND' if v['V1']['pass'] else 'EJ GODKÄND'}")
        lines.append(f"  V2 hatat och billigt mot marknaden 95 % {v['V2']['ci']} → {'GODKÄND' if v['V2']['pass'] else 'EJ GODKÄND'}")
    lines.append("  (Bara rank 11–30 avgör; topp 10 är redan sett.)")
    return lines


def run(api=None, save=None, no_save=False, today=None, n_boot=None) -> dict:
    from valhall import hate_study as hs
    rng = np.random.default_rng(hs.SEED)
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    res = {"spec_version": 1, "spec": SPEC, "generated": datetime.now(tz=timezone.utc).isoformat()}
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
