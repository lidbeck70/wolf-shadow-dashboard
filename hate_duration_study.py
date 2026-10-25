#!/usr/bin/env python3
"""
hate_duration_study.py — hatets varaktighet (docs/forskning/hatets_varaktighet_spec.md, version 1).

  Norden    per Valhall-segment de 10 (variant 5) största bolagen efter föregående års börsvärde (Börsdata),
            kurser från Börsdata, marknaden = likaviktade Nordens storbolag (alla sektorers topp 10)
  Råvaror   producent-ETF:erna och råvarudrivarna, marknaden = SPY

D1 varaktighet per sektor/grupp, D2 köp vid start/6/12 månader, D3 hat-bredd B1/B2 — mot de låsta kriterierna.
Körs manuellt i Actions (hate-duration-study.yml). Resultatet i loggen och i Gisten (hate_duration.json) om
GITHUB_TOKEN finns. Ändrar inga regler. Env: BORSDATA_API_KEY.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("hate_duration_study")

BLOB = "hate_duration.json"
START_YEAR = 2006


def _close(df):
    from valhall_event_study import _close as c
    return c(df)


def _ohlc(api, iid):
    try:
        df = api.get_stockprices_df(int(iid), max_count=5040)
    except Exception as exc:
        log.warning("kurser %s: %s", iid, exc)
        return None, None
    if df is None or not len(df):
        return None, None
    c = _close(df)
    v = df["Volume"].reindex(c.index) if c is not None and "Volume" in df else None
    return c, v


def nordic(api, today, rng) -> dict:
    """Norden: D1 per sektor, D2 och D3 för topp 10 och topp 5 (variant), dagens hatade storbolag."""
    import markets as _markets
    from valhall import cycle
    from valhall import hate_study as hs
    from valhall import long_study as lst
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
    cand = hs.top_by_year(mcap, info, segment_of, years, n=hs.TOP_N + 4)      # reserv om kurser saknas
    union = sorted({i for per in cand.values() for ys in per.values() for i in ys})
    closes, vols = {}, {}
    for iid in union:
        c, v = _ohlc(api, iid)
        if c is not None and len(c) > 300:
            closes[iid], vols[iid] = c, v
    have = set(closes)
    top = {n: {seg: hs.restrict(per, have, n) for seg, per in cand.items()} for n in (hs.TOP_N, hs.TOP_VARIANT)}
    all10 = {y: [i for seg in top[hs.TOP_N].values() for i in seg.get(y, [])] for y in years}
    market = hs.universe_index(closes, all10)
    log.info("Norden: %d bolag med kurser, marknadsindex från %s", len(closes),
             market.index[0].date() if len(market) else "—")
    ev = api.get_kpi_history_batch(sorted(have), 10, "year", "mean", max_count=20)
    pb = api.get_kpi_history_batch(sorted(have), 4, "year", "mean", max_count=20)
    states, eps_of = {}, {}
    for iid, c in closes.items():
        fr = cycle.cycle_frame(c, vols.get(iid), market, 50, cykel_base=market)
        h = hs.hated(fr)
        states[iid] = h
        eps = hs.episodes(h)
        for e in eps:
            e["undervalued"] = hs.undervalued_at(ev.get(iid), pb.get(iid), e["start"])
        eps_of[iid] = eps
    label = {s.key: s.label for s in sgm.SEGMENTS}
    out = {"companies": len(closes), "variants": {}}
    for n in (hs.TOP_N, hs.TOP_VARIANT):
        d1, d2rows, d3rows, cur = {}, [], [], []
        alln = {y: [i for seg in top[n].values() for i in seg.get(y, [])] for y in years}
        uni = {h: lst.basket_forward({i: closes[i] for ys in alln.values() for i in ys}, alln, h) for h in hs.BREADTH_H}
        for seg, per in top[n].items():
            mine = []
            for iid in sorted({i for ys in per.values() for i in ys}):
                for e in eps_of.get(iid, []):
                    if e["start"] >= pd.Timestamp(hs.START) and iid in per.get(int(e["start"].year), []):
                        mine.append((iid, e))
            k = hs.km([e["length"] for _i, e in mine], [e["censored"] for _i, e in mine])
            uv = [e["undervalued"] for _i, e in mine if e["undervalued"] is not None]
            k["undervalued_share"] = round(sum(uv) / len(uv) * 100, 1) if uv else None
            d1[label.get(seg, seg)] = k
            for iid, e in mine:
                d2rows += [{**r, "sector": seg} for r in hs.delay_rows(closes[iid], market, [e])]
            b = hs.breadth({i: states[i] for ys in per.values() for i in ys if i in states}, per, market.index)
            sec_c = {i: closes[i] for ys in per.values() for i in ys}
            rel = {}
            for h in hs.BREADTH_H:
                sb = lst.basket_forward(sec_c, per, h)
                rel[h] = (sb - uni[h].reindex(sb.index)).reindex(market.index)
            d3rows += hs.d3_rows(seg, rel, hs.breadth_events(b))
            if n == hs.TOP_N:
                for iid in per.get(pd.Timestamp(today).year, []):
                    c = hs.current(str(info[iid].get("ticker")), states.get(iid), eps_of.get(iid, []))
                    if c:
                        cur.append({**c, "sector": label.get(seg, seg), "typical_m": k.get("median_m")})
        out["variants"][n] = {"d1": d1, "d2": hs.d2_summary(d2rows, rng), "d3": hs.d3_summary(d3rows, rng),
                              "d2_undervalued": hs.d2_summary([r for r in d2rows if r.get("undervalued")], rng),
                              "d3_events": d3rows}
        if n == hs.TOP_N:
            out["current"] = sorted(cur, key=lambda r: -r["months"])
    return out


def commodities(getter, rng) -> dict:
    """Råvaror: D1 per grupp (producent, råvara), D2 för båda tillsammans, dagens hatade."""
    from berserk import backtest as bbt
    from berserk import themes as bth
    from valhall import board as vb
    from valhall import cycle
    from valhall import hate_study as hs
    from valhall import long_paper as vlp
    from valhall import segments as sgm
    spy = _close(getter("SPY", "max"))
    series = []
    for sym in vlp.PRODUCER_SEGMENT:
        df = getter(sym, "max")
        c = _close(df)
        if c is not None:
            series.append(("producent", sym, c, cycle.cycle_frame(c, df["Volume"] if "Volume" in df else None, spy, 80,
                                                                  cykel_base=spy)))
    cache, seen = {}, set()
    for theme in dict.fromkeys(t for s in sgm.SEGMENTS for t in s.themes):
        for sym in bth.drivers(theme):
            if sym not in cache:
                cache[sym] = _close(getter(sym, "max"))
        sym, d = bbt.pick_driver(theme, "2008-01-01", cache)
        if d is None or sym in seen:
            continue
        seen.add(sym)
        series.append(("råvara", f"{theme}:{sym}", d, cycle.cycle_frame(d, None, spy, vb.THEME_NECESSITY.get(theme, 50))))
    d1, rows, cur = {}, [], []
    for grp in ("producent", "råvara"):
        eps_all = []
        for g, name, c, fr in series:
            if g != grp:
                continue
            h = hs.hated(fr)
            eps = [e for e in hs.episodes(h) if e["start"] >= pd.Timestamp(hs.START)]
            eps_all += eps
            rows += hs.delay_rows(c, spy, eps)
            now = hs.current(name, h, hs.episodes(h))
            if now:
                cur.append({**now, "group": grp})
        d1[grp] = hs.km([e["length"] for e in eps_all], [e["censored"] for e in eps_all])
    for r in cur:
        r["typical_m"] = d1[r["group"]].get("median_m")
    return {"series": len(series), "d1": d1, "d2": hs.d2_summary(rows, rng), "current": sorted(cur, key=lambda r: -r["months"])}


def report_lines(res: dict) -> list:
    from valhall import hate_study as hs
    lines = [f"HATETS VARAKTIGHET — spec version {hs.SPEC_VERSION} ({hs.SPEC_PATH})"]
    nd = res.get("nordic") or {}
    for n, v in (nd.get("variants") or {}).items():
        lines.append(f"== NORDEN TOPP {n} — D1 längd (Kaplan–Meier, månader)")
        for sec, k in v["d1"].items():
            if k.get("n"):
                lines.append(f"  {sec:<18} n={k['n']:>3} (pågår {k['ongoing']}) median {k['median_m']} · kvartiler "
                             f"{k['q25_m']}–{k['q75_m']} · kvar 12 mån {k['left_12m']} % · 24 mån {k['left_24m']} % · "
                             f"undervärderad vid start {k['undervalued_share']} %")
        for title, d2 in (("D2", v["d2"]), ("D2 hatad och undervärderad (beskrivande)", v["d2_undervalued"])):
            lines.append(f"== NORDEN TOPP {n} — {title}: n={d2['n']} från {d2['years']} år · rel 2 år "
                         + " · ".join(f"{d // 21} mån {d2['levels'][d]['mean']}/{d2['levels'][d]['median']}" for d in hs.DELAYS))
            for k in ("V6", "V12"):
                x = d2[k]
                lines.append(f"  {k}: snitt {x['mean']} · median {x['median']} · 95 % {x['ci']} → {x['verdict']}")
        lines.append(f"== NORDEN TOPP {n} — D3 hat-bredd")
        for kind, x in v["d3"].items():
            lines.append(f"  {kind} {'GODKÄND' if x['pass'] else 'EJ GODKÄND'} · " + " · ".join(
                f"{h}d n={x[h]['n']} år={x[h]['years']} över={x[h]['mean']} median={x[h]['median']} 95%={x[h]['ci']}"
                for h in hs.BREADTH_H))
    cm = res.get("commodities") or {}
    if cm:
        lines.append("== RÅVAROR — D1 längd (månader)")
        for g, k in cm["d1"].items():
            if k.get("n"):
                lines.append(f"  {g:<10} n={k['n']:>3} (pågår {k['ongoing']}) median {k['median_m']} · kvartiler "
                             f"{k['q25_m']}–{k['q75_m']} · kvar 12 mån {k['left_12m']} % · 24 mån {k['left_24m']} %")
        d2 = cm["d2"]
        lines.append(f"== RÅVAROR — D2: n={d2['n']} från {d2['years']} år · rel 2 år "
                     + " · ".join(f"{d // 21} mån {d2['levels'][d]['mean']}/{d2['levels'][d]['median']}" for d in hs.DELAYS))
        for k in ("V6", "V12"):
            x = d2[k]
            lines.append(f"  {k}: snitt {x['mean']} · median {x['median']} · 95 % {x['ci']} → {x['verdict']}")
    lines.append("== HATADE I DAG")
    for r in (nd.get("current") or []) + (cm.get("current") or []):
        lines.append(f"  {r.get('sector') or r.get('group')}: {r['name']} sedan {r['since']} ({r['months']} mån; "
                     f"typiskt {r.get('typical_m')} mån)")
    return lines


def run(getter=None, api=None, save=None, no_save=False, today=None) -> dict:
    from valhall import hate_study as hs
    rng = np.random.default_rng(hs.SEED)
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    if getter is None:
        from market_prices import ohlcv as getter
    res = {"spec_version": hs.SPEC_VERSION, "spec": hs.SPEC_PATH, "generated": datetime.now(tz=timezone.utc).isoformat()}
    try:
        if api is None:
            from borsdata_api import get_api
            api = get_api()
        ok = getattr(api, "is_configured", True) if api is not None else False
        if callable(ok) and not isinstance(ok, bool):
            ok = ok()
        res["nordic"] = nordic(api, today, rng) if ok else {"error": "Börsdata saknas"}
    except Exception as exc:
        log.exception("Norden fallerade: %s", exc)
        res["nordic"] = {"error": str(exc)}
    try:
        res["commodities"] = commodities(getter, rng)
    except Exception as exc:
        log.exception("Råvarorna fallerade: %s", exc)
        res["commodities"] = {"error": str(exc)}
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
