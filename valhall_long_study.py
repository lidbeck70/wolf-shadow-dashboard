#!/usr/bin/env python3
"""
valhall_long_study.py — långsidans cykeltest (docs/forskning/lang_cykel_spec.md, version 1).

  1. Serierna: producent-ETF:er och de 11 SPDR-fonderna (cykelläget mot SPY), råvarudrivarna (beskrivande)
     och per sektor en likaviktad korg av de 5 största nordiska bolagen efter föregående års börsvärde
     (Börsdata KPI 50) — korgen följer sektorfondens händelser.
  2. Avnoteringskontrollen: har Börsdatas lista med avnoterade bolag? Bara då avgör bolagsgruppen.
  3. valhall.long_study.run_study → L1–L5 mot de låsta kriterierna.

Körs manuellt i Actions (valhall-long-study.yml). Resultatet i loggen och i Gisten (valhall_long_study.json)
om GITHUB_TOKEN finns. Ändrar inga regler. Env: BORSDATA_API_KEY.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("valhall_long_study")

BLOB = "valhall_long_study.json"
PRODUCERS = ("XOP", "XES", "FCG", "XME", "GDX", "GDXJ", "SIL", "COPX", "URA", "PICK", "SLX", "WOOD", "MOO",
             "REMX", "LIT")
TOP_N, CANDIDATES = 5, 8
MCAP_KPI = 50
START_YEAR = 2008


def _close(df):
    from valhall_event_study import _close as c
    return c(df)


def market_series(getter) -> tuple:
    """(serier, sektorhändelser {segment: {inträde: [datum]}}) för producent, sektor och råvara."""
    from berserk import backtest as bbt
    from berserk import themes as bth
    from valhall import board as vb
    from valhall import cycle
    from valhall import long_study as lst
    from valhall import segments as sgm
    spy = _close(getter("SPY", "max"))
    out, sector_events = [], {}

    def add(name, group, df, kind, base, necessity=50):
        c = _close(df) if isinstance(df, pd.DataFrame) else df
        if c is None:
            log.warning("%s saknas", name)
            return None
        vol = df["Volume"] if isinstance(df, pd.DataFrame) and "Volume" in df else None
        fr = cycle.cycle_frame(c, vol, spy, necessity, cykel_base=base)
        ev = lst.entry_events(fr, kind)
        out.append(lst.series_from_close(name, group, c, ev))
        return ev

    for sym in PRODUCERS:
        add(sym, lst.PRODUCENT, getter(sym, "max"), cycle.RAVARA, spy, 80)
    for s in sgm.SEGMENTS:
        ev = add(s.spdr, lst.SEKTOR, getter(s.spdr, "max"), cycle.SEGMENT, spy, s.necessity)
        if ev is not None:
            sector_events[s.key] = ev
    cache, seen = {}, set()
    for theme in dict.fromkeys(t for s in sgm.SEGMENTS for t in s.themes):
        for sym in bth.drivers(theme):
            if sym not in cache:
                cache[sym] = _close(getter(sym, "max"))
        sym, d = bbt.pick_driver(theme, f"{START_YEAR}-01-01", cache)
        if d is None or sym in seen:
            continue
        seen.add(sym)
        add(f"{theme}:{sym}", lst.RAVARA, d, cycle.RAVARA, None, vb.THEME_NECESSITY.get(theme, 50))
    return out, sector_events


def _stem(name: str) -> str:
    """Bolaget utan aktieslag ('Volvo B' → 'volvo') — A- och B-aktien är samma bolag."""
    return re.sub(r"\s+(ser\.?\s*)?[A-D]$", "", str(name or "").strip(), flags=re.I).lower()


def top_members(mcap: dict, info: dict, segment_of: dict, years) -> dict:
    """{segment: {år: [insId]}} — de CANDIDATES största per sektor efter föregående års börsvärde i kronor
    (en aktie per bolag, B-aktien före A). mcap = {insId: {år: värde}}."""
    from valhall import nordic_index as ni
    out: dict = {}
    for y in years:
        by_seg: dict = {}
        for iid, hist in mcap.items():
            v = hist.get(y - 1)
            seg = segment_of.get(iid)
            if v is None or not seg or v <= 0:
                continue
            i = info[iid]
            fx = ni.FX_TO_SEK.get(str(i.get("reportCurrency") or i.get("stockPriceCurrency") or "SEK").upper(), 1.0)
            by_seg.setdefault(seg, []).append((v * fx, str(i.get("ticker") or "").upper().endswith("B"), iid))
        for seg, rows in by_seg.items():
            picked, stems = [], set()
            for _v, _b, iid in sorted(rows, key=lambda r: (-r[0], not r[1])):
                st = _stem(info[iid].get("name"))
                if st in stems:
                    continue
                stems.add(st)
                picked.append(iid)
                if len(picked) == CANDIDATES:
                    break
            out.setdefault(seg, {})[y] = picked
    return out


def bolag_series(api, sector_events: dict, today) -> tuple:
    """(serier, avnoteringskontroll) för bolagsgruppen. Tomt utan Börsdata."""
    import markets as _markets
    from valhall import long_study as lst
    from valhall import segments as sgm
    table = _markets.load(api=api)
    nordic_ids = _markets.nordic_stock_ids(table)
    info = {i["insId"]: i for i in api.get_instruments() if i.get("marketId") in nordic_ids and i.get("insId")}
    last = {}
    for p in api.get_last_stockprices() or []:
        iid = p.get("i", p.get("insId"))
        if iid is not None:
            last[iid] = p.get("d")
    check = lst.delisted_check(set(info), last, today)
    log.info("Avnoteringskontroll: %d nordiska instrument, %d med senaste kurs äldre än %d dagar → %s",
             check["nordic"], check["old"], lst.DELISTED_DAYS, "avgör" if check["covered"] else "beskrivande")
    segment_of = {iid: sgm.segment_for(i.get("sectorId"), i.get("branchId")) for iid, i in info.items()}
    ids = [iid for iid, seg in segment_of.items() if seg]
    raw = api.get_kpi_history_batch(ids, MCAP_KPI, "year", "mean", max_count=20)
    mcap = {iid: {int(v["y"]): float(v["v"]) for v in vals if v.get("y") is not None and v.get("v") is not None}
            for iid, vals in raw.items()}
    years = range(START_YEAR - 2, pd.Timestamp(today).year + 1)
    cands = top_members(mcap, info, segment_of, years)
    prices = {}
    for iid in sorted({i for per in cands.values() for ys in per.values() for i in ys}):
        try:
            df = api.get_stockprices_df(int(iid), max_count=5040)
        except Exception as exc:
            log.warning("kurser %s: %s", iid, exc)
            df = None
        c = _close(df) if df is not None and len(df) else None
        if c is not None:
            prices[iid] = c
    out = []
    for s in sgm.SEGMENTS:
        if s.key not in sector_events or s.key not in cands:
            continue
        members = {}
        for y, ids_y in cands[s.key].items():
            have = [i for i in ids_y if i in prices and ((prices[i].index.year == y).any())]
            members[y] = have[:TOP_N]
        closes = {i: prices[i] for ys in members.values() for i in ys}
        if not closes:
            continue
        fwd = {h: lst.basket_forward(closes, members, h) for h in lst.HORIZONS}
        out.append({"name": f"Norden {s.label}", "group": lst.BOLAG, "fwd": fwd, "mae": None,
                    "events": sector_events[s.key],
                    "members": {str(y): [str(info[i].get("ticker")) for i in ys] for y, ys in members.items()}})
    return out, check


def run(getter=None, api=None, save=None, no_save=False, today=None, n_boot=None) -> dict:
    from valhall import long_study as lst
    if getter is None:
        from market_prices import ohlcv as getter
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    series, sector_events = market_series(getter)
    check = {"nordic": 0, "old": 0, "covered": False, "error": None}
    try:
        if api is None:
            from borsdata_api import get_api
            api = get_api()
        ok = getattr(api, "is_configured", True) if api is not None else False
        if callable(ok) and not isinstance(ok, bool):
            ok = ok()                                  # BorsdataAPI har en egenskap; testernas falska API en metod
        if ok:
            bs, check = bolag_series(api, sector_events, today)
            series += bs
        else:
            check["error"] = "Börsdata saknas"
    except Exception as exc:
        log.exception("Bolagsgruppen fallerade: %s", exc)
        check["error"] = str(exc)
    deciding = (lst.PRODUCENT, lst.SEKTOR) + ((lst.BOLAG,) if check.get("covered") else ())
    log.info("Serier: %s", ", ".join(f"{g} {sum(1 for s in series if s['group'] == g)}" for g in lst.GROUPS))
    result = lst.run_study(series, deciding=deciding, n_boot=n_boot or lst.N_BOOT)
    result["generated"] = datetime.now(tz=timezone.utc).isoformat()
    result["delisted_check"] = check
    result["series"] = [{"name": s["name"], "group": s["group"], **({"members": s["members"]} if "members" in s else {})}
                        for s in series]
    for line in lst.report_lines(result):
        log.info("%s", line)
    for s in result["series"]:
        if "members" in s:
            picks = [f"{y}: {','.join(m)}" for y, m in s["members"].items() if int(y) % 4 == 0]
            log.info("  korg %s: %s", s["name"], " · ".join(picks))
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
