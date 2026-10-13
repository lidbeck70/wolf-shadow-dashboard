"""
valhall/swing.py — swingdelen av Valhalls rutor: två bästa Viking Nine-affärer i Norden per grönt
segment och råvarutema, plus segmentets nischsignaler.

All swing görs i Norden (där backtesten visat kant). Grunden är Viking Nine (viking_screen.scan,
oförändrad); rutan visar affärer bara när segmentet (nordiska sektorindexet) respektive temat är GRÖNT
enligt cykelmotorn. Universumet är de likvida nordiska aktierna (snittomsättning minst 1 milj kr/dag de
senaste 60 dagarna).

Nischen är segmentets särdrag — i fas 2 samlas de befintliga signalerna per segment (BERSERK för
råvaruteman, Insider Norden för alla). Fas 3 prövar vilken nisch som fungerar bäst i vilket segment.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import pandas as pd

from berserk import universe as buv
from valhall import cycle
from valhall import nordic_index as ni
from valhall import segments as sgm

N_PICKS = 2
SEGMENT_CANDIDATES, THEME_CANDIDATES = 8, 5     # Viking Nine steg 2 per segment / tema (steg 1 rankar på momentum)
GOOD = ("GOLDEN TICKET", "READY", "DEVELOPING")


def nordic_universe(instruments: Iterable[dict], nordic_ids: set, to_yf: Callable) -> dict:
    """{ticker: {segment, currency, name}} för nordiska aktier med känd sektor."""
    out = {}
    for i in instruments:
        if i.get("marketId") not in nordic_ids or not i.get("ticker"):
            continue
        seg = sgm.segment_for(i.get("sectorId"), i.get("branchId"))
        if seg:
            out[to_yf(i.get("ticker"), i.get("marketId"))] = {
                "segment": seg, "currency": str(i.get("stockPriceCurrency") or "SEK").upper(),
                "name": str(i.get("name") or "")}
    return out


def liquid(data: dict, universe: dict, min_turnover: float = ni.MIN_TURNOVER_SEK) -> set:
    """Tickers vars snittomsättning de senaste 60 dagarna är minst min_turnover kr."""
    out = set()
    for t, df in (data or {}).items():
        if df is None or len(df) == 0 or "Volume" not in df or t not in universe:
            continue
        fx = ni.FX_TO_SEK.get(universe[t]["currency"], 1.0)
        tv = (df["Close"] * df["Volume"] * fx).tail(ni.TURNOVER_DAYS).mean()
        if pd.notna(tv) and tv >= min_turnover:
            out.add(t)
    return out


def compact(row: dict, segment: str, theme: Optional[str] = None, name: str = "") -> Optional[dict]:
    """Viking Nine-raden i JSON-form (viking_screen.log_entry) + segment/tema."""
    import viking_screen as vs
    e = vs.log_entry(row, source="valhall")
    if e is None:
        return None
    e.update(segment=segment, theme=theme, name=name, category=row.get("category"))
    for k in ("timestamp", "label", "source"):
        e.pop(k, None)
    return e


def pick(rows: list, n: int = N_PICKS) -> list:
    """De n bästa: kategori (GOLDEN TICKET → DEVELOPING), sedan Nine. Avvisade och fel räknas inte."""
    import viking_screen as vs
    ok = [r for r in vs.results(rows) if r.get("nine") is not None and r.get("category") in GOOD]
    return ok[:n]


def _closes_from(data: dict) -> Callable:
    def get(tickers):
        return {t: data[t]["Close"] for t in tickers if t in data and data[t] is not None and len(data[t])}
    return get


def candidates(board: dict, universe: dict, data: dict, scan_fn: Optional[Callable] = None,
               min_turnover: float = ni.MIN_TURNOVER_SEK, n: int = N_PICKS, **kw) -> dict:
    """{segments: {key: {status, picks, scanned}}, themes: {tema: {...}}}. Bara gröna rutor skannas.
    scan_fn = viking_screen.scan (utbytbar i tester); kw går vidare till evaluate_ticker (getter m.m.)."""
    if scan_fn is None:
        import viking_screen as vs
        scan_fn = vs.scan
    liq = liquid(data, universe, min_turnover)
    closes_getter = _closes_from(data)
    kw.setdefault("held", {})                                   # inga öppna positioner styr kartan
    out = {"segments": {}, "themes": {}, "liquid": len(liq)}
    for r in board.get("segments") or []:
        st = (r.get("nordic") or {}).get("status", cycle.NODATA)
        entry = {"status": st, "picks": [], "scanned": 0}
        tickers = sorted(t for t in liq if universe[t]["segment"] == r["key"])
        if st == cycle.GRON and tickers:
            res = scan_fn(tickers, closes_getter=closes_getter, max_candidates=SEGMENT_CANDIDATES, **kw)
            entry["scanned"] = len(res.get("rows") or [])
            entry["picks"] = [c for c in (compact(x, r["key"], None, universe[x["ticker"]]["name"])
                                          for x in pick(res.get("rows") or [], n)) if c]
        out["segments"][r["key"]] = entry
        for t in r["themes"]:
            tst = t["state"].get("status", cycle.NODATA)
            te = {"status": tst, "picks": [], "scanned": 0}
            members = sorted(x for x in liq if buv.theme_of(x) == t["theme"])
            if tst == cycle.GRON and members:
                res = scan_fn(members, closes_getter=closes_getter, max_candidates=THEME_CANDIDATES, **kw)
                te["scanned"] = len(res.get("rows") or [])
                te["picks"] = [c for c in (compact(x, r["key"], t["theme"], universe.get(x["ticker"], {}).get("name", ""))
                                           for x in pick(res.get("rows") or [], n)) if c]
            out["themes"][t["theme"]] = te
    return out


def niche(universe: dict, berserk_blob: Optional[dict] = None, insider_blob: Optional[dict] = None) -> dict:
    """Befintliga nischsignaler per segment: BERSERK (KÖP/BEVAKA i råvaruteman) och Insider Norden-kluster."""
    out = {s.key: {"berserk": [], "insider": []} for s in sgm.SEGMENTS}
    for r in (berserk_blob or {}).get("rows") or []:
        seg = (universe.get(r.get("ticker")) or {}).get("segment") or sgm.segment_of_theme(r.get("theme"))
        if seg and r.get("status") in ("KÖP", "BEVAKA") and buv.region_of(r.get("ticker", "")) == "Norden":
            out[seg]["berserk"].append({k: r.get(k) for k in ("ticker", "status", "setup", "theme", "close", "stop")})
    for c in (insider_blob or {}).get("clusters") or []:
        seg = (universe.get(c.get("ticker")) or {}).get("segment")
        if seg:
            out[seg]["insider"].append({k: c.get(k) for k in ("ticker", "name", "score", "insiders", "role", "amount",
                                                            "cluster_avg", "price_now", "gate", "trigger")})
    for v in out.values():
        v["insider"].sort(key=lambda x: -(x.get("score") or 0))
    return out
