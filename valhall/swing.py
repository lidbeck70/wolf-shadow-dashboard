"""
valhall/swing.py — swingdelen av Valhalls rutor: de två bästa affärerna i Norden per segment och
råvarutema, med den strategi som har kant i segmentet, plus segmentets nischsignaler.

All swing görs i Norden. Styrning A (fas 3a, backtest 2008–idag): cykelstatusen VISAS men spärrar inte
— gröna grinden hjälpte 2008–2020 och skadade 2021–. I stället väljs strategin per segment:

  Viking Nine          Teknik, Finans — kant i båda perioderna (viking_screen.scan, oförändrad)
  BERSERK              Energi, Material och alla råvaruteman — starkast i Norden (berserk_scan.json:s KÖP)
  ingen swing          Kraftförsörjning — negativ i båda perioderna
  Viking Nine (prövas) övriga segment — ingen säker kant; papperskontona (valhall/paper.py) avgör

Universumet för Viking Nine är de likvida nordiska aktierna (snittomsättning minst 1 milj kr/dag de
senaste 60 dagarna). Nischen är de befintliga signalerna per segment (BERSERK, Insider Norden).
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import pandas as pd

from berserk import universe as buv
from valhall import cycle
from valhall import nordic_index as ni
from valhall import segments as sgm

N_PICKS = 2
SEGMENT_CANDIDATES = 8           # Viking Nine steg 2 per segment (steg 1 rankar på momentum)
GOOD = ("GOLDEN TICKET", "READY", "DEVELOPING")

VIKING, BERSERK, NONE, TEST = "Viking Nine", "BERSERK", "ingen swing", "Viking Nine (prövas)"
SWING_BY_SEGMENT = {"technology": VIKING, "financials": VIKING, "energy": BERSERK, "materials": BERSERK,
                    "utilities": NONE}


def swing_of(segment: str) -> str:
    """Segmentets swingstrategi (styrning A). Övriga segment: Viking Nine som prövas i papperskontot."""
    return SWING_BY_SEGMENT.get(segment, TEST)


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


def berserk_picks(berserk_blob: Optional[dict], segment: str, theme: Optional[str] = None,
                  n: int = N_PICKS) -> list:
    """BERSERK:s KÖP i Norden för segmentet (eller temat) — sällsyntaste setupen först (signals.PRIORITY)."""
    from berserk import signals as bsg
    rows = [r for r in (berserk_blob or {}).get("rows") or []
            if r.get("status") == "KÖP" and not r.get("error") and buv.region_of(r.get("ticker") or "") == "Norden"
            and (r.get("theme") == theme if theme else sgm.segment_of_theme(r.get("theme")) == segment)]
    rows.sort(key=lambda r: (-bsg.PRIORITY.get(r.get("setup"), 0), r.get("ticker") or ""))
    return [{**{k: r.get(k) for k in ("ticker", "setup", "theme", "close", "stop", "atr", "risk_pct", "date")},
             "segment": segment, "strategy": BERSERK} for r in rows[:n]]


def candidates(board: dict, universe: dict, data: dict, scan_fn: Optional[Callable] = None,
               min_turnover: float = ni.MIN_TURNOVER_SEK, n: int = N_PICKS, berserk_blob: Optional[dict] = None,
               **kw) -> dict:
    """{segments: {key: {status, strategy, picks, scanned}}, themes: {tema: {...}}}. Styrning A: strategin per
    segment (swing_of), cykelstatusen följer med men spärrar inte. Råvaruteman får BERSERK:s KÖP.
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
        mode = swing_of(r["key"])
        entry = {"status": st, "strategy": mode, "picks": [], "scanned": 0}
        tickers = sorted(t for t in liq if universe[t]["segment"] == r["key"])
        if mode in (VIKING, TEST) and tickers:
            res = scan_fn(tickers, closes_getter=closes_getter, max_candidates=SEGMENT_CANDIDATES, **kw)
            entry["scanned"] = len(res.get("rows") or [])
            entry["picks"] = [c for c in (compact(x, r["key"], None, universe[x["ticker"]]["name"])
                                          for x in pick(res.get("rows") or [], n)) if c]
            for c in entry["picks"]:
                c["strategy"] = mode
        elif mode == BERSERK:
            entry["picks"] = berserk_picks(berserk_blob, r["key"], n=n)
        out["segments"][r["key"]] = entry
        for t in r["themes"]:
            out["themes"][t["theme"]] = {"status": t["state"].get("status", cycle.NODATA), "strategy": BERSERK,
                                         "picks": berserk_picks(berserk_blob, r["key"], t["theme"], n), "scanned": 0}
    return out


def complex_candidates(board: dict, universe: dict, data: dict, complex_map: dict, getter: Optional[Callable] = None,
                       today=None, now=None, n: int = N_PICKS, scan_fn: Optional[Callable] = None,
                       min_turnover: float = ni.MIN_TURNOVER_SEK, **kw) -> dict:
    """Viking Nine per råvarukomplex (K4): komplexprofilen (complex_profiles) genom omslaget complex_nine över de
    likvida nordiska bolag vars komplex (complex_map = valhall.complexes.member_complexes) är rutans. Körs
    BREDVID BERSERK i Energi/Material (beslut 8) — papperet avgör. {complexes: {key: {status, turn, strategy,
    profile, picks, scanned, driver}}, liquid}. scan_fn = complex_nine.scan_complex (utbytbar i tester)."""
    import complex_nine as cn
    import complex_profiles as cp
    scan = scan_fn or cn.scan_complex
    liq = liquid(data, universe, min_turnover)
    closes_getter = _closes_from(data)
    kw.setdefault("held", {})
    states = {c["key"]: (c.get("state") or {}) for c in board.get("complexes") or [] if isinstance(c, dict)}
    out = {"complexes": {}, "liquid": len(liq)}
    for p in cp.complexes():
        st = states.get(p.key) or {}
        tickers = sorted(t for t in liq if (complex_map or {}).get(t) == p.key)
        entry = {"status": st.get("status", cycle.NODATA), "turn": st.get("turn"), "strategy": f"{VIKING} · {p.label}",
                 "profile": p.key, "picks": [], "scanned": 0, "universe": len(tickers), "driver": None}
        if tickers:
            res = scan(p, tickers, closes_getter=closes_getter, max_candidates=SEGMENT_CANDIDATES, getter=getter,
                       today=today, now=now, **kw)
            entry["scanned"] = len(res.get("rows") or [])
            entry["driver"] = res.get("driver")
            for x in pick(res.get("rows") or [], n):
                u = universe.get(x["ticker"], {})
                c = compact(x, u.get("segment"), None, u.get("name", ""))
                if c:
                    c.update(strategy=entry["strategy"], profile=p.key, complex=p.key)
                    entry["picks"].append(c)
        out["complexes"][p.key] = entry
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
