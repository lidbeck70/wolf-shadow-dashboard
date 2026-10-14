"""
Valhall fas 2a/4a — likviditetsfilter (dag för dag), swingkandidater per segment enligt styrning A
(Viking Nine utbytbar, BERSERK ur skanningen), nischsignaler och den schemalagda körningen. Syntetiska data — inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valhall import cycle  # noqa: E402
from valhall import nordic_index as ni  # noqa: E402
from valhall import swing  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = pd.bdate_range(end="2026-10-09", periods=300)


def _cv(close=100.0, volume=1e5, n=300, drift=0.0):
    c = close * (1 + drift) ** np.arange(n)
    return pd.DataFrame({"Close": c, "Volume": np.full(n, float(volume))}, index=IDX[-n:])


# ── Likviditetsfiltret ──────────────────────────────────────────────────────
def test_liquid_mask_is_point_in_time():
    tv = pd.DataFrame({"A": np.r_[np.full(100, 5e5), np.full(200, 5e6)]}, index=IDX)
    m = ni.liquid_mask(tv, 1e6, days=60)
    assert not m["A"].iloc[100] and not m["A"].iloc[105]                       # snittet FÖRE dagen räknas
    assert m["A"].iloc[115] and m["A"].iloc[-1]                                 # 45 × 0,5 + 15 × 5 milj / 60


def test_index_ignores_illiquid_members():
    a = _cv(100, 1e6, drift=0.001)                                            # 100 milj kr/dag, stiger
    b = _cv(100, 1e6, drift=-0.001)                                           # likvid, faller
    c = _cv(10, 1e-6, drift=0.05)                                             # illikvid, rusar
    turnover = {k: d["Close"] * d["Volume"] for k, d in (("A", a), ("B", b), ("C", c))}
    closes = {k: d["Close"] for k, d in (("A", a), ("B", b), ("C", c))}
    with_filter = ni.equal_weight_index(closes, min_members=2, turnover=turnover)
    without = ni.equal_weight_index(closes, min_members=2)
    assert without.iloc[-1] > 1000 and with_filter.iloc[-1] == pytest.approx(100, rel=0.02)   # C styr inte


def test_build_indices_keep_and_liquid_count():
    data = {"A.ST": _cv(100, 1e5), "B.ST": _cv(50, 1e5), "C.ST": _cv(20, 1e5), "D.ST": _cv(5, 10)}
    mem = {s.key: [] for s in ni.sgm.SEGMENTS}
    mem["industrials"] = ["A.ST", "B.ST", "C.ST", "D.ST"]
    keep = {}
    out = ni.build_indices(mem, {t: "SEK" for t in data}, lambda ts, p: {t: data[t] for t in ts}, keep=keep)
    assert out["industrials"]["liquid"] == 3 and out["industrials"]["with_data"] == 4
    assert out["industrials"]["index"] is not None and out[ni.ALL]["members"] == 4 and set(keep) == set(data)
    assert out["energy"]["index"] is None


def test_closes_volumes_batch(monkeypatch):
    import market_prices as mp
    cols = pd.MultiIndex.from_product([["A", "B"], ["Open", "Close", "Volume"]])
    raw = pd.DataFrame(np.ones((3, 6)), index=IDX[:3], columns=cols)
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **k: raw)
    mp.clear()
    out = mp.closes_volumes(["A", "B", "A"], "max")
    assert set(out) == {"A", "B"} and list(out["A"].columns) == ["Close", "Volume"]
    mp.clear()


# ── Universum och kandidater ────────────────────────────────────────────────
def test_nordic_universe_and_liquid():
    inst = [{"ticker": "VOLV B", "marketId": 1, "sectorId": 5, "branchId": 23, "stockPriceCurrency": "SEK",
             "name": "Volvo"},
            {"ticker": "EQNR", "marketId": 2, "sectorId": 3, "stockPriceCurrency": "NOK", "name": "Equinor"},
            {"ticker": "X", "marketId": 9, "sectorId": 3}]
    uni = swing.nordic_universe(inst, {1, 2}, lambda t, m: t.replace(" ", "-") + (".ST" if m == 1 else ".OL"))
    assert uni == {"VOLV-B.ST": {"segment": "industrials", "currency": "SEK", "name": "Volvo"},
                   "EQNR.OL": {"segment": "energy", "currency": "NOK", "name": "Equinor"}}
    data = {"VOLV-B.ST": _cv(250, 1e6), "EQNR.OL": _cv(250, 100)}
    assert swing.liquid(data, uni) == {"VOLV-B.ST"}


def _row(ticker, category="GOLDEN TICKET", passed=9, weighted=90.0):
    nine = SimpleNamespace(passed=passed, weighted=weighted, layer_passed=lambda layer: 1, sector_etf="XLI",
                           get=lambda k: None)
    pos = SimpleNamespace(stop=90.0, atr=2.0, shares=10, risk_amount=100.0)
    d = SimpleNamespace(execution_passed=6, execution_total=7, entry=100.0, position=pos, rr=2.5, status="GO",
                        trigger_date="2026-10-09", reasons=["ok"], checks=[])
    return {"ticker": ticker, "error": None, "nine": nine, "decision": d, "category": category, "sector_busy": None}


def _board(nordic_status=cycle.GRON, theme_status=cycle.GRON):
    segs = []
    for s in ni.sgm.SEGMENTS:
        segs.append({"key": s.key, "label": s.label, "nordic": {"status": nordic_status if s.key == "industrials"
                                                                 else cycle.ROD},
                     "themes": [{"theme": t, "label": t, "state": {"status": theme_status if t == "koppar"
                                                                   else cycle.ROD}} for t in s.themes]})
    return {"segments": segs}


def test_swing_of_control_a():
    assert swing.swing_of("technology") == swing.swing_of("financials") == swing.VIKING
    assert swing.swing_of("energy") == swing.swing_of("materials") == swing.BERSERK
    assert swing.swing_of("utilities") == swing.NONE and swing.swing_of("industrials") == swing.TEST


BERSERK_BLOB = {"rows": [{"ticker": "BOL.ST", "status": "KÖP", "setup": "S1 Divergens", "theme": "koppar", "close": 300.0,
                          "stop": 280.0, "atr": 8.0, "date": "2026-10-09"},
                         {"ticker": "LUMI.ST", "status": "KÖP", "setup": "S2 Stark", "theme": "koppar"},
                         {"ticker": "FCX", "status": "KÖP", "setup": "S1 Divergens", "theme": "koppar"},
                         {"ticker": "EQNR.OL", "status": "BEVAKA", "setup": "S1 Divergens", "theme": "olja"}]}


def test_candidates_follow_strategy_per_segment():
    uni = {"VOLV-B.ST": {"segment": "industrials", "currency": "SEK", "name": "Volvo"},
           "SAND.ST": {"segment": "industrials", "currency": "SEK", "name": "Sandvik"},
           "ATCO-A.ST": {"segment": "industrials", "currency": "SEK", "name": "Atlas"},
           "BOL.ST": {"segment": "materials", "currency": "SEK", "name": "Boliden"},
           "EQNR.OL": {"segment": "energy", "currency": "NOK", "name": "Equinor"},
           "FORTUM.HE": {"segment": "utilities", "currency": "EUR", "name": "Fortum"}}
    data = {t: _cv(100, 1e6) for t in uni}
    calls = []

    def scan(tickers, closes_getter=None, max_candidates=0, **kw):
        calls.append((tuple(tickers), max_candidates, kw.get("held")))
        rows = [_row(t, "READY" if t == "SAND.ST" else "GOLDEN TICKET", weighted=80 + len(t)) for t in tickers]
        rows.append({**_row("BAD.ST", "REJECTED"), "nine": None})
        assert set(closes_getter(tickers)) == set(tickers)
        return {"rows": rows}

    from berserk import signals as bsg
    out = swing.candidates(_board(cycle.ROD), uni, data, scan_fn=scan, berserk_blob=BERSERK_BLOB)
    ind = out["segments"]["industrials"]
    assert ind["status"] == cycle.ROD and ind["strategy"] == swing.TEST and len(ind["picks"]) == 2   # röd spärrar inte
    assert [p["ticker"] for p in ind["picks"]] == ["ATCO-A.ST", "VOLV-B.ST"]           # GOLDEN före READY
    assert ind["picks"][0]["name"] == "Atlas" and ind["picks"][0]["entry"] == 100.0 and ind["picks"][0]["stop"] == 90.0
    assert ind["picks"][0]["strategy"] == swing.TEST
    assert out["segments"]["utilities"]["picks"] == [] and out["segments"]["utilities"]["scanned"] == 0
    mat = out["segments"]["materials"]
    assert mat["strategy"] == swing.BERSERK and mat["scanned"] == 0                     # ingen Viking-skanning
    expect = sorted(["BOL.ST", "LUMI.ST"], key=lambda t: -bsg.PRIORITY.get(
        {"BOL.ST": "S1 Divergens", "LUMI.ST": "S2 Stark"}[t], 0))
    assert [p["ticker"] for p in mat["picks"]] == expect                                # FCX är inte Norden
    assert out["segments"]["energy"]["picks"] == []                                     # BEVAKA är inget köp
    assert [p["ticker"] for p in out["themes"]["koppar"]["picks"]] == expect
    assert out["themes"]["koppar"]["strategy"] == swing.BERSERK and out["themes"]["olja"]["picks"] == []
    assert all(c[2] == {} for c in calls) and {c[1] for c in calls} == {swing.SEGMENT_CANDIDATES}
    assert all(set(c[0]) <= {"VOLV-B.ST", "SAND.ST", "ATCO-A.ST"} for c in calls)      # bara Viking-segment


def test_niche_signals_by_segment():
    uni = {"BOL.ST": {"segment": "materials", "currency": "SEK", "name": "Boliden"},
           "INVE-B.ST": {"segment": "financials", "currency": "SEK", "name": "Investor"}}
    berserk = {"rows": [{"ticker": "BOL.ST", "status": "KÖP", "setup": "S1 Divergens", "theme": "koppar"},
                        {"ticker": "FCX", "status": "KÖP", "theme": "koppar"},
                        {"ticker": "LUMI.ST", "status": "BEVAKA", "theme": "koppar"}]}
    insider = {"clusters": [{"ticker": "INVE-B.ST", "score": 6}, {"ticker": "BOL.ST", "score": 8},
                            {"ticker": "OKÄND.ST", "score": 9}]}
    n = swing.niche(uni, berserk, insider)
    assert [x["ticker"] for x in n["materials"]["berserk"]] == ["BOL.ST", "LUMI.ST"]       # FCX är inte Norden
    assert n["financials"]["insider"][0]["ticker"] == "INVE-B.ST" and n["materials"]["insider"][0]["score"] == 8
    assert swing.niche(uni)["energy"] == {"berserk": [], "insider": []}


# ── Den schemalagda körningen ───────────────────────────────────────────────
def test_valhall_scan_run(monkeypatch):
    import markets
    import valhall_scan as vsn
    inst = [{"ticker": t, "marketId": 1, "sectorId": 5, "stockPriceCurrency": "SEK", "name": t}
            for t in ("A", "B", "C")]
    monkeypatch.setattr(markets, "load", lambda api=None, **k: {})
    monkeypatch.setattr(markets, "nordic_stock_ids", lambda table=None: {1})
    monkeypatch.setattr(markets, "to_yf", lambda t, m=None, table=None: t + ".ST")
    data = {t + ".ST": _cv(100, 1e5, n=300) for t in ("A", "B", "C")}
    saved = {}
    out = vsn.run(api=SimpleNamespace(get_instruments=lambda: inst),
                  data_getter=lambda ts, p: {t: data[t] for t in ts if t in data}, getter=lambda t, p: None,
                  load=lambda f, fb: fb, save=lambda f, d: saved.__setitem__(f, d) or True,
                  scan_fn=lambda *a, **k: {"rows": []}, nordic_market=None, sources={},
                  long_fundamentals=lambda ts: {})
    p = out["payload"]
    assert out["saved"] and vsn.BLOB in saved and len(p["board"]["segments"]) == 11
    assert p["nordic"]["industrials"]["members"] == 3 and "swing" in p and "niche" in p
    assert "buys" in p and p["sources"] == {"ca_runs": 0, "screens": False, "sheets": []}
    assert p["long"]["scanned"] == 3 and set(p["long"]["segments"]) == {s.key for s in ni.sgm.SEGMENTS}
    assert set(p["wolf"]["segments"]) == {s.key for s in ni.sgm.SEGMENTS}             # Wolf-raderna finns alltid
    assert p["wolf"]["arc_live"] is False and p["wolf"]["momentum_live"] is False      # listorna saknas → tomma
    lines = vsn.summary(p)
    assert len(lines) >= 13 and any(x.startswith("Källor") for x in lines) and lines[-1].startswith("Papperskonton")
    assert vsn.PAPER_BLOB in saved and p["paper"]["by_strategy"] == [] and saved[vsn.PAPER_BLOB]["orders"] == []
    import json
    json.dumps(saved[vsn.BLOB], default=str)                                   # går att spara som JSON


def test_scan_workflow():
    src = open(os.path.join(ROOT, ".github", "workflows", "valhall-scan.yml"), encoding="utf-8").read()
    for text in ('cron: "47 16 * * 1-5"', "schedule_gate.sh valhall-scan.yml", "python valhall_scan.py",
                 "secrets.GIST_TOKEN", "secrets.BORSDATA_API_KEY", "workflow_dispatch"):
        assert text in src, text
