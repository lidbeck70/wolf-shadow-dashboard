"""
Viking Nine över hela registret (viking_nine_scan.py): universumen ur registret, raderna i loggens form med namn
och segment, REJECTED bort, ordningen, felvägen, bloben och arbetsflödessteget — och Thor läser raderna.
Viking Nine-motorn är fejkad; inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import viking_nine_scan as vns  # noqa: E402
from thor import candidates as tc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _row(ticker, category, passed=9, weighted=90.0):
    nine = SimpleNamespace(passed=passed, weighted=weighted, layer_passed=lambda layer: 1)
    pos = SimpleNamespace(stop=90.0, atr=2.0, shares=10, risk_amount=100.0)
    d = SimpleNamespace(position=pos, execution_passed=7, execution_total=7, entry=100.0, rr=2.5, status="GO",
                        trigger_date="2026-10-20", reasons=["ok"])
    return {"ticker": ticker, "nine": nine, "decision": d, "category": category, "sector_busy": None}


def _scan(tickers, max_candidates=40, **kw):
    rows = [_row(t, cat, w) for t, cat, w in (("AAA.ST", "READY", 8), ("BBB.ST", "GOLDEN TICKET", 9),
                                              ("CCC.ST", "REJECTED", 5), ("LLY", "GOLDEN TICKET", 9))
            if t in tickers]
    return {"rows": rows, "funnel": {"universe": len(tickers), "data": len(tickers), "passed": 2, "momentum": {}}}


MEMBERS = {"norden": {"AAA.ST": ("Alfa", "industrials"), "BBB.ST": ("Beta", "healthcare"), "CCC.ST": ("Gamma", None)},
           "usa": {"LLY": ("Eli Lilly", "healthcare")}}


def test_scan_market_keeps_good_rows_in_log_form():
    res = vns.scan_market("norden", MEMBERS["norden"], scan_fn=_scan)
    assert [r["ticker"] for r in res["rows"]] == ["BBB.ST", "AAA.ST"]             # GOLDEN före READY, REJECTED bort
    b = res["rows"][0]
    assert b["category"] == "GOLDEN TICKET" and b["name"] == "Beta" and b["segment"] == "healthcare"
    assert b["ovtlyr_nine"] == "9/9" and b["entry"] == 100.0 and b["stop"] == 90.0 and b["market"] == "norden"
    assert "momentum" not in res["funnel"] and vns.scan_market("usa", {}, scan_fn=_scan)["rows"] == []


def test_run_saves_both_markets_and_survives_a_failure():
    saved = {}
    out = vns.run(members=MEMBERS, scan_fn=_scan, save=lambda n, d: saved.__setitem__(n, d) or True)
    p = out["payload"]
    assert out["saved"] and vns.BLOB_NAME in saved and set(p["markets"]) == {"norden", "usa"}
    assert [r["ticker"] for r in p["rows"]] == ["BBB.ST", "AAA.ST", "LLY"] and p["markets"]["usa"]["error"] is None

    def boom(tickers, **kw):
        if "LLY" in tickers:
            raise RuntimeError("Yahoo nere")
        return _scan(tickers, **kw)
    bad = vns.run(members=MEMBERS, scan_fn=boom, dry_run=True)["payload"]
    assert bad["markets"]["usa"]["error"] == "Yahoo nere" and [r["ticker"] for r in bad["rows"]] == ["BBB.ST", "AAA.ST"]


def test_thor_reads_the_scan_rows():
    payload = vns.run(members=MEMBERS, scan_fn=_scan, dry_run=True)["payload"]
    rows = {(c.strategy, c.ticker): c for c in tc.from_viking_nine(None, tc.segment_context(None), payload)}
    b, a, l_ = rows[("viking_nine", "BBB.ST")], rows[("viking_nine", "AAA.ST")], rows[("viking_nine", "LLY")]
    assert b.status_class == tc.KOP and b.segment == "healthcare" and b.nordic and b.score == 90.0
    assert a.status_class == tc.BEVAKA and a.segment == "industrials"
    assert l_.status_class == tc.KOP and not l_.nordic and l_.segment == "healthcare"
    assert tc.BLOBS["viking_nine"] == "viking_nine_scan.json"


def test_workflow_runs_viking_nine_after_arc():
    src = open(os.path.join(ROOT, ".github", "workflows", "scheduled-scan.yml"), encoding="utf-8").read()
    assert "python viking_nine_scan.py" in src
    assert src.index("python arc_scan.py") < src.index("python viking_nine_scan.py") < src.index("python alert_scan.py")
