"""Viking Nine-skanningen loggar marknadslagret och poängfördelningen (diagnostik — inga regler ändras)."""
from types import SimpleNamespace

import viking_nine_scan as vns


def _row(t, n, cat="REJECTED"):
    return {"ticker": t, "category": cat, "nine": SimpleNamespace(passed=n)}


def test_score_counts():
    rows = [_row("A", 9), _row("B", 8), _row("C", 7), _row("D", 6), _row("E", 3), {"ticker": "F"}]
    assert vns.score_counts(rows) == {"9": 1, "8": 1, "7": 1, "≤6": 2}


def test_market_layer_status_and_failure():
    f = SimpleNamespace(label="Trend", status="FAIL", detail="under EMA50")
    assert vns.market_layer_status("norden", lambda m: [f]) == [{"label": "Trend", "status": "FAIL", "detail": "under EMA50"}]

    def boom(m):
        raise RuntimeError("nät")
    assert vns.market_layer_status("usa", boom)[0]["status"] == "DATA UNAVAILABLE"


def test_run_saves_layer_and_scores():
    seen = []
    layer = lambda m: seen.append(m) or [SimpleNamespace(label="Trend", status="FAIL", detail=""),
                                         SimpleNamespace(label="Signal", status="FAIL", detail=""),
                                         SimpleNamespace(label="Breadth", status="PASS", detail="")]
    scan = lambda tickers, **kw: {"funnel": {"universe": len(tickers), "passed": 2},
                                  "rows": [_row("A.ST", 6), _row("B.ST", 5)]}
    out = vns.run(members={"norden": {"A.ST": ("A", None), "B.ST": ("B", None)}, "usa": {}}, scan_fn=scan,
                  layer_fn=layer, dry_run=True)
    nd = out["payload"]["markets"]["norden"]
    assert seen == ["OMXS30"] and [f["status"] for f in nd["market_layer"]] == ["FAIL", "FAIL", "PASS"]
    assert nd["funnel"]["scores"] == {"9": 0, "8": 0, "7": 0, "≤6": 2}
    assert out["payload"]["markets"]["usa"]["market_layer"] == []


def test_thor_explains_a_blocked_market():
    from thor import candidates as tc
    blob = {"markets": {
        "norden": {"market_layer": [{"label": "Trend", "status": "FAIL"}, {"label": "Signal", "status": "FAIL"},
                                    {"label": "Breadth", "status": "PASS"}]},
        "usa": {"market_layer": [{"label": "Trend", "status": "PASS"}, {"label": "Signal", "status": "PASS"},
                                 {"label": "Breadth", "status": "FAIL"}]},
        "gammal": {}}}
    notes = tc.vn_market_notes(blob)
    assert len(notes) == 1 and "OMXS30 1/3 (Trend FAIL, Signal FAIL)" in notes[0] and "inte ett fel" in notes[0]
    assert tc.vn_market_notes(None) == []
