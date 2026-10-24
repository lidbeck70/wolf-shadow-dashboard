"""Designförslaget PR 2: larmloggen i Gisten och 'Öppna →' till larmets mål."""
from alerts import history
from ui import links


def _a(i, **kw):
    return {"kind": "k", "title": f"T{i}", "body": "B", "channels": ["discord"], **kw}


def test_append_keeps_the_last_fifty():
    log = history.append(None, [_a(i) for i in range(30)], "t1")
    log = history.append(log, [_a(i) for i in range(30, 60)], "t2")
    assert len(log) == history.LIMIT == 50
    assert log[0]["title"] == "T10" and log[-1]["title"] == "T59"


def test_entry_carries_target_and_delivery():
    log = history.append([], [_a(1, target="thor", ticker="ABC.ST", strategy="wolf")], "2026-10-24 06:00 UTC",
                         {"T1": False})
    e = log[0]
    assert e == {"time": "2026-10-24 06:00 UTC", "kind": "k", "title": "T1", "body": "B", "target": "thor",
                 "ticker": "ABC.ST", "strategy": "wolf", "delivered": False}
    assert "channels" not in e


def test_garbage_log_is_ignored():
    assert history.append({"x": 1}, [_a(1)], "t")[0]["title"] == "T1"
    assert history.append(["str", None, {"title": "ok"}], [], "t") == [{"title": "ok"}]


def test_latest_newest_first():
    log = history.append([], [_a(i) for i in range(5)], "t")
    assert [r["title"] for r in history.latest(log, 2)] == ["T4", "T3"]


def test_go_opens_thor_with_ticker_and_strategy():
    state = {}
    links.go("thor", "VOLV-B.ST", "viking_nine", state=state)
    assert state["nav_goto"] == "thor" and state["thor_ticker"] == "VOLV-B.ST"
    assert state["thor_strategy"] == "Viking Nine" and state["_thor_strategy_set"]


def test_go_other_target_only_navigates():
    state = {}
    links.go("market_risk", "SPY", state=state)
    assert state == {"nav_goto": "odin/laget", "laget_risk": True}
