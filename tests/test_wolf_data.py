"""
Tests for wolf_data.py — the Swing screener/regime data engine.

The engine only reaches Börsdata over the network at run time, so these tests
mock the single HTTP entry point (wolf_data._get) with synthetic, Börsdata-
shaped responses and drive the *real* main() pipeline: universe filtering,
per-instrument metrics, qualification, ranking, regime classification, index
block, history append and the JSON files the panel tabs read.

This is the offline guard for the flow that cannot run in CI/sandbox against the
live API (egress-blocked): everything except the literal HTTP call is exercised.
"""
import os
import sys
import json
import math
import datetime as dt
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wolf_data as w


# ── Synthetic Börsdata responses ─────────────────────────────────────────────
_INSTR = [
    {"insId": 1, "name": "Alfa AB",  "ticker": "AAA", "instrument": 0},
    {"insId": 2, "name": "Beta AB",  "ticker": "BBB", "instrument": 0},
    {"insId": 3, "name": "Gamma AB", "ticker": "CCC", "instrument": 0},
    {"insId": 9, "name": "Sank AB",  "ticker": "SNK", "instrument": 0},  # downtrend
    {"insId": 7, "name": "En fond",  "ticker": "FND", "instrument": 1},  # not a stock
]
_IDX = [{"insId": 100, "name": "OMX Stockholm PI"}]


def _series(ins_id, n=260):
    if ins_id == 100:
        return [900.0 * (1 + 0.06 * t / n) for t in range(n)]      # index up ~6%
    if ins_id == 9:
        return [120.0 * (1 - 0.30 * t / n) for t in range(n)]      # -30% downtrend
    drift = {1: 0.35, 2: 0.20, 3: 0.14}.get(ins_id, 0.1)
    return [80.0 * (1 + drift * t / n) + 2.0 * math.sin(t / 9.0) for t in range(n)]


def _fake_get(ep, params=None):
    if ep == "/instruments":
        return {"instruments": _INSTR}
    if ep == "/instruments/indexes":
        return {"indexes": _IDX}
    if ep.endswith("/stockprices"):
        ins_id = int(ep.split("/")[2])
        base = dt.date(2025, 1, 1)
        rows = [{"d": (base + dt.timedelta(days=i)).isoformat(), "c": round(v, 3)}
                for i, v in enumerate(_series(ins_id))]
        return {"stockPricesList": rows}
    raise AssertionError("unexpected endpoint " + ep)


def _run_pipeline():
    """Run the real main() against mocked Börsdata into a temp dir; return JSON."""
    tmp = tempfile.mkdtemp()
    w.CONFIG["OUTPUT_DIR"] = tmp
    w.CONFIG["CACHE_DIR"] = os.path.join(tmp, "cache")
    w.CONFIG["UNIVERSE_CSV"] = os.path.join(tmp, "absent.csv")  # -> full mocked universe
    w.CONFIG["API_KEY"] = "test-key"                            # passes the guard
    w.CONFIG["PUSH_TO_GIST"] = False                            # no network
    w._get = _fake_get
    w._last = [0.0]
    w.main()
    scr = json.load(open(os.path.join(tmp, "wolf_screener.json"), encoding="utf-8"))
    reg = json.load(open(os.path.join(tmp, "wolf_regime.json"), encoding="utf-8"))
    return scr, reg


# ── Pure helpers ─────────────────────────────────────────────────────────────
def test_metrics_for_uptrend():
    import pandas as pd
    idx = pd.date_range("2025-01-01", periods=260, freq="D")
    closes = pd.Series([80 + 0.15 * t for t in range(260)], index=idx)
    m = w.metrics_for(closes)
    assert m is not None
    assert m["px"] > m["ma200"] > 0
    assert m["r3"] is not None and m["r6"] is not None
    assert 0 <= m["rsi"] <= 100


def test_metrics_for_too_short():
    import pandas as pd
    closes = pd.Series(range(50))
    assert w.metrics_for(closes) is None


def test_classify_regime_all_branches():
    assert w.classify_regime(None, 0.6)[0] == "OKÄND"
    assert w.classify_regime({"above": False, "dist": -0.05}, 0.6)[0] == "RÖD"
    assert w.classify_regime({"above": True, "dist": 0.05}, 0.30)[0] == "GUL"   # thin breadth
    assert w.classify_regime({"above": True, "dist": 0.01}, 0.60)[0] == "GUL"   # thin margin
    assert w.classify_regime({"above": True, "dist": 0.05}, 0.60)[0] == "GRÖN"
    # each returns a non-empty rules list
    for idx_block, breadth in [(None, 0.6), ({"above": False, "dist": 0}, 0.6),
                               ({"above": True, "dist": 0.01}, 0.6),
                               ({"above": True, "dist": 0.05}, 0.6)]:
        assert w.classify_regime(idx_block, breadth)[1]


# ── End-to-end pipeline ──────────────────────────────────────────────────────
def test_pipeline_filters_non_stocks_and_downtrends():
    scr, _ = _run_pipeline()
    tickers = [r["ticker"] for r in scr["top"]]
    assert "FND" not in tickers          # instrument != 0 filtered by build_universe
    assert "SNK" not in tickers          # downtrend never qualifies (px < MA200)
    assert scr["universe_size"] == 4     # 4 stocks scanned (fund excluded)


def test_pipeline_ranking_and_schema():
    scr, _ = _run_pipeline()
    assert scr["top"], "expected at least one qualifier"
    # ranked by score desc, rank starts at 1
    scores = [r["score"] for r in scr["top"]]
    assert scores == sorted(scores, reverse=True)
    assert scr["top"][0]["rank"] == 1
    assert scr["top"][0]["ticker"] == "AAA"   # strongest drift
    # UI contract: every field the screener tab reads is present
    for key in ("ticker", "name", "score", "mom3", "mom6", "rsi",
                "dist_ma20", "setupA", "nearMA", "nearHigh", "rank"):
        assert key in scr["top"][0], f"missing screener field {key}"


def test_pipeline_regime_schema_and_history():
    _, reg = _run_pipeline()
    assert reg["regime"] in ("GRÖN", "GUL", "RÖD", "OKÄND")
    assert isinstance(reg["rules"], list) and reg["rules"]
    # index block + UI contract
    assert reg["index"] is not None
    for key in ("close", "ma200", "above", "dist", "spark"):
        assert key in reg["index"], f"missing index field {key}"
    assert len(reg["index"]["spark"]) == 30
    for key in ("breadth", "qualifying", "universe", "history"):
        assert key in reg, f"missing regime field {key}"
    assert reg["history"] and reg["history"][-1]["regime"] == reg["regime"]
