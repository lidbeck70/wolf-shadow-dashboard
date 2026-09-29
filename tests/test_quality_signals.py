"""
Quality-köpsignaler: Alpha Regimes fyra gates (trend, värdering, cykel,
kvalitet) körs i den schemalagda skanningen på Quality-listans topp, och ett
larm går när ett bolag NYTT står i BUY (4/4) — strategins egen köpregel.
"""
import json
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import alert_rules as ar  # noqa: E402
from alpha_regime import quality_scan as qs  # noqa: E402


def _res(verdict="BUY", passed=4, error=None):
    names = ("TREND", "DISCOUNT", "CYCLE", "QUALITY")
    sigs = [NS(name=n, passed=i < passed, label="OK" if i < passed else "NEJ") for i, n in enumerate(names)]
    return NS(error=error, signals=sigs, signals_passed=passed, quality_verdict=verdict, market_phase="HOPE")


# ── skanningen ───────────────────────────────────────────────────────────────
def test_benchmark_follows_the_home_market():
    assert qs.benchmark_for("ATCO-A.ST") == "^OMX" and qs.benchmark_for("EQNR.OL") == "OBX.OL"
    assert qs.benchmark_for("NOVO-B.CO") == "^OMXC25" and qs.benchmark_for("NESTE.HE") == "^OMXH25"
    assert qs.benchmark_for("MSFT") == "SPY"


def test_scan_runs_the_top_n_with_yf_tickers_and_records_gates():
    calls = []

    def analyze(t, bm, kap):
        calls.append((t, bm, kap))
        return {"ATCO-A.ST": _res(), "SAND.ST": _res("WATCH", 3)}.get(t) or _res(error="No price data")
    rows = [NS(ticker="ATCO A.ST", kap_badge=True), {"ticker": "SAND.ST"}, NS(ticker="X.ST", kap_badge=False),
            NS(ticker="NOPE.ST", kap_badge=False)]
    out = qs.scan(rows, top=3, analyze=analyze)
    assert calls == [("ATCO-A.ST", "^OMX", True), ("SAND.ST", "^OMX", False), ("X.ST", "^OMX", False)]
    a = out["ATCO A.ST"]
    assert a["verdict"] == "BUY" and a["passed"] == 4 and a["total"] == 4 and a["phase"] == "HOPE"
    assert a["gates"]["Trend"]["passed"] and set(a["gates"]) == {"Trend", "Värdering", "Cykel", "Kvalitet"}
    assert out["SAND.ST"]["verdict"] == "WATCH" and not out["SAND.ST"]["gates"]["Kvalitet"]["passed"]
    assert out["X.ST"]["verdict"] == "ERROR" and "NOPE.ST" not in out


def test_scan_survives_an_exception():
    def analyze(t, bm, kap):
        raise RuntimeError("yfinance nere")
    assert qs.scan([NS(ticker="A.ST", kap_badge=False)], analyze=analyze)["A.ST"]["verdict"] == "ERROR"


def test_the_engine_can_skip_sentiment():
    import inspect
    from alpha_regime.engine import run_regime_analysis
    assert inspect.signature(run_regime_analysis).parameters["with_sentiment"].default is True


# ── larmbenet ────────────────────────────────────────────────────────────────
def _q(signals, ts="2026-09-29T10:00"):
    return {"timestamp": ts, "mode": "quality",
            "results": [{"ticker": t, "name": t.split(".")[0].title(), "rank": i + 1, "composite_score": 70.0,
                         "sector": "Industri"} for i, t in enumerate(signals)],
            "quality_signals": {t: {"verdict": v, "passed": 4 if v == "BUY" else 3, "total": 4,
                                    "benchmark": "^OMX", "phase": "HOPE",
                                    "gates": {"Trend": {"passed": True}, "Värdering": {"passed": True},
                                              "Cykel": {"passed": True}, "Kvalitet": {"passed": v == "BUY"}}}
                                for t, v in signals.items()}}


def test_alerts_only_on_the_transition_to_buy():
    _a, state = ar.quality_alerts(_q({"ATCO-A.ST": "BUY", "SAND.ST": "WATCH"}), None)
    assert _a == [] and set(state["buy"]) == {"ATCO-A.ST"}               # baslinje, tyst
    alerts, state = ar.quality_alerts(_q({"ATCO-A.ST": "BUY", "SAND.ST": "BUY"}), state)
    assert [a["title"] for a in alerts] == ["💎 Quality: SAND.ST köpsignal (4/4)"]
    body = alerts[0]["body"]
    assert "Sand · Industri · #2 i listan" in body and "Kvalitet ✓" in body and "mot ^OMX" in body
    alerts, state = ar.quality_alerts(_q({"ATCO-A.ST": "BUY", "SAND.ST": "BUY"}), state)
    assert alerts == []                                                   # läget upprepas inte
    alerts, state = ar.quality_alerts(_q({"ATCO-A.ST": "BUY", "SAND.ST": "WATCH"}), state)
    alerts, state = ar.quality_alerts(_q({"ATCO-A.ST": "BUY", "SAND.ST": "BUY"}), state)
    assert len(alerts) == 1                                               # tillbaka till BUY larmar igen


def test_a_list_without_signals_freezes_the_baseline():
    _a, state = ar.quality_alerts(_q({"ATCO-A.ST": "BUY"}), None)
    old = _q({})
    old.pop("quality_signals")
    assert ar.quality_alerts(old, state) == ([], state)
    assert ar.quality_alerts(None, state) == ([], state)


def test_evaluate_routes_the_quality_leg():
    regime = {"regime": "GREEN"}
    _a, state = ar.evaluate(regime, {"top": []}, {"positions": []}, [], None, {},
                            quality_data=_q({"ATCO-A.ST": "BUY"}))
    assert "quality" in state
    alerts, _ = ar.evaluate(regime, {"top": []}, {"positions": []}, [], state,
                            {"quality": {"enabled": True, "channels": ["email"]}},
                            quality_data=_q({"ATCO-A.ST": "BUY", "VOLV-B.ST": "BUY"}))
    q = [a for a in alerts if a["kind"] == "quality_buy"]
    assert len(q) == 1 and q[0]["channels"] == ["email"]
    alerts, _ = ar.evaluate(regime, {"top": []}, {"positions": []}, [], state,
                            {"quality": {"enabled": False}},
                            quality_data=_q({"ATCO-A.ST": "BUY", "VOLV-B.ST": "BUY"}))
    assert not [a for a in alerts if a["kind"] == "quality_buy"]


def test_signals_are_saved_with_the_quality_list(monkeypatch, tmp_path):
    from contrarian_alpha import cache
    from contrarian_alpha.engine import PipelineConfig, PipelineResult
    monkeypatch.setattr(cache, "_LOCAL_FALLBACK", str(tmp_path / ".ca_results.json"))
    monkeypatch.setattr(cache, "_get_github_token", lambda: "")
    res = PipelineResult(results=[], universe_count=10, necessity_passed=0, hate_passed=0, bs_passed=0,
                         composite_ranked=0, run_duration_s=1.0, config=PipelineConfig(mode="quality"),
                         eliminated=[], timestamp="t")
    cache.save_screener_results(res, mode="quality", extra={"quality_signals": {"A.ST": {"verdict": "BUY"}}})
    saved = json.load(open(tmp_path / ".ca_results_quality.json"))
    assert saved["quality_signals"]["A.ST"]["verdict"] == "BUY" and saved["mode"] == "quality"


def test_scheduled_scan_and_alert_scan_are_wired():
    sched = open(os.path.join(ROOT, "scheduled_scan.py"), encoding="utf-8").read()
    assert "quality_scan import scan" in sched and 'extra = {"quality_signals"' in sched
    scan = open(os.path.join(ROOT, "alert_scan.py"), encoding="utf-8").read()
    assert 'load_screener_results(mode="quality")' in scan and "quality_data=quality_data" in scan
    tab = open(os.path.join(ROOT, "tabs", "alerts.py"), encoding="utf-8").read()
    assert '"quality":   {"enabled": True' in tab and "💎 Quality-köpsignal" in tab
