"""
🌩️ Marknadsrisk PR 3 — riskspärren: HÖG spärrar nya entries i Viking Nine och
halverar Wolf; Discord-larm när SPY/OMXS30 går in i och lämnar HÖG.
"""
import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import alert_rules as ar  # noqa: E402
import market_risk_gate as mg  # noqa: E402
import viking_execution as vx  # noqa: E402


def _risk(level, market="SPY", points=5):
    return {"market": market, "label": "S&P 500 (SPY)" if market == "SPY" else "OMXS30", "level": level,
            "points": points, "possible": 9, "date": "2026-10-02", "active": ["VIX-spik", "Under SMA200"]}


def test_market_for_ticker():
    assert mg.market_for("VOLV-B.ST") == "OMXS30" and mg.market_for("EQNR.OL") == "OMXS30"
    assert mg.market_for("NVDA") == "SPY" and mg.market_for("") == "SPY"


def test_gate_rules():
    assert mg.blocks_entry(_risk("HÖG")) and not mg.blocks_entry(_risk("FÖRHÖJD")) and not mg.blocks_entry(None)
    assert mg.blocks_viking_entry(_risk("HÖG")) and mg.blocks_viking_entry(_risk("FÖRHÖJD"))
    assert not mg.blocks_viking_entry(_risk("LÅG")) and not mg.blocks_viking_entry(None)
    assert mg.size_factor(_risk("FÖRHÖJD")) == 1.0                            # Wolf halveras bara vid HÖG
    assert mg.size_factor(_risk("HÖG")) == 0.5 and mg.size_factor(_risk("LÅG")) == 1.0 and mg.size_factor(None) == 1.0
    assert "okänd" in mg.describe(None) and "HÖG (5 av 9 varningar: VIX-spik, Under SMA200)" in mg.describe(_risk("HÖG"))


def test_current_is_cached_and_survives_errors(monkeypatch):
    monkeypatch.setattr(mg, "_CACHE", {})
    calls = []
    r = SimpleNamespace(market="SPY", label="S&P 500 (SPY)", level="HÖG", points=4, possible=9, date="d", error=None,
                        signals=[{"label": "VIX-spik", "active": True}, {"label": "Eufori", "active": False}])

    def ev(m):
        calls.append(m)
        return r
    assert mg.current("SPY", ev)["active"] == ["VIX-spik"] and mg.current("SPY", ev)["level"] == "HÖG"
    assert calls == ["SPY"]                                                   # cachad
    assert mg.current("OMXS30", lambda m: (_ for _ in ()).throw(RuntimeError("nät"))) is None


def test_viking_nine_blocks_new_entries_at_high_risk():
    from test_viking_execution import AFTER_CLOSE, FAR, _Nine, _df
    ok = vx.evaluate_entry("NVDA", _df(), nine=_Nine(), earnings_date=FAR, now=AFTER_CLOSE, trades=[],
                           market_risk=_risk("LÅG"))
    assert ok.status == vx.GO
    hi = vx.evaluate_entry("NVDA", _df(), nine=_Nine(), earnings_date=FAR, now=AFTER_CLOSE, trades=[],
                           market_risk=_risk("HÖG"))
    assert hi.status == vx.NO_TRADE and vx.MARKET_RISK_HIGH in hi.flags
    assert any("inga nya entries" in r for r in hi.reasons)
    mid = vx.evaluate_entry("NVDA", _df(), nine=_Nine(), earnings_date=FAR, now=AFTER_CLOSE, trades=[],
                            market_risk=_risk("FÖRHÖJD"))
    assert mid.status == vx.NO_TRADE and vx.MARKET_RISK_ELEVATED in mid.flags   # FÖRHÖJD spärrar också Viking Nine


def test_screen_passes_the_risk_per_ticker():
    import viking_screen as vs
    from test_viking_screen import END, NOW, _getter, _sector
    asked = []
    rows = vs.run(["GOOD"], getter=_getter, sector_getter=_sector, now=NOW, today=pd.Timestamp(END),
                  earnings_getter=lambda t: pd.Timestamp("2026-12-01"),
                  risk_getter=lambda t: asked.append(t) or _risk("HÖG"))
    assert asked == ["GOOD"] and rows[0]["decision"].status == vx.NO_TRADE


def test_wolf_halves_the_position():
    src = open(os.path.join(ROOT, "tabs", "regime.py"), encoding="utf-8").read()
    assert "_mg.size_factor(_mr)" in src and "Wolf-positionen halveras" in src


# ── Larmen ───────────────────────────────────────────────────────────────────
def test_alerts_on_entering_and_leaving_high():
    alerts, state = ar.market_risk_alerts({"SPY": _risk("FÖRHÖJD"), "OMXS30": _risk("LÅG", "OMXS30")}, None)
    assert alerts == [] and state == {"levels": {"SPY": "FÖRHÖJD", "OMXS30": "LÅG"}}      # baslinje
    alerts, state = ar.market_risk_alerts({"SPY": _risk("HÖG"), "OMXS30": _risk("FÖRHÖJD", "OMXS30")}, state)
    assert [a["title"] for a in alerts] == ["🌩️ Marknadsrisk HÖG: S&P 500 (SPY)"]          # FÖRHÖJD larmar inte
    assert "VIX-spik" in alerts[0]["body"] and "inga nya entries" in alerts[0]["body"]
    alerts, state = ar.market_risk_alerts({"SPY": _risk("HÖG"), "OMXS30": None}, state)
    assert alerts == [] and state["levels"]["OMXS30"] == "FÖRHÖJD"                         # okänd → behåll
    alerts, state = ar.market_risk_alerts({"SPY": _risk("FÖRHÖJD")}, state)
    assert [a["title"] for a in alerts] == ["✅ Marknadsrisk S&P 500 (SPY) ner till FÖRHÖJD"]
    frozen, st2 = ar.market_risk_alerts(None, state)
    assert frozen == [] and st2 == state


def test_evaluate_routes_the_leg():
    prev = {"market_risk": {"levels": {"SPY": "LÅG"}}}
    out, state = ar.evaluate({}, {}, {"positions": []}, [], prev, {"market_risk": {"enabled": True}},
                             market_risk_data={"SPY": _risk("HÖG")})
    assert any(a["kind"] == "market_risk_high" for a in out) and state["market_risk"]["levels"]["SPY"] == "HÖG"
    off, _ = ar.evaluate({}, {}, {"positions": []}, [], prev, {"market_risk": {"enabled": False}},
                         market_risk_data={"SPY": _risk("HÖG")})
    assert not any(a["kind"].startswith("market_risk") for a in off)


def test_scan_and_workflow_wiring():
    scan = open(os.path.join(ROOT, "alert_scan.py"), encoding="utf-8").read()
    assert "market_risk_data=market_risk_data" in scan and "_mg.current(m)" in scan
    wf = open(os.path.join(ROOT, ".github", "workflows", "scheduled-scan.yml"), encoding="utf-8").read()
    step = wf[wf.index("- name: Alert scan"):]
    assert "BORSDATA_API_KEY" in step
    tab = open(os.path.join(ROOT, "tabs", "alerts.py"), encoding="utf-8").read()
    assert '"market_risk"' in tab and "🌩️ Marknadsrisk" in tab


def test_log_ticks():
    import market_risk_ui as ui
    assert ui.log_ticks(60, 700) == [100, 200, 500]
    assert ui.log_ticks(900, 3600) == [1000, 2000]
    assert ui.log_ticks(0, 10) == []


# ── SPY visade DATA UNAVAILABLE (Yahoo-fel cachades i sex timmar) ────────────
def test_failed_yahoo_download_is_not_cached(monkeypatch):
    import yfinance as yf
    import market_prices as mp
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return pd.DataFrame()                                  # t.ex. Too Many Requests
        idx = pd.bdate_range(end="2026-09-30", periods=5)
        return pd.DataFrame({"Close": [1.0, 2, 3, 4, 5]}, index=idx)
    monkeypatch.setattr(yf, "download", flaky)
    mp.clear()
    assert mp.ohlcv("SPY", "max").empty                            # första försöket misslyckas …
    assert len(mp.ohlcv("SPY", "max")) == 5 and len(calls) == 2    # … och provas igen direkt
    assert len(mp.ohlcv("SPY", "max")) == 5 and len(calls) == 2    # lyckat resultat cachas
    mp.clear()


def test_spy_falls_back_to_gspc():
    import market_risk as mr
    from test_market_risk import _data
    _idx, d = _data(crash=True)
    d2 = {k: v for k, v in d.items() if k != "SPY"}
    d2["^GSPC"] = d["SPY"]
    r = mr.evaluate("SPY", getter=lambda t, p: d2.get(t), fred_getter=lambda s: d2.get(s))
    assert r.error is None and "^GSPC (reserv — SPY saknades)" in r.source


def test_light_mode_shares_the_tab_download():
    import market_risk as mr
    from test_market_risk import _data
    _idx, d = _data()
    periods = set()
    mr.evaluate("SPY", getter=lambda t, p: periods.add(p) or d.get(t), fred_getter=lambda s: d.get(s), light=True)
    assert periods == {"max"}


def test_failed_level_is_retried_soon(monkeypatch):
    import time as _t
    monkeypatch.setattr(mg, "_CACHE", {"SPY": (_t.time() - mg.TTL_FAIL_S - 1, None)})
    r = SimpleNamespace(market="SPY", label="S&P 500 (SPY)", level="LÅG", points=0, possible=9, date="d",
                        error=None, signals=[])
    assert mg.current("SPY", lambda m: r)["level"] == "LÅG"


def test_tab_does_not_cache_an_error():
    src = open(os.path.join(ROOT, "market_risk_ui.py"), encoding="utf-8").read()
    assert "if not res.error:" in src


# ── Viking Nine spärras från FÖRHÖJD ────────────────────────────────────────
def test_screen_blocks_viking_at_elevated_risk():
    import viking_screen as vs
    from test_viking_screen import END, NOW, _getter, _sector
    rows = vs.run(["GOOD"], getter=_getter, sector_getter=_sector, now=NOW, today=pd.Timestamp(END),
                  earnings_getter=lambda t: pd.Timestamp("2026-12-01"),
                  risk_getter=lambda t: _risk("FÖRHÖJD", points=3))
    d = rows[0]["decision"]
    assert d.status == vx.NO_TRADE and vx.MARKET_RISK_ELEVATED in d.flags


def test_risk_banner_warns_at_elevated(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("MRG_TEST_ROOT", ROOT)
    monkeypatch.setattr(mg, "current", lambda m, evaluator=None: {"market": m, "label": m, "level": "FÖRHÖJD",
                                                                  "points": 3, "possible": 9, "date": "d",
                                                                  "active": []})

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["MRG_TEST_ROOT"])
        from ovtlyr.ui.viking_screens import _risk_banner
        _risk_banner(("SPY",))

    at = AppTest.from_function(app, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "SPY: FÖRHÖJD" in html and "inga nya Viking Nine-entries" in html
