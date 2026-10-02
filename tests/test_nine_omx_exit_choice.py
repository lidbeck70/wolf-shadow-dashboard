"""
Viking Nine: nordiska aktier mäts mot OMXS30 (index + svensk Large Cap-bredd)
i Nine, exekveringen, exitmotorn och backtestet — och backtestet kan köras
med olika exitregler och jämföra dem. Syntetiska kurser — inget nätverk.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import ovtlyr_nine as on  # noqa: E402
import viking_backtest as vb  # noqa: E402
from test_ovtlyr_nine import BULL_OB, END, _df  # noqa: E402

UP, DOWN = _df(), _df(step=-0.4, start=300)
YAHOO = {"SPY": UP, **{t: UP for t in on.SECTOR_ETFS.values()}, "VOLV-B.ST": UP}


def _nordic(close=DOWN, breadth=True):
    b = pd.Series([20.0] * len(close), index=close.index) if breadth else None
    return lambda: {"close": close["Close"], "breadth": b, "source": "Börsdata · OMX Stockholm 30",
                    "breadth_source": "Börsdata · 164 Large Cap-aktier"}


def _eval(ticker, provider):
    return on.evaluate(ticker, ob_analysis=BULL_OB, getter=lambda t, p: YAHOO.get(t, pd.DataFrame()),
                       sector_getter=lambda t: "Industrials", bd_sector=lambda t: None, today=END,
                       nordic_provider=provider)


def test_nordic_ticker_uses_omxs30_not_spy():
    r = _eval("VOLV-B.ST", _nordic())                   # SPY stiger, OMXS30 faller
    assert r.market_label == "OMXS30" and on.market_for("VOLV-B.ST") == "OMXS30" and on.market_for("NVDA") == "SPY"
    assert r.get("market.trend").status == on.FAIL and r.get("market.signal").status == on.FAIL
    assert r.get("market.trend").source.startswith("Börsdata · OMX Stockholm 30")
    br = r.get("market.breadth")
    assert br.status == on.FAIL and "av aktierna över EMA50" in br.detail and "Large Cap" in br.source
    us = _eval("NVDA", _nordic())                       # amerikansk aktie: SPY som förut
    assert us.market_label == "SPY" and us.get("market.trend").status == on.PASS


def test_missing_omx_data_is_marked():
    r = _eval("VOLV-B.ST", lambda: None)
    assert r.market_label == "SPY (OMXS30 saknas)"
    nb = _eval("VOLV-B.ST", _nordic(UP, breadth=False))
    assert nb.get("market.breadth").status == on.UNAVAILABLE and nb.get("market.trend").status == on.PASS


def test_card_execution_and_exit_name_the_market():
    import viking_execution as vx
    import viking_exit as vex
    from ovtlyr.ui.nine_card import card_html
    r = _eval("VOLV-B.ST", _nordic())
    assert "MARKET 40 % · OMXS30" in card_html(r)
    d = vx.evaluate_entry("VOLV-B.ST", UP, nine=r, earnings_date=pd.Timestamp("2026-12-01"), trades=[])
    assert any("OMXS30 under EMA20" in x for x in d.reasons)
    e = vex.evaluate_exit("VOLV-B.ST", UP, float(UP["Close"].iloc[-30]), UP.index[-30], nine=r,
                          earnings_date=pd.Timestamp("2026-12-01"), today=pd.Timestamp(END))
    assert e.status == vex.CLOSE_ALL and "OMXS30 under EMA20" in e.reasons[0]


def test_market_layer_for_both_markets():
    spy = on.market_layer("SPY", getter=lambda t, p: YAHOO.get(t, pd.DataFrame()), today=END)
    assert [f.status for f in spy] == [on.PASS, on.PASS, on.PASS]
    omx = on.market_layer("OMXS30", nordic_provider=_nordic(), today=END)
    assert omx[0].status == on.FAIL and "Börsdata" in omx[0].source
    gone = on.market_layer("OMXS30", nordic_provider=lambda: None, today=END)
    assert all(f.status == on.UNAVAILABLE for f in gone)


# ── Backtestet ───────────────────────────────────────────────────────────────
from test_viking_backtest import DATA, _walk  # noqa: E402


def _bt(tickers, provider=None, **cfg):
    return vb.run(tickers, getter=lambda t, p: DATA.get(t), sector_getter=lambda t: "Technology",
                  cfg=vb.Config(min_nine=7, **cfg), today=pd.Timestamp("2026-09-30"), nordic_provider=provider)


def test_backtest_uses_omxs30_for_nordic_tickers():
    DATA["NORD.ST"], DATA["OMX"] = DATA["S0"], _walk(-0.0006, 0.012, seed=77)
    try:
        prov = lambda: {"close": DATA["OMX"]["Close"], "breadth": None, "source": "Börsdata", "breadth_source": ""}  # noqa: E731,E501
        res = _bt(["NORD.ST", "S0"], provider=prov)
    finally:
        DATA.pop("NORD.ST")
    per = {p["ticker"]: p for p in res["per_ticker"]}
    assert per["NORD.ST"]["market"] == "OMXS30" and per["S0"]["market"] == "SPY"
    reasons = {t.exit_reason for t in res["trades"] if t.ticker == "NORD.ST"}
    assert "SPY < EMA20" not in reasons                                   # svensk aktie säljs inte på SPY


def test_exit_presets_change_the_exits():
    tickers = [f"S{i}" for i in range(4)]
    allr = _bt(tickers)
    only = _bt(tickers, exit_rules=("trail",))
    assert {t.exit_reason for t in only["trades"] if not t.open} <= {"stopp", "breakeven-stopp", "trailing EMA10"}
    assert len({t.exit_reason for t in allr["trades"]}) >= len({t.exit_reason for t in only["trades"]})
    after = _bt(tickers, trail_after_be=True)
    early = [t for t in after["trades"] if t.exit_reason == "trailing EMA10"]
    before = [t for t in allr["trades"] if t.exit_reason == "trailing EMA10"]
    assert len(early) <= len(before)                                       # EMA10 väntar på breakeven
    assert set(vb.EXIT_PRESETS) == {"Alla regler", "EMA10 först efter breakeven",
                                    "Kärnan (stopp, breakeven, EMA10, marknad)", "Bara stopp + EMA10"}


def test_comparison_view(monkeypatch):
    from streamlit.testing.v1 import AppTest
    tickers = [f"S{i}" for i in range(3)]
    runs = {}
    for name, (rules, after) in vb.EXIT_PRESETS.items():
        runs[name] = _bt(tickers, exit_rules=rules, trail_after_be=after)
    monkeypatch.setenv("VNB_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["VNB_TEST_ROOT"])
        from ovtlyr.ui.viking_nine_backtest import render_viking_nine_backtest
        render_viking_nine_backtest()

    at = AppTest.from_function(app, default_timeout=90)
    at.session_state["vnb_result"] = {"selected": "Alla regler", "runs": runs}
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "JÄMFÖRELSE AV EXITREGLER" in html and "EMA10 först efter breakeven" in html and "Expectancy R" in html
    at.selectbox(key="vnb_show").set_value("Bara stopp + EMA10").run()
    assert not at.exception and "Exit: <b" in " ".join(m.value for m in at.markdown)
