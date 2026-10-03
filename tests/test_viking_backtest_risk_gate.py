"""
Viking Nine-backtestet med marknadsriskspärren (🌩️ Marknadsrisk): en
signaldag där marknadens risknivå är spärrad ger ingen entry — som live.
Syntetiska kurser och riskpoäng — inget nätverk.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ovtlyr_nine as on  # noqa: E402
import viking_backtest as vb  # noqa: E402
from tests.test_viking_backtest import DATA, IDX, ROOT  # noqa: E402

TODAY = pd.Timestamp("2026-09-30")
CUT = IDX[600]


def _pts(value, start=None):
    s = pd.Series(0, index=IDX)
    s[IDX >= (start or IDX[0])] = value
    return s


def _bt(ticker="S0", risk=None, name=None, **cfg):
    stock, sec = DATA[ticker], DATA["XLK"]
    breadth = on.breadth_series({t: DATA[t]["Close"] for t in on.SECTOR_ETFS.values()})
    return vb.backtest_ticker(name or ticker, stock, DATA["SPY"], sec, breadth, vb.Config(min_nine=8, years=5, **cfg),
                              risk=risk)


def _run(provider=None, tickers=("S0", "S1", "S2", "S3"), **cfg):
    return vb.run(list(tickers), getter=lambda t, p: DATA.get(t), sector_getter=lambda t: "Technology",
                  cfg=vb.Config(min_nine=8, years=3, **cfg), today=TODAY, risk_provider=provider)


def _key(trades):
    return [(t.signal_date, t.exit_reason, t.r) for t in trades]


def test_risk_levels_use_last_known_points():
    pts = pd.Series([0, 2, 5], index=[IDX[10], IDX[20], IDX[30]])
    lv = vb.risk_levels(pts, IDX[:40])
    assert lv.iloc[5] == "" and lv.iloc[10] == "LÅG" and lv.iloc[19] == "LÅG"
    assert lv.iloc[25] == "FÖRHÖJD" and lv.iloc[39] == "HÖG"
    assert (vb.risk_levels(None, IDX[:5]) == "").all()


def test_default_gate_is_per_market_like_live():
    import market_risk_gate as mrg
    assert vb.Config().risk_gate == mrg.VIKING_BLOCK_BY_MARKET == {"OMXS30": ("FÖRHÖJD", "HÖG"), "SPY": ("HÖG",)}
    first, levels = next(iter(vb.RISK_GATES.items()))
    assert first == "Per marknad (som live)" and levels == mrg.VIKING_BLOCK_BY_MARKET
    assert vb.RISK_GATES["FÖRHÖJD eller HÖG"] == ("FÖRHÖJD", "HÖG")
    assert vb.RISK_GATES["Bara HÖG"] == ("HÖG",) and vb.RISK_GATES["Av"] == ()
    assert vb.gate_levels(levels, "OMXS30") == ("FÖRHÖJD", "HÖG") and vb.gate_levels(levels, "SPY") == ("HÖG",)
    assert vb.gate_levels(("HÖG",), "OMXS30") == ("HÖG",) and vb.gate_levels((), "SPY") == ()


def test_high_risk_blocks_every_entry():
    free = _bt()
    assert free["trades"] and free["risk_blocked"] == 0
    blocked = _bt(risk=_pts(5))
    assert blocked["trades"] == [] and blocked["risk_blocked"] == blocked["signals"] > 0


def test_low_risk_or_gate_off_changes_nothing():
    free = _bt()
    assert _key(_bt(risk=_pts(0))["trades"]) == _key(free["trades"])
    assert _key(_bt(risk=_pts(5), risk_gate=())["trades"]) == _key(free["trades"])


def test_elevated_blocks_nordic_but_not_us_by_default():
    free = _bt()
    assert _key(_bt(risk=_pts(3))["trades"]) == _key(free["trades"])                      # SPY: FÖRHÖJD handlas
    nord = _bt(risk=_pts(3), name="NORD.ST")
    assert nord["trades"] == [] and nord["risk_blocked"] > 0                               # OMXS30: spärr
    assert _key(_bt(risk=_pts(3), name="NORD.ST", risk_gate=vb.RISK_GATES["Bara HÖG"])["trades"]) == \
        _key(free["trades"])
    strict = _bt(risk=_pts(3), risk_gate=vb.RISK_GATES["FÖRHÖJD eller HÖG"])
    assert strict["trades"] == [] and strict["risk_blocked"] > 0


def test_gate_only_acts_from_the_day_risk_is_known():
    free = _bt()
    gated = _bt(risk=_pts(5, start=CUT))
    before = [t for t in free["trades"] if pd.Timestamp(t.signal_date) < CUT]
    assert _key([t for t in gated["trades"] if pd.Timestamp(t.signal_date) < CUT]) == _key(before)
    assert all(pd.Timestamp(t.signal_date) < CUT for t in gated["trades"])
    assert len(gated["trades"]) < len(free["trades"])


def test_run_fetches_risk_per_market_and_reports_it():
    asked = []

    def provider(m):
        asked.append(m)
        return _pts(5, start=CUT)

    DATA["NORD.ST"] = DATA["S0"]
    try:
        res = _run(provider, tickers=("S0", "S1", "NORD.ST"))
    finally:
        DATA.pop("NORD.ST")
    assert sorted(asked) == ["OMXS30", "SPY"]
    assert set(res["risk"]) == {"OMXS30", "SPY"} and res["risk"]["SPY"]["status"] == "ok"
    assert 0 < res["risk"]["SPY"]["blocked_pct"] < 100
    assert res["risk_blocked"] == sum(p["risk_blocked"] for p in res["per_ticker"]) > 0


def test_missing_risk_never_blocks():
    free = _run(None)
    assert free["risk"]["SPY"]["status"].startswith("DATA UNAVAILABLE") and free["risk_blocked"] == 0

    def broken(m):
        raise RuntimeError("nätverk")

    res = _run(broken)
    assert _key(res["trades"]) == _key(free["trades"]) and res["risk_blocked"] == 0
    asked = []
    _run(lambda m: asked.append(m), risk_gate=())
    assert asked == []                                                   # avstängd spärr hämtar inget


def test_run_names():
    from ovtlyr.ui.viking_nine_backtest import run_names
    first = next(iter(vb.EXIT_PRESETS))
    assert run_names(first, "Av", False, False) == [(first, first, "Av")]
    names = run_names(first, "Av", False, True)
    assert [n for n, _e, _g in names] == [f"Spärr {g}" for g in vb.RISK_GATES]
    both = run_names(first, "Av", True, True)
    assert len(both) == len(vb.EXIT_PRESETS) * len(vb.RISK_GATES) and " · spärr " in both[0][0]


def test_page_compares_risk_gates(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("VNB_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["VNB_TEST_ROOT"])
        import market_prices
        import pandas as _pd
        import ovtlyr.ui.viking_nine_backtest as page
        from tests.test_viking_backtest import DATA as _D, IDX as _I
        market_prices.ohlcv = lambda t, p="1y", **k: _D.get(t, _pd.DataFrame())
        page._sector = lambda t: "Technology"
        page._risk_points = lambda m: _pd.Series(5, index=_I[_I >= _I[600]]).reindex(_I, fill_value=0)
        page.render_viking_nine_backtest()

    at = AppTest.from_function(app, default_timeout=120)
    at.run()
    at.text_area(key="vnb_tickers").set_value("S0, S1, S2, S3")
    at.selectbox(key="vnb_min_nine").set_value(8)
    at.checkbox(key="vnb_compare_risk").check()
    at.button(key="FormSubmitter:vnb_form-⚔️ Kör backtest").click().run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "JÄMFÖRELSE AV EXITREGLER OCH RISKSPÄRR" in html and "Spärrade" in html
    assert "Spärr Per marknad (som live)" in html and "Spärr Av" in html
    assert "Marknadsriskspärr:" in html and "signaler spärrade" in html
    runs = at.session_state["vnb_result"]["runs"]
    assert runs["Spärr Av"]["risk_blocked"] == 0 and runs["Spärr Per marknad (som live)"]["risk_blocked"] > 0
    assert runs["Spärr Per marknad (som live)"]["metrics"]["trades"] < runs["Spärr Av"]["metrics"]["trades"]
