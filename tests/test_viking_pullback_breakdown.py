"""
Viking Nine-backtestet: entry efter pullback till EMA20 (en hypotes att
testa, inte live-regeln) och uppdelningen per land och per ticker.
Syntetiska kurser — inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import viking_backtest as vb  # noqa: E402
from tests.test_viking_backtest import DATA, IDX, ROOT  # noqa: E402

TODAY = pd.Timestamp("2026-09-30")


def _run(tickers=("S0", "S1", "S2", "S3"), **cfg):
    return vb.run(list(tickers), getter=lambda t, p: DATA.get(t), sector_getter=lambda t: "Technology",
                  cfg=vb.Config(min_nine=8, years=3, **cfg), today=TODAY)


def test_pullback_flag_is_a_recent_touch_of_ema20():
    n = 120
    idx = pd.bdate_range(end="2026-09-30", periods=n)
    close = 100 + 0.5 * np.arange(n, dtype=float)
    df = pd.DataFrame({"Open": close - 0.1, "High": close + 0.3, "Low": close - 0.3, "Close": close,
                       "Volume": 1e6}, index=idx)
    x = vb.execution_frame(df)
    assert not x["pullback"].iloc[-1]                                          # lång rusning, ingen pullback
    dip = df.copy()
    k = n - 4
    dip.iloc[k, dip.columns.get_loc("Low")] = float(x["ema20"].iloc[k]) - 0.1   # låget når EMA20 för 3 dagar sedan
    xd = vb.execution_frame(dip)
    assert xd["pullback"].iloc[-1] and xd["pullback"].iloc[k]
    assert not xd["pullback"].iloc[k - 1]                                      # inte före dippen (kausalt)
    old = df.copy()
    k = n - 1 - vb.PULLBACK_DAYS
    old.iloc[k, old.columns.get_loc("Low")] = float(x["ema20"].iloc[k]) - 0.1   # för länge sedan
    assert not vb.execution_frame(old)["pullback"].iloc[-1]


def test_pullback_is_causal():
    stock = DATA["S0"]
    full = vb.execution_frame(stock)["pullback"]
    for i in (300, 500, 700):
        assert full.iloc[i] == vb.execution_frame(stock.iloc[:i + 1])["pullback"].iloc[-1]


def test_pullback_entry_only_takes_signals_after_a_pullback():
    live, pull = _run(), _run(pullback=True)
    assert vb.Config().pullback is False                                       # live-regeln oförändrad
    assert pull["no_pullback"] > 0 and live["no_pullback"] == 0
    assert len(pull["trades"]) < len(live["trades"])
    for t in pull["trades"]:
        x = vb.execution_frame(DATA[t.ticker])
        assert x["pullback"].loc[pd.Timestamp(t.signal_date)]


def test_run_names_with_entry():
    from ovtlyr.ui.viking_nine_backtest import run_names
    first, gate = next(iter(vb.EXIT_PRESETS)), next(iter(vb.RISK_GATES))
    runs = run_names(first, gate, False, False, None, True)
    assert [n for n, *_ in runs] == [f"Entry {e}" for e in vb.ENTRY_MODES]
    assert [en for *_, en in runs] == list(vb.ENTRY_MODES)
    assert len(run_names(first, gate, True, True, None, True)) == \
        len(vb.EXIT_PRESETS) * len(vb.RISK_GATES) * len(vb.ENTRY_MODES)


def test_breakdown_per_country_and_ticker():
    from ovtlyr.ui.viking_nine_backtest import country_of, group_rows, ticker_rows
    assert country_of("VOLV-B.ST") == "Sverige" and country_of("EQNR.OL") == "Norge"
    assert country_of("NOVO-B.CO") == "Danmark" and country_of("NOKIA.HE") == "Finland"
    assert country_of("NVDA") == "USA/övriga"
    res = _run()
    rows = group_rows(res["trades"], lambda t: t.ticker)
    assert [r["Summa R"] for r in rows] == sorted(r["Summa R"] for r in rows)   # sämst först
    closed = [t for t in res["trades"] if not t.open]
    assert sum(r["Affärer"] for r in rows) == len(closed)
    assert sum(r["Summa R"] for r in rows) == pytest.approx(sum(t.r for t in closed), abs=0.05)
    tr = ticker_rows(res)
    assert len(tr) == 4 and tr[0]["Summa R"] == rows[0]["Summa R"] and tr[0]["Land"] == "USA/övriga"


def test_page_shows_entry_and_breakdown(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("VNB_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["VNB_TEST_ROOT"])
        import market_prices
        import pandas as _pd
        import ovtlyr.ui.viking_nine_backtest as page
        from tests.test_viking_backtest import DATA as _D
        market_prices.ohlcv = lambda t, p="1y", **k: _D.get(t, _pd.DataFrame())
        page._sector = lambda t: "Technology"
        page._risk_points = lambda m: None
        page.render_viking_nine_backtest()

    at = AppTest.from_function(app, default_timeout=120)
    at.run()
    at.text_area(key="vnb_tickers").set_value("S0, S1, S2, S3")
    at.selectbox(key="vnb_min_nine").set_value(8)
    at.checkbox(key="vnb_compare_entry").check()
    at.button(key="FormSubmitter:vnb_form-⚔️ Kör backtest").click().run()
    assert not at.exception, at.exception
    runs = at.session_state["vnb_result"]["runs"]
    assert set(runs) == {f"Entry {e}" for e in vb.ENTRY_MODES}
    html = " ".join(m.value for m in at.markdown)
    assert "PER LAND" in html and "USA/övriga" in html and "Entry:" in html
    at.selectbox(key="vnb_show").set_value("Entry Efter pullback till EMA20").run()
    html = " ".join(m.value for m in at.markdown)
    assert "efter pullback till EMA20" in html and "signaler utan pullback" in html
