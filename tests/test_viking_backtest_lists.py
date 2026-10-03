"""
Viking Nine-backtestet: fasta tickerlistor (Norden 50, USA 25) som fyller
tickerfältet. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ovtlyr_nine as on  # noqa: E402
import viking_backtest as vb  # noqa: E402
import viking_screen as vs  # noqa: E402
from tests.test_viking_backtest import ROOT  # noqa: E402


def test_lists_are_clean():
    assert len(vb.NORDIC_50) == 50 and len(vb.US_25) == 25
    both = vb.TICKER_LISTS["Norden 50 + USA 25"]
    assert len(both) == len(set(both)) == 75
    assert all(t == t.upper() for t in both)
    assert all(on.market_for(t) == on.NORDIC_LABEL for t in vb.NORDIC_50)        # testas mot OMXS30
    assert all(on.market_for(t) == "SPY" for t in vb.US_25)
    assert "FNV" in vb.US_25 and "BOL.ST" in vb.NORDIC_50 and "VAR.OL" in vb.NORDIC_50
    assert vs.parse_tickers(", ".join(vb.NORDIC_50)) == list(vb.NORDIC_50)


def _app():
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.environ["VNB_TEST_ROOT"])
    from ovtlyr.ui.viking_nine_backtest import render_viking_nine_backtest
    render_viking_nine_backtest()


def test_choosing_a_list_fills_the_field(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("VNB_TEST_ROOT", ROOT)
    at = AppTest.from_function(_app, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    assert at.text_area(key="vnb_tickers").value == ""                         # ingen skanning än
    at.selectbox(key="vnb_list").set_value("USA 25").run()
    assert at.text_area(key="vnb_tickers").value == ", ".join(vb.US_25)
    at.selectbox(key="vnb_list").set_value("Norden 50").run()
    assert at.text_area(key="vnb_tickers").value.startswith("CBRAIN.CO, DLAB.ST")
    at.text_area(key="vnb_tickers").set_value("NVDA, MSFT").run()                # går att ändra
    assert at.text_area(key="vnb_tickers").value == "NVDA, MSFT" and not at.exception
