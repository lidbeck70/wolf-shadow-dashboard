"""
⚔️ Viking Nine Regime — beslutssidan för en aktie: marknadslagret, OVTLYR
Nine, graf, Viking Execution + risk, exit och signallogg. Tickern kan väljas
ur senaste ⚔️ Viking Nine-skanningen. Syntetiska kurser — inget nätverk.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import ovtlyr_nine as on  # noqa: E402
from test_viking_screen import DATA, _run  # noqa: E402


def _patch(monkeypatch):
    import market_prices
    import storage
    import streamlit as st
    from ovtlyr.ui import viking_screens
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, default))
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(market_prices, "ohlcv", lambda t, p="1y": DATA.get(t, pd.DataFrame()))
    monkeypatch.setattr(viking_screens, "_sector", lambda t: {"GOOD": "Technology", "BAD": "Energy"}.get(t))
    monkeypatch.setattr(viking_screens, "_earnings", lambda t: pd.Timestamp("2026-12-01"))
    import ovtlyr.ui.viking_nine_regime as page
    monkeypatch.setattr(page, "_sector", lambda t: {"GOOD": "Technology", "BAD": "Energy"}.get(t))
    monkeypatch.setattr(page, "_earnings", lambda t: pd.Timestamp("2026-12-01"))
    monkeypatch.setattr(on, "STALE_BDAYS", 10 ** 6)          # testdatan slutar 2026-09-30 — får inte bli gammal
    monkeypatch.setenv("VNR_TEST_ROOT", ROOT)


def _app():
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.environ["VNR_TEST_ROOT"])
    from ovtlyr.ui.viking_nine_regime import render_viking_nine_regime_page
    render_viking_nine_regime_page()


def test_page_shows_the_full_decision_chain(monkeypatch):
    from streamlit.testing.v1 import AppTest
    _patch(monkeypatch)
    at = AppTest.from_function(_app, default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    at.text_input(key="vnr_ticker").set_value("GOOD").run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "MARKET SPY 3/3" in html and "OVTLYR NINE" in html and "TOTAL 9 / 9" in html
    assert "VIKING EXECUTION" in html and "RISK ENGINE" in html and "VIKING EXIT ENGINE" in html
    assert any("GOOD" in c.proto.spec and "EMA10" in c.proto.spec for c in at.get("plotly_chart"))
    assert "Kör ⚔️ Viking Nine under SCREENING" in html                    # ingen skanning än
    at.text_input(key="vnr_ticker").set_value("NOPE").run()
    assert not at.exception and "DATA UNAVAILABLE" in " ".join(m.value for m in at.markdown)


def test_ticker_can_be_picked_from_the_last_scan(monkeypatch):
    from streamlit.testing.v1 import AppTest
    _patch(monkeypatch)
    rows = _run(("GOOD", "BAD"))
    at = AppTest.from_function(_app, default_timeout=90)
    at.session_state["vn_screen_rows"] = {"rows": rows, "funnel": {}, "when": "x"}
    at.run()
    assert not at.exception, at.exception
    opts = at.selectbox(key="vnr_pick").options
    assert opts[0] == "—" and opts[1].startswith("GOOD · ") and opts[2] == "BAD · REJECTED"
    at.selectbox(key="vnr_pick").set_value("BAD · REJECTED").run()
    assert at.text_input(key="vnr_ticker").value == "BAD"
    html = " ".join(m.value for m in at.markdown)
    assert "NO TRADE" in html                                               # BAD ≤ 6/9
    at.text_input(key="vnr_ticker").set_value("GOOD").run()                 # fritext vinner efter valet
    assert at.text_input(key="vnr_ticker").value == "GOOD"


def test_navigation_and_guide():
    from ui import nav
    from ovtlyr.ui.rules_page import _PANEL_GUIDE
    opts = nav.options("regime/Marknad/Arc Regime")
    assert "⚔️ Viking Nine Regime" in opts and "Viking Regime" in opts and "Wolf Regime" in opts
    assert any(t == "REGIME → Marknad → Arc Regime → ⚔️ Viking Nine Regime" for t, _r, _u in _PANEL_GUIDE)
    src = open(os.path.join(ROOT, "wolf_panel.py"), encoding="utf-8").read()
    assert 'elif inner == "⚔️ Viking Nine Regime":' in src
