"""
Thor-granskningen 2026-10-23, PR D — cacher och bryggor: Market Cycle Engine cachar inte ett nätverksfel i en timme
(och skyller inte på symbolen), Long Screener-layouten säger när fundamenta saknas, momentumsidans sessionscache
sparar inte None, Thors 📡 Uppdatera rensar även dem, och bryggknapparna förifyller målsidornas tickerfält.
Inget nätverk.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_thor_regim import _prices, _run, engines  # noqa: E402,F401,F811
from thor import ui as tui  # noqa: E402


def test_market_cycle_network_failure_is_not_cached(monkeypatch):
    import cycle as cy
    from market_cycle import cache as mc
    from market_cycle import indicators as mi
    tk = "NEMTEST"
    monkeypatch.setattr(mc, "compute_indicators", lambda t, p: {})
    assert mc.cached_market_cycle_analysis(tk, "1y") == {}
    state, err = cy.market_phase(tk)
    assert state is None and "Kursdata kunde inte hämtas" in err and "Kontrollera" not in err
    good = mi._compute_from_df(_prices("NEM"))
    monkeypatch.setattr(mc, "compute_indicators", lambda t, p: good)
    data = mc.cached_market_cycle_analysis(tk, "1y")                     # nätet tillbaka: inget tomt svar låg kvar
    assert data and data["result"].get("phase")
    state, err = cy.market_phase(tk)
    assert err is None and state["phase"] == data["result"]["phase"]
    monkeypatch.setattr(mc, "compute_indicators", lambda t, p: {})
    assert mc.cached_market_cycle_analysis(tk, "1y")["result"]["phase"] == state["phase"]   # det lyckade svaret cachas


def test_long_screener_layout_says_when_fundamentals_are_missing(engines):  # noqa: F811
    _at, text = _run("long", "BOL.ST")                     # engines-fixturen ger {} för fundamenta
    assert "Fundamenta saknas" in text and "LONG SCREENER" in text
    from thor import regim
    assert regim._no_fundamentals({}) and regim._no_fundamentals({"EV/EBITDA": "N/A", "ROIC": None, "_data_source": "yfinance"})
    assert regim._no_fundamentals({"_data_source": "none", "ROIC": 12.0})
    assert not regim._no_fundamentals({"ROIC": 12.0, "_data_source": "borsdata"})


def test_momentum_session_cache_does_not_keep_none(monkeypatch):
    import streamlit as st
    import wolf_regime_ui
    import wolf_screener_ui
    calls = {"s": 0, "r": 0}

    def flaky(name):
        calls["s"] += 1
        return None if calls["s"] == 1 else {"generated": "2026-10-14T04:20", "top": []}
    monkeypatch.setattr(wolf_screener_ui, "_load_wolf", flaky)
    st.session_state.pop(wolf_screener_ui._CACHE, None)
    assert wolf_screener_ui._get_data() is None
    assert wolf_screener_ui._get_data() == {"generated": "2026-10-14T04:20", "top": []} and calls["s"] == 2
    assert wolf_screener_ui._get_data() and calls["s"] == 2                                   # och sedan cachat

    def flaky_r(name):
        calls["r"] += 1
        return None if calls["r"] == 1 else {"regime": "GRÖN"}
    monkeypatch.setattr(wolf_regime_ui, "_load_wolf", flaky_r)
    st.session_state.pop(wolf_regime_ui._CACHE, None)
    assert wolf_regime_ui._get_data() is None and wolf_regime_ui._get_data() == {"regime": "GRÖN"}
    for k in (wolf_screener_ui._CACHE, wolf_regime_ui._CACHE):
        st.session_state.pop(k, None)


def test_momentum_layout_names_the_unreadable_file(engines, monkeypatch):  # noqa: F811
    import wolf_screener_ui
    monkeypatch.setattr(wolf_screener_ui, "_get_data", lambda force=False: None)
    _at, text = _run("momentum", "ERIC-B.ST")
    assert "wolf_screener.json kunde inte läsas" in text


def test_bridges_prefill_the_target_pages_ticker_fields():
    want = {"wolf": "reg_ticker", "viking": "ovtlyr_ticker", "ember": "ember_regime_ticker_search",
            "viking_nine": "vnr_ticker", "momentum": "swreg_ticker"}
    for layout, key in want.items():
        assert tui.BRIDGE[layout][2] == key, layout
    src = {"reg_ticker": "tabs/regime.py", "ovtlyr_ticker": "ovtlyr/ui/layout.py",
           "ember_regime_ticker_search": "ember/regime.py"}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for key, path in src.items():
        with open(os.path.join(root, path), encoding="utf-8") as f:
            assert f'key="{key}"' in f.read(), (key, path)                      # målsidans widget heter så
    import streamlit as st
    tui._jump("odin/marknad/arc-regime/wolf-regime", "reg_ticker", "BOL.ST")
    assert st.session_state["reg_ticker"] == "BOL.ST" and st.session_state["nav_goto"].endswith("wolf-regime")
    st.session_state.pop("reg_ticker", None)
    st.session_state.pop("nav_goto", None)


def test_thor_refresh_clears_the_page_caches(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import test_thor_ui as tu
    tu._patch(monkeypatch)
    at = AppTest.from_function(tu._script)
    at.session_state["_wolf_screener_cache"] = None
    at.session_state["_wolf_regime_cache"] = {"regime": "RÖD"}
    at.run(timeout=60)
    assert not at.exception
    assert "_wolf_screener_cache" not in at.session_state and "_wolf_regime_cache" not in at.session_state
    assert isinstance(pd.Timestamp.now(), pd.Timestamp)
