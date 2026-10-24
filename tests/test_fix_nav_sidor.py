"""Granskning 2 (navigering och sidor): länkar landar rätt och stannar, sidorna klarar trasig data."""
import pathlib

from tabs import home_today as ht
from ui import links, nav

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _script():
    import streamlit as st
    from ui import nav
    import wolf_panel as wp
    wp._apply_deep_link()
    tabs = st.tabs([nav.tab_label(k) for k, _l in nav.TOP], on_change="rerun", key=nav.TABS_KEY)
    keys = [k for k, _l in nav.TOP]
    with tabs[keys.index("portfolio")]:
        st.radio("p", nav.options("portfolio"), key=nav.STATE_KEY["portfolio"])
        st.button("klick", key="klick")
    wp._sync_url()


def test_deep_link_keeps_its_tab_after_several_clicks():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_script)
    at.query_params["p"] = "portfolio/mot-malet"
    at.run()
    for _ in range(4):
        at.button(key="klick").click().run()
        assert not at.exception
    p = at.query_params["p"]
    assert (p[0] if isinstance(p, list) else p) == "portfolio/mot-malet"


def test_ticker_only_travels_to_thor():
    assert nav.target_link("holdings", "https://p", ticker="ABC.ST") == "https://p/?p=portfolio/positioner"
    assert "&t=ABC.ST" in nav.target_link("thor", "https://p", ticker="ABC.ST")
    src = (ROOT / "wolf_panel.py").read_text(encoding="utf-8")
    assert 'if segs[0] != "thor":' in src and 'del st.query_params[k]' in src


def test_market_risk_link_and_backtests_button_open_the_hidden_part():
    assert nav.target_link("market_risk").endswith("&risk=1")
    state = {}
    links.go("backtests", state=state)
    assert state["goal_backtests"] is True


def test_old_links_open_their_inner_tab():
    assert nav.inner_seed("odin/ravaror/guldkvoter") == ("odin_gold_tabs", "🥇 Guldkvoter")
    assert nav.inner_seed("regime/ravaror/guldkvoter") == ("odin_gold_tabs", "🥇 Guldkvoter")
    assert nav.inner_seed("intelligence/retail-pulse") == ("odin_sentiment_tabs", "Retail")
    assert nav.inner_seed("odin/makro/sentiment") is None


def test_partial_sell_shows_on_the_start_page():
    state = {"swing": {"verdicts": {"AAA.ST": {"verdict": "DELVINST — sälj halva", "reason": "+20 %"},
                                    "BBB.ST": {"verdict": "SÄLJ", "reason": "MA50"},
                                    "CCC.ST": {"verdict": "BEVAKA", "reason": "under EMA20"}}}}
    assert ht.holdings_lines(state) == [("BBB.ST", "SÄLJ", "MA50"), ("AAA.ST", "MINSKA", "+20 %"),
                                        ("CCC.ST", "VAKSAM", "under EMA20")]


def test_garbage_shapes_never_crash():
    from odin import laget
    from portfolio import mot_malet as mm
    for bad in ([], "x", {"swing": []}, {"swing": {"verdicts": []}}):
        assert ht.holdings_lines(bad) == []
    assert ht.turn_cards([]) == [] and ht.turn_cards({"turn_plan": []}) == [] and ht.on_the_way({"turns": "x"}) == 0
    assert laget.risk_light({"levels": []})[0] == "Saknas" and laget.nordic_light({"segments": "x"})[0] == "Saknas"
    assert mm.strategy_rows([], []) == [] and "för få" in mm.today_line(None, [], [])


def test_journal_page_has_no_shadowed_color_function():
    src = (ROOT / "trade_journal.py").read_text(encoding="utf-8")
    assert "_cagr_color = _cagr_color(" not in src


def test_shares_field_is_per_stock_and_strategy():
    src = (ROOT / "thor" / "ui.py").read_text(encoding="utf-8")
    assert 'key=f"thor_buy_shares_{strategy}_{ident.yf}"' in src
