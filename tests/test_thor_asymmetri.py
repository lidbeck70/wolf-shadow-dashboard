"""
Thor — Asymmetri-fliken: riktpriset (median av skattningar ur egen historik), kurshävstången (drivaren ±10/20/30 %
med multipeln fast) och tillväxtscenarierna, räknade för hand på samma syntetiska bolag som Snabbkollens egna
tester; HTML-byggarna; cachen; och fliken under AppTest med fejkad hämtning (inget nätverk).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asymmetry import quick_config as qc  # noqa: E402
from thor import asymmetri as ta  # noqa: E402
from thor import ui as tui  # noqa: E402

YEARS = list(range(2016, 2026))
REV = {y: 10000.0 * 1.05 ** i for i, y in enumerate(YEARS)}                 # +5 %/år
MARGINS = dict(zip(YEARS, [12, 13, 12, 14, 12, 11, 12, 9, 8, 8]))             # median 12, nu 8
PRICES = dict(zip(YEARS, [2.5, 2.8, 3.0, 2.7, 2.9, 4.2, 4.0, 3.8, 4.1, 4.3]))  # koppar


def value_co(**kw) -> dict:
    eb = {y: REV[y] * MARGINS[y] / 100 for y in YEARS}
    ebitda_now = REV[2025] * 0.08
    d = {"ticker": "VAL.ST", "name": "Value AB", "source": "Börsdata", "currency": "SEK", "price_currency": "SEK",
         "revenue_series": list(REV.items()), "ebitda_series": list(eb.items()), "ebitda_margin_pct": 8.0,
         "ev_ebitda_hist": [8.0, 9.0, 10.0, 11.0, 12.0], "ev_ebitda": 6.0, "net_debt": 2000.0, "nd_ebitda": 1.6,
         "mcap_bd": 6 * ebitda_now - 2000.0, "price": 100.0, "shares_growth_3y_pct": 2.0,
         "roic_series": [(y, 15.0) for y in YEARS], "commodity_px": {"commodity": "", "ticker": ""},
         "prices": [("2026-10-13", 99.0), ("2026-10-14", 100.0)], "fetched": "nu"}
    d.update(kw)
    return d


def copper_co(**kw) -> dict:
    eb = [(y, 1000.0 * p - 2000) for y, p in PRICES.items()]                 # EBITDA = 1000·P − 2000
    d = {"ticker": "CU.X", "name": "Copper Co", "source": "Börsdata", "currency": "USD", "price_currency": "USD",
         "revenue_series": [(y, 1000.0 * p) for y, p in PRICES.items()], "ebitda_series": eb,
         "fcf_series": [(y, v - 500) for y, v in eb],
         "commodity_px": {"commodity": "koppar", "ticker": "HG=F", "prices": dict(PRICES), "p0": 4.0, "fx_now": 1.0},
         "ev_ebitda_hist": [4.0, 5.0, 6.0, 7.0, 8.0], "ev_ebitda": 7.0, "net_debt": 1000.0, "nd_ebitda": 0.5,
         "mcap_bd": 11000.0, "price": 50.0, "shares_growth_3y_pct": 2.0,
         "prices": [("2026-10-13", 49.0), ("2026-10-14", 50.0)], "fetched": "nu"}
    d.update(kw)
    return d


# ── Riktpris ────────────────────────────────────────────────────────────────
def test_riktpris_value_company_is_the_median_of_the_estimates():
    d = value_co()
    r = ta.riktpris(d)
    assert r.error is None and r.mode == qc.MODE_VALUE and r.ccy == "SEK" and r.price_now == 100.0
    ebitda_now, mcap = REV[2025] * 0.08, 6 * REV[2025] * 0.08 - 2000
    by = {e.key: e for e in r.estimates}
    assert by["rerating"].price == pytest.approx(100 * (ebitda_now * 10 - 2000) / mcap, abs=0.01)      # ≈ 191
    assert by["normal"].price == pytest.approx(100 * (REV[2025] * 0.12 * 10 - 2000) / mcap, abs=0.01)  # ≈ 305
    rev5 = REV[2025] * 1.05 ** 5
    disc = (rev5 * 0.12 * 10 - 2000) / 1.1 ** 5
    dil = 1 / 1.02 ** (5 / 3)
    assert by["growth"].price == pytest.approx(100 * disc / mcap * dil, abs=0.02)                        # ≈ 240
    assert "fcf" not in by and "base" not in by                                   # ingen FCF-historik, inget råvarutema
    assert r.target == pytest.approx(by["growth"].price, abs=0.01) and r.low == by["rerating"].price
    assert r.high == by["normal"].price and r.upside_pct == pytest.approx(r.target - 100, abs=0.1)
    assert "median av 3 skattningar" in r.method and "2016–2025" in r.method
    assert "diskonterat 10%/år" in by["growth"].formula and "utspädning" in by["growth"].formula


def test_riktpris_adds_cashflow_and_commodity_base():
    d = value_co(fcf=800.0, p_fcf_hist=[10.0, 12.0, 14.0], p_fcf=9.0)
    by = {e.key: e for e in ta.riktpris(d).estimates}
    mcap = 6 * REV[2025] * 0.08 - 2000
    assert by["fcf"].price == pytest.approx(100 * 800 * 12 / mcap, abs=0.01) and "median-P/FCF 12" in by["fcf"].formula
    c = ta.riktpris(copper_co())
    by = {e.key: e for e in c.estimates}
    assert c.mode == qc.MODE_COMMODITY and by["base"].price == 50.0 and "5×-motorns BASE" in by["base"].note
    assert by["rerating"].price == pytest.approx(50 * (2300 * 6 - 1000) / 11000, abs=0.01)         # EBITDA 2025 = 2300
    from statistics import median
    assert set(by) == {"rerating", "normal", "growth", "base"}             # värdekedjan gäller även råvarubolag
    assert c.target == pytest.approx(median(e.price for e in c.used), abs=0.01) and len(c.used) == 4
    assert c.low <= c.target <= c.high and c.error is None


def test_riktpris_data_gaps():
    assert "kursen saknas" in ta.riktpris(value_co(price=None)).error
    bank = ta.riktpris(value_co(bd_branch_id=68))
    assert "för få skattningar" in bank.error and "bank" in bank.error and bank.target is None
    short = ta.riktpris(value_co(revenue_series=list(REV.items())[-3:], ebitda_series=[]))
    assert short.target is None and short.error
    assert "DATA_GAP" in ta.riktpris_html(bank)


# ── Kurshävstång ────────────────────────────────────────────────────────────
def test_share_leverage_value_company_is_ev_over_equity():
    lv = ta.share_leverage(value_co())
    assert lv.error is None and lv.driver == "EBITDA" and lv.unit == "M SEK"
    ebitda_now = REV[2025] * 0.08
    mcap = 6 * ebitda_now - 2000
    ev_over_eq = 6 * ebitda_now / mcap                                                  # ≈ 1,37
    assert [r["pct"] for r in lv.rows] == list(ta.LEVERAGE_STEPS)
    for r in lv.rows:
        assert r["elasticity"] == pytest.approx(ev_over_eq, abs=0.01)
        assert r["price"] == pytest.approx(100 * (1 + ev_over_eq * r["pct"] / 100), abs=0.05)
    assert lv.up == pytest.approx(ev_over_eq, abs=0.01) and lv.down == pytest.approx(ev_over_eq, abs=0.01)
    assert lv.financial == pytest.approx(2000 / mcap, abs=0.01) and "multipeln hålls fast" in lv.note


def test_share_leverage_commodity_company_follows_the_engine_chain():
    lv = ta.share_leverage(copper_co())
    assert lv.error is None and lv.driver == "Koppar" and lv.unit == "USD/lb"
    assert lv.financial == pytest.approx(1000 / 11000, abs=0.01)
    by = {r["pct"]: r for r in lv.rows}
    assert by[10.0]["driver"] == 4.4 and by[10.0]["price"] == pytest.approx(50 * (2400 * 6 - 1000) / 11000, abs=0.01)
    assert by[-30.0]["driver"] == 2.8 and by[-30.0]["price"] == pytest.approx(50 * (800 * 6 - 1000) / 11000, abs=0.01)
    assert by[10.0]["elasticity"] == pytest.approx(2.18, abs=0.01) and lv.up == lv.down == pytest.approx(2.18, abs=0.01)
    no_px = ta.share_leverage(copper_co(commodity_px={}), mode=qc.MODE_COMMODITY)
    assert "råvarupriset saknas" in no_px.error and "DATA_GAP" in ta.leverage_html(no_px)
    assert ta.share_leverage(copper_co(commodity_px={})).driver == "EBITDA"          # utan tema: värdekedjan
    assert "kursen saknas" in ta.share_leverage(value_co(price=0)).error


# ── Tillväxtscenarier ───────────────────────────────────────────────────────
def test_growth_table_rows_and_own_rate():
    g = ta.growth_table(value_co())
    assert g.error is None and g.own_pct == 5.0 and g.margin == 12.0 and g.multiple == 10.0 and g.ccy == "SEK"
    assert [r["growth_pct"] for r in g.rows] == [0.0, 5.0, 10.0, 15.0, 20.0]
    own = next(r for r in g.rows if r["own"])
    mcap = 6 * REV[2025] * 0.08 - 2000
    rev5 = REV[2025] * 1.05 ** 5
    assert own["growth_pct"] == 5.0 and own["revenue"] == round(rev5) and own["ebitda"] == round(rev5 * 0.12)
    assert own["ratio"] == pytest.approx((rev5 * 0.12 * 10 - 2000) / mcap, abs=0.01)
    assert own["price"] == pytest.approx(100 * own["ratio"], abs=0.5)
    assert own["cagr_pct"] == pytest.approx((own["ratio"] ** 0.2 - 1) * 100, abs=0.1)
    prices = [r["price"] for r in g.rows]
    assert prices == sorted(prices) and sum(r["own"] for r in g.rows) == 1
    assert "DATA_GAP" in ta.growth_html(ta.growth_table(value_co(bd_branch_id=68)))
    capped = ta.growth_table(value_co(revenue_series=[(y, 1000.0 * 1.6 ** i) for i, y in enumerate(YEARS)]))
    assert capped.own_pct == 30.0                                                  # egen tillväxt kapad (GROWTH_CAP)


def test_html_builders_are_plain_html():
    d = value_co(fcf=800.0, p_fcf_hist=[10.0, 12.0, 14.0])
    html = ta.riktpris_html(ta.riktpris(d))
    for text in ("Riktpris", "Spann", "median av 4", "Omvärdering", "Normal marginal", "Kassaflöde", "Tillväxt 5 år"):
        assert text in html, text
    lv = ta.leverage_html(ta.share_leverage(d))
    assert "Hävstång upp" in lv and "Hävstång ned" in lv and "Finansiell hävstång" in lv and "+30 %" in lv
    gh = ta.growth_html(ta.growth_table(d))
    assert "(egen takt)" in gh and "Omsättning år 5" in gh and "Per år" in gh
    assert "<script" not in html + lv + gh


def test_cache_is_shared_with_wolf_asymmetry_and_refreshes():
    from asymmetry import quick_ui as qui
    assert ta.CACHE_KEY == qui.CACHE_KEY and ta.CACHE_TTL_S == qui.CACHE_TTL_S

    class _St:
        session_state: dict = {}
    calls = []

    def fetch(ticker, api=None):
        calls.append((ticker, api))
        return {"ticker": ticker}
    st = _St()
    assert ta.cached(st, "VAL.ST", "api", fetch) == {"ticker": "VAL.ST"} and calls == [("VAL.ST", "api")]
    assert ta.cached(st, "VAL.ST", "api", fetch) == {"ticker": "VAL.ST"} and len(calls) == 1           # träff
    ta.cached(st, "VAL.ST", "api", fetch, force=True)
    assert len(calls) == 2 and "VAL.ST" in st.session_state[ta.CACHE_KEY]


# ── Fliken ──────────────────────────────────────────────────────────────────
def _script():
    import streamlit as st
    from thor import dossier as d
    from thor import ui as t
    ident = d.identify(st.session_state.get("_ticker", "VAL.ST"))
    t._tab_asymmetri(st, {}, ident, "rule", {}, [])


@pytest.mark.parametrize("ticker,fixture,expect", [("VAL.ST", value_co, "VÄRDEDRIVARE"),
                                                   ("CU.X", copper_co, "RÅVARUHÄVSTÅNG")])
def test_tab_renders_snabbkoll_with_thor_additions(monkeypatch, ticker, fixture, expect):
    from streamlit.testing.v1 import AppTest
    from asymmetry import quick_data
    monkeypatch.setattr(quick_data, "fetch", lambda t, api=None, **kw: fixture())
    monkeypatch.setattr(tui, "_api", lambda: None)
    at = AppTest.from_function(_script)
    at.session_state["_ticker"] = ticker
    at.run(timeout=60)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    for needle in ("WOLF ASYMMETRY SCORE", "RIKTPRIS", "HÄVSTÅNG PÅ AKTIEKURSEN", "TILLVÄXTSCENARIER", expect):
        assert needle in text, needle
    assert any("Wolf Asymmetry" in b.label for b in at.button)
    assert tui.TABS == ("Regim", "Teknik", "Cykel", "Råvara", "Asymmetri", "Arken", "Backtest", "Regler")
    assert tui.ASYMMETRY[0] == "review/wolf-asymmetry"


def test_tab_without_data_explains(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from asymmetry import quick_data
    monkeypatch.setattr(quick_data, "fetch", lambda t, api=None, **kw: {"ticker": t, "source": "—", "prices": []})
    monkeypatch.setattr(tui, "_api", lambda: None)
    at = AppTest.from_function(_script)
    at.run(timeout=60)
    assert not at.exception and "Hittade inget" in " ".join(m.value for m in at.markdown)
