"""
Thor fas 5 — thor/ravara: temats ordförråd mot varje motor (ren logik) och Råvara-fliken under AppTest med
fejkade motorer: BERSERK-kortet, Valhall-temat, Triple Signal med kartboken, Odins blindspot, hävstången,
debasement och kvoterna — och att ett block som fallerar blir en notering, inte ett fel. Inget nätverk.
"""
import os
import sys
import zlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_candidates as fx  # noqa: E402
from test_wolf_regime_score import _walk  # noqa: E402
from thor import dossier as td  # noqa: E402
from thor import ravara as rv  # noqa: E402
from thor import ui as tui  # noqa: E402

TODAY = pd.Timestamp.today().normalize()
ROT = {"month": "2026-10", "grades": {"koppar": {"hatred": 5, "fundamentals": 4, "catalyst": 4, "case_intact": True}},
       "history": []}


def _prices(ticker, period="1y"):
    return _walk(zlib.crc32(ticker.encode()) % 1000, 0.0006, end=TODAY)


def patch_engines(monkeypatch):
    """Alla motorer bakom Råvara-fliken utan nätverk — delas med test_thor_ui."""
    import market_prices
    import storage
    import streamlit as st
    from alpha_regime import commodity_ratios as cr
    from berserk import live
    from blindspot import theme_board as tb
    from commodity_leverage import engine as ce
    from commodity_leverage import ui as clu
    from fiat_debasement import environment as env
    from fiat_debasement import ui as fu
    from gold_ratios import ui as gu
    monkeypatch.setattr(market_prices, "ohlcv", _prices)
    monkeypatch.setattr(tui, "_prices", _prices)
    monkeypatch.setattr(tui, "_ohlcv", lambda t: _prices(t))
    monkeypatch.setattr(live, "load_markets", lambda regions, getter=None, nordic_provider=None: {})
    monkeypatch.setattr(tb, "build_theme_board", lambda: [
        tb.ThemeResult(key="koppar", label="Koppar", necessity=80, cykel_label="TIDIG", percentile_10y=22.0, ma200w_slope_pct=1.2,
                       hat_score=61.0, blindspot_score=30.0, current_price=4.1),
        tb.ThemeResult(key="guld", label="Guld", necessity=90, cykel_label="TOPP", percentile_10y=99.0, hat_score=10.0,
                       blindspot_score=3.0)])

    def leverage(ticker, commodity, force):
        return ce.CompanyLeverage(ticker=ticker, name="Bolaget", commodity=commodity, beta_error="test")

    def environment(ccy="USD", compute=None):
        return {"currency": "USD", "level": "Neutral", "value": 55.0, "as_of": "2026-09"}

    monkeypatch.setattr(clu, "_get", leverage)
    monkeypatch.setattr(env, "current", environment)
    idx = pd.date_range("2005-01-01", periods=250, freq="MS")
    monkeypatch.setattr(fu, "real_commodity", lambda name, base_year, loader=None, asset_loader=None: {
        "nominal": pd.Series(np.linspace(2.0, 4.5, 250), index=idx), "real": pd.Series(np.linspace(2.5, 3.8, 250), index=idx),
        "vs_gold": None if name == "gold" else pd.Series(np.linspace(0.004, 0.002, 250), index=idx),
        "base": pd.Timestamp(f"{base_year}-01-01"), "label": {"copper": "Koppar", "gold": "Guld"}.get(name, name),
        "unit": "USD/lb", "notes": [], "segments": []})
    monkeypatch.setattr(fu, "gs_ratio", lambda asset_loader=None: (
        pd.Series(np.linspace(60, 85, 300), index=pd.bdate_range(end=TODAY, periods=300)),
        [{"Period": "10 år", "Från": "2016", "Snitt": 78.0, "Median": 79.0, "Min": 60.0, "Max": 120.0,
          "Nu mot perioden (percentil)": 71.0}]))
    monkeypatch.setattr(gu, "_load", lambda: {"koppar": {"series": pd.Series(np.linspace(600, 900, 3000),
                                                                             index=pd.bdate_range(end=TODAY, periods=3000))}})
    monkeypatch.setattr(cr, "fetch_all_ratios", lambda: {
        "copper_gold": cr.RatioResult(key="copper_gold", label="Copper / Gold", denominator_label="Gold", cheap_direction="low",
                                      current=0.0011, percentile=8.0, zscore=-1.9, status="RUBBER_BAND_STRETCHED",
                                      driver="NUMERATOR_CHEAP")})
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None: st.session_state.setdefault(name, default))


@pytest.fixture
def engines(monkeypatch):
    patch_engines(monkeypatch)


# ── Ordförråden ──────────────────────────────────────────────────────────────
def test_vocabularies_map_the_berserk_theme_to_every_engine():
    assert rv.rotation_key("koppar") == "koppar" and rv.rotation_key("naturgas") == "gas" and rv.rotation_key("zink_nickel") == "zink"
    assert rv.rotation_key("lax") is None
    assert rv.blindspot_key("majs") == "agri" and rv.blindspot_key("uran") == "uran" and rv.blindspot_key("tank") is None
    assert rv.leverage_key("guld") == "guld" and rv.leverage_key("uran") is None
    assert rv.fiat_asset("olja") == "oil" and rv.fiat_asset("uran") is None
    assert rv.gold_pair("koppar")["ticker"] == "HG=F" and rv.gold_pair("guld") is None
    assert rv.ratio_keys("koppar") == ["copper_gold"] and "metal_miners" in rv.ratio_keys("guld") and rv.ratio_keys("kol") == []
    assert rv.coverage("uran") == {"rotation": "uran", "blindspot": "uran", "leverage": None, "fiat": None, "gold_pair": None,
                                   "ratios": [], "gold_silver": False}
    assert rv.coverage("silver")["gold_silver"] and rv.coverage("silver")["ratios"] == ["gold_silver", "silver_juniors"]
    assert rv.label("koppar") == "Koppar" and rv.label("okänt") == "Okänt"


def test_rotation_state_and_valhall_theme():
    s = rv.rotation_state("koppar", ROT)
    assert s["status"] == "AGERA" and s["sum"] == 13 and s["max"] == 15 and s["month"] == "2026-10"
    assert rv.rotation_state("guld", ROT) is None and rv.rotation_state("lax", ROT) is None and rv.rotation_state("koppar", None) is None
    t, sw, buys = rv.valhall_theme(fx.BOARD, "guld")
    assert t["label"] == "Guld" and sw is None and buys["leverage"]["strategy"] == "Durrett"
    assert rv.valhall_theme(fx.BOARD, "koppar") is None and rv.valhall_theme(None, "guld") is None
    assert td.sheets_for("BOL.ST", {"rotation": ROT})["rotation"] == ROT                     # arket följer med dossiern


# ── Fliken ───────────────────────────────────────────────────────────────────
def _script():
    import streamlit as st
    from thor import dossier as d
    from thor import ravara as r
    from thor import ui as t
    ident = d.identify(st.session_state["_ticker"])
    r.render(st, ident, {"board": st.session_state["_board"]}, st.session_state.get("_rot"), t._prices)


def _run(ticker, rot=ROT):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_script, default_timeout=120)
    at.session_state["_ticker"], at.session_state["_board"], at.session_state["_rot"] = ticker, fx.BOARD, rot
    at.run()
    assert not at.exception, at.exception
    return at, " ".join(m.value for m in at.markdown)


def test_tab_renders_every_block_for_a_copper_stock(engines):
    at, text = _run("BOL.ST")
    for part in ("Koppar · källa: BERSERK-listan", "Råvarucykeln", "HG=F", "finns inte som tema på Valhall-kartan",
                 ">AGERA<", "13/15", "TIDIG", "61/100", "plats 1 av 2", "HIGHT POTENTIELL BLINDSPOT",
                 "Hävstång mot råvaran", "RESULTATHÄVSTÅNG", "KURSBETA", "FIAT ENVIRONMENT USD", "Neutral (55/100)",
                 "Koppar nominellt", "I 2010 års dollar", "Mätt i guld", "Guld/koppar", "Copper / Gold", "P8",
                 "rubber band stretched", "Samma motorer som Odins visdom"):
        assert part in text, part
    assert "Kartboken: Koppar" in [e.label for e in at.expander] and len(at.get("plotly_chart")) == 1   # realprisgrafen
    assert "median 79" not in text                                                                      # guld/silver: inte för koppar


def test_tab_for_a_gold_stock_shows_valhall_theme_ungraded_rotation_and_gold_silver(engines):
    at, text = _run("NEM")
    for part in ("Guld · källa", "BAISSE · VÄNDER", "Hävstång (Durrett)", "INTE BETYGSATT", "saknar månadens Triple Signal",
                 "TOPP", "Guld nominellt", ">85.0<", "median 79", "P71", "ackumuleringszon"):
        assert part in text, part
    assert "Kartboken: Guld" in [e.label for e in at.expander] and "Guld/koppar" not in text


def test_tab_without_a_theme_says_so(engines):
    _at, text = _run("ERIC-B.ST")
    assert "inte knuten till någon råvara" in text and "Råvarucykeln" not in text


def test_a_failing_block_is_a_note_and_the_rest_renders(engines, monkeypatch):
    from blindspot import theme_board as tb

    def boom():
        raise RuntimeError("nätet nere")
    monkeypatch.setattr(tb, "build_theme_board", boom)
    _at, text = _run("BOL.ST")
    assert "Odins blindspot kunde inte läsas: nätet nere" in text and "Guld/koppar" in text and ">AGERA<" in text


def test_engines_that_do_not_cover_the_theme_say_so(engines):
    from thor import ticker as tt
    assert tt.identify("CCJ").theme == "uran"                                    # Cameco → uran (BERSERK-listan)
    at, text = _run("CCJ")
    assert "Hävstångsmotorn täcker" in text and "Realprisvyn finns för guld, silver, koppar och olja" in text
    assert "Inga kvoter för uran" in text and "Kartboken: Uran" in [e.label for e in at.expander]
