"""
Odins Blindspot: temakartan utökad med platina, palladium, vete, skog, kaffe
och kakao (sällsynta metaller fanns redan). Varje tema når EMBER (sektor-ETF,
komplex, nyckelord) och Confidence-registret, så cykelfasen följer med.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blindspot import theme_board as tb  # noqa: E402
from confidence import commodities as com  # noqa: E402
from ember import config as ec  # noqa: E402

NEW = {"platina": ("PPLT", "Platina"), "palladium": ("PALL", "Palladium"), "vete": ("WEAT", "Vete"),
       "skog": ("WOOD", "Skog"), "kaffe": ("KC=F", "Kaffe"), "kakao": ("CC=F", "Kakao")}


def test_the_board_has_the_new_themes_with_their_series():
    specs = {s.key: s for s in tb._THEMES}
    assert len(specs) == 15 and "sallsynta" in specs
    for key, (ticker, label) in NEW.items():
        assert specs[key].primary_tickers[0] == ticker and specs[key].label == label
        assert 0 < specs[key].necessity <= 100
    assert specs["platina"].primary_tickers[1] == "PL=F" and specs["vete"].primary_tickers[1] == "ZW=F"
    assert specs["kaffe"].proxy_flag and "KC=F" in specs["kaffe"].proxy_note
    assert specs["vete"].necessity > specs["kaffe"].necessity > specs["kakao"].necessity


def test_a_futures_theme_is_computed_like_the_others(monkeypatch):
    idx = pd.bdate_range("2016-01-01", periods=2600)
    price = pd.Series(np.linspace(100, 300, 2600), index=idx)
    price.iloc[-60:] = np.linspace(300, 120, 60)                         # rasat — nära botten
    monkeypatch.setattr(tb, "_download_close", lambda t, p="10y": price if t in ("KC=F", "SPY") else pd.Series(dtype=float))
    monkeypatch.setattr(tb, "_download_volume", lambda t, p="2y": pd.Series(np.ones(504) * 1000.0, index=idx[-504:]))
    spec = next(s for s in tb._THEMES if s.key == "kaffe")
    r = tb._compute_theme(spec)
    assert r.error is None and r.ticker_used == "KC=F"
    assert r.cykel_label in ("TIDIG", "MITTEN") and r.blindspot_score > 0 and r.sparkline_values


def test_ember_knows_every_board_theme():
    for s in tb._THEMES:
        assert s.key in ec.EMBER_SECTOR_ETF, s.key
        assert ec.THEME_TO_COMPLEX.get(s.key) in ec.COMPLEX_LABEL, s.key
        assert s.key in ec._THEME_LABEL, s.key
    assert ec.THEME_TO_COMPLEX["platina"] == ec.THEME_TO_COMPLEX["palladium"] == "adelmetaller"
    assert {ec.THEME_TO_COMPLEX[k] for k in ("vete", "kaffe", "kakao", "skog")} == {"agri"}
    assert ec.COMMODITY_TO_THEME["platinum"] == "platina" and ec.COMMODITY_TO_THEME["palladium"] == "palladium"


def _first(table, text):
    t = text.lower()
    return next((theme for kw, theme in table if kw in t), None)


def test_keywords_route_to_the_new_themes_before_the_broad_ones():
    ind = ec.INDUSTRY_KEYWORD_THEME
    assert _first(ind, "Basic Materials Other Precious Metals & Mining Sibanye Platinum") == "platina"
    assert _first(ind, "Basic Materials Lumber & Wood Production") == "skog"
    assert _first(ind, "Basic Materials Paper & Paper Products Holmen") == "skog"
    assert _first(ind, "Consumer Defensive Confectioners cocoa") == "kakao"
    assert _first(ind, "Basic Materials Gold") == "guld"                    # guld oförändrat
    sec = ec.SECTOR_KEYWORD_THEME
    assert _first(sec, "Skogsbolag") == "skog" and _first(sec, "Palladium") == "palladium"
    assert _first(sec, "Kaffe & kakao") == "kaffe" and _first(sec, "Vete") == "vete"


def test_confidence_registry_crosswalks_the_new_themes():
    assert com.resolve_key("vete") == "wheat" and com.resolve_key("Kaffe") == "coffee"
    assert com.resolve_key("kakao") == "cocoa" and com.resolve_key("Skog") == "timber"
    assert com.crosswalk("platinum")["theme"] == "platina" and com.crosswalk("palladium")["theme"] == "palladium"
    assert com.resolve_key("Gruv - Guld & Silver") == "gold"               # oförändrat


def test_contrarian_counts_pgm_as_commodity_but_not_the_softs():
    from contrarian_alpha.commodity_gate import COMMODITY_THEMES
    assert {"platina", "palladium"} <= COMMODITY_THEMES
    assert not ({"vete", "kaffe", "kakao", "skog"} & COMMODITY_THEMES)     # Contrarian: inte jordbruk/skog
