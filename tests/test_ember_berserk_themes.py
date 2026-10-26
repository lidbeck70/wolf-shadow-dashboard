"""
Fas 8: EMBER har samma temalista som BERSERK, Valhall-kartan och universum-registret (berserk/themes.py är
källan). Varje tema har sektor-ETF, etikett, komplex och en plats på Odins temakarta; stål, lax, torrbulk och
de andra får sitt komplex i stället för "cykel okänd". FRAKT är det sjätte komplexet. Inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from berserk import themes as bth  # noqa: E402
from blindspot import theme_board as tb  # noqa: E402
from confidence import commodities as com  # noqa: E402
from ember import config as cfg  # noqa: E402
from ember import regime as rg  # noqa: E402
from ember import universe as eu  # noqa: E402

NEW = ("naturgas_eu", "aluminium", "zink_nickel", "stal", "jarnmalm", "litium", "majs", "soja", "socker",
       "godsel", "lax", "tank", "torrbulk")


def test_ember_has_every_berserk_theme_with_etf_label_complex_and_board_entry():
    board = {s.key: s for s in tb._THEMES}
    for theme in bth.THEMES:
        assert theme in cfg.THEME_TO_COMPLEX, theme
        assert theme in cfg.EMBER_SECTOR_ETF and theme in cfg._THEME_LABEL and theme in board, theme
        assert cfg.THEME_TO_COMPLEX[theme] in cfg.COMPLEX_LABEL
        assert board[theme].primary_tickers and 0 < board[theme].necessity <= 100
    for theme in NEW:
        assert cfg._THEME_LABEL[theme] == bth.label(theme), theme
        assert board[theme].label == bth.label(theme), theme
    # samma komplex som BERSERK — utom skog (EMBER:s eget) och EMBER:s samlingstema agri (jordbrukskorgen)
    for theme, (_lbl, cx, _drv) in bth.THEMES.items():
        if theme != "skog":
            assert cfg.THEME_TO_COMPLEX[theme] == cx, theme
    assert cfg.THEME_TO_COMPLEX["skog"] == "skog" and cfg.THEME_TO_COMPLEX["agri"] == "agri"
    assert cfg.THEME_TO_COMPLEX["tank"] == cfg.THEME_TO_COMPLEX["torrbulk"] == "frakt"
    assert cfg.COMPLEX_LABEL["frakt"] == "FRAKT" and cfg.COMPLEX_DEFAULT_THEME["frakt"] == "torrbulk"


def test_old_labels_and_etfs_are_unchanged():
    assert cfg._THEME_LABEL["sallsynta"] == "Sällsynta Jordartsmetaller" and cfg._THEME_LABEL["skog"] == "Skog"
    assert cfg.EMBER_SECTOR_ETF["koppar"] == "COPX" and cfg.EMBER_SECTOR_ETF["uran"] == "URA"
    assert cfg.EMBER_SECTOR_ETF["stal"] == "SLX" and cfg.EMBER_SECTOR_ETF["torrbulk"] == "BDRY"


def test_sheet_commodities_speak_berserk():
    assert cfg.COMMODITY_TO_THEME["nickel"] == cfg.COMMODITY_TO_THEME["zinc"] == "zink_nickel"
    assert cfg.COMMODITY_TO_THEME["steel"] == "stal" and cfg.COMMODITY_TO_THEME["aluminum"] == "aluminium"
    assert cfg.COMMODITY_TO_THEME["lithium"] == cfg.COMMODITY_TO_THEME["cobalt"] == "litium"
    assert cfg.COMMODITY_TO_THEME["potash"] == "godsel" and cfg.COMMODITY_TO_THEME["tin"] == "koppar"
    for key, theme in cfg.COMMODITY_TO_THEME.items():
        assert cfg.COMMODITY_TO_COMPLEX[key] == cfg.THEME_TO_COMPLEX[theme], key     # härledd, inte handskriven
    assert cfg.COMMODITY_TO_COMPLEX["nickel"] == "basmetaller" and cfg.COMMODITY_TO_COMPLEX["potash"] == "agri"
    for key, c in com.REGISTRY.items():
        if c.theme_key:
            assert c.theme_key in cfg.THEME_TO_COMPLEX, key
            assert cfg.THEME_TO_COMPLEX[c.theme_key] == c.ember_complex, key
    assert com.REGISTRY["steel"].theme_key == "stal" and com.REGISTRY["zinc"].theme_key == "zink_nickel"


def test_sector_and_industry_keywords_reach_the_new_themes():
    for text, theme in (("Stål", "stal"), ("Steel", "stal"), ("Aluminium", "aluminium"), ("Järnmalm", "jarnmalm"),
                        ("Nickel", "zink_nickel"), ("Zink", "zink_nickel"), ("Litium", "litium"),
                        ("Sällsynta jordartsmetaller", "sallsynta"), ("Lax", "lax"), ("Sjömat", "lax"),
                        ("Gödsel", "godsel"), ("Majs", "majs"), ("Socker", "socker"), ("Tankrederi", "tank"),
                        ("Torrbulk", "torrbulk"), ("Olja & offshore", "olja"), ("Guldgruva", "guld")):
        assert rg.theme_from_sector_text(text) == theme, text
    yahoo = {"MOWI": {"industry": "Farm Products", "longName": "Mowi ASA — salmon farming"},
             "SSAB-A.ST": {"industry": "Steel"}, "GOGL": {"industry": "Marine Shipping"},
             "FRO": {"industry": "Oil & Gas Midstream", "longName": "Frontline plc tanker"},
             "BOL.ST": {"industry": "Other Industrial Metals & Mining"},
             "ENV": {"industry": "Environmental Services"}}

    def info(t):
        return yahoo.get(t, {})
    rg._THEME_CACHE.clear()
    assert rg.yahoo_theme("mowi", info) == "lax" and rg.yahoo_theme("SSAB-A.ST", info) == "stal"
    assert rg.yahoo_theme("GOGL", info) == "torrbulk" and rg.yahoo_theme("FRO", info) == "tank"
    assert rg.yahoo_theme("BOL.ST", info) == "koppar" and rg.yahoo_theme("ENV", info) is None
    rg._THEME_CACHE.clear()


def test_golden_in_a_company_name_is_not_gold():
    yahoo = {"GEOS.OL": {"sector": "Energy", "industry": "Oil & Gas Equipment & Services",
                         "longName": "Golden Energy Offshore Services AS"},
             "GOGL": {"industry": "Marine Shipping", "longName": "Golden Ocean Group Limited"},
             "NAME.OL": {"longName": "Golden Energy Offshore Services AS"},       # utan bransch: namnets ord
             "AEM.TO": {"industry": "Gold"}, "YAR.OL": {"industry": "Agricultural Inputs — Fertilizers"}}
    rg._THEME_CACHE.clear()
    got = {t: rg.yahoo_theme(t, lambda s: yahoo.get(s, {})) for t in yahoo}
    assert got == {"GEOS.OL": "olja", "GOGL": "torrbulk", "NAME.OL": "olja", "AEM.TO": "guld", "YAR.OL": "godsel"}
    rg._THEME_CACHE.clear()


def test_registry_members_need_no_alias_and_every_berserk_theme_registers():
    assert eu.REGISTRY_THEME_ALIAS == {}
    rg._THEME_OVERLAY.clear()
    assert rg.register_themes({"SSAB-A.ST": "stal", "MOWI.OL": "lax", "GOGL": "torrbulk", "X": "finns-inte"}) == 3
    assert rg.detect_theme("ssab-a.st") == "stal" and rg.detect_complex("GOGL") == "frakt"
    rg._THEME_OVERLAY.clear()


def _series(up=True, n=1500):
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.Series(100 + np.arange(n) * (0.05 if up else -0.05), index=idx)


def test_frakt_regime_has_five_pillars_on_bdry_bwet_and_china(monkeypatch):
    seen = []

    def close(ticker, period):
        seen.append(ticker)
        return _series()

    monkeypatch.setattr(rg, "_close", close)
    china = rg.PillarResult(name="MCHI/FXI 3M", status="AMBER", value="x", detail="y")
    monkeypatch.setattr(rg, "_bm_china_demand", lambda: china)
    monkeypatch.setattr(rg, "_theme_cycle", lambda key, label: rg.PillarResult(
        name=f"{label.upper()}-TEMA CYKEL", status="GREEN", value="Cykel: TIDIG", detail=key))
    getattr(rg.compute_frakt_regime, "clear", lambda: None)()
    res = rg.compute_frakt_regime()
    assert res.key == "frakt" and res.label == "FRAKT" and len(res.pillars) == 5
    assert [p.name for p in res.pillars] == ["BDRY vs 50V EMA + LUTNING", "BDRY/SPY RELATIV STYRKA",
                                             "TANK (BWET) vs 50V EMA", "MCHI/FXI 3M", "TORRBULK-TEMA CYKEL"]
    assert {"BDRY", "BWET"} <= set(seen) and "DBA" not in seen
    assert res.verdict in (rg.VERDICT_PA, rg.VERDICT_SELEKTIV, rg.VERDICT_AV)
    getattr(rg.compute_frakt_regime, "clear", lambda: None)()
