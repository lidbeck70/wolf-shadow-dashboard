"""
Valhall fas 2b — kvalitets- och hävstångsköp per ruta: Quality Contrarian per segment, Rick Rule-
producenter per råvara (undantag 1: guld/koppar/uran globalt), Durrett med 15-BAGGER-regeln för
guld/silver och Tiggre för övriga (undantag 2: PGM och uran). Syntetiska ark — inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valhall import picks  # noqa: E402

CA_Q = {"mode": "quality",
        "results": [{"ticker": "VOLV-B.ST", "name": "Volvo", "sector": "Industrials", "branch": "Industrimaskiner",
                     "composite_score": 70, "rank": 2},
                    {"ticker": "SAND.ST", "name": "Sandvik", "sector": "Industrials", "branch": "Industrimaskiner",
                     "composite_score": 75, "rank": 1},
                    {"ticker": "ATCO-A.ST", "name": "Atlas", "sector": "Industrials", "composite_score": 60},
                    {"ticker": "CAST.ST", "name": "Castellum", "sector": "Financials", "branch": "Fastighetsbolag",
                     "composite_score": 55},
                    {"ticker": "BOL.ST", "name": "Boliden", "sector": "Materials", "composite_score": 65}],
        "quality_signals": {"VOLV-B.ST": {"verdict": "BUY"}, "SAND.ST": {"verdict": "WAIT"},
                            "ATCO-A.ST": {"verdict": "WATCH"}}}
CA_D = {"mode": "deep_contrarian",
        "results": [{"ticker": "SAND.ST", "name": "Sandvik", "sector": "Industrials", "composite_score": 50}]}

SHEETS = {
    "producers": {"producers": [
        {"ticker": "NEM", "name": "Newmont", "commodity": "Guld", "price": 2400, "unit_cost": 1300,
         "jurisdiktion": True, "kapitaldisciplin": True, "insyn": True},                         # 2 + 3 = 5
        {"ticker": "B2G.TO", "name": "B2Gold", "commodity": "Guld", "price": 2400, "unit_cost": 1900,
         "jurisdiktion": True},                                                                   # 0 + 1
        {"ticker": "IMP.JO", "name": "Impala", "commodity": "Platina", "price": 1000, "unit_cost": 500,
         "jurisdiktion": True, "kapitaldisciplin": True, "insyn": True}]},
    "scoring": {"durrett": [
        {"ticker": "ABC.TO", "factors": {"balans": 2, "vardering": 2, "tillvaxt": 2, "agare": 1, "havstang": 2}},
        {"ticker": "DEF.V", "factors": {"balans": 1, "vardering": 1, "tillvaxt": 1, "agare": 1, "havstang": 1}}],
        "sprott": [{"ticker": "PLAT.V", "commodity": "Platina"}]},
    "confidence": {"companies": {
        "ABC.TO": {"ticker": "ABC.TO", "name": "Abc Gold", "commodity": "gold", "stage": "developer", "fields": {}},
        "DEF.V": {"ticker": "DEF.V", "name": "Def Gold", "commodity": "gold", "stage": "explorer", "fields": {}},
        "SLV1.TO": {"ticker": "SLV1.TO", "name": "Silv", "commodity": "silver", "stage": "producer", "fields": {}},
        "URX.TO": {"ticker": "URX.TO", "name": "Urex", "commodity": "uranium", "stage": "developer", "fields": {}},
        "CU.TO": {"ticker": "CU.TO", "name": "Copperco", "commodity": "copper", "stage": "developer", "fields": {}}}},
    "tiggre": {"candidates": [
        {"ticker": "URX.TO", "name": "Urex", "screen": {"fs": True, "permits": True, "funded": False}, "mcap": 50,
         "nav": 200, "downside": 30, "factors": {"stadium": 2, "finansiering": 2, "manniskor": 2, "jurisdiktion": 2},
         "catalysts": [{"name": "a", "date": "2026-12-01", "status": "Väntar"},
                       {"name": "b", "date": "2027-03-01", "status": "Väntar"}]},
        {"ticker": "PLAT.V", "name": "Plat", "screen": {"fs": True}, "mcap": 80, "nav": 100, "downside": 50,
         "factors": {"stadium": 1}},
        {"ticker": "MYST.V", "name": "Mystery", "mcap": 10, "nav": 20, "factors": {}}]},
}
SCREENS = {"screens": {
    "rule": {"rows": [{"ticker": "CCO.TO", "name": "Cameco", "branch_id": 7, "m": {"ev_ebitda": 9.0}},
                      {"ticker": "EQNR.OL", "name": "Equinor", "branch_id": 4, "m": {"ev_ebitda": 3.0}},
                      {"ticker": "XOM", "name": "Exxon", "branch_id": 4, "m": {"ev_ebitda": 6.0}}]},
    "durrett": {"rows": [{"ticker": "GHI.V", "name": "Ghi", "mcap_musd": 80}, {"ticker": "ABC.TO", "name": "Abc"}]},
    "tiggre": {"rows": [{"ticker": "UUU.V", "name": "Uuu", "branch_id": 7}, {"ticker": "PPP.V", "branch_id": 16}]}}}


def fake_analyze(company):
    bull = {"ABC.TO": 18.0, "DEF.V": 22.0, "SLV1.TO": 4.0}.get(company.ticker, 1.0)
    return SimpleNamespace(bull_case=SimpleNamespace(upside_multiple=bull), quality_score=SimpleNamespace(value=70))


def test_commodity_and_sector_mapping():
    assert picks.theme_of_commodity("Guld") == "guld" and picks.theme_of_commodity("uranium") == "uran"
    assert picks.theme_of_commodity("Platina") == "platina" and picks.theme_of_commodity("Zink") == "zink_nickel"
    assert picks.theme_of_commodity(None) is None and picks.theme_of_commodity("Royalty") is None
    assert picks.segment_of_ca({"sector": "Financials", "branch": "Fastighetsbolag"}) == "real_estate"
    assert picks.segment_of_ca({"sector": "Financials", "branch": "Banker"}) == "financials"
    assert picks.segment_of_ca({"sector": "Informationsteknik"}) == "technology"


def test_segment_quality_ranks_buy_signals_first():
    q = picks.segment_quality([CA_Q, CA_D])
    ind = q["industrials"]["picks"]
    assert [p["ticker"] for p in ind] == ["VOLV-B.ST", "ATCO-A.ST"]                 # BUY, WATCH före WAIT
    assert q["real_estate"]["picks"][0]["ticker"] == "CAST.ST"
    assert q["industrials"]["etf"] == {"signal": "XLI", "ucits": ["IUIS.L"]} and q["energy"]["picks"] == []


def test_theme_quality_rick_rule_and_global_exception():
    tq = picks.theme_quality(SHEETS, SCREENS, [CA_Q])
    assert [p["ticker"] for p in tq["guld"]["picks"]] == ["NEM", "B2G.TO"]            # arket, Köpkandidat först
    assert tq["guld"]["picks"][0]["verdict"] == "Köpkandidat" and tq["guld"]["global_allowed"]
    assert [p["ticker"] for p in tq["uran"]["picks"]] == ["CCO.TO"]                  # uran: globalt tillåtet
    assert [p["ticker"] for p in tq["olja"]["picks"]] == ["EQNR.OL"]                 # olja: bara Norden (inte XOM)
    assert tq["platina"]["picks"] == [] and "PPLT" in tq["platina"]["etf"]           # platina: inget globalt
    assert tq["koppar"]["picks"][0]["ticker"] == "BOL.ST"                            # nordisk Quality Contrarian


def test_durrett_15_bagger_rule():
    rows = {r["ticker"]: r for r in picks.durrett_rows(SHEETS, analyze=fake_analyze)}
    assert set(rows) == {"ABC.TO", "DEF.V", "SLV1.TO"}                               # bara guld och silver
    assert rows["ABC.TO"]["bagger"] and rows["ABC.TO"]["poang"] == 9                 # 18× och 9 poäng
    assert not rows["DEF.V"]["bagger"] and rows["DEF.V"]["bull_multiple"] == 22.0     # 22× men bara 5 poäng
    assert not rows["SLV1.TO"]["bagger"] and rows["SLV1.TO"]["quality"] == 70


def test_theme_leverage_durrett_and_tiggre():
    lv = picks.theme_leverage(SHEETS, SCREENS, analyze=fake_analyze)
    assert lv["guld"]["strategy"] == "Durrett" and [p["ticker"] for p in lv["guld"]["picks"]] == ["ABC.TO", "DEF.V"]
    assert [r["ticker"] for r in lv["guld"]["to_analyse"]] == ["GHI.V"]              # håven, ej analyserad
    assert lv["uran"]["strategy"] == "Tiggre" and lv["uran"]["exception"]
    assert lv["uran"]["picks"][0]["ticker"] == "URX.TO" and lv["uran"]["picks"][0]["buy"]
    assert [r["ticker"] for r in lv["uran"]["to_analyse"]] == ["UUU.V"]              # urangren i Tiggre-håven
    assert lv["platina"]["picks"][0]["ticker"] == "PLAT.V" and not lv["platina"]["picks"][0]["buy"]
    assert [p["ticker"] for p in lv["_okand"]["picks"]] == ["MYST.V"]                 # tiggararket saknar råvara
    assert lv["koppar"]["strategy"] == "Tiggre" and "COPX" in lv["koppar"]["etf"]


def test_build_and_empty_sources():
    b = picks.build([CA_Q], SCREENS, SHEETS, analyze=fake_analyze)
    assert set(b["segments"]) == {s.key for s in picks.sgm.SEGMENTS}
    assert b["themes"]["guld"]["leverage"]["picks"][0]["bagger"]
    assert [p["ticker"] for p in b["unknown_commodity"]] == ["MYST.V"]
    empty = picks.build([], None, {}, analyze=fake_analyze)
    assert all(not v["picks"] for v in empty["segments"].values())
    assert all(not v["quality"]["picks"] and not v["leverage"]["picks"] for v in empty["themes"].values())
