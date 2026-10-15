"""
Skannrarna mot registret — håvarna (screens_scan): instrumentlistorna ur universum-registret (Norden och globalt,
alla listor, bara Börsdata-bolag) med håvarnas geografi läst ur registrets marknadsnyckel; utan register precis
som förut. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import markets as mk  # noqa: E402
import screens_scan as sc  # noqa: E402
from test_screens_scan import _API  # noqa: E402
from universe import registry as ur  # noqa: E402

NORDIC = [{"insId": 1, "ticker": "BOL", "name": "Boliden", "marketId": 1, "branchId": 17, "sectorId": 7,
           "stockPriceCurrency": "SEK"},
          {"insId": 2, "ticker": "OMX", "name": "Index", "marketId": 7}]
GLOBAL = [{"insId": 101, "ticker": "JR", "name": "Junior Corp", "marketId": 36, "branchId": 16,
           "stockPriceCurrency": "CAD"},                                                 # TSX Venture — räknas
          {"insId": 102, "ticker": "AU", "name": "Gold Prod", "marketId": 35, "branchId": 18,
           "stockPriceCurrency": "CAD"},
          {"insId": 103, "ticker": "OTCG", "name": "OTC Gold", "marketId": 34, "branchId": 18,
           "stockPriceCurrency": "USD"}]                                                 # OTC — inte i registret


def _loader():
    u = ur.build(table=mk.FALLBACK, nordic=NORDIC, global_=GLOBAL, include_lists=True)
    saved = {}
    ur.save(u, lambda name, data: saved.setdefault(name, data) is not None)
    return lambda name, fb: saved.get(name, fb)


def test_registry_instruments_are_borsdata_members_with_market_keys_as_geography():
    nordic, glob, ctry = sc.registry_instruments(_loader())
    assert [i["ticker"] for i in nordic] == ["BOL"] and nordic[0]["countryId"] == "norden"
    assert {i["ticker"] for i in glob} == {"JR", "AU"} and all(i["countryId"] == "kanada" for i in glob)
    assert nordic[0]["insId"] == 1 and nordic[0]["marketId"] == 1 and nordic[0]["stockPriceCurrency"] == "SEK"
    assert ctry == {sc.CA: {"kanada"}, sc.AU: {"australien"}, sc.US: {"usa"}, sc.NORDIC: {"norden"}}
    assert sc.registry_instruments(lambda name, fb: fb) is None
    assert all(i.get("insId") for lst in (nordic, glob) for i in lst)            # listornas bolag saknar insId


def test_scan_reads_the_register_and_screens_the_same_way():
    out = sc.scan(_API(), loader=_loader())
    s = out["screens"]
    assert out["universe_source"] == "universum-registret" and out["global_available"] is True
    assert [r["ticker"] for r in s["rule"]["rows"]] == ["BOL.ST"]
    assert [r["ticker"] for r in s["sprott"]["rows"]] == ["JR.V"]                  # suffix ur registrets marknad
    assert [r["ticker"] for r in s["tiggre"]["rows"]] == ["JR.V"]
    assert [r["ticker"] for r in s["durrett"]["rows"]] == ["AU.TO"]
    assert s["royalty"]["rows"] == [] and all(v["error"] is None for v in s.values())
    assert "markets" in out


def test_scan_without_register_is_borsdata_as_before():
    out = sc.scan(_API(), loader=lambda name, fb: fb)
    assert out["universe_source"] == "Börsdata" and [r["ticker"] for r in out["screens"]["rule"]["rows"]] == ["BOL.ST"]
    assert [r["ticker"] for r in out["screens"]["sprott"]["rows"]] == ["JR"]
    broken = sc.scan(_API(), loader=lambda name, fb: (_ for _ in ()).throw(RuntimeError("nere")))
    assert broken["universe_source"] == "Börsdata" and broken["screens"]["rule"]["rows"]
    none = sc.scan(_API(global_ok=False), loader=_loader_without_global())
    assert none["global_available"] is False and "Pro+ global" in none["screens"]["sprott"]["error"]


def _loader_without_global():
    u = ur.build(table=mk.FALLBACK, nordic=NORDIC, global_=[], include_lists=False)
    saved = {}
    ur.save(u, lambda name, data: saved.setdefault(name, data) is not None)
    return lambda name, fb: saved.get(name, fb)
