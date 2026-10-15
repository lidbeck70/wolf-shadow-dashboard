"""
Skannrarna mot registret — arc_scan: Wolf får registrets nordiska huvudlistor som extra marknad (Large Cap först,
tak, bara bolag utöver de fyra listorna) och Viking får registrets Norden som tickerlista; utan register precis
som förut. Screenrarna själva är fejkade — inget nätverk.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arc_scan as ac  # noqa: E402
import markets as mk  # noqa: E402
import screener_ovtlyr as so  # noqa: E402
import wolf_shadow_screener as ws  # noqa: E402
from universe import registry as ur  # noqa: E402

NORDIC = [{"insId": 1, "ticker": "ERIC B", "name": "Ericsson B", "marketId": 1, "stockPriceCurrency": "SEK"},
          {"insId": 2, "ticker": "GULD", "name": "Guldgruvan", "marketId": 3, "stockPriceCurrency": "SEK"},
          {"insId": 3, "ticker": "MIDCO", "name": "Mid Co", "marketId": 2, "stockPriceCurrency": "SEK"},
          {"insId": 4, "ticker": "FN", "name": "First North AB", "marketId": 4, "stockPriceCurrency": "SEK"},
          {"insId": 5, "ticker": "EQNR", "name": "Equinor", "marketId": 9, "stockPriceCurrency": "NOK"},
          {"insId": 6, "ticker": "OMXS30", "name": "Index", "marketId": 7}]


def _loader():
    u = ur.build(table=mk.FALLBACK, nordic=NORDIC, global_=[], include_lists=False)
    saved = {}
    ur.save(u, lambda name, data: saved.setdefault(name, data) is not None)
    return lambda name, fb: saved.get(name, fb)


def test_registry_lists_for_wolf_and_viking():
    wolf = ac.registry_wolf(_loader())
    assert list(wolf) == ["EQNR.OL", "ERIC-B.ST", "MIDCO.ST", "GULD.ST"]          # huvudbörserna, Mid, Small
    assert wolf["GULD.ST"] == "Guldgruvan" and "FN.ST" not in wolf                 # First North inte för Wolf
    assert ac.registry_viking(_loader()) == ("EQNR.OL", "ERIC-B.ST", "FN.ST", "GULD.ST", "MIDCO.ST")   # alla listor
    assert ac.registry_wolf(lambda name, fb: fb) is None and ac.registry_viking(lambda name, fb: fb) is None


def test_run_wolf_adds_the_register_as_an_extra_market(monkeypatch):
    seen = {}

    def fake_screener(markets=None, min_score=0, pre_fetched=None):
        seen["markets"] = list(markets)
        return pd.DataFrame([{"Ticker": "GULD.ST", "Name": "Guldgruvan", "Total Score": 77.5}])
    monkeypatch.setattr(ws, "run_screener", fake_screener)
    import utils.bd_api as bd
    monkeypatch.setattr(bd, "load_api_key", lambda: None)
    monkeypatch.setattr(ac, "WOLF_EXTRA_MAX", 2)
    out = ac.run_wolf(loader=_loader())
    assert out["error"] is None and out["universe_source"] == "universum-registret"
    assert seen["markets"] == ac.WOLF_MARKETS + [ac.WOLF_REGISTRY_KEY]
    extra = ws.MARKETS[ac.WOLF_REGISTRY_KEY]
    assert "ERIC-B.ST" not in extra and "EQNR.OL" not in extra                   # finns redan i listorna
    assert list(extra) == ["MIDCO.ST", "GULD.ST"]                                # taket 2, Mid före Small
    assert out["rows"] == [{"ticker": "GULD.ST", "name": "Guldgruvan", "score": 77.5}]
    plain = ac.run_wolf(loader=lambda name, fb: fb)
    assert plain["universe_source"] == "listorna" and seen["markets"] == ac.WOLF_MARKETS
    assert ac.WOLF_REGISTRY_KEY not in ws.MARKETS
    broken = ac.run_wolf(loader=lambda name, fb: (_ for _ in ()).throw(RuntimeError("nere")))
    assert broken["universe_source"] == "listorna" and broken["error"] is None


def test_run_viking_passes_the_register_as_ticker_list(monkeypatch):
    seen = {}

    def fake(universe="Nordic", min_volume=100_000, period="1y", ticker_list=None):
        seen["ticker_list"] = ticker_list
        return pd.DataFrame([{"Ticker": "GULD.ST", "Name": "Guldgruvan", "V9": "7/9", "_eligible": False,
                              "Composite": 1.2, "Signal": "WATCH"}])
    monkeypatch.setattr(so, "run_ovtlyr_screener", fake)
    out = ac.run_viking(loader=_loader())
    assert out["universe_source"] == "universum-registret" and seen["ticker_list"] == ac.registry_viking(_loader())
    assert out["rows"] and out["rows"][0]["ticker"] == "GULD.ST" and out["rows"][0]["v9"] == 7
    plain = ac.run_viking(loader=lambda name, fb: fb)
    assert plain["universe_source"] == "Börsdata" and seen["ticker_list"] is None
