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
        seen["extra"] = dict(ws.MARKETS.get(ac.WOLF_REGISTRY_KEY) or {})         # finns bara under körningen
        return pd.DataFrame([{"Ticker": "GULD.ST", "Name": "Guldgruvan", "Total Score": 77.5}])
    monkeypatch.setattr(ws, "run_screener", fake_screener)
    import utils.bd_api as bd
    monkeypatch.setattr(bd, "load_api_key", lambda: None)
    monkeypatch.setattr(ac, "WOLF_EXTRA_MAX", 2)
    out = ac.run_wolf(loader=_loader())
    assert out["error"] is None and out["universe_source"] == "universum-registret"
    assert seen["markets"] == ac.WOLF_MARKETS + [ac.WOLF_REGISTRY_KEY]
    extra = seen["extra"]
    assert "ERIC-B.ST" not in extra and "EQNR.OL" not in extra                   # finns redan i listorna
    assert list(extra) == ["MIDCO.ST", "GULD.ST"]                                # taket 2, Mid före Small
    assert ac.WOLF_REGISTRY_KEY not in ws.MARKETS                                # listorna lämnas som de var
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


# ── USA: registrets NYSE/Nasdaq utan small cap ───────────────────────────────
GLOBAL = [{"insId": 501, "ticker": "LLY", "name": "Eli Lilly", "marketId": 32, "sectorId": 4, "stockPriceCurrency": "USD"},
          {"insId": 502, "ticker": "UNH", "name": "UnitedHealth", "marketId": 32, "sectorId": 4, "stockPriceCurrency": "USD"},
          {"insId": 503, "ticker": "TINY", "name": "Tiny Biotech", "marketId": 33, "sectorId": 4, "stockPriceCurrency": "USD"},
          {"insId": 504, "ticker": "NOCAP", "name": "No Cap Inc", "marketId": 33, "sectorId": 4, "stockPriceCurrency": "USD"},
          {"insId": 505, "ticker": "TECK B", "name": "Teck", "marketId": 35, "sectorId": 7, "stockPriceCurrency": "CAD"}]
CAPS = {501: 800000.0, 502: 450000.0, 503: 900.0, 505: 30000.0}              # MUSD; 504 saknar värde


class _CapAPI:
    def __init__(self, caps=None):
        self.caps, self.calls = (CAPS if caps is None else caps), []

    def get_kpi_screener_global(self, kpi_id, calc_group="last", calc="latest"):
        self.calls.append(kpi_id)
        return [{"i": i, "n": n} for i, n in self.caps.items()]


def _usa_loader():
    u = ur.build(table=mk.FALLBACK, nordic=NORDIC, global_=GLOBAL, include_lists=False)
    saved = {}
    ur.save(u, lambda name, data: saved.setdefault(name, data) is not None)
    return lambda name, fb: saved.get(name, fb)


def test_registry_usa_is_mid_and_large_cap_only():
    api = _CapAPI()
    reg = ac.registry_usa(_usa_loader(), api)
    assert list(reg) == ["LLY", "UNH"] and reg["LLY"] == "Eli Lilly" and api.calls == [50]      # störst först
    assert "TINY" not in reg and "NOCAP" not in reg and "TECK-B.TO" not in reg                   # small cap, utan värde, Kanada
    assert ac.registry_usa(_usa_loader(), None) is None                                           # utan Börsdata: inget USA
    assert ac.registry_usa(lambda name, fb: fb, api) is None                                      # utan register
    assert ac.registry_usa(_usa_loader(), _CapAPI({})) is None                                    # inga börsvärden (licens)
    assert ac.USA_MIN_MCAP_MUSD == 2000.0


def test_usa_runs_use_the_register_and_skip_without_it(monkeypatch):
    seen = {}

    def fake_wolf(markets=None, min_score=0, pre_fetched=None):
        seen["markets"], seen["pre"] = list(markets), pre_fetched
        seen["usa_list"] = dict(ws.MARKETS.get(ac.USA_WOLF_KEY) or {})          # finns bara under körningen
        return pd.DataFrame([{"Ticker": "LLY", "Name": "Eli Lilly", "Total Score": 91.0},
                             {"Ticker": "UNH", "Name": "UnitedHealth", "Total Score": 55.5}])

    def fake_viking(universe="Nordic", min_volume=100_000, period="1y", ticker_list=None):
        seen["universe"], seen["ticker_list"] = universe, ticker_list
        return pd.DataFrame([{"Ticker": "UNH", "Name": "UnitedHealth", "V9": "8/9", "_eligible": True,
                              "Composite": 2.0, "Signal": "BUY"}])
    monkeypatch.setattr(ws, "run_screener", fake_wolf)
    monkeypatch.setattr(so, "run_ovtlyr_screener", fake_viking)
    reg = ac.registry_usa(_usa_loader(), _CapAPI())
    w = ac.run_wolf_usa(reg=reg)
    assert seen["markets"] == [ac.USA_WOLF_KEY] and seen["pre"] is None and seen["usa_list"] == reg
    assert ac.USA_WOLF_KEY not in ws.MARKETS                                     # listorna lämnas som de var
    assert w["rows"] == [{"ticker": "LLY", "name": "Eli Lilly", "score": 91.0}, {"ticker": "UNH", "name": "UnitedHealth", "score": 55.5}]
    assert w["universe_source"] == "universum-registret (USA)" and w["universe"] == 2 and w["min_mcap_musd"] == 2000.0
    v = ac.run_viking_usa(reg=reg)
    assert seen["universe"] == "US" and seen["ticker_list"] == ("LLY", "UNH") and v["rows"][0]["v9"] == 8
    skipped = ac.run_wolf_usa(loader=_usa_loader(), api=None)                                     # utan Börsdata
    assert skipped["rows"] == [] and skipped["universe_source"] == "hoppades över" and skipped["error"] is None
    assert ac.run_viking_usa(loader=lambda name, fb: fb, api=_CapAPI())["universe_source"] == "hoppades över"


def test_scan_all_has_norden_and_usa_parts(monkeypatch):
    monkeypatch.setattr(ws, "run_screener", lambda markets=None, min_score=0, pre_fetched=None: pd.DataFrame())
    monkeypatch.setattr(so, "run_ovtlyr_screener", lambda universe="Nordic", min_volume=100_000, period="1y",
                        ticker_list=None: pd.DataFrame())
    import utils.bd_api as bd
    monkeypatch.setattr(bd, "load_api_key", lambda: None)
    out = ac.scan_all(loader=_usa_loader(), usa_api=_CapAPI())
    assert set(out) == {"wolf", "viking", "wolf_usa", "viking_usa"} and all(o["error"] is None for o in out.values())
    assert out["wolf_usa"]["universe"] == 2 and out["viking_usa"]["universe_source"] == "universum-registret (USA)"
    assert out["wolf"]["universe_source"] == "universum-registret" and out["viking"]["universe_source"] == "universum-registret"
