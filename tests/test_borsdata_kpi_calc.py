"""
Börsdata KPI-screenern: rätt calcGroup/calc för KPI:er som svarar 400 på last/latest (probe 2026-10-14) och
Long Screeners batch (cagr_loader) — tillväxt, F-score, Magic Formula (rangplats → percentil), RS-rank och
omsättningens CAGR 5 år. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import borsdata_api as bd  # noqa: E402
from borsdata_api import KPI  # noqa: E402


def test_screener_calc_maps_only_the_default():
    assert bd.screener_calc(94) == ("1year", "cagr") and bd.screener_calc(97) == ("1year", "cagr")
    assert bd.screener_calc(163) == ("last", "rank") and bd.screener_calc(167) == ("last", "point")
    assert bd.screener_calc(192) == ("last", "rank")
    assert bd.screener_calc(33) == ("last", "latest")                                       # ROE: oförändrad
    assert bd.screener_calc(94, "5year", "cagr") == ("5year", "cagr")                       # uttryckligt val står
    assert {174, 179} == set(bd.KPI_SCREENER_UNAVAILABLE)
    assert KPI["revenue_growth"] == 94 and KPI["earnings_growth"] == 97 and KPI["f_score"] == 167
    assert KPI["magic_formula"] == 163 and KPI["rs_rank"] == 192


def test_get_kpi_screener_calls_the_working_path():
    api = bd.BorsdataAPI.__new__(bd.BorsdataAPI)
    paths = []
    api._get = lambda path, params=None: paths.append(path) or {"values": [{"i": 1, "n": 5.0}]}
    assert api.get_kpi_screener(167) == [{"i": 1, "n": 5.0}]
    api.get_kpi_screener(97)
    api.get_kpi_screener(94, "5year", "cagr")
    api.get_kpi_screener(KPI["roe"])
    api.get_kpi_screener_global(192)
    assert paths == ["/instruments/kpis/167/last/point", "/instruments/kpis/97/1year/cagr",
                     "/instruments/kpis/94/5year/cagr", "/instruments/kpis/33/last/latest",
                     "/instruments/global/kpis/192/last/rank"]


def test_magic_percentile():
    from cagr.cagr_loader import _magic_percentile
    vals = [{"i": i, "n": float(i)} for i in range(1, 201)] + [{"i": 999, "n": None}]
    out = {v["i"]: v["n"] for v in _magic_percentile(vals)}
    assert out[1] == 1.0 and out[60] == 30.0 and out[200] == 100.0 and 999 not in out
    assert _magic_percentile([]) == []


def test_batch_fast_fills_the_long_screener_fields(monkeypatch):
    import cagr.cagr_loader as cl
    calls = []
    screener = {  # (kpi, grupp, calc) → {insId: värde}, som Börsdata svarar
        (94, "1year", "cagr"): {1: 20.0}, (97, "1year", "cagr"): {1: 25.0}, (94, "5year", "cagr"): {1: 12.0},
        (167, "last", "point"): {1: 8.0}, (192, "last", "rank"): {1: 85.0},
        (163, "last", "rank"): {i: float(i) for i in range(1, 101)}, (33, "last", "latest"): {1: 25.0},
    }

    class Api:
        def resolve_instrument_id(self, t):
            return {"VOLV-B.ST": 1}.get(t)

        def get_instrument_info(self, i):
            return {"name": "Volvo B"}

        def get_kpi_screener(self, kpi_id, calc_group="last", calc="latest"):
            g, c = bd.screener_calc(kpi_id, calc_group, calc)
            calls.append((kpi_id, g, c))
            return [{"i": i, "n": v} for i, v in screener.get((kpi_id, g, c), {}).items()]

    monkeypatch.setattr(cl, "_borsdata_available", lambda: True)
    monkeypatch.setattr(cl, "_get_borsdata_api", lambda: Api())
    fn = getattr(cl.fetch_fundamentals_batch_fast, "__wrapped__", cl.fetch_fundamentals_batch_fast)
    out = fn(("VOLV-B.ST",))["VOLV-B.ST"]
    assert out["revenue_growth"] == 0.20 and out["earnings_growth"] == 0.25 and out["revenue_cagr_5y"] == 0.12
    assert out["f_score"] == 8.0 and out["rs_rank"] == 85.0 and out["magic_formula"] == 1.0   # rang 1 av 100
    assert out["returnOnEquity"] == 0.25
    assert (94, "5year", "cagr") in calls and (163, "last", "rank") in calls
    from cagr.cagr_fundamentals import score_fundamentals
    det = score_fundamentals(out)["details"]
    for crit in ("Revenue Growth", "Earnings Growth", "Revenue CAGR 5y", "RS Rank", "Magic Formula"):
        assert det[crit]["value"] != "N/A", crit                                           # förut alltid N/A
