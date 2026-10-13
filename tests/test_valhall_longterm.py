"""
Valhall fas 4b — långsiktigt backtest: framåtavkastning per cykelläge, sälj på RÖD, ROIC-grinden punkt
för punkt (rapporten känd fyra månader efter årets slut) och kvartalsportföljerna. Syntetiska data.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valhall import cycle  # noqa: E402
from valhall import longterm as lt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = pd.bdate_range("2015-01-01", "2020-12-31")


def _series(rate_per_year=0.10, idx=IDX):
    t = np.arange(len(idx)) / 252
    return pd.Series(100 * (1 + rate_per_year) ** t, index=idx)


def _frame(status_by_year: dict, close: pd.Series):
    st = [status_by_year.get(d.year, cycle.GRON) for d in close.index]
    return pd.DataFrame({"close": close, "status": st, "phase": cycle.STARK, "cykel": cycle.MITTEN,
                         "cycle_start": False}, index=close.index)


def test_month_ends_and_fwd():
    me = lt.month_ends(IDX, "2016-01-01")
    assert me[0] == pd.Timestamp("2016-01-29") and len(me) == 60
    assert lt.month_ends(IDX, "2016-01-01", step=3)[1] == pd.Timestamp("2016-04-29")
    s = _series(0.10)
    assert lt.fwd(s, "2016-01-04", 12) == pytest.approx(10.0, abs=0.6)
    assert lt.fwd(s, "2020-06-30", 12) is None                                       # slutet saknas
    assert lt.fwd(None, "2016-01-04", 12) is None


def test_red_exit_sells_at_first_red_month_end():
    s = _series(0.10)
    f = _frame({2017: cycle.ROD}, s)
    full = lt.fwd(s, "2016-01-04", 24)
    red = lt.red_exit(f, s, "2016-01-04", 24)
    assert red == pytest.approx(lt.fwd(s, "2016-01-04", 0) + (s.loc["2017-01-31"] / s.loc["2016-01-04"] - 1) * 100)
    assert red < full
    assert lt.red_exit(_frame({}, s), s, "2016-01-04", 24) == pytest.approx(full)     # aldrig röd = behåll


def test_segment_rows_relative_and_absolute():
    s, bench = _series(0.20), _series(0.10)
    f = _frame({2016: cycle.ROD}, s)
    rows = lt.segment_rows({"tech": f}, bench, "2016-01-01", "Norden")
    r = rows[0]
    assert r["status"] == cycle.ROD and r["where"] == "Norden" and r["period"] == "2008–2020"
    assert r["x12"] == pytest.approx(r["r12"] - lt.fwd(bench, r["date"], 12))
    assert r["x12"] > 0 and r["red12"] is not None
    absolute = lt.segment_rows({"guld": f}, None, "2016-01-01", "tema")[0]
    assert absolute["x12"] == pytest.approx(absolute["r12"])
    nodata = _frame({}, s).assign(status=cycle.NODATA)
    assert lt.segment_rows({"x": nodata}, bench, "2016-01-01", "Norden") == []


def test_known_roic_respects_publication_lag():
    vals = [{"y": 2015, "v": 20.0}, {"y": 2016, "v": 10.0}, {"y": 2014, "v": 5.0}, {"y": 2017, "v": None}]
    assert lt.known_roic(vals, "2016-04-29") == [5.0]                                 # 2015 känd först 1 maj
    assert lt.known_roic(vals, "2016-05-02") == [5.0, 20.0]
    assert lt.known_roic(vals, "2017-06-30") == [5.0, 20.0, 10.0]
    assert lt.gates([5.0, 20.0]) == {"q": True, "d": True}
    assert lt.gates([20.0, 9.0, 5.0]) == {"q": False, "d": True}                      # medianen 9 > 8
    assert lt.gates([4.0, 5.0, 6.0]) == {"q": False, "d": False} and lt.gates([])["q"] is False


def test_stock_rows_filters_liquidity_gate_and_tags_segment():
    s, bench = _series(0.15), _series(0.05)
    vol = pd.Series(1e5, index=IDX)                                                   # 100 kr × 1e5 = 10 milj/dag
    thin = pd.Series(1.0, index=IDX)
    closes = {"A.ST": s, "B.ST": s, "C.ST": s}
    turnover = {"A.ST": s * vol, "B.ST": s * thin, "C.ST": s * vol}
    uni = {"A.ST": {"segment": "technology", "ins_id": 1}, "B.ST": {"segment": "technology", "ins_id": 2},
           "C.ST": {"segment": "technology", "ins_id": 3}}
    roic = {1: [{"y": y, "v": 20.0} for y in range(2010, 2020)], 2: [{"y": 2014, "v": 30.0}],
            3: [{"y": y, "v": 2.0} for y in range(2010, 2020)]}
    frames = {"technology": _frame({2016: cycle.ROD}, bench)}
    rows = lt.stock_rows(closes, turnover, uni, roic, frames, bench, "2016-01-01")
    assert {r["ticker"] for r in rows} == {"A.ST"}                                    # B illikvid, C grinden
    r = rows[0]
    assert r["q"] and r["status"] == cycle.ROD and r["r3"] is not None and r["date"] == "2016-01-29"
    assert len({x["date"] for x in rows}) == len(lt.month_ends(bench.index, "2016-01-01", 3))
    assert lt.stock_rows(closes, turnover, uni, roic, frames, None, "2016-01-01") == []


def test_summarize_filters_and_portfolio():
    rows = [{"date": "2016-03-31", "period": "2008–2020", "q": True, "d": True, "status": cycle.ROD, "cykel": cycle.TIDIG,
             "cycle_start": False, "r3": 10.0, "x12": 5.0, "r12": 15.0, "xred12": 1.0},
            {"date": "2016-03-31", "period": "2008–2020", "q": False, "d": True, "status": cycle.GRON,
             "cykel": cycle.SEN, "cycle_start": False, "r3": -10.0, "x12": -5.0, "r12": 0.0},
            {"date": "2021-06-30", "period": "2021–", "q": True, "d": True, "status": cycle.GRON, "cykel": cycle.SEN,
             "cycle_start": True, "r3": 0.0, "x12": -1.0, "r12": 2.0}]
    sm = {(r["status"], r["period"]): r for r in lt.summarize(rows, ("status",), keys=(12,))}
    assert sm[(cycle.ROD, "alla")]["x12"] == 5.0 and sm[(cycle.ROD, "alla")]["hit12"] == 100
    assert sm[(cycle.GRON, "alla")]["n"] == 2 and sm[(cycle.GRON, "2021–")]["x12"] == -1.0
    assert sm[(cycle.ROD, "alla")]["xred12"] == 1.0
    ft = {(r["filter"], r["period"]): r for r in lt.filter_table(rows)}
    assert ft[("Kvalitet — alla", "alla")]["n"] == 2 and ft[("Kvalitet — segment RÖD (hatat)", "alla")]["n"] == 1
    assert ft[("Kvalitet — cykelstart (VÄNDER)", "alla")]["n"] == 1 and ft[("Djup — alla", "alla")]["n"] == 3
    p = lt.portfolio(rows, lambda r: r["q"])
    assert p["quarters"] == 2 and p["cagr"] == pytest.approx(((1.10 * 1.0) ** (1 / 0.5) - 1) * 100, abs=0.1)
    assert p["dd"] == 0.0 and p["avg_n"] == 1.0 and p["bench_cagr"] is None
    empty = lt.portfolio(rows, lambda r: r["q"] and r["status"] == cycle.GUL)
    assert empty["cagr"] == 0.0                                                       # kassa när inget klarar
    by = {(r["filter"], r["period"]): r for r in lt.portfolios(rows, None)}
    assert by[("Kvalitet — alla", "2021–")]["quarters"] == 1


def test_fas4b_runner(monkeypatch):
    import markets
    import valhall_fas3 as fas3
    import valhall_fas4b as vf
    bench = _series(0.05)
    s = _series(0.15)
    fr = {"nordic": {"technology": _frame({2016: cycle.ROD}, bench)}, "global": {"technology": _frame({}, bench)},
          "themes": {"guld": _frame({}, s)}, "bench": {"nordic": bench, "global": bench},
          "universe": {"A.ST": {"segment": "technology", "currency": "SEK"}}}

    def frames(api, getter, data_getter, nordic_market=None, keep=None):
        keep["A.ST"] = pd.DataFrame({"Close": s, "Volume": 1e5})
        return fr

    monkeypatch.setattr(fas3, "frames", frames)
    monkeypatch.setattr(markets, "load", lambda api=None, **k: {})
    monkeypatch.setattr(markets, "nordic_stock_ids", lambda table=None: {1})
    monkeypatch.setattr(markets, "to_yf", lambda t, m=None, table=None: t + ".ST")
    api = SimpleNamespace(get_instruments=lambda: [{"ticker": "A", "marketId": 1, "insId": 7}])
    out = vf.run(api=api, getter=lambda t, p: None, data_getter=lambda ts, p: {},
                 roic_getter=lambda ids: {7: [{"y": y, "v": 25.0} for y in range(2010, 2020)]}, no_save=True)
    assert out["counts"]["with_roic"] == 1 and out["counts"]["stock_rows"] > 0
    assert {r["where"] for r in out["segments"]["by_status"]} == {"Norden", "globalt", "tema"}
    assert any(r["filter"] == "Kvalitet — alla" and r.get("quarters") for r in out["stocks"]["portfolios"])
    import json
    json.dumps(out, default=str)


def test_longterm_workflow():
    src = open(os.path.join(ROOT, ".github", "workflows", "valhall-longterm.yml"), encoding="utf-8").read()
    assert "workflow_dispatch" in src and "python valhall_fas4b.py" in src and "BORSDATA_API_KEY" in src


def test_clean_removes_data_jumps():
    s = pd.Series([100.0, 101.0, 1010.0, 1020.0, 102.0, 103.0], index=IDX[:6])        # felaktigt ×10-hopp och tillbaka
    c = lt.clean(s)
    assert c.iloc[0] == 100.0 and c.iloc[1] == pytest.approx(101.0)
    assert c.iloc[-1] == pytest.approx(101.0 * 1020 / 1010 * 103 / 102)                # hoppen räknas som 0 %
    assert lt.clean(s.iloc[:1]).iloc[0] == 100.0


def test_kpi_history_batch_passes_max_count():
    from borsdata_api import BorsdataAPI
    calls = []
    api = BorsdataAPI.__new__(BorsdataAPI)
    api._get = lambda path, params=None: calls.append(params) or {"kpisList": [{"instrument": 7, "values": [{"y": 2006}]}]}
    assert api.get_kpi_history_batch([7], 37, max_count=20) == {7: [{"y": 2006}]}
    assert calls[-1] == {"instList": "7", "maxcount": 20}
    api.get_kpi_history_batch([7], 37)
    assert calls[-1] == {"instList": "7"}                                             # oförändrat utan max_count
