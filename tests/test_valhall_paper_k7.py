"""
K7 — nya papperskonton (komplex_viking, komplex_deep, komplex_long, komplex_etf, ember, deep, royalty), märkning
med komplex och vändning på signaldagen, EMBER:s stopp/tidsstopp, drivarexit för komplexvarianten och de nya
grupperingarna. Inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_valhall_paper import D, _board, _df  # noqa: E402
from valhall import paper as vpa  # noqa: E402

TURN = {"level": "komplex", "key": "adelmetaller", "label": "Ädelmetaller", "where": "komplex", "kind": "VÄNDER",
        "days": 12, "confirmed": True, "confirmed_since": "2026-01-10", "since": "2026-01-02"}


def test_new_accounts_exist_with_the_right_kind():
    for k in ("komplex_viking", "ember"):
        assert vpa.STRATEGIES[k][1] == vpa.SWING
    for k in ("komplex_deep", "komplex_long", "komplex_etf", "deep", "royalty"):
        assert vpa.STRATEGIES[k][1] == vpa.LONG


def test_complex_signals_cover_every_new_source():
    cs = {"complexes": {"basmetaller": {"strategy": "Viking Nine · Basmetaller", "picks": [
        {"ticker": "BOL.ST", "name": "Boliden", "segment": "materials", "category": "GOLDEN TICKET", "ovtlyr_nine": "9/9"},
        {"ticker": "LUMI.ST", "segment": "materials", "category": "READY"}]}}}
    plan = {"window": [{**TURN, "deep": [{"ticker": "GOLD.ST", "name": "Guld", "segment": "materials", "score": 61}],
                        "long": [{"ticker": "NEM.ST", "segment": "materials", "signal": "BUY", "source": "Long Screener"}],
                        "etf": {"ticker": "IGLN.L", "name": "Gold ETC", "exposure": "fysiskt guld", "score": 80, "verified": False}}]}
    ember = {"eligible": [{"ticker": "SCA-B.ST", "theme_key": "skog", "stop": 90.0, "setup_score": 75, "typ": "A"}]}
    deep = {"mode": "deep_contrarian", "results": [
        {"ticker": "ERIC-B.ST", "name": "Ericsson B", "composite_score": 60, "sector": "Informationsteknik"},
        {"ticker": "HEXA-B.ST", "name": "Hexagon", "composite_score": 50, "sector": "Informationsteknik"},
        {"ticker": "SINCH.ST", "name": "Sinch", "composite_score": 40, "sector": "Informationsteknik"}]}
    roy = {"rows": [{"ticker": "WPM", "verdict": "BUY CANDIDATE"}, {"ticker": "FNV", "verdict": "WATCH"}]}
    uni = {"SCA-B.ST": {"segment": "materials", "name": "SCA"}}
    sigs = vpa.complex_signals(cs, plan, ember, deep, roy, uni)
    by = {(s["strategy"], s["ticker"]): s for s in sigs}
    assert ("komplex_viking", "BOL.ST") in by and ("komplex_viking", "LUMI.ST") not in by
    assert by[("komplex_viking", "BOL.ST")]["complex"] == "basmetaller" and by[("komplex_viking", "BOL.ST")]["profile"] == "basmetaller"
    assert by[("komplex_deep", "GOLD.ST")]["complex"] == "adelmetaller" and "vändning Ädelmetaller dag 12" in by[("komplex_deep", "GOLD.ST")]["why"]
    assert ("komplex_long", "NEM.ST") in by and "inte verifierad" in by[("komplex_etf", "IGLN.L")]["why"]
    assert by[("ember", "SCA-B.ST")]["stop"] == 90.0 and by[("ember", "SCA-B.ST")]["complex"] == "skog"
    assert {t for (k, t) in by if k == "deep"} == {"ERIC-B.ST", "HEXA-B.ST"}                       # 2 per segment
    assert ("royalty", "WPM") in by and ("royalty", "FNV") not in by
    assert vpa.complex_signals(None, None, None, None, None, {}) == []


def test_orders_are_tagged_with_the_turn_and_rows_group_by_it():
    data = {"GOLD.ST": _df(o=np.arange(100, 140, 1.0))}
    sig = vpa._sig("komplex_deep", "GOLD.ST", "Guld", "materials", "guld", True, "test", complex="adelmetaller")
    turns = [{**TURN, "level": "tema", "key": "guld"}]
    st, _ = vpa.step(None, [sig], _board(), lambda t: data[t].loc[:D[2]], today=D[2], turns=turns)
    o = st["orders"][0]
    assert o["turn"]["kind"] == "VÄNDER" and o["turn"]["level"] == "tema" and o["turn"]["days"] == 12
    st, _ = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[3]], today=D[3])
    rs = vpa.rows(st)
    assert rs[0]["complex"] == "adelmetaller" and rs[0]["turn_in"] == "VÄNDER" and rs[0]["confirmed_in"] and rs[0]["days_since_turn"] == 12
    res = vpa.results(st)
    assert res["by_complex"][0]["complex"] == "adelmetaller" and res["by_turn"][0]["turn_in"] == "VÄNDER"
    assert "by_phase" in res
    assert vpa.turn_tag(turns, "materials", "silver", None) == {} and vpa.turn_tag(None, "x") == {}


def test_ember_position_has_stop_and_time_stop():
    data = {"SCA-B.ST": _df(o=np.full(40, 100.0), lo=np.r_[np.full(10, 99.0), np.full(30, 80.0)])}
    sig = vpa._sig("ember", "SCA-B.ST", "SCA", "materials", "skog", True, "KÖPLÄGE", stop=90.0)
    st, _ = vpa.step(None, [sig], _board(), lambda t: data[t].loc[:D[2]], today=D[2])
    st, _ = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[5]], today=D[5])
    assert st["positions"] and st["positions"][0]["init_stop"] == 90.0
    st, _ = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[12]], today=D[12])
    assert not st["positions"] and st["closed"][0]["reason"].startswith("stopp (EMBER")


def test_complex_viking_uses_the_driver_exit(monkeypatch):
    import viking_exit as vex
    monkeypatch.setattr(vex, "evaluate_exit", lambda *a, **k: SimpleNamespace(status=vex.HOLD, reasons=[]))
    data = {"BOL.ST": _df(o=np.arange(100, 140, 1.0))}
    sig = vpa._sig("komplex_viking", "BOL.ST", "Boliden", "materials", None, True, "9/9", complex="basmetaller",
                   profile="basmetaller")
    st, _ = vpa.step(None, [sig], _board(), lambda t: data[t].loc[:D[2]], today=D[2])
    st, _ = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[3]], today=D[3])
    assert st["positions"][0]["init_stop"] is not None                                      # Viking-stoppen vid fyllning
    ctx = {"driver": lambda prof: {"triggered": prof == "basmetaller", "detail": "HG=F: under EMA50"}}
    st, ev = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[4]], today=D[4], exit_ctx=ctx)
    assert st["positions"][0]["pending"]["reason"].startswith("drivarexit") and ev[0]["kind"] == vpa.SALJ_OPEN


def test_closed_stats_unit_curve_and_history():
    def r(ret, ent, ex, open_=False, excess=None):
        return {"open": open_, "ret_pct": ret, "excess_pct": excess, "entry_date": ent, "exit_date": None if open_ else ex,
                "held_days": 30}
    g = [r(10, "2025-01-01", "2025-02-01", excess=3), r(-5, "2025-03-01", "2025-04-01", excess=-1),
         r(20, "2025-05-01", "2026-01-01"), r(50, "2025-06-01", None, open_=True)]
    st = vpa.closed_stats(g)
    assert st["closed_win"] == 66.7 and st["payoff"] == round(15 / 5, 2) and st["profit_factor"] == 6.0
    assert st["avg_hold"] == 30.0 and st["excess_30d"] == 1.0 and st["closed_n"] == "3/20" and not st["enough"]
    assert st["unit_dd"] == round((105 / 110 - 1) * 100, 2) and st["unit_cagr"] is not None and st["unit_cagr"] > 0
    assert vpa.unit_curve([]) == {"unit_cagr": None, "unit_dd": None}
    res = {"by_strategy": [{"strategy": "viking", "n": 3, "closed": 3, "payoff": 3.0}]}
    snap = vpa.snapshot(res, "2026-10-24")
    assert snap["date"] == "2026-10-24" and snap["by_strategy"]["viking"]["payoff"] == 3.0
    h = vpa.append_history([{"date": "2026-10-23"}, {"date": "2026-10-24", "old": 1}], snap)
    assert [x["date"] for x in h] == ["2026-10-23", "2026-10-24"] and "old" not in h[-1]
    assert len(vpa.append_history([{"date": str(i)} for i in range(900)], snap, max_rows=800)) == 800
