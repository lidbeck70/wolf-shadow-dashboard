"""
K5+K6 — händelsemotorn (valhall/turns.py) och ETF-benet (valhall/etf.py): köpfönstret efter bekräftad vändning,
rutans bolag, 2 Deep Contrarian + 2 lång, ETF-exponeringarna rankade på den långa signalserien, planen och dess
rader i kartan. Inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_valhall_cycle import CRASH_TURN, IDX, UP, _ohlcv, _ser  # noqa: E402
from valhall import complexes as vcx  # noqa: E402
from valhall import cycle  # noqa: E402
from valhall import etf as ve  # noqa: E402
from valhall import segments as sgm  # noqa: E402
from valhall import turns as vt  # noqa: E402

TODAY = "2026-10-20"
T_CX = {"level": "komplex", "key": "adelmetaller", "label": "Ädelmetaller", "where": "komplex", "kind": "VÄNDER",
        "since": "2026-10-01", "days": 13, "confirmed": True, "confirmed_since": "2026-10-16"}
T_SEG = {"level": "segment", "key": "technology", "label": "Teknik", "where": "Norden", "kind": "RÖD→GRÖN",
         "since": "2026-10-06", "days": 10, "confirmed": True, "confirmed_since": "2026-10-19"}
T_OLD = {**T_CX, "key": "energi", "label": "Energi", "confirmed_since": "2026-09-01"}
T_SOON = {**T_CX, "key": "skog", "confirmed": False, "confirmed_since": None}
T_GLOBAL = {**T_SEG, "where": "global"}
UNIVERSE = {"BOL.ST": {"segment": "materials", "name": "Boliden"}, "ERIC-B.ST": {"segment": "technology", "name": "Ericsson"},
            "HEXA-B.ST": {"segment": "technology", "name": "Hexagon"}, "GOLD.ST": {"segment": "materials", "name": "Guldbolaget"},
            "MOWI.OL": {"segment": "consumer_staples", "name": "Mowi"}}
CMAP = {"BOL.ST": "basmetaller", "GOLD.ST": "adelmetaller", "MOWI.OL": "agri"}


def test_window_is_confirmed_and_fresh():
    assert vt.in_window(T_CX, TODAY) and vt.in_window(T_SEG, TODAY)
    assert not vt.in_window(T_OLD, TODAY) and not vt.in_window(T_SOON, TODAY) and not vt.in_window({}, TODAY)
    assert vt.in_window({"confirmed": True}, TODAY)                                   # datum saknas: släpps in
    assert [t["key"] for t in vt.window([T_CX, T_SEG, T_OLD, T_SOON, T_GLOBAL, "x"], TODAY)] == ["adelmetaller", "technology"]


def test_members_per_level(monkeypatch):
    from berserk import universe as buv
    assert vt.members(T_CX, UNIVERSE, CMAP) == {"GOLD.ST"}
    assert vt.members(T_SEG, UNIVERSE, CMAP) == {"ERIC-B.ST", "HEXA-B.ST"}
    monkeypatch.setattr(buv, "by_theme", lambda tickers=None: {"guld": ["NEM", "GOLD.ST"], "lax": ["MOWI.OL", "SALM.OL"]})
    monkeypatch.setattr(buv, "region_of", lambda t: "Norden" if t.endswith((".ST", ".OL")) else "USA")
    assert vt.members({"level": "tema", "key": "guld"}, UNIVERSE, CMAP) == {"GOLD.ST"}
    assert vt.members({"level": "tema", "key": "lax"}, UNIVERSE, CMAP) == {"MOWI.OL", "SALM.OL"}


def test_deep_and_long_picks_follow_the_turns_room():
    deep = {"mode": "deep_contrarian", "results": [
        {"ticker": "GOLD.ST", "name": "Guldbolaget", "composite_score": 61.0, "rank": 3, "sector": "Material"},
        {"ticker": "BOL.ST", "name": "Boliden", "composite_score": 70.0, "rank": 1, "sector": "Material"},
        {"ticker": "ERIC-B.ST", "name": "Ericsson B", "composite_score": 55.0, "rank": 5, "sector": "Informationsteknik"},
        {"ticker": "ERIC-A.ST", "name": "Ericsson A", "composite_score": 54.0, "rank": 6, "sector": "Informationsteknik"},
        {"ticker": "HEXA-B.ST", "name": "Hexagon", "composite_score": 40.0, "rank": 9, "sector": "Informationsteknik"}]}
    assert [p["ticker"] for p in vt.deep_picks(deep, T_CX, {"GOLD.ST"})] == ["GOLD.ST"]
    seg = vt.deep_picks(deep, T_SEG, set())
    assert [p["ticker"] for p in seg] == ["ERIC-B.ST", "HEXA-B.ST"] and seg[0]["source"] == "Deep Contrarian"   # ett aktieslag
    assert vt.deep_picks(None, T_SEG, set()) == []
    long_scan = {"segments": {"technology": {"picks": [{"ticker": "HEXA-B.ST", "name": "Hexagon", "signal": "BUY", "score_pct": 70},
                                                       {"ticker": "ERIC-B.ST", "name": "Ericsson", "signal": "STRONG BUY", "score_pct": 60}]},
                              "materials": {"picks": [{"ticker": "GOLD.ST", "name": "Guldbolaget", "signal": "BUY", "score_pct": 66}]}}}
    buys = {"segments": {"technology": {"picks": [{"ticker": "ABB.ST", "name": "ABB", "verdict": "BUY", "score": 80},
                                                  {"ticker": "ATCO-A.ST", "name": "Atlas", "verdict": "WATCH", "score": 90}]},
                         "materials": {"picks": [{"ticker": "BOL.ST", "name": "Boliden", "verdict": "BUY", "score": 75}]}}}
    seg = vt.long_picks(long_scan, buys, T_SEG, set())
    assert [p["ticker"] for p in seg] == ["ERIC-B.ST", "HEXA-B.ST"]                                 # STRONG BUY först
    assert [p["ticker"] for p in vt.long_picks(long_scan, buys, T_SEG, set(), n=3)] == ["ERIC-B.ST", "HEXA-B.ST", "ABB.ST"]
    cx = vt.long_picks(long_scan, buys, T_CX, {"GOLD.ST", "BOL.ST"})
    assert [p["ticker"] for p in cx] == ["GOLD.ST", "BOL.ST"] and cx[1]["source"] == "Quality Contrarian"
    assert vt.long_picks(None, None, T_CX, set()) == []


def test_plan_and_lines():
    deep = {"mode": "deep_contrarian", "results": [{"ticker": "GOLD.ST", "name": "Guldbolaget", "composite_score": 61.0}]}
    etfs = {"adelmetaller": {"ticker": "IGLN.L", "exposure": "fysiskt guld", "verified": False}}
    p = vt.plan([T_CX, T_SEG, T_OLD], UNIVERSE, CMAP, deep_blob=deep, etf_pick=lambda k: etfs.get(k), today=TODAY)
    assert [w["key"] for w in p["window"]] == ["adelmetaller", "technology"] and p["window_days"] == vt.WINDOW_DAYS
    w = p["window"][0]
    assert w["members"] == 1 and [x["ticker"] for x in w["deep"]] == ["GOLD.ST"] and w["long"] == [] and w["etf"]["ticker"] == "IGLN.L"
    assert p["window"][1]["etf"] is None
    ls = vt.lines(p)
    assert ls[0] == "Ädelmetaller (komplex, VÄNDER dag 13 ✓): Deep GOLD.ST · ETF IGLN.L (fysiskt guld, att verifiera)"
    assert ls[1] == "Teknik (Norden, RÖD→GRÖN dag 10 ✓): inga kandidater i rutan i dag"
    assert vt.lines(None) == [] and vt.plan([], UNIVERSE, CMAP, today=TODAY)["window"] == []
    boom = vt.plan([T_CX], UNIVERSE, CMAP, etf_pick=lambda k: 1 / 0, today=TODAY)
    assert boom["window"][0]["etf"] is None                                                    # ETF-fel stoppar inte planen


def test_etf_candidates_cover_every_room_and_are_unverified():
    for k in list(vcx.KEYS) + [s.key for s in sgm.SEGMENTS]:
        cs = ve.candidates(k)
        assert cs and all(not e.verified for e in cs) and all(e.signal for e in cs), k
    assert {e.exposure for e in ve.candidates("adelmetaller")} >= {"fysiskt guld", "guldgruvor"}
    assert ve.candidates("frakt")[0].ticker is None and ve.candidates("technology")[0].ticker == "IUIT.L"
    assert ve.level_of("energi") == "komplex" and ve.level_of("technology") == "segment"
    assert ve.necessity_of("energi") == vcx.NECESSITY["energi"] and ve.necessity_of("technology") == 20 and ve.necessity_of("x") == 50


def test_exposures_are_ranked_on_the_long_signal_series():
    data = {"GLD": _ohlcv(CRASH_TURN), "GDX": _ohlcv(np.linspace(100, 60, len(IDX))), "GDXJ": _ohlcv(UP), "SLV": None}
    ranked = ve.rank_exposures("adelmetaller", getter=lambda t, p: data.get(t), spy=_ser(UP))
    by = {r["signal"]: r for r in ranked}
    assert by["GLD"]["score"] is not None and by["GLD"]["phase"] == cycle.VANDER and by["GLD"]["blocked"] is None
    assert by["GDXJ"]["blocked"] == "cykeltopp — köp ej" and by["GDXJ"]["score"] is None          # rak uppgång = topp
    assert by["SLV"]["blocked"].startswith("SLV: ingen serie") and by["SLV"]["score"] is None
    assert ranked[0]["signal"] == "GLD" and ranked[-1]["score"] is None                            # bäst först, spärrade sist
    assert ve.pick("adelmetaller", getter=lambda t, p: data.get(t), spy=_ser(UP))["ticker"] == "IGLN.L"
    assert ve.pick("frakt", getter=lambda t, p: None) is None
    st = {"pct10": 20.0, "cykel": cycle.TIDIG, "phase": cycle.VANDER, "blindspot": 30.0, "rs_spy": 6.0}
    assert ve.score_exposure(st, "komplex") == round(30 * 1.0 + 100 * 0.5 + 60 * 0.25, 1)
    assert ve.score_exposure(st, "segment") == round(30 * 0.5 + 100 * 1.0 + 60 * 0.25, 1)
    assert ve.score_exposure({**st, "cykel": cycle.TOPP}, "komplex") is None and ve.score_exposure({**st, "pct10": None}, "komplex") is None


def test_valhall_map_lists_the_plan(monkeypatch):
    import test_valhall_ui as tv
    from valhall import ui as vui
    plan = {"window": [{**T_CX, "members": 1, "deep": [{"ticker": "GOLD.ST"}], "long": [],
                        "etf": {"ticker": "IGLN.L", "exposure": "fysiskt guld", "verified": False}}], "window_days": 5}
    board = {**tv.BOARD, "turn_plan": plan, "turns": [T_CX]}
    from streamlit.testing.v1 import AppTest

    def script():
        import streamlit as st
        from valhall import ui as v
        v.render_map(st.session_state["_b"])
    at = AppTest.from_function(script)
    at.session_state["_b"] = board
    at.run(timeout=60)
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "VÄNDNING — KÖP I DAG" in text and "Deep GOLD.ST" in text and "IGLN.L" in text and "att verifiera" in text
