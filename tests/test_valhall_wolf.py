"""
Valhall: Wolf på Kartan och i papperet — Arc-Wolfs rader (färskhet, universum), strategins köpregel per rad,
Kartans Wolf- och Momentum-rader per segment och papperets Wolf-signaler. Syntetiska data — inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valhall import wolf as vw  # noqa: E402

UNI = {"VOLV-B.ST": {"segment": "industrials", "name": "Volvo"}, "ATCO-A.ST": {"segment": "industrials", "name": "Atlas"},
       "BOL.ST": {"segment": "materials", "name": "Boliden"}}
ARC = {"wolf": {"generated": "2026-10-14T06:11:00+00:00", "error": None,
                "rows": [{"ticker": "VOLV-B.ST", "name": "Volvo", "score": 91.6},
                         {"ticker": "ATCO-A.ST", "name": "Atlas", "score": 97.0},
                         {"ticker": "BOL.ST", "name": "Boliden", "score": 66.0},
                         {"ticker": "NEM", "name": "Newmont", "score": 99.0}]}}
ENTRIES = {"VOLV-B.ST": {"signal": "BUY", "entry": 310.0, "stop": 300.0, "tp1": 336.0, "tp2": 362.0, "gates": 4, "gates_n": 4},
           "ATCO-A.ST": {"signal": "HOLD", "entry": None, "stop": None, "tp1": None, "tp2": None, "gates": 3, "gates_n": 4}}


def _df(n=260, step=1.0):
    idx = pd.bdate_range("2025-01-01", periods=n)
    c = 100 + np.arange(n) * step
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1e6}, index=idx)


def test_arc_live_freshness_universe_and_errors():
    live = vw.arc_live(ARC, UNI, "2026-10-15")
    assert set(live) == {"VOLV-B.ST", "ATCO-A.ST", "BOL.ST"}                       # NEM är inte i universumet
    assert vw.arc_live(ARC, UNI, "2026-10-20") is None                               # äldre än fyra dagar
    assert vw.arc_live({"wolf": {**ARC["wolf"], "error": "boom"}}, UNI, "2026-10-15") is None
    assert vw.arc_live(None, UNI, "2026-10-15") is None and vw.arc_live({"viking": {}}, UNI, "2026-10-15") is None


def test_arc_entries_uses_strategy_rule_and_needs_history():
    def fake_entry(df):
        buy = df["Close"].iloc[-1] > 300
        return {"signal": "BUY" if buy else "HOLD", "entry_price": 310.0 if buy else None,
                "stop_loss": 300.0 if buy else None, "tp1_price": 336.0 if buy else None,
                "tp2_price": 362.0 if buy else None,
                "gates": [{"passed": True}] * (4 if buy else 2) + [{"passed": False}] * (0 if buy else 2)}

    data = {"VOLV-B.ST": _df(260, 1.0), "ATCO-A.ST": _df(260, 0.1), "BOL.ST": _df(100, 1.0)}   # BOL: för kort historik
    arc = vw.arc_live(ARC, UNI, "2026-10-15")
    e = vw.arc_entries(arc, lambda t: data[t], entry_fn=fake_entry)
    assert e["VOLV-B.ST"]["signal"] == "BUY" and e["VOLV-B.ST"]["stop"] == 300.0 and e["VOLV-B.ST"]["gates"] == 4
    assert e["ATCO-A.ST"]["signal"] == "HOLD" and e["ATCO-A.ST"]["gates"] == 2 and "BOL.ST" not in e
    assert vw.arc_entries(None, lambda t: None) == {}


def test_real_entry_rule_runs_on_price_data():
    """strategies/wolf.entry_fn på dagsdata: regeln är strategins egen — svaret har signal och fyra grindar."""
    e = vw.arc_entries({"VOLV-B.ST": {"score": 90}}, lambda t: _df(260, 1.0))
    assert e["VOLV-B.ST"]["signal"] in ("BUY", "HOLD") and e["VOLV-B.ST"]["gates_n"] == 4


def test_map_rows_per_segment():
    arc = vw.arc_live(ARC, UNI, "2026-10-15")
    picks = vw.arc_picks(arc, ENTRIES, UNI, "industrials")
    assert [p["ticker"] for p in picks] == ["VOLV-B.ST", "ATCO-A.ST"]                 # ENTRY före högre poäng
    assert picks[0]["tag"] == "ENTRY · 92" and picks[0]["entry"] and picks[0]["stop"] == 300.0
    assert picks[0]["gates"] == "4/4" and picks[0]["strategy"] == vw.WOLF and picks[0]["name"] == "Volvo"
    assert picks[1]["tag"] == "97" and not picks[1]["entry"] and picks[1]["stop"] is None
    assert vw.arc_picks(arc, ENTRIES, UNI, "materials")[0]["ticker"] == "BOL.ST"
    assert vw.arc_picks(None, {}, UNI, "energy") == []
    wolf = {"VOLV-B.ST": {"rank": 3, "setupA": True, "name": "Volvo", "price": 280.0},
            "ATCO-A.ST": {"rank": 1, "setupA": False}, "BOL.ST": {"rank": 25, "setupA": True}}
    mom = vw.momentum_picks(wolf, UNI, "industrials")
    assert [p["ticker"] for p in mom] == ["VOLV-B.ST"] and mom[0]["tag"] == "#3" and mom[0]["strategy"] == vw.MOMENTUM
    assert vw.momentum_picks(wolf, UNI, "materials") == []                                 # rad 25 är inget köp
    rows = vw.by_segment(arc, ENTRIES, wolf, UNI)
    assert rows["arc_live"] and rows["momentum_live"] and set(rows["segments"]) == {s.key for s in vw.sgm.SEGMENTS}
    assert rows["segments"]["industrials"]["arc"][0]["ticker"] == "VOLV-B.ST"
    assert rows["segments"]["energy"] == {"arc": [], "momentum": []}
    off = vw.by_segment(None, {}, None, UNI)
    assert off["arc_live"] is False and off["momentum_live"] is False


def test_paper_signals_only_when_entry_rule_holds():
    arc = vw.arc_live(ARC, UNI, "2026-10-15")
    sigs = vw.arc_signals(arc, ENTRIES, UNI)
    assert [(s["strategy"], s["ticker"], s["segment"], s["nordic"]) for s in sigs] == \
        [(vw.ARC_ACCOUNT, "VOLV-B.ST", "industrials", True)]
    assert "poäng 92" in sigs[0]["why"] and "4/4" in sigs[0]["why"] and sigs[0]["stop"] is None
    assert vw.arc_signals(None, {}, UNI) == []
