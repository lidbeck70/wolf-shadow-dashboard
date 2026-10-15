"""
Thor fas 2 — dossierdatan: kandidatraderna för en aktie, vald strategi, ögonblicksbilden, cykeln med
playbookens grind (injicerad fasfunktion) och arken (bara läsning ur stores). Inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thor import candidates as tc  # noqa: E402
from thor import dossier as td  # noqa: E402
from thor import ticker as tt  # noqa: E402

ROWS = [tc.Candidate("ERIC-B.ST", "Ericsson", "momentum", "rad 1 · setup A", tc.KOP, score=100.0),
        tc.Candidate("ERIC-B.ST", "Ericsson", "viking_nine", "READY", tc.BEVAKA, score=71.0),
        tc.Candidate("BOL.ST", "Boliden", "berserk", "KÖP · S2", tc.KOP, score=100.0, theme="koppar", segment="materials")]
BOARD = {"generated": "2026-10-14", "board": {"segments": [
    {"key": "materials", "nordic": {"status": "RÖD", "phase": "SVAG", "cykel": "TOPP", "close": 1.0},
     "global": {"status": "GUL", "phase": "STARK", "cykel": "SEN", "close": 5.0},
     "themes": [{"theme": "koppar", "state": {"status": "GRÖN", "phase": "STARK", "cykel": "TIDIG", "close": 2.0}}]}]}}


def _df(n=80):
    idx = pd.bdate_range("2026-01-01", periods=n)
    c = 100 + np.arange(n) * 0.5
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1e6}, index=idx)


def test_candidates_for_matches_every_form():
    assert [c.strategy for c in td.candidates_for(ROWS, "eric b")] == ["momentum", "viking_nine"]
    assert [c.ticker for c in td.candidates_for(ROWS, "BOL")] == ["BOL.ST"]           # bar form hittar .ST
    assert td.candidates_for(ROWS, "ZZZ") == [] and td.candidates_for(ROWS, "") == [] and td.candidates_for(None, "BOL") == []


def test_default_strategy():
    nordic = tt.Ticker(raw="x", yf="X.ST", nordic=True)
    assert td.default_strategy(nordic, td.candidates_for(ROWS, "ERIC-B.ST")) == "momentum"   # bästa raden
    assert td.default_strategy(nordic, []) == "viking_nine"
    assert td.default_strategy(tt.Ticker(raw="x", yf="X", theme="guld"), []) == "berserk"
    assert td.default_strategy(tt.Ticker(raw="x", yf="NEM"), []) == "wolf"


def test_snapshot_is_pure():
    snap, err = td.snapshot(_df(), "X.ST")
    assert err == "" and snap["price"] == pytest.approx(100 + 79 * 0.5) and snap["bars"] == 80 and snap["ema50"] is not None
    none, err2 = td.snapshot(None, "X.ST")
    assert none is None and "X.ST" in err2


def test_cycle_with_injected_phase_and_board():
    calls = []

    def phase(t):
        calls.append(t)
        return ({"phase": "HOPE" if t == "BOL.ST" else "OPTIMISM", "confidence": 69.0}, None)

    ident = tt.identify("BOL.ST")                                   # utan Börsdata: tema ur BERSERK-listan
    out = td.cycle(ident, "quality", BOARD, phase)
    assert calls == ["BOL.ST", "^OMX"] and out["phase"]["phase"] == "HOPE" and out["benchmark"]["phase"]["phase"] == "OPTIMISM"
    assert out["benchmark"]["index"] == "^OMX" and out["error"] == ""
    assert out["segment"]["view"] == "temat" and out["segment"]["status"] == "GRÖN" and out["segment"]["cykel"] == "TIDIG"
    assert out["segment"]["all"]["n_status"] == "RÖD"                            # segmentets läge följer med
    assert out["gate"]["status"] == "PASS" and "HOPE" in out["phases"]["buy"] and out["playbook"] == "quality"
    gate = td.cycle(ident, "berserk", BOARD, phase)["gate"]            # BERSERK:s playbook nämner inga cykelfaser
    assert gate["status"] == "MANUAL" and "nämns inte i playbookens regler" in gate["note"]
    assert td.cycle(ident, "hav", BOARD, phase)["gate"] == {"status": "MANUAL", "note": td.NO_PLAYBOOK}

    def boom(t):
        raise RuntimeError("nät")

    bad = td.cycle(ident, "quality", None, boom)
    assert bad["phase"] is None and "nät" in bad["error"] and bad["segment"] == {} and bad["gate"]["status"] == "MANUAL"
    plain = td.cycle(tt.identify("VOLV-B.ST"), "quality", BOARD, phase)
    assert plain["segment"] == {}                                                # inget segment utan Börsdata


def test_sheets_for_reads_every_store_read_only():
    stores = {
        "producers": {"producers": [{"ticker": "EQX", "name": "Equinox", "commodity": "Guld", "price": 2400.0,
                                     "unit_cost": 1600.0, "ev_ebitda": 5.0, "nd_ebitda": 0.5, "mine_life": 12.0,
                                     "rp_ratio": 15.0, "jurisdiktion": True, "kapitaldisciplin": True, "insyn": True}],
                      "royalty": []},
        "confidence": {"companies": {"EQX": {"ticker": "EQX", "name": "Equinox", "commodity": "gold"}}},
        "scorecard": {"cards": {"equinox|EQX": {"strategy": "rule"}, "annat|XX": {}}},
        "holdings": {"long": [{"ticker": "EQX.TO", "name": "Equinox", "strategy": "Quality", "entry_price": 10.0,
                               "shares": 100}],
                     "closed": [{"ticker": "EQX", "exit_price": 12.0}]},
        "journal": [{"ticker": "EQX.TO", "pnl_pct": 5.0}, {"ticker": "BOL.ST"}],
        "copilot_journal": {"entries": [{"ticker": "EQX", "note": "x"}]},
    }
    before = {k: repr(v) for k, v in stores.items()}
    s = td.sheets_for("EQX.TO", stores)
    assert set(s["reviews"]) == {"rule"} and s["reviews"]["rule"]["found"]
    assert s["reviews"]["rule"]["status"] in ("PASS", "MANUAL", "FAIL") and s["reviews"]["rule"]["sheet"] == "Rick Rule"
    assert s["confidence"]["name"] == "Equinox" and list(s["scorecard"]) == ["equinox|EQX"]
    assert [p["_bucket"] for p in s["positions"]] == ["long"] and s["positions"][0]["ticker"] == "EQX.TO"
    assert len(s["closed"]) == 1 and len(s["journal"]) == 1 and len(s["copilot"]) == 1 and s["found"]
    assert {k: repr(v) for k, v in stores.items()} == before                      # bara läsning
    empty = td.sheets_for("ZZZ.ST", {})
    assert not empty["found"] and empty["reviews"] == {} and empty["confidence"] is None and empty["positions"] == []


def test_assemble_everything_optional():
    d = td.assemble("eric b", rows=ROWS)
    assert d["ticker"]["yf"] == "ERIC-B.ST" and d["strategy"] == "momentum" and d["strategy_label"] == "Momentum Swing"
    assert [c["strategy"] for c in d["candidates"]] == ["momentum", "viking_nine"]
    assert d["snapshot"] is None and d["cycle"] is None and d["sheets"]["found"] is False
    d2 = td.assemble("BOL.ST", rows=ROWS, board=BOARD, df=_df(), strategy="quality",
                     phase_fn=lambda t: ({"phase": "HOPE", "confidence": 50.0}, None))
    assert d2["strategy"] == "quality" and d2["snapshot"]["bars"] == 80 and d2["cycle"]["gate"]["status"] == "PASS"
    assert d2["ticker"]["theme"] == "koppar" and d2["candidates"][0]["strategy"] == "berserk"
