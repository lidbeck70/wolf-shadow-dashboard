"""
Thor fas 5 — thor/cykel: Cykel-fliken under AppTest med Market Cycle Engine fejkad: grinden, fasremsan,
chipsen (hemmamarknaden, Valhall, ⚡ cykelstart), mätaren, fasens beskrivning, regelutfallet, indikatorerna
och tidslinjen — och felvägarna (motorn fallerar, fasen saknas). Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_candidates as fx  # noqa: E402
from thor import cykel  # noqa: E402

BOARD = {**fx.BOARD, "cycle_starts": [{"segment": "materials", "where": "Norden"}]}


def test_cycle_start_and_phase_config():
    assert cykel.cycle_start(BOARD, "materials") == "Norden" and cykel.cycle_start(BOARD, "technology") is None
    assert cykel.cycle_start(fx.BOARD, "materials") is None and cykel.cycle_start(None, "materials") is None
    assert cykel.cycle_start(BOARD, None) is None
    assert cykel.phase_config("HOPE")["emoji"] and cykel.phase_config("HOPE")["color"].startswith("#")
    assert cykel.phase_config(None) == {} and cykel.phase_config("NOPE") == {}


def _script():
    import streamlit as st
    from thor import cykel as c
    from thor import dossier as d
    mode = st.session_state["_mode"]
    ident = d.identify("BOL.ST")

    def phase(t):
        return (None, "motorn tyst") if mode == "nophase" else ({"phase": "HOPE", "confidence": 69.0}, None)

    def analysis(t, p):
        if mode == "fail":
            raise RuntimeError("nätet nere")
        return {"indicators": {"rsi": 55.0, "price_vs_ma50": 2.0, "price_vs_ma200": 5.0, "momentum_30": 3.0, "momentum_60": 6.0,
                               "momentum_90": 9.0, "drawdown_90": -4.0, "volume_vs_avg20": 1.1},
                "result": {"phase": "HOPE", "confidence": 69.0, "phase_scores": {},
                           "matched_rules": {"matched": [{"field": "rsi", "op": "between", "value": [45, 60]}],
                                             "unmatched": [{"field": "momentum_30", "op": "gt", "value": 5, "actual": 3.0}]}}}

    def history(t, p):
        return [{"date": "2026-01-05", "phase": "DISBELIEF", "confidence": 50.0, "phase_index": 0},
                {"date": "2026-06-05", "phase": "HOPE", "confidence": 69.0, "phase_index": 1}]
    c.render(st, {"board": st.session_state["_board"]}, ident, "quality", phase, analysis, history)


def _run(mode):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_script, default_timeout=120)
    at.session_state["_mode"], at.session_state["_board"] = mode, BOARD
    at.run()
    assert not at.exception, at.exception
    return at, " ".join(m.value for m in at.markdown)


def test_tab_shows_gate_strip_chips_rules_indicators_and_timeline():
    at, text = _run("ok")
    for part in ("Cykelgrinden (quality): PASS", "köpfas enligt playbooken", "Hope", "^OMX: HOPE 69 %",
                 "Valhall Norden · RÖD · SVAG · TOPP", "⚡ Cykelstart (Norden)", "Early signs of recovery",
                 "Fasens villkor: 1 av 2 uppfyllda", "Matched (1)", "Unmatched (1)", "momentum_30", "Key Indicators",
                 "55.0", "Market Cycle Engine, 14 faser"):
        assert part in text, part
    assert len(at.get("plotly_chart")) == 2                                                 # mätaren och tidslinjen


def test_engine_failure_is_a_note_but_the_gauge_stays():
    at, text = _run("fail")
    assert "Regelutfallet kunde inte läsas: nätet nere" in text and "Cykelgrinden (quality): PASS" in text
    assert len(at.get("plotly_chart")) == 1 and "Key Indicators" not in text


def test_missing_phase_is_a_note_without_charts():
    at, text = _run("nophase")
    assert "Cykelgrinden (quality): MANUAL" in text and "Aktiens fas kunde inte läsas: motorn tyst" in text
    assert len(at.get("plotly_chart")) == 0 and "Fasens villkor" not in text
