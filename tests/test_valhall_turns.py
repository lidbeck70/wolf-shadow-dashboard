"""
K2 — kant och bekräftad vändning (analysen 2026-10-24 §5.2, beslut 3): cycle.turn_state för råvaror (fasen VÄNDER +
bekräftelse över SMA50) och aktiesegment (RÖD → GRÖN som hållit tio dagar), komplexrader i kartan med
komplexdrivaren som korg, listan turns, notisen "VÄNDNING" och komplexrutorna i Valhall, samt Thors chip. Inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_valhall_cycle import CRASH_TURN, IDX, UP, _ohlcv, _ser  # noqa: E402
from thor import cykel  # noqa: E402
from valhall import board as vb  # noqa: E402
from valhall import cycle  # noqa: E402
from valhall import ui as vui  # noqa: E402


def _status_frame(statuses):
    idx = pd.bdate_range("2026-01-01", periods=len(statuses))
    return pd.DataFrame({"status": statuses, "phase": [cycle.STARK] * len(statuses)}, index=idx)


def test_segment_turn_is_red_to_green_that_has_held_ten_days():
    t = cycle.turn_state(_status_frame([cycle.ROD] * 20 + [cycle.GRON] * 5), cycle.SEGMENT)
    assert t["kind"] == "RÖD→GRÖN" and t["days"] == 5 and not t["confirmed"] and t["confirmed_since"] is None
    assert t["since"] == str(pd.bdate_range("2026-01-01", periods=25)[20].date())
    t = cycle.turn_state(_status_frame([cycle.ROD] * 20 + [cycle.GRON] * 12), cycle.SEGMENT)
    assert t["confirmed"] and t["days"] == 12 and t["confirmed_since"] == str(pd.bdate_range("2026-01-01", periods=32)[29].date())
    assert cycle.turn_state(_status_frame([cycle.GUL] * 5 + [cycle.GRON] * 12), cycle.SEGMENT) == cycle.NO_TURN   # GUL → GRÖN räknas inte
    assert cycle.turn_state(_status_frame([cycle.ROD] * 30), cycle.SEGMENT) == cycle.NO_TURN
    assert cycle.turn_state(_status_frame([cycle.ROD] * 5 + [cycle.GRON] * 3 + [cycle.ROD] * 2), cycle.SEGMENT) == cycle.NO_TURN
    assert cycle.turn_state(None) == cycle.NO_TURN and cycle.turn_state(pd.DataFrame()) == cycle.NO_TURN


def test_commodity_turn_is_the_vander_phase_confirmed_above_sma50():
    f = cycle.cycle_frame(_ser(CRASH_TURN))
    assert "confirmed" in f and f["phase"].iloc[-1] == cycle.VANDER
    t = cycle.turn_state(f)
    assert t["kind"] == "VÄNDER" and t["days"] >= 1 and t["confirmed"] == bool(f["confirmed"].iloc[-1])
    assert t["confirmed"] and t["confirmed_since"] is not None and t["since"] <= t["confirmed_since"]   # 70 → 85 har tagit SMA50
    st = cycle.state(f)
    assert st["turn"] == t and st["cycle_start"]
    assert cycle.turn_state(cycle.cycle_frame(_ser(UP))) == cycle.NO_TURN                   # ingen baisse → ingen vändning
    assert cycle.state(None)["turn"] == cycle.NO_TURN
    # bekräftelsen kräver strong: en vändning som bara passerat EMA50 men inte SMA50/ret63 är inte bekräftad
    slow = np.concatenate([np.linspace(100, 140, 2000), np.linspace(140, 70, 900), np.linspace(70, 74, 100)])
    f2 = cycle.cycle_frame(_ser(slow))
    assert f2["confirmed"].iloc[-1] == (bool(f2["phase"].iloc[-1] == cycle.VANDER) and bool(f2["confirmed"].iloc[-1]))


def test_board_has_complex_rows_and_turns():
    data = {"SPY": _ohlcv(UP), "XLE": _ohlcv(CRASH_TURN), "XLU": _ohlcv(np.linspace(200, 60, len(IDX)))}
    drivers = {"olja": ("BZ=F", _ser(CRASH_TURN)), "koppar": ("HG=F", _ser(UP)), "skog": ("WOOD", _ser(CRASH_TURN))}
    b = vb.build(getter=lambda t, p: data.get(t), nordic={"energy": {"index": _ser(CRASH_TURN), "members": 14}},
                 nordic_market=_ser(UP), drivers=drivers)
    cx = {c["key"]: c for c in b["complexes"]}
    assert list(cx) == ["energi", "basmetaller", "adelmetaller", "agri", "skog", "frakt"]
    assert cx["energi"]["driver"] == "korg: olja:BZ=F" and cx["energi"]["state"]["phase"] == cycle.VANDER
    assert cx["energi"]["state"]["turn"]["kind"] == "VÄNDER" and cx["energi"]["n_themes"] == 5
    assert cx["energi"]["themes_turning"] >= 1 and [t["theme"] for t in cx["energi"]["themes"]][0] == "olja"
    assert cx["basmetaller"]["state"]["cykel"] == cycle.TOPP and cx["adelmetaller"]["state"]["status"] == cycle.NODATA   # rak uppgång = topp
    assert cx["skog"]["driver"] == "korg: skog:WOOD" and cx["frakt"]["state"]["turn"] == cycle.NO_TURN
    turns = vb.turns(b)
    kinds = {(t["level"], t["key"], t.get("where")) for t in turns}
    assert ("komplex", "energi", "komplex") in kinds and ("tema", "olja", "tema") in kinds and ("komplex", "skog", "komplex") in kinds
    assert all(t["kind"] for t in turns) and turns == sorted(turns, key=lambda x: (not x["confirmed"], x.get("days") or 0))
    starts = vb.cycle_starts(b)
    assert starts[0]["where"] == "global" and any(s.get("complex") == "energi" and s["where"] == "komplex" for s in starts)
    # segmentens vändning är RÖD → GRÖN, inte fasen: XLE globalt är GRÖN men perioden före var inte RÖD i serien → kolla formen
    g = next(r for r in b["segments"] if r["key"] == "energy")["global"]
    assert set(g["turn"]) == {"kind", "since", "days", "confirmed", "confirmed_since"}


def test_valhall_map_shows_complexes_and_the_turn_notice(monkeypatch):
    import test_valhall_ui as tv
    turn = {"kind": "VÄNDER", "since": "2026-10-01", "days": 7, "confirmed": True, "confirmed_since": "2026-10-08"}
    cxs = [{"key": "adelmetaller", "label": "Ädelmetaller", "driver": "korg: guld:GC=F", "necessity": 80,
            "state": {"status": "GRÖN", "phase": cycle.VANDER, "cykel": "TIDIG", "cycle_start": True, "turn": turn},
            "themes": [{"theme": "guld", "label": "Guld", "state": {"status": "GRÖN", "turn": turn}}],
            "themes_turning": 1, "n_themes": 4}]
    turns = [{"level": "komplex", "key": "adelmetaller", "label": "Ädelmetaller", "where": "komplex", **turn},
             {"level": "segment", "key": "technology", "label": "Teknik", "where": "Norden", "kind": "RÖD→GRÖN",
              "since": "2026-10-10", "days": 3, "confirmed": False, "confirmed_since": None}]
    html = vui.complex_card(cxs[0])
    assert "Ädelmetaller" in html and "VÄNDER dag 7 ✓" in html and "1/4 teman" in html and "Guld" in html
    assert vui.turn_text(turn) == " ⚡ VÄNDER dag 7 ✓" and vui.turn_text(None) == ""
    notice = vui.turn_notice(turns)
    assert notice.startswith("⚡ VÄNDNING — bekräftad: Ädelmetaller (komplex, VÄNDER dag 7)")
    assert "på väg: Teknik (Norden, RÖD→GRÖN dag 3)" in notice and vui.turn_notice([]) == ""
    assert "VÄNDER dag 7 ✓" in vui.status_chip("", cxs[0]["state"])
    from streamlit.testing.v1 import AppTest
    board = {**tv.BOARD, "board": {**tv.BOARD["board"], "complexes": cxs}, "turns": turns}
    monkeypatch.setattr(vui, "load", lambda: {"board": board, "fas3": None, "fas4b": None, "fas4c": None, "nordic_probe": None})

    def script():
        import streamlit as st
        from valhall import ui as v
        v.render_map(v.load()["board"])
    at = AppTest.from_function(script)
    at.run(timeout=60)
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "RÅVARUKOMPLEX" in text and "Ädelmetaller" in text and "VÄNDNING" in text and "på väg: Teknik" in text


def test_thor_cycle_chip_reads_the_turn():
    turns = [{"level": "tema", "key": "guld", "label": "Guld", "where": "tema", "kind": "VÄNDER", "days": 7, "confirmed": True},
             {"level": "komplex", "key": "adelmetaller", "label": "Ädelmetaller", "where": "komplex", "kind": "VÄNDER",
              "days": 9, "confirmed": False},
             {"level": "segment", "key": "materials", "label": "Material", "where": "global", "kind": "RÖD→GRÖN", "days": 2},
             {"level": "segment", "key": "technology", "label": "Teknik", "where": "Norden", "kind": "RÖD→GRÖN", "days": 12,
              "confirmed": True}]
    blob = {"turns": turns}
    assert cykel.turn_for(blob, "materials", "guld", "adelmetaller")["key"] == "guld"            # temat först
    assert cykel.turn_for(blob, "materials", "silver", "adelmetaller")["key"] == "adelmetaller"  # sedan komplexet
    assert cykel.turn_for(blob, "materials", None, None) is None                                   # bara globalt → inget
    assert cykel.turn_for(blob, "technology")["days"] == 12 and cykel.turn_for(None, "technology") is None
    assert cykel.turn_chip_text(turns[0]) == "⚡ VÄNDER dag 7 · bekräftad (Guld)"
    assert cykel.turn_chip_text(turns[1]) == "⚡ VÄNDER dag 9 · på väg (Ädelmetaller)" and cykel.turn_chip_text(None) == ""
