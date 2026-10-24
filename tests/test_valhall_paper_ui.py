"""
K8 — ARKEN → 📊 Papperskonton: sidan läser Valhalls och BERSERK:s papper ur Gisten (fejkat) och visar
sammanställning, utveckling, öppna/stängda affärer och händelser. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thor import registry as reg  # noqa: E402
from ui import nav  # noqa: E402
from valhall import paper_ui as pui  # noqa: E402

STATE = {"last_run": "2026-10-23", "positions": [
    {"strategy": "komplex_deep", "ticker": "GOLD.ST", "name": "Guld", "complex": "adelmetaller", "segment": "materials",
     "signal_date": "2026-10-20", "entry_date": "2026-10-21", "entry": 100.0, "last_close": 110.0, "init_stop": None,
     "turn": {"kind": "VÄNDER", "days": 12, "confirmed": True}, "why": "vändning"}],
    "orders": [{"strategy": "ember", "ticker": "SCA-B.ST", "signal_date": "2026-10-23", "why": "KÖPLÄGE"}],
    "closed": [{"strategy": "viking", "ticker": "VOLV-B.ST", "entry_date": "2026-10-01", "exit_date": "2026-10-10",
                "entry": 100.0, "exit": 106.0, "init_stop": 97.0, "reason": "Trailing"}],
    "events": [{"date": "2026-10-23", "kind": "KÖP PÅ ÖPPNING", "strategy": "ember", "ticker": "SCA-B.ST", "text": "x"}]}
HISTORY = [{"date": "2026-10-22", "by_strategy": {"viking": {"avg_r": 0.5}}},
           {"date": "2026-10-23", "by_strategy": {"viking": {"avg_r": 0.7}, "komplex_deep": {"avg_r": None}}}]
BOARD = {"paper": {"since": "2026-10-01", "positions": 1, "orders": 1,
                   "by_strategy": [{"strategy": "viking", "n": 1, "closed": 1, "closed_n": "1/20", "avg_r": 2.0}]}}


def test_tables_and_history():
    pos = pui.positions_table(STATE)
    assert list(pos["Läge"]) == ["position", "order"] and pos.iloc[0]["Avk. %"] == 10.0
    assert pos.iloc[0]["Vändning"] == "VÄNDER dag 12 ✓" and pos.iloc[0]["Strategi"] == "Deep Contrarian · vändning"
    cl = pui.closed_table(STATE)
    assert cl.iloc[0]["R"] == 2.0 and cl.iloc[0]["Strategi"] == "Viking Nine"
    assert len(pui.events_table(STATE)) == 1 and pui.positions_table(None).empty
    df = pui.history_frame(HISTORY)
    assert list(df["Viking Nine"]) == [0.5, 0.7] and pui.history_frame(None).empty


def test_navigation_and_registry():
    assert "📊 Papperskonton" in nav.SUBS["review"]
    for k in ("ember", "deep", "royalty"):
        assert reg.get(k).paper == k


def test_page_renders(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import berserk.screen_ui as bsu
    monkeypatch.setattr(bsu, "render_paper_tab", lambda: None)
    data = {"board": BOARD, "state": STATE, "history": HISTORY}
    monkeypatch.setattr(pui, "load", lambda loader=None: {**data, "loaded": "nu"})

    def script():
        from valhall import paper_ui as p
        p.render_paper_page()
    at = AppTest.from_function(script)
    at.run(timeout=60)
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "PAPPERSKONTON" in text.upper() and "ÖPPNA POSITIONER" in text and "STÄNGDA AFFÄRER" in text
