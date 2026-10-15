"""
Thor fas 6 — thor/arken: insiderklustret och Confidence-bolaget (ren logik) och Arken-fliken under AppTest
med arken i sessionen: granskningarna, Durrett/Wolf Asymmetry-motorerna, scorecardets kryss, insiderklustret,
öppna och stängda positioner, journalen och Copilotens anteckningar — och att ett block som fallerar blir en
notering. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_candidates as fx  # noqa: E402
from thor import arken  # noqa: E402
from thor import dossier as td  # noqa: E402

STORES = {
    "producers": {"producers": [{"ticker": "BOL", "name": "Boliden", "commodity": "Koppar", "price": 9000.0, "unit_cost": 6000.0,
                                 "ev_ebitda": 5.0, "nd_ebitda": 0.5, "mine_life": 12.0, "rp_ratio": 15.0}], "royalty": []},
    "holdings": {"long": [{"ticker": "BOL.ST", "name": "Boliden", "strategy": "Quality", "entry_price": 300.0, "shares": 10}],
                 "closed": [{"ticker": "BOL.ST", "strategy": "Wolf", "entry_price": 280.0, "exit_price": 310.0, "shares": 20,
                             "entry_date": "2026-03-01", "exit_date": "2026-05-01", "exit_reason": "TP1"}]},
    "confidence": {"companies": {"BOL.ST": {"ticker": "BOL.ST", "name": "Boliden", "commodity": "Koppar", "stage": "producer",
                                            "fields": {}}}, "commodity_overrides": {}, "strategies": {}},
    "scorecard": {"cards": {"Quality|BOL.ST": {"strategy": "Quality", "position_pct_total": 3.0, "tes_skriven": True}}},
    "copilot_journal": {"entries": [{"date": "2026-10-01", "ticker": "BOL.ST", "strategy": "Wolf", "entry": 300, "stop": 280,
                                     "target": 350, "note": "Pullback mot EMA10"}]},
}
INSIDER = {"generated": "2026-10-14", "clusters": [
    {"ticker": "BOL", "name": "Boliden", "score": 8, "status": "KÖP — logga i journalen", "stop": 285.0, "price_now": 312.0,
     "chase": False, "insiders": 3, "role": "VD/CFO", "cluster_avg": 300.0}]}


def test_insider_cluster_and_company_lookup_accept_every_ticker_form():
    assert arken.insider_cluster(INSIDER, "BOL.ST")["insiders"] == 3 and arken.insider_cluster(INSIDER, "bol")["score"] == 8
    assert arken.insider_cluster(INSIDER, "NEM") is None and arken.insider_cluster(None, "BOL.ST") is None
    assert arken.insider_cluster({"clusters": ["skräp", None]}, "BOL.ST") is None
    c = arken.company_for(STORES["confidence"], "bol.st")
    assert c is not None and c.name == "Boliden" and c.commodity == "Koppar"
    assert arken.company_for(STORES["confidence"], "NEM") is None and arken.company_for(None, "BOL.ST") is None
    sheets = td.sheets_for("BOL.ST", STORES)
    assert sheets["confidence_store"] is STORES["confidence"] and sheets["closed"][0]["exit_price"] == 310.0


def _script():
    import streamlit as st
    from thor import arken as a
    from thor import dossier as d
    from thor import ui as t
    ident = d.identify("BOL.ST")
    sheets = d.sheets_for("BOL.ST", st.session_state["_stores"])
    a.render(st, {"insider": st.session_state["_insider"]}, ident, sheets, [], t.SHEET_LINK, t._jump)


def _run(monkeypatch, stores=STORES, insider=INSIDER):
    import storage
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None: st.session_state.setdefault(name, default))
    at = AppTest.from_function(_script, default_timeout=120)
    at.session_state["_stores"], at.session_state["_insider"] = stores, insider
    at.run()
    assert not at.exception, at.exception
    return at, " ".join(m.value for m in at.markdown)


def test_tab_shows_every_section_from_the_sheets(monkeypatch):
    at, text = _run(monkeypatch)
    for part in ("Rick Rule:", "CONFIDENCE · DURRETT · WOLF ASYMMETRY", "Boliden · Koppar · producer", "Durrett kvalitet",
                 "Hävstång", "Säkerhetsmarginal", "Break-even", "SCORECARDET", "Scorecard: NO TRADE · Quality",
                 "Strategin är aktiv", "INSIDERKLUSTRET", ">8/10<", "KÖP — logga i journalen", "VD/CFO", "stopp 285.0",
                 "POSITIONER", "STÄNGDA POSITIONER", "COPILOTENS ANTECKNINGAR", "Pullback mot EMA10"):
        assert part in text, part
    assert len(at.dataframe) == 2                                                           # öppna och stängda
    labels = [b.label for b in at.button]
    assert "Rick Rule →" in labels and "Royalty →" in labels


def test_tab_without_sheets_says_so_and_a_failing_block_is_a_note(monkeypatch):
    _at, text = _run(monkeypatch, stores={}, insider=None)
    assert "finns inte i något ark än" in text and "INSIDERKLUSTRET" not in text and "SCORECARDET" not in text
    import scorecard

    def boom(card, entry=None):
        raise RuntimeError("kortet trasigt")
    monkeypatch.setattr(scorecard, "gate_state", boom)
    _at, text = _run(monkeypatch)
    assert "Scorecardet kunde inte läsas: kortet trasigt" in text and "INSIDERKLUSTRET" in text
    assert fx.BOARD                                                                          # fixturen laddas (importvakt)
