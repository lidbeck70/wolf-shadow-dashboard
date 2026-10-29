"""
Arkens dom läser riktiga scorecarden (arken, PR A): Thor räknar köpgrinden på samma kandidatrad som 🎯 Scorecard-
fliken — inte på en attrapp utan arkrad, där DS alltid blev 'inte bedömd' och domen aldrig kunde säga KÖP.
Rick Rule-kortet med DS 0/10 och alla kryss: KLAR FÖR KÖP i fliken, READY i Thors Arken-flik och KÖP i Arkens dom.
Inga regler ändras. Inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scorecard as sc  # noqa: E402
from thor import dom  # noqa: E402
from thor import dossier as td  # noqa: E402

ROW = {"id": "a1", "ticker": "BOL", "name": "Boliden", "commodity": "Koppar", "price": 9000.0, "unit_cost": 6000.0,
       "ev_ebitda": 5.0, "nd_ebitda": 0.5, "mine_life": 12.0, "rp_ratio": 15.0, "jurisdiktion": True,
       "kapitaldisciplin": True, "insyn": True,
       "ds_runway": 0, "ds_capex": 0, "ds_warrants": 0, "ds_aktier_3ar": 0, "ds_historik": 0}
CARD = {**{k: True for k in ("strategi_aktiv", "screener_kval", "granskning_klar", "sakerhetsmarginal",
                             "trigger_definierad", "position_saljregel")},
        "strategy": "producenter", "position_pct_total": 1.5}
KEY = "boliden|BOL"


def _stores():
    return {"producers": {"producers": [dict(ROW)], "royalty": []}, "scorecard": {"cards": {KEY: dict(CARD)}}}


class _Score:
    def __init__(self, total, rating):
        self.total, self.rating, self.band = total, rating, rating


class _Analysis:
    recommendation, recommendation_why = "BUY CANDIDATE", "ok"
    case, confidence = _Score(85.0, "PICK"), _Score(75.0, "GOOD")


def test_sources_from_stores_are_the_tabs_sources():
    s = sc.sources_from_stores({
        **_stores(), "scoring": {"sprott": [{"ticker": "SPR.V"}], "durrett": []},
        "tiggre": {"candidates": [{"ticker": "T1.V"}]},
        "holdings": {"tiggre": [{"ticker": "T2.V", "strategy": "Tiggre", "entry_price": 1.0, "shares": 10}],
                     "long": [{"ticker": "L.ST", "strategy": "Quality", "entry_price": 1.0, "shares": 1}]},
        "insider": {"signals": [{"ticker": "EVO.ST"}]},
        "confidence": {"companies": {"EQX": {"ticker": "EQX", "name": "Equinox", "fields": {"x": 1}}}}})
    assert set(s) == {"sprott", "durrett", "tiggre", "producenter", "royalty", "insider", "confidence"}
    assert [r["ticker"] for r in s["tiggre"]] == ["T1.V", "T2.V"]                     # kandidater + registrets positioner
    assert s["producenter"][0]["ticker"] == "BOL" and s["insider"][0]["ticker"] == "EVO.ST"
    assert s["confidence"][0]["ticker"] == "EQX" and "fields" not in s["confidence"][0]
    assert sc.collect(sc.sources_from_stores(None)) == [] and sc.collect(sc.sources_from_stores({"scoring": 7})) == []


def test_thor_judges_the_same_entry_as_the_scorecard_tab():
    stores = _stores()
    entries = sc.entries_for(["BOL.ST", "BOL"], stores)
    assert list(entries) == [KEY] and entries[KEY]["rows"]["producenter"]["ds_runway"] == 0
    assert sc.gate_state(CARD, entries[KEY])["ready"]                                  # fliken: KLAR FÖR KÖP
    assert sc.gate_state(CARD, {"ticker": "BOL.ST"})["gaps"] == ["DS är inte bedömd"]  # attrappen: aldrig klar
    sheets = td.sheets_for("BOL.ST", stores)
    assert sheets["scorecard_entries"] == {KEY: entries[KEY]}
    d = dom.judge("rule", sheets, {}, "BOL.ST", company_fn=lambda s, t: object(), analyze_fn=lambda c: _Analysis())
    by = {c.key: c for c in d.checks}
    assert by["scorecard"].status == dom.PASS and by["scorecard"].note == "KLAR FÖR KÖP · producenter"
    assert by["ctl_DS"].status == dom.PASS and by["review"].status == dom.PASS and d.word == dom.KOP
    assert sc.entries_for(["NEM"], stores) == {} and sc.entries_for([], stores) == {}


def test_a_card_without_its_sheet_row_is_a_gap_not_a_pass():
    sheets = td.sheets_for("BOL.ST", {"scorecard": {"cards": {KEY: dict(CARD)}}})     # raden borttagen ur arket
    assert sheets["scorecard_entries"] == {KEY: None}
    check = dom._scorecard_check("rule", sheets, "BOL.ST", None)
    assert check.status == dom.MANUAL and "Kontrollerna utan permanent-risk-flagga" in check.note


def _script():
    import streamlit as st
    import scorecard
    from thor import arken as a
    from thor import dossier as d
    from thor import ui as t
    scorecard.render_scorecard_page()
    stores = {k: st.session_state.get(k) for k in ("producers", "scorecard")}
    a.render(st, {}, d.identify("BOL.ST"), d.sheets_for("BOL.ST", stores), [], t.SHEET_LINK, t._jump)


def test_scorecard_tab_and_thors_arken_tab_agree(monkeypatch):
    import storage
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(storage, "session_load",
                        lambda name, default=None, legacy_file=None: st.session_state.setdefault(name, default))
    at = AppTest.from_function(_script, default_timeout=120)
    for k, v in _stores().items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, at.exception
    assert {m.label: m.value for m in at.metric}["Klara för köp"] == "1"
    text = " ".join(m.value for m in at.markdown)
    assert "KLAR FÖR KÖP" in text and "Scorecard: READY · producenter" in text and "DS är inte bedömd" not in text
