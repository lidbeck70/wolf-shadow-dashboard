"""
Thor-verifieringens rättelser: underlaget till ChatGPT, fokusläget och importen läser arken på samma former som
dossiern ('TEL' i arket är TE Connectivity, aldrig Telenor TEL.OL; en bar 'EQX' är inte EQX.TO), assemble tar
registrets karta, och Arken-fliken säger vilka rader på kortnamnet som inte räknas och vilka arkrader som matchats
utan kartan. Kandidatkortet visar segmentets läge i aktiens egen vy — globalt utanför Norden, som huvudet.
Inga regler ändras. Inget nätverk.
"""
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_candidates as fx  # noqa: E402
import test_thor_ui as tu  # noqa: E402
from review import answers as an  # noqa: E402
from review import export as ex  # noqa: E402
from review import focus  # noqa: E402
from review import importer as im  # noqa: E402
from test_review_focus import _cards, page  # noqa: E402,F401,F811  (fixturen: arken under AppTest)
from test_thor_lage_historik_bors import _board, _held, _mine, _universe  # noqa: E402
from thor import candidates as tc  # noqa: E402
from thor import cards  # noqa: E402
from thor import dossier as td  # noqa: E402
from thor import registry as reg  # noqa: E402
from thor import ui as tui  # noqa: E402


@pytest.fixture(autouse=True)
def _no_net(monkeypatch):
    """Inget nätverk: varje uppkoppling stoppas och listas — testet kräver att listan är tom."""
    import socket
    tried = []

    def offline(self, address, *_a, **_k):
        tried.append(address)
        raise OSError("inget nätverk i testet")
    monkeypatch.setattr(socket.socket, "connect", offline)
    yield
    assert tried == [], tried


def _rule(ticker, name, **kw) -> dict:
    return {"id": f"p_{ticker}", "ticker": ticker, "name": name, "commodity": "Koppar", "date": "2026-09-01",
            "price": 4.1, "unit_cost": 2.0, **kw}


# Arken som de skrevs: TE Connectivitys bara 'TEL', Equinox bara 'EQX', Ericsson och Boliden i Börsdatas form
STORES = {
    "producers": {"producers": [_rule("TEL", "TE Connectivity"), _rule("EQX", "Equinox"), _rule("ERIC B", "Ericsson")],
                  "royalty": []},
    "confidence": {"companies": {"TEL": {"ticker": "TEL", "name": "TE Connectivity", "commodity": "x"},
                                 "EQX": {"ticker": "EQX", "name": "Equinox", "commodity": "guld"}},
                   "commodity_overrides": {}},
    "scorecard": {"cards": {k: {"strategy": "producenter"}
                            for k in ("te connectivity|TEL", "equinox|EQX", "ericsson|ERIC B", "boliden|BOL")}},
}
TICKERS = ("TEL.OL", "TEL", "EQX.TO", "EQX", "ERIC-B.ST", "BOL.ST")


def _dossier_sheets(s: dict) -> set:
    return set(s["reviews"]) | {k for k in ("confidence", "scorecard") if s[k]}


# ── Underlaget till ChatGPT läser samma rader som dossiern ──────────────────
@pytest.mark.parametrize("registry", [False, True])
def test_export_finds_the_same_sheet_rows_as_the_dossier(registry):
    smap = _universe().sheet_map() if registry else None
    for t in TICKERS:
        assert set(ex.sheets_with(t, STORES, smap)) == _dossier_sheets(td.sheets_for(t, STORES, smap)), t
    assert ex.sheets_with("EQX.TO", STORES, smap) == [] and ex.find("confidence", "EQX.TO", STORES, smap) is None
    assert ex.find("rule", "ERIC-B.ST", STORES, smap)["name"] == "Ericsson"   # Börsdatas form når den nordiska
    assert ex.card_for("BOL.ST", STORES, smap)[0] == "boliden|BOL"
    assert set(ex.sheets_with("TEL", STORES, smap)) == {"rule", "confidence", "scorecard"}
    if registry:                                                                # TE Connectivity är inte Telenor
        assert ex.sheets_with("TEL.OL", STORES, smap) == [] and ex.card_for("TEL.OL", STORES, smap)[0] is None


def test_export_on_telenor_is_an_empty_template_not_te_connectivitys_row():
    smap = _universe().sheet_map()
    exp = ex.build("TEL.OL", STORES, ["rule"], blob=None, sheet_map=smap)
    assert exp.absent == ["rule"] and exp.filled == 0 and exp.name == "" and exp.source_rows["rule"] is None
    assert exp.sheet_map == smap and all(r["ticker"] == "TEL.OL" for r in exp.rows)
    te = ex.build("TEL", STORES, ["rule"], blob=None, sheet_map=smap)
    assert te.absent == [] and te.filled > 0 and te.name == "TE Connectivity"
    eric = ex.build("ERIC-B.ST", STORES, ["rule"], blob=None, sheet_map=smap)
    assert eric.absent == [] and eric.name == "Ericsson"
    assert ex.build("EQX.TO", STORES, ["rule"], blob=None).absent == ["rule"]  # en bar 'EQX' är den amerikanska


# ── Importen prövar svarets ticker mot samma former ─────────────────────────
def _answer(exp, *lines):
    return im.check(an.read_text("\n".join([ex.ANSWER_HEADER] + list(lines))), exp)


def _other(res) -> list:
    return [t for _l, _k, t in res.issues if t.startswith("annan ticker")]


def _one(row, smap=None):
    """Underlaget för arkets egen rad (som arkens kort bygger det), med eller utan registrets karta."""
    return ex.build(row["ticker"], {"producers": {"producers": [row], "royalty": []}}, ["rule"], blob=None,
                    rows={"rule": row}, sheet_map=smap)


def test_import_checks_the_answer_ticker_against_the_same_forms():
    exp = _one(_rule("EQX.TO", "Equinox"))
    assert _other(_answer(exp, "EQX;rule:mine_life;9;år;x;primary;;high;"))[0].startswith("annan ticker (EQX)")
    ok = _answer(exp, "eqx.to;rule:mine_life;9;år;x;primary;;high;")
    assert _other(ok) == [] and [c.key for c in ok.changes] == ["rule:mine_life"]
    smap = _universe().sheet_map()                                             # med registrets karta, som i Thor
    exp = _one(_rule("TEL.OL", "Telenor"), smap)
    assert _other(_answer(exp, "TEL;rule:mine_life;9;år;x;primary;;high;"))
    assert _other(_answer(exp, "TEL.OL;rule:mine_life;9;år;x;primary;;high;")) == []
    exp = _one(_rule("BOL.ST", "Boliden"), smap)
    assert _other(_answer(exp, "BOL;rule:mine_life;9;år;x;primary;;high;")) == []   # Börsdatas form: samma bolag
    exp = _one(_rule("BOL", "Boliden"))                                       # arkets egen form, utan karta: som förut
    assert _other(_answer(exp, "BOL.ST;rule:mine_life;9;år;x;primary;;high;")) == []


# ── Fokusläget öppnar arket på raden dossiern läste ─────────────────────────
def test_focus_uses_the_forms_thor_read_the_company_on():
    assert not focus.matches("TEL", "TEL.OL", ["TEL.OL"]) and focus.matches("tel.ol", "TEL.OL", ["TEL.OL"])
    assert focus.matches("ERIC B", "ERIC-B.ST", td.sheet_forms("ERIC-B.ST", _universe().sheet_map()))
    assert focus.matches("BOL", "BOL.ST", ["BOL.ST", "BOL"]) and focus.matches("BOL", "BOL.ST")   # utan: som förut
    assert not focus.matches("EQX", "EQX.TO") and not focus.matches("EQX.TO", "EQX")
    assert focus.pick(["TEL", "NEM"], "TEL.OL", ["TEL.OL"]) is None
    assert focus.pick(["TEL", "NEM"], "TEL.OL") == "TEL"                       # utan Thors former: går inte att skilja


def test_sheet_opens_on_the_row_thor_read(page):  # noqa: F811
    prod = {"producers": [_rule("TEL", "TE Connectivity"), _rule("NEM", "Newmont"), _rule("BOL", "Boliden")],
            "royalty": []}
    tel, bol = {"ticker": "TEL.OL", "forms": ["TEL.OL"]}, {"ticker": "BOL.ST", "forms": ["BOL.ST", "BOL"]}
    at = page("producers", "render_producers_page", {"sheet": "Rick Rule"},
              {"producers": copy.deepcopy(prod), focus.FOCUS: "TEL.OL", focus.FORMS: tel})
    assert sorted(c.label.split(" ·")[0] for c in _cards(at, "/5 p")) == ["BOL", "NEM", "TEL"]   # TE ≠ Telenor
    assert any("TEL.OL finns inte i arket — visar alla bolag." in c.value for c in at.caption)
    at = page("producers", "render_producers_page", {"sheet": "Rick Rule"},
              {"producers": copy.deepcopy(prod), focus.FOCUS: "BOL.ST", focus.FORMS: bol})
    assert [c.label.split(" ·")[0] for c in _cards(at, "/5 p")] == ["BOL"]
    at = page("producers", "render_producers_page", {"sheet": "Rick Rule"},                  # formerna för en annan
              {"producers": copy.deepcopy(prod), focus.FOCUS: "NEM", focus.FORMS: tel})     # ticker gäller inte
    assert [c.label.split(" ·")[0] for c in _cards(at, "/5 p")] == ["NEM"]
    at.button(key="arken_focus_rule_all").click().run()                         # Visa alla släpper formerna också
    assert at.session_state[focus.FOCUS] == "" and focus.FORMS not in at.session_state


# ── assemble tar kartan, och inget försvinner tyst ──────────────────────────
def test_assemble_takes_the_registry_map():
    smap = _universe().sheet_map()
    s = td.assemble("TEL.OL", stores=STORES, sheet_map=smap)["sheets"]
    assert s["confidence"] is None and s["scorecard"] == {} and s["reviews"] == {} and s["sheet_forms"] == ["TEL.OL"]
    assert s["ambiguous"] == [] and "Durrett/Confidence 'TEL'" in s["elsewhere"]
    loose = td.assemble("TEL.OL", stores=STORES)["sheets"]                     # utan kartan: matchas, men sägs
    assert loose["confidence"]["name"] == "TE Connectivity" and loose["ambiguous"] == ["TEL"]


def test_rows_on_the_short_name_that_do_not_count_are_listed():
    smap = _universe().sheet_map()
    s = td.sheets_for("BOL.ST", _held("BOL", "Boliden"), smap)
    assert _mine(s) == (0, 0, 0, 0)
    assert s["elsewhere"] == ["innehav 'BOL'", "stängd position 'BOL'", "journalen 'BOL'", "Copiloten 'BOL'"]
    assert td.sheets_for("ERIC-B.ST", _held("ERIC-B", "Ericsson"), smap)["elsewhere"][0] == "innehav 'ERIC-B'"
    eqx = td.sheets_for("EQX.TO", STORES, smap)
    assert eqx["elsewhere"] == ["Rick Rule 'EQX'", "Durrett/Confidence 'EQX'", "Scorecardet 'EQX'"]
    assert td.sheets_for("BOL.ST", _held("BOL.ST", "Boliden"), smap)["elsewhere"] == []     # rätt form: inget att säga
    assert td.sheets_for("TEL", {**STORES, **_held("TEL", "TE")}, smap)["elsewhere"] == []
    ok = td.sheets_for("ERIC-B.ST", STORES, smap)
    assert ok["elsewhere"] == [] and ok["ambiguous"] == [] and ok["found"]
    bol = td.sheets_for("BOL.ST", STORES)                                       # utan kartan: 'BOL' gissas nordisk
    assert list(bol["scorecard"]) == ["boliden|BOL"] and bol["ambiguous"] == ["BOL"]
    assert td.sheets_for("ERIC-B.ST", STORES)["ambiguous"] == []               # 'ERIC B' är alltid nordisk
    assert td.sheets_for("TEL", STORES)["ambiguous"] == []                     # en bar sida är aldrig tvetydig


def _arken_script():
    import streamlit as st
    from thor import arken as a
    from thor import dossier as d
    from thor import ui as t
    smap = st.session_state["_smap"]
    ident = d.identify(st.session_state["_t"])
    sheets = d.sheets_for(ident.yf, st.session_state["_stores"], smap)
    a.render(st, {"sheet_map": smap}, ident, sheets, [], t.SHEET_LINK, t._jump, strategy="rule")


def _arken(monkeypatch, ticker, stores, smap):
    import storage
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
    at = AppTest.from_function(_arken_script, default_timeout=120)
    at.session_state["_t"], at.session_state["_stores"], at.session_state["_smap"] = ticker, stores, smap
    at.run()
    assert not at.exception, at.exception
    return at


def test_arken_tab_on_telenor_keeps_te_connectivity_out_of_every_part(monkeypatch):
    stores = {**copy.deepcopy(STORES), **_held("TEL", "TE Connectivity")}
    at = _arken(monkeypatch, "TEL.OL", stores, _universe().sheet_map())
    text = " ".join(m.value for m in at.markdown)
    assert ("Visas inte på TEL.OL: Rick Rule 'TEL', Durrett/Confidence 'TEL', Scorecardet 'TEL', innehav 'TEL', "
            "stängd position 'TEL', journalen 'TEL', Copiloten 'TEL'.") in text
    assert "skriv om raden som TEL.OL" in text and "TE Connectivity · " not in text and "POSITIONER" not in text
    caps = " ".join(c.value for c in at.caption)                               # underlaget: en tom mall, inte TE:s rad
    assert "Finns inte i Rick Rule än" in caps and "0/" in caps
    at.button(key="thor_sheet_rule").click().run()
    assert at.session_state[focus.FOCUS] == "TEL.OL" and at.session_state["nav_goto"] == "review/rick-rule"
    assert at.session_state[focus.FORMS] == {"ticker": "TEL.OL", "forms": ["TEL.OL"]}


def test_arken_tab_on_ericsson_reads_the_borsdata_row_and_says_nothing_extra(monkeypatch):
    at = _arken(monkeypatch, "ERIC-B.ST", copy.deepcopy(STORES), _universe().sheet_map())
    text = " ".join(m.value for m in at.markdown)
    assert "Rick Rule:" in text and "Visas inte på" not in text and "tickerkarta saknas" not in text
    assert "Finns inte i Rick Rule" not in " ".join(c.value for c in at.caption)
    at.button(key="thor_sheet_rule").click().run()
    assert at.session_state[focus.FORMS]["forms"] == ["ERIC-B.ST", "ERIC-B", "ERIC B"]


def test_arken_tab_without_the_map_says_the_match_is_a_guess(monkeypatch):
    at = _arken(monkeypatch, "TEL.OL", copy.deepcopy(STORES), {})
    text = " ".join(m.value for m in at.markdown)
    assert "Registrets tickerkarta saknas, så arkraderna 'TEL' matchas här på kortnamnet" in text


# ── Kortet visar segmentets läge i aktiens egen vy ──────────────────────────
def _rows(board):
    return {(c.strategy, c.ticker): c for c in tc.build({**fx.BLOBS, "board": board}, today="2026-10-15")}


def test_card_shows_the_segment_status_in_the_stocks_own_view():
    rows = _rows(_board("GUL"))
    fcx = rows[("berserk", "FCX")]                                              # USA, koppar saknas på kartan
    assert fcx.status_of == "segment" and not fcx.nordic and fcx.segment_status == "RÖD" and fcx.global_status == "GUL"
    card = cards.candidate_card(fcx)
    assert ">Material · globalt GUL<" in card and "Material · RÖD" not in card and ">Koppar<" in card
    head = cards.header_card(td.identify("FCX").as_dict(), [fcx], board=_board("GUL"))
    assert ">Material · globalt GUL<" in head                                    # kortet och huvudet säger samma
    bol = cards.candidate_card(rows[("berserk", "BOL.ST")])                     # Norden: Nordens läge som förut
    assert ">Material · RÖD · TOPP<" in bol and "globalt" not in bol
    assert ">Guld · GRÖN · TIDIG<" in cards.candidate_card(rows[("rule", "NEM")])   # temats läge oförändrat
    plain = cards.candidate_card(_rows(fx.BOARD)[("berserk", "FCX")])          # ingen global vy: inget läge
    assert ">Material<" in plain and "RÖD" not in plain and "globalt" not in plain


def test_list_shows_the_us_stock_in_the_global_view(monkeypatch):
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    monkeypatch.setattr(tui, "load", lambda loader=None: {**tu.DATA, "board": _board("GUL")})
    at = AppTest.from_function(tu._script)
    at.run(timeout=90)
    at.selectbox(key="thor_pick").select(reg.label("berserk")).run(timeout=90)   # BERSERK: FCX och BOL.ST
    assert not at.exception, at.exception
    fcx = [m.value for m in at.markdown if ">FCX</b>" in m.value]
    assert fcx and all(">Material · globalt GUL<" in v and "Material · RÖD" not in v for v in fcx)
    bol = [m.value for m in at.markdown if ">BOL.ST</b>" in m.value]
    assert bol and any(">Material · RÖD · TOPP<" in v for v in bol)
