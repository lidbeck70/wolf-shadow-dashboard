"""
Thor-granskningen (T9, T11, T13): segmentchippen säger vems läge det är (temats GRÖN står under temats namn, aldrig
under segmentets — kartans Material kan vara RÖD/TOPP när guldet är GRÖN/TIDIG), backtestet säger när aktiens
historik är för kort i stället för 'reglerna gav ingen entry', och innehav/journal/Copilot/arken matchas på rätt
börs ('TEL' är TE Connectivity, aldrig Telenor TEL.OL). Inga regler ändras. Inget nätverk.
"""
import copy
import os
import sys
import zlib

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_candidates as fx  # noqa: E402
import test_thor_ui as tu  # noqa: E402
from test_wolf_regime_score import _walk  # noqa: E402
from thor import backtest as tb  # noqa: E402
from thor import candidates as tc  # noqa: E402
from thor import cards  # noqa: E402
from thor import dossier as td  # noqa: E402
from thor import ui as tui  # noqa: E402
from universe import registry as ur  # noqa: E402


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


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown)


# ── T9: vems läge chippen visar ─────────────────────────────────────────────
def _rows():
    return {(c.strategy, c.ticker): c for c in tc.build(fx.BLOBS, today="2026-10-15")}


def _board(global_status=None):
    """fx.BOARD: Material Norden RÖD/TOPP, temat guld GRÖN/TIDIG — med Materials globala vy om den ges."""
    b = copy.deepcopy(fx.BOARD)
    if global_status:
        b["board"]["segments"][1]["global"] = {"status": global_status}
    return b


def test_the_row_records_whose_status_it_carries():
    rows = _rows()
    nem = rows[("rule", "NEM")]
    assert nem.segment == "materials" and nem.segment_status == "GRÖN" and nem.cykel == "TIDIG"   # temats läge som förut
    assert nem.status_of == "tema" and nem.as_dict()["status_of"] == "tema"
    bol = rows[("berserk", "BOL.ST")]                                             # koppar finns inte på kartan
    assert bol.status_of == "segment" and bol.segment_status == "RÖD" and bol.cykel == "TOPP"
    hexa = rows[("viking_nine", "HEXA-B.ST")]
    assert hexa.status_of == "segment" and hexa.segment_status == "GRÖN"
    assert tc.Candidate("X", "", "wolf", "", tc.HALL).status_of == ""              # inget läge alls


def test_segment_chip_names_the_owner_of_the_status():
    assert ">Guld · GRÖN · TIDIG<" in cards.segment_chip("materials", "GRÖN", "TIDIG", theme="guld")
    assert ">Material · RÖD · TOPP<" in cards.segment_chip("materials", "RÖD", "TOPP")
    assert ">Material · globalt GUL<" in cards.segment_chip("materials", "GUL", where="globalt")
    assert ">Material<" in cards.segment_chip("materials") and cards.segment_chip(None) == ""
    assert ">Guld<" in cards.segment_chip(None, theme="guld")                     # temat utan segment


def test_candidate_card_shows_the_theme_status_under_the_theme():
    rows = _rows()
    nem = cards.candidate_card(rows[("rule", "NEM")])
    assert ">Guld · GRÖN · TIDIG<" in nem and ">Material<" in nem
    assert "Material · GRÖN" not in nem and "Material · RÖD" not in nem          # segmentets läge påstås inte
    bol = cards.candidate_card(rows[("berserk", "BOL.ST")])                       # segmentets läge, temat som förut
    assert ">Material · RÖD · TOPP<" in bol and ">Koppar<" in bol
    assert ">Teknik · GRÖN · MITTEN<" in cards.candidate_card(rows[("viking_nine", "HEXA-B.ST")])


def test_header_reads_the_segment_status_from_the_map_for_the_stocks_market():
    nem_row = _rows()[("rule", "NEM")]                                            # temats GRÖN/TIDIG överst
    nem = td.identify("NEM").as_dict()
    assert nem["segment"] == "materials" and nem["theme"] == "guld" and not nem["nordic"]
    head = cards.header_card(nem, [nem_row], board=_board("GUL"))
    assert ">Material · globalt GUL<" in head and ">Guld · GRÖN · TIDIG<" in head
    assert "Material · GRÖN" not in head and "Material · RÖD" not in head         # inte Nordens läge för NEM
    assert ">Material<" in cards.header_card(nem, [nem_row], board=_board())      # ingen global vy: inget läge
    gold_se = {"yf": "XGLD.ST", "nordic": True, "segment": "materials", "theme": "guld", "theme_label": "Guld",
               "theme_source": "arket"}
    head = cards.header_card(gold_se, [nem_row], board=_board("GUL"))
    assert ">Material · RÖD · TOPP<" in head and ">Guld · GRÖN · TIDIG<" in head and "Guld · arket" in head
    bare = cards.header_card(gold_se, [nem_row])                                  # utan karta: inget läge alls
    assert ">Material<" in bare and "GRÖN" not in bare and "RÖD" not in bare


def test_page_header_and_list_never_put_the_theme_status_under_the_segment(monkeypatch):
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    at = AppTest.from_function(tu._script)
    at.run(timeout=90)
    assert not at.exception, at.exception
    assert dict(tui.segment_choices(fx.BOARD))["materials"] == "Material · RÖD"   # rullgardinen
    text = _text(at)
    assert "Guld · GRÖN · TIDIG" in text and "Material · GRÖN" not in text
    at.text_input(key="thor_ticker").set_value("NEM").run(timeout=120)
    assert not at.exception, at.exception
    head = next(m.value for m in at.markdown if "finns i dag i" in m.value)
    assert ">NEM</b>" in head and ">Guld · GRÖN · TIDIG<" in head and "Material · GRÖN" not in head


# ── T11: för kort historik i backtestet ─────────────────────────────────────
TODAY = pd.Timestamp("2026-10-15")
BARS = {"1y": 260, "2y": 520, "3y": 780, "4y": 1040, "5y": 1300, "6y": 1560, "7y": 1820, "8y": 2080, "max": 2080}


def _prices(short: str = "", n: int = 0):
    """Deterministiska kurser (som test_thor_backtest); tickern short har bara n dagar, som en ny notering."""
    def get(ticker, period="1y"):
        bars = n if ticker == short else BARS.get(period, 1300)
        return _walk(zlib.crc32(ticker.encode()) % 1000, 0.0006, n=bars, end=TODAY)
    return get


def _first(n: int) -> str:
    """Första handelsdagen i en n dagars historik som slutar TODAY (samma index som _walk)."""
    return str(pd.bdate_range(end=TODAY, periods=n)[0].date())


@pytest.fixture
def engines(monkeypatch):
    import market_prices
    import ovtlyr_nine as on
    monkeypatch.setattr(market_prices, "ohlcv", _prices())
    monkeypatch.setattr(on, "nordic_market", lambda *a, **k: None)            # ingen Börsdata i testet


@pytest.mark.parametrize("n", [30, 61, 100])
def test_viking_nine_says_the_history_is_too_short(engines, n):
    first = _first(n)
    r = tb.run("viking_nine", td.identify("NEM"), 5, _prices("NEM", n), today=TODAY)
    assert r.period == (first, "2026-10-15")                                      # perioden börjar där datan börjar
    if n < 62:                                                                    # WARMUP_BARS 60 + 2: motorn ger tomt
        assert r.error == (f"bara {n} dagars historik (från {first}) — för kort för reglerna, som behöver 62 dagar "
                           f"(60 dagars uppvärmning).")
        assert r.metrics == [] and r.trades == []
    else:                                                                         # reglerna körs, på de sista 40 dagarna
        assert r.error == "" and r.metrics[0][0] == "Affärer"
        assert r.warnings[0].startswith(f"bara {n} dagars historik (från {first}) — perioden börjar där, inte 2021-10-15")
        assert f"reglerna kunde handla först från {str(pd.bdate_range(end=TODAY, periods=n)[60].date())}" in r.warnings[0]


@pytest.mark.parametrize("n", [30, 61, 100])
def test_berserk_says_the_history_is_too_short(engines, n):
    first = _first(n)
    r = tb.run("berserk", td.identify("BOL.ST"), 5, _prices("BOL.ST", n), today=TODAY)
    assert r.error.startswith(f"bara {n} dagars historik (från {first}) — för kort för reglerna, som behöver 262 dagar")
    assert r.period == (first, "2026-10-15") and r.metrics == []


def test_a_full_history_is_unchanged(engines):
    r = tb.run("viking_nine", td.identify("NEM"), 3, _prices(), today=TODAY)
    assert r.error == "" and r.period == ("2023-10-15", "2026-10-15") and not any("bara" in w for w in r.warnings)
    assert r.metrics[0][0] == "Affärer" and r.close is not None and len(r.close) > 700
    b = tb.run("berserk", td.identify("BOL.ST"), 3, _prices(), today=TODAY)
    assert b.error == "" and b.period == ("2023-10-15", "2026-10-15") and not any("bara" in w for w in b.warnings)


def test_short_history_without_data_or_period_says_nothing():
    res = tb.Result("viking_nine", "Viking Nine", 5, period=("2021-10-15", "2026-10-15"))
    assert tb.short_history(res, None, 60) is False and res.error == "" and res.warnings == []
    long_ = tb.Result("viking_nine", "Viking Nine", 5, period=("2021-10-15", "2026-10-15"))
    assert tb.short_history(long_, _walk(1, n=1600, end=TODAY), 60) is False and long_.warnings == []
    assert long_.period == ("2021-10-15", "2026-10-15")


def _bt_script():
    import streamlit as st
    import test_thor_lage_historik_bors as m
    from thor import backtest as b
    from thor import dossier as d

    def runner(strategy, ident, years, prices, today=None):
        return b.run(strategy, ident, years, m._prices("NEM", st.session_state["_n"]), today=m.TODAY)
    b.render(st, d.identify("NEM"), "viking_nine", lambda t, p="1y": None, runner=runner)


def test_tab_says_too_short_instead_of_no_entry(engines):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_bt_script, default_timeout=120)
    at.session_state["_n"] = 30
    at.run()
    at.button(key="thor_bt_run").click().run()
    assert not at.exception, at.exception
    text = _text(at)
    first = _first(30)
    assert f"Viking Nine på NEM: bara 30 dagars historik (från {first}) — för kort för reglerna" in text
    assert f"{first} → 2026-10-15" in text and "Inga affärer" not in text and "reglerna gav ingen entry" not in text


# ── T13: innehav, journal och arken på rätt börs ────────────────────────────
def _universe():
    m = ur.Member
    return ur.Universe(members={
        "BOL.ST": m("BOL.ST", "Boliden", ur.NORDEN, bd_ticker="BOL"),
        "ERIC-B.ST": m("ERIC-B.ST", "Ericsson B", ur.NORDEN, bd_ticker="ERIC B"),
        "TEL.OL": m("TEL.OL", "Telenor", ur.NORDEN, bd_ticker="TEL"),
        "TEL": m("TEL", "TE Connectivity", ur.USA)})


def _held(ticker, name):
    return {"holdings": {"long": [{"ticker": ticker, "name": name, "strategy": "Quality", "entry_price": 100.0,
                                   "shares": 10}],
                         "closed": [{"ticker": ticker, "exit_price": 110.0}]},
            "journal": [{"ticker": ticker, "pnl_pct": 5.0}],
            "copilot_journal": {"entries": [{"ticker": ticker, "note": "x"}]}}


def _mine(s) -> tuple:
    return len(s["positions"]), len(s["closed"]), len(s["journal"]), len(s["copilot"])


def test_te_connectivity_holdings_never_show_on_telenor_and_back():
    te = _held("TEL", "TE Connectivity")
    assert _mine(td.sheets_for("TEL.OL", te)) == (0, 0, 0, 0)                     # bar 'TEL' = USA
    assert _mine(td.sheets_for("TEL", te)) == (1, 1, 1, 1)
    telenor = _held("TEL.OL", "Telenor")
    assert _mine(td.sheets_for("TEL", telenor)) == (0, 0, 0, 0)
    assert _mine(td.sheets_for("tel.ol", telenor)) == (1, 1, 1, 1)
    assert _mine(td.sheets_for("BOL.ST", _held("BOL", "x"))) == (0, 0, 0, 0)      # bar rad: USA-ticker i panelen
    assert _mine(td.sheets_for("ERIC-B.ST", _held("ERIC B", "Ericsson"))) == (1, 1, 1, 1)   # Börsdatas form: nordisk
    assert td.held_forms("ERIC-B.ST") == ["ERIC-B.ST", "ERIC B"] and td.held_forms("TEL.OL") == ["TEL.OL"]
    assert td.held_forms("EQX.TO") == ["EQX.TO"] and td.held_forms("tel") == ["TEL"] and td.held_forms("") == []


def test_borsdata_form_sheet_rows_still_reach_the_nordic_listing():
    stores = {"producers": {"producers": [{"ticker": "ERIC B", "name": "Ericsson", "commodity": "Koppar", "price": 1.0,
                                           "unit_cost": 1.0}], "royalty": []},
              "confidence": {"companies": {"ERIC B": {"ticker": "ERIC B", "name": "Ericsson", "commodity": "Koppar"}}},
              "scorecard": {"cards": {"ericsson|ERIC B": {"strategy": "producenter"}}}}
    for smap in (None, _universe().sheet_map()):                                  # med och utan registrets karta
        s = td.sheets_for("ERIC-B.ST", stores, smap)
        assert set(s["reviews"]) == {"rule"} and s["confidence"]["name"] == "Ericsson"
        assert list(s["scorecard"]) == ["ericsson|ERIC B"] and s["found"]
    bol = {"scorecard": {"cards": {"boliden|BOL": {"strategy": "producenter"}}}}
    assert list(td.sheets_for("BOL.ST", bol, _universe().sheet_map())["scorecard"]) == ["boliden|BOL"]
    assert td.sheets_for("NEM", stores)["scorecard"] == {}


def test_the_register_keeps_te_connectivitys_sheet_rows_off_telenor():
    stores = {"confidence": {"companies": {"TEL": {"ticker": "TEL", "name": "TE Connectivity"}}},
              "scorecard": {"cards": {"te connectivity|TEL": {"strategy": "Quality"}}}}
    smap = _universe().sheet_map()
    s = td.sheets_for("TEL.OL", stores, smap)
    assert s["confidence"] is None and s["scorecard"] == {} and not s["found"] and s["sheet_forms"] == ["TEL.OL"]
    us = td.sheets_for("TEL", stores, smap)
    assert us["confidence"]["name"] == "TE Connectivity" and list(us["scorecard"]) == ["te connectivity|TEL"]
    # utan registret går arkets bara 'TEL' inte att skilja från Telenors Börsdata-form — som förut
    assert td.sheets_for("TEL.OL", stores)["confidence"]["name"] == "TE Connectivity"
    assert td.sheet_forms("TEL.OL", smap) == ["TEL.OL"] and td.sheet_forms("BOL.ST", smap) == ["BOL.ST", "BOL"]
    assert td.sheet_forms("ERIC-B.ST", smap) == ["ERIC-B.ST", "ERIC-B", "ERIC B"]
    assert td.sheet_forms("EQX.TO") == ["EQX.TO"] and td.sheet_forms("eqx") == ["EQX"]


def test_company_lookup_follows_the_same_forms():
    from thor import arken
    store = {"companies": {"TEL": {"ticker": "TEL", "name": "TE Connectivity", "commodity": "x"},
                           "EQX": {"ticker": "EQX", "name": "Equinox", "commodity": "guld"}}}
    assert arken.company_for(store, "EQX.TO") is None and arken.company_for(store, "EQX").name == "Equinox"
    assert arken.company_for(store, "TEL.OL", ["TEL.OL"]) is None                 # dossierns former (med registret)
    assert arken.company_for(store, "TEL").name == "TE Connectivity"


def test_page_shows_innehav_only_on_the_right_listing(monkeypatch):
    from streamlit.testing.v1 import AppTest
    tu._patch(monkeypatch)
    monkeypatch.setattr(tui, "_stores", lambda: _held("TEL", "TE Connectivity"))
    for ticker, held in (("TEL.OL", False), ("TEL", True)):
        at = AppTest.from_function(tu._script)
        at.query_params["t"], at.query_params["s"] = ticker, "berserk"       # Regim-fliken utan Viking Nines riskmotor
        at.run(timeout=120)
        assert not at.exception, at.exception
        head = next(m.value for m in at.markdown if "inte i någon strategis lista" in m.value or "finns i dag i" in m.value)
        assert (">INNEHAV<" in head) is held, ticker
