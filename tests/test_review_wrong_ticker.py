"""
review — ett svar för fel bolag i importen: EN ruta per bolag och ett tryck till rätt kort.

Ägarens fall: hela svaret bar en annan ticker än kortet. Förr en stor gul lista ('114 rader kunde inte läsas in:'
och en rad per avvisad rad); nu en ruta per bolag — vad som hände och de två vanliga orsakerna (fel kort, eller
ChatGPT svarade om fel bolag). Finns bolaget i ett ark svaret pekar på: "Öppna XXX och läs in där" — svaret följer
med (rv_imp_carry) och arken öppnar på XXX (review.focus). Där granskas svaret som vanligt; inget skrivs förrän
"Skriv in", inget sparas till GitHub, och ingenting händer när sidan bara öppnas. importer.check:s res.issues är
orörda — rutan bygger på rena hjälpare (other_tickers, split_issues, answer_sheets, routing.find_company,
routing.same_listing, routing.unsure_company). Utan registrets karta flyttas inget till en bar rad som kan vara ett
annat bolag ('TEL' för TEL.OL), och fokus låses på raden svaret flyttades till.

Påhittade bolag (AAA.TO, BBB.V, NEM, EQX …), ingen riktig data, inget nätverk.
"""
import copy
import json
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import durrett_cases as dcs  # noqa: E402
import storage  # noqa: E402
from review import answers as an  # noqa: E402
from review import chatgpt_ui  # noqa: E402
from review import export as ex  # noqa: E402
from review import focus  # noqa: E402
from review import importer as im  # noqa: E402
from review import routing as rt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAD = ex.ANSWER_HEADER
CARRY = chatgpt_ui.CARRY


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


# ── De rena hjälparna ────────────────────────────────────────────────────────
def _rule_exp(ticker="BOL.ST"):
    row = {"id": "p1", "ticker": ticker, "name": "Exempel", "commodity": "Koppar"}
    return ex.build(ticker, {"producers": {"producers": [row], "royalty": []}}, ["rule"], rows={"rule": row})


def _read(*lines):
    return an.read_text("\n".join([HEAD] + list(lines)))


def test_other_tickers_groups_rows_per_company_and_counts_blank_rows_once():
    exp = _rule_exp("BOL.ST")
    answers = _read("NEM;rule:mine_life;9;år;x;primary;;high;",
                    "NEM;rule:unit_cost;;;;;;;",                       # tom rad för NEM — samma grupp
                    "nem;rule:rp_ratio;;;;;;;",                        # samma ticker i gemener
                    "BOL;rule:mine_life;9;år;x;primary;;high;",        # kortets bolag i Börsdatas form
                    ";rule:nav_musd;5;MUSD;x;;;high;",                 # tom ticker = kortets
                    "EQX;rule:mine_life;9;år;x;primary;;high;",
                    "EQX.TO;rule:mine_life;9;år;x;primary;;high;")     # en annan notering än EQX
    assert im.other_tickers(answers, exp) == {"NEM": 3, "EQX": 1, "EQX.TO": 1}
    assert list(im.other_tickers(answers, exp)) == ["NEM", "EQX", "EQX.TO"]           # flest först
    # ett bolag i två former är en grupp (kortet är ett annat bolag)
    two = _read("ABC;rule:mine_life;9;år;x;;;high;", "ABC.ST;rule:unit_cost;1;USD/oz;x;;;high;")
    assert im.other_tickers(two, _rule_exp("NEM")) == {"ABC": 2}
    assert im.other_tickers(_read("BOL.ST;rule:mine_life;9;år;x;;;high;"), exp) == {}


def test_split_issues_leaves_check_issues_untouched():
    """check:s res.issues är exakt som förr — rutan delar bara upp dem för visningen."""
    exp = _rule_exp("BOL.ST")
    answers = _read("NEM;rule:mine_life;9;år;x;primary;;high;",
                    "BOL.ST;rule:hittepa;9;;x;;;high;",
                    "NEM;rule:unit_cost;2;USD/oz;x;primary;;high;",
                    "BOL.ST;rule:jurisdiktion;kanske;;x;;;high;")
    res = im.check(answers, exp)
    before = copy.deepcopy(res.issues)
    other, rest = im.split_issues(res.issues)
    assert res.issues == before                                             # orörda
    assert [t for _l, _k, t in other] == ["annan ticker (NEM) — underlaget gäller BOL.ST"] * 2
    assert [k for _l, k, _t in rest] == ["rule:hittepa", "rule:jurisdiktion"]
    assert sorted(other + rest) == sorted(res.issues)
    assert im.other_tickers(answers, exp) == {"NEM": 2}
    assert im.split_issues([]) == ([], [])


def test_only_blank_rows_of_another_company_are_one_group():
    """Fel fil: underlaget för ett annat bolag, helt tomt. check lägger det i ETT skäl ('… N tomma rader för XXX
    räknas inte'); rutan räknar samma rader en gång."""
    exp = _rule_exp("BOL.ST")
    answers = _read("NEM;rule:mine_life;;;;;;;", "NEM;rule:unit_cost;;;;;;;", "NEM;rule:rp_ratio;;;;;;;")
    res = im.check(answers, exp)
    other, rest = im.split_issues(res.issues)
    assert len(other) == 1 and "3 tomma rader för NEM" in other[0][2] and rest == []
    assert im.other_tickers(answers, exp) == {"NEM": 3}


def test_answer_sheets_follow_the_field_key():
    exp = _rule_exp("BOL.ST")
    answers = _read("NEM;confidence:fields.aisc;;;;;;;", "NEM;asymmetry:fields.cash_musd;;;;;;;",
                    "NEM;rule:mine_life;9;år;x;;;high;", "NEM;mine_life;9;år;x;;;high;",   # utan prefix: underlaget
                    "NEM;hittepa:x;1;;;;;;", "EQX;tiggre:nav;1;;;;;;")
    assert im.answer_sheets(answers, exp, "NEM") == ["confidence", "rule"]
    assert im.answer_sheets(answers, exp, "EQX") == ["tiggre"]
    assert im.answer_sheets(answers, exp, "ZZZ") == []


def test_same_company_uses_the_sheet_forms():
    assert im.same_company("BOL", "BOL.ST") and im.same_company("bol.st", "BOL")
    assert not im.same_company("EQX", "EQX.TO") and not im.same_company("AAA", "AAA.TO")
    assert not im.same_company("", "NEM") and im.same_company("NEM", "nem")


def test_find_company_looks_in_the_given_sheets_only():
    data = _durrett_data()
    stores = {"confidence": data,
              "producers": {"producers": [{"id": "p1", "ticker": "BOL.ST", "name": "Exempel"}], "royalty": []},
              "scoring": {"sprott": [], "durrett": []}, "tiggre": {"candidates": []}, "insider": {"signals": []},
              "scorecard": {"cards": {}}}
    assert rt.find_company("BBB.V", ["confidence"], stores) == ("confidence", "BBB.V")
    assert rt.find_company("bbb.v", ["asymmetry"], stores) == ("confidence", "BBB.V")    # samma lager
    assert rt.find_company("BBB", ["confidence"], stores) is None                         # bar = amerikansk
    smap = {"BOL": "BOL.ST"}                                                              # registrets karta
    assert rt.find_company("BOL", ["rule"], stores, smap) == ("rule", "BOL.ST")           # Börsdatas form
    assert rt.find_company("BOL", ["rule"], stores) is None                               # utan karta: kanske inte
    assert rt.unsure_company("BOL", ["rule"], stores) == ("rule", "BOL.ST")
    assert rt.unsure_company("BOL", ["rule"], stores, smap) is None
    assert rt.find_company("BOL.ST", ["confidence", "rule"], stores) == ("rule", "BOL.ST")
    assert rt.find_company("BBB.V", ["rule"], stores) is None                             # inte i det arket
    assert rt.find_company("NEM", ["confidence", "rule"], stores) is None
    assert rt.find_company("NEM", ["confidence"], None) is None and rt.find_company("", ["rule"], stores) is None


def test_same_listing_never_guesses_a_bare_row_without_the_registry_map():
    """Utan registrets karta kan en bar 'TEL' vara TE Connectivity och TEL.OL Telenor — åt båda hållen, som
    routing.ambiguous/_suffixed_twin. Med kartan avgör den. Exakt samma ticker är alltid samma."""
    assert not rt.same_listing("TEL", "TEL.OL") and not rt.same_listing("TEL.OL", "TEL")
    assert im.same_company("TEL", "TEL.OL")                                          # formerna möts ändå
    assert rt.same_listing("TEL", "TEL.OL", {"TEL": "TEL.OL"})                       # kartan: TEL är Telenor
    assert not rt.same_listing("TEL", "TEL.OL", {"BBB": "BBB.ST"})                   # kartan: TEL är inte Telenor
    assert not rt.same_listing("BOL", "BOL.ST") and rt.same_listing("BOL", "BOL.ST", {"BOL": "BOL.ST"})
    assert rt.same_listing("bbb.v", "BBB.V") and rt.same_listing("ERIC B", "ERIC-B.ST")
    assert not rt.same_listing("EQX", "EQX.TO") and not rt.same_listing("", "NEM")


def test_find_company_takes_the_exact_row_first_and_never_a_bare_twin_without_the_map():
    """NEM, TEL (en amerikansk rad) och TEL.OL i Durrett-arket: ett svar för TEL.OL går till TEL.OL-raden — med och
    utan karta, i vilken ordning raderna än står. Finns bara 'TEL': utan karta ingen flytt (unsure_company säger att
    raden finns), med kartan avgör den."""
    for order in (("NEM", "TEL", "TEL.OL"), ("NEM", "TEL.OL", "TEL")):
        stores = {"confidence": _durrett_data(order)}
        for smap in (None, {"BBB": "BBB.ST"}):
            assert rt.find_company("TEL.OL", ["confidence"], stores, smap) == ("confidence", "TEL.OL")
            assert rt.find_company("TEL", ["confidence"], stores, smap) == ("confidence", "TEL")
            assert rt.unsure_company("TEL.OL", ["confidence"], stores, smap) is None
    bare = {"confidence": _durrett_data(("NEM", "TEL"))}
    assert rt.find_company("TEL.OL", ["confidence"], bare) is None
    assert rt.unsure_company("TEL.OL", ["confidence"], bare) == ("confidence", "TEL")
    assert rt.find_company("TEL.OL", ["confidence"], bare, {"BBB": "BBB.ST"}) is None   # TEL är ett annat bolag
    assert rt.unsure_company("TEL.OL", ["confidence"], bare, {"BBB": "BBB.ST"}) is None
    assert rt.find_company("TEL.OL", ["confidence"], bare, {"TEL": "TEL.OL"}) == ("confidence", "TEL")
    suffixed = {"confidence": _durrett_data(("NEM", "TEL.OL"))}                          # omvänt: svaret bar 'TEL'
    assert rt.find_company("TEL", ["confidence"], suffixed) is None
    assert rt.unsure_company("TEL", ["confidence"], suffixed) == ("confidence", "TEL.OL")


# ── Durrett-arket under AppTest: två påhittade bolag ─────────────────────────
def _durrett_data(tickers=("AAA.TO", "BBB.V")) -> dict:
    from confidence import store as cs
    data = cs.default()
    for t in tickers:
        c = dcs.gold_producer()
        c.ticker, c.name = t, {"AAA.TO": "Exempel Guld A", "BBB.V": "Exempel Guld B"}.get(t, f"Exempel {t}")
        cs.put(data, c)
    return data


BBB_ANSWER = "\n".join([HEAD,
                        "BBB.V;confidence:fields.debt_maturity_year;2029;år;Årsredovisning 2025 s. 61;primary;"
                        "2026-03-01;medium;",
                        "BBB.V;confidence:fields.aisc;;;;;;;",
                        "BBB.V;confidence:fields.commodity_price;;;;;;;"])


@pytest.fixture
def durrett(monkeypatch):
    """Durrett-arket (ARKEN) med AAA.TO och BBB.V — och 💾 får aldrig anropas."""
    return _durrett_page(monkeypatch, {"confidence": _durrett_data()})


def _durrett_page(monkeypatch, stores: dict):
    """run(state) → Durrett-arket under AppTest med lagren stores (standard: fokus på AAA.TO)."""
    import refresh_ui
    import screens_ui
    import gist_storage
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)

    def _no_save(*a, **k):
        raise AssertionError("importen får aldrig spara till GitHub")
    monkeypatch.setattr(storage, "save_json", _no_save)
    monkeypatch.setattr(storage, "save_session", _no_save)
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
    monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)
    monkeypatch.setenv("WT_TEST_ROOT", ROOT)

    def run(state: dict = None):
        from streamlit.testing.v1 import AppTest

        def app():
            import os as _o
            import sys as _s
            _s.path.insert(0, _o.environ["WT_TEST_ROOT"])
            from engines.durrett.ui import render_durrett_page
            render_durrett_page()

        at = AppTest.from_function(app, default_timeout=90)
        for k, v in (state or {"arken_focus": "AAA.TO"}).items():
            at.session_state[k] = v
        at.run()
        assert not at.exception, at.exception
        return at
    return run


def _state(at) -> dict:
    return at.session_state.filtered_state


def _company(at, t) -> dict:
    return at.session_state["confidence"]["companies"][t]


def _underlag(at) -> list:
    return [e for e in at.expander if e.label == chatgpt_ui.TITLE]


def _warns(at) -> str:
    return "\n".join(w.value for w in at.warning)


def _caps(at) -> str:
    return "\n".join(c.value for c in at.caption)


def _keys(at) -> set:
    return {b.key for b in at.button}


def test_an_answer_for_another_company_is_one_box_not_a_row_per_row(durrett):
    at = durrett()
    before = copy.deepcopy(at.session_state["confidence"])
    at.text_area(key="rv_imp_txt_durrett_AAA.TO_0").set_value(BBB_ANSWER).run()
    assert not at.exception, at.exception
    boxes = [w.value for w in at.warning if "annan ticker" in w.value]
    assert len(boxes) == 1
    box = boxes[0]
    assert box.startswith("**Svaret gäller BBB.V — det här kortet är AAA.TO.**")
    assert "3 rader har annan ticker (BBB.V) och lästes inte in — inget skrivs in i AAA.TO från dem." in box
    assert "Läste du in i fel kort?" in box and "Öppna BBB.V och läs in där" in box
    gpt = "Svarade ChatGPT om fel bolag (t.ex. i en gammal chatt)? Be om ett nytt svar för AAA.TO i en ny chatt."
    assert gpt in box
    assert "rad 2 ·" not in _warns(at) and "kunde inte läsas in:" not in _warns(at)     # ingen rad-för-rad-lista
    assert "3 rader i svaret. Inget nytt att skriva in · 3 med annan ticker." in _caps(at)
    assert not at.info                                                                   # rutan säger allt
    btn = at.button(key="rv_imp_move_durrett_AAA.TO")
    assert btn.label == "Öppna BBB.V och läs in där"
    # att rita rutan gör ingenting: inget flyttat svar, samma fokus, inget skrivet
    assert CARRY not in _state(at) and at.session_state[focus.FOCUS] == "AAA.TO"
    assert _same(at.session_state["confidence"], before)
    assert not _underlag(at)[0].proto.expanded                                           # som förr


def test_move_opens_the_right_card_and_writes_only_after_skriv_in(durrett):
    at = durrett()
    before = copy.deepcopy(at.session_state["confidence"])
    at.text_area(key="rv_imp_txt_durrett_AAA.TO_0").set_value(BBB_ANSWER).run()
    at.button(key="rv_imp_move_durrett_AAA.TO").click().run()
    assert not at.exception, at.exception
    carry = at.session_state[CARRY]
    assert carry["ticker"] == "BBB.V" and carry["from_ticker"] == "AAA.TO" and carry["text"] == BBB_ANSWER
    assert carry["data"] is None and carry["sheet"] == "confidence"
    assert at.session_state[focus.FOCUS] == "BBB.V"                                     # arket öppnar på BBB.V
    assert _same(at.session_state["confidence"], before)                                # flytten skriver inget
    (exp_,) = _underlag(at)                                                              # bara BBB.V-kortet
    assert exp_.proto.expanded                                                           # underlaget öppet
    assert "Svaret du läste in i AAA.TO-kortet — inläst här." in _caps(at)
    assert "rv_imp_drop_durrett_BBB.V" in _keys(at)
    assert "annan ticker" not in _warns(at)
    go = at.button(key="rv_imp_go_durrett_BBB.V")
    assert go.label == "Skriv in 1 ändring"
    assert at.text_area(key="rv_imp_txt_durrett_BBB.V_0").value in ("", None)          # kortets egen ruta är tom
    go.click().run()
    assert not at.exception, at.exception
    f = _company(at, "BBB.V")["fields"]
    assert f["debt_maturity_year"]["value"] == 2029
    assert f["debt_maturity_year"]["source"] == "ChatGPT: Årsredovisning 2025 s. 61"
    assert _same(_company(at, "AAA.TO"), before["companies"]["AAA.TO"])                  # fel kort rörs aldrig
    assert CARRY not in _state(at)                                                       # inläst — släppt
    assert any("1 ändring inskriven i Durrett 10-steg" in s.value for s in at.success)
    (exp_,) = _underlag(at)
    assert exp_.proto.expanded                                                           # kvittot syns kvar
    at.button(key="rv_imp_close_durrett_BBB.V").click().run()
    assert not at.exception, at.exception
    assert not _underlag(at)[0].proto.expanded                                           # som förr


def test_slapp_drops_the_moved_answer_and_nothing_is_written(durrett):
    at = durrett()
    before = copy.deepcopy(at.session_state["confidence"])
    at.text_area(key="rv_imp_txt_durrett_AAA.TO_0").set_value(BBB_ANSWER).run()
    at.button(key="rv_imp_move_durrett_AAA.TO").click().run()
    at.button(key="rv_imp_drop_durrett_BBB.V").click().run()
    assert not at.exception, at.exception
    assert CARRY not in _state(at) and "rv_imp_go_durrett_BBB.V" not in _keys(at)
    assert "inläst här" not in _caps(at) and not _underlag(at)[0].proto.expanded
    assert _same(at.session_state["confidence"], before)


def test_pasting_your_own_answer_in_the_card_drops_the_moved_one(durrett):
    at = durrett()
    at.text_area(key="rv_imp_txt_durrett_AAA.TO_0").set_value(BBB_ANSWER).run()
    at.button(key="rv_imp_move_durrett_AAA.TO").click().run()
    own = "\n".join([HEAD, "BBB.V;confidence:fields.debt_maturity_year;2031;år;PR 2026;primary;2026-09-01;high;"])
    at.text_area(key="rv_imp_txt_durrett_BBB.V_0").set_value(own).run()
    assert not at.exception, at.exception
    assert CARRY not in _state(at) and "inläst här" not in _caps(at)
    table = at.dataframe[0].value
    assert list(table["Nytt"]) == ["2031"]                                               # ditt svar gäller
    # att tömma rutan igen släpper inget nytt — och tar inte tillbaka det flyttade
    at.text_area(key="rv_imp_txt_durrett_BBB.V_0").set_value("").run()
    assert CARRY not in _state(at) and "rv_imp_go_durrett_BBB.V" not in _keys(at)


def test_a_moved_answer_wins_over_a_stale_answer_in_the_card_and_says_so(durrett):
    """BBB.V-kortet hade ett äldre, oinläst svar i rutan. Flytten är det senaste du gjorde: det flyttade svaret
    läses in, och kortet säger att texten i rutan inte används medan det ligger där."""
    at = durrett({"arken_focus": ""})                                                    # Visa alla: båda korten
    stale = "\n".join([HEAD, "BBB.V;confidence:fields.debt_maturity_year;2035;år;Gammal;primary;2020-01-01;high;"])
    at.text_area(key="rv_imp_txt_durrett_BBB.V_0").set_value(stale).run()
    at.text_area(key="rv_imp_txt_durrett_AAA.TO_0").set_value(BBB_ANSWER).run()
    at.button(key="rv_imp_move_durrett_AAA.TO").click().run()
    assert not at.exception, at.exception
    assert at.session_state[focus.FOCUS] == "BBB.V"
    assert "Svaret du läste in i AAA.TO-kortet — inläst här. Filen eller texten ovan används inte" in _caps(at)
    assert list(at.dataframe[0].value["Nytt"]) == ["2029"]


def test_a_company_not_in_the_sheets_gets_no_button_but_a_hint(durrett):
    at = durrett()
    answer = "\n".join([HEAD, "NEM;confidence:fields.debt_maturity_year;2029;år;x;primary;2026-03-01;medium;"])
    at.text_area(key="rv_imp_txt_durrett_AAA.TO_0").set_value(answer).run()
    assert not at.exception, at.exception
    box = _warns(at)
    assert "**Svaret gäller NEM — det här kortet är AAA.TO.**" in box
    assert "1 rad har annan ticker (NEM) och lästes inte in — inget skrivs in i AAA.TO från den." in box
    assert "NEM finns inte i arken än — lägg in bolaget först" in box
    assert not [k for k in _keys(at) if str(k).startswith("rv_imp_move_")]
    assert CARRY not in _state(at)


def test_several_other_companies_are_at_most_three_boxes_then_plus_n(durrett):
    at = durrett()
    rows = [f"{t};confidence:fields.aisc;;;;;;;" for t in ("NEM", "NEM", "EQX", "EQX", "BBB.V", "CCC.TO", "DDD.V")]
    at.text_area(key="rv_imp_txt_durrett_AAA.TO_0").set_value("\n".join([HEAD] + rows)).run()
    assert not at.exception, at.exception
    boxes = [w.value for w in at.warning if "annan ticker" in w.value]
    assert len(boxes) == 3
    heads = [b.split("**")[1].split(" —")[0] for b in boxes]
    assert heads == ["En del av svaret gäller NEM", "En del av svaret gäller EQX", "En del av svaret gäller BBB.V"]
    assert "+2 bolag till i svaret, inte heller inlästa: CCC.TO (1 rad) · DDD.V (1 rad)" in _caps(at)
    moves = {b.key: b.label for b in at.button if str(b.key).startswith("rv_imp_move_")}
    assert moves == {"rv_imp_move_durrett_AAA.TO_2": "Öppna BBB.V och läs in där"}       # bara BBB.V finns i arken


def test_an_uploaded_file_moves_with_its_bytes(durrett):
    at = durrett()
    data = BBB_ANSWER.encode("utf-8-sig")
    at.file_uploader(key="rv_imp_file_durrett_AAA.TO_0").set_value(("svar_BBB.csv", data, "text/csv")).run()
    assert not at.exception, at.exception
    at.button(key="rv_imp_move_durrett_AAA.TO").click().run()
    assert not at.exception, at.exception
    carry = at.session_state[CARRY]
    assert carry["data"] == data and carry["file"] == "svar_BBB.csv" and carry["text"] is None
    assert at.button(key="rv_imp_go_durrett_BBB.V").label == "Skriv in 1 ändring"


def test_nothing_happens_on_page_open_even_with_a_moved_answer_waiting(durrett):
    """En sida som bara öppnas skriver inget, flyttar inget och byter inte fokus — inte heller när ett flyttat svar
    väntar (det läses först när du tryckt Skriv in i kortet)."""
    at = durrett()
    assert CARRY not in _state(at) and at.session_state[focus.FOCUS] == "AAA.TO"
    carry = {"ticker": "BBB.V", "sheet": "confidence", "from_ticker": "AAA.TO", "from_key": "durrett_AAA.TO",
             "data": None, "file": None, "text": BBB_ANSWER}
    at2 = durrett({"arken_focus": "BBB.V", CARRY: carry})
    before = copy.deepcopy(at2.session_state["confidence"])
    at2.run()
    assert not at2.exception, at2.exception
    assert _same(at2.session_state["confidence"], before) and at2.session_state[CARRY] == carry
    assert at2.session_state[focus.FOCUS] == "BBB.V"
    # ett annat bolags kort öppnar inte sitt underlag för ett svar som inte gäller det
    at3 = durrett({"arken_focus": "AAA.TO", CARRY: carry})
    assert not _underlag(at3)[0].proto.expanded and "inläst här" not in _caps(at3)


# ── Ett blandat svar i Rick Rule: rutan för NEM, resten som förr ─────────────
STORES = {
    "producers": {"producers": [{"id": "p1", "ticker": "BOL.ST", "name": "Exempel", "commodity": "Guld",
                                 "date": "2026-09-01", "price": 3000.0},
                                {"id": "p2", "ticker": "NEM", "name": "Exempel N", "commodity": "Guld",
                                 "date": "2026-09-01"}],
                  "royalty": []},
    "scoring": {"sprott": [], "durrett": []},
    "insider": {"signals": []},
    "tiggre": {"candidates": [], "positions": [], "closed": [], "parked": []},
    "holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0},
    "scorecard": {"cards": {}},
    "confidence": {"companies": {}, "commodity_overrides": {}},
}
MIXED = "\n".join([HEAD,
                   "BOL.ST;rule:unit_cost;1 250,5;USD/oz;Q2-rapporten s. 12;primary;2026-08-01;high;",
                   "BOL.ST;rule:hittepa;9;;x;;;high;",
                   "NEM;rule:ev_musd;5;MUSD;x;primary;2026-01-01;high;"])


@pytest.fixture
def rule_page(monkeypatch):
    """Rick Rule-arket — eller Durrett-arket när session_state['_page'] == 'durrett' (samma session, som att byta
    flik i ARKEN). Durrett-lagret har AAA.TO."""
    stores = copy.deepcopy(STORES)
    stores["confidence"] = {k: v for k, v in _durrett_data().items()}
    stores["confidence"]["companies"].pop("BBB.V", None)
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)

    def _no_save(*a, **k):
        raise AssertionError("importen får aldrig spara till GitHub")
    monkeypatch.setattr(storage, "save_json", _no_save)
    monkeypatch.setattr(storage, "save_session", _no_save)
    import gist_storage
    import refresh_ui
    import screens_ui
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
    monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)
    monkeypatch.setenv("WT_TEST_ROOT", ROOT)

    def run(state: dict = None):
        from streamlit.testing.v1 import AppTest

        def app():
            import os as _o
            import sys as _s
            _s.path.insert(0, _o.environ["WT_TEST_ROOT"])
            import streamlit as _st
            if _st.session_state.get("_page") == "durrett":
                from engines.durrett.ui import render_durrett_page
                render_durrett_page()
                return
            import producers
            producers.render_producers_page(sheet="Rick Rule")

        at = AppTest.from_function(app, default_timeout=90)
        for k, v in (state or {}).items():
            at.session_state[k] = v
        at.run()
        assert not at.exception, at.exception
        return at
    return run


def test_a_mixed_answer_keeps_its_own_rows_and_lists_the_rest_as_before(rule_page):
    at = rule_page({focus.FOCUS: "BOL.ST"})
    at.text_area(key="rv_imp_txt_rule_p1_0").set_value(MIXED).run()
    assert not at.exception, at.exception
    warns = _warns(at)
    assert "annan ticker (NEM)" in warns                                                 # som rundturstestet läser
    assert "**En del av svaret gäller NEM — det här kortet är BOL.ST.**" in warns
    assert "1 rad har annan ticker (NEM) och lästes inte in — inget skrivs in i BOL.ST från den." in warns
    assert "1 rad kunde inte läsas in:\n\nrad 3 · rule:hittepa — okänd fältnyckel" in warns    # övriga som förr
    assert "rad 4 · rule:ev_musd" not in warns
    assert "3 rader i svaret. 1 att skriva in → Rick Rule 1 · 1 med annan ticker · 1 kunde inte läsas in." in _caps(at)
    move = at.button(key="rv_imp_move_rule_p1")
    assert move.label == "Öppna NEM och läs in där"
    assert at.button(key="rv_imp_go_rule_p1").label == "Skriv in 1 ändring"             # BOL.ST-raden som förr
    at.button(key="rv_imp_go_rule_p1").click().run()
    assert not at.exception, at.exception
    rows = at.session_state["producers"]["producers"]
    assert rows[0]["unit_cost"] == 1250.5 and rows[1].get("ev_musd") is None
    assert CARRY not in _state(at)                                                       # inget flyttat


def test_move_from_rick_rule_opens_nem_in_the_same_sheet(rule_page):
    at = rule_page({focus.FOCUS: "BOL.ST"})
    at.text_area(key="rv_imp_txt_rule_p1_0").set_value(MIXED).run()
    at.button(key="rv_imp_move_rule_p1").click().run()
    assert not at.exception, at.exception
    assert at.session_state[focus.FOCUS] == "NEM" and at.selectbox(key="arken_focus_rule").value == "NEM"
    assert "Svaret du läste in i BOL.ST-kortet — inläst här." in _caps(at)
    # i NEM-kortet är BOL.ST-raderna nu det andra bolaget — med en väg tillbaka
    assert "**En del av svaret gäller BOL.ST — det här kortet är NEM.**" in _warns(at)
    assert at.button(key="rv_imp_move_rule_p2").label == "Öppna BOL.ST och läs in där"
    go = at.button(key="rv_imp_go_rule_p2")
    assert go.label == "Skriv in 1 ändring"
    go.click().run()
    assert not at.exception, at.exception
    rows = at.session_state["producers"]["producers"]
    assert rows[1]["ev_musd"] == 5.0 and rows[0].get("unit_cost") is None               # bara NEM-raden
    assert CARRY not in _state(at)


def test_an_answer_for_a_company_in_another_sheet_waits_in_its_card(rule_page):
    """Svaret gäller AAA.TO, som bara finns i Durrett-arket. Rick Rule har inget AAA.TO-kort: fokus följer med,
    Rick Rule visar alla, och kortet du läste in i säger var svaret väntar. I Durrett-arket öppnar AAA.TO-kortet
    med svaret — och skriver först vid Skriv in."""
    at = rule_page({focus.FOCUS: "BOL.ST"})
    answer = "\n".join([HEAD, "AAA.TO;confidence:fields.debt_maturity_year;2029;år;Årsredovisning 2025 s. 61;"
                              "primary;2026-03-01;medium;"])
    at.text_area(key="rv_imp_txt_rule_p1_0").set_value(answer).run()
    assert not at.exception, at.exception
    assert "svaret följer med till AAA.TO-kortet i Durrett 10-steg" in _warns(at)
    at.button(key="rv_imp_move_rule_p1").click().run()
    assert not at.exception, at.exception
    assert at.session_state[focus.FOCUS] == "AAA.TO"
    assert any("AAA.TO finns inte i arket — visar alla bolag." in c.value for c in at.caption)
    # kortet du läste in i syns kvar: rutan säger var svaret väntar och hur du kommer dit — ingen knapp som inte
    # verkar göra något när du trycker igen
    assert ("**Svaret väntar i AAA.TO-kortet (Durrett 10-steg).** Öppna Durrett 10-steg i ARKEN — fokus står redan "
            "på AAA.TO. Inget skrivs in förrän du tryckt Skriv in där.") in _warns(at)
    assert "Tryck **Öppna AAA.TO" not in _warns(at)
    assert not [k for k in _keys(at) if str(k).startswith("rv_imp_move_")]
    before = copy.deepcopy(at.session_state["confidence"])
    at.session_state["_page"] = "durrett"                                               # fliken Durrett
    at.run()
    assert not at.exception, at.exception
    assert _same(at.session_state["confidence"], before)                                # att öppna skriver inget
    (exp_,) = _underlag(at)
    assert exp_.proto.expanded and "Svaret du läste in i BOL.ST-kortet — inläst här." in _caps(at)
    at.button(key="rv_imp_go_durrett_AAA.TO").click().run()
    assert not at.exception, at.exception
    assert _company(at, "AAA.TO")["fields"]["debt_maturity_year"]["value"] == 2029
    assert CARRY not in _state(at)
    assert at.session_state["producers"]["producers"][0].get("debt_maturity_year") is None


def _asym_page(monkeypatch, stores: dict):
    """Wolf Asymmetry i läget Ark, på AAA.TO — 💾 får aldrig anropas."""
    import commodity_prices
    import positions
    import refresh_ui
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)

    def _no_save(*a, **k):
        raise AssertionError("importen får aldrig spara till GitHub")
    monkeypatch.setattr(storage, "save_json", _no_save)
    monkeypatch.setattr(storage, "save_session", _no_save)
    monkeypatch.setattr(positions, "open_positions", lambda strategy=None, bucket=None: [])
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})
    monkeypatch.setattr(commodity_prices, "spot", lambda name: None)
    monkeypatch.setenv("WT_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["WT_TEST_ROOT"])
        from asymmetry.ui import render_asymmetry_page
        render_asymmetry_page()

    at = AppTest.from_function(app, default_timeout=90)
    at.session_state["asym_mode"] = "Ark"
    at.session_state["asym_pick"] = "AAA.TO"
    at.run()
    assert not at.exception, at.exception
    return at


def test_wolf_asymmetry_opens_the_right_company_on_the_tap(monkeypatch):
    """Wolf Asymmetry har ingen fokusrad (bolaget väljs i 'Bolag'). Ett tryck på "Öppna BBB.V och läs in där" byter
    'Bolag' till BBB.V (sidans egen asym_goto): dess underlag öppnar med svaret — och skriver först vid Skriv in."""
    from asymmetry import store as ast
    at = _asym_page(monkeypatch, {"confidence": ast.normalize(_durrett_data())})
    at.text_area(key="rv_imp_txt_asym_AAA.TO_0").set_value(BBB_ANSWER).run()
    assert not at.exception, at.exception
    assert "svaret följer med till BBB.V-kortet i Wolf Asymmetry" in _warns(at)
    before = copy.deepcopy(at.session_state["confidence"])
    at.button(key="rv_imp_move_asym_AAA.TO").click().run()
    assert not at.exception, at.exception
    assert at.session_state[CARRY]["ticker"] == "BBB.V"
    assert at.session_state["asym_pick"] == "BBB.V"                                     # sidan bytte bolag
    assert "asym_goto" not in _state(at)                                                 # förbrukad av sidan
    (exp_,) = _underlag(at)
    assert exp_.proto.expanded and "Svaret du läste in i AAA.TO-kortet — inläst här." in _caps(at)
    assert "annan ticker" not in _warns(at)                                              # AAA.TO-kortet syns inte
    assert "rv_imp_drop_asym_BBB.V" in _keys(at)
    assert _same(at.session_state["confidence"], before)                                # inget skrivet ännu
    go = at.button(key="rv_imp_go_asym_BBB.V")
    assert go.label == "Skriv in 1 ändring"
    go.click().run()
    assert not at.exception, at.exception
    assert _company(at, "BBB.V")["fields"]["debt_maturity_year"]["value"] == 2029
    assert _same(_company(at, "AAA.TO"), before["companies"]["AAA.TO"])
    assert CARRY not in _state(at)


def test_wolf_asymmetry_says_where_an_answer_for_another_sheet_waits(monkeypatch):
    """Svaret gäller NEM, som bara finns i Rick Rule. Wolf Asymmetry kan inte öppna det kortet: efter trycket står
    AAA.TO kvar, och rutan säger var svaret väntar och hur du kommer dit — ingen knapp som ser död ut."""
    from asymmetry import store as ast
    stores = {"confidence": ast.normalize(_durrett_data()),
              "producers": {"producers": [{"id": "p2", "ticker": "NEM", "name": "Exempel N", "commodity": "Guld"}],
                            "royalty": []}}
    at = _asym_page(monkeypatch, stores)
    answer = "\n".join([HEAD, "NEM;rule:ev_musd;5;MUSD;x;primary;2026-01-01;high;"])
    at.text_area(key="rv_imp_txt_asym_AAA.TO_0").set_value(answer).run()
    assert not at.exception, at.exception
    assert "svaret följer med till NEM-kortet i Rick Rule" in _warns(at)
    before = copy.deepcopy(at.session_state["producers"])
    at.button(key="rv_imp_move_asym_AAA.TO").click().run()
    assert not at.exception, at.exception
    assert at.session_state[CARRY]["ticker"] == "NEM" and at.session_state[CARRY]["sheet"] == "rule"
    assert at.session_state["asym_pick"] == "AAA.TO" and at.session_state[focus.FOCUS] == "NEM"
    assert ("**Svaret väntar i NEM-kortet (Rick Rule).** Öppna Rick Rule i ARKEN — fokus står redan på NEM. "
            "Inget skrivs in förrän du tryckt Skriv in där.") in _warns(at)
    assert "rv_imp_move_asym_AAA.TO" not in _keys(at)
    assert _same(at.session_state["producers"], before)


# ── TEL och TEL.OL: aldrig ett annat bolag med samma bara ticker ─────────────
TEL_ANSWER = "\n".join([HEAD, "TEL.OL;confidence:fields.debt_maturity_year;2029;år;Rapport s. 1;primary;2026-03-01;"
                              "medium;"])


def _maturity(at, t):
    return (_company(at, t)["fields"].get("debt_maturity_year") or {}).get("value")


def test_no_move_to_a_bare_row_that_may_be_another_company_without_the_map(monkeypatch):
    """Durrett-arket har NEM och TEL (en amerikansk rad), registrets karta kunde inte läsas. Ett svar för TEL.OL i
    NEM-kortet: ingen "Öppna"-knapp — rutan säger att raden TEL finns men kanske är ett annat bolag. Och ett flyttat
    svar för TEL.OL läses aldrig in i TEL-kortet."""
    run = _durrett_page(monkeypatch, {"confidence": _durrett_data(("NEM", "TEL"))})
    at = run({"arken_focus": "NEM"})
    before = copy.deepcopy(at.session_state["confidence"])
    at.text_area(key="rv_imp_txt_durrett_NEM_0").set_value(TEL_ANSWER).run()
    assert not at.exception, at.exception
    box = _warns(at)
    assert "**Svaret gäller TEL.OL — det här kortet är NEM.**" in box and "annan ticker (TEL.OL)" in box
    assert ("Durrett 10-steg har raden TEL, men registrets tickerkarta kunde inte läsas — det kan vara ett annat "
            "bolag än TEL.OL, så svaret flyttas inte dit.") in box
    assert not [k for k in _keys(at) if str(k).startswith("rv_imp_move_")]
    assert CARRY not in _state(at) and _same(at.session_state["confidence"], before)
    carry = {"ticker": "TEL.OL", "sheet": "confidence", "from_ticker": "NEM", "from_key": "durrett_NEM",
             "data": None, "file": None, "text": TEL_ANSWER}
    at2 = run({"arken_focus": "TEL", CARRY: carry})
    assert not _underlag(at2)[0].proto.expanded and "inläst här" not in _caps(at2)
    assert "rv_imp_go_durrett_TEL" not in _keys(at2) and _maturity(at2, "TEL") is None


@pytest.mark.parametrize("smap", [None, {"BBB": "BBB.ST"}])
def test_move_opens_tel_ol_not_the_bare_tel_row(monkeypatch, smap):
    """NEM, TEL och TEL.OL i Durrett-arket (TEL sorteras före TEL.OL). "Öppna TEL.OL" flyttar svaret till TEL.OL och
    fokus låses på just den raden — med och utan registrets karta. Skriv in skriver bara TEL.OL."""
    run = _durrett_page(monkeypatch, {"confidence": _durrett_data(("NEM", "TEL", "TEL.OL"))})
    at = run({"arken_focus": "NEM", **({"rv_sheet_map": smap} if smap else {})})
    before = copy.deepcopy(at.session_state["confidence"])
    at.text_area(key="rv_imp_txt_durrett_NEM_0").set_value(TEL_ANSWER).run()
    assert not at.exception, at.exception
    assert "svaret följer med till TEL.OL-kortet i Durrett 10-steg" in _warns(at)
    at.button(key="rv_imp_move_durrett_NEM").click().run()
    assert not at.exception, at.exception
    assert at.session_state[CARRY]["ticker"] == "TEL.OL" and at.session_state[focus.FOCUS] == "TEL.OL"
    assert at.session_state[focus.FORMS] == {"ticker": "TEL.OL", "forms": ["TEL.OL"]}
    assert at.selectbox(key="arken_focus_durrett10").value == "TEL.OL"
    (exp_,) = _underlag(at)
    assert exp_.proto.expanded and "Svaret du läste in i NEM-kortet — inläst här." in _caps(at)
    assert "rv_imp_drop_durrett_TEL.OL" in _keys(at)
    assert "rv_imp_go_durrett_TEL" not in _keys(at) and "rv_imp_drop_durrett_TEL" not in _keys(at)
    assert _same(at.session_state["confidence"], before)
    at.button(key="rv_imp_go_durrett_TEL.OL").click().run()
    assert not at.exception, at.exception
    assert _maturity(at, "TEL.OL") == 2029
    assert _same(_company(at, "TEL"), before["companies"]["TEL"])                         # TE Connectivity orörd
    assert _same(_company(at, "NEM"), before["companies"]["NEM"])
    assert CARRY not in _state(at)


# ── Granskningens småsaker: kortet säger vilket det är, ditt eget svar syns, samma symbol, kvittot ────────────
def test_the_moved_answer_names_the_card_now_open(durrett):
    at = durrett()
    at.text_area(key="rv_imp_txt_durrett_AAA.TO_0").set_value(BBB_ANSWER).run()
    at.button(key="rv_imp_move_durrett_AAA.TO").click().run()
    assert not at.exception, at.exception
    assert ("Svaret du läste in i AAA.TO-kortet — inläst här. Det här är BBB.V-kortet — Skriv in skriver hit."
            in _caps(at))


def test_your_own_answer_after_a_move_keeps_the_underlag_open_through_the_receipt(durrett):
    """Du klistrar in ett eget svar i kortet som läste det flyttade: det flyttade släpps, men underlaget fälls inte
    ihop — ditt svar och sedan kvittot syns. Stäng kvittot fäller ihop det som förr."""
    at = durrett()
    at.text_area(key="rv_imp_txt_durrett_AAA.TO_0").set_value(BBB_ANSWER).run()
    at.button(key="rv_imp_move_durrett_AAA.TO").click().run()
    own = "\n".join([HEAD, "BBB.V;confidence:fields.debt_maturity_year;2031;år;PR 2026;primary;2026-09-01;high;"])
    at.text_area(key="rv_imp_txt_durrett_BBB.V_0").set_value(own).run()
    assert not at.exception, at.exception
    assert CARRY not in _state(at) and _underlag(at)[0].proto.expanded
    at.button(key="rv_imp_go_durrett_BBB.V").click().run()
    assert not at.exception, at.exception
    assert _company(at, "BBB.V")["fields"]["debt_maturity_year"]["value"] == 2031
    assert _underlag(at)[0].proto.expanded and any("1 ändring inskriven" in s.value for s in at.success)
    at.button(key="rv_imp_close_durrett_BBB.V").click().run()
    assert not at.exception, at.exception
    assert not _underlag(at)[0].proto.expanded and "rv_imp_keep_durrett_BBB.V" not in _state(at)


def test_same_symbol_without_the_listing_says_change_the_ticker_not_add_the_company(durrett):
    at = durrett()
    before = copy.deepcopy(at.session_state["confidence"])
    answer = "\n".join([HEAD, "AAA;confidence:fields.debt_maturity_year;2029;år;x;primary;2026-03-01;medium;",
                        "AAA;confidence:fields.aisc;;;;;;;"])
    at.text_area(key="rv_imp_txt_durrett_AAA.TO_0").set_value(answer).run()
    assert not at.exception, at.exception
    box = _warns(at)
    assert "annan ticker (AAA)" in box
    assert "ChatGPT skrev AAA i stället för AAA.TO — en annan notering" in box
    assert "ändra Ticker till AAA.TO i svaret" in box and "lägg in bolaget först" not in box
    assert not [k for k in _keys(at) if str(k).startswith("rv_imp_move_")]
    assert _same(at.session_state["confidence"], before)


def test_a_mixed_answer_receipt_says_which_rows_were_not_read(durrett):
    at = durrett()
    answer = "\n".join([HEAD, "AAA.TO;confidence:fields.debt_maturity_year;2030;år;x;primary;2026-03-01;medium;",
                        "BBB.V;confidence:fields.debt_maturity_year;2029;år;y;primary;2026-03-01;medium;"])
    at.text_area(key="rv_imp_txt_durrett_AAA.TO_0").set_value(answer).run()
    assert "**En del av svaret gäller BBB.V — det här kortet är AAA.TO.**" in _warns(at)
    before_b = copy.deepcopy(_company(at, "BBB.V"))
    at.button(key="rv_imp_go_durrett_AAA.TO").click().run()
    assert not at.exception, at.exception
    assert _company(at, "AAA.TO")["fields"]["debt_maturity_year"]["value"] == 2030
    assert _same(_company(at, "BBB.V"), before_b)
    assert "1 rad för BBB.V lästes inte in här — gäller den BBB.V: läs in samma svar i BBB.V-kortet." in _caps(at)
