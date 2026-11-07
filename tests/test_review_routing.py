"""review.routing + filläsningen: ett svar läses in i alla ark bolaget finns i, och en tom eller omdöpt fil ger ett
begripligt besked. Rena funktioner — AppTest-fallen står i test_review_roundtrip_apptest."""
import copy
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from review import answers
from review import export as ex
from review import importer as im
from review import routing as rt

HEAD = "Ticker;Fältnyckel;Värde;Enhet;Källa;Källtyp;Datum;Säkerhet;Kommentar"


def _stores():
    return {
        "producers": {"producers": [{"id": "p1", "ticker": "VAR.OL", "name": "Vår Energi", "commodity": "Olja",
                                     "date": "2026-09-01"}], "royalty": []},
        "scoring": {"sprott": [], "durrett": []},
        "insider": {"signals": [{"id": "s1", "ticker": "VAR.OL", "name": "Vår Energi", "comment": ""}]},
        "tiggre": {"candidates": [], "positions": [], "closed": [], "parked": []},
        "holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [],
                     "tiggre": [{"id": "h1", "ticker": "LOB", "name": "Lobo", "strategy": "Tiggre"}],
                     "closed": [], "cash": 0},
        "scorecard": {"cards": {}},
        "confidence": {"companies": {}, "commodity_overrides": {}},
    }


def _card(stores, sheet="insider"):
    """Insider-kortets underlag, byggt som kortet bygger det (bara sitt eget lager)."""
    sig = stores["insider"]["signals"][0]
    exp = ex.build("VAR.OL", {"insider": stores["insider"]}, [sheet], rows={sheet: sig})
    return exp, {sheet: sig}


def test_plan_routes_other_sheets_to_the_live_rows():
    stores = _stores()
    exp, targets = _card(stores)
    p = rt.plan(exp, targets, (["s1"], ()), stores)
    assert p.targets["insider"] is stores["insider"]["signals"][0]
    assert p.targets["rule"] is stores["producers"]["producers"][0]          # samma objekt — inte en kopia
    assert p.writable == {"insider", "rule", "scorecard"}                    # kandidaten finns → kortet skapas vid skrivning
    assert callable(p.targets["scorecard"]) and stores["scorecard"]["cards"] == {}
    assert p.reset["rule"] == (["p1"], ()) and p.reset["insider"] == (("s1",), ())
    assert "rule:unit_cost" in p.exp.fields and "tiggre:screen.fs" in p.exp.fields   # tom mall där bolaget saknas
    assert p.exp.source_rows["tiggre"] is None


def test_without_stores_the_card_keeps_its_own_sheet_only():
    stores = _stores()
    exp, targets = _card(stores)
    p = rt.plan(exp, targets, (), None)
    assert set(p.targets) == {"insider"} and p.exp is exp


def test_check_routes_and_explains_every_row():
    stores = _stores()
    exp, targets = _card(stores)
    p = rt.plan(exp, targets, (), stores)
    text = "\n".join([HEAD, "VAR.OL;rule:unit_cost;30;USD/boe;Q3;primary;2026-10-01;high;",
                      "VAR.OL;insider:insiders;2;;Insyn;primary;2026-10-02;high;",
                      "VAR.OL;tiggre:screen.fs;Ja;;DFS;primary;2026-01-01;high;",
                      "VAR.OL;sprott:ds_runway;2;;;;;high;",
                      "VAR.OL;rule:mine_life;;;;;;;",
                      "VAR.OL;rule:nonsense;1;;x;primary;2026-01-01;high;"])
    res = im.check(answers.read_text(text), p.exp, p.writable, p.blocked)
    assert {c.key for c in res.changes} == {"rule:unit_cost", "insider:insiders"}
    why = {k: t for _l, k, t in res.issues}
    assert why["tiggre:screen.fs"].startswith("bolaget finns inte i Tiggre")
    assert "rule:nonsense" in why and why["rule:nonsense"] == "okänd fältnyckel"
    assert not any("läs in svaret där" in t for _l, _k, t in res.issues + res.ignored)
    assert [f.key for f in res.blank] == ["rule:mine_life"]                  # ett ChatGPT-fält som står tomt
    n = im.apply(res.changes, rt.resolved_targets(p, ["rule", "insider"]))
    assert n == 2 and stores["producers"]["producers"][0]["unit_cost"] == 30.0
    assert stores["insider"]["signals"][0]["insiders"] == 2


def test_a_tiggre_position_in_the_register_is_not_written():
    """Tiggres position i registret är en vy — svaret skrivs inte dit, och skälet står i importen."""
    stores = _stores()
    stores["insider"]["signals"][0]["ticker"] = "LOB"
    sig = stores["insider"]["signals"][0]
    exp = ex.build("LOB", {"insider": stores["insider"]}, ["insider"], rows={"insider": sig})
    p = rt.plan(exp, {"insider": sig}, (), stores)
    assert "tiggre" in p.blocked and "position i registret" in p.blocked["tiggre"]
    assert "tiggre" not in p.writable


def test_load_failure_blocks_the_sheet():
    stores = _stores()
    exp, targets = _card(stores)
    p = rt.plan(exp, targets, (), stores, lambda name: "HTTP 500" if name == "producers" else None)
    assert "rule" in p.blocked and "rule" not in p.writable and "insider" in p.writable


def test_durrett_and_wolf_asymmetry_share_the_company_row():
    """asymmetry:fields.x och confidence:fields.x är samma värde i samma bolagsrad — en ändring, inte två."""
    from confidence import store as cs
    import durrett_cases as dcs
    data = cs.default()
    company = dcs.gold_producer()
    cs.put(data, company)
    t = company.ticker
    stores = _stores()
    stores["confidence"] = data
    row = data["companies"][t]
    exp = ex.build(t, {cs.STORE: data}, ["asymmetry"], rows={"asymmetry": company.as_dict()})
    p = rt.plan(exp, {"asymmetry": lambda: data["companies"][t]}, ((), (f"cf_{t}_",)), stores)
    assert p.targets["confidence"] is row
    text = "\n".join([HEAD, f"{t};asymmetry:fields.aisc;1 300;USD/oz;A;primary;2026-08-01;high;",
                      f"{t};confidence:fields.aisc;1 310;USD/oz;B;primary;2026-08-02;high;"])
    res = im.check(answers.read_text(text), p.exp, p.writable, p.blocked)
    assert len(res.changes) == 1 and res.changes[0].new == 1310.0
    assert any("dubbel rad" in t for _l, _k, t in res.ignored)


def test_missing_now_counts_what_is_still_missing():
    stores = _stores()
    rows = {"rule": stores["producers"]["producers"][0]}
    exp, _t = _card(stores)
    out = rt.missing_now(exp, rows, stores)
    assert set(out) == {"rule"} and out["rule"]


# ── Filen ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("data", [b"", b"  \n"])
def test_an_empty_file_says_so(data):
    with pytest.raises(answers.ImportFormatError) as exc:
        answers.read_file(data, "underlag_VAR.OL_2026-11-06_ifylld.xlsx")
    assert "är tom (0 byte)" in str(exc.value) and "Ladda ner arket från ChatGPT igen" in str(exc.value)


def test_xlsx_with_a_renamed_sheet_and_instruktion_first_is_read():
    """ChatGPT döper ibland om 'Fyll i' och lägger Instruktion först — fliken med Fältnyckel hittas ändå."""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    help_ws = wb.active
    help_ws.title = "Instruktion"
    help_ws["A1"] = "Fyll i bara de blå kolumnerna."
    ws = wb.create_sheet("Sheet1")
    ws.append(["Ticker", "Fältnyckel", "Fyll i: Värde", "Fyll i: Källa (URL/dokument+sida)"])
    ws.append(["VAR.OL", "rule:unit_cost", 30, "Q3"])
    wb.active = 0
    buf = io.BytesIO()
    wb.save(buf)
    got = answers.read_file(buf.getvalue(), "svar.xlsx")
    assert [(a.key, a.value, a.source) for a in got] == [("rule:unit_cost", 30, "Q3")]


def test_the_exported_xlsx_still_round_trips():
    pytest.importorskip("openpyxl")
    stores = _stores()
    exp, _t = _card(stores)
    data = ex.to_xlsx(exp.rows, "prompt")
    got = answers.read_file(data, "underlag.xlsx")
    assert {a.key for a in got} >= {"insider:insiders"}
    assert copy.deepcopy(got)                                                   # rena dataobjekt


# ── Granskningens fynd ───────────────────────────────────────────────────────
def test_the_row_compared_is_the_row_written_with_two_signals_for_one_ticker():
    """Fynd 1/6: Scorecardet läser arkets senaste signal — jämförelsen och skrivningen gäller samma rad, inte den
    första signalen med samma ticker."""
    stores = _stores()
    old = {"id": "s_old", "ticker": "VAR.OL", "name": "Vår Energi", "insiders": 2, "comment": ""}
    new = {"id": "s_new", "ticker": "VAR.OL", "name": "Vår Energi", "comment": ""}
    stores["insider"]["signals"] = [old, new]
    card = {"strategy": "insider"}
    exp = ex.build("VAR.OL", stores, ["insider", "scorecard"], rows={"insider": new, "scorecard": card})
    p = rt.plan(exp, {"scorecard": lambda: card}, (), stores)
    assert p.targets["insider"] is new and p.exp.source_rows["insider"] is new
    res = im.check(answers.read_text("\n".join([HEAD, "VAR.OL;insider:insiders;3;;Insyn;primary;2026-10-02;high;"])),
                   p.exp, p.writable, p.blocked)
    (c,) = res.changes
    assert c.old is None and c.default_use
    im.apply(res.changes, rt.resolved_targets(p, ["insider"]))
    assert new["insiders"] == 3 and old["insiders"] == 2


def test_without_the_registry_map_a_bare_row_of_maybe_another_company_is_not_written():
    """Fynd 2/12: Telenor TEL.OL och en Insider-rad 'TEL' (TE Connectivity i USA). Utan karta skrivs inget dit;
    med registrets karta ('TEL' finns inte i den — TE Connectivity) räknas raden inte som Telenor alls."""
    stores = _stores()
    stores["insider"]["signals"] = [{"id": "s1", "ticker": "TEL", "name": "TE Connectivity", "comment": ""}]
    stores["producers"]["producers"] = [{"id": "p1", "ticker": "TEL.OL", "name": "Telenor"}]
    row = stores["producers"]["producers"][0]
    exp = ex.build("TEL.OL", {"producers": stores["producers"]}, ["rule"], rows={"rule": row})
    p = rt.plan(exp, {"rule": row}, (), stores)
    assert "insider" in p.blocked and "tickerkarta" in p.blocked["insider"] and "insider" not in p.writable
    p2 = rt.plan(exp, {"rule": row}, (), stores, sheet_map={"BOL": "BOL.ST"})
    assert p2.exp.source_rows["insider"] is None and "insider" not in p2.targets


def test_a_bare_card_ticker_finds_the_suffixed_rows_with_the_registry_map():
    """Fynd 3/11: Rick Rule-raden står som 'BOL', Insider-signalen som 'BOL.ST' — med registrets karta hittar
    importen i Rick Rule-kortet signalen i stället för att säga att bolaget saknas."""
    stores = _stores()
    stores["producers"]["producers"] = [{"id": "p1", "ticker": "BOL", "name": "Boliden"}]
    stores["insider"]["signals"] = [{"id": "s1", "ticker": "BOL.ST", "name": "Boliden", "comment": ""}]
    row = stores["producers"]["producers"][0]
    exp = ex.build("BOL", {"producers": stores["producers"]}, ["rule"], rows={"rule": row})
    p = rt.plan(exp, {"rule": row}, (), stores, sheet_map={"BOL": "BOL.ST"})
    assert p.targets["insider"] is stores["insider"]["signals"][0]
    res = im.check(answers.read_text("\n".join([HEAD, "BOL.ST;insider:insiders;2;;Insyn;primary;2026-10-02;high;",
                                                "BOL;rule:mine_life;9;år;43-101;primary;2025-01-01;high;"])),
                   p.exp, p.writable, p.blocked)
    assert {c.key for c in res.changes} == {"insider:insiders", "rule:mine_life"} and not res.issues


@pytest.mark.parametrize("key, want", [("aisc", "rule:unit_cost"), ("nav_musd", "rule:nav_musd"),
                                       ("unit_cost", "rule:unit_cost"), ("ds_runway", "rule:ds_runway")])
def test_bare_keys_resolve_in_the_card_first(key, want):
    """Fynd 5/7/10: nycklar utan arkprefix (extraktionens JSON) prövas först mot kortets eget ark — en tom mall
    för ett ark där bolaget saknas gör dem inte tvetydiga."""
    stores = _stores()
    stores["insider"]["signals"] = []
    row = stores["producers"]["producers"][0]
    exp = ex.build("VAR.OL", {"producers": stores["producers"]}, ["rule"], rows={"rule": row})
    p = rt.plan(exp, {"rule": row}, (), stores)
    f, why = im.resolve(key, p.exp.fields, p.exp.source_rows, p.exp.own_sheets)
    assert f is not None and f.key == want, why


def test_missing_after_import_follows_the_sheets_own_position():
    """Fynd 9: kvittots 'saknas fortfarande' räknas som arket själv räknar — Rick Rule-kortets position 8 %, inte
    Scorecardets 1 % — när importen körs i ett ark."""
    from review import missing as mis
    stores = _stores()
    row = stores["producers"]["producers"][0]
    row["position_pct"] = 8.0
    stores["scorecard"]["cards"] = {"vår energi|VAR.OL": {"strategy": "producenter", "position_pct_total": 1.0}}
    exp, _t = _card(stores)
    p = rt.plan(exp, _t, (), stores)
    got = rt.missing_now(p.exp, {"rule": row}, stores, p.own)["rule"]
    want = list(dict.fromkeys(it.label for it in mis.missing_for("rule", row)))
    assert got == want


def test_a_store_that_failed_to_load_says_reload_not_add_the_company():
    """Fynd 13: ett lager som inte gick att läsa står tomt i sessionen — beskedet är 'ladda om', aldrig 'lägg in
    bolaget i arket' (det finns i GitHub)."""
    stores = _stores()
    stores["producers"] = {"producers": [], "royalty": []}                     # standardformen efter läsfelet
    exp, targets = _card(stores)
    p = rt.plan(exp, targets, (), stores, lambda name: "HTTP 500" if name == "producers" else None)
    res = im.check(answers.read_text("\n".join([HEAD, "VAR.OL;rule:unit_cost;30;USD/boe;Q3;primary;2026-10-01;high;"])),
                   p.exp, p.writable, p.blocked)
    ((_l, _k, why),) = res.issues
    assert "kunde inte läsas i den här sessionen (HTTP 500)" in why and "lägg in" not in why


def test_blank_counts_only_fields_chatgpt_should_fill_that_are_still_empty():
    """Fynd 14: 'Bara du', Börsdata-rader och fält som redan har ett värde räknas inte som tomma svar."""
    stores = _stores()
    stores["producers"]["producers"][0]["mine_life"] = 12.0
    exp, targets = _card(stores)
    p = rt.plan(exp, targets, (), stores)
    text = "\n".join([HEAD, "VAR.OL;rule:position_pct;;;;;;;", "VAR.OL;rule:price;;;;;;;",
                      "VAR.OL;rule:mine_life;;;;;;;", "VAR.OL;rule:unit_cost;;;;;;;", "VAR.OL;tiggre:screen.fs;;;;;;;"])
    res = im.check(answers.read_text(text), p.exp, p.writable, p.blocked)
    assert [f.key for f in res.blank] == ["rule:unit_cost"]


# ── Andra granskningsrundan ──────────────────────────────────────────────────
def test_a_field_the_sheet_shows_at_its_own_position_is_imported_from_another_card():
    """Rond 2 fynd 1: Rick Rule-raden 8 % visar AQS, Scorecardet står på 1 %. Importen i Insider-kortet tar emot
    rule:aqs_capex (arket visar fältet) och kvittot listar det inte som saknat efteråt."""
    stores = _stores()
    row = stores["producers"]["producers"][0]
    row["position_pct"] = 8.0
    stores["scorecard"]["cards"] = {"vår energi|VAR.OL": {"strategy": "producenter", "position_pct_total": 1.0}}
    exp, targets = _card(stores)
    p = rt.plan(exp, targets, (), stores)
    text = "\n".join([HEAD, "VAR.OL;rule:aqs_capex;2;;43-101;primary;2026-10-01;high;"])
    res = im.check(answers.read_text(text), p.exp, p.writable, p.blocked)
    assert [c.key for c in res.changes] == ["rule:aqs_capex"] and not res.ignored
    im.apply(res.changes, rt.resolved_targets(p, ["rule"]))
    miss = rt.missing_now(p.exp, {"rule": row}, stores, p.own)["rule"]
    assert not any(m.startswith("AQS · Capex") for m in miss)


def test_the_receipt_in_the_scorecard_uses_the_card_it_ran_in():
    """Rond 2 fynd 2/9: två kandidater för Boliden ('BOL' i Sprott, 'BOL.ST' i Rick Rule). Kvittot i kortet
    'boliden|BOL.ST' (8 %) räknar Rick Rule efter just det kortets position — inte det andra kortets."""
    import scorecard as sc
    stores = _stores()
    stores["producers"]["producers"] = [{"id": "p1", "ticker": "BOL.ST", "name": "Boliden", "position_pct": 1.0}]
    stores["scoring"]["sprott"] = [{"id": "q1", "ticker": "BOL", "name": "Boliden"}]
    stores["scorecard"]["cards"] = {"boliden|BOL.ST": {"strategy": "producenter", "position_pct_total": 8.0},
                                    "boliden|BOL": {"strategy": "sprott", "position_pct_total": 1.0}}
    entry = sc.entries_for(["BOL.ST"], stores)["boliden|BOL.ST"]
    card = sc._shown(stores["scorecard"]["cards"]["boliden|BOL.ST"], entry)
    rule = stores["producers"]["producers"][0]
    exp = ex.build("BOL.ST", stores, ["rule", "scorecard"], rows={"rule": rule, "scorecard": card}, entry=entry)
    p = rt.plan(exp, {"scorecard": lambda: stores["scorecard"]["cards"]["boliden|BOL.ST"]}, (), stores)
    miss = rt.missing_now(p.exp, {"rule": rule}, stores, p.own)["rule"]
    assert any(m.startswith("AQS") for m in miss)                              # köpgrindens 8 % kräver AQS


def test_a_failed_store_explains_reload_also_for_position_dependent_fields():
    """Rond 2 fynd 3: AQS/CSM-fält i ett lager som inte gick att läsa får 'ladda om', inte 'visas inte just nu'."""
    stores = _stores()
    stores["producers"] = {"producers": [], "royalty": []}
    exp, targets = _card(stores)
    p = rt.plan(exp, targets, (), stores, lambda name: "HTTP 500" if name == "producers" else None)
    text = "\n".join([HEAD, "VAR.OL;rule:aqs_capex;2;;x;primary;2026-10-01;high;",
                      "VAR.OL;rule:csm.Bear.price;60;USD/bbl;x;primary;2026-10-01;high;",
                      "VAR.OL;rule:påhittad;1;;x;primary;2026-10-01;high;"])
    res = im.check(answers.read_text(text), p.exp, p.writable, p.blocked)
    why = {k: t for _l, k, t in res.issues + res.ignored}
    assert "ladda om" in why["rule:aqs_capex"] and "ladda om" in why["rule:csm.Bear.price"]
    assert why["rule:påhittad"] == "okänd fältnyckel"


def test_without_a_map_the_exact_scorecard_card_is_used_not_the_bare_one():
    """Rond 2 fynd 5/10: utan karta sorteras 'TEL' (TE Connectivity) före 'TEL.OL' — importen skriver i Telenors
    eget kort, och Insider-raden 'TEL' förblir spärrad."""
    stores = _stores()
    stores["producers"]["producers"] = [{"id": "p1", "ticker": "TEL.OL", "name": "Telenor"}]
    stores["insider"]["signals"] = [{"id": "s1", "ticker": "TEL", "name": "TE Connectivity", "comment": ""}]
    stores["scorecard"]["cards"] = {"telenor|TEL.OL": {"strategy": "producenter"}}
    row = stores["producers"]["producers"][0]
    exp = ex.build("TEL.OL", {"producers": stores["producers"]}, ["rule"], rows={"rule": row})
    p = rt.plan(exp, {"rule": row}, (), stores)
    assert "scorecard" not in p.blocked and "insider" in p.blocked
    text = "\n".join([HEAD, "TEL.OL;scorecard:valuation;P/E 9;;Analys;secondary;2026-10-01;high;"])
    res = im.check(answers.read_text(text), p.exp, p.writable, p.blocked)
    im.apply(res.changes, rt.resolved_targets(p, ["scorecard"]))
    assert stores["scorecard"]["cards"]["telenor|TEL.OL"]["valuation"] == "P/E 9"
    assert set(stores["scorecard"]["cards"]) == {"telenor|TEL.OL"}


def test_blank_follows_who_the_export_showed():
    """Rond 2 fynd 6: ett Börsdata-fält utan tal visades som ChatGPT i underlaget — lämnas det tomt räknas det."""
    stores = _stores()
    row = stores["producers"]["producers"][0]
    exp = ex.build("VAR.OL", {"producers": stores["producers"]}, ["rule"], rows={"rule": row}, blob={})
    assert {r["key"]: r["who"] for r in exp.rows}["rule:ev_musd"] == "ChatGPT"
    p = rt.plan(exp, {"rule": row}, (), stores)
    res = im.check(answers.read_text("\n".join([HEAD, "VAR.OL;rule:ev_musd;;;;;;;"])), p.exp, p.writable, p.blocked)
    assert [f.key for f in res.blank] == ["rule:ev_musd"]


def test_the_tiggre_position_message_promises_nothing_impossible():
    stores = _stores()
    stores["insider"]["signals"][0]["ticker"] = "LOB"
    sig = stores["insider"]["signals"][0]
    exp = ex.build("LOB", {"insider": stores["insider"]}, ["insider"], rows={"insider": sig})
    msg = rt.plan(exp, {"insider": sig}, (), stores).blocked["tiggre"]
    assert "för hand" not in msg and "sparas inte" in msg


def _map_app():
    import streamlit as st
    from review import chatgpt_ui
    st.session_state["got"] = chatgpt_ui.registry_sheet_map()


def test_registry_map_is_trusted_only_when_complete_and_retried_after_failure(monkeypatch):
    """Rond 2 fynd 4/8/11/12: ett misslyckat eller ofullständigt register sparas inte — då kunde 'TEL' tas för
    Telenor — och prövas igen; Thors karta används när registrets filer gick att läsa."""
    from streamlit.testing.v1 import AppTest
    import gist_storage as gs
    from universe import registry as ur
    calls = []

    def _broken(name, fallback):
        calls.append(name)
        if name == "universe_usa.json":
            gs.UNREADABLE_NAMES.add(name)
            return fallback
        return fallback
    monkeypatch.setattr(ur, "default_loader", lambda: _broken)
    monkeypatch.setattr(gs, "UNREADABLE_NAMES", set())
    at = AppTest.from_function(_map_app)
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["got"] is None and "rv_sheet_map" not in at.session_state
    assert at.session_state["rv_sheet_map_failed"] > 0
    n = len(calls)
    at.run()
    assert len(calls) == n                                                    # inget nytt försök inom en minut
    at.session_state["rv_sheet_map_failed"] = 0.0                            # en minut senare: nytt försök
    at.run()
    assert len(calls) > n
    gs.UNREADABLE_NAMES.clear()
    at.session_state["thor_data"] = {"sheet_map": {"BOL": "BOL.ST"}}       # Thors karta bär inget besked om
    at.session_state["rv_sheet_map_failed"] = 0.0                            # att registret var helt läst
    at.run()
    assert at.session_state["got"] is None                                   # — den används aldrig


def test_a_partly_read_registry_gives_no_map(monkeypatch):
    """Registret lästes men USA-delen gick inte att läsa: kartan skulle tro att 'TEL' är Telenor. Den används inte,
    sparas inte — och Thors karta används inte heller medan registrets filer står som oläsliga."""
    from streamlit.testing.v1 import AppTest
    import gist_storage as gs
    from universe import registry as ur

    class _Partial:
        def sheet_map(self):
            return {"TEL": "TEL.OL"}

    def _load(markets=None, loader=None):
        loader("universe_norden.json", None)
        loader("universe_usa.json", None)
        return _Partial()

    def _loader(name, fallback):
        if name == "universe_usa.json":
            gs.UNREADABLE_NAMES.add(name)
        return fallback
    monkeypatch.setattr(gs, "UNREADABLE_NAMES", {"universe_usa.json"})     # Thors läsning av USA-delen fallerade
    monkeypatch.setattr(ur, "default_loader", lambda: _loader)
    monkeypatch.setattr(ur, "load", _load)
    at = AppTest.from_function(_map_app)
    at.session_state["thor_data"] = {"sheet_map": {"TEL": "TEL.OL"}}
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["got"] is None and "rv_sheet_map" not in at.session_state
    gs.UNREADABLE_NAMES.clear()                                              # allt läst: kartan gäller och sparas
    monkeypatch.setattr(ur, "load", lambda markets=None, loader=None: _Partial())
    at.session_state["rv_sheet_map_failed"] = 0.0
    del at.session_state["thor_data"]
    at.run()
    assert at.session_state["got"] == {"TEL": "TEL.OL"} and at.session_state["rv_sheet_map"] == {"TEL": "TEL.OL"}


# ── Tredje granskningsrundan ─────────────────────────────────────────────────
def test_a_stale_thor_map_is_never_used_even_after_the_unreadable_flag_clears(monkeypatch):
    """Rond 3 fynd 1/3: Thors karta byggdes medan USA-delen var oläslig ('TEL' → Telenor). Importens egen hela
    läsning gäller hela sessionen — också på nästa omritning, när flaggan för den oläsliga filen har släppt."""
    from streamlit.testing.v1 import AppTest
    import gist_storage as gs
    from universe import registry as ur

    class _Complete:
        def sheet_map(self):
            return {"BOL": "BOL.ST"}
    monkeypatch.setattr(gs, "UNREADABLE_NAMES", {"universe_usa.json"})
    monkeypatch.setattr(ur, "default_loader", lambda: (lambda name, fb: (gs.UNREADABLE_NAMES.discard(name), fb)[1]))
    monkeypatch.setattr(ur, "load", lambda markets=None, loader=None: (loader("universe_usa.json", None), _Complete())[1])
    at = AppTest.from_function(_map_app)
    at.session_state["thor_data"] = {"sheet_map": {"BOL": "BOL.ST", "TEL": "TEL.OL"}}
    at.run()
    assert at.session_state["got"] == {"BOL": "BOL.ST"}
    assert not gs.UNREADABLE_NAMES                                           # flaggan har släppt
    at.run()
    assert at.session_state["got"] == {"BOL": "BOL.ST"}                      # fortfarande den hela kartan


def test_the_scorecard_receipt_uses_the_strategy_the_card_shows():
    """Rond 3 fynd 2: kortet sparades med strategin Sprott, men bolaget finns inte längre där — Scorecardet visar
    och grindar kortet som producent. Kvittot gör likadant."""
    import scorecard as sc
    stores = _stores()
    stores["producers"]["producers"] = [{"id": "p1", "ticker": "BOL.ST", "name": "Boliden", "position_pct": 1.0}]
    stores["scorecard"]["cards"] = {"boliden|BOL.ST": {"strategy": "sprott", "position_pct_total": 8.0}}
    entry = sc.entries_for(["BOL.ST"], stores)["boliden|BOL.ST"]
    raw = stores["scorecard"]["cards"]["boliden|BOL.ST"]
    rule = stores["producers"]["producers"][0]
    exp = ex.build("BOL.ST", stores, ["rule", "scorecard"], rows={"rule": rule, "scorecard": sc._shown(raw, entry)},
                   entry=entry)
    p = rt.plan(exp, {"scorecard": lambda: raw}, (), stores)
    miss = rt.missing_now(p.exp, {"rule": rule, "scorecard": raw}, stores, p.own)["rule"]
    assert any(m.startswith("AQS") for m in miss)


def test_a_bare_card_without_the_map_does_not_say_add_the_company():
    """Rond 3 fynd 5: Rick Rule 'BOL', Insider 'BOL.ST', ingen karta — beskedet är att raden kan vara ett annat
    bolag, inte 'lägg in det i arket' (då blev det en dubblett)."""
    stores = _stores()
    stores["producers"]["producers"] = [{"id": "p1", "ticker": "BOL", "name": "Boliden"}]
    stores["insider"]["signals"] = [{"id": "s1", "ticker": "BOL.ST", "name": "Boliden", "comment": ""}]
    row = stores["producers"]["producers"][0]
    exp = ex.build("BOL", {"producers": stores["producers"]}, ["rule"], rows={"rule": row})
    p = rt.plan(exp, {"rule": row}, (), stores)
    res = im.check(answers.read_text("\n".join([HEAD, "BOL.ST;insider:insiders;2;;Insyn;primary;2026-10-02;high;"])),
                   p.exp, p.writable, p.blocked)
    ((_l, _k, why),) = res.issues
    assert "tickerkarta" in why and "lägg in" not in why


def test_blank_uses_who_from_the_answer_file():
    """Rond 3 fynd 6: Thors underlag (utan sifferuppdatering) sade Börsdata — läses filen in i ett kort som visar
    raden som ChatGPT räknas den ändå inte som tom."""
    stores = _stores()
    row = stores["producers"]["producers"][0]
    thor = ex.build("VAR.OL", stores, ["rule"], blob=None)
    got = answers.read_file(ex.to_csv(thor.rows), "underlag.csv")
    card = ex.build("VAR.OL", {"producers": stores["producers"]}, ["rule"], rows={"rule": row}, blob={})
    p = rt.plan(card, {"rule": row}, (), stores)
    res = im.check(got, p.exp, p.writable, p.blocked)
    whos = {r["key"]: r["who"] for r in thor.rows}
    assert res.blank and all(whos[f.key] in ("ChatGPT", "Bedömning") for f in res.blank)
    assert "rule:ev_musd" not in {f.key for f in res.blank}                  # Börsdata i Thors fil


def test_a_last_row_equal_to_the_sheet_cancels_the_earlier_duplicate():
    """Rond 3 fynd 7: svarets sista rad bekräftar arkets 1200 — den tidigare 1300 erbjuds inte."""
    stores = _stores()
    stores["producers"]["producers"][0]["unit_cost"] = 1200.0
    exp, targets = _card(stores)
    p = rt.plan(exp, targets, (), stores)
    text = "\n".join([HEAD, "VAR.OL;rule:unit_cost;1300;USD/oz;A;primary;2026-10-01;high;",
                      "VAR.OL;rule:unit_cost;1200;USD/oz;B;primary;2026-10-02;high;"])
    res = im.check(answers.read_text(text), p.exp, p.writable, p.blocked)
    assert not res.changes and res.same == 1 and any("dubbel rad" in t for _l, _k, t in res.ignored)
