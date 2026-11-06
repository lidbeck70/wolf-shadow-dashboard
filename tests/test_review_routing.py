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
    assert res.blank == [(6, "rule:mine_life")]
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
