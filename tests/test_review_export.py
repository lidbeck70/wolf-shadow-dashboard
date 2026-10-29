"""
review/export — underlaget till ChatGPT: rader, CSV, Excel och prompten.

Kolumnerna är stabila, CSV:n har BOM och semikolon, värdena förifylls ur arken och Börsdatas förslag (aldrig
omräknade), det som saknas för köpbeslut står först, och prompten säger reglerna: bara Fyll i-kolumnerna, källa och
datum eller tomt, ingen enhets- eller valutaomräkning, NAV/NPV efter skatt, 0–2 med en mening, rör aldrig "Bara du".
Excel-arket har låsta frågekolumner, blå ifyllnadskolumner och rullgardiner. Inget nätverk.
"""
import io
import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import controls as ctl  # noqa: E402
from review import catalog as cat  # noqa: E402
from review import export as ex  # noqa: E402

DS0 = {f.key: 0 for f in ctl.DS_FIELDS}
STORES = {
    "producers": {"producers": [{"id": "p1", "ticker": "BOL", "name": "Boliden", "commodity": "Koppar", "price": 4.1,
                                 "jurisdiktion": True, "nav_source": "=HYPERLINK(\"x\")"}], "royalty": []},
    "scoring": {"sprott": [], "durrett": []},
    "tiggre": {"candidates": [], "positions": [], "closed": [], "parked": []},
    "insider": {"signals": []},
    "holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0},
    "scorecard": {"cards": {"boliden|BOL": {"strategy": "producenter", "position_pct_total": 1.5,
                                            "sell_rule": "Stopp −15 %"}}},
    "confidence": {"companies": {"BOL.ST": {
        "ticker": "BOL.ST", "name": "Boliden", "stage": "producer", "commodity": "copper",
        "fields": {"aisc": {"value": 2.5, "kind": "ACTUAL", "source": "Q2-rapport", "pub_date": "2026-07-15",
                            "unit": "USD/lb"}}}}, "commodity_overrides": {}},
}
BLOB = {"rows": {"producers:p1": {"asof": "2026-10-28", "ev_ebitda": 5.2, "commodity_price": 4.3,
                                  "commodity_unit": "USD/lb", "shares_growth_3y_pct": 12.0}}}


def _by_key(rows):
    return {r["key"]: r for r in rows}


def test_columns_are_stable():
    assert [h for _k, h in ex.EXPORT_COLUMNS] == [
        "Ticker", "Ark", "Fältnyckel", "Fält", "Typ", "Enhet", "Tillåtna värden", "Värde nu", "Källa nu", "Vem fyller",
        "Krävs för köpbeslut", "Instruktion/definition", "Fyll i: Värde", "Fyll i: Enhet",
        "Fyll i: Källa (URL/dokument+sida)",
        "Fyll i: Källtyp (primary/independent/secondary/mixed/weak/unsupported)", "Fyll i: Datum (ÅÅÅÅ-MM-DD)",
        "Fyll i: Säkerhet (high/medium/low)", "Fyll i: Kommentar/citat"]
    assert set(ex.INPUT_KEYS) == {k for k, _h in ex.EXPORT_COLUMNS if k.startswith("in_")}


def test_build_finds_the_sheets_and_prefills_from_the_stores():
    exp = ex.build("BOL.ST", STORES)
    assert exp.sheets == ["rule", "scorecard", "confidence"] and exp.name == "Boliden" and exp.absent == []
    rows = _by_key(exp.rows)
    assert rows["rule:price"]["value"] == 4.1 and rows["rule:jurisdiktion"]["value"] == "Ja"
    assert rows["scorecard:sell_rule"]["value"] == "Stopp −15 %" and rows["scorecard:sell_rule"]["who"] == cat.BARA_DU
    aisc = rows["confidence:fields.aisc"]
    assert aisc["value"] == 2.5 and aisc["unit"] == "USD/lb" and aisc["source"] == "Q2-rapport · 2026-07-15"
    assert exp.filled == sum(1 for r in exp.rows if r["_stored"]) >= 5
    assert all(r["ticker"] == "BOL.ST" for r in exp.rows)


def test_missing_for_the_buy_decision_comes_first():
    rows = ex.build("BOL.ST", STORES).rows
    ranks = [ex._rank(r) for r in rows]
    assert ranks == sorted(ranks) and ranks[0] == 0
    need = _by_key(rows)["rule:unit_cost"]
    assert need["required"].startswith("Ja — ") and "kostnad per enhet" in need["required"]
    assert _by_key(rows)["rule:price"]["required"] == ""


def test_borsdata_suggestions_fill_empty_fields_but_never_convert():
    rows = _by_key(ex.build("BOL.ST", STORES, sheets=["rule"], blob=BLOB).rows)
    ev = rows["rule:ev_ebitda"]
    assert ev["value"] == 5.2 and ev["who"] == cat.BORSDATA and "förslag, inte inskrivet" in ev["source"]
    assert rows["rule:price"]["value"] == 4.1                    # det sparade värdet står kvar (4.3 är ett förslag)
    assert "+12 %" in rows["rule:ds_aktier_3ar"]["hint"]          # DS-förslaget ur aktiehistoriken — som text
    assert rows["rule:ds_aktier_3ar"]["value"] == ""
    # Börsdata har inget tal → ChatGPT med källa; utan läst blob står fältet kvar som Börsdata
    assert rows["rule:ev_musd"]["who"] == cat.CHATGPT and "Börsdata har inget tal" in rows["rule:ev_musd"]["hint"]
    assert _by_key(ex.build("BOL.ST", STORES, sheets=["rule"], blob=None).rows)["rule:ev_musd"]["who"] == cat.BORSDATA
    ok = ex.build("BOL.ST", STORES, sheets=["rule"], blob=BLOB)
    assert ok.filled == ex.build("BOL.ST", STORES, sheets=["rule"], blob=None).filled   # förslag räknas inte som ifyllt


def test_a_sheet_without_the_stock_is_a_blank_template():
    exp = ex.build("BOL.ST", STORES, sheets=["tiggre", "rule"])
    assert exp.absent == ["tiggre"] and exp.sheets == ["tiggre", "rule"]
    tg = [r for r in exp.rows if r["key"].startswith("tiggre:")]
    assert tg and all(r["value"] == "" for r in tg)
    assert {"tiggre:catalysts.0.name", "tiggre:nav"} <= {r["key"] for r in tg if r["required"]}


def test_explicit_rows_and_scorecard_entry_win_over_the_lookup():
    other = {"id": "p2", "ticker": "BOL", "price": 9.9}
    rows = _by_key(ex.build("BOL", STORES, sheets=["rule"], rows={"rule": other}).rows)
    assert rows["rule:price"]["value"] == 9.9
    card = {"strategy": "producenter", "position_pct_total": 3.0}
    exp = ex.build("BOL", STORES, sheets=["rule", "scorecard"], rows={"scorecard": card})
    keys = {r["key"] for r in exp.rows}
    assert "rule:aqs_kostnad" in keys                             # scorecardets 3 % kräver AQS i Rick Rule-raden
    assert "scorecard:sakerhetsmarginal" not in keys               # … och räknar steg 5 med Lukacs FV


def test_csv_has_bom_semicolons_and_swedish_text():
    rows = ex.build("BOL.ST", STORES, sheets=["rule"]).rows
    data = ex.to_csv(rows)
    assert data.startswith(b"\xef\xbb\xbf")
    text = data.decode("utf-8-sig")
    head = text.splitlines()[0].split(";")
    assert head == [h for _k, h in ex.EXPORT_COLUMNS] and "Kostnad per enhet (AISC/C1)" in text
    assert "Fältnyckel" in head and "rule:unit_cost" in text and "Ja/Nej" in text


def test_prompt_says_the_rules_and_lists_what_is_missing():
    exp = ex.build("BOL.ST", STORES)
    p = ex.build_prompt(exp.ticker, exp.name, exp.rows, "Rick Rule", "Arkens dom: BEVAKA", exp.missing,
                        ex.filename("BOL.ST", "xlsx", date(2026, 10, 29)))
    assert p.startswith("Underlag för köpbeslut: Boliden (BOL.ST) · strategi Rick Rule")
    assert "underlag_BOL-ST_2026-10-29.xlsx" in p
    for must in ("\"Fyll i:\"", "ChatGPT eller Bedömning", "\"Bara du\" rör du aldrig", "källa", "Gissa aldrig",
                 "Konvertera aldrig enheter eller valutor", "EFTER skatt", "Poäng 0–2", "en mening",
                 "Ja eller Nej", ex.ANSWER_HEADER, "Saknas för köpbeslut", "Arkens dom: BEVAKA"):
        assert must in p, must
    assert "- Rick Rule · Kostnad per enhet (AISC/C1) — " in p and "[rule:unit_cost]" in p
    assert "[beslutas av mig — fyll inte i]" in p                 # scorecardets beslut och kontrollernas luckor
    todo = p.split("Fält att fylla i", 1)[1]
    assert "rule:unit_cost" in todo and "scorecard:sell_rule" not in todo and "rule:position_pct" not in todo


def test_filename_is_safe():
    assert ex.filename("ERIC-B.ST", "csv", date(2026, 1, 2)) == "underlag_ERIC-B-ST_2026-01-02.csv"
    assert ex.filename("", ".xlsx", date(2026, 1, 2)) == "underlag_aktie_2026-01-02.xlsx"


def test_xlsx_has_locked_questions_blue_inputs_and_dropdowns():
    openpyxl = pytest.importorskip("openpyxl")
    exp = ex.build("BOL.ST", STORES, blob=BLOB)
    prompt = ex.build_prompt(exp.ticker, exp.name, exp.rows, missing=exp.missing)
    wb = openpyxl.load_workbook(io.BytesIO(ex.to_xlsx(exp.rows, prompt)))
    assert wb.sheetnames == [ex.SHEET_FILL, ex.SHEET_HELP]
    ws = wb[ex.SHEET_FILL]
    assert [c.value for c in ws[1]] == [h for _k, h in ex.EXPORT_COLUMNS]
    assert ws.max_row == len(exp.rows) + 1 and ws.protection.sheet and ws.freeze_panes == "E2"
    col = {k: i for i, (k, _h) in enumerate(ex.EXPORT_COLUMNS, 1)}
    rows = {ws.cell(i, col["key"]).value: i for i in range(2, ws.max_row + 1)}
    unit_cost = rows["rule:unit_cost"]
    assert ws.cell(unit_cost, col["in_value"]).protection.locked is False          # blått: fyll i
    assert ws.cell(unit_cost, col["in_value"]).fill.fgColor.rgb.endswith(ex._CELL_IN)
    assert ws.cell(unit_cost, col["key"]).protection.locked is True               # frågan står still
    mine = rows["scorecard:sell_rule"]
    assert ws.cell(mine, col["in_value"]).protection.locked is True               # "Bara du" är låst
    assert ws.cell(rows["rule:price"], col["value"]).value == 4.1                  # tal som tal, oomräknat
    assert ws.cell(rows["rule:nav_source"], col["value"]).data_type == "s"        # text, aldrig en formel
    dvs = {dv.formula1: str(dv.sqref) for dv in ws.data_validations.dataValidation}
    v = ex.EXPORT_COLUMNS.index(("in_value", "Fyll i: Värde")) + 1
    letter = openpyxl.utils.get_column_letter(v)
    assert f"{letter}{rows['rule:kapitaldisciplin']}" in dvs['"Ja,Nej"'].split()
    assert f"{letter}{rows['rule:ds_runway']}" in dvs['"0,1,2"'].split()
    assert '"primary,independent,secondary,mixed,weak,unsupported"' in dvs and '"high,medium,low"' in dvs
    assert f"{letter}{mine}" not in " ".join(dvs.values()).split()
    help_ws = wb[ex.SHEET_HELP]
    assert help_ws["A1"].value.startswith("Underlag för köpbeslut: Boliden (BOL.ST)")


def test_without_openpyxl_there_is_no_xlsx_but_csv_still_works(monkeypatch):
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    rows = ex.build("BOL.ST", STORES, sheets=["rule"]).rows
    assert ex.to_xlsx(rows, "p") is None and not ex.xlsx_available()
    assert ex.to_csv(rows).startswith(b"\xef\xbb\xbf")


def test_choice_list_and_allowed_text_match():
    f = {x.path: x for x in cat.fields_for("insider", {})}
    assert ex.choice_list(f["role"]) == ("VD/CFO", "Styrelse", "Övrig") and ex.allowed(f["role"]) == "VD/CFO, Styrelse, Övrig"
    assert ex.allowed(f["amount"]) == "tal ≥ 0" and ex.allowed(f["okar_25"]) == "Ja, Nej"
    r = {x.path: x for x in cat.fields_for("rule", {"position_pct": 3.0, "fcf_kvalitet": "A"})}
    assert ex.allowed(r["fv.Base.target_yield"]) == "tal 6–8" and ex.allowed(r["fin_catalyst_date"]) == "ÅÅÅÅ-MM"
    assert ex._quoted(["a,b"]) is None and ex._quoted(["x" * 300]) is None
