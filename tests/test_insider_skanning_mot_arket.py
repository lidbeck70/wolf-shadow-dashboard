"""
Insider steg 1 (batch 10): skanningen kopplas till arkets rader — inga API-ändringar, inga regler ändras.

  Matchning      skanningens kluster mot arkets rader på ins_id, sedan tickern i någon form (thor.ticker.forms):
                 'VOLV B' skrivet för hand och 'VOLV-B.ST' är samma bolag — 'Finns i arket' i stället för en dubblett
  Förslag        refresh_ui.offer (vilken typ som helst, Använd via on_click) för antal, roll, belopp, återkommande,
                 snittkurs och efter fall — aldrig en nedgradering; grind och trigger bara som text
  Visning        säljregel 3 (F-score nu ur nattjobbet) och säljregel 5 (upptäckt + 18 mån)
  Texter         F-score finns i licensen sedan PR #186 (KPI 167 last/point)
  Katalogen      skanningens fält är Börsdata i ChatGPT-underlaget när panelen har talet

Att öppna Insider-sidan skriver ingenting; allt skrivs först med användarens eget klick.
"""
import copy
import json
import os
import sys
from datetime import date

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import insider as ins  # noqa: E402
import refresh_ui as rui  # noqa: E402
import storage  # noqa: E402
from review import catalog as cat  # noqa: E402
from review import export as ex  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cluster(**over) -> dict:
    """Ett kluster som insider_scan.scan sparar det (bara fälten arket läser)."""
    base = {"ticker": "VOLV-B.ST", "name": "Volvo B", "ins_id": 123, "currency": "NOK", "found": "2026-11-09",
            "insiders": 3, "role": ins.ROLE_TOP, "amount": 1235.0, "okar_25": False, "efter_fall": True,
            "aterkommande": True, "cluster_avg": 48.2, "price_now": 50.0, "gate": "Ja", "trigger": "A",
            "score": 9, "status": ins.S_BUY, "vs_cluster": 3.7, "stop": 40.97, "chase": False,
            "auto": {"gate_checks": [], "trigger_note": "stängning över MA20",
                     "efter_fall_note": "34 % under 52v-topp, F-score 6", "drawdown_52w": 34.0,
                     "amount_local": 1300.0, "first_buy": "2026-10-20", "last_buy": "2026-11-05", "buys": []}}
    base.update(over)
    return base


def _row(**over) -> dict:
    """En rad som lagts in för hand ('Ny signal'): tickern som den skrevs, roll Övrig, inga tal."""
    base = {"id": "v1", "ticker": "VOLV B", "name": "Volvo", "found": "2026-11-06", "role": ins.ROLE_OTHER,
            "gate": ins.GATE_BLANK, "trigger": ""}
    base.update(over)
    return base


# ── Matchningen ──────────────────────────────────────────────────────────────
def test_the_hand_typed_form_and_the_scans_form_are_the_same_company():
    assert ins.same_company("VOLV B", "VOLV-B.ST")
    assert ins.same_company("volv-b", "VOLV-B.ST")
    assert ins.same_company("VOLV-B.ST", "VOLV-B.ST")
    assert not ins.same_company("VOLV A", "VOLV-B.ST")          # annan aktieklass
    assert not ins.same_company("ABB.ST", "ABB.HE")             # två börssuffix måste vara lika
    assert not ins.same_company("", "VOLV-B.ST")


def test_a_cluster_matches_on_ins_id_first_then_on_the_ticker_form():
    cl = _cluster()
    by_form = _row()
    assert ins.match_signal(cl, [by_form]) is by_form
    by_id = _row(id="x", ticker="VOLVO", ins_id="123")          # annan tickertext, samma Börsdata-id
    assert ins.match_signal(cl, [by_form, by_id]) is by_id
    other = _row(id="o", ins_id=999)                            # 'VOLV B' men ett annat bolag enligt ins_id
    assert ins.match_signal(cl, [other]) is None
    assert ins.match_signal(cl, []) is None
    assert ins.cluster_for(by_form, [cl], [by_form]) is cl
    assert ins.clusters_by_row([cl], [by_form]) == {id(by_form): cl}


# ── Förslagen per fält ───────────────────────────────────────────────────────
def test_a_hand_typed_row_gets_every_scan_field_but_never_gate_or_trigger():
    offers = ins.scan_offers(_row(), _cluster())
    assert offers == {"insiders": 3, "role": ins.ROLE_TOP, "amount": 1235.0, "aterkommande": True,
                      "cluster_avg": 48.2, "efter_fall": True}
    assert isinstance(offers["insiders"], int)
    assert "gate" not in ins.SCAN_FIELDS and "trigger" not in ins.SCAN_FIELDS and "okar_25" not in ins.SCAN_FIELDS


def test_nothing_is_ever_downgraded():
    """Fönstret rullar från i dag: ett lägre antal eller belopp är köp som fallit ur fönstret, inte ett fel."""
    row = _row(insiders=4, role=ins.ROLE_TOP, amount=2000.0, aterkommande=True, efter_fall=True, cluster_avg=48.2)
    assert ins.scan_offers(row, _cluster()) == {}
    lower = _cluster(insiders=2, role=ins.ROLE_BOARD, amount=300.0, aterkommande=False, efter_fall=False)
    assert ins.scan_offers(_row(insiders=3, role=ins.ROLE_TOP, amount=900.0, cluster_avg=48.2), lower) == {}
    # skanningens 'nej' i ett kryss föreslås aldrig — inte ens på en tom rad
    assert "efter_fall" not in ins.scan_offers(_row(), lower) and "aterkommande" not in ins.scan_offers(_row(), lower)


def test_higher_numbers_are_offered_within_the_same_cluster():
    row = _row(insiders=2, role=ins.ROLE_BOARD, amount=900.0, cluster_avg=50.0)
    assert ins.same_cluster(row, _cluster()) is True
    assert ins.scan_offers(row, _cluster()) == {"insiders": 3, "role": ins.ROLE_TOP, "amount": 1235.0,
                                                "aterkommande": True, "cluster_avg": 48.2, "efter_fall": True}


def _full_row(**over) -> dict:
    """En rad som lades in ur skanningen: alla skanningsfält ifyllda, samma roll och kryss som klustret."""
    return _row(**{"role": ins.ROLE_TOP, "aterkommande": True, "efter_fall": True, **over})


def test_a_rolled_window_average_is_not_offered():
    """Köp A 09-25 @40, B 10-05 @50, C 10-18 @60 → raden: 3 insiders, 1 500 tkr, snitt 50. Sex dagar senare har A
    fallit ur fönstret: B och C ger 2 insiders, 1 100 tkr, snitt 55 — samma kluster, men snittet är bara de
    senaste köpen och skulle flytta stoppen från 42,50 till 46,75."""
    row = _full_row(found="2026-10-20", insiders=3, amount=1500.0, cluster_avg=50.0)
    cl = _cluster(insiders=2, amount=1100.0, cluster_avg=55.0,
                  auto={**_cluster()["auto"], "first_buy": "2026-10-05", "last_buy": "2026-10-18"})
    assert ins.same_cluster(row, cl) is True and ins.rolled_out(row, cl)
    assert ins.scan_offers(row, cl) == {}
    scan = {"generated": "2026-10-26T07:01:00+00:00", "clusters": [cl]}
    assert "cluster_avg" not in ins.scan_suggestions(row, scan)
    assert ins.held_back_avg(row, cl) == 55.0                     # bara som text, utan Använd
    assert ins.stop_price(row) == pytest.approx(42.5)             # raden orörd


def test_the_three_week_old_row_keeps_its_average():
    """Raden 2026-10-12: 3 insiders, 1 310 tkr, snitt 43,67 (stopp 37,12). 2026-11-02 har 10-01-köpet fallit ur
    fönstret: 2 insiders, 510 tkr, snitt 51 — inget förslag, inte heller till ChatGPT-underlaget."""
    row = _full_row(found="2026-10-12", insiders=3, amount=1310.0, cluster_avg=43.6667)
    cl = _cluster(insiders=2, amount=510.0, cluster_avg=51.0,
                  auto={**_cluster()["auto"], "first_buy": "2026-10-05", "last_buy": "2026-10-12"})
    assert ins.same_cluster(row, cl) is True
    assert "cluster_avg" not in ins.scan_offers(row, cl)
    assert "cluster_avg" not in ins.scan_suggestions(row, {"generated": "2026-11-02", "clusters": [cl]})
    # bara beloppet lägre räcker — och en tom snittkurs fylls som förut
    assert "cluster_avg" not in ins.scan_offers(_full_row(found="2026-10-12", amount=1310.0, cluster_avg=43.6667), cl)
    assert ins.scan_offers(_full_row(found="2026-10-12", insiders=3, amount=1310.0), cl) == {"cluster_avg": 51.0}
    assert ins.held_back_avg(_full_row(found="2026-10-12", insiders=3, amount=1310.0), cl) is None


def test_a_growing_cluster_still_offers_its_average():
    row = _full_row(insiders=3, amount=1235.0, cluster_avg=50.0)          # lika många, lika mycket: inget föll bort
    assert not ins.rolled_out(row, _cluster())
    assert ins.scan_offers(row, _cluster()) == {"cluster_avg": 48.2}
    assert ins.held_back_avg(row, _cluster()) is None
    assert not ins.rolled_out(_row(cluster_avg=50.0), _cluster(insiders=1))   # radens tal saknas: inget att jämföra
    empty = _full_row(cluster_avg=50.0)                                     # antal och belopp tomma
    assert ins.scan_offers(empty, _cluster()) == {"insiders": 3, "amount": 1235.0, "cluster_avg": 48.2}


def test_a_new_cluster_after_the_row_was_found_gets_no_suggestions():
    old = _row(found="2026-06-01", insiders=2, cluster_avg=80.0)
    assert ins.same_cluster(old, _cluster()) is False
    assert ins.scan_offers(old, _cluster()) == {}
    assert ins.scan_offers(_row(found="2026-06-01"), _cluster()) == {}     # inte ens tomma fält


def test_without_dates_only_empty_fields_are_filled():
    row = _row(found="", insiders=2, cluster_avg=50.0)
    assert ins.same_cluster(row, _cluster()) is None
    offers = ins.scan_offers(row, _cluster())
    assert "insiders" not in offers and "cluster_avg" not in offers and "role" not in offers   # ifyllda
    assert offers["amount"] == 1235.0 and offers["efter_fall"] is True                          # tomma / ja
    assert ins.same_cluster(_row(), _cluster(auto={})) is None


def test_the_amount_shows_the_local_currency_and_the_rate_used():
    assert ins.amount_note(_cluster()) == "1 300 tNOK × kurs 0,95 (grov)"
    assert ins.amount_note(_cluster(currency="SEK")) == ""
    assert ins.amount_note(_cluster(auto={})) == ""


def test_scan_suggestions_carry_date_and_unit_for_the_export():
    scan = {"generated": "2026-11-09T07:01:00+00:00", "clusters": [_cluster()]}
    sug = ins.scan_suggestions(_row(), scan)
    assert sug["insiders"] == (3, "2026-11-09", "st") and sug["amount"] == (1235.0, "2026-11-09", "tkr")
    assert sug["cluster_avg"] == (48.2, "2026-11-09", "NOK") and sug["role"] == (ins.ROLE_TOP, "2026-11-09", "")
    assert ins.scan_suggestions(_row(ticker="ERIC B"), scan) == {} and ins.scan_suggestions(_row(), None) == {}


# ── Säljregel 3 och 5 (bara visning) ─────────────────────────────────────────
def test_the_sell_rule_numbers_are_the_sheets_own():
    rule3, rule5 = ins.SELL_RULES[2], ins.SELL_RULES[4]
    assert f"F-score < {ins.SELL_FSCORE_BELOW}" in rule3[1]
    assert f"{ins.TIME_STOP_MONTHS} månader" in rule5[1] and f"Upptäckt + {ins.TIME_STOP_MONTHS} mån" in rule5[2]


def test_sell_watch_reads_fscore_and_the_time_stop():
    today = date(2026, 11, 9)
    lines = ins.sell_watch(_row(found="2025-03-31"), 3, "2026-11-08", today=today)
    assert lines == [("Säljregel 3: F-score nu 3 < 4 — grinden bryts (Börsdata 2026-11-08)", "stop"),
                     ("Säljregel 5: tidsstopp 2026-09-30 (upptäckt + 18 mån) — passerat", "warn")]
    assert ins.sell_watch(_row(), 6, "", today=today) == [
        ("Säljregel 3: F-score nu 6", ""), ("Säljregel 5: tidsstopp 2028-05-06 (upptäckt + 18 mån)", "")]
    assert ins.sell_watch(_row(found="okänt"), None) == []                 # inget F-score, inget datum: inget
    assert ins.time_stop({"found": "2026-08-31"}) == date(2028, 2, 29)
    assert ins.time_stop({"found": "2025-08-31"}) == date(2027, 2, 28)


# ── Inaktuella texter ────────────────────────────────────────────────────────
def test_fscore_is_no_longer_called_missing_from_the_licence():
    import borsdata_api as bd
    assert bd.KPI_SCREENER_CALC[bd.KPI["f_score"]] == ("last", "point")
    assert bd.KPI["f_score"] not in bd.KPI_SCREENER_UNAVAILABLE
    for name in ("insider.py", "insider_scan.py"):
        with open(os.path.join(ROOT, name), encoding="utf-8") as f:
            src = f.read()
        assert "ligger inte i API-licensen" not in src and "ligger inte i licensen" not in src, name
    import insider_scan as sc
    _gate, checks = sc.auto_gate({"market_cap": 900.0, "f_score": None, "net_debt_ebitda": 1.0, "fcf_m": 5.0,
                                  "revenue_growth": None})
    details = " ".join(c["detail"] for c in checks)
    assert "saknas i Börsdata för bolaget" in details and "licens" not in details


# ── Katalogen och ChatGPT-underlaget ─────────────────────────────────────────
def test_the_catalog_marks_the_scan_fields_as_borsdata():
    fields = {f.path: f for f in cat.fields_for("insider", {})}
    for k in ins.SCAN_FIELDS:
        assert fields[k].who == cat.BORSDATA and fields[k].scan == k, k
    assert fields["okar_25"].who == cat.CHATGPT and not fields["okar_25"].scan
    assert fields["gate"].who == cat.BARA_DU and fields["trigger"].who == cat.BARA_DU
    assert fields["price_now"].refresh == "price" and not fields["price_now"].scan


def _rows(export) -> dict:
    return {r["key"]: r for r in export.rows}


def test_the_export_does_not_ask_chatgpt_for_numbers_the_scan_has():
    row = _row()
    scan = {"generated": "2026-11-09T07:01", "clusters": [_cluster()]}
    exp = ex.build("VOLV B", {"insider": {"signals": [row]}}, ["insider"], blob={}, rows={"insider": row},
                   scan=scan)
    r = _rows(exp)
    assert r["insider:insiders"]["who"] == cat.BORSDATA and r["insider:insiders"]["value"] == 3
    assert r["insider:insiders"]["source"] == "Insider-skanningen 2026-11-09 — förslag, inte inskrivet"
    assert r["insider:amount"]["unit"] == "tkr" and r["insider:cluster_avg"]["unit"] == "NOK"
    assert r["insider:efter_fall"]["value"] == "Ja" and r["insider:okar_25"]["who"] == cat.CHATGPT
    assert exp.filled == 1                                         # bara rollen är sparad — förslagen räknas inte
    prompt = ex.build_prompt(exp.ticker, exp.name, exp.rows)
    assert "insider:insiders" not in prompt and "insider:amount" not in prompt and "insider:okar_25" in prompt
    assert row == _row()                                                      # underlaget skriver ingenting


def test_without_the_scan_the_export_asks_chatgpt_as_before():
    row = _row(aterkommande=False)
    for scan in (None, {"clusters": [_cluster(ticker="ERIC-B.ST", ins_id=7)]}):
        r = _rows(ex.build("VOLV B", {"insider": {"signals": [row]}}, ["insider"], blob=None,
                           rows={"insider": row}, scan=scan))
        assert r["insider:insiders"]["who"] == cat.CHATGPT and r["insider:role"]["who"] == cat.BORSDATA  # roll ifylld
        assert r["insider:insiders"]["hint"].startswith("Skanningen har inget tal för raden")
        assert r["insider:aterkommande"]["who"] == cat.CHATGPT          # ett sparat 'nej' är inget besked


# ── refresh_ui.offer (AppTest) ───────────────────────────────────────────────
def _offer_app(root):
    import sys as _s
    _s.path.insert(0, root)
    import streamlit as st
    import refresh_ui
    row = st.session_state.setdefault("row", {"id": "r1", "n": None, "role": "Övrig", "flag": False, "amt": 900.0})
    st.number_input("Antal", min_value=0, step=1, value=row["n"], key="w_n")
    st.selectbox("Roll", ["VD/CFO", "Styrelse", "Övrig"], index=["VD/CFO", "Styrelse", "Övrig"].index(row["role"]),
                 key="w_role")
    st.checkbox("Kryss", value=bool(row["flag"]), key="w_flag")
    refresh_ui.offer(row, "n", 3.0, "Antal", key="n", source="Skanningen", asof="2026-11-09", widget_key="w_n",
                     cast=int)
    refresh_ui.offer(row, "role", "VD/CFO", "Roll", key="role", widget_key="w_role")
    refresh_ui.offer(row, "flag", True, "Kryss", key="flag", widget_key="w_flag")
    refresh_ui.offer(row, "amt", 1234.4, "Belopp", key="amt", fmt=lambda v: f"{v:,.0f}".replace(",", " "),
                     unit="tkr", note="1 300 tNOK × kurs 0,95 (grov)")
    refresh_ui.offer(row, "amt", 900.0, "Samma", key="same")             # redan ifyllt: ritas inte


def test_offer_writes_any_type_only_on_click():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_offer_app, args=(ROOT,), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "🤖 Skanningen 2026-11-09: Antal 3" in html and "Roll VD/CFO" in html and "Kryss ja" in html
    assert "Belopp 1 234 tkr — 1 300 tNOK × kurs 0,95 (grov)" in html
    keys = {b.key for b in at.button}
    assert {"rf_offer_n", "rf_offer_role", "rf_offer_flag", "rf_offer_amt"} <= keys and "rf_offer_same" not in keys
    assert at.session_state["row"] == {"id": "r1", "n": None, "role": "Övrig", "flag": False, "amt": 900.0}
    at.button(key="rf_offer_n").click().run()
    assert at.session_state["row"]["n"] == 3 and isinstance(at.session_state["row"]["n"], int)
    assert at.number_input(key="w_n").value == 3
    at.button(key="rf_offer_role").click().run()
    assert at.session_state["row"]["role"] == "VD/CFO" and at.selectbox(key="w_role").value == "VD/CFO"
    at.button(key="rf_offer_flag").click().run()
    assert at.session_state["row"]["flag"] is True and at.checkbox(key="w_flag").value is True
    assert not at.exception, at.exception
    assert "rf_offer_n" not in {b.key for b in at.button}                    # använt förslag föreslås inte igen


def test_the_offer_text_escapes_html_and_dollars():
    assert rui._shown(True) == "ja" and rui._shown(False) == "nej" and rui._shown(3) == "3"
    assert rui._shown("VD/CFO") == "VD/CFO" and rui._shown(1234.5, lambda v: f"{v:.0f}") == "1234"
    assert rui._same(3, 3.0) and rui._same(None, False) and not rui._same(None, 0) and rui._same(" a ", "a")


# ── Insider-sidan (AppTest) ──────────────────────────────────────────────────
SCAN = {"generated": "2026-11-09T07:01:00+00:00", "clusters": [
    _cluster(),
    _cluster(ticker="EKTA-B.ST", name="Elekta", ins_id=55, currency="SEK"),
]}
REFRESH = {"rows": {"insider:v1": {"price": 50.0, "asof": "2026-11-08", "currency": "SEK", "f_score": 3}}}
STORES = {"insider": {"signals": [_row()]}}


def _same_json(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


@pytest.fixture
def page(monkeypatch):
    """Insider-sidan ur en fejkad lagring och fejkad skanning — inget GitHub, ingen Gist. run() tar egna
    arkdata, nattjobbsrader och skanning i stället för STORES, REFRESH och SCAN."""
    state = {"stores": copy.deepcopy(STORES), "refresh": REFRESH, "scan": SCAN}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(state["stores"].get(name, default))))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    import gist_storage
    monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)
    monkeypatch.setattr(rui, "load_refresh", lambda: state["refresh"])
    monkeypatch.setattr(ins, "_load_auto_scan", lambda: copy.deepcopy(state["scan"]))

    def run(stores=None, refresh=None, scan=None):
        for k, v in (("stores", stores), ("refresh", refresh), ("scan", scan)):
            if v is not None:
                state[k] = copy.deepcopy(v)
        from streamlit.testing.v1 import AppTest

        def app(root):
            import sys as _s
            _s.path.insert(0, root)
            import insider as i
            i.render_insider_page()
        at = AppTest.from_function(app, args=(ROOT,), default_timeout=90)
        at.run()
        assert not at.exception, at.exception
        return at
    return run


def _texts(at) -> str:
    return " ".join(getattr(e, "value", "") or "" for coll in (at.markdown, at.caption) for e in coll)


def test_a_cluster_already_in_the_sheet_shows_suggestions_instead_of_add(page):
    at = page()
    txt = _texts(at)
    keys = {b.key for b in at.button}
    assert "Finns i arket som VOLV B." in txt
    assert "ins_auto_add_VOLV-B.ST" not in keys and "ins_auto_add_EKTA-B.ST" in keys     # ingen dubblett
    for k in ins.SCAN_FIELDS:
        assert f"rf_offer_ins_scan_v1_{k}" in keys, k
    assert not any("gate" in k or "trigger" in k for k in keys if k.startswith("rf_offer"))
    assert "Belopp 1 235 tkr — 1 300 tNOK × kurs 0,95 (grov)" in txt
    assert "räknar köparnamn: en person och hens bolag blir två" in txt
    assert "Klustrets snittkurs 48.2 NOK — köp 2026-10-20 – 2026-11-05" in txt
    assert "Grind och trigger ur skanningen är bara text" in txt
    assert "🤖 Skanningen har 6 förslag för raden" in txt
    assert "Säljregel 3: F-score nu 3 &lt; 4 — grinden bryts (Börsdata 2026-11-08)" in txt
    assert "Säljregel 5: tidsstopp 2028-05-06 (upptäckt + 18 mån)" in txt
    assert "ligger inte i API-licensen" not in txt
    assert _same_json(at.session_state["insider"], STORES["insider"])          # att öppna sidan skriver ingenting
    at.run()
    assert _same_json(at.session_state["insider"], STORES["insider"])


def test_anvand_writes_the_field_and_the_box(page):
    at = page()
    at.button(key="rf_offer_ins_scan_v1_insiders").click().run()
    sig = at.session_state["insider"]["signals"][0]
    assert sig["insiders"] == 3 and isinstance(sig["insiders"], int)
    assert at.number_input(key="ins_n_v1").value == 3
    at.button(key="rf_offer_ins_scan_v1_role").click().run()
    assert at.session_state["insider"]["signals"][0]["role"] == ins.ROLE_TOP
    assert at.selectbox(key="ins_r_v1").value == ins.ROLE_TOP
    at.button(key="rf_offer_ins_scan_v1_amount").click().run()
    assert at.session_state["insider"]["signals"][0]["amount"] == 1235.0
    at.button(key="rf_offer_ins_scan_v1_efter_fall").click().run()
    assert at.session_state["insider"]["signals"][0]["efter_fall"] is True
    assert at.checkbox(key="ins_b_v1_efter_fall").value is True
    assert not at.exception, at.exception
    sig = at.session_state["insider"]["signals"][0]
    assert sig["gate"] == ins.GATE_BLANK and sig["trigger"] == ""                 # grind och trigger orörda
    keys = {b.key for b in at.button}
    assert "rf_offer_ins_scan_v1_insiders" not in keys and "rf_offer_ins_scan_v1_cluster_avg" in keys
    assert len(at.session_state["insider"]["signals"]) == 1


def test_a_rolled_window_average_gets_no_button_and_the_stop_stays(page):
    row = _full_row(found="2026-10-12", insiders=3, amount=1310.0, cluster_avg=43.6667, price_now=50.0)
    cl = _cluster(insiders=2, amount=510.0, cluster_avg=51.0, currency="SEK",
                  auto={**_cluster()["auto"], "first_buy": "2026-10-05", "last_buy": "2026-10-12"})
    stores = {"insider": {"signals": [row]}}
    at = page(stores=stores, scan={"generated": "2026-11-02T07:01:00+00:00", "clusters": [cl]})
    txt = _texts(at)
    keys = {b.key for b in at.button}
    assert not any(k.startswith("rf_offer_ins_scan_v1_") for k in keys)
    assert "Inga förslag — arket har samma eller högre tal." in txt
    assert "Klustrets snittkurs i fönstret 51 SEK — köp har fallit ur fönstret, inget förslag." in txt
    assert "🤖 Skanningen har" not in txt
    assert [m.value for m in at.metric if m.label == "Stopp (−15 %)"] == ["37.12"]
    assert _same_json(at.session_state["insider"], stores["insider"])             # att öppna sidan skriver ingenting


def test_the_fscore_line_names_the_matched_company(page):
    """En handskriven rad vars nattrad matchats mot ett annat bolag: kursen är redan inskriven, så kursförslaget
    (och dess matchat-not) ritas inte — säljregel 3 säger själv vilket bolag F-score kommer från."""
    stores = {"insider": {"signals": [_row(price_now=50.0)]}}
    other = {"rows": {"insider:v1": {"price": 50.0, "asof": "2026-11-08", "currency": "SEK", "f_score": 2,
                                     "bd_name": "Volati AB"}}}
    at = page(stores=stores, refresh=other)
    lines = [m.value for m in at.markdown if "Säljregel" in (m.value or "")]
    rule3 = [x for x in lines if "Säljregel 3" in x]
    assert len(rule3) == 1
    assert "Säljregel 3: F-score nu 2 &lt; 4 — grinden bryts (Börsdata 2026-11-08) (matchat: Volati AB)" in rule3[0]
    assert not any("matchat" in x for x in lines if "Säljregel 5" in x)
    assert "rf_insider_v1_price_now" not in {b.key for b in at.button}            # kursen har redan värdet
    assert ins._refresh_fscore(_row()) == (2.0, "2026-11-08", " (matchat: Volati AB)")
    # nattradens namn liknar radens: ingen not
    same = copy.deepcopy(other)
    same["rows"]["insider:v1"]["bd_name"] = "Volvo AB"
    txt = _texts(page(stores=stores, refresh=same))
    assert "Säljregel 3: F-score nu 2 &lt; 4" in txt and "(matchat:" not in txt
    # noten är redan escapad och escapas inte en gång till
    amp = copy.deepcopy(other)
    amp["rows"]["insider:v1"]["bd_name"] = "A&B Mining"
    txt = _texts(page(stores=stores, refresh=amp))
    assert "(matchat: A&amp;B Mining)" in txt and "&amp;amp;" not in txt
