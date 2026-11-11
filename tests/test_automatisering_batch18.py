"""
Automatisering, batch 18 — befintliga avvikelser, nu bestämda av ägaren.

1. Lukacs FV: en FCF-kvalitetsklass som aldrig valts visas som '—' (inte som C), och köpgrindens steg 5 kräver en
   vald klass — gate_gaps namnger valet som saknas. Kort med en sparad klass räknar exakt som förut.
2. AQS: ingenting fylls i när sidan ritas. Jurisdiktion och management ur arkets egna frågor visas som förslag med
   Använd. Rick Rules 'Jurisdiktion OK' (Fraser-rankingens topphalva) ger 1 p, inte 2; insyn ger fortfarande 2 p.
3. Okänd valuta i nattjobbet (och i 'Hämta nu'): talen som räknas om med en kurs utanför FX_TO_USD blir None i
   stället för att räknas som USD. Kursen i sin egen valuta står kvar.
4. Tiggre: 'Försenad 2:a ggn' i kalendern blir ett förslag med en knapp — registret sparas bara på klicket.
   Stäng utan Markera får ändå säljregeln som orsak (arket, registret, journalen), som förut.

Inga grindar, trösklar, poängformler eller storlekar ändrade utöver ägarens beslut ovan.
"""
import copy
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import controls as ctl   # noqa: E402
import lukacs            # noqa: E402
import sheets_refresh as sr   # noqa: E402
import storage           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASOF = "2026-11-07"


# ═════════════════════════════════════════════════════════════════════════════
#  1. Lukacs FV — klassen måste väljas
# ═════════════════════════════════════════════════════════════════════════════
def _fv_row(**over):
    """Guidens EQX-exempel (klass B, yield 9 %): fair value Base 8,33 mot kurs 5 → MoS 40 %."""
    row = {"fcf_kvalitet": "B", "framtida_antal_aktier": 1200.0, "aktuell_kurs": 5.0, "nd_ebitda": 0.15,
           "what_must_go_right": "Guld 2 400 · 500 koz",
           "fv": {ctl.BEAR: {"forward_fcf_musd": 400, "target_yield": 9},
                  ctl.BASE: {"forward_fcf_musd": 900, "target_yield": 9},
                  ctl.BULL: {"forward_fcf_musd": 1600, "target_yield": 9}}}
    row.update(over)
    return row


@pytest.mark.parametrize("unset", [None, "", "  ", "X"])
def test_an_unchosen_class_keeps_step_five_from_turning_green(unset):
    row = _fv_row(fcf_kvalitet=unset)
    if unset is None:
        row.pop("fcf_kvalitet")
    assert lukacs.gate_gaps(row, 5.0, "producenter") == [lukacs.CLASS_MISSING]
    assert not lukacs.gate_ok(row, 5.0, "producenter")
    assert "FCF-kvalitet är inte vald" in lukacs.CLASS_MISSING


def test_a_yield_outside_every_band_can_no_longer_pass_without_a_class():
    """Förut: ingen klass → inga yield-fel → steg 5 grönt med 4 % (under alla band, FV 18,75 mot kurs 5)."""
    row = _fv_row(fcf_kvalitet=None, fv={s: {"forward_fcf_musd": f, "target_yield": 4}
                                         for s, f in ((ctl.BEAR, 400), (ctl.BASE, 900), (ctl.BULL, 1600))})
    ev = lukacs.evaluate(row)
    assert ev["quality"] is None and ev["yield_errors"] == []           # evaluate räknar som förut
    assert lukacs.gate_gaps(row, 5.0, "producenter") == [lukacs.CLASS_MISSING]
    row["fcf_kvalitet"] = "C"                                           # vald klass: bandet låser yielden
    assert any("utanför klass C:s band" in g for g in lukacs.gate_gaps(row, 5.0, "producenter"))


def test_a_chosen_class_gives_the_same_results_as_before():
    assert lukacs.gate_gaps(_fv_row(), 5.0, "producenter") == []
    assert lukacs.gate_gaps(_fv_row(fcf_kvalitet="b"), 5.0, "producenter") == []      # gemener som förut
    a = lukacs.gate_gaps(_fv_row(fcf_kvalitet="A"), 5.0, "producenter")
    assert len(a) == 3 and all("utanför klass A:s band" in g for g in a)
    d = lukacs.gate_gaps(_fv_row(fcf_kvalitet="D"), 5.0, "producenter")
    assert d == [f"{lukacs.NOT_FCF_VALUED} — säkerhetsmarginalen kan inte härledas ur FCF för den här klassen."]
    thin = lukacs.gate_gaps(_fv_row(aktuell_kurs=7.0), 5.0, "producenter")
    assert thin == ["Säkerhetsmarginal 16.0 % — kravet är 25 %."]
    for code in lukacs.CLASS_CODES:
        assert lukacs.CLASS_MISSING not in lukacs.gate_gaps(_fv_row(fcf_kvalitet=code), 5.0, "producenter")


def test_the_class_is_only_required_where_the_module_is():
    row = _fv_row(fcf_kvalitet=None)
    assert lukacs.gate_gaps(row, 1.5, "producenter") == []              # under 2 %: frivillig, som förut
    assert lukacs.gate_gaps(row, 5.0, "tiggre") == []


def test_the_scorecard_gate_names_the_missing_class():
    import scorecard as sc
    entry = {"key": "x|X", "ticker": "X", "name": "X", "strategies": ["producenter"],
             "rows": {"producenter": _fv_row(fcf_kvalitet=None)}}
    card = {"strategy": "producenter", "position_pct_total": 5.0}
    assert not sc.gate_state(card, entry)["ready"]
    assert sc.valuation_gaps(entry, card) == [lukacs.CLASS_MISSING]
    entry["rows"]["producenter"]["fcf_kvalitet"] = "B"
    assert sc.valuation_gaps(entry, card) == []


def test_the_review_list_asks_for_the_class():
    from review import missing as mis
    row = dict(_fv_row(fcf_kvalitet=None), id="p1", ticker="EQX", position_pct=3.0)
    assert any(i.path == "fcf_kvalitet" and i.reason == lukacs.CLASS_MISSING for i in mis.missing_for("rule", row))
    assert not any(i.path == "fcf_kvalitet" for i in mis.missing_for("rule", dict(row, fcf_kvalitet="C")))


def test_the_scorecard_valuation_text_follows_the_card():
    import scorecard as sc
    rule = {"id": "r1", "ticker": "AAA", "ev_musd": 620.0, "nav_musd": 1000.0, "framtida_antal_aktier": 100.0,
            "aktuell_kurs": 4.0, "fv": {"Base": {"forward_fcf_musd": 120.0, "target_yield": 12.0}}}
    unset = sc.valuation_suggestion({"rows": {"producenter": rule}}, "producenter")
    assert "— OBS: arket varnar för saknad FCF-klass" in unset
    assert "OBS" not in sc.valuation_suggestion({"rows": {"producenter": dict(rule, fcf_kvalitet="C")}},
                                                "producenter")


# ═════════════════════════════════════════════════════════════════════════════
#  2. AQS — förslag i stället för ifyllnad
# ═════════════════════════════════════════════════════════════════════════════
def test_fraser_top_half_is_one_point_and_insider_ownership_stays_two():
    assert ctl.aqs_prefill_from_producer({"jurisdiktion": True, "insyn": True}) == {
        "aqs_jurisdiktion": 1, "aqs_management": 2}
    assert ctl.aqs_prefill_from_producer({"jurisdiktion": False, "insyn": False}) == {
        "aqs_jurisdiktion": 0, "aqs_management": 0}
    j = [f for f in ctl.AQS_FIELDS if f.key == "aqs_jurisdiktion"][0]
    assert j.one == "Topphalvan" and j.two == "Toppkvartilen"            # 1 p = topphalvan i fältets egen text


def test_tiggre_factors_are_suggested_unchanged():
    assert ctl.aqs_prefill_from_tiggre({"jurisdiktion": 2, "manniskor": 1}) == {
        "aqs_jurisdiktion": 2, "aqs_management": 1}


def test_every_suggestion_says_where_it_comes_from():
    for strategy, notes in ctl.AQS_PREFILL_NOTE.items():
        assert set(notes) == set(ctl.AQS_PREFILLED), strategy
    assert "Fraser-rankingens topphalva" in ctl.AQS_PREFILL_NOTE["producenter"]["aqs_jurisdiktion"]


# ── Rick Rule-sidan (AppTest) ────────────────────────────────────────────────
RULE = {"id": "p1", "ticker": "EQX.TO", "name": "Equinox Gold", "commodity": "Guld", "date": "2026-09-01",
        "price": 4000.0, "unit_cost": 2000.0, "nd_ebitda": 0.4, "position_pct": 3.0}


@pytest.fixture
def rule(monkeypatch):
    """Rick Rule-arket med fejkad lagring — inget nätverk, ingen sifferuppdatering."""
    import gist_storage
    import refresh_ui
    import screens_ui

    def make(row=None, night=None):
        stores = {"producers": {"producers": [dict(RULE, **(row or {}))], "royalty": []},
                  "holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [],
                               "cash": 0}}
        blob = {"rows": {"producers:p1": night}} if night else {}
        monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                            streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
        monkeypatch.setattr(storage, "load_error", lambda name: None)
        monkeypatch.setattr(storage, "is_dirty", lambda name: False)
        monkeypatch.setattr(storage, "last_saved", lambda name: None)
        monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)
        monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob)
        monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
        from streamlit.testing.v1 import AppTest

        def app(root):
            import sys as _s
            _s.path.insert(0, root)
            import producers as p
            p.render_producers_page(sheet="Rick Rule")
        at = AppTest.from_function(app, args=(ROOT,), default_timeout=90)
        at.run()
        assert not at.exception, at.exception
        at.stores = stores
        return at
    return make


def _md(at) -> str:
    return " ".join(m.value for m in at.markdown)


def _row(at) -> dict:
    return at.session_state["producers"]["producers"][0]


def _unwritten(at) -> bool:
    """Arket som det sparades — bortsett från FV-modulens tomma scenarioskelett ({'Bear': {}, …}), som
    lukacs_ui.render_fv lägger dit sedan förut (setdefault) och som inte är ett tal."""
    rows = copy.deepcopy(at.session_state["producers"])
    for r in rows["producers"]:
        if isinstance(r.get("fv"), dict):
            r["fv"] = {k: v for k, v in r["fv"].items() if v}
            if not r["fv"]:
                r.pop("fv")
    return rows == at.stores["producers"]


def test_rule_card_shows_a_dash_for_an_unchosen_class_and_writes_nothing(rule):
    at = rule(row={"framtida_antal_aktier": 1200.0, "aktuell_kurs": 5.0, "what_must_go_right": "x",
                   "fv": {"Base": {"forward_fcf_musd": 900.0, "target_yield": 20.0}}})
    assert at.selectbox(key="fv_p1_class").value is None                 # '—', inte C
    assert any(lukacs.CLASS_MISSING in i.value for i in at.info)
    assert not any("utanför klass" in e.value for e in at.error)        # inget band att mäta mot ännu
    assert _unwritten(at) and "fcf_kvalitet" not in _row(at)
    at.selectbox(key="fv_p1_class").set_value("C").run()                # valet: bandet låser yielden direkt
    assert not at.exception, at.exception
    assert _row(at)["fcf_kvalitet"] == "C"
    assert any("utanför klass C:s band" in e.value for e in at.error)
    assert not any(lukacs.CLASS_MISSING in i.value for i in at.info)


def test_rule_card_with_a_saved_class_is_as_before(rule):
    at = rule(row={"fcf_kvalitet": "B", "framtida_antal_aktier": 1200.0, "aktuell_kurs": 5.0,
                   "fv": {"Base": {"forward_fcf_musd": 900.0, "target_yield": 9.0}}})
    assert at.selectbox(key="fv_p1_class").value == "B"
    assert not any(lukacs.CLASS_MISSING in i.value for i in at.info)
    assert any("Klass B: target-yield 8–10 %" in c.value for c in at.caption)
    assert _unwritten(at)


def test_rule_aqs_suggestions_are_shown_and_nothing_is_filled_in(rule):
    at = rule(row={"jurisdiktion": True, "insyn": True})
    md = _md(at)
    assert "🤖 Kortet: Jurisdiktion 1 (Topphalvan) — ur krysset Jurisdiktion OK (Fraser-rankingens topphalva)" in md
    assert "🤖 Kortet: Management / ägande 2 (Byggt förr, äger själva) — ur krysset Insynsägande" in md
    assert at.selectbox(key="aqs_p1_aqs_jurisdiktion").value == "–"
    assert at.selectbox(key="aqs_p1_aqs_management").value == "–"
    assert _unwritten(at) and "aqs_jurisdiktion" not in _row(at) and "aqs_management" not in _row(at)
    at.run()                                                            # en omritning till skriver inte heller
    assert _unwritten(at)


def test_rule_aqs_use_writes_only_that_field(rule):
    at = rule(row={"jurisdiktion": True, "insyn": True})
    at.button(key="rf_offer_aqs_aqs_p1_aqs_jurisdiktion").click().run()
    assert not at.exception, at.exception
    assert _row(at)["aqs_jurisdiktion"] == 1 and "aqs_management" not in _row(at)
    assert at.selectbox(key="aqs_p1_aqs_jurisdiktion").value == 1
    md = _md(at)
    assert "Jurisdiktion 1 (Topphalvan)" not in md                     # använt: förslaget borta
    assert "Management / ägande 2" in md                               # det andra står kvar
    assert ctl.aqs_total(_row(at)) == 1


def test_rule_aqs_no_suggestion_for_an_answered_field(rule):
    at = rule(row={"jurisdiktion": True, "insyn": False, "aqs_jurisdiktion": 2})
    md = _md(at)
    assert "🤖 Kortet: Jurisdiktion" not in md                          # ditt svar (toppkvartilen) står
    assert "🤖 Kortet: Management / ägande 0 (Serieutspädare, inga meriter) — ur krysset Insynsägande" in md
    assert at.selectbox(key="aqs_p1_aqs_jurisdiktion").value == 2 and _unwritten(at)


def test_rule_aqs_without_the_checkboxes_has_no_suggestions(rule):
    at = rule()
    assert "🤖 Kortet:" not in _md(at) and _unwritten(at)


# ═════════════════════════════════════════════════════════════════════════════
#  Tiggre-sidan (AppTest) — AQS-förslagen och 'Försenad 2:a ggn'
# ═════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def tiggre_page(monkeypatch):
    """Tiggre-arket med fejkad lagring och sifferuppdatering — inget nätverk. Sparningarna spelas in."""
    from streamlit.testing.v1 import AppTest
    import gist_storage
    import refresh_ui
    import screens_ui

    saves: list = []

    def make(stores: dict, blob=None):
        blob = blob or {}
        monkeypatch.setattr(gist_storage, "load_blob", lambda n, default=None: default)
        monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob)
        monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
        monkeypatch.setattr(storage, "load_json", lambda n, default=None: copy.deepcopy(stores.get(n, default)))
        monkeypatch.setattr(storage, "legacy_gist", lambda f: None)

        def _save(n):
            import streamlit as st
            saves.append((n, copy.deepcopy(st.session_state[n])))
            return storage.SaveResult(n, "abc1234def", "https://x", "2026-11-08T10:00:00Z")
        monkeypatch.setattr(storage, "save_session", _save)

        def app(root):
            import sys as _s
            _s.path.insert(0, root)
            import tiggre
            tiggre.render_tiggre_page()
        at = AppTest.from_function(app, args=(ROOT,), default_timeout=90)
        at.run()
        assert not at.exception, at.exception
        return at

    make.saves = saves
    return make


def _registry(**buckets):
    return {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0, **buckets}


def _stores(candidates=(), held=()):
    return {"tiggre": {"candidates": list(candidates), "positions": [], "closed": [], "parked": []},
            "holdings": _registry(tiggre=list(held))}


def test_tiggre_aqs_suggestions_carry_the_factors_and_write_nothing(tiggre_page):
    cand = {"id": "c1", "ticker": "LOB.TO", "name": "Lobo", "screen": {}, "downside": -40.0,
            "factors": {"jurisdiktion": 2, "manniskor": 1}, "catalysts": []}
    stores = _stores(candidates=[cand])
    at = tiggre_page(stores)
    md = _md(at)
    assert "🤖 Kortet: Jurisdiktion 2 (Toppkvartilen) — ur faktorn Jurisdiktion" in md
    assert "🤖 Kortet: Management / ägande 1 (Delvis meriter) — ur faktorn Människor" in md
    assert at.session_state["tiggre"] == stores["tiggre"]               # inget ifyllt när sidan öppnas
    assert not tiggre_page.saves
    at.button(key="rf_offer_aqs_aqs_c1_aqs_management").click().run()
    assert not at.exception, at.exception
    c = at.session_state["tiggre"]["candidates"][0]
    assert c["aqs_management"] == 1 and "aqs_jurisdiktion" not in c
    assert not tiggre_page.saves                                        # 💾 sparar, som förut


def _held(**extras):
    base = {"current": 1.1, "mcap": 50.0, "nav": 400.0, "half_sold": False, "triggers": {}, "catalysts": []}
    return {"id": "t1", "ticker": "LOB.TO", "name": "Lobo", "strategy": "Tiggre", "entry_price": 1.0,
            "shares": 0, "entry_date": "2026-06-01", "source": "tiggre", "extras": {**base, **extras}}


LATE_2 = [{"id": "k1", "name": "Finansieringsbesked", "date": "2026-03", "status": "Försenad 2:a ggn — SÄLJREGEL"}]


def test_opening_tiggre_with_a_second_delay_leaves_the_registry_untouched(tiggre_page):
    stores = _stores(held=[_held(catalysts=LATE_2)])
    at = tiggre_page(stores)
    assert not tiggre_page.saves                                        # förut: positions.put → sparat vid öppning
    held = at.session_state["holdings"]["tiggre"]
    assert len(held) == 1 and held[0]["extras"] == stores["holdings"]["tiggre"][0]["extras"]   # triggern orörd
    md = _md(at)
    assert "Finansieringsbesked står som Försenad 2:a ggn — säljregeln gäller (sälj allt). " \
           "Försenad två gånger — markera?" in md
    assert "SÄLJ ALLT — Katalysator försenad ANDRA gången" in md        # följden syns i brickan
    assert at.checkbox(key="tg_t_t1_delayed_twice").value is False
    assert not any("Sälj allt samma vecka" in e.value for e in at.error)
    at.run()                                                            # en omritning till sparar inte heller
    assert not tiggre_page.saves


def test_marking_the_second_delay_saves_only_on_the_click(tiggre_page):
    at = tiggre_page(_stores(held=[_held(catalysts=LATE_2)]))
    at.button(key="tg_dt_t1").click().run()
    assert not at.exception, at.exception
    assert [n for n, _ in tiggre_page.saves] == ["holdings"]
    saved = tiggre_page.saves[-1][1]["tiggre"][0]
    assert saved["extras"]["triggers"] == {"delayed_twice": True}
    assert at.checkbox(key="tg_t_t1_delayed_twice").value is True
    assert any("Sälj allt samma vecka: Katalysator försenad ANDRA gången" in e.value for e in at.error)
    assert "markera?" not in _md(at) and "tg_dt_t1" not in {b.key for b in at.button}


def test_no_suggestion_when_already_marked_or_not_delayed_twice(tiggre_page):
    at = tiggre_page(_stores(held=[_held(catalysts=LATE_2, triggers={"delayed_twice": True})]))
    assert "markera?" not in _md(at) and not tiggre_page.saves
    late_1 = [dict(LATE_2[0], status="Försenad 1:a ggn")]
    at = tiggre_page(_stores(held=[_held(catalysts=late_1)]))
    assert "markera?" not in _md(at) and not tiggre_page.saves


# Stäng utan Markera: säljorsaken följer brickan (som på main), räknad i minnet vid klicket.
DELAYED_TWICE = "Katalysator försenad ANDRA gången utan god förklaring"


@pytest.fixture
def journal(monkeypatch):
    """journal_bridge.record_close utan Gisten: raden som skickas spelas in, journalraden byggs som förut."""
    import journal_bridge
    rows: list = []

    def _record(row):
        rows.append(copy.deepcopy(row))
        return journal_bridge.trade_from_closed(row)
    monkeypatch.setattr(journal_bridge, "record_close", _record)
    return rows


def _close(at):
    at.button(key="tg_close_t1").click().run()
    assert not at.exception, at.exception
    return at.session_state["tiggre"]["closed"], at.session_state["holdings"]["closed"]


def test_closing_a_second_delay_without_marking_records_the_sell_rule(tiggre_page, journal):
    import journal_bridge
    at = tiggre_page(_stores(held=[_held(catalysts=LATE_2)]))
    assert not tiggre_page.saves                                        # inget skrivet när sidan öppnas
    assert at.checkbox(key="tg_t_t1_delayed_twice").value is False      # Markera inte klickad
    sheet, reg = _close(at)
    assert [c["reason"] for c in sheet] == [DELAYED_TWICE]              # förut (batch 18): 'manuell'
    assert [c["exit_reason"] for c in reg] == [DELAYED_TWICE]
    assert [r["exit_reason"] for r in journal] == [DELAYED_TWICE]
    trade = journal_bridge.trade_from_closed(journal[0])
    assert trade["exit_reason"] == DELAYED_TWICE.lower().replace(" ", "_")
    assert at.session_state["holdings"]["tiggre"] == []


def test_a_ticked_trigger_still_comes_first_and_no_sell_signal_closes_as_manual(tiggre_page, journal):
    at = tiggre_page(_stores(held=[_held(catalysts=LATE_2, triggers={"key_person": True})]))
    sheet, reg = _close(at)
    assert sheet[0]["reason"] == reg[0]["exit_reason"] == "Nyckelperson lämnar"
    late_1 = [dict(LATE_2[0], status="Försenad 1:a ggn")]
    at = tiggre_page(_stores(held=[_held(catalysts=late_1)]))
    sheet, reg = _close(at)
    assert sheet[0]["reason"] == reg[0]["exit_reason"] == journal[-1]["exit_reason"] == "manuell"


# ═════════════════════════════════════════════════════════════════════════════
#  3. Okänd valuta — inget tal i stället för 1,0
# ═════════════════════════════════════════════════════════════════════════════
def test_fx_rate_knows_the_table_and_gives_nothing_for_the_rest():
    assert sr.fx_rate("NOK") == 0.095 and sr.fx_rate("cad") == 0.73
    assert sr.fx_rate(None) == 1.0 and sr.fx_rate("") == 1.0           # saknad valuta = USD, som förut
    for unknown in ("GBX", "ZAR", "JPY"):
        assert sr.fx_rate(unknown) is None


class _FakeBD:
    """Börsdata-klienten med ett bolag per arkrad. Kursvaluta och rapportvaluta väljs per test."""

    def __init__(self, ccy: str, rccy: str = ""):
        self.ccy, self.rccy = ccy, rccy or ccy

    def get_instruments(self):
        return []

    def get_global_instruments_list(self):
        return [{"insId": 900 + i, "ticker": f"T{i}", "name": f"Bolag {i}", "stockPriceCurrency": self.ccy,
                 "reportCurrency": self.rccy} for i in range(10)]

    def resolve_instrument_id(self, form):
        return None

    def get_fundamentals_snapshot_fast(self, ids, scope="nordic"):
        return {i: {"ins_id": i, "market_cap": 5000.0, "ev": 5500.0, "ev_ebitda": 4.0, "net_debt_ebitda": 0.5,
                    "net_debt_m": 500.0, "revenue_m": 3000.0, "fcf_m": 400.0, "ocf_m": 700.0, "roic": 0.12,
                    "p_fcf": 12.5, "ev_ebit": 6.0, "dividend_yield": 0.02, "f_score": 6.0} for i in ids}

    def get_stockprices(self, ins_id, max_count=5):
        return [{"d": ASOF, "c": 1520.0}]

    def get_reports(self, ins_id, kind, max_count=7):
        if kind == "r12":
            return [{"cashAndEquivalents": 800.0, "netDebt": 500.0, "freeCashFlow": -120.0}]
        return [{"year": y, "numberOfShares": n} for y, n in ((2022, 90.0), (2025, 100.0))]

    def get_kpi_history(self, ins_id, kpi, period, calc):
        return [{"v": 8.0}, {"v": 10.0}]


def _every_sheet() -> dict:
    """En rad i varje ark och hink som nattjobbet läser — alla med ins_id ur den globala listan."""
    def r(i, **kw):
        return {"id": f"r{i}", "ticker": f"T{i}", "ins_id": 900 + i, **kw}
    return {"insider": {"signals": [r(0, cluster_avg=1500.0)]},
            "tiggre": {"candidates": [r(1, nav=500.0)],
                       "positions": [r(2, entry=1000.0, current=1500.0, nav=300.0, half_sold=False)]},
            "producers": {"producers": [r(3, nd_ebitda=0.4)],
                          "royalty": [r(4, ev_now=5.0, ev_median=8.0)]},
            "scoring": {"sprott": [r(5)], "durrett": [r(6, mcap=300.0, profit=20.0)]},
            "confidence": {"companies": {"T7": {"ticker": "T7", "ins_id": 907}}},
            "holdings": {"swing": [], "ovtlyr": [], "long": [{"id": "h8", "ticker": "T8", "ins_id": 908,
                                                              "strategy": "Quality", "entry_price": 10.0,
                                                              "shares": 5}],
                         "momentum": [], "tiggre": [], "closed": [], "cash": 0}}


MUSD_FIELDS = ("mcap_musd", "ev_musd", "cash_musd", "debt_musd", "burn_musd", "net_debt_musd", "revenue_musd",
               "fcf_musd", "fcf12_musd", "ocf_musd")


@pytest.mark.parametrize("ccy", ["GBX", "ZAR"])
def test_an_unknown_currency_writes_no_usd_numbers_on_any_sheet(ccy):
    out = sr.refresh(_FakeBD(ccy), _every_sheet())
    assert out["error"] is None
    rows = out["rows"]
    assert len(rows) == 9
    for key, s in rows.items():
        assert s["price"] == 1520.0 and s["currency"] == ccy, key       # kursen i sin egen valuta står kvar
        assert s["shares_now_m"] == 100.0, key                          # aktieantalen räknas inte om
        for f in MUSD_FIELDS:
            assert s.get(f) is None, (key, f)
        assert s.get("fx_to_usd") is None, key
    assert rows["confidence:T7"]["ev_ebitda"] == 4.0 and rows["confidence:T7"]["roic_pct"] == 12.0   # kvoter står kvar
    assert rows["producers:r4"]["ev_ebitda_median"] == 9.0
    assert out["events"] == [] or all(e["kind"] not in ("tiggre_nav_target", "durrett_buy_rule",
                                                        "durrett_engine_buy_rule") for e in out["events"])


def test_a_known_currency_converts_as_before():
    out = sr.refresh(_FakeBD("CAD"), _every_sheet())
    s = out["rows"]["confidence:T7"]
    assert s["mcap_musd"] == 3650.0 and s["ev_musd"] == 4015.0 and s["fx_to_usd"] == 0.73
    assert s["cash_musd"] == 584.0 and s["debt_musd"] == 949.0 and s["burn_musd"] == 87.6
    assert s["fcf_musd"] == 292.0 and s["revenue_musd"] == 2190.0
    assert out["rows"]["producers:r3"]["fcf12_musd"] == 292.0


def test_an_unknown_report_currency_empties_only_the_report_numbers():
    """Kursen i CAD (i tabellen), rapporten i ZAR (utanför): börsvärde och EV räknas om, rapportens tal inte."""
    s = sr.refresh(_FakeBD("CAD", "ZAR"), _every_sheet())["rows"]["confidence:T7"]
    assert s["mcap_musd"] == 3650.0 and s["ev_musd"] == 4015.0 and s["fx_to_usd"] == 0.73
    for f in ("cash_musd", "debt_musd", "burn_musd", "net_debt_musd", "revenue_musd", "fcf_musd", "ocf_musd"):
        assert s[f] is None, f


def test_the_nightly_events_skip_an_unknown_currency():
    """Tiggre-positionen: 1 520 GBX × 5 000 M som USD vore P/NAV långt över 0,8× — inget larm på ett sådant tal."""
    events = sr.refresh(_FakeBD("GBX"), _every_sheet())["events"]
    assert not any(e["kind"] == "tiggre_nav_target" for e in events)
    known = sr.refresh(_FakeBD("CAD"), _every_sheet())["events"]
    assert any(e["kind"] == "tiggre_nav_target" for e in known)       # samma rad med känd valuta larmar som förut


@pytest.mark.parametrize("ccy", ["GBX", "ZAR"])
def test_fetch_row_follows_the_same_rule(ccy):
    s = sr.fetch_row(_FakeBD(ccy), "T3", 903)
    assert s["price"] == 1520.0 and s["currency"] == ccy and s["shares_now_m"] == 100.0
    for f in ("mcap_musd", "ev_musd", "cash_musd", "debt_musd", "burn_musd", "fcf_musd", "revenue_musd"):
        assert s[f] is None, f
    assert s["fx_to_usd"] is None
    assert sr.fetch_row(_FakeBD("NOK"), "T3", 903)["mcap_musd"] == 475.0


def test_asymmetry_fetch_now_proposes_no_usd_numbers_for_an_unknown_currency():
    from asymmetry import fetch
    from confidence.data.models import CompanyInput
    # producent: EV/EBITDA gäller bara producenter och royaltybolag, så bara de får förslaget (batch 8)
    blob, props, msg = fetch.borsdata_now(CompanyInput(ticker="T7", ins_id=907, stage="producer"), _FakeBD("GBX"))
    keys = {k for k, _dp, _cur in props}
    assert blob is not None and "kurs 1520 GBX" in msg
    assert not keys & {"market_cap_musd", "enterprise_value_musd", "fx_to_usd", "cash_musd", "debt_musd",
                       "free_cash_flow_musd", "revenue_musd", "share_price"}
    assert "basic_shares_m" in keys and "ev_ebitda" in keys
    _b, known, _m = fetch.borsdata_now(CompanyInput(ticker="T7", ins_id=907), _FakeBD("CAD"))
    assert {"market_cap_musd", "cash_musd"} <= {k for k, _dp, _cur in known}


def test_asymmetry_yahoo_fallback_gives_no_usd_numbers_for_an_unknown_currency():
    from asymmetry import fetch
    info = {"marketCap": 2_000_000_000, "currency": "ZAR", "financialCurrency": "ZAR", "currentPrice": 150.0,
            "totalCash": 300_000_000, "totalDebt": 100_000_000, "sharesOutstanding": 50_000_000}
    row = fetch.yahoo_row("HAR", lambda sym: info if sym == "HAR" else {})
    assert row["price"] == 150.0 and row["currency"] == "ZAR" and row["shares_now_m"] == 50.0
    assert row["mcap_musd"] is None and row["cash_musd"] is None and row["debt_musd"] is None
    assert row["fx_to_usd"] is None
    mixed = fetch.yahoo_row("HAR", lambda sym: dict(info, currency="USD") if sym == "HAR" else {})
    assert mixed["mcap_musd"] == 2000.0 and mixed["cash_musd"] is None    # rapporten i ZAR: inget MUSD-tal


def test_consumers_take_the_empty_numbers_quietly():
    import controls_ui
    import refresh_ui
    import tiggre
    s = sr.fetch_row(_FakeBD("GBX"), "T3", 903)
    blob = {"rows": {"producers:p1": s, "tiggre:t1": s, "scoring:s1": s}}
    for sheet, rid, field in (("producers", "p1", "ev_musd"), ("tiggre", "t1", "mcap_musd"),
                              ("scoring", "s1", "cash_musd"), ("scoring", "s1", "burn_musd")):
        assert refresh_ui.suggestion(blob, sheet, {"id": rid}, field) is None      # inget Använd-förslag
    assert controls_ui.refresh_runway(s) == (None, "")                  # DS-runwayn: inget förslag
    assert lukacs.ref_fcf_musd(s) is None
    pos = {"id": "t1", "ticker": "T3", "entry": 1000.0, "current": 1100.0, "mcap": 50.0, "nav": 400.0,
           "catalysts": []}
    assert tiggre.export_rows([pos], blob)[0]["_pnav"] == pytest.approx(0.12)      # arkets börsvärde, inte GBX


def test_rick_rule_fetch_ev_says_the_currency_is_unknown(monkeypatch):
    import borsdata_api
    import producers
    state: dict = {}
    monkeypatch.setattr(streamlit, "session_state", state)
    monkeypatch.setattr(borsdata_api, "BorsdataAPI", lambda: _FakeBD("GBX"))
    row = {"id": "p1", "ticker": "T3", "ins_id": 903}
    producers._fetch_ev(row, "k")
    assert "ev_musd" not in row and "k" not in state
    assert state["k_msg"] == "Börsdata har T3 i GBX, som saknas i växelkurstabellen — skriv EV själv i MUSD."
    monkeypatch.setattr(borsdata_api, "BorsdataAPI", lambda: _FakeBD("NOK"))
    producers._fetch_ev(row, "k")
    assert row["ev_musd"] == 522.5 and state["k"] == 522.5
