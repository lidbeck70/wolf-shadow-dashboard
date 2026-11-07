"""Kraschar med delvis ifyllda rader — en lucka eller '–' i stället för ett fel som fäller sidan.

Genomsökningen efter Scorecard-kraschen (lukacs.gate_gaps, #307): rader som importen eller Börsdata-förslagen fyllt
delvis, eller med 0 / negativa tal, nådde rutor med golvet 0, divisioner och talformat. Reglerna är oförändrade:
för varje positivt tal räknas allt som förut — bara det som inte kan räknas blir en lucka.
"""
import copy
import logging
import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Arken: rutor med golvet 0 ────────────────────────────────────────────────
@pytest.fixture
def sheet(monkeypatch):
    """Arksidan med en fejkad sifferuppdatering och lagring — inget nätverk, och inget läcker till andra tester."""
    from streamlit.testing.v1 import AppTest
    import gist_storage
    import refresh_ui
    import storage

    def make(page: str, blob: dict, stores: dict = None):
        """stores = lagren som 'GitHub' har — de läses med storage.session_load som i drift (ren baslinje)."""
        monkeypatch.setattr(gist_storage, "load_blob",
                            lambda name, default=None: blob if name == "sheets_refresh.json" else default)
        monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob)     # annars cachad mellan testerna
        monkeypatch.setattr(storage, "load_json",
                            lambda name, default=None: copy.deepcopy((stores or {}).get(name, default)))
        monkeypatch.setattr(storage, "legacy_gist", lambda f: None)

        def app(page, root):
            import sys as _s
            _s.path.insert(0, root)
            if page in ("Rick Rule", "Royalty C"):
                import producers
                producers.render_producers_page(page)
            elif page == "Sprott":
                import scoring
                scoring.render_scoring_page()
            else:
                import tiggre
                tiggre.render_tiggre_page()
        return AppTest.from_function(app, args=(page, ROOT), default_timeout=90)
    return make


@pytest.mark.parametrize("field, page", [("ev_ebitda", "Rick Rule"), ("ev_musd", "Rick Rule")])
def test_a_negative_borsdata_multiple_is_neither_offered_nor_a_crash(field, page, sheet):
    """Fynd 1/2: Börsdatas EV/EBITDA (negativt EBITDA) eller EV (mer nettokassa än börsvärde) är negativt. 'Använd'
    erbjuds inte — rutan har golvet 0 — och en redan sparad negativ siffra visas som tom i stället för att fälla
    hela Rick Rule-sidan."""
    blob = {"rows": {"producers:r1": {field: -8.4, "asof": "2026-11-06"}}}
    at = sheet(page, blob)
    at.session_state["producers"] = {"producers": [{"id": "r1", "ticker": "AAA", "commodity": "Guld",
                                                    "nav_musd": 400.0}], "royalty": []}
    at.run()
    assert not at.exception, at.exception
    assert not [b for b in at.button if b.key == f"rf_producers_r1_{field}"]           # inget negativt förslag
    at2 = sheet(page, {})
    at2.session_state["producers"] = {"producers": [{"id": "r1", "ticker": "AAA", "commodity": "Guld",
                                                     field: -8.4}], "royalty": []}
    at2.run()
    assert not at2.exception, at2.exception                                            # sparad negativ: ingen krasch


def test_a_positive_suggestion_is_still_offered_and_applied(sheet):
    blob = {"rows": {"producers:r1": {"ev_ebitda": 6.5, "asof": "2026-11-06"}}}
    at = sheet("Rick Rule", blob)
    at.session_state["producers"] = {"producers": [{"id": "r1", "ticker": "AAA", "commodity": "Guld"}], "royalty": []}
    at.run()
    [b for b in at.button if b.key == "rf_producers_r1_ev_ebitda"][0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["producers"]["producers"][0]["ev_ebitda"] == 6.5


def test_royalty_ev_now_negative_is_not_offered_or_a_crash(sheet):
    """Fynd 3: Royalty C:s 'EV/EBITDA nu' ur Börsdatas negativa EV/EBITDA."""
    blob = {"rows": {"producers:k1": {"ev_ebitda": -8.4, "asof": "2026-11-06"}}}
    at = sheet("Royalty C", blob)
    at.session_state["producers"] = {"producers": [], "royalty": [{"id": "k1", "ticker": "RYL", "level": 2,
                                                                   "ev_now": -3.2}]}
    at.run()
    assert not at.exception, at.exception
    assert not [b for b in at.button if b.key == "rf_producers_k1_ev_now"]


def test_fetch_ev_refuses_a_negative_ev(monkeypatch):
    """Fynd 2: 'Hämta EV nu' skriver inte in ett negativt EV — det säger varför."""
    import producers
    import sheets_refresh as sr
    import streamlit as st
    monkeypatch.setattr(sr, "fetch_row", lambda api, ticker, ins_id=None: {"ev_musd": -35.0, "asof": "2026-11-06"})
    import borsdata_api
    monkeypatch.setattr(borsdata_api, "BorsdataAPI", lambda *a, **k: None)
    row = {"id": "r1", "ticker": "AAA"}
    producers._fetch_ev(row, "pr_evm_r1")
    assert "ev_musd" not in row and "negativt EV" in st.session_state["pr_evm_r1_msg"]


def test_sprott_negative_burn_shows_a_gap_not_a_crash(sheet):
    """Fynd 4: en negativ burn (extraktionens 'Använd' med operativt kassaflöde) fäller inte Poängmodellen."""
    at = sheet("Sprott", {})
    at.session_state["scoring"] = {"sprott": [{"id": "q1", "ticker": "AAA", "commodity": "Guld", "factors": {},
                                               "cash": 30.0, "burn": -12.4}], "durrett": []}
    at.run()
    assert not at.exception, at.exception


def test_durrett_helper_does_not_offer_a_negative_profit(sheet):
    """Fynd 4: hjälpräknaren erbjuder inte en negativ framtida vinst (målpris under AISC)."""
    at = sheet("Sprott", {})
    at.session_state["scoring"] = {"sprott": [], "durrett": [{"id": "d1", "ticker": "DDD", "commodity": "Guld",
                                                              "factors": {}, "prod": 100.0, "target": 1000.0,
                                                              "aisc": 1500.0}]}
    import scoring
    at.session_state[scoring.WHICH_KEY] = scoring.WHICH_FOR[scoring.DURRETT]
    at.run()
    assert not at.exception, at.exception
    assert [n for n in at.number_input if n.key == "sc_aisc_d1"]                       # hjälpräknaren ritades
    assert not [b for b in at.button if b.key == "sc_use_d1"]
    assert any("Målpriset ligger under AISC" in c.value for c in at.caption)


def test_tiggre_position_with_a_fresh_market_cap_but_no_date(sheet):
    """Fynd 5: sifferuppdateringen har börsvärde men inget datum (kursen gick inte att hämta) — positionen visas
    med 'datum saknas' och kandidaterna under den ritas."""
    blob = {"rows": {"tiggre:p1": {"ticker": "TGR.TO", "price": None, "asof": None, "mcap_musd": 250.0}}}
    at = sheet("Tiggre", blob)
    at.session_state["tiggre"] = {"candidates": [{"id": "c1", "ticker": "CAND", "screen": {}, "downside": -40.0,
                                                  "factors": {}, "catalysts": []}],
                                  "positions": [], "closed": [], "parked": []}
    at.session_state["holdings"] = {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "closed": [], "cash": 0,
                                    "tiggre": [{"id": "p1", "ticker": "TGR.TO", "strategy": "Tiggre",
                                                "entry_price": 1.2, "shares": 1000, "entry_date": "2026-05-01",
                                                "extras": {"current": 1.4, "mcap": 120.0, "nav": 600.0,
                                                           "catalysts": [], "triggers": {}}}]}
    at.run()
    assert not at.exception, at.exception
    assert not [e for e in at.error if "kunde inte renderas" in e.value]
    helps = [m.help for m in at.metric if m.label == "P/NAV nu"]
    assert helps and "datum saknas" in helps[0]


# ── Motorerna: 0 i en nämnare, text i ett årtal ──────────────────────────────
def _company(stage: str, **fields):
    from confidence.data.models import CompanyInput
    from confidence.data.provenance import dp
    c = CompanyInput(ticker="EQX", name="Equinox", commodity="gold", stage=stage,
                     maturity="production" if stage == "producer" else "construction")
    for k, v in fields.items():
        c.set(k, dp(v, kind="ESTIMATE", source="ChatGPT: rapport", source_type="primary"))
    return c


@pytest.mark.parametrize("fields", [
    {"commodity_price": 2400, "annual_production": 0, "aisc": 1200, "market_cap_musd": 500},
    {"commodity_price": 0, "annual_production": 100000, "aisc": 1200, "market_cap_musd": 500},
    {"commodity_price": 2400, "annual_production": 100000, "aisc": 1200, "market_cap_musd": 500, "target_ev_ebitda": 0},
])
def test_wolf_asymmetry_producer_with_a_zero_divisor(fields):
    """Fynd 6/8: produktion, råvarupris eller EV/EBITDA-mål 0 — scenariomotorn, prismarginalen och break-even
    säger DATA_MISSING i stället för division med noll (Wolf Asymmetrys Analys föll)."""
    import asymmetry
    from confidence import reports
    c = _company("producer", **fields)
    reports.analyze(c)
    asymmetry.analyze(c)


def test_wolf_asymmetry_developer_with_p_nav_zero():
    """Fynd 7: P/NAV vid omvärdering 0 (en bedömning importen tar emot)."""
    from confidence import reports
    rep = reports.analyze(_company("developer", npv_musd=800, commodity_price=2400, market_cap_musd=300,
                                   target_p_nav=0))
    assert rep is not None


def test_price_margin_with_price_zero_is_data_missing():
    import asymmetry
    a = asymmetry.analyze(_company("producer", commodity_price=0, aisc=1200))
    assert a is not None


@pytest.mark.parametrize("stage, fields", [
    ("producer", {"commodity_price": 0, "aisc": 1200, "production_current": 100000}),       # fynd 9
    ("developer", {"commodity_price": 2400, "breakeven_price": 0}),                          # fynd 10
    ("developer", {"basic_shares_m": 0, "shares_1y_ago_m": 100, "shares_3y_ago_m": 90,
                   "shares_5y_ago_m": 80}),                                                  # fynd 11
])
def test_durrett_engine_with_a_zero_divisor(stage, fields):
    """Fynd 9–11: Durrett 10-steg föll för alla bolag när ett hade pris, break-even eller aktieantal 0."""
    from engines.durrett.engine import analyze
    analyze(_company(stage, **fields), today=date(2026, 11, 7))


def test_durrett_results_unchanged_for_positive_values():
    """Reglerna är desamma: med positiva tal ger kostnadssteget samma marginal som förut."""
    from engines.durrett.engine import analyze
    a = analyze(_company("producer", commodity_price=2400, aisc=1200, production_current=100000),
                today=date(2026, 11, 7))
    assert a.cost_score.value is not None


def test_growth_with_a_text_cash_flow_year():
    """Fynd 12: 'Q3 2027' som första kassaflöde-år (arkets fritextfält) fäller inte tillväxtpelaren."""
    from confidence.data.models import CompanyInput
    from confidence.data.provenance import dp
    from confidence.scoring import growth
    c = CompanyInput(ticker="EQX", name="Equinox", commodity="gold", stage="producer", maturity="production")
    c.set("first_cashflow_year", dp("Q3 2027", kind="ESTIMATE", source="arket"))
    growth.score(c)


def test_momentum_caption_shows_text_values():
    """Fynd 13: ett momentumfält med text ('n/a') formateras som text i Durretts 'Mer'-flik, inte med :g."""
    from confidence import ui as cui
    assert cui._fmt("n/a") == "n/a" and cui._fmt(12.5) == "12.5"


# ── Att titta skriver inget (granskningen av rättelserna) ────────────────────
NEG_RULE = {"id": "r1", "ticker": "AAA", "commodity": "Guld", "ev_ebitda": -8.4, "ev_musd": -35.0,
            "unit_cost": -2.5, "nav_musd": -45.0}


@pytest.mark.parametrize("page, stores, store", [
    ("Rick Rule", {"producers": {"producers": [NEG_RULE], "royalty": []}}, "producers"),
    ("Royalty C", {"producers": {"producers": [], "royalty": [{"id": "k1", "ticker": "RYL", "level": 2,
                                                               "ev_now": -3.2, "ev_median": 12.0}]}}, "producers"),
    ("Sprott", {"scoring": {"sprott": [{"id": "q1", "ticker": "AAA", "commodity": "Guld", "factors": {},
                                        "cash": 30.0, "burn": -12.4, "profit": -50.0}], "durrett": []}}, "scoring"),
    ("Tiggre", {"tiggre": {"candidates": [{"id": "c1", "ticker": "CAND", "screen": {}, "downside": -40.0,
                                           "factors": {}, "catalysts": [], "nav": -30.0}],
                           "positions": [], "closed": [], "parked": []}}, "tiggre"),
])
def test_opening_a_sheet_with_stored_negatives_changes_nothing(sheet, page, stores, store):
    """Ett sparat negativt tal (från före golvet) fäller inte sidan, skrivs inte över när arket bara öppnas och gör
    inte arket osparat — det står kvar med en notis tills du skriver in rätt tal."""
    import storage
    at = sheet(page, {}, stores)
    at.run()
    assert not at.exception, at.exception
    at.run()                                                                   # en omritning till
    assert not at.exception, at.exception
    assert at.session_state[store] == stores[store]                           # raden orörd
    fp = storage._fingerprint(at.session_state[store])
    assert fp == at.session_state[storage._clean_key(store)]                  # inget osparat av att titta
    assert any("Sparat men negativt" in c.value for c in at.caption)


def test_the_extractor_does_not_offer_a_negative_number(sheet):
    """Dokumentläsaren erbjuder inte 'Använd' för ett negativt tal (burn som negativt kassaflöde, ett negativt
    NPV) — arkens rutor tar ≥ 0. Ett positivt förslag erbjuds som förut."""
    at = sheet("Sprott", {}, {"scoring": {"sprott": [{"id": "q1", "ticker": "AAA", "commodity": "Guld",
                                                      "factors": {}, "cash": 30.0}], "durrett": []}})
    at.session_state["xt_doc:AAA"] = {"parsed": {"fields": {
        "burn": {"value": "-12,4", "unit": "MUSD", "confidence": "medium"},
        "cash": {"value": "40", "unit": "MUSD", "confidence": "high"}}}}
    at.run()
    assert not at.exception, at.exception
    keys = {b.key for b in at.button}
    assert "xt_sprott_q1_use_burn" not in keys and "xt_sprott_q1_use_cash" in keys
    assert any("negativt — rutan tar ≥ 0" in c.value for c in at.caption)


# ── Domen räknar som rutan visar (sista granskningen) ────────────────────────
def test_a_negative_profit_or_burn_is_a_gap_in_the_verdict():
    """En sparad negativ framtida vinst gav Kärninnehav (−4× < 10×), en negativ burn gav Passa (runway −2,4 år).
    Rutan visar dem som tomma — domen räknar dem nu också som luckor. Positiva tal ger samma dom som förut."""
    import scoring as sc
    nine = {"balans": 2, "vardering": 2, "tillvaxt": 2, "agare": 2, "havstang": 1}
    assert sc.gated_verdict(sc.DURRETT, {"factors": nine, "mcap": 200, "profit": -50})[0] == sc.WATCH
    assert sc.gated_verdict(sc.DURRETT, {"factors": nine, "mcap": 200, "profit": 50})[0] == sc.CORE      # 4× < 10×
    assert sc.gated_verdict(sc.SPROTT, {"factors": nine, "cash": 30, "burn": -12.4})[0] == sc.CORE      # runway okänd
    assert sc.gated_verdict(sc.SPROTT, {"factors": nine, "cash": 30, "burn": 30})[0] == sc.PASS         # 12 mån < 18
    assert sc.runway_years(30, 12) == 2.5 and sc.mcap_per_earnings(200, 50) == 4.0 and sc.mcap_per_oz(200, 2) == 100


def test_a_negative_ev_multiple_is_a_gap_in_the_royalty_signal():
    """En negativ EV/EBITDA (negativt EBITDA) är inte 'billigare än vanligt'. Positiva tal räknas som förut."""
    import producers as pr
    assert pr.vs_median(-3.2, 12.0) is None and pr.vs_median(9.0, 12.0) == -25.0
    assert pr.ev_nav(-35, 400) is None and pr.ev_nav(200, 400) == 0.5


def test_opening_the_durrett_view_writes_no_target_and_no_downside(sheet):
    """Durretts målpris (3000 som förval) och Tiggres nedsida (−40 som förval) skrevs in när arket bara öppnades."""
    import scoring
    import storage
    stores = {"scoring": {"sprott": [], "durrett": [{"id": "d1", "ticker": "DDD", "commodity": "Guld",
                                                     "factors": {}, "profit": -50.0}]}}
    at = sheet("Sprott", {}, stores)
    at.session_state[scoring.WHICH_KEY] = scoring.WHICH_FOR[scoring.DURRETT]
    at.run()
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["scoring"] == stores["scoring"]
    assert storage._fingerprint(at.session_state["scoring"]) == at.session_state[storage._clean_key("scoring")]
    tstores = {"tiggre": {"candidates": [{"id": "c1", "ticker": "CAND", "screen": {}, "factors": {}, "catalysts": []}],
                          "positions": [], "closed": [], "parked": []}}
    at2 = sheet("Tiggre", {}, tstores)
    at2.run()
    at2.run()
    assert not at2.exception, at2.exception
    assert at2.session_state["tiggre"] == tstores["tiggre"]


def test_the_helpers_profit_sticks(sheet):
    """'Använd som framtida vinst' skriver vinsten i raden och rutan — nästa omritning tar inte tillbaka den."""
    import scoring
    stores = {"scoring": {"sprott": [], "durrett": [{"id": "d1", "ticker": "DDD", "commodity": "Guld", "factors": {},
                                                     "profit": 40.0, "prod": 100.0, "target": 3000.0,
                                                     "aisc": 1500.0}]}}
    at = sheet("Sprott", {}, stores)
    at.session_state[scoring.WHICH_KEY] = scoring.WHICH_FOR[scoring.DURRETT]
    at.run()
    [b for b in at.button if b.key == "sc_use_d1"][0].click().run()
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["scoring"]["durrett"][0]["profit"] == 150.0                    # 100 koz × 1500 $/oz
    assert [n.value for n in at.number_input if n.key == "sc_profit_d1"] == [150.0]


def test_a_stored_negative_is_listed_as_missing_for_the_buy_decision():
    """Den sista granskningen: grinden räknar ett sparat negativt tal som en lucka — då står det också i listan
    'saknas för köpbeslut', med skälet. Ett tomt fält listas som förut, ett positivt inte alls."""
    from review import missing as mis
    nine = {"balans": 2, "vardering": 2, "tillvaxt": 2, "agare": 2, "havstang": 1}

    def keys(sheet, row):
        return {it.path: it.reason for it in mis.missing_for(sheet, row)}
    neg = keys("sprott", {"factors": nine, "cash": 30, "burn": -12.4})
    assert "burn" in neg and "Sparat negativt (-12.4)" in neg["burn"]
    assert "burn" in keys("sprott", {"factors": nine, "cash": 30})
    assert "burn" not in keys("sprott", {"factors": nine, "cash": 30, "burn": 12.4})
    assert "Sparat negativt" in keys("durrett", {"factors": nine, "mcap": 200, "profit": -50}).get("profit", "")
    assert "Sparat negativt" in keys("royalty", {"level": 2, "ev_now": -3.2, "ev_median": 12}).get("ev_now", "")


def test_the_tiggre_downside_box_shows_what_is_stored(sheet):
    """Rutan visar den sparade nedsidan — tom när den saknas (U/N-grinden säger då att den saknas) — och det du
    skriver in, också −40, sparas."""
    stores = {"tiggre": {"candidates": [{"id": "c1", "ticker": "CAND", "screen": {}, "factors": {}, "catalysts": [],
                                         "mcap": 100.0, "nav": 650.0}],
                         "positions": [], "closed": [], "parked": []}}
    at = sheet("Tiggre", {}, stores)
    at.run()
    assert not at.exception, at.exception
    box = [n for n in at.number_input if n.key == "tg_down_c1"][0]
    assert box.value is None
    box.set_value(-40.0).run()
    assert not at.exception, at.exception
    assert at.session_state["tiggre"]["candidates"][0]["downside"] == -40.0


def test_every_floor_zero_field_is_a_gap_in_its_rule_and_in_the_missing_list():
    """Genomgången av alla fält med golvet 0 i arken: regeln räknar ett sparat negativt tal som en lucka (som rutan),
    och saknas-listan nämner det. Positiva tal räknas som förut."""
    import producers as pr
    import scoring as sc
    from review import missing as mis
    assert pr.margin_pct(2000, -500) is None and pr.margin_pct(-100, 50) is None and pr.margin_pct(2000, 1000) == 50.0
    assert pr.producer_score({"price": 2000, "unit_cost": -500, "kapitaldisciplin": True}) is None
    assert pr.discount_vs_bottom(-1.0, 0.9) is None and pr.discount_vs_bottom(1.0, 0.8) == 25.0
    assert pr.geo_growth(-0.1, 0.1) is None and pr.geo_growth(1.2, 1.0) == pytest.approx(20.0)
    assert sc.future_profit(100, 3000, -10) is None and sc.future_profit(100, 3000, 1500) == 150000

    def keys(sheet, row):
        return {it.path: it.reason for it in mis.missing_for(sheet, row)}
    rule = {"price": 2000, "unit_cost": 1000, "mine_life": 15, "position_pct": 1, "ev_musd": 500, "nav_musd": -300,
            "kapitaldisciplin": True, "jurisdiktion": True, "insiderägande": True}
    assert "Sparat negativt (-300)" in keys("rule", rule).get("nav_musd", "")
    assert "Sparat negativt (-500)" in keys("rule", {**rule, "nav_musd": 900, "unit_cost": -500}).get("unit_cost", "")
    roy = {"level": 2, "pnav_now": -1.0, "pnav_bottom": 0.9, "geo_now": -0.1, "geo_3y": 0.1}
    got = keys("royalty", roy)
    assert "Sparat negativt" in got.get("pnav_now", "") and "Sparat negativt" in got.get("geo_now", "")


def _var_row(**kw) -> dict:
    row = {"id": "r1", "ticker": "VAR.OL", "commodity": "Olja", "price": 90.0, "unit_cost": 10.0,
           "position_pct": 10.0, "csm_kind": "producer", "fcf_kvalitet": "C", "framtida_antal_aktier": 2744.8,
           "aktuell_kurs": 49.53, "ev_musd": 15051.4, "what_must_go_right": "x",
           "fv": {"Base": {"forward_fcf_musd": 12.0, "target_yield": 12.0},
                  "Bear": {"forward_fcf_musd": 14.0, "target_yield": 14.0},
                  "Bull": {"forward_fcf_musd": 10.0, "target_yield": 10.0}},
           "csm": {"Base": {"fcf_musd": 3550.0, "price": 95.0}, "Bear": {"fcf_musd": 2781.0, "price": 70.0},
                   "Bull": {"fcf_musd": 4165.0, "price": 115.0}}}
    row.update(kw)
    return row


def _job_blob() -> dict:
    """Bloben som nattjobbet skriver den: Rick Rule-raden utan fx_to_usd (granskningens fynd — det handbyggda
    testet med fx_to_usd dolde att förslaget aldrig visades)."""
    from tests.test_lukacs import producer_refresh_row
    ref = producer_refresh_row()
    assert "fx_to_usd" not in ref
    return {"rows": {"producers:r1": ref}}


def _price_warn(at) -> str:
    return " ".join(w.value for w in at.warning if "ser ut att stå i" in w.value)


def test_the_lukacs_panel_warns_about_fcf_units_and_offers_the_price_in_usd(sheet):
    """Vår Energi-fallet: forward FCF 12/14/10 MUSD (yielden i fel ruta) och kursen 49,53 i NOK. Panelen varnar för
    båda, och 'Använd USD' skriver Börsdatas kurs omräknad till USD i raden och rutan."""
    at = sheet("Rick Rule", _job_blob(), {"producers": {"producers": [_var_row()], "royalty": []}})
    at.run()
    assert not at.exception, at.exception
    warns = " ".join(w.value for w in at.warning)
    assert "Base: forward FCF 12 MUSD mot CSM:s Base-FCF 3 550 MUSD" in warns
    assert "Kursen 49,53 ser ut att stå i NOK (Börsdatas kurs: 49,53 NOK)" in _price_warn(at)
    assert not [b for b in at.button if b.key == "rf_producers_r1_aktuell_kurs"]     # ingen NOK-kurs i USD-rutan
    btn = [b for b in at.button if b.label == "Använd USD"]
    assert btn
    btn[0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["producers"]["producers"][0]["aktuell_kurs"] == pytest.approx(4.7054)
    assert at.session_state["fv_r1_price"] == pytest.approx(4.7054)
    assert not _price_warn(at) and not [b for b in at.button if b.label == "Använd USD"]


def test_an_empty_price_box_is_offered_the_usd_price_not_the_nok_one(sheet):
    at = sheet("Rick Rule", _job_blob(), {"producers": {"producers": [_var_row(aktuell_kurs=None)], "royalty": []}})
    at.run()
    assert not at.exception, at.exception
    assert not [b for b in at.button if b.key == "rf_producers_r1_aktuell_kurs"]
    assert any("kurs 49,53 NOK ≈ 4,71 USD" in m.value for m in at.markdown)
    assert not _price_warn(at)
    [b for b in at.button if b.label == "Använd USD"][0].click().run()
    assert at.session_state["producers"]["producers"][0]["aktuell_kurs"] == pytest.approx(4.7054)


def test_a_nok_price_typed_into_the_box_is_flagged_in_the_same_run(sheet):
    """Varningen läser rutans värde i körningen, inte radens förra — den kommer inte ett varv för sent."""
    at = sheet("Rick Rule", _job_blob(), {"producers": {"producers": [_var_row(aktuell_kurs=4.7054)],
                                                        "royalty": []}})
    at.run()
    assert not at.exception and not _price_warn(at)
    at.number_input(key="fv_r1_price").set_value(52.0).run()
    assert not at.exception, at.exception
    assert "Kursen 52,00 ser ut att stå i NOK" in _price_warn(at)
    at.number_input(key="fv_r1_price").set_value(4.75).run()
    assert not _price_warn(at)


def test_a_yahoo_fallback_price_gets_its_currency_from_the_ticker(sheet):
    """Utan Börsdata-träff tar nattjobbet Yahoos kurs utan valuta — VAR.OL är NOK, och förslaget räknas om."""
    blob = {"rows": {"producers:r1": {"ticker": "VAR.OL", "price": 49.53, "asof": "2026-11-07", "currency": None,
                                      "source": "yfinance"}}}
    at = sheet("Rick Rule", blob, {"producers": {"producers": [_var_row()], "royalty": []}})
    at.run()
    assert not at.exception, at.exception
    assert "Kursen 49,53 ser ut att stå i NOK (Yahoos kurs: 49,53 NOK)" in _price_warn(at)
    assert not [b for b in at.button if b.key == "rf_producers_r1_aktuell_kurs"]
    assert [b for b in at.button if b.label == "Använd USD"]


def test_a_usd_listed_producer_keeps_the_plain_suggestion(sheet):
    blob = {"rows": {"producers:r1": {"ticker": "NEM", "price": 52.0, "asof": "2026-11-07", "currency": "USD",
                                      "source": "borsdata"}}}
    at = sheet("Rick Rule", blob, {"producers": {"producers": [_var_row(ticker="NEM", aktuell_kurs=None)],
                                                 "royalty": []}})
    at.run()
    assert not at.exception, at.exception
    assert [b for b in at.button if b.key == "rf_producers_r1_aktuell_kurs"]
    assert not [b for b in at.button if b.label == "Använd USD"] and not _price_warn(at)


def test_a_currency_outside_the_table_is_not_converted(sheet):
    blob = {"rows": {"producers:r1": {"ticker": "5711.T", "price": 1500.0, "asof": "2026-11-07", "currency": "JPY",
                                      "source": "borsdata", "fx_to_usd": 1.0}}}
    at = sheet("Rick Rule", blob, {"producers": {"producers": [_var_row(aktuell_kurs=1500.0)], "royalty": []}})
    at.run()
    assert not at.exception, at.exception
    assert "ser ut att stå i JPY" in _price_warn(at) and "Skriv kursen i USD" in _price_warn(at)
    assert not [b for b in at.button if b.label == "Använd USD" or b.key == "rf_producers_r1_aktuell_kurs"]
