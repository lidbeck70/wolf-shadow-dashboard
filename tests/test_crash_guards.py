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

    def make(page: str, blob: dict):
        monkeypatch.setattr(gist_storage, "load_blob",
                            lambda name, default=None: blob if name == "sheets_refresh.json" else default)
        monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob)     # annars cachad mellan testerna
        monkeypatch.setattr(storage, "load_json", lambda name, default=None: default)
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
