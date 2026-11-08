"""
Automatisering, batch 1 — förslag som redan fanns men inte var inkopplade.

1. Poängmodellen: DS 'Aktier 3 år' och 'Historik' får Börsdatas aktiehistorik (scoring:<id>).
2. Tiggre och 3. Rick Rule: DS-runwayn räknas ur nattradens kassa och burn (tiggre:<id>, producers:<id>).
4. Swing: 'Kurs nu' får nattens kurs (holdings:<id>) med Använd, som Tiggres positioner.
5. Wolf Asymmetry: 'Förslag ur granskningsarken' (Rick Rule/Tiggre) som Durrett-arket redan visar.
6. Durrett-arket: rätt text om ins_id, och Använd för nattradens Börsdata-id.

Inget skrivs när en sida öppnas; reglerna och trösklarna är oförändrade.
"""
import copy
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.CRITICAL)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASOF = "2026-11-07"


# ── rena funktioner ──────────────────────────────────────────────────────────
def test_refresh_runway_is_cash_over_burn_with_its_source():
    import controls as ctl
    import controls_ui as cu
    years, note = cu.refresh_runway({"cash_musd": 12.0, "burn_musd": 8.0, "asof": ASOF + "T06:00"})
    assert years == pytest.approx(1.5)
    assert note == (f"Börsdata {ASOF}: kassa 12,0 MUSD ÷ burn 8,0 MUSD/år (negativt FCF 12 mån) "
                    f"= runway 18 mån")
    assert ctl.ds_runway_suggestion(years) == 1                         # fältets egna gränser, oförändrade
    big = cu.refresh_runway({"cash_musd": 1250.0, "burn_musd": 100.0})
    assert big[0] == pytest.approx(12.5) and big[1].startswith("Börsdata: kassa 1 250,0 MUSD")


@pytest.mark.parametrize("row", [
    {"cash_musd": 30.0, "burn_musd": 0.0},           # positivt FCF: burn 0 → ingen runway (som förut)
    {"cash_musd": 30.0},                              # burn saknas
    {"burn_musd": 8.0},                               # kassa saknas
    {"cash_musd": -5.0, "burn_musd": 8.0},            # negativ kassa är en lucka, som i Sprott-rutan
    None, "inte en rad"])
def test_refresh_runway_without_both_numbers_gives_nothing(row):
    import controls_ui as cu
    assert cu.refresh_runway(row) == (None, "")


def test_runway_months_never_round_across_the_ds_limits():
    import controls_ui as cu
    assert cu._runway_months(1.5) == "18"
    assert cu._runway_months(11.96 / 12) == "11,96"          # inte '12' bredvid '< 12 mån'
    assert cu._runway_months(24.04 / 12) == "24,04"          # inte '24' bredvid '> 24 mån'
    assert cu._runway_months(30.4 / 12) == "30"


def test_sheet_runway_reads_the_rows_own_key(monkeypatch):
    import controls_ui as cu
    import refresh_ui
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {"rows": {
        "tiggre:c1": {"cash_musd": 6.0, "burn_musd": 12.0, "asof": ASOF}}})
    assert cu.sheet_runway("tiggre", "c1")[0] == pytest.approx(0.5)
    assert cu.sheet_runway("tiggre", "c2") == (None, "") and cu.sheet_runway("producers", "c1") == (None, "")
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: None)
    assert cu.sheet_runway("tiggre", "c1") == (None, "")


def test_ins_id_offer_only_when_the_company_lacks_one():
    from confidence.data.models import CompanyInput
    from engines.durrett import refresh as dr
    c = CompanyInput(ticker="GPR")
    assert dr.ins_id_offer({"ins_id": 105}, c) == 105
    assert dr.ins_id_offer({"ins_id": "105"}, c) == 105
    for row in ({"ins_id": None, "source": "yfinance"}, {}, None, {"ins_id": "x"}, {"ins_id": 0}, {"ins_id": True}):
        assert dr.ins_id_offer(row, c) is None, row
    assert dr.ins_id_offer({"ins_id": 105}, CompanyInput(ticker="GPR", ins_id=105)) is None
    assert dr.ins_id_offer({"ins_id": 999}, CompanyInput(ticker="GPR", ins_id=105)) is None   # ditt id står


# ── sidorna (AppTest) ────────────────────────────────────────────────────────
@pytest.fixture
def page(monkeypatch):
    """Arksidan med fejkad sifferuppdatering och lagring — inget nätverk. Sparningar spelas in."""
    from streamlit.testing.v1 import AppTest
    import gist_storage
    import lukacs_ui
    import refresh_ui
    import screens_ui
    import storage

    saves: list = []

    def make(name: str, blob: dict, stores: dict):
        monkeypatch.setattr(lukacs_ui, "_live_fx", lambda ccy: None)                  # ingen dagskurs ur nätet
        monkeypatch.setattr(gist_storage, "load_blob",
                            lambda n, default=None: blob if n == "sheets_refresh.json" else default)
        monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob)
        monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
        monkeypatch.setattr(storage, "load_json", lambda n, default=None: copy.deepcopy(stores.get(n, default)))
        monkeypatch.setattr(storage, "legacy_gist", lambda f: None)

        def _save(n):
            import streamlit as st
            saves.append((n, copy.deepcopy(st.session_state[n])))
            return storage.SaveResult(n, "abc1234def", "https://x", "2026-11-08T10:00:00Z")
        monkeypatch.setattr(storage, "save_session", _save)

        def app(name, root):
            import sys as _s
            _s.path.insert(0, root)
            if name == "Rick Rule":
                import producers
                producers.render_producers_page(name)
            elif name == "Poängmodell":
                import scoring
                scoring.render_scoring_page()
            elif name == "Tiggre":
                import tiggre
                tiggre.render_tiggre_page()
            elif name == "Swing":
                import swing
                swing.render_swing_page()
            else:
                from engines.durrett.ui import render_durrett_page
                render_durrett_page()
        return AppTest.from_function(app, args=(name, ROOT), default_timeout=90)

    make.saves = saves
    return make


def _captions(at) -> str:
    return " ".join(c.value for c in at.caption)


def _markdown(at) -> str:
    return " ".join(m.value for m in at.markdown)


# 1. Poängmodellen
@pytest.mark.parametrize("bucket", ["sprott", "durrett"])
def test_scoring_ds_offers_the_share_history_without_writing(bucket, page):
    import scoring
    row = {"id": "q1", "ticker": "AAA", "commodity": "Guld", "factors": {}}
    blob = {"rows": {"scoring:q1": {"asof": ASOF, "shares_growth_3y_pct": 27.3, "shares_growth_5y_pct": 60.0}}}
    stores = {"scoring": {"sprott": [], "durrett": [], bucket: [copy.deepcopy(row)]}}
    at = page("Poängmodell", blob, stores)
    at.session_state[scoring.WHICH_KEY] = scoring.WHICH_FOR[bucket]
    at.run()
    assert not at.exception, at.exception
    text = _captions(at)
    assert (f"Börsdata {ASOF}: antalet aktier är +27 % mot för tre år sedan — motsvarar 2 riskpoäng på "
            f"Aktier 3 år") in text
    assert "+60 % på fem år — motsvarar 2 riskpoäng på Historik (serieutspädare)" in text
    saved = at.session_state["scoring"][bucket][0]
    assert "ds_aktier_3ar" not in saved and "ds_historik" not in saved           # förslag, inte ifyllt
    assert not page.saves


def test_scoring_ds_share_text_hides_once_the_field_is_set(page):
    row = {"id": "q1", "ticker": "AAA", "commodity": "Guld", "factors": {}, "ds_aktier_3ar": 1, "ds_historik": 0}
    blob = {"rows": {"scoring:q1": {"asof": ASOF, "shares_growth_3y_pct": 27.3, "shares_growth_5y_pct": 60.0}}}
    at = page("Poängmodell", blob, {"scoring": {"sprott": [row], "durrett": []}})
    at.run()
    assert not at.exception, at.exception
    assert "riskpoäng på Aktier 3 år" not in _captions(at) and "riskpoäng på Historik" not in _captions(at)


# 2. Tiggre
def _tiggre_store(**cand):
    return {"tiggre": {"candidates": [{"id": "c1", "ticker": "TGR", "screen": {}, "downside": -40.0,
                                       "factors": {}, "catalysts": [], **cand}],
                       "positions": [], "closed": [], "parked": []},
            "holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [],
                         "cash": 0}}


def test_tiggre_ds_runway_text_from_the_nightly_cash_and_burn(page):
    blob = {"rows": {"tiggre:c1": {"ticker": "TGR", "asof": ASOF, "cash_musd": 12.0, "burn_musd": 8.0}}}
    at = page("Tiggre", blob, _tiggre_store())
    at.run()
    assert not at.exception, at.exception
    assert (f"Börsdata {ASOF}: kassa 12,0 MUSD ÷ burn 8,0 MUSD/år (negativt FCF 12 mån) = runway 18 mån — "
            f"motsvarar 1 riskpoäng på Runway. Förslag, inte ifyllt.") in _captions(at)
    assert "ds_runway" not in at.session_state["tiggre"]["candidates"][0]
    assert not page.saves


@pytest.mark.parametrize("fresh", [{"cash_musd": 12.0, "burn_musd": 0.0}, {}])
def test_tiggre_without_burn_shows_no_runway_text(fresh, page):
    blob = {"rows": {"tiggre:c1": {"ticker": "TGR", "asof": ASOF, **fresh}}}
    at = page("Tiggre", blob, _tiggre_store())
    at.run()
    assert not at.exception, at.exception
    assert "riskpoäng på Runway" not in _captions(at) and "Runwayen motsvarar" not in _captions(at)


def test_tiggre_runway_text_hides_once_ds_runway_is_set(page):
    blob = {"rows": {"tiggre:c1": {"ticker": "TGR", "asof": ASOF, "cash_musd": 12.0, "burn_musd": 8.0}}}
    at = page("Tiggre", blob, _tiggre_store(ds_runway=2))
    at.run()
    assert not at.exception, at.exception
    assert "riskpoäng på Runway" not in _captions(at)


# 3. Rick Rule
@pytest.mark.parametrize("burn, expect", [(40.0, "runway 9 mån — motsvarar 2 riskpoäng på Runway"), (0.0, None)])
def test_rick_rule_ds_runway_text_from_the_nightly_row(burn, expect, page):
    blob = {"rows": {"producers:r1": {"ticker": "AAA", "asof": ASOF, "cash_musd": 30.0, "burn_musd": burn}}}
    at = page("Rick Rule", blob, {"producers": {"producers": [{"id": "r1", "ticker": "AAA", "commodity": "Guld"}],
                                                "royalty": []}})
    at.run()
    assert not at.exception, at.exception
    if expect:
        assert expect in _captions(at)
    else:
        assert "riskpoäng på Runway" not in _captions(at)
    assert "ds_runway" not in at.session_state["producers"]["producers"][0]


# 4. Swing
def _swing_stores():
    reg = {"swing": [], "ovtlyr": [], "long": [], "tiggre": [], "closed": [], "cash": 0,
           "momentum": [{"id": pid, "ticker": t, "strategy": "Momentum", "entry_price": 250.0, "shares": 0,
                         "entry_date": "2026-10-01", "extras": {"current": 260.0, "setup": "A"}}
                        for pid, t in (("m1", "VOLV-B.ST"), ("m2", "ERIC-B.ST"))]}
    swing_data = {"market": {"aboveMA200": True, "checked": "2026-11-01"}, "checklist": {"week": "", "done": []},
                  "watchlist": [], "positions": [], "closed": []}
    return {"holdings": reg, "swing": swing_data}


def test_swing_current_price_suggestion_saves_only_on_click(page):
    blob = {"rows": {"holdings:m1": {"ticker": "VOLV-B.ST", "price": 270.5, "currency": "SEK", "asof": ASOF,
                                     "source": "borsdata"},
                     "holdings:m2": {"ticker": "ERIC-B.ST", "price": 88.0, "currency": "SEK", "asof": ASOF,
                                     "source": "borsdata"}}}
    at = page("Swing", blob, _swing_stores())
    at.run()
    assert not at.exception, at.exception
    assert f"🤖 Börsdata {ASOF}: kurs 270.5 SEK" in _markdown(at)
    assert {b.key for b in at.button} >= {"rf_holdings_m1_current", "rf_holdings_m2_current"}
    assert not page.saves                                                       # att öppna sparar inget
    at.button(key="rf_holdings_m1_current").click().run()
    assert not at.exception, at.exception
    assert at.number_input(key="swing_cur_m1").value == 270.5
    assert at.number_input(key="swing_cur_m2").value == 260.0
    saved = [s for n, s in page.saves if n == "holdings"]
    assert len(saved) == 1                                                      # ett klick, en sparning
    rows = {r["id"]: r for r in saved[0]["momentum"]}
    assert rows["m1"]["extras"]["current"] == 270.5                             # rätt position, inte den sista
    assert rows["m2"]["extras"]["current"] == 260.0
    assert not [b for b in at.button if b.key == "rf_holdings_m1_current"]      # samma tal: förslaget döljs


def test_swing_without_a_nightly_row_shows_no_suggestion(page):
    at = page("Swing", {}, _swing_stores())
    at.run()
    assert not at.exception, at.exception
    assert not [b for b in at.button if b.key.startswith("rf_holdings_")]


def test_the_nightly_job_keys_momentum_positions_like_the_swing_page():
    """sheets_refresh hämtar registrets momentum-hink som holdings:<id> — samma id som Swing-vyn har."""
    import positions
    import sheets_refresh as sr
    assert positions.STRATEGY_BUCKET["Momentum"] == "momentum" and "momentum" in sr._BUCKETS["holdings"]
    reg = _swing_stores()["holdings"]
    keys = {r["key"] for r in sr.collect_rows({"holdings": reg})}
    view_ids = {v["id"] for v in positions.view_rows_from(reg, "Momentum")}
    assert keys == {f"holdings:{i}" for i in view_ids} == {"holdings:m1", "holdings:m2"}


# 6. Durrett-arket
def _durrett_store():
    import durrett_cases as dcs
    from confidence import store as cs
    data = cs.default()
    cs.put(data, dcs.gold_producer())                                           # utan ins_id
    return {"confidence": data, "producers": {}, "tiggre": {}}


def test_durrett_offers_the_nightly_ins_id_and_saves_it_on_click(page):
    blob = {"generated": "2026-11-07T06:00", "rows": {"confidence:GPR": {
        "ticker": "GPR", "ins_id": 105, "price": 6.4, "currency": "USD", "asof": ASOF, "source": "borsdata"}}}
    at = page("Durrett", blob, _durrett_store())
    at.session_state["durrett_row_sub_GPR"] = "Arket"
    at.run()
    assert not at.exception, at.exception
    assert (f"🤖 Börsdata {ASOF}: tickern matchades mot Börsdata-id 105 (senaste kurs 6.4 USD). Med id:t på "
            f"bolaget hämtas samma bolag varje natt.") in _captions(at)
    assert at.session_state["confidence"]["companies"]["GPR"]["ins_id"] is None   # att öppna skriver inget
    at.button(key="durrett_insid_GPR").click().run()
    assert not at.exception, at.exception
    assert at.session_state["confidence"]["companies"]["GPR"]["ins_id"] == 105
    assert not [b for b in at.button if b.key == "durrett_insid_GPR"]          # sparat: inget förslag kvar


def test_durrett_without_a_row_points_to_a_field_that_exists(page):
    at = page("Durrett", {}, _durrett_store())
    at.session_state["durrett_row_sub_GPR"] = "Arket"
    at.run()
    assert not at.exception, at.exception
    text = _captions(at)
    assert "Mer → Alla fält" not in text
    assert "🐺 Wolf Asymmetry → Ark → Identitet och strategi" in text
    # fältet som texten pekar på finns där
    src = open(os.path.join(ROOT, "asymmetry", "ui.py"), encoding="utf-8").read()
    assert 'with st.expander("Identitet och strategi"' in src and '"Börsdata-id (ins_id)"' in src
