"""
Automatisering, batch 13 — Royalty C: P/NAV och GEO per aktie med Börsdatas aktieantal och kurs.

1. GEO-räknaren under GEO: total GEO nu och för tre år sedan i koz (tillfälliga rutor, sparas inte) → GEO/aktie nu
   och för 3 år sedan = koz × 1 000 / (miljoner aktier × 1 000 000), i kortets enhet (uns per aktie). Bara som par,
   med en Använd. Aktieåret visas, och att GEO-året kan skilja sig. Uns i stället för koz, eller styck i stället för
   miljoner, ger inget förslag.
2. NAV per aktie och valuta (tillfälliga) → 'P/NAV nu = kurs / NAV per aktie' med Använd, bara när kursens valuta
   är NAV:ns (NYSE i USD, TSX i CAD).
3. 'Ny egen botten?' med Använd när P/NAV nu ligger under den sparade botten — med påminnelsen om samma NAV-bas.

Inget skrivs när sidan öppnas. Signalen och nattlarmet läser de sparade fälten: det räknade talet når dem först efter
Använd. Inga grindar, trösklar eller formler ändrade.
"""
import copy
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import producers as p    # noqa: E402
import storage           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASOF = "2026-11-07"
NIGHT = {"ticker": "FNV", "price": 120.5, "currency": "USD", "asof": ASOF, "source": "borsdata",
         "shares_now_m": 192.4, "shares_3y_ago_m": 191.0, "shares_year": 2025}


# ── 1. GEO per aktie (rent) ──────────────────────────────────────────────────
def test_geo_per_share_is_koz_times_thousand_over_shares_in_millions():
    assert p.geo_per_share(650, 192.4) == pytest.approx(650_000 / 192_400_000)
    assert p.geo_per_share(650, 192.4) == 0.00337838                      # sex värdesiffror, uns per aktie
    assert p.geo_per_share(7, 170.0) == pytest.approx(7_000 / 170_000_000, rel=1e-5)
    for bad in ((None, 10.0), (650, None), (0, 10.0), (650, 0), (-5, 10.0), (650, -1)):
        assert p.geo_per_share(*bad) is None


def test_geo_pair_suggestion_gives_both_numbers_and_shows_the_inputs():
    s = p.geo_pair_suggestion(650, 560, NIGHT)
    assert set(s.values) == {"geo_now", "geo_3y"}
    assert s.values["geo_now"] == p.geo_per_share(650, 192.4)
    assert s.values["geo_3y"] == p.geo_per_share(560, 191.0)
    growth = p.geo_growth(s.values["geo_now"], s.values["geo_3y"])
    assert growth == pytest.approx((650 / 192.4) / (560 / 191.0) * 100 - 100, rel=1e-4)
    assert s.text == ("🤖 Räknat med Börsdatas aktieantal: GEO/aktie nu 0,003378 (650 koz / 192,40 milj. aktier 2025) "
                      "och för 3 år sedan 0,002932 (560 koz / 191,00 milj. 2022) = +15,2 %")
    assert s.note == ""


def test_geo_pair_suggestion_without_the_share_year_still_shows_the_numbers():
    s = p.geo_pair_suggestion(90, 100, {"shares_now_m": 100.0, "shares_3y_ago_m": 100.0})
    assert s.text == ("🤖 Räknat med Börsdatas aktieantal: GEO/aktie nu 0,000900 (90 koz / 100,00 milj. aktier) och för "
                      "3 år sedan 0,001000 (100 koz / 100,00 milj.) = −10,0 %")


def test_geo_pair_is_offered_only_as_a_pair():
    assert p.geo_pair_suggestion(None, None, NIGHT) == p.Calc({})
    for one in ((650, None), (None, 560), (650, 0)):
        s = p.geo_pair_suggestion(*one, NIGHT)
        assert s.values == {} and "bara som par" in s.text


@pytest.mark.parametrize("fresh", [{}, None, {"shares_now_m": 192.4}, {"shares_now_m": 0, "shares_3y_ago_m": 191.0}])
def test_geo_pair_needs_both_share_counts(fresh):
    s = p.geo_pair_suggestion(650, 560, fresh)
    assert s.values == {} and "aktieantalet" in s.text


def test_geo_pair_guards_ounces_typed_as_koz():
    """650 000 uns skrivet i koz-rutan hade gett 1000× för hög GEO per aktie — inget förslag."""
    s = p.geo_pair_suggestion(650_000, 560_000, NIGHT)
    assert s.values == {} and "uns" in s.text and "650000 koz" in s.text
    mixed = p.geo_pair_suggestion(650, 560_000, NIGHT)                     # ett i koz, ett i uns
    assert mixed.values == {} and "uns" in mixed.text
    half = p.geo_pair_suggestion(650, 0.56, NIGHT)                         # ett i koz, ett i Moz
    assert half.values == {} and "står båda i koz" in half.text


def test_geo_pair_guards_shares_in_units_instead_of_millions():
    s = p.geo_pair_suggestion(650, 560, dict(NIGHT, shares_now_m=192_400_000.0, shares_3y_ago_m=191_000_000.0))
    assert s.values == {} and "styck" in s.text
    jump = p.geo_pair_suggestion(650, 560, dict(NIGHT, shares_3y_ago_m=191_000.0))
    assert jump.values == {} and "styck" in jump.text


def test_geo_pair_warns_when_the_saved_value_is_a_thousand_times_off():
    s = p.geo_pair_suggestion(650, 560, NIGHT, saved_now=3.378)
    assert s.values and "mer än 100× från det sparade" in s.note
    assert p.geo_pair_suggestion(650, 560, NIGHT, saved_now=0.0031).note == ""
    moz = p.geo_pair_suggestion(0.65, 0.56, NIGHT)                            # Moz i koz-rutorna: 1000× för lågt
    assert moz.values and moz.note.startswith("Under 1 koz — står talen i Moz?")


def test_shares_note_names_the_years_and_the_basis():
    assert p.shares_note(NIGHT) == ("Börsdatas aktieantal: 192,40 milj. (2025) och 191,00 milj. (2022), från "
                                    "årsrapporterna (basic vid årsskiftet, inte vägt). Ta GEO för samma år — "
                                    "GEO-året kan skilja sig från aktieåret.")
    assert "nu och 191,00 milj. för tre år sedan" in p.shares_note(dict(NIGHT, shares_year=None))
    for bad in ({}, None, {"shares_now_m": 1.0}):
        assert p.shares_note(bad) == ""


# ── 2. P/NAV nu (rent) ───────────────────────────────────────────────────────
def test_pnav_is_price_over_navps_in_the_same_currency():
    s = p.pnav_suggestion(NIGHT, 147.0, "usd", bottom=0.75)
    assert s.values == {"pnav_now": 0.82}                                  # 120,5 / 147 = 0,8197 → två decimaler
    assert s.text == ("🤖 P/NAV nu = kurs 120,50 USD (Börsdata 2026-11-07) / NAV per aktie 147 USD = 0,82 — mot "
                      "botten 0,75: +9,3 %")
    assert p.pnav_suggestion(NIGHT, 147.0, "USD").text.endswith("= 0,82")  # utan botten: ingen grindrad


def test_pnav_is_not_offered_when_the_currencies_differ():
    tsx = dict(NIGHT, price=165.0, currency="CAD")
    s = p.pnav_suggestion(tsx, 147.0, "USD")
    assert s.values == {}
    assert "Kursen är i CAD, NAV per aktie i USD" in s.text and "NYSE" in s.text and "TSX" in s.text


def test_pnav_needs_a_currency_on_both_sides_and_a_price():
    assert p.pnav_suggestion(NIGHT, None, "USD") == p.Calc({})
    assert p.pnav_suggestion(NIGHT, 0, "USD") == p.Calc({})
    assert "Välj NAV:ns valuta" in p.pnav_suggestion(NIGHT, 147.0, "").text
    yahoo = dict(NIGHT, currency=None, source="yfinance")
    assert p.pnav_suggestion(yahoo, 147.0, "USD").values == {}
    assert "valuta är okänd" in p.pnav_suggestion(yahoo, 147.0, "USD").text
    assert "Ingen kurs" in p.pnav_suggestion(dict(NIGHT, price=None), 147.0, "USD").text
    assert p.pnav_suggestion(None, 147.0, "USD").values == {}


def test_pnav_guards_a_total_nav_typed_as_per_share():
    s = p.pnav_suggestion(NIGHT, 28_000.0, "USD")                         # total NAV i miljoner
    assert s.values == {} and "orimligt" in s.text
    assert p.pnav_suggestion(NIGHT, 1.0, "USD").values == {}               # P/NAV 120 — fel enhet


# ── 3. Ny egen botten (rent) ─────────────────────────────────────────────────
def test_new_bottom_only_below_the_saved_bottom():
    assert p.new_bottom(0.70, 0.75) == 0.70
    assert p.new_bottom(0.75, 0.75) is None
    assert p.new_bottom(0.80, 0.75) is None
    assert p.new_bottom(0.70, None) is None and p.new_bottom(0.70, 0) is None   # ingen botten sparad
    assert p.new_bottom(None, 0.75) is None and p.new_bottom(-0.1, 0.75) is None


def test_offer_fields_compares_small_numbers_relatively():
    """GEO per aktie för ett litet royaltybolag (0,0000412 mot 0,0000418) är inte 'samma tal' — tillväxten kan vända."""
    import refresh_ui
    assert not refresh_ui._same_rel(0.0000412, 0.0000418)
    assert refresh_ui._same_rel(0.00337838, 0.00337838) and refresh_ui._same_rel(0.0, 0.0)
    assert not refresh_ui._same_rel(None, 0.1) and not refresh_ui._same_rel("", 0.1)
    assert refresh_ui.offer_fields({}, {"geo_now": None, "geo_3y": 0.1}, "x", "k") is False      # bara hela paret
    assert refresh_ui.offer_fields({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0}, "x", "k") is False  # redan inskrivet


def test_the_royalty_rules_are_untouched():
    assert p.DISCOUNT_MAX == 15.0
    row = {"pnav_now": 1.15, "pnav_bottom": 1.0, "ev_now": 18.0, "ev_median": 20.0, "geo_now": 0.05, "geo_3y": 0.04}
    assert p.royalty_signal(row).label == p.R_BUY


# ── sidan (AppTest) ──────────────────────────────────────────────────────────
@pytest.fixture
def page(monkeypatch):
    """Royalty C med fejkad sifferuppdatering och lagring — inget nätverk. Sparningar till repot spelas in."""
    from streamlit.testing.v1 import AppTest
    import gist_storage
    import refresh_ui
    import screens_ui

    saves: list = []

    def make(blob: dict, stores: dict):
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

        def app(root):
            import sys as _s
            _s.path.insert(0, root)
            import producers
            producers.render_producers_page("Royalty C")
        return AppTest.from_function(app, args=(ROOT,), default_timeout=90)

    make.saves = saves
    return make


def _store(**row) -> dict:
    base = {"id": "r1", "ticker": "FNV", "name": "Franco", "level": 1, "date": "2026-09-01"}
    return {"producers": {"producers": [], "royalty": [dict(base, **row)]}}


def _blob(**night) -> dict:
    return {"rows": {"producers:r1": dict(NIGHT, **night)}}


def _captions(at) -> str:
    return " ".join(c.value for c in at.caption)


def _markdown(at) -> str:
    return " ".join(m.value for m in at.markdown)


def _btn(at, key):
    return next((b for b in at.button if b.key == key), None)


def _row(at) -> dict:
    return at.session_state["producers"]["royalty"][0]


def _head(at) -> str:
    """Kortets rubrik: 'FNV · nivå 1 · <signal>'."""
    return next(e.label for e in at.expander if e.label.startswith("FNV · "))


def test_opening_the_card_writes_nothing_and_offers_nothing(page):
    store = _store(pnav_now=0.9, pnav_bottom=0.8, geo_now=0.003, geo_3y=0.0029)
    at = page(_blob(), store)
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["producers"] == store["producers"]                 # inget skrivet
    assert at.number_input(key="ro_navps_r1").value is None                    # tillfälliga rutor, tomma
    assert at.number_input(key="ro_geokoz_r1").value is None
    assert _btn(at, "rf_offer_ro_pnav_calc_r1") is None and _btn(at, "rf_offer_ro_geo_calc_r1") is None
    assert _btn(at, "rf_offer_ro_newbottom_r1") is None
    assert "Kurs 120,50 USD · 2026-11-07" in _captions(at)
    assert "192,40 milj. (2025) och 191,00 milj. (2022)" in _captions(at)
    assert p.NAV_BASIS_NOTE in _captions(at)
    assert not page.saves


# 1. GEO-räknaren
def test_geo_calculator_offers_the_pair_and_one_click_writes_both(page):
    store = _store(geo_now=0.05, geo_3y=0.04)
    at = page(_blob(), store)
    at.run()
    assert at.number_input(key="ro_geokoz_r1").label == "Total GEO 2025 (koz)"
    assert at.number_input(key="ro_geokoz3_r1").label == "Total GEO 2022 (koz)"
    at.number_input(key="ro_geokoz_r1").set_value(650.0).run()
    assert not at.exception, at.exception
    assert _btn(at, "rf_offer_ro_geo_calc_r1") is None                       # bara som par
    assert "bara som par" in _captions(at)
    at.number_input(key="ro_geokoz3_r1").set_value(560.0).run()
    assert not at.exception, at.exception
    assert "GEO/aktie nu 0,003378 (650 koz / 192,40 milj. aktier 2025)" in _markdown(at)
    assert _row(at)["geo_now"] == 0.05 and _row(at)["geo_3y"] == 0.04        # inget skrivet förrän Använd
    _btn(at, "rf_offer_ro_geo_calc_r1").click().run()
    assert not at.exception, at.exception
    row = _row(at)
    assert row["geo_now"] == p.geo_per_share(650, 192.4) and row["geo_3y"] == p.geo_per_share(560, 191.0)
    assert at.number_input(key="ro_gn_r1").value == row["geo_now"]
    assert at.number_input(key="ro_g3_r1").value == row["geo_3y"]
    assert _btn(at, "rf_offer_ro_geo_calc_r1") is None                       # inskrivet
    assert "står redan i rutorna" in _captions(at)
    assert "ro_geokoz" not in str(row) and set(row) == {"id", "ticker", "name", "level", "date", "geo_now", "geo_3y"}
    assert not page.saves                                                    # 💾 sparar till repot, som förut


def test_geo_calculator_refuses_ounces_typed_as_koz(page):
    at = page(_blob(), _store())
    at.run()
    at.number_input(key="ro_geokoz_r1").set_value(650_000.0)
    at.number_input(key="ro_geokoz3_r1").set_value(560_000.0).run()
    assert not at.exception, at.exception
    assert _btn(at, "rf_offer_ro_geo_calc_r1") is None
    assert "står talet i uns?" in _captions(at)


def test_geo_calculator_without_share_counts_says_so(page):
    at = page(_blob(shares_now_m=None, shares_3y_ago_m=None), _store())
    at.run()
    assert not at.exception, at.exception
    assert "Börsdata har inget aktieantal nu och för tre år sedan" in _captions(at)
    assert not [n for n in at.number_input if n.key == "ro_geokoz_r1"]


# 2. P/NAV nu
def test_pnav_calculator_offers_price_over_navps_and_use_writes_the_field(page):
    store = _store(pnav_now=1.2, pnav_bottom=0.75)
    at = page(_blob(), store)
    at.run()
    at.number_input(key="ro_navps_r1").set_value(147.0).run()
    assert not at.exception, at.exception
    assert _btn(at, "rf_offer_ro_pnav_calc_r1") is None                       # valutan inte vald
    assert "Välj NAV:ns valuta — kursen är i USD." in _captions(at)
    assert at.selectbox(key="ro_navccy_r1").options[:3] == ["– välj", "USD", "CAD"]
    at.selectbox(key="ro_navccy_r1").set_value("USD").run()
    assert not at.exception, at.exception
    assert ("P/NAV nu = kurs 120,50 USD (Börsdata 2026-11-07) / NAV per aktie 147 USD = 0,82 — mot botten 0,75: "
            "+9,3 %") in _markdown(at)
    assert _row(at)["pnav_now"] == 1.2                                       # signalen läser det sparade talet
    _btn(at, "rf_offer_ro_pnav_calc_r1").click().run()
    assert not at.exception, at.exception
    assert _row(at)["pnav_now"] == 0.82 and at.number_input(key="ro_pn_r1").value == 0.82
    assert _row(at)["pnav_bottom"] == 0.75
    assert "navps" not in str(_row(at)) and "USD" not in str(_row(at))       # NAV per aktie och valutan sparas inte
    assert _btn(at, "rf_offer_ro_pnav_calc_r1") is None
    assert not page.saves


def test_pnav_calculator_says_why_when_the_currencies_differ(page):
    at = page(_blob(price=165.0, currency="CAD"), _store(pnav_bottom=0.75))
    at.run()
    at.number_input(key="ro_navps_r1").set_value(147.0)
    at.selectbox(key="ro_navccy_r1").set_value("USD").run()
    assert not at.exception, at.exception
    assert _btn(at, "rf_offer_ro_pnav_calc_r1") is None
    assert "Kursen är i CAD, NAV per aktie i USD — inget förslag." in _captions(at)
    assert _btn(at, "rf_offer_ro_newbottom_r1") is None                      # inget räknat tal → ingen ny botten


def test_pnav_calculator_without_a_price_says_so(page):
    at = page(_blob(price=None), _store())
    at.run()
    assert not at.exception, at.exception
    assert "Ingen kurs i nattens rad ännu" in _captions(at)
    assert not [n for n in at.number_input if n.key == "ro_navps_r1"]


# 3. Ny egen botten
def test_new_bottom_from_the_calculated_pnav_with_the_basis_reminder(page):
    at = page(_blob(), _store(pnav_now=1.0, pnav_bottom=0.90))
    at.run()
    assert _btn(at, "rf_offer_ro_newbottom_r1") is None                      # 1,0 över botten 0,90
    at.number_input(key="ro_navps_r1").set_value(147.0)
    at.selectbox(key="ro_navccy_r1").set_value("USD").run()
    assert not at.exception, at.exception
    assert ("Ny egen botten? P/NAV nu 0,82 (räknat ovan) är under den sparade botten 0,90. Använd bara om "
            "NAV-basen är densamma.") in _markdown(at)
    _btn(at, "rf_offer_ro_newbottom_r1").click().run()
    assert not at.exception, at.exception
    assert _row(at)["pnav_bottom"] == 0.82 and at.number_input(key="ro_pb_r1").value == 0.82
    assert _row(at)["pnav_now"] == 1.0                                       # bara botten skrivs
    assert _btn(at, "rf_offer_ro_newbottom_r1") is None


def test_new_bottom_from_the_saved_pnav_without_the_calculator(page):
    store = _store(pnav_now=0.70, pnav_bottom=0.75)
    at = page({}, store)
    at.run()
    assert not at.exception, at.exception
    assert "Ny egen botten? P/NAV nu 0,70 (i rutan) är under den sparade botten 0,75." in _markdown(at)
    assert at.session_state["producers"] == store["producers"]                 # att visa skriver inget
    _btn(at, "rf_offer_ro_newbottom_r1").click().run()
    assert not at.exception, at.exception
    assert _row(at)["pnav_bottom"] == 0.70


def test_no_new_bottom_without_a_saved_bottom(page):
    at = page({}, _store(pnav_now=0.70))
    at.run()
    assert not at.exception, at.exception
    assert _btn(at, "rf_offer_ro_newbottom_r1") is None


def test_the_signal_reads_the_saved_numbers_until_use(page):
    """Huvudet (royalty_signal) och nattlarmet läser de sparade fälten — det räknade P/NAV:t når dem först med Använd."""
    store = _store(pnav_now=1.30, pnav_bottom=1.0, ev_now=18.0, ev_median=20.0, geo_now=0.05, geo_3y=0.04)
    at = page(_blob(price=104.0), store)
    at.run()
    assert _head(at).endswith(p.R_NEUTRAL)
    at.number_input(key="ro_navps_r1").set_value(100.0)
    at.selectbox(key="ro_navccy_r1").set_value("USD").run()
    assert not at.exception, at.exception
    assert _head(at).endswith(p.R_NEUTRAL)                        # 1,04 räknat men inte använt
    _btn(at, "rf_offer_ro_pnav_calc_r1").click().run()
    assert not at.exception, at.exception
    assert _head(at).endswith(p.R_BUY)                            # efter Använd: 1,04 mot botten 1,0
