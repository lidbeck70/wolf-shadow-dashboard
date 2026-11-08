"""
Scorecard: Värdering föreslås ur arket strategin hör till — samma tal och gränser som arket räknar med.

Strategin durrett är Poängmodellens Durrett-del (snabbpoängen): MCap/framtida vinst mot köpregeln under 10× och
MCap/uns mot $100/$50. Strategin confidence är ARKEN → 🧭 Durrett (10-stegsarket): Durrett-motorns MCap/framtida vinst
efter skatt, EV per resursenhet och EV/NPV. Båda får Börsdatas EV/EBITDA när sifferuppdateringen har den, och
förslaget säger vilket ark talen kommer ur — de två räknar köpregeln olika (före/efter skatt).

Förslaget visas av sig självt under rutan och skrivs in först med "Använd": att öppna Scorecardet skriver ingenting,
och inget sparas utan 💾. Ingen grind- eller poängregel är ändrad.
"""
import copy
import json
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scorecard  # noqa: E402
import storage    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = "gold|AU"
DURRETT = {"id": "d1", "ticker": "AU", "name": "Gold", "commodity": "Guld", "date": "2026-09-01",
           "factors": {}, "mcap": 300.0, "moz": 4.0, "profit": 55.0}
STORES = {
    "producers": {"producers": [], "royalty": []},
    "scoring": {"sprott": [dict(DURRETT, id="s1", factors={})], "durrett": [DURRETT]},
    "insider": {"signals": []},
    "tiggre": {"candidates": [], "positions": [], "closed": [], "parked": []},
    "holdings": {"swing": [], "ovtlyr": [], "long": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0},
    "scorecard": {"cards": {}},
    "confidence": {"companies": {}, "commodity_overrides": {}},
}
TEXT = "MCap/framtida vinst 5,5× (under köpregelns 10×) · MCap/uns $75 (billigt) · EV/EBITDA 4,2× (Börsdata)"


def _entry(**row):
    return {"rows": {"durrett": dict(DURRETT, **row)}}


# ── Den rena funktionen ─────────────────────────────────────────────────────
def test_the_suggestion_uses_the_sheets_own_numbers_and_limits():
    assert scorecard.valuation_suggestion(_entry(), "durrett", {"ev_ebitda": 4.2}) == TEXT
    assert scorecard.valuation_suggestion(_entry(profit=20.0, moz=1.5), "durrett") == (
        "MCap/framtida vinst 15,0× (över köpregelns 10×) · MCap/uns $200 (över $100)")
    assert scorecard.valuation_suggestion(_entry(moz=10.0), "durrett").endswith("MCap/uns $30 (mycket billigt)")


def test_only_what_can_be_computed_is_shown():
    assert scorecard.valuation_suggestion(_entry(moz=None), "durrett") == (
        "MCap/framtida vinst 5,5× (under köpregelns 10×)")
    assert scorecard.valuation_suggestion(_entry(profit=None, moz=None), "durrett", {"ev_ebitda": 4.2}) == (
        "EV/EBITDA 4,2× (Börsdata)")
    assert scorecard.valuation_suggestion(_entry(mcap=None), "durrett") is None
    # Negativa tal är luckor som i arket — aldrig ett tal under 10× som ser ut som ett köp
    assert scorecard.valuation_suggestion(_entry(profit=-5.0, moz=None), "durrett") is None
    assert scorecard.valuation_suggestion(_entry(moz=None, profit=None), "durrett", {"ev_ebitda": -3.0}) is None


def test_other_strategies_get_no_suggestion():
    e = {"rows": {"durrett": DURRETT, "sprott": DURRETT}}
    assert scorecard.valuation_suggestion(e, "sprott") is None
    assert scorecard.valuation_suggestion({"rows": {}}, "durrett") is None
    assert scorecard.valuation_suggestion(None, "durrett") is None


# ── Kortet (AppTest) ────────────────────────────────────────────────────────
def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


@pytest.fixture
def page(monkeypatch):
    """Scorecardet ur en fejkad lagring — inget GitHub, ingen Gist; sifferuppdateringen styrs per test."""
    stores = copy.deepcopy(STORES)
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        streamlit.session_state.setdefault(name, copy.deepcopy(stores.get(name, default))))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    import gist_storage
    import refresh_ui
    import screens_ui
    monkeypatch.setattr(screens_ui, "load_screens", lambda: {})
    monkeypatch.setattr(gist_storage, "load_blob", lambda *a, **k: None)

    def run(blob=None):
        from streamlit.testing.v1 import AppTest
        monkeypatch.setattr(refresh_ui, "load_refresh", lambda: blob or {})

        def app(root):
            import sys as _s
            _s.path.insert(0, root)
            import scorecard as sc
            sc.render_scorecard_page()
        at = AppTest.from_function(app, args=(ROOT,), default_timeout=90)
        at.run()
        assert not at.exception, at.exception
        return at
    return run


def _hint(at) -> str:
    return " ".join(c.value for c in at.caption if c.value.startswith("🤖 Ur "))


def test_the_card_shows_the_valuation_by_itself_and_writes_nothing(page):
    at = page({"rows": {"scoring:d1": {"ev_ebitda": 4.2, "asof": "2026-11-07"}}})
    at.selectbox(key=f"sc_strat_{KEY}").set_value("durrett").run()
    assert f"🤖 Ur Poängmodellen → Durrett (snabbpoäng): {TEXT.replace('$', chr(92) + '$')} — förslag, inte inskrivet." \
        in _hint(at)
    assert at.text_input(key=f"sc_val_{KEY}").value == ""
    # Att byta strategi i kortet är ett eget val — men förslaget i sig skriver ingenting
    assert at.session_state["scorecard"]["cards"][KEY].get("valuation", "") == ""
    for other in ("scoring", "producers", "tiggre", "insider"):
        assert _same(at.session_state[other], STORES[other]), other


def test_opening_with_durrett_first_writes_no_card(page):
    """Durrett som enda/förvalda strategi: förslaget syns direkt och lagringen är orörd."""
    at = page()
    at.session_state["scoring"] = dict(copy.deepcopy(STORES["scoring"]), sprott=[])
    at.run()
    assert not at.exception, at.exception
    assert "MCap/framtida vinst 5,5×" in _hint(at)
    assert _same(at.session_state["scorecard"], STORES["scorecard"])


def test_use_fills_the_box_and_the_hint_goes_away(page):
    at = page({"rows": {"scoring:d1": {"ev_ebitda": 4.2}}})
    at.selectbox(key=f"sc_strat_{KEY}").set_value("durrett").run()
    at.button(key=f"sc_useval_{KEY}").click().run()
    assert not at.exception, at.exception
    assert at.text_input(key=f"sc_val_{KEY}").value == TEXT
    card = at.session_state["scorecard"]["cards"][KEY]
    assert card["valuation"] == TEXT and card["strategy"] == "durrett"
    assert not _hint(at)


def test_the_hint_stays_beside_an_own_text_and_sprott_gets_none(page):
    at = page()
    at.selectbox(key=f"sc_strat_{KEY}").set_value("durrett").run()
    at.text_input(key=f"sc_val_{KEY}").input("Egen bedömning").run()
    assert at.session_state["scorecard"]["cards"][KEY]["valuation"] == "Egen bedömning"
    assert "MCap/framtida vinst 5,5×" in _hint(at)                    # förslaget skriver inte över ens egen text
    at.selectbox(key=f"sc_strat_{KEY}").set_value("sprott").run()
    assert not _hint(at)


def test_two_dollar_signs_are_escaped_on_screen_but_not_in_the_text(page):
    """MCap/uns $200 (över $100): två '$' på en rad blev en formel i Markdown och dollartecknen försvann."""
    at = page()
    at.session_state["scoring"] = dict(copy.deepcopy(STORES["scoring"]), sprott=[],
                                       durrett=[dict(DURRETT, profit=20.0, moz=1.5)])
    at.run()
    assert "MCap/uns \\$200 (över \\$100)" in _hint(at)
    at.button(key=f"sc_useval_{KEY}").click().run()
    assert at.text_input(key=f"sc_val_{KEY}").value.endswith("MCap/uns $200 (över $100)")     # rutan: ren text


# ── ARKEN → 🧭 Durrett (strategin confidence) ───────────────────────────────
def _metrics(**vals):
    return {k: {"value": v, "unit": "USD/oz" if k == "ev_per_resource_unit" else "×"} for k, v in vals.items()}


def test_the_durrett_sheet_suggestion_uses_the_engines_own_numbers():
    e = {"ticker": "GPR", "rows": {"confidence": {"ticker": "GPR"}}}
    m = _metrics(mcap_future_earnings=10.9, ev_per_resource_unit=443.75, ev_npv=0.62)
    assert scorecard.valuation_suggestion(e, "confidence", {"ev_ebitda": 4.1}, m) == (
        "MCap/framtida vinst 10,9× efter skatt (över köpregelns 10×) · EV/resurs 444 USD/oz · EV/NPV 0,62× · "
        "EV/EBITDA 4,1× (Börsdata)")
    assert scorecard.valuation_suggestion(e, "confidence", None, _metrics(mcap_future_earnings=4.42)) == (
        "MCap/framtida vinst 4,4× efter skatt (under köpregelns 10×)")
    assert scorecard.valuation_suggestion(e, "confidence", None, {}) is None
    assert scorecard.valuation_suggestion(e, "confidence") is None        # utan motorns mått — inget gissat


def test_the_engine_numbers_match_what_the_durrett_sheet_shows():
    """Samma körning som 🧭 Durrett (analyze): förslagets MCap/framtida vinst är motorns, inte Poängmodellens."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import durrett_cases as dcs
    from datetime import date
    from engines.durrett.engine import analyze
    a = analyze(dcs.gold_producer(), today=date.today())
    fe = a.metrics["mcap_future_earnings"]["value"]
    text = scorecard.valuation_suggestion({"ticker": "GPR"}, "confidence", None, a.metrics)
    assert text.startswith(f"MCap/framtida vinst {scorecard._sv(fe)}× efter skatt (under köpregelns 10×)")


def test_a_shared_card_names_each_sheet_for_its_own_strategy(page):
    """Samma bolag i Poängmodellen och ARKEN → 🧭 Durrett: ett kort, två strategier — varje förslag säger sitt ark."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import durrett_cases as dcs
    company = dcs.gold_producer().as_dict()
    at = page({"rows": {"confidence:GPR": {"ev_ebitda": 4.1}}})
    at.session_state["scoring"] = dict(copy.deepcopy(STORES["scoring"]), sprott=[],
                                       durrett=[dict(DURRETT, id="d2", ticker="GPR", name="Test Gold Producer")])
    at.session_state["confidence"] = {"companies": {"GPR": company}, "commodity_overrides": {}}
    at.run()
    assert not at.exception, at.exception
    key = "test gold producer|GPR"
    assert "🤖 Ur Poängmodellen → Durrett (snabbpoäng): MCap/framtida vinst 5,5×" in _hint(at)
    at.selectbox(key=f"sc_strat_{key}").set_value("confidence").run()
    hint = _hint(at)
    assert "🤖 Ur Durrett-arket (ARKEN → 🧭 Durrett): MCap/framtida vinst" in hint and "efter skatt" in hint
    assert "EV/EBITDA 4,1× (Börsdata)" in hint and "Poängmodellen" not in hint
    at.button(key=f"sc_useval_{key}").click().run()
    assert at.session_state["scorecard"]["cards"][key]["valuation"].startswith("MCap/framtida vinst")
    assert at.session_state["scorecard"]["cards"][key]["strategy"] == "confidence"


# ── Rick Rule (producenter) och Royalty C ────────────────────────────────────
RULE = {"id": "r1", "ticker": "AAA", "name": "Rule Co", "commodity": "Guld", "date": "2026-09-01",
        "ev_musd": 620.0, "nav_musd": 1000.0, "nav_price": 2500.0, "price": 2600.0, "ev_ebitda": 3.0,
        "fcf_kvalitet": "C", "framtida_antal_aktier": 100.0, "aktuell_kurs": 4.0,
        "fv": {"Base": {"forward_fcf_musd": 120.0, "target_yield": 12.0}}}
ROY = {"id": "y1", "ticker": "BBB", "name": "Roy Co", "level": 2, "date": "2026-09-01",
       "pnav_now": 1.05, "pnav_bottom": 1.0, "ev_now": 12.0, "ev_median": 15.0, "geo_now": 1.1, "geo_3y": 1.0}


def _rule(**kw):
    return {"rows": {"producenter": dict(RULE, **kw)}}


def _roy(**kw):
    return {"rows": {"royalty": dict(ROY, **kw)}}


def test_rick_rule_uses_the_sheets_ev_nav_and_lukacs():
    """EV/NAV 0,62× billigt (under 0,7×) och Lukacs: FV Base = 120 / 12 % / 100 M = 10,00, kurs 4 → MoS 60 %."""
    assert scorecard.valuation_suggestion(_rule(), "producenter") == (
        "EV/NAV 0,62× (billigt mot NAV) · Lukacs: säkerhetsmarginal 60,0 % mot fair value Base 10,00 "
        "(mycket attraktiv) · EV/EBITDA 3,00×")
    assert "(rimligt mot NAV)" in scorecard.valuation_suggestion(_rule(ev_musd=900.0), "producenter")
    assert "(dyrt mot NAV)" in scorecard.valuation_suggestion(_rule(ev_musd=1200.0), "producenter")
    assert "säkerhetsmarginal −25,0 %" in scorecard.valuation_suggestion(_rule(aktuell_kurs=12.5), "producenter")


def test_rick_rule_carries_the_sheets_warnings():
    """NAV räknad långt från spot, och Lukacs-panelens varning om FCF-enhet/kursvaluta, följer med i förslaget."""
    far = scorecard.valuation_suggestion(_rule(nav_price=1800.0), "producenter")
    assert "EV/NAV 0,62× (billigt mot NAV), NAV räknad på 1 800 mot spot 2 600 (+44 %)" in far
    nok = scorecard.valuation_suggestion(_rule(aktuell_kurs=49.53), "producenter",
                                         {"price": 49.53, "currency": "NOK"})
    assert "— OBS: arket varnar för FCF-enhet eller kursvaluta" in nok
    yield_in_box = scorecard.valuation_suggestion(
        _rule(fv={"Base": {"forward_fcf_musd": 12.0, "target_yield": 12.0}},
              csm={"Base": {"fcf_musd": 3550.0}}), "producenter")
    assert "OBS: arket varnar" in yield_in_box


def test_rick_rule_shows_only_what_can_be_computed():
    assert scorecard.valuation_suggestion(_rule(nav_musd=None, aktuell_kurs=None, ev_ebitda=None), "producenter") \
        is None
    assert scorecard.valuation_suggestion(_rule(nav_musd=None, aktuell_kurs=None, ev_ebitda=None), "producenter",
                                          {"ev_ebitda": 4.2}) == "EV/EBITDA 4,2× (Börsdata)"
    assert scorecard.valuation_suggestion(_rule(fcf_kvalitet="D", nav_musd=None, ev_ebitda=None), "producenter") == (
        "Lukacs: klass D — värderas inte på FCF")
    assert scorecard.valuation_suggestion(_rule(ev_musd=-50.0, nav_musd=1000.0, aktuell_kurs=None, ev_ebitda=-2.0),
                                          "producenter") is None            # negativt = lucka, som i arket


def test_royalty_uses_the_sheets_own_comparisons_and_signal():
    assert scorecard.valuation_suggestion(_roy(), "royalty") == (
        "P/NAV 1,05 (5,0 % över egen botten 1,00 — köpläge högst 15 % över) · EV/EBITDA 12,00× (20,0 % under egen "
        "median 15,00×) · GEO/aktie +10,0 % på 3 år · signal \"KÖPLÄGE\"")
    shrink = scorecard.valuation_suggestion(_roy(geo_now=0.9), "royalty")
    assert "GEO/aktie −10,0 % på 3 år" in shrink and shrink.endswith('signal "Varning: GEO/aktie krymper"')
    assert scorecard.valuation_suggestion(_roy(pnav_now=1.3), "royalty").endswith('signal "Neutral"')


def test_royalty_shows_only_what_can_be_computed():
    assert scorecard.valuation_suggestion({"rows": {"royalty": {"id": "y1"}}}, "royalty") is None
    only_bd = scorecard.valuation_suggestion(_roy(pnav_now=None, ev_now=None, geo_now=None), "royalty",
                                             {"ev_ebitda": 11.0})
    assert only_bd == "EV/EBITDA 11,0× (Börsdata)"                       # ingen signal utan arkets jämförelser
    assert "(på egen median 15,00×)" in scorecard.valuation_suggestion(_roy(ev_now=15.0), "royalty")


def test_the_rule_and_royalty_cards_show_their_sheets_suggestion(page):
    at = page({"rows": {"producers:y1": {"ev_ebitda": 11.0}}})
    at.session_state["producers"] = {"producers": [RULE], "royalty": [ROY]}
    at.run()
    assert not at.exception, at.exception
    hint = _hint(at)
    assert "🤖 Ur Rick Rule: EV/NAV 0,62× (billigt mot NAV)" in hint
    assert "🤖 Ur Royalty C: P/NAV 1,05" in hint and 'signal "KÖPLÄGE"' in hint
    assert _same(at.session_state["scorecard"], STORES["scorecard"])         # att öppna skriver inget kort
    assert _same(at.session_state["producers"], {"producers": [RULE], "royalty": [ROY]})
    at.button(key="sc_useval_roy co|BBB").click().run()
    card = at.session_state["scorecard"]["cards"]["roy co|BBB"]
    assert card["valuation"].startswith("P/NAV 1,05") and card["strategy"] == "royalty"


# ── Granskningens fynd (Rick Rule/Royalty) ──────────────────────────────────
def test_the_margin_keeps_the_cards_decimal_at_the_band_edges():
    """24,6 % är 'Bevaka' på kortet — avrundat till '25 %' såg förslaget ut att klara köpgrindens 25 %."""
    assert "säkerhetsmarginal 24,6 % mot fair value Base 10,00 (bevaka)" in \
        scorecard.valuation_suggestion(_rule(aktuell_kurs=7.54), "producenter")
    assert "säkerhetsmarginal 25,0 % mot fair value Base 10,00 (attraktiv)" in \
        scorecard.valuation_suggestion(_rule(aktuell_kurs=7.5), "producenter")
    assert "säkerhetsmarginal 9,6 %" in scorecard.valuation_suggestion(_rule(aktuell_kurs=9.04), "producenter")


def test_an_out_of_band_target_yield_is_flagged_like_the_card():
    """Klass C har bandet 10–14 %: 8 % visar ett fel på kortet — förslaget får inte se rent ut."""
    bad = scorecard.valuation_suggestion(_rule(aktuell_kurs=9.0, fv={"Base": {"forward_fcf_musd": 120.0,
                                                                              "target_yield": 8.0}}), "producenter")
    assert "säkerhetsmarginal 40,0 %" in bad and "— OBS: arket varnar för target-yield utanför klassens band" in bad
    unset = scorecard.valuation_suggestion(_rule(fcf_kvalitet=None, aktuell_kurs=9.0,
                                                 fv={"Base": {"forward_fcf_musd": 120.0, "target_yield": 8.0}}),
                                           "producenter")
    assert "target-yield utanför klassens band" in unset                  # kortets förval är klass C
    assert "OBS" not in scorecard.valuation_suggestion(_rule(), "producenter")


def test_a_yahoo_fallback_row_gets_its_currency_like_the_lukacs_panel(page):
    """Börsdata saknar bolaget → Yahoos kurs utan valuta. Panelen läser valutan ur tickern (VAR.OL = NOK) och varnar;
    förslaget gör nu samma sak i stället för att visa en ren marginal."""
    blob = {"rows": {"producers:r1": {"ticker": "VAR.OL", "price": 30.0, "currency": None, "source": "yfinance",
                                      "asof": "2026-11-07"}}}
    at = page(blob)
    at.session_state["producers"] = {"producers": [dict(RULE, ticker="VAR.OL", aktuell_kurs=30.0)], "royalty": []}
    at.run()
    assert not at.exception, at.exception
    assert "— OBS: arket varnar för FCF-enhet eller kursvaluta" in _hint(at)


def test_royalty_keeps_the_cards_decimal_at_the_fifteen_percent_limit():
    """+15,4 % är 'Neutral' på kortet; '15 % över … högst 15 % över' läste som köpläge."""
    edge = scorecard.valuation_suggestion(_roy(pnav_now=0.75, pnav_bottom=0.65), "royalty")
    assert "P/NAV 0,75 (15,4 % över egen botten 0,65" in edge and edge.endswith('signal "Neutral"')
    assert "(0,3 % över egen median 15,00×)" in scorecard.valuation_suggestion(_roy(ev_now=15.05), "royalty")
    assert "GEO/aktie −0,2 % på 3 år" in scorecard.valuation_suggestion(_roy(geo_now=0.998), "royalty")
    # Börsdatas oavrundade EV/EBITDA efter 'Använd' på kortet: två decimaler som rutan, inte '15,0× … under 15,0×'
    assert "EV/EBITDA 14,96× (0,2 % under egen median 15,00×)" in \
        scorecard.valuation_suggestion(_roy(ev_now=14.9637), "royalty")
    assert "EV/EBITDA 3,25×" in scorecard.valuation_suggestion(_rule(ev_ebitda=3.25), "producenter")


def test_the_geo_warning_shows_even_before_bottom_and_median_are_filled():
    """Kortet varnar för krympande GEO/aktie utan P/NAV-botten och median — förslaget också."""
    fresh = scorecard.valuation_suggestion(_roy(pnav_bottom=None, ev_median=None, geo_now=0.9), "royalty")
    assert fresh.endswith('signal "Varning: GEO/aktie krymper"')
    assert scorecard.valuation_suggestion(_roy(pnav_bottom=None, ev_median=None, ev_now=None), "royalty").endswith(
        'signal "Neutral"')


# ── Tiggre och Sprott ───────────────────────────────────────────────────────
LOB = {"id": "c1", "ticker": "LOB", "name": "Lobo", "added": "2026-09-01", "screen": {}, "factors": {},
       "catalysts": [], "mcap": 200.0, "nav": 650.0, "downside": -40.0}
JR = {"id": "s1", "ticker": "JR", "name": "Junior", "commodity": "Guld", "date": "2026-09-01", "factors": {},
      "cash": 45.0, "burn": 20.0}


def _tg(**kw):
    return {"rows": {"tiggre": dict(LOB, **kw)}}


def _sp(**kw):
    return {"rows": {"sprott": dict(JR, **kw)}}


def test_tiggre_uses_the_lobo_sheets_numbers():
    """Masterguidens exempel: MCap 200, NAV 650 → 0,31× NAV, +160 % till 0,8× NAV; nedsida −40 % → U/N 4,0:1."""
    assert scorecard.valuation_suggestion(_tg(), "tiggre") == (
        "P/NAV 0,31× (köpzon 0,2–0,4×) · uppsida till 0,8× NAV +160 % · U/N 4,0:1 (klarar kravet 3:1)")
    assert scorecard.valuation_suggestion(_tg(downside=-80.0), "tiggre").endswith("U/N 2,0:1 (under kravet 3:1)")
    assert scorecard.valuation_suggestion(_tg(downside=None), "tiggre") == (
        "P/NAV 0,31× (köpzon 0,2–0,4×) · uppsida till 0,8× NAV +160 %")


def test_tiggre_un_at_the_limit_never_rounds_across_it():
    """U/N 2,96 visas inte som '3,0:1' bredvid 'under kravet' — en decimal till när avrundningen korsar gränsen."""
    below = scorecard.valuation_suggestion(_tg(downside=-54.0), "tiggre")            # 160 / 54 = 2,963
    assert below.endswith("U/N 2,96:1 (under kravet 3:1)")
    assert scorecard.valuation_suggestion(_tg(downside=-53.3), "tiggre").endswith("U/N 3,0:1 (klarar kravet 3:1)")


def test_a_held_tiggre_position_uses_todays_market_cap_and_the_exit_rule():
    """Positionen räknas som positionskortet: dagens börsvärde (tiggre:<id>) och slutsäljregeln 0,8–1,0× NAV."""
    pos = {"rows": {"tiggre": {"id": "p1", "ticker": "AU", "strategy": "Tiggre", "entry": 10.0, "nav": 300.0,
                               "mcap": 200.0}}}
    assert scorecard.valuation_suggestion(pos, "tiggre", {"mcap_musd": 292.0}) == (
        "P/NAV 0,97× (0,8–1× NAV — slutsälj i etapper)")
    assert scorecard.valuation_suggestion(pos, "tiggre", {"mcap_musd": 310.0}) == (
        "P/NAV 1,03× (≥ 1× NAV — omvärderingen gjord, sälj resten)")
    assert scorecard.valuation_suggestion(pos, "tiggre") == (
        "P/NAV 0,67× (under 0,8× NAV — slutsäljregeln inte nådd)")
    assert scorecard.valuation_suggestion(pos, "tiggre", {"mcap_musd": 239.9}) == (
        "P/NAV 0,7997× (under 0,8× NAV — slutsäljregeln inte nådd)")              # inte '0,80×' bredvid 'under 0,8'


def test_tiggre_shows_nothing_for_gaps_or_negatives():
    assert scorecard.valuation_suggestion(_tg(mcap=None), "tiggre") is None
    assert scorecard.valuation_suggestion(_tg(nav=-650.0), "tiggre") is None
    assert scorecard.valuation_suggestion({"rows": {"tiggre": {"id": "c9"}}}, "tiggre") is None


def test_sprott_runway_against_the_eighteen_month_gate():
    assert scorecard.valuation_suggestion(_sp(), "sprott") == (
        "runway 2,2 år = 27 mån (klarar Sprotts grind 18 mån)")
    assert scorecard.valuation_suggestion(_sp(cash=10.0), "sprott") == (
        "runway 0,5 år = 6 mån (under Sprotts grind 18 mån — avstå oavsett poäng)")
    # 17,6 mån visas inte som '18 mån' bredvid 'under grinden'
    assert scorecard.valuation_suggestion(_sp(cash=29.3), "sprott") == (
        "runway 1,5 år = 17,6 mån (under Sprotts grind 18 mån — avstå oavsett poäng)")
    assert scorecard.valuation_suggestion(_sp(cash=30.0), "sprott") == (
        "runway 1,5 år = 18 mån (klarar Sprotts grind 18 mån)")
    assert scorecard.valuation_suggestion(_sp(burn=0.0), "sprott") is None
    assert scorecard.valuation_suggestion(_sp(cash=-5.0), "sprott") is None


def test_the_edge_rule_also_guards_the_earlier_sheets():
    """Samma kantregel i de tidigare arken: talet hamnar aldrig på andra sidan gränsen domen bygger på."""
    assert "15,04 % över egen botten" in scorecard.valuation_suggestion(_roy(pnav_now=1.1504), "royalty")
    assert "säkerhetsmarginal 24,96 %" in scorecard.valuation_suggestion(_rule(aktuell_kurs=7.504), "producenter")
    assert "MCap/framtida vinst 9,97×" in scorecard.valuation_suggestion(_entry(profit=30.1, moz=None), "durrett")
    assert "EV/NAV 0,6996× (billigt mot NAV)" in scorecard.valuation_suggestion(_rule(ev_musd=699.6), "producenter")


def test_the_tiggre_and_sprott_cards_show_their_sheets_suggestion(page):
    at = page()
    at.session_state["tiggre"] = dict(copy.deepcopy(STORES["tiggre"]), candidates=[LOB])
    at.session_state["scoring"] = {"sprott": [JR], "durrett": []}
    at.run()
    assert not at.exception, at.exception
    hint = _hint(at)
    assert "🤖 Ur Tiggre (Lobo-arket): P/NAV 0,31×" in hint
    assert "🤖 Ur Poängmodellen → Sprott (optionalitet): runway 2,2 år = 27 mån" in hint
    assert _same(at.session_state["scorecard"], STORES["scorecard"])
    at.button(key="sc_useval_junior|JR").click().run()
    card = at.session_state["scorecard"]["cards"]["junior|JR"]
    assert card["valuation"].startswith("runway 2,2 år") and card["strategy"] == "sprott"
