"""
Automatisering, batch 7 — Lukacs FV och Rick Rule-kortet: härledda förslag och visningar.

1. 'Framtida antal aktier': Börsdatas aktieantal (shares_now_m, året ur shares_year) som förslag med Använd, och en
   varning — bara text — när talet i rutan ligger mer än 10× därifrån (tusental mot miljoner).
2. Under 'Tänkt position': 'Tak 4 % av totalen (2 % med skuld/EBITDA 1,4)' ur allokeraren och
   lukacs.max_position_pct. Bara visning, inget storleksförslag.
3. 'Detta är ett innehav': ett textförslag när positionerna har tickern i någon form — tyst när innehaven är låsta.
   Rutan kryssas aldrig.
4. CSM: 'Spot i dag X · NAV räknad på Y' över scenarierna. Inga rutor fylls; USD/lb jämförs aldrig med USD/oz.

Inget skrivs när sidan öppnas; inga grindar, trösklar eller formler ändrade.
"""
import copy
import os
import sys

import pytest
import streamlit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import allocator    # noqa: E402
import controls_ui  # noqa: E402
import lukacs       # noqa: E402
import lukacs_ui    # noqa: E402
import producers    # noqa: E402
from portfolio import private_lock   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── 1. Framtida antal aktier (ren) ───────────────────────────────────────────
def test_shares_typed_in_thousands_is_flagged():
    w = lukacs_ui.shares_scale_warning(1_234_000, 1234, 2025)
    assert w == ("Framtida antal aktier 1 234 000 är 1 000× Börsdatas 1 234 miljoner, årsrapporten 2025. Står talet "
                 "i tusental? Rutan är i miljoner — FV per aktie blir då 1 000× för låg.")


def test_shares_too_small_says_fv_gets_too_high_and_can_turn_step_five_green():
    w = lukacs_ui.shares_scale_warning(1.234, 1234)
    assert w.startswith("Framtida antal aktier 1,23 är bara 1/1 000 av Börsdatas 1 234 miljoner. Står talet i miljarder?")
    assert "1 000× för hög och kan göra köpgrindens steg 5 grönt" in w


def test_shares_within_ten_times_or_missing_give_no_warning():
    assert lukacs_ui.shares_scale_warning(1500, 1234) == ""          # återköp/utspädning: inget fel
    assert lukacs_ui.shares_scale_warning(12_340, 1234) == ""        # precis 10×: inte över gränsen
    assert lukacs_ui.shares_scale_warning(123.4, 1234) == ""
    for typed, bd in ((None, 1234), (0, 1234), (1234, None), (1234, 0), (-5, 1234), ("", "")):
        assert lukacs_ui.shares_scale_warning(typed, bd) == "", (typed, bd)


def test_the_shares_check_changes_no_calculation():
    row = {"framtida_antal_aktier": 1_234_000, "aktuell_kurs": 10.0, "what_must_go_right": "x",
           "fv": {s: {"forward_fcf_musd": 100.0, "target_yield": 10.0} for s in lukacs.FV_SCENARIOS}}
    before = copy.deepcopy(row)
    ev = lukacs.evaluate(row)
    lukacs_ui.shares_scale_warning(row["framtida_antal_aktier"], 1234)
    assert lukacs.evaluate(row) == ev and row == before


# ── 2. Taket under 'Tänkt position' (ren) ────────────────────────────────────
def test_cap_line_halves_with_debt_over_one():
    assert producers.position_cap_line({"nd_ebitda": 1.4}) == (
        "📏 Tak 4 % av totalen (2 % med skuld/EBITDA 1,4) — en gräns, inte ett förslag på storlek.")


def test_cap_line_without_debt_or_at_one_is_the_plain_cap():
    for nd in (None, "", 0.5, 1.0, -0.3):
        assert producers.position_cap_line({"nd_ebitda": nd}) == (
            "📏 Tak 4 % av totalen — en gräns, inte ett förslag på storlek."), nd


def test_cap_line_reads_the_same_rule_and_function_as_the_scorecard():
    import scorecard
    cap = allocator.rule_for_strategy("rule").hard_cap
    assert f"Tak {cap:g} %" in producers.position_cap_line({})
    entry = {"rows": {scorecard.RULE_STRATEGY: {"nd_ebitda": 2.5}}}
    half = scorecard.deleveraging_cap(entry, {"strategy": scorecard.RULE_STRATEGY})
    assert half == lukacs.max_position_pct(2.5, cap)
    assert f"({half:g} % med skuld/EBITDA 2,5)" in producers.position_cap_line({"nd_ebitda": 2.5})


# ── 3. Innehav (ren) ─────────────────────────────────────────────────────────
def _finder(*tickers):
    held = {t: {"ticker": t} for t in tickers}
    return lambda t: held.get(str(t).strip().upper())


def test_holding_hint_finds_the_ticker_in_any_form():
    assert lukacs_ui.holding_hint("EQX.TO", _finder("EQX")) == \
        "💡 Positionerna har EQX — kryssa om det här är ett innehav."
    assert lukacs_ui.holding_hint("eric-b.st", _finder("ERIC-B.ST")) == \
        "💡 Positionerna har ERIC-B.ST — kryssa om det här är ett innehav."
    assert lukacs_ui.holding_hint("ERIC-B.ST", _finder("ERIC B")) != ""          # Börsdatas form med mellanslag


def test_holding_hint_is_empty_without_a_match_or_a_ticker():
    assert lukacs_ui.holding_hint("NEM", _finder("EQX")) == ""
    assert lukacs_ui.holding_hint("", _finder("EQX")) == "" and lukacs_ui.holding_hint(None, _finder()) == ""


# ── 4. CSM-kontext (ren) ─────────────────────────────────────────────────────
GOLD_REF = {"commodity_price": 4012.3, "commodity_unit": "USD/oz", "commodity_asof": "2026-11-07"}


def test_spot_context_names_spot_and_the_cards_nav_price():
    assert controls_ui.spot_context(GOLD_REF, {"commodity": "Guld", "nav_price": 2000.0}) == \
        "🤖 Spot i dag 4 012 USD/oz (2026-11-07) · NAV räknad på 2 000"
    assert controls_ui.spot_context(GOLD_REF, {"commodity": "Guld"}) == "🤖 Spot i dag 4 012 USD/oz (2026-11-07)"
    cu = {"commodity_price": 4.5234, "commodity_unit": "USD/lb"}
    assert controls_ui.spot_context(cu, {"commodity": "Koppar", "nav_price": 3.5}) == \
        "🤖 Spot i dag 4,52 USD/lb · NAV räknad på 3,50"


def test_spot_context_never_compares_different_units():
    # råvaran bytt sedan natten: nattens guldpris i USD/oz står inte på ett kopparkort (USD/lb)
    assert controls_ui.spot_context(GOLD_REF, {"commodity": "Koppar", "nav_price": 4.0}) == ""
    # kopparkortets NAV-pris skrivet i USD/t: inte en jämförelse med USD/lb
    cu = {"commodity_price": 4.52, "commodity_unit": "USD/lb"}
    txt = controls_ui.spot_context(cu, {"commodity": "Koppar", "nav_price": 9000.0})
    assert txt == "🤖 Spot i dag 4,52 USD/lb · NAV-priset 9 000 ser ut att stå i en annan enhet än USD/lb"
    assert "räknad på" not in txt


def test_spot_context_is_empty_when_data_is_missing():
    for ref, row in ((None, {"commodity": "Guld", "nav_price": 2000.0}),
                     ({}, {"commodity": "Guld"}),
                     (dict(GOLD_REF, commodity_price=None), {"commodity": "Guld"}),
                     (dict(GOLD_REF, commodity_price=0), {"commodity": "Guld"}),
                     (dict(GOLD_REF, commodity_unit=""), {"commodity": "Guld"}),
                     (GOLD_REF, {}),                                           # kortet saknar råvara
                     (dict(GOLD_REF, commodity_unit="USD/lb"), {"commodity": "Uran"})):   # ingen Yahoo-termin
        assert controls_ui.spot_context(ref, row) == "", (ref, row)


# ── Rick Rule-sidan (AppTest) ────────────────────────────────────────────────
RULE = {"id": "p1", "ticker": "EQX.TO", "name": "Equinox Gold", "commodity": "Guld", "date": "2026-09-01",
        "price": 4000.0, "unit_cost": 2000.0, "nd_ebitda": 1.4, "mine_life": 10.0, "position_pct": 3.0,
        "nav_price": 2000.0}
NIGHT = {"ticker": "EQX.TO", "price": 8.0, "currency": "USD", "asof": "2026-11-07", "source": "borsdata",
         "bd_name": "Equinox Gold Corp", "shares_now_m": 456.78, "shares_year": 2025,
         "commodity_price": 4012.3, "commodity_unit": "USD/oz", "commodity_asof": "2026-11-07"}
HOLDINGS = {"swing": [], "ovtlyr": [], "momentum": [], "tiggre": [], "closed": [], "cash": 0,
            "long": [{"id": "q1", "ticker": "EQX", "strategy": "Quality", "entry_price": 7.0, "shares": 10}]}


@pytest.fixture
def page(monkeypatch):
    """Rick Rule-arket med fejkad lagring och sifferuppdatering — inget nätverk."""
    import gist_storage
    import refresh_ui
    import screens_ui
    import storage

    def make(row=None, night=None, unlocked=False):
        stores = {"producers": {"producers": [dict(RULE, **(row or {}))], "royalty": []},
                  "holdings": copy.deepcopy(HOLDINGS)}
        blob = {"rows": {"producers:p1": dict(NIGHT, **(night or {}))}}
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
        if unlocked:
            at.session_state[private_lock.STATE] = True
        at.run()
        assert not at.exception, at.exception
        at.stores = stores
        return at
    return make


def _caps(at) -> str:
    return " ".join(c.value for c in at.caption)


def _md(at) -> str:
    return " ".join(m.value for m in at.markdown)


def _warn(at) -> str:
    return " ".join(w.value for w in at.warning)


def _unwritten(at) -> bool:
    """Arket som det sparades — bortsett från FV-modulens tomma scenarioskelett ({'Bear': {}, …}), som
    lukacs_ui.render_fv lägger dit sedan förut (setdefault) och som inte är ett tal."""
    rows = copy.deepcopy(at.session_state["producers"])
    for r in rows["producers"]:
        if isinstance(r.get("fv"), dict) and not any(r["fv"].values()):
            r.pop("fv")
    return rows == at.stores["producers"]


def test_opening_the_card_shows_everything_and_writes_nothing(page):
    at = page()
    caps, md = _caps(at), _md(at)
    assert "📏 Tak 4 % av totalen (2 % med skuld/EBITDA 1,4) — en gräns, inte ett förslag på storlek." in caps
    assert "🤖 Spot i dag 4 012 USD/oz (2026-11-07) · NAV räknad på 2 000" in md
    assert "🤖 Börsdata 2026-11-07: aktier i dag (miljoner, årsrapporten 2025) 456.78" in md
    assert "rf_producers_p1_framtida_antal_aktier" in {b.key for b in at.button}
    assert "Står talet i" not in _warn(at)                             # tom ruta: ingen varning
    assert "Positionerna har" not in caps + md                         # låst: tyst
    assert _unwritten(at)                                             # inget skrivet när sidan öppnas
    assert at.number_input(key="fv_p1_shares").value is None


def test_use_writes_borsdatas_share_count_into_the_box(page):
    at = page()
    at.button(key="rf_producers_p1_framtida_antal_aktier").click().run()
    assert not at.exception, at.exception
    assert at.session_state["producers"]["producers"][0]["framtida_antal_aktier"] == 456.78
    assert at.number_input(key="fv_p1_shares").value == 456.78
    assert "rf_producers_p1_framtida_antal_aktier" not in {b.key for b in at.button}   # förslaget är använt


def test_shares_in_thousands_get_a_text_warning_without_a_button(page):
    at = page(row={"framtida_antal_aktier": 456_780.0})
    w = _warn(at)
    assert "Framtida antal aktier 456 780 är 1 000× Börsdatas 456,78 miljoner, årsrapporten 2025." in w
    assert _unwritten(at)
    at.number_input(key="fv_p1_shares").set_value(460.0).run()        # rättat: varningen går bort i samma körning
    assert not at.exception, at.exception
    assert "Står talet i" not in _warn(at)


def test_without_the_year_the_suggestion_reads_today(page):
    at = page(night={"shares_year": None})
    assert "aktier i dag (miljoner) 456.78" in _md(at)


def test_holding_hint_only_when_unlocked_and_never_ticks_the_box(page):
    at = page(unlocked=True)
    assert "💡 Positionerna har EQX — kryssa om det här är ett innehav." in _caps(at)
    assert at.checkbox(key="fv_p1_holding").value is False
    assert "is_holding" not in at.session_state["producers"]["producers"][0]
    at = page(row={"is_holding": True}, unlocked=True)                  # redan kryssad: inget att föreslå
    assert "Positionerna har" not in _caps(at)


def test_cap_line_without_debt(page):
    at = page(row={"nd_ebitda": 0.4})
    assert "📏 Tak 4 % av totalen — en gräns, inte ett förslag på storlek." in _caps(at)


def test_spot_context_is_silent_without_the_night_price_or_in_another_unit(page):
    at = page(night={"commodity_price": None})
    assert "Spot i dag" not in _md(at)
    at = page(row={"commodity": "Koppar"})                              # nattens pris är guldets USD/oz
    assert "Spot i dag" not in _md(at)
    assert "csm" not in at.session_state["producers"]["producers"][0]  # inga rutor fylls
