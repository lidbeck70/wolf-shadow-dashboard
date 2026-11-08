"""
Scorecard (batch 9): förslag ur arken som redan har svaret — inga regler ändras.

  Granskningen är klar   arkets egen dom (review_link._REVIEWERS) på raden för kortets strategi: PASS ger ett förslag
                         med Använd, MANUAL och FAIL står som text som säger vad som saknas
  Position (% av total)  taket, normalbandet, halveringen vid skuld och strömbrytaren ur allokeraren — bara visning,
                         och nuvarande vikt när bolaget redan ägs
  Vald säljregel         en rullista ur playbookens exitregler (Insider även stoppen), utan förval
  Trigger / katalysator  nästa väntande katalysator — aldrig en försenad, utebliven eller passerad

Att öppna Scorecardet skriver ingenting; allt skrivs först med användarens eget klick.
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
import review_link  # noqa: E402
import scorecard as sc  # noqa: E402
import storage  # noqa: E402
import strategy_rules  # noqa: E402
import tiggre as tg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = date(2026, 11, 8)

RULE = {"id": "a1", "ticker": "BOL.ST", "name": "Boliden", "commodity": "Koppar", "price": 9000.0,
        "unit_cost": 6000.0, "ev_ebitda": 5.0, "nd_ebitda": 0.5, "mine_life": 12.0, "rp_ratio": 15.0,
        "jurisdiktion": True, "kapitaldisciplin": True, "insyn": True}
INS = {"id": "i1", "ticker": "EVO.ST", "name": "Evolution", "insiders": 1, "role": ins.ROLE_OTHER, "amount": 100.0,
       "cluster_avg": 50.0}
LOB = {"id": "c1", "ticker": "LOB", "name": "Lobo", "added": "2026-09-01", "screen": {}, "factors": {},
       "downside": -40.0, "catalysts": [
           {"id": "k1", "name": "PFS", "date": "2020-01", "status": tg.CAT_WAITING},             # passerad
           {"id": "k2", "name": "Tillstånd", "date": "2098-01", "status": tg.CAT_LATE_1},        # försenad
           {"id": "k3", "name": "Borrsvar", "date": "2098-02", "status": tg.CAT_MISSED},         # utebliven
           {"id": "k4", "name": "Finansieringsbesked", "date": "2099-06", "status": tg.CAT_WAITING}]}
STORES = {
    "producers": {"producers": [RULE], "royalty": []},
    "scoring": {"sprott": [], "durrett": []},
    "insider": {"signals": [INS]},
    "tiggre": {"candidates": [LOB], "positions": [], "closed": [], "parked": []},
    "holdings": {"swing": [], "ovtlyr": [], "long": [{"id": "h1", "ticker": "BOL.ST", "strategy": "Rule",
                                                      "shares": 100, "entry_price": 200.0}],
                 "momentum": [], "tiggre": [], "closed": [], "cash": 0},
    "allocator": {"values": {"producenter": 100000.0, "kassa": 300000.0}, "positions": [], "peak": 1000000.0,
                  "current": 950000.0, "quarters_cash_high": 0, "updated": ""},
    "scorecard": {"cards": {}},
    "confidence": {"companies": {}, "commodity_overrides": {}},
}
BOL, EVO, LOBO = "boliden|BOL.ST", "evolution|EVO.ST", "lobo|LOB"


def _entry(strategy: str, row: dict) -> dict:
    return sc.collect({strategy: [row]})[0]


# ── Granskningen är klar ────────────────────────────────────────────────────
def test_the_review_verdict_is_the_sheets_own_for_the_cards_strategy():
    e = sc.collect({"producenter": [dict(RULE, name="X")], "insider": [dict(INS, ticker="BOL.ST", name="X")]})[0]
    assert sc.review_verdict(e, "producenter") == ("Rick Rule",) + review_link._REVIEWERS["rule"](e["rows"]["producenter"])
    assert sc.review_verdict(e, "producenter")[1:] == ("PASS", "4/5 p — Köpkandidat. Marginal och disciplin på plats.")
    assert sc.review_verdict(e, "insider") == ("Insiderbevakaren",) + review_link._REVIEWERS["insider"](INS)
    assert sc.review_verdict(e, "insider")[1] == "FAIL"
    missing = sc.review_verdict(_entry("producenter", dict(RULE, price=None)), "producenter")
    assert missing[1] == "MANUAL" and "fyll i råvarupris" in missing[2]


def test_strategies_without_a_reviewer_or_row_get_no_verdict():
    assert sc.review_verdict(_entry("confidence", {"ticker": "GPR", "name": "G"}), "confidence") is None
    assert sc.review_verdict(_entry("producenter", RULE), "tiggre") is None
    assert sc.review_verdict(None, "producenter") is None


# ── Position (% av total) ───────────────────────────────────────────────────
def test_the_position_frame_reads_the_allocator_and_the_debt_rule():
    e = _entry("producenter", RULE)
    assert sc.position_frame(e, "producenter", 1000000.0, 950000.0) == (
        "Tak 4 % av totalen · normalt 5–10 % av strategidelen · strömbrytaren NORMAL (5,0 % från topp)")
    debt = _entry("producenter", dict(RULE, nd_ebitda=1.8))
    assert "nettoskulden 1,8× EBITDA halverar taket till 2 %" in sc.position_frame(debt, "producenter")
    assert sc.position_frame(_entry("sprott", {"ticker": "JR", "name": "J"}), "sprott").startswith(
        "Tak 1,5 % av totalen · normalt 10–20 % av strategidelen")
    assert sc.position_frame(e, "producenter").endswith("strömbrytaren okänd — fyll i topp och nuvarande i Allokeringen")
    sharp = sc.position_frame(e, "producenter", 1000000.0, 850000.0)
    assert "strömbrytaren SKÄRPT (15,0 % från topp): Inga nya köp i strategin som driver nedgången." in sharp


def test_confidence_has_no_cap_in_the_allocator():
    e = _entry("confidence", {"ticker": "GPR", "name": "G", "nd_ebitda": 3.0})
    assert sc._strategy_cap("confidence") is None
    assert sc.position_frame(e, "confidence", 100.0, 100.0) == "Inget tak i allokeraren · strömbrytaren NORMAL (0,0 % från topp)"


def test_the_cap_helper_is_unchanged():
    for s, cap in (("tiggre", 4.0), ("sprott", 1.5), ("producenter", 4.0), ("royalty", 10.0), ("insider", 4.0),
                   ("durrett", 3.0), ("okänd", None)):
        assert sc._strategy_cap(s) == cap, s


def test_held_weight_against_the_allocator_total():
    assert sc.held_weight("BOL.ST", [{"ticker": "BOL.ST", "value": 20000.0, "_currency": "SEK"}], 400000.0) == (
        "Äger redan: 5,0 % av totalen (20 000 kr av 400 000 kr).")
    two = [{"ticker": "AU", "value": 10000.0, "_currency": "USD"}, {"ticker": "au", "value": 10000.0, "_currency": "USD"}]
    assert sc.held_weight("AU", two, 100000.0) == "Äger redan: 20,0 % av totalen (20 000 kr av 100 000 kr, grovt omräknat till SEK)."
    assert "antal eller kurs saknas" in sc.held_weight("AU", [{"ticker": "AU", "value": None}], 100000.0)
    assert "totalen saknas" in sc.held_weight("AU", [{"ticker": "AU", "value": 5.0}], 0)
    assert sc.held_weight("NEM", two, 100000.0) is None and sc.held_weight("", [], 1.0) is None


# ── Vald säljregel ──────────────────────────────────────────────────────────
def test_sell_rules_come_from_the_playbook_of_the_cards_strategy():
    exits = {k: [r.text for r in strategy_rules.PLAYBOOKS[k].exit] for k in ("rule", "durrett", "tiggre", "insider")}
    assert sc.sell_rule_options(_entry("producenter", RULE), "producenter") == exits["rule"]
    assert sc.sell_rule_options(_entry("confidence", {"ticker": "G", "name": "G"}), "confidence") == exits["durrett"]
    assert sc.sell_rule_options(_entry("tiggre", LOB), "tiggre") == exits["tiggre"]
    opts = sc.sell_rule_options(_entry("insider", INS), "insider")
    assert opts == exits["insider"] + ["Stopp 42,50 (klustersnittet 50,00 × 0,85)"]
    assert sc.sell_rule_options(_entry("insider", dict(INS, cluster_avg=None)), "insider") == exits["insider"]
    assert sc.sell_rule_options(_entry("producenter", RULE), "okänd") == []


# ── Trigger / katalysator ───────────────────────────────────────────────────
def test_the_next_pending_tiggre_catalyst_is_suggested():
    assert sc.next_catalyst(_entry("tiggre", LOB), "tiggre", TODAY) == ("Tiggre (Lobo-arket)", "Finansieringsbesked (2099-06)")
    soon = dict(LOB, catalysts=LOB["catalysts"] + [{"name": "PEA", "date": "2026-11", "status": tg.CAT_WAITING}])
    assert sc.next_catalyst(_entry("tiggre", soon), "tiggre", TODAY)[1] == "PEA (2026-11)"      # månaden pågår


def test_a_late_missed_delivered_or_passed_catalyst_is_never_suggested():
    bad = [{"name": "A", "date": "2099-01", "status": tg.CAT_LATE_1}, {"name": "B", "date": "2099-01", "status": tg.CAT_LATE_2},
           {"name": "C", "date": "2099-01", "status": tg.CAT_MISSED}, {"name": "D", "date": "2099-01", "status": tg.CAT_DELIVERED},
           {"name": "E", "date": "2026-10", "status": tg.CAT_WAITING}, {"name": "F", "date": "snart", "status": tg.CAT_WAITING},
           {"name": "", "date": "2099-01"}]
    assert sc.next_catalyst(_entry("tiggre", dict(LOB, catalysts=bad)), "tiggre", TODAY) is None


def test_the_durrett_sheet_catalysts_and_the_ds_financing_catalyst():
    gpr = {"ticker": "GPR", "name": "G", "catalysts": [
        {"name": "Gammal PEA", "expected": "2026-Q2"}, {"name": "Byggbeslut", "expected": "2027-Q2"},
        {"name": "Tillstånd", "expected": "Q1 2027"}, {"name": "Utan datum", "expected": ""}]}
    assert sc.next_catalyst(_entry("confidence", gpr), "confidence", TODAY) == (
        "Durrett-arket (ARKEN → 🧭 Durrett)", "Tillstånd (Q1 2027)")
    ds = dict(RULE, fin_catalyst_text="Riktad emission", fin_catalyst_date="2027-02")
    assert sc.next_catalyst(_entry("producenter", ds), "producenter", TODAY) == (
        "Rick Rule · DS-finansiering", "Riktad emission (2027-02)")
    assert sc.next_catalyst(_entry("producenter", dict(ds, fin_catalyst_date="2026-09")), "producenter", TODAY) is None
    early = dict(LOB, fin_catalyst_text="Brygglån", fin_catalyst_date="2027-01")
    assert sc.next_catalyst(_entry("tiggre", early), "tiggre", TODAY)[1] == "Brygglån (2027-01)"   # den som kommer först


def test_period_end_reads_the_sheets_formats():
    assert sc._period_end("2027-03") == date(2027, 3, 31) and sc._period_end("2027-02-14") == date(2027, 2, 14)
    assert sc._period_end("2027-Q1") == date(2027, 3, 31) and sc._period_end("Q3 2027") == date(2027, 9, 30)
    assert sc._period_end("2027-H1") == date(2027, 6, 30) and sc._period_end("2027") == date(2027, 12, 31)
    for junk in ("", None, "snart", "2027-13", "2027-02-30", "2027-H3"):
        assert sc._period_end(junk) is None, junk


# ── Kortet (AppTest) ────────────────────────────────────────────────────────
def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


@pytest.fixture
def page(monkeypatch):
    """Scorecardet ur en fejkad lagring — inget GitHub, ingen Gist, ingen sifferuppdatering."""
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
    monkeypatch.setattr(refresh_ui, "load_refresh", lambda: {})

    def run():
        from streamlit.testing.v1 import AppTest

        def app(root):
            import sys as _s
            _s.path.insert(0, root)
            import scorecard as s
            s.render_scorecard_page()
        at = AppTest.from_function(app, args=(ROOT,), default_timeout=90)
        at.run()
        assert not at.exception, at.exception
        return at
    return run


def _caps(at) -> str:
    return " ".join(c.value for c in at.caption)


def _md(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_opening_the_page_shows_the_suggestions_and_writes_nothing(page):
    at = page()
    caps = _caps(at)
    assert "🤖 Ur Rick Rule: PASS — 4/5 p — Köpkandidat. Marginal och disciplin på plats. — förslag, inte inskrivet." in caps
    assert "📏 Tak 4 % av totalen · normalt 5–10 % av strategidelen · strömbrytaren NORMAL (5,0 % från topp)" in caps
    assert "Äger redan: 5,0 % av totalen (20 000 kr av 400 000 kr)." in caps
    assert ("🤖 Ur Tiggre (Lobo-arket): Finansieringsbesked (2099-06) — nästa väntande katalysator, förslag, inte "
            "inskrivet.") in caps
    assert "Tillstånd" not in caps and "Borrsvar" not in caps and "PFS (" not in caps
    note = review_link._REVIEWERS["insider"](INS)[1]
    assert f"🤖 Ur Insiderbevakaren: FAIL — {note}" in _md(at)
    keys = {b.key for b in at.button}
    assert f"sc_usegr_{BOL}" in keys and f"sc_usegr_{EVO}" not in keys and f"sc_usegr_{LOBO}" not in keys
    assert at.selectbox(key=f"sc_sellpick_{EVO}").value is None                     # inget förval
    for name in ("scorecard", "producers", "insider", "tiggre", "allocator"):
        assert _same(at.session_state[name], STORES[name]), name
    held = at.session_state["holdings"]["long"]                 # registret normaliseras vid läsning, inget mer
    assert [(r["ticker"], r["shares"], r["entry_price"]) for r in held] == [("BOL.ST", 100, 200.0)]


def test_anvand_ticks_the_review_box_and_the_hint_goes_away(page):
    at = page()
    assert at.checkbox(key=f"sc_g_{BOL}_granskning_klar").value is False
    at.button(key=f"sc_usegr_{BOL}").click().run()
    assert not at.exception, at.exception
    card = at.session_state["scorecard"]["cards"][BOL]
    assert card["granskning_klar"] is True and card["strategy"] == "producenter"
    assert at.checkbox(key=f"sc_g_{BOL}_granskning_klar").value is True
    assert "🤖 Ur Rick Rule: PASS" not in _caps(at)
    assert set(at.session_state["scorecard"]["cards"]) == {BOL}


def test_the_sell_rule_list_copies_the_text_and_resets(page):
    at = page()
    stop = "Stopp 42,50 (klustersnittet 50,00 × 0,85)"
    assert stop in at.selectbox(key=f"sc_sellpick_{EVO}").options
    at.selectbox(key=f"sc_sellpick_{EVO}").select(stop).run()
    assert not at.exception, at.exception
    assert at.text_input(key=f"sc_sell_{EVO}").value == stop
    card = at.session_state["scorecard"]["cards"][EVO]
    assert card["sell_rule"] == stop and card["strategy"] == "insider"
    assert at.selectbox(key=f"sc_sellpick_{EVO}").value is None


def test_anvand_writes_the_pending_catalyst_into_the_trigger(page):
    at = page()
    at.button(key=f"sc_usetrig_{LOBO}").click().run()
    assert not at.exception, at.exception
    assert at.text_input(key=f"sc_trig_{LOBO}").value == "Finansieringsbesked (2099-06)"
    card = at.session_state["scorecard"]["cards"][LOBO]
    assert card["trigger"] == "Finansieringsbesked (2099-06)" and card["strategy"] == "tiggre"
    assert "Finansieringsbesked" not in _caps(at)
    assert set(at.session_state["scorecard"]["cards"]) == {LOBO}
