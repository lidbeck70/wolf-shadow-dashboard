"""
Tests for scorecard.py — Master Scorecard och köpgrinden (tilläggsspec E).

The rule that matters most here: "Luckor i tabellen = standardbeslut INGEN
AFFÄR". An unassessed control must read as a no, not as a silent yes — which
is the opposite of how the strategies' own gates treat a blank DS, and the
difference is deliberate.
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import controls as ctl
import lukacs as lk
import scorecard as sc


def _fv_filled() -> dict:
    """Lukacs FV ifylld så att köpgrindens steg 5 blir mekaniskt grönt."""
    return {"fcf_kvalitet": "B", "framtida_antal_aktier": 1200.0,
            "aktuell_kurs": 5.0,
            "what_must_go_right": "Guld $2 400 · 500 koz · capex $180M",
            "fv": {ctl.BEAR: {"forward_fcf_musd": 400, "target_yield": 9},
                   ctl.BASE: {"forward_fcf_musd": 900, "target_yield": 9},
                   ctl.BULL: {"forward_fcf_musd": 1600, "target_yield": 9}}}


def _sources() -> dict:
    return {
        "sprott": [{"id": "1", "ticker": "AAA", "name": "Alfa Mining"}],
        "tiggre": [{"id": "2", "ticker": "BBB", "name": "Beta Gold"}],
        "producenter": [{"id": "3", "ticker": "AAA", "name": "Alfa Mining"}],
    }


def _full_card(**kw) -> dict:
    card = {g.key: True for g in sc.GATES}
    card.update({"strategy": "swing", "position_pct_total": 5.0})
    card.update(kw)
    return card


# ── Sammanställningen ────────────────────────────────────────────────────────
def test_the_same_company_in_two_tabs_becomes_one_candidate():
    """That is the point of the bolag+ticker key — nothing is entered twice."""
    entries = sc.collect(_sources())
    assert [e["ticker"] for e in entries] == ["AAA", "BBB"]
    alfa = entries[0]
    assert set(alfa["strategies"]) == {"sprott", "producenter"}
    assert set(alfa["rows"]) == {"sprott", "producenter"}


def test_the_key_ignores_case_and_padding():
    entries = sc.collect({
        "sprott": [{"ticker": "aaa ", "name": " Alfa Mining"}],
        "durrett": [{"ticker": "AAA", "name": "alfa mining"}]})
    assert len(entries) == 1
    assert entries[0]["ticker"] == "AAA"


def test_rows_without_a_ticker_are_skipped():
    entries = sc.collect({"sprott": [{"name": "Namnlös"}, {"ticker": "  "},
                                     None, "skräp"]})
    assert entries == []


def test_collect_survives_empty_input():
    assert sc.collect({}) == []
    assert sc.collect(None) == []
    assert sc.collect({"sprott": None}) == []


def test_source_row_prefers_the_named_strategy():
    entry = sc.collect(_sources())[0]
    assert sc.source_row(entry, "sprott")["id"] == "1"
    assert sc.source_row(entry, "producenter")["id"] == "3"
    assert sc.source_row(entry, "saknas")["id"] in ("1", "3")   # faller tillbaka
    assert sc.source_row({}) == {}


# ── Köpgrinden ───────────────────────────────────────────────────────────────
def test_the_gate_has_the_specs_seven_checks():
    assert len(sc.GATES) == 7
    assert [g.key for g in sc.GATES] == [
        "strategi_aktiv", "screener_kval", "granskning_klar",
        "inga_roda_flaggor", "sakerhetsmarginal", "trigger_definierad",
        "position_saljregel"]


def test_all_seven_green_means_ready():
    """Swing at 5 %: proportionality asks for nothing beyond the strategy."""
    entry = _entry_with({}, "swing")
    assert sc.is_ready(_full_card(), entry)
    assert sc.gate_state(_full_card(), entry)["missing"] == []


def test_a_single_unchecked_box_blocks_and_names_itself():
    entry = _entry_with({}, "swing")
    card = _full_card()
    card["sakerhetsmarginal"] = False
    state = sc.gate_state(card, entry)
    assert not state["ready"]
    assert state["missing"] == ["Värderingen ger säkerhetsmarginal"]


def test_an_empty_card_lists_every_missing_check():
    state = sc.gate_state({}, _entry_with({}, "swing"))
    assert not state["ready"]
    assert len(state["missing"]) == len(sc.GATES)


def test_a_card_without_candidate_data_never_signs_off():
    """No source row is a gap, not a pass — the controls cannot be read."""
    state = sc.gate_state(_full_card())
    assert not state["ready"]
    assert "Kontrollerna utan permanent-risk-flagga" in state["missing"]
    assert state["gaps"] == ["Kandidatdata saknas — kontrollerna går inte att läsa"]


# ── Luck-regeln ──────────────────────────────────────────────────────────────
def _entry_with(row: dict, strategy: str = "tiggre") -> dict:
    return sc.collect({strategy: [dict(row, ticker="CCC", name="Gamma")]})[0]


def test_an_unassessed_control_reads_as_a_no_not_a_silent_yes():
    """The scorecard is stricter than the strategy gates, by design."""
    entry = _entry_with({})
    card = _full_card(strategy="tiggre", position_pct_total=4.0)
    state = sc.gate_state(card, entry)
    assert not state["ready"]
    assert "DS är inte bedömd" in state["gaps"]
    assert "AQS är inte bedömd" in state["gaps"]
    assert "CSM är inte ifylld för alla scenarier" in state["gaps"]
    # ...och i strategins egen grind gör en tom DS inget alls
    assert not ctl.ds_blocks_buy({})


def test_the_control_check_cannot_be_ticked_past():
    """It is the box one ticks out of habit, so it is not a checkbox."""
    entry = _entry_with({})
    card = _full_card(strategy="tiggre", position_pct_total=4.0)
    card["inga_roda_flaggor"] = True          # försök kryssa förbi
    state = sc.gate_state(card, entry)
    assert state["checks"]["inga_roda_flaggor"] is False
    assert "Kontrollerna utan permanent-risk-flagga" in state["missing"]


def _producer_row() -> dict:
    row = {f.key: 0 for f in ctl.DS_FIELDS}
    row.update({f.key: 2 for f in ctl.AQS_FIELDS})
    row["csm"] = {s: {"price": 100, "fcf_musd": 10} for s in ctl.SCENARIOS_3}
    row["csm_kind"] = ctl.PRODUCER
    return row


def test_a_complete_workup_clears_the_gaps():
    row = {**_producer_row(), **_fv_filled()}
    entry = _entry_with(row, "producenter")
    card = _full_card(strategy="producenter", position_pct_total=4.0)
    state = sc.gate_state(card, entry)
    assert state["gaps"] == []
    assert state["valuation_gaps"] == []
    assert state["ready"]


# ── Steg 5: säkerhetsmarginalen ──────────────────────────────────────────────
def test_step_five_cannot_be_ticked_past_where_lukacs_fv_is_required():
    """Över 2 % av totalen är säkerhetsmarginalen en uträkning, inte ett omdöme."""
    entry = _entry_with(_producer_row(), "producenter")
    card = _full_card(strategy="producenter", position_pct_total=4.0)
    card["sakerhetsmarginal"] = True          # försök kryssa förbi
    state = sc.gate_state(card, entry)
    assert state["fv_mechanical"] is True
    assert state["checks"]["sakerhetsmarginal"] is False
    assert "Värderingen ger säkerhetsmarginal" in state["missing"]
    assert state["valuation_gaps"]


def test_step_five_stays_manual_where_the_module_does_not_apply():
    """Insider och swing värderas inte på forward FCF — krysset står kvar."""
    entry = _entry_with({}, "insider")
    card = _full_card(strategy="insider", position_pct_total=4.0)
    state = sc.gate_state(card, entry)
    assert state["fv_mechanical"] is False
    assert state["checks"]["sakerhetsmarginal"] is True
    assert state["valuation_gaps"] == []


def test_step_five_stays_manual_under_two_percent():
    """Proportionalitetsregeln: en liten position kostar inte fullt arbete."""
    entry = _entry_with(_producer_row(), "producenter")
    card = _full_card(strategy="producenter", position_pct_total=1.5)
    state = sc.gate_state(card, entry)
    assert state["fv_mechanical"] is False
    assert state["checks"]["sakerhetsmarginal"] is True


def test_a_thin_margin_of_safety_blocks_the_gate():
    row = {**_producer_row(), **_fv_filled()}
    row["aktuell_kurs"] = 7.5                 # FV base 8,33 -> MoS ~10 %
    entry = _entry_with(row, "producenter")
    card = _full_card(strategy="producenter", position_pct_total=4.0)
    state = sc.gate_state(card, entry)
    assert not state["ready"]
    assert any("Säkerhetsmarginal" in g for g in state["valuation_gaps"])


def test_what_must_go_right_is_required_for_green():
    row = {**_producer_row(), **_fv_filled(), "what_must_go_right": ""}
    entry = _entry_with(row, "producenter")
    card = _full_card(strategy="producenter", position_pct_total=4.0)
    state = sc.gate_state(card, entry)
    assert state["valuation_gaps"] == [lk.WMGR_MISSING]
    assert not state["ready"]


# ── Deleveraging-taket ───────────────────────────────────────────────────────
def test_debt_over_one_halves_the_position_cap():
    """Rule-benets tak är 4 % — med skuld över 1,0x blir det 2 %."""
    row = {**_producer_row(), **_fv_filled(),
           "nd_ebitda": 1.8, "ar_till_lag_skuld": 2.0}
    entry = _entry_with(row, "producenter")
    card = _full_card(strategy="producenter", position_pct_total=4.0)
    assert sc.deleveraging_cap(entry, card) == 2.0
    state = sc.gate_state(card, entry)
    assert state["position_gaps"]
    assert state["checks"]["position_saljregel"] is False
    assert not state["ready"]
    # inom taket passerar samma kort
    card["position_pct_total"] = 2.0
    ok = sc.gate_state(card, entry)
    assert ok["position_gaps"] == []
    assert ok["ready"]


def test_low_debt_leaves_the_cap_alone():
    row = {**_producer_row(), **_fv_filled(), "nd_ebitda": 0.15}
    entry = _entry_with(row, "producenter")
    card = _full_card(strategy="producenter", position_pct_total=4.0)
    assert sc.deleveraging_cap(entry, card) is None
    assert sc.gate_state(card, entry)["position_gaps"] == []


def test_a_red_flag_is_a_gap_even_when_everything_is_filled_in():
    row = {f.key: 0 for f in ctl.DS_FIELDS}
    row.update({f.key: 2 for f in ctl.AQS_FIELDS})
    row["csm_kind"] = ctl.DEVELOPER
    row["csm"] = {s: {"price": 100, "nav_musd": 500, "financing_need": 0}
                  for s in ctl.SCENARIOS_3}
    row["csm"][ctl.BEAR]["financing_need"] = 200
    entry = _entry_with(row)
    card = _full_card(strategy="tiggre", position_pct_total=4.0)
    assert ctl.CSM_BEAR_FAIL in sc.gate_state(card, entry)["gaps"]


def test_a_failed_aqs_band_is_a_gap():
    row = {f.key: 0 for f in ctl.DS_FIELDS}
    row.update({f.key: 0 for f in ctl.AQS_FIELDS})      # AQS 0 -> PASS
    row["csm"] = {s: {"price": 1, "fcf_musd": 1} for s in ctl.SCENARIOS_3}
    entry = _entry_with(row, "producenter")
    card = _full_card(strategy="producenter", position_pct_total=4.0)
    gaps = sc.gate_state(card, entry)["gaps"]
    assert any(ctl.AQS_PASS in g for g in gaps)


def test_high_dilution_without_financing_is_a_gap():
    row = {f.key: 2 for f in ctl.DS_FIELDS}
    entry = _entry_with(row, "sprott")
    card = _full_card(strategy="sprott", position_pct_total=1.5)
    gaps = sc.gate_state(card, entry)["gaps"]
    assert any("låser köpet" in g for g in gaps)
    # dateras finansieringen försvinner luckan
    row.update({"fin_catalyst_text": "Emission klar", "fin_catalyst_date": "2027-02"})
    entry = _entry_with(row, "sprott")
    assert sc.gate_state(card, entry)["gaps"] == []


# ── Proportionalitetsregeln i praktiken ──────────────────────────────────────
def test_a_small_position_is_not_asked_for_aqs_or_csm():
    entry = _entry_with({})
    card = _full_card(strategy="sprott", position_pct_total=1.5)
    gaps = sc.gate_state(card, entry)["gaps"]
    assert gaps == ["DS är inte bedömd"]        # AQS och CSM efterfrågas inte


def test_swing_needs_no_controls_at_all():
    entry = _entry_with({}, "swing")
    card = _full_card(strategy="swing", position_pct_total=6.0)
    assert sc.gate_state(card, entry)["gaps"] == []
    assert sc.is_ready(card, entry)


def test_swing_with_the_dilution_flag_gets_ds_back():
    entry = _entry_with({"dilution_risk": True}, "swing")
    card = _full_card(strategy="swing", position_pct_total=6.0)
    assert sc.gate_state(card, entry)["gaps"] == ["DS är inte bedömd"]


def test_control_state_reports_what_is_required():
    entry = _entry_with({})
    big = sc.control_state(entry, 5.0, "tiggre")["required"]
    small = sc.control_state(entry, 1.0, "tiggre")["required"]
    assert ctl.SEC_CSM in big and ctl.SEC_CSM not in small


# ── Årsregeln ────────────────────────────────────────────────────────────────
def test_cards_untouched_for_a_year_are_listed():
    cards = [{"strategy": "sprott", "last_decision": "2025-08-15"},
             {"strategy": "tiggre", "last_decision": "2026-06-01"},
             {"strategy": "durrett"}]
    stale = sc.stale_controls(cards, today=date(2026, 8, 15))
    assert len(stale) == 1
    assert stale[0]["card"]["strategy"] == "sprott"
    assert stale[0]["days"] == 365
    assert sc.STALE_DAYS == 365


def test_stale_survives_junk_dates():
    assert sc.stale_controls([{"last_decision": "inte ett datum"},
                              {"last_decision": None}, {}, None],
                             today=date(2026, 8, 15)) == []
    assert sc.stale_controls([], today=date(2026, 8, 15)) == []
    assert sc.stale_controls(None) == []


def test_the_default_decision_text_is_the_guides():
    assert sc.NO_TRADE == "Luckor i tabellen = standardbeslut INGEN AFFÄR"
    assert sc.READY == "KLAR FÖR KÖP"
