"""
Tester för cycle.py — cykelläget in i Copilotens köpgrind.

Det viktiga här är översättningen: rotationsflikens AGERA/Bevaka/Vila ska bli
PASS/MANUAL/FAIL utan att något obetygsatt läge slinker igenom som ett ja.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cycle
import rotation


def _grade(hatred=5, fundamentals=5, catalyst=5, intact=True):
    return {"hatred": hatred, "fundamentals": fundamentals,
            "catalyst": catalyst, "case_intact": intact}


def _rotation_data(**grades):
    return {"month": "2026-08", "grades": grades, "history": []}


# ── Vilka strategier som alls berörs ─────────────────────────────────────────
def test_only_the_guides_commodity_strategies_require_a_cycle():
    for key in ("rule", "durrett", "sprott", "tiggre", "royalty"):
        assert cycle.requires_cycle(key) is True, key
    for key in ("momentum", "insider", "quality", "", None):
        assert cycle.requires_cycle(key) is False, key


# ── Ticker → råvara ──────────────────────────────────────────────────────────
def test_the_commodity_comes_from_the_rick_rule_sheet():
    data = {"producers": [{"ticker": "EQX", "commodity": "Guld"},
                          {"ticker": "CCJ", "commodity": "Uran"}],
            "royalty": [{"ticker": "FNV"}]}
    assert cycle.commodity_for_ticker("eqx", data) == "Guld"
    assert cycle.commodity_for_ticker("CCJ", data) == "Uran"
    # royalty-arket har ingen råvarukolumn — inget att hämta
    assert cycle.commodity_for_ticker("FNV", data) is None
    assert cycle.commodity_for_ticker("OKÄND", data) is None
    assert cycle.commodity_for_ticker("", data) is None
    assert cycle.commodity_for_ticker("EQX", {}) is None


def test_commodity_key_accepts_name_and_key():
    assert cycle.commodity_key("Guld") == "guld"
    assert cycle.commodity_key("guld") == "guld"
    assert cycle.commodity_key("Uran") == "uran"
    assert cycle.commodity_key("Bitcoin") is None
    assert cycle.commodity_key("") is None


# ── Cykelläget ───────────────────────────────────────────────────────────────
def test_an_agera_grade_reads_through():
    data = _rotation_data(guld=_grade(5, 5, 5))
    state = cycle.cycle_state("Guld", data)
    assert state["status"] == rotation.AGERA
    assert state["sum"] == 15 and state["max"] == 15
    assert state["month"] == "2026-08"


def test_an_ungraded_commodity_is_none_not_vila():
    """Obetygsatt betyder att månadens gradering inte är gjord. Vila betyder
    att den är gjord och sa nej. De får inte blandas ihop."""
    assert cycle.cycle_state("Guld", _rotation_data()) is None
    assert cycle.cycle_state("Guld", {}) is None
    assert cycle.cycle_state("", _rotation_data(guld=_grade())) is None


def test_a_broken_case_is_vila_no_matter_the_sum():
    data = _rotation_data(guld=_grade(5, 5, 5, intact=False))
    state = cycle.cycle_state("Guld", data)
    assert state["status"] == rotation.VILA
    assert "brutet" in state["why"]


def test_the_value_trap_warning_travels_with_the_state():
    data = _rotation_data(guld=_grade(hatred=5, fundamentals=1, catalyst=5))
    state = cycle.cycle_state("Guld", data)
    assert rotation.WARN_VALUE_TRAP in state["warnings"]


# ── Grindöversättningen ──────────────────────────────────────────────────────
def test_agera_passes_bevaka_is_manual_vila_fails():
    agera = cycle.cycle_state("Guld", _rotation_data(guld=_grade(5, 5, 5)))
    bevaka = cycle.cycle_state("Guld", _rotation_data(guld=_grade(4, 3, 3)))
    vila = cycle.cycle_state("Guld", _rotation_data(guld=_grade(2, 2, 2)))
    assert cycle.gate_from_cycle(agera, "Guld")[0] == "PASS"
    assert cycle.gate_from_cycle(bevaka, "Guld")[0] == "MANUAL"
    assert cycle.gate_from_cycle(vila, "Guld")[0] == "FAIL"


def test_the_gate_note_carries_the_source_and_month():
    state = cycle.cycle_state("Guld", _rotation_data(guld=_grade(5, 5, 5)))
    status, note = cycle.gate_from_cycle(state, "Guld")
    assert "Guld" in note and "15/15" in note and "2026-08" in note


def test_ungraded_and_unchosen_are_manual_with_instructions():
    """Aldrig ett tyst godkännande — men inte heller ett falskt nej."""
    status, note = cycle.gate_from_cycle(None, "Guld")
    assert status == "MANUAL" and "rotationsfliken" in note
    status, note = cycle.gate_from_cycle(None, "")
    assert status == "MANUAL" and "råvara" in note.lower()


def test_warnings_reach_the_gate_note():
    state = cycle.cycle_state(
        "Guld", _rotation_data(guld=_grade(hatred=5, fundamentals=1,
                                           catalyst=5)))
    _status, note = cycle.gate_from_cycle(state, "Guld")
    assert "VARNING" in note


# ── Blindspot-läsningen ──────────────────────────────────────────────────────
def test_blindspot_latest_reads_the_committed_history():
    """Historikfilen ligger i repot med riktiga rader — CCJ finns i den."""
    row = cycle.blindspot_latest("CCJ")
    assert row is not None
    assert row["ticker"] == "CCJ"
    assert "opportunity" in row and "timestamp" in row


def test_blindspot_latest_is_none_for_unknown_or_empty():
    assert cycle.blindspot_latest("FINNSINTE123") is None
    assert cycle.blindspot_latest("") is None
    assert cycle.blindspot_latest(None) is None


# ── Marknadscykelfasen (contrarian/quality) ──────────────────────────────────
def test_phase_sets_are_read_from_the_playbooks_verbatim():
    """Fasmängderna PARSAS ur strategy_rules — ändras playbooken följer
    grinden med. Testet pinnar dagens innehåll så en oavsiktlig ändring syns."""
    con = cycle.playbook_phases("contrarian")
    assert con["buy"] == {"CAPITULATION", "DEPRESSION", "DISBELIEF",
                          "ANGER", "PANIC", "HOPE"}
    assert con["hold"] == {"OPTIMISM", "BELIEF"}
    assert con["sell"] == {"THRILL", "EUPHORIA", "COMPLACENCY",
                           "ANXIETY", "DENIAL"}
    qua = cycle.playbook_phases("quality")
    assert qua["buy"] == {"DISBELIEF", "HOPE", "OPTIMISM", "BELIEF",
                          "DISBELIEF_NEW"}
    assert qua["hold"] == set()
    assert qua["sell"] == {"THRILL", "EUPHORIA", "COMPLACENCY",
                           "ANXIETY", "DENIAL"}
    # HOLD-faser får aldrig läcka in i köp- eller säljmängden
    assert not con["buy"] & con["hold"] and not con["sell"] & con["hold"]


def test_only_contrarian_and_quality_use_the_market_cycle():
    assert cycle.requires_market_cycle("contrarian")
    assert cycle.requires_market_cycle("quality")
    for key in ("rule", "momentum", "insider", "", None):
        assert not cycle.requires_market_cycle(key), key


def test_the_phase_gate_translates_buy_hold_sell():
    def st_(phase):
        return {"phase": phase, "confidence": 70.0}
    assert cycle.gate_from_market_phase(st_("CAPITULATION"),
                                        "contrarian")[0] == "PASS"
    status, note = cycle.gate_from_market_phase(st_("OPTIMISM"), "contrarian")
    assert status == "MANUAL" and "HOLD" in note
    status, note = cycle.gate_from_market_phase(st_("EUPHORIA"), "contrarian")
    assert status == "FAIL" and "distributionsfas" in note
    # quality köper i OPTIMISM — samma fas, olika playbook, olika svar
    assert cycle.gate_from_market_phase(st_("OPTIMISM"), "quality")[0] == "PASS"


def test_an_unknown_phase_is_manual_never_a_yes():
    assert cycle.gate_from_market_phase(None, "contrarian")[0] == "MANUAL"
    status, note = cycle.gate_from_market_phase(
        {"phase": "PÅHITTAD", "confidence": 50.0}, "quality")
    assert status == "MANUAL" and "nämns inte" in note


def test_market_phase_reports_errors_instead_of_hiding_them(monkeypatch):
    state, err = cycle.market_phase("")
    assert state is None and err
