"""
Tests for insider.py — Insiderbevakaren.

Asserted against insiderbevakaren.xlsx cell by cell. The status flow is a
nested IF whose ORDER is the rule, so the tests walk the same branches the
formula does rather than spot-checking outcomes.

  J: =IF(OR(D="",E="",F=""),"",IF(D>=3,3,IF(D=2,2,0))+IF(E="VD/CFO",2,
        IF(E="Styrelse",1,0))+IF(F>=1000,2,IF(F>=500,1,0))+G+H+I)
  N: =IF(J="","",IF(J<5,"Ignorera — brus",IF(J<7,"Bevaka — vänta på fler köp",
        IF(L="Nej","Stoppad i kvalitetsgrinden",IF(L="","Kör kvalitetsgrinden!",
        IF(OR(M="",M="Nej"),"Väntar på teknisk trigger",
        "KÖP — logga i journalen"))))))
  P: =IF(OR(O="",K="",K=0),"",O/K-1)
  Q: =IF(K="","",K*0.85)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import insider as ins


def _sig(**kw) -> dict:
    """A fully scored signal; override single fields per test."""
    base = {"id": "x", "ticker": "ABC", "insiders": 3, "role": ins.ROLE_TOP,
            "amount": 1200.0, "okar_25": False, "efter_fall": False,
            "aterkommande": False, "gate": ins.GATE_BLANK, "trigger": ""}
    base.update(kw)
    return base


# ── Poängen (kolumn J) ───────────────────────────────────────────────────────
def test_cluster_points():
    assert ins.cluster_points(3) == 3
    assert ins.cluster_points(7) == 3
    assert ins.cluster_points(2) == 2
    assert ins.cluster_points(1) == 0       # ensam köpare ger noll
    assert ins.cluster_points(0) == 0
    assert ins.cluster_points(None) == 0


def test_role_points():
    assert ins.role_points(ins.ROLE_TOP) == 2
    assert ins.role_points(ins.ROLE_BOARD) == 1
    assert ins.role_points(ins.ROLE_OTHER) == 0
    assert ins.role_points("Suppleant") == 0


def test_amount_points_at_the_sheets_boundaries():
    assert ins.amount_points(1000) == 2     # exakt 1 MSEK
    assert ins.amount_points(999.9) == 1
    assert ins.amount_points(500) == 1      # exakt 500 tkr
    assert ins.amount_points(499.9) == 0
    assert ins.amount_points(None) == 0


def test_score_sums_all_six_components():
    s = _sig(insiders=3, role=ins.ROLE_TOP, amount=1200,
             okar_25=True, efter_fall=True, aterkommande=True)
    assert ins.score(s) == 3 + 2 + 2 + 1 + 1 + 1 == ins.MAX_SCORE


def test_score_is_blank_until_the_three_required_fields_are_filled():
    """The sheet shows an empty cell, not a zero — a zero would read as 'brus'."""
    assert ins.score(_sig(insiders=None)) is None
    assert ins.score(_sig(amount=None)) is None
    assert ins.score(_sig(role="")) is None
    assert ins.status(_sig(insiders=None)) is None


def test_a_lone_large_buy_by_the_ceo_is_still_only_noise():
    """The point of the model: one buyer is not a cluster."""
    s = _sig(insiders=1, role=ins.ROLE_TOP, amount=5000)
    assert ins.score(s) == 0 + 2 + 2 == 4
    assert ins.status(s) == ins.S_NOISE


# ── Statusflödet (kolumn N) — ordningen är regeln ────────────────────────────
def test_below_five_is_noise_and_five_to_six_is_watch():
    assert ins.status(_sig(insiders=2, role=ins.ROLE_OTHER, amount=100)) == ins.S_NOISE
    # 2 + 1 + 2 = 5 -> bevaka
    s5 = _sig(insiders=2, role=ins.ROLE_BOARD, amount=1200)
    assert ins.score(s5) == 5
    assert ins.status(s5) == ins.S_WATCH
    s6 = _sig(insiders=2, role=ins.ROLE_BOARD, amount=1200, okar_25=True)
    assert ins.score(s6) == 6
    assert ins.status(s6) == ins.S_WATCH


def test_seven_points_sends_you_to_the_gate():
    s = _sig()                       # 3 + 2 + 2 = 7
    assert ins.score(s) == 7
    assert ins.status(s) == ins.S_RUN_GATE


def test_a_failed_gate_stops_the_signal_however_high_the_score():
    s = _sig(okar_25=True, efter_fall=True, aterkommande=True,
             gate=ins.GATE_NO, trigger="A")
    assert ins.score(s) == ins.MAX_SCORE
    assert ins.status(s) == ins.S_GATE_FAIL


def test_a_passed_gate_without_a_trigger_waits():
    assert ins.status(_sig(gate=ins.GATE_YES)) == ins.S_WAIT_TRIGGER
    assert ins.status(_sig(gate=ins.GATE_YES, trigger="Nej")) == ins.S_WAIT_TRIGGER


def test_gate_plus_trigger_is_the_only_route_to_buy():
    for trigger in ("A", "B", "C"):
        assert ins.status(_sig(gate=ins.GATE_YES, trigger=trigger)) == ins.S_BUY


def test_the_full_flow_in_order():
    """Walk one signal from noise to buy, one field at a time."""
    s = _sig(insiders=1, role=ins.ROLE_OTHER, amount=100)
    assert ins.status(s) == ins.S_NOISE
    s["role"] = ins.ROLE_BOARD; s["insiders"] = 2; s["amount"] = 1200
    assert ins.status(s) == ins.S_WATCH
    s["insiders"] = 3; s["role"] = ins.ROLE_TOP; s["amount"] = 1200
    assert ins.status(s) == ins.S_RUN_GATE
    s["gate"] = ins.GATE_NO
    assert ins.status(s) == ins.S_GATE_FAIL
    s["gate"] = ins.GATE_YES
    assert ins.status(s) == ins.S_WAIT_TRIGGER
    s["trigger"] = "B"
    assert ins.status(s) == ins.S_BUY


# ── Kurskolumnerna (P och Q) ─────────────────────────────────────────────────
def test_vs_cluster_and_stop():
    s = _sig(cluster_avg=100.0, price_now=130.0)
    assert round(ins.vs_cluster(s), 6) == 30.0
    assert ins.stop_price(s) == 85.0
    assert ins.STOP_FRAC == 0.85


def test_price_columns_are_blank_without_a_cluster_average():
    """The sheet guards K=0 explicitly — a zero average must not divide."""
    assert ins.vs_cluster(_sig(cluster_avg=0, price_now=50)) is None
    assert ins.vs_cluster(_sig(cluster_avg=None, price_now=50)) is None
    assert ins.vs_cluster(_sig(cluster_avg=100, price_now=None)) is None
    assert ins.stop_price(_sig(cluster_avg=None)) is None


def test_the_chase_rule_triggers_above_thirty_percent():
    assert not ins.is_chase(_sig(cluster_avg=100, price_now=130))   # exakt +30
    assert ins.is_chase(_sig(cluster_avg=100, price_now=131))
    assert not ins.is_chase(_sig(cluster_avg=100, price_now=90))
    assert not ins.is_chase(_sig())                # inga kurser = ingen flagga
    assert ins.CHASE_PCT == 30.0


# ── Listan ───────────────────────────────────────────────────────────────────
def test_ranked_sorts_by_score_and_survives_incomplete_rows():
    sigs = [_sig(id="a", ticker="LOW", insiders=1, role=ins.ROLE_OTHER, amount=100),
            _sig(id="b", ticker="HIGH", okar_25=True, efter_fall=True,
                 aterkommande=True),
            {"id": "c", "ticker": "EMPTY"}]
    rows = ins.ranked(sigs)
    assert [r["signal"]["ticker"] for r in rows] == ["HIGH", "LOW", "EMPTY"]
    assert rows[-1]["score"] is None and rows[-1]["status"] is None
    assert ins.ranked([]) == []
    assert ins.ranked(None) == []


def test_buy_candidates_exclude_chases():
    ready = _sig(id="a", ticker="OK", gate=ins.GATE_YES, trigger="A",
                 cluster_avg=100, price_now=110)
    chased = _sig(id="b", ticker="LATE", gate=ins.GATE_YES, trigger="A",
                  cluster_avg=100, price_now=140)
    waiting = _sig(id="c", ticker="WAIT", gate=ins.GATE_YES)
    picks = [r["signal"]["ticker"] for r in
             ins.buy_candidates([ready, chased, waiting])]
    assert picks == ["OK"]


# ── Mot resten av panelen ────────────────────────────────────────────────────
def test_the_gate_matches_the_snabbreferens_screener():
    """Both describe the same Börsdata filter — drift would be a real bug."""
    import reference
    filters = reference.screener("insider").filters
    assert "300 MSEK" in filters
    assert any("300 MSEK" in c for c in ins.GATE_CRITERIA)
    assert "F-score ≥ 5" in filters
    assert any("F-score ≥ 5" in c for c in ins.GATE_CRITERIA)


def test_the_sell_rules_match_the_snabbreferens_row():
    import reference
    row = reference.sell_rule("insider").rule
    assert "−15 %" in row
    assert any("−15 %" in what for _l, what, _w in ins.SELL_RULES)
    assert "18 mån" in row
    assert any("18 månader" in what for _l, what, _w in ins.SELL_RULES)
