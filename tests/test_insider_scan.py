"""
Insiderbevakaren headless — insider_scan.py fyller arkets INMATNINGSFÄLT ur
Börsdatas insynsregister; poäng/status/stopp räknas med insider.py:s egna
funktioner. Inga trösklar ändras här — testerna kontrollerar att fälten
fylls som arket förväntar sig och att larmbenet bara larmar på övergångar.
"""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import insider as ins
import insider_scan as sc
import alert_rules as ar

TODAY = date(2026, 9, 10)


def _d(days_ago):
    return (TODAY - timedelta(days=days_ago)).isoformat() + "T00:00:00"


def _tx(owner, pos, shares, price, days_ago, eq=False):
    return {"ownerName": owner, "ownerPosition": pos, "shares": shares,
            "price": price, "amount": shares * price, "transactionType": 19,
            "equityProgram": eq, "transactionDate": _d(days_ago)}


# ── Klustret ur transaktionerna ──────────────────────────────────────────────
def test_cluster_counts_real_buys_in_the_30_day_window():
    values = [
        _tx("Anna", "cfo", 1000, 500.0, 5),                 # 500 tkr
        _tx("Bertil", "board member", 2000, 480.0, 12),     # 960 tkr
        _tx("Carl", "other management", 100, 490.0, 40),    # utanför 30 dgr
        _tx("Doris", "ceo", 5000, 470.0, 3, eq=True),       # program → räknas ej
        {"ownerName": "Erik", "ownerPosition": "board member", "shares": -300,
         "price": 500.0, "amount": -150000, "transactionDate": _d(2)},   # sälj
        _tx("Bertil", "board member", 500, 400.0, 200),     # återkommande (12 mån)
    ]
    cl = sc.build_cluster(values, TODAY)
    assert cl["insiders"] == 2
    assert cl["role"] == ins.ROLE_TOP            # cfo slår styrelse
    assert cl["amount"] == 1460.0                # tkr
    assert cl["aterkommande"] is True            # Bertil köpte även för 200 dgr sedan
    # aktieviktad snittkurs: (1000*500 + 2000*480) / 3000
    assert abs(cl["cluster_avg"] - 486.6667) < 0.001
    assert cl["first_buy"] == (TODAY - timedelta(days=12)).isoformat()
    assert sc.build_cluster([_tx("X", "ceo", 10, 1.0, 45)], TODAY) is None


def test_role_mapping_follows_the_sheet():
    assert sc.role_from_position("external ceo") == ins.ROLE_TOP
    assert sc.role_from_position("CFO") == ins.ROLE_TOP
    assert sc.role_from_position("board member") == ins.ROLE_BOARD
    assert sc.role_from_position("other management") == ins.ROLE_OTHER
    assert sc.role_from_position(None) == ins.ROLE_OTHER


def test_fx_scales_amount_to_sek():
    cl = sc.build_cluster([_tx("A", "ceo", 100, 100.0, 1)], TODAY, fx=11.3)
    assert cl["amount_local"] == 10.0 and cl["amount"] == 113.0


# ── Grind, trigger, efter fall ───────────────────────────────────────────────
def _snap(**over):
    base = {"market_cap": 900.0, "f_score": 6, "net_debt_m": 50.0,
            "net_debt_ebitda": 1.2, "fcf_m": 30.0, "revenue_growth": 0.05}
    base.update(over)
    return base


def test_gate_yes_only_when_everything_measurable_is_ok():
    gate, checks = sc.auto_gate(_snap())
    assert gate == ins.GATE_YES and all(c["status"] == "ok" for c in checks)
    assert [c["label"] for c in checks] == list(ins.GATE_CRITERIA)

    gate, _ = sc.auto_gate(_snap(market_cap=120.0))
    assert gate == ins.GATE_NO                       # 120 MSEK < 300
    gate, _ = sc.auto_gate(_snap(net_debt_ebitda=2.6))
    assert gate == ins.GATE_NO
    gate, _ = sc.auto_gate(_snap(net_debt_m=-10.0, net_debt_ebitda=None))
    assert gate == ins.GATE_YES                      # nettokassa passerar

    # F-score utanför licensen → inte Ja, inte Nej: kör grinden själv
    gate, checks = sc.auto_gate(_snap(f_score=None))
    assert gate == ins.GATE_BLANK
    assert any(c["status"] == "unknown" and "F-score" in c["label"] for c in checks)
    # negativt FCF är en bedömning ("tydlig väg dit"), inte ett automatiskt nej
    gate, _ = sc.auto_gate(_snap(fcf_m=-5.0))
    assert gate == ins.GATE_BLANK
    # valuta: 40 MEUR = 452 MSEK → passerar
    gate, _ = sc.auto_gate(_snap(market_cap=40.0), fx=11.3)
    assert gate == ins.GATE_YES


def test_trigger_a_then_b_then_no():
    flat = [100.0] * 30
    rising = [100.0] * 10 + [100 + i for i in range(20)]      # över MA20, MA20 vänt
    assert sc.auto_trigger(rising, 90.0)[0] == "A"
    # under MA20 men positiv månad och över klustersnittet → B
    b = [90.0] * 10 + [130.0] * 15 + [100.0, 101.0, 102.0, 103.0, 104.0]
    trig, note = sc.auto_trigger(b, 90.0)
    assert trig == "B" and "klustersnitt" in note
    assert sc.auto_trigger(flat, 120.0)[0] == "Nej"
    assert sc.auto_trigger([100.0] * 5, 90.0)[0] == ""


def test_efter_fall_needs_both_drawdown_and_known_fscore():
    closes = [100.0] * 100 + [70.0]
    dd = sc.drawdown_52w(closes)
    assert dd == 30.0
    assert sc.auto_efter_fall(dd, 6)[0] is True
    assert sc.auto_efter_fall(dd, 3)[0] is False
    ok, note = sc.auto_efter_fall(dd, None)
    assert ok is False and "kryssa själv" in note
    assert sc.auto_efter_fall(10.0, 8)[0] is False


# ── Hela skanningen mot ett fejkat API ──────────────────────────────────────
class _FakeAPI:
    def __init__(self):
        self.calls = []

    def get_instruments(self):
        return [
            {"insId": 1, "ticker": "EKTA B", "name": "Elekta", "marketId": 1,
             "stockPriceCurrency": "SEK"},
            {"insId": 2, "ticker": "NOD", "name": "Nordic", "marketId": 4,
             "stockPriceCurrency": "NOK"},
            {"insId": 3, "ticker": "OMXS30", "name": "Index", "marketId": 7},
            {"insId": 4, "ticker": "US", "name": "Utanför", "marketId": 30},
        ]

    def get_insider_transactions_batch(self, ids):
        self.calls.append(("insider", list(ids)))
        return {
            1: [_tx("Anna", "ceo", 2000, 600.0, 4),          # 1200 tkr
                _tx("Bertil", "board member", 500, 590.0, 9),
                _tx("Carl", "other management", 100, 595.0, 20)],   # 3 insiders
            2: [_tx("Ola", "other management", 10, 50.0, 3)],      # ensam, litet → brus
        }

    def get_fundamentals_snapshot_fast(self, ids):
        self.calls.append(("fund", list(ids)))
        return {1: _snap()}

    def get_stockprices(self, ins_id, max_count=260):
        self.calls.append(("prices", ins_id))
        closes = [100.0] * 10 + [100 + i for i in range(20)]
        return [{"d": f"2026-0{1 + i // 28}-{1 + i % 28:02d}", "c": c}
                for i, c in enumerate(closes)]


def test_scan_fills_the_sheet_fields_and_uses_sheet_scoring():
    api = _FakeAPI()
    out = sc.scan(api, today=TODAY)
    assert out["error"] is None
    assert out["universe"] == 2                       # index och icke-nordiskt bort
    assert out["with_buys"] == 2
    assert [c["ticker"] for c in out["clusters"]] == ["EKTA-B.ST"]
    row = out["clusters"][0]
    # bara det kvalificerade klustret hämtade fundamenta och kurser
    assert ("fund", [1]) in api.calls and ("prices", 1) in api.calls
    assert ("prices", 2) not in api.calls
    assert row["insiders"] == 3 and row["role"] == ins.ROLE_TOP
    assert row["amount"] == 1554.5
    assert row["gate"] == ins.GATE_YES and row["trigger"] == "A"
    # 3 insiders (3p) + VD (2p) + > 1 MSEK (2p) = 7 — okar_25 aldrig auto,
    # efter_fall bara vid fall (kursen har stigit)
    assert row["okar_25"] is False and row["efter_fall"] is False
    assert row["score"] == 7 == ins.score(row)
    assert row["status"] == ins.status(row) == ins.S_BUY
    assert row["price_now"] == 119.0
    assert row["stop"] == round(row["cluster_avg"] * ins.STOP_FRAC, 2)
    assert row["auto"]["okar_25_note"]
    # arkraden ur klustret innehåller bara arkets fält + kommentar
    sig = ins.auto_to_signal(row)
    assert set(sig) <= set(ins.AUTO_FIELDS) | {"id", "comment"}
    assert ins.score(sig) == 7 and "ökar > 25 %" in sig["comment"]


def test_scan_survives_an_api_error():
    class _Broken(_FakeAPI):
        def get_instruments(self):
            raise RuntimeError("401")
    out = sc.scan(_Broken(), today=TODAY)
    assert out["error"] and out["clusters"] == []


# ── Larmbenet ────────────────────────────────────────────────────────────────
def _blob(*rows, error=None):
    return {"generated": "2026-09-10T06:00:00", "error": error,
            "clusters": [dict({"ticker": t, "name": t, "insiders": 3,
                               "role": "VD/CFO", "amount": 1200.0,
                               "gate": "", "trigger": "", "cluster_avg": 100.0,
                               "price_now": 104.0, "stop": 85.0,
                               "vs_cluster": 4.0, "chase": False,
                               "score": s, "status": st}, **extra)
                         for t, s, st, extra in rows]}


def test_insider_alerts_on_new_cluster_and_new_buy_state():
    assert ar._S_INSIDER_BUY == ins.S_BUY
    a, state = ar.insider_alerts(_blob(("A.ST", 7, ins.S_RUN_GATE, {})), None)
    assert a == [] and "A.ST" in state["qualified"]          # baslinje, tyst

    a, state = ar.insider_alerts(
        _blob(("A.ST", 7, ins.S_RUN_GATE, {}),
              ("B.ST", 8, ins.S_WAIT_TRIGGER, {}),
              ("C.ST", 6, ins.S_WATCH, {})), state)
    assert [x["kind"] for x in a] == ["insider_cluster"]
    assert "B.ST" in a[0]["title"] and "1,200 tkr" in a[0]["body"]

    # A når KÖP-läge → eget larm; C under ribban larmar aldrig
    a, state = ar.insider_alerts(
        _blob(("A.ST", 7, ins.S_BUY, {"gate": "Ja", "trigger": "A"}),
              ("B.ST", 8, ins.S_WAIT_TRIGGER, {})), state)
    assert [x["kind"] for x in a] == ["insider_buy"]
    assert "A.ST" in a[0]["title"] and "stoppen" in a[0]["body"]
    # KÖP men > 30 % över klustersnittet = passa → inget köp-larm
    a, _s = ar.insider_alerts(
        _blob(("D.ST", 9, ins.S_BUY, {"chase": True})), state)
    assert [x["kind"] for x in a] == ["insider_cluster"]


def test_insider_alerts_freeze_on_error_or_missing_source():
    _a, state = ar.insider_alerts(_blob(("A.ST", 7, ins.S_RUN_GATE, {})), None)
    _a, kept = ar.insider_alerts(_blob(error="401"), state)
    assert kept == state
    _a, kept = ar.insider_alerts(None, state)
    assert kept == state
    # ribban från inställningarna
    a, _s = ar.insider_alerts(_blob(("E.ST", 5, ins.S_WATCH, {})), state, min_score=5)
    assert len(a) == 1


def test_evaluate_routes_insider_leg():
    regime = {"regime": "GRÖN", "rules": []}
    _a, state = ar.evaluate(regime, {"top": []}, {"positions": []}, [], None, {})
    assert "insider" in state
    alerts, state2 = ar.evaluate(
        regime, {"top": []}, {"positions": []}, [], state,
        {"insider": {"enabled": True, "channels": ["email"], "min_score": 7}},
        insider_data=_blob(("A.ST", 7, ins.S_RUN_GATE, {})))
    kinds = {a["kind"]: a for a in alerts}
    assert kinds["insider_cluster"]["channels"] == ["email"]
    assert "A.ST" in state2["insider"]["qualified"]
