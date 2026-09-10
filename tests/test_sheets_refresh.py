"""
Arkens sifferuppdatering — sheets_refresh.py hämtar färska tal som FÖRSLAG
och räknar övergångar med arkens egna funktioner. Testerna låser att
händelserna följer arkens regler (stopp × 0,85, +30 %, +100 %, 0,8× NAV,
1,0× skuld, 10× vinst) och att larmbenet bara larmar på nya händelser.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sheets_refresh as sr
import refresh_ui as rui
import alert_rules as ar


def _sheets():
    return {
        "insider": {"signals": [
            {"id": "i1", "ticker": "EKTA-B.ST", "cluster_avg": 100.0, "price_now": 100.0,
             "insiders": 3, "role": "VD/CFO", "amount": 1500.0},
            {"id": "i2", "ticker": "HEXA-B.ST", "cluster_avg": 100.0, "price_now": 100.0}]},
        "tiggre": {"candidates": [{"id": "c1", "ticker": "JR", "ins_id": 101, "nav": 500.0}],
                   "positions": [{"id": "p1", "ticker": "AU", "ins_id": 102, "entry": 10.0,
                                  "current": 12.0, "nav": 300.0, "half_sold": False}]},
        "producers": {"producers": [{"id": "r1", "ticker": "BOL.ST", "nd_ebitda": 0.7}],
                      "royalty": [{"id": "y1", "ticker": "FNV", "ins_id": 103,
                                   "ev_now": 20.0, "ev_median": 15.0,
                                   "pnav_now": 1.0, "pnav_bottom": 1.0,
                                   "geo_now": 1.1, "geo_3y": 1.0}]},
        "scoring": {"sprott": [], "durrett": [{"id": "d1", "ticker": "MJS", "ins_id": 104,
                                              "mcap": 300.0, "profit": 20.0}]},
    }


def test_collect_rows_finds_every_bucket():
    refs = sr.collect_rows(_sheets())
    assert {(r["sheet"], r["bucket"]) for r in refs} == {
        ("insider", "signals"), ("tiggre", "candidates"), ("tiggre", "positions"),
        ("producers", "producers"), ("producers", "royalty"), ("scoring", "durrett")}
    assert sr.ref_key("insider", {"id": "i1"}) == "insider:i1"


def test_ticker_forms_strip_suffix_and_dash():
    assert sr.ticker_forms("EKTA-B.ST")[:2] == ["EKTA B", "EKTA-B"]
    assert sr.ticker_forms("bol.st")[0] == "BOL"


class _API:
    def __init__(self):
        self.prices = {1: 84.0, 2: 135.0, 40: 300.0, 102: 21.0, 103: 50.0, 104: 50.0}
        self.snaps = {40: {"ev_ebitda": 5.0, "net_debt_ebitda": 1.4, "market_cap": 90000.0},
                      102: {"market_cap": 400.0},
                      103: {"ev_ebitda": 12.0},
                      104: {"market_cap": 150.0}}

    def get_instruments(self):
        return [{"insId": 1, "ticker": "EKTA B", "stockPriceCurrency": "SEK"},
                {"insId": 2, "ticker": "HEXA B", "stockPriceCurrency": "SEK"},
                {"insId": 40, "ticker": "BOL", "stockPriceCurrency": "SEK"}]

    def get_global_instruments_list(self):
        return [{"insId": i, "ticker": t, "stockPriceCurrency": "CAD"}
                for i, t in ((101, "JR"), (102, "AU"), (103, "FNV"), (104, "MJS"))]

    def resolve_instrument_id(self, form):
        return {"EKTA B": 1, "HEXA B": 2, "BOL": 40}.get(form)

    def get_fundamentals_snapshot_fast(self, ids):
        return {i: dict(self.snaps.get(i, {}), ins_id=i) for i in ids}

    def get_stockprices(self, ins_id, max_count=5):
        p = self.prices.get(ins_id)
        return [{"d": "2026-09-10", "c": p}] if p is not None else []


def test_refresh_builds_suggestions_and_sheet_rule_events():
    out = sr.refresh(_API(), _sheets())
    assert out["error"] is None
    rows = out["rows"]
    assert rows["insider:i1"]["price"] == 84.0 and rows["insider:i1"]["currency"] == "SEK"
    assert rows["producers:r1"]["ev_ebitda"] == 5.0 and rows["producers:r1"]["nd_ebitda"] == 1.4
    assert rows["producers:r1"]["mcap_musd"] == 8550.0
    assert rows["tiggre:p1"]["mcap_musd"] == 292.0                 # 400 CAD × 0.73
    assert rows["tiggre:c1"]["price"] is None                      # ingen kurs → tomt

    kinds = {e["key"]: e for e in out["events"]}
    assert "insider_stop:insider:i1" in kinds                      # 84 ≤ 100 × 0,85
    assert "insider_chase:insider:i2" in kinds                     # 135 > +30 %
    assert "tiggre_free_ride:tiggre:p1" in kinds                   # 21 / 10 = +110 %
    assert "tiggre_nav_target:tiggre:p1" in kinds                  # 292 / 300 ≥ 0,8
    assert "royalty_signal:producers:y1" in kinds                  # 12 < median 15 → byter
    assert "rule_deleveraging:producers:r1" in kinds               # 0,7 → 1,4 korsar 1,0
    assert "durrett_buy_rule:scoring:d1" in kinds                  # 300/20 = 15× → 109/20 = 5.5×
    assert "sälj halva" in kinds["tiggre_free_ride:tiggre:p1"]["title"]


def test_no_events_when_nothing_crosses():
    api = _API()
    api.prices[1] = 100.0
    api.prices[2] = 100.0
    api.prices[102] = 12.0
    api.snaps[40]["net_debt_ebitda"] = 0.8
    api.snaps[102]["market_cap"] = 100.0
    api.snaps[103]["ev_ebitda"] = 20.0
    api.snaps[104]["market_cap"] = 400.0       # 292/20 = 14.6× → fortfarande över 10
    out = sr.refresh(api, _sheets())
    assert out["events"] == []


def test_refresh_survives_api_error():
    class _Broken(_API):
        def get_instruments(self):
            raise RuntimeError("401")
    out = sr.refresh(_Broken(), _sheets())
    assert out["error"] and out["events"] == []


def test_suggestion_lookup_from_blob():
    blob = {"rows": {"insider:i1": {"price": 84.0, "asof": "2026-09-10", "currency": "SEK"}}}
    assert rui.suggestion(blob, "insider", {"id": "i1"}, "price") == (84.0, "2026-09-10", "SEK")
    assert rui.suggestion(blob, "insider", {"id": "i1"}, "ev_ebitda") is None
    assert rui.suggestion({}, "insider", {"id": "i1"}, "price") is None


# ── Larmbenet ────────────────────────────────────────────────────────────────
def _blob(*keys, error=None):
    return {"generated": "2026-09-10T06:00:00", "error": error,
            "events": [{"key": k, "kind": k.split(":")[0], "ticker": "X",
                        "title": f"T {k}", "body": "b"} for k in keys]}


def test_sheet_alerts_only_on_new_events():
    _a, state = ar.sheet_alerts(_blob("insider_stop:insider:i1"), None)
    assert _a == [] and "insider_stop:insider:i1" in state["events"]
    alerts, state = ar.sheet_alerts(
        _blob("insider_stop:insider:i1", "tiggre_free_ride:tiggre:p1"), state)
    assert [a["kind"] for a in alerts] == ["sheet_tiggre_free_ride"]
    # händelsen försvinner och kommer igen → larmar igen
    _a, state = ar.sheet_alerts(_blob(), state)
    alerts, _s = ar.sheet_alerts(_blob("insider_stop:insider:i1"), state)
    assert [a["kind"] for a in alerts] == ["sheet_insider_stop"]


def test_sheet_alerts_freeze_on_error():
    _a, state = ar.sheet_alerts(_blob("insider_stop:insider:i1"), None)
    _a, kept = ar.sheet_alerts(_blob(error="401"), state)
    assert kept == state
    _a, kept2 = ar.sheet_alerts(None, kept)
    assert kept2 == kept


def test_evaluate_routes_sheets_leg():
    regime = {"regime": "GRÖN", "rules": []}
    _a, state = ar.evaluate(regime, {"top": []}, {"positions": []}, [], None, {},
                            sheets_data=_blob())
    alerts, state2 = ar.evaluate(
        regime, {"top": []}, {"positions": []}, [], state,
        {"sheets": {"enabled": True, "channels": ["email"]}},
        sheets_data=_blob("insider_stop:insider:i1"))
    kinds = {a["kind"]: a for a in alerts}
    assert kinds["sheet_insider_stop"]["channels"] == ["email"]
    assert "insider_stop:insider:i1" in state2["sheets"]["events"]
