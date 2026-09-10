"""
Håvarna headless — screens_scan.py kör guidens fem Börsdata-screeners
med exakt kriterietexten (reference.SCREENERS). Testerna kontrollerar
att varje kriterium fäller precis det det ska, att geografin bara
tillämpas där håven har den, och att larmbenet larmar per håv och bara
på nya bolag.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import screens_scan as sc
import screens_ui as sui
import alert_rules as ar


def test_criteria_texts_match_the_guide():
    """Håvarnas kriterier i jobbet är samma text som i RULES → SNABBREFERENS."""
    from reference import SCREENERS
    ref = {s.key: s.filters for s in SCREENERS if s.key}
    for s in sc.SCREENS:
        assert s.criteria == ref[s.key], s.key


# ── Kriterierna ──────────────────────────────────────────────────────────────
def _rule_ok():
    return {"nd_ebitda": 0.2, "equity_ratio": 62.0, "ev_ebitda": 4.1, "pb": 0.9,
            "fcf": 120.0}


def test_rule_survivors():
    ok, _ = sc.rule_check(_rule_ok(), {"branch_id": 17})
    assert ok == []
    assert "inte råvarubransch" in sc.rule_check(_rule_ok(), {"branch_id": 55})[0]
    # olja får 1,0 i skuld/EBITDA, gruvor 0,5
    assert sc.rule_check(dict(_rule_ok(), nd_ebitda=0.8), {"branch_id": 4})[0] == []
    assert any("skuld/EBITDA" in f for f in
               sc.rule_check(dict(_rule_ok(), nd_ebitda=0.8), {"branch_id": 17})[0])
    assert any("soliditet" in f for f in
               sc.rule_check(dict(_rule_ok(), equity_ratio=48.0), {"branch_id": 17})[0])
    assert any("P/B" in f for f in sc.rule_check(dict(_rule_ok(), pb=1.6), {"branch_id": 17})[0])
    assert any("FCF" in f for f in sc.rule_check(dict(_rule_ok(), fcf=-3.0), {"branch_id": 17})[0])
    assert any("EV/EBITDA" in f for f in
               sc.rule_check(dict(_rule_ok(), ev_ebitda=7.0), {"branch_id": 17})[0])
    # negativ EBITDA är inte "billigt"
    assert any("EBITDA ≤ 0" in f for f in
               sc.rule_check(dict(_rule_ok(), ev_ebitda=-2.0), {"branch_id": 17})[0])
    assert any("saknas" in f for f in
               sc.rule_check(dict(_rule_ok(), pb=None), {"branch_id": 17})[0])


def test_sprott_optionality():
    m = {"mcap_musd": 120.0, "net_debt": -15.0, "pb": 0.7}
    assert sc.sprott_check(m, {"branch_id": 16})[0] == []
    assert sc.sprott_check(dict(m, mcap_musd=250.0), {"branch_id": 16})[0]
    assert sc.sprott_check(dict(m, net_debt=5.0), {"branch_id": 16})[0]
    assert sc.sprott_check(dict(m, pb=1.1), {"branch_id": 16})[0]
    assert sc.sprott_check(m, {"branch_id": 21})[0]


def test_durrett_leverage():
    m = {"mcap_musd": 220.0, "ps": 1.4, "gross_margin": 35.0, "nd_ebitda": 0.9,
         "revenue_growth": 0.12}
    assert sc.durrett_check(m, {"branch_id": 18})[0] == []
    assert sc.durrett_check(m, {"branch_id": 17})[0]                 # inte guld/silver
    assert sc.durrett_check(dict(m, mcap_musd=40.0), {"branch_id": 18})[0]
    assert sc.durrett_check(dict(m, mcap_musd=600.0), {"branch_id": 18})[0]
    assert sc.durrett_check(dict(m, revenue_growth=-0.05), {"branch_id": 18})[0]
    assert any("historik" in f for f in
               sc.durrett_check(dict(m, revenue_growth=None), {"branch_id": 18})[0])


def test_tiggre_sweet_spot_notes_debt_instead_of_failing():
    m = {"mcap_musd": 300.0, "net_debt": -40.0, "revenue_musd": 0.0}
    fails, notes = sc.tiggre_check(m, {"branch_id": 16})
    assert fails == [] and notes == []
    fails, notes = sc.tiggre_check(dict(m, net_debt=80.0), {"branch_id": 16})
    assert fails == [] and any("byggkrediten" in n for n in notes)
    assert sc.tiggre_check(dict(m, revenue_musd=40.0), {"branch_id": 16})[0]
    assert sc.tiggre_check(dict(m, mcap_musd=1500.0), {"branch_id": 16})[0]


def test_royalty_margins():
    m = {"gross_margin": 82.0, "ebit_margin": 55.0, "nd_ebitda": 0.4}
    assert sc.royalty_check(m, {})[0] == []
    assert sc.royalty_check(dict(m, gross_margin=65.0), {})[0]
    assert sc.royalty_check(dict(m, ebit_margin=30.0), {})[0]
    assert sc.royalty_check(dict(m, nd_ebitda=2.0), {})[0]


def test_metrics_convert_market_cap_to_musd():
    inst = {"insId": 7, "stockPriceCurrency": "SEK"}
    kpi = {"mcap": {7: 2000.0}, "revenue": {7: 100.0}, "pb": {7: 0.8}}
    m = sc.metrics_for(inst, kpi)
    assert m["mcap_musd"] == 190.0 and m["revenue_musd"] == 9.5 and m["pb"] == 0.8
    assert m["nd_ebitda"] is None


def test_country_ids_resolve_by_name():
    c = sc.country_ids([{"id": 1, "name": "Sverige"}, {"id": 9, "name": "Kanada"},
                        {"id": 12, "name": "Australia"}, {"id": 30, "name": "USA"}])
    assert c[sc.CA] == {9} and c[sc.AU] == {12} and c[sc.US] == {30}


# ── Hela körningen mot ett fejkat API ───────────────────────────────────────
class _API:
    """Ett nordiskt gruvbolag som klarar Rule, ett kanadensiskt som klarar
    Sprott + Tiggre, ett kanadensiskt guldbolag för Durrett."""
    def __init__(self, global_ok=True):
        self.global_ok = global_ok

    def get_countries(self):
        return [{"id": 1, "name": "Sverige"}, {"id": 9, "name": "Kanada"}]

    def get_instruments(self):
        return [{"insId": 1, "ticker": "BOL", "name": "Boliden", "marketId": 1,
                 "branchId": 17, "countryId": 1, "stockPriceCurrency": "SEK"},
                {"insId": 2, "ticker": "OMX", "name": "Index", "marketId": 7}]

    def get_kpi_screener(self, kid, g, c):
        v = {42: 0.3, 39: 60.0, 11: 4.0, 4: 1.1, 63: 500.0, 50: 90000.0,
             3: 1.0, 28: 30.0, 29: 20.0, 60: 1000.0, 53: 80000.0}
        return [{"i": 1, "n": v[kid]}]

    def get_global_instruments_list(self):
        if not self.global_ok:
            return []
        return [{"insId": 101, "ticker": "JR", "name": "Junior Corp", "branchId": 16,
                 "countryId": 9, "stockPriceCurrency": "CAD"},
                {"insId": 102, "ticker": "AU", "name": "Gold Prod", "branchId": 18,
                 "countryId": 9, "stockPriceCurrency": "CAD"}]

    def get_kpi_screener_global(self, kid, g, c):
        rows = {101: {42: None, 39: 90.0, 11: -3.0, 4: 0.6, 63: -5.0, 50: 150.0,
                      3: None, 28: None, 29: None, 60: -20.0, 53: 0.0},
                102: {42: 0.8, 39: 55.0, 11: 5.0, 4: 1.3, 63: 30.0, 50: 300.0,
                      3: 1.5, 28: 40.0, 29: 25.0, 60: 10.0, 53: 200.0}}
        return [{"i": i, "n": r[kid]} for i, r in rows.items() if r[kid] is not None]

    def get_kpi_history(self, ins_id, kpi_id, rt, pt):
        return [{"y": 2024, "v": 150.0}, {"y": 2025, "v": 200.0}]


def test_scan_runs_all_five_screens():
    out = sc.scan(_API())
    s = out["screens"]
    assert out["global_available"] is True
    assert [r["ticker"] for r in s["rule"]["rows"]] == ["BOL.ST"]
    assert s["rule"]["rows"][0]["mcap_musd"] == 8550.0
    assert [r["ticker"] for r in s["sprott"]["rows"]] == ["JR"]
    assert [r["ticker"] for r in s["tiggre"]["rows"]] == ["JR"]
    assert [r["ticker"] for r in s["durrett"]["rows"]] == ["AU"]
    assert s["durrett"]["rows"][0]["m"]["revenue_growth"] == round(200 / 150 - 1, 3)
    assert s["royalty"]["rows"] == []                  # 40 % brutto < 70
    assert all(v["error"] is None for v in s.values())


def test_scan_without_global_licence_marks_geographic_screens():
    out = sc.scan(_API(global_ok=False))
    s = out["screens"]
    assert out["global_available"] is False
    assert [r["ticker"] for r in s["rule"]["rows"]] == ["BOL.ST"]
    for k in ("sprott", "tiggre", "royalty"):
        assert "Pro+ global" in s[k]["error"]
    assert s["durrett"]["error"] is None and s["durrett"]["rows"] == []


# ── Arkens förifyllning ──────────────────────────────────────────────────────
def test_row_to_fields_only_prefills_what_the_sheet_uses():
    row = {"ticker": "bol.st", "name": "Boliden", "mcap_musd": 210.0,
           "m": {"ev_ebitda": 4.0, "nd_ebitda": 0.3, "pb": 1.1}}
    assert sui.row_to_fields(row, "rule") == {"ticker": "BOL.ST", "name": "Boliden",
                                              "ev_ebitda": 4.0, "nd_ebitda": 0.3}
    assert sui.row_to_fields(row, "durrett")["mcap"] == 210.0
    assert "mcap" not in sui.row_to_fields(row, "sprott")
    assert sui.row_to_fields(row, "royalty")["ev_now"] == 4.0


# ── Larmbenet ────────────────────────────────────────────────────────────────
def _blob(**rows_by_key):
    return {"generated": "2026-09-10T06:00:00", "screens": {
        k: {"label": k.title(), "sheet": k, "rows": [
            {"ticker": t, "name": t, "mcap_musd": 100.0, "universe": "nordic"}
            for t in v] if isinstance(v, list) else [],
            "error": v if isinstance(v, str) else None}
        for k, v in rows_by_key.items()}}


def test_screen_alerts_per_screen_on_new_tickers_only():
    _a, state = ar.screen_alerts(_blob(rule=["A.ST"], sprott=["X"]), None)
    assert _a == [] and set(state) == {"rule", "sprott"}
    alerts, state = ar.screen_alerts(_blob(rule=["A.ST", "B.ST"], sprott=["X"],
                                           durrett=["G"]), state)
    # rule: B ny → larm; sprott oförändrad; durrett ny håv → baslinje, tyst
    assert [a["kind"] for a in alerts] == ["screen_rule"]
    assert "B.ST" in alerts[0]["body"] and "A.ST" not in alerts[0]["body"]
    assert "durrett" in state
    alerts, _s = ar.screen_alerts(_blob(rule=["A.ST", "B.ST"], sprott=["X"],
                                        durrett=["G", "H"]), state)
    assert [a["kind"] for a in alerts] == ["screen_durrett"]


def test_screen_alerts_freeze_per_screen_on_error():
    _a, state = ar.screen_alerts(_blob(rule=["A.ST"], sprott=["X"]), None)
    _a, kept = ar.screen_alerts(_blob(rule=["A.ST"], sprott="Pro+ global saknas"), state)
    assert kept["sprott"] == state["sprott"]
    _a, kept2 = ar.screen_alerts(None, kept)
    assert kept2 == kept


def test_evaluate_routes_screens_leg():
    regime = {"regime": "GRÖN", "rules": []}
    _a, state = ar.evaluate(regime, {"top": []}, {"positions": []}, [], None, {},
                            screens_data=_blob(rule=[]))
    alerts, state2 = ar.evaluate(
        regime, {"top": []}, {"positions": []}, [], state,
        {"screens": {"enabled": True, "channels": ["email"]}},
        screens_data=_blob(rule=["A.ST"]))
    kinds = {a["kind"]: a for a in alerts}
    assert kinds["screen_rule"]["channels"] == ["email"]
    assert "A.ST" in state2["screens"]["rule"]
