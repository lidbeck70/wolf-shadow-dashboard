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
        "confidence": {"companies": {"GPR": _durrett_company()}, "commodity_overrides": {}},
    }


def _durrett_company() -> dict:
    """Durrett-arkets rad (data/confidence.json): 300 koz × (3 000 − 1 450) = 465 → FCF 349 MUSD;
    börsvärde 5 000 = 14,3× → färskt 2 000 MUSD = 5,7× korsar 10×."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import durrett_cases as dcs
    c = dcs.gold_producer()
    c.ins_id = 105
    c.set("market_cap_musd", type(c.get("aisc"))(value=5000.0, kind="ACTUAL", source="Börsdata"))
    return c.as_dict()


def test_collect_rows_finds_every_bucket():
    refs = sr.collect_rows(_sheets())
    assert {(r["sheet"], r["bucket"]) for r in refs} == {
        ("insider", "signals"), ("tiggre", "candidates"), ("tiggre", "positions"),
        ("producers", "producers"), ("producers", "royalty"), ("scoring", "durrett"),
        ("confidence", "companies")}
    assert sr.ref_key("insider", {"id": "i1"}) == "insider:i1"
    conf = [r for r in refs if r["sheet"] == "confidence"][0]
    assert conf["key"] == "confidence:GPR" and conf["ins_id"] == 105 and conf["ticker"] == "GPR"


def test_ticker_forms_strip_suffix_and_dash():
    assert sr.ticker_forms("EKTA-B.ST")[:2] == ["EKTA B", "EKTA-B"]
    assert sr.ticker_forms("bol.st")[0] == "BOL"


class _API:
    def __init__(self):
        self.prices = {1: 84.0, 2: 135.0, 40: 300.0, 102: 21.0, 103: 50.0, 104: 50.0, 105: 8.0}
        self.snaps = {40: {"ev_ebitda": 5.0, "net_debt_ebitda": 1.4, "market_cap": 90000.0},
                      102: {"market_cap": 400.0},
                      103: {"ev_ebitda": 12.0},
                      104: {"market_cap": 150.0},
                      105: {"market_cap": 2000.0, "ev": 1900.0, "ev_ebitda": 4.1, "net_debt_ebitda": -0.3,
                            "revenue_m": 600.0, "fcf_m": 110.0, "pe": 9.0, "rs_rank": 71.0, "ebitda_margin": 0.48}}

    def get_instruments(self):
        return [{"insId": 1, "ticker": "EKTA B", "name": "Elekta B", "stockPriceCurrency": "SEK"},
                {"insId": 2, "ticker": "HEXA B", "name": "Hexagon B", "stockPriceCurrency": "SEK"},
                {"insId": 40, "ticker": "BOL", "name": "Boliden", "stockPriceCurrency": "SEK"}]

    def get_global_instruments_list(self):
        return [{"insId": i, "ticker": t, "name": n, "stockPriceCurrency": "CAD"}
                for i, t, n in ((101, "JR", "Jaguar"), (102, "AU", "AngloGold"), (103, "FNV", "Franco-Nevada"),
                                (104, "MJS", "Majestic"))] + \
            [{"insId": 105, "ticker": "GPR", "name": "Great Panther", "stockPriceCurrency": "USD",
              "reportCurrency": "USD"}]

    def resolve_instrument_id(self, form):
        return {"EKTA B": 1, "HEXA B": 2, "BOL": 40}.get(form)

    def get_fundamentals_snapshot_fast(self, ids, scope="nordic"):
        # Globala id:n (>100) finns bara i den globala screenern — samma
        # gräns som Börsdatas egen: fel universum ger tomt.
        if scope == "global":
            return {i: dict(self.snaps.get(i, {}), ins_id=i) for i in ids if i > 100}
        return {i: dict(self.snaps.get(i, {}), ins_id=i) for i in ids if i <= 100}

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
    # Durrett-arket (confidence.json): fler tal ur snapshoten + motorns köpregel
    g = rows["confidence:GPR"]
    assert g["mcap_musd"] == 2000.0 and g["ev_musd"] == 1900.0 and g["revenue_musd"] == 600.0
    assert g["fcf_musd"] == 110.0 and g["pe"] == 9.0 and g["rs_rank"] == 71.0 and g["fx_to_usd"] == 1.0
    assert g["price"] == 8.0 and g["currency"] == "USD"
    assert "producers:r1" in rows and "fx_to_usd" not in rows["producers:r1"]   # bara Durrett-arket
    ev = kinds["durrett_engine_buy_rule:confidence:GPR"]                          # 14,3× → 5,7× korsar 10×
    assert "köpregeln uppfylld" in ev["body"] and ev["ticker"] == "GPR" and ev["sheet"] == "confidence"


def test_no_events_when_nothing_crosses():
    api = _API()
    api.prices[1] = 100.0
    api.prices[2] = 100.0
    api.prices[102] = 12.0
    api.snaps[40]["net_debt_ebitda"] = 0.8
    api.snaps[102]["market_cap"] = 100.0
    api.snaps[103]["ev_ebitda"] = 20.0
    api.snaps[104]["market_cap"] = 400.0       # 292/20 = 14.6× → fortfarande över 10
    api.snaps[105]["market_cap"] = 5000.0      # Durrett-arket: 14,3× → 14,3×, ingen korsning
    out = sr.refresh(api, _sheets())
    assert out["events"] == []


def test_refresh_survives_api_error():
    class _Broken(_API):
        def get_instruments(self):
            raise RuntimeError("401")
    out = sr.refresh(_Broken(), _sheets())
    assert out["error"] and out["events"] == []


# ── Batch 2: nya nattfält utan extra Börsdata-anrop ──────────────────────────
class _Counting(_API):
    """Räknar anropen mot Börsdata-klienten — de nya fälten får inte kosta fler."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if callable(attr) and not name.startswith("_") and name not in ("calls",):
            object.__getattribute__(self, "calls").append(name)
        return attr


def test_snapshot_musd_converts_like_the_confidence_block():
    """EV med kursvalutans fx, resten med rapportvalutans — None när Börsdata saknar talet."""
    snap = {"ev": 1000.0, "net_debt_m": -200.0, "revenue_m": 500.0, "fcf_m": 100.0}
    out = sr._snapshot_musd(snap, 0.095, 0.73)
    assert out == {"ev_musd": 95.0, "net_debt_musd": -146.0, "revenue_musd": 365.0, "fcf_musd": 73.0,
                   "ocf_musd": None}


def test_producer_rows_get_the_musd_block_and_dividend_yield():
    api = _API()
    api.snaps[40].update(ev=100000.0, fcf_m=20000.0, net_debt_m=5000.0, revenue_m=80000.0, ocf_m=30000.0,
                         dividend_yield=0.034)                   # BOL, SEK; snapshoten ger 3,4 % som 0,034
    rows = sr.refresh(api, _sheets())["rows"]
    r1 = rows["producers:r1"]
    assert r1["fcf12_musd"] == 1900.0 and r1["net_debt_musd"] == 475.0 and r1["revenue_musd"] == 7600.0
    assert "fcf_musd" not in r1                                  # bara referens: Lukacs enhetskontroll läser fcf_musd
    assert r1["ocf_musd"] == 2850.0 and r1["ev_musd"] == 9500.0
    assert r1["dividend_yield_pct"] == 3.4
    y1 = rows["producers:y1"]                                    # Royalty C: samma block, tomt där talen saknas
    assert y1["fcf12_musd"] is None and y1["dividend_yield_pct"] is None and y1["ev_ebitda_median"] is None
    assert "fx_to_usd" not in r1                                 # fx_to_usd stannar på Durrett-arket
    assert "fcf_musd" not in rows["insider:i1"] and "dividend_yield_pct" not in rows["scoring:d1"]


def test_every_borsdata_row_says_which_company_was_matched():
    rows = sr.refresh(_API(), _sheets())["rows"]
    assert rows["insider:i1"]["bd_name"] == "Elekta B" and rows["insider:i1"]["bd_ticker"] == "EKTA B"
    assert rows["producers:r1"]["bd_name"] == "Boliden" and rows["producers:r1"]["report_currency"] == "SEK"
    assert rows["confidence:GPR"]["bd_name"] == "Great Panther" and rows["confidence:GPR"]["report_currency"] == "USD"
    assert rows["tiggre:p1"]["bd_ticker"] == "AU"                 # global lista
    assert rows["tiggre:c1"]["bd_name"] == "Jaguar" and rows["tiggre:c1"]["price"] is None   # matchad utan kurs


def test_f_score_on_insider_rows_only():
    api = _API()
    api.snaps[1] = {"f_score": 7.0}
    api.snaps[40]["f_score"] = 5.0
    rows = sr.refresh(api, _sheets())["rows"]
    assert rows["insider:i1"]["f_score"] == 7.0 and rows["insider:i2"]["f_score"] is None
    assert "f_score" not in rows["producers:r1"]


def test_the_new_fields_cost_no_extra_calls(monkeypatch):
    """Samma Börsdata-anrop som före batch 2: en instrumentlista, en global, två snapshot-omgångar, en kurs- och
    rapporthämtning per bolag och EV/EBITDA-historiken för Royalty C."""
    api = _Counting()
    out = sr.refresh(api, _sheets())
    assert out["error"] is None
    from collections import Counter
    c = Counter(api.calls)
    n_ids = len({v["ins_id"] for v in out["rows"].values() if v.get("ins_id") is not None})
    assert c["get_instruments"] == 1 and c["get_global_instruments_list"] == 1
    assert c["get_fundamentals_snapshot_fast"] == 2 and c["get_stockprices"] == n_ids
    assert set(c) <= {"get_instruments", "get_global_instruments_list", "get_fundamentals_snapshot_fast",
                      "get_stockprices", "resolve_instrument_id", "get_reports", "get_kpi_history"}


def test_report_fields_say_which_year_the_share_count_is_from():
    class _Rep:
        def get_reports(self, ins_id, kind, max_count=7):
            if kind == "r12":
                return []
            return [{"year": y, "numberOfShares": n} for y, n in ((2020, 80.0), (2022, 90.0), (2025, 100.0))]
    out = sr.report_fields(_Rep(), 1)
    assert out["shares_now_m"] == 100.0 and out["shares_year"] == 2025 and out["shares_3y_ago_m"] == 90.0
    assert out["shares_growth_5y_pct"] == 25.0
    assert "shares_year" not in sr.report_fields(type("_None", (), {"get_reports": lambda *a, **k: []})(), 1)


def _two_country_api(tmp_path):
    """Riktiga BorsdataAPI.resolve_instrument_id mot en instrumentlista där GENO finns i två länder (Genovis i
    Sverige, General Oceans i Norge — Norge läses sist). Inget nätverk: _get är fejkad."""
    from borsdata_api import BorsdataAPI
    api = BorsdataAPI(api_key="test", cache_dir=str(tmp_path))
    insts = [{"insId": 7, "ticker": "GENO", "name": "Genovis AB", "countryId": 1, "stockPriceCurrency": "SEK",
              "reportCurrency": "SEK"},
             {"insId": 8, "ticker": "GENO", "name": "General Oceans", "countryId": 2, "stockPriceCurrency": "NOK",
              "reportCurrency": "NOK"},
             {"insId": 9, "ticker": "EKTA B", "name": "Elekta B", "countryId": 1, "stockPriceCurrency": "SEK"}]
    api._get = lambda path, params=None, use_cache=True: {"instruments": insts} if path == "/instruments" else {}
    return api


def test_resolve_picks_the_company_in_the_suffix_country(tmp_path):
    api = _two_country_api(tmp_path)
    assert api.resolve_instrument_id("GENO") == 8                # den nakna formen: den som lästes sist
    assert sr.resolve(api, "GENO.ST") == 7                       # förut 8 — fel lands bolag
    assert sr.resolve(api, "GENO.OL") == 8
    assert sr.resolve(api, "geno.st") == 7 and sr.resolve(api, "EKTA-B.ST") == 9
    assert sr.resolve(api, "GENO") == 8                          # utan suffix som förut
    assert sr.resolve(api, "GENO.ST", ins_id=8) == 8             # sparat ins_id vinner alltid


def test_refresh_matches_the_suffix_country_and_shows_the_name(tmp_path):
    api = _two_country_api(tmp_path)
    api.get_global_instruments_list = lambda: []
    api.get_fundamentals_snapshot_fast = lambda ids, scope="nordic": {i: {"ins_id": i} for i in ids}
    api.get_stockprices = lambda ins_id, max_count=5: [{"d": "2026-11-06", "c": 20.0}]
    api.get_reports = lambda ins_id, kind, max_count=7: []
    sheets = {"insider": {"signals": [{"id": "s1", "ticker": "GENO.ST", "name": "Genovis"},
                                      {"id": "s2", "ticker": "GENO.OL", "name": "Genovis"}]}}
    rows = sr.refresh(api, sheets)["rows"]
    assert rows["insider:s1"]["ins_id"] == 7 and rows["insider:s1"]["bd_name"] == "Genovis AB"
    assert rows["insider:s2"]["ins_id"] == 8 and rows["insider:s2"]["bd_name"] == "General Oceans"
    blob = {"rows": rows}
    assert rui.matched_note(blob, "insider", sheets["insider"]["signals"][0]) == ""
    assert rui.matched_note(blob, "insider", sheets["insider"]["signals"][1]) == " (matchat: General Oceans)"


def test_fetch_row_carries_the_same_new_fields(monkeypatch):
    class _One(_API):
        def get_reports(self, ins_id, kind, max_count=7):
            return [] if kind == "r12" else [{"year": 2025, "numberOfShares": 50.0}]
    api = _One()
    api.snaps[40].update(fcf_m=1000.0, ev=2000.0)
    row = sr.fetch_row(api, "BOL.ST")
    assert row["ins_id"] == 40 and row["bd_name"] == "Boliden" and row["bd_ticker"] == "BOL"
    assert row["fcf_musd"] == 95.0 and row["ev_musd"] == 190.0 and row["shares_year"] == 2025


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
