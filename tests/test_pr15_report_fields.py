"""
PR 15 — rapportfälten ur Börsdata som förslag.

Kassa och burn (Sprott), skuld, antal aktier nu och 1/3/5 år tillbaka
(Durrett-arket och DS "Aktier 3 år"), ROIC, FCF-yield och EV/EBIT
(Durrett-arket) och EV/EBITDA-medianen (Royalty C). Allt via
sifferuppdateringen och "Använd" — inget skrivs över.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sheets_refresh as sr   # noqa: E402


def _src(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


class _Reports:
    """Börsdatas rapporter i miljoner, rapportvalutan."""
    def __init__(self, fcf=-40.0, years=(2021, 2022, 2023, 2024, 2025)):
        self.fcf = fcf
        self.years = years
        self.calls = []

    def get_reports(self, ins_id, report_type="year", max_count=20, original=True):
        self.calls.append((ins_id, report_type))
        if report_type == "year":
            return [{"year": y, "numberOfShares": 100.0 + 10 * (y - 2021), "cashAndEquivalents": 50.0}
                    for y in self.years]
        return [{"year": 2025, "cashAndEquivalents": 120.0, "netDebt": -70.0, "freeCashFlow": self.fcf}]

    def get_kpi_history(self, ins_id, kpi_id, report_type="year", price_type="mean"):
        assert kpi_id == 11
        return [{"y": 2016 + i, "v": v} for i, v in enumerate([8.0, 12.0, -3.0, 6.0, 9.0, None])]


# ── report_fields ────────────────────────────────────────────────────────────
def test_report_fields_read_cash_burn_debt_and_the_share_history():
    api = _Reports()
    out = sr.report_fields(api, 7, rfx=0.5)          # rapportvaluta → USD × 0,5
    assert out["cash_musd"] == 60.0                   # 120 × 0,5
    assert out["burn_musd"] == 20.0                   # −40 FCF → 40 burn × 0,5
    assert out["debt_musd"] == 25.0                   # (−70 + 120) × 0,5 — bruttoskuld ≈ nettoskuld + kassa
    assert out["shares_now_m"] == 140.0 and out["shares_1y_ago_m"] == 130.0
    assert out["shares_3y_ago_m"] == 110.0 and "shares_5y_ago_m" not in out     # bara fem år i historiken
    assert out["shares_growth_3y_pct"] == 27.3        # 140/110 − 1
    assert api.calls == [(7, "year"), (7, "r12")]
    # positivt FCF = ingen burn; skulden aldrig negativ
    out = sr.report_fields(_Reports(fcf=30.0), 7)
    assert out["burn_musd"] == 0.0 and out["debt_musd"] == 50.0


def test_report_fields_survive_a_broken_api():
    class _Bad:
        def get_reports(self, *a, **k):
            raise RuntimeError("403")
    assert sr.report_fields(_Bad(), 7) == {}
    assert sr.ev_ebitda_median(_Bad(), 7) is None


def test_ev_ebitda_median_ignores_negative_and_missing_years():
    assert sr.ev_ebitda_median(_Reports(), 7) == 8.5           # median av 8, 12, 6, 9
    assert sr._median([]) is None and sr._median([3.0]) == 3.0


# ── jobbet lägger fälten på raderna ──────────────────────────────────────────
def test_refresh_puts_report_fields_on_the_rows_that_need_them():
    from test_sheets_refresh import _API, _sheets

    class _Api(_API, _Reports):
        def __init__(self):
            _API.__init__(self)
            _Reports.__init__(self)
            self.snaps[105].update({"roic": 0.18, "p_fcf": 8.0, "ev_ebit": 6.5})

    out = sr.refresh(_Api(), _sheets())
    rows = out["rows"]
    d = rows["scoring:d1"]                                          # Durrett-snabbpoäng: rapportfälten, CAD × 0,73
    assert d["cash_musd"] == 87.6 and d["burn_musd"] == 29.2 and d["shares_growth_3y_pct"] == 27.3
    assert rows["producers:y1"]["ev_ebitda_median"] == 8.5        # Royalty C
    assert "ev_ebitda_median" not in rows["producers:r1"]          # inte Rick Rule
    g = rows["confidence:GPR"]                                      # Durrett-arket: KPI-fälten också
    assert g["roic_pct"] == 18.0 and g["fcf_yield_pct"] == 12.5 and g["ev_ebit"] == 6.5
    assert g["shares_now_m"] == 140.0 and g["debt_musd"] == 50.0
    assert "roic_pct" not in d                                      # bara Durrett-arket


# ── Durrett-arket föreslår de nya fälten ─────────────────────────────────────
def test_durrett_proposals_include_the_report_fields():
    from engines.durrett import refresh as dr
    import durrett_cases as dcs
    c = dcs.gold_producer()
    blob = {"generated": "2026-09-22T06:00", "rows": {"confidence:GPR": {
        "asof": "2026-09-22", "currency": "USD", "cash_musd": 60.0, "debt_musd": 25.0,
        "shares_now_m": 140.0, "shares_3y_ago_m": 110.0, "roic_pct": 18.0,
        "fcf_yield_pct": 12.5, "ev_ebit": 6.5}}}
    props = {k: v for k, v, _cur in dr.proposals(blob, c)}
    assert props["cash_musd"].value == 60.0 and props["basic_shares_m"].value == 140.0
    assert props["shares_3y_ago_m"].value == 110.0 and props["roic_pct"].unit == "%"
    assert props["debt_musd"].kind == "ESTIMATE" and "nettoskuld + kassa" in props["debt_musd"].note
    assert props["ev_ebit"].value == 6.5 and props["fcf_yield_pct"].value == 12.5


# ── DS "Aktier 3 år" och de andra arken ──────────────────────────────────────
def test_ds_shares_suggestion_follows_the_fields_own_thresholds():
    import controls as ctl
    assert ctl.ds_shares_suggestion(4.0) == 0
    assert ctl.ds_shares_suggestion(10.0) == 1 and ctl.ds_shares_suggestion(25.0) == 1
    assert ctl.ds_shares_suggestion(27.3) == 2
    assert ctl.ds_shares_suggestion(None) is None
    import inspect
    import controls_ui
    assert "sheet" in inspect.signature(controls_ui.render_ds).parameters
    src = _src("controls_ui.py")
    assert 'render_ds(row, key, runway_years,\n                                 sheet=SHEET_BY_STRATEGY.get(strategy, ""))' in src
    assert '"shares_growth_3y_pct", row_id=key' in src


def test_sprott_and_royalty_sheets_offer_the_new_suggestions():
    sc = _src("scoring.py")
    assert 'refresh_ui.suggest("scoring", row, "cash", "cash_musd"' in sc
    assert 'refresh_ui.suggest("scoring", row, "burn", "burn_musd"' in sc
    assert '_suggest("producers", row, "ev_median", "ev_ebitda_median"' in _src("producers.py")
