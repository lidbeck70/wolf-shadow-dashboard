"""
Börsdatas rapportfält har understreck (number_Of_Shares, free_Cash_Flow …);
panelen läser numberOfShares, freeCashFlow. get_reports och
get_reports_batch lägger till namnet utan understreck — utspädning,
kassa, skuld och FCF-historik i Snabbkollen, Durrett-arket och Contrarian
Alpha läses då ur rapporterna i stället för att bli DATA_GAP.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import borsdata_api as bd  # noqa: E402

ROW = {"year": 2024, "period": 5, "number_Of_Shares": 273.5, "free_Cash_Flow": 4532.0,
       "cash_And_Equivalents": 3100.0, "net_Debt": 5200.0, "total_Assets": 110_000.0, "total_Equity": 60_000.0}


def test_aliases_keep_both_forms_and_leave_camel_case_alone():
    out = bd._report_aliases(ROW)
    assert out["numberOfShares"] == 273.5 and out["freeCashFlow"] == 4532.0
    assert out["cashAndEquivalents"] == 3100.0 and out["netDebt"] == 5200.0
    assert out["totalAssets"] == 110_000.0 and out["totalEquity"] == 60_000.0
    assert out["number_Of_Shares"] == 273.5                          # originalet finns kvar
    camel = {"year": 2024, "numberOfShares": 10.0}
    assert bd._report_aliases(camel) == camel
    assert bd._report_aliases({"number_Of_Shares": 1.0, "numberOfShares": 2.0})["numberOfShares"] == 2.0
    assert bd._report_aliases(None) is None


def test_get_reports_and_batch_return_aliased_rows(monkeypatch):
    api = bd.BorsdataAPI.__new__(bd.BorsdataAPI)
    monkeypatch.setattr(api, "_get", lambda path, params=None: (
        {"reportsYear": [dict(ROW, instrument=7)]}), raising=False)
    rows = api.get_reports(7, "year", max_count=5)
    assert rows[0]["numberOfShares"] == 273.5
    batch = api.get_reports_batch([7])
    assert batch[7][0]["freeCashFlow"] == 4532.0


def test_consumers_now_read_dilution_and_cash_from_reports():
    import sheets_refresh as sr
    from asymmetry import quick_data

    years = [bd._report_aliases({"year": 2020 + i, "number_Of_Shares": 250.0 + 5 * i, "free_Cash_Flow": 1000.0 * i,
                                 "total_Assets": 100.0, "total_Equity": 55.0}) for i in range(6)]
    r12 = bd._report_aliases({"free_Cash_Flow": 4532.0, "cash_And_Equivalents": 3100.0, "net_Debt": 5200.0})

    class _Api:
        is_configured = True

        def resolve_instrument_id(self, q):
            return 7

        def get_instruments(self):
            return [{"insId": 7, "ticker": "BOL", "name": "Boliden", "marketId": 1}]

        def get_global_instruments_list(self):
            return []

        def get_fundamentals_snapshot_fast(self, ids, scope="nordic"):
            return {7: {}}

        def get_reports(self, iid, kind, max_count=10):
            return [r12] if kind == "r12" else years

        def get_kpi_history(self, *a):
            return []

    rf = sr.report_fields(_Api(), 7)
    assert rf["shares_now_m"] == 275.0 and rf["shares_3y_ago_m"] == 260.0 and rf["cash_musd"] == 3100.0
    d = quick_data.fetch("BOL", api=_Api(), price_getter=lambda s: None, info_getter=lambda s: {})
    assert d["shares_growth_3y_pct"] == round((275 / 260 - 1) * 100, 1)
    assert d["fcf"] == 4532.0 and d["cash"] == 3100.0 and d["equity_ratio_pct"] == 55.0
