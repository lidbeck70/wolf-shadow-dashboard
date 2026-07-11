"""
Tests for the Börsdata data-layer fix: per-KPI history fallback when the batch
screener returns 400 (endpoint not in licence) or never fetches a KPI at all.

Regression cover for the "BS gate can only flag BS_DATA_SAKNAS" bug: the batch
endpoint /instruments/kpis/{id}/last/latest 400s for several KPIs and never
fetches equity / Altman inputs, so fund_snap is None for equity, ebitda_margin
and the Altman components — the gate could not evaluate. The fix back-fills those
critical fields from 24h-cached KPI history, survivor-only and guarded to KPIs
that actually failed at the batch level (never re-fetching working KPIs).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import borsdata_api as bd
from borsdata_api import KPI
import contrarian_alpha.cache as cache
from contrarian_alpha.engine import (
    _augment_fund_snap_from_history,
    _build_fundamentals_dict,
    _critical_kpi_fallback,
)
from contrarian_alpha.strength import calculate_strength_score


class _StubAPI:
    """Records every history call and returns canned oldest-first annual data."""
    is_configured = True

    def __init__(self):
        self.calls = []
        self.hist = {
            KPI["total_equity_m"]: [{"y": 2023, "v": 1800.0}, {"y": 2024, "v": 2000.0}],
            KPI["ebitda_margin"]:  [{"y": 2023, "v": 30.0}, {"y": 2024, "v": 38.0}],   # raw %
            KPI["fcf_m"]:          [{"y": 2023, "v": 300.0}, {"y": 2024, "v": 450.0}],
            KPI["total_assets_m"]: [{"y": 2023, "v": 4500.0}, {"y": 2024, "v": 5000.0}],
            KPI["ebit_m"]:         [{"y": 2023, "v": 600.0}, {"y": 2024, "v": 700.0}],
            KPI["revenue_m"]:      [{"y": 2023, "v": 3000.0}, {"y": 2024, "v": 3500.0}],
            KPI["market_cap"]:     [{"y": 2023, "v": 4000.0}, {"y": 2024, "v": 4200.0}],
            KPI["ev_ebitda"]:      [{"y": 2023, "v": 8.0}, {"y": 2024, "v": 7.2}],
            KPI["debt_to_equity"]: [{"y": 2023, "v": 30.0}, {"y": 2024, "v": 25.0}],   # raw %
        }

    def get_kpi_history(self, ins_id, kpi_id, report_type, price_type):
        self.calls.append(kpi_id)
        return self.hist.get(kpi_id, [])


def _no_cache(monkeypatch):
    # Bypass the 24h TTLCache so the stub history is always exercised.
    monkeypatch.setattr(cache, "get_fundamentals", lambda k: None)
    monkeypatch.setattr(cache, "set_fundamentals", lambda k, v: None)


def _reset_batch_state():
    bd.KPI_BATCH_ATTEMPTED.clear()
    bd.KPI_BATCH_FAILED.clear()


class TestHistoryFallback:
    def test_backfills_failed_and_never_attempted_kpis(self, monkeypatch):
        _no_cache(monkeypatch)
        _reset_batch_state()
        # Batch attempted these; ebitda_margin + fcf_m 400'd. equity/assets/ebit
        # were never attempted by the batch at all.
        for k in ("ebitda_margin", "fcf_m", "debt_to_equity", "revenue_m",
                  "market_cap", "ev_ebitda"):
            bd.KPI_BATCH_ATTEMPTED.add(KPI[k])
        bd.KPI_BATCH_FAILED.update({KPI["ebitda_margin"], KPI["fcf_m"]})

        fund_snap = {
            "ins_id": 123,
            "debt_to_equity": 0.25,  # batch OK
            "revenue_m": 3500.0,     # batch OK
            "market_cap": 4200.0,    # batch OK
            "ev_ebitda": 7.2,        # batch OK
        }
        api = _StubAPI()
        filled = _augment_fund_snap_from_history(fund_snap, 123, api)

        # Back-filled: the two 400'd KPIs + the three never-attempted ones.
        assert set(filled) == {
            KPI["ebitda_margin"], KPI["fcf_m"],
            KPI["total_equity_m"], KPI["total_assets_m"], KPI["ebit_m"],
        }
        # Divisor is applied identically to the batch (38.0 raw % -> 0.38).
        assert fund_snap["ebitda_margin"] == 0.38
        assert fund_snap["total_equity_m"] == 2000.0

    def test_guard_does_not_refetch_working_kpis(self, monkeypatch):
        _no_cache(monkeypatch)
        _reset_batch_state()
        for k in ("debt_to_equity", "revenue_m", "market_cap", "ev_ebitda"):
            bd.KPI_BATCH_ATTEMPTED.add(KPI[k])
        fund_snap = {
            "ins_id": 1, "debt_to_equity": 0.25, "revenue_m": 3500.0,
            "market_cap": 4200.0, "ev_ebitda": 7.2,
        }
        api = _StubAPI()
        _augment_fund_snap_from_history(fund_snap, 1, api)
        for k in ("debt_to_equity", "revenue_m", "market_cap", "ev_ebitda"):
            assert KPI[k] not in api.calls, f"guard failed: re-fetched working {k}"

    def test_guard_skips_working_but_genuinely_none_kpi(self, monkeypatch):
        _no_cache(monkeypatch)
        _reset_batch_state()
        # ev_ebitda was attempted and succeeded, but the value is genuinely None.
        bd.KPI_BATCH_ATTEMPTED.add(KPI["ev_ebitda"])
        fund_snap = {"ins_id": 9, "ev_ebitda": None}
        api = _StubAPI()
        _augment_fund_snap_from_history(fund_snap, 9, api)
        assert KPI["ev_ebitda"] not in api.calls
        # ...but never-attempted equity is still covered.
        assert KPI["total_equity_m"] in api.calls

    def test_gate_evaluates_and_altman_computable_after_backfill(self, monkeypatch):
        _no_cache(monkeypatch)
        _reset_batch_state()
        bd.KPI_BATCH_FAILED.update({KPI["ebitda_margin"], KPI["fcf_m"]})
        fund_snap = {"ins_id": 5, "debt_to_equity": 0.25}
        api = _StubAPI()
        _augment_fund_snap_from_history(fund_snap, 5, api)
        fund = _build_fundamentals_dict(fund_snap)

        # total_liabilities derived from assets - equity.
        assert fund["total_liabilities"] == 3000.0
        res = calculate_strength_score(fund)
        # Gates now evaluate to real pass/fail (not "missing").
        for gate in ("fcf_positive", "ebitda_margin_positive",
                     "equity_positive", "altman_z_ok"):
            assert res.gate_status[gate] != "missing", res.gate_status
        assert res.altman_z is not None

    def test_no_api_or_empty_snapshot_is_noop(self, monkeypatch):
        _no_cache(monkeypatch)
        _reset_batch_state()
        assert _augment_fund_snap_from_history({}, 1, _StubAPI()) == []
        assert _augment_fund_snap_from_history({"ins_id": 1}, None, _StubAPI()) == []
        assert _augment_fund_snap_from_history({"ins_id": 1}, 1, None) == []


class TestBatchFailureTracking:
    def test_critical_map_ids_exist_in_kpi_table(self):
        # Every fallback field maps to a real Börsdata KPI id.
        for key, (kpi_id, _divisor) in _critical_kpi_fallback().items():
            assert isinstance(kpi_id, int) and kpi_id > 0

    def test_snapshot_fast_records_attempted_and_failed(self, monkeypatch):
        _reset_batch_state()

        # Fake a BorsdataAPI.get_kpi_screener that 400s (empty + path logged)
        # for one KPI and returns data for another.
        from borsdata_api import BorsdataAPI

        api = BorsdataAPI.__new__(BorsdataAPI)

        def fake_screener(kpi_id, calc_group="last", calc="latest"):
            path = f"/instruments/kpis/{kpi_id}/last/latest"
            if kpi_id == KPI["ebitda_margin"]:
                bd._KPI_ERROR_LOGGED.add(path)   # simulate _get's 400 bookkeeping
                return []
            return [{"i": 1, "n": 10.0}]

        monkeypatch.setattr(api, "get_kpi_screener", fake_screener)
        snaps = api.get_fundamentals_snapshot_fast([1])

        assert KPI["ebitda_margin"] in bd.KPI_BATCH_ATTEMPTED
        assert KPI["ebitda_margin"] in bd.KPI_BATCH_FAILED
        # A KPI that returned data must be attempted but not failed.
        assert KPI["market_cap"] in bd.KPI_BATCH_ATTEMPTED
        assert KPI["market_cap"] not in bd.KPI_BATCH_FAILED
        assert snaps[1]["ins_id"] == 1
