"""
borsdata_probe — PROBE_SECTION=kpicalc: hittar vilka calcGroup/calc som ger värden för Long Screeners KPI:er
(syntetiska svar, inget nätverk).
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import borsdata_probe as bp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_probe_kpi_calc_finds_working_combinations(monkeypatch):
    calls = []

    def raw(path, params=None, pause=0):
        calls.append(path)
        if path.endswith("/metadata"):
            return 200, {"kpiHistoryMetadatas": [{"kpiId": 97, "nameSv": "Vinsttillväxt", "nameEn": "Earnings growth"}]}, ""
        if path == "/instruments/kpis/97/1year/latest" or path == "/instruments/kpis/33/last/latest":
            return 200, {"values": [{"i": 1, "n": 12.5}, {"i": 2, "n": None}]}, ""
        if path.endswith("/history"):
            return 200, {"kpisList": [{"instrument": 1, "values": [{"y": 2025, "v": 11.0}]}]}, ""
        return 400, None, '{"error":"invalid calcgroup"}'

    monkeypatch.setattr(bp, "_raw", raw)
    api = SimpleNamespace(get_instruments=lambda: [{"insId": 1, "ticker": "VOLV B"}, {"insId": 2, "ticker": "X"}])
    lines = []
    found = bp.probe_kpi_calc(api, out=lines.append)
    assert found[97] == [("1year", "latest", 1, {"VOLV B": 12.5})] and found[94] == []
    assert found[33][0][:2] == ("last", "latest")
    text = "\n".join(lines)
    assert "Vinsttillväxt" in text and "invalid calcgroup" in text and "INGEN" in text
    assert sum(1 for c in calls if c.startswith("/instruments/kpis/33/")) < 20              # kontrollen bryter tidigt
    assert sum(1 for c in calls if c.startswith("/instruments/kpis/94/") and not c.endswith("history")) == \
        len(bp.CALC_GROUPS) * len(bp.CALCS) + 1                                          # + felmeddelandet


def test_probe_workflow_offers_kpicalc():
    src = open(os.path.join(ROOT, ".github", "workflows", "borsdata-probe.yml"), encoding="utf-8").read()
    assert "kpicalc" in src and "PROBE_SECTION" in src
