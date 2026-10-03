"""
🐺 Fiat Debasement PR 1 — datalagret (sources) och datasonden (probe).
Falska HTTP-svar — inget nätverk.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiat_debasement import probe as pr  # noqa: E402
from fiat_debasement import sources as src  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Resp:
    def __init__(self, text="", payload=None, status=200):
        self.text, self._payload, self.status = text, payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class _Http:
    """Svarar ur en tabell {url-delsträng: _Resp}; registrerar anropen."""
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def _find(self, url):
        self.calls.append(url)
        for part, resp in self.routes.items():
            if part in url:
                return resp
        return _Resp(status=404)

    def get(self, url, timeout=None, **kw):
        return self._find(url)

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        return self._find(url + "#post")


def test_parse_period_and_frequency():
    assert src.parse_period("2024-03") == pd.Timestamp("2024-03-01")
    assert src.parse_period("2024M03") == pd.Timestamp("2024-03-01")
    assert src.parse_period("2024-Q2") == pd.Timestamp("2024-04-01")
    assert src.parse_period("2024K4") == pd.Timestamp("2024-10-01")
    assert src.parse_period("2024") == pd.Timestamp("2024-01-01")
    assert src.parse_period("2024-03-15") == pd.Timestamp("2024-03-15")
    assert src.parse_period("skräp") is None
    assert src.frequency_of(pd.bdate_range("2024-01-01", periods=30)) == "D"
    assert src.frequency_of(pd.date_range("2020-01-01", periods=30, freq="MS")) == "M"
    assert src.frequency_of(pd.date_range("2000-01-01", periods=30, freq="QS")) == "Q"
    assert src.frequency_of(pd.date_range("1990-01-01", periods=30, freq="YS")) == "A"
    assert src.frequency_of(pd.DatetimeIndex([])) == ""


def test_fred_parses_csv_and_skips_missing_values():
    csv = "observation_date,M2SL\n2024-01-01,20800.1\n2024-02-01,.\n2024-03-01,20900.5\n"
    sd = src.fred("M2SL", http=_Http({"id=M2SL": _Resp(csv)}), unit="mdr USD", currency="USD")
    assert sd.ok and list(sd.values) == [20800.1, 20900.5]                    # '.' = saknas, aldrig 0
    assert sd.first == "2024-01-01" and sd.last == "2024-03-01" and sd.last_value == 20900.5
    assert sd.source == "FRED" and sd.unit == "mdr USD" and sd.last_updated


def test_failures_never_become_zero():
    sd = src.fred("NOPE", http=_Http({}))
    assert not sd.ok and sd.values is None and "HTTP 404" in sd.error
    sd = src.fred("X", http=_Http({"id=X": _Resp("observation_date,X\n2024-01-01,.\n")}))
    assert not sd.ok and sd.error == "inga observationer"


def test_ecb_csv():
    csv = "KEY,FREQ,TIME_PERIOD,OBS_VALUE,UNIT\nX,M,2024-01,15000000,EUR\nX,M,2024-02,15100000,EUR\n"
    sd = src.ecb("BSI", "M.U2.Y.V.M20.X.1.U2.2300.Z01.E", http=_Http({"data-api.ecb.europa.eu": _Resp(csv)}))
    assert sd.ok and sd.series_id.startswith("BSI/") and sd.last_value == 15100000
    assert sd.meta["unit_code"] == "EUR"
    bad = src.ecb("BSI", "X", http=_Http({"data-api": _Resp("A,B\n1,2\n")}))
    assert not bad.ok and "oväntat" in bad.error


def test_eurostat_json_stat():
    payload = {"id": ["freq", "geo", "time"], "size": [1, 1, 3],
               "dimension": {"time": {"category": {"index": {"2024-01": 0, "2024-02": 1, "2024-03": 2}}}},
               "value": {"0": 120.1, "2": 121.0}}
    sd = src.eurostat("prc_hicp_midx", {"geo": "SE"}, http=_Http({"eurostat": _Resp(payload=payload)}))
    assert sd.ok and list(sd.values) == [120.1, 121.0] and "geo=SE" in sd.series_id
    many = dict(payload, size=[1, 2, 3])
    sd = src.eurostat("x", {}, http=_Http({"eurostat": _Resp(payload=many)}))
    assert not sd.ok and "fler än ett värde" in sd.error


def test_scb_table_locks_every_variable_and_reads_the_time_series():
    meta = {"title": "KPI", "variables": [
        {"code": "ContentsCode", "values": ["000004VU", "X"], "valueTexts": ["KPI 1980=100", "annat"]},
        {"code": "Tid", "time": True, "values": ["2024M01", "2024M02"]}]}
    data = {"columns": [{"code": "Tid"}, {"code": "000004VU"}],
            "data": [{"key": ["2024M01"], "values": ["410.2"]}, {"key": ["2024M02"], "values": ["411.0"]}]}
    http = _Http({"KPItotM#post": _Resp(payload=data), "KPItotM": _Resp(payload=meta)})
    sd = src.scb_table("PR/PR0101/PR0101A/KPItotM", http=http)
    assert sd.ok and list(sd.values) == [410.2, 411.0] and sd.frequency == ""          # för få punkter
    post = [c for c in http.calls if isinstance(c, tuple)][0][2]
    assert post["query"] == [{"code": "ContentsCode", "selection": {"filter": "item", "values": ["000004VU"]}}]
    assert sd.meta["chosen"]["ContentsCode"].startswith("000004VU")
    pref = src.scb_table("PR/PR0101/PR0101A/KPItotM", prefer={"ContentsCode": "X"}, http=http)
    assert pref.meta["chosen"]["ContentsCode"].startswith("X")


def test_riksbank_yahoo_borsdata():
    rows = [{"date": "2024-01-02", "value": 10.4}, {"date": "2024-01-03", "value": 10.5}]
    sd = src.riksbank("SEKUSDPMI", http=_Http({"Observations/SEKUSDPMI": _Resp(payload=rows)}))
    assert sd.ok and sd.last_value == 10.5
    idx = pd.date_range("2024-01-01", periods=5, tz="UTC")
    sd = src.yahoo("GC=F", getter=lambda t, p: pd.DataFrame({"Close": [1, 2, 3, 4, 5.0]}, index=idx))
    assert sd.ok and sd.values.index.tz is None and sd.frequency == "D"
    assert not src.yahoo("X", getter=lambda t, p: pd.DataFrame()).ok

    class _Api:
        def get_stockprices_df(self, ins, max_count=None):
            return pd.DataFrame({"Close": [3.0, 1.0]}, index=pd.to_datetime(["2024-01-03", "2024-01-02"]))
    sd = src.borsdata(21031, api=_Api())
    assert sd.ok and list(sd.values) == [1.0, 3.0]                              # sorterad


def _sd(dates, source="FRED", sid="S", err=None):
    if err:
        return src.SeriesData(source, sid, error=err)
    s = pd.Series(range(1, len(dates) + 1), index=pd.DatetimeIndex(dates), dtype=float)
    sd = src.SeriesData(source, sid, values=s)
    sd.frequency = src.frequency_of(s.index)
    return sd


def test_probe_rows_preference_and_stale_series():
    today = pd.Timestamp("2026-10-03")
    monthly = pd.date_range("2020-01-01", "2026-08-01", freq="MS")
    old = pd.date_range("2000-01-01", "2022-01-01", freq="MS")
    cands = [("KPI", "SEK", lambda: _sd([], "SCB", "KPItotM", err="HTTP 404")),
             ("KPI", "SEK", lambda: _sd(old, "FRED", "OLD")),
             ("KPI", "SEK", lambda: _sd(monthly, "Eurostat", "HICP")),
             ("M2", "EUR", lambda: (_ for _ in ()).throw(RuntimeError("krasch")))]
    lines = []
    rows, disc = pr.run(cands, lister=lambda p: [], out=lines.append, today=today)
    assert [r["status"] for r in rows] == ["FEL", "GAMMAL", "OK", "FEL"]          # nedlagd serie flaggas
    best = pr.chosen(rows)
    assert best[("KPI", "SEK")]["series"] == "HICP" and best[("M2", "EUR")] is None
    md = pr.markdown(rows, [("FM/FM5001/X", "Penningmängden")])
    assert "| M2 | EUR | **DATA UNAVAILABLE**" in md and "**Saknas:** M2 EUR" in md
    assert "HTTP 404" in md and "FM/FM5001/X" in md and len(lines) == 4


def test_scb_discovery_walks_folders():
    tree = {"FM": [("FM5001", "l", "Finansmarknad")],
            "FM/FM5001": [("A", "l", "Penningmängd")],
            "FM/FM5001/A": [("PM1", "t", "Penningmängden M1, M2, M3"), ("X", "t", "Utlåning")],
            "PR/PR0101": [("KPItotM", "t", "KPI, fastställda tal")]}
    found = pr.discover_scb(lambda p: tree.get(p, []))
    assert ("FM/FM5001/A/PM1", "Penningmängden M1, M2, M3") in found
    assert ("PR/PR0101/KPItotM", "KPI, fastställda tal") in found
    assert all("Utlåning" not in t for _p, t in found)


def test_candidates_are_exactly_the_configured_sources():
    from fiat_debasement import config as cfg
    cands = pr.candidates()
    n = sum(len(v) for v in cfg.SERIES.values()) + 2 * len(cfg.ASSET_SPLICE)
    assert len(cands) == n
    keys = {(c, cur) for c, cur, _f in cands}
    for cur in ("SEK", "EUR", "USD"):
        for concept in (cfg.M2, cfg.CPI, cfg.CORE, cfg.GDP, cfg.DEBT):
            assert (cfg.CONCEPT_LABEL[concept], cur) in keys, (concept, cur)
    for concept in (cfg.GOLD, cfg.SILVER, cfg.COPPER, cfg.OIL, cfg.BTC):
        assert (cfg.CONCEPT_LABEL[concept], "USD") in keys
    assert pr.STALE_DAYS["Q"] >= 300                         # Q1-siffran är aktuell långt in på hösten


def test_main_writes_the_step_summary(monkeypatch, tmp_path):
    out = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(out))
    monkeypatch.setattr(pr, "run", lambda: ([pr.row_of("KPI", "USD", _sd(pd.date_range(
        "2020-01-01", periods=12, freq="MS")))], []))
    assert pr.main() == 0
    assert "Fiat Debasement — datasond" in out.read_text(encoding="utf-8")


def test_workflow_exists():
    wf = open(os.path.join(ROOT, ".github", "workflows", "fiat-probe.yml"), encoding="utf-8").read()
    assert "workflow_dispatch" in wf and "python -m fiat_debasement.probe" in wf and "BORSDATA_API_KEY" in wf


def test_scb_prefer_text_picks_m2_and_outstanding_amounts():
    meta = {"title": "Penningmängd", "variables": [
        {"code": "Penningmangdsmatt", "values": ["M1", "M2", "M3"], "valueTexts": ["M1", "M2", "M3"]},
        {"code": "ContentsCode", "values": ["T1", "T2"],
         "valueTexts": ["Tillväxttakt, procent", "Utestående belopp, mnkr"]},
        {"code": "Tid", "time": True, "values": ["2026M07", "2026M08"]}]}
    data = {"columns": [{"code": "Penningmangdsmatt"}, {"code": "Tid"}, {"code": "T2"}],
            "data": [{"key": ["M2", "2026M07"], "values": ["5000000"]}, {"key": ["M2", "2026M08"], "values": ["5010000"]}]}
    http = _Http({"pm#post": _Resp(payload=data), "pm": _Resp(payload=meta)})
    sd = src.scb_table("FM/pm", prefer_text=("M2", "utestående"), http=http)
    assert sd.ok and sd.last_value == 5010000
    assert sd.meta["chosen"] == {"Penningmangdsmatt": "M2 (M2)", "ContentsCode": "T2 (Utestående belopp, mnkr)"}
