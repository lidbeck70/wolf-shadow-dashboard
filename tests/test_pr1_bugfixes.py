"""
PR 1 — buggfixar ur panelgenomgången:
  resursscoring läser CSV-stavningar · håvträffar får Yahoo-suffix ·
  inget fuzzy tickeruppslag · Gist-läsare hanterar trunkering ·
  yfinance-anrop utan döda kwargs · Actions-schemat kör en gång per timme ·
  Wolf-checklistan har inga hårdkodade godkännanden.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── resursscoring ─────────────────────────────────────────────────────────────
def test_resource_scoring_reads_csv_spellings():
    from contrarian_alpha import resource_scoring as rs
    for name, expected in (("oil_gas", 80.0), ("Oil_Gas", 80.0), ("iron_ore", 79.0), ("iron ore", 79.0),
                           ("rare_earth", 90.0), ("rare earths", 90.0), ("natural_gas", 78.0),
                           ("steel", 80.0), ("aluminum", 82.0), ("aluminium", 82.0), ("uranium", 95.0)):
        score, flags = rs.score_commodity(name)
        assert score == expected and "COMMODITY_UNKNOWN" not in flags, name
    score, flags = rs.score_commodity("betting")
    assert score == rs._COMMODITY_DEFAULT and flags == ["COMMODITY_UNKNOWN"]
    assert rs.score_commodity("")[1] == ["COMMODITY_UNKNOWN"]
    # sekundär råvara med understreck ger samma knuff som klartext
    assert rs.score_commodity("gold", "rare_earth")[0] == rs.score_commodity("gold", "rare earth")[0] == 88.0


def test_every_row_in_the_resource_csv_is_scored():
    import csv
    from contrarian_alpha import resource_scoring as rs
    path = os.path.join(ROOT, "config", "universes", "us_ca_resource.csv")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    unknown = [r["ticker"] for r in rows if "COMMODITY_UNKNOWN" in rs.score_commodity(r.get("primary_commodity", ""))[1]]
    assert rows and unknown == [], unknown


# ── håvens tickers ────────────────────────────────────────────────────────────
def test_screen_hits_get_yahoo_suffix_and_dash():
    import screens_scan as ss
    assert ss.yahoo_ticker({"ticker": "KDK", "marketId": 36}, "global") == "KDK.V"
    assert ss.yahoo_ticker({"ticker": "TECK B", "marketId": 35}, "global") == "TECK-B.TO"
    assert ss.yahoo_ticker({"ticker": "CCJ", "marketId": 32}, "global") == "CCJ"
    assert ss.yahoo_ticker({"ticker": "ANTO", "marketId": 38}, "global") == "ANTO.L"
    assert ss.yahoo_ticker({"ticker": "BOL", "marketId": 1}, "nordic") == "BOL.ST"
    assert ss.yahoo_ticker({"ticker": "VOLV B", "marketId": 1}, "nordic") == "VOLV-B.ST"
    assert ss.yahoo_ticker({"ticker": "XYZ", "marketId": 999}, "global") == "XYZ"      # okänd marknad: inget suffix


# ── inget fuzzy uppslag ───────────────────────────────────────────────────────
def test_borsdata_resolve_never_matches_substrings():
    from borsdata_api import BorsdataAPI
    api = BorsdataAPI.__new__(BorsdataAPI)
    api._ticker_map = {"BOL": 40, "BOLIDEN": 40, "GOLDEN HEIGHTS": 77, "VOLV B": 12, "AGES": 5}
    assert api.resolve_instrument_id("BOL.ST") == 40 and api.resolve_instrument_id("VOLV-B.ST") == 12
    assert api.resolve_instrument_id("GOLD") is None                 # tidigare → GOLDEN HEIGHTS
    assert api.resolve_instrument_id("AG") is None and api.resolve_instrument_id("X") is None

    from utils import bd_api
    src = open(os.path.join(ROOT, "utils", "bd_api.py"), encoding="utf-8").read()
    assert "query in key or key in query" not in src and "query in key or key in query" not in \
        open(os.path.join(ROOT, "borsdata_api.py"), encoding="utf-8").read()
    assert hasattr(bd_api, "BDClient")


# ── Gist-trunkering ───────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


def _truncated_gist(filename: str, real: str):
    files = {filename: {"content": "", "truncated": True, "raw_url": "https://gist.example/raw/" + filename}}

    def fake_get(url, *a, **kw):
        if url.startswith("https://gist.example/raw/"):
            return _Resp(200, text=real)
        return _Resp(200, payload={"files": files})
    return fake_get


def test_gist_readers_follow_raw_url_when_truncated(monkeypatch, tmp_path):
    import requests
    import gist_storage
    monkeypatch.setattr(requests, "get", _truncated_gist("any.json", '[{"id": "t1"}]'))
    assert gist_storage._gist_file_content({"any.json": {"content": "", "truncated": True,
                                                         "raw_url": "https://gist.example/raw/any.json"}}, "any.json") == '[{"id": "t1"}]'

    # tradejournalen: trunkerad fil får inte bli en tom lista (som sedan skriver över Gisten)
    import streamlit as st
    import trade_journal as tj
    st.session_state.pop("journal_trades", None)
    monkeypatch.setattr(tj, "LOCAL_FALLBACK", str(tmp_path / "nope.json"))
    monkeypatch.setattr(requests, "get", _truncated_gist(tj.JOURNAL_FILENAME, '[{"id": "t1", "ticker": "BOL"}]'))
    trades = tj.load_journal()
    assert trades and trades[0]["ticker"] == "BOL"
    st.session_state.pop("journal_trades", None)

    # Contrarian-cachen
    from contrarian_alpha import cache as ca
    monkeypatch.setattr(requests, "get", _truncated_gist("contrarian_alpha_results_quality.json",
                                                         json.dumps({"results": [{"ticker": "BOL"}], "timestamp": "x"})))
    assert ca.load_screener_results("quality")["results"][0]["ticker"] == "BOL"

    # EMBER-cachen
    from ember import cache as ec
    monkeypatch.setattr(ec, "_GIST_API_URL", "https://api.github.com/gists/x")
    monkeypatch.setattr(requests, "get", _truncated_gist(ec._GIST_FILENAME, json.dumps({"eligible": [{"t": 1}]})))
    assert ec.load_ember_results()["eligible"] == [{"t": 1}]

    # annoteringarna
    from contrarian_alpha import annotations as an
    monkeypatch.setattr(an, "_LOCAL_PATH", str(tmp_path / "ann.json"))
    monkeypatch.setattr(requests, "get", _truncated_gist(an._GIST_FILE, json.dumps({"BOL": {"note": "x"}})))
    assert an.load_annotations()["BOL"]["note"] == "x"


# ── yfinance-kwargs som inte finns ────────────────────────────────────────────
def test_no_dead_yfinance_kwargs_in_ticker_history():
    import inspect
    import yfinance as yf
    assert "progress" not in inspect.signature(yf.Ticker.history).parameters
    bad = []
    for path in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
        if "/tests/" in path or "/.venv/" in path:
            continue
        src = open(path, encoding="utf-8", errors="ignore").read()
        for m in re.finditer(r"\.history\((?:[^()]|\([^()]*\))*progress\s*=", src):
            bad.append(f"{os.path.relpath(path, ROOT)}: {m.group(0)[:60]}")
    assert bad == [], bad


# ── Actions-schemat ───────────────────────────────────────────────────────────
def test_scheduled_scan_runs_once_per_local_hour_and_wolf_data_does_not_overlap():
    import yaml
    wf = yaml.safe_load(open(os.path.join(ROOT, ".github", "workflows", "scheduled-scan.yml"), encoding="utf-8"))
    steps = wf["jobs"]["screen"]["steps"]
    assert steps[0]["id"] == "gate" and "Europe/Stockholm" in steps[0]["run"]
    for s in steps[1:]:
        assert "steps.gate.outputs.run == 'true'" in str(s.get("if", "")), s["name"]
    crons = [c["cron"] for c in wf[True]["schedule"]] if True in wf else [c["cron"] for c in wf["on"]["schedule"]]
    assert crons == ["0 6,10,16 * * 1-5", "0 7,11,17 * * 1-5"]
    wd = yaml.safe_load(open(os.path.join(ROOT, ".github", "workflows", "wolf-data.yml"), encoding="utf-8"))
    on = wd[True] if True in wd else wd["on"]
    assert on["schedule"][0]["cron"] == "30 4 * * 1-5"                    # 90 min före scheduled-scan


# ── Wolf-checklistan ──────────────────────────────────────────────────────────
def test_wolf_checklist_has_no_hardcoded_passes():
    src = open(os.path.join(ROOT, "tabs", "regime.py"), encoding="utf-8").read()
    block = src[src.index("swing_gates = ["):src.index("MÄTBARA GRINDAR")]
    assert '"passed": True' not in block                                 # inget godkänns av sig självt
    assert block.count('"passed": None') == 8                           # 4 manuella × 2 grenar
    assert 'Data ej tillgänglig ({type(_gate_exc).__name__})' in block   # fel → ej godkänd, med orsak
    assert "measurable = [g for g in swing_gates if g[\"passed\"] is not None]" in src
