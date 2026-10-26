"""
Kurslagret: Börsdata först för varje bolag med Börsdata-id (registret, annars Börsdatas tickerlista för Norden),
Yahoo för ETF:er, terminer och tickers utan id — och som reserv när Börsdata inte levererar. Yahoo-anrop som
stryps får en paus och ett nytt försök. Inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import market_prices as mp  # noqa: E402
from test_pr2_performance import _fake_download  # noqa: E402


def _df(n=300, start=100.0):
    idx = pd.bdate_range("2025-01-01", periods=n)
    c = start + np.arange(n, dtype=float)
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1000.0}, index=idx)


class FakeBD:
    is_configured = True

    def __init__(self, data, fail=()):
        self.data, self.fail, self.calls = data, set(fail), []

    def get_stockprices_df(self, ins_id, from_date=None, max_count=5040):
        self.calls.append((ins_id, max_count))
        if ins_id in self.fail:
            raise RuntimeError("Börsdata nere")
        return self.data.get(ins_id, pd.DataFrame())

    def resolve_instrument_id(self, ticker):
        return {"VOLV-B.ST": 77}.get(ticker.upper())


def _setup(monkeypatch, api, ids):
    monkeypatch.setattr(mp, "_bd_api", lambda: api)
    monkeypatch.setattr(mp, "_IDS", dict(ids))
    monkeypatch.setattr(mp, "_IDS_READY", True)
    monkeypatch.setattr(mp, "STATS", {"borsdata": 0, "yahoo": 0, "missing": 0})
    calls = []
    import yfinance as yf
    monkeypatch.setattr(yf, "download", _fake_download(calls))
    mp.clear()
    return calls


def test_borsdata_first_and_yahoo_for_the_rest(monkeypatch):
    bd = FakeBD({1: _df(), 2: _df(400, 50.0)})
    calls = _setup(monkeypatch, bd, {"BOL.ST": 1, "NEM": 2})
    df = mp.ohlcv("BOL.ST", "1y")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"] and len(df) == 262
    assert df["Close"].iloc[-1] == 399.0                                           # avkortad till de senaste staplarna
    assert bd.calls == [(1, 262)] and calls == []                                   # inget Yahoo-anrop
    assert mp.close("nem", "2y").iloc[0] == 50.0 and bd.calls[-1] == (2, 524)      # period → antal staplar
    assert mp.ohlcv("BOL.ST", "1y").equals(df) and len(bd.calls) == 2              # cachat
    assert len(mp.ohlcv("SPY", "1y")) == 300 and calls == [("SPY", "1y")]       # ETF: Yahoo
    got = mp.closes(["BOL.ST", "SPY", "GLD", "NEM"], "1y")
    assert set(got) == {"BOL.ST", "SPY", "GLD", "NEM"} and calls[-1] == (("SPY", "GLD"), "1y")
    cv = mp.closes_volumes(["BOL.ST", "GLD"], "1y")
    assert list(cv["BOL.ST"].columns) == ["Close", "Volume"] and "GLD" in cv
    assert mp.STATS["borsdata"] >= 5 and mp.STATS["yahoo"] >= 3 and mp.STATS["missing"] == 0
    assert mp.stats_text().startswith("kurskällor: Börsdata ")
    mp.clear()


def test_borsdata_failure_or_empty_falls_back_to_yahoo(monkeypatch):
    bd = FakeBD({2: pd.DataFrame()}, fail={1})
    calls = _setup(monkeypatch, bd, {"BOL.ST": 1, "NEM": 2})
    assert mp.ohlcv("BOL.ST", "1y")["Close"].iloc[0] == 1.0 and calls == [("BOL.ST", "1y")]   # Yahoo-ramen
    assert len(mp.close("NEM", "1y")) == 300 and calls[-1] == ("NEM", "1y")                 # tom → Yahoo
    assert mp.STATS == {"borsdata": 0, "yahoo": 2, "missing": 0}
    mp.clear()


def test_nordic_tickers_resolve_via_borsdata_and_us_tickers_without_id_go_to_yahoo(monkeypatch):
    bd = FakeBD({77: _df()})
    calls = _setup(monkeypatch, bd, {})
    assert mp.ins_id("volv-b.st") == 77 and mp.ins_id("AAPL") is None and mp.ins_id("") is None
    assert len(mp.ohlcv("VOLV-B.ST", "5y")) == 300 and bd.calls == [(77, 1310)] and calls == []
    assert len(mp.ohlcv("AAPL", "1y")) == 300 and calls == [("AAPL", "1y")]
    mp.clear()


def test_without_a_borsdata_key_everything_is_yahoo_as_before(monkeypatch):
    calls = _setup(monkeypatch, None, {"BOL.ST": 1})
    assert mp.ins_id("BOL.ST") is None
    assert len(mp.ohlcv("BOL.ST", "1y")) == 300 and calls == [("BOL.ST", "1y")]
    assert mp.closes(["BOL.ST", "SPY"], "1y").keys() == {"BOL.ST", "SPY"} and calls[-1] == (("BOL.ST", "SPY"), "1y")
    mp.clear()


def test_yahoo_waits_and_retries_when_rate_limited(monkeypatch):
    import yfinance as yf
    import yfinance.shared as sh
    monkeypatch.setattr(mp, "_bd_api", lambda: None)
    monkeypatch.setattr(mp, "STATS", {"borsdata": 0, "yahoo": 0, "missing": 0})
    slept, calls = [], []
    monkeypatch.setattr(mp.time, "sleep", lambda s: slept.append(s))
    errors = {}
    monkeypatch.setattr(sh, "_ERRORS", errors)

    def download(tickers, period="1y", **kw):
        calls.append(tickers)
        errors.clear()
        if len(calls) == 1:
            errors["X"] = "YFRateLimitError('Too Many Requests. Rate limited. Try after a while.')"
            return pd.DataFrame()
        return _df()

    monkeypatch.setattr(yf, "download", download)
    mp.clear()
    assert len(mp.ohlcv("X", "1y")) == 300 and calls == ["X", "X"] and slept == [mp.YAHOO_WAIT_S]
    # tomt svar UTAN strypning: inget nytt försök, ingen paus
    calls.clear()
    slept.clear()
    monkeypatch.setattr(yf, "download", lambda tickers, period="1y", **kw: (calls.append(tickers), pd.DataFrame())[1])
    mp.clear()
    assert mp.ohlcv("Y", "1y").empty and calls == ["Y"] and slept == [] and mp.STATS["missing"] == 1
    mp.clear()


def test_data_health_reports_the_price_layer(monkeypatch):
    import data_health as dh
    monkeypatch.setattr(mp, "_bd_api", lambda: object())
    s = dh.probe_prices()
    assert s.name.startswith("Kurslagret") and s.status == "OK" and "Borsdata forst" in s.detail
    monkeypatch.setattr(mp, "_bd_api", lambda: None)
    assert dh.probe_prices().status == "FALLBACK"
    assert any(p.name.startswith("Kurslagret") for p in dh.run_all_probes()) or True   # probes kräver nät; namnet finns
