"""Granskning 2 (identitet): rätt bolag och rätt notering, tomma körningar och nyaste EMBER-resultatet."""
import json

import markets
from thor import candidates as tc
from thor import dossier as td
from thor import ticker as tt


class _Api:
    def __init__(self, inst):
        self.inst = inst

    def get_instruments(self, *a, **k):
        return self.inst

    def __getattr__(self, name):
        return lambda *a, **k: self.inst


def test_bare_us_ticker_in_the_list_is_not_rewritten_to_a_nordic_listing(monkeypatch):
    monkeypatch.setattr(tt, "lookup", lambda api, t: {"insId": 7, "ticker": "TEL", "name": "Telenor", "marketId": 9,
                                                       "stockPriceCurrency": "NOK", "sectorId": 9, "branchId": 70})
    assert tt.identify("TEL", api=object(), table=markets.FALLBACK, keep_bare=True).yf == "TEL"
    assert tt.identify("TEL", api=object(), table=markets.FALLBACK).yf != "TEL"      # utan listrad som förut


def test_cross_listed_rows_are_not_merged():
    rows = [tc.Candidate("EQNR", "Equinor ADR", "wolf", "BULL", tc.BEVAKA),
            tc.Candidate("EQNR.OL", "Equinor", "viking_nine", "READY", tc.BEVAKA)]
    assert [c.ticker for c in td.candidates_for(rows, "EQNR.OL")] == ["EQNR.OL"]
    assert [c.ticker for c in td.candidates_for(rows, "EQNR")] == ["EQNR", "EQNR.OL"]
    assert [c.ticker for c in td.candidates_for(rows[:1] + [tc.Candidate("BOL.ST", "B", "wolf", "x", tc.HALL)],
                                                "BOL")] == ["BOL.ST"]


def test_empty_contrarian_run_is_an_answer(monkeypatch):
    import gist_storage
    from contrarian_alpha import cache
    run = {"timestamp": "2026-10-24T06:00", "results": [], "commodity_only": True}
    monkeypatch.setattr(gist_storage, "load_blob", lambda name, fb=None: run)
    assert cache.load_screener_results(mode="deep_contrarian") == run


def test_newest_ember_result_wins(monkeypatch):
    import gist_storage
    from ember import cache as ec
    old = {"timestamp": "2026-10-20T06:00", "eligible": []}
    new = {"timestamp": "2026-10-24T06:00", "eligible": [{"ticker": "CCJ"}]}

    class _R:
        status_code = 200

        def json(self):
            return {"files": {ec._GIST_FILENAME: {"content": json.dumps(old)}}}
    monkeypatch.setattr(ec, "_GIST_API_URL", "https://api.github.com/gists/x")
    monkeypatch.setattr("requests.get", lambda *a, **k: _R())
    monkeypatch.setattr(gist_storage, "load_blob", lambda name, fb=None: new)
    assert ec.load_ember_results()["timestamp"] == "2026-10-24T06:00"
