"""
EMBER-resultaten går även via huvudgisten (gist_storage) — panelen hittar dem utan EMBER_GIST_ID, så Thor ser
EMBER:s köp. Läsare och skrivare är fejkade; inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gist_storage  # noqa: E402
from ember import cache as ec  # noqa: E402

BLOB = {"timestamp": "2026-10-20T16:40:25", "eligible": [{"ticker": "SCA-B.ST", "verdict": "KÖPLÄGE"}],
        "near_misses": [{"ticker": "UPM.HE", "verdict": "BEVAKA"}]}


def test_load_without_ember_gist_id_reads_the_main_gist(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, "_GIST_API_URL", "")
    monkeypatch.setattr(ec, "_LOCAL_FALLBACK", str(tmp_path / "saknas.json"))
    seen = []
    monkeypatch.setattr(gist_storage, "load_blob", lambda name, fb: seen.append(name) or (BLOB if name == ec._GIST_FILENAME else fb))
    assert ec.load_ember_results() == BLOB and seen == ["ember_results.json"]
    monkeypatch.setattr(gist_storage, "load_blob", lambda name, fb: fb)          # inget någonstans
    assert ec.load_ember_results() == {"eligible": [], "near_misses": [], "timestamp": None}


def test_save_writes_the_main_gist_even_without_ember_gist_id(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, "_GIST_API_URL", "")
    monkeypatch.setattr(ec, "_LOCAL_FALLBACK", str(tmp_path / "ember.json"))
    saved = {}
    monkeypatch.setattr(gist_storage, "save_blob", lambda name, data: saved.__setitem__(name, data) or True)
    res = SimpleNamespace(timestamp="2026-10-20T16:40:25", eligible=[], near_misses=[])
    assert ec.save_ember_results(res) is True
    assert saved["ember_results.json"]["timestamp"] == "2026-10-20T16:40:25" and saved["ember_results.json"]["eligible"] == []
    monkeypatch.setattr(gist_storage, "save_blob", lambda name, data: False)
    assert ec.save_ember_results(res) is False
