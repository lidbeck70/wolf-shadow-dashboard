"""
Contrarian Alpha-resultaten (Quality, Deep Contrarian) går även via huvudgisten (gist_storage, med token) —
panelen på Streamlit Cloud läste förut CA-gisten utan token och fick ofta tomt (60 anrop/h per delad IP), så Thor
visade "inte i någon strategis lista" medan Discord-larmet såg bolaget. Samma lösning som EMBER (#219).
Läsare och skrivare är fejkade; inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gist_storage  # noqa: E402
from contrarian_alpha import cache as cc  # noqa: E402

DEEP = {"timestamp": "2026-10-23T16:40:00", "mode": "deep_contrarian", "commodity_only": True,
        "results": [{"ticker": "SOSI.ST", "rank": 3, "composite_score": 38.2}]}


def test_load_reads_the_main_gist_first(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "_GIST_API_URL", "http://127.0.0.1:9/none")           # CA-gisten nås inte
    monkeypatch.setattr(cc, "_LOCAL_FALLBACK", str(tmp_path / ".ca_results.json"))
    seen = []
    monkeypatch.setattr(gist_storage, "load_blob",
                        lambda name, fb: seen.append(name) or (DEEP if name == "contrarian_alpha_results_deep_contrarian.json" else fb))
    assert cc.load_screener_results("deep_contrarian") == DEEP
    assert seen == ["contrarian_alpha_results_deep_contrarian.json"]
    assert cc.load_screener_results("quality")["results"] == []                   # inget någonstans → tomt, inte Deep
    monkeypatch.setattr(gist_storage, "load_blob", lambda name, fb: fb)
    assert cc.load_screener_results("deep_contrarian")["results"] == []


def test_save_writes_the_main_gist_and_skips_the_duplicate_patch(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "_LOCAL_FALLBACK", str(tmp_path / ".ca_results.json"))
    saved, patched = {}, []
    monkeypatch.setattr(gist_storage, "save_blob", lambda name, data: saved.__setitem__(name, data) or True)
    monkeypatch.setattr(cc, "_get_github_token", lambda: "tok")
    import requests
    monkeypatch.setattr(requests, "patch", lambda *a, **k: patched.append(a) or SimpleNamespace(status_code=200))
    res = SimpleNamespace(timestamp="2026-10-23T16:40:00", universe_count=100, run_duration_s=1.0,
                          results=[], config=SimpleNamespace(mode="deep_contrarian"), commodity_passed=12)
    monkeypatch.setattr(cc, "_GIST_ID", gist_storage.GIST_ID)                      # CA-gisten ÄR huvudgisten
    assert cc.save_screener_results(res, mode="deep_contrarian") is True
    assert saved["contrarian_alpha_results_deep_contrarian.json"]["mode"] == "deep_contrarian" and patched == []
    monkeypatch.setattr(cc, "_GIST_ID", "annan-gist")                              # en egen CA-gist: båda skrivs
    assert cc.save_screener_results(res, mode="quality") is True and len(patched) == 1
    monkeypatch.setattr(gist_storage, "save_blob", lambda name, data: False)        # utan token: bara lokalt
    monkeypatch.setattr(cc, "_get_github_token", lambda: "")
    assert cc.save_screener_results(res, mode="quality") is False


def test_thor_reads_the_same_list_as_the_alert(monkeypatch):
    from thor import candidates as tc
    monkeypatch.setattr(gist_storage, "load_blob", lambda name, fb: DEEP if "deep" in name else fb)
    monkeypatch.setattr(cc, "_GIST_API_URL", "http://127.0.0.1:9/none")
    blobs = tc.load_blobs(loader=lambda name, fb: fb)
    rows = tc.from_contrarian(blobs["ca_deep"], {}, mode="deep_contrarian")
    assert [(r.ticker, r.status, r.status_class) for r in rows] == [("SOSI.ST", "rank 3", tc.BEVAKA)]
