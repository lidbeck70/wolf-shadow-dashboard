"""
Gistens 1 MB-tak. Probe 2026-09-11: 18 filer, 1,13 MB — GitHub flaggade
screens.json truncated (klippt på 256 KB) och de efterföljande filerna
(sheets_refresh, swing_data, wolf_regime, wolf_screener) kom tillbaka med
content="". Panelens load_blob gav None → "0 träffar" i håvarna och tomma
swing-flikar. Läsvägen ska hämta raw_url när filen är trunkerad.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gist_storage as g


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def _gist(files):
    return {"files": files}


def test_truncated_file_is_fetched_from_raw_url(monkeypatch):
    full = json.dumps({"screens": {"rule": {"rows": [1, 2, 3]}}})
    calls = []

    def fake_get(url, headers=None, timeout=10):
        calls.append(url)
        if url == g.GIST_API_URL:
            return _Resp(200, _gist({
                "screens.json": {"content": full[:20], "truncated": True,
                                 "raw_url": "https://gist.githubusercontent.com/x/raw/screens.json"},
                "swing_data.json": {"content": "", "truncated": True,
                                    "raw_url": "https://gist.githubusercontent.com/x/raw/swing_data.json"},
                "small.json": {"content": '{"ok": 1}', "truncated": False},
            }))
        if url.endswith("raw/screens.json"):
            return _Resp(200, text=full)
        if url.endswith("raw/swing_data.json"):
            return _Resp(200, text='{"positions": []}')
        raise AssertionError(url)

    monkeypatch.setattr(g.requests, "get", fake_get)
    assert g.load_blob("screens.json", None) == json.loads(full)
    assert g.load_blob("swing_data.json", None) == {"positions": []}
    assert g.load_blob("small.json", None) == {"ok": 1}
    assert g.load_wolf_json("swing_data.json") == {"positions": []}
    assert any(u.endswith("raw/screens.json") for u in calls)


def test_missing_file_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(g.requests, "get",
                        lambda url, headers=None, timeout=10: _Resp(200, _gist({})))
    monkeypatch.chdir(tmp_path)
    assert g.load_blob("nope.json", "fallback") == "fallback"
    (tmp_path / ".nope.json").write_text('{"local": true}')
    assert g.load_blob("nope.json", None) == {"local": True}


def test_reads_use_token_when_available(monkeypatch):
    seen = {}

    def fake_get(url, headers=None, timeout=10):
        seen["headers"] = headers or {}
        return _Resp(200, _gist({}))

    monkeypatch.setattr(g.requests, "get", fake_get)
    monkeypatch.setattr(g, "_get_github_token", lambda: "github_pat_abc")
    g.load_blob("x.json", None)
    assert seen["headers"].get("Authorization") == "Bearer github_pat_abc"
    monkeypatch.setattr(g, "_get_github_token", lambda: None)
    g.load_blob("x.json", None)
    assert "Authorization" not in seen["headers"]


def test_save_blob_writes_compact_json(monkeypatch, tmp_path):
    sent = {}

    def fake_patch(url, headers=None, json=None, timeout=10):
        sent.update(json)
        return _Resp(200)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(g.requests, "patch", fake_patch)
    monkeypatch.setattr(g, "_get_github_token", lambda: "tok")
    assert g.save_blob("b.json", {"a": [1, 2], "ö": "å"}) is True
    content = sent["files"]["b.json"]["content"]
    assert content == '{"a":[1,2],"ö":"å"}'
