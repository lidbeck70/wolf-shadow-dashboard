"""Granskning 2: ett läsfel i Gisten är inte en tom fil — larm, larmlogg och Thor skiljer på dem."""
import json
from types import SimpleNamespace

import gist_storage as gs


class _Resp:
    def __init__(self, code, files=None, text=""):
        self.status_code, self._files, self.text = code, files, text

    def json(self):
        return {"files": self._files or {}}


def _patch(monkeypatch, files=None, code=200, raw=None):
    calls = []

    def get(url, headers=None, timeout=None, **kw):
        calls.append(url)
        if url == gs.GIST_API_URL:
            return _Resp(code, files)
        return raw if raw is not None else _Resp(500)
    monkeypatch.setattr(gs.requests, "get", get)
    gs._invalidate()
    return calls


def test_read_blob_tells_missing_from_unreadable(monkeypatch):
    _patch(monkeypatch, {"a.json": {"content": json.dumps({"x": 1})}, "bad.json": {"content": "{trasig"},
                         "big.json": {"content": "", "truncated": True, "raw_url": "https://raw/big"}})
    assert gs.read_blob("a.json") == ("ok", {"x": 1})
    assert gs.read_blob("nope.json") == ("missing", None)
    assert gs.read_blob("bad.json")[0] == "unreadable"
    assert gs.read_blob("big.json")[0] == "unreadable"                 # raw-hämtningen fallerade
    _patch(monkeypatch, code=502)
    assert gs.read_blob("a.json")[0] == "unreadable"


def test_one_gist_get_per_page_load(monkeypatch):
    calls = _patch(monkeypatch, {f"f{i}.json": {"content": "{}"} for i in range(8)})
    for i in range(8):
        gs.load_blob(f"f{i}.json", None)
    assert calls.count(gs.GIST_API_URL) == 1


def test_load_blob_remembers_unreadable_files(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _patch(monkeypatch, {"bad.json": {"content": "{trasig"}})
    assert gs.load_blob("bad.json", None) is None and "bad.json" in gs.UNREADABLE_NAMES
    _patch(monkeypatch, {"bad.json": {"content": "{}"}})
    assert gs.load_blob("bad.json", None) == {} and "bad.json" not in gs.UNREADABLE_NAMES


def test_thor_shows_unreadable_blob_as_error():
    from thor import live
    errs = live.blob_errors({"unreadable": ["berserk"], "berserk": None})
    assert errs["berserk"][0] == "Gist-filen gick inte att läsa"


def test_alert_scan_stops_on_unreadable_state(monkeypatch):
    import alert_scan
    monkeypatch.setattr(gs, "read_blob", lambda name: ("unreadable", "HTTP 502"))
    sent = []
    monkeypatch.setattr("alerts.engine.send_alert", lambda *a, **k: sent.append(a) or {"discord": True})
    saved = []
    monkeypatch.setattr(gs, "save_blob", lambda *a: saved.append(a) or True)
    monkeypatch.setattr("sys.argv", ["alert_scan.py", "--no-blindspot"])
    assert alert_scan.main() == 1 and not sent and not saved


def test_universe_member_does_not_freeze_a_failed_read(monkeypatch):
    from thor import ui as tui
    st = SimpleNamespace(session_state={})
    monkeypatch.setattr(tui.ur, "load", lambda keys, loader=None: None)
    assert tui.universe_member(st, "VOLV-B.ST") is None
    assert not st.session_state.get(tui._UNIVERSE)
