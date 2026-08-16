"""
Tests for storage.py — persistensen mot GitHub Contents API.

Bakgrunden, som testerna finns för att förhindra en upprepning av: den gamla
vägen skrev till en Gist via save_blob(), som RETURNERAR False vid fel i
stället för att kasta — och samtliga nio anropare kastade bort returvärdet.
En saknad token gav därför exakt samma tysta beteende som en lyckad sparning.

Därför handlar de flesta testerna nedan om att fel BLIR SYNLIGA.
"""
import json
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import storage


CFG = storage.Config(token="github_pat_x", repo="lidbeck70/wolf-shadow-dashboard",
                     branch="panel-data")


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


def _content(data) -> dict:
    import base64
    raw = json.dumps(data).encode()
    return {"content": base64.b64encode(raw).decode(), "sha": "abc123"}


# ── Konfigurationen ──────────────────────────────────────────────────────────
def test_missing_secrets_names_the_exact_fix():
    """Ett fel som inte säger vad man ska göra är nästan lika illa som tyst."""
    with mock.patch.object(storage, "_secrets_section", return_value=None):
        with pytest.raises(storage.StorageError) as exc:
            storage.get_config()
    msg = str(exc.value)
    assert "[github]" in msg
    assert "token" in msg and "repo" in msg and "branch" in msg


def test_partial_secrets_say_which_key_is_missing():
    with mock.patch.object(storage, "_secrets_section",
                           return_value={"token": "x", "repo": "a/b"}):
        with pytest.raises(storage.StorageError) as exc:
            storage.get_config()
    assert "branch" in str(exc.value)


def test_a_malformed_repo_is_caught_before_the_request():
    with mock.patch.object(storage, "_secrets_section",
                           return_value={"token": "x", "repo": "wolf-shadow",
                                         "branch": "main"}):
        with pytest.raises(storage.StorageError) as exc:
            storage.get_config()
    assert "ägare/namn" in str(exc.value)


def test_fine_grained_and_classic_tokens_get_different_auth_schemes():
    assert _h(CFG)["Authorization"].startswith("Bearer ")
    classic = storage.Config(token="ghp_old", repo="a/b", branch="c")
    assert _h(classic)["Authorization"].startswith("token ")


def _h(cfg):
    return storage._headers(cfg)


def test_writing_to_the_deploy_branch_is_flagged():
    """Sparar man till den deployade grenen startar Cloud om appen — vilket
    slänger osparade ändringar, alltså precis buggen det här ska laga."""
    assert storage.Config(token="t", repo="a/b",
                          branch=storage.DEPLOY_BRANCH).deploys_the_app
    assert not storage.Config(token="t", repo="a/b",
                              branch="panel-data").deploys_the_app


def test_path_is_sanitised_since_it_builds_a_url():
    assert storage.path_for("rotation") == "data/rotation.json"
    assert storage.path_for("../../etc/passwd") == "data/etcpasswd.json"
    with pytest.raises(storage.StorageError):
        storage.path_for("../")


# ── Läsning ──────────────────────────────────────────────────────────────────
def test_load_returns_the_default_when_the_file_does_not_exist_yet():
    with mock.patch.object(storage, "get_config", return_value=CFG), \
         mock.patch.object(storage.requests, "get", return_value=_Resp(404)):
        assert storage.load_json("rotation", {"grades": {}}) == {"grades": {}}


def test_load_decodes_the_file():
    payload = {"grades": {"uran": {"hatred": 5}}}
    with mock.patch.object(storage, "get_config", return_value=CFG), \
         mock.patch.object(storage.requests, "get",
                           return_value=_Resp(200, _content(payload))):
        assert storage.load_json("rotation") == payload


@pytest.mark.parametrize("code,needle", [
    (401, "ogiltig"), (403, "skrivrätt"), (422, "branchen")])
def test_read_failures_raise_with_an_actionable_message(code, needle):
    """Ett läsfel som tyst ger tom data ser ut som att allt är borta."""
    with mock.patch.object(storage, "get_config", return_value=CFG), \
         mock.patch.object(storage.requests, "get", return_value=_Resp(code)):
        with pytest.raises(storage.StorageError) as exc:
            storage.load_json("rotation", {})
    assert needle in str(exc.value)


def test_corrupt_json_names_the_file_and_the_way_out():
    bad = {"content": "eyJub3QiOiAianNvbg==", "sha": "x"}   # trasig base64/JSON
    with mock.patch.object(storage, "get_config", return_value=CFG), \
         mock.patch.object(storage.requests, "get", return_value=_Resp(200, bad)):
        with pytest.raises(storage.StorageError) as exc:
            storage.load_json("rotation", {})
    assert "data/rotation.json" in str(exc.value)


def test_a_network_error_is_not_mistaken_for_an_empty_file():
    with mock.patch.object(storage, "get_config", return_value=CFG), \
         mock.patch.object(storage.requests, "get",
                           side_effect=storage.requests.RequestException("timeout")):
        with pytest.raises(storage.StorageError) as exc:
            storage.load_json("rotation", {})
    assert "Nådde inte GitHub" in str(exc.value)


# ── Skrivning ────────────────────────────────────────────────────────────────
def test_save_fetches_the_sha_first_because_the_api_requires_it():
    put = mock.Mock(return_value=_Resp(200, {"commit": {"sha": "deadbeef1234",
                                                        "html_url": "u"}}))
    with mock.patch.object(storage, "get_config", return_value=CFG), \
         mock.patch.object(storage.requests, "get",
                           return_value=_Resp(200, _content({"a": 1}))), \
         mock.patch.object(storage.requests, "put", put):
        result = storage.save_json("rotation", {"a": 2})
    body = put.call_args.kwargs["json"]
    assert body["sha"] == "abc123"           # hämtad, inte gissad
    assert body["branch"] == "panel-data"
    assert body["message"].startswith("panel: rotation ")
    assert result.short_sha == "deadbee"      # 7 tecken, som git


def test_a_new_file_is_created_without_a_sha():
    put = mock.Mock(return_value=_Resp(201, {"commit": {"sha": "n", "html_url": ""}}))
    with mock.patch.object(storage, "get_config", return_value=CFG), \
         mock.patch.object(storage.requests, "get", return_value=_Resp(404)), \
         mock.patch.object(storage.requests, "put", put):
        storage.save_json("rotation", {"a": 1})
    assert "sha" not in put.call_args.kwargs["json"]


def test_the_saved_body_is_readable_json_not_a_blob():
    """Filen ska gå att läsa i GitHubs diff — det är halva poängen."""
    import base64
    put = mock.Mock(return_value=_Resp(200, {"commit": {"sha": "s", "html_url": ""}}))
    with mock.patch.object(storage, "get_config", return_value=CFG), \
         mock.patch.object(storage.requests, "get", return_value=_Resp(404)), \
         mock.patch.object(storage.requests, "put", put):
        storage.save_json("rotation", {"b": 1, "a": {"x": "å"}})
    body = base64.b64decode(put.call_args.kwargs["json"]["content"]).decode()
    assert body.startswith("{\n")            # indenterad
    assert "å" in body                       # inte å
    assert body.index('"a"') < body.index('"b"')      # sorterad -> små diffar


@pytest.mark.parametrize("code,needle", [
    (401, "ogiltig"), (403, "Contents: Read and write"),
    (404, "fine-grained"), (409, "ändrades av någon annan")])
def test_every_write_failure_raises_rather_than_returning_false(code, needle):
    """Kärnan i buggen: save_blob returnerade False och ingen tittade."""
    with mock.patch.object(storage, "get_config", return_value=CFG), \
         mock.patch.object(storage.requests, "get", return_value=_Resp(404)), \
         mock.patch.object(storage.requests, "put", return_value=_Resp(code)):
        with pytest.raises(storage.StorageError) as exc:
            storage.save_json("rotation", {})
    assert needle in str(exc.value)


def test_save_never_returns_a_falsy_success_signal():
    """Signaturen ska göra det omöjligt att ignorera ett fel av misstag.

    Den gamla save_blob returnerade bool, vilket gjorde `save(...)` till ett
    giltigt uttryck som tyst slängde felet. Nu finns ingen falsk returväg:
    antingen ett SaveResult eller ett kastat StorageError.
    """
    import typing
    hints = typing.get_type_hints(storage.save_json)
    assert hints["return"] is storage.SaveResult


# ── Sessionsmönstret ─────────────────────────────────────────────────────────
class _State(dict):
    """st.session_state räcker som dict för de här testerna."""


@pytest.fixture
def session(monkeypatch):
    state = _State()
    monkeypatch.setattr(storage.st, "session_state", state)
    return state


def test_the_file_is_read_once_per_session(session):
    load = mock.Mock(return_value={"grades": {"uran": 1}})
    with mock.patch.object(storage, "load_json", load):
        storage.session_load("rotation", {})
        storage.session_load("rotation", {})
        storage.session_load("rotation", {})
    assert load.call_count == 1


def test_a_reload_never_overwrites_unsaved_session_changes(session):
    """Det uttryckliga kravet: ALDRIG reload ovanpå osparade ändringar."""
    with mock.patch.object(storage, "load_json", return_value={"a": 1}):
        storage.session_load("rotation", {})
    session["rotation"]["a"] = 999                      # osparad ändring
    with mock.patch.object(storage, "load_json", return_value={"a": 1}) as load:
        again = storage.session_load("rotation", {})
    assert again["a"] == 999
    assert load.call_count == 0


def test_dirty_tracking(session):
    with mock.patch.object(storage, "load_json", return_value={"a": 1}):
        storage.session_load("rotation", {})
    assert not storage.is_dirty("rotation")
    session["rotation"]["a"] = 2
    assert storage.is_dirty("rotation")
    storage.mark_saved("rotation")
    assert not storage.is_dirty("rotation")


def test_an_unloaded_store_is_not_dirty(session):
    assert not storage.is_dirty("aldrig-laddad")
    assert storage.dirty_stores() == []


def test_a_load_error_is_recorded_rather_than_raised_into_the_tab(session):
    """Fliken ska rendera med tom data OCH visa felet — inte krascha."""
    with mock.patch.object(storage, "load_json",
                           side_effect=storage.StorageError("token saknas")):
        data = storage.session_load("rotation", {"grades": {}})
    assert data == {"grades": {}}
    assert "token saknas" in storage.load_error("rotation")


def test_gist_data_is_migrated_when_the_repo_file_is_missing(session):
    """Byte av lagringsplats får inte tappa det som redan matats in."""
    with mock.patch.object(storage, "load_json", return_value=None), \
         mock.patch.object(storage, "legacy_gist",
                           return_value={"grades": {"uran": {"hatred": 5}}}):
        data = storage.session_load("rotation", {"grades": {}},
                                    legacy_file="rotation_data.json")
    assert data["grades"]["uran"]["hatred"] == 5
    assert storage.meta("rotation")["migrated"] is True
    # ...och den räknas som osparad tills den skrivits till repot
    assert storage.is_dirty("rotation")


def test_the_repo_file_wins_over_the_gist(session):
    legacy = mock.Mock(return_value={"grades": {"gammalt": 1}})
    with mock.patch.object(storage, "load_json", return_value={"grades": {"nytt": 1}}), \
         mock.patch.object(storage, "legacy_gist", legacy):
        data = storage.session_load("rotation", {}, legacy_file="rotation_data.json")
    assert data == {"grades": {"nytt": 1}}
    assert legacy.call_count == 0


def test_save_session_marks_the_store_clean(session):
    with mock.patch.object(storage, "load_json", return_value={"a": 1}):
        storage.session_load("rotation", {})
    session["rotation"]["a"] = 2
    result = storage.SaveResult("rotation", "abc1234def", "url", "2026-08-16T10:00:00+00:00")
    with mock.patch.object(storage, "save_json", return_value=result):
        storage.save_session("rotation")
    assert not storage.is_dirty("rotation")
    assert storage.last_saved("rotation")["sha"] == "abc1234"


def test_the_footer_lists_dirty_and_saved_stores(session):
    # Färska objekt per anrop — annars delar lagren dict och testet mäter fel.
    with mock.patch.object(storage, "load_json",
                           side_effect=lambda *a, **k: {"a": 1}):
        storage.session_load("rotation", {})
        storage.session_load("tiggre", {})
    session["tiggre"]["a"] = 2
    assert storage.dirty_stores() == ["tiggre"]
    assert storage.saved_stores() == []


# ── Datafilerna i repot ──────────────────────────────────────────────────────
def test_every_tab_has_an_empty_structure_committed():
    """data/ ska finnas från start, annars är första laddningen ett 404-fall."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("rotation", "tiggre", "insider", "scoring", "producers",
                 "scorecard", "allocator", "swing"):
        path = os.path.join(root, "data", f"{name}.json")
        assert os.path.exists(path), f"data/{name}.json saknas"
        with open(path, encoding="utf-8") as f:
            json.load(f)


def test_the_committed_files_hold_no_computed_fields():
    """Endast inmatningar lagras — beräknat räknas vid rendering."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "data", "rotation.json"), encoding="utf-8") as f:
        rotation = json.load(f)
    assert set(rotation) == {"month", "grades", "history"}
    assert "priority" not in json.dumps(rotation)
    assert "status" not in json.dumps(rotation)
