"""F12: innehavens tickers i Gisten — krypterade när PANEL_DATA_KEY finns, klartext med varning annars.

Gisten är publik (id:t står i koden). Larmtillståndet, larmloggen, sifferuppdateringen och tradejournalen bär
innehaven och sparas därför i private_data-kuvertet när nyckeln finns. Läsarna tar båda formerna; ett kuvert som
inte går att öppna är oläsbart (baslinjen står still), aldrig "tomt". Inget test rör nätet: Gisten och panel-data är
fejkade i minnet, och en okänd adress fäller testet.
"""
import base64
import json
import logging
import pathlib
from types import SimpleNamespace

import pytest

import gist_storage as gs
import private_data as pdata

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEY = "en lång testfras"
HELD = "VOLV-B.ST"
TIGGRE = "LUG.TO"
HOLDINGS = {"momentum": [{"id": "m1", "ticker": HELD, "strategy": "Momentum", "entry_price": 200.0,
                          "shares": 50}], "cash": 12345}
TIGGRE_EVENT = {"key": "tiggre_free_ride:tiggre:p1", "kind": "tiggre_free_ride", "sheet": "tiggre",
                "ticker": TIGGRE, "title": f"🎯 Tiggre: {TIGGRE} +100 % — sälj halva",
                "body": "Kurs 12.30 mot entry 6.00 (+105 %)."}
TIGGRE_STATE = {k: TIGGRE_EVENT[k] for k in ("title", "body", "kind", "ticker")}   # så sparar sheet_alerts den


class _Resp:
    def __init__(self, code=200, body=None, text=""):
        self.status_code, self._body, self.text = code, body, text

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def net(monkeypatch, tmp_path):
    """Gisten ({fil: innehåll}) och panel-data ({sökväg: json}) i minnet. Varje PATCH hamnar i gist."""
    import requests
    monkeypatch.chdir(tmp_path)                       # save_blob skriver sin lokala reserv här, aldrig i repot
    monkeypatch.setattr(gs, "_get_github_token", lambda: "t")
    monkeypatch.setattr(gs, "UNREADABLE_NAMES", set())
    monkeypatch.setattr(pdata, "_warned", False)
    box = SimpleNamespace(gist={}, repo={}, patches=[])

    def get(url, headers=None, params=None, timeout=None, **kw):
        if url == gs.GIST_API_URL:
            return _Resp(200, {"files": {n: {"content": c} for n, c in box.gist.items()}})
        if "/contents/" in url:
            path = url.split("/contents/", 1)[1]
            if path not in box.repo:
                return _Resp(404)
            return _Resp(200, {"content": base64.b64encode(json.dumps(box.repo[path]).encode()).decode()})
        raise AssertionError(f"nätanrop i testet: {url}")

    def patch(url, headers=None, timeout=None, **kw):
        assert url == gs.GIST_API_URL
        for name, f in kw["json"]["files"].items():
            box.gist[name] = f["content"]
            box.patches.append(name)
        return _Resp(200, {})

    def blocked(*a, **k):
        raise AssertionError("nätanrop i testet")
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "patch", patch)
    monkeypatch.setattr(requests, "post", blocked)
    monkeypatch.setattr(requests, "put", blocked)
    gs._invalidate()
    yield box
    gs._invalidate()


def _key(monkeypatch, phrase):
    if phrase:
        monkeypatch.setenv(pdata.KEY_NAME, phrase)
    else:
        monkeypatch.delenv(pdata.KEY_NAME, raising=False)
        monkeypatch.setattr(pdata, "passphrase", lambda: None)     # inte heller ur en lokal secrets.toml


def _put(box, name, data, phrase=None):
    box.gist[name] = json.dumps(pdata.seal(data, phrase) if phrase else data)
    gs._invalidate()


def _opened(box, name, phrase=KEY):
    raw = json.loads(box.gist[name])
    assert pdata.is_encrypted(raw)
    return pdata.unseal(raw, phrase)


# ── Gisten: runda med nyckel, klartext utan ─────────────────────────────────
def test_round_trip_with_a_key_hides_the_holdings(net, monkeypatch):
    _key(monkeypatch, KEY)
    state = {"swing": {"light": "GRÖN", "buyable": {"ABB.ST": "rank 3"},
                       "verdicts": {HELD: {"verdict": "SÄLJ", "reason": "Ur topp 30"}}},
             "sheets": {"events": {TIGGRE_EVENT["key"]: TIGGRE_STATE}}}
    refresh = {"generated": "2026-10-28", "rows": {"holdings:m1": {"ticker": HELD, "price": 231.4}},
               "events": [TIGGRE_EVENT], "error": None}
    log = [{"time": "t", "kind": "swing_exit", "title": f"Säljsignal på innehav: {HELD} — SÄLJ", "ticker": HELD}]
    for name, data in (("alert_state.json", state), ("sheets_refresh.json", refresh), ("alert_log.json", log)):
        assert gs.save_blob(name, data) is True
        assert HELD not in net.gist[name] and TIGGRE not in net.gist[name] and pdata.SCHEME in net.gist[name]
        assert gs.read_blob(name) == ("ok", data)                 # samma dict: baslinjerna nollställs inte
        assert gs.load_blob(name, None) == data
    gs.save_blob("screens.json", {"rows": ["ABB.ST"]})               # inte privat → klartext som förut
    assert json.loads(net.gist["screens.json"]) == {"rows": ["ABB.ST"]}
    from tabs import home_today as ht                              # startsidans "Sälj och bevaka" läser samma dict
    assert ht.holdings_lines(gs.load_blob("alert_state.json", None)) == [(HELD, "SÄLJ", "Ur topp 30")]


def test_the_scrypt_key_is_derived_once_per_phrase():
    """Varje Gist-läsning av en privat fil öppnar ett kuvert; scrypt (~40 ms) räknas en gång per fras och process."""
    pdata._derived.cache_clear()
    for _ in range(3):
        pdata.unseal(pdata.seal({"a": 1}, KEY), KEY)
    assert pdata._derived.cache_info().misses == 1
    assert pdata.unseal(pdata.seal({"a": 1}, KEY + "x"), KEY + "x") == {"a": 1}
    assert pdata._derived.cache_info().misses == 2


def test_without_a_key_the_payload_is_unchanged_and_one_warning_is_logged(net, monkeypatch, caplog):
    _key(monkeypatch, None)
    caplog.set_level(logging.INFO)
    state = {"swing": {"verdicts": {HELD: {"verdict": "HÅLL", "reason": "x"}}}}
    for name, data in (("alert_state.json", state), ("alert_log.json", [{"title": HELD}]),
                       ("sheets_refresh.json", {"rows": {"holdings:m1": {"ticker": HELD}}})):
        assert gs.save_blob(name, data) is True
        assert net.gist[name] == json.dumps(data, separators=(",", ":"), ensure_ascii=False, default=str)
        assert gs.read_blob(name) == ("ok", data)
    warnings = [r for r in caplog.records if r.name == "private_data" and r.levelno == logging.WARNING]
    assert len(warnings) == 1                                      # en rad per körning, inte en per fil
    msg = warnings[0].getMessage()
    assert pdata.KEY_NAME in msg and "GitHub Actions secrets" in msg and "klartext" in msg


def test_the_old_plaintext_form_is_still_read(net, monkeypatch):
    _key(monkeypatch, KEY)
    old = {"swing": {"verdicts": {HELD: {"verdict": "SÄLJ", "reason": "x"}}}, "blindspot": {"themes": {}}}
    _put(net, "alert_state.json", old)
    assert gs.read_blob("alert_state.json") == ("ok", old)


def test_a_wrong_or_missing_key_is_unreadable_not_empty(net, monkeypatch, tmp_path):
    _put(net, "alert_state.json", {"swing": {"verdicts": {HELD: {}}}}, phrase="en annan fras")
    _key(monkeypatch, KEY)
    status, why = gs.read_blob("alert_state.json")
    assert status == "unreadable" and "passar inte" in why and KEY not in why
    assert gs.load_blob("alert_state.json", None) is None and "alert_state.json" in gs.UNREADABLE_NAMES
    _key(monkeypatch, None)
    status, why = gs.read_blob("alert_state.json")
    assert status == "unreadable" and pdata.KEY_NAME in why


# ── Tradejournalen (antal och resultat i kronor) ────────────────────────────
@pytest.fixture
def session(monkeypatch):
    import streamlit
    state: dict = {}
    monkeypatch.setattr(streamlit, "session_state", state)
    return state


def test_the_trade_journal_is_sealed_with_the_key_and_read_back(net, monkeypatch, session):
    import trade_journal as tj
    monkeypatch.setattr(tj, "_get_github_token", lambda: "t")
    trades = [{"id": "t1", "ticker": HELD, "shares": 50, "pnl_amount": 1570.0}]
    _key(monkeypatch, KEY)
    assert tj.save_journal(trades) is True
    assert HELD not in net.gist["journal_data.json"] and pdata.SCHEME in net.gist["journal_data.json"]
    session.clear()
    gs._invalidate()
    assert tj.load_journal() == trades and tj.journal_unreadable() is None
    session.clear()
    monkeypatch.setenv(pdata.KEY_NAME, "fel fras")                 # fel nyckel: läsfel, och sparandet spärras
    assert tj.load_journal() == [] and tj.journal_unreadable()
    assert tj.save_journal([]) is False and json.loads(net.gist["journal_data.json"])["data"]


def test_the_trade_journal_without_a_key_is_saved_as_before(net, monkeypatch, session):
    import trade_journal as tj
    monkeypatch.setattr(tj, "_get_github_token", lambda: "t")
    _key(monkeypatch, None)
    trades = [{"id": "t1", "ticker": HELD, "shares": 50}]
    assert tj.save_journal(trades) is True
    assert net.gist["journal_data.json"] == json.dumps(trades, indent=2, default=str)


# ── alert_scan: migreringen utan falska larm ────────────────────────────────
def _scan(monkeypatch):
    """alert_scan.main med fejkad Gist och panel-data; kanalerna fångas. (returkod, skickade texter)."""
    import alert_scan
    sent = []
    monkeypatch.setattr(alert_scan, "UNREADABLE_PATHS", set())
    monkeypatch.setattr("alerts.engine.send_alert",
                        lambda text, ch, metadata=None: sent.append(text) or {"discord": True})
    monkeypatch.setattr("ember.cache.load_ember_results", lambda: None)
    monkeypatch.setattr("contrarian_alpha.cache.load_screener_results", lambda mode=None: None)
    monkeypatch.setattr("market_risk_gate.current", lambda m, evaluator=None: None)
    monkeypatch.setattr("fiat_debasement.alerts.collect", lambda: None)
    monkeypatch.setattr("sys.argv", ["alert_scan.py", "--no-blindspot"])
    gs._invalidate()
    return alert_scan.main(), sent


def test_old_plaintext_state_migrates_to_encrypted_without_spurious_alerts(net, monkeypatch):
    _key(monkeypatch, KEY)
    net.repo["data/holdings.json"] = pdata.seal(HOLDINGS, KEY)
    _put(net, "alert_state.json", {"swing": {"light": "", "buyable": {},
                                             "verdicts": {HELD: {"verdict": "SÄLJ", "reason": "x"}}}})
    code, sent = _scan(monkeypatch)
    assert code == 0 and sent == []                               # samma SÄLJ som förut: inget larm
    assert HELD not in net.gist["alert_state.json"]
    first = _opened(net, "alert_state.json")
    assert first["swing"]["verdicts"][HELD]["verdict"] == "SÄLJ"

    code, sent = _scan(monkeypatch)                                # nästa körning läser kuvertet
    assert code == 0 and sent == [] and _opened(net, "alert_state.json") == first

    # Kontroll: utan baslinjen hade samma läge larmat — det är det krypteringen inte får orsaka.
    _put(net, "alert_state.json", {"swing": {"light": "", "buyable": {}, "verdicts": {}}}, phrase=KEY)
    code, sent = _scan(monkeypatch)
    assert code == 0 and len(sent) == 1 and HELD in sent[0]
    assert HELD not in net.gist["alert_log.json"] and _opened(net, "alert_log.json")[-1]["ticker"] == HELD


@pytest.mark.parametrize("phrase", ["fel fras", None])
def test_a_state_that_cannot_be_opened_fires_nothing_and_saves_nothing(net, monkeypatch, phrase):
    net.repo["data/holdings.json"] = HOLDINGS
    _put(net, "alert_state.json", {"swing": {"verdicts": {}}}, phrase=KEY)
    before = dict(net.gist)
    _key(monkeypatch, phrase)
    code, sent = _scan(monkeypatch)
    assert code == 1 and sent == [] and net.patches == [] and net.gist == before


def test_an_unreadable_register_keeps_the_held_baselines(net, monkeypatch):
    """Registret krypterat med en annan nyckel: innehavens verdikt och Tiggre-händelser står kvar — annars larmade
    de en gång till när nyckeln rättats."""
    _key(monkeypatch, KEY)
    net.repo["data/holdings.json"] = pdata.seal(HOLDINGS, "panelens andra fras")
    public = {"title": "🛑 Insider: X under stoppen", "body": "b", "kind": "insider_stop", "ticker": "X.ST"}
    _put(net, "alert_state.json", {"swing": {"light": "", "buyable": {},
                                             "verdicts": {HELD: {"verdict": "SÄLJ", "reason": "x"}}},
                                   "sheets": {"events": {TIGGRE_EVENT["key"]: TIGGRE_STATE,
                                                         "insider_stop:insider:i1": public}}}, phrase=KEY)
    _put(net, "sheets_refresh.json", {"generated": "g", "rows": {}, "events": [], "error": None,
                                      "holdings_unreadable": True}, phrase=KEY)
    code, sent = _scan(monkeypatch)
    assert code == 0 and sent == []
    state = _opened(net, "alert_state.json")
    assert state["swing"]["verdicts"][HELD]["verdict"] == "SÄLJ"
    assert set(state["sheets"]["events"]) == {TIGGRE_EVENT["key"]}   # den publika händelsen tog slut som vanligt

    net.repo["data/holdings.json"] = pdata.seal(HOLDINGS, KEY)       # nyckeln rättad, händelsen kvar
    _put(net, "sheets_refresh.json", {"generated": "g", "rows": {}, "events": [TIGGRE_EVENT], "error": None},
         phrase=KEY)
    code, sent = _scan(monkeypatch)
    assert code == 0 and sent == []


# ── sheets_refresh: flaggan och den publika loggen ──────────────────────────
def _refresh_main(monkeypatch, net):
    import alert_scan
    import borsdata_api
    import sheets_refresh as sr
    monkeypatch.setenv("BORSDATA_API_KEY", "x")
    monkeypatch.setattr(borsdata_api, "BorsdataAPI", lambda api_key=None: object())
    monkeypatch.setattr(sr, "load_sheets", lambda: {"holdings": {"momentum": HOLDINGS["momentum"]}, "insider": {}})
    monkeypatch.setattr(sr, "refresh", lambda api, sheets: {
        "generated": "g", "rows": {"holdings:m1": {"ticker": HELD, "price": 231.4}}, "events": [TIGGRE_EVENT],
        "error": None})
    monkeypatch.setattr(alert_scan, "UNREADABLE_PATHS", {sr.SHEET_FILES["holdings"]})
    monkeypatch.setattr("sys.argv", ["sheets_refresh.py"])
    return sr.main()


def test_sheets_refresh_seals_flags_and_keeps_holdings_out_of_the_log(net, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _key(monkeypatch, KEY)
    assert _refresh_main(monkeypatch, net) == 0
    blob = _opened(net, "sheets_refresh.json")
    assert blob["holdings_unreadable"] is True and blob["rows"]["holdings:m1"]["ticker"] == HELD
    assert HELD not in net.gist["sheets_refresh.json"] and TIGGRE not in net.gist["sheets_refresh.json"]
    assert HELD not in caplog.text and TIGGRE not in caplog.text and KEY not in caplog.text
    assert "tiggre_free_ride (rubriken dold)" in caplog.text


def test_sheets_refresh_without_a_key_logs_and_publishes_as_before(net, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _key(monkeypatch, None)
    assert _refresh_main(monkeypatch, net) == 0
    assert json.loads(net.gist["sheets_refresh.json"])["rows"]["holdings:m1"]["ticker"] == HELD
    assert TIGGRE_EVENT["title"] in caplog.text and "'holdings'" in caplog.text
    assert sum(1 for r in caplog.records if r.name == "private_data") == 1


def test_mail_subject_is_hidden_in_the_log_only_with_a_key(monkeypatch, caplog):
    from alerts.channels import email as em

    class _SMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def ehlo(self):
            pass

        def sendmail(self, *a):
            pass
    monkeypatch.setattr(em.smtplib, "SMTP", _SMTP)
    monkeypatch.setattr(em, "secret", lambda name, default="": {"SMTP_HOST": "smtp.test"}.get(name, default))
    caplog.set_level(logging.INFO)
    subject = f"Säljsignal på innehav: {HELD} — SÄLJ"
    _key(monkeypatch, KEY)
    assert em._smtp_send(subject, "b", ["a@b"], "a@b") is True
    assert HELD not in caplog.text and "[dold]" in caplog.text
    caplog.clear()
    _key(monkeypatch, None)
    assert em._smtp_send(subject, "b", ["a@b"], "a@b") is True
    assert HELD in caplog.text


# ── Panelen läser kuvertet (AppTest) ────────────────────────────────────────
@pytest.fixture
def panel_caches():
    """Panelens st.cache_data-läsare töms efter testet, så att inga testrader följer med till nästa test."""
    yield
    import refresh_ui
    from tabs import alerts as ta
    refresh_ui.load_refresh.clear()
    ta._saved_log.clear()


def _suggest_app():
    import refresh_ui
    refresh_ui.load_refresh.clear()
    refresh_ui.suggest("holdings", {"id": "m1", "price": 200.0}, "price", "price", "kurs")


def _log_app():
    from tabs import alerts as ta
    ta._saved_log.clear()
    ta._render_saved_log()


def _texts(at):
    return " ".join(m.value for m in at.markdown)


def test_panel_reads_the_sealed_blobs(net, monkeypatch, panel_caches):
    from streamlit.testing.v1 import AppTest
    _key(monkeypatch, KEY)
    _put(net, "sheets_refresh.json", {"generated": "g", "rows": {"holdings:m1": {"ticker": HELD, "price": 231.4,
                                                                                 "asof": "2026-10-28"}},
                                      "events": [], "error": None}, phrase=KEY)
    _put(net, "alert_log.json", [{"time": "2026-10-28 06:20 UTC", "kind": "swing_exit", "target": "holdings",
                                  "title": f"Säljsignal på innehav: {HELD} — SÄLJ", "body": "Ur topp 30",
                                  "ticker": HELD}], phrase=KEY)
    at = AppTest.from_function(_suggest_app, default_timeout=60).run()
    assert not at.exception, at.exception
    assert "🤖 Börsdata 2026-10-28: kurs 231.4" in _texts(at) and [b.label for b in at.button] == ["Använd"]
    at = AppTest.from_function(_log_app, default_timeout=60).run()
    assert not at.exception, at.exception
    assert f"Säljsignal på innehav: {HELD}" in _texts(at)


def test_panel_with_the_wrong_key_shows_nothing_and_does_not_crash(net, monkeypatch, panel_caches):
    from streamlit.testing.v1 import AppTest
    _put(net, "alert_log.json", [{"time": "t", "kind": "swing_exit", "title": HELD}], phrase=KEY)
    _put(net, "sheets_refresh.json", {"rows": {"holdings:m1": {"price": 231.4}}}, phrase=KEY)
    _key(monkeypatch, "fel fras")
    at = AppTest.from_function(_log_app, default_timeout=60).run()
    assert not at.exception, at.exception
    assert HELD not in _texts(at) and "Inga sparade larm än" in _texts(at)
    at = AppTest.from_function(_suggest_app, default_timeout=60).run()
    assert not at.exception and not at.button


# ── Arbetsflödena ───────────────────────────────────────────────────────────
def test_every_step_that_reads_or_writes_the_private_blobs_gets_the_key():
    wf = ROOT / ".github" / "workflows"
    line = "PANEL_DATA_KEY: ${{ secrets.PANEL_DATA_KEY }}"
    sched = (wf / "scheduled-scan.yml").read_text(encoding="utf-8")
    for step in ("python sheets_refresh.py", "python alert_scan.py"):
        block = sched[:sched.index(step)].rsplit("- name:", 1)[1]
        assert line in block, step
    wolf = (wf / "wolf-data.yml").read_text(encoding="utf-8")
    assert line in wolf[:wolf.index("python alert_scan.py")].rsplit("- name:", 1)[1]
    probe = (wf / "token-probe.yml").read_text(encoding="utf-8")
    assert line in probe[:probe.index("python gist_probe.py")].rsplit("- name:", 1)[1]
    assert "cryptography" in probe
    assert pdata.PRIVATE_BLOBS == {"alert_state.json", "alert_log.json", "sheets_refresh.json", "journal_data.json"}
