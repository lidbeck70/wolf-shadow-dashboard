"""
Tester för alert_rules.py och e-postkanalen.

Kärnregeln som testas hårdast: ÖVERGÅNGAR larmar, lägen gör det inte.
Första körningen lägger baslinjen tyst, och samma läge två körningar i rad
ger noll larm — annars tapetseras kanalen och slutar läsas.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alert_rules as ar
import swing_verdict as sv


def _regime(light="GRÖN"):
    return {"regime": light, "generated": "2026-08-20T06:00",
            "rules": ["Full positionsstorlek enligt reglerna (12–20 %)."]}


def _screener(*rows):
    return {"generated": "2026-08-20T06:00", "top": list(rows)}


def _row(ticker="ANOT", rank=5, setupA=True, nearHigh=False,
         price=10.0, dist_ma50=0.05):
    return {"ticker": ticker, "rank": rank, "setupA": setupA,
            "nearHigh": nearHigh, "price": price, "dist_ma50": dist_ma50}


def _swing(positions=None):
    return {"positions": positions or [], "market": {}}


def _themes(**labels):
    return [{"name": n, "cykel_label": v, "blindspot_score": 20.0,
             "hat_score": 70.0} for n, v in labels.items()]


# ── Baslinjen ────────────────────────────────────────────────────────────────
def test_the_first_run_lays_a_baseline_and_stays_silent():
    alerts, state = ar.swing_alerts(_regime(), _screener(_row()), _swing(),
                                    prev=None)
    assert alerts == []
    assert state["light"] == "GRÖN" and "ANOT" in state["buyable"]

    alerts, state = ar.blindspot_alerts(_themes(Uran="TIDIG"), prev=None)
    assert alerts == []
    assert state["themes"]["Uran"]["label"] == "TIDIG"


def test_an_unchanged_state_never_alerts():
    _a, state = ar.swing_alerts(_regime(), _screener(_row()), _swing(), None)
    alerts, _s = ar.swing_alerts(_regime(), _screener(_row()), _swing(), state)
    assert alerts == []
    _a, bstate = ar.blindspot_alerts(_themes(Uran="TIDIG"), None)
    alerts, _s = ar.blindspot_alerts(_themes(Uran="TIDIG"), bstate)
    assert alerts == []


# ── Swing-övergångarna ───────────────────────────────────────────────────────
def test_a_regime_shift_alerts_with_the_new_rules():
    _a, state = ar.swing_alerts(_regime("GRÖN"), _screener(), _swing(), None)
    alerts, _s = ar.swing_alerts(_regime("GUL"), _screener(), _swing(), state)
    assert len(alerts) == 1
    assert alerts[0]["kind"] == "swing_regime"
    assert "GRÖN → GUL" in alerts[0]["title"]


def test_a_new_top20_setup_candidate_alerts_once():
    _a, state = ar.swing_alerts(_regime(), _screener(), _swing(), None)
    alerts, state = ar.swing_alerts(_regime(), _screener(_row()), _swing(),
                                    state)
    assert len(alerts) == 1 and alerts[0]["kind"] == "swing_setup"
    assert "ANOT" in alerts[0]["title"] and "setup A" in alerts[0]["body"]
    # samma kandidat nästa körning: tyst
    alerts, _s = ar.swing_alerts(_regime(), _screener(_row()), _swing(), state)
    assert alerts == []


def test_rank_21_or_no_setup_is_not_a_candidate():
    _a, state = ar.swing_alerts(_regime(), _screener(), _swing(), None)
    quiet = _screener(_row(rank=25), _row(ticker="X", setupA=False,
                                          nearHigh=False))
    alerts, _s = ar.swing_alerts(_regime(), quiet, _swing(), state)
    assert alerts == []


def test_a_sell_signal_on_a_holding_alerts_once():
    held = [{"ticker": "ANOT", "entry": 10.0, "date": "2026-08-01"}]
    ok = _screener(_row(price=10.5))
    _a, state = ar.swing_alerts(_regime(), ok, _swing(held), None)
    # kursen faller under stoppen
    crashed = _screener(_row(price=8.9, setupA=False))
    alerts, state = ar.swing_alerts(_regime(), crashed, _swing(held), state)
    exits = [a for a in alerts if a["kind"] == "swing_exit"]
    assert len(exits) == 1
    assert sv.SELL in exits[0]["title"] and "stoppen" in exits[0]["body"]
    # samma dom nästa körning: tyst
    alerts, _s = ar.swing_alerts(_regime(), crashed, _swing(held), state)
    assert [a for a in alerts if a["kind"] == "swing_exit"] == []


# ── Blindspot-övergångarna ───────────────────────────────────────────────────
def test_entering_early_cycle_alerts_and_leaving_does_not():
    _a, state = ar.blindspot_alerts(_themes(Uran="MITTEN", Guld="SEN"), None)
    alerts, state = ar.blindspot_alerts(_themes(Uran="TIDIG", Guld="SEN"),
                                        state)
    assert len(alerts) == 1
    assert "Uran" in alerts[0]["title"] and "TIDIG" in alerts[0]["title"]
    assert "Triple Signal" in alerts[0]["body"]     # nästa steg pekas ut
    # ut ur TIDIG: inget larm (lämnanden är inte köplägen)
    alerts, _s = ar.blindspot_alerts(_themes(Uran="SEN", Guld="SEN"), state)
    assert alerts == []


def test_a_theme_already_early_at_baseline_never_retro_alerts():
    _a, state = ar.blindspot_alerts(_themes(Uran="TIDIG"), None)
    alerts, _s = ar.blindspot_alerts(_themes(Uran="TIDIG"), state)
    assert alerts == []


# ── Sammanvägningen och inställningarna ──────────────────────────────────────
def test_evaluate_routes_channels_from_settings():
    _a, state = ar.evaluate(_regime("GRÖN"), _screener(), _swing(),
                            _themes(Uran="MITTEN"), None, {})
    settings = {"swing": {"enabled": True, "channels": ["email"]},
                "blindspot": {"enabled": True, "channels": ["discord",
                                                            "email"]}}
    alerts, _s = ar.evaluate(_regime("RÖD"), _screener(), _swing(),
                             _themes(Uran="TIDIG"), state, settings)
    kinds = {a["kind"]: a["channels"] for a in alerts}
    assert kinds["swing_regime"] == ["email"]
    assert kinds["blindspot_early"] == ["discord", "email"]


def test_a_disabled_leg_computes_state_but_sends_nothing():
    """Avstängd = tyst, men baslinjen uppdateras ändå — annars exploderar
    återaktivering i retro-larm för allt som hänt under tystnaden."""
    _a, state = ar.evaluate(_regime("GRÖN"), _screener(), _swing(),
                            _themes(Uran="MITTEN"), None, {})
    off = {"swing": {"enabled": False}, "blindspot": {"enabled": False}}
    alerts, state = ar.evaluate(_regime("RÖD"), _screener(), _swing(),
                                _themes(Uran="TIDIG"), state, off)
    assert alerts == []
    assert state["swing"]["light"] == "RÖD"
    assert state["blindspot"]["themes"]["Uran"]["label"] == "TIDIG"


def test_missing_settings_default_to_discord_on():
    _a, state = ar.evaluate(_regime("GRÖN"), _screener(), _swing(), [], None,
                            None)
    alerts, _s = ar.evaluate(_regime("RÖD"), _screener(), _swing(), [], state,
                             None)
    assert alerts and alerts[0]["channels"] == ["discord"]


def test_format_alert_carries_title_and_body():
    text = ar.format_alert({"kind": "x", "title": "Rubrik", "body": "Kropp"})
    assert "Rubrik" in text and "Kropp" in text


# ── E-postkanalen ────────────────────────────────────────────────────────────
def test_email_without_recipient_is_a_failure_not_a_delivery(monkeypatch):
    """Den gamla attrappen returnerade True — samma mönster som sparbuggen."""
    from alerts.channels import email
    monkeypatch.delenv("EMAIL_TO", raising=False)
    assert email.send("test") is False


def test_email_without_host_fails_loudly(monkeypatch):
    from alerts.channels import email
    monkeypatch.setenv("EMAIL_TO", "till@example.com")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    assert email.send("test") is False


def test_email_sends_via_smtp(monkeypatch):
    from alerts.channels import email
    sent = {}

    class _FakeSMTP:
        def __init__(self, host, port, timeout=None):
            sent["host"], sent["port"] = host, port

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def ehlo(self):
            pass

        def starttls(self):
            sent["tls"] = True

        def login(self, user, password):
            sent["login"] = (user, password)

        def sendmail(self, from_addr, to_addrs, msg):
            sent["from"], sent["to"], sent["msg"] = from_addr, to_addrs, msg

    monkeypatch.setattr(email.smtplib, "SMTP", _FakeSMTP)
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_USER", "mig@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-lösen")
    monkeypatch.setenv("EMAIL_TO", "till@example.com, två@example.com")
    ok = email.send("Larmtext", metadata={"subject": "Testrubrik"})
    assert ok is True
    assert sent["host"] == "smtp.gmail.com" and sent["tls"]
    assert sent["login"] == ("mig@gmail.com", "app-lösen")
    assert sent["to"] == ["till@example.com", "två@example.com"]
    assert "Larmtext" in sent["msg"] or "TGFybXRleHQ" in sent["msg"]


def test_channels_read_streamlit_secrets_before_env(monkeypatch):
    """Panelens testknapp felade: kanalerna läste bara os.environ, men på
    Streamlit Cloud bor nycklarna i st.secrets. secret() ska ta secrets
    först och miljön som reserv."""
    from alerts import config

    class _Secrets(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)

    import streamlit as st
    monkeypatch.setattr(st, "secrets", _Secrets(
        {"DISCORD_WEBHOOK_URL": "https://discord/secrets-vägen"}))
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord/env-vägen")
    assert config.secret("DISCORD_WEBHOOK_URL") == "https://discord/secrets-vägen"
    # saknas i secrets → miljön
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    assert config.secret("SMTP_HOST") == "smtp.gmail.com"
    # finns ingenstans → default
    assert config.secret("FINNS_INTE", "x") == "x"


def test_discord_failures_explain_themselves(monkeypatch):
    """'Misslyckades' utan varför gjorde felsökningen till gissningslek —
    varje False-väg ska lämna en läsbar orsak i last_error."""
    from alerts.channels import discord
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    assert discord.send("test") is False
    assert "DISCORD_WEBHOOK_URL" in discord.last_error
    # satt men fel sak kopierad (inte en webhook-URL): stoppas före anropet
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.gg/inbjudan")
    assert discord.send("test") is False
    assert "webhook-URL" in discord.last_error


def test_email_failures_explain_themselves(monkeypatch):
    from alerts.channels import email
    monkeypatch.delenv("EMAIL_TO", raising=False)
    assert email.send("test") is False
    assert "EMAIL_TO" in email.last_error
    monkeypatch.setenv("EMAIL_TO", "till@example.com")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    assert email.send("test") is False
    assert "SMTP_HOST" in email.last_error


def test_theme_board_mapping_uses_real_dataclass_fields(monkeypatch):
    """Första skarpa körningen kraschade på t.name — ThemeResult har
    label/key. Mappa genom den RIKTIGA dataklassen så attributdrift
    fångas här och inte i Actions-loggen."""
    from blindspot.theme_board import ThemeResult
    import alert_scan

    themes = [ThemeResult(key="uran", label="Uran", necessity=5,
                          cykel_label="TIDIG", blindspot_score=42.0,
                          hat_score=71.0)]
    monkeypatch.setattr("blindspot.theme_board.build_theme_board",
                        lambda: themes)
    out = alert_scan._themes(skip=False)
    assert out == [{"name": "Uran", "cykel_label": "TIDIG",
                    "blindspot_score": 42.0, "hat_score": 71.0}]
    assert alert_scan._themes(skip=True) == []


# ── EMBER-benet ──────────────────────────────────────────────────────────────
def _ember(*tickers, ts="2026-08-24T08:00"):
    return {"timestamp": ts,
            "eligible": [{"ticker": t, "entry": 10.0, "stop": 9.0,
                          "t1": 13.0, "rr": 3.0, "cykel_label": "TIDIG",
                          "setup_quality": "A"} for t in tickers],
            "near_misses": []}


def test_ember_alerts_on_entering_eligible_once():
    _a, state = ar.ember_alerts(_ember("FCX"), None)
    assert _a == []                                   # baslinje tyst
    alerts, state = ar.ember_alerts(_ember("FCX", "UUUU"), state)
    assert len(alerts) == 1 and "UUUU" in alerts[0]["title"]
    assert "R:R 3.00" in alerts[0]["body"]
    alerts, _s = ar.ember_alerts(_ember("FCX", "UUUU"), state)
    assert alerts == []                               # oförändrat tyst


def test_ember_unreadable_source_preserves_baseline():
    """timestamp None = källan gick inte att läsa — baslinjen får inte
    nollas, annars 'återupptäcks' hela listan nästa lyckade läsning."""
    _a, state = ar.ember_alerts(_ember("FCX"), None)
    _a2, kept = ar.ember_alerts({"timestamp": None, "eligible": []}, state)
    assert kept == state
    alerts, _s = ar.ember_alerts(_ember("FCX"), kept)
    assert alerts == []                               # FCX larmar INTE igen


# ── Wolf- och Viking-benen ───────────────────────────────────────────────────
def _wolf(*rows):
    return {"generated": "2026-08-24T08:00",
            "rows": [{"ticker": t, "name": t, "score": s} for t, s in rows]}


def _viking(*rows):
    return {"generated": "2026-08-24T08:00",
            "rows": [{"ticker": t, "name": t, "v9": v, "eligible": e,
                      "signal": "BUY", "composite": 1.0}
                     for t, v, e in rows]}


def test_wolf_alerts_when_a_ticker_newly_crosses_the_bar():
    _a, state = ar.wolf_alerts(_wolf(("BOL", 85.0)), None)
    assert _a == []
    alerts, state = ar.wolf_alerts(_wolf(("BOL", 88.0), ("SSAB", 81.0)), state)
    assert len(alerts) == 1 and "SSAB" in alerts[0]["title"]
    assert "81/125" in alerts[0]["title"]
    # under ribban räknas inte
    alerts, _s = ar.wolf_alerts(_wolf(("BOL", 85.0), ("EQNR", 79.9)), state)
    assert alerts == []


def test_wolf_unreadable_source_preserves_baseline():
    _a, state = ar.wolf_alerts(_wolf(("BOL", 85.0)), None)
    _a2, kept = ar.wolf_alerts(None, state)
    assert kept == state


def test_viking_requires_both_nine_and_absolute_gate():
    _a, state = ar.viking_alerts(_viking(), None)
    high_not_eligible = _viking(("ANOT", 9, False))
    alerts, state = ar.viking_alerts(high_not_eligible, state)
    assert alerts == []                               # 9/9 men under EMA200
    alerts, state = ar.viking_alerts(_viking(("ANOT", 8, True)), state)
    assert len(alerts) == 1 and "8/9" in alerts[0]["title"]
    alerts, _s = ar.viking_alerts(_viking(("ANOT", 9, True)), state)
    assert alerts == []                               # redan kvalificerad


def test_evaluate_routes_the_new_legs_and_reads_thresholds():
    _a, state = ar.evaluate(_regime(), _screener(), _swing(), [], None, {})
    settings = {"wolf": {"enabled": True, "channels": ["email"],
                         "min_score": 100},
                "viking": {"enabled": False},
                "ember": {"enabled": True, "channels": ["discord"]}}
    alerts, state2 = ar.evaluate(
        _regime(), _screener(), _swing(), [], state, settings,
        ember_data=_ember("FCX"),
        wolf_data=_wolf(("BOL", 99.0), ("SSAB", 101.0)),
        viking_data=_viking(("ANOT", 9, True)))
    kinds = {a["kind"]: a for a in alerts}
    # wolf: bara SSAB över den höjda ribban 100, till email
    assert "SSAB" in kinds["wolf_score"]["title"]
    assert kinds["wolf_score"]["channels"] == ["email"]
    # viking avstängd: inget larm — men baslinjen räknad ändå
    assert "viking_nine" not in kinds
    assert "ANOT" in state2["viking"]["qualified"]
    # ember: FCX in i eligible
    assert "FCX" in kinds["ember_eligible"]["title"]


# ── 👀 Övervakningen ─────────────────────────────────────────────────────────
# Ägarens beslut: bara KÖP (in i KÖP från vad som helst) och "föll ur listan" larmar. Raderna är Thors egna,
# byggda ur testernas filer på ett fast datum (inget är gammalt).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_thor_ui as tu  # noqa: E402
from thor import candidates as tc  # noqa: E402
from thor import registry as reg  # noqa: E402

WATCH = {"items": [
    {"id": "w1", "ticker": "HEXA-B.ST", "strategy": "viking_nine", "name": "Hexagon", "added": "2026-10-20"},
    {"id": "w2", "ticker": "ERIC-B.ST", "strategy": "wolf", "added": "2026-10-21"},
    {"id": "w3", "ticker": "ABB.ST", "strategy": "momentum", "added": "2026-10-22"}]}


def _wrows(today="2026-10-15", drop=()):
    """Thors rader; drop = (ticker, strategi)-par, eller strategier, som inte står i körningen."""
    return [c for c in tc.build(tu.DATA, today=today)
            if (c.ticker, c.strategy) not in drop and c.strategy not in drop]


def _was(state, key, cls, status, in_list=True, ident=None):
    """Förra körningens läge för ett par — resten av tillståndet som det var."""
    items = dict(state["items"])
    items[key] = {"id": ident if ident is not None else items[key]["id"], "in_list": in_list, "class": cls,
                  "status": status}
    return {"items": items}


def test_watch_baseline_unchanged_and_new_pairs_are_silent():
    alerts, s0 = ar.watch_alerts(WATCH, _wrows(), None)
    assert alerts == []                                               # baslinjen tyst, även för KÖP-raderna
    assert s0["items"]["HEXA-B.ST|viking_nine"] == {"id": "w1", "in_list": True, "class": "KÖP",
                                                    "status": "GOLDEN TICKET"}
    assert s0["items"]["ABB.ST|momentum"]["class"] == "BEVAKA"
    alerts, s1 = ar.watch_alerts(WATCH, _wrows(), s0)
    assert alerts == [] and s1 == s0                                  # oförändrat: tyst
    more = {"items": WATCH["items"] + [{"id": "w4", "ticker": "MYCR.ST", "strategy": "quality"}]}
    alerts, s2 = ar.watch_alerts(more, _wrows(), s0)
    assert alerts == [] and s2["items"]["MYCR.ST|quality"]["class"] == "KÖP"   # nytillagd KÖP: baslinje, inget larm


def test_watch_alerts_when_a_watched_pair_becomes_kop():
    _a, s0 = ar.watch_alerts(WATCH, _wrows(), None)
    alerts, s1 = ar.watch_alerts(WATCH, _wrows(), _was(s0, "HEXA-B.ST|viking_nine", "BEVAKA", "READY"))
    assert len(alerts) == 1
    a = alerts[0]
    assert a["kind"] == "watch_buy" and a["title"] == "👀 HEXA-B.ST blev KÖP — Viking Nine"
    assert a["body"].startswith("Hexagon: READY → GOLDEN TICKET") and "2026-10-14" in a["body"]
    assert "Bevakad sedan 2026-10-20" in a["body"] and "pröva storlek" in a["body"]
    assert ar.format_alert(a, "https://p").endswith("?p=thor&t=HEXA-B.ST&s=viking_nine")
    assert s1["items"]["HEXA-B.ST|viking_nine"]["class"] == "KÖP"
    alerts, _s = ar.watch_alerts(WATCH, _wrows(), s1)
    assert alerts == []                                               # samma KÖP nästa körning: tyst
    # från att inte stå i listan rakt in i KÖP larmar också
    alerts, _s = ar.watch_alerts(WATCH, _wrows(), _was(s0, "ERIC-B.ST|wolf", "", "", in_list=False))
    assert [x["kind"] for x in alerts] == ["watch_buy"]
    assert "ERIC-B.ST: inte i listan → ENTRY · BULL" in alerts[0]["body"] and "stopp 70.00" in alerts[0]["body"]


def test_watch_alerts_when_a_watched_pair_falls_out_of_the_list():
    _a, s0 = ar.watch_alerts(WATCH, _wrows(), None)
    alerts, s1 = ar.watch_alerts(WATCH, _wrows(drop={("ERIC-B.ST", "wolf")}), s0)
    assert len(alerts) == 1
    a = alerts[0]
    assert a["kind"] == "watch_out" and a["title"] == "👀 ERIC-B.ST föll ur listan — Wolf"
    assert "stod som ENTRY · BULL" in a["body"] and "(2026-10-14)" in a["body"] and "👀 Övervakning" in a["body"]
    assert a["target"] == "thor" and a["strategy"] == "wolf"
    assert s1["items"]["ERIC-B.ST|wolf"] == {"id": "w2", "in_list": False, "class": "", "status": ""}
    alerts, _s = ar.watch_alerts(WATCH, _wrows(drop={("ERIC-B.ST", "wolf")}), s1)
    assert alerts == []                                               # fortfarande ute: tyst


def test_watch_other_class_changes_are_silent():
    _a, s0 = ar.watch_alerts(WATCH, _wrows(), None)
    prev = _was(_was(s0, "ABB.ST|momentum", "HÅLL", "rad 25"), "HEXA-B.ST|viking_nine", "KÖP", "GOLDEN TICKET")
    items = {"items": WATCH["items"] + [{"id": "w5", "ticker": "ERIC-B.ST", "strategy": "viking_nine"}]}
    _a, base = ar.watch_alerts(items, _wrows(), None)
    prev["items"]["ERIC-B.ST|viking_nine"] = {"id": "w5", "in_list": True, "class": "KÖP", "status": "GOLDEN TICKET"}
    alerts, s1 = ar.watch_alerts(items, _wrows(), prev)
    assert alerts == []                                               # HÅLL → BEVAKA och KÖP → BEVAKA: tysta
    assert s1["items"]["ABB.ST|momentum"]["class"] == "BEVAKA"
    assert s1["items"]["ERIC-B.ST|viking_nine"] == base["items"]["ERIC-B.ST|viking_nine"]


def test_watch_alert_flag_readded_pairs_and_items_without_id():
    _a, s0 = ar.watch_alerts(WATCH, _wrows(), None)
    quiet = {"items": [dict(WATCH["items"][0], alert=False)] + WATCH["items"][1:]}
    alerts, s1 = ar.watch_alerts(quiet, _wrows(), _was(s0, "HEXA-B.ST|viking_nine", "BEVAKA", "READY"))
    assert alerts == [] and s1["items"]["HEXA-B.ST|viking_nine"]["class"] == "KÖP"     # räknas, larmar inte
    # borttagen och tillagd igen mellan två körningar (nytt id): ett nytt par, ny baslinje
    alerts, s2 = ar.watch_alerts(WATCH, _wrows(), _was(s0, "HEXA-B.ST|viking_nine", "BEVAKA", "READY", ident="old"))
    assert alerts == [] and s2["items"]["HEXA-B.ST|viking_nine"]["id"] == "w1"
    # en handskriven rad utan id får ett nytt id vid varje läsning — den larmar ändå
    bare = {"items": [{"ticker": "hexa-b.st", "strategy": "viking_nine"}]}
    _a, b0 = ar.watch_alerts(bare, _wrows(), None)
    assert b0["items"]["HEXA-B.ST|viking_nine"]["id"] == ""
    alerts, _s = ar.watch_alerts(bare, _wrows(), _was(b0, "HEXA-B.ST|viking_nine", "BEVAKA", "READY"))
    assert [x["kind"] for x in alerts] == ["watch_buy"]


def test_watch_freezes_on_unreadable_missing_or_stale_runs():
    _a, s0 = ar.watch_alerts(WATCH, _wrows(), None)
    prev = _was(s0, "HEXA-B.ST|viking_nine", "BEVAKA", "READY")
    # körningarna (eller listan) gick inte att läsa: hela benet står still
    assert ar.watch_alerts(WATCH, None, prev) == ([], prev)
    assert ar.watch_alerts(None, _wrows(), prev) == ([], prev)
    assert ar.watch_alerts(WATCH, None, None) == ([], {"items": {}})
    # strategins källa saknas eller felar (watch_frozen): paret behåller förra läget, och larmar när den är tillbaka
    for frozen in ({"viking_nine"}, {"HEXA-B.ST|viking_nine"}):
        alerts, s1 = ar.watch_alerts(WATCH, _wrows(), prev, frozen)
        assert alerts == [] and s1["items"]["HEXA-B.ST|viking_nine"] == prev["items"]["HEXA-B.ST|viking_nine"]
        alerts, _s = ar.watch_alerts(WATCH, _wrows(), s1)
        assert [x["kind"] for x in alerts] == ["watch_buy"]
    # strategin har inga rader alls i dag (saknad körning): inget "föll ur listan"
    alerts, s2 = ar.watch_alerts(WATCH, _wrows(drop={"wolf"}), s0)
    assert alerts == [] and s2["items"]["ERIC-B.ST|wolf"] == s0["items"]["ERIC-B.ST|wolf"]
    # gamla körningar (äldre än tc.STALE_DAYS): varken KÖP eller "ur listan" larmar
    old = _wrows(today="2026-11-30", drop={("ERIC-B.ST", "wolf")})
    assert all(c.stale for c in old)
    alerts, s3 = ar.watch_alerts(WATCH, old, prev)
    assert alerts == [] and s3 == prev
    # ett nytt par i en fryst strategi läggs inte i baslinjen (det får sin baslinje när källan är tillbaka)
    _a, s4 = ar.watch_alerts(WATCH, _wrows(), {"items": {}}, {"wolf"})
    assert "ERIC-B.ST|wolf" not in s4["items"] and "HEXA-B.ST|viking_nine" in s4["items"]


SHEETS = {"rule", "durrett", "tiggre"}
# Vad valhall_scan skriver i brädans sources när allt gick att läsa (testernas bräda saknar fältet)
SOURCES = {"ca_runs": 2, "screens": True, "sheets": ["confidence", "holdings", "insider", "producers", "tiggre"]}


def _full():
    """Testernas filer med alla källor: Viking Nine, Royalty, registrets tickerkarta (som Thor bygger med) och
    brädans sources."""
    vn = {"generated": "2026-10-14", "rows": [], "markets": {"norden": {"error": None}}}
    return {**tu.DATA, "board": {**tu.DATA["board"], "sources": SOURCES}, "viking_nine": vn,
            "royalty": {"generated": "2026-10-14", "rows": []}, "sheet_map": {"BOL": "BOL.ST"}}


def test_watch_frozen_names_missing_and_failing_sources():
    keys = set(tc.BLOBS) | {"ca_quality", "ca_deep", "ember", "regime", "sheet_map"}
    assert set(ar.WATCH_SOURCES) == set(reg.STRATEGIES)               # varje strategi har sina källor
    assert all(set(v) <= keys for v in ar.WATCH_SOURCES.values())
    base = dict(tu.DATA)
    # testernas filer saknar Viking Nine, Royalty och registrets karta (arkens 'BOL' → 'BOL.ST')
    assert ar.watch_frozen(base) == {"viking_nine", "royalty"} | SHEETS
    full = _full()
    vn = full["viking_nine"]
    assert ar.watch_frozen(full) == set()
    for no_map in (None, {}):                                          # registret oläsbart: arkstrategierna står still
        assert ar.watch_frozen({**full, "sheet_map": no_map}) == SHEETS
    usa_down = {**vn, "markets": {"usa": {"error": "HTTP 500"}}}
    assert ar.watch_frozen({**full, "viking_nine": usa_down}) == {"viking_nine"}
    arc = {**full["arc"], "wolf_usa": {"rows": [], "error": "licens"}}
    assert ar.watch_frozen({**full, "arc": arc}) == {"wolf"}          # Viking-delen är frisk
    screens = {**full["screens"], "screens": {**full["screens"]["screens"], "rule": {"rows": [], "error": "nät"}}}
    assert ar.watch_frozen({**full, "screens": screens}) == {"rule", "hav"}
    assert ar.watch_frozen({**full, "regime": {}}) == {"momentum"}
    assert ar.watch_frozen({**full, "board": None}) == {"viking_nine", "wolf", "quality", "long", "rule", "durrett",
                                                        "tiggre"}
    assert ar.watch_frozen({**full, "ca_quality": {"timestamp": None, "results": []}}) == {"quality"}
    assert ar.watch_frozen({**full, "insider": {"error": "Börsdata nere"}}) == {"insider"}
    berserk = {**full["berserk"], "missing": ["bol.st"]}
    ember = {**full["ember"], "near_misses": full["ember"]["near_misses"] + [{"ticker": "NHY.OL", "error": "yf"}]}
    assert ar.watch_frozen({**full, "berserk": berserk, "ember": ember}) == {"BOL.ST|berserk", "NHY.OL|ember"}
    assert ar.watch_frozen(None) == set(ar.WATCH_SOURCES)
    # en del som föll i Thors bygge (thor.candidates.build errors) gav inga rader — ingen tom lista
    assert ar.watch_frozen(full, {"berserk": "KeyError: 'rows'"}) == {"berserk"}
    assert ar.watch_frozen(full, {"themes": "x"}) == SHEETS and ar.watch_frozen(full, {"arc": "x"}) == {"wolf", "viking"}
    assert ar.watch_frozen(full, {"screens": "x"}) == SHEETS | {"hav"}
    assert ar.watch_frozen(full, {"board": "x"}) == {"viking_nine", "wolf", "quality", "long"} | SHEETS
    assert ar.watch_frozen(full, {"okänd": "x"}) == set(ar.WATCH_SOURCES)


def test_watch_fresh_reads_each_lists_own_stamp():
    """Färsk = varje lista strategin byggs ur är högst tc.STALE_DAYS gammal enligt sin egen stämpel, och felfri."""
    full = _full()
    fresh = ar.watch_fresh(full, today="2026-10-15")
    assert set(fresh) == set(ar.WATCH_SOURCES) and fresh["berserk"] == "2026-10-14" and fresh["ember"] == "2026-10-14"
    assert ar.watch_fresh(full, today="2026-10-18") == fresh                       # fyra dagar: fortfarande färsk
    assert ar.watch_fresh(full, today="2026-10-19") == {}                          # fem: gammal, som Thors GAMMAL
    # utan användbar stämpel räknas listan inte som färsk (då gäller raderna, som förut)
    no_stamp = {**full, "berserk": {"rows": []}, "insider": {**full["insider"], "generated": "trasig"}}
    assert {"berserk", "insider"}.isdisjoint(ar.watch_fresh(no_stamp, today="2026-10-15"))
    # varje del räknas: Wolfs USA-del gammal → Wolf inte färsk; Viking Nines marknad med eget datum
    arc = {**full["arc"], "wolf_usa": {"generated": "2026-10-01", "rows": []}}
    got = ar.watch_fresh({**full, "arc": arc}, today="2026-10-15")
    assert "wolf" not in got and "viking" in got
    vn = {**full["viking_nine"], "markets": {"norden": {"generated": "2026-10-14"}, "usa": {"generated": "2026-10-02"}}}
    assert "viking_nine" not in ar.watch_fresh({**full, "viking_nine": vn}, today="2026-10-15")
    # en källa som saknas eller felar är aldrig färsk (den står i watch_frozen)
    assert not {"momentum"} & set(ar.watch_fresh({**full, "regime": None}, today="2026-10-15"))
    assert not SHEETS & set(ar.watch_fresh({**full, "sheet_map": None}, today="2026-10-15"))
    assert ar.watch_fresh(None) == {}


def test_watch_a_fresh_but_empty_list_means_out_of_the_list():
    """BERSERK bär bara KÖP/BEVAKA: en färsk körning utan BOL.ST — även helt tom — är 'föll ur listan'. Bara KÖP och
    'föll ur listan' larmar (ägarens beslut); en gammal, ostämplad eller i bygget fallerad lista står still."""
    import copy
    item = {"items": [{"id": "b1", "ticker": "BOL.ST", "strategy": "berserk", "added": "2026-10-01"}]}
    blobs = _full()
    _a, s0 = ar.watch_alerts(item, tc.build(blobs, today="2026-10-15"), None)
    assert s0["items"]["BOL.ST|berserk"]["class"] == "KÖP"
    empty = {**blobs, "berserk": {"generated": "2026-10-16T21:40:00+00:00", "rows": []}}
    rows = tc.build(empty, today="2026-10-17")
    assert not [c for c in rows if c.strategy == "berserk"]
    # utan källornas besked (bara raderna): en strategi utan rader står still, som förut
    assert ar.watch_alerts(item, rows, s0) == ([], s0)
    # med listans egen stämpel: färsk och felfri → ur listan
    alerts, s1 = ar.watch_alerts(item, rows, s0, ar.watch_frozen(empty), ar.watch_fresh(empty, today="2026-10-17"))
    assert [a["kind"] for a in alerts] == ["watch_out"]
    assert "stod som KÖP · S2 Cykelvändning" in alerts[0]["body"] and "(2026-10-16)" in alerts[0]["body"]
    assert s1["items"]["BOL.ST|berserk"]["in_list"] is False
    # gammal tom lista, tom lista utan stämpel, eller en del som föll i bygget: ingen övergång
    for blob, today, errors in (({"generated": "2026-10-09", "rows": []}, "2026-10-17", None),
                                ({"rows": []}, "2026-10-17", None),
                                (empty["berserk"], "2026-10-17", {"berserk": "KeyError: 'rows'"})):
        b = {**blobs, "berserk": blob}
        r = tc.build(b, today=today)
        assert ar.watch_alerts(item, r, s0, ar.watch_frozen(b, errors), ar.watch_fresh(b, today=today)) == ([], s0)
    # tillbaka som BEVAKA: tyst (bara KÖP larmar); sedan KÖP: larm
    bev = copy.deepcopy(blobs)
    bev["berserk"]["rows"][0]["status"] = "BEVAKA"
    alerts, s2 = ar.watch_alerts(item, tc.build(bev, today="2026-10-15"), s1, ar.watch_frozen(bev),
                                 ar.watch_fresh(bev, today="2026-10-15"))
    assert alerts == [] and s2["items"]["BOL.ST|berserk"]["class"] == "BEVAKA"
    alerts, _s = ar.watch_alerts(item, tc.build(blobs, today="2026-10-15"), s2, ar.watch_frozen(blobs),
                                 ar.watch_fresh(blobs, today="2026-10-15"))
    assert [a["kind"] for a in alerts] == ["watch_buy"]


def _run(items, blobs, today, prev, errors=None):
    """En körning av benet som larmjobbet gör den: Thors rader, watch_frozen och watch_fresh ur samma filer."""
    rows = tc.build(blobs, today=today, errors=errors, sheet_map=blobs.get("sheet_map"))
    return ar.watch_alerts(items, rows, prev, ar.watch_frozen(blobs, errors), ar.watch_fresh(blobs, today=today))


def test_watch_a_long_screener_that_fell_is_no_empty_list(monkeypatch):
    """valhall_scan sparar long = None när Long Screenern kastar — utan felfält, och brädans stämpel är färsk. Det är
    ingen tom lista: inget 'föll ur listan' den dagen och inget 'blev KÖP' när screenern går igen."""
    import copy

    import alert_scan
    item = {"items": [{"id": "l1", "ticker": "LAGR-B.ST", "strategy": "long", "added": "2026-10-01"}]}
    blobs = _full()
    _a, s0 = _run(item, blobs, "2026-10-15", None)
    assert s0["items"]["LAGR-B.ST|long"]["class"] == "KÖP"
    fell = copy.deepcopy(blobs)
    fell["board"].update(generated="2026-10-16T16:50:00+00:00", long=None)
    errors: dict = {}
    assert not [c for c in tc.build(fell, today="2026-10-17", errors=errors) if c.strategy == "long"]
    assert errors == {}                                               # bygget ser inget fel
    assert "long" in ar.watch_frozen(fell) and "long" not in ar.watch_fresh(fell, today="2026-10-17")
    assert _run(item, fell, "2026-10-17", s0) == ([], s0)
    # larmjobbets väg med samma bräda
    _no_net(monkeypatch)
    _r, frozen, fresh = alert_scan._watch_rows(item, _register_loader(fell["board"]), tu.DATA["regime"],
                                               today="2026-10-17")
    assert "long" in frozen and "long" not in fresh
    # nästa dag går screenern igen: samma KÖP — tyst
    back = copy.deepcopy(blobs)
    back["board"]["generated"] = "2026-10-17T16:50:00+00:00"
    assert _run(item, back, "2026-10-18", s0) == ([], s0)
    # en läst och färsk Long Screener utan aktien är däremot 'föll ur listan' — också när den är tom
    back["board"]["long"] = {"segments": {}}
    alerts, _s = _run(item, back, "2026-10-18", s0)
    assert [a["kind"] for a in alerts] == ["watch_out"] and "(2026-10-17)" in alerts[0]["body"]


def _sheets_unread(board):
    """Brädan dagen efter, när valhall_scan inte kunde läsa arken (load_sheets kastade → {}): arkens val saknas och
    syns bara som en lucka i sources.sheets — inget felfält. Rule-håven och Quality Contrarian finns kvar."""
    import copy
    b = copy.deepcopy(board)
    b["generated"] = "2026-10-16T16:50:00+00:00"
    sheet = ("Rick Rule-arket", "Durrett", "Tiggre")
    for th in b["buys"]["themes"].values():
        for part in ("quality", "leverage"):
            th[part]["picks"] = [p for p in th[part]["picks"] if p.get("source") not in sheet]
    b["buys"]["unknown_commodity"] = []
    b["sources"] = {**b["sources"], "sheets": []}
    return b


def test_watch_sheets_that_were_not_read_are_no_empty_list():
    """Ett ark som valhall_scan inte kunde läsa syns bara som en lucka i brädans sources. Rick Rules, Durretts och
    Tiggres par på brädans rader står då still — även när strategin har andra rader (håvarna, Rule-håven) — så inget
    'föll ur listan' i dag och inget 'blev KÖP' när arket är tillbaka. Håvarnas TRÄFF-par larmar som vanligt."""
    import copy
    items = {"items": [{"id": "d1", "ticker": "MMY.V", "strategy": "durrett", "added": "2026-10-01"},
                       {"id": "t1", "ticker": "CU.V", "strategy": "tiggre", "added": "2026-10-01"},
                       {"id": "t2", "ticker": "UNK.V", "strategy": "tiggre", "added": "2026-10-01"},
                       {"id": "t3", "ticker": "SCR.V", "strategy": "tiggre", "added": "2026-10-01"},
                       {"id": "r1", "ticker": "NEM", "strategy": "rule", "added": "2026-10-01"}]}
    blobs = copy.deepcopy(_full())
    blobs["screens"]["screens"]["tiggre"]["rows"].append({"ticker": "SCR.V", "name": "Screen", "universe": "CA"})
    _a, s0 = _run(items, blobs, "2026-10-15", None)
    assert {k: (v["class"], v["in_list"]) for k, v in s0["items"].items()} == {
        "MMY.V|durrett": ("KÖP", True), "CU.V|tiggre": ("KÖP", True), "UNK.V|tiggre": ("BEVAKA", True),
        "SCR.V|tiggre": ("TRÄFF", True), "NEM|rule": ("KÖP", True)}
    # dag 2: arken gick inte att läsa; håvarna körde i dag utan SCR.V
    day2 = {**blobs, "board": _sheets_unread(blobs["board"]), "screens": copy.deepcopy(blobs["screens"])}
    day2["screens"]["generated"] = "2026-10-16T16:50:00+00:00"
    day2["screens"]["screens"]["tiggre"]["rows"] = [r for r in day2["screens"]["screens"]["tiggre"]["rows"]
                                                    if r["ticker"] != "SCR.V"]
    rows = tc.build(day2, today="2026-10-17", sheet_map=day2["sheet_map"])
    assert {(c.ticker, c.status_class) for c in rows if c.strategy == "tiggre"} == {("CU.V", "TRÄFF")}
    assert {c.ticker for c in rows if c.strategy == "rule"} == {"AEM"}           # Rule-håven finns kvar
    assert not [c for c in rows if c.strategy == "durrett"]
    gaps = {ar._board_gap(s) for s in SHEETS}
    assert gaps <= ar.watch_frozen(day2) and not SHEETS & set(ar.watch_fresh(day2, today="2026-10-17"))
    alerts, s1 = _run(items, day2, "2026-10-17", s0)
    assert [(a["kind"], a["ticker"]) for a in alerts] == [("watch_out", "SCR.V")]   # bara håvens eget par
    assert {k: v for k, v in s1["items"].items() if k != "SCR.V|tiggre"} == \
        {k: v for k, v in s0["items"].items() if k != "SCR.V|tiggre"}
    # utan luckan i frozen hade arkets par fallit ur (NEM, UNK.V) och CU.V blivit TRÄFF — det som larmade fel
    no_gap = ar.watch_alerts(items, rows, s0, ar.watch_frozen(day2) - gaps, ar.watch_fresh(day2, today="2026-10-17"))
    assert {a["ticker"] for a in no_gap[0] if a["kind"] == "watch_out"} == {"NEM", "UNK.V", "SCR.V"}
    # dag 3: arken är tillbaka — inget 'blev KÖP' för det som stod still
    day3 = {**day2, "board": {**blobs["board"], "generated": "2026-10-17T16:50:00+00:00"}}
    alerts, s2 = _run(items, day3, "2026-10-18", s1)
    assert alerts == [] and s2 == s1


def test_watch_a_read_but_empty_sheet_list_means_out_and_partial_gaps():
    """Ett ark som lästes men inte längre har aktien är 'föll ur listan' — även när strategins lista är tom. Luckan
    gäller bara strategin vars del saknas; en bräda utan sources går inte att pröva (färsk nej, fryser inget)."""
    import copy
    item = {"items": [{"id": "d1", "ticker": "MMY.V", "strategy": "durrett", "added": "2026-10-01"}]}
    blobs = _full()
    _a, s0 = _run(item, blobs, "2026-10-15", None)
    empty = copy.deepcopy(blobs)
    empty["board"]["generated"] = "2026-10-16T16:50:00+00:00"
    empty["board"]["buys"]["themes"]["guld"]["leverage"]["picks"] = []        # Durrett-arket läst, utan bolag i dag
    assert "durrett" in ar.watch_fresh(empty, today="2026-10-17") and not ar.watch_frozen(empty)
    alerts, s1 = _run(item, empty, "2026-10-17", s0)
    assert [(a["kind"], a["title"]) for a in alerts] == [("watch_out", "👀 MMY.V föll ur listan — Durrett")]
    assert s1["items"]["MMY.V|durrett"]["in_list"] is False
    # varje strategi har sin egen del: Tiggre-arket, och för Rick Rule även håvarna och båda Quality Contrarian
    src = blobs["board"]["sources"]
    for sources, gap in (({**src, "sheets": ["confidence", "producers"]}, {"tiggre"}),
                         ({**src, "sheets": ["producers", "tiggre"]}, {"durrett"}),
                         ({**src, "ca_runs": 1}, {"rule"}), ({**src, "screens": False}, {"rule"}),
                         ({"sheets": "trasig"}, SHEETS)):
        b = {**blobs, "board": {**blobs["board"], "sources": sources}}
        assert ar.watch_frozen(b) == {ar._board_gap(s) for s in gap}, sources
        assert set(ar.watch_fresh(b, today="2026-10-15")) == set(ar.WATCH_SOURCES) - gap
    # en bräda utan sources: ingen lucka att se — inget fryses, men brädan gör inte arkens listor färska
    legacy = {**blobs, "board": {k: v for k, v in blobs["board"].items() if k != "sources"}}
    assert ar.watch_frozen(legacy) == set() and not SHEETS & set(ar.watch_fresh(legacy, today="2026-10-15"))


def test_evaluate_routes_the_watch_leg():
    rows = _wrows()
    _a, s0 = ar.evaluate({}, {}, {}, [], None, {}, watch_data=WATCH, watch_rows=rows)
    assert _a == [] and "HEXA-B.ST|viking_nine" in s0["watch"]["items"]
    s0["watch"] = _was(s0["watch"], "ERIC-B.ST|wolf", "HÅLL", "NEUTRAL")
    alerts, s1 = ar.evaluate({}, {}, {}, [], s0, None, watch_data=WATCH, watch_rows=rows)
    assert [(a["kind"], a["channels"]) for a in alerts if a["kind"].startswith("watch")] == \
        [("watch_buy", ["discord"])]
    off, s2 = ar.evaluate({}, {}, {}, [], s0, {"watch": {"enabled": False}}, watch_data=WATCH, watch_rows=rows)
    assert not [a for a in off if a["kind"].startswith("watch")] and s2["watch"] == s1["watch"]   # avstängt: räknas
    frozen, s3 = ar.evaluate({}, {}, {}, [], s0, None, watch_data=WATCH, watch_rows=rows, frozen_watch={"wolf"})
    assert not [a for a in frozen if a["kind"].startswith("watch")]
    assert s3["watch"]["items"]["ERIC-B.ST|wolf"]["class"] == "HÅLL"
    _a, s4 = ar.evaluate({}, {}, {}, [], s0, None)                    # ingen lista inläst: benet fryser
    assert s4["watch"] == s0["watch"]


# ── alert_scan: listan och raderna ───────────────────────────────────────────
def _gist(name, fallback=None):
    names = {v: k for k, v in tc.BLOBS.items()}
    return tu.DATA.get(names.get(name), fallback) if name in names else fallback


def _no_net(monkeypatch):
    """Inget nätverk: CA- och EMBER-läsarna ger testernas filer; ett anrop som ändå går ut stoppas och hamnar i
    listan som returneras (testet kräver att den är tom)."""
    tried = []

    def _offline(url, *_a, **_k):
        tried.append(url)
        raise OSError("inget nätverk i testet")
    monkeypatch.setattr("requests.get", _offline)
    monkeypatch.setattr("contrarian_alpha.cache.load_screener_results",
                        lambda mode=None: tu.DATA["ca_quality" if mode == "quality" else "ca_deep"])
    monkeypatch.setattr("ember.cache.load_ember_results", lambda: tu.DATA["ember"])
    return tried


def test_alert_scan_builds_watch_rows_only_when_something_is_watched(monkeypatch):
    import alert_scan
    tried = _no_net(monkeypatch)
    calls = []

    def loader(name, fallback=None):
        calls.append(name)
        return _gist(name, fallback)
    assert alert_scan._watch_rows(None, loader, {}) == (None, set(), {})         # listan oläsbar: benet fryser
    assert alert_scan._watch_rows({"items": []}, loader, {}) == ([], set(), {})  # inget bevakas: ingen läsning
    assert calls == []
    rows, frozen, fresh = alert_scan._watch_rows(WATCH, loader, tu.DATA["regime"], today="2026-10-15")
    got = {(c.ticker, c.strategy, c.status_class) for c in rows}
    assert got >= {("ERIC-B.ST", "wolf", "KÖP"), ("ABB.ST", "momentum", "BEVAKA")}
    assert not [c for c in rows if c.stale]                           # samma dag för raderna och listornas stämplar
    # testernas Gist saknar Viking Nine, Royalty och registret (arkstrategierna står still utan Thors tickerkarta)
    assert frozen == {"viking_nine", "royalty"} | SHEETS and "valhall_board.json" in calls
    assert fresh["wolf"] == "2026-10-14" and not ({"viking_nine", "royalty"} | SHEETS) & set(fresh)
    assert alert_scan._watch_rows(WATCH, lambda n, f=None: None, {}) == (None, set(), {})   # Gisten oläsbar
    assert tried == []


def _register_loader(board, register=True):
    """Gisten med en annan karta och (valfritt) universum-registret: BOL i Stockholm (Börsdatas form 'BOL')."""
    import markets as mk
    from universe import registry as ur
    saved: dict = {}
    if register:
        nordic = [{"insId": 7, "ticker": "BOL", "name": "Boliden", "marketId": 1, "sectorId": 3,
                   "stockPriceCurrency": "SEK"}]
        ur.save(ur.build(table=mk.FALLBACK, nordic=nordic, global_=[], include_lists=False),
                lambda name, data: saved.setdefault(name, data) is not None)

    def loader(name, fallback=None):
        if name == "valhall_board.json":
            return board
        return saved[name] if name in saved else _gist(name, fallback)
    return loader


def test_alert_scan_maps_sheet_tickers_like_thor(monkeypatch):
    """Arken sparar tickern som den skrevs ('BOL'). Thor gör den till 'BOL.ST' med registrets karta — larmbenet
    bygger med samma karta; går registret inte att läsa står Rick Rule, Durrett och Tiggre still i stället för att
    larma 'föll ur listan' (och nästa dag 'blev KÖP') för en bevakad BOL.ST."""
    import copy

    import alert_scan
    from thor import ui as tui
    tried = _no_net(monkeypatch)
    board = copy.deepcopy(_full()["board"])
    board["buys"]["themes"]["guld"]["quality"]["picks"].append(
        {"ticker": "BOL", "name": "Boliden", "source": "Rick Rule-arket", "score": 4, "verdict": "Köpkandidat"})
    item = {"items": [{"id": "r1", "ticker": "BOL.ST", "strategy": "rule", "added": "2026-10-01"}]}
    loader = _register_loader(board)
    rows, frozen, fresh = alert_scan._watch_rows(item, loader, tu.DATA["regime"], today="2026-10-15")
    thor_rows = tui.rebuild(tui.load(loader))["rows"]                 # Thors egen väg (📡 Uppdatera)

    def sheet(rs):
        return sorted((c.ticker, c.strategy, c.status_class) for c in rs if c.strategy in SHEETS)
    assert sheet(rows) == sheet(thor_rows) and ("BOL.ST", "rule") in {(t, s) for t, s, _c in sheet(rows)}
    assert not SHEETS & frozen and "rule" in fresh
    _a, s0 = ar.watch_alerts(item, rows, None, frozen, fresh)
    assert s0["items"]["BOL.ST|rule"]["in_list"] is True
    # samma dag utan registret: arkets bara 'BOL' blir ingen 'BOL.ST'-rad — benet står still, inget larm
    bare_rows, bare_frozen, bare_fresh = alert_scan._watch_rows(item, _register_loader(board, register=False),
                                                                tu.DATA["regime"], today="2026-10-15")
    assert ("BOL", "rule") in {(c.ticker, c.strategy) for c in bare_rows} and SHEETS <= bare_frozen
    assert ar.watch_alerts(item, bare_rows, s0, bare_frozen, bare_fresh) == ([], s0)
    # ett trasigt register fäller inte benet
    monkeypatch.setattr("universe.registry.load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nere")))
    rows2, frozen2, _f = alert_scan._watch_rows(item, loader, tu.DATA["regime"], today="2026-10-15")
    assert rows2 is not None and SHEETS <= frozen2
    assert tried == []


def test_alert_scan_sends_the_watch_alert_and_saves_the_state(monkeypatch):
    import alert_scan
    import gist_storage as gs
    tried = _no_net(monkeypatch)
    reads = []                                                        # CA och EMBER läses en gång, inte om för benet

    def _ca(mode=None):
        reads.append(mode)
        return tu.DATA["ca_quality" if mode == "quality" else "ca_deep"]
    monkeypatch.setattr("contrarian_alpha.cache.load_screener_results", _ca)
    monkeypatch.setattr("ember.cache.load_ember_results", lambda: reads.append("ember") or tu.DATA["ember"])
    build = tc.build
    monkeypatch.setattr(tc, "build", lambda blobs, **k: build(blobs, **{**k, "today": "2026-10-15"}))
    prev = {"watch": {"items": {"ERIC-B.ST|wolf": {"id": "w2", "in_list": True, "class": "BEVAKA",
                                                   "status": "MODERATE"}}}}
    monkeypatch.setattr(gs, "read_blob",
                        lambda name: ("ok", prev) if name == alert_scan.STATE_FILE else ("missing", None))
    monkeypatch.setattr(gs, "load_blob", _gist)
    monkeypatch.setattr(gs, "load_wolf_json", lambda name: {"wolf_regime.json": tu.DATA["regime"],
                                                            "wolf_screener.json": tu.DATA["momentum"]}.get(name))
    saved = {}
    monkeypatch.setattr(gs, "save_blob", lambda name, data: saved.update({name: data}) or True)
    files = {alert_scan.WATCH_PATH: WATCH}
    monkeypatch.setattr(alert_scan, "_repo_file", lambda path, default: files.get(path, default))
    monkeypatch.setattr("market_risk_gate.current", lambda m, evaluator=None: None)
    monkeypatch.setattr("fiat_debasement.alerts.collect", lambda: None)
    sent = []
    monkeypatch.setattr("alerts.engine.send_alert",
                        lambda text, ch, metadata=None: sent.append(text) or {"discord": True})
    monkeypatch.setenv("PANEL_URL", "https://panel")
    monkeypatch.setattr("sys.argv", ["alert_scan.py", "--no-blindspot"])
    assert alert_scan.main() == 0
    assert len(sent) == 1 and "👀 ERIC-B.ST blev KÖP — Wolf" in sent[0]          # övriga ben lägger baslinje
    assert sent[0].endswith("→ https://panel/?p=thor&t=ERIC-B.ST&s=wolf")
    watch = saved[alert_scan.STATE_FILE]["watch"]["items"]
    assert watch["ERIC-B.ST|wolf"]["class"] == "KÖP" and "HEXA-B.ST|viking_nine" not in watch   # Viking Nine fryst
    assert sorted(reads) == ["deep_contrarian", "ember", "quality"] and tried == []


def test_alert_scan_and_settings_wire_the_watch_leg():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scan = open(os.path.join(root, "alert_scan.py"), encoding="utf-8").read()
    assert 'WATCH_PATH = "data/watchlist.json"' in scan and "watch_data=watch_data" in scan
    assert "frozen_watch=frozen_watch" in scan
    tab = open(os.path.join(root, "tabs", "alerts.py"), encoding="utf-8").read()
    assert '"watch": {"enabled": True, "channels": ["discord"]}' in tab and '("watch", "👀 Övervakning",' in tab
    assert ar.KIND_TARGET["watch_buy"][1] == ar.KIND_TARGET["watch_out"][1] == "thor"


# ── Inställningarna (MER → Larm) och texterna ────────────────────────────────
def _settings_page():
    from tabs.alerts import _render_scheduled_settings
    _render_scheduled_settings(None)


def test_the_alert_settings_have_the_watch_leg_on_with_discord(monkeypatch):
    """En sparad alerts.json från före benet får benet påslaget med Discord; att stänga av det hamnar i filen."""
    from streamlit.testing.v1 import AppTest
    import test_watch_ui as twu
    twu.fake_repo(monkeypatch, {"alerts": {"swing": {"enabled": True, "channels": ["discord"]}}})
    at = AppTest.from_function(_settings_page, default_timeout=60).run()
    assert not at.exception
    tog = at.toggle(key="sched_watch_on")
    assert tog.label == "👀 Övervakning" and tog.value is True
    assert at.multiselect(key="sched_watch_ch").value == ["discord"]
    assert any("BLIR KÖP" in c.value and "faller ur strategins lista" in c.value
               and "Long Screener som föll" in c.value and "ark som inte gick att läsa" in c.value for c in at.caption)
    tog.set_value(False).run()
    assert at.session_state["alerts"]["watch"] == {"enabled": False, "channels": ["discord"]}


def test_the_watch_texts_name_the_two_alerts(monkeypatch):
    from ovtlyr.ui.rules_page import _PANEL_GUIDE
    from ui import nav
    import test_watch_ui as twu
    cards = dict(nav.HOME_ZONES[[k for k, _t, _c in nav.HOME_ZONES].index("portfolio")][2])
    assert "larm när en blir KÖP" in cards["👀 Övervakning"]
    row = next(r for r in _PANEL_GUIDE if r[0] == "PORTFÖLJ → 👀 Övervakning")
    assert "BLIR KÖP eller faller ur listan" in row[2] and "MER → Larm" in row[2]
    assert "Long Screener som föll" in row[2] and "ark som inte gick att läsa" in row[2]
    at, _repo = twu._app(monkeypatch, twu.ITEMS)                     # sidan säger samma sak under tabellen
    at.run()
    assert "Larm på Discord när en bevakad aktie BLIR KÖP eller faller ur listan" in \
        " ".join(m.value for m in at.markdown)
