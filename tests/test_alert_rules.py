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
