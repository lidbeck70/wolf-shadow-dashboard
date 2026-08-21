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
