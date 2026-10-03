"""
🐺 Fiat Debasement PR 6 — larmbenet "fiat", FIAT ENVIRONMENT på gruvsidorna
och REAL COMMODITY PRICES. Syntetiska serier — inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alert_rules as ar  # noqa: E402
from fiat_debasement import config as cfg  # noqa: E402
from fiat_debasement import environment as env  # noqa: E402
from fiat_debasement import snapshot as fs  # noqa: E402
from tests.test_fiat_ui import ROOT, _app, fake_data  # noqa: E402,F401


def _data(m2_usd=5.0, gs=70.0, gold_sek=-5.0):
    m = {c: {"m2_yoy": 3.0, "cpi_yoy": 2.0, "monetary_gap": 1.0, "gold_1y": 0.0} for c in cfg.CURRENCIES}
    m["USD"]["m2_yoy"] = m2_usd
    m["SEK"]["gold_1y"] = gold_sek
    return {"metrics": m, "gs_ratio": gs, "dates": {"USD": {"m2_yoy": "2026-08-01"}, "gs_ratio": "2026-10-02"}}


def test_first_run_only_sets_the_baseline():
    alerts, state = ar.fiat_alerts(_data(m2_usd=9.0), None)
    assert alerts == [] and state["on"]["m2_high|USD"] is True


def test_only_transitions_alert():
    _a, s0 = ar.fiat_alerts(_data(), None)
    a1, s1 = ar.fiat_alerts(_data(m2_usd=7.5, gs=82.0, gold_sek=-12.0), s0)
    kinds = sorted(a["kind"] for a in a1)
    assert kinds == ["fiat_gold_fall", "fiat_gs_high", "fiat_m2_high"]
    m2 = next(a for a in a1 if a["kind"] == "fiat_m2_high")
    assert "USD" in m2["title"] and "+7.5" in m2["body"] and "2026-08-01" in m2["body"]
    assert all("ingen köp- eller säljsignal" in a["body"] for a in a1)
    a2, s2 = ar.fiat_alerts(_data(m2_usd=8.0, gs=85.0, gold_sek=-15.0), s1)
    assert a2 == []                                                     # läget upprepas inte
    _a3, s3 = ar.fiat_alerts(_data(m2_usd=6.0, gs=75.0, gold_sek=-2.0), s2)
    a4, _s4 = ar.fiat_alerts(_data(m2_usd=7.2), s3)
    assert [a["kind"] for a in a4] == ["fiat_m2_high"]                  # ny övergång efter att ha lämnat


def test_rules_can_be_switched_off_and_unknown_values_keep_state():
    _a, s0 = ar.fiat_alerts(_data(), None)
    a1, s1 = ar.fiat_alerts(_data(m2_usd=9.0, gs=45.0), s0, rules=["gs_low"])
    assert [a["kind"] for a in a1] == ["fiat_gs_low"] and s1["on"]["m2_high|USD"] is True
    unknown = _data()
    unknown["metrics"]["USD"]["m2_yoy"] = None
    _a, s2 = ar.fiat_alerts(unknown, s1)
    assert s2["on"]["m2_high|USD"] is True                              # okänt värde → gamla läget
    alerts, state = ar.fiat_alerts(None, s1)
    assert alerts == [] and state == s1                                 # benet fryser


def test_evaluate_routes_the_fiat_leg():
    empty = dict(regime_data={}, screener_data={}, swing_data={}, themes=[])
    _o, st0 = ar.evaluate(**empty, prev_state=None, fiat_data=_data())
    assert "fiat" in st0
    out, _st = ar.evaluate(**empty, prev_state=st0, fiat_data=_data(m2_usd=9.0))
    assert any(a["kind"] == "fiat_m2_high" and a["channels"] == ["discord"] for a in out)
    off, _st = ar.evaluate(**empty, prev_state=st0, fiat_data=_data(m2_usd=9.0), settings={"fiat": {"enabled": False}})
    assert not any(a["kind"].startswith("fiat_") for a in off)


def test_scan_and_settings_are_wired():
    scan = open(os.path.join(ROOT, "alert_scan.py"), encoding="utf-8").read()
    assert "fiat_data=fiat_data" in scan and "fiat_debasement.alerts" in scan
    tab = open(os.path.join(ROOT, "tabs", "alerts.py"), encoding="utf-8").read()
    assert '"fiat":' in tab and "sched_fiat_rules" in tab and "🐺 Fiat Debasement" in tab
    assert set(cfg.FIAT_ALERT_RULES) == set(ar.FIAT_DEFAULT_RULES)


def test_collect(fake_data):
    from fiat_debasement.alerts import collect
    data = collect()
    assert set(data["metrics"]) == set(cfg.CURRENCIES)
    assert data["metrics"]["USD"]["m2_yoy"] == pytest.approx(7.0, abs=0.05) and data["gs_ratio"] > 0
    assert collect(snapshot_fn=lambda c: fs.Snapshot(c), asset_loader=lambda n: SimpleNamespace(ok=False)) is None


# ── FIAT ENVIRONMENT ────────────────────────────────────────────────────────
def test_environment_levels_cache_and_failures():
    assert env.level_of(None) == env.NA and env.level_of(20) == "Low"
    assert env.level_of(55) == "Neutral" and env.level_of(85) == "Elevated" and env.level_of(100) == "Elevated"
    env.clear_cache()
    calls = []

    def compute(ccy):
        calls.append(ccy)
        return SimpleNamespace(value=62.0, as_of="2026-10-01")
    e = env.current("USD", compute)
    assert e == {"currency": "USD", "value": 62.0, "as_of": "2026-10-01", "level": "Neutral"}
    env.current("USD", compute)
    assert calls == ["USD"]                                             # cachad
    env.clear_cache()
    bad = env.current("USD", lambda c: (_ for _ in ()).throw(RuntimeError("nät")))
    assert bad["level"] == env.NA and bad["value"] is None
    html = env.badge_html(e)
    assert "FIAT ENVIRONMENT USD" in html and "Neutral (62/100)" in html and "ingen köp- eller säljsignal" in html
    env.clear_cache()


def test_mining_pages_show_the_environment():
    src = open(os.path.join(ROOT, "wolf_panel.py"), encoding="utf-8").read()
    assert "render_fiat_environment()" in src
    assert '("Rick Rule", "Royalty C", "🐺 Wolf Asymmetry", "🚀 Råvaruhävstång")' in src


# ── REAL COMMODITY PRICES ───────────────────────────────────────────────────
def test_real_commodity_prices(fake_data):
    from fiat_debasement import ui as fu
    r = fu.real_commodity(cfg.COPPER, 2010)
    assert r["nominal"] is not None and r["real"] is not None and r["vs_gold"] is not None
    assert r["real"].loc["2010-01-01"] == pytest.approx(r["nominal"].loc["2010-01-01"])   # samma vid basåret
    assert r["real"].iloc[-1] < r["nominal"].iloc[-1]                   # KPI har stigit sedan 2010
    early = fu.real_commodity(cfg.OIL, 2000)
    assert any("finns från" in n for n in early["notes"])                # olja från 2006
    assert fu.real_commodity(cfg.GOLD, 2010)["vs_gold"] is None          # guld mot guld visas inte


def test_page_shows_real_commodity_prices(fake_data, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("FD_TEST_ROOT", ROOT)
    at = AppTest.from_function(_app, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "REAL COMMODITY PRICES" in html and "I 2010 ÅRS DOLLAR" in html
    at.selectbox(key="fd_rc_name").set_value(cfg.OIL).run()
    at.selectbox(key="fd_rc_base").set_value(2020).run()
    assert not at.exception
