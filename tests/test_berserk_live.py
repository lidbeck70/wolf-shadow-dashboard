"""
🪓 BERSERK PR 2 — regimen per råvarutema, skannern (KÖP/BEVAKA, plan, spärrar),
signalloggen och de två sidorna. Syntetiska kurser — inget nätverk.
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from berserk import backtest as bt  # noqa: E402
from berserk import live  # noqa: E402
from berserk import signals as sg  # noqa: E402
from berserk import themes as th  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 600
IDX = pd.bdate_range(end="2026-09-30", periods=N)


def _ser(values):
    return pd.Series(np.asarray(values, dtype=float), index=IDX)


def _ohlcv(c, vol=1e6):
    c = np.asarray(c, dtype=float)
    return pd.DataFrame({"Open": c * 1.002, "High": c * 1.006, "Low": c * 0.994, "Close": c, "Volume": vol},
                        index=IDX)


CRASH_TURN = np.concatenate([np.full(300, 100.0), np.linspace(100, 55, 200), np.linspace(55, 75, 100)])
UP = np.linspace(100, 200, N)
DOWN = np.linspace(200, 100, N)


# ── Regimen ─────────────────────────────────────────────────────────────────
def test_phases_from_driver_state():
    drivers = {"koppar": ("HG=F", _ser(CRASH_TURN)), "guld": ("GC=F", _ser(UP)), "olja": ("BZ=F", _ser(DOWN)),
               "lax": (None, None)}
    rows = {r["theme"]: r for r in live.theme_states(drivers)}
    assert rows["koppar"]["phase"] == "BAISSE · VÄNDER" and sg.S2 in rows["koppar"]["setups"]
    assert rows["guld"]["phase"] == "STARK" and sg.S3 in rows["guld"]["setups"]
    assert rows["olja"]["phase"] == "BAISSE" and rows["olja"]["setups"] == []     # i nedre 20 % av intervallet
    assert rows["lax"]["phase"] == "INGEN DRIVARE" and rows["lax"]["setups"] == [sg.S3]
    assert rows["guld"]["range5y"] == 100 and rows["guld"]["ret63"] > 0
    assert set(rows) == set(th.THEMES)


def test_producer_divergence_and_s1_permission():
    gold = _ser(100 * np.exp(np.linspace(0, 2, N)))                         # +23 % på 63 d
    drivers = {"guld": ("GC=F", gold)}
    flat = _ser(np.full(N, 50.0))
    rows = {r["theme"]: r for r in live.theme_states(drivers, {"NEM": flat, "AEM": flat, "B": gold})}
    g = rows["guld"]
    assert g["lagging"] == 2 and g["divergence"] < 0                       # två producenter efter guldet
    assert sg.S1 in g["setups"]


def test_market_states():
    rows = {r["region"]: r for r in live.market_states({"SPY": _ser(UP), "^GSPTSE": _ser(DOWN)})}
    assert rows["USA"]["ok"] is True and rows["USA"]["vs_sma200"] > 0
    assert rows["Kanada"]["ok"] is False and rows["Australien"]["ok"] is None


# ── Skannern ────────────────────────────────────────────────────────────────
def _panic_stock():
    c = np.linspace(60, 140, N)
    c[-3:] = [134, 129, 124]                                               # S3: panik i upptrend
    return _ohlcv(c)


def test_evaluate_buy_with_plan():
    r = live.evaluate("BOL.ST", _panic_stock(), _ser(UP), _ser(UP), capital=200_000, driver_symbol="HG=F")
    assert r["status"] == live.KOP and r["setup"] == sg.S3 and "S3 Snapback idag" in r["why"][0]
    atr = sg.atr(_panic_stock()).iloc[-1]
    assert r["stop"] == pytest.approx(124 - bt.STOP_ATR[sg.S3] * atr, abs=1e-3)
    stop_pct = (124 - r["stop"]) / 124 * 100
    assert r["position_pct"] == pytest.approx(min(1.0 / stop_pct * 100, 20.0), abs=0.1)
    assert abs(r["shares"] - 200_000 * r["position_pct"] / 100 / 124) <= 2       # position avrundad till 0,1 %


def test_gates_turn_buy_into_watch():
    falling = live.evaluate("BOL.ST", _panic_stock(), _ser(UP), _ser(DOWN))
    assert falling["status"] == live.BEVAKA and any("under SMA200" in w for w in falling["why"])
    thin = live.evaluate("BOL.ST", _ohlcv(_panic_stock()["Close"].values, vol=100), _ser(UP), _ser(UP))
    assert thin["status"] == live.BEVAKA and any("omsättning" in w for w in thin["why"])


def test_watch_reasons():
    c = np.linspace(60, 140, N)
    c[-2:] = [136, 133]                                                    # RSI(2) faller, men inte under 10 ännu
    r = live.evaluate("BOL.ST", _ohlcv(c), _ser(UP), _ser(UP))
    if r["status"] == live.BEVAKA:
        assert r["setup"] in sg.SETUPS and r["why"]
    s2 = np.concatenate([np.full(300, 100.0), np.linspace(100, 45, 200), np.linspace(45, 50, 100)])
    r2 = live.evaluate("BOL.ST", _ohlcv(s2), _ser(CRASH_TURN), _ser(UP))
    assert r2["status"] == live.BEVAKA and r2["setup"] == sg.S2 and r2["why"][0].startswith("S2")
    short = live.evaluate("BOL.ST", _ohlcv(s2).iloc[:100], _ser(UP), _ser(UP))
    assert "DATA UNAVAILABLE" in short["error"]


def test_scan_sort_and_unknown():
    data = {"BOL.ST": _panic_stock(), "FCX": _ohlcv(UP), "HG=F": _ohlcv(UP), "SPY": _ohlcv(UP)}
    res = live.scan(["FCX", "BOL.ST", "XYZ"], getter=lambda t, p: data.get(t),
                    nordic_provider=lambda: {"close": _ser(UP)}, today=IDX[-1])
    rows = live.sort_rows(res["rows"])
    assert rows[0]["ticker"] == "BOL.ST" and rows[0]["status"] == live.KOP
    assert next(r for r in rows if r["ticker"] == "XYZ")["error"].startswith("okänd ticker")
    assert res["drivers"]["koppar"] == "HG=F"


def test_portfolio_flags_against_holdings():
    rows = [{"ticker": t, "status": live.KOP, "theme": theme, "complex": th.complex_of(theme)}
            for t, theme in (("FCX", "koppar"), ("AA", "aluminium"), ("NEM", "guld"))]
    live.portfolio_flags(rows, ["BOL.ST", "LUMI.ST", "NHY.OL", "SSAB-A.ST", "OUT1V.HE", "fcx", "UNKNOWN"])
    by = {r["ticker"]: r["flags"] for r in rows}
    assert "TEMA FULLT" in by["FCX"] and "ÄGS REDAN" in by["FCX"]          # två kopparbolag ägs redan
    assert "KOMPLEX FULLT" in by["AA"]                                     # fem basmetallbolag ägs
    assert "KOMPLEX FULLT" not in by["NEM"] and "TEMA FULLT" not in by["NEM"]
    many = [f"T{k}" for k in range(8)]
    rows2 = [{"ticker": "NEM", "status": live.KOP, "theme": "guld", "complex": "adelmetaller"}]
    live.portfolio_flags(rows2, ["EQNR.OL", "YAR.OL", "MOWI.OL", "SCA-B.ST", "FRO.OL", "2020.OL", "VAR.OL", "BOL.ST"]
                         + many)
    assert "MAX 8 POSITIONER" in rows2[0]["flags"]


# ── Signalloggen ────────────────────────────────────────────────────────────
def test_signal_log_one_row_per_ticker_and_day():
    from berserk.screen_ui import append_log
    rows = [{"ticker": "BOL.ST", "status": live.KOP, "setup": sg.S3, "label": "Koppar", "close": 124.0, "why": []},
            {"ticker": "FCX", "status": live.INGET, "setup": None}]
    log, n = append_log([], rows, now=datetime(2026, 9, 30, 18, 15))
    assert n == 1 and log[0]["ticker"] == "BOL.ST" and log[0]["setup"] == sg.S3
    log, n = append_log(log, rows, now=datetime(2026, 9, 30, 18, 20))
    assert n == 0 and len(log) == 1                                        # samma dag, samma innehåll
    rows[0]["close"] = 125.0
    log, n = append_log(log, rows, now=datetime(2026, 9, 30, 18, 30))
    assert n == 1 and len(log) == 1 and log[0]["close"] == 125.0
    log, n = append_log(log, rows, now=datetime(2026, 10, 1, 18, 15))
    assert len(log) == 2


# ── Navigation och sidor ────────────────────────────────────────────────────
def test_navigation():
    from ui import nav
    assert "🪓 BERSERK" in nav.options("screening/Arc Screener")
    assert "🪓 BERSERK Regime" in nav.options("regime/Råvaror")
    src = open(os.path.join(ROOT, "wolf_panel.py"), encoding="utf-8").read()
    assert 'elif inner == "🪓 BERSERK":' in src and 'elif sub == "🪓 BERSERK Regime":' in src
    assert "render_berserk_screen_page" in src and "render_berserk_regime_page" in src


def _app(func_name, state, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("BZ_TEST_ROOT", ROOT)
    monkeypatch.setenv("BZ_FUNC", func_name)

    def app():
        import importlib
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["BZ_TEST_ROOT"])
        mod, fn = _o.environ["BZ_FUNC"].rsplit(".", 1)
        getattr(importlib.import_module(mod), fn)()

    at = AppTest.from_function(app, default_timeout=90)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, at.exception
    return " ".join(m.value for m in at.markdown)


def test_regime_page(monkeypatch):
    drivers = {"koppar": ("HG=F", _ser(CRASH_TURN)), "guld": ("GC=F", _ser(UP))}
    data = {"themes": live.theme_states(drivers), "markets": live.market_states({"SPY": _ser(UP)}),
            "when": "2026-09-30 18:20"}
    html = _app("berserk.regime_ui.render_berserk_regime_page", {"bz_regime": data}, monkeypatch)
    for text in ("MARKNADERNA", "BASMETALLER", "ÄDELMETALLER", "BAISSE · VÄNDER", "Cykelvändning (S2) möjlig nu",
                 "Koppar", "FRAKT"):
        assert text in html, text


def test_screen_page(monkeypatch):
    data = {"BOL.ST": _panic_stock(), "HG=F": _ohlcv(UP), "SPY": _ohlcv(UP)}
    res = live.scan(["BOL.ST", "XYZ"], getter=lambda t, p: data.get(t), nordic_provider=lambda: {"close": _ser(UP)},
                    today=IDX[-1])
    html = _app("berserk.screen_ui.render_berserk_screen_page", {"bz_scan": res, "berserk_signals": []},
                monkeypatch)
    assert "BOL.ST" in html and "KÖP" in html and "S3" in html and "1 KÖP" in html
