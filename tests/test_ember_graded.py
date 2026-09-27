"""
Ember med graderad setup-poäng i stället för nio hårda grindar.

Hårda grindar: pris > 50V EMA, inte sen cykel. Resten är poäng 0–100 med
vikterna i ember/config.SETUP_WEIGHTS, minus avdrag för ATR-surge och
DXY-rally. ≥ 70 = KÖPLÄGE, 50–70 = BEVAKA, annars AVVAKTA. Chop-zonen är
borta. Syntetiska kursserier — inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ember import config as cfg  # noqa: E402
from ember import engine as e  # noqa: E402
from ember import gates as g  # noqa: E402


def _series(n=320, start=100.0, slope=0.35, wave=2.5, period=7, pull=-0.04, pull_days=6, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = start + slope * t + wave * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.3, n)
    if pull_days:
        close[-pull_days:] = close[-pull_days - 1] * (1 + np.linspace(0, pull, pull_days))
    idx = pd.bdate_range("2025-06-01", periods=n)
    c = pd.Series(close, index=idx)
    df = pd.DataFrame({"Open": c.shift(1).fillna(c), "High": c * 1.01, "Low": c * 0.99, "Close": c,
                       "Volume": 1_000_000 + rng.integers(0, 200_000, n)})
    df.loc[df.index[-1], "Volume"] = 1_400_000
    return c, df


def _etf(rising=True):
    ys = np.linspace(50, 60, 130) if rising else np.full(130, 55.0)
    return pd.DataFrame({"Close": ys}, index=pd.bdate_range("2026-03-01", periods=130))


def _gates(monkeypatch, c, df, etf, pct=40.0):
    monkeypatch.setattr(g, "_download_robust", lambda tk, period: etf)      # sektor-ETF och DXY
    w = c.resample("W").last().dropna()
    tg = g.compute_trend_gates(c, w, "GDX")
    eg = g.compute_entry_gates(c, df)[0]
    nf = g.compute_notrade_flags(c, df, pct)
    return tg, eg, nf


def test_weights_and_thresholds():
    assert sum(cfg.SETUP_WEIGHTS.values()) == 100
    assert cfg.VERDICT_BUY_MIN > cfg.VERDICT_WATCH_MIN > 0
    assert cfg.RSI_ZERO > cfg.RSI_FULL and cfg.PULLBACK_MAX_PCT > cfg.PULLBACK_EMA_PCT
    # RULES-fliken läser fortfarande de gamla namnen
    assert cfg.HIGHER_LOWS_MIN == 3 and cfg.RSI_ENTRY_MAX == 45 and cfg.PULLBACK_EMA_PCT == 3.0
    assert g.graded(0, -5, 5) == 0.5 and g.graded(-7, -5, 5) == 0 and g.graded(9, -5, 5) == 1
    assert g.graded(3, 6, 0) == 0.5                                   # åt andra hållet (pullback)


def test_trend_in_pullback_is_watch_not_rejected(monkeypatch):
    """Uppåttrend, rekyl 2,8 % till 20D EMA, RSI 35, men bara två stigande
    bottnar och 2,8 % sämre än sektor-ETF:en. Gamla reglerna: REJECT på
    två hårda grindar. Nu: BEVAKA med poängen synlig."""
    c, df = _series()
    tg, eg, nf = _gates(monkeypatch, c, df, _etf(rising=True))
    score, verdict, hard = e.score_setup(tg, eg, nf)
    assert hard and verdict == cfg.SETUP_BEVAKA and 55 <= score <= 65
    by = {x.name.split(" (")[0].split(" vs")[0]: x for x in tg + eg}
    rs = next(x for x in tg if x.name.startswith("Relativ styrka"))
    assert not rs.passed and not rs.is_blocker and 0 < rs.points < rs.max_points   # graderad, inte stopp
    hl = next(x for x in tg if x.name.startswith("Stigande bottnar"))
    assert hl.points == cfg.SETUP_WEIGHTS["higher_lows"] / 2                        # 2 bottnar = halva
    assert next(x for x in tg if x.name == "Pris > 50V EMA").is_blocker
    assert all(not x.is_blocker for x in eg)                                         # inga hårda entrygrindar
    assert not any("chop" in f.name.lower() for f in nf)                             # chop-zonen är borta
    late = next(f for f in nf if f.name.startswith("Sen cykel"))
    assert late.is_blocker and not late.passed


def test_same_setup_with_positive_rs_is_a_buy(monkeypatch):
    c, df = _series()
    tg, eg, nf = _gates(monkeypatch, c, df, _etf(rising=False))          # ETF platt → RS +5,9 % → full poäng
    score, verdict, hard = e.score_setup(tg, eg, nf)
    assert verdict == cfg.SETUP_KOP and score >= cfg.VERDICT_BUY_MIN
    rs = next(x for x in tg if x.name.startswith("Relativ styrka"))
    assert rs.points == rs.max_points


def test_hard_gates_still_block(monkeypatch):
    c, df = _series()
    # sen cykel: samma poäng, AVVAKTA
    tg, eg, nf = _gates(monkeypatch, c, df, _etf(rising=False), pct=92.0)
    score, verdict, hard = e.score_setup(tg, eg, nf)
    assert not hard and verdict == cfg.SETUP_AVVAKTA and score >= cfg.VERDICT_BUY_MIN
    # under 50V EMA: nedåttrend
    c2, df2 = _series(slope=-0.35, start=250.0, pull=0.0, pull_days=0)
    tg, eg, nf = _gates(monkeypatch, c2, df2, _etf(rising=False))
    assert not next(x for x in tg if x.name == "Pris > 50V EMA").passed
    assert e.score_setup(tg, eg, nf)[1] == cfg.SETUP_AVVAKTA


def test_atr_surge_is_a_penalty_not_a_stop(monkeypatch):
    c, df = _series()
    df2 = df.copy()
    df2.loc[df2.index[-8:], "High"] = df2["Close"].iloc[-8:] * 1.08          # volatilitetsspik sista veckan
    df2.loc[df2.index[-8:], "Low"] = df2["Close"].iloc[-8:] * 0.92
    tg, eg, nf = _gates(monkeypatch, c, df2, _etf(rising=False))
    surge = next(f for f in nf if f.name.startswith("ATR-surge"))
    assert surge.passed and not surge.is_blocker and surge.points == -cfg.PENALTY_ATR_SURGE
    score, verdict, hard = e.score_setup(tg, eg, nf)
    assert hard                                                            # inte blockerad …
    base = e.score_setup(tg, eg, [f for f in nf if not f.name.startswith("ATR-surge")])[0]
    assert score == pytest.approx(base - cfg.PENALTY_ATR_SURGE, abs=0.11)  # … men avdraget syns


def test_scan_ticker_end_to_end_without_network(monkeypatch):
    c, df = _series()
    weekly = pd.DataFrame({"Close": c})
    frames = {"2y": df, "5y": weekly}
    monkeypatch.setattr(e, "_download_robust", lambda tk, period: frames.get(period, df))
    monkeypatch.setattr(g, "_download_robust", lambda tk, period: _etf(rising=False))
    monkeypatch.setattr(e, "compute_macro_score", lambda ratios, cykel: e.MacroScore())
    monkeypatch.setattr(e, "compute_sentiment_score", lambda t: e.SentimentScore())
    r = e._scan_ticker("TST", "Aktie", "Guld", "GDX", {}, None, 100_000.0)
    assert r.error is None and r.verdict == cfg.SETUP_KOP and r.eligible and r.hard_pass
    assert r.setup_score >= cfg.VERDICT_BUY_MIN and r.entry_pass and r.trend_pass
    assert r.stop is not None and r.stop < r.entry and r.shares and r.shares > 0
    from ember.cache import _setup_to_dict
    d = _setup_to_dict(r)
    assert d["verdict"] == cfg.SETUP_KOP and d["setup_score"] == r.setup_score


def test_rules_page_reads_the_new_thresholds():
    import strategy_rules as sr
    text = " ".join(str(x) for x in sr.EMBER_PB.entry)
    assert "KÖPLÄGE" in text and "70" in text and "50" in text
    assert "HÅRD" in text and "Chop-zonen" in text
    assert "alla hårda, alla måste passera" not in text
