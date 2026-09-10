"""
Alpha Regime — Deep Contrarian-läget ska tala samma språk som screenern.

Tre saker: (1) samma SMA200-tak (5 %) i stegen och i tactical entry,
(2) trendfiltren ersätts av contrarian-vakter i tactical entry, och
(3) fliken kan läsa senaste Deep Contrarian-körningen och hitta ett bolag
oavsett ticker-form ("EKTA B.ST" vs "EKTA-B.ST").
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_regime import contrarian_signals as cs
from alpha_regime import tactical_entry as te
from alpha_regime import screener_link as sl
from alpha_regime.engine import _sma


# ── (1) Samma 200-dagarstak överallt ─────────────────────────────────────────
def test_sma200_ceiling_is_shared_with_the_screener():
    from contrarian_alpha.engine import PipelineConfig
    assert cs.MAX_ABOVE_MA200_PCT == PipelineConfig().deep_max_above_sma200_pct == 5.0
    assert te._CONTRARIAN_MAX_ABOVE_SMA200_PCT == cs.MAX_ABOVE_MA200_PCT


def test_hope_phase_window_stops_at_the_screener_ceiling():
    """Förut räknades upp till +15 % som 'sista ackumuleringsfönstret' —
    screenern kastar ut allt över +5 %."""
    inside = cs.get_contrarian_stage("HOPE", 3.0, None, 55)
    assert inside.stage == "ACCUMULATE_3"
    assert any("final accumulation window" in x for x in inside.rationale)

    loved = cs.get_contrarian_stage("HOPE", 10.0, None, 55)
    assert loved.stage == "ACCUMULATE_3"
    assert any("priced in" in x for x in loved.rationale)
    assert not any("final accumulation window" in x for x in loved.rationale)

    early = cs.get_contrarian_stage("DISBELIEF", 12.0, None, 55)
    assert any("priced in" in x for x in early.rationale)
    assert all("200D" not in x for x in early.rationale)


def test_engine_sma_helper():
    s = pd.Series(np.arange(1, 301, dtype=float))
    assert _sma(s, 200) == float(np.mean(np.arange(101, 301)))
    assert np.isnan(_sma(s.iloc[:100], 200))


# ── (2) Tactical entry: vakter i stället för trendfilter ─────────────────────
def _daily(closes, volumes=None):
    n = len(closes)
    idx = pd.bdate_range(end="2026-09-09", periods=n)
    c = np.asarray(closes, dtype=float)
    v = np.asarray(volumes if volumes is not None else np.full(n, 100_000.0), dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99,
                         "Close": c, "Volume": v}, index=idx)


def _hated_base():
    """200 dagar fall 200→100, 40 dagar botten kring 100 (lägsta 98), sedan 20 dagar."""
    decline = np.linspace(200, 100, 200)
    base = np.full(40, 100.0)
    base[20] = 98.0
    vol = np.concatenate([np.full(200, 100_000.0), np.full(60, 50_000.0)])
    return decline, base, vol


def _run(monkeypatch, closes, vol, mode):
    df = _daily(closes, vol)
    monkeypatch.setattr(te, "_download_robust", lambda ticker, period: df)
    return te.compute_tactical_entry("X.ST", mode=mode)


def test_contrarian_mode_has_guards_not_trend_filters(monkeypatch):
    decline, base, vol = _hated_base()
    tail = np.linspace(99.5, 103.0, 20)            # högre botten (99.5 > 98), lugn uppgång
    r = _run(monkeypatch, np.concatenate([decline, base, tail]), vol, "contrarian")
    names = [c.name for c in r.checks]
    assert not any("50-veckors" in n or "20D EMA > 50D" in n for n in names)
    assert any("SMA200" in n for n in names)
    assert any("Bottnen håller" in n for n in names)
    assert any("Volymtorka" in n for n in names)
    guards = [c for c in r.checks if c.is_trend]
    assert len(guards) == 2 and all(c.passed for c in guards), [(c.name, c.detail) for c in guards]
    assert r.verdict != "INGEN ENTRY"
    vol_chk = next(c for c in r.checks if "Volymtorka" in c.name)
    assert vol_chk.passed


def test_contrarian_guard_blocks_when_already_loved(monkeypatch):
    decline, base, vol = _hated_base()
    tail = np.linspace(120, 140, 20)               # rusat >5 % över SMA200
    r = _run(monkeypatch, np.concatenate([decline, base, tail]), vol, "contrarian")
    g = next(c for c in r.checks if "SMA200" in c.name)
    assert not g.passed and "prissatt" in g.detail
    assert r.verdict == "INGEN ENTRY"


def test_contrarian_guard_blocks_a_falling_knife(monkeypatch):
    decline, base, vol = _hated_base()
    tail = np.linspace(97, 90, 20)                 # ny lägsta under 98
    r = _run(monkeypatch, np.concatenate([decline, base, tail]), vol, "contrarian")
    g = next(c for c in r.checks if "Bottnen" in c.name)
    assert not g.passed and "fallande kniv" in g.detail
    assert r.verdict == "INGEN ENTRY"


def test_quality_mode_keeps_its_trend_filters(monkeypatch):
    decline, base, vol = _hated_base()
    tail = np.linspace(99.5, 103.0, 20)
    r = _run(monkeypatch, np.concatenate([decline, base, tail]), vol, "quality")
    names = [c.name for c in r.checks]
    assert "Pris > 50-veckors EMA" in names and "20D EMA > 50D EMA" in names
    assert not any("Bottnen" in n for n in names)
    assert len(r.checks) == 6


# ── (3) Screener-koppling ────────────────────────────────────────────────────
def _payload():
    return {"timestamp": "2026-09-10T12:00:00", "results": [
        {"ticker": "EKTA B.ST", "name": "Elekta", "rank": 8, "composite_score": 53.5,
         "hat_score": 45.2, "necessity_score": 60, "altman_z": 2.1, "net_debt_ebitda": 1.64,
         "roic": -3.6, "close": 50.0, "sma200": 52.0, "branch": "Medicinsk Utrustning",
         "all_flags": ["ROIC_TROUGH"]},
        {"ticker": "INFRA.CO", "name": "Infracom", "rank": 1, "composite_score": 65.7,
         "hat_score": 68.8, "necessity_score": 66},
    ]}


def test_screener_rows_are_rank_ordered_and_yf_formatted():
    rows = sl.rows_from_payload(_payload())
    assert [r.yf_ticker for r in rows] == ["INFRA.CO", "EKTA-B.ST"]
    assert rows[0].label == "#1 INFRA.CO — Infracom"
    assert rows[1].pct_vs_sma200 == -3.8
    assert rows[1].flags == ["ROIC_TROUGH"]


def test_find_row_ignores_ticker_form():
    rows = sl.rows_from_payload(_payload())
    assert sl.find_row(rows, "ekta-b.st").ticker == "EKTA B.ST"
    assert sl.find_row(rows, "EKTA B.ST").rank == 8
    assert sl.find_row(rows, "BOL.ST") is None
    assert sl.rows_from_payload({}) == []


def test_cache_payload_carries_the_deep_gates():
    """Alpha Regime-kortet läser ND/EBITDA, ROIC och bransch — de måste
    sparas i Gist-payloaden."""
    from contrarian_alpha import cache
    src = open(cache.__file__, encoding="utf-8").read()
    for key in ('"net_debt_ebitda"', '"roic"', '"branch"'):
        assert key in src
