"""
Tester för market_data.py.

Inget test rör nätverket: build() matas med en konstruerad OHLCV-ram och varje
tal kontrolleras mot en egen uträkning. Det som testas hårdast är att en
ofullständig ram ger ett FEL eller ett None — aldrig ett påhittat tal.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import market_data as md


def _frame(n=300, start=100.0, step=0.1, volume=1000.0):
    """Sakta stigande serie — förutsägbar, så talen går att räkna för hand."""
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    close = pd.Series([start + i * step for i in range(n)], index=idx)
    return pd.DataFrame({
        "Open": close, "High": close + 1.0, "Low": close - 1.0,
        "Close": close, "Volume": pd.Series([volume] * n, index=idx)})


def _zigzag(n=300, start=100.0):
    """Sågtandad serie — både upp- och nedgångar, så RSI är definierad."""
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    vals = [start + (i * 0.3 if i % 3 else -i * 0.1) for i in range(n)]
    close = pd.Series(vals, index=idx)
    return pd.DataFrame({
        "Open": close, "High": close + 1.0, "Low": close - 1.0,
        "Close": close, "Volume": pd.Series([1000.0] * n, index=idx)})


# ── Felvägarna ───────────────────────────────────────────────────────────────
def test_an_empty_frame_raises_instead_of_returning_nothing():
    with pytest.raises(md.MarketDataError):
        md.build(pd.DataFrame(), "ABB")
    with pytest.raises(md.MarketDataError):
        md.build(None, "ABB")


def test_missing_columns_name_themselves():
    df = _frame(50).drop(columns=["High", "Low"])
    with pytest.raises(md.MarketDataError) as e:
        md.build(df, "ABB")
    assert "High" in str(e.value) and "Low" in str(e.value)


def test_an_unusable_last_price_raises():
    df = _frame(50)
    df.loc[df.index[-1], "Close"] = 0.0
    with pytest.raises(md.MarketDataError):
        md.build(df, "ABB")


def test_an_empty_ticker_raises_before_any_download():
    with pytest.raises(md.MarketDataError):
        md.fetch("   ")


def test_try_snapshot_returns_the_reason_instead_of_raising(monkeypatch):
    monkeypatch.setattr(md, "_cached_download",
                        lambda name: pd.DataFrame())
    snap, err = md.try_snapshot("ABB")
    assert snap is None
    assert "ABB" in err and "börssuffix" in err


def test_a_download_failure_keeps_the_cause(monkeypatch):
    def _boom(name):
        raise RuntimeError("nätverket nere")
    monkeypatch.setattr(md, "_cached_download", _boom)
    with pytest.raises(md.MarketDataError) as e:
        md.fetch("ABB")
    assert "nätverket nere" in str(e.value)


# ── Talen ────────────────────────────────────────────────────────────────────
def test_the_snapshot_carries_the_basics():
    snap = md.build(_frame(300), "abb")
    assert snap.ticker == "ABB"
    assert snap.bars == 300
    assert snap.price == round(100.0 + 299 * 0.1, 4)
    assert snap.as_of == "2025-10-27"   # 2025-01-01 + 299 dagar


def test_a_rising_series_is_above_both_averages():
    snap = md.build(_frame(300), "ABB")
    assert snap.above_ema50 is True
    assert snap.above_ema200 is True
    assert snap.dist_ema50_pct > 0 and snap.dist_ema200_pct > 0


def test_a_falling_series_is_below_both():
    snap = md.build(_frame(300, start=200.0, step=-0.2), "ABB")
    assert snap.above_ema50 is False
    assert snap.above_ema200 is False


def test_atr_is_the_true_range_of_a_two_krona_band():
    """High − Low är konstant 2,0 och serien rör sig 0,1/dag → ATR ≈ 2,1."""
    snap = md.build(_frame(300), "ABB")
    assert 2.0 <= snap.atr14 <= 2.2
    assert snap.atr_pct == round(snap.atr14 / snap.price * 100, 2)


def test_volume_ratio_is_one_for_a_flat_volume():
    assert md.build(_frame(300), "ABB").vol_ratio == 1.0


def test_a_volume_spike_shows_up():
    df = _frame(300)
    df.loc[df.index[-1], "Volume"] = 5000.0
    assert md.build(df, "ABB").vol_ratio > 1.15


def test_the_52_week_window_never_exceeds_the_data():
    snap = md.build(_frame(60), "ABB")
    assert snap.bars == 60
    assert snap.high_52w == round(100.0 + 59 * 0.1 + 1.0, 4)
    assert snap.from_high_pct < 0        # kursen ligger under periodens topp


# ── Det som saknas ska vara None, inte gissat ────────────────────────────────
def test_short_history_leaves_the_long_average_empty():
    """Under 200 dagar finns ingen EMA200 — och då ska den vara None."""
    snap = md.build(_frame(60), "ABB")
    assert snap.ema200 is None
    assert snap.dist_ema200_pct is None
    assert snap.above_ema200 is None
    assert snap.ema50 is not None        # 60 dagar räcker för EMA50


def test_a_very_short_history_leaves_almost_everything_empty():
    snap = md.build(_frame(10), "ABB")
    assert snap.ema50 is None and snap.atr14 is None and snap.rsi14 is None
    assert snap.swing_low_20 is None and snap.vol_ratio is None
    assert snap.price > 0                # men kursen finns


def test_a_frame_without_volume_gives_no_ratio():
    df = _frame(300).drop(columns=["Volume"])
    assert md.build(df, "ABB").vol_ratio is None


def test_nan_never_leaks_out_as_a_number():
    df = _frame(300)
    df.loc[df.index[-1], "Volume"] = np.nan
    snap = md.build(df, "ABB")
    for name, value in snap.as_dict().items():
        assert value == value, f"{name} är NaN"


def test_as_dict_round_trips_every_field():
    snap = md.build(_frame(300), "ABB")
    d = snap.as_dict()
    assert d["ticker"] == "ABB" and d["price"] == snap.price
    assert "ema200" in d and "swing_low_20" in d


# ── Definitionerna får inte glida isär från optim/indicators.py ──────────────
def test_the_indicator_definitions_match_optim():
    """market_data kopierar optim/indicators.py rad 73–116, för den modulen går
    inte att importera från roten. Kopian måste ge SAMMA tal — annars säger
    Copiloten en sak om trenden och regimmotorn en annan.

    Testet läser optims källkod och kör den i ett eget namnrum, så det upptäcker
    en ändring där även om importen fortfarande är omöjlig.
    """
    import types
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "optim", "indicators.py")).read()
    start = src.index("def ema(")
    end = src.index("def donchian(")
    mod = types.ModuleType("optim_primitives")
    mod.__dict__.update({"pd": pd, "np": np})
    exec(compile(src[start:end], "optim/indicators.py", "exec"), mod.__dict__)

    df = _zigzag(300)          # RSI kräver både upp- och nedgångar
    close, high, low = df["Close"], df["High"], df["Low"]

    for name, ours, theirs in (
            ("ema", md._ema(close, 50), mod.ema(close, 50)),
            ("rsi", md._rsi(close, 14), mod.rsi(close, 14)),
            ("atr", md._atr(high, low, close, 14),
             mod.atr(high, low, close, 14))):
        a, b = float(ours.iloc[-1]), float(theirs.iloc[-1])
        assert a == a, f"{name} gav NaN"
        assert a == b, f"{name} skiljer sig: {a} vs {b}"


def test_an_undefined_rsi_becomes_none_not_nan():
    """En serie som bara stiger har inga förlustdagar — Wilder-RSI är då
    odefinierad. Den ska bli None, inte NaN som smyger vidare i prompten."""
    assert md.build(_frame(300), "ABB").rsi14 is None
    assert md.build(_zigzag(300), "ABB").rsi14 is not None
