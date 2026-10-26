"""
INTELLIGENCE → 🕯️ Teknisk analys: sökbar ticker, pris + order blocks, order
block-tabell, entry/exit-mönster (grupperade med när de kom), risk, momentum,
volatilitetsfördelning, oscillator och Bull List %. Syntetiska kurser.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import tech_analysis as ta  # noqa: E402


def _walk(n=500, seed=0, drift=0.0005):
    g = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-02", periods=n)
    c = 100 * np.exp(np.cumsum(g.normal(drift, 0.015, n)))
    o = c * np.exp(g.normal(0, 0.006, n))
    h = np.maximum(c, o) * (1 + abs(g.normal(0, 0.008, n)))
    lo = np.minimum(c, o) * (1 - abs(g.normal(0, 0.008, n)))
    return pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c, "Volume": 1e6 * np.exp(g.normal(0, 0.4, n))},
                        index=idx)


def test_load_prefers_yahoo_and_falls_back_to_borsdata():
    df, src = ta.load_ohlcv("NVDA", "2y", getter=lambda t, p: _walk(), bd=lambda t: None)
    assert src == "Yahoo Finance NVDA" and list(df.columns)[:1] == ["Date"] and len(df) == 500
    df, src = ta.load_ohlcv("VOLV-B.ST", "2y", getter=lambda t, p: pd.DataFrame(), bd=lambda t: _walk(seed=2))
    assert src == "Börsdata VOLV-B.ST" and len(df) == 500
    assert ta.load_ohlcv("X", "1y", getter=lambda t, p: pd.DataFrame(), bd=lambda t: None) == (None, "")


def test_period_limits_the_view_even_when_the_source_sends_full_history():
    full = _walk(n=3000)                                                  # ~12 år, som Börsdata kan svara med
    full = pd.concat([full, full.iloc[[-1]]])                             # och en dubblerad sista dag
    for period, first in (("6mo", "2026-04-02"), ("1y", "2025-10-02"), ("2y", "2024-10-02")):
        df, _src = ta.load_ohlcv("AMC", period, getter=lambda t, p: full, bd=lambda t: None)
        assert str(df["Date"].iloc[0].date()) >= first and df["Date"].iloc[0] < pd.Timestamp(first) + pd.Timedelta(days=5)
        assert df["Date"].is_unique and str(df["Date"].iloc[-1].date()) == "2026-10-02"


def test_source_label_follows_the_price_layer(monkeypatch):
    import market_prices
    monkeypatch.setattr(market_prices, "STATS", {"borsdata": 0, "yahoo": 0, "missing": 0})

    def from_bd(t, p):
        market_prices.STATS["borsdata"] += 1
        return _walk()
    monkeypatch.setattr(market_prices, "ohlcv", from_bd)
    assert ta.load_ohlcv("AMC", "1y")[1] == "Börsdata AMC"
    monkeypatch.setattr(market_prices, "ohlcv", lambda t, p: _walk())
    assert ta.load_ohlcv("SPY", "1y")[1] == "Yahoo Finance SPY"


def test_same_candle_from_both_sensitivities_is_one_block():
    ob = lambda t, lo, hi, d, s: SimpleNamespace(type=t, low=lo, high=hi, status="Active", date=d,  # noqa: E731
                                                 vol_strength=1.2, sensitivity=s)
    obs = [ob("bullish", 2.85, 2.96, "2026-10-07", 1), ob("bullish", 2.85, 2.96, "2026-10-07", 2),
           ob("bearish", 2.85, 2.96, "2026-10-07", 1), ob("bullish", 2.85, 2.96, "2026-09-01", 2)]
    assert [(o.type, o.date) for o in ta.unique_orderblocks(obs)] == [
        ("bullish", "2026-10-07"), ("bearish", "2026-10-07"), ("bullish", "2026-09-01")]
    assert len(ta.orderblock_rows(ta.unique_orderblocks(obs), 2.9)) == 3


def test_repeated_patterns_are_grouped_with_the_latest_day():
    P = lambda name, ago, conf="Strong": SimpleNamespace(name=name, bar_index=ago, confidence=conf,  # noqa: E731
                                                         description="d", visual="⬇")
    dates = pd.Series(pd.bdate_range(end="2026-10-02", periods=10))
    groups = ta.grouped_patterns([P("Three Black Crows", 2), P("Three Black Crows", 0), P("Three Black Crows", 1),
                                  P("Evening Star", 3)], dates)
    assert [(g[0].name, g[1], g[2]) for g in groups] == [("Three Black Crows", 3, 0), ("Evening Star", 1, 3)]
    assert groups[0][3] == "2026-10-02" and ta._when(0) == "i dag" and ta._when(3) == "för 3 dagar sedan"
    html = ta.patterns_html("EXIT WARNINGS", "#f00", groups, "inga")
    assert html.count("Three Black Crows") == 1 and "×3" in html and "i dag (2026-10-02)" in html


def test_orderblock_rows():
    ob = lambda t, lo, hi, st="Active": SimpleNamespace(type=t, low=lo, high=hi, status=st,  # noqa: E731
                                                        date="2026-09-01", vol_strength=1.8)
    rows = ta.orderblock_rows([ob("bearish", 110, 114), ob("bullish", 95, 99), ob("bullish", 99, 101),
                               ob("bearish", 120, 125, "Mitigated")], price=100.0)
    assert [r[0] for r in rows] == ["bullish", "bullish", "bearish"]              # inne först, mitigerade bort
    assert rows[0][6] is True and round(rows[1][3], 1) == -3.0 and round(rows[2][3], 1) == 12.0


def test_page_renders_all_sections(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import market_prices
    monkeypatch.setattr(market_prices, "ohlcv", lambda t, p="1y": _walk(seed=hash(t) % 50))
    monkeypatch.setenv("TA_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["TA_TEST_ROOT"])
        from tech_analysis import render_tech_analysis_page
        render_tech_analysis_page()

    at = AppTest.from_function(app, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    at.text_input(key="ta_ticker").set_value("nvda").run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    for part in ("<b>NVDA</b>", "ORDER BLOCKS", "ENTRY PATTERNS", "EXIT WARNINGS", "RISK · MOMENTUM", "RSI 14",
                 "AVANCERAT"):
        assert part in html, part
    keys = {c.proto.id for c in at.get("plotly_chart")}
    assert len(at.get("plotly_chart")) >= 5, keys                         # pris, risk, fördelning, oscillator, Bull List


def test_navigation_and_guide():
    from ui import nav
    from ovtlyr.ui.rules_page import _PANEL_GUIDE
    # Fas 8: Thors Teknik-flik ersätter fliken — modulen ligger kvar, gamla länkar landar i Thor.
    assert "🕯️ Teknisk analys" not in nav.leaves()
    assert not any("Teknisk analys" in t for t, _r, _u in _PANEL_GUIDE)
    assert nav.resolve("intel/teknisk-analys") == ["thor"]
