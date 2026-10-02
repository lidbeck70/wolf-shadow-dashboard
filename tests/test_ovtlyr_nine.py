"""
Viking PR 2 — OVTLYR Nine (Market 40 % · Sector 30 % · Stock 30 %) på
riktiga priser: SPY, de 11 SPDR-sektor-ETF:erna och aktien. Allt är
WOLF APPROXIMATION; DATA UNAVAILABLE och STALE DATA räknas aldrig som PASS.
Syntetiska kurser — inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import ovtlyr_nine as on  # noqa: E402

END = pd.Timestamp("2026-09-30")


def _df(n=260, start=100.0, step=0.4, wiggle=0.0, end=END, last_drop=0.0, vol=1e6, pop=4.0):
    """Stigande kurs; de fem sista dagarna stiger extra (pop) så att F&G stiger."""
    idx = pd.bdate_range(end=end, periods=n)
    base = start + step * np.arange(n) + wiggle * np.sin(np.arange(n) / 3.0)
    if pop and n >= 5:
        base[-5:] += np.linspace(pop / 5, pop, 5)
    base[-1] -= last_drop
    c = pd.Series(base, index=idx)
    v = vol * (1 + 0.3 * np.cos(np.arange(n) / 4.0))
    if pop and n >= 5:
        v[-5:] *= np.linspace(1.2, 2.0, 5)
    v = pd.Series(v, index=idx)
    return pd.DataFrame({"Open": c * 0.998, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": v})


def _etfs(up=11, down=0):
    out = {}
    names = list(on.SECTOR_ETFS.values())
    for i, t in enumerate(names):
        out[t] = _df(step=0.4)["Close"] if i < up else _df(step=-0.4, start=300)["Close"]
    return out


BULL_OB = {"signal_bias": "BUY", "approaching_bearish": False}


def _bull(**kw):
    args = dict(ticker="NVDA", stock_df=_df(), spy_df=_df(), sector_etf="XLK", sector_df=_df(),
                etf_closes=_etfs(), ob_analysis=BULL_OB, today=END)
    args.update(kw)
    return on.compute(**args)


def test_structure_follows_the_spec():
    r = _bull()
    assert [f.key for f in r.market] == ["market.trend", "market.signal", "market.breadth"]
    assert [f.key for f in r.sector] == ["sector.fear_greed", "sector.breadth"]
    assert [f.key for f in r.stock] == ["stock.trend", "stock.signal", "stock.fear_greed", "stock.blocks"]
    d = r.as_dict()
    assert set(d["market"]) == {"trend", "signal", "breadth"} and set(d["sector"]) == {"fear_greed", "breadth"}
    assert set(d["stock"]) == {"trend", "signal", "fear_greed", "blocks"}
    assert d["nine_total"] == 9 and d["label"] == on.APPROXIMATION
    f = d["market"]["trend"]
    assert {"value", "timestamp", "source", "status"} <= set(f) and f["source"] == "Yahoo Finance SPY"
    assert f["timestamp"] == "2026-09-30" and f["label"] == on.APPROXIMATION


def test_trend_and_signal_are_computed_from_spy():
    r = _bull(spy_df=_df(step=-0.4, start=300))                       # SPY faller
    assert r.get("market.trend").status == on.FAIL and r.get("market.signal").status == on.FAIL
    assert "EMA20" in r.get("market.signal").detail and "SELL" in r.get("market.signal").detail
    r = _bull(spy_df=_df(last_drop=8.0))                               # trend upp, men stängning under EMA20
    assert r.get("market.trend").status == on.FAIL or r.get("market.signal").status == on.FAIL


def test_market_breadth_from_sector_etfs():
    assert _bull(etf_closes=_etfs(up=11)).get("market.breadth").status == on.PASS
    weak = _bull(etf_closes=_etfs(up=3))
    assert weak.get("market.breadth").status == on.FAIL and weak.get("market.breadth").value == pytest.approx(27.3, abs=0.1)
    gap = _bull(etf_closes={k: v for i, (k, v) in enumerate(_etfs().items()) if i < 5})
    assert gap.get("market.breadth").status == on.UNAVAILABLE and "5 av 11" in gap.get("market.breadth").detail
    assert len(_bull().etf_states) == 11


def test_weighting_and_status():
    r = _bull()
    assert r.passed == 9 and r.weighted == 100 and r.status == on.FULL
    r = _bull(stock_df=_df(step=-0.4, start=300), ob_analysis={"signal_bias": "SELL"})
    assert r.layer_passed("market") == 3 and r.layer_passed("sector") == 2
    assert r.status == on.NOT_ALIGNED and r.passed < 9
    assert r.weighted == pytest.approx(40 + 30 + r.layer_passed("stock") / 4 * 30)


def test_missing_and_stale_data_never_pass():
    r = _bull(sector_etf=None, sector_df=None)
    assert [f.status for f in r.sector] == [on.UNAVAILABLE, on.UNAVAILABLE]
    assert r.passed == 7 and r.weighted == 70 and r.status == on.UNAVAILABLE
    short = _bull(stock_df=_df(n=30))
    assert all(f.status == on.UNAVAILABLE for f in short.stock) and "kräver 60" in short.stock[0].detail
    old = _bull(spy_df=_df(end=END - pd.Timedelta(days=14)))
    assert old.get("market.trend").status == on.STALE and not old.get("market.trend").passed
    assert old.get("market.trend").as_dict()["status"] == on.STALE
    assert _bull(spy_df=_df(end=END - pd.Timedelta(days=3)), today=END).get("market.trend").status != on.STALE


def test_blocks_and_fear_greed():
    assert _bull(ob_analysis={"signal_bias": "REDUCE"}).get("stock.blocks").status == on.FAIL
    assert _bull(ob_analysis={"signal_bias": "BUY", "approaching_bearish": True}).get("stock.blocks").status == on.FAIL
    fg = _bull().get("stock.fear_greed")
    assert fg.value is not None and 0 <= fg.value <= 100 and "för 5 dagar sedan" in fg.detail
    assert on.fear_greed(_df(n=20)) is None


def test_sector_mapping_and_evaluate_with_a_getter():
    assert on.sector_etf_for("Technology") == "XLK" and on.sector_etf_for("Basic Materials") == "XLB"
    assert on.sector_etf_for("Okänd") is None and on.sector_etf_for(None) is None
    data = {"SPY": _df(), "NVDA": _df(), **{t: _df() for t in on.SECTOR_ETFS.values()}}
    calls = []

    def getter(t, period):
        calls.append((t, period))
        return data.get(t, pd.DataFrame())
    r = on.evaluate("NVDA", ob_analysis=BULL_OB, getter=getter, sector_getter=lambda t: "Technology", today=END)
    assert r.sector_etf == "XLK" and r.passed == 9
    assert ("SPY", "1y") in calls and ("NVDA", "1y") in calls
    gone = on.evaluate("X", getter=lambda t, p: pd.DataFrame(), sector_getter=lambda t: None, today=END)
    assert gone.passed == 0 and gone.status == on.UNAVAILABLE and not gone.failed


# ── Signalmotorn får riktig data ─────────────────────────────────────────────
def _lt(nine):
    from ovtlyr.signals.longterm_signals import compute_longterm_signal
    trend = {"price": 110.0, "ema10": 105.0, "ema20": 100.0, "ema50": 95.0, "ema200": 80.0,
             "trend_state": "bullish", "regime_color": "green"}
    return compute_longterm_signal(trend, {"score": None, "available": False}, {"risk_score": 40},
                                   BULL_OB, None, nine=nine)


def test_longterm_signal_uses_nine():
    s = _lt(_bull())
    assert s["ovtlyr_nine"] == 100 and s["signal"] == "BUY"
    assert all(g["status"] == "PASS" for g in s["gates"][:9])
    assert on.APPROXIMATION in s["gates"][0]["detail"] and "Yahoo Finance SPY" in s["gates"][0]["detail"]
    down = _lt(_bull(spy_df=_df(step=-0.4, start=300)))
    assert down["signal"] == "SELL" and any("SPY < 20EMA" in r for r in down["reasons"])
    assert down["exit_triggers"][0]["active"] is True                # SPY under EMA20 → stäng allt
    gap = _lt(_bull(sector_etf=None, sector_df=None))
    assert [g["status"] for g in gap["gates"][3:5]] == [on.UNAVAILABLE, on.UNAVAILABLE]


# ── Kortet och sidan ─────────────────────────────────────────────────────────
def test_card_shows_layers_score_and_label():
    from ovtlyr.ui.nine_card import card_html
    html = card_html(_bull())
    assert "OVTLYR NINE" in html and "SLINGSHOT SETUP" in html and on.APPROXIMATION in html
    assert "MARKET 40 %" in html and "SECTOR 30 % · XLK" in html and "STOCK 30 %" in html
    assert "TOTAL 9 / 9" in html and "100 / 100" in html and "STATUS: FULL ALIGNMENT" in html
    html = card_html(_bull(sector_etf=None, sector_df=None, ob_analysis={"signal_bias": "SELL"}))
    assert "STATUS: NOT ALIGNED" in html and "Failed: Stock blocks" in html
    assert "Data saknas (räknas inte som PASS): Sector fear &amp; greed" in html or \
           "Data saknas (räknas inte som PASS): Sector fear & greed" in html
    assert "Market 3/3 × 40 = 40" in html and "Sector 0/2 × 30 = 0" in html


def test_viking_labels_no_longer_claim_to_be_ovtlyr_nine():
    src = open(os.path.join(ROOT, "ovtlyr", "ui", "layout.py"), encoding="utf-8").read()
    assert "VIKING'S NINE" not in src and "VIKING EXECUTION FILTER" in src and "inte OVTLYR Nine" in src
    assert "render_nine_card(nine)" in src and "nine=nine" in src
    assert "Vikings Nine" not in open(os.path.join(ROOT, "alert_rules.py"), encoding="utf-8").read()
