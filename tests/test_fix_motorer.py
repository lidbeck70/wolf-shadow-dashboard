"""Granskning 2 (motorerna): Thor matar motorerna som skannrarna gör, och fel cachas inte."""
import pathlib
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_viking_nine_list_producers_pass_the_market_risk_gate():
    import valhall_scan
    assert "risk_getter" in valhall_scan.scan_kwargs(None)
    assert valhall_scan.scan_kwargs(lambda *a, **k: {}) == {}
    for f in ("viking_nine_scan.py", "thor/live.py"):
        assert "risk_getter=_risk_for" in (ROOT / f).read_text(encoding="utf-8"), f


def test_kap_badge_is_read_from_the_quality_scan():
    from thor import regim
    data = {"ca_quality": {"results": [{"ticker": "INVE B"}],
                           "quality_signals": {"INVE B": {"gates": {"Kvalitet": {"label": "KAP SCREENED"}}}}}}
    assert regim.kap_badge_for(data, "INVE-B.ST") is True
    assert regim.kap_badge_for(data, "ERIC-B.ST") is False
    assert regim.kap_badge_for({"ca_quality": {"results": [{"ticker": "ERIC B", "kap_badge": True}]}}, "ERIC-B.ST")


def test_long_screener_prefers_the_scanners_saved_market():
    src = (ROOT / "thor" / "regim.py").read_text(encoding="utf-8")
    assert '.get("markets") or {}).get(ident.home_index)' in src and "_close, _api())" in src


def test_berserk_uses_the_scanners_theme_and_your_capital():
    src = (ROOT / "thor" / "regim.py").read_text(encoding="utf-8")
    assert "theme = uv.theme_of(ident.yf) or ident.theme or None" in src
    assert "capital=_capital_default()" in src


def test_wolf_rule_matches_the_list_rule():
    import numpy as np
    import pandas as pd
    from strategies.wolf import entry_fn
    from thor import regim
    idx = pd.bdate_range("2024-01-01", periods=400)
    close = pd.Series(np.linspace(50, 120, 400), index=idx)
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6})
    r, e = regim.wolf_rule(df), entry_fn(df)
    assert r["signal"] == e["signal"] and r["stop"] == e["stop_loss"] and r["tp1"] == e["tp1_price"]
    assert regim.wolf_rule(df.tail(100)) is None


def test_wolf_rows_carry_their_entry_price():
    from valhall import wolf as vw
    arc = {"ABC.ST": {"score": 90, "name": "ABC"}}
    entries = {"ABC.ST": {"signal": "BUY", "entry": 101.0, "stop": 95.0, "tp1": 116.6, "gates": 5, "gates_n": 5}}
    [p] = vw.arc_picks(arc, entries, {"ABC.ST": {"segment": "tech"}}, "tech")
    assert p["price"] == 101.0


def test_failed_asymmetry_fetch_is_not_cached():
    from thor import asymmetri as ta
    st = SimpleNamespace(session_state={})
    ta.cached(st, "NEM", fetch_fn=lambda t, api=None: {"source": "", "prices": None})
    good = {"source": "Börsdata", "prices": [1]}
    assert ta.cached(st, "NEM", fetch_fn=lambda t, api=None: good) is good


def test_viking_backtest_no_longer_runs_viking_nine():
    from thor import backtest
    assert "viking" not in backtest.ENGINES and "viking_nine" in backtest.ENGINES


def test_closed_bars_in_wolf_and_berserk():
    src = (ROOT / "thor" / "regim.py").read_text(encoding="utf-8")
    assert '_closed(ohlcv(ident.yf, "1y"), ident.yf)' in src and "_closed(ohlcv(ident.yf, live.STOCK_PERIOD), ident.yf)" in src
