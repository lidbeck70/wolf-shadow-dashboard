"""
🪓 BERSERK PR 1 — de tre setupen, backtestet i R, portföljgränserna och sidan.
Syntetiska kurser — inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import viking_backtest as vb  # noqa: E402
import viking_portfolio as vp  # noqa: E402
from berserk import backtest as bt  # noqa: E402
from berserk import signals as sg  # noqa: E402
from berserk import themes as th  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = pd.bdate_range(end="2026-09-30", periods=2000)
TODAY = pd.Timestamp("2026-09-30")


def _walk(seed, drift=0.0003, vol=0.02, cyc=0.25):
    g = np.random.default_rng(seed)
    n = len(IDX)
    lr = g.normal(drift, vol, n) + cyc * np.sin(np.arange(n) / 120) * 0.01
    c = 100 * np.exp(np.cumsum(lr))
    o = c * np.exp(g.normal(0, 0.006, n))
    h = np.maximum(c, o) * (1 + abs(g.normal(0, 0.008, n)))
    lo = np.minimum(c, o) * (1 - abs(g.normal(0, 0.008, n)))
    return pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c, "Volume": 2e5 * np.exp(g.normal(0, 0.5, n))},
                        index=IDX)


NAMES = ["BOL.ST", "EQNR.OL", "MOWI.OL", "GLD", "FCX", "HG=F", "GC=F", "BZ=F", "SPY", "DBC"]
DATA = {t: _walk(i) for i, t in enumerate(NAMES)}
TICKERS = ["BOL.ST", "EQNR.OL", "MOWI.OL", "GLD", "FCX"]


def _run(**kw):
    cfg = bt.Config(**{"years": 5, "min_turnover_m": 1.0, **kw})
    return bt.run(TICKERS + ["XYZ"], getter=lambda t, p: DATA.get(t), cfg=cfg, today=TODAY,
                  nordic_provider=lambda: {"close": DATA["SPY"]["Close"]})


@pytest.fixture(scope="module")
def res():
    return _run()


# ── Setupen ─────────────────────────────────────────────────────────────────
def test_signals_are_causal():
    stock, drv = DATA["BOL.ST"], DATA["HG=F"]["Close"]
    full = sg.frame(stock, drv, market=DATA["SPY"]["Close"])
    for i in (900, 1400, 1900):
        d = IDX[i]
        cut = sg.frame(stock.iloc[:i + 1], drv[drv.index <= d], market=DATA["SPY"]["Close"][DATA["SPY"].index <= d])
        for col in sg.SETUPS + ("divergence", "rsi2", "market_ok"):
            a, b = full[col].iloc[i], cut[col].iloc[-1]
            assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), (col, d)


def _flat(n=400, price=100.0):
    idx = pd.bdate_range(end="2026-09-30", periods=n)
    c = np.full(n, price)
    return pd.DataFrame({"Open": c, "High": c * 1.005, "Low": c * 0.995, "Close": c, "Volume": 1e6}, index=idx)


def test_s1_divergence_needs_strong_driver_lagging_stock_and_reversal():
    stock = _flat()
    idx = stock.index
    drv = pd.Series(np.linspace(100, 300, len(idx)), index=idx)          # råvaran stiger kraftigt
    last = len(idx) - 1
    stock.iloc[last, stock.columns.get_loc("Open")] = 99.5
    stock.iloc[last, stock.columns.get_loc("Close")] = 101.5               # över gårdagens high, grön
    stock.iloc[last, stock.columns.get_loc("High")] = 101.6
    stock.iloc[last, stock.columns.get_loc("Volume")] = 3e6                # relativ volym 3×
    f = sg.frame(stock, drv)
    assert f[sg.S1].iloc[-1] and f["divergence"].iloc[-1] <= sg.DIV_LAG
    assert not f[sg.S1].iloc[-2]                                           # ingen vändning dagen före
    assert not sg.frame(stock, drv, is_etf=True)[sg.S1].iloc[-1]           # en ETF ÄR råvaran
    flat_drv = pd.Series(100.0, index=idx)
    assert not sg.frame(stock, flat_drv)[sg.S1].iloc[-1]                   # råvaran inte stark


def test_s3_snapback_after_panic_in_commodity_uptrend():
    stock = _flat()
    idx = stock.index
    c = np.linspace(60, 140, len(idx))
    c[-3:] = [134, 129, 124]                                               # tre panikdagar, fortfarande över SMA200
    stock["Close"], stock["Open"] = c, c * 1.002
    stock["High"], stock["Low"] = np.maximum(c, c * 1.002) * 1.003, c * 0.997
    drv = pd.Series(np.linspace(100, 150, len(idx)), index=idx)
    f = sg.frame(stock, drv)
    assert f["rsi2"].iloc[-1] < sg.RSI2_MAX and f[sg.S3].iloc[-1]
    falling = pd.Series(np.linspace(150, 100, len(idx)), index=idx)
    assert not sg.frame(stock, falling)[sg.S3].iloc[-1]                    # råvaran under SMA200
    assert sg.frame(stock, None)[sg.S3].iloc[-1]                           # utan drivare: aktiens egen trend


def test_s2_cycle_turn_needs_bear_turn_hated_and_breakout():
    n = 600
    idx = pd.bdate_range(end="2026-09-30", periods=n)
    d = np.concatenate([np.linspace(100, 100, 300), np.linspace(100, 55, 200), np.linspace(55, 75, 100)])
    drv = pd.Series(d, index=idx)                                          # krasch, sedan vändning
    c = np.concatenate([np.linspace(100, 100, 300), np.linspace(100, 45, 200), np.linspace(45, 50, 99), [53.0]])
    stock = pd.DataFrame({"Open": c * 0.99, "High": c * 1.005, "Low": c * 0.985, "Close": c, "Volume": 1e6}, index=idx)
    stock.iloc[-1, stock.columns.get_loc("Volume")] = 3e6
    f = sg.frame(stock, drv)
    assert f[sg.S2].iloc[-1]
    calm = pd.Series(np.linspace(100, 110, n), index=idx)                  # ingen baisse i råvaran
    assert not sg.frame(stock, calm)[sg.S2].iloc[-1]


def test_pick_driver_prefers_coverage_from_period_start():
    late = pd.Series(1.0, index=pd.bdate_range("2022-08-05", "2026-09-30"))
    early = pd.Series(1.0, index=pd.bdate_range("2008-06-25", "2026-09-30"))
    series = {"LBR=F": late, "WOOD": early}
    assert bt.pick_driver("skog", "2008-01-01", series)[0] == "WOOD"       # ingen täcker 2008 → längst historik
    assert bt.pick_driver("skog", "2010-01-01", series)[0] == "WOOD"
    assert bt.pick_driver("skog", "2023-01-01", series)[0] == "LBR=F"
    assert bt.pick_driver("lax", "2010-01-01", series) == (None, None)


# ── Backtestet ──────────────────────────────────────────────────────────────
def test_trades_enter_next_open_and_exit_by_setup_rules(res):
    assert res["trades"]
    allowed = {sg.S1: {"stopp", "breakeven-stopp", "trailing EMA20", "råvaran under SMA50", "tidsstopp", "öppen"},
               sg.S2: {"stopp", "trailing EMA50", "råvaran under EMA50", "öppen"},
               sg.S3: {"katastrofstopp", "över SMA5", "efter 5 dagar", "öppen"}}
    for t in res["trades"]:
        s = t.features["setup"]
        df = DATA[t.ticker]
        i = df.index.get_loc(pd.Timestamp(t.signal_date))
        assert t.entry == pytest.approx(df["Open"].iloc[i + 1], abs=1e-3)
        assert t.exit_reason in allowed[s], (s, t.exit_reason)
        atr0 = sg.atr(df).iloc[i]
        assert t.risk == pytest.approx(bt.STOP_ATR[s] * atr0, abs=1e-3)
        assert t.r == pytest.approx((t.exit - t.entry) / t.risk, abs=1e-3)
        assert t.features["risk_pct"] == bt.RISK_BY_SETUP[s] and t.features["complex"] == th.complex_of(t.sector)
        if s == sg.S3 and not t.open:
            assert t.days <= bt.S3_MAX_DAYS + 1
        assert t.dates[0] == t.entry_date and t.dates[-1] == t.exit_date


def test_s3_exits_within_five_days_and_s1_time_stop(res):
    s3 = [t for t in res["trades"] if t.features["setup"] == sg.S3 and not t.open]
    assert s3 and all(t.exit_reason in ("över SMA5", "efter 5 dagar", "katastrofstopp") for t in s3)


def test_run_reports_drivers_unknown_tickers_and_benchmarks(res):
    assert res["drivers"]["koppar"] == "HG=F" and res["drivers"]["lax"] is None
    bad = next(p for p in res["per_ticker"] if p["ticker"] == "XYZ")
    assert "okänd ticker" in bad["error"]
    assert {"SPY", "OMXS30", "DBC (råvarukorg)"} <= set(res["benchmarks"])
    assert all(t.features["setup"] == sg.S3 for t in res["trades"] if t.ticker == "MOWI.OL")   # lax: bara S3
    assert not [t for t in res["trades"] if t.ticker == "GLD" and t.features["setup"] == sg.S1]


def test_setup_selection_and_gates():
    only3 = _run(setups=(sg.S3,))
    assert only3["trades"] and all(t.features["setup"] == sg.S3 for t in only3["trades"])
    thin = _run(min_turnover_m=1e6)
    assert thin["trades"] == [] and thin["thin"] > 0
    no_gate = _run(market_gate=False)
    assert no_gate["market_blocked"] == 0 and len(no_gate["trades"]) >= len(_run()["trades"])


def test_free_period_cuts_data():
    r = _run(start_year=2024, end_year=2025)
    assert r["period"]["start"] == "2024-01-01" and r["period"]["end"] == "2025-12-31"
    assert all("2024-01-01" <= t.signal_date and t.exit_date <= "2025-12-31" for t in r["trades"])


# ── Portföljen ──────────────────────────────────────────────────────────────
D = [str(d.date()) for d in pd.bdate_range("2026-01-05", periods=30)]


def _pt(tk, theme, cx, risk_pct=1.25, stop=4.0, setup=sg.S1):
    return vb.Trade(tk, D[0], D[1], 100.0, 100 - stop, stop, sg.PRIORITY[setup], exit_date=D[10], exit=104.0, r=1.0,
                    sector=theme, features={"theme": theme, "complex": cx, "risk_pct": risk_pct, "setup": setup})


def test_portfolio_caps():
    pc = bt.portfolio_config()
    assert (pc.max_positions, pc.max_position_pct, pc.sector_cap, pc.max_heat_pct) == (8, 20.0, 2, 6.0)
    trades = [_pt(f"K{k}", "koppar", "basmetaller") for k in range(3)]               # 2 per tema
    trades += [_pt("AL", "aluminium", "basmetaller"), _pt("ST", "stal", "basmetaller"),
               _pt("ZN", "zink_nickel", "basmetaller")]                              # 4 per komplex
    p = vp.simulate(trades, pc=pc)
    assert p["skipped_sector"] == 1 and p["skipped_group"] == 1 and p["taken"] == 4
    assert p["rows"][0]["position_pct"] == pytest.approx(20.0)                       # 1,25 % / 4 % = 31 % → tak 20


def test_portfolio_heat_and_max_positions():
    trades = [_pt(f"T{k}", f"tema{k}", f"cx{k}", stop=10.0) for k in range(10)]      # 1,25 % risk var
    p = vp.simulate(trades, pc=bt.portfolio_config())
    assert p["taken"] == 4 and p["skipped_heat"] == 6                                # 4 × 1,25 = 5 % ≤ 6 %
    small = [_pt(f"S{k}", f"tema{k}", f"cx{k}", risk_pct=0.5, stop=10.0) for k in range(10)]
    q = vp.simulate(small, pc=bt.portfolio_config())
    assert q["taken"] == 8 and q["skipped_full"] == 2                                # max 8 positioner


def test_viking_defaults_unchanged():
    pc = vp.PortfolioConfig()
    assert (pc.sector_cap, pc.group_caps, pc.max_positions, pc.max_heat_pct) == (1, (), None, None)


# ── Sidan ───────────────────────────────────────────────────────────────────
def test_variants():
    from berserk.ui import ALL_SETUPS, variants
    v = variants(list(sg.SETUPS), True)
    assert [n for n, _s in v] == list(sg.SETUPS) + [ALL_SETUPS] and v[-1][1] == sg.SETUPS
    assert variants([sg.S3], False) == [("S3", (sg.S3,))]


def test_backtest_tab_lists_berserk():
    src = open(os.path.join(ROOT, "tabs", "backtest.py"), encoding="utf-8").read()
    assert '"🪓 BERSERK"' in src and "render_berserk_backtest" in src


def test_page_renders_result(monkeypatch, res):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("BZ_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["BZ_TEST_ROOT"])
        from berserk.ui import render_berserk_backtest
        render_berserk_backtest()

    at = AppTest.from_function(app, default_timeout=120)
    at.session_state["bz_result"] = {"runs": {"Alla tre": dict(res), "S3 Snapback": dict(_run(setups=(sg.S3,)))},
                                     "selected": "Alla tre"}
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    for text in ("JÄMFÖRELSE AV SETUPS", "PER SETUP", "PER TEMA", "PER KOMPLEX", "PORTFÖLJ — ETT KONTO",
                 "ROBUSTHET", "MONTE CARLO", "BERSERK-portföljen"):
        assert text in html, text
    assert "PER LISTA" not in html and "position:sticky" in html             # inga Viking-listor · låst radnamn


def test_cost_table_uses_berserk_portfolio_rules(res):
    """Kostnadstabellens 0 bp-rad = portföljkortet (samma spärrar), och byte av regler räknar om."""
    from berserk.ui import portfolio_of
    from ovtlyr.ui.viking_robustness_ui import robustness_of
    r = dict(res)
    p = portfolio_of(r)
    rob = robustness_of(r, p, bt.portfolio_config())
    zero = rob["costs"][0]
    assert zero["Kostnad bp"] == 0 and zero["Avkastning %"] == p["return_pct"] and zero["Max DD %"] == p["max_dd_pct"]
    viking = robustness_of(r, p)                                             # Vikings förval → räknas om
    assert viking["pc"] != rob["pc"] and robustness_of(r, p) is viking


def test_market_gate_uses_the_regions_index():
    """En kanadensisk aktie spärras av TSX under SMA200 — även när SPY stiger."""
    data = dict(DATA)
    data["SU.TO"] = DATA["EQNR.OL"]
    falling = DATA["SPY"].copy()
    falling["Close"] = np.linspace(200, 100, len(IDX))                    # TSX i baisse hela vägen
    data["^GSPTSE"] = falling
    rising = DATA["SPY"].copy()
    rising["Close"] = np.linspace(100, 200, len(IDX))
    data["SPY"] = rising
    r = bt.run(["SU.TO", "FCX"], getter=lambda t, p: data.get(t), cfg=bt.Config(years=5, min_turnover_m=1.0),
               today=TODAY)
    by = {p["ticker"]: p for p in r["per_ticker"]}
    assert by["SU.TO"]["trades"] == [] and by["SU.TO"]["market_blocked"] > 0
    assert by["FCX"]["market_blocked"] == 0
    assert "^GSPTSE" in r["benchmarks"] and "SPY" in r["benchmarks"]


def test_all_list_fits_the_page():
    from berserk.ui import LISTS
    import viking_screen as vs
    allt = next(v for k, v in LISTS.items() if k.startswith("Allt"))
    assert len(vs.parse_tickers(", ".join(allt), limit=400)) == len(set(allt)) >= 250
