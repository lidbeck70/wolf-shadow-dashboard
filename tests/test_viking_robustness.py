"""
Viking Nine PR A — robusthet utan regeländringar: dagsvärderad drawdown,
breakeven som live, fri period, kostnader, Monte Carlo, konfidensintervall,
kantanalys, köp och behåll, Norden OOS 100 och tickergränsen i backtestet.
Syntetiska kurser — inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import viking_backtest as vb  # noqa: E402
import viking_exit as vex  # noqa: E402
import viking_portfolio as vp  # noqa: E402
import viking_robustness as rb  # noqa: E402
import viking_screen as vs  # noqa: E402
from tests.test_viking_backtest import DATA, IDX, ROOT  # noqa: E402

TICKERS = [f"S{i}" for i in range(4)]
TODAY = pd.Timestamp("2026-09-30")


def _run(**cfg):
    base = dict(min_nine=8, years=3, require_volume=False)
    base.update(cfg)
    return vb.run(TICKERS, getter=lambda t, p: DATA.get(t), sector_getter=lambda t: "Technology",
                  cfg=vb.Config(**base), today=TODAY)


@pytest.fixture(scope="module")
def res():
    return _run()


# ── Affärernas kursbana, mått och breakeven ─────────────────────────────────
def test_trade_path_runs_from_entry_to_exit(res):
    assert res["trades"]
    for t in res["trades"]:
        assert t.dates[0] == t.entry_date and t.dates[-1] == t.exit_date and len(t.dates) == len(t.path)
        d = DATA[t.ticker]["Close"]
        assert t.path[-1] == pytest.approx(d.loc[pd.Timestamp(t.exit_date)], rel=1e-4)


def test_entry_features_are_causal(res):
    t = res["trades"][-1]
    stock = DATA[t.ticker]
    i = IDX.get_loc(pd.Timestamp(t.signal_date))
    x = vb.execution_frame(stock.iloc[:i + 2])                         # +1 dag: gapet är nästa öppning
    cut = vb.entry_features(stock.iloc[:i + 2], x, i, vb._close_on(DATA["SPY"], stock.index[:i + 2]))
    assert set(t.features) >= {"atr_pct", "rvol", "rsi", "clv", "upper_wick", "gap_atr", "dist_ema20_atr", "rs63"}
    for k, v in t.features.items():
        assert cut[k] == pytest.approx(v, abs=0.02), k


def test_breakeven_lookback_matches_live(res):
    """Backtestets 'ny högre topp' = viking_exit.breakeven_armed: 10 dagar t.o.m. entrydagen."""
    checked = 0
    for t in res["trades"]:
        if t.exit_reason not in ("BE exit", "breakeven-stopp"):
            continue
        df = DATA[t.ticker]
        x = IDX.get_loc(pd.Timestamp(t.exit_date))
        assert vex.breakeven_armed(df.iloc[:x], t.entry_date), t
        checked += 1
    assert checked


# ── Dagsvärderad drawdown ───────────────────────────────────────────────────
D = [str(d.date()) for d in pd.bdate_range("2026-01-05", periods=30)]


def _tr(tk, a, b, path, exit_px, open_=False):
    return vb.Trade(tk, D[a - 1], D[a], 100.0, 96.25, 3.75, 9, exit_date=D[b], exit=exit_px,
                    r=round((exit_px - 100) / 3.75, 3), open=open_, dates=tuple(D[a:b + 1]), path=tuple(path))


def test_mtm_drawdown_sees_open_losses():
    t = _tr("A", 1, 4, [100, 92, 90, 110], 110.0)                       # dyker 10 %, stänger +10 %
    p = vp.simulate([t])
    assert p["closed_dd_pct"] == 0.0
    assert p["max_dd_pct"] == pytest.approx(25 * 0.10, abs=0.05)        # 25 % position × −10 %
    assert p["mtm"]["max_exposure_pct"] == 25.0 and 0 < p["mtm"]["avg_exposure_pct"] <= 25
    assert p["cap_share"] == 100.0                                       # stopp 3,75 % → taket 25 %


def test_mtm_realises_on_exit_day_and_handles_open_trades():
    a = _tr("A", 1, 3, [100, 101, 99], 95.0)                             # exitkursen, inte stängningen
    b = _tr("B", 2, 6, [100, 104, 106, 108, 107], 107.0, open_=True)
    p = vp.simulate([a, b])
    curve = dict(p["mtm"]["curve"])
    assert curve[D[3]] == pytest.approx((1 + 0.25 * -0.05) * (1 + 0.25 * 0.04) * 100 - 100, abs=0.02)
    assert D[6] in curve


def test_without_paths_falls_back_to_closed_drawdown():
    t = vb.Trade("A", D[0], D[1], 100.0, 96.25, 3.75, 9, exit_date=D[3], exit=90.0, r=-2.667)
    p = vp.simulate([t])
    assert p["mtm"] is None and p["max_dd_pct"] == p["closed_dd_pct"]


def test_run_portfolio_mtm_is_at_least_closed(res):
    p = vp.simulate(res["trades"], years=res["period"]["years"])
    assert p["mtm"] is not None and p["max_dd_pct"] >= p["closed_dd_pct"] - 1e-9


# ── Fri period och omsättning ───────────────────────────────────────────────
def test_period_of():
    s, e, y = vb.period_of(vb.Config(start_year=2008, end_year=2020), TODAY)
    assert str(s.date()) == "2008-01-01" and str(e.date()) == "2020-12-31" and y == pytest.approx(13.0, abs=0.01)
    s, e, y = vb.period_of(vb.Config(years=3), TODAY)
    assert e == TODAY and y == pytest.approx(3.0, abs=0.01)


def test_free_period_cuts_data_at_the_end():
    calls = []

    def getter(t, period):
        calls.append(period)
        return DATA.get(t)
    r = vb.run(TICKERS, getter=getter, sector_getter=lambda t: "Technology",
               cfg=vb.Config(min_nine=8, require_volume=False, start_year=2024, end_year=2025), today=TODAY)
    assert set(calls) == {"max"}
    assert r["period"] == {"start": "2024-01-01", "end": "2025-12-31", "years": pytest.approx(2.0, abs=0.01)}
    assert r["trades"] and all("2024-01-01" <= t.signal_date <= "2025-12-31" for t in r["trades"])
    assert all(t.exit_date <= "2025-12-31" for t in r["trades"])
    assert r["benchmarks"]["SPY"].index.max() <= pd.Timestamp("2025-12-31")


def test_min_turnover_blocks_thin_signals():
    r = _run(min_turnover_m=1e6)                                         # omöjligt högt
    assert r["trades"] == [] and r["thin"] > 0


# ── Kostnader ───────────────────────────────────────────────────────────────
def test_costs_in_r():
    t = vb.Trade("A", D[0], D[1], 100.0, 96.0, 4.0, 9, exit_date=D[3], exit=108.0, r=2.0, exit_reason="BE exit")
    s = vb.Trade("B", D[0], D[1], 100.0, 96.0, 4.0, 9, exit_date=D[3], exit=96.0, r=-1.0, exit_reason="stopp")
    a, b = rb.with_costs([t, s], 20, gap_bps=20)
    assert a.r == pytest.approx(2.0 - 0.002 / 0.04, abs=1e-3)           # 20 bp / 4 % stopp = 0,05R
    assert b.r == pytest.approx(-1.0 - 0.004 / 0.04, abs=1e-3)          # stopp: + gap-slippage
    assert t.r == 2.0                                                    # originalet orört
    assert rb.breakeven_cost_bps([t, s]) == pytest.approx(0.5 / 0.0025, abs=1)


def test_cost_table_gets_worse_with_cost(res):
    rows = rb.cost_table(res["trades"], 3)
    exps = [r["Expectancy R"] for r in rows if r["Gap-slippage bp"] == 0]
    assert [r["Kostnad bp"] for r in rows if r["Gap-slippage bp"] == 0] == list(rb.COST_LEVELS_BPS)
    assert exps == sorted(exps, reverse=True)


# ── Monte Carlo ─────────────────────────────────────────────────────────────
def test_monte_carlo_on_actual_trades(res):
    p = vp.simulate(res["trades"], years=3)
    mc = rb.monte_carlo(p["rows"], 3, sims=2000)
    by = {r["Metod"]: r for r in mc}
    hist = by["Historiskt (faktisk ordning)"]
    assert hist["Avkastning p50 %"] == pytest.approx(p["return_pct"], abs=0.11)
    assert hist["DD p50 %"] == pytest.approx(p["closed_dd_pct"], abs=0.11)
    shuf = next(r for k, r in by.items() if k.startswith("Slumpad ordning"))
    assert shuf["CAGR p5 %"] == shuf["CAGR p95 %"]                      # samma affärer → samma slutresultat
    assert shuf["DD p95 %"] >= hist["DD p50 %"] - 1e-9 or shuf["DD p95 %"] >= shuf["DD p50 %"]
    boot = by["Bootstrap (slumpat urval)"]
    assert boot["CAGR p5 %"] <= boot["CAGR p50 %"] <= boot["CAGR p95 %"]
    assert boot["DD p50 %"] <= boot["DD p95 %"] <= boot["DD p99 %"]
    for r in mc:
        assert 0 <= r["P(förlust) %"] <= 100 and r["P(DD>10%) %"] >= r["P(DD>20%) %"]
    costly = next(r for k, r in by.items() if "kostnad" in k)
    assert costly["CAGR p50 %"] <= boot["CAGR p50 %"]
    assert rb.monte_carlo(p["rows"], 3, sims=500) == rb.monte_carlo(p["rows"], 3, sims=500)   # fast frö
    assert rb.monte_carlo(p["rows"][:3], 3) == []


# ── Statistik, koncentration, kantanalys och index ──────────────────────────
def test_ci_and_concentration(res):
    ci = rb.expectancy_ci(res["trades"])
    assert ci["low"] <= ci["mean"] <= ci["high"] and ci["n"] == res["metrics"]["trades"]
    c = rb.concentration(res["trades"])
    assert c["total_r"] == pytest.approx(res["metrics"]["total_r"], abs=0.01)
    assert len(c["top"]) == rb.TOP_N and c["top"][0][2] == max(t.r for t in res["trades"] if not t.open)
    assert c["total_without_top"] == pytest.approx(c["total_r"] - c["top_r"], abs=0.01)


def test_buckets():
    assert rb._bucket(1.0, (1.5, 2.5, 3.5), "%") == "< 1.5%"
    assert rb._bucket(2.0, (1.5, 2.5, 3.5), "%") == "1.5–2.5%"
    assert rb._bucket(4.0, (1.5, 2.5, 3.5), "%") == "≥ 3.5%"
    assert rb._bucket(None, (1.0,)) is None


def test_breakdown_covers_all_trades(res):
    bd = rb.breakdown(res["trades"])
    n = res["metrics"]["trades"]
    for key in ("År", "Exittyp", "ATR % av kursen", "Innehavstid (handelsdagar)"):
        assert sum(r["Affärer"] for r in bd[key]) == n, key
    assert {"Relativ volym", "Stängning i dagens spann (CLV)", "RS63 mot index (procentenheter)"} <= set(bd)


def test_benchmark_buy_and_hold():
    idx = pd.bdate_range("2020-01-01", "2022-01-01")
    s = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    s.iloc[100] = 80                                                     # ett ras
    b = rb.benchmark(s)
    assert b["CAGR %"] == pytest.approx(41.4, abs=0.6) and b["Max DD %"] > 30
    rows = rb.strategy_vs_benchmarks({"return_pct": 10.0, "cagr_pct": 5.0, "max_dd_pct": 4.0}, {"OMXS30": s})
    assert rows[0]["CAGR/DD"] == 1.25 and rows[1]["Vad"] == "Köp och behåll OMXS30"


# ── Listor och tickergränsen ────────────────────────────────────────────────
def test_oos_list_is_new_and_lists_fit_in_the_backtest():
    assert len(vb.NORDIC_OOS_100) == 100 == len(set(vb.NORDIC_OOS_100))
    assert not set(vb.NORDIC_OOS_100) & set(vb.NORDIC_50)
    assert all(t.endswith((".ST", ".OL", ".CO", ".HE")) for t in vb.NORDIC_OOS_100)
    from ovtlyr.ui.viking_nine_backtest import MAX_BACKTEST_TICKERS
    for name, lst in vb.TICKER_LISTS.items():
        assert len(vs.parse_tickers(", ".join(lst), limit=MAX_BACKTEST_TICKERS)) == len(set(lst)), name
    assert len(vs.parse_tickers(", ".join(vb.NORDIC_50 + vb.US_25))) == vs.MAX_TICKERS   # skannerns gräns kvar


def test_list_of():
    from ovtlyr.ui.viking_robustness_ui import list_of
    assert list_of("VOLV-B.ST") == "Norden 50" and list_of("ALFA.ST") == "Norden OOS 100"
    assert list_of("AAPL") == "USA 25" and list_of("XYZ") == "Egna"


def test_page_shows_robustness(monkeypatch, res):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("VNB_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["VNB_TEST_ROOT"])
        from ovtlyr.ui.viking_nine_backtest import render_viking_nine_backtest
        render_viking_nine_backtest()

    at = AppTest.from_function(app, default_timeout=120)
    at.session_state["vnb_result"] = {"selected": "X", "runs": {"X": dict(res)}}
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    for text in ("ROBUSTHET", "MOT KÖP OCH BEHÅLL", "KOSTNADER", "MONTE CARLO", "VAR KOMMER KANTEN IFRÅN",
                 "EXPECTANCY 95 %-INTERVALL", "KAPITAL I ARBETE", "dagsvärderad", "2023-09-30 – 2026-09-30"):
        assert text in html, text
    assert "Egen period" in at.selectbox(key="vnb_years").options
