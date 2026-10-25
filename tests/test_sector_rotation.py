"""Sektorrotationen (odin/rotation_engine.py, odin/rotation_backtest.py): låsta regler, inget facit, rätt mått."""
import numpy as np
import pandas as pd
import pytest

from odin import rotation_backtest as rb
from odin import rotation_engine as re_


def _daily(n=3000, k=9, seed=0, start="1999-01-04", drifts=None):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    cols = {}
    for j in range(k):
        mu = (drifts or {}).get(j, 0.0002)
        cols[f"S{j}"] = 100 * np.exp(np.cumsum(rng.normal(mu, 0.01, n)))
    daily = pd.DataFrame(cols, index=idx)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, n))), index=idx)
    return daily, spy


def test_parameters_are_the_spec():
    assert re_.WEIGHTS == {12: 0.40, 6: 0.35, 3: 0.25} and (re_.SMA_MONTHS, re_.RS_SMA_MONTHS) == (10, 6)
    assert (rb.COST_SIDE, rb.TOP_N, rb.MIN_FUNDS, rb.SPLIT) == (0.0045, 3, 8, "2015-01-01")
    assert (rb.N_BOOT, rb.BLOCK, rb.SEED) == (5000, 12, 20261025)
    text = open(re_.SPEC_PATH, encoding="utf-8").read()
    for w in ("0,40 × R12 + 0,35 × R6 + 0,25 × R3", "45 bp", "20261025", "2015–", "S ≥ 60", "R3 ≤ 40"):
        assert w in text, w


def test_month_ends_are_real_trading_days():
    d, _ = _daily(300)
    m = re_.month_ends(d)
    assert all(x in d.index for x in m.index)
    assert all(m.index[i].month != m.index[i + 1].month for i in range(len(m) - 1))
    first = d.index[(d.index.year == d.index[0].year) & (d.index.month == d.index[0].month)]
    assert m.index[0] == first[-1]


def test_percentile_rank_is_zero_to_hundred():
    df = pd.DataFrame([[1.0, 3.0, 2.0], [5.0, np.nan, np.nan]])
    r = re_._pct_rank(df)
    assert list(r.iloc[0]) == [0.0, 100.0, 50.0]
    assert r.iloc[1, 0] == 100.0 and np.isnan(r.iloc[1, 1])


def test_signals_need_no_future_data():
    d, spy = _daily(2500, seed=3)
    full = re_.compute(d, spy)
    cut = d.index[1800]
    part = re_.compute(d.loc[:cut], spy.loc[:cut])
    j = len(part["S"]) - 2                                  # en hel månad före klippet
    date = part["S"].index[j]
    pd.testing.assert_series_equal(full["S"].loc[date], part["S"].loc[date])
    pd.testing.assert_series_equal(full["phase"].loc[date], part["phase"].loc[date])


def test_leader_and_lost_phases():
    d, spy = _daily(2000, k=9, seed=5, drifts={0: 0.0015, 1: -0.0015})
    sig = re_.compute(d, spy)
    last = sig["phase"].iloc[-1]
    assert last["S0"] == re_.F2                             # stark och stigande hela vägen
    assert last["S1"] == re_.F4                             # svag och fallande
    rows = re_.table(sig)
    s0 = next(r for r in rows if r["ticker"] == "S0")
    assert rows.index(s0) <= 1 and s0["R12"] == 100.0 and s0["f1"] and s0["f2"] and s0["f3"]


def test_rotation_holds_the_leader_and_pays_costs():
    d, spy = _daily(3000, k=9, seed=7, drifts={0: 0.0012})
    sig = re_.compute(d, spy)
    p = rb.simulate(d, sig, rb.w_rotation)
    assert len(p) > 100
    i0 = rb.start_index(sig)
    assert sig["eligible"].iloc[i0].sum() >= rb.MIN_FUNDS
    held = [rb.w_rotation(sig, i) for i in range(i0, len(sig["S"]))]
    assert sum("S0" in w for w in held) / len(held) > 0.6
    assert (p["net"] <= p["gross"] + 1e-12).all()
    assert p["turnover"].iloc[0] == pytest.approx(p["invested"].iloc[0])     # första köpet från kassa


def test_cost_math_on_a_known_period(monkeypatch):
    monkeypatch.setattr(rb, "MIN_FUNDS", 1)
    idx = pd.bdate_range("2020-01-01", periods=5)
    d = pd.DataFrame({"A": [100, 100, 110, 110, 121.0]}, index=idx)
    sig = {"S": pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx[[0, 1, 3]]),
           "eligible": pd.DataFrame({"A": [True] * 3}, index=idx[[0, 1, 3]])}
    p = rb.simulate(d, {**sig}, lambda s, i: {"A": 1.0}, cost=0.01, calendar=idx)
    assert p["gross"].iloc[0] == pytest.approx(0.10)                          # handel dagen efter: 100 → 110
    assert p["net"].iloc[0] == pytest.approx(0.99 * 1.10 - 1)
    assert p["turnover"].iloc[1] == pytest.approx(0.0)                       # samma vikt efter drift


def test_stats_and_bootstrap():
    idx = pd.date_range("2000-01-31", periods=24, freq="ME")
    p = pd.DataFrame({"start": idx[:-1], "end": idx[1:], "net": [0.01] * 23, "turnover": [0.5] * 23,
                      "invested": [1.0] * 23})
    s = rb.stats(p)
    assert s["max_dd"] == 0.0 and s["cagr"] > 12
    lo, hi = rb.block_boot(np.full(60, 0.01), np.random.default_rng(0), n=200)
    assert lo == pytest.approx(12.0) and hi == pytest.approx(12.0)


def _by(rot, spy, eq, mom, trend, dd_rot=20, dd_spy=50):
    row = lambda c, sh, dd=30: {"cagr": c, "sharpe": sh, "max_dd": dd}   # noqa: E731
    p = {rb.ROTATION: row(rot, 0.8), rb.SPY: row(spy, 0.5), rb.EQUAL: row(eq, 0.5), rb.MOM12: row(mom, 0.4),
         rb.TREND: row(trend, 0.4)}
    return {"Hela": {rb.ROTATION: row(rot, 0.8, dd_rot), rb.SPY: row(spy, 0.5, dd_spy)}, "2000–2014": p, "2015–": p}


def test_verdict_levels():
    v = rb.verdict(_by(9, 7, 7, 8, 6), [0.5, 4.0])
    assert v["R1"] and v["R2"] and v["R3"] and v["S"] and "sektorgrind" in v["use"]
    v2 = rb.verdict(_by(6, 7, 7, 8, 6), [-1.0, 4.0])
    assert not v2["R1"] and v2["R2"] and not v2["S"] and "riskfilter" in v2["use"]
    v3 = rb.verdict(_by(6, 7, 7, 8, 6, dd_rot=60), [-1.0, 4.0])
    assert not v3["R2"] and "ej validerad" in v3["use"]


def test_script_runs_end_to_end():
    import sector_rotation_backtest as srb

    def getter(sym, period):
        seed = sum(map(ord, sym))
        idx = pd.bdate_range("1998-12-01", "2026-10-23")
        c = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0.0003, 0.01, len(idx)))), index=idx)
        if sym in ("XLRE", "XLC"):
            c = c.loc["2016-01-01":]
        return pd.DataFrame({"Close": c, "Volume": 1e6})

    saved = {}
    res = srb.run(getter=getter, save=lambda n, d: saved.setdefault(n, d) is not None, n_boot=100)
    assert res["start"] and res["start"] < "2000-03-01"
    assert set(res["by_period"]) == {"Hela", "2000–2014", "2015–"}
    assert {r["ticker"] for r in res["latest"]} >= {"XLE", "XLC", "XLRE"}
    assert res["latest"][0]["label"] != res["latest"][0]["ticker"]
    assert srb.BLOB in saved and res["saved"]
    assert any("UTFALL" in x for x in rb.report_lines(res))
