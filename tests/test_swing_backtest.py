"""
📈 Momentum Swing-backtest: veckorutinen som portfölj utan look-ahead, med
screenerns egna regler (wolf_data). Syntetiska kurser — inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import swing_backtest as sb  # noqa: E402
import wolf_data as wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = pd.bdate_range(end="2026-09-30", periods=1900)


def _universe(n=80, seed=0, idx_drift=0.0004):
    rng = np.random.default_rng(seed)
    prices = {}
    for i in range(n):
        dr = rng.normal(0.0003, 0.0004)
        c = 100 * np.exp(np.cumsum(rng.normal(dr, 0.018, len(IDX))))
        o = c * np.exp(rng.normal(0, 0.004, len(IDX)))
        h = np.maximum(c, o) * (1 + abs(rng.normal(0, 0.008, len(IDX))))
        lo = np.minimum(c, o) * (1 - abs(rng.normal(0, 0.008, len(IDX))))
        prices[f"S{i}"] = pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c, "Volume": 1e6}, index=IDX)
    ix = pd.Series(100 * np.exp(np.cumsum(rng.normal(idx_drift, 0.008, len(IDX)))), index=IDX)
    return prices, ix


PRICES, INDEX = _universe()
PN = sb.panels(PRICES, INDEX)
RES = sb.run(PN, INDEX, sb.Config(years=5))


def test_indicators_are_the_screeners_own():
    ind = sb.indicators(PN["C"])
    d = IDX[-300]
    for t in ("S0", "S7", "S33"):
        m = wd.metrics_for(PRICES[t]["Close"].loc[:d])
        qualifies = m["px"] > m["ma200"] and m["r3"] > 0 and m["r6"] > wd.CONFIG["MOM_LONG_MIN"]
        assert ind["score"].at[d, t] == pytest.approx(0.5 * m["r3"] + 0.5 * m["r6"])
        assert bool(ind["qualifies"].at[d, t]) == qualifies
        near_ma = abs(m["px"] / m["ma20"] - 1) <= 0.02 or abs(m["px"] / m["ma50"] - 1) <= 0.02
        assert bool(ind["setup_a"].at[d, t]) == bool(qualifies and near_ma and 35 <= m["rsi"] <= 55)


def test_regime_uses_the_swing_regime_function():
    reg = sb.regimes(INDEX, sb.indicators(PN["C"])["breadth"])
    assert set(reg.unique()) <= {sb.GREEN, sb.YELLOW, sb.RED, sb.UNKNOWN}
    assert reg.iloc[0] == sb.UNKNOWN                                   # MA200 saknas i början
    assert (sb.regimes(None, reg.map(lambda _: 0.5)) == sb.UNKNOWN).all()


def test_entries_are_next_open_after_a_week_end():
    weekly = sb.week_ends(PN["C"].index)
    closed = [t for t in RES["trades"] if not t.open]
    assert len(closed) >= 20
    for t in RES["trades"]:
        d = pd.Timestamp(t.entry_date)
        prev = PN["C"].index[PN["C"].index.get_loc(d) - 1]
        assert prev in weekly                                          # signal på veckans sista dag
        assert t.entry == pytest.approx(PN["O"].at[d, t.ticker], abs=1e-3)


def test_no_look_ahead():
    cut = IDX[-400]
    trunc = {t: df.loc[:cut] for t, df in PRICES.items()}
    r_cut = sb.run(sb.panels(trunc, INDEX.loc[:cut]), INDEX.loc[:cut], sb.Config(years=10), today=cut)
    r_full = sb.run(PN, INDEX, sb.Config(years=10))
    early = lambda trades: sorted((t.ticker, t.entry_date, t.entry) for t in trades  # noqa: E731
                                  if pd.Timestamp(t.entry_date) < cut - pd.Timedelta(days=7))
    assert early(r_cut["trades"]) and early(r_cut["trades"]) == early(r_full["trades"])


def test_position_and_weekly_buy_limits():
    trades = RES["trades"]
    for d in IDX[-1250:]:
        held = [t for t in trades if pd.Timestamp(t.entry_date) <= d and
                (t.open or pd.Timestamp(t.exit_date) > d)]
        assert len(held) <= 8
    by_week = pd.Series(1, index=[pd.Timestamp(t.entry_date) for t in trades]).groupby(
        lambda d: (d.isocalendar()[0], d.isocalendar()[1])).sum()
    assert by_week.max() <= 2


def test_exit_rules():
    reasons = {t.reason for t in RES["trades"] if not t.open}
    assert reasons <= {"stopp −10 %", "breakeven-stopp", "under MA50", "ur topp 40"}
    fee = 2 * sb.Config().fee_pct
    for t in RES["trades"]:
        if t.reason == "stopp −10 %":
            assert t.ret_pct <= -10 + 0.01 and not t.half_sold                 # stop (eller gap under)
        if t.reason == "breakeven-stopp":
            assert t.half_sold and t.ret_pct >= 10 - fee - 0.5                 # halva +20 %, resten ±0
        if t.half_sold and not t.open:
            assert t.ret_pct > -1


def test_red_regime_blocks_buys():
    prices, falling = _universe(n=40, seed=3, idx_drift=-0.001)
    pn = sb.panels(prices, falling)
    blocked = sb.run(pn, falling, sb.Config(years=5))
    free = sb.run(pn, falling, sb.Config(years=5, regime_filter=False))
    assert blocked["metrics"]["trades"] == 0 and blocked["regime_share"].get(sb.RED, 0) > 0.9
    assert free["metrics"]["trades"] > 0


def test_variants_and_metrics():
    m = RES["metrics"]
    for k in ("total_return_pct", "cagr_pct", "max_dd_pct", "exposure_pct", "win_rate", "payoff", "avg_days"):
        assert m.get(k) is not None, k
    assert 0 <= m["exposure_pct"] <= 100 and m["max_dd_pct"] >= 0
    assert RES["benchmark"]["cagr_pct"] is not None and len(RES["equity"]) > 1000
    no_setup = sb.run(PN, INDEX, sb.Config(years=5, require_setup=False))
    assert no_setup["metrics"]["trades"] >= m["trades"] * 0.5
    assert set(sb.VARIANTS) == {"Reglerna", "Utan setup-krav", "Utan regimfilter"}
    rows = sb.exit_table(RES["trades"])
    assert sum(r["Antal"] for r in rows) == m["trades"]


def test_load_data_from_borsdata_and_fallback():
    class _Api:
        def get_instruments(self):
            return [{"insId": 1, "ticker": "AAA", "marketId": 1, "instrumentType": 0},
                    {"insId": 2, "ticker": "BBB", "marketId": 2, "instrumentType": 0},
                    {"insId": 3, "ticker": "CCC", "marketId": 3, "instrumentType": 0},          # Small Cap
                    {"insId": 9, "ticker": "OMXSPI", "name": "OMX Stockholm PI", "marketId": 7}]

        def get_stockprices_df(self, ins, max_count=None):
            return PRICES["S0"] if ins in (1, 2, 9) else pd.DataFrame()
    data = sb.load_data(5, api=_Api())
    assert set(data["prices"]) == {"AAA", "BBB"} and data["index"] is not None
    assert "Large + Mid Cap" in data["source"]
    fb = sb.load_data(5, api=None, yahoo_getter=lambda t, p: PRICES["S1"])
    assert fb["prices"] and fb["prices"].keys() <= {t for t in __import__("viking_backtest").NORDIC_50}
    assert "RESERV" in fb["source"] and all(t.endswith(".ST") for t in fb["prices"])


def test_backtest_mode_and_page(monkeypatch):
    src = open(os.path.join(ROOT, "tabs", "backtest.py"), encoding="utf-8").read()
    assert '"📈 Momentum Swing"' in src and "render_swing_backtest_page" in src
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("SBT_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["SBT_TEST_ROOT"])
        import swing_backtest as _sb
        from tests.test_swing_backtest import INDEX as _I, PRICES as _P
        _sb.load_data = lambda years, **k: {"prices": _P, "index": _I, "source": "test", "universe": len(_P),
                                            "missing": 0}
        from swing_backtest_ui import render_swing_backtest_page
        render_swing_backtest_page()

    at = AppTest.from_function(app, default_timeout=180)
    at.run()
    at.button(key="FormSubmitter:sbt_form-📈 Kör backtest").click().run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "CAGR" in html and "PAYOFF-KVOT" in html and "EXITORSAKER" in html and "OMXSPI" in html
    runs = at.session_state["sbt_result"]["runs"]
    assert set(runs) == set(sb.VARIANTS)
    at.selectbox(key="sbt_show").set_value("Utan regimfilter").run()
    assert not at.exception
