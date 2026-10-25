"""Rule-köp i Norden (valhall/rule_study.py, rule_kop_study.py): grindar med fördröjning, katalysator, utfall."""
import numpy as np
import pandas as pd

from valhall import hate_study as hs
from valhall import rule_study as rs


def _idx(n, start="2010-01-01"):
    return pd.bdate_range(start, periods=n)


def test_spec_matches_the_code():
    text = open(rs.SPEC_PATH, encoding="utf-8").read()
    assert (rs.TOP_N, rs.CAT_SMA, rs.CAT_SLOPE, rs.CAT_AFTER) == (30, 50, 10, 63)
    assert (rs.ROIC_MIN, rs.ROIC_YEARS, rs.ROIC_MIN_YEARS, rs.DE_MAX, rs.MIN_N, rs.MIN_YEARS) == (8.0, 4, 2, 0.6, 30, 5)
    assert rs.KPIS == {"roic": 37, "fcf": 63, "de": 40, "ebitda": 54, "equity": 58}
    for w in ("30 största", "SMA50", "10 handelsdagar", "63 handelsdagar", "≥ 8", "< 0,6", "minst 30 händelser",
              "rule_kop_study.py"):
        assert w in text, w


def test_catalyst_turns_on_above_a_rising_sma50():
    c = pd.Series(np.r_[np.linspace(100, 60, 200), np.linspace(60, 90, 100)], index=_idx(300))
    cat = rs.catalyst(c)
    assert not cat.iloc[:200].any() and cat.iloc[-1]
    k = int(np.flatnonzero(cat.to_numpy())[0])
    sma = c.rolling(50).mean()
    assert c.iloc[k] > sma.iloc[k] and sma.iloc[k] > sma.iloc[k - 10]


def test_catalyst_day_window():
    idx = _idx(400)
    cat = pd.Series(False, index=idx)
    cat.iloc[300:] = True
    ep = {"start": idx[100], "end": idx[200]}
    assert rs.catalyst_day(cat, ep) is None                         # 300 > 200 + 63
    ep2 = {"start": idx[100], "end": idx[250]}
    assert rs.catalyst_day(cat, ep2) == idx[300]


def _hist(roic=(12, 10, 9, 15), fcf=50, de=0.4, ebitda=80, equity=500, last_year=2014):
    ys = range(last_year - len(roic) + 1, last_year + 1)
    one = lambda v: [{"y": y, "v": v} for y in ys]                  # noqa: E731
    return {"roic": [{"y": y, "v": v} for y, v in zip(ys, roic)], "fcf": one(fcf), "de": one(de),
            "ebitda": one(ebitda), "equity": one(equity)}


def test_gates_use_known_years_only():
    h = _hist()
    assert rs.gates(h, "2015-05-01") == {"quality": True, "balance": True}
    early = rs.gates(h, "2012-03-01")                               # bara 2010 känt → för få ROIC-år
    assert early["quality"] is None
    assert rs.gates(_hist(roic=(5, 6, 7, 20)), "2015-05-01")["quality"] is False   # median 6,5
    assert rs.gates(_hist(de=0.8), "2015-05-01")["balance"] is False
    assert rs.gates(_hist(fcf=-1), "2015-05-01")["balance"] is False
    assert rs.gates({**_hist(), "equity": []}, "2015-05-01")["balance"] is None


def test_event_rows_group_and_horizon():
    idx = pd.bdate_range("2014-01-01", periods=1200)
    c = pd.Series(np.r_[np.linspace(100, 60, 300), np.linspace(60, 120, 900)], index=idx)
    m = pd.Series(100.0, index=idx)
    ep = {"start": idx[150], "end": idx[330], "length": 181, "censored": False}
    good = rs.event_rows(c, m, [ep], _hist(last_year=2013))
    assert len(good) == 1 and good[0]["group"] == "rule" and good[0]["rel"] > 0
    bad = rs.event_rows(c, m, [ep], _hist(de=1.2, last_year=2013))
    assert bad[0]["group"] == "utan" and bad[0]["balance"] is False
    assert rs.event_rows(c, m, [{**ep, "start": idx[1000], "end": idx[1100]}], _hist()) == []


def _rows(gap_rule, gap_utan, n=50, years=8):
    rng = np.random.default_rng(5)
    out = []
    for i in range(n):
        y = 2009 + i % years
        out.append({"year": y, "group": "rule", "quality": True, "balance": True,
                    "rel": float(rng.normal(gap_rule, 0.05)), "rel_start": float(rng.normal(0, 0.05))})
        out.append({"year": y, "group": "utan", "quality": False, "balance": True,
                    "rel": float(rng.normal(gap_utan, 0.05)), "rel_start": None})
    return out


def test_summary_verdicts():
    rng = np.random.default_rng(0)
    s = rs.summary(_rows(0.10, -0.05), rng, 500)
    assert s["K1"]["pass"] and s["K2"]["pass"] and s["K3"]["n"] == 50 and s["fail_quality"] == 50
    s2 = rs.summary(_rows(-0.02, -0.10), rng, 500)
    assert not s2["K1"]["pass"] and s2["K2"]["pass"]
    s3 = rs.summary(_rows(0.10, -0.05, n=20), rng, 500)
    assert not s3["K1"]["pass"] and not s3["K2"]["pass"]


def test_script_runs(monkeypatch):
    import markets

    import rule_kop_study as rks
    from tests.test_hate_study import _Api
    monkeypatch.setattr(markets, "load", lambda api=None: {})
    monkeypatch.setattr(markets, "nordic_stock_ids", lambda table=None: {1})
    saved = {}
    res = rks.run(api=_Api(), today="2026-10-25", n_boot=100, save=lambda n, d: saved.setdefault(n, d) is not None)
    assert "error" not in res and "K1" in res and res["companies"] == 30
    assert rks.BLOB in saved and any("K2" in x for x in rks.report_lines(res))
