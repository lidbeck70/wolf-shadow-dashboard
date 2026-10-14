"""
Valhall fas 4c — Long Screener punkt för punkt: nyckeltalen som de var kända då, RS-rank och Magic Formula
ur tvärsnittet, cykelvarianterna, signalerna och screenerns portfölj. Syntetiska data — inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valhall import cycle  # noqa: E402
from valhall import long_backtest as lb  # noqa: E402
from valhall import long_screener as ls  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = pd.bdate_range("2012-01-02", "2019-12-31")


def _close(rate=0.15, idx=IDX):
    return pd.Series(100 * (1 + rate) ** (np.arange(len(idx)) / 252), index=idx)


def _hist(good=True):
    v = (lambda x, bad: x if good else bad)
    yrs = range(2005, 2019)
    one = lambda x: [{"y": y, "v": x} for y in yrs]                                      # noqa: E731
    return {"returnOnEquity": one(v(25.0, 2.0)), "operating_margin": one(v(20.0, 1.0)), "f_score": one(v(8, 2)),
            "fcf_stability": one(v(80, 10)), "enterpriseToEbitda": one(v(10, 40)), "priceToBook": one(v(3, 9)),
            "trailingPE": one(v(15, 80)), "revenue_growth": one(v(20.0, -5.0)), "earnings_growth": one(v(25.0, -10.0)),
            "net_debt_ebitda": one(v(1.0, 6.0)), "current_ratio": one(v(1.5, 0.5)), "equity_ratio": one(v(50.0, 5.0)),
            "earnings_stability": one(v(80, 10)), "dividend_yield": one(v(2.0, 0.0)),
            "revenue_m": [{"y": y, "v": 100 * 1.15 ** (y - 2005)} for y in yrs],
            "ev_ebit": one(v(8.0, 50.0)), "roc": one(v(30.0, 1.0)), "roic": one(v(20.0, 1.0))}


def test_fundamentals_at_lag_scaling_and_cagr():
    h = {"returnOnEquity": [{"y": 2015, "v": 25.0}, {"y": 2016, "v": 5.0}],
         "revenue_m": [{"y": 2010, "v": 100.0}, {"y": 2015, "v": 200.0}, {"y": 2016, "v": 400.0}],
         "roic": [{"y": 2014, "v": 9.0}, {"y": 2015, "v": 20.0}]}
    a = lb.fundamentals_at(h, "2016-04-29")                                             # 2015 känt först 1 maj
    assert "returnOnEquity" not in a and a["_roic_hist"] == [9.0] and a["_data_source"] == "borsdata"
    b = lb.fundamentals_at(h, "2016-05-02")
    assert b["returnOnEquity"] == pytest.approx(0.25) and b["_roic_hist"] == [9.0, 20.0]
    assert b["revenue_cagr_5y"] == pytest.approx(2 ** 0.2 - 1)
    assert "revenue_cagr_5y" not in lb.fundamentals_at(h, "2017-06-01")                 # 2011 saknas
    assert set(lb.kpi_ids()) == set(lb.KPIS) | set(lb.EXTRA)


def test_cross_section_rs_and_magic():
    idx = pd.bdate_range("2015-01-01", "2016-12-30")
    closes = {"A": _close(0.40, idx), "B": _close(0.0, idx), "C": _close(-0.30, idx)}
    infos = {"A": {"_ev_ebit": 5.0, "_roc": 30.0}, "B": {"_ev_ebit": 20.0, "_roc": 10.0}, "C": {"_ev_ebit": None, "_roc": 5.0}}
    lb.cross_section(infos, closes, "2016-12-30")
    assert infos["A"]["rs_rank"] == 100 and infos["C"]["rs_rank"] < infos["B"]["rs_rank"]
    assert infos["A"]["magic_formula"] < infos["B"]["magic_formula"] and "magic_formula" not in infos["C"]


def test_cycle_variants_and_market_regimes():
    assert lb.cycle_for("ingen", "technology", cycle.ROD, cycle.ROD) == 3
    assert lb.cycle_for("valhall", "technology", cycle.GRON, cycle.GRON) == 3
    assert lb.cycle_for("valhall", "technology", cycle.GRON, cycle.ROD) == 0
    assert lb.cycle_for("forval", "technology", cycle.GRON, cycle.GRON) == 0             # SECTOR_CONFIG: Technology 0
    assert lb.cycle_for("forval", "energy", cycle.ROD, cycle.ROD) == 3
    d = [pd.Timestamp("2012-03-30"), pd.Timestamp("2015-06-30")]
    reg = lb.market_regimes({"^OMX": _close(0.10), "OBX.OL": _close(-0.10), "X": None}, d)
    assert reg["^OMX"][d[0]] == cycle.NODATA and reg["^OMX"][d[1]] == cycle.GRON and reg["OBX.OL"][d[1]] == cycle.ROD
    assert reg["X"] == {}


def _rows_setup():
    closes = {"A.ST": _close(0.20), "B.ST": _close(0.10), "C.ST": _close(-0.15)}
    hist = {"A.ST": _hist(True), "B.ST": _hist(True), "C.ST": _hist(False)}
    uni = {t: {"segment": "technology"} for t in closes}
    vol = pd.Series(1e5, index=IDX)
    turnover = {t: c * vol for t, c in closes.items()}
    frames = {"technology": pd.DataFrame({"status": cycle.GRON, "phase": cycle.STARK, "cykel": cycle.MITTEN,
                                          "cycle_start": False}, index=IDX)}
    return closes, hist, uni, turnover, frames


def test_signals_rows_per_month_and_variant():
    closes, hist, uni, turnover, frames = _rows_setup()
    dates = [pd.Timestamp("2016-06-30"), pd.Timestamp("2016-12-30")]
    seen = []
    rows = lb.signals(closes, turnover, uni, hist, frames, {"^OMX": _close(0.10)}, dates,
                      progress=lambda i, n, k: seen.append((i, n, k)))
    assert len(rows) == 6 and seen[-1] == (2, 2, 3)
    by = {(r["ticker"], r["date"]): r for r in rows}
    a = by[("A.ST", "2016-06-30")]
    assert a["sig_valhall"] in ls.BUY_SIGNALS and a["sig_ingen"] in ls.BUY_SIGNALS and a["market"] == cycle.GRON
    assert a["sig_forval"] == "STRONG SELL" or a["sig_forval"] in ls.SELL_SIGNALS        # Technology-förval 0 = sälj
    assert a["q"] and a["status"] == cycle.GRON and a["fund"] >= 15
    c = by[("C.ST", "2016-06-30")]
    assert c["sig_ingen"] in ls.SELL_SIGNALS and not c["q"]
    thin = lb.signals(closes, {t: c * 0 for t, c in closes.items()}, uni, hist, frames, {}, dates)
    assert thin == []                                                                   # illikvida räknas inte
    lb.add_returns(rows, closes, _close(0.10))
    assert rows[0]["r1"] is not None and rows[0]["x12"] == pytest.approx(rows[0]["r12"] - (1.10 - 1) * 100, abs=0.5)


def test_fund_max_rescales_thresholds():
    closes, hist, uni, turnover, frames = _rows_setup()
    for f in lb.OPTIONAL_HISTORY:                                                        # som i Börsdata-licensen
        for h in hist.values():
            h.pop(f, None)
    d = [pd.Timestamp("2016-06-30")]
    full = lb.signals(closes, turnover, uni, hist, frames, {"^OMX": _close(0.10)}, d)
    fair = lb.signals(closes, turnover, uni, hist, frames, {"^OMX": _close(0.10)}, d, fund_max=16)
    a_full = next(r for r in full if r["ticker"] == "A.ST")
    a_fair = next(r for r in fair if r["ticker"] == "A.ST")
    assert a_fair["pct_ingen"] > a_full["pct_ingen"] and a_fair["fund"] == a_full["fund"]


def _prow(t, d, sig, r1, seg="technology", pct=0.8):
    return {"ticker": t, "date": d, "segment": seg, "sig_v": sig, "pct_v": pct, "r1": r1}


def test_portfolio_rules():
    d1, d2, d3 = "2016-01-29", "2016-02-29", "2016-03-31"
    rows = [_prow("A", d1, "STRONG BUY", 10.0), _prow("B", d1, "BUY", 0.0), _prow("C", d1, "BUY", 0.0, seg="energy"),
            _prow("D", d1, "BUY", 0.0, pct=0.6),
            _prow("A", d2, "SELL", -50.0), _prow("B", d2, "HOLD", 0.0), _prow("C", d2, "HOLD", 0.0),
            _prow("A", d3, "HOLD", 0.0)]
    p = lb.portfolio(rows, "v")
    # mån 1: A 10 % + B 7 % + D 7 % i tech (24 % ≤ 25 %) och C i energi; mån 2 säljs A på SELL
    assert p["months"] == 3 and p["buys"] == 4
    assert p["cagr"] > 0 and p["dd"] == 0.0                                            # A såld före −50 %
    capped = [_prow(t, d1, "STRONG BUY", 0.0) for t in "ABCD"]
    q = lb.portfolio(capped, "v")
    assert q["buys"] == 2                                                              # 10 + 10 = 20, en till > 25 %
    many = [_prow(f"T{i}", d1, "BUY", 0.0, seg=f"s{i}") for i in range(14)]
    assert lb.portfolio(many, "v")["avg_n"] == 10.0                                    # högst 10 innehav
    assert lb.portfolio([], "v") == {"months": 0}


def test_buy_table_and_portfolios_shape():
    rows = [{"ticker": "A", "date": "2016-01-29", "period": "2008–2020", "segment": "technology", "status": cycle.GRON,
             "r1": 1.0, "x12": 5.0, "r12": 10.0, **{f"sig_{v}": "BUY" for v in lb.VARIANTS},
             **{f"pct_{v}": 0.6 for v in lb.VARIANTS}}]
    t = lb.buy_table(rows)
    assert {r["variant"] for r in t} == set(lb.VARIANTS) | {"alla likvida"}
    assert any(r.get("status") == cycle.GRON for r in t)
    assert {(r["variant"], r["period"]) for r in lb.portfolios(rows, None)} >= {("valhall", "alla"), ("ingen", "2021–")}


def test_fas4c_runner(monkeypatch):
    import valhall_fas3 as fas3
    import valhall_fas4b as fas4b
    import valhall_fas4c as vf
    closes, hist, uni, turnover, frames = _rows_setup()
    bench = _close(0.08)
    fr = {"nordic": frames, "global": {}, "themes": {}, "bench": {"nordic": bench, "global": bench},
          "universe": {t: {"segment": "technology", "currency": "SEK"} for t in closes}}
    monkeypatch.setattr(fas3, "frames", lambda api, getter, data_getter, nordic_market=None, keep=None: fr)
    monkeypatch.setattr(fas4b, "stock_inputs", lambda api, fr_, keep: (
        {t: {"segment": "technology", "ins_id": i} for i, t in enumerate(closes)}, closes, turnover))
    by_id = {i: hist[t] for i, t in enumerate(closes)}
    kid_field = {v: k for k, v in lb.kpi_ids().items()}
    out = vf.run(api=SimpleNamespace(), getter=lambda t, p: pd.DataFrame({"Close": _close(0.10)}),
                 data_getter=lambda ts, p: {},
                 kpi_getter=lambda ids, kid: {i: by_id[i][kid_field[kid]] for i in ids}, no_save=True)
    assert out["counts"]["with_kpis"] == 3 and out["counts"]["rows"] > 0 and out["counts"]["buys_ingen"] > 0
    assert out["fund_max"] == 19 and out["missing"] == ["insider"]
    no_hist = {lb.kpi_ids()[f] for f in lb.OPTIONAL_HISTORY}                             # licensen saknar historik
    out2 = vf.run(api=SimpleNamespace(), getter=lambda t, p: pd.DataFrame({"Close": _close(0.10)}),
                  data_getter=lambda ts, p: {}, no_save=True,
                  kpi_getter=lambda ids, kid: {} if kid in no_hist else {i: by_id[i][kid_field[kid]] for i in ids})
    assert out2["fund_max"] == 16 and out2["missing"] == ["insider", *lb.OPTIONAL_HISTORY]
    assert any(r["variant"].startswith("ROIC") for r in out["buys"])
    assert any(r["variant"] == "valhall" and r.get("months") for r in out["portfolios"])
    import json
    json.dumps(out, default=str)


def test_longscreener_workflow():
    src = open(os.path.join(ROOT, ".github", "workflows", "valhall-longscreener.yml"), encoding="utf-8").read()
    assert "workflow_dispatch" in src and "python valhall_fas4c.py" in src and "BORSDATA_API_KEY" in src


def test_green_only_variants():
    g = lambda s, m: lb.cycle_for("gron", "technology", s, m)                          # noqa: E731
    assert g(cycle.GRON, cycle.GRON) == 3 and g(cycle.GRON, cycle.GUL) == 2
    assert g(cycle.GUL, cycle.GRON) == 1 and g(cycle.ROD, cycle.GRON) == 1           # GUL köper inte (valhall: 2)
    assert g(cycle.GRON, cycle.ROD) == 0 and lb.cycle_for("valhall", "technology", cycle.GUL, cycle.GRON) == 2
    n = lambda s, m: lb.cycle_for("gron_utan_marknad", "technology", s, m)             # noqa: E731
    assert n(cycle.GRON, cycle.ROD) == 3 and n(cycle.GUL, cycle.GRON) == 1 and n(cycle.ROD, cycle.ROD) == 1
    assert set(lb.VARIANTS) >= {"gron", "gron_utan_marknad"}
