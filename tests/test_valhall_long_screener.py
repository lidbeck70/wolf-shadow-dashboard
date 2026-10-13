"""
Valhall fas 4c — Long Screener i Valhall: cykelpoängen ur sektorindex + hemmamarknadens index, screenerns
egen poäng (cagr, oförändrad), korten och papperskontots sälj på screenerns SELL. Syntetiska data.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valhall import cycle  # noqa: E402
from valhall import long_screener as ls  # noqa: E402
from valhall import paper as vpa  # noqa: E402
from valhall import segments as sgm  # noqa: E402

IDX = pd.bdate_range(end="2026-10-09", periods=400)


def _close(drift):
    return pd.Series(100 * (1 + drift) ** np.arange(400), index=IDX)


def _board(status):
    return {"segments": [{"key": s.key, "nordic": {"status": status.get(s.key, cycle.ROD)}, "themes": []}
                         for s in sgm.SEGMENTS]}


GOOD = {"_data_source": "borsdata", "revenue_growth": 0.20, "earnings_growth": 0.25, "revenue_cagr_5y": 0.12,
        "returnOnEquity": 0.25, "fcf_stability": 80, "f_score": 8, "operating_margin": 0.20,
        "enterpriseToEbitda": 10, "priceToBook": 3, "trailingPE": 15, "rs_rank": 90, "magic_formula": 10,
        "net_debt_ebitda": 1.0, "current_ratio": 1.5, "equity_ratio": 0.5, "earnings_stability": 80,
        "dividend_yield": 0.02}


def test_market_index_and_regime():
    assert ls.market_index("VOLV-B.ST") == "^OMX" and ls.market_index("EQNR.OL") == "OBX.OL"
    assert ls.market_index("NOVO-B.CO") == "^OMXC25" and ls.market_index("NOKIA.HE") == "^OMXH25"
    assert ls.market_regime(_close(0.001)) == cycle.GRON
    assert ls.market_regime(_close(-0.001)) == cycle.ROD
    assert ls.market_regime(_close(0.001).iloc[:100]) == cycle.NODATA and ls.market_regime(None) == cycle.NODATA


def test_cycle_score_mapping():
    assert ls.cycle_score(cycle.GRON, cycle.GRON) == 3 and ls.cycle_score(cycle.GRON, cycle.GUL) == 2
    assert ls.cycle_score(cycle.GUL, cycle.GRON) == 2 and ls.cycle_score(cycle.GUL, cycle.GUL) == 1
    assert ls.cycle_score(cycle.ROD, cycle.GRON) == 1 and ls.cycle_score(cycle.NODATA, cycle.NODATA) == 1
    assert ls.cycle_score(cycle.GRON, cycle.ROD) == 0                                    # röd regim = sälj


def test_score_uses_screener_unchanged():
    up = ls.score(_close(0.001), GOOD, 3)
    assert up["signal"] in ls.BUY_SIGNALS and up["fund_max"] == 20 and up["cycle"] == 3
    assert ls.score(_close(0.001), GOOD, 1)["signal"] == "HOLD"                          # köpgrinden: cykel ≥ 2
    assert ls.score(_close(0.001), GOOD, 0)["signal"] in ls.SELL_SIGNALS                 # regel 7
    assert ls.score(_close(-0.001), GOOD, 3)["signal"] in ls.SELL_SIGNALS                # regel 6: under EMA200


def test_scan_picks_per_segment():
    uni = {"A.ST": {"segment": "technology", "name": "A"}, "B.ST": {"segment": "technology", "name": "B"},
           "C.ST": {"segment": "technology", "name": "C"}, "D.ST": {"segment": "industrials", "name": "D"},
           "E.ST": {"segment": "technology", "name": "E"}}
    data = {"A.ST": pd.DataFrame({"Close": _close(0.001)}), "B.ST": pd.DataFrame({"Close": _close(0.0012)}),
            "C.ST": pd.DataFrame({"Close": _close(0.0011)}), "D.ST": pd.DataFrame({"Close": _close(0.001)}),
            "E.ST": pd.DataFrame({"Close": _close(-0.001)})}
    fund = {t: dict(GOOD) for t in uni}
    fund["C.ST"] = {**GOOD, "returnOnEquity": 0.01, "f_score": 2, "operating_margin": 0.01}
    asked = []
    out = ls.scan(_board({"technology": cycle.GRON, "industrials": cycle.ROD}), uni, data, sorted(uni),
                  lambda ts: asked.append(ts) or fund, lambda sym: _close(0.001))
    assert asked == [tuple(sorted(uni))] and out["markets"] == {"^OMX": cycle.GRON} and out["scanned"] == 5
    tech = out["segments"]["technology"]
    assert tech["sector_status"] == cycle.GRON and len(tech["picks"]) == 2
    assert {p["ticker"] for p in tech["picks"]} <= {"A.ST", "B.ST", "C.ST"}                      # E under EMA200
    assert tech["picks"][0]["source"] == "Long Screener" and tech["picks"][0]["cycle"] == 3
    assert out["segments"]["industrials"]["picks"] == []                                 # RÖD sektor → cykel 1
    assert out["signals"]["E.ST"] in ls.SELL_SIGNALS and out["signals"]["D.ST"] == "HOLD"
    red = ls.scan(_board({"technology": cycle.GRON}), uni, data, sorted(uni), lambda ts: fund,
                  lambda sym: _close(-0.001))
    assert red["markets"] == {"^OMX": cycle.ROD} and red["buys"] == 0


def test_paper_long_account_buys_picks_and_sells_on_sell_signal():
    long_scan = {"segments": {"technology": {"picks": [{"ticker": "A.ST", "name": "A", "signal": "STRONG BUY",
                                                        "score_pct": 0.8, "cycle": 3}]}}}
    sigs = vpa.signals({}, {}, {}, long_scan=long_scan)
    assert [(s["strategy"], s["ticker"], s["nordic"]) for s in sigs] == [("long", "A.ST", True)]
    idx = pd.bdate_range("2026-01-01", periods=20)
    d = [str(x.date()) for x in idx]
    df = pd.DataFrame({"Open": np.arange(100, 120.0), "High": 130.0, "Low": 90.0, "Close": np.arange(100, 120.0)},
                      index=idx)
    board = _board({"technology": cycle.GRON})
    st, _ = vpa.step(None, sigs, board, lambda t: df.loc[:d[1]], today=d[1])
    st, _ = vpa.step(st, [], board, lambda t: df.loc[:d[2]], today=d[2], long_signals={"A.ST": "STRONG BUY"})
    st, _ = vpa.step(st, [], board, lambda t: df.loc[:d[3]], today=d[3], long_signals=None)   # skanning saknas
    assert st["positions"] and not st["positions"][0]["pending"]
    st, ev = vpa.step(st, [], board, lambda t: df.loc[:d[4]], today=d[4], long_signals={"A.ST": "SELL"})
    assert ev[0]["kind"] == vpa.SALJ_OPEN and "Long Screener SELL" in ev[0]["text"]
    st, _ = vpa.step(st, [], board, lambda t: df.loc[:d[5]], today=d[5], long_signals={"A.ST": "SELL"})
    c = st["closed"][0]
    assert c["entry"] == 102.0 and c["exit"] == 105.0 and c["reason"] == "Long Screener SELL"
    assert vpa.results(st)["by_strategy"][0]["strategy"] == "long"
