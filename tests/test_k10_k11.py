"""
K10 — larmet vid bekräftad cykelvändning (alert_rules.turn_alerts) och K11 — Momentum-kontot följer
playbooken (stopp −10 %, halva vid +20 % och stoppen till entry, stängning under MA50) med likviditet och
minsta kurs. Inget nätverk.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alert_rules as ar  # noqa: E402
from test_valhall_paper import D, _board, _df, _sig  # noqa: E402
from valhall import paper as vpa  # noqa: E402

W = {"level": "komplex", "key": "adelmetaller", "label": "Ädelmetaller", "where": "komplex", "kind": "VÄNDER",
     "confirmed_since": "2026-10-16", "deep": [{"ticker": "GOLD.ST"}], "long": [{"ticker": "NEM.ST"}],
     "etf": {"ticker": "IGLN.L", "exposure": "fysiskt guld", "verified": False}}


def test_turn_alert_fires_once_per_confirmed_turn():
    board = {"turn_plan": {"window": [W]}}
    alerts, st = ar.turn_alerts(board, None)
    assert alerts == [] and st["rooms"] == {"komplex:adelmetaller:komplex": "2026-10-16"}      # baslinje
    alerts, st2 = ar.turn_alerts(board, {"rooms": {}})
    assert len(alerts) == 1 and "VÄNDNING: Ädelmetaller" in alerts[0]["title"]
    assert "GOLD.ST" in alerts[0]["body"] and "NEM.ST" in alerts[0]["body"] and "verifiera hos Nordnet" in alerts[0]["body"]
    assert ar.turn_alerts(board, st2)[0] == []                                                # samma vändning: tyst
    again = {"turn_plan": {"window": [{**W, "confirmed_since": "2027-01-10"}]}}
    assert len(ar.turn_alerts(again, st2)[0]) == 1                                             # ny vändning: larm
    assert ar.turn_alerts(None, st2) == ([], st2)
    out, state = ar.evaluate({}, {}, {}, [], {"turns": {"rooms": {}}}, board_data=board)
    assert any(a["kind"] == "cycle_turn" for a in out) and "turns" in state


def test_momentum_signal_needs_liquidity_and_a_real_price():
    wolf = {"A.ST": {"rank": 1, "setupA": True, "price": 50.0}, "B.ST": {"rank": 2, "setupA": True, "price": 0.06},
            "C.ST": {"rank": 3, "setupA": True, "price": 80.0}}
    uni = {t: {"segment": "industrials"} for t in wolf}
    sigs = vpa.signals({}, {}, uni, wolf=wolf, momentum_ok={"A.ST", "B.ST"})
    assert [s["ticker"] for s in sigs if s["strategy"] == "wolf"] == ["A.ST"]                   # B penny, C illikvid
    assert {s["ticker"] for s in vpa.signals({}, {}, uni, wolf=wolf) if s["strategy"] == "wolf"} == {"A.ST", "C.ST"}


def _run(prices, lows=None, highs=None):
    data = {"A.ST": _df(o=prices, c=prices, lo=lows, h=highs)}
    wolf = {"A.ST": {"rank": 1}}
    st, _ = vpa.step(None, [_sig("wolf", "A.ST")], _board(), lambda t: data[t].loc[:D[2]], today=D[2], wolf=wolf)
    st, _ = vpa.step(st, [], _board(), lambda t: data[t], today=D[-1], wolf=wolf)
    return st


def test_momentum_stop_minus_ten_percent():
    p = np.full(40, 100.0)
    lows = p - 1
    lows[10] = 85.0
    st = _run(p, lows=lows)
    c = st["closed"][0]
    assert c["exit"] == 90.0 and c["reason"] == "stopp −10 %" and c["init_stop"] == 90.0


def test_momentum_half_at_twenty_then_stop_at_entry():
    p = np.r_[np.full(5, 100.0), np.full(5, 121.0), np.full(30, 100.0)]
    highs = p + 1
    lows = p - 1
    lows[12] = 99.0
    st = _run(p, lows=lows, highs=highs)
    c = st["closed"][0]
    assert c["half"]["price"] == 121.0 and isinstance(c["half"]["price"], float)
    assert c["exit"] == (121.0 + 100.0) / 2 and c["reason"].startswith("stopp vid entry") and "halvan" in c["reason"]


def test_momentum_close_under_ma50_sells_next_open():
    p = np.r_[np.linspace(100, 120, 55), np.full(5, 105.0)]
    from test_valhall_paper import IDX
    import pandas as pd
    idx = pd.bdate_range("2026-01-01", periods=60)
    df = _df(o=p, c=p, n=60, idx=idx)
    wolf = {"A.ST": {"rank": 1}}
    d = [str(x.date()) for x in idx]
    st, _ = vpa.step(None, [_sig("wolf", "A.ST")], _board(), lambda t: df.loc[:d[1]], today=d[1], wolf=wolf)
    st, _ = vpa.step(st, [], _board(), lambda t: df, today=d[-1], wolf=wolf)
    c = st["closed"][0]
    assert c["reason"].startswith("stängning under MA50") and IDX is not None
