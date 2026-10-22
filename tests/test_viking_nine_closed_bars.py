"""
Viking Nine räknar på STÄNGDA candles (playbooken): dagens halva stapel från Yahoo tas bort medan tickerns börs
är öppen — i Thor, i Viking Nine Regime och i schemat (samma motor). Kortet och motorn i Thor säger vilken
stängning de bygger på. Inget nätverk; klockslaget skickas in.
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ovtlyr_nine as on  # noqa: E402
import viking_screen as vs  # noqa: E402
from thor import candidates as tc  # noqa: E402
from thor import cards  # noqa: E402
from thor import regim  # noqa: E402

END = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=1)[-1]      # senaste handelsdag
PREV = pd.bdate_range(end=END, periods=2)[0]
OPEN = datetime(END.year, END.month, END.day, 10, 0)                            # börsen öppen (lokal tid)
CLOSED = datetime(END.year, END.month, END.day, 23, 30)                         # alla börser stängda


def _df(n=300):
    idx = pd.bdate_range(end=END, periods=n)
    c = 100 + np.arange(n) * 0.3
    return pd.DataFrame({"Open": c - 0.2, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1e6}, index=idx)


def test_closed_only_drops_todays_bar_while_the_exchange_is_open():
    df = _df()
    assert on.closed_only(df, "SCA-B.ST", OPEN).index[-1] == PREV                # Stockholm 10:00 — öppen
    assert on.closed_only(df, "SCA-B.ST", CLOSED).index[-1] == END               # efter 17:30 — stängd
    assert on.closed_only(df, "SPY", OPEN).index[-1] == PREV                     # New York 10:00 — öppen
    assert on.closed_only(df["Close"], "NEM", CLOSED).index[-1] == END
    old = _df().iloc[:-1]                                                        # slutar i går: rörs aldrig
    assert on.closed_only(old, "SCA-B.ST", OPEN).index[-1] == PREV and len(on.closed_only(old, "SPY", OPEN)) == len(old)
    assert on.closed_only(None, "X", OPEN) is None and len(on.closed_only(pd.DataFrame(), "X", OPEN)) == 0


def test_evaluate_and_market_layer_use_the_last_closed_bar():
    df = _df()
    getter = lambda t, p: _df()                                                  # noqa: E731 — SPY och ETF:erna
    nordic = lambda: {"close": _df()["Close"], "source": "test", "breadth": None}  # noqa: E731
    live = on.evaluate("SCA-B.ST", stock_df=df, getter=getter, sector_getter=lambda t: None, nordic_provider=nordic,
                       today=END, now=OPEN)
    assert all(f.timestamp == str(PREV)[:10] for f in live.stock if f.timestamp)
    assert all(f.timestamp == str(PREV)[:10] for f in live.market if f.timestamp)
    done = on.evaluate("SCA-B.ST", stock_df=df, getter=getter, sector_getter=lambda t: None, nordic_provider=nordic,
                       today=END, now=CLOSED)
    assert all(f.timestamp == str(END)[:10] for f in done.stock + done.market if f.timestamp)
    us = on.evaluate("NEM", stock_df=df, getter=getter, sector_getter=lambda t: None, today=END, now=OPEN)
    assert all(f.timestamp == str(PREV)[:10] for f in us.market + us.stock if f.timestamp)
    layer = on.market_layer(on.MARKET_TICKER, getter=getter, today=END, now=OPEN)
    assert all(f.timestamp == str(PREV)[:10] for f in layer if f.timestamp)
    assert vs.bar_date(live) == str(PREV)[:10] and vs.bar_date(done) == str(END)[:10]


def test_the_scan_row_and_thors_card_carry_the_closing_date():
    scan = {"generated": "2026-10-22T06:11:00+00:00",
            "rows": [{"ticker": "SCA-B.ST", "name": "SCA", "category": "READY", "ovtlyr_nine": "9/9",
                      "nine_weighted": 100.0, "entry": 57.67, "stop": 54.25, "rr": 1.2, "bar_date": "2026-10-21",
                      "reason": "Momentum: FAIL", "segment": "materials"}]}
    row = tc.from_viking_nine(None, {}, scan)[0]
    assert row.asof == "2026-10-22" and row.data_asof == "2026-10-21" and row.status_class == tc.BEVAKA
    html = cards.verdict_for(row, "viking_nine")
    assert "körd 2026-10-22 på stängning 2026-10-21" in html
    plain = tc.from_viking_nine(None, {}, {"generated": "2026-10-22", "rows": [{"ticker": "X", "category": "READY"}]})[0]
    assert plain.data_asof == "" and "körd 2026-10-22" in cards.verdict_for(plain, "viking_nine")
    assert "på stängning" not in cards.verdict_for(plain, "viking_nine")


def test_bars_caption_names_both_closings():
    nine = on.NineResult(ticker="SCA-B.ST", market_label="OMXS30")
    nine.market = [on.Factor("market.trend", "Trend", "market", on.PASS, timestamp="2026-10-21")]
    nine.stock = [on.Factor("stock.trend", "Trend", "stock", on.PASS, timestamp="2026-10-22")]
    text = regim.bars_caption(nine)
    assert "aktien t.o.m. 2026-10-22" in text and "marknaden (OMXS30) t.o.m. 2026-10-21" in text
    assert "Dagens stapel räknas först när börsen har stängt" in text
