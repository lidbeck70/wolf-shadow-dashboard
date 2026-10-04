"""
Viking Nine live: OVTLYR:s breddregler i OVTLYR Nine (båda marknaderna) och
en aktie per sektor i skannern (SEKTOR UPPTAGEN). Backtestets förval följer
live. Syntetiska kurser och fejkade positioner — inget nätverk.
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ovtlyr_nine as on  # noqa: E402
import viking_backtest as vb  # noqa: E402
import viking_portfolio as vp  # noqa: E402
import viking_screen as vs  # noqa: E402
from test_viking_screen import END, NOW, _getter, _sector  # noqa: E402


def _b(values):
    return pd.Series(values, index=pd.bdate_range("2026-01-05", periods=len(values)), dtype=float)


# ── Breddregeln (live) ──────────────────────────────────────────────────────
def test_breadth_rule_cases():
    assert on.breadth_ok(_b(np.linspace(30, 60, 40))).iloc[-1]                       # ökar
    assert not on.breadth_ok(_b(np.linspace(60, 30, 40))).iloc[-1]                   # krymper
    assert on.breadth_ok(_b([82.0] * 60)).iloc[-1]                                   # ligger still = OK
    low = [40] * 20 + list(np.linspace(40, 10, 15))
    assert on.breadth_ok(_b(low + [14, 19])).iloc[-1]                                # < 25, vänder upp
    assert not on.breadth_ok(_b(low + [14, 19, 19])).iloc[-1]                        # < 25 utan uppvändning
    high = list(np.linspace(50, 90, 30))
    assert not on.breadth_ok(_b(high + [89])).iloc[-1]                               # > 75 och nedvänd = stopp
    assert on.breadth_ok(_b(high + [91])).iloc[-1]


def test_breadth_check_explains_the_rule():
    ok, value, detail = on._breadth_check("aktierna")(_b(list(np.linspace(50, 90, 30)) + [89]))
    assert not ok and value == 89 and "över 75" in detail and "inga nya affärer" in detail
    ok, _v, detail = on._breadth_check("aktierna")(_b(np.linspace(30, 60, 40)))
    assert ok and "ökar" in detail and "EMA10" in detail


def test_backtest_defaults_follow_live():
    assert vb.Config().ovt_breadth is True and vb.ovtlyr_breadth_ok is on.breadth_ok
    assert next(iter(vb.BREADTH_RULES)) == "OVTLYR (som live)" and vb.BREADTH_RULES["OVTLYR (som live)"]
    assert vp.PortfolioConfig().one_per_sector is True


# ── En aktie per sektor (live) ──────────────────────────────────────────────
def test_held_sectors_from_viking_positions():
    rows = [{"ticker": "msft"}, {"ticker": "NVDA"}, {"ticker": "XOM"}, {"ticker": ""}, {"ticker": "UNKNOWN"}]
    sectors = {"MSFT": "Technology", "NVDA": "Technology", "XOM": "Energy"}
    held = vs.held_sectors(sector_getter=sectors.get, positions_getter=lambda: rows, bd_sector=lambda t: None)
    assert held == {"XLK": "MSFT", "XLE": "XOM"}                                   # första i sektorn, okänd hoppas


def test_sector_busy():
    held = {"XLK": "MSFT"}
    assert vs.sector_busy("NVDA", "XLK", held) == "MSFT"
    assert vs.sector_busy("MSFT", "XLK", held) is None                              # aktien själv
    assert vs.sector_busy("XOM", "XLE", held) is None and vs.sector_busy("X", None, held) is None
    assert vs.sector_busy("NVDA", "XLK", None) is None


def _good(held=None):
    return vs.evaluate_ticker("GOOD", getter=_getter, sector_getter=_sector, now=NOW, today=pd.Timestamp(END),
                              earnings_getter=lambda t: pd.Timestamp("2026-12-01"), held=held)


def test_golden_ticket_waits_when_the_sector_is_busy():
    free = _good()
    assert free["sector_busy"] is None and free["nine"].sector_etf == "XLK"
    busy = _good(held={"XLK": "MSFT"})
    assert busy["sector_busy"] == "MSFT"
    assert busy["category"] == ("READY" if free["category"] in ("GOLDEN TICKET", "READY") else free["category"])
    assert busy["category"] != "GOLDEN TICKET"
    e = vs.log_entry(busy, now=datetime(2026, 9, 30, 21, 0))
    assert e["reason"].startswith(f"{vs.SECTOR_BUSY} (MSFT)")
    assert _good(held={"XLE": "XOM"})["sector_busy"] is None


def test_scan_passes_held_to_every_row():
    res = vs.run(["GOOD", "BAD"], getter=_getter, sector_getter=_sector, now=NOW, today=pd.Timestamp(END),
                 held={"XLK": "MSFT"})
    by = {r["ticker"]: r for r in res}
    assert by["GOOD"]["sector_busy"] == "MSFT" and by["BAD"]["sector_busy"] is None


def test_scanner_table_shows_sector_busy_and_us_note():
    from ovtlyr.ui.viking_screens import _table
    html = _table([_good(held={"XLK": "MSFT"})])
    assert f"{vs.SECTOR_BUSY} · MSFT" in html
    assert "+0,07R" in vs.US_NOTE
