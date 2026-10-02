"""
Viking PR 5 — OVTLYR SCREEN, VIKING MOMENTUM SCREEN, bevakningslistan och
signalloggen. Syntetiska kurser och en fejkad hämtare — inget nätverk.
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import ovtlyr_nine as on  # noqa: E402
import viking_execution as vx  # noqa: E402
import viking_screen as vs  # noqa: E402
from test_ovtlyr_nine import END, _df  # noqa: E402

NOW = datetime(2026, 9, 30, 22, 0, tzinfo=timezone.utc)
UP, DOWN = _df(), _df(step=-0.4, start=300)
DATA = {"SPY": UP, "GOOD": UP, "BAD": DOWN, **{t: UP for t in on.SECTOR_ETFS.values()}, "XLE": DOWN}
SECTORS = {"GOOD": "Technology", "BAD": "Energy"}                      # BAD:s sektor (XLE) faller också


def _sector(t):
    return SECTORS.get(t)


def _getter(t, period):
    return DATA.get(t, pd.DataFrame())


def _run(tickers=("GOOD", "BAD", "NODATA")):
    return vs.run(list(tickers), getter=_getter, sector_getter=_sector,
                  earnings_getter=lambda t: pd.Timestamp("2026-12-01"), now=NOW, today=pd.Timestamp(END))


def test_parse_tickers():
    assert vs.parse_tickers("nvda, msft\nVOLV-B.ST;nvda  eqnr.ol") == ["NVDA", "MSFT", "VOLV-B.ST", "EQNR.OL"]
    assert len(vs.parse_tickers(",".join(f"T{i}" for i in range(100)))) == vs.MAX_TICKERS


def test_rows_categories_and_sorting():
    rows = _run()
    by = {r["ticker"]: r for r in rows}
    assert by["GOOD"]["nine"].passed == 9 and by["GOOD"]["category"] in ("GOLDEN TICKET", "READY")
    assert by["BAD"]["nine"].passed <= 6 and by["BAD"]["category"] == "REJECTED"
    assert by["NODATA"]["nine"] is None and "DATA UNAVAILABLE" in by["NODATA"]["error"]
    assert [r["ticker"] for r in vs.ovtlyr_screen(rows)] == ["GOOD", "BAD", "NODATA"]
    wl = vs.watchlist(rows)
    assert list(wl) == list(vs.CATEGORY_ORDER) and by["BAD"] in wl["REJECTED"]


def test_momentum_screen_filters_and_reasons():
    rows = _run()
    by = {r["ticker"]: r for r in rows}
    ok, why = vs.momentum_pass(by["BAD"])
    assert not ok and any("Nine" in w for w in why) and "trendstruktur" in why
    assert vs.momentum_pass(by["NODATA"]) == (False, ["DATA UNAVAILABLE"])
    hits = vs.momentum_screen(rows)
    assert all(vs.momentum_pass(r)[0] for r in hits) and by["BAD"] not in hits
    good_ok, good_why = vs.momentum_pass(by["GOOD"])
    assert good_ok == (by["GOOD"] in hits)
    assert ("volym" in vs.momentum_pass(by["GOOD"], require_volume=True)[1]) == \
           (not next(c for c in by["GOOD"]["decision"].checks if c.key == "volume").passed)


def test_signal_log_one_row_per_ticker_and_day():
    rows = _run()
    log, n = vs.append_log([], rows, now=datetime(2026, 9, 30, 21, 0))
    assert n == 1 and log[0]["ticker"] == "GOOD" and log[0]["action"] in vs.LOGGED_CATEGORIES
    e = log[0]
    for k in ("timestamp", "ticker", "ovtlyr_nine", "market", "sector", "stock", "viking", "entry", "stop",
              "atr", "shares", "rr", "reason", "action", "trigger_date"):
        assert k in e, k
    assert e["ovtlyr_nine"] == "9/9" and e["market"] == "3/3" and e["label"] == on.APPROXIMATION
    again, n2 = vs.append_log(log, rows, now=datetime(2026, 9, 30, 21, 30))
    assert n2 == 0 and len(again) == 1                                   # samma dag, oförändrat → ingen ny rad
    next_day, n3 = vs.append_log(log, rows, now=datetime(2026, 10, 1, 21, 0))
    assert n3 == 1 and len(next_day) == 2


def _closes(tickers):
    return {t: DATA[t]["Close"] for t in tickers if t in DATA}


def test_stage1_scans_the_universe_on_closes():
    closes = {**_closes(["GOOD", "BAD"]), "SHORT": UP["Close"].tail(20), "OLD": UP["Close"].iloc[:-20]}
    ranked, funnel = vs.stage1(closes, today=pd.Timestamp(END))
    assert [t for t, _r in ranked] == ["GOOD"]                         # BAD faller, SHORT/OLD saknar data
    assert funnel == {"universe": 4, "data": 2, "passed": 1}
    assert ranked[0][1] > 0                                             # avkastning tre månader


def test_scan_runs_stage2_on_the_best_candidates():
    up2 = _df(step=0.8)                                                 # starkare momentum → först
    DATA["FAST"] = up2
    try:
        res = vs.scan(["GOOD", "BAD", "FAST", "NODATA"], closes_getter=_closes, max_candidates=1,
                      getter=_getter, sector_getter=lambda t: "Technology",
                      earnings_getter=lambda t: pd.Timestamp("2026-12-01"), now=NOW, today=pd.Timestamp(END))
    finally:
        DATA.pop("FAST")
    f = res["funnel"]
    assert f["universe"] == 4 and f["data"] == 3 and f["passed"] == 2 and f["candidates"] == 1
    assert [r["ticker"] for r in res["rows"]] == ["FAST"]


def test_earnings_only_fetched_for_nine_of_nine():
    asked = []
    vs.run(["GOOD", "BAD"], getter=_getter, sector_getter=_sector, now=NOW, today=pd.Timestamp(END),
           earnings_getter=lambda t: asked.append(t) or pd.Timestamp("2026-12-01"))
    assert asked == ["GOOD"]


def test_page_scans_markets(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import market_prices
    import storage
    import streamlit as st
    from ovtlyr.ui import viking_screens as ui
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, default))
    monkeypatch.setattr(storage, "is_dirty", lambda name: True)
    monkeypatch.setattr(market_prices, "ohlcv", lambda t, p="1y": DATA.get(t, pd.DataFrame()))
    monkeypatch.setattr(ui, "_regions", lambda: (["Norden", "USA"], lambda m: ["GOOD", "BAD"] if "Norden" in m else []))
    real_scan = vs.scan
    monkeypatch.setattr(vs, "scan", lambda tickers, max_candidates=40, progress=None, **kw: real_scan(
        tickers, closes_getter=_closes, max_candidates=max_candidates, progress=progress, getter=_getter,
        sector_getter=_sector, earnings_getter=lambda t: pd.Timestamp("2026-12-01"), now=NOW,
        today=pd.Timestamp(END)))
    monkeypatch.setenv("VN_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["VN_TEST_ROOT"])
        from ovtlyr.ui.viking_screens import render_viking_nine_page
        render_viking_nine_page()

    at = AppTest.from_function(app, default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "MARKET SPY 3/3" in html and "MARKET OMXS30" in html and "Välj marknader" in html
    assert at.multiselect(key="vn_markets").value == ["Norden"]
    at.text_input(key="vn_extra").set_value("nodata")
    at.button(key="FormSubmitter:vn_form-⚔️ SCAN").click().run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "universum <b" in html and "steg 1 (trend + signal)" in html
    assert "<b>GOOD</b>" in html and "9/9" in html                     # BAD och NODATA föll i steg 1
    assert "<b>BAD</b>" not in html and "Momentum</th>" in html
    assert "READY 1" in html or "GOLDEN TICKET 1" in html
    at.checkbox(key="vn_momentum_only").check().run()
    assert not at.exception, at.exception
    assert at.session_state[vs.LOG_STORE][0]["ticker"] == "GOOD"


def test_one_result_list_sorted_by_category_with_momentum_filter():
    rows = _run()
    out = vs.results(rows)
    assert [r["ticker"] for r in out] == ["GOOD", "BAD", "NODATA"]          # kategori → Nine; saknad data sist
    only = vs.results(rows, momentum_only=True)
    assert all(vs.momentum_pass(r)[0] for r in only) and all(r["ticker"] != "BAD" for r in only)


def test_navigation():
    from ui import nav
    assert "⚔️ Viking Nine" in nav.options("screening/Arc Screener") and "Viking" in nav.options("screening/Arc Screener")
    src = open(os.path.join(ROOT, "wolf_panel.py"), encoding="utf-8").read()
    assert 'elif inner == "⚔️ Viking Nine":' in src and "render_viking_nine_page" in src
