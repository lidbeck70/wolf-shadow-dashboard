"""
Valhall fas 3a — affärerna märks med cykelläget på signaldagen (utan facit), grupperas per segment,
status och period, och portföljen körs med filter. Syntetiska affärer — inget nätverk.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import viking_backtest as vbt  # noqa: E402
from valhall import backtest as vbk  # noqa: E402
from valhall import cycle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _frame(statuses: dict):
    idx = pd.to_datetime(list(statuses))
    return pd.DataFrame({"status": list(statuses.values()), "phase": cycle.STARK, "cykel": cycle.MITTEN}, index=idx)


NORDIC_IND = _frame({"2015-01-02": cycle.GRON, "2016-01-04": cycle.ROD, "2022-01-03": cycle.GRON})
GLOBAL_IND = _frame({"2015-01-02": cycle.ROD, "2022-01-03": cycle.GRON})
THEME_CU = _frame({"2015-01-02": cycle.GUL})
FRAMES = {"nordic": {"industrials": NORDIC_IND}, "global": {"industrials": GLOBAL_IND},
          "themes": {"koppar": THEME_CU}}


def _t(ticker, signal, r, entry="2015-01-05", exit_="2015-02-05"):
    t = vbt.Trade(ticker, signal, entry, 100.0, 95.0, 5.0, 9, exit_date=exit_, exit=100 + 5 * r, r=r)
    t.dates, t.path = (entry, exit_), (100.0, 100 + 5 * r)
    return t


TRADES = [_t("VOLV-B.ST", "2015-06-01", 2.0), _t("VOLV-B.ST", "2016-06-01", -1.0),
          _t("SAND.ST", "2022-06-01", 1.0, "2022-06-02", "2022-07-01"), _t("SAND.ST", "2014-06-01", -1.0),
          _t("BOL.ST", "2015-06-01", 0.5)]
SEG = {"VOLV-B.ST": "industrials", "SAND.ST": "industrials", "BOL.ST": "materials"}.get


def test_asof_uses_last_known_row():
    assert vbk.asof(NORDIC_IND, "2015-12-31")["status"] == cycle.GRON
    assert vbk.asof(NORDIC_IND, "2016-01-04")["status"] == cycle.ROD
    assert vbk.asof(NORDIC_IND, "2014-12-31") == {} and vbk.asof(None, "2020-01-01") == {}


def test_tag_and_period():
    tagged = vbk.tag(TRADES, SEG, FRAMES, theme_of={"BOL.ST": "koppar"}.get)
    by = {(r["trade"].ticker, r["trade"].signal_date): r for r in tagged}
    assert by[("VOLV-B.ST", "2015-06-01")]["n_status"] == cycle.GRON and by[("VOLV-B.ST", "2015-06-01")]["g_status"] == cycle.ROD
    assert by[("VOLV-B.ST", "2016-06-01")]["n_status"] == cycle.ROD
    assert by[("SAND.ST", "2022-06-01")]["period"] == "2021–" and by[("SAND.ST", "2022-06-01")]["g_status"] == cycle.GRON
    assert by[("SAND.ST", "2014-06-01")]["n_status"] == cycle.NODATA                   # före första raden
    assert by[("BOL.ST", "2015-06-01")]["t_status"] == cycle.GUL and by[("BOL.ST", "2015-06-01")]["n_status"] == cycle.NODATA


def test_table_and_stats():
    tagged = vbk.tag(TRADES, SEG, FRAMES)
    rows = {(r["group"], r["status"], r["period"]): r for r in vbk.table(tagged, "segment")}
    assert rows[("industrials", "alla", "alla")]["n"] == 4
    green = rows[("industrials", cycle.GRON, "alla")]
    assert green["n"] == 2 and green["exp"] == pytest.approx(1.5) and green["win"] == 100.0
    assert rows[("industrials", cycle.ROD, "alla")]["exp"] == -1.0
    assert rows[("industrials", cycle.GRON, "2021–")]["n"] == 1
    assert vbk.stats([])["n"] == 0
    assert vbk.table(tagged, "all")[0]["group"] == "alla"


def test_portfolio_filters():
    tagged = vbk.tag(TRADES, SEG, FRAMES)
    out = {r["filter"]: r for r in vbk.portfolios(tagged, 10.0)}
    assert out["Alla affärer"]["trades"] == 5 and out["Norden GRÖN"]["trades"] == 2
    assert out["Norden ej RÖD"]["trades"] == 4 and out["Norden och globalt GRÖN"]["trades"] == 1
    assert out["Norden GRÖN"]["exp"] == pytest.approx(1.5) and "mc_dd95" in out["Alla affärer"]
    one = vbk.portfolios(tagged, 5.0, period="2021–")
    assert {r["filter"]: r["trades"] for r in one}["Alla affärer"] == 1


def test_fas3_runner(monkeypatch):
    import valhall_fas3 as vf
    monkeypatch.setattr(vf, "frames", lambda api, getter, data_getter, nordic_market=None:
                        {**FRAMES, "universe": {t: {"segment": s} for t, s in
                                                (("VOLV-B.ST", "industrials"), ("SAND.ST", "industrials"),
                                                 ("BOL.ST", "materials"))}})
    out = vf.run(api=object(), getter=lambda t, p: None, data_getter=lambda ts, p: {},
                 viking_run=lambda tickers, cfg=None, today=None: {"trades": TRADES[:4]},
                 berserk_run=lambda tickers, cfg=None, today=None: {"trades": [TRADES[4]]},
                 today="2026-10-13", no_save=True)
    assert {r["filter"] for r in out["viking"]["portfolio"]} >= {"Alla affärer", "Norden GRÖN"}
    assert not any(r["filter"].startswith("Tema") for r in out["viking"]["portfolio"])
    assert out["berserk"]["by_theme"][0]["group"] == "koppar"
    assert set(out["viking"]["portfolio_by_period"]) == {"2008–2020", "2021–"}


def test_backtest_workflow():
    src = open(os.path.join(ROOT, ".github", "workflows", "valhall-backtest.yml"), encoding="utf-8").read()
    assert "workflow_dispatch" in src and "python valhall_fas3.py" in src and "BORSDATA_API_KEY" in src


# ── Fas 3b (K9): vändningsgrinden och Viking Nine per komplex ───────────────
def _turn_frame():
    idx = pd.bdate_range("2015-01-01", periods=120)
    ph = [cycle.BAISSE] * 40 + [cycle.VANDER] * 50 + [cycle.STARK] * 30
    conf = [False] * 60 + [True] * 30 + [False] * 30
    return pd.DataFrame({"status": cycle.GRON, "phase": ph, "cykel": cycle.TIDIG, "confirmed": conf}, index=idx)


def test_asof_carries_the_turn():
    f = _turn_frame()
    a = vbk.asof(f, f.index[45])
    assert a["phase"] == cycle.VANDER and a["vander_days"] == 6 and not a["confirmed"]
    b = vbk.asof(f, f.index[70])
    assert b["vander_days"] == 31 and b["confirmed"]
    assert vbk.asof(f, f.index[100])["vander_days"] == 0
    assert vbk.asof(_frame({"2015-01-02": cycle.GRON}), "2015-06-01")["confirmed"] is False


def test_fas3b_filters_select_on_phase_and_age():
    f = _turn_frame()
    trades = [_t("VOLV-B.ST", str(f.index[45].date()), 1.0), _t("VOLV-B.ST", str(f.index[70].date()), 2.0),
              _t("VOLV-B.ST", str(f.index[100].date()), -1.0)]
    tagged = vbk.tag(trades, SEG, {"nordic": {"industrials": f}, "global": {}, "themes": {}})
    sel = {name: [r["trade"].r for r in tagged if fn(r)] for name, fn in vbk.FILTERS_3B.items()}
    assert sel["Norden VÄNDER"] == [1.0, 2.0] and sel["Norden VÄNDER bekräftad"] == [2.0]
    assert sel["Norden VÄNDER ≤ 20 d"] == [1.0] and sel["Norden VÄNDER ≤ 60 d"] == [1.0, 2.0]
    tt = vbk.tag(trades, SEG, {"themes": {"koppar": f}}, theme_of=lambda t: "koppar")
    th = {name: [r["trade"].r for r in tt if fn(r)] for name, fn in vbk.FILTERS_3B_THEME.items()}
    assert th["Tema VÄNDER bekräftad"] == [2.0] and th["Tema VÄNDER ≤ 20 d"] == [1.0]
    out = {r["filter"]: r for r in vbk.portfolios(tagged, 1.0, filters=vbk.FILTERS_3B)}
    assert out["Norden VÄNDER"]["trades"] == 2 and out["Alla affärer"]["trades"] == 3


def test_complex_backtest_compares_standard_complex_and_driver(monkeypatch):
    import numpy as np
    import complex_backtest as cbt
    import complex_profiles as cp
    idx = pd.bdate_range("2015-01-01", periods=400)
    up = pd.Series(np.linspace(100, 200, 400), index=idx)
    down = pd.Series(np.linspace(200, 100, 400), index=idx)
    assert cbt.gate_ok(cbt.driver_turn_series(up), idx[-1]) is True
    assert cbt.gate_ok(cbt.driver_turn_series(down), idx[-1]) is False and cbt.gate_ok(None, idx[-1]) is None
    seen = []

    def fake_bt(t, df, m, sec, breadth, cfg, start=None, market_label="SPY"):
        seen.append((t, sec is not None))
        return {"trades": [_t(t, str(idx[-1].date()), 1.0 if sec is not None else -1.0)]}
    monkeypatch.setattr(cbt.vbt, "backtest_ticker", fake_bt)
    frame = pd.DataFrame({"Open": up, "High": up, "Low": up, "Close": up, "Volume": 1e6})
    getter = lambda t, p: frame if t in ("BOL.ST", "COPX") else None  # noqa: E731
    res = cbt.run_complex(cp.get("basmetaller"), ["BOL.ST"], getter, {"close": up, "breadth": None}, driver=down)
    assert [t.r for t in res["standard"]] == [-1.0] and [t.r for t in res["komplex"]] == [1.0]
    assert res["drivare"] == []                                                 # drivaren har inte vänt
    rows = cbt.run(getter, members={"basmetaller": ["BOL.ST"]}, nordic={"close": up}, drivers={"basmetaller": up})
    by = {r["variant"]: r for r in rows}
    assert set(by) == {"standard", "komplex", "drivare"} and by["drivare"]["n"] == 1 and by["komplex"]["exp"] == 1.0
    assert cbt.summary([])["n"] == 0
