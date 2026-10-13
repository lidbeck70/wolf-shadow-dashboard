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
