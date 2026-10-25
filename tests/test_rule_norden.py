"""Papperskontot rule_norden: testets regler live (valhall.rule_study.live), skanningen och inkopplingen."""
import numpy as np
import pandas as pd

from valhall import paper as vpa
from valhall import rule_study as rs
from tests.test_rule_study import _hist


def _series():
    idx = pd.bdate_range("2014-01-01", periods=400)
    c = pd.Series(np.r_[np.linspace(100, 60, 380), np.linspace(60, 75, 20)], index=idx)
    return idx, c


def test_live_signal_when_the_catalyst_is_recent_and_gates_pass():
    _idx_full, full = _series()
    k = int(np.flatnonzero(rs.catalyst(full).to_numpy())[0])
    c = full.iloc[:k + 3]                                              # vändningen syns sista dagarna
    idx = c.index
    ep = {"start": idx[200], "end": idx[k - 1], "censored": True}
    res = rs.live(c, pd.Series(100.0, index=idx), [ep], _hist(last_year=2013))
    assert res["signal"] == {"day": str(idx[k].date()), "hate_since": str(idx[200].date())} and res["watch"] is None
    bad = rs.live(c, pd.Series(100.0, index=idx), [ep], _hist(de=1.5, last_year=2013))
    assert bad["signal"] is None


def test_live_watch_when_hated_without_catalyst():
    idx = pd.bdate_range("2014-01-01", periods=400)
    c = pd.Series(np.linspace(100, 50, 400), index=idx)
    ep = {"start": idx[300], "end": idx[399], "censored": True}
    res = rs.live(c, pd.Series(100.0, index=idx), [ep], _hist(last_year=2013))
    assert res["signal"] is None and res["watch"] == {"hate_since": str(idx[300].date())}


def test_rule_norden_account_sells_after_two_years():
    assert vpa.STRATEGIES["rule_norden"][1] == vpa.LONG
    idx = pd.bdate_range("2024-01-01", periods=800)
    df = pd.DataFrame({k: 100.0 for k in ("Open", "High", "Low", "Close")}, index=idx)
    df["Volume"] = 1e6
    st = vpa.new_state("2024-01-01")
    sig = vpa._sig("rule_norden", "ABC.ST", "ABC", "industrials", None, True, "x")
    st, _ = vpa.step(st, [sig], {}, lambda t: df.iloc[:2], today="2024-01-01")
    st, _ = vpa.step(st, [], {}, lambda t: df.iloc[:3], today="2024-01-03")
    st, _ = vpa.step(st, [], {}, lambda t: df, today=idx[-1])
    assert st["closed"][0]["reason"].startswith("tidsstopp 2 år")


def test_valhall_scan_reads_fresh_signals_only():
    import valhall_scan as vs
    blob = {"date": "2026-10-23", "signals": [
        {"ticker": "ABC.ST", "name": "ABC", "segment": "industrials", "day": "2026-10-22", "hate_since": "2026-03-01"},
        {"ticker": "OLD.ST", "name": "Old", "segment": "energy", "day": "2026-09-01", "hate_since": "2026-01-01"}]}
    sigs = vs._rule_signals(lambda name, default=None: blob if name == vs.RULE_BLOB else default, "2026-10-25")
    assert [(s["strategy"], s["ticker"], s["nordic"]) for s in sigs] == [("rule_norden", "ABC.ST", True)]
    assert vs._rule_signals(lambda name, default=None: None, "2026-10-25") == []


def test_scan_runs_end_to_end(monkeypatch):
    import markets

    import rule_norden_scan as rns
    from tests.test_hate_study import _Api
    monkeypatch.setattr(markets, "load", lambda api=None: {})
    monkeypatch.setattr(markets, "nordic_stock_ids", lambda table=None: {1})
    monkeypatch.setattr(markets, "to_yf", lambda t, m, table=None: f"{t}.ST")
    saved = {}
    out = rns.run(api=_Api(), today="2026-10-25", save=lambda n, d: saved.setdefault(n, d) is not None)
    assert out["universe"] == 30 and isinstance(out["signals"], list) and isinstance(out["watch"], list)
    assert set(out["funnel"]) >= {"hatade nu", "båda OK", "katalysator redan i perioden"}
    assert rns.BLOB in saved and out["saved"]
    assert all(r["ticker"].endswith(".ST") for r in out["signals"] + out["watch"])
