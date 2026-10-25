"""Långsidans cykeltest (valhall/long_study.py): tre inträden utan facit, korgar utan överlevnadsfusk, låsta kriterier."""
import numpy as np
import pandas as pd

from valhall import cycle
from valhall import long_study as ls


def _idx(n, start="2008-01-01"):
    return pd.bdate_range(start, periods=n)


def _walk(n, seed=1, start="2000-01-03", drift=0.0003):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n))), index=_idx(n, start))


def test_parameters_are_the_spec():
    assert ls.HORIZONS == (252, 504, 756) and ls.MAIN_H == 504 and ls.LONG_H == 756 and ls.DECLUSTER == 252
    assert (ls.CHEAP_PCT, ls.HAT_MIN, ls.RECENT) == (30.0, 60.0, 126)
    assert (ls.N_BOOT, ls.SEED, ls.MIN_EVENTS, ls.MIN_YEARS) == (5000, 20261025, 20, 5)
    assert (ls.DELISTED_MIN, ls.DELISTED_DAYS) == (20, 365)
    text = open(ls.SPEC_PATH, encoding="utf-8").read()
    for word in ("504", "756", "≤ 30", "≥ 60", "126", "252", "20261025", "minst 20 nordiska"):
        assert word in text, word
    import valhall_long_study as vls
    for sym in vls.PRODUCERS:
        assert sym in text


def _frame(pct, hat, close=None):
    idx = _idx(len(pct))
    c = close if close is not None else pd.Series(100.0, index=idx)
    return pd.DataFrame({"pct10": pct, "hat": hat, "close": c.to_numpy()}, index=idx)


def test_e1_is_the_first_cheap_and_hated_day():
    pct = [np.nan, 20, 20, 40, 25, 25, 25, 10]
    hat = [70, 70, 70, 70, 50, 65, 65, 65]
    fr = _frame(pct, hat)
    assert ls.e1_events(fr) == [fr.index[5]]       # dag 1 är uppvärmning, dag 4 är inte hatad


def test_e2_needs_a_recent_bottom_and_a_new_turn():
    n = 900
    c = pd.Series(np.r_[np.linspace(100, 40, 600), np.linspace(40, 70, 300)], index=_idx(n))
    pct = np.full(n, 50.0)
    pct[560:600] = 10.0                             # botten nyligen
    fr = _frame(pct, np.full(n, 80.0), c)
    ev = ls.e2_events(fr)
    assert len(ev) == 1 and fr.index.get_loc(ev[0]) > 600
    assert fr.index.get_loc(ev[0]) - 599 <= ls.RECENT
    pct2 = np.full(n, 50.0)
    pct2[100:140] = 10.0                            # botten för länge sedan
    assert ls.e2_events(_frame(pct2, np.full(n, 80.0), c)) == []


def test_entries_need_no_future_data():
    c = _walk(3000, seed=11, drift=-0.0001)
    full = cycle.cycle_frame(c)
    cut = c.index[2300]
    part = cycle.cycle_frame(c.loc[:cut])
    a, b = ls.entry_events(full, cycle.RAVARA), ls.entry_events(part, cycle.RAVARA)
    for e in ls.ENTRIES:
        assert [d for d in a[e] if d <= cut] == b[e], e


def test_mae_is_the_deepest_drop_after_entry():
    c = pd.Series([100.0, 100.0, 80.0, 120.0, 90.0], index=_idx(5))
    m = ls.mae(c, 2)
    assert abs(m.iloc[0] - (80 / 100 - 1)) < 1e-12   # inträde 100 (dag 1), botten 80
    assert m.iloc[2:].isna().all()


def test_basket_uses_last_price_after_delisting_and_skips_unlisted():
    idx = _idx(10, "2010-01-01")
    a = pd.Series(np.linspace(100, 190, 10), index=idx)
    b = pd.Series([50.0, 50, 60, 60], index=idx[:4])               # avnoteras efter dag 3
    late = pd.Series([10.0] * 5, index=idx[5:])                     # noteras dag 5
    f = ls.basket_forward({"A": a, "B": b, "L": late}, {2010: ["A", "B", "L"]}, 4)
    ra = a.iloc[5] / a.iloc[1] - 1
    rb = 60 / 50 - 1
    assert abs(f.iloc[0] - (ra + rb) / 2) < 1e-12                   # L inte med, B på sista kursen
    assert np.isnan(f.iloc[-1])


def test_basket_follows_the_year_membership():
    idx = pd.bdate_range("2010-12-20", "2011-01-20")
    a = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    b = pd.Series(np.linspace(100, 50, len(idx)), index=idx)
    f = ls.basket_forward({"A": a, "B": b}, {2010: ["A"], 2011: ["B"]}, 2)
    assert f[f.index.year == 2010].dropna().gt(0).all()
    assert f[f.index.year == 2011].dropna().lt(0).all()


def test_us_signal_maps_to_the_nordic_day_before():
    idx = pd.DatetimeIndex(["2015-03-02", "2015-03-04"])
    assert ls._on_or_before(idx, pd.Timestamp("2015-03-03")) == idx[0]
    assert ls._on_or_before(idx, pd.Timestamp("2015-03-01")) is None


def test_cluster_bootstrap_draws_whole_years():
    rng = np.random.default_rng(0)
    rows = [{"year": 2010, "excess": {504: 0.1}}] * 30
    lo, hi = ls.cluster_boot(rows, 504, rng, 200)
    assert abs(lo - 0.1) < 1e-12 and abs(hi - 0.1) < 1e-12    # ett år: inget att dra mellan
    rows2 = rows + [{"year": 2011, "excess": {504: -0.1}}] * 30
    lo2, hi2 = ls.cluster_boot(rows2, 504, rng, 2000)
    assert lo2 <= -0.09 and hi2 >= 0.09                       # två år: allt från det ena till det andra


def test_verdict_rules():
    ok = {"n": 25, "years": 6, "ci": [1.0, 9.0], "median_excess": 2.0}
    assert ls.verdict(ok, {"mean_excess": 1.0})["pass"]
    assert not ls.verdict({**ok, "n": 19}, {"mean_excess": 1.0})["pass"]
    assert not ls.verdict({**ok, "years": 4}, {"mean_excess": 1.0})["pass"]
    assert not ls.verdict({**ok, "ci": [-0.1, 9.0]}, {"mean_excess": 1.0})["pass"]
    assert not ls.verdict({**ok, "median_excess": -0.1}, {"mean_excess": 1.0})["pass"]
    assert not ls.verdict(ok, {"mean_excess": -1.0})["pass"]


def _planted(k, plant_e1=True):
    idx = _idx(5200, "2002-01-01")
    rng = np.random.default_rng(300 + k)
    r = rng.normal(0, 0.01, len(idx))
    e1 = list(idx[np.arange(1600 + 37 * k, 4300, 300)])
    if plant_e1:
        for d in e1:
            j = idx.get_loc(d)
            r[j + 2:j + 2 + 504] += 0.0012
    c = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    e3 = list(idx[np.arange(1750 + 37 * k, 4300, 300)])
    return ls.series_from_close(f"P{k}", ls.PRODUCENT, c, {ls.E1: e1, ls.E2: [], ls.E3: e3})


def test_run_study_planted_edge_passes_and_noise_does_not():
    series = [_planted(k) for k in range(4)]
    res = ls.run_study(series, n_boot=300)
    assert res["hypotheses"]["L1 producent"]["pass"], res["hypotheses"]["L1 producent"]["reasons"]
    assert not res["hypotheses"]["L3 producent"]["pass"]
    assert not res["hypotheses"]["L2 producent"]["pass"]            # inga händelser
    lines = ls.report_lines(res)
    assert any("L1 producent" in x and "GODKÄND" in x for x in lines)
    flat = ls.run_study([_planted(k, plant_e1=False) for k in range(4)], n_boot=300)
    assert not flat["hypotheses"]["L1 producent"]["pass"]


def test_strong_needs_two_groups():
    series = [_planted(k) for k in range(4)]
    sek = [{**s, "name": s["name"] + "s", "group": ls.SEKTOR} for s in series]
    res = ls.run_study(series + sek, n_boot=300)
    assert ls.E1 in res["strong"]
    assert ls.run_study(series, n_boot=300)["strong"] == []


def test_delisted_check():
    today = pd.Timestamp("2026-10-25")
    last = {i: "2026-10-24" for i in range(100)}
    assert not ls.delisted_check(set(range(100)), last, today)["covered"]
    last.update({i: "2019-05-01" for i in range(20)})
    chk = ls.delisted_check(set(range(100)), last, today)
    assert chk["covered"] and chk["old"] == 20


def test_top_members_uses_last_year_one_share_class_and_sek():
    import valhall_long_study as vls
    info = {1: {"name": "Volvo A", "ticker": "VOLV A", "reportCurrency": "SEK"},
            2: {"name": "Volvo B", "ticker": "VOLV B", "reportCurrency": "SEK"},
            3: {"name": "Norsk Hydro", "ticker": "NHY", "reportCurrency": "NOK"},
            4: {"name": "Liten", "ticker": "LIT", "reportCurrency": "EUR"}}
    mcap = {1: {2010: 100.0, 2011: 100.0}, 2: {2010: 100.0, 2011: 100.0}, 3: {2010: 90.0, 2011: 200.0},
            4: {2010: 1.0, 2011: 50.0}}
    seg = {i: "materials" for i in info}
    got = vls.top_members(mcap, info, seg, [2011, 2012])
    assert got["materials"][2011] == [2, 3, 4]                     # 2010 års värde, B före A
    assert got["materials"][2012][0] == 4                          # 50 EUR × 11,3 > 200 NOK × 0,95


class _Api:
    def __init__(self, old):
        self.old = old

    def is_configured(self):
        return True

    def get_instruments(self):
        return [{"insId": i, "marketId": 1, "ticker": f"T{i} B", "name": f"Bolag {i}", "sectorId": 3,
                 "branchId": None, "reportCurrency": "SEK"} for i in range(1, 31)]

    def get_last_stockprices(self):
        return [{"i": i, "d": "2018-01-02" if i <= self.old else "2026-10-23"} for i in range(1, 31)]

    def get_kpi_history_batch(self, ids, kpi, rt, pt, max_count=None):
        return {i: [{"y": y, "p": 5, "v": float(i * 10)} for y in range(2005, 2027)] for i in ids}

    def get_stockprices_df(self, iid, max_count=5040):
        c = _walk(5000, seed=iid, start="2006-01-02")
        return pd.DataFrame({"Close": c, "Volume": 1e5}, index=c.index)


def _getter(sym, period):
    c = _walk(6000, seed=sum(map(ord, sym)), start="2001-01-02")
    return pd.DataFrame({"Close": c, "Volume": 1e6}, index=c.index)


def test_script_decides_on_bolag_only_when_delisted_are_covered(monkeypatch):
    import markets
    import valhall_long_study as vls
    monkeypatch.setattr(markets, "load", lambda api=None: {})
    monkeypatch.setattr(markets, "nordic_stock_ids", lambda table=None: {1})
    saved = {}
    res = vls.run(getter=_getter, api=_Api(old=25), today="2026-10-25", n_boot=50,
                  save=lambda n, d: saved.setdefault(n, d) is not None)
    groups = {s["group"] for s in res["series"]}
    assert groups == {ls.PRODUCENT, ls.SEKTOR, ls.BOLAG, ls.RAVARA}
    assert res["delisted_check"]["covered"] and ls.BOLAG in res["deciding"]
    korg = next(s for s in res["series"] if s["group"] == ls.BOLAG)
    assert all(len(m) <= vls.TOP_N for m in korg["members"].values())
    assert vls.BLOB in saved and res["saved"]
    res2 = vls.run(getter=_getter, api=_Api(old=3), today="2026-10-25", n_boot=50, no_save=True)
    assert not res2["delisted_check"]["covered"] and ls.BOLAG not in res2["deciding"]
    assert not any(k.endswith(ls.BOLAG) for k in res2["hypotheses"])


class _ApiProperty(_Api):
    """Som BorsdataAPI: is_configured är en egenskap, inte en metod."""
    is_configured = True


def test_script_accepts_is_configured_as_a_property(monkeypatch):
    import markets
    import valhall_long_study as vls
    monkeypatch.setattr(markets, "load", lambda api=None: {})
    monkeypatch.setattr(markets, "nordic_stock_ids", lambda table=None: {1})
    res = vls.run(getter=_getter, api=_ApiProperty(old=25), today="2026-10-25", n_boot=50, no_save=True)
    assert res["delisted_check"].get("error") is None and any(s["group"] == ls.BOLAG for s in res["series"])
