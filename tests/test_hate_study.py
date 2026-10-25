"""Hatets varaktighet (valhall/hate_study.py): perioder, Kaplan–Meier, väntan, bredd och värdering utan facit."""
import numpy as np
import pandas as pd
import pytest

from valhall import hate_study as hs


def _idx(n, start="2008-01-01"):
    return pd.bdate_range(start, periods=n)


def test_parameters_are_the_spec():
    assert (hs.GAP, hs.DELAYS, hs.HOLD, hs.START) == (21, (0, 126, 252), 504, "2008-01-01")
    assert (hs.TOP_N, hs.TOP_VARIANT, hs.B1_LEVEL, hs.B2_LEVEL, hs.B2_WINDOW, hs.DECLUSTER) == (10, 5, 0.5, 0.3, 756, 252)
    assert (hs.VAL_PCT, hs.VAL_MIN_YEARS, hs.LAG_MONTHS, hs.N_BOOT, hs.SEED) == (30.0, 5, 4, 5000, 20261025)
    assert (hs.D2_MIN_N, hs.D2_MIN_YEARS, hs.D3_MIN_N, hs.D3_MIN_YEARS) == (30, 5, 15, 5)
    text = open(hs.SPEC_PATH, encoding="utf-8").read()
    for w in ("21 handelsdagar", "126 och\n252", "504 handelsdagar", "≥ 50 %", "≤ 30 %", "756 dagar", "20261025",
              "minst 30 perioder", "minst 15 händelser", "10 största"):
        assert w in text, w


def test_episodes_merge_short_gaps_and_flag_censoring():
    h = pd.Series([np.nan] * 3 + [0] * 5 + [1] * 10 + [0] * 5 + [1] * 5 + [0] * 30 + [1] * 4 + [0] * 3, index=_idx(65),
                  dtype=float)
    eps = hs.episodes(h)
    assert len(eps) == 2
    assert eps[0]["start"] == h.index[8] and eps[0]["end"] == h.index[27] and not eps[0]["censored"]
    assert eps[0]["length"] == 20                                  # glappet på 5 dagar räknas in
    assert eps[1]["censored"]                                      # bara tre icke-hatade dagar efter


def test_episode_at_the_first_known_day_is_dropped():
    h = pd.Series([np.nan, 1, 1, 0] + [0] * 30, index=_idx(34), dtype=float)
    assert hs.episodes(h) == []


def test_kaplan_meier_with_and_without_censoring():
    k = hs.km([21, 42, 63, 84, 105], [False] * 5)
    assert k["median_m"] == 3.0 and k["completed"] == 5
    k2 = hs.km([21, 42, 600, 700], [False, False, True, True])      # två pågående, långa
    assert k2["median_m"] == 2.0 and k2["ongoing"] == 2 and k2["left_24m"] == 50.0
    assert hs.km([], [])["n"] == 0


def test_delay_rows_are_relative_and_need_full_horizon():
    idx = _idx(1200)
    c = pd.Series(np.linspace(100, 200, 1200), index=idx)
    m = pd.Series(100.0, index=idx)
    e = {"start": idx[10], "end": idx[50], "length": 41, "censored": False}
    rows = hs.delay_rows(c, m, [e])
    assert len(rows) == 1
    a, b = 11, 11 + hs.HOLD
    assert rows[0]["rel"][0] == pytest.approx(c.iloc[b] / c.iloc[a] - 1)
    late = {**e, "start": idx[500]}
    assert hs.delay_rows(c, m, [late]) == []                        # 12 mån + 2 år ryms inte


def _rows(diff6, n=40, years=8):
    rng = np.random.default_rng(1)
    out = []
    for i, x in enumerate(rng.normal(0, 0.1, n)):
        rel = {0: float(x), 126: float(x) + diff6 + rng.normal(0, 0.02), 252: float(x)}
        out.append({"year": 2008 + i % years, "rel": rel})
    return out


def test_d2_verdicts():
    rng = np.random.default_rng(0)
    assert hs.d2_summary(_rows(0.10), rng, 500)["V6"]["verdict"] == "VÄNTA LÖNAR SIG"
    assert hs.d2_summary(_rows(-0.10), rng, 500)["V6"]["verdict"] == "KÖP DIREKT"
    assert hs.d2_summary(_rows(0.10, n=20), rng, 500)["V6"]["verdict"] == "INGEN SKILLNAD BEVISAD"   # för få
    assert hs.d2_summary(_rows(0.10), rng, 500)["V12"]["verdict"] == "INGEN SKILLNAD BEVISAD"


def test_top_by_year_one_share_class_previous_year():
    info = {1: {"name": "Stora Enso A", "ticker": "STE A", "reportCurrency": "EUR"},
            2: {"name": "Stora Enso R", "ticker": "STE R", "reportCurrency": "EUR"},
            3: {"name": "Boliden", "ticker": "BOL", "reportCurrency": "SEK"},
            4: {"name": "Liten", "ticker": "LIT", "reportCurrency": "SEK"}}
    mcap = {1: {2019: 10.0}, 2: {2019: 10.0}, 3: {2019: 50.0}, 4: {2019: 1.0, 2020: 500.0}}
    got = hs.top_by_year(mcap, info, {i: "materials" for i in info}, [2020, 2021], n=2)
    assert got["materials"][2020] == [1, 3]                         # 10 EUR × 11,3 > 50 SEK; Stora Enso en gång
    assert got["materials"][2021] == [4]


def test_universe_index_follows_yearly_membership():
    idx = pd.bdate_range("2019-12-20", "2020-01-10")
    a = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    b = pd.Series(np.linspace(100, 80, len(idx)), index=idx)
    ix = hs.universe_index({"A": a, "B": b}, {2019: ["A"], 2020: ["B"]})
    assert ix[ix.index.year == 2019].is_monotonic_increasing
    assert ix[ix.index.year == 2020].iloc[1:].is_monotonic_decreasing


def test_breadth_and_events():
    cal = _idx(1500, "2009-01-01")
    states = {}
    for k in range(6):
        v = np.zeros(len(cal))
        v[200:500] = 1 if k < 4 else 0                              # 4 av 6 hatade → 67 %
        v[700:] = 1 if k < 1 else 0                                 # 1 av 6 → 17 %
        states[k] = pd.Series(v, index=cal)
    members = {y: list(range(6)) for y in range(2009, 2016)}
    b = hs.breadth(states, members, cal)
    assert b.iloc[300] == pytest.approx(4 / 6) and b.iloc[800] == pytest.approx(1 / 6)
    ev = hs.breadth_events(b)
    assert ev["B1"] == [cal[200]] and ev["B2"] == [cal[500]]
    few = hs.breadth({0: states[0]}, {y: [0] for y in range(2009, 2016)}, cal)
    assert few.isna().all()                                         # färre än 5 med läge


def test_cheap_years_and_lagged_undervaluation():
    ev = [{"y": y, "v": v} for y, v in zip(range(2010, 2017), [20, 18, 16, 15, 4, 25, -3])]
    flags = hs.cheap_years(ev, positive_only=True)
    assert flags[2014] is True and flags[2015] is False and 2016 not in flags
    assert hs.undervalued_at(ev, [], "2015-04-30") is None or hs.undervalued_at(ev, [], "2015-04-30") is False
    assert hs.undervalued_at(ev, [], "2015-05-01") is True          # 2014 känt fyra månader efter årsskiftet
    assert hs.undervalued_at([], [], "2015-05-01") is None


def test_current_reports_ongoing_hate():
    h = pd.Series([0.0] * 10 + [1.0] * 50, index=_idx(60))
    c = hs.current("X", h, hs.episodes(h))
    assert c["since"] == str(h.index[10].date()) and c["months"] == pytest.approx(50 / 21, abs=0.1)
    h2 = pd.Series([0.0] * 10 + [1.0] * 5 + [0.0] * 45, index=_idx(60))
    assert hs.current("X", h2, hs.episodes(h2)) is None


class _Api:
    is_configured = True

    def get_instruments(self):
        return [{"insId": i, "marketId": 1, "ticker": f"T{i}", "name": f"Bolag {i}", "sectorId": 3 if i <= 15 else 7,
                 "branchId": None, "reportCurrency": "SEK"} for i in range(1, 31)]

    def get_kpi_history_batch(self, ids, kpi, rt, pt, max_count=None):
        return {i: [{"y": y, "v": float(i * 10 + (y % 3))} for y in range(2004, 2027)] for i in ids}

    def get_stockprices_df(self, iid, max_count=5040):
        idx = pd.bdate_range("2006-01-02", "2026-10-23")
        rng = np.random.default_rng(iid)
        c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, len(idx)))), index=idx)
        return pd.DataFrame({"Close": c, "Volume": 1e5}, index=idx)


def _getter(sym, period):
    idx = pd.bdate_range("2001-01-02", "2026-10-23")
    c = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(sum(map(ord, sym))).normal(0.0002, 0.015, len(idx)))),
                  index=idx)
    return pd.DataFrame({"Close": c, "Volume": 1e6})


def test_script_runs_both_groups(monkeypatch):
    import markets

    import hate_duration_study as hds
    monkeypatch.setattr(markets, "load", lambda api=None: {})
    monkeypatch.setattr(markets, "nordic_stock_ids", lambda table=None: {1})
    saved = {}
    res = hds.run(getter=_getter, api=_Api(), today="2026-10-25", save=lambda n, d: saved.setdefault(n, d) is not None)
    nd, cm = res["nordic"], res["commodities"]
    assert "error" not in nd and "error" not in cm
    assert set(nd["variants"]) == {10, 5} and nd["companies"] == 28     # 10 + 4 i reserv per sektor
    v = nd["variants"][10]
    assert set(v["d1"]) <= {"Energi", "Material"} and v["d2"]["V6"]["verdict"]
    assert {"B1", "B2"} == set(v["d3"])
    assert set(cm["d1"]) == {"producent", "råvara"}
    assert hds.BLOB in saved and res["saved"]
    assert any("HATADE I DAG" in x for x in hds.report_lines(res))
