"""Eventtest nivå 1 (valhall/event_study.py): händelsedagarna är turn_states, inget facit, låsta parametrar."""
import numpy as np
import pandas as pd

from valhall import cycle
from valhall import event_study as es


def _idx(n, start="2008-01-01"):
    return pd.bdate_range(start, periods=n)


def _status_frame(statuses, start="2008-01-01"):
    return pd.DataFrame({"status": statuses}, index=_idx(len(statuses), start))


def _walk(n, seed=1, start="2000-01-03"):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n))), index=_idx(n, start))


def test_parameters_are_the_spec():
    assert es.HORIZONS == (21, 63, 126) and es.MAIN_H == 63 and es.DECLUSTER == 63
    assert (es.N_BOOT, es.N_PERM, es.SEED, es.MIN_EVENTS, es.ALPHA) == (5000, 2000, 20261025, 30, 0.05)
    assert [p[0] for p in es.PERIODS] == ["2008–2020", "2021–"]
    text = open(es.SPEC_PATH, encoding="utf-8").read()
    assert f"version {es.SPEC_VERSION}" in text and "20261025" in text and "2 000" in text


def test_segment_event_is_turn_states_confirmed_day():
    R, G, Y = cycle.ROD, cycle.GRON, cycle.GUL
    st = [R] * 5 + [G] * 12 + [Y] * 3 + [G] * 15 + [R] * 2 + [G] * 9
    fr = _status_frame(st)
    ev = es.segment_events(fr)
    assert ev == [fr.index[5 + 9]]                      # GUL→GRÖN räknas inte, nio GRÖNA räcker inte
    assert cycle.turn_state(fr.iloc[:5 + 12], cycle.SEGMENT)["confirmed_since"] == str(ev[0].date())


def test_ravara_event_is_first_confirmed_day_in_each_vander_period():
    V, B = cycle.VANDER, cycle.BAISSE
    ph = [B] * 3 + [V] * 6 + [B] * 2 + [V] * 4 + [V] * 0
    conf = [False] * 3 + [False, False, True, True, False, True] + [False] * 2 + [False] * 4
    fr = pd.DataFrame({"phase": ph, "confirmed": conf}, index=_idx(len(ph)))
    ev = es.ravara_events(fr)
    assert ev == [fr.index[5]]                          # andra VÄNDER-perioden bekräftas aldrig
    assert cycle.turn_state(fr.iloc[:9], cycle.RAVARA)["confirmed_since"] == str(ev[0].date())


def test_trend_event_is_the_first_day_above_both_averages():
    c = pd.Series(np.r_[np.linspace(100, 50, 300), np.linspace(50, 120, 200)], index=_idx(500))
    ev = es.trend_events(c)
    assert len(ev) == 1
    k = c.index.get_loc(ev[0])
    sma, ema = c.rolling(200).mean(), c.ewm(span=50, adjust=False).mean()
    assert c.iloc[k] > sma.iloc[k] and c.iloc[k] > ema.iloc[k]
    assert not (c.iloc[k - 1] > sma.iloc[k - 1] and c.iloc[k - 1] > ema.iloc[k - 1])


def test_decluster_drops_events_within_63_days_of_the_previous():
    idx = _idx(400)
    d = [idx[10], idx[50], idx[120], idx[150], idx[300]]
    assert es.decluster(d, idx) == [idx[10], idx[120], idx[300]]


def test_forward_return_enters_the_day_after():
    c = pd.Series([100.0, 110.0, 121.0, 133.1], index=_idx(4))
    f = es.forward(c, 1)
    assert abs(f.iloc[0] - 0.10) < 1e-12                # 110 → 121, inte 100 → 110
    assert f.iloc[2:].isna().all()                      # ingen hel horisont


def test_events_need_no_future_data():
    c = _walk(2600, seed=7)
    full = cycle.cycle_frame(c)
    cut = c.index[2000]
    part = cycle.cycle_frame(c.loc[:cut])
    for fn in (es.ravara_events, es.segment_events):
        a = [d for d in fn(full) if d <= cut]
        b = fn(part)
        assert a == b, fn.__name__
    assert [d for d in es.trend_events(c) if d <= cut] == es.trend_events(c.loc[:cut])


def test_excess_is_against_the_same_series_and_period():
    c = _walk(4400, seed=3, start="2006-01-02")
    rows, pools = es.series_rows("X", es.SEGMENT, c, {es.TURN: [c.index[800], c.index[4100]]})
    assert {r["period"] for r in rows} == {"2008–2020", "2021–"}
    r = next(r for r in rows if r["period"] == "2008–2020")
    base = es.forward(c, 63)[(c.index >= "2008-01-01") & (c.index <= "2020-12-31")].dropna().mean()
    assert abs(r["excess"][63] - (r["ret"][63] - base)) < 1e-12
    assert abs(pools[("2008–2020", 63)].mean()) < 1e-12


def test_events_before_2008_are_not_counted():
    c = _walk(1500, seed=4, start="2003-01-01")
    rows, _ = es.series_rows("X", es.SEGMENT, c, {es.TURN: [c.index[100]]})
    assert rows == []


def test_permutation_p_separates_planted_edge_from_noise():
    rng = np.random.default_rng(0)
    pool = rng.normal(0, 1, 2000)
    pool -= pool.mean()
    top = np.sort(pool)[-40:]
    assert es.perm_p(float(top.mean()), {"s": 40}, {"s": pool}, rng, n=500) < 0.01
    rand = rng.choice(pool, 40, replace=False)
    assert es.perm_p(float(rand.mean()), {"s": 40}, {"s": pool}, rng, n=500) > 0.01


def test_verdict_needs_both_periods():
    ok = {"n": 40, "ci": [0.5, 3.0], "p": 0.01}
    assert es.verdict({"2008–2020": ok, "2021–": ok})["pass"]
    assert not es.verdict({"2008–2020": ok, "2021–": {**ok, "n": 29}})["pass"]
    assert not es.verdict({"2008–2020": ok, "2021–": {**ok, "ci": [-0.1, 2.0]}})["pass"]
    assert not es.verdict({"2008–2020": ok, "2021–": {**ok, "p": 0.05}})["pass"]
    assert es.verdict({"2008–2020": ok, "2021–": {**ok, "p": None}}, need_p=False)["pass"]


def test_run_study_planted_edge_passes_and_noise_does_not():
    series = []
    for k in range(12):
        idx = _idx(5000, "2004-01-01")
        rng = np.random.default_rng(100 + k)
        r = rng.normal(0, 0.01, len(idx))
        ev = list(idx[np.arange(1100, 4900, 140)])
        for d in ev:                                   # plantera en uppgång efter varje händelse
            j = idx.get_loc(d)
            r[j + 2:j + 2 + 63] += 0.004
        c = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
        noise = list(idx[np.arange(1170, 4900, 140)])
        series.append({"name": f"S{k}", "group": es.SEGMENT, "close": c, "events": {es.TURN: ev, es.TREND: noise}})
    res = es.run_study(series, n_boot=300, n_perm=200)
    assert res["hypotheses"]["H1"]["pass"], res["hypotheses"]["H1"]["reasons"]
    assert res["hypotheses"][f"H3 {es.SEGMENT}"]["pass"]
    assert res["level2"]
    lines = es.report_lines(res)
    assert any("H1" in x and "GODKÄND" in x for x in lines)
    flipped = [{**s, "events": {es.TURN: s["events"][es.TREND], es.TREND: s["events"][es.TURN]}} for s in series]
    res2 = es.run_study(flipped, n_boot=300, n_perm=200)
    assert not res2["hypotheses"]["H1"]["pass"] and not res2["level2"]


def test_script_builds_all_three_groups_from_a_getter():
    import valhall_event_study as ves

    def getter(sym, period):
        seed = sum(map(ord, sym))
        c = _walk(5200, seed=seed, start="2001-01-02")
        return pd.DataFrame({"Close": c, "Volume": 1e6}, index=c.index)

    saved = {}
    res = ves.run(getter=getter, save=lambda name, data: saved.setdefault(name, data) is not None, n_perm=20)
    groups = {s["group"] for s in res["series"]}
    assert groups == {es.SEGMENT, es.RAVARA, es.KOMPLEX}
    names = [s["name"] for s in res["series"] if s["group"] == es.RAVARA]
    syms = [n.split(":", 1)[1] for n in names]
    assert len(syms) == len(set(syms))                 # en drivarsymbol räknas en gång
    assert sum(1 for s in res["series"] if s["group"] == es.SEGMENT) == 11
    assert ves.BLOB in saved and res["saved"]
