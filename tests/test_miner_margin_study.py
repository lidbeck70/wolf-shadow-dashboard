"""
Gruvorna mot marginalen — förregistrerat backtest (miner_margin_study.py, docs/forskning/gruvor_marginal_spec.md).

Signalerna S0–S3 ska vara panelens egna funktioner räknade point-in-time: ingen framåtblick (data efter t ändrar
aldrig signalen på t, AISC används först från sitt publiceringsdatum), episoder skilda med 63 dagar, hela
horisonter, en bootstrap som ger samma svar med samma frö och en rapport som går utan nätverk. Inget nätverk och
ingen skrivning till repot eller Gisten — fejkade kursserier och fejkade AISC-rader.
"""
import json
import math
import os
import socket

import numpy as np
import pandas as pd
import pytest

import miner_margin_study as ms
from alpha_regime import commodity_ratios as cr
from alpha_regime import miner_margin as mm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _no_network_and_no_gist(monkeypatch):
    """Inget nätverk och aldrig den riktiga Gisten (eller dess lokala reservfil i repot)."""
    def _blocked(*_a, **_k):
        raise RuntimeError("nätverk är avstängt i testerna")
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    import gist_storage
    monkeypatch.setattr(gist_storage, "save_blob", lambda *a, **k: pytest.fail("riktiga save_blob anropades"))


# ── Fejkade kurser ───────────────────────────────────────────────────────────
def _path(idx, anchors, vol, seed):
    """Log-linjär väg genom ankarpunkterna plus brus som dras tillbaka mot vägen."""
    t = pd.to_datetime([a[0] for a in anchors])
    base = np.interp(idx.asi8, t.asi8, np.log([a[1] for a in anchors]))
    noise = np.cumsum(np.random.default_rng(seed).normal(0, vol, len(idx)))
    noise -= pd.Series(noise).rolling(60, min_periods=1).mean().to_numpy()
    return pd.Series(np.exp(base + noise), index=idx)


def _make_series():
    idx = pd.bdate_range("2004-11-18", "2026-10-30")
    gold = _path(idx, [("2004-11-18", 440), ("2008-03-01", 980), ("2008-10-24", 720), ("2011-09-01", 1880),
                       ("2013-06-28", 1200), ("2015-12-17", 1050), ("2016-07-06", 1370), ("2018-08-16", 1175),
                       ("2020-08-06", 2060), ("2022-09-28", 1630), ("2024-10-30", 2790), ("2025-10-20", 4350),
                       ("2026-10-30", 4100)], 0.008, 1)
    gld = gold / 10.0 * np.exp(np.random.default_rng(9).normal(0, 0.001, len(idx)))
    gdx = _path(idx[idx >= "2006-05-22"],
                [("2006-05-22", 38), ("2008-03-01", 55), ("2008-10-24", 17), ("2011-04-08", 64), ("2013-06-26", 19),
                 ("2015-09-11", 13), ("2016-08-11", 31), ("2018-09-11", 18), ("2020-08-04", 44), ("2022-09-26", 22),
                 ("2024-10-21", 44), ("2025-10-15", 85), ("2026-10-30", 80)], 0.02, 2)
    gdxj = _path(idx[idx >= "2009-11-11"],
                 [("2009-11-11", 40), ("2010-12-01", 55), ("2011-04-08", 60), ("2013-06-26", 22), ("2015-09-11", 17),
                  ("2016-08-11", 51), ("2018-09-11", 28), ("2020-08-04", 63), ("2022-09-26", 27),
                  ("2024-10-21", 52), ("2025-10-15", 110), ("2026-10-30", 100)], 0.025, 3)
    gold = gold.drop(idx[::97])                         # några dagar utan guldkurs, som terminens luckor
    return {"GLD": gld, "GDX": gdx, "GDXJ": gdxj, "GC=F": gold}


SERIES = _make_series()


def _row(period, aisc, published, series="MF_Q"):
    return mm.AiscRow(period, aisc, published, series, "Testkälla", "testkälla", "https://example.org/aisc")


FLAT = [_row("2000", 300, "2000-06-30", "MF_A")]       # AISC 300 för alla datum från 2000 (känt hela tiden)


@pytest.fixture(scope="module")
def gdx_frame():
    return ms.signal_frame(SERIES["GLD"], SERIES["GDX"], SERIES["GC=F"], rows=FLAT)


def _window(s, t, years=ms.WINDOW_YEARS):
    t = pd.Timestamp(t)
    return s[(s.index > t - pd.DateOffset(years=years)) & (s.index <= t)]


# ── Specen och panelens regler ───────────────────────────────────────────────
def test_spec_is_registered_and_matches_the_code():
    text = open(os.path.join(ROOT, ms.SPEC), encoding="utf-8").read()
    assert ms.SPEC == "docs/forskning/gruvor_marginal_spec.md" and ms.SPEC_VERSION == 1
    for w in ("version 1, 2026-11-03", "innan testet kördes första gången", "Det ändrar inga regler",
              "(t − 10 år, t]", "minst **3 år**", "2012-01-01", "minst **63 handelsdagar**",
              "**63, 126 och 252 handelsdagar**", "P_M ≤ 20", "P_M ≤ 30", "5 000", "frö 20261103",
              "block om 63 handelsdagar", "cirkulära block", "minst 3 episoder med hel 252-dagarshorisont",
              "ägarens beslut", "aisc_series(point_in_time=True)", "2015-11-12", "Överlevnadsbias",
              "miner-margin-study.yml", "miner_margin_study.json",
              "I H1 räknas S0:s episoder **på samma dagar som S3:s**", "S0 räknad på S3:s"):
        assert w in text, w
    assert (ms.START, ms.PERIOD, ms.WINDOW_YEARS, ms.MIN_YEARS) == ("2012-01-01", "max", 10, 3)
    assert (ms.GAP, ms.HORIZONS, ms.DD_DAYS, ms.BASE_STEP) == (63, (63, 126, 252), 252, 63)
    assert (ms.S2_M_MAX, ms.S3_M_MAX, ms.MIN_EPISODES) == (20.0, 30.0, 3)
    assert (ms.BLOCK, ms.N_BOOT, ms.SEED) == (63, 5000, 20261103)
    assert ms.TICKERS == ("GLD", "GDX", "GDXJ", "GC=F") and ms.BLOB == "miner_margin_study.json"


def test_signals_use_the_live_ratio_specs_and_the_engines_confirmation():
    """S0 är regeln före 2026-11-03 — den som övriga kvoter fortfarande har i buy_support — och S3 är regeln
    som ägaren valde för GLD/GDX och GLD/GDXJ: samma kvotnycklar, gruv-ETF:er och tröskel som panelen läser.
    (Förr låste testet S0 mot motorns egen villkorsrad; motorn läser nu buy_support.evaluate.)"""
    from alpha_regime import buy_support as bs
    for miner, key in ms.MINERS:
        spec = ms.live_spec(key)
        assert (spec.primary_num, spec.primary_den, spec.cheap_direction) == ("GLD", miner, "high")
    engine = open(os.path.join(ROOT, "alpha_regime", "engine.py"), encoding="utf-8").read()
    assert "_bs.evaluate(_rk, _ratio, result.miner_margins)" in engine
    assert bs.STRETCHED == ms.STRETCHED == "RUBBER_BAND_STRETCHED"
    assert bs.ENGINE_DRIVERS == ms.S0_DRIVERS == ("DENOMINATOR_CHEAP", "BOTH")
    assert bs.MARGIN_RULE == {key: miner for miner, key in ms.MINERS}
    assert bs.MARGIN_MAX_PCT == ms.S3_M_MAX == 30.0 and bs.SPEC == ms.SPEC


def test_no_rule_or_panel_code_reads_the_study():
    """Studien är forskning: ingen regel, räkning eller panelkod läser den."""
    for folder, _dirs, files in os.walk(os.path.join(ROOT, "alpha_regime")):
        for name in files:
            if name.endswith(".py"):
                assert "miner_margin_study" not in open(os.path.join(folder, name), encoding="utf-8").read(), name


# ── Signalerna är panelens funktioner ────────────────────────────────────────
def _check_days(fr, n_random=6, seed=4):
    """Några dagar per signal (första signaldagen) plus slumpade dagar där allt kan räknas."""
    ok = (fr["ok_S3"] & fr["gold"].notna()).to_numpy()
    days = []
    for sig in ("S0", "S1", "S2", "S3"):
        hit = np.nonzero(fr[sig].to_numpy() & ok)[0]
        assert len(hit), f"{sig} ger aldrig signal i testdatan"
        days += [fr.index[hit[0]], fr.index[hit[-1]]]
    rng = np.random.default_rng(seed)
    days += list(fr.index[rng.choice(np.nonzero(ok)[0], n_random, replace=False)])
    first_ratio = fr.index[np.argmax(fr["ok_S1"].to_numpy())]
    days.append(first_ratio)                            # kortaste fönstret: exakt tre år
    return days


@pytest.mark.parametrize("miner,key", ms.MINERS)
def test_signal_definitions_match_the_live_functions(monkeypatch, miner, key):
    fr = ms.signal_frame(SERIES["GLD"], SERIES[miner], SERIES["GC=F"], rows=FLAT)
    for t in _check_days(fr):
        win = {"GLD": _window(SERIES["GLD"], t), miner: _window(SERIES[miner], t)}
        monkeypatch.setattr(cr, "_download_close", lambda tk, period="10y": win[tk])
        live = cr._compute_ratio(ms.live_spec(key))
        row = fr.loc[t]
        assert live.status == row["status"] and live.driver == row["driver"], t
        assert live.percentile == round(row["p_ratio"], 1), t
        assert (live.numerator_pctile, live.denominator_pctile) == (round(row["p_gld"], 1), round(row["p_miner"], 1))
        engine_plus_one = live.status == "RUBBER_BAND_STRETCHED" and live.driver in ("DENOMINATOR_CHEAP", "BOTH")
        assert bool(row["S0"]) == engine_plus_one, t
        assert bool(row["S1"]) == (live.status == "RUBBER_BAND_STRETCHED"), t
        if not math.isnan(row["p_m"]):
            gauge = mm.build_result(miner, _window(SERIES[miner], t), _window(SERIES["GC=F"], t),
                                    today=t, rows=FLAT)
            assert gauge.status == "OK" and gauge.as_of == str(t.date())
            assert gauge.percentile == round(row["p_m"], 1), t
            assert gauge.current == pytest.approx(row["M"], rel=1e-5)
    pm = fr["p_m"].to_numpy()
    s1 = fr["S1"].to_numpy(dtype=bool)
    assert (fr["S2"].to_numpy(dtype=bool) == (np.nan_to_num(pm, nan=1e9) <= 20)).all()
    assert (fr["S3"].to_numpy(dtype=bool) == (s1 & (np.nan_to_num(pm, nan=1e9) <= 30))).all()
    assert not (fr["S0"] & ~fr["S1"]).any()             # S0 är alltid också S1


def test_percentile_window_is_ten_years_and_needs_three():
    idx = pd.bdate_range("2000-01-03", "2014-12-31")
    s = pd.Series(np.linspace(1.0, 2.0, len(idx)), index=idx)
    p = ms.pit_percentiles(s)
    three = idx[0] + pd.DateOffset(years=3)
    assert p[idx < three].isna().all() and p[idx >= three].notna().all()
    assert (p[idx >= three] == 100.0).all()             # stigande serie: dagens värde är alltid högst
    # det som ligger mer än tio år före t påverkar aldrig t
    t = pd.Timestamp("2014-06-02")
    s2 = s.copy()
    s2[s2.index <= t - pd.DateOffset(years=10)] = 99.0
    assert ms.pit_percentiles(s2)[t] == p[t]
    s3 = s.copy()
    s3[(s3.index > t - pd.DateOffset(years=10)) & (s3.index < t)] = 99.0
    assert ms.pit_percentiles(s3)[t] == pytest.approx(cr._percentile_of(_window(s3, t).to_numpy(), s3[t]))
    assert ms.pit_percentiles(s3)[t] < 1.0


# ── Point-in-time ────────────────────────────────────────────────────────────
_COLS = ["gld", "miner", "ratio", "p_ratio", "p_gld", "p_miner", "gold", "aisc", "M", "p_m", "status", "driver",
         "S0", "S1", "S2", "S3", "ok_S0", "ok_S1", "ok_S2", "ok_S3"]


@pytest.mark.parametrize("t", ["2013-03-28", "2019-06-28", "2023-01-31"])
def test_changing_data_after_t_never_changes_the_signal_at_t(gdx_frame, t):
    t = pd.Timestamp(t)
    rng = np.random.default_rng(11)

    def shake(s):
        s = s.copy()
        after = s.index > t
        s[after] = s[after] * rng.uniform(0.3, 3.0, int(after.sum()))
        return s
    rows = FLAT + [_row("2026Q1", 50, (t + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))]   # ny AISC efter t
    shaken = ms.signal_frame(shake(SERIES["GLD"]), shake(SERIES["GDX"]), shake(SERIES["GC=F"]), rows=rows)
    cut = ms.signal_frame(SERIES["GLD"][:t], SERIES["GDX"][:t], SERIES["GC=F"][:t], rows=FLAT)
    before = gdx_frame.loc[:t, _COLS]
    pd.testing.assert_frame_equal(shaken.loc[:t, _COLS], before)
    pd.testing.assert_frame_equal(cut.loc[:t, _COLS], before)
    assert not shaken.loc[t:, "p_ratio"].iloc[1:].equals(gdx_frame.loc[t:, "p_ratio"].iloc[1:])   # skakningen bet


def test_aisc_is_not_used_before_its_published_date():
    rows = FLAT + [_row("2019Q4", 900, "2020-04-15")]
    fr = ms.signal_frame(SERIES["GLD"], SERIES["GDX"], SERIES["GC=F"], rows=rows)
    pub = pd.Timestamp("2020-04-15")
    a = fr["aisc"].dropna()
    assert (a[a.index < pub] == 300.0).all() and (a[a.index >= pub] == 900.0).all()
    # 2019Q4-raden används inte för Q4-dagarna 2019 — den fanns inte då
    assert (a["2019-10-01":"2019-12-31"] == 300.0).all()
    other = ms.signal_frame(SERIES["GLD"], SERIES["GDX"], SERIES["GC=F"],
                            rows=FLAT + [_row("2019Q4", 1500, "2020-04-15")])
    pd.testing.assert_frame_equal(other.loc[:pub - pd.Timedelta(days=1), _COLS],
                                  fr.loc[:pub - pd.Timedelta(days=1), _COLS])
    assert not other.loc[pub:, "M"].equals(fr.loc[pub:, "M"])
    on = fr.loc[pub:].dropna(subset=["M"]).iloc[0]
    assert on["M"] == pytest.approx(on["miner"] / (on["gold"] - 900.0))


def test_real_table_point_in_time_starts_the_margin_signals_in_late_2018():
    fr = ms.signal_frame(SERIES["GLD"], SERIES["GDX"], SERIES["GC=F"])      # riktiga AISC_TABLE
    m = fr["M"].dropna()
    assert m.index[0] >= pd.Timestamp("2015-11-12") and fr.loc[:"2015-11-11", "aisc"].isna().all()
    first_pm = fr["p_m"].dropna().index[0]
    assert pd.Timestamp("2018-11-12") <= first_pm <= pd.Timestamp("2018-11-30")
    for d, v in fr["aisc"].dropna().items():             # aldrig ett värde före sitt publiceringsdatum
        known = [r for r in mm.AISC_TABLE if pd.Timestamp(r.published) <= d]
        assert known and v in {float(r.aisc) for r in known}, d
    assert not fr.loc[:first_pm - pd.Timedelta(days=1), ["S2", "S3"]].any().any()


# ── Episoder och utfall ──────────────────────────────────────────────────────
def test_episodes_need_63_days_without_the_signal():
    sig = [False] * 5 + [True] * 3 + [False] * 63 + [True] + [False] * 62 + [True, True]
    assert ms.episode_starts(sig) == [5, 71]
    assert ms.episode_starts([True] + [False] * 63 + [True]) == [0, 64]
    assert ms.episode_starts([True] + [False] * 62 + [True]) == [0]
    assert ms.episode_starts([False] * 10) == []
    assert ms.episode_starts(sig, gap=10) == [5, 71, 134]


def test_entry_is_the_next_close_and_only_complete_horizons_count():
    m = np.array([10.0, 11.0, 12.0, 9.0, 15.0, 6.0, 8.0])
    g = np.array([100.0, 100.0, 110.0, 100.0, 100.0, 100.0, 120.0])
    tot, rel = ms.forward_returns(m, g, 2)
    assert tot[0] == pytest.approx(9.0 / 11.0 - 1)      # köp dagen efter (11), sälj två dagar senare (9)
    assert rel[0] == pytest.approx((9.0 / 11.0 - 1) - (100.0 / 100.0 - 1))
    assert rel[1] == pytest.approx((15.0 / 12.0 - 1) - (100.0 / 110.0 - 1))
    assert np.isnan(tot[4:]).all() and np.isnan(rel[4:]).all() and np.isfinite(tot[:4]).all()
    dd = ms.worst_drawdown(m, 3)
    assert dd[0] == pytest.approx(9.0 / 12.0 - 1)       # 11 → 12 → 9 → 15
    assert dd[1] == pytest.approx(6.0 / 15.0 - 1)       # 12 → 9 → 15 → 6
    assert np.isnan(dd[3:]).all()


def test_analyze_episodes_coverage_and_baselines(gdx_frame):
    res = ms.analyze(gdx_frame, n_boot=200)
    cov = res["coverage"]
    assert cov["S0"] == cov["S1"] and cov["S2"]["first"] >= cov["S1"]["first"]
    for sig, _ in ms.SIGNALS:
        eps = res["signals"][sig]["episodes"]
        starts = [pd.Timestamp(e["start"]) for e in eps]
        assert all(s >= pd.Timestamp(ms.START) for s in starts)
        pos = [gdx_frame.index.get_loc(s) for s in starts]
        assert all(b - a > ms.GAP for a, b in zip(pos, pos[1:]))
        for e, p in zip(eps, pos):
            assert e["entry"] == str(gdx_frame.index[p + 1].date())
            assert bool(gdx_frame[sig].iloc[p])
            h = 252
            if p + 1 + h < len(gdx_frame):
                m, g = gdx_frame["miner"].to_numpy(), gdx_frame["gld"].to_numpy()
                assert e["rel_252"] == pytest.approx((m[p + 1 + h] / m[p + 1]) - (g[p + 1 + h] / g[p + 1]), abs=1e-5)
            else:
                assert e["rel_252"] is None and e["dd_252"] is None
        assert res["signals"][sig]["summary"]["n"] == len(eps)
    base = res["baseline"]["kvotperioden"]
    assert base["alla dagar"]["n"] == cov["S1"]["days"]
    assert base["var 63:e dag"]["n"] == -(-cov["S1"]["days"] // 63)
    assert base["alla dagar"]["n_252"] == cov["S1"]["days"] - 253
    marg = res["baseline"]["marginalperioden"]
    assert marg["first"] == cov["S2"]["first"] == res["hypotheses"]["H2"]["period_from"]
    h2 = res["hypotheses"]["H2"]
    assert marg["alla dagar"]["n_252"] == h2["baseline_days"]
    assert marg["alla dagar"]["rel_252"] == pytest.approx(h2["baseline_mean"], abs=1e-5)


# ── Bootstrap ────────────────────────────────────────────────────────────────
def test_bootstrap_is_deterministic_with_the_seed():
    rng = np.random.default_rng(1)
    x = rng.normal(0.05, 0.2, 1500)
    mask = np.zeros(1500, dtype=bool)
    mask[50::100] = True                                 # 15 episoder: intervallets ändar beror på dragningen
    a = ms.block_bootstrap(x, mask, seed=ms.SEED)
    b = ms.block_bootstrap(x, mask, seed=ms.SEED)
    c = ms.block_bootstrap(x, mask, seed=ms.SEED + 1)
    assert a == b and a["ci"] != c["ci"]
    assert a["draws"] + a["empty"] == ms.N_BOOT and (a["block"], a["seed"], a["days"]) == (63, 20261103, 1500)
    lo, hi = a["ci"]
    assert lo <= float(x[mask].mean()) <= hi and min(x[mask]) - 1e-9 <= lo and hi <= max(x[mask]) + 1e-9


def test_bootstrap_edge_cases_and_circular_blocks():
    x = np.linspace(-0.1, 0.3, 700)
    one = np.zeros(700, dtype=bool)
    one[350] = True
    r = ms.block_bootstrap(x, one, n_boot=500)
    assert r["ci"] == [pytest.approx(x[350], abs=1e-12), pytest.approx(x[350], abs=1e-12)]
    assert ms.block_bootstrap(x, np.zeros(700, dtype=bool))["ci"] == [None, None]
    assert ms.block_bootstrap(x[:50], one[:50] | True)["ci"] == [None, None]      # kortare än ett block
    # cirkulära block: en episod på första dagen dras lika ofta som en mitt i perioden
    first = np.zeros(700, dtype=bool)
    first[0] = True
    e_first = ms.block_bootstrap(x, first, n_boot=4000)["empty"]
    e_mid = ms.block_bootstrap(x, one, n_boot=4000)["empty"]
    assert abs(e_first - e_mid) < 250


# ── Domar och förslag ────────────────────────────────────────────────────────
def _arrays(values):
    return {"rel_252": np.asarray(values, dtype=float)}


def test_h1_more_episodes_positive_and_at_least_s0():
    rel = np.full(2000, np.nan)
    rel[[100, 300, 500, 700]] = [0.10, 0.20, 0.05, 0.15]
    rel[[900]] = [0.30]
    cov = {"S3": {"first": "2018-11-12", "pos": 50}}
    h = ms._h1(_arrays(rel), {"S3": [100, 300, 500, 700], "S0": []}, cov)
    assert h["pass"] and h["S0_no_signal"] and h["S3_mean"] == pytest.approx(0.125)
    h = ms._h1(_arrays(rel), {"S3": [100, 300, 500, 700], "S0": [10, 900]}, cov)   # S0 före perioden räknas inte
    assert not h["pass"] and h["S0_episodes"] == 1 and "S3:s snitt är under S0:s" in h["why"]
    h = ms._h1(_arrays(rel), {"S3": [100, 300], "S0": []}, cov)
    assert not h["pass"] and "S3 har färre än 3 hela episoder" in h["why"]
    h = ms._h1(_arrays(rel), {"S3": [100, 300, 500], "S0": [200, 400, 600]}, cov)
    assert not h["pass"] and "S3 har inte fler episoder än S0" in h["why"]
    assert not ms._h1(_arrays(rel), {"S3": [], "S0": []}, {"S3": None})["pass"]


def _boundary_frame(s0, s3, margin_from, n=1400):
    """Fejkad signalram från 2012: S0 (= S1) på dagarna i s0, S3 på dagarna i s3, M från margin_from."""
    pos = np.arange(n)
    ok3 = pos >= margin_from
    fr = pd.DataFrame(index=pd.bdate_range("2012-01-02", periods=n))
    fr["miner"], fr["gld"] = 10.0 * 1.001 ** pos, 100.0
    for c in ("p_ratio", "p_gld", "p_miner", "p_m"):
        fr[c] = 50.0
    fr["driver"] = "BOTH"
    on0 = np.isin(pos, list(s0))
    fr["S0"], fr["S1"], fr["S2"], fr["S3"] = on0, on0, False, np.isin(pos, list(s3)) & ok3
    fr["ok_S0"], fr["ok_S1"], fr["ok_S2"], fr["ok_S3"] = True, True, ok3, ok3
    return fr


def test_h1_counts_s0_on_the_same_days_as_s3_at_the_start_of_the_margin_period():
    """En S0-period som redan pågår när marginalperioden börjar öppnar en episod för S0 också, precis som S3:s
    första signaldag — annars fick S3 en extra episod som kunde avgöra H1."""
    run = range(400, 701)
    fr = _boundary_frame(run, run, 500)
    assert ms.h1_starts(fr) == {"S3": [500], "S0": [500]}
    res = ms.analyze(fr, n_boot=50)
    h1 = res["hypotheses"]["H1"]
    assert (h1["S3_episodes"], h1["S0_episodes"]) == (1, 1) and "S3 har inte fler episoder än S0" in h1["why"]
    assert h1["S3_mean"] == h1["S0_mean"] and h1["S3_dates"] == h1["S0_dates"] == [str(fr.index[500].date())]
    assert [e["start"] for e in res["signals"]["S0"]["episodes"]] == [str(fr.index[400].date())]   # tabellen: hela
    text = "\n".join(ms._hyp_lines(res["hypotheses"]))
    assert f"H1:s episoder (S0 räknad på S3:s dagar): S3 {fr.index[500].date()} · S0 {fr.index[500].date()}" in text
    # S3 slår till senare i samma S0-period: S0 öppnar ändå på periodens första dag
    assert ms.h1_starts(_boundary_frame(run, range(600, 701), 500)) == {"S3": [600], "S0": [500]}
    # en S0-period som tog slut före marginalperioden räknas inte, och glappet mäts på samma dagar för båda
    assert ms.h1_starts(_boundary_frame(range(300, 450), [], 500)) == {"S3": [], "S0": []}
    both = list(range(400, 521)) + list(range(560, 600)) + list(range(700, 720))
    assert ms.h1_starts(_boundary_frame(both, both, 500)) == {"S3": [500, 700], "S0": [500, 700]}
    assert ms.h1_starts(_boundary_frame(run, run, 1400)) == {"S3": [], "S0": []}            # ingen marginal


def test_h1_starts_on_the_real_table_follow_the_definition(gdx_frame):
    fr = ms.signal_frame(SERIES["GLD"], SERIES["GDX"], SERIES["GC=F"])      # riktiga AISC_TABLE
    pos = ms.coverage(fr, "S3")["pos"]
    h1 = ms.h1_starts(fr)
    ok = fr["ok_S3"].to_numpy(dtype=bool)
    assert not (fr["S3"].to_numpy(dtype=bool) & ~ok).any()           # S3 finns bara där M finns
    assert h1["S3"] == [i for i in ms.episode_starts(fr["S3"].to_numpy()) if i >= pos]
    assert h1["S0"] == [i for i in ms.episode_starts(fr["S0"].to_numpy(dtype=bool) & ok) if i >= pos]
    res = ms.analyze(fr, n_boot=50)["hypotheses"]["H1"]
    assert (res["S3_episodes"], res["S0_episodes"]) == (len(h1["S3"]), len(h1["S0"]))


def test_h2_needs_mean_and_lower_bound_above_the_baseline():
    rng = np.random.default_rng(2)
    rel = rng.normal(0.0, 0.02, 1500)
    starts = [200, 500, 800, 1100, 1400]
    rel[starts] = 0.5
    cov = {"S2": {"first": "2018-11-12", "pos": 0}}
    h = ms._h2(_arrays(rel), {"S2": starts}, cov, ms.SEED, 1000)
    assert h["pass"] and h["S2_complete"] == 5 and h["baseline_days"] == 1500
    assert h["bootstrap"]["ci"][0] > h["baseline_mean"]
    below = rel.copy()
    below[starts] = -0.05
    h = ms._h2(_arrays(below), {"S2": starts}, cov, ms.SEED, 1000)
    assert not h["pass"] and "S2:s snitt är inte över baslinjen" in h["why"]
    wide = rel.copy()
    wide[starts] = [0.5, -0.4, 0.3, -0.3, 0.2]          # snittet över baslinjen, men intervallet når under
    h = ms._h2(_arrays(wide), {"S2": starts}, cov, ms.SEED, 1000)
    assert h["S2_mean"] > h["baseline_mean"] and h["why"] == ["intervallets nedre gräns är inte över baslinjen"]
    few = ms._h2(_arrays(rel), {"S2": starts[:2]}, cov, ms.SEED, 1000)
    assert not few["pass"] and "S2 har färre än 3 hela episoder" in few["why"]
    tail = rel.copy()
    tail[-252:] = np.nan                                 # ofullständig horisont: dagarna räknas inte
    assert ms._h2(_arrays(tail), {"S2": starts}, cov, ms.SEED, 1000)["S2_complete"] == 4


def test_h3_needs_three_complete_positive_episodes():
    rel = np.full(1000, np.nan)
    rel[[10, 200, 400]] = [0.1, -0.05, 0.2]
    cov = {"S1": {"first": "2012-01-03", "pos": 0}}
    assert ms._h3(_arrays(rel), {"S1": [10, 200, 400]}, cov)["pass"]
    h = ms._h3(_arrays(rel), {"S1": [10, 200, 400, 990]}, cov)        # 990 utan hel horisont
    assert h["pass"] and (h["S1_episodes"], h["S1_complete"]) == (4, 3)
    assert not ms._h3(_arrays(rel), {"S1": [10, 400]}, cov)["pass"]
    rel[400] = -0.3
    assert not ms._h3(_arrays(rel), {"S1": [10, 200, 400]}, cov)["pass"]


def test_proposals_follow_the_spec():
    yes, no = {"pass": True}, {"pass": False}
    p = ms.proposals("GDX", "GLD/GDX", {"H1": yes, "H2": no, "H3": no})
    assert len(p) == 1 and "ersätt drivarens prisvillkor" in p[0] and "P_M ≤ 30" in p[0]
    p = ms.proposals("GDX", "GLD/GDX", {"H1": yes, "H2": no, "H3": yes})
    assert "ersätt" in p[0] and "alternativet" in p[1]
    p = ms.proposals("GDX", "GLD/GDX", {"H1": no, "H2": no, "H3": yes})
    assert p == ["GLD/GDX: H3 håller men inte H1 → förslag: ta bort drivarens nedgradering, så att kvoten ≥ 90 "
                 "räcker."]
    p = ms.proposals("GDXJ", "GLD/GDXJ", {"H1": no, "H2": yes, "H3": no})
    assert "egen bekräftelse" in p[0] and "approximation" in p[0]
    assert ms.proposals("GDX", "GLD/GDX", {"H1": no, "H2": no, "H3": no}) == [
        "GLD/GDX: inget håller → regeln behålls, och marginalmätaren är kvar som kontext."]


# ── Hela körningen utan nätverk ──────────────────────────────────────────────
def test_report_runs_without_network_and_saves_one_json_blob():
    saved = {}
    res = ms.run(getter=lambda t: SERIES[t], save=lambda name, data: saved.setdefault(name, data) is not None,
                 today="2026-11-03")
    assert list(saved) == [ms.BLOB] and res["saved"] is True and "error" not in res
    json.dumps(saved[ms.BLOB], allow_nan=False)          # ren JSON, inga NaN
    assert set(res["miners"]) == {"GDX", "GDXJ"}
    for miner, m in res["miners"].items():
        assert set(m["hypotheses"]) == {"H1", "H2", "H3"}
        assert all(isinstance(h["pass"], bool) and h["horizon"] == 252 for h in m["hypotheses"].values())
        assert m["ratio"] == f"GLD/{miner}" and m["proposals"]
        assert m["hypotheses"]["H2"]["bootstrap"]["n_boot"] == ms.N_BOOT
    a = res["aisc"]
    assert a["rows"] == len(mm.AISC_TABLE) and a["pit_from"] == "2015-11-12" and a["newest_period"] == "2026Q1"
    assert res["data"]["GDXJ"]["first"] == "2009-11-11"
    text = "\n".join(ms.report_lines(res))
    for w in ("GRUVORNA MOT MARGINALEN", "== GDX (GLD/GDX", "== GDXJ (GLD/GDXJ", "S3 föreslagen ersättare",
              "bas alla dagar, marginal", "Episoder", "H1 S3 mot S0", "H2 S2 mot baslinjen", "H3 S1 utan drivaren",
              "AISC-tabellen: 35 rader", "Förslag enligt specen", "ändrar inga regler"):
        assert w in text, w
    # Efter ägarens beslut A (2026-11-03): S0 är regeln före beslutet, och mätaren är inte längre 'bara kontext'.
    # Bara text — signalernas nycklar och beräkningarna är desamma.
    assert [s for s, _ in ms.SIGNALS] == ["S0", "S1", "S2", "S3"]
    assert all(m["signals"]["S0"]["label"] == "regeln före 2026-11-03" for m in res["miners"].values())
    assert "S0 regeln före 2026-11-03" in text and "nuvarande regel" not in text and "bara kontext" not in text
    assert ("Sedan 2026-11-03 (ägarens beslut A efter första körningen) är S3 panelens köpstöd för GLD/GDX och "
            "GLD/GDXJ; S0 är regeln före dess.") in text


def test_report_runs_when_the_margin_signals_cannot_be_computed():
    """Utan någon känd AISC kan S2/S3 aldrig räknas — rapporten går ändå, och H1/H2 faller med skäl."""
    res = ms.run(getter=lambda t: SERIES[t], no_save=True, rows=[], today="2026-11-03")
    gdx = res["miners"]["GDX"]
    assert gdx["coverage"]["S2"] is None and gdx["coverage"]["S3"] is None
    assert gdx["baseline"]["marginalperioden"] is None
    assert gdx["hypotheses"]["H1"]["why"] == ["S3 kan inte räknas (ingen marginalhistorik)"]
    assert gdx["hypotheses"]["H2"]["why"] == ["S2 kan inte räknas (ingen marginalhistorik)"]
    assert gdx["signals"]["S2"]["signal_days"] == 0 and gdx["signals"]["S3"]["episodes"] == []
    text = "\n".join(ms.report_lines(res))
    assert "AISC-tabellen: tom" in text and "S2/S3 kan inte räknas" in text and "bas marginal: kan inte räknas" in text
    assert "H1 → EJ GODKÄND (S3 kan inte räknas" in text


def test_missing_prices_exit_non_zero_and_save_nothing(monkeypatch):
    calls = []
    short = dict(SERIES, GDXJ=pd.Series(dtype=float))
    res = ms.run(getter=lambda t: short[t], save=lambda n, d: calls.append(n) or True)
    assert res["missing"] == ["GDXJ"] and "GDXJ" in res["error"] and calls == [] and "miners" not in res
    assert any("FEL:" in line for line in ms.report_lines(res))
    monkeypatch.setattr(cr, "_download_close", lambda t, period="10y": short[t])
    assert ms.main(["--no-save"]) == 1
    full = dict(SERIES)
    monkeypatch.setattr(cr, "_download_close", lambda t, period="10y": full[t] if period == "max" else None)
    assert ms.main(["--no-save"]) == 0                   # 0 även när hypoteserna faller


def test_workflow_is_manual_and_runs_the_study():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load(open(os.path.join(ROOT, ".github", "workflows", "miner-margin-study.yml"), encoding="utf-8"))
    on = wf[True] if True in wf else wf["on"]
    assert list(on) == ["workflow_dispatch"]
    step = wf["jobs"]["miner-margin"]["steps"][-1]
    assert step["run"] == "python miner_margin_study.py"
    assert step["env"] == {"BORSDATA_API_KEY": "${{ secrets.BORSDATA_API_KEY }}",
                           "GITHUB_TOKEN": "${{ secrets.GIST_TOKEN }}"}
