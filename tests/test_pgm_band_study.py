"""
PGM-snoddarna — förregistrerat backtest (pgm_band_study.py, docs/forskning/pgm_snoddar_spec.md).

Signalerna S0–S2 ska vara panelens egna funktioner räknade point-in-time: S0 är motorns +1 i dag
(buy_support.evaluate med ENGINE_DRIVERS), S1 kvoten ≥ 90 och S2 kvoten och metallterminen över MA200. Ingen
framåtblick (data efter t ändrar aldrig signalen eller MA200-flaggan på t), episoder skilda med 63 dagar, köp
första utfallsdagen efter signaldagen, hela horisonter, en bootstrap som ger samma svar med samma frö, gruvorna och
korsutfallen som beskrivande tabeller och en rapport som går utan nätverk. Studien läser miner_margin_study:s
funktioner, och gruvornas studie räknar som förut. Inget nätverk och ingen skrivning till repot eller Gisten —
fejkade kursserier.
"""
import json
import os
import socket

import numpy as np
import pandas as pd
import pytest

import miner_margin_study as ms
import pgm_band_study as pg
from alpha_regime import buy_support as bs
from alpha_regime import commodity_ratios as cr

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
    idx = pd.bdate_range("2000-08-30", "2026-10-30")
    gold = _path(idx, [("2000-08-30", 275), ("2004-11-18", 440), ("2008-03-01", 980), ("2008-10-24", 720),
                       ("2011-09-01", 1880), ("2013-06-28", 1200), ("2015-12-17", 1050), ("2016-07-06", 1370),
                       ("2018-08-16", 1175), ("2020-08-06", 2060), ("2022-09-28", 1630), ("2024-10-30", 2790),
                       ("2025-10-20", 4350), ("2026-10-30", 4100)], 0.008, 1)
    plat = _path(idx, [("2000-08-30", 580), ("2003-01-01", 600), ("2008-03-04", 2250), ("2008-10-27", 780),
                       ("2011-08-02", 1880), ("2013-01-01", 1650), ("2015-12-01", 830), ("2016-08-10", 1180),
                       ("2018-09-01", 780), ("2020-03-16", 620), ("2021-02-16", 1290), ("2022-09-01", 850),
                       ("2024-10-01", 1000), ("2025-10-15", 1650), ("2026-10-30", 1550)], 0.014, 4)
    pall = _path(idx, [("2000-08-30", 760), ("2001-01-15", 1090), ("2003-04-01", 160), ("2008-03-04", 580),
                       ("2008-12-01", 180), ("2011-02-01", 830), ("2016-01-20", 470), ("2018-01-01", 1100),
                       ("2020-02-27", 2800), ("2022-03-07", 3300), ("2024-08-05", 480), ("2025-10-15", 640),
                       ("2026-10-30", 600)], 0.018, 5)
    gld = (gold / 10.0 * np.exp(np.random.default_rng(9).normal(0, 0.001, len(idx))))[idx >= "2004-11-18"]
    pplt = (plat / 10.0 * np.exp(np.random.default_rng(10).normal(0, 0.002, len(idx))))[idx >= "2010-01-08"]
    pall_etf = (pall / 10.0 * np.exp(np.random.default_rng(11).normal(0, 0.002, len(idx))))[idx >= "2010-01-08"]
    impuy = (plat / 100.0) ** 1.6 * np.exp(np.cumsum(np.random.default_rng(12).normal(0, 0.01, len(idx))) * 0.2)
    sbsw = ((plat * pall) ** 0.5 / 100.0) ** 1.5 * np.exp(np.random.default_rng(13).normal(0, 0.01, len(idx)))
    # Olika luckor i varje serie: terminerna och ETF:erna handlas inte exakt samma dagar.
    return {"GC=F": gold.drop(idx[::97]), "PL=F": plat.drop(idx[3::89]), "PA=F": pall.drop(idx[5::101]),
            "GLD": gld.drop(gld.index[7::113]), "PPLT": pplt.drop(pplt.index[11::131]), "PALL": pall_etf,
            "IMPUY": impuy[idx >= "2005-01-03"], "SBSW": sbsw[idx >= "2017-02-13"]}


SERIES = _make_series()
METAL_OF = {"gold_platinum": "PL=F", "gold_palladium": "PA=F"}
ETF_OF = {"gold_platinum": "PPLT", "gold_palladium": "PALL"}


@pytest.fixture(scope="module")
def frames():
    return {key: pg.signal_frame(SERIES["GC=F"], SERIES[METAL_OF[key]]) for _, key in pg.BANDS}


@pytest.fixture(scope="module")
def result():
    return pg.study(SERIES, n_boot=300)


def _window(s, t, years=pg.WINDOW_YEARS):
    t = pd.Timestamp(t)
    return s[(s.index > t - pd.DateOffset(years=years)) & (s.index <= t)]


def _starts(fr, eps):
    return [fr.index.get_loc(pd.Timestamp(e["start"])) for e in eps]


# ── Specen och panelens regler ───────────────────────────────────────────────
def test_spec_is_registered_and_matches_the_code():
    text = open(os.path.join(ROOT, pg.SPEC), encoding="utf-8").read()
    assert pg.SPEC == "docs/forskning/pgm_snoddar_spec.md" and pg.SPEC_VERSION == 1
    for section in ("## Bakgrund", "## Data", "## Definitioner (point-in-time)", "### Signaler",
                    "## Episoder och utfall", "## Hypoteser", "## Godkännande", "## Vad utfallet leder till",
                    "## Kända begränsningar", "## Ändringslogg"):
        assert section in text, section
    for w in ("version 1, 2026-11-04", "innan testet kördes första gången", "Det ändrar inga regler", "#300",
              "(t − 10 år, t]", "minst **3 år**", "2012-01-01", "minst **63 handelsdagar**",
              "**63, 126 och 252 handelsdagar**", "`buy_support.ENGINE_DRIVERS`", "P_kvot ≥ 90", "MA200",
              "200 senaste stängningar", "5 000", "frö 20261104", "block om 63 handelsdagar", "cirkulära block",
              "minst 3 episoder med hel 252-dagarshorisont", "för få episoder", "bara kontext", "ägarens beslut",
              "IMPUY", "SBSW", "PPLT", "PALL", "Terminsrullning", "Platina och palladium är korrelerade",
              "Överlevnadsbias", "pgm-band-study.yml", "pgm_band_study.json"):
        assert w in text, w
    assert (pg.START, pg.PERIOD, pg.WINDOW_YEARS, pg.MIN_YEARS) == ("2012-01-01", "max", 10, 3)
    assert (pg.GAP, pg.HORIZONS, pg.DD_DAYS, pg.BASE_STEP) == (63, (63, 126, 252), 252, 63)
    assert (pg.RATIO_MIN, pg.MA_DAYS, pg.MIN_EPISODES) == (90.0, 200, 3)
    assert (pg.BLOCK, pg.N_BOOT, pg.SEED) == (63, 5000, 20261104)
    # samma värden som gruvornas studie, vars funktioner (summarize, outcome_arrays, missing) läser sina egna
    assert (pg.HORIZONS, pg.DD_DAYS, pg.MIN_DAYS, pg.PERIOD) == (ms.HORIZONS, ms.DD_DAYS, ms.MIN_DAYS, ms.PERIOD)


def test_the_study_reads_the_live_bands_and_the_engines_driver_rule():
    for (_name, key), (den, etf) in zip(pg.BANDS, (("PL=F", "PPLT"), ("PA=F", "PALL"))):
        spec = pg.band_spec(key)
        assert (spec.primary_num, spec.primary_den, spec.fallback_num, spec.fallback_den,
                spec.cheap_direction) == ("GC=F", den, "GLD", etf, "high")
        assert not bs.uses_margin(key)                   # drivarregeln, inte gruvornas marginalregel
    assert pg.S0_DRIVERS is bs.ENGINE_DRIVERS and pg.S0_DRIVERS == ("DENOMINATOR_CHEAP", "BOTH")
    assert pg.STRETCHED == bs.STRETCHED == "RUBBER_BAND_STRETCHED" and pg.RATIO_MIN == cr._STRETCHED_PCT == 90.0
    assert cr._LEG_CHEAP_MAX == 20.0                     # P_metall ≤ 20 i specen
    assert cr.EXPOSURE_TO_RATIO["platinum"] == ["gold_platinum", "gold_palladium"]      # därför korsvis
    assert cr.EXPOSURE_TO_RATIO["palladium"] == ["gold_palladium", "gold_platinum"]
    assert pg.REQUIRED == ("GC=F", "PL=F", "PA=F", "GLD", "PPLT", "PALL") and pg.MINERS == ("IMPUY", "SBSW")
    assert pg.TICKERS == pg.REQUIRED + pg.MINERS and pg.BLOB == "pgm_band_study.json"


def test_the_study_reuses_the_gold_miner_studys_functions():
    """Ingen egen kopia av percentilerna, episoderna, utfallen, sammanfattningen eller bootstrapen."""
    for name in ("pit_percentiles", "episode_starts", "forward_returns", "worst_drawdown", "outcome_arrays",
                 "summarize", "block_bootstrap", "coverage", "_clean", "_table_line"):
        assert not hasattr(pg, name), name
    src = open(os.path.join(ROOT, "pgm_band_study.py"), encoding="utf-8").read()
    for call in ("ms.pit_percentiles(", "ms.episode_starts(", "ms.outcome_arrays(", "ms.summarize(",
                 "ms.block_bootstrap(", "ms.coverage(", "ms._complete(", "ms._table_line(", "ms.load(",
                 "ms.data_info(", "ms.missing("):
        assert call in src, call


def test_no_rule_or_panel_code_reads_the_study():
    """Studien är forskning: ingen regel, räkning eller panelkod läser den."""
    for folder, _dirs, files in os.walk(os.path.join(ROOT, "alpha_regime")):
        for name in files:
            if name.endswith(".py"):
                assert "pgm_band_study" not in open(os.path.join(folder, name), encoding="utf-8").read(), name


def test_gold_miner_study_computes_as_before():
    """Generaliseringarna i miner_margin_study är bakåtkompatibla: förvalen ger exakt samma sak som förut."""
    rng = np.random.default_rng(3)
    fr = pd.DataFrame({"miner": np.exp(np.cumsum(rng.normal(0, 0.02, 700))),
                       "gld": np.exp(np.cumsum(rng.normal(0, 0.01, 700)))})
    a = ms.outcome_arrays(fr)
    b = ms.outcome_arrays(fr.rename(columns={"miner": "x", "gld": "y"}), asset="x", bench="y")
    assert set(a) == {"tot_63", "rel_63", "tot_126", "rel_126", "tot_252", "rel_252", "dd_252"} == set(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    tot, rel = ms.forward_returns(fr["miner"], fr["gld"], 252)
    np.testing.assert_array_equal(a["rel_252"], rel)
    np.testing.assert_array_equal(a["dd_252"], ms.worst_drawdown(fr["miner"], 252))
    assert list(ms.data_info({})) == list(ms.TICKERS) and ms.missing({}) == list(ms.TICKERS)
    assert list(ms.load(lambda t: t)) == list(ms.TICKERS)
    assert list(ms.load(lambda t: t, ("A", "B"))) == ["A", "B"] and ms.missing({}, ("A",)) == ["A"]


# ── Signalerna är panelens funktioner ────────────────────────────────────────
def _check_days(fr, n_random=6, seed=4):
    """Första och sista signaldagen per signal, slumpade dagar där allt kan räknas och första dagen med fönster."""
    ok = fr["ok_S2"].to_numpy(dtype=bool)
    days = []
    for sig, _ in pg.SIGNALS:
        hit = np.nonzero(fr[sig].to_numpy(dtype=bool) & ok)[0]
        assert len(hit), f"{sig} ger aldrig signal i testdatan"
        days += [fr.index[hit[0]], fr.index[hit[-1]]]
    rng = np.random.default_rng(seed)
    days += list(fr.index[rng.choice(np.nonzero(ok)[0], n_random, replace=False)])
    days.append(fr.index[np.argmax(fr["ok_S1"].to_numpy())])         # kortaste fönstret: exakt tre år
    return days


@pytest.mark.parametrize("name,key", pg.BANDS)
def test_signal_definitions_match_the_live_functions(monkeypatch, frames, name, key):
    fr = frames[key]
    spec = pg.band_spec(key)
    den = spec.primary_den
    for t in _check_days(fr):
        win = {"GC=F": _window(SERIES["GC=F"], t), den: _window(SERIES[den], t)}
        monkeypatch.setattr(cr, "_download_close", lambda tk, period="10y": win.get(tk, pd.Series(dtype=float)))
        live = cr._compute_ratio(spec)
        row = fr.loc[t]
        assert live.error is None and (live.numerator_label, live.denominator_leg_label) == ("GC=F", den)
        assert live.status == row["status"] and live.driver == row["driver"], t
        assert live.percentile == round(row["p_ratio"], 1), t
        assert (live.numerator_pctile, live.denominator_pctile) == (round(row["p_gc"], 1), round(row["p_metal"], 1))
        engine = bs.evaluate(key, live)                  # motorns +1 i dag: sträckt och ENGINE_DRIVERS
        assert engine.details["rule"] == "driver"
        assert bool(row["S0"]) == engine.supported, t
        assert bool(row["S1"]) == (live.status == "RUBBER_BAND_STRETCHED") == (row["p_ratio"] >= 90), t
        own = SERIES[den][:t].iloc[-200:]                # terminens egna 200 senaste stängningar till och med t
        assert row["ma200"] == pytest.approx(own.mean(), rel=1e-9)
        assert bool(row["S2"]) == (bool(row["S1"]) and SERIES[den][t] > own.mean()), t
    s0, s1, s2 = (fr[s].to_numpy(dtype=bool) for s in ("S0", "S1", "S2"))
    assert not (s0 & ~s1).any() and not (s2 & ~s1).any()  # S0 och S2 är alltid också S1
    assert (s2 == (s1 & (fr["above_ma"].to_numpy() == 1.0))).all()
    assert (fr["ok_S2"] <= fr["ok_S1"]).all() and (fr["ok_S0"] == fr["ok_S1"]).all()


def test_ma200_is_the_mean_of_the_last_200_closes_on_the_metals_own_days(frames):
    s = SERIES["PL=F"]
    ma = pg.ma_frame(s)
    assert ma["ma200"].iloc[:199].isna().all() and ma["above_ma"].iloc[:199].isna().all()
    for i in (199, 1000, 4321, len(s) - 1):
        mean = s.iloc[i - 199:i + 1].mean()
        assert ma["ma200"].iloc[i] == pytest.approx(mean, rel=1e-9)
        assert ma["above_ma"].iloc[i] == float(s.iloc[i] > mean)
    # terminens egna dagar: en dag med platina men utan guldkurs är ingen signaldag men räknas in i MA200
    fr = frames["gold_platinum"]
    no_gold = s.index.difference(SERIES["GC=F"].index)
    assert len(no_gold) > 50 and not no_gold.isin(fr.index).any()
    t = fr.index[fr.index > no_gold[50]][0]
    assert fr.loc[t, "ma200"] == pytest.approx(s[:t].iloc[-200:].mean(), rel=1e-9)
    # trenden: en stigande serie står över MA200, en fallande under
    up = pd.Series(np.linspace(1.0, 2.0, 400), index=pd.bdate_range("2020-01-01", periods=400))
    assert (pg.ma_frame(up)["above_ma"].dropna() == 1.0).all()
    assert (pg.ma_frame(5.0 - up)["above_ma"].dropna() == 0.0).all()
    assert pg.ma_frame(up)["above_ma"].notna().sum() == 201


def test_s2_follows_the_trend_and_not_the_ratio_alone():
    """Kvoten sträckt hela vägen; metallen faller först (fallande kniv: ingen S2) och vänder sedan upp. S2 slår
    till först när stängningen har gått över MA200 och står kvar så länge uppgången håller."""
    idx = pd.bdate_range("2005-01-03", "2013-12-31")
    n = len(idx)
    turn = idx.get_loc(idx[idx >= "2012-06-01"][0])
    metal = np.concatenate([np.linspace(2000.0, 600.0, turn), np.linspace(600.0, 1200.0, n - turn)])
    gold = pd.Series(metal * np.linspace(1.0, 5.0, n), index=idx)       # kvoten stiger hela tiden: P_kvot = 100
    fr = pg.signal_frame(gold, pd.Series(metal, index=idx))
    on = fr.loc["2012-01-01":]
    assert on["S1"].all()
    s2 = on["S2"].to_numpy(dtype=bool)
    first = int(np.argmax(s2))
    cross = on.index[first]
    assert s2.any() and cross > idx[turn] and s2[first:].all() and not s2[:first].any()   # väntar ut kniven
    metal_on, ma_on = on["metal"].to_numpy(), on["ma200"].to_numpy()
    assert metal_on[first] > ma_on[first] and metal_on[first - 1] <= ma_on[first - 1]   # dagen den går över
    assert on.loc[:idx[turn - 1], "S0"].all()            # drivaren: metallen billig mot sig själv — S0 köper i fallet
    res = pg.analyze(fr, pd.Series(metal, index=idx) / 10, gold / 10, n_boot=20)
    assert [e["start"] for e in res["signals"]["S2"]["episodes"]] == [str(cross.date())]
    assert res["signals"]["S2"]["episodes"][0]["above_ma"] is True
    assert res["signals"]["S1"]["episodes"] == [] and res["signals"]["S0"]["episodes"] == []   # pågick redan 2012


# ── Point-in-time ────────────────────────────────────────────────────────────
_COLS = ["gc", "metal", "ratio", "p_ratio", "p_gc", "p_metal", "ma200", "above_ma", "status", "driver",
         "S0", "S1", "S2", "ok_S0", "ok_S1", "ok_S2"]


@pytest.mark.parametrize("key,t", [("gold_platinum", "2014-03-31"), ("gold_platinum", "2019-06-28"),
                                   ("gold_palladium", "2023-01-31")])
def test_changing_data_after_t_never_changes_the_signal_or_the_ma200_flag_at_t(frames, key, t):
    t = pd.Timestamp(t)
    rng = np.random.default_rng(11)
    den = METAL_OF[key]

    def shake(s):
        s = s.copy()
        after = s.index > t
        s[after] = s[after] * rng.uniform(0.3, 3.0, int(after.sum()))
        return s
    shaken = pg.signal_frame(shake(SERIES["GC=F"]), shake(SERIES[den]))
    cut = pg.signal_frame(SERIES["GC=F"][:t], SERIES[den][:t])
    before = frames[key].loc[:t, _COLS]
    pd.testing.assert_frame_equal(shaken.loc[:t, _COLS], before)
    pd.testing.assert_frame_equal(cut.loc[:t, _COLS], before)
    later = frames[key].loc[t:].iloc[1:]
    assert not shaken.loc[t:, "p_ratio"].iloc[1:].equals(later["p_ratio"])       # skakningen bet
    assert not shaken.loc[t:, "above_ma"].iloc[1:].equals(later["above_ma"])
    # episodstarterna till och med t beror inte heller på det som kom sedan
    for sig, _ in pg.SIGNALS:
        a = [i for i in ms.episode_starts(shaken[sig].to_numpy(dtype=bool)) if shaken.index[i] <= t]
        b = [i for i in ms.episode_starts(frames[key][sig].to_numpy(dtype=bool)) if frames[key].index[i] <= t]
        assert a == b, sig


# ── Episoder och utfall ──────────────────────────────────────────────────────
def test_episodes_need_63_days_without_the_signal_and_count_from_2012(frames, result):
    for name, key in pg.BANDS:
        fr = frames[key]
        for sig, _ in pg.SIGNALS:
            eps = result["metals"][name]["signals"][sig]["episodes"]
            pos = _starts(fr, eps)
            assert pos == [i for i in ms.episode_starts(fr[sig].to_numpy(dtype=bool))
                           if fr.index[i] >= pd.Timestamp(pg.START)]
            assert all(b - a > pg.GAP for a, b in zip(pos, pos[1:]))
            for p in pos:
                assert fr[sig].iloc[p] and not fr[sig].iloc[max(0, p - pg.GAP):p].any()
            assert result["metals"][name]["signals"][sig]["summary"]["n"] == len(eps)
    assert all(result["metals"][n]["signals"][s]["episodes"] for n, _ in pg.BANDS for s, _ in pg.SIGNALS)


def _fake_frame(on, n=900, first="2011-06-01"):
    """Fejkad signalram: S0 = S1 = S2 på positionerna i `on`, allt kan räknas."""
    fr = pd.DataFrame(index=pd.bdate_range(first, periods=n))
    for c in ("p_ratio", "p_gc", "p_metal"):
        fr[c] = 50.0
    fr["metal"], fr["ma200"], fr["above_ma"], fr["driver"] = 100.0, 99.0, 1.0, "BOTH"
    sig = np.isin(np.arange(n), list(on))
    for s in ("S0", "S1", "S2"):
        fr[s], fr[f"ok_{s}"] = sig, True
    return fr


def test_a_signal_already_running_in_2012_opens_no_episode_and_the_gap_is_63_days():
    fr = _fake_frame(range(100, 200))                                       # pågår över 2012-01-01
    first_2012 = fr.index.get_loc(fr.index[fr.index >= "2012-01-01"][0])
    assert 100 < first_2012 < 200
    flat = pd.Series(10.0, index=fr.index)
    res = pg.analyze(fr, flat, flat, n_boot=20)
    assert res["signals"]["S1"]["episodes"] == [] and res["coverage"]["S1"]["first"] == str(fr.index[first_2012].date())
    res = pg.analyze(_fake_frame(list(range(100, 200)) + [263, 264]), flat, flat, n_boot=20)
    assert [e["start"] for e in res["signals"]["S1"]["episodes"]] == [str(fr.index[263].date())]   # 63 dagar utan
    res = pg.analyze(_fake_frame(list(range(100, 200)) + [262, 400]), flat, flat, n_boot=20)
    assert [e["start"] for e in res["signals"]["S1"]["episodes"]] == [str(fr.index[400].date())]   # 62 räcker inte


def test_entry_is_the_first_outcome_day_after_the_signal_day_and_only_complete_horizons_count():
    out_idx = pd.bdate_range("2015-01-01", periods=420)
    sig_idx = out_idx.delete([10, 11, 300]).union(pd.DatetimeIndex(["2014-12-31", "2015-01-03", "2015-07-04"]))
    rng = np.random.default_rng(5)
    etf = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.02, len(out_idx)))), index=out_idx)
    gld = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, len(out_idx)))), index=out_idx)
    out = pg.outcomes_on(sig_idx, etf, gld)
    a = out["arrays"]
    for i, t in enumerate(sig_idx):
        later = out_idx[out_idx > t]
        if t < out_idx[0] or not len(later):                     # före utfallens historik eller sista dagen
            assert pg.entry_date(out, i) is None and np.isnan(a["rel_63"][i]) and np.isnan(a["tot_63"][i]), t
            continue
        e = out_idx.get_loc(later[0])
        assert pg.entry_date(out, i) == str(later[0].date()), t
        for h in pg.HORIZONS:
            if e + h < len(out_idx):
                tot = etf.iloc[e + h] / etf.iloc[e] - 1.0
                assert a[f"tot_{h}"][i] == pytest.approx(tot)
                assert a[f"rel_{h}"][i] == pytest.approx(tot - (gld.iloc[e + h] / gld.iloc[e] - 1.0))
            else:
                assert np.isnan(a[f"tot_{h}"][i]) and np.isnan(a[f"rel_{h}"][i])
        if e + pg.DD_DAYS < len(out_idx):
            seg = etf.iloc[e:e + pg.DD_DAYS + 1].to_numpy()
            assert a["dd_252"][i] == pytest.approx(float(np.min(seg / np.maximum.accumulate(seg) - 1.0)))
        else:
            assert np.isnan(a["dd_252"][i])
    # en signaldag utan ETF-kurs (lördag 2015-01-03) köper måndagen efter, som fredagen före
    sat = sig_idx.get_loc(pd.Timestamp("2015-01-03"))
    assert pg.entry_date(out, sat) == "2015-01-05" == pg.entry_date(out, sat - 1)
    assert a["rel_63"][sat] == a["rel_63"][sat - 1]
    empty = pg.outcomes_on(sig_idx, pd.Series(dtype=float), gld)
    assert all(np.isnan(v).all() for v in empty["arrays"].values()) and pg.entry_date(empty, 5) is None


def test_an_episode_before_the_first_quote_gets_no_outcome_and_the_buy_is_not_moved_to_the_listing_day():
    """Verifieringen: en episod före gruvans (eller ETF:ens) första notering får inget utfall. Köpet flyttas inte
    fram till noteringsdagen, inte ens när den är dagen efter t. Specen säger detsamma."""
    text = open(os.path.join(ROOT, pg.SPEC), encoding="utf-8").read()
    for w in ("En episod som börjar före gruvans första notering får därför inget utfall",
              "flyttas inte fram till noteringsdagen", "Samma sak gäller ETF:erna, i huvudutfallen och korsvis",
              "på t eller tidigare"):
        assert w in text, w
    listing = pd.Timestamp("2017-02-13")                                   # en måndag, som SBSW i fejkdatan
    out_idx = pd.bdate_range("2004-11-18", "2020-12-31")
    rng = np.random.default_rng(21)
    gld = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, len(out_idx)))), index=out_idx)
    miner = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.03, len(out_idx)))), index=out_idx)[out_idx >= listing]
    sig_idx = pd.DatetimeIndex(["2012-03-01", "2016-06-01", "2017-02-10", "2017-02-11", listing, "2017-02-14"])
    out = pg.outcomes_on(sig_idx, miner, gld)
    a = out["arrays"]
    for i in range(4):                       # år före noteringen, fredagen före och lördagen före: inget utfall
        assert pg.entry_date(out, i) is None, sig_idx[i]
        assert all(np.isnan(v[i]) for v in a.values()), sig_idx[i]
        assert out["pos"][i] == -1
    m = miner[miner.index > listing]
    assert pg.entry_date(out, 4) == "2017-02-14" and pg.entry_date(out, 5) == "2017-02-15"
    g = gld[gld.index > listing]
    assert a["rel_252"][4] == pytest.approx(m.iloc[252] / m.iloc[0] - g.iloc[252] / g.iloc[0])
    # genom analyze: gruvans utfall saknas på episoderna före noteringen men inte efter, och baslinjen likaså
    idx = pd.bdate_range("2012-01-02", "2020-12-31")
    fr = _fake_frame([idx.get_loc(pd.Timestamp(d)) for d in ("2016-06-01", "2017-02-10", "2018-03-01")],
                     n=len(idx), first="2012-01-02")
    res = pg.analyze(fr, gld, gld, miners={"SBSW": miner}, n_boot=20)
    eps = res["signals"]["S1"]["episodes"]
    assert [e["start"] for e in eps] == ["2016-06-01", "2017-02-10", "2018-03-01"]
    assert [e["miners_rel_252"]["SBSW"] is None for e in eps] == [True, True, False]
    assert res["miners"]["SBSW"]["signals"]["S1"]["n_252"] == 1
    base = res["miners"]["SBSW"]["baseline"]["n_252"]
    assert base == int(np.isfinite(pg.outcomes_on(idx, miner, gld)["arrays"]["rel_252"]).sum())
    assert base == len(miner) - 1 - 252          # bara signaldagar från noteringen: köp dagen efter, 252 dagar till


def test_episode_outcomes_are_the_metal_etf_against_gld(frames, result):
    for name, key in pg.BANDS:
        fr = frames[key]
        of = pg.outcome_frame(SERIES[ETF_OF[key]], SERIES["GLD"])
        n_checked = 0
        for sig, _ in pg.SIGNALS:
            eps = result["metals"][name]["signals"][sig]["episodes"]
            for e, p in zip(eps, _starts(fr, eps)):
                later = of.index[of.index > fr.index[p]]
                if not len(later):
                    assert e["entry"] is None and e["rel_63"] is None
                    continue
                entry = later[0]
                assert e["entry"] == str(entry.date())
                assert e["above_ma"] == bool(fr["above_ma"].iloc[p] == 1.0)
                k = of.index.get_loc(entry)
                if k + 252 < len(of):
                    n_checked += 1
                    etf, gld = of["asset"].to_numpy(), of["bench"].to_numpy()
                    assert e["tot_252"] == pytest.approx(etf[k + 252] / etf[k] - 1.0, abs=1e-5)
                    assert e["rel_252"] == pytest.approx(etf[k + 252] / etf[k] - gld[k + 252] / gld[k], abs=1e-5)
                else:
                    assert e["rel_252"] is None and e["dd_252"] is None
        assert n_checked >= 2, name


def test_baselines_cover_every_day_and_every_63rd_from_2012(frames, result):
    for name, key in pg.BANDS:
        fr, m = frames[key], result["metals"][name]
        cov = m["coverage"]["S1"]
        assert cov["first"] == str(fr.index[fr.index >= pd.Timestamp(pg.START)][0].date())
        assert m["coverage"]["S0"] == m["coverage"]["S2"] == cov            # samma täckning i praktiken
        base = m["baseline"]
        assert base["alla dagar"]["n"] == cov["days"] and base["var 63:e dag"]["n"] == -(-cov["days"] // 63)
        own = pg.outcomes_on(fr.index, SERIES[ETF_OF[key]], SERIES["GLD"])
        rel = own["arrays"]["rel_252"][fr.index.get_loc(pd.Timestamp(cov["first"])):]
        assert base["alla dagar"]["n_252"] == int(np.isfinite(rel).sum())
        assert base["alla dagar"]["rel_252"] == pytest.approx(np.nanmean(rel), abs=1e-5)
        for h in m["hypotheses"].values():
            assert h["baseline_mean"] == pytest.approx(np.nanmean(rel), abs=1e-5)
            assert h["baseline_days"] == int(np.isfinite(rel).sum()) and h["horizon"] == 252


# ── Bootstrap ────────────────────────────────────────────────────────────────
def test_bootstrap_is_deterministic_with_the_seed(frames):
    fr = frames["gold_platinum"]
    a = pg.analyze(fr, SERIES["PPLT"], SERIES["GLD"], n_boot=400)["signals"]
    b = pg.analyze(fr, SERIES["PPLT"], SERIES["GLD"], n_boot=400)["signals"]
    c = pg.analyze(fr, SERIES["PPLT"], SERIES["GLD"], seed=pg.SEED + 1, n_boot=400)["signals"]
    assert all(a[s]["bootstrap"] == b[s]["bootstrap"] for s, _ in pg.SIGNALS)
    assert any(a[s]["bootstrap"]["ci"] != c[s]["bootstrap"]["ci"] for s, _ in pg.SIGNALS)
    for s, _ in pg.SIGNALS:
        boot = a[s]["bootstrap"]
        assert (boot["seed"], boot["block"], boot["n_boot"]) == (20261104, 63, 400)
        assert boot["draws"] + boot["empty"] == 400
        lo, hi = boot["ci"]
        if boot["episodes"] >= 2:
            assert lo <= boot["mean"] + 1e-5 and boot["mean"] - 1e-5 <= hi
            assert boot["vs_baseline"] == [pytest.approx(lo - boot["baseline_mean"], abs=2e-5),
                                           pytest.approx(hi - boot["baseline_mean"], abs=2e-5)]
    # samma intervall som gruvornas bootstrap med fröet 20261104
    rng = np.random.default_rng(1)
    x = rng.normal(0.05, 0.2, 1500)
    starts = list(range(50, 1500, 100))
    mask = np.isin(np.arange(1500), starts)
    mine = pg.bootstrap(x, np.arange(1500), starts)
    theirs = ms.block_bootstrap(x, mask, seed=20261104, n_boot=5000, block=63)
    assert mine["ci"] == [ms._f(v) for v in theirs["ci"]] and mine["n_boot"] == 5000
    assert mine["empty"] == theirs["empty"] and mine["episodes"] == 15


# ── Domar och förslag ────────────────────────────────────────────────────────
def _arrays(values):
    return {"rel_252": np.asarray(values, dtype=float)}


def test_verdicts_need_three_complete_episodes_above_zero_and_the_baseline():
    rel = np.full(2000, np.nan)
    rel[[100, 300, 500, 700]] = [0.10, 0.20, 0.05, 0.15]
    cov = {s: {"first": "2012-01-03", "pos": 0} for s in ("S0", "S1", "S2")}
    st = {"S0": [100, 300, 500], "S1": [100, 300, 500, 1990], "S2": [100, 300, 700]}
    h = pg.judge("S0", _arrays(rel), st, cov, base=0.01)
    assert h["pass"] and h["verdict"] == pg.PASS and h["floor"] and h["mean"] == pytest.approx(0.35 / 3, abs=1e-5)
    h = pg.judge("S1", _arrays(rel), st, cov, base=0.01)                 # 1990 utan hel horisont räknas inte
    assert h["pass"] and (h["episodes"], h["complete"]) == (4, 3)
    h = pg.judge("S0", _arrays(rel), st, cov, base=0.2)
    assert h["verdict"] == pg.FAIL and h["why"] == ["S0:s snitt är inte över baslinjen"]
    neg = -rel
    h = pg.judge("S0", _arrays(neg), st, cov, base=-0.5)
    assert h["verdict"] == pg.FAIL and h["why"] == ["S0:s snitt är inte > 0"]
    h = pg.judge("S0", _arrays(rel), dict(st, S0=[100, 300]), cov, base=0.01)
    assert not h["pass"] and not h["floor"] and h["verdict"] == pg.TOO_FEW
    assert h["why"] == ["S0 har färre än 3 hela episoder"]
    h = pg.judge("S2", _arrays(rel), st, {"S2": None}, base=0.01)
    assert h["verdict"] == pg.TOO_FEW and h["why"] == ["S2 kan inte räknas"]
    assert pg.judge("S0", _arrays(rel), st, cov, base=None)["why"] == ["S0:s snitt är inte över baslinjen"]


def test_h3_needs_the_trend_filter_to_match_the_current_rule():
    rel = np.full(2000, np.nan)
    rel[[100, 300, 500]] = [0.10, 0.20, 0.06]                             # S2: snitt 0,12
    rel[[200, 400]] = [0.30, 0.10]                                        # S0: snitt 0,20
    cov = {s: {"first": "2012-01-03", "pos": 0} for s in ("S0", "S1", "S2")}
    st = {"S0": [200, 400], "S1": [100], "S2": [100, 300, 500]}
    h = pg.hypotheses(_arrays(rel), st, cov, base=0.0, base_days=1500)["H3"]
    assert not h["pass"] and h["why"] == ["S2:s snitt är under S0:s"] and h["S0_complete"] == 2
    st["S0"] = [400]                                                      # S0 snitt 0,10: S2 är bättre
    assert pg.hypotheses(_arrays(rel), st, cov, base=0.0)["H3"]["pass"]
    st["S0"] = [1990]                                                     # S0 utan hel episod
    h = pg.hypotheses(_arrays(rel), st, cov, base=0.0)["H3"]
    assert h["pass"] and h["S0_no_signal"] and h["S0_mean"] is None
    hyp = pg.hypotheses(_arrays(rel), st, cov, base=0.0, base_days=1500)
    assert [hyp[k]["signal"] for k in ("H1", "H2", "H3")] == ["S0", "S1", "S2"]
    assert hyp["H2"]["verdict"] == pg.TOO_FEW and hyp["H1"]["baseline_days"] == 1500


def test_proposals_follow_the_spec_in_order():
    P = {"pass": True, "floor": True, "verdict": pg.PASS}
    F = {"pass": False, "floor": True, "verdict": pg.FAIL}
    T = {"pass": False, "floor": False, "verdict": pg.TOO_FEW}

    def prop(h1, h2, h3):
        return pg.proposal("GC=F/PL=F", "gold_platinum", "PL=F", {"H1": h1, "H2": h2, "H3": h3})
    assert prop(P, P, P)["case"] == "behåll" and "behåll regeln" in prop(P, F, F)["text"]
    p = prop(F, P, P)
    assert p["case"] == "trendfilter" and "ersätt drivarvillkoret med trendfiltret" in p["text"]
    assert "PL=F över MA200" in p["text"] and p["text"].startswith("GC=F/PL=F (gold_platinum): ")
    assert prop(T, F, P)["case"] == "trendfilter"                         # H1 för få räknas som "håller inte"
    p = prop(F, P, F)
    assert p["case"] == "utan drivaren" and "ta bort drivarens nedgradering" in p["text"]
    assert prop(T, P, T)["case"] == "utan drivaren"
    p = prop(F, F, F)
    assert p["case"] == "bara kontext" and "inget +1 och ingen KÖP-STÖD-räkning" in p["text"]
    p = prop(F, F, T)
    assert p["case"] == "för få episoder" and "H1 ej godkänd, H2 ej godkänd, H3 för få episoder" in p["text"]
    assert "snodden behålls som den är, med en varning" in p["text"]
    assert prop(T, T, T)["case"] == "för få episoder"


# ── Det beskrivande: korsvis och gruvorna ────────────────────────────────────
def test_cross_metal_and_miner_tables(frames, result):
    pl, pa = result["metals"]["platina"], result["metals"]["palladium"]
    assert (pl["cross"]["etf"], pa["cross"]["etf"]) == ("PALL", "PPLT")
    fr = frames["gold_platinum"]
    cross = pg.outcomes_on(fr.index, SERIES["PALL"], SERIES["GLD"])
    sbsw = pg.outcomes_on(fr.index, SERIES["SBSW"], SERIES["GLD"])
    first_sbsw = SERIES["SBSW"].index[0]
    before_sbsw = 0
    for sig, _ in pg.SIGNALS:
        eps = pl["signals"][sig]["episodes"]
        pos = _starts(fr, eps)
        assert pl["cross"]["signals"][sig] == ms.summarize(cross["arrays"], pos)
        expect = [v for v in sbsw["arrays"]["rel_252"][pos] if np.isfinite(v)]
        assert pl["miners"]["SBSW"]["signals"][sig]["n_252"] == len(expect)
        for e, p in zip(eps, pos):
            assert e["cross_rel_252"] == ms._f(cross["arrays"]["rel_252"][p])
            assert set(e["miners_rel_252"]) == {"IMPUY", "SBSW"}
            if pd.Timestamp(e["start"]) < first_sbsw:                       # SBSW finns från 2017
                before_sbsw += 1
                assert e["miners_rel_252"]["SBSW"] is None
    assert before_sbsw > 0
    assert pl["miners"]["SBSW"]["data"]["first"] == str(pg.outcome_frame(SERIES["SBSW"], SERIES["GLD"]).index[0].date())
    assert pl["miners"]["SBSW"]["data"]["first"] >= "2017-02-13" and pl["miners"]["IMPUY"]["baseline"]["n_252"] > 0
    # det beskrivande avgör ingenting: domarna är desamma med och utan korsvis och gruvor
    bare = pg.analyze(fr, SERIES["PPLT"], SERIES["GLD"], n_boot=50)
    full = pg.analyze(fr, SERIES["PPLT"], SERIES["GLD"], cross=("PALL", SERIES["PALL"]),
                      miners={m: SERIES[m] for m in pg.MINERS}, n_boot=50)
    assert bare["hypotheses"] == full["hypotheses"] and "cross" not in bare and bare["miners"] == {}
    gone = pg.analyze(fr, SERIES["PPLT"], SERIES["GLD"], miners={"IMPUY": pd.Series(dtype=float), "SBSW": None},
                      n_boot=50)
    assert gone["miners"] == {"IMPUY": None, "SBSW": None}
    assert all(e["miners_rel_252"] == {"IMPUY": None, "SBSW": None}
               for e in gone["signals"]["S1"]["episodes"])


# ── Hela körningen utan nätverk ──────────────────────────────────────────────
def test_report_runs_without_network_and_saves_one_json_blob():
    saved = {}
    res = pg.run(getter=lambda t: SERIES[t], save=lambda name, data: saved.setdefault(name, data) is not None)
    assert list(saved) == [pg.BLOB] and res["saved"] is True and "error" not in res
    json.dumps(saved[pg.BLOB], allow_nan=False)                         # ren JSON, inga NaN
    assert set(res["metals"]) == {"platina", "palladium"} and res["optional_missing"] == []
    for name, m in res["metals"].items():
        assert set(m["hypotheses"]) == {"H1", "H2", "H3"}
        assert all(isinstance(h["pass"], bool) and h["verdict"] in (pg.PASS, pg.FAIL, pg.TOO_FEW)
                   for h in m["hypotheses"].values())
        assert m["proposal"]["case"] in ("behåll", "trendfilter", "utan drivaren", "bara kontext", "för få episoder")
        assert m["signals"]["S0"]["bootstrap"]["n_boot"] == pg.N_BOOT and m["etf"] in ("PPLT", "PALL")
    assert res["proposals"] == [m["proposal"]["text"] for m in res["metals"].values()]
    assert res["data"]["PPLT"]["first"] == "2010-01-08" and res["data"]["SBSW"]["first"] == "2017-02-13"
    text = "\n".join(pg.report_lines(res))
    for w in ("PGM-SNODDARNA — spec version 1", "ändrar inga regler",
              "== Platina (GC=F/PL=F, panelens gold_platinum; utfall PPLT mot GLD) ==",
              "== Palladium (GC=F/PA=F, panelens gold_palladium; utfall PALL mot GLD) ==",
              "S0 nuvarande regel", "S1 bara kvoten", "S2 kvoten och trenden", "bas alla dagar", "bas var 63:e dag",
              "PPLT 252", "Episoder (start → köp · percentil kvot/GC/PL=F · över MA200", "MA200 ja", "MA200 nej",
              "Domar (252 dagar mot GLD; baslinje", "H1 S0 nuvarande regel:", "H2 S1 bara kvoten:",
              "H3 S2 kvoten och trenden:", "Bootstrap, beskrivande (95 %, cirkulära block om 63 dagar, 5 000 "
              "dragningar, frö 20261104)", "mot baslinjen [", "Korsvis, beskrivande (PALL mot GLD på platinans "
              "episoder)", "Korsvis, beskrivande (PPLT mot GLD på palladiums episoder)",
              "Gruvorna, beskrivande", "SBSW: kurser 2017-02-13", "Förslag enligt specen", "Förslag: GC=F/PL=F"):
        assert w in text, w


def test_missing_miners_are_reported_but_missing_required_prices_stop_the_study(monkeypatch):
    no_miners = {t: s for t, s in SERIES.items() if t != "SBSW"}
    no_miners["IMPUY"] = pd.Series(dtype=float)
    res = pg.run(getter=lambda t: no_miners.get(t), no_save=True, n_boot=50)
    assert "missing" not in res and res["optional_missing"] == ["IMPUY", "SBSW"]
    assert all(m["miners"] == {"IMPUY": None, "SBSW": None} for m in res["metals"].values())
    text = "\n".join(pg.report_lines(res))
    assert "Saknas (krävs inte, bara beskrivande): IMPUY, SBSW" in text and "IMPUY: saknas" in text
    calls = []
    short = dict(SERIES, PALL=pd.Series(dtype=float))
    res = pg.run(getter=lambda t: short[t], save=lambda n, d: calls.append(n) or True)
    assert res["missing"] == ["PALL"] and "PALL" in res["error"] and calls == [] and "metals" not in res
    assert any("FEL:" in line for line in pg.report_lines(res))
    monkeypatch.setattr(cr, "_download_close", lambda t, period="10y": short[t])
    assert pg.main(["--no-save"]) == 1
    monkeypatch.setattr(cr, "_download_close", lambda t, period="10y": SERIES[t] if period == "max" else None)
    assert pg.main(["--no-save"]) == 0                   # 0 även när hypoteserna faller


def test_workflow_is_manual_and_runs_the_study():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load(open(os.path.join(ROOT, ".github", "workflows", "pgm-band-study.yml"), encoding="utf-8"))
    on = wf[True] if True in wf else wf["on"]
    assert list(on) == ["workflow_dispatch"]
    steps = wf["jobs"]["pgm-band"]["steps"]
    assert steps[-1]["run"] == "python pgm_band_study.py"
    assert steps[-1]["env"] == {"BORSDATA_API_KEY": "${{ secrets.BORSDATA_API_KEY }}",
                                "GITHUB_TOKEN": "${{ secrets.GIST_TOKEN }}"}
    twin = yaml.safe_load(open(os.path.join(ROOT, ".github", "workflows", "miner-margin-study.yml"),
                               encoding="utf-8"))
    mine = [{k: v for k, v in s.items() if k != "name"} for s in steps[:-1]]
    theirs = [{k: v for k, v in s.items() if k != "name"} for s in twin["jobs"]["miner-margin"]["steps"][:-1]]
    assert mine == theirs                                # samma steg som gruvornas studie
