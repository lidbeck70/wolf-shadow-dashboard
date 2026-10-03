"""
🐺 Fiat Debasement PR 2 — beräkningarna (engine), laddningen med reserver och
guld/silver-skarven (data) samt nyckeltalen per valuta (snapshot).
Syntetiska serier — inget nätverk.
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiat_debasement import config as cfg  # noqa: E402
from fiat_debasement import data as fd  # noqa: E402
from fiat_debasement import engine as fe  # noqa: E402
from fiat_debasement import snapshot as fs  # noqa: E402
from fiat_debasement import sources as src  # noqa: E402

TODAY = pd.Timestamp("2026-10-03")


def _monthly(start, n, growth_pct_per_year, base=100.0):
    idx = pd.date_range(start, periods=n, freq="MS")
    g = (1 + growth_pct_per_year / 100) ** (1 / 12)
    return pd.Series(base * g ** np.arange(n), index=idx)


def _quarterly(start, n, growth_pct_per_year, base=1000.0):
    idx = pd.date_range(start, periods=n, freq="QS")
    g = (1 + growth_pct_per_year / 100) ** (1 / 4)
    return pd.Series(base * g ** np.arange(n), index=idx)


def _daily(start, end, growth_pct_per_year, base=100.0):
    idx = pd.bdate_range(start, end)
    yrs = (idx - idx[0]).days / 365.25
    return pd.Series(base * (1 + growth_pct_per_year / 100) ** yrs, index=idx)


# ── engine ──────────────────────────────────────────────────────────────────
def test_m2_yoy_and_cagr():
    m2 = _monthly("2010-01-01", 200, 6.0)
    y = fe.yoy(m2)
    assert y.index[0] == pd.Timestamp("2011-01-01")                          # första året saknar jämförelse
    assert y.iloc[-1] == pytest.approx(6.0, abs=1e-9)
    assert fe.cagr(m2, 5) == pytest.approx(6.0, abs=1e-9)
    assert fe.cagr(m2, 10) == pytest.approx(6.0, abs=1e-9)
    assert fe.cagr(m2, 50) is None                                            # för kort historik
    assert fe.cagr_since(m2, "2015-01-01") == pytest.approx(6.0, abs=0.01)


def test_yoy_formula_exact():
    s = pd.Series([100.0, 125.0], index=pd.to_datetime(["2024-03-01", "2025-03-01"]))
    assert fe.yoy(s).iloc[-1] == pytest.approx(25.0)                          # (125/100 − 1) × 100
    assert fe.cagr(pd.Series([100.0, 200.0], index=pd.to_datetime(["2015-01-01", "2025-01-01"])), 10) == \
        pytest.approx((2 ** 0.1 - 1) * 100)


def test_date_gaps_never_shift_the_comparison():
    m2 = _monthly("2015-01-01", 120, 5.0).drop(pd.Timestamp("2023-08-01"))   # en månad saknas
    y = fe.yoy(m2)
    assert pd.Timestamp("2024-08-01") not in y.index                         # ingen gissad jämförelse
    assert pd.Timestamp("2023-08-01") not in y.index
    assert y.loc["2024-09-01"] == pytest.approx(5.0, abs=1e-9)


def test_missing_and_negative_values():
    assert fe.yoy(None) is None and fe.cagr(None, 5) is None and fe.purchasing_power(pd.Series(dtype=float)) is None
    neg = pd.Series([100.0, -5.0, 110.0], index=pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]))
    assert fe.yoy(neg) is None                                                # negativa värden ger ingen YoY
    assert fe.cagr(pd.Series([-1.0, 2.0], index=pd.to_datetime(["2015-01-01", "2020-01-01"])), 5) is None
    nan = pd.Series([100.0, np.nan, np.inf], index=pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]))
    assert fe.latest(nan) == (pd.Timestamp("2020-01-01"), 100.0)
    assert fe.gap(None, 2.0) is None and fe.gap(float("nan"), 2.0) is None


def test_purchasing_power_index():
    cpi = pd.Series([100.0, 110.0, 125.0], index=pd.to_datetime(["2000-01-01", "2010-01-01", "2020-01-01"]))
    pp = fe.purchasing_power(cpi, "2000-01-01")
    assert pp.iloc[0] == 100 and pp.iloc[-1] == pytest.approx(80.0)          # 100 / 1,25
    assert fe.cumulative_change(cpi, "2000-01-01") == pytest.approx(25.0)
    pp10 = fe.purchasing_power(cpi, "2005-06-01")                              # första observation efter start
    assert pp10.index[0] == pd.Timestamp("2010-01-01") and pp10.iloc[-1] == pytest.approx(88.0)
    assert fe.purchasing_power(cpi, "2030-01-01") is None


def test_monetary_gap():
    assert fe.gap(6.0, 2.0) == pytest.approx(4.0)                             # specens exempel
    m2 = _monthly("2010-01-01", 200, 6.0)
    gdp = _quarterly("2010-01-01", 66, 2.0)
    gs = fe.gap_series(m2, gdp)
    assert gs.iloc[-1] == pytest.approx(4.0, abs=0.01)                        # olika frekvens → kvartal


def test_currency_conversion():
    usd = pd.Series([2000.0, 2100.0, 2200.0], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    sekusd = pd.Series([10.0, 10.5], index=pd.to_datetime(["2024-01-02", "2024-01-04"]))   # 3 jan saknas
    sek = fe.price_in(usd, "SEK", sekusd)
    assert list(sek) == [20000.0, 23100.0]                                    # ingen ifylld växelkurs
    eurusd = pd.Series([1.10, 1.05, 1.0], index=usd.index)
    assert fe.price_in(usd, "EUR", eurusd).iloc[-1] == pytest.approx(2200.0)
    assert fe.price_in(usd, "USD") is usd or fe.price_in(usd, "USD").equals(usd)
    assert fe.price_in(usd, "SEK", None) is None
    with pytest.raises(ValueError):
        fe.price_in(usd, "GBP", sekusd)


def test_gold_and_silver_normalization():
    gold = _daily("2000-01-03", "2026-10-02", 9.0, base=280.0)
    vs = fe.fiat_vs_asset(gold, "2000-01-03")
    assert vs.iloc[0] == pytest.approx(100.0)
    years = (vs.index[-1] - vs.index[0]).days / 365.25
    assert vs.iloc[-1] == pytest.approx(100 / 1.09 ** years, rel=1e-6)        # valutan köper mindre guld
    units = fe.units_of_100(gold, "2000-01-03")
    assert units.iloc[0] == pytest.approx(100.0) and units.iloc[-1] == pytest.approx(100 * 1.09 ** years, rel=1e-6)
    silver = _daily("2000-01-03", "2026-10-02", 6.0, base=5.0)
    assert fe.normalize_100(silver, "2010-01-01").iloc[0] == pytest.approx(100.0)
    assert fe.change_pct(fe.fiat_vs_asset(gold), 1) == pytest.approx((1 / 1.09 - 1) * 100, abs=0.05)
    assert fe.change_pct(gold, 40) is None                                     # före seriens start


def test_real_price_and_percentile():
    price = _monthly("2010-01-01", 120, 5.0, base=100.0)
    cpi = _monthly("2010-01-01", 120, 5.0, base=100.0)
    real = fe.real_price(price, cpi, "2010-01-01")
    assert real.iloc[-1] == pytest.approx(100.0)                               # stiger bara med inflationen
    s = pd.Series(range(1, 101), index=pd.date_range("2000-01-01", periods=100, freq="MS"), dtype=float)
    assert fe.percentile(s) == pytest.approx(99.5)
    assert fe.percentile(s, value=50.5) == pytest.approx(50.0)
    assert fe.percentile(s.iloc[:1]) is None


# ── data: reserver, cache, skarv ────────────────────────────────────────────
def _sd(values, label="x", source="FRED", sid="S", err=None):
    if err:
        return src.SeriesData(source, sid, label=label, error=err)
    sd = src.SeriesData(source, sid, values=values, label=label)
    sd.frequency = src.frequency_of(values.index)
    return sd


def test_load_uses_the_first_working_source(monkeypatch):
    fd.clear_cache()
    calls = []
    monkeypatch.setitem(cfg.SERIES, ("cpi", "TST"), [{"kind": "scb", "id": "A"}, {"kind": "fred", "id": "B"}])

    def fetcher(spec):
        calls.append(spec["id"])
        return _sd(None, err="HTTP 404") if spec["id"] == "A" else _sd(_monthly("2020-01-01", 80, 2.0), "B")
    ld = fd.load("cpi", "TST", fetcher=fetcher, today=TODAY)
    assert ld.ok and ld.data.label == "B" and [a.series_id for a in ld.attempts] == ["S", "S"]
    assert calls == ["A", "B"] and not ld.stale
    fd.load("cpi", "TST", fetcher=fetcher, today=TODAY)
    assert calls == ["A", "B"]                                                # båda cachade
    fd.clear_cache()


def test_api_failure_gives_data_unavailable_and_stale_is_flagged(monkeypatch):
    fd.clear_cache()
    monkeypatch.setitem(cfg.SERIES, ("m2", "TST"), [{"kind": "fred", "id": "X"}])
    ld = fd.load("m2", "TST", fetcher=lambda s: _sd(None, err="timeout"), today=TODAY)
    assert not ld.ok and ld.values is None and ld.attempts[0].error == "timeout"
    fd.clear_cache()
    ld = fd.load("m2", "TST", fetcher=lambda s: _sd(_monthly("2000-01-01", 200, 2.0)), today=TODAY)
    assert ld.ok and ld.stale                                                 # slutar 2016 → inaktuell
    assert not fd.load("m2", "NONE", fetcher=lambda s: None).ok
    fd.clear_cache()


def test_gold_splice_marks_futures_before_spot():
    gold = _daily("2000-08-30", "2026-10-02", 9.0, base=273.0)
    spot, fut = gold[gold.index >= "2006-10-03"], gold * 1.005               # termin 0,5 % över spot
    values, segs = fd.splice(_sd(spot, "Guld spot (Börsdata)"), _sd(fut, "Guld terminspris (Yahoo GC=F)"))
    assert values.index[0] == fut.index[0] and values.index[-1] == spot.index[-1]
    assert values.loc["2010-01-04"] == spot.loc["2010-01-04"]                  # spot efter 2006
    assert values.loc["2003-01-02"] == fut.loc["2003-01-02"]                   # terminspris före
    assert [s["kind"] for s in segs] == [cfg.FUTURES, cfg.SPOT]
    assert segs[0]["to"] < segs[1]["from"] == "2006-10-03"
    assert fd.splice_gap_pct(_sd(spot), _sd(fut)) == pytest.approx(0.5, abs=0.01)
    only_fut, segs = fd.splice(_sd(None, err="ingen nyckel"), _sd(fut, "GC=F"))
    assert only_fut.equals(fut) and segs[0]["kind"] == cfg.FUTURES             # allt märkt terminspris
    assert fd.splice(None, None) == (None, [])


def test_load_asset_gold(monkeypatch):
    fd.clear_cache()
    spot = _daily("2006-10-03", "2026-10-02", 9.0, base=600.0)
    fut = _daily("2000-08-30", "2026-10-02", 9.0, base=273.0)

    def fetcher(spec):
        return _sd(spot, spec["label"]) if spec["kind"] == "borsdata" else _sd(fut, spec["label"])
    ld = fd.load_asset(cfg.GOLD, fetcher=fetcher, today=TODAY)
    assert ld.ok and ld.values.index[0] == fut.index[0] and len(ld.segments) == 2
    assert "terminspris" in ld.note and ld.data.unit == "USD/oz"
    fd.clear_cache()


# ── snapshot ────────────────────────────────────────────────────────────────
def _fake_world():
    series = {
        cfg.M2: _monthly("2000-01-01", 320, 6.0), cfg.CPI: _monthly("2000-01-01", 320, 2.5),
        cfg.CORE: _monthly("2000-01-01", 320, 2.0), cfg.GDP: _quarterly("2000-01-01", 106, 2.0),
        cfg.DEBT: pd.Series([60.0, 62.5], index=pd.to_datetime(["2025-10-01", "2026-01-01"])),
        cfg.FX: _daily("2000-01-03", "2026-10-02", 1.0, base=9.0),
    }

    def loader(concept, cur):
        ld = fd.Loaded(concept, cur)
        if concept in series:
            ld.data = _sd(series[concept], f"{concept} {cur}")
        return ld

    gold = _daily("2000-01-03", "2026-10-02", 9.0, base=280.0)

    def asset_loader(name):
        ld = fd.Loaded(name, "USD")
        ld.data = _sd(gold, "Guld")
        return ld
    return loader, asset_loader


def test_snapshot_keeps_the_concepts_apart():
    loader, asset_loader = _fake_world()
    snap = fs.snapshot("SEK", loader=loader, asset_loader=asset_loader, today=TODAY)
    g = snap.get
    assert g("m2_yoy").value == pytest.approx(6.0, abs=0.01) and "inte konsumentinflation" in g("m2_yoy").note
    assert g("cpi_yoy").value == pytest.approx(2.5, abs=0.01) and g("core_yoy").value == pytest.approx(2.0, abs=0.01)
    assert g("gdp_yoy").value == pytest.approx(2.0, abs=0.01)
    assert g("monetary_gap").value == pytest.approx(4.0, abs=0.05) and g("monetary_gap").unit == "pe"
    assert g("monetary_gap5").value == pytest.approx(4.0, abs=0.05)
    assert g("debt_gdp").value == 62.5 and g("debt_gdp").as_of == "2026-01-01"
    assert g("m2_cagr10").value == pytest.approx(6.0, abs=0.01)
    assert g("gold_1y").value < 0 and "×" in g("gold_1y").source              # guld × växelkurs
    assert g("cpi_yoy").source.startswith("cpi SEK [FRED S]")


def test_snapshot_missing_data_is_unavailable_not_zero():
    def loader(concept, cur):
        return fd.Loaded(concept, cur)

    def asset_loader(name):
        return fd.Loaded(name, "USD")
    snap = fs.snapshot("EUR", loader=loader, asset_loader=asset_loader, today=TODAY)
    for key in ("m2_yoy", "cpi_yoy", "gdp_yoy", "monetary_gap", "debt_gdp", "gold_5y"):
        assert snap.get(key).value is None and not snap.get(key).ok, key
    assert snap.get("nope").note == "inte beräknad"


def test_config_covers_every_currency():
    for cur in cfg.CURRENCIES:
        for concept in (cfg.M2, cfg.CPI, cfg.CORE, cfg.GDP, cfg.DEBT):
            assert cfg.SERIES.get((concept, cur)), (concept, cur)
    assert cfg.SERIES[(cfg.FX, "SEK")][0]["kind"] == "riksbank"
    assert cfg.ASSET_SPLICE[cfg.GOLD]["primary"]["kind"] == "borsdata"
    assert cfg.ASSET_SPLICE[cfg.GOLD]["backfill"]["id"] == "GC=F"
    assert all(cfg.SERIES[(cfg.M2, c)][0].get("definition") for c in cfg.CURRENCIES)
