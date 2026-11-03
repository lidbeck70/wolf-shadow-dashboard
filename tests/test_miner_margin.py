"""
Gruvorna mot marginalen — GDX och GDXJ mot guldpris minus branschens AISC (kontext).

M = gruv-ETF:ns kurs / (guldpris − AISC). Låg percentil = gruvorna billiga per dollar marginal.
Mätaren är BARA kontext: den får aldrig nå _compute_ratio_summary ('Gummisnoddar: …'), motorns +1
eller någon annan signal. Inget nätverk — fejkade kursserier och fejkade AISC-rader.
"""
import inspect
import math
import os
import re

import numpy as np
import pandas as pd
import pytest

from alpha_regime import commodity_ratios as cr
from alpha_regime import miner_margin as mm
from alpha_regime import ui as aui

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = " "
WGC_MF = "World Gold Council/Metals Focus"
GOLD_MINER_KEYS = ["metal_miners", "gdxj_gdx", "gold_gdxj", "gdx_spy"]


@pytest.fixture(autouse=True)
def _fresh_cache():
    """compute() läser en 6 h-cache — testerna får aldrig se varandras fejkade serier."""
    clear = getattr(mm._compute_cached, "clear", lambda: None)
    clear()
    yield
    clear()


def _row(period, aisc, published, series="MF_Q"):
    return mm.AiscRow(period, aisc, published, series, "Testkälla", "testkälla", "https://example.org/aisc")


def _at(dates, pit=False, rows=None):
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    return [None if math.isnan(v) else v for v in mm.aisc_series(idx, point_in_time=pit, rows=rows).values]


# ── Tabellen ─────────────────────────────────────────────────────────────────
def test_table_rows_are_auditable_and_never_published_before_their_period_ends():
    seen = set()
    for row in mm.AISC_TABLE:
        y, q = mm._parse_period(row.period)
        assert row.period not in seen, f"dubbel period {row.period}"
        seen.add(row.period)
        assert row.aisc > 0 and row.source and row.url.startswith("http") and row.publisher
        assert row.series in ("MF_Q", "MF_A", "MF_A_DERIVED", "WGC_GDT_EARLY", "SP_A", "GFMS_A")
        pub = pd.Timestamp(row.published)
        period_end = pd.Period(f"{y}Q{q or 4}", "Q").end_time
        assert pub > period_end, f"{row.period} publicerad {row.published} innan perioden tog slut"


def test_docstring_example_is_a_template_not_data():
    """Exemplet i instruktionen får inte se ut som en riktig rad som ägaren kan kopiera."""
    doc = mm.__doc__
    assert 'AiscRow("YYYYQn", <USD/uns>, "YYYY-MM-DD", "MF_Q"' in doc
    assert not re.search(r'AiscRow\("\d{4}', doc)
    assert "1840" not in doc


def test_table_holds_the_researched_first_prints():
    by = {r.period: r for r in mm.AISC_TABLE}
    assert len(mm.AISC_TABLE) == 35
    assert (by["2012"].aisc, by["2012"].series) == (1116, "MF_A_DERIVED")
    assert (by["2019"].aisc, by["2019"].published) == (941, "2020-06-24")
    assert (by["2020Q2"].aisc, by["2020Q2"].published) == (972, "2022-03-31")
    assert (by["2024Q1"].aisc, by["2025"].aisc) == (1439, 1552)
    assert (by["2026Q1"].aisc, by["2026Q1"].published, by["2026Q1"].publisher) == (1785, "2026-08-25", WGC_MF)


# ── aisc_series ──────────────────────────────────────────────────────────────
def test_quarter_value_applies_to_its_quarter_and_is_forward_filled():
    rows = [_row("2020Q2", 100, "2020-09-30"), _row("2020Q3", 110, "2020-12-31")]
    assert _at(["2020-03-31", "2020-04-01", "2020-06-30", "2020-07-01", "2020-09-30"], rows=rows) == [
        None, 100, 100, 110, 110]
    # efter tabellens slut: senaste kända värdet ligger kvar
    assert _at(["2020-10-01", "2021-06-01", "2026-11-03"], rows=rows) == [110, 110, 110]


def test_annual_only_year_applies_to_all_four_quarters():
    rows = [_row("2019", 90, "2020-06-24", "MF_A"), _row("2020Q2", 100, "2020-09-30")]
    assert _at(["2018-12-31", "2019-01-02", "2019-05-15", "2019-08-15", "2019-12-31"], rows=rows) == [
        None, 90, 90, 90, 90]
    assert _at(["2020-02-03", "2020-04-01"], rows=rows) == [90, 100]      # lucka 2020Q1 → föregående värde


def test_quarter_rows_win_over_the_full_year_in_any_order():
    rows = [_row("2021", 80, "2022-03-31", "MF_A"), _row("2021Q3", 120, "2021-12-31")]
    dates = ["2021-02-01", "2021-05-03", "2021-08-02", "2021-11-01"]
    assert _at(dates, rows=rows) == [80, 80, 120, 80]
    assert _at(dates, rows=list(reversed(rows))) == [80, 80, 120, 80]


def test_real_table_daily_values():
    dates = ["2011-12-30", "2012-06-01", "2014-06-02", "2015-03-02", "2015-08-03", "2015-11-02",
             "2016-05-02", "2017-08-01", "2018-08-01", "2020-02-03", "2020-05-01"]
    assert _at(dates) == [None, 1116, 1113, 1113, 900, 832, 879, 879, 897, 941, 972]
    # år med alla fyra kvartal: kvartalen, aldrig helåret (1 068 / 1 399 / 1 552)
    assert _at(["2021-02-01", "2021-05-03", "2021-08-02", "2021-11-01"]) == [1048, 1067, 1123, 1129]
    assert _at(["2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01"]) == [1439, 1388, 1456, 1438]
    assert _at(["2025-02-03", "2025-05-01", "2025-08-01", "2025-11-03"]) == [1536, 1578, 1605, 1706]
    assert _at(["2026-02-02", "2026-06-01", "2026-11-03"]) == [1785, 1785, 1785]


def test_point_in_time_uses_a_value_only_from_its_published_date():
    rows = [_row("2020Q1", 100, "2020-05-15"), _row("2020Q2", 120, "2020-08-15")]
    dates = ["2020-02-03", "2020-05-14", "2020-05-15", "2020-08-14", "2020-08-15", "2021-03-01"]
    assert _at(dates, pit=True, rows=rows) == [None, None, 100, 100, 120, 120]
    assert _at(dates, rows=rows) == [100, 120, 120, 120, 120, 120]        # utan PIT: kvartalets eget värde


def test_point_in_time_on_the_real_table():
    assert _at(["2026-08-24", "2026-08-25"], pit=True) == [1706, 1785]
    assert _at(["2021-01-15"], pit=True) == [953]       # 2020Q2:s 972 publicerades först 2022-03-31
    assert _at(["2015-11-11", "2015-11-12"], pit=True) == [None, 900]
    assert _at(["2012-06-01", "2023-05-02"], pit=True) == [None, 1279]   # 2012 (härlett) kom först 2023
    idx = pd.bdate_range("2015-01-01", "2026-11-03")
    applied = mm._applied_rows(idx, point_in_time=True)
    for d, row in zip(idx, applied):                    # ser aldrig in i framtiden
        assert row is None or pd.Timestamp(row.published) <= d, (d, row)


def test_point_in_time_equals_the_table_as_it_looked_that_day():
    rng = np.random.default_rng(3)
    for d in pd.to_datetime(rng.choice(pd.bdate_range("2015-06-01", "2026-11-03"), 60)):
        known = [r for r in mm.AISC_TABLE if pd.Timestamp(r.published) <= d]
        assert _at([d], pit=True) == _at([d], rows=known), d


# ── margin_ratio ─────────────────────────────────────────────────────────────
def test_margin_at_or_below_50_usd_is_masked():
    idx = pd.bdate_range("2026-01-05", periods=7)
    gold = pd.Series([2000.0, 1000.0, 1050.0, 1050.01, 900.0, 1500.0, 1500.0], index=idx)
    miner = pd.Series([20.0, 10.0, 10.0, 10.0, 10.0, 0.0, np.inf], index=idx)
    m = mm.margin_ratio(miner, gold, 1000.0)
    assert m.iloc[0] == pytest.approx(0.02)
    assert m.iloc[3] == pytest.approx(10.0 / 50.01)
    assert m.iloc[[1, 2, 4, 5, 6]].isna().all()          # marginal 0, 50, −100; trasig kurs 0 och inf
    aisc = pd.Series(1000.0, index=idx)
    pd.testing.assert_series_equal(mm.margin_ratio(miner, gold, aisc), m)


def test_non_finite_gold_or_aisc_is_masked_never_m_zero():
    """Guld = inf gav marginal inf > 50 och M = 0,0 som räknades som historik."""
    idx = pd.bdate_range("2026-01-05", periods=5)
    gold = pd.Series([2000.0, np.inf, np.nan, 2000.0, 2000.0], index=idx)
    aisc = pd.Series([1000.0, 1000.0, 1000.0, -np.inf, np.nan], index=idx)
    m = mm.margin_ratio(pd.Series(20.0, index=idx), gold, aisc)
    assert m.iloc[0] == pytest.approx(0.02)
    assert m.iloc[1:].isna().all()
    assert not (m == 0.0).any()


# ── build_result: percentil, median, fönster ─────────────────────────────────
_FLAT = [_row("2000", 1000, "2000-06-30", "MF_A")]       # AISC 1 000 för alla datum från 2000


def _series(values, end="2026-10-30"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.bdate_range(end=end, periods=len(values)))


def test_percentile_and_median_over_the_whole_window():
    n, k = 800, 176
    rng = np.random.default_rng(5)
    others = np.sort(rng.uniform(0.02, 0.08, n - 1))
    cur = (others[k - 2] + others[k - 1]) / 2           # exakt k värden ≤ cur → råpercentil k/n
    m_vals = np.concatenate([rng.permutation(others), [cur]])
    gold = _series(rng.uniform(1500, 2500, n))
    miner = gold.copy()
    miner[:] = m_vals * (gold.values - 1000.0)
    res = mm.build_result("GDX", miner, gold, today="2026-11-03", rows=_FLAT)
    assert res.status == "OK" and res.error is None
    assert res.percentile == round(k / n * 100, 1) == 22.0
    assert res.median == pytest.approx(float(np.median(m_vals)), rel=1e-5)
    assert res.current == pytest.approx(cur, rel=1e-5)
    assert res.history_start == str(gold.index[0].date())
    assert res.history_years == round((gold.index[-1] - gold.index[0]).days / 365.25, 1)
    assert (res.aisc, res.aisc_period, res.aisc_source) == (1000.0, "2000", "Testkälla")
    assert res.margin == pytest.approx(gold.iloc[-1] - 1000.0, abs=0.01)
    assert res.as_of == "2026-10-30"
    weekly = (miner / (gold - 1000.0)).resample("W").last()
    assert res.sparkline_dates == [str(d.date()) for d in weekly.index]
    assert res.sparkline_values[-1] == pytest.approx(cur, rel=1e-5)


def test_masked_days_are_left_out_of_percentile_and_median():
    n = 300
    gold = _series(np.full(n, 2000.0))
    gold.iloc[10:20] = 1040.0                          # marginal 40 → maskerad
    miner = _series(np.linspace(10.0, 40.0, n))
    res = mm.build_result("GDX", miner, gold, today="2026-11-03", rows=_FLAT)
    valid = (miner / (gold - 1000.0)).drop(gold.index[10:20])
    assert res.median == pytest.approx(float(np.median(valid.values)), rel=1e-5)
    assert res.percentile == round(cr._percentile_of(valid.values, valid.iloc[-1]), 1) == 100.0


def test_real_table_result_names_the_aisc_used():
    idx = pd.bdate_range(end="2026-10-30", periods=2600)
    gold = pd.Series(np.linspace(1250.0, 4010.0, len(idx)), index=idx)
    gdx = pd.Series(np.linspace(22.0, 75.0, len(idx)), index=idx)
    res = mm.build_result("GDX", gdx, gold, today="2026-11-03")
    assert res.status == "OK"
    assert (res.aisc, res.aisc_period, res.aisc_source) == (1785.0, "2026Q1", WGC_MF)
    assert res.margin == pytest.approx(4010.0 - 1785.0)
    assert res.current == pytest.approx(75.0 / 2225.0, rel=1e-5)


# ── Inaktuell tabell ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("newest,today,stale", [
    ("2026Q2", "2026-11-03", False), ("2026Q1", "2026-11-03", True), ("2026Q1", "2026-09-30", False),
    ("2025", "2026-06-30", False), ("2025", "2026-07-01", True), ("", "2026-11-03", True),
])
def test_staleness_is_more_than_two_quarters(newest, today, stale):
    assert mm.is_stale(newest, today) is stale


def test_newest_period_and_the_stale_flag_on_the_result():
    assert mm.newest_period() == "2026Q1"
    assert mm.newest_period([_row("2025", 1, "2026-06-05", "MF_A"), _row("2025Q4", 1, "2026-04-29")]) == "2025Q4"
    assert mm.newest_period([_row("2025Q3", 1, "2026-01-29"), _row("2025", 1, "2026-06-05", "MF_A")]) == "2025"
    gold, gdx = _series(np.full(300, 3000.0)), _series(np.full(300, 50.0))
    assert mm.build_result("GDX", gdx, gold, today="2026-11-03").stale is True
    fresh = mm.build_result("GDX", gdx, gold, today="2026-09-30")
    assert fresh.stale is False and fresh.newest_period == "2026Q1"
    # flaggan gäller tabellen — den sätts även när kursen saknas
    assert mm.build_result("GDX", gdx, None, today="2026-11-03").stale is True


# ── DATA_GAP ─────────────────────────────────────────────────────────────────
def test_data_gap_when_gold_is_missing():
    gdx = _series(np.full(300, 50.0))
    for gold in (None, pd.Series(dtype=float)):
        res = mm.build_result("GDX", gdx, gold, today="2026-11-03")
        assert res.status == "DATA_GAP" and res.current is None
        assert res.error.startswith("Guldpriset (GC=F) saknas")


def test_compute_gold_missing_is_data_gap_without_fallback(monkeypatch):
    asked = []

    def fake(ticker, period="10y"):
        asked.append(ticker)
        return pd.Series(dtype=float) if ticker == "GC=F" else _series(np.full(300, 50.0))
    monkeypatch.setattr(mm, "_download_close", fake)
    res = mm.compute("GDX")
    assert res.status == "DATA_GAP" and "Guldpriset (GC=F) saknas" in res.error
    assert asked == ["GC=F"]                            # ingen reserv, ingen onödig gruvhämtning


def test_other_data_gaps():
    gold = _series(np.full(300, 3000.0))
    assert mm.build_result("GDXJ", None, gold).error == "Kursen för GDXJ saknas."
    short = mm.build_result("GDX", _series(np.full(50, 50.0)), gold)
    assert short.status == "DATA_GAP" and short.error.startswith("För kort historik")
    thin = gold.copy()
    thin.iloc[-1] = 1800.0                              # 1 800 − 1 785 = 15 USD/uns
    res = mm.build_result("GDX", _series(np.full(300, 50.0)), thin)
    assert res.status == "DATA_GAP" and res.error.startswith("Marginalen är bara 15 USD/uns")


def test_broken_last_bar_names_the_right_leg_without_raw_floats():
    gold, gdx = _series(np.full(300, 3000.0)), _series(np.full(300, 50.0))
    broken_gold = gold.copy()
    broken_gold.iloc[-1] = np.inf
    res = mm.build_result("GDX", gdx, broken_gold)
    assert res.status == "DATA_GAP" and res.error == "Sista stängningen för guldet (GC=F) är trasig."
    for bad in (0.0, -1.0, np.inf):
        broken = gdx.copy()
        broken.iloc[-1] = bad
        res = mm.build_result("GDX", broken, gold)
        assert res.status == "DATA_GAP" and res.error == "Sista stängningen för GDX är trasig.", bad
    # tidigare 'trasig (0.0)': aldrig en rå Python-float med decimalpunkt i texten
    neg = gold.copy()
    neg.iloc[-1] = 500.0                                 # 500 − 1 785 = −1 285 USD/uns
    res = mm.build_result("GDX", gdx, neg)
    assert res.error == f"Marginalen är bara −1{NB}285 USD/uns (≤ 50) — måttet säger inget då."


def test_inf_gold_bar_inside_the_window_is_not_history():
    gold, gdx = _series(np.full(300, 3000.0)), _series(np.linspace(40.0, 60.0, 300))
    gold.iloc[100] = np.inf
    res = mm.build_result("GDX", gdx, gold, rows=_FLAT)
    valid = (gdx / (gold - 1000.0)).drop(gold.index[100])
    assert res.status == "OK"
    assert res.median == pytest.approx(float(np.median(valid.values)), rel=1e-5)
    assert min(res.sparkline_values) > 0


# ── compute(): ett tillfälligt fel cachas inte ───────────────────────────────
def _flaky(fail: dict, calls: list):
    """Fejkad nedladdning: tickern i `fail` ger tom serie så många gånger som anges, sedan data."""
    def fake(ticker, period="10y"):
        calls.append(ticker)
        if fail.get(ticker, 0) > 0:
            fail[ticker] -= 1
            return pd.Series(dtype=float)
        return _series(np.full(300, 3000.0 if ticker == "GC=F" else 50.0))
    return fake


@pytest.mark.parametrize("ticker,error", [("GC=F", "Guldpriset (GC=F) saknas"), ("GDX", "Kursen för GDX saknas.")])
def test_transient_download_failure_is_retried_on_the_next_render(monkeypatch, ticker, error):
    """Ett tomt svar (t.ex. Yahoos strypning) fick förut kortet att visa DATA_GAP i sex timmar."""
    calls = []
    monkeypatch.setattr(mm, "_download_close", _flaky({ticker: 1}, calls))
    first = mm.compute("GDX")
    assert first.status == "DATA_GAP" and error in first.error
    second = mm.compute("GDX")                          # ingen cacheclear emellan
    assert second.status == "OK" and second.error is None
    n = len(calls)
    assert mm.compute("GDX").status == "OK" and len(calls) == n     # ett lyckat resultat cachas


def test_compute_caches_a_deterministic_data_gap(monkeypatch):
    """Riktiga kurser som ger för kort historik är inget nätverksfel — det får ligga i cachen."""
    calls = []

    def fake(ticker, period="10y"):
        calls.append(ticker)
        return _series(np.full(50, 3000.0 if ticker == "GC=F" else 50.0))
    monkeypatch.setattr(mm, "_download_close", fake)
    assert mm.compute("GDX").error.startswith("För kort historik")
    assert mm.compute("GDX").error.startswith("För kort historik") and calls == ["GC=F", "GDX"]


# ── Kortets texter ───────────────────────────────────────────────────────────
def _margin(miner="GDX", pct=22.0, **kw):
    base = dict(miner=miner, status="OK", current=0.031, percentile=pct, median=0.045, miner_price=75.0,
                gold_price=4010.0, aisc=1620.0, aisc_period="2026Q2", aisc_source=WGC_MF, margin=2390.0,
                as_of="2026-10-30", history_start="2016-11-03", history_years=10.0, newest_period="2026Q2",
                sparkline_dates=["2026-10-25", "2026-11-01"], sparkline_values=[0.032, 0.031])
    base.update(kw)
    return mm.MarginResult(**base)


def test_value_and_margin_lines():
    assert aui._margin_value_text(_margin()) == (
        "GDX kostar 0,031 per dollar marginal per uns — 22:a percentilen sedan 2016 (median 0,045): "
        "billigt mot marginalen")
    assert aui._margin_line_text(_margin()) == (
        f"Guld 4{NB}010 − AISC 1{NB}620 (2026Q2, World Gold Council/Metals Focus) = 2{NB}390 USD/uns")
    # marginalen räknas på de avrundade talen — raden går alltid ihop
    assert aui._margin_line_text(_margin(gold_price=4010.6, aisc=1785.0, aisc_period="2026Q1")).endswith(
        f"4{NB}011 − AISC 1{NB}785 (2026Q1, World Gold Council/Metals Focus) = 2{NB}226 USD/uns")
    assert "sedan" not in aui._margin_value_text(_margin(history_start=""))


@pytest.mark.parametrize("pct,word", [(0.0, "billigt"), (22.0, "billigt"), (25.0, "billigt"), (25.4, "billigt"),
                                      (25.6, "normalt"), (50.0, "normalt"), (74.4, "normalt"), (74.6, "dyrt"),
                                      (100.0, "dyrt")])
def test_margin_words(pct, word):
    assert aui._margin_word(pct)[0] == f"{word} mot marginalen"


def test_word_follows_the_ordinal_on_the_card():
    for p in np.arange(0.0, 100.01, 0.1):
        shown = int(aui._sv_ord(p).split(":")[0])
        word = aui._margin_word(p)[0]
        assert (word == "billigt mot marginalen") == (shown <= 25), p
        assert (word == "dyrt mot marginalen") == (shown >= 75), p


@pytest.mark.parametrize("kw", [dict(current=None), dict(current=0.0), dict(current=float("nan")),
                                dict(current=float("inf")), dict(percentile=None), dict(percentile=float("nan"))])
def test_value_line_rejects_missing_values(kw):
    assert aui._margin_value_text(_margin(**kw)) is None


_ANNUAL_SPANS = [("2016", "S&P Global", "SP_A", "2016Q4", "2017Q4"),
                 ("2018", "Refinitiv GFMS", "GFMS_A", "2018Q1", "2018Q4"),
                 ("2019", "Metals Focus", "MF_A", "2019Q1", "2020Q1")]
_SOURCE_10Y = ("AISC är mest branschsnittet per kvartal (World Gold Council/Metals Focus), men en del av fönstret "
               "(sedan 2016-11-03) bygger på helårstal eller andra källor och metoder: S&P Global helår 2016 "
               "(gäller 2016Q4–2017Q4), Refinitiv GFMS helår 2018 och Metals Focus helår 2019 (gäller "
               "2019Q1–2020Q1) — där är M mindre jämförbart.")


def test_result_records_which_aisc_rows_the_window_uses():
    idx = pd.bdate_range("2016-11-03", "2026-10-30")
    gold = pd.Series(np.linspace(1250.0, 4010.0, len(idx)), index=idx)
    gdx = pd.Series(np.linspace(22.0, 75.0, len(idx)), index=idx)
    res = mm.build_result("GDX", gdx, gold, today="2026-11-03")
    assert res.aisc_spans[:3] == _ANNUAL_SPANS
    assert res.aisc_spans[3] == ("2020Q2", WGC_MF, "MF_Q", "2020Q2", "2020Q2")
    assert all(s[2] == "MF_Q" for s in res.aisc_spans[3:])
    assert res.aisc_spans[-1] == ("2026Q1", WGC_MF, "MF_Q", "2026Q1", "2026Q4")     # framåtfyllt
    assert aui._margin_source_text([res]) == _SOURCE_10Y


def test_source_text_never_claims_all_quarterly_when_it_is_not():
    """Förr: 'AISC är branschsnittet per kvartal (World Gold Council/Metals Focus)' även för 2016–2019."""
    spans = _ANNUAL_SPANS + [("2020Q2", WGC_MF, "MF_Q", "2020Q2", "2026Q4")]
    assert aui._margin_source_text([_margin(aisc_spans=spans, history_start="2016-11-03")]) == _SOURCE_10Y
    # bara huvudserien i fönstret → då stämmer den korta meningen
    only_q = [("2020Q2", WGC_MF, "MF_Q", "2020Q2", "2020Q2"), ("2020Q3", WGC_MF, "MF_Q", "2020Q3", "2026Q4")]
    assert aui._margin_source_text([_margin(aisc_spans=only_q)]) == (
        "AISC är branschsnittet per kvartal (World Gold Council/Metals Focus).")
    # inget resultat med data → en mening som gäller hela tabellen
    gap = mm.MarginResult(miner="GDX", error="x")
    assert aui._margin_source_text([gap]) == (
        "AISC är mest branschsnittet per kvartal (World Gold Council/Metals Focus); äldre år är helårstal, "
        "delvis från andra källor och metoder.")
    # det längsta fönstret avgör när GDX och GDXJ skiljer sig
    short = _margin("GDXJ", aisc_spans=spans[2:], history_start="2019-03-01")
    long_ = _margin("GDX", aisc_spans=spans, history_start="2016-11-03")
    assert aui._margin_source_text([short, long_]) == _SOURCE_10Y


@pytest.mark.parametrize("span,text", [
    (("2018", "Refinitiv GFMS", "GFMS_A", "2018Q1", "2018Q4"), "Refinitiv GFMS helår 2018"),
    (("2016", "S&P Global", "SP_A", "2017Q1", "2017Q4"), "S&P Global helår 2016 (gäller 2017Q1–2017Q4)"),
    (("2015Q4", "World Gold Council", "WGC_GDT_EARLY", "2015Q4", "2015Q4"), "World Gold Council 2015Q4"),
    (("2015Q4", "World Gold Council", "WGC_GDT_EARLY", "2016Q1", "2016Q1"),
     "World Gold Council 2015Q4 (gäller 2016Q1)"),
])
def test_span_text(span, text):
    assert aui._span_text(span) == text


def test_miners_follow_the_shown_ratio_keys():
    assert aui._margin_miners(GOLD_MINER_KEYS) == ["GDX", "GDXJ"]
    assert aui._margin_miners(cr.EXPOSURE_TO_RATIO["junior_miner"]) == ["GDXJ"]
    assert aui._margin_miners(["metal_miners"]) == ["GDX"]
    for keys in (cr.EXPOSURE_TO_RATIO["silver"], [], None, ["gdx_spy"]):
        assert aui._margin_miners(keys) == []


# ── Aldrig en signal ─────────────────────────────────────────────────────────
def _ratio(key, **kw):
    spec = next(s for s in cr._RATIO_SPECS if s.key == key)
    base = dict(key=key, label=spec.label, denominator_label=spec.denominator_label,
                cheap_direction=spec.cheap_direction, current=1.0, percentile=50.0, zscore=0.0, status="NEUTRAL",
                numerator_label=spec.primary_num, denominator_leg_label=spec.primary_den, numerator_pctile=50.0,
                denominator_pctile=50.0, driver="MIXED", history_start="2016-11-03", history_years=10.0, median=1.0)
    base.update(kw)
    return cr.RatioResult(**base)


def _gold_miner_ratios():
    ratios = {k: _ratio(k) for k in GOLD_MINER_KEYS}
    ratios["metal_miners"] = _ratio("metal_miners", current=8.4, percentile=96.0, status="RUBBER_BAND_STRETCHED",
                                    driver="DENOMINATOR_CHEAP", numerator_pctile=55.0, denominator_pctile=9.0,
                                    median=6.25)
    return ratios


def test_gauge_is_not_a_ratio_and_not_in_the_summary():
    assert cr.EXPOSURE_TO_RATIO["gold_miner"] == GOLD_MINER_KEYS
    assert cr.EXPOSURE_TO_RATIO["junior_miner"] == ["gdxj_gdx", "gold_gdxj"]
    assert all("margin" not in s.key for s in cr._RATIO_SPECS)
    assert all("margin" not in k for keys in cr.EXPOSURE_TO_RATIO.values() for k in keys)
    src = inspect.getsource(aui._compute_ratio_summary)
    assert "margin" not in src.lower() and "_mm" not in src
    assert aui._compute_ratio_summary(_gold_miner_ratios(), GOLD_MINER_KEYS) == "Gummisnoddar: 1 KÖP-STÖD, 3 NEUTRAL"


def test_engine_never_imports_the_gauge():
    for name in ("engine.py", "commodity_ratios.py", "contrarian_signals.py"):
        src = open(os.path.join(ROOT, "alpha_regime", name), encoding="utf-8").read()
        assert "miner_margin" not in src and "margin_ratio" not in src, name


def test_engine_run_never_calls_the_gauge_and_keeps_one_confirmation(monkeypatch):
    from alpha_regime import engine as eng
    idx = pd.bdate_range(end="2026-10-30", periods=300)
    c = np.linspace(100.0, 80.0, 300)
    df = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e5}, index=idx)
    calls = []
    monkeypatch.setattr(eng, "_fetch_price", lambda t, period="2y": df)
    monkeypatch.setattr(eng, "_fetch_fundamentals", lambda t: {"branch_name": "Gold Miners"})
    monkeypatch.setattr(eng, "_fetch_ratios", _gold_miner_ratios)
    monkeypatch.setattr(eng, "_fetch_context_gauges", lambda: {})
    monkeypatch.setattr(mm, "compute", lambda *a, **k: calls.append(("compute", a)))
    monkeypatch.setattr(mm, "_compute_cached", lambda *a, **k: calls.append(("_compute_cached", a)))
    monkeypatch.setattr(mm, "build_result", lambda *a, **k: calls.append(("build_result", a)))
    r = eng.run_regime_analysis("NEM", mode="contrarian", market_ticker="SPY", with_sentiment=False)
    assert calls == []
    assert r.detected_exposure == GOLD_MINER_KEYS
    assert [x for x in r.contrarian.rationale if x.startswith("Rubber Band:")] == [
        "Rubber Band: Gold ETF / Gold Miners at 96th percentile — Gold Miners historically stretched cheap"]
    assert not any("marginal" in x.lower() for x in r.contrarian.rationale)


# ── AppTest (inget nätverk) ──────────────────────────────────────────────────
def _margin_app():
    import streamlit as st
    from alpha_regime import ui as u
    u._render_miner_margin(st.session_state.get("_keys", []))


def _contrarian_app():
    import streamlit as st
    from alpha_regime import ui as u
    u._render_contrarian_mode(st.session_state["_r"], None)


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [w.value for w in at.warning])


def _fake_prices():
    idx = pd.bdate_range(end="2026-10-30", periods=2600)
    n = len(idx)
    return {"GC=F": pd.Series(np.linspace(1250.0, 4010.0, n), index=idx),
            "GDX": pd.Series(np.linspace(22.0, 75.0, n), index=idx),
            "GDXJ": pd.Series(np.linspace(35.0, 100.0, n), index=idx)}


def _run_margin(keys):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_margin_app, default_timeout=60)
    at.session_state["_keys"] = keys
    at.run()
    assert not at.exception, at.exception
    return at


def test_apptest_gauge_renders_from_fake_prices(monkeypatch):
    prices = _fake_prices()
    monkeypatch.setattr(mm, "_download_close", lambda t, period="10y": prices.get(t, pd.Series(dtype=float)))
    monkeypatch.setattr(mm, "is_stale", lambda newest, today=None: False)
    at = _run_margin(GOLD_MINER_KEYS)
    html = _texts(at)
    assert "Gruvorna mot marginalen (kontext)" in html
    for miner in ("GDX", "GDXJ"):
        res = mm.build_result(miner, prices[miner], prices["GC=F"])
        assert res.status == "OK"
        assert aui._margin_value_text(res) in html
        assert f"{miner} mot marginalen" in html
    assert f"Guld 4{NB}010 − AISC 1{NB}785 (2026Q1, World Gold Council/Metals Focus) = 2{NB}225 USD/uns" in html
    assert "← BILLIGT MOT MARGINALEN" in html and "flex-wrap" in html
    caption = " ".join(c.value for c in at.caption)
    assert "Bara kontext — ingår aldrig i signalerna" in caption
    assert "GDXJ räknas mot samma bransch-AISC" in caption
    # källraden läses ur raderna fönstret använder — 2016–2019 är helårstal, inte WGC/MF-kvartal
    start = mm.build_result("GDX", prices["GDX"], prices["GC=F"]).history_start
    assert f"en del av fönstret (sedan {start}) bygger på helårstal" in caption
    assert ("S&P Global helår 2016 (gäller 2016Q4–2017Q4), Refinitiv GFMS helår 2018 och Metals Focus helår "
            "2019 (gäller 2019Q1–2020Q1)") in caption
    assert "AISC är branschsnittet per kvartal (World Gold Council/Metals Focus), senaste" not in caption
    assert not at.warning                                  # färsk tabell, data finns
    for word in ("KÖP-STÖD", "SÄLJ-STÖD", "köpstöd"):
        assert word not in html


def test_apptest_stale_table_warns_once(monkeypatch):
    prices = _fake_prices()
    monkeypatch.setattr(mm, "_download_close", lambda t, period="10y": prices.get(t, pd.Series(dtype=float)))
    monkeypatch.setattr(mm, "is_stale", lambda newest, today=None: True)
    at = _run_margin(GOLD_MINER_KEYS)
    warns = [w.value for w in at.warning]
    assert warns == ["AISC-tabellen är äldre än två kvartal — uppdatera alpha_regime/miner_margin.py"]


def test_apptest_junior_exposure_shows_only_gdxj_and_silver_shows_nothing(monkeypatch):
    prices = _fake_prices()
    monkeypatch.setattr(mm, "_download_close", lambda t, period="10y": prices.get(t, pd.Series(dtype=float)))
    monkeypatch.setattr(mm, "is_stale", lambda newest, today=None: False)
    html = _texts(_run_margin(cr.EXPOSURE_TO_RATIO["junior_miner"]))
    assert "GDXJ kostar" in html and "GDX kostar" not in html and "GDXJ räknas mot samma bransch-AISC" in html
    at = _run_margin(cr.EXPOSURE_TO_RATIO["silver"])
    assert not at.markdown and not at.caption and not at.warning


def test_apptest_gold_missing_shows_a_swedish_data_gap(monkeypatch):
    prices = _fake_prices()
    prices.pop("GC=F")
    monkeypatch.setattr(mm, "_download_close", lambda t, period="10y": prices.get(t, pd.Series(dtype=float)))
    monkeypatch.setattr(mm, "is_stale", lambda newest, today=None: False)
    at = _run_margin(["metal_miners"])
    assert [w.value for w in at.warning] == [
        "DATA_GAP — GDX mot marginalen: Guldpriset (GC=F) saknas — utan det går marginalen inte att räkna "
        "(ingen reservkälla)."]


def _regime(ratios):
    from alpha_regime.contrarian_signals import ContrairianStageResult
    from alpha_regime.engine import RegimeResult
    return RegimeResult(mode="contrarian", ticker="NEM", market_ticker="SPY", market_phase="DISBELIEF",
                        market_confidence=60.0, price_vs_ma200=-12.0, trend_phase="Bearish",
                        contrarian=ContrairianStageResult("HOLD", "HOLD", "#607080", rationale=["Kontrollrad"]),
                        commodity_ratios=ratios, detected_exposure=list(GOLD_MINER_KEYS))


@pytest.mark.parametrize("pct", [3.0, 97.0])
def test_apptest_contrarian_mode_places_the_gauge_and_keeps_the_summary(monkeypatch, pct):
    """Billiga eller dyra gruvor mot marginalen — banderollens 'Gummisnoddar: …' och checklistan är desamma."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(aui, "_cached_deep_rows", lambda: ([], ""))
    monkeypatch.setattr(mm, "compute", lambda miner="GDX": _margin(miner, pct, newest_period="2026Q1"))
    at = AppTest.from_function(_contrarian_app, default_timeout=60)
    at.session_state["_r"] = _regime(_gold_miner_ratios())
    at.run()
    assert not at.exception, at.exception
    md = [m.value for m in at.markdown]
    summary = [v for v in md if "Gummisnoddar:" in v]
    assert len(summary) == 1 and "Gummisnoddar: 1 KÖP-STÖD, 3 NEUTRAL</div>" in summary[0]
    # ordningen: snoddarna → marginalmätaren → checklistan
    i_band = next(i for i, v in enumerate(md) if "Rubber Band Gauges" in v)
    i_last_ratio = max(i for i, v in enumerate(md) if "GDX / SPY (Miners/Market)" in v)
    i_margin = next(i for i, v in enumerate(md) if "Gruvorna mot marginalen (kontext)" in v)
    i_check = next(i for i, v in enumerate(md) if "Checklist" in v)
    assert i_band < i_last_ratio < i_margin < i_check
    assert "- Kontrollrad" in md and not any("marginal" in v.lower() for v in md[i_check:])
    word = "billigt" if pct < 50 else "dyrt"
    assert sum(f"{word} mot marginalen" in v for v in md) == 2           # GDX och GDXJ
