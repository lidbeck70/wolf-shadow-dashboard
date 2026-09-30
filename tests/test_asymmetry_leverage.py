"""
Asymmetry 1.1, PR 3: Commodity Leverage (0–10) och break-even i Snabbkollen,
skattade ur bolagets egen historik — utanför 300. Syntetiska bolag med kända
kostnader, så facit går att räkna för hand.
"""
import copy
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from asymmetry import quick, quick_data  # noqa: E402
from asymmetry import quick_config as qc  # noqa: E402
from asymmetry import quick_leverage as ql  # noqa: E402

YEARS = list(range(2016, 2026))
PRICES = dict(zip(YEARS, [2.5, 2.8, 3.0, 2.7, 2.9, 4.2, 4.0, 3.8, 4.1, 4.3]))   # koppar USD/lb


def _company(costs, capex=500.0, per_unit=1000.0, prices=PRICES):
    rev = {y: per_unit * p for y, p in prices.items()}
    ebitda = {y: per_unit * p - costs for y, p in prices.items()}
    fcf = {y: e - capex for y, e in ebitda.items()}
    return rev, ebitda, fcf


# ── motorn ───────────────────────────────────────────────────────────────────
def test_fit_is_ordinary_least_squares():
    f = ql.fit([(1, 3), (2, 5), (3, 7)])
    assert (round(f.a, 6), round(f.b, 6), f.r2, f.n) == (1.0, 2.0, 1.0, 3)
    assert ql.fit([(1, 3)]) is None and ql.fit([(2, 1), (2, 5)]) is None


def test_solid_producer_high_leverage_strong_downside():
    rev, eb, fcf = _company(costs=2000)
    lev = ql.estimate(rev, eb, fcf, PRICES, p0=4.0, commodity="koppar", ticker="HG=F")
    # FCF = 1000·P − 2500. Vid 4,0: 1500. +20 % → +800 = +53 % → 8 p
    assert lev.basis == "FCF" and lev.response_pct == 53.3 and lev.score == 8 and lev.r2 == 1.0
    assert lev.break_even_price == 2.5 and lev.break_even_margin_pct == 37.5 and lev.band == "STARK"
    assert lev.downside == {-20.0: 700.0, -30.0: 300.0} and lev.downside_label == "STARK"
    assert lev.flag == "🟢 Hög hävstång + stark nedsida" and lev.unit == "USD/lb"
    base = next(r for r in lev.sensitivity if r["pct"] == 0)
    assert base == {"pct": 0.0, "price": 4.0, "revenue": 4000.0, "ebitda": 2000.0, "fcf": 1500.0,
                    "ebitda_margin": 50.0, "fcf_margin": 37.5}
    assert [r["pct"] for r in lev.sensitivity] == [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 50.0]


def test_highly_leveraged_producer_is_flagged_fragile():
    rev, eb, fcf = _company(costs=3200)
    lev = ql.estimate(rev, eb, fcf, PRICES, p0=4.0, commodity="koppar", ticker="HG=F")
    assert lev.score == 10 and lev.response_pct > 75                       # FCF 300 → +800
    assert lev.break_even_margin_pct == 7.5 and lev.band == "SVAG"
    assert lev.downside[-20.0] < 0 and lev.downside_label == "SKÖR"
    assert lev.flag == "🔴 Hög hävstång + skör nedsida"                     # hög hävstång ≠ bra


def test_low_leverage_business_scores_low():
    eb = {y: 100.0 * p + 5000 for y, p in PRICES.items()}
    lev = ql.estimate({}, eb, {}, PRICES, p0=4.0, commodity="guld", ticker="GC=F")
    assert lev.basis == "EBITDA" and lev.score == 0 and lev.response_pct < 10
    assert lev.break_even_price == 0.0 and lev.break_even_margin_pct == 100.0 and lev.band == "UTMÄRKT"


def test_ebitda_is_used_when_fcf_does_not_follow_the_price():
    rev, eb, _ = _company(costs=2000)
    noisy_fcf = dict(zip(YEARS, [900, -400, 1200, -50, 3000, 200, -800, 1500, 60, 2500]))
    lev = ql.estimate(rev, eb, noisy_fcf, PRICES, p0=4.0, commodity="koppar", ticker="HG=F")
    assert lev.basis == "EBITDA" and lev.downside_basis == "EBITDA"
    assert lev.fits["FCF"].r2 < qc.LEV_MIN_R2


def test_no_score_when_the_link_is_weak_negative_or_short():
    noise = dict(zip(YEARS, [900, 400, 1200, 50, 3000, 200, 800, 1500, 60, 2500]))
    lev = ql.estimate({}, noise, {}, PRICES, p0=4.0, commodity="koppar", ticker="HG=F")
    assert lev.score is None and "för svagt samband" in lev.error and lev.sensitivity  # tabellen visas ändå
    falling = {y: 10000 - 1000 * p for y, p in PRICES.items()}
    assert "negativ lutning" in ql.estimate({}, falling, {}, PRICES, 4.0, commodity="koppar", ticker="HG=F").error
    short = {y: v for y, v in _company(2000)[1].items() if y >= 2022}
    assert "för få år" in ql.estimate({}, short, {}, PRICES, 4.0, commodity="koppar", ticker="HG=F").error


def test_missing_commodity_or_price_is_data_gap_not_zero():
    rev, eb, fcf = _company(2000)
    assert "okänd" in ql.estimate(rev, eb, fcf, PRICES, 4.0).error
    assert "ingen prisserie för uran" in ql.estimate(rev, eb, fcf, PRICES, 4.0, commodity="uran").error
    assert "prishistorik" in ql.estimate(rev, eb, fcf, {}, None, commodity="koppar", ticker="HG=F").error


def test_break_even_is_shown_in_the_commodity_unit_after_fx():
    fx = 10.0                                                            # SEK per USD
    rev, eb, fcf = _company(costs=20000, capex=5000, per_unit=10000.0)   # samma bolag i SEK
    prices_sek = {y: p * fx for y, p in PRICES.items()}
    rev = {y: v * 1.0 for y, v in rev.items()}
    lev = ql.estimate(rev, eb, fcf, prices_sek, p0=40.0, fx_now=fx, commodity="koppar", ticker="HG=F")
    assert lev.price_now == 4.0 and lev.break_even_price == 2.5 and lev.score == 8


# ── datahämtningen ──────────────────────────────────────────────────────────
def _daily(values_by_year: dict) -> pd.Series:
    idx, vals = [], []
    for y, v in values_by_year.items():
        for d in pd.bdate_range(f"{y}-01-01", f"{y}-12-31")[:20]:
            idx.append(d)
            vals.append(v)
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


def test_commodity_prices_convert_to_the_report_currency():
    px = _daily({2024: 4.0, 2025: 5.0})
    fx = _daily({2024: 10.0, 2025: 11.0})
    got = quick_data.commodity_prices("koppar", "SEK", lambda s, p="10y": {"HG=F": px, "USDSEK=X": fx}[s])
    assert got["ticker"] == "HG=F" and got["prices"] == {2024: 40.0, 2025: 55.0}
    assert got["p0"] == 55.0 and got["fx_now"] == 11.0
    usd = quick_data.commodity_prices("guld", "USD", lambda s, p="10y": px)
    assert usd["prices"] == {2024: 4.0, 2025: 5.0} and usd["fx_now"] == 1.0
    assert quick_data.commodity_prices("uran", "USD", lambda s, p="10y": px)["ticker"] == ""
    no_fx = quick_data.commodity_prices("koppar", "SEK", lambda s, p="10y": px if s == "HG=F" else pd.Series(dtype=float))
    assert no_fx["prices"] == {} and no_fx["p0"] is None                  # utan valuta hellre inget


def test_leverage_is_outside_the_300():
    from test_asymmetry_quick import _good
    a = _good()
    b = copy.deepcopy(a)
    rev, eb, fcf = _company(3200)
    b.update(revenue_series=list(rev.items()), ebitda_series=list(eb.items()), fcf_series=list(fcf.items()),
             commodity_px={"commodity": "koppar", "ticker": "HG=F", "prices": PRICES, "p0": 4.0, "fx_now": 1.0})
    assert quick.score(a).total == quick.score(b).total
    assert ql.from_data(b).score == 10 and ql.from_data(a).score is None


def test_the_quick_tab_shows_leverage_and_break_even(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import storage
    from confidence import store as cs
    from test_asymmetry_quick import _good
    good = _good()
    rev, eb, fcf = _company(2000)
    good.update(revenue_series=list(rev.items()), ebitda_series=list(eb.items()), fcf_series=list(fcf.items()),
                commodity_px={"commodity": "koppar", "ticker": "HG=F", "prices": PRICES, "p0": 4.0, "fx_now": 1.0},
                yf_ticker="BOL.ST", fetched="2026-09-30 18:00 UTC")
    monkeypatch.setattr(quick_data, "fetch", lambda t, **kw: dict(good, ticker=t))
    stores = {"confidence": cs.default()}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, stores.get(name) if stores.get(name) is not None else default))
    monkeypatch.setattr(storage, "load_error", lambda name: None)
    monkeypatch.setattr(storage, "is_dirty", lambda name: False)
    monkeypatch.setattr(storage, "last_saved", lambda name: None)
    monkeypatch.setenv("ASYM_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["ASYM_TEST_ROOT"])
        from asymmetry.ui import render_asymmetry_page
        render_asymmetry_page()

    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    at.text_input(key="asym_quick_ticker").set_value("bol.st")
    at.button(key="FormSubmitter:asym_quick_form-🔍 Analysera").click().run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "RÅVARUHÄVSTÅNG" in html and "8/10" in html and "Hög hävstång + stark nedsida" in html
    assert "BREAK-EVEN-MARGINAL" in html and "38 %" in html and "STARK" in html
    caps = [c.value for c in at.caption]
    assert any("Samband — " in c and "FCF: R² 1.00 (10 år)" in c for c in caps)
    assert any("KOPPAR-PRIS MOT EBITDA OCH FCF" in ch.proto.spec for ch in at.get("plotly_chart"))


def test_fetch_builds_the_series_and_falls_back_to_margin_times_revenue():
    from test_asymmetry_quick import _Api, _closes

    class _A(_Api):
        def get_kpi_history(self, iid, kpi, rt, pt):
            data = {53: [1000.0, 1200.0, 1100.0], 32: [30.0, 40.0, 35.0], 11: [6.0, 7.0, 5.5], 76: [14.0, 12.0, 16.0]}
            if kpi == 54:
                return []                                               # EBITDA-KPI:n saknas
            return [{"y": 2023 + i, "v": v} for i, v in enumerate(data[kpi])]
    px = _daily({2023: 3.8, 2024: 4.1, 2025: 4.3})
    d = quick_data.fetch("BOL.ST", api=_A(), price_getter=lambda s: _closes(), info_getter=lambda s: {},
                         series_getter=lambda s, p="10y": px if s == "HG=F" else _daily({2023: 10.0, 2024: 10.5,
                                                                                         2025: 10.0}),
                         theme_getter=lambda s: "koppar")
    assert d["revenue_series"] == [(2023, 1000.0), (2024, 1200.0), (2025, 1100.0)]
    assert d["ebitda_series"] == [(2023, 300.0), (2024, 480.0), (2025, 385.0)]
    cp = d["commodity_px"]
    assert cp["commodity"] == "koppar" and cp["ticker"] == "HG=F" and cp["prices"][2024] == round(4.1 * 10.5, 10)
    assert ql.from_data(d).error.startswith("för få år")                 # tre år räcker inte
