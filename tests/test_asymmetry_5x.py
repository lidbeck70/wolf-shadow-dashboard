"""
Asymmetry 1.1, PR 5: 5×-motorn i Snabbkollen, automatisk. Råvarupris →
EBITDA (egen linje) → EV (egen EV/EBITDA) → − nettoskuld → mot börsvärdet →
kurs. Syntetiskt bolag, facit räknat för hand.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from asymmetry import quick, quick_data  # noqa: E402
from asymmetry import quick_scenarios as qs  # noqa: E402

YEARS = list(range(2016, 2026))
PRICES = dict(zip(YEARS, [2.5, 2.8, 3.0, 2.7, 2.9, 4.2, 4.0, 3.8, 4.1, 4.3]))   # koppar, högsta 4,3


def _data(**kw):
    eb = [(y, 1000.0 * p - 2000) for y, p in PRICES.items()]          # EBITDA = 1000·P − 2000
    d = {"ticker": "CU.X", "name": "Copper Co", "source": "Börsdata", "currency": "USD", "price_currency": "USD",
         "revenue_series": [(y, 1000.0 * p) for y, p in PRICES.items()], "ebitda_series": eb,
         "fcf_series": [(y, v - 500) for y, v in eb],
         "commodity_px": {"commodity": "koppar", "ticker": "HG=F", "prices": dict(PRICES), "p0": 4.0, "fx_now": 1.0},
         "ev_ebitda_hist": [4.0, 5.0, 6.0, 7.0, 8.0], "ev_ebitda": 7.0, "net_debt": 1000.0, "nd_ebitda": 0.5,
         "mcap_bd": 11000.0, "price": 50.0, "shares_growth_3y_pct": 2.0}
    d.update(kw)
    return d


def test_own_multiples_are_quartiles_of_the_history():
    assert qs.own_multiples([4, 5, 6, 7, 8]) == {"low": 5.0, "median": 6.0, "high": 7.0}
    assert qs.own_multiples([6, 7]) == {} and qs.own_multiples([5, -3, 150, 6, 7]) == {"low": 5.5, "median": 6.0, "high": 6.5}


def test_scenarios_follow_the_chain():
    e = qs.run(_data())
    assert e.error is None and e.multiples == {"low": 5.0, "median": 6.0, "high": 7.0}
    by = {s.name: s for s in e.scenarios}
    base = by["BASE"]                         # EBITDA 2000 × 6 = 12000 − 1000 = 11000 = börsvärdet
    assert (base.ebitda, base.ev, base.equity, base.ratio, base.share_price) == (2000, 12000, 11000, 1.0, 50.0)
    bear = by["BEAR"]                         # 3,2 → 1200 × 5 − 1000 = 5000
    assert bear.price == 3.2 and bear.multiple == 5.0 and bear.equity == 5000 and bear.ratio == 0.45
    assert by["BULL"].equity == 18200 and by["BULL"].ratio == 1.65 and by["BULL"].outside_history   # 5,2 > 4,3
    assert by["SUPER BULL"].price == 6.4 and by["SUPER BULL"].ratio == 2.31
    assert not base.outside_history


def test_what_is_required_for_2x_to_10x():
    e = qs.run(_data())
    req = {r.multiple: r for r in e.requirements}
    # 2×: EBITDA (22000 + 1000) / 6 = 3833 → pris 5,83 (+46 %) ≤ 4,3 × 1,5 → VILLKORAT
    assert req[2].price == 5.83 and req[2].price_pct == 46 and req[2].verdict == "VILLKORAT"
    # 5×: (55000 + 1000) / 6 = 9333 → 11,33 (+183 %) → NEJ
    assert req[5].price == 11.33 and req[5].price_pct == 183 and req[5].verdict == "NEJ"
    assert e.five_x == "NEJ" and "Koppar +183 % (11.33 USD/lb)" in e.five_x_text and "median 6×" in e.five_x_text


def test_cheap_company_gets_five_x_yes():
    e = qs.run(_data(mcap_bd=1500.0))           # base = 11000 / 1500 = 7,3×
    assert e.five_x == "JA" and next(s for s in e.scenarios if s.name == "BASE").ratio == 7.33


def test_stress_matrix_price_against_multiple():
    e = qs.run(_data())
    rows = dict(e.stress)
    assert list(rows) == [-30.0, -20.0, -10.0, 0.0, 20.0, 40.0]
    assert rows[0.0]["median"] == (1.0, 50.0)
    assert rows[-30.0]["low"][0] < rows[0.0]["median"][0] < rows[40.0]["high"][0]


def test_killers_come_from_measured_numbers_only():
    e = qs.run(_data())
    labels = [k.label for k in e.killers]
    assert "Värderingen redan hög" in labels and "Bull kräver nya pristoppar" in labels
    assert "Skuld" not in labels and "Utspädning" not in labels and "Valuta" not in labels
    assert [k.label for k in e.killers if not k.measured] == ["En råvara räknas", "Capex, produktion, tillstånd"]
    e = qs.run(_data(nd_ebitda=3.1, shares_growth_3y_pct=40.0, currency="SEK", price_currency="SEK",
                     commodity_px={"commodity": "koppar", "ticker": "HG=F", "prices": dict(PRICES), "p0": 4.0,
                                   "fx_now": 1.0}))
    labels = [k.label for k in e.killers]
    assert {"Skuld", "Utspädning", "Valuta"} <= set(labels)


def test_market_cap_is_converted_to_the_report_currency():
    from sheets_refresh import FX_TO_USD
    mc, note = qs.market_cap_report({"currency": "USD", "mcap_bd": 10000.0, "price_currency": "CAD"})
    assert mc == 10000.0 * FX_TO_USD["CAD"] and "CAD→USD" in note
    mc, note = qs.market_cap_report({"currency": "SEK", "mcap_yahoo": 500.0, "mcap_yahoo_ccy": "SEK"})
    assert mc == 500.0 and "Yahoo" in note


def test_data_gaps_are_named():
    assert "EV/EBITDA-historik" in qs.run(_data(ev_ebitda_hist=[6.0])).error
    assert qs.run(_data(net_debt=None)).error == "nettoskulden saknas"
    assert qs.run(_data(mcap_bd=None)).error == "börsvärdet saknas"
    noisy = [(y, v) for y, v in zip(YEARS, [900, 400, 1200, 50, 3000, 200, 800, 1500, 60, 2500])]
    assert "EBITDA följer inte" in qs.run(_data(ebitda_series=noisy)).error
    assert "okänd" in qs.run(_data(commodity_px={})).error


def test_engine_is_outside_the_300():
    from test_asymmetry_quick import _good
    a = _good()
    a.update({k: v for k, v in _data().items() if k not in ("ticker", "name", "source")})
    b = copy.deepcopy(a)
    b.pop("commodity_px")                                   # utan motor
    assert quick.score(a).total == quick.score(b).total
    assert qs.run(a).five_x == "NEJ" and qs.run(b).error


def test_the_quick_tab_shows_the_engine(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import storage
    from confidence import store as cs
    from test_asymmetry_quick import _good
    good = _good()
    good.update({k: v for k, v in _data().items() if k not in ("ticker", "name", "source", "fcf_series")})
    good.update(yf_ticker="CU.X", fetched="2026-09-30 19:00 UTC")
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
    at.text_input(key="asym_quick_ticker").set_value("cu.x")
    at.button(key="FormSubmitter:asym_quick_form-🔍 Analysera").click().run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "5×-MOTORN" in html and "5× POTENTIAL" in html and ">NEJ<" in html
    assert "SUPER BULL" in html and "Vad krävs?" in html and "Stressmatris" in html
    assert "VAD DÖDAR CASET?" in html and "Värderingen redan hög" in html
    assert any("Sannolikhet: UNKNOWN" in c.value for c in at.caption)
