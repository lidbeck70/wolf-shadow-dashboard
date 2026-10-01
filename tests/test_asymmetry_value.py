"""
Wolf Asymmetry för värdebolag (PR 1): läge, Margin of Safety-kortet,
värdedrivarna (omvärdering, marginalåterhämtning, dubbel hävstång,
tillväxt) och värdefälle-kontrollerna. Syntetiskt bolag, facit räknat
ur samma formler som modulen visar.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from asymmetry import quick  # noqa: E402
from asymmetry import quick_config as qc  # noqa: E402
from asymmetry import quick_value as qv  # noqa: E402

YEARS = list(range(2016, 2026))
REV = {y: 10000.0 * 1.05 ** i for i, y in enumerate(YEARS)}               # +5 %/år
MARGINS = dict(zip(YEARS, [12, 13, 12, 14, 12, 11, 12, 9, 8, 8]))           # median 12, nu 8


def _value_co(**kw):
    eb = {y: REV[y] * MARGINS[y] / 100 for y in YEARS}
    ebitda_now = REV[2025] * 0.08
    d = {"ticker": "VAL.ST", "name": "Value AB", "source": "Börsdata", "currency": "SEK", "price_currency": "SEK",
         "revenue_series": list(REV.items()), "ebitda_series": list(eb.items()), "ebitda_margin_pct": 8.0,
         "ev_ebitda_hist": [8.0, 9.0, 10.0, 11.0, 12.0], "ev_ebitda": 6.0, "net_debt": 2000.0, "nd_ebitda": 1.6,
         "mcap_bd": 6 * ebitda_now - 2000.0, "price": 100.0, "shares_growth_3y_pct": 2.0,
         "roic_series": [(y, 15.0) for y in YEARS], "commodity_px": {"commodity": "", "ticker": ""}}
    d.update(kw)
    return d


def test_mode_follows_the_commodity_theme():
    assert qv.detect_mode(_value_co()) == qc.MODE_VALUE
    assert qv.detect_mode(_value_co(commodity_px={"commodity": "koppar"})) == qc.MODE_COMMODITY
    assert qv.detect_mode(_value_co(commodity_px={"commodity": "uran"})) == qc.MODE_COMMODITY


def test_financials_are_a_data_gap():
    assert "bank" in qv.analyze(_value_co(bd_branch_id=68)).error
    assert "Yahoo" in qv.analyze(_value_co(sector_text="Financial Services Banks - Regional")).error
    assert "Yahoo" in qv.analyze(_value_co(sector_text="Real Estate REIT - Office")).error
    assert qv.analyze(_value_co(bd_branch_id=71)).error is None              # fondförvaltare: EBITDA fungerar


def test_base_and_drivers():
    r = qv.analyze(_value_co())
    b = r.base
    assert r.error is None and b.implied_multiple == 6.0 and b.margin == 8.0
    assert b.margin_q["median"] == 12.0 and b.multiples["median"] == 10.0 and b.years == "2016–2025"
    assert b.growth == pytest.approx(0.05, abs=1e-4)
    by = {dv.key: dv for dv in r.drivers}
    ebitda_now, norm = REV[2025] * 0.08, REV[2025] * 0.12
    mcap = 6 * ebitda_now - 2000
    assert by["rerating"].ratio == round((ebitda_now * 10 - 2000) / mcap, 2)          # 1,91×
    assert by["margin"].ratio == round((norm * 6 - 2000) / mcap, 2)                    # 1,68×
    assert by["double"].ratio == round((norm * 10 - 2000) / mcap, 2)                   # 3,05×
    assert by["rerating"].upside_pct == 91 and by["double"].upside_pct == 205
    assert by["growth"].upside_pct == 5.0 and "senaste 3 år +5.0 %/år" in by["growth"].formula
    assert "egen median 10×" in by["rerating"].formula


def test_value_traps_from_measured_numbers():
    traps = {t.label: t for t in qv.analyze(_value_co()).traps}
    assert traps["Fallande marginal"].flagged is True                     # 12 → 8, under median
    assert traps["Krympande omsättning"].flagged is False
    assert traps["Låg avkastning på kapitalet"].flagged is False
    assert traps["Hög skuld"].flagged is False and traps["Utspädning"].flagged is False
    bad = {t.label: t for t in qv.analyze(_value_co(roic_series=[(y, 4.0) for y in YEARS], nd_ebitda=3.5,
                                                    shares_growth_3y_pct=25.0)).traps}
    assert bad["Låg avkastning på kapitalet"].flagged and bad["Hög skuld"].flagged and bad["Utspädning"].flagged
    unknown = {t.label: t for t in qv.analyze(_value_co(roic_series=[])).traps}
    assert unknown["Låg avkastning på kapitalet"].flagged is None


def test_data_gaps():
    short = [(y, v) for y, v in REV.items() if y >= 2023]
    assert "för få år" in qv.analyze(_value_co(revenue_series=short)).error
    assert "EV/EBITDA-historik" in qv.analyze(_value_co(ev_ebitda_hist=[6.0])).error
    assert qv.analyze(_value_co(net_debt=None)).error == "nettoskulden saknas"
    neg = [(y, -REV[y] * 0.05) for y in YEARS]
    assert "normalt negativ" in qv.analyze(_value_co(ebitda_series=neg)).error


def test_cagr_and_slope():
    assert qv.cagr({2020: 100.0, 2022: 121.0}) == pytest.approx(0.10)
    assert qv.cagr({2024: 100.0, 2025: 110.0}, years=3) is None
    assert qv.slope({2021: 10.0, 2022: 9.0, 2023: 8.0, 2024: 7.0}, 5) == pytest.approx(-1.0)


def test_value_mode_swaps_the_price_buffer_card():
    d = _value_co()
    mos = quick.margin_of_safety(d, qc.MODE_VALUE)
    card = mos.pillars[0]
    assert card.key == "margin_hist" and card.status == "RED"           # 8 / 12 = 0,67 < 0,7
    assert card.value == "8.0 % mot median 12.0 %"
    assert quick.margin_of_safety(d).pillars[0].key == "buffer"         # råvaruläget som förut
    ok = quick.margin_of_safety(_value_co(ebitda_margin_pct=11.5), qc.MODE_VALUE).pillars[0]
    assert ok.status == "GREEN"


def test_quick_tab_value_mode(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import storage
    from asymmetry import quick_data
    from confidence import store as cs
    from test_asymmetry_quick import _good
    good = _good()
    good.update({k: v for k, v in _value_co().items() if k not in ("ticker", "name")})
    good.update(yf_ticker="VAL.ST", fetched="2026-10-01 08:00 UTC")
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
    at.text_input(key="asym_quick_ticker").set_value("val.st")
    at.button(key="FormSubmitter:asym_quick_form-🔍 Analysera").click().run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Läge: <b>Värde</b>" in html and "VÄRDEDRIVARE" in html and "RÅVARUHÄVSTÅNG" not in html
    assert "OMVÄRDERING" in html and "+91 %" in html and "DUBBEL HÄVSTÅNG" in html
    assert "VÄRDEFÄLLA?" in html and "Fallande marginal" in html
    assert "Marginal mot egen historik" in html
    at.radio(key="asym_quick_mode").set_value(qc.MODE_COMMODITY).run()
    html = " ".join(m.value for m in at.markdown)
    assert "RÅVARUHÄVSTÅNG" in html and "VÄRDEDRIVARE" not in html
