"""
Wolf Asymmetry för värdebolag (PR 2): 5×-motorn — scenarier, tillväxt som
krävs för 2×/3×/5×/10×, stressmatris och thesis killers. Samma syntetiska
bolag som i test_asymmetry_value (omsättning +5 %/år, marginal nu 8 % mot
median 12 %, EV/EBITDA 8–12×, nettoskuld 2 000, dagens multipel 6×).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asymmetry import quick_value_scenarios as qvs  # noqa: E402
from test_asymmetry_value import REV, YEARS, _value_co  # noqa: E402

R = REV[2025]
EB_NOW = R * 0.08
MCAP = 6 * EB_NOW - 2000


def _ratio(rev, margin, mult):
    return round((rev * margin / 100 * mult - 2000) / MCAP, 2)


def test_scenarios_follow_the_chain():
    e = qvs.run(_value_co())
    assert e.error is None and e.growth_used == pytest.approx(0.05, abs=1e-4)
    by = {s.name: s for s in e.scenarios}
    # BEAR: P25-marginalen 9,5 är BÄTTRE än dagens 8 → bear tar dagens; låg multipel 9×
    assert by["BEAR"].margin == 8.0 and by["BEAR"].margin_key == "now" and by["BEAR"].multiple == 9.0
    assert by["BEAR"].ratio == _ratio(R, 8, 9)
    assert by["BASE"].ratio == _ratio(R, 8, 10) and by["BASE"].share_price == pytest.approx(100 * by["BASE"].ratio, abs=0.6)
    bull = by["BULL"]                                    # median-marginal 12, median 10×, 3 år à 5 %
    assert bull.margin == 12.0 and bull.years == 3 and bull.revenue == round(R * 1.05 ** 3, 0)
    assert bull.ratio == _ratio(R * 1.05 ** 3, 12, 10)
    sb = by["SUPER BULL"]                                # P75 12, hög multipel 11×, 5 år
    assert sb.multiple == 11.0 and sb.ratio == _ratio(R * 1.05 ** 5, 12, 11)
    assert by["BEAR"].ratio <= by["BASE"].ratio <= bull.ratio <= sb.ratio


def test_bull_never_assumes_a_worse_margin_than_today():
    e = qvs.run(_value_co(ebitda_margin_pct=14.5))      # över medianen
    bull = next(s for s in e.scenarios if s.name == "BULL")
    assert bull.margin == 14.5 and bull.margin_key == "now"


def test_growth_required_for_2x_to_10x():
    e = qvs.run(_value_co())
    req = {r.multiple: r for r in e.requirements}
    for k in (2, 3, 5, 10):
        rev_req = (k * MCAP + 2000) / 10 / 0.12
        want = 0.0 if rev_req <= R else ((rev_req / R) ** (1 / 5) - 1) * 100
        assert req[k].growth_pct == round(want, 1) and req[k].revenue == round(rev_req, 0)
    assert req[2].verdict == "JA"                       # dubbel hävstång räcker (≈ 3×)
    assert req[5].verdict in ("VILLKORAT", "NEJ") and req[10].verdict == "NEJ"
    assert e.five_x == req[5].verdict and "egen tillväxt 5.0 %/år" in e.five_x_text


def test_stress_matrix_margin_against_multiple():
    e = qvs.run(_value_co())
    rows = {k: (m, r) for k, m, r in e.stress}
    assert list(rows) == ["min", "p25", "median", "p75", "max"]
    m, r = rows["median"]
    assert m == 12.0 and r["median"][0] == _ratio(R, 12, 10)
    assert rows["min"][1]["low"][0] < rows["max"][1]["high"][0]


def test_killers_are_the_traps_that_fired_plus_unmeasured():
    e = qvs.run(_value_co(nd_ebitda=3.2))
    labels = [k[0] for k in e.killers]
    assert "Fallande marginal" in labels and "Hög skuld" in labels and "Utspädning" not in labels
    assert [k[0] for k in e.killers if not k[2]] == ["Multipeln kan stanna låg", "Konkurrens, teknik, ledning"]


def test_high_growth_is_capped_and_named():
    rev = [(y, 1000.0 * 1.6 ** i) for i, y in enumerate(YEARS)]
    eb = [(y, v * 0.12) for y, v in rev]
    e = qvs.run(_value_co(revenue_series=rev, ebitda_series=eb, ebitda_margin_pct=12.0))
    assert e.growth_used == 0.30 and any(k[0] == "Tillväxten kapad" for k in e.killers)


def test_value_engine_data_gaps():
    assert "bank" in qvs.run(_value_co(bd_branch_id=68)).error
    assert "för få år" in qvs.run(_value_co(revenue_series=[(2025, 1.0)])).error


def test_quick_tab_shows_the_value_engine(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import storage
    from asymmetry import quick_data
    from confidence import store as cs
    from test_asymmetry_quick import _good
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    monkeypatch.setenv("ASYM_TEST_ROOT", root)

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
    assert "5×-MOTORN (VÄRDE)" in html and "5× POTENTIAL" in html and "BASE MOT BÖRSVÄRDET" in html
    assert "SUPER BULL" in html and "Tillväxt som" in html and "Stressmatris — marginal mot multipel" in html
    assert "VAD DÖDAR CASET?" in html and "Multipeln kan stanna låg" in html
