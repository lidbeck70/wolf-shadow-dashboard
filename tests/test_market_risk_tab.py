"""
🌩️ Marknadsrisk PR 2 — fliken: nivån för SPY och OMXS30, varningarna,
grafen och den historiska träffbilden. Syntetiska serier — inget nätverk.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import market_risk as mr  # noqa: E402
from test_market_risk import _data  # noqa: E402


def _results():
    _idx, d = _data(crash=True)
    omx = dict(d, **{"^OMX": d["SPY"]})
    spy = mr.evaluate("SPY", getter=lambda t, p: d.get(t), fred_getter=lambda s: d.get(s))
    om = mr.evaluate("OMXS30", getter=lambda t, p: omx.get(t), fred_getter=lambda s: omx.get(s))
    return {"SPY": spy, "OMXS30": om}


def test_evaluate_keeps_history_for_the_chart():
    r = _results()["SPY"]
    assert r.history is not None and len(r.history) == len(r.close)
    assert int(r.history.iloc[-1]) == r.points
    import market_risk_ui as ui
    fig = ui.history_chart(r)
    assert len(fig.data) == 2 and "ANTAL VARNINGAR" in fig.layout.title.text


def test_page_renders(monkeypatch):
    from streamlit.testing.v1 import AppTest
    res = _results()
    monkeypatch.setattr(mr, "evaluate", lambda market, **kw: res[market])
    monkeypatch.setenv("MR_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["MR_TEST_ROOT"])
        from market_risk_ui import render_market_risk_page
        render_market_risk_page()

    at = AppTest.from_function(app, default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "S&amp;P 500 (SPY)" in html or "S&P 500 (SPY)" in html
    assert "OMXS30" in html and "VARNINGAR NU" in html and "NIVÅ:" in html
    assert "HISTORISK TRÄFFBILD" in html and "basfrekvens" in html and "Mot normalt" in html
    assert "Nedgångar ≥ 10 %: <b>2</b>" in html and "falsklarm" in html
    assert any("ANTAL VARNINGAR" in c.proto.spec for c in at.get("plotly_chart"))
    at.radio(key="mr_market").set_value("OMXS30").run()
    assert not at.exception, at.exception
    assert "Breddivergens" not in " ".join(m.value for m in at.markdown)


def test_page_handles_missing_data(monkeypatch):
    from streamlit.testing.v1 import AppTest
    gone = mr.MarketRisk("SPY", "S&P 500 (SPY)", error="DATA UNAVAILABLE — ingen kurshistorik för SPY")
    monkeypatch.setattr(mr, "evaluate", lambda market, **kw: gone)
    monkeypatch.setenv("MR_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["MR_TEST_ROOT"])
        from market_risk_ui import render_market_risk_page
        render_market_risk_page()

    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert "DATA UNAVAILABLE" in " ".join(m.value for m in at.markdown)


def test_navigation_and_guide():
    from ui import nav
    from ovtlyr.ui.rules_page import _PANEL_GUIDE
    assert "🌩️ Marknadsrisk" in nav.options("regime/Marknad")
    assert any(t == "REGIME → Marknad → 🌩️ Marknadsrisk" for t, _r, _u in _PANEL_GUIDE)
    src = open(os.path.join(ROOT, "wolf_panel.py"), encoding="utf-8").read()
    assert 'elif sub == "🌩️ Marknadsrisk":' in src
