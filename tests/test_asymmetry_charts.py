"""
Wolf Asymmetry, steg C: de fem diagrammen.

Varje figur ritas för producenten och utvecklaren, blir None (inte ett
diagram med nollor) för det tomma bolaget, och fliken ritar dem utan
nätverk. Serier bär husets färger; status-färgerna används bara med etikett.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import durrett_cases as dcs  # noqa: E402
from asymmetry import analyze  # noqa: E402
from asymmetry import charts  # noqa: E402
from ui.tokens import AMBER, CYAN, GOLD, GREEN, RED  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _gpr():
    return analyze(dcs.gold_producer(), 75)


def test_price_grid_chart_has_ebitda_and_fcf_series():
    fig = charts.price_grid_chart(_gpr())
    names = [t.name for t in fig.data]
    assert names == ["EBITDA", "FCF"]
    assert list(fig.data[0].x) == [-30, -20, -10, 0, 10, 20, 30, 50]
    assert fig.data[1].line.color == CYAN and fig.data[0].line.color == GOLD
    assert fig.layout.showlegend is not False                       # två serier → legend
    assert "poäng mäts här (6/10)" in " ".join(a.text for a in fig.layout.annotations)
    assert charts.price_grid_chart(analyze(dcs.missing_everything())) is None


def test_safety_chart_colors_by_band_and_skips_not_applicable():
    fig = charts.safety_chart(_gpr())
    bar = fig.data[0]
    assert "CapEx-marginal" not in list(bar.y)                        # ej tillämpligt för producent
    assert len(bar.y) == 4 and list(bar.text)[::-1] == ["2/2", "2/2", "2/2", "1/2"]
    colors = list(bar.marker.color)[::-1]
    assert colors[:3] == [GREEN, GREEN, GREEN] and colors[3] == AMBER  # 1/2 = 50 % → gult band (40–70 %)
    roy = charts.safety_chart(analyze(dcs.royalty_company()))
    assert "DATA_MISSING" in list(roy.data[0].text)
    assert charts.safety_chart(analyze(dcs.missing_everything())) is None


def test_scenario_chart_signs_and_asymmetry_title():
    fig = charts.scenario_chart(_gpr())
    bar = fig.data[0]
    assert list(bar.x) == ["Bear", "Base", "Bull", "Super Bull"]
    assert list(bar.marker.color) == [RED, GREEN, GREEN, GREEN]
    assert "BULL/|BEAR| 1.3×" in fig.layout.title.text and "SYMMETRISK" in fig.layout.title.text
    assert charts.scenario_chart(analyze(dcs.lithium_explorer())) is None


def test_matrix_chart_is_diverging_around_zero():
    cdv = analyze(dcs.copper_developer())
    fig = charts.matrix_chart(cdv)
    hm = fig.data[0]
    assert hm.zmid == 0 and hm.zmin == -hm.zmax
    assert len(hm.z) == 3 and len(hm.z[0]) == 4
    assert hm.text[0][3] == "+24 %"                                   # pris −20 %, capex +40 %
    assert list(hm.colorscale)[1][1] == "#1a1f25"                     # neutral mitt, ingen kulör vid 0
    assert charts.matrix_chart(analyze(dcs.missing_everything())) is None


def test_adjusted_upside_chart_marks_this_company():
    r = _gpr()
    fig = charts.adjusted_upside_chart(r)
    line, dot = fig.data
    assert line.y[-1] == r.base_upside_pct and line.y[0] == 0
    assert dot.x[0] == 75 and abs(dot.y[0] - 6.5) < 0.01
    assert charts.adjusted_upside_chart(analyze(dcs.missing_everything(), 40)) is None
    # utan confidence: linjen finns, ingen markör
    fig2 = charts.adjusted_upside_chart(analyze(dcs.gold_producer()))
    assert len(fig2.data) == 1


def test_every_chart_runs_for_every_archetype():
    for name, mk in dcs.ALL.items():
        r = analyze(mk(), 60)
        for fn in charts.ALL:
            fig = fn(r)
            assert fig is None or fig.data, (name, fn.__name__)


def test_tab_draws_the_charts_without_network(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import storage
    from confidence import store as cs

    data = cs.default()
    for mk in (dcs.gold_producer, dcs.missing_everything):
        cs.put(data, mk())
    stores = {"confidence": data}
    monkeypatch.setattr(storage, "session_load", lambda name, default=None, legacy_file=None:
                        st.session_state.setdefault(name, stores.get(name, default)))
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

    expected = {"Översikt": 2, "Varför?": 1, "Scenarier": 1, "Stressmatris": 1}
    for sub, n in expected.items():
        at = AppTest.from_function(app, default_timeout=60)
        at.session_state["asym_sub"] = sub
        at.run()
        assert not at.exception, (sub, at.exception)
        assert len(at.get("plotly_chart")) == n, sub
        # tomt bolag: inga diagram, en förklaring i stället
        at = AppTest.from_function(app, default_timeout=60)
        at.session_state["asym_pick"] = "NUL"
        at.session_state["asym_sub"] = sub
        at.run()
        assert not at.exception, (sub, at.exception)
        assert len(at.get("plotly_chart")) == 0, sub
        assert any("Diagrammet kan inte ritas" in c.value for c in at.caption), sub
