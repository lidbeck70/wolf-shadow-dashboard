"""
GRANSKNING → 🚀 Råvaruhävstång: resultathävstång (Snabbkollens motor) och
kurshävstång (veckobeta, upp/ned) per bolag, jämförelse och detalj.
"""
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from commodity_leverage import beta as cb  # noqa: E402
from commodity_leverage import config as cc  # noqa: E402
from commodity_leverage import engine as ce  # noqa: E402


def _pair(up_beta=2.0, down_beta=2.0, weeks=200, seed=3):
    """Råvara och aktie (dagliga serier) där aktien rör sig up_beta × råvaran i
    uppveckor och down_beta × i nedveckor."""
    idx = pd.date_range(end="2026-09-25", periods=weeks + 1, freq="W-FRI")
    c, s = [100.0], [50.0]
    for i in range(weeks):
        r = 0.03 * math.sin(i * 1.7 + seed) + 0.01 * math.cos(i * 0.37)
        c.append(c[-1] * (1 + r))
        s.append(s[-1] * (1 + (up_beta if r > 0 else down_beta) * r))
    return pd.Series(s, index=idx), pd.Series(c, index=idx)


# ── kursbeta ─────────────────────────────────────────────────────────────────
def test_beta_recovers_a_known_multiplier():
    s, c = _pair(2.0, 2.0)
    b, err = cb.stock_beta(s, c)
    assert err is None and b.beta == 2.0 and b.r2 == 1.0
    assert b.up_beta == 2.0 and b.down_beta == 2.0 and not b.asymmetric and not b.weak
    assert b.weeks == 156                                            # 3 år veckor


def test_up_and_down_beta_show_asymmetry():
    s, c = _pair(2.5, 1.2)
    b, _ = cb.stock_beta(s, c)
    assert b.up_beta == 2.5 and b.down_beta == 1.2 and b.asymmetric
    assert b.up_weeks + b.down_weeks == b.weeks


def test_beta_gaps():
    s, c = _pair(weeks=30)
    b, err = cb.stock_beta(s, c)
    assert b is None and "för kort kurshistorik" in err
    assert cb.stock_beta(None, c)[0] is None
    flat = pd.Series([100.0] * 200, index=pd.date_range(end="2026-09-25", periods=200, freq="W-FRI"))
    assert cb.stock_beta(_pair()[0], flat)[1] == "råvarans pris har inte rört sig"


def test_weak_link_is_marked():
    s, c = _pair(2.0, 2.0)
    noise = pd.Series([50 * (1 + 0.05 * math.sin(i * 2.9)) for i in range(len(s))], index=s.index)
    b, _ = cb.stock_beta(noise, c)
    assert b.weak and b.r2 < cc.BETA_WEAK_R2


# ── analysen ────────────────────────────────────────────────────────────────
def _company_data(theme):
    from test_asymmetry_5x import _data
    d = _data()
    d["commodity_px"]["commodity"] = theme or "koppar"
    d.update(yf_ticker="CU.X")
    return d


def test_analyze_combines_the_quick_engine_and_the_beta():
    s, c = _pair(2.0, 1.0)
    seen = {}

    def fetcher(t, series_getter=None, theme_getter=None):
        seen["theme"] = theme_getter("X") if theme_getter else None
        return _company_data(seen["theme"])

    def getter(sym, period):
        return {"CU.X": s, "HG=F": c}[sym]
    r = ce.analyze("cu.x", fetcher=fetcher, series_getter=getter)
    assert r.ticker == "CU.X" and seen["theme"] is None and not r.locked          # auto
    assert r.lev.score == 8 and r.eng.five_x == "NEJ" and r.beta.beta > 1.0 and r.beta.asymmetric
    r2 = ce.analyze("cu.x", "koppar", fetcher=fetcher, series_getter=getter)
    assert seen["theme"] == "koppar" and r2.locked


def test_beta_needs_a_commodity_series():
    def fetcher(t, series_getter=None, theme_getter=None):
        return {"ticker": t, "name": t, "commodity_px": {"commodity": "uran", "ticker": ""}}
    r = ce.analyze("NXE", fetcher=fetcher, series_getter=lambda s, p: pd.Series(dtype=float))
    assert r.beta is None and "ingen prisserie för uran" in r.beta_error
    assert r.lev.score is None


def test_ranking_and_ticker_parsing():
    class L:
        def __init__(self, score):
            self.score = score
    a = ce.CompanyLeverage("A", lev=L(4))
    b = ce.CompanyLeverage("B", lev=L(8))
    c = ce.CompanyLeverage("C", lev=L(None), beta=cb.StockBeta(3.0, 0.5, 100, None, None, 0, 0, "", ""))
    assert [x.ticker for x in sorted([a, c, b], key=ce.rank_key)] == ["B", "A", "C"]
    assert ce.parse_tickers("fcx, scco;teck\nfcx, ") == ["FCX", "SCCO", "TECK"]
    assert len(ce.parse_tickers(",".join(f"T{i}" for i in range(20)))) == cc.MAX_TICKERS


# ── fliken ──────────────────────────────────────────────────────────────────
def test_tab_compares_and_details(monkeypatch):
    from streamlit.testing.v1 import AppTest
    s, c = _pair(2.4, 1.1)
    real = ce.analyze

    def fake_analyze(t, commodity=None, fetcher=None, series_getter=None):
        def fetcher_(tt, series_getter=None, theme_getter=None):
            return _company_data("koppar")
        return real(t, commodity, fetcher=fetcher_, series_getter=lambda sym, p: {"HG=F": c}.get(sym, s))
    monkeypatch.setattr(ce, "analyze", fake_analyze)
    monkeypatch.setattr(ce, "_series_default", lambda sym, p: {"HG=F": c}.get(sym, s))
    monkeypatch.setenv("CL_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["CL_TEST_ROOT"])
        from commodity_leverage.ui import render_commodity_leverage_page
        render_commodity_leverage_page()

    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert "Skriv ett eller flera bolag" in " ".join(m.value for m in at.markdown)
    at.text_input(key="cl_tickers").set_value("aaa, bbb")
    at.button(key="FormSubmitter:cl_form-🚀 Mät").click().run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "JÄMFÖRELSE" in html and "8/10" in html and "⬆" in html and ">NEJ<" in html
    assert "RESULTATHÄVSTÅNG" in html and "KURSBETA" in html and "Valt pris" in html
    assert any("VECKOAVKASTNING" in ch.proto.spec for ch in at.get("plotly_chart"))
    at.slider(key="cl_pct_AAA").set_value(50).run()
    assert not at.exception, at.exception
    assert "+50 %" in " ".join(m.value for m in at.markdown)
