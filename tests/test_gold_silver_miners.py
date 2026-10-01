"""
Guld/Silver PR B: silverbolag genom Snabbkollens motor per kvotscenario.
Syntetiskt silverbolag: EBITDA = 100·silver − 1000 (MUSD), egen EV/EBITDA
median 8×, nettoskuld 200, börsvärde 3 000, kurs 20 USD.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from gold_silver import engine as ge  # noqa: E402
from gold_silver import miners as gm  # noqa: E402

SILVER = dict(zip(range(2016, 2026), [17.0, 17.1, 15.7, 16.2, 20.5, 25.1, 21.8, 23.4, 28.3, 40.0]))


def _miner(**kw):
    eb = [(y, 100.0 * p - 1000) for y, p in SILVER.items()]
    d = {"ticker": "SLVR", "name": "Silver Co", "source": "Börsdata", "currency": "USD", "price_currency": "USD",
         "revenue_series": [(y, 250.0 * p) for y, p in SILVER.items()], "ebitda_series": eb,
         "fcf_series": [(y, v - 300) for y, v in eb],
         "commodity_px": {"commodity": "silver", "ticker": "SI=F", "prices": dict(SILVER), "p0": 50.0, "fx_now": 1.0},
         "ev_ebitda_hist": [6.0, 7.0, 8.0, 9.0, 10.0], "ev_ebitda": 7.5, "net_debt": 200.0, "nd_ebitda": 0.1,
         "mcap_bd": 3000.0, "price": 20.0, "shares_growth_3y_pct": 3.0}
    d.update(kw)
    return d


def test_price_points_per_ratio():
    pts = ge.miner_price_points(5000, 50.0)                             # kvot 100
    assert pts[0] == ("BASE", 100.0, 50.0)
    assert [(n, r) for n, r, _ in pts[1:]] == [("RATIO COMPRESSION", 50.0), ("STRONG SILVER", 40.0),
                                              ("SILVER BULL", 30.0), ("REFERENCE SCENARIO", 19.0)]
    assert dict((n, p) for n, _, p in pts)["STRONG SILVER"] == 125.0     # 5000 / 40
    assert ge.miner_price_points(None, 50.0) == []


def test_the_chain_reuses_the_quick_engine():
    lev, res, rows = gm.miner_scenarios(_miner(), 5000, 50.0)
    assert res.error is None and lev.commodity == "silver" and res.multiples["median"] == 8.0
    by = {n: p for n, _r, p in rows}
    base = by["BASE"]                           # silver 50 → EBITDA 4000 × 8 = 32000 − 200 = 31800
    assert (base.ebitda, base.ev, base.equity) == (4000, 32000, 31800) and base.ratio == 10.6
    assert base.price_pct == 0.0 and base.share_price == 212.0 and base.revenue == 12500 and base.fcf == 3700
    comp = by["RATIO COMPRESSION"]              # silver 100 → EBITDA 9000
    assert comp.price == 100.0 and comp.ebitda == 9000 and comp.outside_history
    ref = by["REFERENCE SCENARIO"]              # 5000 / 19 = 263,16
    assert ref.price == 263.16 and ref.ebitda == 25316 and ref.outside_history


def test_miner_without_a_silver_link_is_a_data_gap():
    noisy = [(y, v) for y, v in zip(SILVER, [900, 400, 1200, 50, 3000, 200, 800, 1500, 60, 2500])]
    lev, res, rows = gm.miner_scenarios(_miner(ebitda_series=noisy, fcf_series=[]), 5000, 50.0)
    assert res.error and rows == []


def test_fetch_locks_the_commodity_to_silver():
    seen = {}

    def fetcher(t, theme_getter=None):
        seen["theme"] = theme_getter("ANY")
        return {"ticker": t}
    assert gm.fetch_miner("PAAS", fetcher) == {"ticker": "PAAS"} and seen["theme"] == "silver"


def test_page_runs_a_silver_miner(monkeypatch):
    import pandas as pd
    from streamlit.testing.v1 import AppTest
    from gold_silver import data as gd
    idx = pd.bdate_range(end="2026-09-30", periods=252 * 6)
    fake = {"gold": {"value": 5000.0, "source": "Yahoo Finance GC=F", "date": "2026-09-30", "kind": "ACTUAL"},
            "silver": {"value": 50.0, "source": "Yahoo Finance SI=F", "date": "2026-09-30", "kind": "ACTUAL"},
            "series": pd.Series([100.0] * len(idx), index=idx), "error": None}
    monkeypatch.setattr(gd, "fetch", lambda getter=None: fake)
    monkeypatch.setattr(gm, "fetch_miner", lambda t, fetcher=None: _miner(ticker=t))
    monkeypatch.setenv("GS_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["GS_TEST_ROOT"])
        from gold_silver.ui import render_gold_silver_page
        render_gold_silver_page()

    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "SILVERBOLAG — HÄVSTÅNG PER KVOT" in html and "Skriv ett silverbolag" in html
    at.text_input(key="gs_miner_ticker").set_value("slvr")
    at.button(key="FormSubmitter:gs_miner_form-⛏️ Räkna").click().run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "COMMODITY LEVERAGE (SILVER)" in html and "10.60×" in html and "REFERENCE SCENARIO ⚠" in html
    assert "263.16" in html and "inte ett fair value" in html and "NAV räknas inte automatiskt" in html
