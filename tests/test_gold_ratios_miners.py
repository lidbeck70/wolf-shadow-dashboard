"""
Guldkvoter PR B: bolag per kvotscenario genom Snabbkollens motor, med
råvaran låst till valt par. Kvoterna = parets egna percentiler.
Syntetiskt kopparbolag: EBITDA = 1000·koppar − 2000 (MUSD), egen EV/EBITDA
median 8×, nettoskuld 200, börsvärde 3 000, kurs 20 USD.
"""
import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from gold_ratios import config as rc  # noqa: E402
from gold_ratios import miners as grm  # noqa: E402

COPPER = dict(zip(range(2016, 2026), [2.2, 2.8, 2.9, 2.7, 2.8, 4.2, 4.0, 3.8, 4.1, 4.4]))
FULL = SimpleNamespace(p10=500.0, p25=800.0, median=1000.0, p75=1100.0, p90=1250.0)


def _co(prices=COPPER, slope=1000.0, cut=2000.0, theme="koppar", ticker="HG=F", p0=4.0, **kw):
    eb = [(y, slope * p - cut) for y, p in prices.items()]
    d = {"ticker": "CU", "name": "Copper Co", "source": "Börsdata", "currency": "USD", "price_currency": "USD",
         "revenue_series": [(y, 2.5 * slope * p) for y, p in prices.items()], "ebitda_series": eb,
         "fcf_series": [(y, v - 300) for y, v in eb],
         "commodity_px": {"commodity": theme, "ticker": ticker, "prices": dict(prices), "p0": p0, "fx_now": 1.0},
         "ev_ebitda_hist": [6.0, 7.0, 8.0, 9.0, 10.0], "ev_ebitda": 7.5, "net_debt": 200.0, "nd_ebitda": 0.1,
         "mcap_bd": 3000.0, "price": 20.0, "shares_growth_3y_pct": 3.0}
    d.update(kw)
    return d


def test_only_pairs_on_the_quick_engines_own_series():
    by = rc.PAIR_BY_KEY
    assert grm.minable(by["platina"]) and grm.minable(by["koppar"]) and grm.minable(by["olja"])
    assert grm.minable(by["vete"]) and grm.minable(by["kakao"])
    assert not grm.minable(by["brent"]) and not grm.minable(by["dow"]) and not grm.minable(by["spx"])


def test_price_points_from_own_percentiles():
    pts = grm.price_points(4000.0, 4.0, FULL)
    assert pts == [("BASE", 1000.0, 4.0), ("SVAG (P90)", 1250.0, 3.2), ("MEDIAN", 1000.0, 4.0),
                   ("STARK (P25)", 800.0, 5.0), ("MYCKET STARK (P10)", 500.0, 8.0)]
    assert grm.price_points(4000.0, 4.0, None) == []


def test_the_chain_reuses_the_quick_engine():
    lev, res, rows = grm.miner_scenarios(rc.PAIR_BY_KEY["koppar"], _co(), 4000.0, 4.0, FULL)
    assert res.error is None and lev.commodity == "koppar" and res.multiples["median"] == 8.0
    by = {n: (price, p) for n, _r, price, p in rows}
    price, base = by["BASE"]                     # koppar 4 → EBITDA 2000 × 8 = 16000 − 200 = 15800
    assert price == 4.0 and (base.ebitda, base.ev, base.equity) == (2000, 16000, 15800) and base.ratio == 5.27
    price, strong = by["MYCKET STARK (P10)"]     # 4000 / 500 = 8 → EBITDA 6000
    assert price == 8.0 and strong.ebitda == 6000 and strong.outside_history
    price, weak = by["SVAG (P90)"]               # 4000 / 1250 = 3,2 → EBITDA 1200
    assert price == 3.2 and weak.ebitda == 1200 and weak.price_pct == -20.0


def test_cents_are_converted_for_the_engine():
    """Vete: Yahoo i US-cent, fliken i USD. Motorn får cent, tabellen visar USD."""
    cents = {y: p * 200 for y, p in COPPER.items()}                    # 440–880 c/bu
    d = _co(prices=cents, slope=10.0, cut=3000.0, theme="vete", ticker="ZW=F", p0=550.0)
    lev, res, rows = grm.miner_scenarios(rc.PAIR_BY_KEY["vete"], d, 4000.0, 5.5,
                                         SimpleNamespace(p10=400.0, p25=600.0, median=727.27, p75=800.0, p90=900.0))
    assert res.error is None
    price, base = next((pr, p) for n, _r, pr, p in rows if n == "BASE")
    assert price == 5.5 and base.ebitda == 2500 and base.price_pct == 0.0          # 10 × 550 − 3000


def test_fetch_locks_the_commodity_to_the_pair():
    seen = {}

    def fetcher(t, theme_getter=None):
        seen["theme"] = theme_getter("ANY")
        return {"ticker": t}
    assert grm.fetch_miner("FCX", "koppar", fetcher) == {"ticker": "FCX"} and seen["theme"] == "koppar"


def _page(monkeypatch, miner):
    from streamlit.testing.v1 import AppTest
    from gold_ratios import data as grd
    idx = pd.bdate_range(end="2026-09-30", periods=260 * 12)
    gold = {"series": pd.Series([4000.0] * len(idx), index=idx),
            "point": {"value": 4000.0, "source": "Yahoo Finance GC=F", "date": "2026-09-30", "kind": "ACTUAL"}}
    copper = pd.Series([4000.0 / (500 + 750 * i / (len(idx) - 1)) for i in range(len(idx))], index=idx)
    yahoo = {"HG=F": copper, "BZ=F": pd.Series([80.0] * len(idx), index=idx)}

    def fake_all(getter=None, bd_getter=None):
        return {p["key"]: grd.fetch_pair(p["key"], gold, lambda t, per: yahoo.get(t, pd.Series(dtype=float)),
                                         lambda i: None) for p in rc.PAIRS}
    monkeypatch.setattr(grd, "fetch_all", fake_all)
    monkeypatch.setattr(grm, "fetch_miner", lambda t, theme, fetcher=None: miner(ticker=t))
    monkeypatch.setenv("GR_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["GR_TEST_ROOT"])
        from gold_ratios.ui import render_gold_ratios_page
        render_gold_ratios_page()

    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    return at


def test_page_runs_a_copper_miner(monkeypatch):
    at = _page(monkeypatch, _co)
    at.selectbox(key="gr_pair").set_value("koppar").run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "BOLAG — HÄVSTÅNG PER KVOT" in html and "låst till koppar" in html and "Skriv ett bolag" in html
    at.text_input(key="gr_miner_ticker_koppar").set_value("cu")
    at.button(key="FormSubmitter:gr_miner_form-⛏️ Räkna").click().run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "COMMODITY LEVERAGE (KOPPAR)" in html and "MYCKET STARK (P10)" in html and "3.13×" in html
    assert "Koppar (USD/lb)" in html and "parets egna percentiler" in html


def test_brent_points_to_wti(monkeypatch):
    at = _page(monkeypatch, _co)
    at.selectbox(key="gr_pair").set_value("brent").run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "välj t.ex. Olja (WTI) i stället för Brent" in html
