"""
Wolf Asymmetry · Snabbkoll: skriv en ticker, få Survival, Margin of Safety
och Confidence (auto) — helt automatiskt ur Börsdata med Yahoo som reserv.
Inget nätverk: fejkad Börsdata-klient, fejkad Yahoo, syntetisk kurs.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asymmetry import quick, quick_data  # noqa: E402
from asymmetry import quick_config as qc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _good() -> dict:
    return {"ticker": "BOL.ST", "name": "Boliden", "source": "Börsdata",
            "fcf": 5000.0, "cash": 3000.0, "net_debt": 2000.0, "nd_ebitda": 0.4, "current_ratio": 1.8,
            "shares_growth_3y_pct": 0.0, "equity_ratio_pct": 55.0, "ebitda_margin_pct": 42.0,
            "ev_ebitda": 4.0, "ev_ebitda_hist": [6.0, 7.0, 5.5, 6.5, 8.0], "p_fcf": 8.0,
            "p_fcf_hist": [14.0, 12.0, 16.0], "from_52w_high_pct": 45.0, "vs_sma200_pct": -12.0,
            "earnings_stability": 0.8, "fcf_stability": 0.75, "f_score": 7, "report_years": 10,
            "source_gap_pct": 2.0, "coverage": (12, 12)}


def test_strong_company_is_green_on_every_card():
    r = quick.score(_good())
    assert r.verdict == "GRÖN" and r.total == 300
    for g in r.groups:
        assert g.score == 100.0 and g.coverage == "5/5 mätta", g.label
        assert all(p.status == "GREEN" for p in g.pillars), [(p.label, p.status) for p in g.pillars]
    assert r.group("mos").pillars[1].value == "4.0× (-38 %)"
    ev = r.group("mos").pillars[1]
    assert "median 6.5×" in ev.why and "5 år" in ev.why


def test_weak_junior_is_red_with_reasons():
    d = {"ticker": "JUN.V", "source": "Yahoo", "fcf": -40.0, "cash": 20.0, "nd_ebitda": None,
         "net_debt": 5.0, "current_ratio": 0.8, "shares_growth_3y_pct": 120.0, "ebitda_margin_pct": -15.0,
         "fcf_yield_pct": -9.0, "from_52w_high_pct": 10.0, "vs_sma200_pct": 25.0, "coverage": (5, 12)}
    r = quick.score(d)
    s = {p.key: p.status for p in r.group("survival").pillars}
    assert s == {"cashflow": "RED", "debt": "DATA_GAP", "liquidity": "RED", "dilution": "RED", "equity": "DATA_GAP"}
    assert r.group("survival").pillars[0].value == "0.5 års runway"
    assert r.group("mos").pillars[0].status == "RED" and r.group("mos").pillars[2].label == "FCF-yield"
    assert r.verdict in ("RÖD", "DATA_GAP")
    assert any("Röda kort" in x for x in r.reasons)


def test_data_gap_is_never_zero():
    d = _good()
    for k in ("current_ratio", "equity_ratio_pct"):
        d[k] = None
    g = quick.survival(d)
    assert g.coverage == "3/5 mätta" and g.score == 100.0           # luckorna drar inte ner
    empty = quick.score({"ticker": "X", "coverage": (0, 12)})
    assert empty.verdict == "DATA_GAP" and empty.total is None
    assert all(g.score is None for g in empty.groups)


def test_thresholds_and_verdict_mix():
    assert quick._higher(2.0, qc.RUNWAY_YEARS) == "GREEN" and quick._higher(1.5, qc.RUNWAY_YEARS) == "AMBER"
    assert quick._lower(1.0, qc.ND_EBITDA) == "GREEN" and quick._lower(3.5, qc.ND_EBITDA) == "RED"
    assert quick._vs_median(4.0, [6.0, 7.0]) == (None, None, 2)      # för kort historik
    assert quick._stability(85.0) == 0.85                             # procentskala → 0–1
    d = _good()
    d.update(vs_sma200_pct=30.0, from_52w_high_pct=5.0, ev_ebitda=9.0, p_fcf=20.0, ebitda_margin_pct=25.0)
    r = quick.score(d)
    assert r.group("mos").score < qc.VERDICT_RED_BELOW and r.verdict == "RÖD"


# ── datahämtning ─────────────────────────────────────────────────────────────
class _Api:
    is_configured = True

    def resolve_instrument_id(self, q):
        return 7 if q in ("BOL", "BOL.ST") else None

    def get_instruments(self):
        return [{"insId": 7, "ticker": "BOL", "name": "Boliden", "marketId": 1, "reportCurrency": "SEK",
                 "stockPriceCurrency": "SEK"}]

    def get_global_instruments_list(self):
        return []

    def get_fundamentals_snapshot_fast(self, ids, scope="nordic"):
        return {7: {"equity_ratio": 0.55, "ebitda_margin": 0.42, "current_ratio": 1.8, "earnings_stability": 0.8,
                    "fcf_stability": 0.7, "f_score": 7, "ev_ebitda": 4.0, "p_fcf": 8.0, "net_debt_ebitda": 0.4,
                    "market_cap": 100_000.0}}

    def get_reports(self, iid, kind, max_count=10):
        if kind == "r12":
            return [{"freeCashFlow": 5000.0, "cashAndEquivalents": 3000.0, "netDebt": 2000.0}]
        return [{"year": 2020 + i, "freeCashFlow": 1000.0 * i, "numberOfShares": 273.5, "totalAssets": 100.0,
                 "totalEquity": 55.0} for i in range(6)]

    def get_kpi_history(self, iid, kpi, rt, pt):
        base = {11: [6.0, 7.0, 5.5, 6.5, 8.0], 76: [14.0, 12.0, 16.0]}[kpi]
        return [{"y": 2020 + i, "v": v} for i, v in enumerate(base)]


def _closes(n=320, start=100.0, end=70.0):
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.Series([start + (end - start) * i / (n - 1) for i in range(n)], index=idx)


def test_fetch_from_borsdata_fills_every_scored_field():
    d = quick_data.fetch("bol.st", api=_Api(), price_getter=lambda s: _closes(),
                         info_getter=lambda s: {"marketCap": 98_000_000_000})
    assert d["source"] == "Börsdata" and d["name"] == "Boliden" and d["yf_ticker"] == "BOL.ST"
    assert d["equity_ratio_pct"] == 55.0 and d["ebitda_margin_pct"] == 42.0 and d["fcf"] == 5000.0
    assert d["shares_growth_3y_pct"] == 0.0 and d["report_years"] == 6
    assert d["ev_ebitda_hist"] == [6.0, 7.0, 5.5, 6.5, 8.0] and d["fcf_yield_pct"] == 12.5
    assert d["source_gap_pct"] == 2.0 and d["coverage"] == (12, 12) and d["filled_yahoo"] == []
    assert d["from_52w_high_pct"] > 0 and d["vs_sma200_pct"] < 0 and len(d["prices"]) == 320
    r = quick.score(d)
    assert r.group("survival").score == 100.0 and r.group("confidence").coverage == "5/5 mätta"


def test_fetch_yahoo_only_marks_what_it_filled():
    info = {"longName": "Newmont", "freeCashflow": 3e9, "totalCash": 5e9, "totalDebt": 8e9, "ebitda": 9e9,
            "ebitdaMargins": 0.45, "enterpriseToEbitda": 7.0, "currentRatio": 2.1, "marketCap": 6e10,
            "currency": "USD", "financialCurrency": "USD"}
    d = quick_data.fetch("NEM", api=None, use_api_default=False, price_getter=lambda s: _closes(),
                         info_getter=lambda s: info)
    assert d["source"] == "Yahoo" and d["name"] == "Newmont"
    assert set(d["filled_yahoo"]) >= {"fcf", "current_ratio", "ebitda_margin_pct", "ev_ebitda", "nd_ebitda"}
    assert d["nd_ebitda"] == round(3e9 / 9e9, 4) and d["fcf_yield_pct"] == 5.0
    assert d["coverage"][0] < 12                                      # stabilitet, F-score m.m. saknas
    r = quick.score(d)
    assert r.group("confidence").pillars[1].status == "DATA_GAP"


class _ApiNoScreener(_Api):
    """Screenern ger inga stabilitets-/F-score-värden (som i appen för BOL och AEM)."""

    def get_fundamentals_snapshot_fast(self, ids, scope="nordic"):
        snap = super().get_fundamentals_snapshot_fast(ids, scope)
        for k in ("earnings_stability", "fcf_stability", "f_score"):
            snap[7].pop(k)
        return snap

    def get_kpi_history(self, iid, kpi, rt, pt):
        if kpi == 167:                                   # F-score bara som r12
            return [] if rt == "year" else [{"y": 2025, "p": 3, "v": 6.0}, {"y": 2025, "p": 4, "v": 8.0}]
        if kpi == 174:
            return [{"y": 2024, "v": 0.6}, {"y": 2025, "v": 0.9}, {"y": 2023, "v": 0.1}]
        if kpi == 179:
            raise RuntimeError("400")
        return super().get_kpi_history(iid, kpi, rt, pt)


def test_fetch_falls_back_to_kpi_history_then_reports():
    d = quick_data.fetch("BOL.ST", api=_ApiNoScreener(), price_getter=lambda s: _closes(),
                         info_getter=lambda s: {})
    assert d["earnings_stability"] == 0.9 and d["f_score"] == 8.0
    assert d["kpi_source"]["earnings_stability"] == "Börsdata historik (år)"
    assert d["kpi_source"]["f_score"] == "Börsdata historik (r12)"
    # FCF 0, 1000 … 5000 är en rak linje → stabilitet 1.0 ur årsrapporterna
    assert d["fcf_stability"] == 1.0 and d["kpi_source"]["fcf_stability"] == "beräknad ur årsrapporterna"
    conf = quick.score(d).group("confidence")
    assert conf.coverage == "5/5 mätta"
    assert "beräknad ur årsrapporterna" in next(p for p in conf.pillars if p.key == "fcf_stability").why


def test_trend_stability():
    assert quick_data.trend_stability([1, 2, 3]) is None                   # för få år
    assert quick_data.trend_stability([1, 2, 3, 4, 5]) == 1.0
    assert quick_data.trend_stability([5, 4, 3, 2, 1]) == 0.0              # krympande
    assert quick_data.trend_stability([3, 3, 3, 3, 3]) == 1.0
    assert 0.0 < quick_data.trend_stability([1, 5, 2, 6, 3, 8]) < 0.7
    assert quick_data.trend_stability([1, None, 2, 3, 4, 5]) == 1.0


def test_fetch_unknown_ticker():
    d = quick_data.fetch("NOPE", api=None, use_api_default=False, price_getter=lambda s: None,
                         info_getter=lambda s: {})
    assert d["source"] == "—" and d["prices"] == [] and quick.score(d).verdict == "DATA_GAP"


# ── fliken ───────────────────────────────────────────────────────────────────
def test_quick_tab_renders_card_gauges_cards_and_charts(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import storage
    from confidence import store as cs

    good = _good()
    good.update(prices=[(str(d)[:10], v) for d, v in _closes().items()],
                sma200=[(str(d)[:10], None) for d in _closes().index],
                ev_ebitda_series=[(2020 + i, v) for i, v in enumerate(good["ev_ebitda_hist"])],
                shares_series=[(2020 + i, 273.5) for i in range(6)],
                fcf_series=[(2020 + i, 1000.0 * i - 1500) for i in range(6)], yf_ticker="BOL.ST",
                fetched="2026-09-29 18:00 UTC", filled_yahoo=["current_ratio"])
    calls = []
    monkeypatch.setattr(quick_data, "fetch", lambda t, **kw: calls.append(t) or dict(good, ticker=t))
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
    assert not at.exception, at.exception
    assert at.radio(key="asym_mode").value == "⚡ Snabbkoll"
    assert any("Skriv en ticker" in c.value for c in at.caption) and not calls
    at.text_input(key="asym_quick_ticker").set_value("bol.st")
    at.button(key="FormSubmitter:asym_quick_form-🔍 Analysera").click().run()
    assert not at.exception, at.exception
    assert calls == ["BOL.ST"]
    html = " ".join(m.value for m in at.markdown)
    assert "WOLF ASYMMETRY SCORE" in html and ">300<" in html and "GRÖN · STARK ASYMMETRI" in html
    assert "15/15 kort mätta" in html
    charts = at.get("plotly_chart")
    assert len(charts) == 3 + 4                                        # tre mätare + fyra grafer
    assert "Prisbuffert" in html and "Piotroski F-score" in html and "Utspädning" in html
    assert any("Ur Yahoo" in c.value for c in at.caption)
    # cachat: ny körning hämtar inte igen
    at.run()
    assert calls == ["BOL.ST"]
    # lägg till i arket
    at.button(key="asym_quick_add_BOL.ST").click().run()
    assert not at.exception, at.exception
    assert "BOL.ST" in at.session_state["confidence"]["companies"]
    assert at.session_state["confidence"]["companies"]["BOL.ST"]["name"] == "Boliden"
