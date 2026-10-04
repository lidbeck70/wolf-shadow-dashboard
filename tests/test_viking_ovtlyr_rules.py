"""
Viking Nine-backtestet: OVTLYR Golden Ticket-reglerna (aktier, inga optioner)
som valbara test. Av som förval — live-reglerna ändras inte. Syntetiska
kurser, inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import viking_backtest as vb  # noqa: E402
import viking_portfolio as vp  # noqa: E402
from tests.test_viking_backtest import DATA, IDX, ROOT  # noqa: E402

TICKERS = [f"S{i}" for i in range(4)]


def _run(**cfg):
    base = dict(min_nine=8, years=3, require_volume=False)
    base.update(cfg)
    return vb.run(TICKERS, getter=lambda t, p: DATA.get(t), sector_getter=lambda t: "Technology",
                  cfg=vb.Config(**base), today=pd.Timestamp("2026-09-30"))


@pytest.fixture(scope="module")
def live():
    return _run()


def _atr_at(t):
    x = vb.execution_frame(DATA[t.ticker])
    return float(x["atr"].loc[pd.Timestamp(t.signal_date)])


# ── Förval: av ──────────────────────────────────────────────────────────────
def test_all_rules_are_off_by_default():
    assert vb.Config().ovtlyr_rules() == []
    pc = vp.PortfolioConfig()
    assert not pc.one_per_sector and not pc.history_first
    assert set(vb.OVTLYR_RULES) == {"ovt_stop", "atr_step", "ovt_breadth", "fg_turn", "liquidity", "history"}


def test_live_run_has_no_ovtlyr_exit_reasons(live):
    reasons = {t.exit_reason for t in live["trades"]}
    assert live["trades"] and not reasons & {"½ ATR-stopp", "nödstopp 2 ATR", "ATR-steg", "F&G vänder"}
    assert live["illiquid"] == 0 and live["neg_history"] == 0


# ── ½ ATR-stopp på stängning, risk på 2 × ATR ───────────────────────────────
def test_half_atr_close_stop_with_risk_on_two_atr(live):
    res = _run(ovt_stop=True)
    assert [t.signal_date for t in res["trades"]][:3] == [t.signal_date for t in live["trades"]][:3]
    hit = 0
    for t in res["trades"]:
        atr = _atr_at(t)
        assert t.risk == pytest.approx(vb.OVT_RISK_ATR * atr, rel=1e-3)
        if t.exit_reason == "½ ATR-stopp":
            rule_day = IDX.get_loc(pd.Timestamp(t.exit_date)) - 1           # stängning → exit nästa öppning
            assert DATA[t.ticker]["Close"].iloc[rule_day] < t.entry - vb.OVT_CLOSE_STOP_ATR * atr
            hit += 1
    assert hit > 0
    # Förlusterna i R blir mindre: snittförloraren är långt från −1R
    assert res["metrics"]["avg_loser"] > live["metrics"]["avg_loser"]


# ── ATR-stegtrailing ────────────────────────────────────────────────────────
def test_atr_step_trailing_locks_half_atr_below_each_step():
    res = _run(atr_step=True)
    steps = [t for t in res["trades"] if t.exit_reason == "ATR-steg"]
    assert steps
    for t in steps:
        atr, d = _atr_at(t), DATA[t.ticker]
        e, x = IDX.get_loc(pd.Timestamp(t.entry_date)), IDX.get_loc(pd.Timestamp(t.exit_date))
        k = int(np.floor((d["High"].iloc[e:x].max() - t.entry) / atr))   # stegen från dagarna FÖRE exitdagen
        assert k >= 1
        level = t.entry + (k - vb.ATR_STEP_GIVEBACK) * atr
        assert t.exit == pytest.approx(min(level, d["Open"].iloc[x]), rel=1e-6)
        assert t.r > 0 or d["Open"].iloc[x] < t.entry                 # förlust bara om dagen gappar under


# ── OVTLYR:s breddregler ────────────────────────────────────────────────────
def _b(values):
    return pd.Series(values, index=pd.bdate_range("2026-01-05", periods=len(values)), dtype=float)


def test_ovtlyr_breadth_rules():
    rising = vb.ovtlyr_breadth_ok(_b(np.linspace(30, 60, 40)))
    assert rising.iloc[-1]                                           # 25–75 och över EMA10
    falling = vb.ovtlyr_breadth_ok(_b(np.linspace(60, 30, 40)))
    assert not falling.iloc[-1]                                      # under EMA10 = krymper
    low_turn = vb.ovtlyr_breadth_ok(_b([40] * 20 + list(np.linspace(40, 10, 15)) + [14, 19]))
    assert low_turn.iloc[-1]                                         # under 25, vänder upp över EMA10
    low_flat = vb.ovtlyr_breadth_ok(_b([40] * 20 + list(np.linspace(40, 10, 15)) + [14, 19, 19]))
    assert not low_flat.iloc[-1]                                     # under 25 utan uppvändning
    high_turn = vb.ovtlyr_breadth_ok(_b(list(np.linspace(50, 90, 30)) + [89]))
    assert not high_turn.iloc[-1]                                    # över 75 och nedvänd
    assert vb.ovtlyr_breadth_ok(_b(list(np.linspace(50, 90, 30)) + [91])).iloc[-1]


def test_ovtlyr_breadth_is_causal():
    b = _b(50 + 30 * np.sin(np.arange(200) / 9))
    full = vb.ovtlyr_breadth_ok(b)
    for i in (40, 90, 150):
        assert full.iloc[i] == vb.ovtlyr_breadth_ok(b.iloc[:i + 1]).iloc[-1]


def test_ovtlyr_breadth_changes_the_market_factor():
    stock, spy, sec = DATA["S0"], DATA["SPY"], DATA["XLK"]
    import ovtlyr_nine as on
    breadth = on.breadth_series({t: DATA[t]["Close"] for t in on.SECTOR_ETFS.values()})
    a = vb.factor_frame(stock, spy, sec, breadth)["market.breadth"]
    b = vb.factor_frame(stock, spy, sec, breadth, ovt_breadth=True)["market.breadth"]
    assert not a.equals(b)


# ── F&G vänder ──────────────────────────────────────────────────────────────
def test_fg_turn_exit_follows_a_falling_fear_greed():
    res = _run(fg_turn=True)
    turns = [t for t in res["trades"] if t.exit_reason == "F&G vänder"]
    assert turns
    import ovtlyr_nine as on
    for t in turns:
        fg = on.fear_greed_series(DATA[t.ticker])
        j = IDX.get_loc(pd.Timestamp(t.exit_date)) - 1
        assert fg.iloc[j] < fg.iloc[j - on.FG_LOOKBACK]


# ── Likviditet ──────────────────────────────────────────────────────────────
def test_liquidity_filter_per_market():
    idx = pd.bdate_range("2026-01-05", periods=40)
    df = lambda c, v: pd.DataFrame({"Close": c, "Volume": v}, index=idx)  # noqa: E731
    assert vb.liquid_series(df(30.0, 2e6), "NVDA").iloc[-1]
    assert not vb.liquid_series(df(15.0, 2e6), "AMC").iloc[-1]          # under 20 $
    assert not vb.liquid_series(df(30.0, 5e5), "KO").iloc[-1]           # under 1 milj aktier
    assert vb.liquid_series(df(100.0, 2e5), "VOLV-B.ST").iloc[-1]       # 20 MSEK om dagen
    assert not vb.liquid_series(df(5.0, 1e5), "ELTEL.ST").iloc[-1]      # 0,5 MSEK
    assert not vb.liquid_series(df(5.0, 1e5), "NOKIA.HE").iloc[-1]      # 0,5 milj EUR
    assert not vb.liquid_series(df(30.0, 2e6), "NVDA").iloc[:vb.LIQ_DAYS - 1].any()   # för kort historik


def test_liquidity_filter_counts_blocked_signals():
    res = _run(liquidity=True)                                   # syntetisk volym ≈ 1 milj → en del spärras
    assert res["illiquid"] > 0
    assert len(res["trades"]) < len(_run()["trades"])


# ── Positiv egen historik ───────────────────────────────────────────────────
def _tr(sig, exit_, r):
    d = [str(x.date()) for x in pd.bdate_range("2026-01-05", periods=60)]
    return vb.Trade("X", d[sig], d[sig + 1], 100.0, 95.0, 5.0, 9, exit_date=d[exit_], exit=100 + 5 * r, r=r)


def test_history_is_walk_forward():
    trades = [_tr(0, 5, -1.0), _tr(4, 8, 2.0), _tr(10, 12, 1.0), _tr(20, 25, -3.0), _tr(30, 33, 1.0)]
    res = vb.apply_history({"trades": list(trades)})
    hist = [t.hist_r for t in res["trades"]]
    assert hist == [None, None, 1.0, 2.0, -1.0]                  # bara affärer stängda FÖRE signalen
    res = vb.apply_history({"trades": list(trades)}, start="2026-01-05", require_positive=True)
    assert [t.hist_r for t in res["trades"]] == [None, None, 1.0, 2.0] and res["neg_history"] == 1


def test_history_drops_warmup_trades_from_the_result():
    trades = [_tr(0, 5, 2.0), _tr(10, 12, 1.0)]
    res = vb.apply_history({"trades": trades}, start="2026-01-12", require_positive=True)
    assert len(res["trades"]) == 1 and res["trades"][0].hist_r == 2.0


def test_history_run_keeps_the_period_and_sets_hist_r(live):
    res = _run(history=True)
    start = str((pd.Timestamp("2026-09-30") - pd.DateOffset(years=3)).date())
    assert all(t.signal_date >= start for t in res["trades"])
    assert all(t.hist_r is None or t.hist_r >= 0 for t in res["trades"])
    assert any(t.hist_r is not None for t in live["trades"])     # hist_r sätts alltid (prioritet i portföljen)
    assert all(t.sector == "XLK" for t in live["trades"])


# ── Portföljen: en per sektor, bäst historik först ──────────────────────────
D = [str(d.date()) for d in pd.bdate_range("2026-01-05", periods=40)]


def _pt(tk, sig, exit_, sector="XLK", hist=None, nine=9, r=1.0):
    t = vb.Trade(tk, D[sig], D[sig + 1], 100.0, 90.0, 10.0, nine, exit_date=D[exit_], exit=100 + 10 * r, r=r,
                 mom63=0.1, sector=sector, hist_r=hist)
    return t


def test_one_position_per_sector():
    trades = [_pt("A", 0, 10), _pt("B", 2, 12), _pt("C", 2, 12, sector="XLE"), _pt("D", 12, 15)]
    p = vp.simulate(trades, pc=vp.PortfolioConfig(one_per_sector=True))
    assert [r["trade"].ticker for r in p["rows"]] == ["A", "C", "D"] and p["skipped_sector"] == 1
    assert vp.simulate(trades)["skipped_sector"] == 0
    assert "en aktie per sektor" in p["note"]


def test_history_first_priority():
    trades = [_pt(f"N{k}", 0, 10, hist=-2.0, nine=9) for k in range(10)] + [_pt("BEST", 0, 10, hist=5.0, nine=8)]
    live = vp.simulate(trades)
    best = vp.simulate(trades, pc=vp.PortfolioConfig(history_first=True))
    assert "BEST" not in [r["trade"].ticker for r in live["rows"]]          # Nine 9 först, kontot fullt
    assert best["rows"][0]["trade"].ticker == "BEST"


# ── Sidan ───────────────────────────────────────────────────────────────────
def test_variants_compare_each_rule_against_live():
    from ovtlyr.ui.viking_nine_backtest import ALL_OVT, LIVE, OVT_ALL, ovtlyr_variants
    v = ovtlyr_variants()
    assert v[0] == (LIVE, []) and v[-1][0] == ALL_OVT and len(v) == len(OVT_ALL) + 2
    v = ovtlyr_variants(["ovt_stop", "one_per_sector"])
    assert [n for n, _k in v] == [LIVE, OVT_ALL["ovt_stop"][0], OVT_ALL["one_per_sector"][0], ALL_OVT]
    assert ovtlyr_variants(["fg_turn"]) == [(LIVE, []), (OVT_ALL["fg_turn"][0], ["fg_turn"])]


def test_page_shows_ovtlyr_controls_and_line(monkeypatch):
    from streamlit.testing.v1 import AppTest
    res = _run(ovt_stop=True, liquidity=True)
    res["labels"] = {"ovtlyr": ["ovt_stop", "liquidity", "one_per_sector"]}
    res["portfolio"] = vp.simulate(res["trades"], years=3, pc=vp.PortfolioConfig(one_per_sector=True))
    monkeypatch.setenv("VNB_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["VNB_TEST_ROOT"])
        from ovtlyr.ui.viking_nine_backtest import render_viking_nine_backtest
        render_viking_nine_backtest()

    at = AppTest.from_function(app, default_timeout=90)
    at.session_state["vnb_result"] = {"selected": "X", "runs": {"X": res, "Som live": _run()}}
    at.run()
    assert not at.exception, at.exception
    labels = [c.label for c in at.checkbox]
    assert "½ ATR-stopp på stängning" in labels and "Jämför OVTLYR-reglerna en och en" in labels
    html = " ".join(m.value for m in at.markdown)
    assert "OVTLYR GOLDEN TICKET" in html and "OVTLYR: <b" in html and "signaler för illikvida" in html
    assert "Portfölj CAGR %" in html
