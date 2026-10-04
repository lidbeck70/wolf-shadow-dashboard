"""
🌩️ Marknadsrisk (PR 1) — riskmodellen: kausala signaler, målet (−10 % inom
63 dagar), poäng och nivåer, kalibreringen mot basfrekvensen, nedgångs-
episoder och larm. Syntetiska serier — inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import market_risk as mr  # noqa: E402

IDX = pd.bdate_range(end="2026-09-30", periods=1500)


def _px(values):
    return pd.Series(np.asarray(values, dtype=float), index=IDX[-len(values):])


def _walk(drift=0.0004, vol=0.01, seed=0):
    g = np.random.default_rng(seed)
    return _px(100 * np.exp(np.cumsum(g.normal(drift, vol, len(IDX)))))


def _df(s):
    return pd.DataFrame({"Close": s})


def _data(index=None, crash=False):
    """Index + alla serier. crash=True: två tydliga nedgångar som föregås av varningar."""
    g = np.random.default_rng(1)
    base = 100 * np.exp(np.cumsum(g.normal(0.0004, 0.006, len(IDX))))
    vix = np.full(len(IDX), 14.0) + g.normal(0, 0.3, len(IDX))
    if crash:
        for start in (700, 1200):
            base[start:start + 40] *= np.linspace(1, 0.82, 40)          # −18 %
            base[start + 40:] *= 0.82
            vix[start - 15:start + 40] = np.linspace(15, 40, 55)         # VIX rusar före och under
    idx = _px(base)
    d = {"SPY": _df(idx), "^VIX": _df(_px(vix)), "^VIX3M": _df(_px(np.full(len(IDX), 18.0))),
         "HYG": _df(_walk(0.0002, 0.004, 2)), "IEF": _df(_walk(0.0001, 0.003, 3)),
         "TLT": _df(_walk(0.0001, 0.004, 4)),
         "T10Y2Y": _px(np.linspace(-0.5, 0.8, len(IDX)))}
    for k, t in enumerate(("XLU", "XLP", "XLK", "XLY") + mr.BREADTH_ETFS):
        d.setdefault(t, _df(_walk(0.0004, 0.01, 10 + k)))
    return idx, d


# ── Målet och episoderna ─────────────────────────────────────────────────────
def test_forward_event_looks_only_at_the_next_63_days():
    c = _px([100.0] * 100 + [89.0] + [100.0] * 99)
    ev = mr.forward_event(c, horizon=63, drawdown=0.10)
    assert ev.iloc[99] == 1.0 and ev.iloc[100 - 63] == 1.0 and ev.iloc[100 - 64] == 0.0
    assert ev.iloc[100] == 0.0                                   # dagen själv räknas inte
    assert ev.iloc[-63:].isna().all() and ev.iloc[-64] == 0.0     # framtiden okänd = NaN


def test_drawdown_episodes():
    c = _px([100, 110, 99, 95, 105, 120, 107, 100, 125])
    eps = mr.drawdown_episodes(c, 0.10)
    assert [(e[3]) for e in eps] == [-13.6, -16.7]               # 110 → 95, 120 → 100
    assert str(eps[0][1].date()) == str(c.index[2].date())         # −10 % nåddes vid 99


# ── Signalerna ───────────────────────────────────────────────────────────────
def test_signals_are_causal():
    idx, d = _data(crash=True)
    act, av = mr.compute_signals(idx, d)
    for i in (400, 800, 1300):
        day = IDX[i]
        cut = {k: (v[v.index <= day] if isinstance(v, (pd.Series, pd.DataFrame)) else v) for k, v in d.items()}
        a2, v2 = mr.compute_signals(idx[idx.index <= day], cut)
        assert act.loc[day].equals(a2.iloc[-1]) and av.loc[day].equals(v2.iloc[-1]), day


def test_each_signal_fires_on_its_condition():
    idx, d = _data(crash=True)
    act, av = mr.compute_signals(idx, d)
    assert set(act.columns) == set(mr.SIGNALS)
    assert act["vix_spike"].iloc[700:740].any() and not act["vix_spike"].iloc[300:600].any()
    assert act["below_200"].iloc[760] and not act["below_200"].iloc[650]
    # T10Y2Y går från −0,5 till +0,8 och korsar noll runt dag 577
    assert act["curve"].iloc[700] and not act["curve"].iloc[100]               # positiv efter inversion
    assert not act["curve"].iloc[-1]                                           # inversionen > två år sedan
    assert not av["vix_inverted"].iloc[0] or av["vix_inverted"].all()
    a_omx, _ = mr.compute_signals(idx, d, breadth=False)
    assert "breadth_div" not in a_omx.columns                                  # OMXS30 saknar bredd


def test_missing_series_are_unavailable_never_active():
    idx, d = _data()
    d.pop("HYG")
    d["T10Y2Y"] = None
    act, av = mr.compute_signals(idx, d)
    assert not av["credit"].any() and not act["credit"].any()
    assert not av["curve"].any() and not act["curve"].any()


def test_levels():
    assert [mr.level_of(p) for p in (0, 1, 2, 3, 4, 7)] == ["LÅG", "LÅG", "FÖRHÖJD", "FÖRHÖJD", "HÖG", "HÖG"]


# ── Kalibreringen ────────────────────────────────────────────────────────────
def test_calibration_against_the_base_rate():
    idx, d = _data(crash=True)
    act, av = mr.compute_signals(idx, d)
    cal = mr.calibrate(idx, act, av)
    assert cal is not None and cal.days > 0 and 0 < cal.base_rate < 100
    assert set(cal.by_level) == {"LÅG", "FÖRHÖJD", "HÖG"}
    assert sum(v["days"] for v in cal.by_level.values()) == cal.days
    assert len(cal.episodes) == 2 and all(e["depth"] <= -10 for e in cal.episodes)
    assert all(e["warned"] for e in cal.episodes)                    # VIX-spiken lyste före båda
    assert cal.alarms["total"] == cal.alarms["hits"] + cal.alarms["false"]


def test_evaluate_with_getters():
    idx, d = _data(crash=True)
    r = mr.evaluate("SPY", getter=lambda t, p: d.get(t), fred_getter=lambda s: d.get(s))
    assert r.error is None and r.date == "2026-09-30" and r.level in ("LÅG", "FÖRHÖJD", "HÖG")
    assert r.points == sum(s["active"] for s in r.signals) and r.possible == 10
    assert r.calibration is not None
    omx = dict(d, **{"^OMX": d["SPY"]})
    r2 = mr.evaluate("OMXS30", getter=lambda t, p: omx.get(t), fred_getter=lambda s: omx.get(s))
    assert r2.possible == 9 and all(s["key"] != "breadth_div" for s in r2.signals)
    gone = mr.evaluate("SPY", getter=lambda t, p: None, fred_getter=lambda s: None)
    assert "DATA UNAVAILABLE" in gone.error


# ── Börsdata (OMXS30) ────────────────────────────────────────────────────────
class _FakeBD:
    """Börsdata: indexet 'OMX Stockholm 30' (marknad 7) och 40 Large Cap-aktier (marknad 1)."""

    def __init__(self, with_index=True, stocks=40):
        self.ins = ([{"insId": 1, "name": "OMX Stockholm 30", "ticker": "OMXS30", "marketId": 7}] if with_index else [])
        self.ins += [{"insId": 100 + k, "name": f"Bolag {k}", "ticker": f"B{k}", "marketId": 1} for k in range(stocks)]
        self.ins += [{"insId": 999, "name": "Småbolag", "ticker": "SM", "marketId": 3}]
        self.calls = []

    def get_instruments(self):
        return self.ins

    def get_stockprices_df(self, ins_id, max_count=0):
        self.calls.append(ins_id)
        s = _walk(0.0004, 0.012, seed=ins_id)
        return pd.DataFrame({"Close": s})


def test_stocks_breadth_pct():
    up, down = _px(np.linspace(100, 200, 300)), _px(np.linspace(200, 100, 300))
    pct = mr.stocks_breadth_pct({"A": up, "B": up, "C": down, "D": down}, min_stocks=4)
    assert pct.iloc[-1] == 50.0 and pct.index[0] == up.index[50]
    assert mr.stocks_breadth_pct({"A": up}, min_stocks=4) is None


def test_omxs30_from_borsdata_with_large_cap_breadth():
    _idx, d = _data(crash=True)
    bd = _FakeBD()
    r = mr.evaluate("OMXS30", getter=lambda t, p: d.get(t), fred_getter=lambda s: d.get(s), bd_api=bd)
    assert r.error is None and r.source.startswith("Börsdata · OMX Stockholm 30")
    assert "40 Large Cap-aktier" in r.breadth_source and r.possible == 10
    assert any(s["key"] == "breadth_div" for s in r.signals)
    assert 999 not in bd.calls                                         # bara marknad 1


def test_omxs30_falls_back_to_yahoo_and_reports_why():
    _idx, d = _data(crash=True)
    omx = dict(d, **{"^OMX": d["SPY"]})
    r = mr.evaluate("OMXS30", getter=lambda t, p: omx.get(t), fred_getter=lambda s: omx.get(s),
                    bd_api=_FakeBD(with_index=False, stocks=5))
    assert r.error is None and "Yahoo Finance ^OMX (reserv" in r.source
    assert "för få aktier" in r.breadth_source and r.possible == 9

def test_duration_stress_fires_on_tlt_shock_while_index_near_high():
    """Ränteshock: TLT −8 %/42 d samtidigt som indexet är inom 5 % från årshögsta."""
    idx, d = _data()                                                   # lugnt index nära sina högsta
    tlt = 100 * np.ones(len(IDX))
    tlt[900:942] = np.linspace(100, 88, 42)                            # −12 % på två månader
    tlt[942:] = 88
    d["TLT"] = _df(_px(tlt))
    act, av = mr.compute_signals(idx, d)
    near = idx >= 0.95 * idx.rolling(252, min_periods=50).max()
    fired = act["duration_stress"]
    assert av["duration_stress"].iloc[-1]
    assert fired.iloc[941] == bool(near.iloc[941])                     # tänder under chocken om nära topp
    assert not fired.iloc[:900].any()                                  # aldrig före chocken
    assert not fired.iloc[1100:].any()                                 # släcks när 42-dagarsfönstret passerat
    d2 = dict(d); d2.pop("TLT")
    a2, v2 = mr.compute_signals(idx, d2)
    assert not a2["duration_stress"].any() and not v2["duration_stress"].any()   # saknad data ≠ aktiv
