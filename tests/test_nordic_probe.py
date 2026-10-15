"""
Norden-proben — samvariation, vem som leder, SMA200-korsningar, sektorernas tidighet och grindvarianterna på
syntetiska serier med känt facit (inget nätverk), körskriptet med fejkade motorer, arbetsflödet och
NORDEN-PROBEN i ⚡ VALHALL › Backtester.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import valhall_nordic_probe as vnp  # noqa: E402
import viking_backtest as vbt  # noqa: E402
from valhall import cycle  # noqa: E402
from valhall import nordic_probe as np_  # noqa: E402
from valhall import ui as vui  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 1600                                             # ≈ 6,3 år handelsdagar: alla fyra fönstren finns
END = pd.Timestamp("2026-10-14")


def _returns(seed, n=N, drift=0.0003, vol=0.01):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(drift, vol, n), index=pd.bdate_range(end=END, periods=n))


def _price(r):
    return 100 * np.exp(r.fillna(0.0).cumsum())


def _sine(period=500, n=N, phase=0.0, amp=10.0):
    t = np.arange(n)
    return pd.Series(100 + amp * np.sin(2 * np.pi * (t + phase) / period), index=pd.bdate_range(end=END, periods=n))


def _trade(r, ticker="ABB.ST", signal="2015-06-01", entry="2015-06-02", exit_="2015-07-01"):
    return vbt.Trade(ticker, signal, entry, 100.0, 95.0, 5.0, 9, exit_date=exit_, exit=100 + 5 * r, r=r)


# ── Samvariation och ledning ───────────────────────────────────────────────
def test_comovement_windows_and_near_identical_series():
    ra = _returns(1)
    a, b = _price(ra), _price(ra + _returns(2, vol=0.002))
    rows = np_.comovement(a, b)
    assert [r["window"] for r in rows] == ["1 år", "3 år", "5 år", "allt"]
    assert [r["days"] for r in rows] == [252, 756, 1260, N]
    for r in rows:
        assert r["corr_daily"] > 0.95 and r["corr_weekly"] > 0.95
        assert r["same_sma200_pct"] > 85 and r["same_direction_pct"] > 85
    assert np_.comovement(a, None) == [] and np_.comovement(a.head(50), b.head(50)) == []   # för lite överlapp


def test_lead_lag_finds_a_one_week_lead():
    ra = _returns(3, vol=0.012)
    a, b = _price(ra), _price(ra.shift(5))                      # b gör a:s vecka en vecka senare
    ll = np_.lead_lag(a, b, names=("Norden", "OMXS30"))
    assert ll["best_lag"] == 1 and ll["best_corr"] > 0.9 and ll["text"].startswith("Norden leder OMXS30 med 1 ")
    assert [r["lag"] for r in ll["by_lag"]] == list(range(-np_.MAX_LAG_WEEKS, np_.MAX_LAG_WEEKS + 1))
    rev = np_.lead_lag(b, a, names=("OMXS30", "Norden"))
    assert rev["best_lag"] == -1 and rev["text"].startswith("Norden leder OMXS30 med 1 ")
    same = np_.lead_lag(a, a)
    assert same["best_lag"] == 0 and same["text"].startswith("samtidigt")
    assert np_.lead_lag(a, None)["text"] == "för lite data"


# ── SMA200-korsningar ──────────────────────────────────────────────────────
def test_sma_crossings_alternate():
    x = np_.sma_crossings(_sine())
    assert len(x) >= 4
    assert all(x[i][1] != x[i + 1][1] for i in range(len(x) - 1)) and {k for _d, k in x} == {"upp", "ned"}
    assert all(x[i][0] < x[i + 1][0] for i in range(len(x) - 1))
    assert np_.sma_crossings(_sine(n=150)) == []                                        # kortare än SMA200


def test_crossing_lead_measures_days_before_market():
    a = _sine()
    b = a.shift(5).dropna()                                     # marknaden korsar fem handelsdagar (sju dagar) efter
    cl = np_.crossing_lead(a, b)
    assert cl["n"] >= 2 and cl["matched"] == cl["n"] and cl["earlier"] == cl["n"] and cl["later"] == 0
    assert cl["median_days"] == -7 and cl["mean_days"] == -7
    assert all(p["days"] == -7 and p["a"] < p["b"] for p in cl["pairs"])
    assert np_.crossing_lead(a, a)["same_day"] == np_.crossing_lead(a, a)["n"]
    opposite = np_.crossing_lead(a, _sine(phase=250))                 # motfas: inga uppkorsningar inom ±60 dagar
    assert opposite["n"] >= 2 and opposite["matched"] == 0 and opposite["median_days"] is None


# ── Sektorernas tidighet ───────────────────────────────────────────────────
def _status(idx, turns):
    """NODATA · RÖD · GRÖN från turns[0] · RÖD · NODATA · GRÖN från turns[1] — två gröna vändningar."""
    s = pd.Series(cycle.NODATA, index=idx, dtype=object)
    s.iloc[300:turns[0]] = cycle.ROD
    s.iloc[turns[0]:turns[0] + 200] = cycle.GRON
    s.iloc[turns[0] + 200:turns[1] - 40] = cycle.ROD
    s.iloc[turns[1] - 40:turns[1]] = cycle.NODATA
    s.iloc[turns[1]:] = cycle.GRON
    return s


def test_green_turns_and_earliness_against_market_gate():
    market = _sine()
    gates = [d for d, k in np_.sma_crossings(market) if k == "upp"]
    assert len(gates) >= 2
    idx = market.index
    i0, i1 = idx.get_loc(gates[0]) - 10, idx.get_loc(gates[1]) + 3           # tio dagar före · tre dagar efter
    status = _status(idx, (i0, i1))
    turns = np_.green_turns(status)
    assert turns == [idx[i0], idx[i1]]                          # GRÖN efter NODATA i början räknas inte
    seg_close = _price(_returns(7))
    e = np_.earliness(seg_close, status, market)
    s = e["summary"]
    assert s["turns"] == 2 and s["near_gate"] == 2 and s["earlier"] == 1
    assert e["rows"][0]["days_vs_gate"] == -14 and e["rows"][1]["days_vs_gate"] > 0
    r63 = (float(seg_close.iloc[i0 + 63]) / float(seg_close.iloc[i0]) - 1) * 100
    assert e["rows"][0]["ret_63"] == pytest.approx(round(r63, 1))
    assert s["avg_ret_63"] is not None and s["hit_63"] in (0.0, 50.0, 100.0)
    assert e["rows"][1]["ret_126"] is None if i1 + 126 >= N else e["rows"][1]["ret_126"] is not None
    assert np_.earliness(seg_close, status, None)["summary"]["near_gate"] == 0


# ── Grindvarianterna ───────────────────────────────────────────────────────
def test_gate_variants_runs_each_provider_and_survives_errors():
    def run_fn(p):
        src = p()["source"]
        if src == "trasig":
            raise RuntimeError("nät borta")
        return {"trades": [_trade(2.0), _trade(1.0), _trade(-1.0)] if src == "OMXS30" else [_trade(3.0), _trade(2.0)]}

    rows = np_.gate_variants({"OMXS30": np_.provider(_sine(), None, "OMXS30"),
                              "Norden": np_.provider(_sine(), None, "Norden"),
                              "Trasig": np_.provider(_sine(), None, "trasig")}, run_fn, vbt.metrics)
    by = {r["variant"]: r for r in rows}
    assert by["OMXS30"]["trades"] == 3 and by["OMXS30"]["win_rate"] == pytest.approx(66.7)
    assert by["OMXS30"]["total_r"] == 2.0 and by["OMXS30"]["profit_factor"] == 3.0 and by["OMXS30"]["error"] is None
    assert by["Norden"]["trades"] == 2 and by["Norden"]["profit_factor"] is None          # inga förluster: inf → None
    assert by["Trasig"]["trades"] == 0 and "RuntimeError: nät borta" in by["Trasig"]["error"]
    assert by["Trasig"]["source"] == "trasig"


def test_provider_and_nordic_breadth():
    tz = _sine().tz_localize("Europe/Stockholm")
    p = np_.provider(tz, None, "Norden", "Nordens bredd")()
    assert p["close"].index.tz is None and p["source"] == "Norden" and p["breadth_source"] == "Nordens bredd"
    assert np_.provider(_sine(), None, "X")()["breadth_source"] == "X"
    closes = {f"T{i}.ST": _price(_returns(i)).iloc[i * 10:] for i in range(35)}      # olika noteringsår
    b = np_.nordic_breadth(closes)
    assert b is not None and 0 <= b.min() and b.max() <= 100 and len(b) > 1000
    assert np_.nordic_breadth({k: closes[k] for k in list(closes)[:5]}) is None     # för få aktier


def test_summary_lines_are_plain_facts():
    early = {"turns": 4, "near_gate": 3, "earlier": 2, "median_days": -9.0, "avg_ret_63": 5.1, "hit_63": 75.0,
             "avg_ret_126": 9.0}
    payload = {"comovement": {"Norden mot OMXS30": [{"window": "allt", "corr_daily": 0.9, "corr_weekly": 0.92,
                                                    "same_sma200_pct": 88.0, "same_direction_pct": 80.0}]},
               "lead_lag": {"Norden mot OMXS30": {"text": "Norden leder OMXS30 med 1 vecka/veckor (0.50 mot 0.40)"}},
               "crossing_lead": {"Norden mot OMXS30": {"n": 5, "earlier": 3, "later": 1, "same_day": 1,
                                                       "median_days": -4.0}},
               "earliness": {"OMXS30": {"Industri": {"summary": early}}},
               "gates": {"Viking Nine": [{"variant": "OMXS30", "trades": 120, "win_rate": 45.0, "expectancy": 0.21,
                                          "profit_factor": 1.6, "max_drawdown_r": 9.5, "total_r": 25.0},
                                         {"variant": "Norden", "trades": 0, "error": "RuntimeError: x"}]}}
    lines = np_.summary(payload)
    assert any(line.startswith("Norden mot OMXS30: korrelation 0.9") for line in lines)
    assert any("Norden leder OMXS30" in line for line in lines)
    assert any("5 uppkorsningar" in line and "före 3" in line for line in lines)
    assert any(line.startswith("Industri mot OMXS30: 4 gröna vändningar, 2 av 3") for line in lines)
    assert any("Viking Nine · OMXS30: 120 affärer" in line for line in lines)
    assert any("Viking Nine · Norden: FEL RuntimeError" in line for line in lines)
    assert np_.summary({}) == []


# ── Körskriptet ────────────────────────────────────────────────────────────
def _fake_frames(api, getter, data_getter, keep=None):
    market = _sine()
    seg = _price(_returns(11))
    frame = pd.DataFrame({"close": seg, "status": _status(seg.index, (700, 1200))})
    if keep is not None:
        keep.update({f"S{i}.ST": pd.DataFrame({"Close": _price(_returns(i)), "Volume": 1e6}) for i in range(35)})
    return {"nordic": {"industrials": frame, "tom": pd.DataFrame()}, "global": {}, "themes": {}, "universe": {},
            "bench": {"nordic": market, "global": None}}


def _fake_series(api, getter, fr, keep):
    omxs30 = {"close": _sine(phase=5), "breadth": pd.Series(55.0, index=_sine().index),
              "source": "Börsdata · OMX Stockholm 30", "breadth_source": "Large Cap"}
    return vnp.series(api, getter, fr, keep, nordic_market=lambda bars: omxs30,
                      index_fn=lambda api, hints, bars: (_sine(phase=2), "Börsdata · OMX Stockholm PI"))


def _fake_engine(tickers, getter=None, cfg=None, today=None, nordic_provider=None, **kw):
    src = nordic_provider()["source"]
    rs = [2.0, -1.0, 1.0] if src.startswith("Börsdata") else [3.0, 1.0, -1.0, 2.0]
    return {"trades": [_trade(r) for r in rs], "per_ticker": []}


def test_runner_probes_series_segments_and_gates():
    saved = {}
    out = vnp.run(api=object(), getter=lambda t, p: None, data_getter=lambda ts, p: {}, today="2026-10-14",
                  frames_fn=_fake_frames, series_fn=_fake_series, viking_run=_fake_engine, berserk_run=_fake_engine,
                  save=lambda name, data: saved.setdefault(name, data) is not None)
    assert set(out["sources"]) == {vnp.OMXS30, vnp.OMXSPI, vnp.NORDEN, vnp.OMXN40}
    assert out["sources"][vnp.OMXN40]["bars"] == 0 and "saknas" in out["sources"][vnp.OMXN40]["source"]
    assert out["sources"][vnp.NORDEN]["bars"] == N and "35 aktier" in out["sources"][vnp.NORDEN]["source"]
    assert set(out["comovement"]) == {f"{vnp.NORDEN} mot {vnp.OMXS30}", f"{vnp.NORDEN} mot {vnp.OMXSPI}",
                                      f"{vnp.OMXSPI} mot {vnp.OMXS30}"}                 # OMXN40-paren hoppas över
    assert set(out["lead_lag"]) == set(out["crossing_lead"]) == set(out["comovement"])
    assert set(out["earliness"]) == {vnp.OMXS30, vnp.NORDEN} and list(out["earliness"][vnp.OMXS30]) == ["Industri"]
    assert out["earliness"][vnp.NORDEN]["Industri"]["summary"]["turns"] == 2
    vik = {r["variant"]: r for r in out["gates"]["Viking Nine"]}
    assert set(vik) == {vnp.OMXS30, vnp.OMXSPI, vnp.NORDEN_LC, vnp.NORDEN}
    assert {r["variant"] for r in out["gates"]["BERSERK"]} == {vnp.OMXS30, vnp.OMXSPI, vnp.NORDEN}
    assert vik[vnp.OMXS30]["trades"] == 3 and vik[vnp.NORDEN]["trades"] == 4 and vik[vnp.NORDEN]["error"] is None
    assert vik[vnp.NORDEN]["source"].startswith("Norden likaviktad")
    assert out["summary"] and out["notes"] == list(vnp.NOTES) and out["start_year"] == 2008
    assert out["saved"] is True and list(saved) == [vnp.BLOB]
    quick = vnp.run(api=object(), getter=lambda t, p: None, data_getter=lambda ts, p: {}, today="2026-10-14",
                    frames_fn=_fake_frames, series_fn=_fake_series, no_save=True, no_gates=True)
    assert "gates" not in quick and "saved" not in quick and quick["comovement"]


def test_probe_workflow():
    src = open(os.path.join(ROOT, ".github", "workflows", "nordic-probe.yml"), encoding="utf-8").read()
    assert "workflow_dispatch" in src and "python valhall_nordic_probe.py" in src
    assert "BORSDATA_API_KEY" in src and "GIST_TOKEN" in src and "--no-gates" in src


# ── UI: NORDEN-PROBEN i Backtester ─────────────────────────────────────────
PAYLOAD = {"generated": "2026-10-15T07:00:00+00:00", "start_year": 2008,
           "sources": {"OMXS30": {"source": "Börsdata · OMX Stockholm 30", "breadth_source": "Large Cap", "bars": 5000,
                                  "from": "2006-10-16", "to": "2026-10-14"}},
           "comovement": {"Norden (likaviktad) mot OMXS30": [{"window": "allt", "days": 4500, "corr_daily": 0.83,
                                                              "corr_weekly": 0.86, "same_sma200_pct": 84.2,
                                                              "same_direction_pct": 78.0}]},
           "lead_lag": {"Norden (likaviktad) mot OMXS30": {"best_lag": 1, "best_corr": 0.5, "lag0_corr": 0.4,
                                                           "by_lag": [], "text": "Norden (likaviktad) leder OMXS30 "
                                                                                 "med 1 vecka/veckor"}},
           "crossing_lead": {"Norden (likaviktad) mot OMXS30": {"n": 12, "matched": 10, "earlier": 7, "later": 2,
                                                                "same_day": 1, "median_days": -6.0,
                                                                "mean_days": -4.1}},
           "earliness": {"OMXS30": {"Industri": {"summary": {"turns": 9, "near_gate": 6, "earlier": 4,
                                                             "median_days": -8.0, "avg_ret_63": 6.2, "hit_63": 67.0,
                                                             "avg_ret_126": 11.0, "hit_126": 78.0}},
                                    "Teknik": {"summary": {"turns": 0}}}},
           "gates": {"Viking Nine": [{"variant": "OMXS30", "source": "Börsdata", "trades": 140, "win_rate": 44.0,
                                      "expectancy": 0.2, "profit_factor": 1.5, "max_drawdown_r": 10.0, "total_r": 28.0,
                                      "error": None},
                                     {"variant": "Norden (likaviktad)", "source": "Norden", "trades": 0,
                                      "error": "RuntimeError: x"}]},
           "summary": ["Norden (likaviktad) mot OMXS30: Norden (likaviktad) leder OMXS30 med 1 vecka/veckor"],
           "notes": ["Proben mäter — kartan läser inte resultatet."]}


def test_probe_tables():
    t = vui.probe_tables(PAYLOAD)
    assert list(t) == ["Samvariation", "SMA200-korsningar", "Sektorernas tidighet", "Grindvarianterna"]
    assert t["Samvariation"].iloc[0]["Korr dag"] == 0.83 and "Samma sida SMA200 %" in t["Samvariation"].columns
    assert t["SMA200-korsningar"].iloc[0]["Före"] == 7 and t["SMA200-korsningar"].iloc[0]["Median dagar"] == -6.0
    assert len(t["Sektorernas tidighet"]) == 1 and t["Sektorernas tidighet"].iloc[0]["Segment"] == "Industri"
    g = t["Grindvarianterna"]
    assert list(g["Källa / fel"]) == ["Börsdata", "RuntimeError: x"] and g.iloc[0]["Summa R"] == 28.0
    assert vui.probe_tables(None) == {} and vui.probe_tables({"gates": {}}) == {}
    assert vui.BLOBS["nordic_probe"] == "valhall_nordic_probe.json"


def _script():
    import streamlit as st
    from valhall import ui as v
    v.render_valhall_page(st.session_state.get("_sub", "Backtester"))


def test_backtests_tab_shows_probe(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(vui, "load", lambda loader=None: {"board": None, "fas3": None, "fas4b": None, "fas4c": None,
                                                          "nordic_probe": PAYLOAD, "loaded": "nu"})
    at = AppTest.from_function(_script)
    at.session_state["_sub"] = "Backtester"
    at.run(timeout=30)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "NORDEN-PROBEN" in text and "leder OMXS30 med 1" in text and "Grindvarianterna" in text
    assert "Börsdata · OMX Stockholm 30" in text and "Inga backtester sparade" not in text
    assert len(at.dataframe) == 4
