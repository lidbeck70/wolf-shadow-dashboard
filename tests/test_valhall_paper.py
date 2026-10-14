"""
Valhall fas 4a — papperskontona per strategi: signalerna ur dagens körning, fyllning på nästa öppning,
varje strategis säljregel, skuggexit på RÖD, märkning med cykelläget och sammanställningen.
Syntetiska data — inget nätverk.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valhall import cycle  # noqa: E402
from valhall import paper as vpa  # noqa: E402
from valhall import segments as sgm  # noqa: E402
from valhall import swing  # noqa: E402

IDX = pd.bdate_range("2026-01-01", periods=40)
D = [str(d.date()) for d in IDX]


def _df(o=100.0, h=None, lo=None, c=None, n=40, idx=IDX):
    o = np.full(n, o) if np.isscalar(o) else np.asarray(o, float)
    c = o if c is None else (np.full(n, c) if np.isscalar(c) else np.asarray(c, float))
    h = np.maximum(o, c) + 1 if h is None else np.asarray(h, float)
    lo = np.minimum(o, c) - 1 if lo is None else np.asarray(lo, float)
    return pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c, "Volume": np.full(n, 1e6)}, index=idx[:n])


def _board(n_status=cycle.GRON, t_status=cycle.GUL, close=100.0):
    segs = []
    for s in sgm.SEGMENTS:
        st = {"status": n_status, "phase": cycle.STARK, "cykel": cycle.MITTEN, "close": close}
        segs.append({"key": s.key, "nordic": dict(st), "global": {**st, "status": cycle.ROD, "close": 50.0},
                     "themes": [{"theme": t, "state": {**st, "status": t_status, "close": 10.0}} for t in s.themes]})
    return {"segments": segs}


UNI = {"VOLV-B.ST": {"segment": "industrials"}, "INVE-B.ST": {"segment": "financials"},
       "BOL.ST": {"segment": "materials"}, "EVO.ST": {"segment": "consumer_disc"}}


# ── Märkning och köpregler ──────────────────────────────────────────────────
def test_cycle_tag_and_governing_view():
    tag = vpa.cycle_tag(_board(), "materials", "koppar")
    assert tag["n_status"] == cycle.GRON and tag["g_status"] == cycle.ROD and tag["t_status"] == cycle.GUL
    assert tag["t_close"] == 10.0 and tag["n_cykel"] == cycle.MITTEN
    assert vpa.cycle_tag(_board(), "industrials")["t_status"] == cycle.NODATA
    assert vpa.cycle_tag({}, "okänd")["n_status"] == cycle.NODATA
    assert vpa._gov({"theme": "guld"}) == "t" and vpa._gov({"nordic": True}) == "n" and vpa._gov({}) == "g"


def test_buy_ok_per_source():
    assert vpa.buy_ok({"source": "Rick Rule-arket", "verdict": "Köpkandidat"})
    assert not vpa.buy_ok({"source": "Rick Rule-arket", "verdict": "Bevaka"})
    assert vpa.buy_ok({"source": "Quality Contrarian", "verdict": "BUY"})
    assert not vpa.buy_ok({"source": "Quality Contrarian", "verdict": "WAIT"})
    assert vpa.buy_ok({"source": "Durrett", "ticker": "NEM", "bagger": True})
    assert not vpa.buy_ok({"source": "Durrett", "ticker": "NEM", "bagger": False})          # utanför Norden: bara 15×
    assert vpa.buy_ok({"source": "Durrett", "ticker": "BOL.ST", "bagger": False})
    assert vpa.buy_ok({"source": "Tiggre", "buy": True}) and not vpa.buy_ok({"source": "Tiggre", "buy": False})
    assert vpa.buy_ok({"source": "Rule-håven"})


def test_wolf_live_freshness_and_mapping():
    blob = {"generated": "2026-10-12T04:20", "top": [{"ticker": "VOLV B", "rank": 3, "setupA": True},
                                                       {"ticker": "OKÄND", "rank": 4}]}
    live = vpa.wolf_live(blob, {"VOLV B": "VOLV-B.ST"}, "2026-10-13")
    assert set(live) == {"VOLV-B.ST"} and live["VOLV-B.ST"]["rank"] == 3
    assert vpa.wolf_live({**blob, "generated": "2026-10-01"}, {"VOLV B": "VOLV-B.ST"}, "2026-10-13") is None
    assert vpa.wolf_live(None, {}, "2026-10-13") is None


def test_signals_per_strategy():
    sw = {"segments": {
        "industrials": {"strategy": swing.TEST, "picks": [{"ticker": "VOLV-B.ST", "category": "GOLDEN TICKET"},
                                                          {"ticker": "SAND.ST", "category": "READY"}]},
        "materials": {"strategy": swing.BERSERK, "picks": [{"ticker": "BOL.ST", "setup": "S1"}]}}}
    buys = {"segments": {"financials": {"picks": [{"ticker": "INVE-B.ST", "source": "Quality Contrarian",
                                                    "verdict": "BUY"},
                                                   {"ticker": "SEB-A.ST", "source": "Quality Contrarian",
                                                    "verdict": "WAIT"}]}},
            "themes": {"guld": {"quality": {"picks": [{"ticker": "NEM", "source": "Rule-håven"}]},
                                "leverage": {"strategy": "Durrett", "picks": [{"ticker": "MMY.V", "source": "Durrett",
                                                                               "bagger": True},
                                                                              {"ticker": "XX.V", "source": "Durrett"}]}},
                       "koppar": {"quality": {"picks": []},
                                  "leverage": {"strategy": "Tiggre", "picks": [{"ticker": "CU.V", "source": "Tiggre",
                                                                                "buy": True}]}}}}
    insider = {"clusters": [{"ticker": "EVO.ST", "name": "Evolution", "insiders": 3, "role": "VD/CFO", "amount": 2000,
                             "okar_25": True, "efter_fall": True, "aterkommande": True, "cluster_avg": 100.0,
                             "price_now": 105.0, "gate": "Ja", "trigger": "A"},
                            {"ticker": "EVO.ST", "insiders": 1, "role": "Övrig", "amount": 10, "cluster_avg": 100.0,
                             "price_now": 105.0}]}
    wolf = {"VOLV-B.ST": {"rank": 3, "setupA": True, "name": "Volvo"}, "INVE-B.ST": {"rank": 30, "setupA": True}}
    sigs = vpa.signals(sw, buys, UNI, insider, wolf)
    got = {(s["strategy"], s["ticker"]) for s in sigs}
    assert got == {("viking", "VOLV-B.ST"), ("insider", "EVO.ST"), ("wolf", "VOLV-B.ST"), ("quality", "INVE-B.ST"),
                   ("rule", "NEM"), ("durrett", "MMY.V"), ("tiggre", "CU.V")}
    by = {(s["strategy"], s["ticker"]): s for s in sigs}
    assert by[("insider", "EVO.ST")]["stop"] == pytest.approx(85.0) and by[("insider", "EVO.ST")]["nordic"]
    assert by[("rule", "NEM")]["segment"] == "materials" and not by[("rule", "NEM")]["nordic"]
    assert by[("tiggre", "CU.V")]["theme"] == "koppar"


# ── Steget ──────────────────────────────────────────────────────────────────
def _sig(strategy, ticker, segment="industrials", theme=None, stop=None, nordic=True):
    return {"strategy": strategy, "ticker": ticker, "name": ticker, "segment": segment, "theme": theme,
            "nordic": nordic, "why": "", "stop": stop}


def test_order_fills_next_open_and_cooldown():
    data = {"A.ST": _df(o=np.arange(100, 140, 1.0))}
    st, ev = vpa.step(None, [_sig("quality", "A.ST")], _board(), lambda t: data[t].loc[:D[5]], today=D[5])
    assert st["orders"][0]["signal_date"] == D[5] and ev[0]["kind"] == vpa.KOP_OPEN
    assert st["orders"][0]["cycle"]["n_status"] == cycle.GRON
    st, ev = vpa.step(st, [_sig("quality", "A.ST")], _board(), lambda t: data[t].loc[:D[6]], today=D[6])
    p = st["positions"][0]
    assert p["entry"] == 106.0 and p["entry_date"] == D[6] and p["init_stop"] is None and not st["orders"]
    assert [e["kind"] for e in ev] == [vpa.KOPT]                                       # redan öppen → ingen ny order
    st["closed"].append({**p, "exit_date": D[6], "exit": 106.0})
    st["positions"] = []
    st2, _ = vpa.step(st, [_sig("quality", "A.ST")], _board(), lambda t: data[t].loc[:D[7]], today=D[7])
    assert not st2["orders"]                                                          # karantän två veckor


def test_order_expires_without_data():
    st, _ = vpa.step(None, [_sig("quality", "X.ST")], _board(), lambda t: None, today="2026-01-05")
    st, ev = vpa.step(st, [], _board(), lambda t: None, today="2026-01-14")
    assert not st["orders"] and ev[0]["kind"] == vpa.UTGANGEN


def test_insider_stop_and_time_stop():
    lo = np.full(40, 95.0)
    lo[10] = 80.0                                                                     # stoppen 85 nås dag 10
    data = {"E.ST": _df(o=100.0, lo=lo, c=100.0)}
    st, _ = vpa.step(None, [_sig("insider", "E.ST", stop=85.0)], _board(), lambda t: data[t].loc[:D[2]], today=D[2])
    st, _ = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[3]], today=D[3])
    assert st["positions"][0]["init_stop"] == 85.0
    st, ev = vpa.step(st, [], _board(), lambda t: data[t], today=D[-1])
    c = st["closed"][0]
    assert c["exit"] == 85.0 and c["exit_date"] == D[10] and ev[0]["kind"] == vpa.STOPPAD
    res = vpa.results(st)
    assert res["by_strategy"][0]["avg_r"] == pytest.approx(-1.0)


def test_insider_order_below_stop_is_blocked():
    data = {"E.ST": _df(o=80.0)}
    st, _ = vpa.step(None, [_sig("insider", "E.ST", stop=85.0)], _board(), lambda t: data[t].loc[:D[2]], today=D[2])
    st, ev = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[3]], today=D[3])
    assert not st["positions"] and ev[0]["kind"] == vpa.SPARRAD


def test_long_time_stop_and_red_shadow_exit():
    idx = pd.bdate_range("2023-01-02", periods=900)
    df = _df(o=np.linspace(100, 190, 900), n=900, idx=idx)
    d = [str(x.date()) for x in idx]
    st, _ = vpa.step(None, [_sig("rule", "G.ST", "materials", "guld")], _board(), lambda t: df.loc[:d[0]], today=d[0])
    st, _ = vpa.step(st, [], _board(), lambda t: df.loc[:d[1]], today=d[1])
    st, ev = vpa.step(st, [], _board(t_status=cycle.ROD, close=100.0), lambda t: df.loc[:d[100]], today=d[100])
    p = st["positions"][0]
    assert p["red"]["date"] == d[100] and p["red"]["price"] == pytest.approx(df["Close"].iloc[100])
    assert any(e["kind"] == vpa.ROD for e in ev)
    st, ev = vpa.step(st, [], _board(t_status=cycle.ROD), lambda t: df, today=d[-1])
    assert not any(e["kind"] == vpa.ROD for e in ev)                                   # bara första gången
    c = st["closed"][0]
    held = (pd.Timestamp(c["exit_date"]) - pd.Timestamp(c["entry_date"])).days
    assert c["reason"] == "tidsstopp 3 år" and vpa.LONG_DAYS <= held < vpa.LONG_DAYS + 5
    g = vpa.results(st)["by_strategy"][0]
    assert g["avg_red_ret"] < g["avg_ret"] and g["red"] == 1 and g["avg_r"] is None


def test_wolf_exit_when_out_of_top40():
    data = {"V.ST": _df(o=np.arange(100, 140, 1.0))}
    live = {"V.ST": {"rank": 3, "setupA": True}}
    st, _ = vpa.step(None, [_sig("wolf", "V.ST")], _board(), lambda t: data[t].loc[:D[1]], today=D[1], wolf=live)
    st, _ = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[2]], today=D[2], wolf=live)
    st, _ = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[3]], today=D[3], wolf=None)   # listan saknas
    assert st["positions"] and not st["positions"][0]["pending"]
    st, ev = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[4]], today=D[4], wolf={})
    assert st["positions"][0]["pending"]["date"] == D[4] and ev[0]["kind"] == vpa.SALJ_OPEN
    st, ev = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[5]], today=D[5], wolf={})
    assert st["closed"][0]["exit"] == 105.0 and st["closed"][0]["exit_date"] == D[5]


def test_momentum_and_wolf_accounts_are_named_apart():
    assert vpa.STRATEGIES["wolf"] == ("Momentum Swing", vpa.SWING) and vpa.STRATEGIES["wolf_arc"] == ("Wolf", vpa.SWING)


def test_wolf_arc_fills_with_atr_stop_and_exits_on_strategy_rule(monkeypatch):
    import strategies.wolf as sw
    data = {"V.ST": _df(o=np.arange(100, 140, 1.0))}
    arc = [_sig("wolf_arc", "V.ST")]
    sigs = vpa.signals({}, {}, UNI, None, None, None, arc_sigs=arc)
    assert [(s["strategy"], s["ticker"]) for s in sigs] == [("wolf_arc", "V.ST")]
    calls = []

    def fake_exit(pos, df):
        calls.append((str(df.index[-1].date()), pos.get("bars_below_ema50"), pos.get("stop_loss")))
        hit = df.index[-1] >= pd.Timestamp(D[32])
        return {"exit": hit, "reason": "TRAIL_EXIT" if hit else None, "exit_price": None,
                "partial_tp1": False, "partial_tp2": False}

    monkeypatch.setattr(sw, "exit_fn", fake_exit)
    st, ev = vpa.step(None, sigs, _board(), lambda t: data[t].loc[:D[25]], today=D[25])   # 26 dagar historik → ATR14
    assert ev[0]["kind"] == vpa.KOP_OPEN and ev[0]["text"].startswith("Wolf ·")
    st, ev = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[26]], today=D[26])
    p = st["positions"][0]
    d = sw.risk_fn(data["V.ST"].loc[:D[26]], 1.0)["stop_distance"]                  # 2,5 × ATR14 på fyllningsdagen
    assert p["entry"] == 126.0 and d > 0 and p["init_stop"] == pytest.approx(126.0 - d, abs=1e-6)
    assert p["below50"] == 0 and calls[0] == (D[26], 0, p["init_stop"])
    st, ev = vpa.step(st, [], _board(), lambda t: data[t], today=D[-1])
    c = st["closed"][0]
    assert c["reason"] == "under Kijun och EMA10" and c["exit_date"] == D[33] and c["exit"] == 133.0   # nästa öppning
    assert all(b == 0 for _d, b, _s in calls)                                          # stigande: aldrig under EMA50


def test_wolf_arc_hard_stop_is_intraday():
    lo = np.full(40, 99.0)
    lo[30] = 90.0                                                                      # stoppen (95) nås dag 30
    data = {"V.ST": _df(o=100.0, lo=lo, c=100.0)}
    st, _ = vpa.step(None, [_sig("wolf_arc", "V.ST")], _board(), lambda t: data[t].loc[:D[25]], today=D[25])
    st, _ = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[26]], today=D[26])
    stop = st["positions"][0]["init_stop"]
    assert stop == pytest.approx(95.0)                                                 # 100 − 2,5 × ATR14 (= 2)
    st, ev = vpa.step(st, [], _board(), lambda t: data[t], today=D[-1])
    c = st["closed"][0]
    assert c["exit"] == pytest.approx(stop) and c["exit_date"] == D[30] and ev[0]["kind"] == vpa.STOPPAD
    assert "2,5 × ATR14" in c["reason"] and vpa.results(st)["by_strategy"][0]["avg_r"] == pytest.approx(-1.0)


def test_viking_uses_live_exit_engine(monkeypatch):
    import viking_exit as vex
    data = {"VOLV-B.ST": _df(o=np.arange(100, 140, 1.0))}
    seen = []

    def fake(ticker, df, entry, entry_date, spy_df=None, today=None, **kw):
        seen.append((str(pd.Timestamp(today).date()), len(spy_df) if spy_df is not None else None))
        status = vex.EXIT if pd.Timestamp(today) >= pd.Timestamp(D[8]) else vex.HOLD
        return SimpleNamespace(status=status, reasons=["Trailing stop (EMA10): test"])

    monkeypatch.setattr(vex, "evaluate_exit", fake)
    market = pd.DataFrame({"Close": np.arange(40.0)}, index=IDX)
    st, _ = vpa.step(None, [_sig("viking", "VOLV-B.ST")], _board(), lambda t: data[t].loc[:D[2]], today=D[2])
    st, _ = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[3]], market=market, today=D[3])
    p = st["positions"][0]
    assert p["entry"] == 103.0 and p["init_stop"] < 103.0                                # entry − 1,5 × ATR
    st, ev = vpa.step(st, [], _board(), lambda t: data[t], market=market, today=D[-1])
    c = st["closed"][0]
    assert c["exit_date"] == D[9] and c["exit"] == 109.0 and c["reason"].startswith("Trailing")   # nästa öppning
    assert seen[0] == (D[3], 4) and (D[8], 9) in seen                                  # marknaden t.o.m. dagen


def test_berserk_trades_are_tagged_and_counted():
    bstate = {"positions": [{"ticker": "BOL.ST", "theme": "koppar", "signal_date": "2026-10-12", "entry": 100.0,
                             "init_stop": 90.0, "last_close": 110.0, "region": "Norden"}],
              "closed": [{"ticker": "LUMI.ST", "theme": "koppar", "signal_date": "2026-01-02", "entry": 50.0,
                          "init_stop": 45.0, "exit": 45.0, "r": -1.0}], "orders": []}
    st, _ = vpa.step(None, [], _board(), lambda t: None, today="2026-10-13", berserk_state=bstate, universe=UNI)
    tags = st["berserk_tags"]
    assert tags["BOL.ST|2026-10-12"]["cycle"]["t_status"] == cycle.GUL and tags["BOL.ST|2026-10-12"]["segment"] == "materials"
    assert tags["LUMI.ST|2026-01-02"]["cycle"] == {}                                   # för gammal → okänt läge
    res = vpa.results(st, bstate)
    by = {r["status_in"]: r for r in res["by_status"] if r["strategy"] == "berserk"}
    assert by[cycle.GUL]["avg_r"] is None and by[cycle.GUL]["open"] == 1 and by[cycle.GUL]["avg_ret"] == 10.0
    assert by["OKÄND"]["avg_r"] == pytest.approx(-1.0)
    st2, _ = vpa.step(st, [], _board(), lambda t: None, today="2026-10-14", berserk_state={"positions": [], "closed": []})
    assert st2["berserk_tags"] == {}                                                   # borta ur kontot → städas


def test_results_group_and_excess():
    st = vpa.new_state("2026-01-01")
    base = {"strategy": "quality", "segment": "industrials", "theme": None, "nordic": True, "entry": 100.0,
            "init_stop": None, "cycle": {"n_status": cycle.GRON, "n_cykel": cycle.TIDIG, "n_close": 200.0}}
    st["positions"] = [{**base, "ticker": "A", "last_close": 120.0, "bench_last": 220.0}]
    st["closed"] = [{**base, "ticker": "B", "exit": 90.0, "bench_out": 180.0, "exit_date": "2026-02-01"}]
    res = vpa.results(st)
    g = res["by_strategy"][0]
    assert g["n"] == 2 and g["open"] == 1 and g["avg_ret"] == 5.0 and g["win"] == 50.0
    assert g["avg_excess"] == pytest.approx(5.0)                                       # (+20 − 10) och (−10 + 10)
    assert res["by_cykel"][0]["cykel_in"] == cycle.TIDIG
    assert vpa.summary_lines(res)[1].strip().startswith("Quality Contrarian")


def test_quality_order_gate_uses_signal_day_status_at_fill():
    data = {"HEXA-B.ST": _df(o=np.arange(100, 140, 1.0)), "ORNAV.HE": _df(o=np.arange(100, 140, 1.0))}
    red = _board()
    for r in red["segments"]:
        if r["key"] == "industrials":
            r["nordic"]["status"] = cycle.ROD
    # en order från före grinden: lagd när Industri redan var RÖD (som HEXA-B 14/10)
    st = vpa.new_state(D[0])
    st["orders"] = [{**_sig("quality", "HEXA-B.ST"), "signal_date": D[1], "cycle": vpa.cycle_tag(red, "industrials")},
                    {**_sig("wolf", "HEXA-B.ST"), "signal_date": D[1], "cycle": vpa.cycle_tag(red, "industrials")}]
    # och en lagd på en GRÖN dag — sektorn blir RÖD på fyllningsdagens stängning (får inte titta framåt)
    st, ev1 = vpa.step(st, [_sig("quality", "ORNAV.HE", segment="industrials")], _board(),
                       lambda t: data[t].loc[:D[1]], today=D[1])
    st, ev2 = vpa.step(st, [], red, lambda t: data[t].loc[:D[2]], today=D[2])
    evs = {(e["strategy"], e["ticker"], e["kind"]): e for e in ev1 + ev2}
    hexa = evs[("quality", "HEXA-B.ST", vpa.SPARRAD)]
    assert hexa in ev1 and "signaldagen" in hexa["text"]                                 # struken direkt
    assert ("quality", "ORNAV.HE", vpa.KOPT) in evs                                      # ingen framåtblick …
    assert ("quality", "ORNAV.HE", vpa.ROD) in evs                                       # … men skuggexit noteras
    assert ("wolf", "HEXA-B.ST", vpa.KOPT) in evs                                        # grinden gäller bara Quality
    assert {(p["strategy"], p["ticker"]) for p in st["positions"]} == {("quality", "ORNAV.HE"), ("wolf", "HEXA-B.ST")}


def test_one_company_per_account_across_share_classes():
    data = {t: _df(o=np.arange(100, 140, 1.0)) for t in ("HLUN-A.CO", "HLUN-B.CO", "INVE-A.ST", "INVE-B.ST")}
    a = {**_sig("long", "HLUN-A.CO", "healthcare"), "name": "H. Lundbeck A"}
    b = {**_sig("long", "HLUN-B.CO", "healthcare"), "name": "H. Lundbeck B"}
    # gamla dubbletter från före rättelsen: två order i samma bolag ligger redan i kontot
    st = vpa.new_state(D[0])
    st["orders"] = [{**a, "signal_date": D[1], "cycle": vpa.cycle_tag(_board(), "healthcare")},
                    {**b, "signal_date": D[1], "cycle": vpa.cycle_tag(_board(), "healthcare")}]
    st, ev = vpa.step(st, [], _board(), lambda t: data[t].loc[:D[2]], today=D[2])
    assert [p["ticker"] for p in st["positions"]] == ["HLUN-A.CO"]
    assert any(e["kind"] == vpa.UTGANGEN and e["ticker"] == "HLUN-B.CO" and "aktieslag" in e["text"] for e in ev)
    # nya signaler: B-aktien i ett ägt bolag blir ingen order; samma bolag i ETT annat konto går bra
    ia = {**_sig("long", "INVE-A.ST", "financials"), "name": "Investor A"}
    ib = {**_sig("long", "INVE-B.ST", "financials"), "name": "Investor B"}
    other = {**_sig("long_ingen", "HLUN-B.CO", "healthcare"), "name": "H. Lundbeck B"}
    st, ev = vpa.step(st, [b, ia, ib, other], _board(), lambda t: data[t].loc[:D[3]], today=D[3])
    orders = {(o["strategy"], o["ticker"]) for o in st["orders"]}
    assert orders == {("long", "INVE-A.ST"), ("long_ingen", "HLUN-B.CO")}
    # karantänen gäller bolaget: stängd HLUN-A → inget återköp via HLUN-B i två veckor
    st["positions"] = [p for p in st["positions"] if p["ticker"] != "HLUN-A.CO"]
    st["closed"].append({**a, "exit_date": D[3], "exit": 100.0})
    st, _ = vpa.step(st, [b], _board(), lambda t: data[t].loc[:D[4]], today=D[4])
    assert ("long", "HLUN-B.CO") not in {(o["strategy"], o["ticker"]) for o in st["orders"]}
    # … och tickern även om namnet stavas annorlunda i dag ('H. Lundbeck A/S')
    renamed = {**a, "name": "H. Lundbeck A/S"}
    st, _ = vpa.step(st, [renamed], _board(), lambda t: data[t].loc[:D[5]], today=D[5])
    assert ("long", "HLUN-A.CO") not in {(o["strategy"], o["ticker"]) for o in st["orders"]}
