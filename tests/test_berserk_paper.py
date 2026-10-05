"""
🪓 BERSERK PR 3 — papperskontot (fyllning, exit med backtestets regler, sälj på
öppning, flyttat stopp, spärrar, omkörning samma dag), den headless skanningen
med larm, arbetsflödet och fliken PAPPERSKONTO. Syntetiska kurser — inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from berserk import backtest as bt  # noqa: E402
from berserk import live  # noqa: E402
from berserk import paper  # noqa: E402
from berserk import signals as sg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 596
IDX = pd.bdate_range(end="2026-09-30", periods=N)
K = 592                                                         # signaldagen (S3-panik)


def _ohlcv(c, idx=IDX):
    c = np.asarray(c, dtype=float)
    return pd.DataFrame({"Open": c * 1.002, "High": c * 1.006, "Low": c * 0.994, "Close": c, "Volume": 1e6},
                        index=idx)


def _ser(values, idx=IDX):
    return pd.Series(np.asarray(values, dtype=float), index=idx)


UP = _ser(np.linspace(100, 200, N))


def _bounce():
    c = np.linspace(60, 140, N)
    c[590:593] = [134, 129, 124]                                # panik i upptrend → S3 dag K
    c[593:] = [126, 135, 136]                                   # dag K+2 över SMA5 → säljs K+3
    return _ohlcv(c)


def _ctx(stock, upto):
    return {"BOL.ST": {"stock": stock.iloc[:upto + 1], "driver": UP.iloc[:upto + 1], "market": None}}


def _signal_row(stock):
    row = live.evaluate("BOL.ST", stock.iloc[:K + 1], UP.iloc[:K + 1], UP.iloc[:K + 1], driver_symbol="HG=F")
    assert row["status"] == live.KOP and row["setup"] == sg.S3
    return row


def _run_to(stock, last):
    """Kör kontot dag för dag från signaldagen t.o.m. dag last."""
    state, all_events = None, []
    for d in range(K, last + 1):
        rows = [_signal_row(stock)] if d == K else []
        state, ev = paper.step(state, rows, _ctx(stock, d), today=IDX[d])
        all_events.append(ev)
    return state, all_events


# ── Papperskontot ───────────────────────────────────────────────────────────
def test_order_fill_sell_on_open_and_exit():
    stock = _bounce()
    state, evs = _run_to(stock, K + 3)
    assert [e["kind"] for e in evs[0]] == [paper.KOP_OPEN] and evs[0][0]["ticker"] == "BOL.ST"
    assert [e["kind"] for e in evs[1]] == [paper.KOPT]                       # dag 1: fylld, under SMA5
    assert [e["kind"] for e in evs[2]] == [paper.SALJ_OPEN]                  # dag 2: över SMA5
    assert [e["kind"] for e in evs[3]] == [paper.SALD]                       # dag 3: säljs på öppningen
    c = state["closed"][0]
    entry, exit_ = stock["Open"].iloc[K + 1], stock["Open"].iloc[K + 3]
    assert c["entry"] == pytest.approx(entry, abs=1e-3) and c["exit"] == pytest.approx(exit_, abs=1e-3)
    assert c["reason"] == "över SMA5" and c["exit_date"] == str(IDX[K + 3].date())
    atr0 = sg.atr(stock.iloc[:K + 1]).iloc[-1]
    assert c["r"] == pytest.approx((exit_ - entry) / (bt.STOP_ATR[sg.S3] * atr0), abs=1e-2)
    assert not state["positions"] and not state["orders"]
    assert paper.equity(state) == pytest.approx(100 + c["pnl"], abs=1e-3)
    assert len(state["curve"]) == 4 and state["last_run"] == str(IDX[K + 3].date())


def test_same_as_backtest_exit():
    """Papperskontot och backtestet ger samma exit för samma affär (en källa)."""
    stock = _bounce()
    state, _ = _run_to(stock, K + 3)
    res = bt.backtest_ticker("BOL.ST", stock, UP, bt.Config(min_turnover_m=0, setups=(sg.S3,)), start=IDX[K])
    t = next(t for t in res["trades"] if t.signal_date == str(IDX[K].date()))
    assert (t.exit, t.exit_reason, t.exit_date) == (pytest.approx(state["closed"][0]["exit"], abs=1e-3),
                                                    state["closed"][0]["reason"], state["closed"][0]["exit_date"])


def test_stop_out():
    c = np.linspace(60, 140, N)
    c[590:593] = [134, 129, 124]
    c[593:] = [123, 100, 98]                                                  # rasar genom 3 ATR-stoppet
    stock = _ohlcv(c)
    state, evs = _run_to(stock, K + 2)
    kinds = [e["kind"] for day in evs for e in day]
    assert paper.STOPPAD in kinds
    c0 = state["closed"][0]
    assert c0["reason"] == "katastrofstopp" and c0["r"] <= -1.0 + 1e-6
    assert paper.equity(state) < 100


def test_rerun_same_day_is_quiet():
    stock = _bounce()
    state, _ = _run_to(stock, K + 2)
    again, ev = paper.step(state, [], _ctx(stock, K + 2), today=IDX[K + 2])
    assert ev == [] and again["positions"] == state["positions"] and again["cash"] == state["cash"]
    assert len(again["curve"]) == len(state["curve"])
    s0, _ = paper.step(None, [_signal_row(stock)], _ctx(stock, K), today=IDX[K])
    s1, ev1 = paper.step(s0, [_signal_row(stock)], _ctx(stock, K), today=IDX[K])
    assert ev1 == [] and len(s1["orders"]) == 1                              # ingen dubbelorder


def test_s1_stop_moves_to_breakeven():
    c = np.linspace(60, 120, N)
    stock = _ohlcv(c)
    i = K
    atr0 = float(sg.atr(stock.iloc[:i + 1]).iloc[-1])
    entry = float(stock["Open"].iloc[i + 1])
    c2 = c.copy()
    c2[i + 2:] = entry + 1.5 * atr0                                          # stängning ≥ +1 ATR
    stock = _ohlcv(c2)
    pos = {"ticker": "BOL.ST", "setup": sg.S1, "theme": "koppar", "complex": "basmetaller", "region": "Norden",
           "signal_date": str(IDX[i].date()), "entry_date": str(IDX[i + 1].date()), "entry": entry,
           "init_stop": entry - 2 * atr0, "cur_stop": entry - 2 * atr0, "atr": atr0, "weight": 10.0,
           "units": 10.0 / entry, "last_close": entry, "last_date": str(IDX[i + 1].date()), "pending": None,
           "armed_be": False, "trailing": False}
    state = {**paper.new_state(IDX[i]), "cash": 90.0, "positions": [pos]}
    state, ev = paper.step(state, [], _ctx(stock, i + 2), today=IDX[i + 2])
    assert [e["kind"] for e in ev] == [paper.FLYTTA]
    p = state["positions"][0]
    assert p["cur_stop"] == pytest.approx(entry - bt.S1_BE_GIVE * atr0, abs=1e-3) and p["armed_be"]


def _pos(t, theme, cx, risk=1.0):
    return {"ticker": t, "theme": theme, "complex": cx, "entry": 100.0, "init_stop": 100.0 - risk * 5,
            "units": 0.2, "last_close": 100.0}


def test_caps():
    st_ = {**paper.new_state(), "positions": [_pos("FCX", "koppar", "basmetaller"),
                                              _pos("BOL.ST", "koppar", "basmetaller")]}
    assert paper.block_reason(st_, "koppar", 10, 1.0) == "TEMA FULLT"
    assert paper.block_reason(st_, "guld", 10, 1.0) is None
    st_["positions"] += [_pos("AA", "aluminium", "basmetaller"), _pos("NHY.OL", "aluminium", "basmetaller")]
    assert paper.block_reason(st_, "zink_nickel", 10, 1.0) == "KOMPLEX FULLT"
    st_["positions"] += [_pos(f"G{k}", f"t{k}", f"c{k}") for k in range(4)]
    assert paper.block_reason(st_, "guld", 10, 1.0) == "MAX 8 POSITIONER"
    hot = {**paper.new_state(), "cash": 0.0,                                # 5 × 20 % med 1 % risk vardera
           "positions": [_pos(f"H{k}", f"t{k}", f"c{k}") for k in range(5)]}
    assert paper.heat(hot) == pytest.approx(5.0, abs=0.1)
    assert paper.block_reason(hot, "guld", 10, 1.25).startswith("VÄRME")
    extra = [{"theme": "guld", "complex": "adelmetaller", "risk_pct": 1.0}] * 2
    assert paper.block_reason(paper.new_state(), "guld", 10, 1.0, extra) == "TEMA FULLT"


def test_capped_signal_is_alerted_once():
    stock = _bounce()
    full = {**paper.new_state(), "positions": [_pos("FCX", "koppar", "basmetaller"),
                                               _pos("LUMI.ST", "koppar", "basmetaller")]}
    s1, ev = paper.step(full, [_signal_row(stock)], _ctx(stock, K), today=IDX[K])
    assert [e["kind"] for e in ev] == [paper.SPARRAD] and "TEMA FULLT" in ev[0]["text"] and not s1["orders"]
    _s2, ev2 = paper.step(s1, [_signal_row(stock)], _ctx(stock, K), today=IDX[K])
    assert ev2 == []


def test_stale_order_expires():
    stock = _bounce()
    s0, _ = paper.step(None, [_signal_row(stock)], _ctx(stock, K), today=IDX[K])
    s1, ev = paper.step(s0, [], {}, today=IDX[K] + pd.Timedelta(days=10))
    assert [e["kind"] for e in ev] == [paper.UTGANGEN] and not s1["orders"]


def test_messages_chunked_under_discord_limit():
    stock = _bounce()
    state, evs = _run_to(stock, K + 1)
    assert paper.messages([], state) == []
    many = [{"date": "2026-09-30", "kind": paper.KOP_OPEN, "ticker": f"T{k}", "text": "x" * 120} for k in range(40)]
    msgs = paper.messages(many, state)
    assert len(msgs) > 1 and all(len(m) <= 2000 for m in msgs)
    assert msgs[0].startswith("🪓 BERSERK ·") and "manuellt" in msgs[-1]
    one = paper.messages(evs[1], state)[0]
    assert "KÖPT" in one and "BOL.ST" in one


def test_summary():
    stock = _bounce()
    state, _ = _run_to(stock, K + 3)
    s = paper.summary(state)
    assert s["trades"] == 1 and s["open"] == 0 and s["avg_r"] == pytest.approx(state["closed"][0]["r"], abs=0.01)
    assert s["max_dd_pct"] <= 0 and s["equity"] == pytest.approx(paper.equity(state), abs=0.01)


# ── Headless skanningen ─────────────────────────────────────────────────────
def _fake_world(day):
    stock = _bounce().iloc[:day + 1]
    up = _ohlcv(np.linspace(100, 200, N)).iloc[:day + 1]
    data = {"BOL.ST": stock, "HG=F": up, "SPY": up}
    return lambda t, p: data.get(t)


def test_script_run_saves_and_alerts(monkeypatch):
    import berserk_scan as bs
    monkeypatch.setattr(bs, "universe", lambda: ["BOL.ST", "FCX", "NEM"])
    store, sent = {}, []
    kw = dict(nordic_provider=lambda: {"close": UP}, load=lambda f, fb: store.get(f, fb),
              save=lambda f, d: store.__setitem__(f, d) or True, send=lambda m: sent.append(m) or True)
    out = bs.run(getter=_fake_world(K), today=IDX[K], **kw)
    assert out["saved"] and out["sent"] == 1 and "KÖP PÅ ÖPPNING" in sent[0] and "BOL.ST" in sent[0]
    assert store[bs.SCAN_BLOB]["counts"]["KÖP"] == 1 and store[bs.SCAN_BLOB]["rows"][0]["ticker"] == "BOL.ST"
    assert store[bs.PAPER_BLOB]["orders"][0]["ticker"] == "BOL.ST"
    out2 = bs.run(getter=_fake_world(K + 1), today=IDX[K + 1], **kw)
    assert [e["kind"] for e in out2["events"]] == [paper.KOPT] and len(sent) == 2
    out3 = bs.run(getter=_fake_world(K + 1), today=IDX[K + 1], **kw)        # reservkörningen samma dag
    assert out3["events"] == [] and len(sent) == 2


def test_script_dry_run_and_unreadable_paper(monkeypatch):
    import berserk_scan as bs
    monkeypatch.setattr(bs, "universe", lambda: ["BOL.ST"])
    store = {bs.SCAN_BLOB: {"paper_last_run": "2026-09-29", "paper": {"equity": 101.0}}}
    saves = []
    kw = dict(getter=_fake_world(K), nordic_provider=lambda: {"close": UP}, load=lambda f, fb: store.get(f, fb),
              save=lambda f, d: saves.append(f) or True, send=lambda m: pytest.fail("ska inte skicka"),
              today=IDX[K])
    out = bs.run(**kw)
    assert out["paper"] is None and saves == [bs.SCAN_BLOB]                  # papperskontot skrivs inte över
    store.pop(bs.SCAN_BLOB)
    out = bs.run(dry_run=True, **kw)
    assert saves == [bs.SCAN_BLOB] and out["messages"] and out["saved"] is False


def test_workflow_file():
    src = open(os.path.join(ROOT, ".github", "workflows", "berserk-scan.yml"), encoding="utf-8").read()
    for text in ('cron: "35 21 * * 1-5"', 'cron: "5 22 * * 1-5"', "workflow_dispatch",
                 "bash scripts/schedule_gate.sh berserk-scan.yml", "secrets.GIST_TOKEN",
                 "secrets.DISCORD_WEBHOOK_URL", "secrets.BORSDATA_API_KEY", "python berserk_scan.py"):
        assert text in src, text
    assert "seasonal" not in src.split("schedule_gate.sh", 1)[1].split("\n", 1)[0]


# ── Fliken ──────────────────────────────────────────────────────────────────
def test_paper_tab(monkeypatch):
    from streamlit.testing.v1 import AppTest
    stock = _bounce()
    state, _ = _run_to(stock, K + 2)
    scan = {"when": "2026-09-30 23:40", "tickers": 255, "counts": {"KÖP": 1, "BEVAKA": 0},
            "rows": [dict(_signal_row(stock), flags=[])]}
    monkeypatch.setenv("BZ_TEST_ROOT", ROOT)

    def app():
        import os as _o
        import sys as _s
        _s.path.insert(0, _o.environ["BZ_TEST_ROOT"])
        from berserk.screen_ui import render_berserk_screen_page
        render_berserk_screen_page()

    at = AppTest.from_function(app, default_timeout=90)
    at.session_state["berserk_signals"] = []
    at.session_state["bz_auto"] = {"paper": state, "scan": scan, "loaded": "2026-10-01 08:00"}
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    for text in ("Papperskonto", "ÖPPNA POSITIONER", "HÄNDELSER", "SÄLJ PÅ ÖPPNING", "SENASTE AUTOMATISKA",
                 "1 KÖP", "manuellt"):
        assert text in html, text
    empty = AppTest.from_function(app, default_timeout=90)
    empty.session_state["berserk_signals"] = []
    empty.session_state["bz_auto"] = {"paper": None, "scan": None, "loaded": "—"}
    empty.run()
    assert not empty.exception and "inte startat" in " ".join(m.value for m in empty.markdown)


def test_removed_tickers_leave_the_paper_account():
    """Australien togs bort ur universumet: order stryks, positioner stängs på senaste kurs."""
    pos = {**_pos("FMG.AX", "jarnmalm", "basmetaller"), "setup": sg.S2, "signal_date": "2026-09-01",
           "entry_date": "2026-09-02", "last_date": "2026-10-02", "last_close": 110.0, "weight": 20.0}
    order = {"ticker": "PLS.AX", "setup": sg.S1, "theme": "litium", "complex": "basmetaller",
             "signal_date": "2026-10-02", "atr": 1.0, "risk_pct": 1.25}
    state = {**paper.new_state(), "cash": 80.0, "positions": [pos], "orders": [order]}
    s, ev = paper.step(state, [], {}, today="2026-10-05")
    assert not s["positions"] and not s["orders"]
    assert {(e["kind"], e["ticker"]) for e in ev} == {(paper.SALD, "FMG.AX"), (paper.UTGANGEN, "PLS.AX")}
    c = s["closed"][0]
    assert c["exit"] == 110.0 and c["reason"] == "borttagen ur universumet" and c["r"] == pytest.approx(2.0)
    assert paper.equity(s) == pytest.approx(80 + 0.2 * 110)
