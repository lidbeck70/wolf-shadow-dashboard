"""
Thor fas 4 — wolf_regime_score: Wolfs regimpoäng utbruten ur tabs/regime.py. Talen låses mot en kopia av
sidans gamla formler (frusen referens här i testet) på syntetiska kurser, etiketten mot trösklarna, de elva
grindarna, SL/TP-kalkylatorn — och Wolf Regime-sidan själv under AppTest med fejkad yfinance. Inget nätverk.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wolf_regime_score as wrs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _walk(seed=0, drift=0.0008, n=300, start=100.0, vol=0.015, end=None):
    """Syntetisk OHLCV-ram: slumpvandring med skuggor och volym, deterministisk per frö."""
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = np.r_[start, close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(500_000, 3_000_000, n).astype(float)
    idx = pd.bdate_range(end=end or pd.Timestamp("2026-10-14"), periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


# ── Frusen referens: sidans gamla formler, ordagrant ─────────────────────────
def _old_regime(spy_df, sec_df, stk_df):
    def _ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    def _rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _atr(df, period=14):
        high, low, close = df["High"], df["Low"], df["Close"]
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, adjust=False).mean()

    def _adx(df, period=14):
        high, low, close = df["High"], df["Low"], df["Close"]
        plus_dm, minus_dm = high.diff(), -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr_v = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_v)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_v)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
        return dx.ewm(span=period, adjust=False).mean()

    def _ichimoku(df):
        hi9, lo9 = df["High"].rolling(9).max(), df["Low"].rolling(9).min()
        hi26, lo26 = df["High"].rolling(26).max(), df["Low"].rolling(26).min()
        hi52, lo52 = df["High"].rolling(52).max(), df["Low"].rolling(52).min()
        tenkan, kijun = (hi9 + lo9) / 2, (hi26 + lo26) / 2
        return tenkan, kijun, ((tenkan + kijun) / 2).shift(26), ((hi52 + lo52) / 2).shift(26)

    def _score_index(df):
        if df is None or len(df) < 60:
            return 0
        c = df["Close"]
        e50, e200, last, rsi = _ema(c, 50).iloc[-1], _ema(c, 200).iloc[-1], c.iloc[-1], _rsi(c).iloc[-1]
        score = (10 if last > e200 else 0) + (8 if e50 > e200 else 0) + (7 if last > e50 else 0) + (5 if rsi > 50 else 0)
        return min(score, 30)

    def _score_stock(df):
        if df is None or len(df) < 60:
            return None
        c = df["Close"]
        e10, e21, e50, e200 = _ema(c, 10), _ema(c, 21), _ema(c, 50), _ema(c, 200)
        tenkan, kijun, span_a, span_b = _ichimoku(df)
        adx_s, last = _adx(df), c.iloc[-1]
        rsi_val, atr_val = float(_rsi(c).iloc[-1]), float(_atr(df).iloc[-1])
        adx_val = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0.0
        e10_last, e21_last, e50_last, e200_last = (float(x.iloc[-1]) for x in (e10, e21, e50, e200))
        s = (10 if last > e200_last else 0) + (8 if e50_last > e200_last else 0) + (8 if last > e50_last else 0)
        ema_stack = e10_last > e21_last > e50_last > e200_last
        ema_trend = last > e21_last
        s += (10 if ema_stack else 0) + (7 if ema_trend else 0) + (7 if 40 < rsi_val < 75 else 0)
        s = min(s, 50)
        sa = float(span_a.iloc[-1]) if not pd.isna(span_a.iloc[-1]) else 0
        sb = float(span_b.iloc[-1]) if not pd.isna(span_b.iloc[-1]) else 0
        tk = float(tenkan.iloc[-1]) if not pd.isna(tenkan.iloc[-1]) else 0
        kj = float(kijun.iloc[-1]) if not pd.isna(kijun.iloc[-1]) else 0
        ichi = 6 if last > max(sa, sb) else 2 if last > min(sa, sb) else 0
        ichi += (5 if tk > kj else 0) + (4 if sa > sb else 0)
        sl_dist = 1.5 * atr_val
        return {"stock_score": s, "ichi_score": min(ichi, 15), "close": float(last), "rsi": rsi_val, "atr": atr_val,
                "adx": round(adx_val, 1), "ema_stack": ema_stack, "ema_trend": ema_trend,
                "has_entry": ema_stack and ema_trend and 45 < rsi_val < 70, "entry_zone": last,
                "sl_level": last - sl_dist, "tp1_2R": last + 2.0 * sl_dist, "tp2_3R": last + 3.0 * sl_dist,
                "ema10": e10_last, "ema21": e21_last, "ema50": e50_last, "ema200": e200_last, "kijun": kj}

    mkt, sec, r = _score_index(spy_df), _score_index(sec_df), _score_stock(stk_df)
    stk, ichi = (r["stock_score"], r["ichi_score"]) if r else (0, 0)
    keys = (("close", 0), ("rsi", 0), ("adx", 0), ("ema_stack", False), ("ema_trend", False), ("has_entry", False),
            ("entry_zone", 0), ("sl_level", 0), ("tp1_2R", 0), ("tp2_3R", 0), ("atr", 0), ("ema10", 0), ("ema21", 0),
            ("ema50", 0), ("ema200", 0), ("kijun", 0))
    out = {"mkt_score": mkt, "sec_score": sec, "stk_score": stk, "ichi_score": ichi, "total": mkt + sec + stk + ichi}
    names = {"tp1_2R": "tp1", "tp2_3R": "tp2"}
    out.update({names.get(k, k): (r[k] if r else d) for k, d in keys})
    return out


def _old_checklist(data, stk_df, atr_mult):
    close, rsi, atr = data["close"], data["rsi"], data["atr"]
    e10, e21, e50, e200 = (data.get(k, 0) for k in ("ema10", "ema21", "ema50", "ema200"))
    stack = data["ema_stack"]
    trend = "BULL" if stack else "NEUTRAL-BULL" if close > e50 else "NEUTRAL" if close > e200 else "BEAR"
    rsi_state = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL" if 40 <= rsi <= 60 else "ACTIVE"
    vol_ratio = 0.0
    if stk_df is not None and len(stk_df) >= 20:
        vol_avg = stk_df["Volume"].rolling(20).mean().iloc[-1]
        if vol_avg > 0:
            vol_ratio = float(stk_df["Volume"].iloc[-1] / vol_avg)
    hist_vol = 0.0
    if stk_df is not None and len(stk_df) >= 21:
        log_ret = np.log(stk_df["Close"] / stk_df["Close"].shift(1)).dropna()
        if len(log_ret) >= 20:
            hist_vol = float(log_ret.tail(20).std() * np.sqrt(252) * 100)
    atr_ratio = 1.0
    if stk_df is not None and len(stk_df) >= 64:
        h, lo, c = stk_df["High"], stk_df["Low"], stk_df["Close"]
        tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
        s = tr.ewm(com=13, adjust=False).mean()
        if s.tail(50).mean() > 0:
            atr_ratio = float(s.iloc[-1] / s.tail(50).mean())
    bull, bear = [], []
    if stk_df is not None and len(stk_df) >= 50:
        ob_atr = atr if atr > 0 else 1.0
        recent = stk_df.tail(100) if len(stk_df) >= 100 else stk_df
        for i in range(2, len(recent)):
            cur, prev = recent.iloc[i], recent.iloc[i - 1]
            body = abs(float(cur["Close"]) - float(cur["Open"]))
            if float(cur["Close"]) > float(cur["Open"]) and body > ob_atr * 1.5 and float(prev["Close"]) < float(prev["Open"]):
                bull.append(float(cur["Low"]))
            if float(cur["Close"]) < float(cur["Open"]) and body > ob_atr * 1.5 and float(prev["Close"]) > float(prev["Open"]):
                bear.append(float(cur["High"]))
        bull, bear = bull[-5:], bear[-5:]
    nb = min(bull, key=lambda x: abs(x - close)) if bull else None
    nbe = min(bear, key=lambda x: abs(x - close)) if bear else None
    nearby = bool(nb and abs(nb - close) / close < 0.03) or bool(nbe and abs(nbe - close) / close < 0.03)
    pats = []
    if stk_df is not None and len(stk_df) >= 3:
        cc, cp = stk_df.iloc[-1], stk_df.iloc[-2]
        body, rng = float(cc["Close"]) - float(cc["Open"]), float(cc["High"]) - float(cc["Low"])
        if rng > 0:
            if min(float(cc["Open"]), float(cc["Close"])) - float(cc["Low"]) > abs(body) * 2 and body > 0:
                pats.append("Hammer (Bull)")
            if (float(cp["Close"]) < float(cp["Open"]) and float(cc["Close"]) > float(cc["Open"])
                    and float(cc["Close"]) > float(cp["Open"]) and float(cc["Open"]) < float(cp["Close"])):
                pats.append("Bullish Engulfing")
            if (float(cp["Close"]) > float(cp["Open"]) and float(cc["Close"]) < float(cc["Open"])
                    and float(cc["Close"]) < float(cp["Open"]) and float(cc["Open"]) > float(cp["Close"])):
                pats.append("Bearish Engulfing")
            if abs(body) < rng * 0.1:
                pats.append("Doji")
    d10 = abs(close - e10) / close * 100 if close > 0 else 99
    d21 = abs(close - e21) / close * 100 if close > 0 else 99
    stop = atr * atr_mult
    rr = abs(nbe - close) / stop if nbe and stop > 0 else 0.0
    return {"trend_dir": trend, "rsi_state": rsi_state, "vol_ratio": vol_ratio, "hist_vol": hist_vol, "atr_ratio": atr_ratio,
            "bull_obs": bull, "bear_obs": bear, "nearest_bull_ob": nb, "nearest_bear_ob": nbe, "has_nearby_ob": nearby,
            "candle_patterns": pats, "is_consolidating": (35 < rsi < 65) and (atr_ratio < 0.7), "ema10_dist": d10,
            "ema21_dist": d21, "is_pullback": (min(d10, d21) < 2.0) and (close > e50), "stop_dist": stop,
            "sl_pct": (stop / close * 100) if close > 0 else 0, "rr_ratio": rr,
            "risk_score": min(100, int((hist_vol / 50) * 100)) if hist_vol > 0 else 50}


def _old_sltp(df, cap, risk, mult, factor=1.0):
    close, high, low = (pd.to_numeric(df[k], errors="coerce") for k in ("Close", "High", "Low"))
    price = float(close.dropna().iloc[-1])
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().dropna().iloc[-1])
    kijun = float((high.rolling(26).max() + low.rolling(26).min()).iloc[-1] / 2)
    sl = max(price - atr * mult, kijun)
    dist = price - sl
    risk_amt = cap * (risk / 100)
    shares = int(risk_amt / dist) if dist > 0 else 0
    if factor < 1:
        shares = int(shares * factor)
        risk_amt = shares * dist
    return {"sl": sl, "tp_2r": price + dist * 2, "tp_3r": price + dist * 3, "shares": shares, "risk_amt": risk_amt,
            "pos_val": shares * price, "kijun": kijun, "ema10": float(close.ewm(span=10).mean().iloc[-1])}


def _same(a, b):
    for k, v in b.items():
        if isinstance(v, float):
            assert a[k] == pytest.approx(v, rel=1e-12, abs=1e-12), k
        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            assert a[k] == pytest.approx(v, rel=1e-12), k
        elif isinstance(v, list):                       # mönsternamn: nyare pytest godtar inte text i approx
            assert a[k] == v, k
        else:
            assert a[k] == v, k


CASES = [(0, 0.0008, 1, 0.0005, 2, 0.001), (3, -0.0015, 4, -0.001, 5, -0.002), (6, 0.0, 7, 0.0, 8, 0.0),
         (9, 0.002, 10, -0.001, 11, 0.0004), (12, -0.0005, 13, 0.001, 14, -0.0015)]


# ── Regimen, checklistan, grindarna och SL/TP ger samma tal som sidan gav ─────
@pytest.mark.parametrize("s1,d1,s2,d2,s3,d3", CASES)
def test_regime_matches_the_frozen_page_formulas(s1, d1, s2, d2, s3, d3):
    spy, sec, stk = _walk(s1, d1), _walk(s2, d2), _walk(s3, d3)
    new, old = wrs.regime(spy, sec, stk, timestamp="t"), _old_regime(spy, sec, stk)
    assert new.pop("timestamp") == "t" and set(new) == set(old)
    _same(new, old)
    assert new["total"] == new["mkt_score"] + new["sec_score"] + new["stk_score"] + new["ichi_score"]
    assert 0 <= new["mkt_score"] <= 30 and 0 <= new["sec_score"] <= 30 and 0 <= new["stk_score"] <= 50 and 0 <= new["ichi_score"] <= 15


@pytest.mark.parametrize("seed,drift", [(0, 0.0008), (3, -0.0015), (6, 0.0), (21, 0.003), (33, -0.003), (44, 0.0002)])
def test_checklist_gates_and_sltp_match_the_frozen_page_formulas(seed, drift):
    stk = _walk(seed, drift)
    data = wrs.regime(_walk(100), _walk(101), stk)
    chk = wrs.checklist(data, stk)
    _same(chk, _old_checklist(data, stk, wrs.ATR_MULT))
    gs = wrs.gates(data, chk)
    assert [g["passed"] for g in gs[:7]] == [data["total"] >= 50, not chk["is_consolidating"], chk["has_nearby_ob"],
                                             chk["is_pullback"], chk["has_candle"], chk["vol_ratio"] >= 1.0,
                                             chk["rr_ratio"] >= 2.0]
    for cap, risk, factor in ((100000, wrs.RISK_PCT, 1.0), (250000, 1.0, 0.5), (50000, 3.5, 1.0)):
        _same(wrs.sltp(stk, cap, risk, wrs.ATR_MULT, factor), _old_sltp(stk, cap, risk, wrs.ATR_MULT, factor))


def test_regime_without_data_is_all_zero():
    d = wrs.regime(None, None, None)
    assert d["total"] == 0 and d["close"] == 0 and d["has_entry"] is False and d["ema_stack"] is False
    short = _walk(1, n=40)
    assert wrs.score_market(short) == 0 and wrs.score_stock(short) is None
    assert wrs.regime(_walk(2), None, short)["sec_score"] == 0 and wrs.regime(_walk(2), None, short)["stk_score"] == 0
    assert wrs.sltp(short.head(10), 100000) is None


def test_wolf_constants_follow_the_strategy():
    from strategies.wolf import DEFAULT_PARAMS
    assert wrs.ATR_MULT == float(DEFAULT_PARAMS["atr_mult"]) and wrs.RISK_PCT == float(DEFAULT_PARAMS["risk_pct"]) * 100
    assert wrs.TOTAL_MAX == 125 == sum(wrs.LAYER_MAX.values()) and wrs.GATE_MIN == 40


@pytest.mark.parametrize("total,expected", [(125, "BULL REGIME"), (85, "BULL REGIME"), (84, "MODERATE BULL"), (65, "MODERATE BULL"),
                                            (64, "NEUTRAL"), (45, "NEUTRAL"), (44, "BEAR / AVOID"), (0, "BEAR / AVOID")])
def test_label_thresholds(total, expected):
    lbl, col = wrs.label(total)
    assert lbl == expected and col.startswith("#")


def test_eleven_gates_four_manual_and_the_summary_colours():
    stk = _walk(0)
    data = wrs.regime(_walk(100), _walk(101), stk)
    gs = wrs.gates(data, wrs.checklist(data, stk))
    assert len(gs) == 11 and [g["rule"][:2].strip(".") for g in gs] == [str(i) for i in range(1, 12)]
    assert [g["passed"] is None for g in gs] == [False] * 7 + [True] * 4
    assert f"{wrs.RISK_PCT:g}%" in gs[7]["rule"] and f"{wrs.ATR_MULT:g}×ATR" in gs[10]["rule"]
    s = wrs.gate_summary(gs)
    assert s["measurable"] == 7 and s["manual"] == 4 and 0 <= s["passed"] <= 7
    mk = lambda flags: [{"rule": "x", "passed": f, "value": ""} for f in flags]  # noqa: E731
    assert wrs.gate_summary(mk([True] * 6 + [False] + [None] * 4))["color"] == "#2d8a4e"      # en saknas → grönt
    assert wrs.gate_summary(mk([True] * 4 + [False] * 3 + [None] * 4))["color"] == "#d4943a"  # minst hälften → gult
    assert wrs.gate_summary(mk([True] * 3 + [False] * 4 + [None] * 4))["color"] == "#c44545"
    na = wrs.gates_unavailable(data, KeyError("rsi"))
    assert len(na) == 11 and [g["passed"] for g in na[1:7]] == [False] * 6 and all(g["passed"] is None for g in na[7:])
    assert "KeyError" in na[2]["value"] and na[0]["passed"] == (data["total"] >= 50)


def test_sltp_sizes_from_the_risk_budget_and_halves_on_high_market_risk():
    frames = [_walk(s, 0.002) for s in range(40)]
    above = [f for f in frames if wrs.sltp(f, 100000, 2.0)["sl_dist"] > 0]             # kursen över Kijun
    below = [f for f in frames if wrs.sltp(f, 100000, 2.0)["sl_dist"] <= 0]            # Kijun över kursen
    assert above and below
    p = wrs.sltp(above[0], 100000, 2.0)
    assert p["sl"] == max(p["price"] - p["atr_stop"], p["kijun"]) and p["sl_dist"] == p["price"] - p["sl"]
    assert p["shares"] == int(2000 / p["sl_dist"]) and p["risk_amt"] == 2000 and p["tp_2r"] == p["price"] + 2 * p["sl_dist"]
    assert p["pos_pct"] == pytest.approx(p["pos_val"] / 1000)
    half = wrs.sltp(above[0], 100000, 2.0, size_factor=0.5)
    assert half["shares"] == int(p["shares"] * 0.5) and half["risk_amt"] == pytest.approx(half["shares"] * p["sl_dist"])
    q = wrs.sltp(below[0], 100000, 2.0)                                                   # sidans regel: ingen position
    assert q["sl"] == q["kijun"] and q["shares"] == 0 and q["pos_val"] == 0


# ── Wolf Regime-sidan läser modulen — samma tal på skärmen ───────────────────
def _page():
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.environ["WRS_TEST_ROOT"])
    from tabs.regime import tab_regime
    tab_regime()


def _frames():
    return {"SPY": _walk(100, 0.0006), "XOM": _walk(7, 0.001), "AAPL": _walk(8, -0.001)}


def test_wolf_regime_page_shows_the_module_numbers(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import market_prices
    import market_risk_gate
    import tabs.regime as page
    from utils.presets import SECTOR_ETF_LIST, etf_from_display
    frames = _frames()

    def fake_download(ticker, period="1y", progress=False, auto_adjust=True, **_k):
        return frames.get(ticker, _walk(abs(hash(ticker)) % 97, 0.0004)).copy()

    monkeypatch.setattr(page.yf, "download", fake_download)
    monkeypatch.setattr(market_prices, "ohlcv", lambda t, p="1y": fake_download(t))
    monkeypatch.setattr(market_risk_gate, "for_ticker", lambda t, evaluator=None: None)
    monkeypatch.setenv("WRS_TEST_ROOT", ROOT)
    at = AppTest.from_function(_page, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    sector = etf_from_display(SECTOR_ETF_LIST[0])
    want = wrs.regime(fake_download("SPY"), fake_download(sector), fake_download("XOM"))
    html = " ".join(m.value for m in at.markdown)
    assert "REGIME SCORE" in html and f">{want['total']}</div>" in html and wrs.label(want["total"])[0] in html
    assert "WOLF ENTRY CHECKLIST" in html and "SWING TRADING GATES" in html and "MÄTBARA GRINDAR" in html
    assert "SL / TP KALKYLATOR — WOLF" in html and "STOP LOSS" in html
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["CLOSE PRICE"] == f"{want['close']:.2f}" and metrics["RSI (14)"] == f"{want['rsi']:.1f}"
    chk = wrs.checklist(want, fake_download("XOM"))
    gs = wrs.gate_summary(wrs.gates(want, chk))
    assert f"{gs['passed']}/{gs['measurable']} MÄTBARA GRINDAR" in html and f"ATR ratio: <b>{chk['atr_ratio']:.2f}x" in html
    p = wrs.sltp(fake_download("XOM"), 100000, wrs.RISK_PCT)
    assert f"{p['sl']:.2f}" in html and f"{p['shares']} aktier" in html
    at.text_input(key="reg_ticker").set_value("AAPL").run()                      # ny ticker → ny hämtning
    assert not at.exception, at.exception
    want2 = wrs.regime(fake_download("SPY"), fake_download(sector), fake_download("AAPL"))
    assert f">{want2['total']}</div>" in " ".join(m.value for m in at.markdown)
    assert {m.label: m.value for m in at.metric}["CLOSE PRICE"] == f"{want2['close']:.2f}"
