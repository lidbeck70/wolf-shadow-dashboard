"""
wolf_regime_score.py — Wolfs regimpoäng 0–125 och entry-checklistan, utbrutna ur tabs/regime.py.

Samma formler och trösklar som ODINS VISDOM → Marknad → Arc Regime → Wolf Regime visar: fyra lager (Market 30 · Sector 30 ·
Stock 50 · Ichimoku 15), etiketten BULL REGIME ≥ 85 / MODERATE BULL ≥ 65 / NEUTRAL ≥ 45 / BEAR, entry-
checklistan (trend, volatilitet, momentum, candlestick, order blocks), de elva swinggrindarna och SL/TP-
kalkylatorn. Koden flyttades hit oförändrad (tests/test_wolf_regime_score.py låser talen mot en kopia av
de gamla formlerna) så att Wolf Regime-sidan och Thors analysblock visar exakt samma siffror.

Rena funktioner över OHLCV-ramar — inget nätverk, ingen Streamlit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Wolfs riktiga stoppavstånd och risk (strategies/wolf.py) — checklistan, grindarna och SL/TP räknar på dem.
try:
    from strategies.wolf import DEFAULT_PARAMS as _WOLF_P
    ATR_MULT = float(_WOLF_P["atr_mult"])
    RISK_PCT = float(_WOLF_P["risk_pct"]) * 100
except Exception:  # pragma: no cover
    ATR_MULT, RISK_PCT = 2.5, 2.0

LAYER_MAX = {"mkt_score": 30, "sec_score": 30, "stk_score": 50, "ichi_score": 15}
TOTAL_MAX = 125
GATE_MIN = 40                      # REGIME GATE: minsta poäng för entry (strategies/wolf.py entry_min_score)
LABELS = ((85, "BULL REGIME", "#2d8a4e"), (65, "MODERATE BULL", "#d4943a"), (45, "NEUTRAL", "#d4943a"))
BEAR = ("BEAR / AVOID", "#c44545")


# ── Indikatorerna (ordagrant ur tabs/regime.py) ─────────────────────────────
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_v = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_v)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_v)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    return dx.ewm(span=period, adjust=False).mean()


def ichimoku(df):
    hi9, lo9 = df["High"].rolling(9).max(), df["Low"].rolling(9).min()
    hi26, lo26 = df["High"].rolling(26).max(), df["Low"].rolling(26).min()
    hi52, lo52 = df["High"].rolling(52).max(), df["Low"].rolling(52).min()
    tenkan = (hi9 + lo9) / 2
    kijun = (hi26 + lo26) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((hi52 + lo52) / 2).shift(26)
    chikou = df["Close"].shift(-26)
    return tenkan, kijun, span_a, span_b, chikou


# ── Lagren ──────────────────────────────────────────────────────────────────
def _score_index(df) -> int:
    if df is None or len(df) < 60:
        return 0
    c = df["Close"]
    score = 0
    e50, e200 = ema(c, 50).iloc[-1], ema(c, 200).iloc[-1]
    last = c.iloc[-1]
    r = rsi(c).iloc[-1]
    if last > e200:
        score += 10
    if e50 > e200:
        score += 8
    if last > e50:
        score += 7
    if r > 50:
        score += 5
    return min(score, 30)


def score_market(spy_df) -> int:
    """MARKET 0–30: SPY över EMA200 (10), EMA50 > EMA200 (8), över EMA50 (7), RSI > 50 (5)."""
    return _score_index(spy_df)


def score_sector(sec_df) -> int:
    """SECTOR 0–30: samma regler på sektor-ETF:en."""
    return _score_index(sec_df)


def score_stock(stk_df) -> Optional[dict]:
    """STOCK 0–50 och ICHIMOKU 0–15 plus nivåerna (entry = senaste stängning, SL 1,5 × ATR, TP 2R/3R)."""
    if stk_df is None or len(stk_df) < 60:
        return None
    c = stk_df["Close"]
    e10, e21, e50, e200 = ema(c, 10), ema(c, 21), ema(c, 50), ema(c, 200)
    rsi_s, atr_s = rsi(c), atr(stk_df)
    tenkan, kijun, span_a, span_b, _chikou = ichimoku(stk_df)
    adx_s = adx(stk_df)
    last = c.iloc[-1]
    rsi_val, atr_val = float(rsi_s.iloc[-1]), float(atr_s.iloc[-1])
    adx_val = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0.0
    e10_last, e21_last, e50_last, e200_last = (float(x.iloc[-1]) for x in (e10, e21, e50, e200))

    stk_score = 0
    if last > e200_last:
        stk_score += 10
    if e50_last > e200_last:
        stk_score += 8
    if last > e50_last:
        stk_score += 8
    ema_stack = (e10_last > e21_last > e50_last > e200_last)
    if ema_stack:
        stk_score += 10
    ema_trend = last > e21_last
    if ema_trend:
        stk_score += 7
    if 40 < rsi_val < 75:
        stk_score += 7
    stk_score = min(stk_score, 50)

    ichi_score = 0
    sa = float(span_a.iloc[-1]) if not pd.isna(span_a.iloc[-1]) else 0
    sb = float(span_b.iloc[-1]) if not pd.isna(span_b.iloc[-1]) else 0
    tk = float(tenkan.iloc[-1]) if not pd.isna(tenkan.iloc[-1]) else 0
    kj = float(kijun.iloc[-1]) if not pd.isna(kijun.iloc[-1]) else 0
    cloud_top, cloud_bot = max(sa, sb), min(sa, sb)
    if last > cloud_top:
        ichi_score += 6
    elif last > cloud_bot:
        ichi_score += 2
    if tk > kj:
        ichi_score += 5
    if sa > sb:
        ichi_score += 4
    ichi_score = min(ichi_score, 15)

    sl_dist = 1.5 * atr_val
    sl_level = last - sl_dist
    return {"stock_score": stk_score, "ichi_score": ichi_score, "close": float(last), "rsi": rsi_val, "atr": atr_val,
            "adx": round(adx_val, 1), "ema_stack": ema_stack, "ema_trend": ema_trend,
            "has_entry": ema_stack and ema_trend and 45 < rsi_val < 70, "entry_zone": last, "sl_level": sl_level,
            "tp1_2R": last + 2.0 * sl_dist, "tp2_3R": last + 3.0 * sl_dist, "ema10": e10_last, "ema21": e21_last,
            "ema50": e50_last, "ema200": e200_last, "kijun": kj}


def regime(spy_df, sec_df, stk_df, timestamp: str = "") -> dict:
    """Regimdatan som Wolf Regime-sidan lägger i session_state: lagren, totalen och nivåerna."""
    mkt, sec = score_market(spy_df), score_sector(sec_df)
    r = score_stock(stk_df)
    stk, ichi = (r["stock_score"], r["ichi_score"]) if r else (0, 0)

    def g(k, d=0):
        return r[k] if r else d

    return {"mkt_score": mkt, "sec_score": sec, "stk_score": stk, "ichi_score": ichi, "total": mkt + sec + stk + ichi,
            "close": g("close"), "rsi": g("rsi"), "adx": g("adx"), "ema_stack": g("ema_stack", False),
            "ema_trend": g("ema_trend", False), "has_entry": g("has_entry", False), "entry_zone": g("entry_zone"),
            "sl_level": g("sl_level"), "tp1": g("tp1_2R"), "tp2": g("tp2_3R"), "atr": g("atr"), "ema10": g("ema10"),
            "ema21": g("ema21"), "ema50": g("ema50"), "ema200": g("ema200"), "kijun": g("kijun"), "timestamp": timestamp}


def label(total) -> tuple:
    """(etikett, färg) för totalen — BULL REGIME ≥ 85, MODERATE BULL ≥ 65, NEUTRAL ≥ 45, annars BEAR / AVOID."""
    for lo, lbl, col in LABELS:
        if total >= lo:
            return lbl, col
    return BEAR


# ── Entry-checklistan ───────────────────────────────────────────────────────
def checklist(data: dict, stk_df, atr_mult: float = ATR_MULT) -> dict:
    """Trend, volatilitet, momentum, candlestick och order blocks ur regimdatan och aktiens dagsdata —
    talen som checklistkorten och grindarna 2–7 läser."""
    close, r, a = float(data.get("close") or 0), float(data.get("rsi") or 0), float(data.get("atr") or 0)
    e10, e21, e50, e200 = (float(data.get(k) or 0) for k in ("ema10", "ema21", "ema50", "ema200"))
    stack = bool(data.get("ema_stack"))
    if stack:
        trend_dir, trend_clr = "BULL", "#2d8a4e"
    elif close > e50:
        trend_dir, trend_clr = "NEUTRAL-BULL", "#d4943a"
    elif close > e200:
        trend_dir, trend_clr = "NEUTRAL", "#d4943a"
    else:
        trend_dir, trend_clr = "BEAR", "#c44545"
    rsi_state = "OVERBOUGHT" if r > 70 else "OVERSOLD" if r < 30 else "NEUTRAL" if 40 <= r <= 60 else "ACTIVE"

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
        high, low, cls = stk_df["High"], stk_df["Low"], stk_df["Close"]
        tr = pd.concat([high - low, (high - cls.shift()).abs(), (low - cls.shift()).abs()], axis=1).max(axis=1)
        atr_series = tr.ewm(com=13, adjust=False).mean()
        atr_avg_50 = atr_series.tail(50).mean()
        if atr_avg_50 > 0:
            atr_ratio = float(atr_series.iloc[-1] / atr_avg_50)

    bull_obs, bear_obs = [], []
    if stk_df is not None and len(stk_df) >= 50:
        ob_atr = a if a > 0 else 1.0
        recent = stk_df.tail(100) if len(stk_df) >= 100 else stk_df
        for i in range(2, len(recent)):
            cur, prev = recent.iloc[i], recent.iloc[i - 1]
            body = abs(float(cur["Close"]) - float(cur["Open"]))
            if float(cur["Close"]) > float(cur["Open"]) and body > ob_atr * 1.5 and float(prev["Close"]) < float(prev["Open"]):
                bull_obs.append(float(cur["Low"]))
            if float(cur["Close"]) < float(cur["Open"]) and body > ob_atr * 1.5 and float(prev["Close"]) > float(prev["Open"]):
                bear_obs.append(float(cur["High"]))
        bull_obs, bear_obs = bull_obs[-5:], bear_obs[-5:]
    nearest_bull = min(bull_obs, key=lambda x: abs(x - close)) if bull_obs else None
    nearest_bear = min(bear_obs, key=lambda x: abs(x - close)) if bear_obs else None
    has_nearby_ob = bool(nearest_bull and close and abs(nearest_bull - close) / close < 0.03) or \
        bool(nearest_bear and close and abs(nearest_bear - close) / close < 0.03)

    patterns = []
    if stk_df is not None and len(stk_df) >= 3:
        cc, cp = stk_df.iloc[-1], stk_df.iloc[-2]
        cbody = float(cc["Close"]) - float(cc["Open"])
        crange = float(cc["High"]) - float(cc["Low"])
        if crange > 0:
            lower_wick = min(float(cc["Open"]), float(cc["Close"])) - float(cc["Low"])
            if lower_wick > abs(cbody) * 2 and cbody > 0:
                patterns.append("Hammer (Bull)")
            if (float(cp["Close"]) < float(cp["Open"]) and float(cc["Close"]) > float(cc["Open"])
                    and float(cc["Close"]) > float(cp["Open"]) and float(cc["Open"]) < float(cp["Close"])):
                patterns.append("Bullish Engulfing")
            if (float(cp["Close"]) > float(cp["Open"]) and float(cc["Close"]) < float(cc["Open"])
                    and float(cc["Close"]) < float(cp["Open"]) and float(cc["Open"]) > float(cp["Close"])):
                patterns.append("Bearish Engulfing")
            if abs(cbody) < crange * 0.1:
                patterns.append("Doji")

    ema10_dist = abs(close - e10) / close * 100 if close > 0 else 99
    ema21_dist = abs(close - e21) / close * 100 if close > 0 else 99
    stop_dist = a * atr_mult
    rr = 0.0
    if nearest_bear and stop_dist > 0:
        rr = abs(nearest_bear - close) / stop_dist
    return {"trend_dir": trend_dir, "trend_color": trend_clr, "stack": stack, "rsi_state": rsi_state,
            "vol_ratio": vol_ratio, "hist_vol": hist_vol, "atr_ratio": atr_ratio, "bull_obs": bull_obs,
            "bear_obs": bear_obs, "nearest_bull_ob": nearest_bull, "nearest_bear_ob": nearest_bear,
            "has_nearby_ob": has_nearby_ob, "candle_patterns": patterns, "has_candle": bool(patterns),
            "pattern_name": ", ".join(patterns) if patterns else "Inget mönster",
            "is_consolidating": (35 < r < 65) and (atr_ratio < 0.7), "ema10_dist": ema10_dist, "ema21_dist": ema21_dist,
            "is_pullback": (min(ema10_dist, ema21_dist) < 2.0) and (close > e50), "stop_dist": stop_dist,
            "sl_pct": (stop_dist / close * 100) if close > 0 else 0, "rr_ratio": rr,
            "risk_score": min(100, int((hist_vol / 50) * 100)) if hist_vol > 0 else 50}


def gates(data: dict, chk: dict, atr_mult: float = ATR_MULT, risk_pct: float = RISK_PCT) -> list:
    """Wolfs elva swinggrindar: 1–7 mätbara ur regimdatan och checklistan, 8–11 manuella (passed None)."""
    total = data.get("total", 0)
    r, kj, e10 = float(data.get("rsi") or 0), float(data.get("kijun") or 0), float(data.get("ema10") or 0)
    nearest = chk.get("nearest_bull_ob") or chk.get("nearest_bear_ob") or 0.0
    return [
        {"rule": "1. Trendriktning", "passed": total >= 50, "value": f"Score: {total}/125"},
        {"rule": "2. Ej konsolidering", "passed": not chk["is_consolidating"], "value": f"RSI: {r:.0f}"},
        {"rule": "3. Key level (OB)", "passed": chk["has_nearby_ob"],
         "value": f"Närmaste OB: {nearest:.2f}" if chk["has_nearby_ob"] else "Ingen OB nära"},
        {"rule": "4. Pullback entry", "passed": chk["is_pullback"], "value": f"Avstånd EMA10: {chk['ema10_dist']:.1f}%"},
        {"rule": "5. Candle trigger", "passed": chk["has_candle"], "value": chk["pattern_name"]},
        {"rule": "6. Volymbekräftelse", "passed": chk["vol_ratio"] >= 1.0, "value": f"Vol ratio: {chk['vol_ratio']:.1f}x"},
        {"rule": "7. R:R ≥ 1:2", "passed": chk["rr_ratio"] >= 2.0, "value": f"R:R 1:{chk['rr_ratio']:.1f}"},
        {"rule": f"8. Max {risk_pct:g}% risk (Wolf)", "passed": None,
         "value": f"SL {atr_mult:g}×ATR: {chk['stop_dist']:.2f} ({chk['sl_pct']:.1f}%) — bedöm själv"},
        {"rule": "9. SL → BE efter HH", "passed": None, "value": "Post-entry regel — manuell"},
        {"rule": "10. Max 2 förluster/dag", "passed": None, "value": "Disciplin — manuell"},
        {"rule": f"11. Kijun trail + {atr_mult:g}×ATR", "passed": None, "value": f"Kijun: {kj:.2f}, EMA10: {e10:.2f} — exit-regel"},
    ]


def gates_unavailable(data: dict, exc: Exception, atr_mult: float = ATR_MULT, risk_pct: float = RISK_PCT) -> list:
    """Grindarna när checklistan inte gick att räkna: 2–7 underkända (datafel är inget ja), 8–11 manuella."""
    na = f"Data ej tillgänglig ({type(exc).__name__})"
    total = data.get("total", 0)
    measured = ("2. Ej konsolidering", "3. Key level (OB)", "4. Pullback entry", "5. Candle trigger", "6. Volymbekräftelse",
                "7. R:R ≥ 1:2")
    return ([{"rule": "1. Trendriktning", "passed": total >= 50, "value": f"Score: {total}/125"}]
            + [{"rule": r, "passed": False, "value": na} for r in measured]
            + [{"rule": f"8. Max {risk_pct:g}% risk (Wolf)", "passed": None, "value": "Position sizing — manuell"},
               {"rule": "9. SL → BE efter HH", "passed": None, "value": "Post-entry — manuell"},
               {"rule": "10. Max 2 förluster/dag", "passed": None, "value": "Disciplin — manuell"},
               {"rule": f"11. Kijun trail + {atr_mult:g}×ATR", "passed": None, "value": "Exit-regel — manuell"}])


def gate_summary(gs: list) -> dict:
    """{passed, measurable, manual, color}: grönt = högst en mätbar grind saknas, gult = minst hälften."""
    measurable = [g for g in gs if g["passed"] is not None]
    passed = sum(1 for g in measurable if g["passed"])
    n = len(measurable)
    color = "#2d8a4e" if passed >= n - 1 else ("#d4943a" if passed * 2 >= n else "#c44545")
    return {"passed": passed, "measurable": n, "manual": len(gs) - n, "color": color}


# ── SL/TP-kalkylatorn ───────────────────────────────────────────────────────
def sltp(stk_df, capital: float, risk_pct: float = RISK_PCT, atr_mult: float = ATR_MULT, size_factor: float = 1.0) -> Optional[dict]:
    """Stoppen = max(kurs − atr_mult × ATR14 (SMA), Kijun), mål 2R/3R, antal aktier ur riskbudgeten
    (halverad när 🌩️ Marknadsrisk är HÖG: size_factor 0,5). None under 20 dagars data."""
    if stk_df is None or len(stk_df) < 20:
        return None
    close = pd.to_numeric(stk_df["Close"], errors="coerce")
    high = pd.to_numeric(stk_df["High"], errors="coerce")
    low = pd.to_numeric(stk_df["Low"], errors="coerce")
    price = float(close.dropna().iloc[-1])
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().dropna().iloc[-1])
    atr_stop = atr14 * atr_mult
    ema10 = float(close.ewm(span=10).mean().iloc[-1])
    kijun = float((high.rolling(26).max() + low.rolling(26).min()).iloc[-1] / 2)
    sl = max(price - atr_stop, kijun)
    sl_dist = price - sl
    risk_amt = capital * (risk_pct / 100)
    shares = int(risk_amt / sl_dist) if sl_dist > 0 else 0
    if size_factor < 1:
        shares = int(shares * size_factor)
        risk_amt = shares * sl_dist
    pos_val = shares * price
    return {"price": price, "atr": atr14, "atr_stop": atr_stop, "ema10": ema10, "kijun": kijun, "sl": sl, "sl_dist": sl_dist,
            "tp_2r": price + sl_dist * 2, "tp_3r": price + sl_dist * 3, "risk_amt": risk_amt, "shares": shares,
            "pos_val": pos_val, "pos_pct": (pos_val / capital * 100) if capital > 0 else 0}
