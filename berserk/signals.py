"""
berserk/signals.py — 🪓 BERSERK:s tre setups som dagliga serier (allt kausalt).

Varje värde en dag bygger bara på data t.o.m. den dagen (EMA adjust=False,
rullande fönster). Råvarans drivare följer aktiens kalender med senast KÄNDA
värde (ffill). Signalen bedöms på stängd dag; entry sker nästa dags öppning
(backtest.py).

  S1 DIVERGENS      råvaran stark (över SMA50, positiv 63 d) men aktien har halkat
                    efter ≥ 10 procentenheter på 63 d — och vänder: stängning över
                    gårdagens high, grön dag, relativ volym ≥ 1,2
  S2 CYKELVÄNDNING  råvaran i baisse (≥ 30 % under 52-veckorshögsta eller i nedre
                    20 % av sitt 5-årsintervall någon gång senaste 126 d) och vänder
                    (över EMA50, EMA20 stigande); aktien hatad (≥ 40 % under sin topp
                    senaste 126 d; ETF: ≥ 30 %) och bryter 20-dagarshögsta med
                    relativ volym ≥ 1,5
  S3 SNAPBACK       råvaran över SMA200 (utan drivare: aktien över stigande SMA200);
                    RSI(2) < 10 och stängning under nedre Bollinger (20, 2)

Gemensamma grindar: likviditet (snittomsättning 20 d) och marknaden (indexet
över SMA200) — se backtest.Config.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

S1, S2, S3 = "S1 Divergens", "S2 Cykelvändning", "S3 Snapback"
SETUPS = (S1, S2, S3)
PRIORITY = {S2: 3, S1: 2, S3: 1}          # flera setups samma dag → sällsyntast och störst först

# Parametrar — satta i förväg, inte optimerade (känslighet testas i ett senare steg)
DIV_LAG = -0.10                  # aktien minst 10 procentenheter efter råvaran på 63 d
RVOL_S1, RVOL_S2 = 1.2, 1.5
BEAR_DD, HATED_DD, HATED_DD_ETF = -0.30, -0.40, -0.30
RANGE_LOW = 0.20                 # nedre 20 % av femårsintervallet
LOOKBACK_BEAR = 126              # "någon gång senaste halvåret"
RSI2_MAX, BB_N, BB_K = 10.0, 20, 2.0
RET_N = 63
JUMP_MAX, JUMP_BLOCK_DAYS = 0.60, 20     # datavakt: dagshopp > 60 % (ojusterad split/utdelning/datafel) spärrar 20 dagar


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(dn.notna())


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, lo, c = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def align(s: Optional[pd.Series], idx) -> Optional[pd.Series]:
    """Senast KÄNDA värde på aktiens dagar — aldrig ett senare."""
    if s is None or len(s) == 0:
        return None
    s = s.astype(float)
    if getattr(s.index, "tz", None) is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def jump_block(close: pd.Series) -> pd.Series:
    """True de dagar då kursen hoppat mer än JUMP_MAX på en dag inom de senaste JUMP_BLOCK_DAYS dagarna —
    typiskt en ojusterad split, extrautdelning eller ett datafel. Indikatorerna är då opålitliga."""
    jump = (close.astype(float).pct_change().abs() > JUMP_MAX).astype(float)
    return jump.rolling(JUMP_BLOCK_DAYS + 1, min_periods=1).max() > 0


def driver_frame(d: pd.Series) -> pd.DataFrame:
    """Råvarans mått på dess egen kalender (innan den läggs på aktiens dagar)."""
    d = d.astype(float).dropna()
    out = pd.DataFrame(index=d.index)
    out["close"] = d
    out["sma50"], out["sma200"] = d.rolling(50).mean(), d.rolling(200).mean()
    out["ema20"], out["ema50"] = _ema(d, 20), _ema(d, 50)
    out["ret63"] = d / d.shift(RET_N) - 1
    dd = d / d.rolling(252, min_periods=60).max() - 1
    lo5, hi5 = d.rolling(1260, min_periods=252).min(), d.rolling(1260, min_periods=252).max()
    pos = (d - lo5) / (hi5 - lo5).replace(0, np.nan)
    out["bear"] = ((dd <= BEAR_DD) | (pos <= RANGE_LOW)).astype(float).rolling(LOOKBACK_BEAR, min_periods=1).max() > 0
    out["turn"] = (d > out["ema50"]) & (out["ema20"] > out["ema20"].shift(5))
    out["strong"] = (d > out["sma50"]) & (out["ret63"] > 0)
    out["uptrend"] = d > out["sma200"]
    out["above_ema50"] = d > out["ema50"]
    out["above_sma50"] = d > out["sma50"]
    return out


def frame(stock: pd.DataFrame, driver: Optional[pd.Series] = None, is_etf: bool = False,
          market: Optional[pd.Series] = None) -> pd.DataFrame:
    """Alla mått och de tre setupens signaler per dag för en aktie (eller ETF)."""
    idx = stock.index
    o, h, lo, c = (stock[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    v = stock["Volume"].astype(float)
    f = pd.DataFrame(index=idx)
    f["atr"] = atr(stock)
    f["ema20"], f["ema50"] = _ema(c, 20), _ema(c, 50)
    f["sma5"], f["sma200"] = c.rolling(5).mean(), c.rolling(200).mean()
    f["rsi2"] = rsi(c, 2)
    mid, sd = c.rolling(BB_N).mean(), c.rolling(BB_N).std(ddof=0)
    f["bb_low"] = mid - BB_K * sd
    f["rvol"] = v / v.shift(1).rolling(20).mean()
    f["turnover20"] = (c * v).rolling(20).mean()
    f["ret63"] = c / c.shift(RET_N) - 1
    f["dd252"] = c / c.rolling(252, min_periods=60).max() - 1
    f["high20_prev"] = h.shift(1).rolling(20).max()
    f["green"] = c > o
    f["up_close"] = c > h.shift(1)
    f["data_jump"] = jump_block(c)
    if market is not None:
        m = align(market, idx)
        f["market_ok"] = m > m.rolling(200).mean() if m is not None else True
    else:
        f["market_ok"] = True

    df = None
    if driver is not None and len(driver.dropna()) > 0:
        dfr = driver_frame(driver)
        df = pd.DataFrame({k: align(dfr[k].astype(float), idx) for k in dfr.columns}, index=idx)
        for k in ("bear", "turn", "strong", "uptrend", "above_ema50", "above_sma50"):
            df[k] = df[k].fillna(0).astype(bool)
    f["has_driver"] = df is not None
    if df is not None:
        f["d_close"], f["d_ret63"] = df["close"], df["ret63"]
        f["d_above_sma50"], f["d_above_ema50"] = df["above_sma50"], df["above_ema50"]
        f["divergence"] = f["ret63"] - df["ret63"]
    else:
        f["d_close"] = f["d_ret63"] = f["divergence"] = np.nan
        f["d_above_sma50"] = f["d_above_ema50"] = True

    # S1 Divergens — bara producentbolag med drivare (en ETF ÄR råvaran)
    if df is not None and not is_etf:
        f["s1_setup"] = df["strong"] & (f["divergence"] <= DIV_LAG)          # allt utom triggern (skannerns BEVAKA)
        f[S1] = f["s1_setup"] & f["up_close"] & f["green"] & (f["rvol"] >= RVOL_S1)
    else:
        f["s1_setup"] = False
        f[S1] = False
    # S2 Cykelvändning — ETF:en är sin egen drivare
    if df is not None or is_etf:
        dd = df if df is not None else driver_frame(c)
        hated = (f["dd252"] <= (HATED_DD_ETF if is_etf else HATED_DD)).astype(float) \
            .rolling(LOOKBACK_BEAR, min_periods=1).max() > 0
        bear = dd["bear"] if df is not None else dd["bear"].reindex(idx).fillna(False)
        turn = dd["turn"] if df is not None else dd["turn"].reindex(idx).fillna(False)
        f["s2_setup"] = bear & turn & hated
        f[S2] = f["s2_setup"] & (c > f["high20_prev"]) & (f["rvol"] >= RVOL_S2)
    else:
        f["s2_setup"] = False
        f[S2] = False
    # S3 Snapback
    trend = df["uptrend"] if df is not None else ((c > f["sma200"]) & (f["sma200"] > f["sma200"].shift(20)))
    f["s3_trend"] = trend
    f[S3] = trend & (f["rsi2"] < RSI2_MAX) & (c < f["bb_low"])
    for k in SETUPS + ("s1_setup", "s2_setup", "s3_trend"):
        f[k] = f[k].fillna(False).astype(bool)
    return f
