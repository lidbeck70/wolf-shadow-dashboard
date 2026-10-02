"""
viking_backtest.py — backtest av Viking Nine (OVTLYR Nine + Viking Execution
+ exitmotorn), rapporterat i R. Utan look-ahead:

  * Varje indikator är en serie där värdet en dag bara bygger på data fram
    till och med den dagen (EMA med adjust=False, rullande fönster). Order
    blocks räknas om på kursdata t.o.m. signaldagen, bara de dagar då de kan
    avgöra signalen.
  * Signalen bedöms på signaldagens STÄNGNING (stängd candle). Entry sker på
    NÄSTA dags öppning — aldrig samma dags pris som signalen räknades på.
  * Marknads- och sektordata följer aktiens kalender med senast KÄNDA värde
    (ffill), aldrig ett senare.
  * Stoppen ligger intradag: low ≤ stopp → exit på stoppen (eller öppningen om
    dagen gappar under). Stängningsregler (SPY < EMA20, EMA10, breakeven,
    gap & crap, signal, bredd, F&G) ger exit på NÄSTA dags öppning.

Ingår inte (historiska data saknas): rapportspärren, max två förluster per
dag (portfölj), bearish block som exit. Universum och sektor är dagens —
överlevnads- och sektorbias. Allt står i resultatets notes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

import ovtlyr_nine as on
import viking_execution as vx
import viking_exit as vex

WARMUP_BARS = 60
NOTES = (
    "Entry på nästa dags öppning efter en stängd signaldag; stängningsregler ger exit på nästa öppning.",
    "Rapportspärren ingår inte — historiska rapportdatum saknas.",
    "Max två förluster per dag (portföljregel) ingår inte — varje ticker testas för sig.",
    "Bearish block används som entryfilter men inte som exit.",
    "Universum och sektor är dagens: aktier som försvunnit saknas (överlevnadsbias).",
    "OVTLYR Nine är WOLF APPROXIMATION — panelens definitioner, inte OVTLYR:s data.",
)


@dataclass
class Config:
    min_nine: int = 9
    require_volume: bool = vx.VOLUME_REQUIRED
    max_chase_pct: float = vx.MAX_CHASE_PCT
    min_rr: float = vx.MINIMUM_RR
    atr_mult: float = vx.ATR_STOP_MULT
    years: int = 3


@dataclass
class Trade:
    ticker: str
    signal_date: str
    entry_date: str
    entry: float
    stop: float
    risk: float
    nine: int
    exit_date: Optional[str] = None
    exit: Optional[float] = None
    exit_reason: str = ""
    r: Optional[float] = None
    days: int = 0
    open: bool = False


# ── Serierna (allt kausalt) ─────────────────────────────────────────────────
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _trend(c: pd.Series) -> pd.Series:
    return (_ema(c, 10) > _ema(c, 20)) & (c > _ema(c, 50))


def _signal(c: pd.Series) -> pd.Series:
    return c >= _ema(c, 20)


def _align(s: Optional[pd.Series], idx) -> pd.Series:
    """Senast KÄNDA värde på aktiens dagar — aldrig ett senare."""
    if s is None or len(s) == 0:
        return pd.Series(False, index=idx)
    return s.astype(float).reindex(s.index.union(idx)).ffill().reindex(idx).fillna(0).astype(bool)


def factor_frame(stock: pd.DataFrame, spy: Optional[pd.DataFrame], sector: Optional[pd.DataFrame],
                 breadth: Optional[pd.Series]) -> pd.DataFrame:
    """OVTLYR Nine:s åtta prisfaktorer per dag (order blocks räknas separat)."""
    idx = stock.index
    c = stock["Close"].astype(float)
    out = pd.DataFrame(index=idx)
    if spy is not None and len(spy):
        sc = spy["Close"].astype(float)
        out["market.trend"] = _align(_trend(sc), idx)
        out["market.signal"] = _align(_signal(sc), idx)
    else:
        out["market.trend"] = out["market.signal"] = False
    if breadth is not None and len(breadth):
        ok = (breadth >= on.BREADTH_MIN_PCT) & (breadth >= _ema(breadth, on.BREADTH_EMA))
        out["market.breadth"] = _align(ok, idx)
    else:
        out["market.breadth"] = False
    if sector is not None and len(sector) >= on.MIN_BARS:
        fg = on.fear_greed_series(sector)
        out["sector.fear_greed"] = _align((fg >= fg.shift(on.FG_LOOKBACK)) & (fg < on.FG_MAX), idx)
        e20 = _ema(sector["Close"].astype(float), 20)
        out["sector.breadth"] = _align((sector["Close"].astype(float) > e20)
                                       & (e20 > e20.shift(on.SECTOR_EMA_RISE_DAYS)), idx)
    else:
        out["sector.fear_greed"] = out["sector.breadth"] = False
    fg = on.fear_greed_series(stock)
    out["stock.trend"] = _trend(c)
    out["stock.signal"] = _signal(c)
    out["stock.fear_greed"] = (fg >= fg.shift(on.FG_LOOKBACK)) & (fg < on.FG_MAX)
    out["fg"] = fg
    return out


def execution_frame(stock: pd.DataFrame) -> pd.DataFrame:
    c, o, lo = (stock[k].astype(float) for k in ("Close", "Open", "Low"))
    r = vx.rsi(c)
    out = pd.DataFrame(index=stock.index)
    out["momentum"] = (r > vx.RSI_MIN) & (r > r.shift(1)) & (c > lo.shift(1))
    out["trend"] = (c > _ema(c, 10)) & (_ema(c, 10) > _ema(c, 20)) & (_ema(c, 20) > _ema(c, 50))
    out["candle"] = (c > o) & (c >= c.shift(1))
    v = stock["Volume"].astype(float)
    out["volume"] = v / v.shift(1).rolling(20).mean() >= vx.RELATIVE_VOLUME_MIN
    out["atr"] = vx.atr(stock)
    out["ema10"], out["ema20"] = _ema(c, 10), _ema(c, 20)
    return out


def _blocks_at(stock: pd.DataFrame, i: int) -> tuple:
    """(fritt?, ob_analysis) på kursdata t.o.m. dag i."""
    try:
        from ovtlyr.indicators.orderblocks import classify_price_vs_ob, detect_orderblocks
        part = stock.iloc[:i + 1]
        obs = detect_orderblocks(part)
        oa = classify_price_vs_ob(float(part["Close"].iloc[-1]), obs) if obs else {"signal_bias": "HOLD"}
    except Exception:
        oa = {"signal_bias": "HOLD"}
    bias = str(oa.get("signal_bias", "HOLD")).upper()
    return bias not in ("SELL", "REDUCE") and not oa.get("approaching_bearish", False), oa


# ── En ticker ───────────────────────────────────────────────────────────────
def backtest_ticker(ticker: str, stock: pd.DataFrame, spy: Optional[pd.DataFrame], sector: Optional[pd.DataFrame],
                    breadth: Optional[pd.Series], cfg: Config = Config(), start=None) -> dict:
    stock = stock.dropna(subset=["Open", "High", "Low", "Close"])
    n = len(stock)
    res = {"ticker": ticker, "trades": [], "signals": 0, "no_chase": 0, "low_rr": 0}
    if n < WARMUP_BARS + 2:
        return res
    f, x = factor_frame(stock, spy, sector, breadth), execution_frame(stock)
    eight = [k for k in f.columns if "." in k]
    count8 = f[eight].sum(axis=1)
    o, h, lo, c = (stock[k].astype(float).values for k in ("Open", "High", "Low", "Close"))
    idx = stock.index
    first = max(WARMUP_BARS, int(np.searchsorted(idx, pd.Timestamp(start))) if start is not None else 0)
    i = first
    while i < n - 1:
        if count8.iloc[i] + 1 < cfg.min_nine or not f["market.signal"].iloc[i]:
            i += 1
            continue
        ex = x.iloc[i]
        if not (ex["momentum"] and ex["trend"] and ex["candle"] and (ex["volume"] or not cfg.require_volume)):
            i += 1
            continue
        free, oa = _blocks_at(stock, i)
        nine = int(count8.iloc[i]) + int(free)
        if nine < cfg.min_nine:
            i += 1
            continue
        res["signals"] += 1
        stop_dist = cfg.atr_mult * float(ex["atr"])
        resist, _src = vx.nearest_resistance(stock.iloc[:i + 1], float(c[i]), oa)
        if resist is not None and stop_dist > 0 and (resist - c[i]) / stop_dist < cfg.min_rr:
            res["low_rr"] += 1
            i += 1
            continue
        entry = float(o[i + 1])                                     # nästa dags öppning
        if entry > c[i] * (1 + cfg.max_chase_pct / 100):
            res["no_chase"] += 1
            i += 1
            continue
        stop = entry - stop_dist
        risk = entry - stop
        if not (risk > 0):
            i += 1
            continue
        t = Trade(ticker, str(idx[i].date()), str(idx[i + 1].date()), round(entry, 4), round(stop, 4),
                  round(risk, 4), nine)
        fg_target = vex.fg_target(float(f["fg"].iloc[i])) if pd.notna(f["fg"].iloc[i]) else None
        pre_high = float(np.max(h[max(0, i + 1 - vex.BE_LOOKBACK):i + 2]))
        armed = False
        j = i + 1
        pending = None                                              # stängningsregel → exit nästa öppning
        while j < n:
            cur_stop = max(stop, entry) if armed else stop
            if pending is not None:
                t.exit, t.exit_reason, t.exit_date = float(o[j]), pending, str(idx[j].date())
                break
            if lo[j] <= cur_stop:
                px = float(o[j]) if o[j] < cur_stop else cur_stop
                t.exit, t.exit_date = px, str(idx[j].date())
                t.exit_reason = "breakeven-stopp" if armed and cur_stop >= entry else "stopp"
                break
            reasons = []
            if not f["market.signal"].iloc[j]:
                reasons.append("SPY < EMA20")
            if c[j] < x["ema10"].iloc[j]:
                reasons.append("trailing EMA10")
            if armed and j >= 1 and c[j] < lo[j - 1]:
                reasons.append("BE exit")
            if j >= 1 and o[j] > h[j - 1] and c[j] < c[j - 1]:
                reasons.append("gap & crap")
            if not f["stock.signal"].iloc[j]:
                reasons.append("stock signal")
            if not f["sector.breadth"].iloc[j] and not f["market.breadth"].iloc[j]:
                reasons.append("sektor + bredd")
            if fg_target is not None and pd.notna(f["fg"].iloc[j]) and f["fg"].iloc[j] >= fg_target:
                reasons.append("F&G-target")
            if h[j] > pre_high:
                armed = True
            if reasons:
                pending = reasons[0]
                if j == n - 1:                                      # ingen nästa dag — stäng på stängningen
                    t.exit, t.exit_reason, t.exit_date = float(c[j]), pending, str(idx[j].date())
                    break
            j += 1
        if t.exit is None:
            t.exit, t.exit_reason, t.exit_date, t.open = float(c[-1]), "öppen", str(idx[-1].date()), True
        t.r = round((t.exit - entry) / risk, 3)
        t.days = int(np.busday_count(pd.Timestamp(t.entry_date).date(), pd.Timestamp(t.exit_date).date()))
        res["trades"].append(t)
        i = max(j, i + 1)                                           # en position åt gången per ticker
    return res


# ── Nyckeltal i R ───────────────────────────────────────────────────────────
def metrics(trades: list) -> dict:
    closed = [t for t in trades if not t.open and t.r is not None]
    rs = [t.r for t in closed]
    if not rs:
        return {"trades": 0}
    wins, losses = [r for r in rs if r > 0], [r for r in rs if r <= 0]
    wr = len(wins) / len(rs)
    avg_w = float(np.mean(wins)) if wins else 0.0
    avg_l = float(np.mean(losses)) if losses else 0.0
    order = sorted(closed, key=lambda t: (t.exit_date, t.ticker))
    curve = np.cumsum([t.r for t in order])
    peak = np.maximum.accumulate(np.concatenate([[0.0], curve]))[1:]
    max_dd = float(np.max(peak - curve)) if len(curve) else 0.0
    streak = best = 0
    for t in order:
        streak = streak + 1 if t.r <= 0 else 0
        best = max(best, streak)
    gross_w, gross_l = sum(wins), -sum(losses)
    return {
        "trades": len(rs), "win_rate": round(wr * 100, 1), "avg_r": round(float(np.mean(rs)), 3),
        "median_r": round(float(np.median(rs)), 3),
        "profit_factor": round(gross_w / gross_l, 2) if gross_l > 0 else (math.inf if gross_w > 0 else None),
        "max_drawdown_r": round(max_dd, 2), "avg_winner": round(avg_w, 3), "avg_loser": round(avg_l, 3),
        "expectancy": round(wr * avg_w - (1 - wr) * abs(avg_l), 3), "max_consecutive_losses": best,
        "avg_holding_days": round(float(np.mean([t.days for t in closed])), 1), "total_r": round(float(sum(rs)), 2),
        "curve": [(t.exit_date, round(float(v), 3)) for t, v in zip(order, curve)],
    }


# ── Flera tickers ───────────────────────────────────────────────────────────
def run(tickers: list, getter: Optional[Callable] = None, sector_getter: Optional[Callable] = None,
        cfg: Config = Config(), progress: Optional[Callable] = None, today=None) -> dict:
    if getter is None:
        from market_prices import ohlcv as getter
    sector_getter = sector_getter or on._sector_default
    period = f"{int(cfg.years) + 1}y"                               # ett extra år för uppvärmning

    def _get(t):
        try:
            df = getter(t, period)
        except Exception:
            return None
        if df is None or len(df) == 0:
            return None
        if getattr(df.index, "tz", None) is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        return df

    spy = _get(on.MARKET_TICKER)
    etfs = {t: _get(t) for t in on.SECTOR_ETFS.values()}
    breadth = on.breadth_series({t: on._close(d) for t, d in etfs.items()})
    end = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    start = end - pd.DateOffset(years=int(cfg.years))
    per, trades = [], []
    for k, t in enumerate(tickers):
        df = _get(t)
        if df is None:
            per.append({"ticker": t, "trades": [], "signals": 0, "no_chase": 0, "low_rr": 0,
                        "error": "DATA UNAVAILABLE"})
        else:
            etf = on.sector_etf_for(sector_getter(t))
            r = backtest_ticker(t, df, spy, etfs.get(etf) if etf else None, breadth, cfg, start=start)
            r["sector_etf"] = etf
            per.append(r)
            trades += r["trades"]
        if progress is not None:
            progress(k + 1, len(tickers), t)
    return {"trades": trades, "per_ticker": per, "metrics": metrics(trades), "notes": NOTES, "config": cfg}
