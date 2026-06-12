"""
ember/gates.py
EMBER strategy — hard gate computations.

Trend gates:  price > 50W EMA, 20D > 50D EMA, RS vs sector ETF (3m),
              ≥3 higher lows in 6 weeks.
Entry gates:  pullback to 20D EMA, RSI(14)<45, MACD histogram higher low,
              volume > 20D avg, bullish candle, ATR falling.
No-trade flags: price in chop zone, ATR surge, late cycle, DXY rally.

DATA_GAP is never silently treated as a pass — missing data scores 0/False.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ember.config import (
    PULLBACK_EMA_PCT, RSI_ENTRY_MAX, RSI_PERIOD, VOL_MIN_RATIO,
    ATR_PERIOD, ATR_SURGE_PCT, ATR_SURGE_LOOKBACK_W,
    LATE_CYCLE_PCT, DXY_SURGE_PCT, DXY_SURGE_LOOKBACK_W,
    HIGHER_LOWS_MIN, HIGHER_LOWS_LOOKBACK_W, RS_LOOKBACK_DAYS,
    DXY_PRIMARY, DXY_FALLBACK,
)

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    is_blocker: bool   # True = hard gate; False = confirmation / advisory


# ── Robust download (reuses the pattern from tactical_entry / commodity_ratios) ──

def _download_robust(ticker: str, period: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        try:
            df = yf.download(
                ticker, period=period, auto_adjust=True,
                progress=False, show_errors=False, multi_level_index=False,
            )
        except TypeError:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.dropna(how="all")
    except Exception as exc:
        logger.debug("_download_robust(%s, %s): %s", ticker, period, exc)
        return pd.DataFrame()


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s: pd.Series, period: int = RSI_PERIOD) -> float:
    s = s.dropna()
    if len(s) < period + 1:
        return float("nan")
    delta = s.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag = gain.ewm(com=period - 1, min_periods=period).mean()
    al = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def _atr_series(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _macd_hist_higher_low(s: pd.Series) -> tuple[bool, float, float]:
    """Return (histogram made higher low vs prior bar, h_now, h_prev)."""
    s = s.dropna()
    if len(s) < 36:
        return False, float("nan"), float("nan")
    macd_line = _ema(s, 12) - _ema(s, 26)
    signal    = _ema(macd_line, 9)
    hist      = macd_line - signal
    h_now  = float(hist.iloc[-1])
    h_prev = float(hist.iloc[-2])
    return h_now > h_prev, h_now, h_prev


def _detect_candle(df: pd.DataFrame) -> str:
    """Detect the last bar's bullish candle pattern."""
    if len(df) < 2:
        return "NONE"
    last = df.iloc[-1]
    prev = df.iloc[-2]

    c_last = float(last["Close"])
    h_last = float(last["High"])
    l_last = float(last["Low"])
    o_last = float(last.get("Open", c_last))

    c_prev = float(prev["Close"])
    h_prev = float(prev["High"])
    l_prev = float(prev["Low"])
    o_prev = float(prev.get("Open", c_prev))

    body   = abs(c_last - o_last)
    rng    = h_last - l_last if h_last > l_last else 1e-10

    # Hammer: lower shadow ≥ 2× body, upper shadow < 33% of range
    lower_shadow = min(o_last, c_last) - l_last
    if (lower_shadow >= 2 * body
            and (h_last - max(o_last, c_last)) / rng < 0.33
            and body / rng < 0.40):
        return "HAMMER"

    # Bullish engulfing: last bar bullish, fully engulfs previous bearish bar
    if (c_last > o_last                     # last bar is bullish
            and o_prev > c_prev             # prev bar is bearish
            and o_last <= c_prev            # last open at or below prev close
            and c_last >= o_prev):          # last close at or above prev open
        return "ENGULFING"

    # Inside-bar breakout: last bar breaks above previous high after an inside bar
    if (l_prev >= l_last and h_prev <= h_last):
        # previous bar was inside this one — not what we want
        pass
    if c_last > h_prev and h_prev > o_prev:
        return "BREAKOUT"

    return "NONE"


def _count_higher_lows(close: pd.Series, lookback_weeks: int = HIGHER_LOWS_LOOKBACK_W) -> int:
    """Count how many consecutive swing lows are progressively higher."""
    n = lookback_weeks * 5
    subset = close.iloc[-n:] if len(close) > n else close
    if len(subset) < 6:
        return 0
    vals = subset.values
    lows = [float(vals[i])
            for i in range(1, len(vals) - 1)
            if vals[i] < vals[i - 1] and vals[i] < vals[i + 1]]
    if len(lows) < 2:
        return 0
    return sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])


# ── Trend gates ───────────────────────────────────────────────────────────────

def compute_trend_gates(
    close_d: pd.Series,
    close_w: pd.Series,
    sector_etf: str,
) -> list[GateResult]:
    """4 trend gates — all must pass (is_blocker=True on all)."""
    gates: list[GateResult] = []
    price = float(close_d.iloc[-1])

    # TREND 1 — price above 50-week EMA
    if len(close_w) >= 50:
        ema50w = float(_ema(close_w, 50).iloc[-1])
        passed = price > ema50w
        gates.append(GateResult(
            name="Pris > 50V EMA",
            passed=passed,
            detail=f"Pris {price:.2f} {'>' if passed else '<'} 50V EMA {ema50w:.2f}",
            is_blocker=True,
        ))
    else:
        gates.append(GateResult(
            name="Pris > 50V EMA",
            passed=False,
            detail="DATA_GAP — veckodata < 50 veckor",
            is_blocker=True,
        ))

    # TREND 2 — 20D EMA > 50D EMA
    ema20_d = float(_ema(close_d, 20).iloc[-1])
    ema50_d = float(_ema(close_d, 50).iloc[-1])
    c2 = ema20_d > ema50_d
    gates.append(GateResult(
        name="20D EMA > 50D EMA",
        passed=c2,
        detail=f"20D EMA {ema20_d:.2f} {'>' if c2 else '<'} 50D EMA {ema50_d:.2f}",
        is_blocker=True,
    ))

    # TREND 3 — relative strength vs sector ETF (3 months)
    gates.append(_rs_gate(close_d, sector_etf))

    # TREND 4 — ≥3 higher lows in 6 weeks
    hl_count = _count_higher_lows(close_d, HIGHER_LOWS_LOOKBACK_W)
    hl_pass  = hl_count >= HIGHER_LOWS_MIN
    gates.append(GateResult(
        name=f"≥{HIGHER_LOWS_MIN} stigande bottnar (6V)",
        passed=hl_pass,
        detail=f"{hl_count} stigande botten{'ar' if hl_count != 1 else ''} detekterade",
        is_blocker=True,
    ))

    return gates


def _rs_gate(close_d: pd.Series, sector_etf: str) -> GateResult:
    name = f"Relativ styrka vs {sector_etf} (3 mån)"
    try:
        etf_df = _download_robust(sector_etf, "6mo")
        if etf_df.empty or "Close" not in etf_df.columns:
            return GateResult(name=name, passed=False,
                              detail=f"DATA_GAP — {sector_etf} ej tillgänglig",
                              is_blocker=True)
        etf_c = etf_df["Close"].squeeze()
        if isinstance(etf_c, pd.DataFrame):
            etf_c = etf_c.iloc[:, 0]
        etf_c = etf_c.dropna()
        n = RS_LOOKBACK_DAYS
        if len(etf_c) < n or len(close_d) < n:
            return GateResult(name=name, passed=False,
                              detail="DATA_GAP — otillräcklig historik för RS",
                              is_blocker=True)
        rs_t   = float(close_d.iloc[-1]) / float(close_d.iloc[-n]) - 1
        rs_etf = float(etf_c.iloc[-1]) / float(etf_c.iloc[-n]) - 1
        diff   = (rs_t - rs_etf) * 100
        passed = diff > 0
        return GateResult(
            name=name,
            passed=passed,
            detail=f"Ticker 3m {rs_t*100:+.1f}% vs {sector_etf} {rs_etf*100:+.1f}% "
                   f"→ RS {diff:+.1f}%",
            is_blocker=True,
        )
    except Exception as exc:
        logger.debug("_rs_gate: %s", exc)
        return GateResult(name=name, passed=False,
                          detail=f"DATA_GAP: {exc}", is_blocker=True)


# ── Entry gates ───────────────────────────────────────────────────────────────

def compute_entry_gates(
    close_d: pd.Series,
    df_daily: pd.DataFrame,
) -> tuple[list[GateResult], float, float, float, float, float, str]:
    """
    Compute 6 entry gates.

    Returns
    -------
    (gates, ema20, atr14, entry_price, stop_price, rr, candle_pattern)
    """
    gates: list[GateResult] = []
    price  = float(close_d.iloc[-1])
    ema20  = float(_ema(close_d, 20).iloc[-1])

    # ATR(14) from full daily OHLC
    atr_val = float("nan")
    if all(c in df_daily.columns for c in ("High", "Low", "Close")):
        atr_s = _atr_series(df_daily)
        if not atr_s.empty:
            atr_val = float(atr_s.iloc[-1])

    # ENTRY 1 — pullback to 20D EMA ≤ 3% (hard gate)
    pct = abs(price - ema20) / ema20 * 100 if ema20 > 0 else 999.0
    e1  = pct <= PULLBACK_EMA_PCT
    gates.append(GateResult(
        name=f"Pullback till 20D EMA (≤{PULLBACK_EMA_PCT}%)",
        passed=e1,
        detail=f"Pris {price:.2f} är {pct:.1f}% från 20D EMA {ema20:.2f}",
        is_blocker=True,
    ))

    # ENTRY 2 — RSI(14) < 45 (hard gate)
    rsi_val = _rsi(close_d)
    e2 = (not np.isnan(rsi_val)) and (rsi_val < RSI_ENTRY_MAX)
    rsi_str = f"{rsi_val:.1f}" if not np.isnan(rsi_val) else "DATA_GAP"
    gates.append(GateResult(
        name=f"RSI({RSI_PERIOD}) < {RSI_ENTRY_MAX}",
        passed=e2,
        detail=f"RSI = {rsi_str}",
        is_blocker=True,
    ))

    # ENTRY 3 — MACD histogram higher low (confirmation)
    hist_rising, h_now, h_prev = _macd_hist_higher_low(close_d)
    if not np.isnan(h_now):
        gates.append(GateResult(
            name="MACD-histogram: stigande botten",
            passed=hist_rising,
            detail=f"Histogram nu {h_now:.4f} / föregående {h_prev:.4f}",
            is_blocker=False,
        ))
    else:
        gates.append(GateResult(
            name="MACD-histogram: stigande botten",
            passed=False,
            detail="DATA_GAP — otillräcklig historik",
            is_blocker=False,
        ))

    # ENTRY 4 — volume > 20D average (confirmation)
    gates.append(_volume_gate(df_daily))

    # ENTRY 5 — bullish candle pattern (confirmation)
    candle = _detect_candle(df_daily)
    candle_pass = candle in ("HAMMER", "ENGULFING", "BREAKOUT")
    gates.append(GateResult(
        name="Bullish candlestick-mönster",
        passed=candle_pass,
        detail=f"Detekterat: {candle}" if candle != "NONE" else "Inget mönster detekterat",
        is_blocker=False,
    ))

    # ENTRY 6 — ATR falling during pullback (advisory)
    gates.append(_atr_falling_gate(df_daily))

    # ── Levels ────────────────────────────────────────────────────────────────
    entry_price = ema20
    if not np.isnan(atr_val) and atr_val > 0:
        stop_price = entry_price - 2.5 * atr_val
    else:
        stop_price = entry_price * 0.95
    risk_abs = entry_price - stop_price
    rr = 2.0 if risk_abs > 0 else float("nan")

    return (
        gates,
        round(ema20, 2),
        round(atr_val, 4) if not np.isnan(atr_val) else float("nan"),
        round(entry_price, 2),
        round(stop_price, 2),
        rr,
        candle,
    )


def _volume_gate(df_daily: pd.DataFrame) -> GateResult:
    name = f"Volym ≥ {VOL_MIN_RATIO:.1f}× 20D snitt"
    if "Volume" not in df_daily.columns:
        return GateResult(name=name, passed=False,
                          detail="DATA_GAP — volymdata saknas", is_blocker=False)
    vol = df_daily["Volume"].dropna()
    if len(vol) < 22:
        return GateResult(name=name, passed=False,
                          detail="DATA_GAP — < 22 volymrader", is_blocker=False)
    avg20 = float(vol.iloc[-21:-1].mean())
    cur   = float(vol.iloc[-1])
    ratio = cur / avg20 if avg20 > 0 else 0.0
    passed = ratio >= VOL_MIN_RATIO
    return GateResult(
        name=name,
        passed=passed,
        detail=f"Volym {cur:,.0f} = {ratio:.2f}× snitt ({avg20:,.0f})",
        is_blocker=False,
    )


def _atr_falling_gate(df_daily: pd.DataFrame) -> GateResult:
    name = "ATR sjunker (lugn pullback)"
    if not all(c in df_daily.columns for c in ("High", "Low", "Close")):
        return GateResult(name=name, passed=False,
                          detail="DATA_GAP — OHLC saknas", is_blocker=False)
    atr_s = _atr_series(df_daily)
    if len(atr_s) < 6:
        return GateResult(name=name, passed=False,
                          detail="DATA_GAP — otillräcklig historik", is_blocker=False)
    atr_now  = float(atr_s.iloc[-1])
    atr_prev = float(atr_s.iloc[-5])
    falling  = atr_now < atr_prev
    return GateResult(
        name=name,
        passed=falling,
        detail=f"ATR nu {atr_now:.4f} / 5 dagar {atr_prev:.4f} "
               f"({'↓ SJUNKER ✓' if falling else '↑ STIGER'})",
        is_blocker=False,
    )


# ── No-trade zone flags ───────────────────────────────────────────────────────

def compute_notrade_flags(
    close_d: pd.Series,
    df_daily: pd.DataFrame,
    cycle_percentile_10y: Optional[float],
) -> list[GateResult]:
    """
    4 no-trade zone checks.
    passed=True means the NO-TRADE condition IS active → blocks the trade.
    """
    flags: list[GateResult] = []
    price  = float(close_d.iloc[-1])
    ema20  = float(_ema(close_d, 20).iloc[-1])
    ema50  = float(_ema(close_d, 50).iloc[-1])

    # FLAG 1 — price in chop zone between 20D and 50D EMA
    lo, hi = min(ema20, ema50), max(ema20, ema50)
    between = lo < price < hi
    flags.append(GateResult(
        name="Pris i chop-zon (mellan 20D/50D EMA)",
        passed=between,
        detail=f"20D {ema20:.2f} · 50D {ema50:.2f} · Pris {price:.2f}"
               f" {'→ CHOP-ZON ⛔' if between else ' ✓'}",
        is_blocker=True,
    ))

    # FLAG 2 — ATR surge > 40% vs 2 weeks ago
    flags.append(_atr_surge_flag(df_daily))

    # FLAG 3 — late cycle (10y percentile > 85)
    if cycle_percentile_10y is not None:
        late = cycle_percentile_10y > LATE_CYCLE_PCT
        flags.append(GateResult(
            name=f"Sen cykel (10å-percentil > {LATE_CYCLE_PCT:.0f}%)",
            passed=late,
            detail=f"Temas 10å-percentil: {cycle_percentile_10y:.1f}% "
                   f"{'→ SEN/TOPP ⛔' if late else '✓'}",
            is_blocker=True,
        ))
    else:
        flags.append(GateResult(
            name=f"Sen cykel (10å-percentil > {LATE_CYCLE_PCT:.0f}%)",
            passed=False,   # DATA_GAP → assume OK (no block)
            detail="DATA_GAP — cykelpositon okänd, antas OK",
            is_blocker=True,
        ))

    # FLAG 4 — DXY rally > 2% in 2 weeks
    flags.append(_dxy_surge_flag())

    return flags


def _atr_surge_flag(df_daily: pd.DataFrame) -> GateResult:
    name = f"ATR-surge > {ATR_SURGE_PCT:.0f}% (2 veckor)"
    if not all(c in df_daily.columns for c in ("High", "Low", "Close")):
        return GateResult(name=name, passed=False,
                          detail="DATA_GAP — OHLC saknas", is_blocker=True)
    atr_s = _atr_series(df_daily)
    n = ATR_SURGE_LOOKBACK_W * 5
    if len(atr_s) < n + 2:
        return GateResult(name=name, passed=False,
                          detail="DATA_GAP — otillräcklig historik", is_blocker=True)
    atr_now  = float(atr_s.iloc[-1])
    atr_prev = float(atr_s.iloc[-(n + 1)])
    change   = (atr_now - atr_prev) / atr_prev * 100 if atr_prev > 0 else 0.0
    surge    = change > ATR_SURGE_PCT
    return GateResult(
        name=name,
        passed=surge,
        detail=f"ATR 2V förändring: {change:+.1f}% "
               f"({'SURGE → STANNA ⛔' if surge else '✓'})",
        is_blocker=True,
    )


def _dxy_surge_flag() -> GateResult:
    name = f"DXY-rally > {DXY_SURGE_PCT:.0f}% (2 veckor)"
    try:
        df = _download_robust(DXY_PRIMARY, "3mo")
        if df.empty or "Close" not in df.columns:
            df = _download_robust(DXY_FALLBACK, "3mo")
        if df.empty or "Close" not in df.columns:
            return GateResult(name=name, passed=False,
                              detail="DATA_GAP — DXY ej tillgänglig, antas OK",
                              is_blocker=True)
        close = df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        n = DXY_SURGE_LOOKBACK_W * 5
        if len(close) < n + 2:
            return GateResult(name=name, passed=False,
                              detail="DATA_GAP — otillräcklig DXY-historik",
                              is_blocker=True)
        change = (float(close.iloc[-1]) / float(close.iloc[-(n + 1)]) - 1) * 100
        surge  = change > DXY_SURGE_PCT
        return GateResult(
            name=name,
            passed=surge,
            detail=f"DXY 2V förändring: {change:+.1f}% "
                   f"({'RALLY → MOTSTÅND ⛔' if surge else '✓'})",
            is_blocker=True,
        )
    except Exception as exc:
        logger.debug("_dxy_surge_flag: %s", exc)
        return GateResult(name=name, passed=False,
                          detail=f"DATA_GAP: {exc}", is_blocker=True)
