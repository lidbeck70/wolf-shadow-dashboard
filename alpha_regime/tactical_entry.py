"""
alpha_regime/tactical_entry.py
Tactical entry timing layer for Alpha Regime.

Activates when regime is ACCUMULATE (any stage) or Quality BUY.
Downloads daily (2y) + weekly (5y) price data and computes 6 entry checks.
Returns TacticalEntryResult with a Swedish verdict, per-check list, and
concrete levels (entry zone, stop, 1:2 / 1:3 targets).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Verdict thresholds
_ENTRY_NOW_MIN_PASS   = 5   # ≥5/6 checks → ENTRY NOW
_WAIT_TREND_CHECKS    = {0, 1}  # check indices for trend filters (fail → INGEN ENTRY)

# Entry zone half-width around 20D EMA
_ENTRY_ZONE_PCT = 0.02  # ±2%

# Swing-low lookback for stop calculation
_SWING_LOOKBACK_DAYS = 63  # ~3 months

# Sector ETF mapping by exposure key (matches EXPOSURE_TO_RATIO in commodity_ratios)
_EXPOSURE_TO_ETF: dict[str, str] = {
    "gold_miner":   "GDX",
    "junior_miner": "GDX",
    "silver":       "SIL",
    "oil":          "XLE",
    "copper":       "COPX",
    "uranium":      "URA",
}
_DEFAULT_SECTOR_ETF = "SPY"

# RSI and MACD parameters
_RSI_PERIOD   = 14
_EMA_FAST     = 12
_EMA_SLOW     = 26
_EMA_SIGNAL   = 9


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class TacticalCheck:
    name: str       # Swedish display name
    passed: bool
    detail: str     # e.g. "Pris 184.40 > 50V EMA 172.30"
    is_trend: bool  # True → trend filter; fail → verdict becomes INGEN ENTRY


@dataclass
class TacticalEntryResult:
    verdict: str                    # "ENTRY NOW" | "WAIT — PULLBACK PÅGÅR" | "INGEN ENTRY"
    checks: list[TacticalCheck]     = field(default_factory=list)
    passed_count: int               = 0

    # Concrete levels
    ema20: Optional[float]          = None
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    stop_level: Optional[float]     = None
    stop_risk_pct: Optional[float]  = None
    target_2r: Optional[float]      = None
    target_3r: Optional[float]      = None

    error: Optional[str]            = None


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema_series(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s: pd.Series, period: int = _RSI_PERIOD) -> float:
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


def _macd_hist_rising(s: pd.Series) -> tuple[bool, float, float]:
    """Return (rising, hist_now, hist_prev)."""
    s = s.dropna()
    if len(s) < _EMA_SLOW + _EMA_SIGNAL + 2:
        return False, float("nan"), float("nan")
    macd_line = _ema_series(s, _EMA_FAST) - _ema_series(s, _EMA_SLOW)
    signal    = _ema_series(macd_line, _EMA_SIGNAL)
    hist      = macd_line - signal
    h_now  = float(hist.iloc[-1])
    h_prev = float(hist.iloc[-2])
    return h_now > h_prev, h_now, h_prev


def _download_robust(ticker: str, period: str) -> pd.DataFrame:
    """Robust yfinance OHLCV download (same pattern as commodity_ratios)."""
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


# ── Main compute function ─────────────────────────────────────────────────────

def compute_tactical_entry(
    ticker: str,
    sector_etf: Optional[str] = None,
) -> TacticalEntryResult:
    """
    Compute tactical entry signals for a ticker.

    Parameters
    ----------
    ticker     : stock / ETF ticker (yfinance format)
    sector_etf : sector ETF for RS check; None = auto-default SPY

    Returns
    -------
    TacticalEntryResult
    """
    if not sector_etf:
        sector_etf = _DEFAULT_SECTOR_ETF

    result = TacticalEntryResult(verdict="INGEN ENTRY")

    # ── 1. Download daily data (2y) ──────────────────────────────────────────
    daily = _download_robust(ticker, "2y")
    if daily.empty or "Close" not in daily.columns:
        result.error = f"Daglig kursdata ej tillgänglig för {ticker}"
        return result

    close_d = daily["Close"].squeeze()
    if isinstance(close_d, pd.DataFrame):
        close_d = close_d.iloc[:, 0]
    close_d = close_d.dropna()

    if len(close_d) < 60:
        result.error = f"För lite historik för {ticker} ({len(close_d)} dagar)"
        return result

    price = float(close_d.iloc[-1])

    # ── 2. Download weekly data (5y) for 50W EMA ─────────────────────────────
    weekly = _download_robust(ticker, "5y")
    ema50w: Optional[float] = None
    if not weekly.empty and "Close" in weekly.columns:
        close_w = weekly["Close"].squeeze()
        if isinstance(close_w, pd.DataFrame):
            close_w = close_w.iloc[:, 0]
        close_w = close_w.dropna().resample("W").last().dropna()
        if len(close_w) >= 50:
            ema50w = float(_ema_series(close_w, 50).iloc[-1])

    # ── Daily indicators ──────────────────────────────────────────────────────
    ema20_s  = _ema_series(close_d, 20)
    ema50_s  = _ema_series(close_d, 50)
    ema20    = float(ema20_s.iloc[-1])
    ema50_d  = float(ema50_s.iloc[-1])
    rsi_val  = _rsi(close_d)
    macd_rising, hist_now, hist_prev = _macd_hist_rising(close_d)

    # Swing low (3M)
    lookback   = min(_SWING_LOOKBACK_DAYS, len(close_d) - 1)
    swing_low  = float(np.min(close_d.values[-lookback:])) if lookback > 0 else price * 0.95
    stop_risk  = round((price - swing_low) / price * 100, 2) if price > 0 else None

    # ── RS vs sector ETF (1 month = 21 trading days) ─────────────────────────
    rs_positive: Optional[bool] = None
    rs_detail   = f"Sektordata ej tillgänglig ({sector_etf})"

    etf_daily = _download_robust(sector_etf, "3mo")
    if not etf_daily.empty and "Close" in etf_daily.columns:
        etf_close = etf_daily["Close"].squeeze()
        if isinstance(etf_close, pd.DataFrame):
            etf_close = etf_close.iloc[:, 0]
        etf_close = etf_close.dropna()
        if len(etf_close) >= 21 and len(close_d) >= 21:
            rs_ticker = float(close_d.iloc[-1]) / float(close_d.iloc[-22]) - 1
            rs_etf    = float(etf_close.iloc[-1]) / float(etf_close.iloc[-22]) - 1
            rs_diff   = (rs_ticker - rs_etf) * 100
            rs_positive = rs_diff > 0
            sign = "+" if rs_diff >= 0 else ""
            rs_detail = (
                f"{ticker} 1m: {rs_ticker*100:+.1f}% vs {sector_etf}: "
                f"{rs_etf*100:+.1f}% → RS {sign}{rs_diff:.1f}%"
            )

    # ── Build checks ─────────────────────────────────────────────────────────
    checks: list[TacticalCheck] = []

    # CHECK 0 — price > 50W EMA (trend filter)
    if ema50w is not None:
        c0_pass = price > ema50w
        checks.append(TacticalCheck(
            name="Pris > 50-veckors EMA",
            passed=c0_pass,
            detail=f"Pris {price:.2f} {'>' if c0_pass else '<'} 50V EMA {ema50w:.2f}",
            is_trend=True,
        ))
    else:
        checks.append(TacticalCheck(
            name="Pris > 50-veckors EMA",
            passed=False,
            detail="Veckodata otillräcklig (< 50 veckor)",
            is_trend=True,
        ))

    # CHECK 1 — 20D EMA > 50D EMA (trend filter)
    c1_pass = ema20 > ema50_d
    checks.append(TacticalCheck(
        name="20D EMA > 50D EMA",
        passed=c1_pass,
        detail=f"20D EMA {ema20:.2f} {'>' if c1_pass else '<'} 50D EMA {ema50_d:.2f}",
        is_trend=True,
    ))

    # CHECK 2 — pullback: price within 3% of 20D EMA
    pct_from_ema20 = abs(price - ema20) / ema20 * 100
    c2_pass = pct_from_ema20 <= 3.0
    checks.append(TacticalCheck(
        name="Pullback till 20D EMA (≤3%)",
        passed=c2_pass,
        detail=f"Pris {price:.2f} är {pct_from_ema20:.1f}% från 20D EMA {ema20:.2f}",
        is_trend=False,
    ))

    # CHECK 3 — RSI(14) < 45
    c3_pass = (not np.isnan(rsi_val)) and (rsi_val < 45)
    rsi_str = f"{rsi_val:.1f}" if not np.isnan(rsi_val) else "N/A"
    checks.append(TacticalCheck(
        name="RSI(14) < 45 (översålt på dagsdiagram)",
        passed=c3_pass,
        detail=f"RSI = {rsi_str}",
        is_trend=False,
    ))

    # CHECK 4 — MACD histogram rising
    if not np.isnan(hist_now):
        c4_pass = macd_rising
        checks.append(TacticalCheck(
            name="MACD-histogram stiger",
            passed=c4_pass,
            detail=f"Histogram nu: {hist_now:.4f}, föregående: {hist_prev:.4f}",
            is_trend=False,
        ))
    else:
        checks.append(TacticalCheck(
            name="MACD-histogram stiger",
            passed=False,
            detail="MACD-data otillräcklig",
            is_trend=False,
        ))

    # CHECK 5 — RS vs sector ETF positive (1 month)
    if rs_positive is not None:
        checks.append(TacticalCheck(
            name=f"Relativ styrka vs {sector_etf} (1 mån)",
            passed=rs_positive,
            detail=rs_detail,
            is_trend=False,
        ))
    else:
        checks.append(TacticalCheck(
            name=f"Relativ styrka vs {sector_etf} (1 mån)",
            passed=False,
            detail=rs_detail,
            is_trend=False,
        ))

    # ── Verdict logic ─────────────────────────────────────────────────────────
    trend_failed = any(not c.passed for c in checks if c.is_trend)
    passed_count = sum(1 for c in checks if c.passed)

    if trend_failed:
        verdict = "INGEN ENTRY"
    elif passed_count >= _ENTRY_NOW_MIN_PASS:
        verdict = "ENTRY NOW"
    else:
        verdict = "WAIT — PULLBACK PÅGÅR"

    # ── Concrete levels ───────────────────────────────────────────────────────
    entry_low  = round(ema20 * (1 - _ENTRY_ZONE_PCT), 2)
    entry_high = round(ema20 * (1 + _ENTRY_ZONE_PCT), 2)
    stop       = round(swing_low * 0.995, 2)      # just below swing low

    risk_abs   = ((entry_low + entry_high) / 2) - stop
    target_2r  = round(((entry_low + entry_high) / 2) + 2 * risk_abs, 2) if risk_abs > 0 else None
    target_3r  = round(((entry_low + entry_high) / 2) + 3 * risk_abs, 2) if risk_abs > 0 else None

    result.verdict       = verdict
    result.checks        = checks
    result.passed_count  = passed_count
    result.ema20         = round(ema20, 2)
    result.entry_zone_low  = entry_low
    result.entry_zone_high = entry_high
    result.stop_level      = stop
    result.stop_risk_pct   = stop_risk
    result.target_2r       = target_2r
    result.target_3r       = target_3r
    return result


def resolve_sector_etf(detected_exposure: Optional[list]) -> str:
    """
    Map a list of ratio keys (from RegimeResult.detected_exposure) to the
    best-fit sector ETF for the RS check.
    """
    if not detected_exposure:
        return _DEFAULT_SECTOR_ETF
    key = detected_exposure[0] if detected_exposure else ""
    # Map ratio key → exposure label
    _RATIO_TO_EXP = {
        "metal_miners": "gold_miner",
        "gdxj_gdx":     "junior_miner",
        "gold_gdxj":    "junior_miner",
        "gdx_spy":      "gold_miner",
        "gold_silver":  "silver",
        "silver_juniors": "silver",
        "silver_miners": "silver",
        "gold_oil":     "oil",
        "copper_gold":  "copper",
    }
    exp = _RATIO_TO_EXP.get(key, "")
    return _EXPOSURE_TO_ETF.get(exp, _DEFAULT_SECTOR_ETF)
