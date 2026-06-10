"""
alpha_regime/commodity_ratios.py
Commodity ratio "rubber band" engine for Deep Contrarian mode.

Five cross-asset ratios at 10-year daily history, cached 6h.
High percentile (≥90) = denominator stretched cheap (high-direction ratios).
Low  percentile (≤10) = numerator  stretched cheap (copper_gold only).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Cache (graceful without Streamlit) ───────────────────────────────────────
try:
    import streamlit as st
    def _cache_6h(fn):
        return st.cache_data(ttl=21600, show_spinner=False)(fn)
except ImportError:
    def _cache_6h(fn):
        return fn


# ── Thresholds (named constants) ──────────────────────────────────────────────
_STRETCHED_PCT   = 90.0   # ≥ → RUBBER_BAND_STRETCHED  (high direction)
_TENSION_PCT     = 80.0   # ≥ → TENSION_BUILDING        (high direction)
_CHEAP_LOW_MAX   = 10.0   # ≤ → RUBBER_BAND_STRETCHED  (low direction: copper_gold / gdx_spy)
_CHEAP_LOW_TENS  = 20.0   # ≤ → TENSION_BUILDING        (low direction)
_MIN_HISTORY_DAYS = 252 * 5  # 5-year guard for silver_juniors (SILJ history)

# Per-leg percentile thresholds (driver classification)
_LEG_EXPENSIVE_MIN = 80.0   # a leg is "expensive" when its own pctile >= this
_LEG_CHEAP_MAX     = 20.0   # a leg is "cheap"    when its own pctile <= this


# ── Ratio specifications ──────────────────────────────────────────────────────
@dataclass
class RatioSpec:
    key: str
    label: str
    denominator_label: str   # asset that is "stretched cheap" when STRETCHED
    primary_num: str
    primary_den: str
    fallback_num: str        # "" = no fallback (DATA_GAP on primary failure)
    fallback_den: str
    cheap_direction: str     # "high" | "low"
    min_years: int = 0       # minimum history required; 0 = no guard


_RATIO_SPECS: list[RatioSpec] = [
    # ── Original five ────────────────────────────────────────────────────────
    RatioSpec("gold_silver",     "Gold / Silver",               "Silver",          "GC=F", "SI=F",  "GLD",  "SLV",  "high"),
    RatioSpec("gold_oil",        "Gold / Oil",                  "Oil",             "GC=F", "CL=F",  "GLD",  "USO",  "high"),
    RatioSpec("metal_miners",    "Gold ETF / Gold Miners",      "Gold Miners",     "GLD",  "GDX",   "GLD",  "GDX",  "high"),
    RatioSpec("silver_miners",   "Silver ETF / Silver Miners",  "Silver Miners",   "SLV",  "SIL",   "SLV",  "SIL",  "high"),
    RatioSpec("copper_gold",     "Copper / Gold",               "Copper",          "HG=F", "GC=F",  "COPX", "GLD",  "low"),
    # ── Miner benchmark ratios ───────────────────────────────────────────────
    RatioSpec("gdxj_gdx",        "GDXJ / GDX (Juniors/Majors)", "Junior Miners",   "GDXJ", "GDX",   "GDXJ", "GDX",  "high"),
    RatioSpec("gold_gdxj",       "GLD / GDXJ (Metal/Juniors)",  "Junior Miners",   "GLD",  "GDXJ",  "GLD",  "GDXJ", "high"),
    RatioSpec("silver_juniors",  "SLV / SILJ (Silver Juniors)", "Silver Juniors",  "SLV",  "SILJ",  "",     "",     "high", min_years=5),
    RatioSpec("gdx_spy",         "GDX / SPY (Miners/Market)",   "Gold Miners",     "GDX",  "SPY",   "GDX",  "SPY",  "low"),
]


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class RatioResult:
    key: str
    label: str
    denominator_label: str
    cheap_direction: str
    current: float = 0.0
    percentile: float = 0.0
    zscore: float = 0.0
    status: str = "DATA_GAP"   # RUBBER_BAND_STRETCHED | TENSION_BUILDING | NEUTRAL | DATA_GAP
    sparkline_dates: list = field(default_factory=list)
    sparkline_values: list = field(default_factory=list)
    error: Optional[str] = None
    # Per-leg breakdown (populated by _compute_ratio)
    numerator_label:       str            = ""
    denominator_leg_label: str            = ""
    numerator_pctile:      Optional[float] = None
    denominator_pctile:    Optional[float] = None
    driver: str = "UNKNOWN"  # DENOMINATOR_CHEAP | NUMERATOR_EXPENSIVE | BOTH | MIXED | UNKNOWN


# ── Fetch helpers ─────────────────────────────────────────────────────────────
def _download_close(ticker: str, period: str = "10y") -> pd.Series:
    """Robust yfinance Close series download (multi_level_index pattern)."""
    try:
        df = yf.download(
            ticker, period=period, auto_adjust=True,
            progress=False, show_errors=False, multi_level_index=False,
        )
    except TypeError:
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        except Exception as exc:
            logger.debug("_download_close(%s): %s", ticker, exc)
            return pd.Series(dtype=float)
    except Exception as exc:
        logger.debug("_download_close(%s): %s", ticker, exc)
        return pd.Series(dtype=float)

    if df is None or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        candidates = [c for c in df.columns if str(c).lower() == "close"]
        if not candidates:
            return pd.Series(dtype=float)
        df = df.rename(columns={candidates[0]: "Close"})

    s = df["Close"].squeeze()
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    if hasattr(s.index, "tz") and s.index.tz is not None:
        s.index = s.index.tz_localize(None)

    return s.dropna()


def _percentile_of(arr: np.ndarray, val: float) -> float:
    """Percentile-of-score without scipy."""
    if len(arr) == 0:
        return 50.0
    return float(np.searchsorted(np.sort(arr), val, side="right") / len(arr) * 100)


def _leg_label(pct: float) -> str:
    """Human-readable Swedish label for a single leg's own percentile."""
    if pct >= 80: return "DYRT"
    if pct >= 60: return "HÖGT"
    if pct >= 40: return "NEUTRALT"
    if pct >= 20: return "LÅGT"
    return "BILLIGT"


def _classify_driver(num_pct: float, den_pct: float, cheap_direction: str) -> str:
    """
    Classify what is actually driving a stretched ratio.

    For 'high' direction (high ratio = denominator cheap, e.g. gold/silver → silver cheap):
      Genuine signal  = denominator actually cheap (den_pct <= _LEG_CHEAP_MAX).
      Misleading      = numerator just expensive   (num_pct >= _LEG_EXPENSIVE_MIN).
    For 'low' direction (low ratio = numerator cheap, e.g. copper/gold → copper cheap):
      Genuine signal  = numerator actually cheap   (num_pct <= _LEG_CHEAP_MAX).
      Misleading      = denominator just expensive (den_pct >= _LEG_EXPENSIVE_MIN).
    """
    if cheap_direction == "high":
        cheap_pct = den_pct
        other_pct = num_pct
    else:
        cheap_pct = num_pct
        other_pct = den_pct

    genuine = cheap_pct <= _LEG_CHEAP_MAX
    mislead = other_pct >= _LEG_EXPENSIVE_MIN

    if genuine and mislead:
        return "BOTH"
    if genuine:
        return "DENOMINATOR_CHEAP"
    if mislead:
        return "NUMERATOR_EXPENSIVE"
    return "MIXED"


def _classify(pct: float, direction: str) -> str:
    if direction == "high":
        if pct >= _STRETCHED_PCT:
            return "RUBBER_BAND_STRETCHED"
        if pct >= _TENSION_PCT:
            return "TENSION_BUILDING"
        return "NEUTRAL"
    else:  # low direction
        if pct <= _CHEAP_LOW_MAX:
            return "RUBBER_BAND_STRETCHED"
        if pct <= _CHEAP_LOW_TENS:
            return "TENSION_BUILDING"
        return "NEUTRAL"


def _compute_ratio(spec: RatioSpec) -> RatioResult:
    """Compute ratio result, trying primary pair then fallback."""
    base = RatioResult(
        key=spec.key, label=spec.label,
        denominator_label=spec.denominator_label,
        cheap_direction=spec.cheap_direction,
    )

    pairs = [(spec.primary_num, spec.primary_den)]
    # Only add fallback when non-empty and different from primary
    if spec.fallback_num and spec.fallback_den and (spec.fallback_num, spec.fallback_den) != (spec.primary_num, spec.primary_den):
        pairs.append((spec.fallback_num, spec.fallback_den))

    for num_t, den_t in pairs:
        try:
            num_s = _download_close(num_t)
            den_s = _download_close(den_t)
            if num_s.empty or den_s.empty:
                continue

            aligned = pd.concat({"n": num_s, "d": den_s}, axis=1).dropna()
            if len(aligned) < 100:
                continue

            # Minimum history guard (e.g. 5y for SILJ)
            if spec.min_years > 0 and len(aligned) < spec.min_years * 252:
                base.error = (
                    f"{spec.key}: only {len(aligned)} trading days of history "
                    f"(need ≥{spec.min_years * 252} for {spec.min_years}y minimum)"
                )
                return base

            ratio = aligned["n"] / aligned["d"]
            arr = ratio.values.astype(float)
            current = float(arr[-1])
            pct = _percentile_of(arr, current)
            mean, std = float(arr.mean()), float(arr.std())
            zscore = round((current - mean) / std, 2) if std > 0 else 0.0

            # Weekly-resampled sparkline (full history)
            ratio_w = ratio.resample("W").last().dropna()
            base.sparkline_dates  = [str(d.date()) for d in ratio_w.index]
            base.sparkline_values = [round(float(v), 5) for v in ratio_w.values]

            base.current    = round(current, 5)
            base.percentile = round(pct, 1)
            base.zscore     = zscore
            base.status     = _classify(pct, spec.cheap_direction)
            base.error      = None

            # Per-leg percentile breakdown
            num_leg = aligned["n"].values.astype(float)
            den_leg = aligned["d"].values.astype(float)
            num_leg_pct = _percentile_of(num_leg, float(num_leg[-1]))
            den_leg_pct = _percentile_of(den_leg, float(den_leg[-1]))
            base.numerator_label       = num_t
            base.denominator_leg_label = den_t
            base.numerator_pctile      = round(num_leg_pct, 1)
            base.denominator_pctile    = round(den_leg_pct, 1)
            base.driver                = _classify_driver(num_leg_pct, den_leg_pct, spec.cheap_direction)
            return base

        except Exception as exc:
            logger.debug("_compute_ratio(%s) %s/%s: %s", spec.key, num_t, den_t, exc)
            continue

    if base.error is None:
        base.error = f"Data unavailable for {spec.key} (primary{' + fallback' if spec.fallback_num else ''} failed)"
    return base


@_cache_6h
def fetch_all_ratios() -> dict:
    """Fetch and compute all 5 commodity ratios. Result cached 6h."""
    return {spec.key: _compute_ratio(spec) for spec in _RATIO_SPECS}


# ── Exposure → ratio key list ─────────────────────────────────────────────────
# Each exposure maps to an ordered list of ratio keys to display.
EXPOSURE_TO_RATIO: dict[str, list[str]] = {
    "gold_miner":   ["metal_miners", "gdxj_gdx", "gold_gdxj", "gdx_spy"],
    "junior_miner": ["gdxj_gdx", "gold_gdxj"],
    "silver":       ["gold_silver", "silver_juniors"],
    "oil":          ["gold_oil"],
    "copper":       ["copper_gold"],
}

# Ordered: most specific patterns first
_KEYWORD_EXPOSURE: list[tuple[list[str], str]] = [
    (["junior miner", "junior mine", "juniormine", "gdxj"],   "junior_miner"),
    (["silver miner", "silver mine", "silvr", "silber"],       "silver"),
    (["gold miner", "gold mine", "goldminer"],                 "gold_miner"),
    (["silver"],                                               "silver"),
    (["gold", "guld", "gld"],                                  "gold_miner"),
    (["mining", "gruv"],                                       "gold_miner"),
    (["oil", "olja", "petroleum", "petro", "crude", "energi"], "oil"),
    (["copper", "koppar"],                                     "copper"),
]


def detect_exposure(branch_name: Optional[str]) -> Optional[str]:
    """
    Map Börsdata branch/sector name (or ticker name) → exposure key.
    Returns None when no commodity match found.
    """
    if not branch_name:
        return None
    text = branch_name.lower()
    for keywords, exposure in _KEYWORD_EXPOSURE:
        if any(kw in text for kw in keywords):
            return exposure
    return None


# ── Gold context gauges (Gold/USD, Gold/SEK, Gold/NOK) ───────────────────────
# CONTEXT ONLY — these gauges must NEVER feed the ACCUMULATE/DISTRIBUTE
# confirmation count. Currency trends have no mean reversion; using FX-adjusted
# gold prices as a contrarian buy signal would be methodologically wrong.

@dataclass
class ContextGaugeSpec:
    key: str
    label: str
    currency: str
    primary_price: str    # "GC=F"
    fallback_price: str   # "GLD"
    fx_ticker: str        # "" for USD; "USDSEK=X" / "USDNOK=X" for local currency


@dataclass
class ContextGaugeResult:
    key: str
    label: str
    currency: str
    current: float = 0.0
    percentile: float = 0.0
    zscore: float = 0.0
    sparkline_dates: list = field(default_factory=list)
    sparkline_values: list = field(default_factory=list)
    # DATA_GAP | DYRT | HÖGT | NEUTRALT | LÅGT | BILLIGT
    status: str = "DATA_GAP"
    error: Optional[str] = None


_CONTEXT_GAUGE_SPECS: list[ContextGaugeSpec] = [
    ContextGaugeSpec("gold_usd", "Gold / USD", "USD", "GC=F", "GLD", ""),
    ContextGaugeSpec("gold_sek", "Gold / SEK", "SEK", "GC=F", "GLD", "USDSEK=X"),
    ContextGaugeSpec("gold_nok", "Gold / NOK", "NOK", "GC=F", "GLD", "USDNOK=X"),
]


def _compute_context_gauge(spec: ContextGaugeSpec) -> ContextGaugeResult:
    base = ContextGaugeResult(key=spec.key, label=spec.label, currency=spec.currency)

    price_s = _download_close(spec.primary_price)
    if price_s.empty:
        price_s = _download_close(spec.fallback_price)
    if price_s.empty:
        base.error = f"No price data for {spec.key} (tried {spec.primary_price} + {spec.fallback_price})"
        return base

    if spec.fx_ticker:
        fx_s = _download_close(spec.fx_ticker)
        if fx_s.empty:
            base.error = f"FX data unavailable for {spec.fx_ticker} — {spec.key} skipped"
            return base
        combined = pd.concat({"p": price_s, "fx": fx_s}, axis=1).dropna()
        if len(combined) < 100:
            base.error = f"Insufficient aligned history for {spec.key} ({len(combined)} days)"
            return base
        series = combined["p"] * combined["fx"]
    else:
        series = price_s.dropna()

    if len(series) < 100:
        base.error = f"Insufficient history for {spec.key} ({len(series)} days)"
        return base

    arr = series.values.astype(float)
    current = float(arr[-1])
    pct = _percentile_of(arr, current)
    mean, std = float(arr.mean()), float(arr.std())
    zscore = round((current - mean) / std, 2) if std > 0 else 0.0

    series_w = series.resample("W").last().dropna()
    base.current = round(current, 2)
    base.percentile = round(pct, 1)
    base.zscore = zscore
    base.status = _leg_label(pct)   # DYRT / HÖGT / NEUTRALT / LÅGT / BILLIGT
    base.sparkline_dates = [str(d.date()) for d in series_w.index]
    base.sparkline_values = [round(float(v), 2) for v in series_w.values]
    base.error = None
    return base


@_cache_6h
def fetch_context_gauges() -> dict:
    """
    Fetch Gold/USD, Gold/SEK, Gold/NOK context gauges. Cached 6h.

    CONTEXT ONLY — never feed the ACCUMULATE/DISTRIBUTE confirmation count.
    Currency (SEK/NOK) trends lack mean reversion and must never act as
    contrarian buy signals.
    """
    return {spec.key: _compute_context_gauge(spec) for spec in _CONTEXT_GAUGE_SPECS}
