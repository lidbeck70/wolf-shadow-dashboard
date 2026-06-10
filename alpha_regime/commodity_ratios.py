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
