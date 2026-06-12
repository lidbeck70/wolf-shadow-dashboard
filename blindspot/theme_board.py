"""
blindspot/theme_board.py
Commodity Theme Board for Odin's Blindspot.

Nine commodity themes ranked by BLINDSPOT_SCORE.
Each theme has:
  CYKELPOSITION  — 10y price percentile + 200W MA slope → TIDIG/MITTEN/SEN/TOPP
  HAT            — distance from 5y high + 12m volume trend + 12m RS vs SPY (0-100)
  NÖDVÄNDIGHET   — static weight (named constants)
  BLINDSPOT_SCORE = (necessity/100) * (hat/100) * (cheapness/100) * 100

Results cached 12h via Streamlit cache_data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Cache (graceful without Streamlit) ────────────────────────────────────────
try:
    import streamlit as st
    def _cache_12h(fn):
        return st.cache_data(ttl=43200, show_spinner=False)(fn)
except ImportError:
    def _cache_12h(fn):
        return fn

# ── Necessity constants (named, with rationale) ───────────────────────────────

# Uranium: irreplaceable baseload energy, no substitute for nuclear
NECESSITY_URAN       = 90
# Copper: critical for electrification, EVs, data centres
NECESSITY_KOPPAR     = 95
# Oil: transportation + petrochemicals, still dominant energy carrier
NECESSITY_OLJA       = 90
# Silver: industrial (solar, EVs, electronics) + monetary
NECESSITY_SILVER     = 75
# Gold: monetary reserve, central bank demand driver
NECESSITY_GULD       = 70
# Natural gas: heating, power generation, transitional energy
NECESSITY_NATURGAS   = 80
# Coal: still ~35% of global electricity; declining but not gone
NECESSITY_KOL        = 60
# Agriculture: food security, fertiliser upstream
NECESSITY_AGRI       = 70
# Rare earth metals: defence, EVs, wind turbines, electronics
NECESSITY_SALLSYNTA  = 85

# ── Cycle classification thresholds ──────────────────────────────────────────

_CYKEL_TIDIG   = 30.0   # percentile ≤ → TIDIG (near bottom)
_CYKEL_SEN     = 70.0   # percentile ≥ → SEN or TOPP
_CYKEL_TOPP    = 90.0   # percentile ≥ → TOPP

# HAT sub-score weights (sum = 100)
_HAT_W_5Y_HIGH  = 40    # how far below the 5y high
_HAT_W_VOL      = 30    # 12m volume trend (declining = more hated)
_HAT_W_RS       = 30    # 12m RS vs SPY (underperforming = more hated)

# Minimum history (trading days) for a theme to avoid DATA_GAP
_MIN_HISTORY = 200


# ── Theme specification ────────────────────────────────────────────────────────

@dataclass
class ThemeSpec:
    key: str
    label: str              # Swedish display name
    primary_tickers: list[str]
    necessity: int
    proxy_flag: bool = False   # True = using a proxy (e.g. BTU for Kol)
    proxy_note: str = ""


_THEMES: list[ThemeSpec] = [
    ThemeSpec("uran",       "Uran",             ["URA"],          NECESSITY_URAN),
    ThemeSpec("silver",     "Silver",            ["SLV", "SIL"],   NECESSITY_SILVER),
    ThemeSpec("guld",       "Guld",              ["GLD", "GDX"],   NECESSITY_GULD),
    ThemeSpec("koppar",     "Koppar",            ["COPX"],         NECESSITY_KOPPAR),
    ThemeSpec("olja",       "Olja",              ["XLE", "USO"],   NECESSITY_OLJA),
    ThemeSpec("naturgas",   "Naturgas",          ["UNG"],          NECESSITY_NATURGAS),
    ThemeSpec("kol",        "Kol",               ["BTU"],          NECESSITY_KOL,
              proxy_flag=True, proxy_note="BTU (Peabody Energy) används som proxy för kol"),
    ThemeSpec("agri",       "Agri",              ["DBA"],          NECESSITY_AGRI),
    ThemeSpec("sallsynta",  "Sällsynta metaller",["REMX"],         NECESSITY_SALLSYNTA),
]

# Rubber-band ratio keys per theme (for cross-linking in the UI)
THEME_RATIO_KEYS: dict[str, list[str]] = {
    "silver":   ["gold_silver", "silver_juniors"],
    "guld":     ["metal_miners", "gdxj_gdx", "gold_gdxj", "gdx_spy"],
    "koppar":   ["copper_gold"],
    "olja":     ["gold_oil"],
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ThemeResult:
    key: str
    label: str
    necessity: int
    proxy_flag: bool = False
    proxy_note: str = ""

    # CYKELPOSITION
    cykel_label: str = "DATA_GAP"   # TIDIG | MITTEN | SEN | TOPP | DATA_GAP
    percentile_10y: Optional[float] = None
    ma200w_slope_pct: Optional[float] = None  # weekly 200W MA slope over last 4 weeks (%)

    # HAT
    hat_score: float = 0.0
    hat_from_5y_high_pct: Optional[float] = None   # % below 5y high (positive = below)
    hat_vol_trend: Optional[float] = None           # 12m volume z-score (negative = declining)
    hat_rs_vs_spy: Optional[float] = None           # 12m RS vs SPY % (negative = underperforming)

    # Blindspot score
    blindspot_score: float = 0.0

    # Current price + sparkline (weekly, last 2y)
    current_price: float = 0.0
    sparkline_dates: list = field(default_factory=list)
    sparkline_values: list = field(default_factory=list)

    # Primary ticker used
    ticker_used: str = ""

    error: Optional[str] = None


# ── Download helper (reuses commodity_ratios pattern) ─────────────────────────

def _download_close(ticker: str, period: str = "10y") -> pd.Series:
    """Robust yfinance Close series download (mirrors commodity_ratios._download_close)."""
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
            return pd.Series(dtype=float)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        col = "Close"
        if col not in df.columns:
            candidates = [c for c in df.columns if str(c).lower() == "close"]
            if not candidates:
                return pd.Series(dtype=float)
            df = df.rename(columns={candidates[0]: col})

        s = df[col].squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        if hasattr(s.index, "tz") and s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s.dropna()

    except Exception as exc:
        logger.debug("_download_close(%s): %s", ticker, exc)
        return pd.Series(dtype=float)


def _download_volume(ticker: str, period: str = "2y") -> pd.Series:
    """Download Volume series via yfinance."""
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
            return pd.Series(dtype=float)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        col = "Volume"
        if col not in df.columns:
            return pd.Series(dtype=float)

        s = df[col].squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        if hasattr(s.index, "tz") and s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s.dropna()

    except Exception as exc:
        logger.debug("_download_volume(%s): %s", ticker, exc)
        return pd.Series(dtype=float)


def _percentile_of(arr: np.ndarray, val: float) -> float:
    if len(arr) == 0:
        return 50.0
    return float(np.searchsorted(np.sort(arr), val, side="right") / len(arr) * 100)


# ── Per-theme computation ─────────────────────────────────────────────────────

def _compute_theme(spec: ThemeSpec) -> ThemeResult:
    res = ThemeResult(
        key=spec.key,
        label=spec.label,
        necessity=spec.necessity,
        proxy_flag=spec.proxy_flag,
        proxy_note=spec.proxy_note,
    )

    # Try tickers in order, use first that has sufficient data
    close_10y: Optional[pd.Series] = None
    ticker_used = ""
    for tkr in spec.primary_tickers:
        s = _download_close(tkr, "10y")
        if not s.empty and len(s) >= _MIN_HISTORY:
            close_10y = s
            ticker_used = tkr
            break

    if close_10y is None or close_10y.empty:
        res.error = f"Kursdata ej tillgänglig för {spec.primary_tickers}"
        return res

    res.ticker_used = ticker_used

    arr_10y  = close_10y.values.astype(float)
    price_now = float(arr_10y[-1])
    res.current_price = round(price_now, 4)

    # ── CYKELPOSITION ─────────────────────────────────────────────────────────

    pct10y = _percentile_of(arr_10y, price_now)
    res.percentile_10y = round(pct10y, 1)

    # 200W MA slope: compare current vs 4 weeks ago
    close_w = close_10y.resample("W").last().dropna()
    ma200w_slope = None
    if len(close_w) >= 204:
        ma200w = close_w.ewm(span=200, adjust=False).mean()
        ma_now  = float(ma200w.iloc[-1])
        ma_4w   = float(ma200w.iloc[-5])
        if ma_4w > 0:
            ma200w_slope = (ma_now / ma_4w - 1) * 100
    res.ma200w_slope_pct = round(ma200w_slope, 2) if ma200w_slope is not None else None

    slope_rising = (ma200w_slope is not None) and (ma200w_slope > 0)

    if pct10y >= _CYKEL_TOPP:
        res.cykel_label = "TOPP"
    elif pct10y >= _CYKEL_SEN:
        res.cykel_label = "SEN"
    elif pct10y <= _CYKEL_TIDIG:
        res.cykel_label = "TIDIG"
    else:
        # MITTEN — refine by slope
        res.cykel_label = "MITTEN" if slope_rising else "TIDIG"

    # ── HAT score ─────────────────────────────────────────────────────────────

    # a) Distance from 5y high
    close_5y = close_10y.iloc[-min(5*252, len(close_10y)):]
    high_5y  = float(close_5y.max())
    pct_from_high = max(0.0, (high_5y - price_now) / high_5y * 100) if high_5y > 0 else 0.0
    # Map 0-100% below 5y high → 0-100 points (cap at 60% below → full score)
    hat_high_raw = min(pct_from_high / 60.0, 1.0) * 100
    res.hat_from_5y_high_pct = round(pct_from_high, 1)

    # b) Volume trend — 12m vs prior 12m (z-score of daily vol change)
    hat_vol_raw = 50.0  # default neutral
    vol_s = _download_volume(ticker_used, "2y")
    if not vol_s.empty and len(vol_s) >= 252:
        vol_arr = vol_s.values.astype(float)
        vol_12m_now  = float(np.mean(vol_arr[-252:]))
        vol_12m_prev = float(np.mean(vol_arr[-504:-252])) if len(vol_arr) >= 504 else float(np.mean(vol_arr[:-252]))
        if vol_12m_prev > 0:
            vol_chg = (vol_12m_now / vol_12m_prev - 1) * 100
            # Declining volume → more hated → higher hat score
            # Map: -50% change → 100 pts, +50% change → 0 pts
            hat_vol_raw = max(0.0, min(100.0, 50.0 - vol_chg))
        res.hat_vol_trend = round(vol_chg if vol_12m_prev > 0 else 0.0, 1)

    # c) RS vs SPY (12 months)
    hat_rs_raw = 50.0  # default neutral
    spy = _download_close("SPY", "2y")
    if not spy.empty and len(spy) >= 252 and len(close_10y) >= 252:
        ticker_ret = float(close_10y.iloc[-1]) / float(close_10y.iloc[-253]) - 1
        spy_ret    = float(spy.iloc[-1]) / float(spy.iloc[-253]) - 1
        rs_12m     = (ticker_ret - spy_ret) * 100
        # Underperforming → more hated → higher hat score
        # Map: -30% RS → 100 pts, +30% RS → 0 pts
        hat_rs_raw = max(0.0, min(100.0, 50.0 - rs_12m / 0.6))
        res.hat_rs_vs_spy = round(rs_12m, 1)

    # Weighted HAT composite
    res.hat_score = round(
        hat_high_raw  * (_HAT_W_5Y_HIGH / 100)
        + hat_vol_raw * (_HAT_W_VOL     / 100)
        + hat_rs_raw  * (_HAT_W_RS      / 100),
        1,
    )

    # ── BLINDSPOT SCORE ───────────────────────────────────────────────────────
    cheapness = 100.0 - pct10y  # 0-100; higher = cheaper
    res.blindspot_score = round(
        (spec.necessity / 100.0) * (res.hat_score / 100.0) * (cheapness / 100.0) * 100.0,
        1,
    )

    # ── Sparkline (weekly, last 2 years) ─────────────────────────────────────
    close_2y   = close_10y.iloc[-min(2*252, len(close_10y)):]
    spark_w    = close_2y.resample("W").last().dropna()
    res.sparkline_dates  = [str(d.date()) for d in spark_w.index]
    res.sparkline_values = [round(float(v), 4) for v in spark_w.values]

    return res


# ── Main cached function ──────────────────────────────────────────────────────

@_cache_12h
def build_theme_board() -> list[ThemeResult]:
    """
    Build all 9 theme results, sorted by blindspot_score descending.
    Cached 12h.
    """
    results = []
    for spec in _THEMES:
        try:
            r = _compute_theme(spec)
        except Exception as exc:
            logger.warning("theme_board: %s failed: %s", spec.key, exc)
            r = ThemeResult(
                key=spec.key, label=spec.label, necessity=spec.necessity,
                proxy_flag=spec.proxy_flag, proxy_note=spec.proxy_note,
                error=str(exc),
            )
        results.append(r)

    results.sort(key=lambda r: r.blindspot_score, reverse=True)
    return results


# ── Swedish verdict text per theme ────────────────────────────────────────────

def theme_verdict_text(r: ThemeResult) -> str:
    """Generate a one-line Swedish verdict for a theme."""
    if r.error:
        return f"{r.label}: DATA SAKNAS — {r.error[:80]}"

    cycle_txt = {
        "TIDIG": "tidig cykel (möjlig bottenkandidatur)",
        "MITTEN": "mittenläge (trend pågår)",
        "SEN":   "sent stadium (var försiktig med nya positioner)",
        "TOPP":  "vid toppen (distribuera, köp ej)",
    }.get(r.cykel_label, r.cykel_label)

    hat_txt = (
        "hatad av marknaden" if r.hat_score >= 60
        else "måttligt ignorerad" if r.hat_score >= 35
        else "relativt populär"
    )

    nec_txt = (
        "civilisationskritisk nödvändighet" if r.necessity >= 85
        else "hög nödvändighet" if r.necessity >= 70
        else "medelhög nödvändighet"
    )

    if r.blindspot_score >= 25:
        action = "— HIGHT POTENTIELL BLINDSPOT — leta bolag i Contrarian Alpha"
    elif r.blindspot_score >= 12:
        action = "— bevaka för läge"
    else:
        action = "— inget tydligt blindspot-läge just nu"

    return f"{r.label}: {cycle_txt}, {hat_txt}, {nec_txt} {action}."
