"""
engine.py — Contrarian Alpha Screener pipeline (Fas 5).

Pipeline stages (in order):
  1. NECESSITY GATE   score >= 60    eliminates low-necessity sectors (SaaS, crypto…)
  2. HATE FILTER      score >= 45    eliminates loved/trending stocks
  3. BALANCE SHEET    FCF>0, D/E<0.6, EBITDA>0%, Equity>0  eliminates weak balance sheets
  4. COMPOSITE RANK   sorted descending

Composite Score (0-100):
  Necessity  × 0.25
  Hat        × 0.25
  Strength   × 0.30
  Catalyst   × 0.15
  VikingBonus× 0.05   (VikingBonus = 100 if OVTLYR regime green, else 0)

Data sources:
  Börsdata Pro+ API (borsdata_api.py) — KPI screener + instrument universe
  yfinance — price history (OHLCV), fallback for missing Börsdata prices
  StockTwits — optional sentiment (hate.py)
  EODHD — optional short interest + analyst data (hate.py)
"""
from __future__ import annotations

import logging
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# ─── Composite weights ────────────────────────────────────────────────────────
# 5-pillar mode-based weights (necessity + hate + quality + value + catalyst = 1.0)
# Viking bonus is an additive flat term on top (+5p when regime is green).

_WEIGHTS: dict[str, dict[str, float]] = {
    "quality": {
        "necessity": 0.15,
        "hate":      0.20,
        "quality":   0.30,
        "value":     0.20,
        "catalyst":  0.15,
    },
    "deep_contrarian": {
        "necessity": 0.15,
        "hate":      0.30,
        "quality":   0.20,
        "value":     0.20,
        "catalyst":  0.15,
    },
}

W_VIKING = 0.05   # VikingBonus: 100 if green else 0  →  flat +5p contribution

# ─── Gate thresholds ─────────────────────────────────────────────────────────

from contrarian_alpha.necessity import NECESSITY_THRESHOLD, get_necessity_score, NecessityEntry
from contrarian_alpha.hate      import HAT_THRESHOLD, calculate_hate_score, HateResult
from contrarian_alpha.hate      import fetch_analyst_data, fetch_short_data
from contrarian_alpha.strength  import calculate_strength_score, StrengthResult
from contrarian_alpha.catalyst  import (
    calculate_catalyst_score, CatalystResult, compute_regime_color, fetch_insider_data,
)
from contrarian_alpha.quality   import (
    calculate_quality_score, QualityResult,
    GATE_ROIC_QUALITY, GATE_ROIC_DEEP,
)
from contrarian_alpha.value     import (
    calculate_value_score, ValueResult,
    check_valuation_bands, ValuationBandsResult,
)

# Börsdata – optional (degrades gracefully when no API key)
try:
    # When deployed inside wolf-shadow-dashboard, borsdata_api.py is in the same
    # package root (parent of contrarian_alpha/).  When running from the standalone
    # wolfpanel dev tree it is three levels up in the sibling Documents folder.
    # Try both locations so the module works in either layout.
    _candidates = [
        Path(__file__).parent.parent,                                          # dashboard root
        Path(__file__).parent.parent.parent / "Documents" / "wolf-shadow-dashboard",  # dev tree
    ]
    for _bd_path in _candidates:
        if _bd_path.exists() and str(_bd_path) not in sys.path:
            sys.path.insert(0, str(_bd_path))
    from borsdata_api import BorsdataAPI, KPI, ALL_NORDIC_MARKETS
    _BORSDATA_AVAILABLE = True
except ImportError:
    _BORSDATA_AVAILABLE = False
    ALL_NORDIC_MARKETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 18, 19]
    logger.warning("borsdata_api not found — universe will be limited to manual tickers")


# ─── Config ───────────────────────────────────────────────────────────────────

# Quality-mode hate floor: compounders need NOT be hated, but we still skip
# obvious blow-off names. 0 = effectively no hate requirement in quality mode.
QUALITY_HATE_FLOOR: float = 0.0


@dataclass
class PipelineConfig:
    """Runtime configuration for run_pipeline()."""

    # Scoring mode — controls pillar weights and quality ROIC gate
    # "quality"         → ROIC>15%, quality weight=30%, hate weight=20%
    # "deep_contrarian" → ROIC>10%, quality weight=20%, hate weight=30%
    mode: str = "quality"

    # Universe
    # "nordic"         → Börsdata Nordic instruments (default, unchanged behavior)
    # "us_ca_resource" → static US/CA resource CSV (PR1 foundation, no scoring change)
    universe: str = "nordic"
    market_ids: list[int] = field(default_factory=lambda: list(ALL_NORDIC_MARKETS))
    # All Nordic: SE Large/Mid/Small/First North/Spotlight/NGM, NO, FI, DK, all exchanges
    include_global: bool   = False   # Requires Börsdata Pro+ global licence
    manual_tickers: list[str] = field(default_factory=list)  # override / supplement

    # Pipeline gates
    necessity_threshold: float = float(NECESSITY_THRESHOLD)   # 60
    hate_threshold:      float = float(HAT_THRESHOLD)          # 45

    # Output
    top_n: int = 25

    # Concurrency
    max_price_workers: int = 6   # parallel yfinance / Börsdata price fetches
    max_fund_workers:  int = 4   # parallel per-instrument report fetches

    # Optional enrichment (slower — requires EODHD key or extra API calls)
    fetch_analyst_data: bool = False
    fetch_short_data:   bool = False
    fetch_sector_etf:   bool = False   # sector-relative performance

    # Progress callback: f(done: int, total: int, ticker: str)
    progress_cb: Callable[[int, int, str], None] | None = None


# ─── Result models ────────────────────────────────────────────────────────────

@dataclass
class ContrairianAlphaResult:
    """Full result for a single instrument passing the pipeline."""

    # Identity
    ticker:     str
    ins_id:     int | None
    name:       str
    market:     str
    sector:     str
    branch:     str

    # Composite
    composite_score: float
    rank: int = 0

    # Sub-scores (0-100 each)
    necessity_score:  float = 0.0
    hat_score:        float = 0.0
    strength_score:   float = 0.0
    catalyst_score:   float = 0.0
    quality_score:    float = 0.0
    value_score:      float = 0.0
    viking_bonus_raw: float = 0.0  # 100 or 0
    viking_bonus_pts: float = 0.0  # flat +5p when green

    # Rich sub-results (for UI drilldown)
    necessity_entry:  NecessityEntry | None    = None
    strength_result:  StrengthResult | None    = None
    hate_result:      HateResult | None        = None
    catalyst_result:  CatalystResult | None    = None
    quality_result:   "QualityResult | None"   = None
    value_result:     "ValueResult | None"     = None

    # Pipeline flags
    eliminated:           bool       = False
    elimination_stage:    str        = ""   # "NECESSITY" | "HATE" | "BALANCE_SHEET"
    elimination_reason:   str        = ""
    all_flags:            list[str]  = field(default_factory=list)

    # Resource-universe metadata (us_ca_resource only; empty for Nordic)
    stage:               str        = ""   # producer|developer|explorer|royalty|energy|services
    primary_commodity:   str        = ""
    resource_flags:      list[str]  = field(default_factory=list)  # stage guardrail notes
    resource_gate_mode:  str        = ""   # ""|"RELAXED"|"MATURE" — see PR2 guardrail

    # Resource-composite v1 (PR3; us_ca_resource only, 0.0 for Nordic)
    resource_composite:  float      = 0.0  # blended resource score 0-100
    survival_score:      float      = 0.0
    dilution_score:      float      = 0.0
    jurisdiction_score:  float      = 0.0
    commodity_score:     float      = 0.0
    resource_confidence: float      = 0.0  # 0-1 data-quality confidence
    resource_stage_profile: str     = ""   # stage weight table used
    # Resource enrichment transparency (PR4; us_ca_resource only)
    resource_data_quality: str      = ""   # HIGH|MEDIUM|LOW (blank for Nordic)
    commodity_proxy:      str        = ""   # ETF/index proxies (metadata only)
    resource_data_as_of:  str        = ""   # enrichment date (may be blank)

    # Existing-source overlay (PR5; us_ca_resource only). Context/watchlist only,
    # NOT a buy trigger and separate from resource_composite. Blank for Nordic.
    resource_overlay_score:    float       = 0.0   # 0-100 cautious overlay score
    market_cap_bucket:         str         = ""    # nano|micro|small|mid|large|unknown
    liquidity_flag:            str         = ""    # OK|THIN|LOW|UNKNOWN
    drawdown_52w_pct:          float | None = None
    commodity_relative_strength: float | None = None  # candidate−proxy %return (pp)
    commodity_rs_flag:         str         = ""    # OUTPERFORMING_PROXY|LAGGING_PROXY|RS_NEUTRAL|COMMODITY_RS_NOT_AVAILABLE
    short_interest_flag:       str         = ""    # HIGH|ELEVATED|NORMAL|UNKNOWN
    analyst_revision_flag:     str         = ""    # NET_DOWNGRADES|NET_UPGRADES|NEUTRAL|UNKNOWN
    sentiment_attention_flag:  str         = ""    # LOW_ATTENTION|HYPE_RISK|NORMAL_ATTENTION|SENTIMENT_NOT_AVAILABLE (placeholder in live pipeline)
    macro_context_flag:        str         = ""    # COMMODITY_MACRO_TAILWIND|COMMODITY_MACRO_HEADWIND|MACRO_NEUTRAL|MACRO_CONTEXT_NOT_AVAILABLE
    existing_source_flags:     list[str]   = field(default_factory=list)

    # Price snapshot (for UI display)
    close:     float = 0.0
    sma50:     float = 0.0
    sma200:    float = 0.0
    high_52w:  float = 0.0
    low_52w:   float = 0.0

    # Fundamental snapshot
    fcf_m:        float | None = None
    ebitda_pct:   float | None = None   # EBITDA margin %
    debt_equity:  float | None = None
    equity_m:     float | None = None
    ev_ebitda:    float | None = None
    altman_z:       float | None = None
    avg_volume_20d: float | None = None   # for LOW_LIQUIDITY flag in flags.py
    roic:           float | None = None   # ROIC % (Quality model)
    p_fcf:          float | None = None   # P/FCF multiple (Value model)

    # KAP quality-mode fields (None in deep_contrarian mode)
    net_debt_ebitda:   float | None = None
    dividend_yield_pct: float | None = None
    pe_ratio:          float | None = None
    ev_ebit_ratio:     float | None = None
    revenue_cagr_5y:   float | None = None
    revenue_cagr_10y:  float | None = None
    eps_cagr_10y:      float | None = None
    kap_badge:         bool = False
    valuation_bands:   "ValuationBandsResult | None" = None

    # Meta
    data_confidence: float = 0.0
    timestamp: str = ""

    @property
    def passes_all_gates(self) -> bool:
        return not self.eliminated

    @property
    def value_trap(self) -> bool:
        return "POTENTIAL_VALUE_TRAP" in self.all_flags


@dataclass
class PipelineResult:
    """Full output from run_pipeline()."""
    results:          list[ContrairianAlphaResult]   # top_n passing instruments
    eliminated:       list[ContrairianAlphaResult]   # all eliminated (for diagnostics)
    timestamp:        str
    universe_count:   int
    necessity_passed: int
    hate_passed:      int
    bs_passed:        int
    composite_ranked: int
    run_duration_s:   float
    config:           PipelineConfig
    delisted_count:   int = 0  # tickers skipped because yfinance returned no price data

    @property
    def pass_rates(self) -> dict[str, str]:
        def _pct(n, d):
            return f"{n}/{d} ({n/d*100:.0f}%)" if d else "0/0"
        return {
            "necessity":  _pct(self.necessity_passed,  self.universe_count),
            "hate":       _pct(self.hate_passed,        self.necessity_passed),
            "balance_sheet": _pct(self.bs_passed,      self.hate_passed),
            "ranked":     _pct(self.composite_ranked,   self.bs_passed),
        }


# ─── Market suffix map (Börsdata marketId → yfinance suffix) ──────────────────

_MARKET_SUFFIX: dict[int, str] = {
    # Sweden
    1:  ".ST",   # Large Cap
    2:  ".ST",   # Mid Cap
    3:  ".ST",   # Small Cap
    7:  ".ST",   # First North
    8:  ".ST",   # Spotlight
    9:  ".ST",   # NGM
    18: ".ST",   # Mid Cap (alt ID)
    19: ".ST",   # Small Cap (alt ID)
    # Norway
    4:  ".OL",   # Oslo Børs
    14: ".OL",   # Euronext Growth
    # Finland
    5:  ".HE",   # Helsinki
    16: ".HE",   # First North
    # Denmark
    6:  ".CO",   # Copenhagen
    15: ".CO",   # First North
    # Global
    11: "",      # US (no suffix)
    12: ".TO",   # Canada TSX
}

_MARKET_NAME: dict[int, str] = {
    1:  "SE Large",       2:  "SE Mid",         3:  "SE Small",
    7:  "SE First North", 8:  "SE Spotlight",    9:  "SE NGM",
    18: "SE Mid",         19: "SE Small",
    4:  "NO",             14: "NO Euronext",
    5:  "FI",             16: "FI First North",
    6:  "DK",             15: "DK First North",
    11: "US",             12: "CA",
}

# yfinance sector/industry → Börsdata-compatible name for necessity lookup
_YF_SECTOR_MAP: dict[str, str] = {
    "Energy":                  "energy",
    "Basic Materials":         "materials",
    "Industrials":             "industrials",
    "Consumer Cyclical":       "consumer discretionary",
    "Consumer Defensive":      "consumer staples",
    "Healthcare":              "health care",
    "Financial Services":      "financials",
    "Technology":              "information technology",
    "Communication Services":  "communication services",
    "Utilities":               "utilities",
    "Real Estate":             "real estate",
}

def _yf_sector_name(ticker: str) -> tuple[str, str]:
    """Return (sector_name, industry_name) from yfinance for necessity fallback."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        sector   = _YF_SECTOR_MAP.get(info.get("sector", ""), info.get("sector", "").lower())
        industry = info.get("industry", "").lower()
        return sector, industry
    except Exception:
        return "", ""


# ─── Price metrics ────────────────────────────────────────────────────────────

def _compute_price_metrics(df) -> dict:
    """
    Derive all price metrics needed by hate.py + catalyst.py from a
    pandas OHLCV DataFrame (DatetimeIndex, columns Open/High/Low/Close/Volume).
    Returns empty dict if df is invalid.
    """
    try:
        import pandas as pd

        close = df["Close"].dropna()
        if len(close) < 20:
            return {}

        current_close = float(close.iloc[-1])

        # SMAs
        sma50  = float(close.tail(50).mean()) if len(close) >= 50 else current_close
        sma200 = float(close.tail(200).mean()) if len(close) >= 200 else current_close

        # Average close over all available history (used for cycle position detection)
        avg_price_5y = float(close.mean()) if len(close) >= 20 else None

        # 52-week range
        high = df["High"].dropna()
        low  = df["Low"].dropna()
        high_52w = float(high.tail(252).max()) if len(high) >= 20 else float(high.max())
        low_52w  = float(low.tail(252).min())  if len(low)  >= 20 else float(low.min())

        # Volume
        vol = df["Volume"].dropna()
        current_vol   = float(vol.iloc[-1])            if len(vol) > 0 else 0.0
        avg_vol_20d   = float(vol.tail(20).mean())     if len(vol) >= 20 else 0.0
        std_vol_20d   = float(vol.tail(20).std())      if len(vol) >= 20 else 0.0
        # 6-månaderssnitt för volymtorkan (hate.py) — kräver rimlig historik
        # så kvoten 20d/6m inte jämför fönstret med sig självt.
        avg_vol_6m    = float(vol.tail(126).mean())    if len(vol) >= 60 else None

        # SMA50 slope (linear regression over last 10 bars of SMA50)
        sma50_slope = 0.0
        if len(close) >= 60:
            sma50_series = close.rolling(50).mean().dropna()
            if len(sma50_series) >= 10:
                recent = sma50_series.tail(10).values
                x = np.arange(10)
                try:
                    sma50_slope = float(np.polyfit(x, recent, 1)[0])
                except (np.linalg.LinAlgError, ValueError):
                    pass

        # Close history — newest-first list, 15 bars for reversal detection
        history_len = min(15, len(close))
        close_history = [float(close.iloc[-(i+1)]) for i in range(history_len)]

        return {
            "close":          round(current_close, 4),
            "sma50":          round(sma50, 4),
            "sma200":         round(sma200, 4),
            "high_52w":       round(high_52w, 4),
            "low_52w":        round(low_52w, 4),
            "current_volume": current_vol,
            "avg_volume_20d": avg_vol_20d,
            "std_volume_20d": std_vol_20d,
            "avg_volume_6m":  avg_vol_6m,
            "sma50_slope":    round(sma50_slope, 6),
            "close_history":  close_history,
            "avg_price_5y":   round(avg_price_5y, 4) if avg_price_5y is not None else None,
            "confidence":     1.0 if len(close) >= 200 else 0.7,
        }
    except Exception as e:
        logger.debug("_compute_price_metrics failed: %s", e)
        return {}


def _fetch_price_df(ticker: str, ins_id: int | None, api) -> object:
    """
    Fetch OHLCV DataFrame for a ticker.
    Tries Börsdata first (if api available + ins_id known), falls back to yfinance.
    Checks the price TTLCache (1 h) before making any network call.
    Returns empty DataFrame on failure.
    """
    import pandas as pd
    try:
        from contrarian_alpha.cache import get_price, set_price
        _cache_key = f"df:{ticker}:{ins_id}"
        cached = get_price(_cache_key)
        if cached is not None:
            return cached
    except Exception:
        get_price = set_price = None
        _cache_key = None

    df = None

    # Börsdata path — 5 år (~1250 handelsdagar) så cykelpositionen i hate.py
    # faktiskt mäter mot ett flerårssnitt (300 dagar gjorde "5y" till 14 mån).
    # Samma antal API-anrop, bara större svar; yfinance-vägen hämtar redan 5y.
    if api is not None and ins_id is not None:
        try:
            df = api.get_stockprices_df(ins_id, max_count=1300)
            if df is not None and len(df) < 50:
                df = None
        except Exception as e:
            logger.debug("Börsdata price failed for %s (id=%s): %s", ticker, ins_id, e)

    # yfinance fallback
    if df is None:
        try:
            from contrarian_alpha.cache import is_delisted, mark_delisted
        except Exception:
            is_delisted = lambda t: False
            mark_delisted = lambda t: None

        if not is_delisted(ticker):
            try:
                import yfinance as yf
                import logging as _std_logging
                _yf_log = _std_logging.getLogger("yfinance")
                _prev_level = _yf_log.level
                _yf_log.setLevel(_std_logging.CRITICAL)
                try:
                    raw = yf.Ticker(ticker).history(period="5y", auto_adjust=True, progress=False)
                finally:
                    _yf_log.setLevel(_prev_level)
                if raw is not None and not raw.empty:
                    raw.index  = raw.index.tz_localize(None) if hasattr(raw.index, "tz") and raw.index.tz else raw.index
                    raw.columns = [c.capitalize() for c in raw.columns]
                    df = raw
                else:
                    mark_delisted(ticker)
            except Exception as e:
                mark_delisted(ticker)
                logger.debug("yfinance price failed for %s: %s", ticker, e)

    result = df if df is not None else pd.DataFrame()
    if set_price is not None and _cache_key and not result.empty:
        try:
            set_price(_cache_key, result)
        except Exception:
            pass
    return result


# ─── Fundamentals dict builder ────────────────────────────────────────────────

def _build_fundamentals_dict(snapshot: dict, reports: list[dict] | None = None) -> dict:
    """
    Map a Börsdata KPI snapshot + optional reports list to the flat fundamentals
    dict expected by strength.py's calculate_strength_score().

    Börsdata KPI divisor notes (from borsdata_api.py KPI dict):
      debt_to_equity  returned as % (×100) by get_kpi_screener  → divide by 100
      ebitda_margin   returned as % (×100)                       → divide by 100
      equity_ratio    returned as % (×100)                       → divide by 100
      fcf_m           raw MSEK                                    → keep as-is
    """
    fund: dict = {}

    # FCF (TTM from r12 or latest annual)
    fcf_m = snapshot.get("fcf_m")
    if fcf_m is not None:
        fund["fcf"] = float(fcf_m)   # MSEK — sign is what matters for the gate

    # EBITDA margin %
    em = snapshot.get("ebitda_margin")
    if em is not None:
        # borsdata_api divides by 100 for ebitda_margin (see get_fundamentals_snapshot_fast)
        # but raw screener value is %, so check divisor applied upstream
        fund["ebitda_margin"] = float(em)

    # D/E ratio — borsdata returns as raw ratio after /100 in snapshot_fast
    de = snapshot.get("debt_to_equity")
    if de is not None:
        fund["debt_to_equity"] = float(de)

    # Equity (absolute MSEK)
    eq = snapshot.get("total_equity_m")
    if eq is not None:
        fund["equity"] = float(eq)

    # Market cap (MSEK)
    mc = snapshot.get("market_cap")
    if mc is not None:
        fund["market_cap"] = float(mc)

    # EV/EBITDA
    ev_e = snapshot.get("ev_ebitda")
    if ev_e is not None:
        fund["ev_ebitda"] = float(ev_e)

    # Revenue (MSEK) — also Altman X5 numerator
    rev = snapshot.get("revenue_m")
    if rev is not None:
        fund["revenue"] = float(rev)

    # Total assets (MSEK) — Altman X1/X2/X3/X5 denominator
    ta = snapshot.get("total_assets_m")
    if ta is not None:
        fund["total_assets"] = float(ta)

    # EBIT (MSEK) — Altman X3 numerator
    ebit = snapshot.get("ebit_m")
    if ebit is not None:
        fund["ebit"] = float(ebit)

    # Total liabilities (Altman X4 denominator). No direct Börsdata KPI, so
    # derive it from assets − equity when both are present (positive only).
    if fund.get("total_assets") is not None and fund.get("equity") is not None:
        _tl = fund["total_assets"] - fund["equity"]
        if _tl > 0:
            fund["total_liabilities"] = _tl

    # FCF yield (derive from market cap if available)
    if fund.get("fcf") and fund.get("market_cap") and fund["market_cap"] > 0:
        fund["fcf_yield"] = round(fund["fcf"] / fund["market_cap"] * 100, 2)

    # FCF history from reports (newest-first)
    if reports:
        fcf_hist = []
        for r in sorted(reports, key=lambda x: x.get("year", 0), reverse=True)[:3]:
            v = r.get("freeCashFlow")
            if v is not None:
                fcf_hist.append(float(v))
        if fcf_hist:
            fund["fcf_history"] = fcf_hist
            fund["fcf"] = fcf_hist[0]   # use actual TTM

    return fund


# ─── Quality / Value data builders ───────────────────────────────────────────

def _fetch_kpi_history(ins_id: int, kpi_id: int, api) -> list[float]:
    """
    Return newest-first list of annual KPI float values for one instrument.
    Uses fundamentals TTLCache (24 h).  Returns [] on any failure.
    """
    _cset = None
    _ckey = None
    try:
        from contrarian_alpha.cache import get_fundamentals as _cget, set_fundamentals as _cset
        _ckey = f"kpi_hist:{ins_id}:{kpi_id}"
        hit = _cget(_ckey)
        if hit is not None:
            return hit
    except Exception:
        pass
    try:
        raw = api.get_kpi_history(ins_id, kpi_id, "year", "mean")
        # Börsdata returns oldest-first → reverse for newest-first
        vals = [float(r["v"]) for r in raw if r.get("v") is not None]
        vals.reverse()
        if _cset and _ckey:
            try:
                _cset(_ckey, vals)
            except Exception:
                pass
        return vals
    except Exception as e:
        logger.debug("_fetch_kpi_history(%s, kpi=%s) failed: %s", ins_id, kpi_id, e)
        return []


# Critical snapshot fields for the balance-sheet gate + Altman Z, mapped to the
# Börsdata KPI id and the divisor get_fundamentals_snapshot_fast applies (so a
# history back-fill scales identically to a successful batch). Some of these
# KPIs (total_equity_m 58, total_assets_m 57, ebit_m 55) are NOT fetched by the
# batch at all — for those, equity/Altman are always missing without this
# fallback; the rest are back-filled only when their batch screener 400'd.
def _critical_kpi_fallback() -> dict[str, tuple[int, float]]:
    return {
        "fcf_m":          (KPI["fcf_m"],          1),    # 63  FCF (TTM) gate
        "ebitda_margin":  (KPI["ebitda_margin"],  100),  # 32  EBITDA-margin gate
        "debt_to_equity": (KPI["debt_to_equity"], 100),  # 40  D/E gate
        "total_equity_m": (KPI["total_equity_m"], 1),    # 58  Equity > 0 gate
        "total_assets_m": (KPI["total_assets_m"], 1),    # 57  Altman denominator
        "ebit_m":         (KPI["ebit_m"],         1),    # 55  Altman X3
        "revenue_m":      (KPI["revenue_m"],      1),    # 53  Altman X5
        "market_cap":     (KPI["market_cap"],     1),    # 50  Altman X4 numerator
        "ev_ebitda":      (KPI["ev_ebitda"],      1),    # 11  strength EV/EBITDA
    }


def _augment_fund_snap_from_history(fund_snap: dict, ins_id: int | None, api) -> list[int]:
    """
    Root-cause data-layer fix: back-fill critical snapshot fields the Börsdata
    batch left None (400 / not in licence, or never batch-fetched) from the
    per-instrument KPI history (already TTL-cached 24 h by _fetch_kpi_history),
    scaling each value with the batch's own divisor so downstream builders see a
    value identical to a successful batch.

    PERFORMANCE GUARD: a KPI is only fetched when the field is missing AND the
    batch either failed on that KPI (KPI_BATCH_FAILED) or never attempted it —
    KPIs the batch retrieved fine are never re-fetched. Runs survivor-only (past
    necessity + hate elimination) and respects the API's shared rate limiter.

    Returns the list of kpi_ids that were back-filled (for logging / flags).
    """
    if not fund_snap or ins_id is None or api is None:
        return []

    try:
        import borsdata_api as _bd
        failed = _bd.KPI_BATCH_FAILED
        attempted = _bd.KPI_BATCH_ATTEMPTED
    except Exception:
        failed = attempted = set()

    filled: list[int] = []
    for key, (kpi_id, divisor) in _critical_kpi_fallback().items():
        if fund_snap.get(key) is not None:
            continue  # batch already has a value — nothing to do
        # Skip KPIs the batch fetched successfully (attempted and not failed):
        # a None there means the value genuinely does not exist, so fetching
        # history would just waste a call. Fetch only for failed / never-tried.
        if kpi_id in attempted and kpi_id not in failed:
            continue
        hist = _fetch_kpi_history(ins_id, kpi_id, api)
        if hist:
            val = hist[0]
            fund_snap[key] = val / divisor if divisor != 1 else val
            filled.append(kpi_id)

    if filled:
        logger.debug(
            "BS history fallback for ins_id=%s back-filled KPI ids %s",
            ins_id, filled,
        )
    return filled


def _build_quality_data(
    fund_snap: dict,
    ins_id: int | None,
    api,
    mode: str = "quality",
) -> dict:
    """
    Assemble quality_data dict for calculate_quality_score().
    Current values come from the batch snapshot (fractions → converted to %).
    History is fetched per-instrument (only for survivors, cached 24 h).
    When mode='quality', also fetches revenue/EPS CAGR from annual reports
    and adds dividend_yield_pct.
    """
    data: dict = {}

    # Current values (snapshot stores % KPIs as fractions → multiply by 100)
    roic_frac = fund_snap.get("roic")
    if roic_frac is not None:
        data["roic"] = roic_frac * 100

    gm_frac = fund_snap.get("gross_margin")
    if gm_frac is not None:
        data["gross_margin"] = gm_frac * 100

    om_frac = fund_snap.get("operating_margin")
    if om_frac is not None:
        data["operating_margin"] = om_frac * 100

    # Dividend yield — snapshot stores as fraction (KPI 1 ÷ 100); convert to %
    div_frac = fund_snap.get("dividend_yield")
    if div_frac is not None:
        data["dividend_yield_pct"] = round(float(div_frac) * 100, 2)

    if ins_id is None or api is None:
        return data

    # ROIC history (KPI 37) — raw values are already in %, no conversion
    roic_hist = _fetch_kpi_history(ins_id, KPI["roic"], api)
    if roic_hist:
        data["roic_history"] = roic_hist
        # Fallback: if the batch snapshot lacked current ROIC, use the most
        # recent history value so the quality ROIC gate can actually evaluate.
        if "roic" not in data:
            data["roic"] = roic_hist[0]

    # ROCE (KPI 36) — fetch from history, use most recent as current
    roce_hist = _fetch_kpi_history(ins_id, KPI["roc"], api)
    if roce_hist:
        data["roce"] = roce_hist[0]   # most recent ROCE %

    # Gross margin history (KPI 28) — raw % values
    gm_hist = _fetch_kpi_history(ins_id, KPI["gross_margin"], api)
    if gm_hist:
        data["gross_margin_history"] = gm_hist
        if "gross_margin" not in data:
            data["gross_margin"] = gm_hist[0]

    # Operating margin history (KPI 29) — raw % values
    om_hist = _fetch_kpi_history(ins_id, KPI["operating_margin"], api)
    if om_hist:
        data["op_margin_history"] = om_hist
        if "operating_margin" not in data:
            data["operating_margin"] = om_hist[0]

    # Quality-mode only: true CAGR figures from annual reports (per-survivor, disk-cached)
    if mode == "quality":
        try:
            growth = api.get_growth_history(ins_id, years=11)
            rev5  = growth.get("revenue_cagr_5y")
            rev10 = growth.get("revenue_cagr_10y")
            earn10 = growth.get("earnings_cagr_10y")
            if rev5  is not None:
                data["revenue_cagr_5y"]  = round(float(rev5)   * 100, 2)
            if rev10 is not None:
                data["revenue_cagr_10y"] = round(float(rev10)  * 100, 2)
            if earn10 is not None:
                data["eps_cagr_10y"]     = round(float(earn10) * 100, 2)
        except Exception as e:
            logger.debug("get_growth_history(%s) failed: %s", ins_id, e)

    return data


def _build_value_data(fund_snap: dict, ins_id: int | None, api) -> dict:
    """
    Assemble value_data dict for calculate_value_score().
    P/FCF and EV/EBITDA are raw multiples (divisor=1 in snapshot).
    """
    data: dict = {}

    p_fcf = fund_snap.get("p_fcf")
    if p_fcf is not None:
        data["p_fcf"] = p_fcf

    ev_e = fund_snap.get("ev_ebitda")
    if ev_e is not None:
        data["ev_ebitda"] = ev_e

    if ins_id is None or api is None:
        return data

    # P/FCF history (KPI 76) — raw multiples, no conversion
    pfcf_hist = _fetch_kpi_history(ins_id, KPI["p_fcf"], api)
    if pfcf_hist:
        data["p_fcf_history"] = pfcf_hist

    # EV/EBITDA history (KPI 11) — raw multiples
    ev_hist = _fetch_kpi_history(ins_id, KPI["ev_ebitda"], api)
    if ev_hist:
        data["ev_ebitda_history"] = ev_hist

    return data


# ─── Resource stage-aware guardrails (PR2) ────────────────────────────────────
#
# GUARDRAIL, NOT full resource scoring. The mature-company hard gates (FCF>0,
# EBITDA margin>0, Altman Z, ROIC, positive equity) were designed for Nordic
# Börsdata fundamentals. Applied unchanged to the US/CA resource universe they
# wrongly eliminate pre-revenue juniors (explorers/developers legitimately burn
# cash and have no ROIC) and data-sparse US/CA rows (ins_id is None → no
# Börsdata snapshot at all). This helper relaxes those gates for the resource
# universe only, converting the failures into transparent flags/notes. Full
# resource composite (survival/cash runway, dilution, jurisdiction, stage
# weights, commodity/regime triggers) is deferred to PR3.

# Pre-revenue stages: never hard-eliminate on missing/negative fundamentals.
_RESOURCE_PRE_REVENUE_STAGES = {"explorer", "developer"}
# Mature/cash-flowing stages: keep gates when data is PRESENT, but never
# eliminate solely because Börsdata-style fundamentals are missing in US/CA.
_RESOURCE_MATURE_STAGES = {"producer", "energy", "services", "royalty"}


def _apply_resource_stage_guardrails(
    stage:      str,
    fund_dict:  dict,
    gates:      dict[str, bool],
    roic:       float | None,
    bs_failures: list[str],
) -> tuple[list[str], bool, str, list[str]]:
    """
    Stage-aware guardrail for the us_ca_resource universe.

    Decides which balance-sheet / ROIC eliminations to suppress and produces
    transparency flags. This does NOT alter any score math — it only rewrites
    elimination reasons. Nordic behavior is untouched because the caller only
    invokes this for config.universe == "us_ca_resource".

    Args:
        stage:       resource stage (lowercased); may be unknown/empty.
        fund_dict:   flat fundamentals dict (values None when data missing).
        gates:       strength gate_results (fcf_positive, ebitda_margin_positive,
                     equity_positive, debt_equity_low).
        roic:        ROIC % or None (missing).
        bs_failures: balance-sheet failure reasons computed by the caller.

    Returns:
        (kept_bs_failures, drop_roic_gate, gate_mode, flags)
          kept_bs_failures : subset of bs_failures that should STILL eliminate
          drop_roic_gate   : True → caller must NOT eliminate on ROIC
          gate_mode        : "RELAXED" (pre-revenue) | "MATURE" (data-relaxed)
          flags            : list[str] notes for ResourceFlags / all_flags
    """
    stg = (stage or "").strip().lower()
    flags: list[str] = []

    # Data presence (None = Börsdata-style fundamental simply absent for US/CA)
    fcf_present     = fund_dict.get("fcf") is not None
    ebitda_present  = fund_dict.get("ebitda_margin") is not None
    equity_present  = fund_dict.get("equity") is not None
    leverage_present = (
        fund_dict.get("debt_to_equity") is not None
        or fund_dict.get("net_debt_ebitda") is not None
    )

    pre_revenue = stg in _RESOURCE_PRE_REVENUE_STAGES

    if pre_revenue:
        # Explorers/developers: relax FCF, EBITDA, equity and ROIC entirely
        # (present-but-negative OR missing). D/E stays enforced only when data
        # exists (juniors normally carry little debt; a genuinely over-levered
        # junior should still be flagged out).
        gate_mode = "RELAXED"
        flags.append("PRE_REVENUE")
        flags.append("STAGE_AWARE_GATE_RELAXED")
        if not gates.get("fcf_positive", False):
            flags.append("NO_FCF_EXPECTED")
        if not gates.get("ebitda_margin_positive", False):
            flags.append("NO_EBITDA_EXPECTED")
        # ROIC is not a meaningful metric for a pre-revenue miner.
        flags.append("ROIC_NOT_APPLICABLE")

        # D/E stays enforced only when data actually exists (comment above);
        # a missing leverage figure must not hard-eliminate a curated junior.
        kept = [
            f for f in bs_failures
            if (f.startswith("D/E") or f.startswith("Net Debt")) and leverage_present
        ]
        if any(f.startswith("D/E") or f.startswith("Net Debt") for f in bs_failures) \
                and not leverage_present:
            flags.append("LEVERAGE_DATA_MISSING")
        drop_roic_gate = True
        return kept, drop_roic_gate, gate_mode, flags

    # Mature/cash-flowing stages (producer/energy/services/royalty) and any
    # unknown stage: keep the gates, but never eliminate purely on MISSING data.
    gate_mode = "MATURE"
    kept: list[str] = []
    for f in bs_failures:
        if f.startswith("FCF") and not fcf_present:
            flags.append("FCF_DATA_MISSING")
        elif f.startswith("EBITDA") and not ebitda_present:
            flags.append("EBITDA_DATA_MISSING")
        elif f.startswith("Equity") and not equity_present:
            flags.append("EQUITY_DATA_MISSING")
        elif (f.startswith("D/E") or f.startswith("Net Debt")) and not leverage_present:
            flags.append("LEVERAGE_DATA_MISSING")
        else:
            # Data present and failing (real weakness) → keep the elimination.
            kept.append(f)

    # ROIC gate is only dropped when data is missing (mature rows with real,
    # sub-threshold ROIC are still legitimately eliminated by the caller).
    drop_roic_gate = roic is None
    if drop_roic_gate:
        flags.append("ROIC_DATA_MISSING")

    if flags:
        flags.append("STAGE_AWARE_MISSING_DATA_RELAXED")
    return kept, drop_roic_gate, gate_mode, flags


# ─── Resource composite v1 (PR3) ──────────────────────────────────────────────
#
# Additive scoring layer for surviving us_ca_resource rows. Blends the
# resource-specific factors (survival/dilution/jurisdiction/commodity) with the
# pipeline's existing hate/catalyst/quality/value scores via stage-aware
# weights. This does NOT change composite_score (the Nordic Contrarian Alpha
# ranking) — it writes separate resource_* fields so the two rankings stay
# independent and Nordic behavior is untouched.

def _apply_resource_composite(
    result: "ContrairianAlphaResult",
    inst_info: dict,
    fund_snap: dict,
) -> None:
    """Populate result.resource_composite + sub-scores/flags (in place)."""
    from contrarian_alpha.resource_scoring import compute_resource_composite

    meta = inst_info.get("resource_meta") or inst_info
    # Quality/value are only meaningful when fundamentals existed; otherwise pass
    # None so the composite treats them as neutral instead of a real 0.
    _has_fund = bool(fund_snap)
    q = result.quality_score if _has_fund else None
    v = result.value_score if _has_fund else None

    rs = compute_resource_composite(
        stage             = result.stage,
        meta              = meta,
        country           = inst_info.get("country", "") or meta.get("country", ""),
        exchange          = inst_info.get("exchange", "") or meta.get("exchange", ""),
        primary_commodity = result.primary_commodity or meta.get("primary_commodity", ""),
        secondary_commodity = inst_info.get("secondary_commodity", "")
                              or meta.get("secondary_commodity", ""),
        hate_score        = result.hat_score,
        catalyst_score    = result.catalyst_score,
        quality_score     = q,
        value_score       = v,
    )

    result.resource_composite     = rs.resource_composite
    result.survival_score         = rs.survival_score
    result.dilution_score         = rs.dilution_score
    result.jurisdiction_score     = rs.jurisdiction_score
    result.commodity_score        = rs.commodity_score
    result.resource_confidence    = rs.resource_confidence
    result.resource_stage_profile = rs.stage_profile
    result.resource_data_quality  = rs.resource_data_quality
    result.commodity_proxy        = rs.commodity_proxy
    result.resource_data_as_of    = rs.data_as_of

    for f in rs.flags:
        if f not in result.resource_flags:
            result.resource_flags.append(f)
        if f not in result.all_flags:
            result.all_flags.append(f)


# Existing-source overlay (PR5). Additive context layer built from data the
# pipeline has already fetched from EXISTING sources (yfinance price snapshot,
# EODHD/yfinance analyst + short dicts, static CSV shares_out_m). Makes no new
# network calls of its own, adds no dependencies, and is kept SEPARATE from
# resource_composite — it never changes the PR3 composite math.

def _closes_from_df(df) -> list[float]:
    """Oldest-first Close list from an OHLCV DataFrame; [] when unavailable."""
    if df is None:
        return []
    try:
        if getattr(df, "empty", True):
            return []
        col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
        if col is None:
            return []
        return [float(v) for v in df[col].dropna().tolist()]
    except Exception:
        return []


def _fetch_resource_macro_context(api=None) -> dict | None:
    """
    Fetch the FRED T10Y2Y yield-curve reading ONCE per run via the project's
    existing disk-cached helper (ember.fred_cache). Returns a plain dict
    {"t10y2y", "t10y2y_change_4w"} for classify_macro_context, or None on any
    failure so the overlay falls back to MACRO_CONTEXT_NOT_AVAILABLE.

    This is the single controlled, cached macro network touch of a run — never
    called from the pure scoring helpers or from tests.
    """
    try:
        from ember.fred_cache import fetch_t10y2y_values
        try:
            from ember.config import FRED_T10Y2Y_URL
        except Exception:
            FRED_T10Y2Y_URL = (
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y"
            )
        vals, _is_stale = fetch_t10y2y_values(FRED_T10Y2Y_URL)
        if not vals or len(vals) < 25:
            return None
        current = float(vals[-1])
        change_4w = current - float(vals[-21])   # ~20 business days ≈ 4 weeks
        return {"t10y2y": round(current, 3), "t10y2y_change_4w": round(change_4w, 3)}
    except Exception as e:
        logger.debug("resource macro context unavailable: %s", e)
        return None


def _apply_existing_source_overlay(
    result: "ContrairianAlphaResult",
    inst_info: dict,
    fund_snap: dict,
    analyst_dict: dict | None,
    short_dict: dict | None,
    price_df=None,
    api=None,
    macro_context: dict | None = None,
) -> None:
    """Populate result.resource_overlay_score + overlay flags (in place)."""
    from contrarian_alpha.existing_source_enrichment import enrich_resource_candidate

    meta = inst_info.get("resource_meta") or inst_info
    # Börsdata market cap is MSEK and absent for US/CA rows (ins_id=None); pass it
    # only when present so the overlay falls back to a flagged CSV estimate.
    mcap = fund_snap.get("market_cap") if fund_snap else None

    # Commodity relative strength: candidate close series (already fetched) vs.
    # the commodity-proxy ETF series (first symbol of the proxy metadata the
    # resource composite attached), fetched through the SAME cached yfinance path
    # so repeat proxies across the run cost one network call each.
    candidate_closes = _closes_from_df(price_df)
    proxy_closes: list[float] = []
    proxy_field = (result.commodity_proxy or "").strip()
    if candidate_closes and proxy_field:
        proxy_ticker = proxy_field.split(",")[0].strip()
        if proxy_ticker:
            try:
                proxy_closes = _closes_from_df(_fetch_price_df(proxy_ticker, None, api))
            except Exception as e:
                logger.debug("proxy price fetch failed for %s: %s", proxy_ticker, e)

    ov = enrich_resource_candidate(
        close            = result.close,
        high_52w         = result.high_52w,
        low_52w          = result.low_52w,
        avg_volume_20d   = result.avg_volume_20d,
        meta             = meta,
        analyst_data     = analyst_dict,
        short_data       = short_dict,
        market_cap_usd   = mcap,
        macro            = macro_context,
        candidate_closes = candidate_closes,
        proxy_closes     = proxy_closes,
    )

    result.resource_overlay_score      = ov.resource_overlay_score
    result.market_cap_bucket           = ov.market_cap_bucket
    result.liquidity_flag              = ov.liquidity_flag
    result.drawdown_52w_pct            = ov.drawdown_52w_pct
    result.commodity_relative_strength = ov.commodity_relative_strength
    result.commodity_rs_flag           = ov.commodity_rs_flag
    result.short_interest_flag         = ov.short_interest_flag
    result.analyst_revision_flag       = ov.analyst_revision_flag
    result.sentiment_attention_flag    = ov.sentiment_attention_flag
    result.macro_context_flag          = ov.macro_context_flag
    result.existing_source_flags       = ov.existing_source_flags


# ─── Single-ticker pipeline ───────────────────────────────────────────────────

def _yf_short_pct(ticker: str) -> float | None:
    """Blanknings-% via yfinance (US-tickers utan Börsdata-id). 24h-cachad
    i fundamentals-cachen; None när yfinance saknar fältet (vanligt utanför
    US) — då räknas komponenten som omätt, inte som noll."""
    try:
        from contrarian_alpha.cache import get_fundamentals, set_fundamentals
        hit = get_fundamentals(f"yfshort:{ticker}")
        if hit is not None:
            return hit.get("pct")
    except Exception:
        get_fundamentals = set_fundamentals = None
    pct = None
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        raw = info.get("shortPercentOfFloat")
        if raw is not None:
            pct = round(float(raw) * 100.0, 2)   # yfinance ger andel, inte %
    except Exception as e:
        logger.debug("yf short fetch failed for %s: %s", ticker, e)
    try:
        if set_fundamentals is not None:
            set_fundamentals(f"yfshort:{ticker}", {"pct": pct})
    except Exception:
        pass
    return pct


def _batch_valuation_data(
    universe: list[dict],
    fund_snapshots: dict,
    api,
) -> dict[int, dict]:
    """Underlag för värderingsdepressionen — hela universumet batchvis.

    EV/EBITDA-historiken hämtas via Börsdatas batch-endpoint (50 instrument
    per anrop → ~15 anrop för hela Norden, inte ett per aktie). P/E som
    reserv för rader med för tunn EV/EBITDA-historik. Returnerar
    {ins_id: {"current", "history", "metric"}} för rader med >= 8 giltiga
    historikpunkter; övriga utelämnas (komponenten räknas då som omätt).
    """
    if api is None or not getattr(api, "is_configured", False):
        return {}
    ins_ids = [u["ins_id"] for u in universe if u["ins_id"] is not None]
    if not ins_ids:
        return {}

    def _histories(kpi_key: str, ids: list[int]) -> dict[int, list[float]]:
        try:
            raw = api.get_kpi_history_batch(ids, KPI[kpi_key])
        except Exception as e:
            logger.warning("valuation batch (%s) failed: %s", kpi_key, e)
            return {}
        out: dict[int, list[float]] = {}
        for iid, entries in raw.items():
            vals = []
            for entry in entries or []:
                v = (entry or {}).get("v")
                if isinstance(v, (int, float)) and v > 0:
                    vals.append(float(v))
            out[iid] = vals
        return out

    result: dict[int, dict] = {}
    ev_hist = _histories("ev_ebitda", ins_ids)
    thin = []
    for iid in ins_ids:
        valid = ev_hist.get(iid, [])
        if len(valid) >= 8:
            current = (fund_snapshots.get(iid) or {}).get("ev_ebitda")
            if not isinstance(current, (int, float)) or current <= 0:
                current = valid[0]
            result[iid] = {"current": current, "history": valid,
                           "metric": "ev_ebitda"}
        else:
            thin.append(iid)

    if thin:
        pe_hist = _histories("pe", thin)
        for iid in thin:
            valid = pe_hist.get(iid, [])
            if len(valid) >= 8:
                current = (fund_snapshots.get(iid) or {}).get("pe")
                if not isinstance(current, (int, float)) or current <= 0:
                    current = valid[0]
                result[iid] = {"current": current, "history": valid,
                               "metric": "pe"}
    return result


def _run_single_ticker(
    ticker:       str,
    ins_id:       int | None,
    inst_info:    dict,
    fund_snap:    dict,
    price_df,
    branch_name:  str,
    sector_name:  str,
    config:       PipelineConfig,
    api,
    resource_macro_context: dict | None = None,
    short_pct:    float | None = None,
    sector_rel_3m: float | None = None,
    valuation_data: dict | None = None,
) -> ContrairianAlphaResult:
    """
    Run the full 4-stage pipeline for a single instrument.
    All exceptions are caught at the caller — this function may raise.
    """
    market_id  = inst_info.get("marketId", 1)
    name       = inst_info.get("name", ticker)
    market_str = _MARKET_NAME.get(market_id, str(market_id))

    result = ContrairianAlphaResult(
        ticker=ticker,
        ins_id=ins_id,
        name=name,
        market=market_str,
        sector=sector_name,
        branch=branch_name,
        composite_score=0.0,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )

    # Resource-universe context (PR2). Only populated for us_ca_resource; Nordic
    # rows leave these empty so their behavior is bit-for-bit unchanged.
    _is_resource = config.universe == "us_ca_resource"
    if _is_resource:
        result.stage             = (inst_info.get("stage") or "").strip().lower()
        result.primary_commodity = inst_info.get("primary_commodity", "") or ""

    # ── 1. NECESSITY GATE ────────────────────────────────────────────────────

    gics_sector = inst_info.get("sectorId")
    gics_branch = inst_info.get("branchId")
    _sect_name  = branch_name or sector_name

    # Resource universe: the static CSV carries no GICS codes, so necessity would
    # otherwise fall through to a per-ticker yfinance sector call that is slow and
    # unreliable for 60+ US/CA tickers (frequently returns nothing → FALLBACK 40
    # → the whole curated universe is wrongly eliminated at NECESSITY). Seed the
    # sector name from the curated commodity metadata (gold/copper/uranium/…),
    # which maps directly onto SECTOR_NAME_MAP, and skip the network lookup.
    if _is_resource and not _sect_name and result.primary_commodity:
        _sect_name = result.primary_commodity.replace("_", " ")

    # Fallback: no GICS codes and no name → try yfinance sector + industry
    if not _is_resource and not gics_sector and not gics_branch and not _sect_name:
        yf_sector, yf_industry = _yf_sector_name(ticker)
        if not sector_name:
            result.sector = yf_sector
            result.branch = yf_industry
        # Try industry first (more specific); if it returns fallback score, try sector
        from contrarian_alpha.necessity import FALLBACK_SCORE
        if yf_industry:
            _candidate = get_necessity_score(sector_name=yf_industry)
            if _candidate.score != FALLBACK_SCORE.score:
                _sect_name = yf_industry
            else:
                _sect_name = yf_sector   # broader match more likely to hit
        else:
            _sect_name = yf_sector

    necessity_entry = get_necessity_score(
        gics_industry=gics_branch,
        gics_sector=gics_sector,
        sector_name=_sect_name,
    )
    result.necessity_score = float(necessity_entry.score)
    result.necessity_entry = necessity_entry

    if result.necessity_score < config.necessity_threshold:
        if _is_resource:
            # The US/CA resource universe is hand-curated for commodity necessity
            # (Rick Rule / Eric Sprott style), so a below-threshold score here means
            # an unrecognised commodity or missing metadata — NOT a low-necessity
            # business. Never hard-eliminate; surface a transparent flag instead so
            # the row survives onto the watchlist (missing data = flag, not cut).
            _nflag = ("LOW_NECESSITY_UNRECOGNISED"
                      if result.primary_commodity else "NECESSITY_DATA_MISSING")
            for _f in ("NECESSITY_BELOW_THRESHOLD", _nflag):
                if _f not in result.resource_flags:
                    result.resource_flags.append(_f)
                if _f not in result.all_flags:
                    result.all_flags.append(_f)
        else:
            result.eliminated       = True
            result.elimination_stage  = "NECESSITY"
            result.elimination_reason = (
                f"Necessity {result.necessity_score:.0f} < {config.necessity_threshold:.0f} "
                f"({necessity_entry.label})"
            )
            return result

    # ── 2. HATE FILTER ───────────────────────────────────────────────────────

    price_dict = _compute_price_metrics(price_df) if (price_df is not None and not price_df.empty) else {}

    # Populate price snapshot fields for display
    result.close          = price_dict.get("close",          0.0)
    result.sma50          = price_dict.get("sma50",          0.0)
    result.sma200         = price_dict.get("sma200",         0.0)
    result.high_52w       = price_dict.get("high_52w",       0.0)
    result.low_52w        = price_dict.get("low_52w",        0.0)
    result.avg_volume_20d = price_dict.get("avg_volume_20d", None)

    # Optional enrichments for hate
    analyst_dict: dict | None = None
    short_dict:   dict | None = None

    if config.fetch_analyst_data:
        try:
            from contrarian_alpha.cache import cached_analyst_data
            analyst_dict = cached_analyst_data(ticker)
        except Exception as e:
            logger.debug("analyst_data failed for %s: %s", ticker, e)

    if config.fetch_short_data:
        try:
            from contrarian_alpha.cache import cached_short_data
            short_dict = cached_short_data(ticker)
        except Exception as e:
            logger.debug("short_data failed for %s: %s", ticker, e)

    # Blankning utan EODHD: FI-registret (short_pct från batchanropet i
    # run_pipeline) för Börsdata-rader — även 0.0 är RIKTIG data där (under
    # registrets 0,5 %-golv). US-rader utan registervärde: yfinance.
    if short_dict is None:
        if short_pct is not None:
            short_dict = {"short_float_pct": short_pct, "source": "fi_registry"}
        elif ins_id is None:
            yf_pct = _yf_short_pct(ticker)
            if yf_pct is not None:
                short_dict = {"short_float_pct": yf_pct, "source": "yfinance"}

    # Sektorrelativ svaghet räknas ur universumets egna priser i run_pipeline;
    # värderingsdepressionen kommer ur batch-hämtad KPI-historik. Båda skickas
    # in färdiga — inga extra anrop per aktie här.
    sector_dict = ({"stock_vs_sector_3m": sector_rel_3m}
                   if sector_rel_3m is not None else None)

    hate_result = calculate_hate_score(
        price_data     = price_dict,
        analyst_data   = analyst_dict,
        short_data     = short_dict,
        sector_data    = sector_dict,
        valuation_data = valuation_data,
    )
    result.hat_score    = hate_result.score
    result.hate_result  = hate_result

    # Mode-aware hate gate:
    #   deep_contrarian -> hard gate (must be hated, Rule/Sprott trough hunting)
    #   quality         -> soft gate (Buffett/KAP compounders need NOT be hated;
    #                      a proven compounder at fair value is still a Quality buy)
    _hate_floor = config.hate_threshold if config.mode == "deep_contrarian" else QUALITY_HATE_FLOOR
    if result.hat_score < _hate_floor:
        if _is_resource:
            # US/CA resource rows are absent from Börsdata, so the Hat Score is
            # built almost entirely from optional sentiment / analyst / short /
            # sector context that is usually unavailable for these tickers (and
            # price context is often missing too). A sub-threshold Hat here almost
            # always means MISSING data — not a loved/hyped name — so hard-
            # eliminating it empties the whole watchlist (the "67 scanned, 0
            # results, all killed at HATE" bug). Mirror the necessity fix: never
            # hard-eliminate the curated resource universe on a weak hate signal;
            # surface transparent flags and let resource_composite / the overlay
            # rank it. Only when real data actually shows the name is not hated do
            # we flag it as hype/low-neglect (still kept, just deprioritised).
            _hate_missing = (
                result.hat_score <= 0.0
                or hate_result.confidence < 0.5
                or "NO_PRICE_DATA" in hate_result.flags
            )
            _hflag = "HATE_DATA_MISSING" if _hate_missing else "HYPE_OR_LOW_NEGLECT"
            for _f in ("HATE_BELOW_THRESHOLD", _hflag):
                if _f not in result.resource_flags:
                    result.resource_flags.append(_f)
                if _f not in result.all_flags:
                    result.all_flags.append(_f)
        else:
            result.eliminated       = True
            result.elimination_stage  = "HATE"
            result.elimination_reason = (
                f"Hat {result.hat_score:.1f} < {_hate_floor:.0f} "
                f"(not sufficiently hated/neglected) [{config.mode} mode]"
            )
            return result

    # ── 3. BALANCE SHEET GATE ────────────────────────────────────────────────

    # Root-cause fix: the Börsdata batch screener returns 400 for several KPIs
    # (not in licence) and never fetches equity/Altman inputs at all, leaving
    # fund_snap None for equity, ebitda_margin and the Altman components — so the
    # BS gate could only flag BS_DATA_SAKNAS. Back-fill those critical fields
    # from 24h-cached KPI history (survivor-only, guarded to failed KPIs) so the
    # gate + Altman Z can actually evaluate. BS_DATA_SAKNAS below still fires as
    # the last-resort net when even the history is empty.
    _augment_fund_snap_from_history(fund_snap, ins_id, api)

    fund_dict = _build_fundamentals_dict(fund_snap)

    # Store fundamental snapshot on result
    result.fcf_m       = fund_dict.get("fcf")
    result.ebitda_pct  = fund_dict.get("ebitda_margin")
    result.debt_equity = fund_dict.get("debt_to_equity")
    result.equity_m    = fund_dict.get("equity")
    result.ev_ebitda   = fund_dict.get("ev_ebitda")

    strength_result = calculate_strength_score(fund_dict)
    result.strength_score  = strength_result.score
    result.strength_result = strength_result
    result.altman_z        = strength_result.altman_z

    # BS gate uses 4 of the 5 hard gates (Altman Z is scoring-only here)
    gates = strength_result.gate_results
    bs_failures = []

    # Missing-data policy (mirrors ROIC_SAKNAS handling below): a hard gate returns
    # False both when the value violates the threshold AND when the underlying
    # Börsdata field is simply absent. Börsdata batch KPI data is frequently empty
    # or not in licence, so treating "missing" as "failed" rejects the whole
    # universe (bs=0). Reject ONLY when the value EXISTS and violates; when it is
    # missing, flag BS_DATA_SAKNAS and let the row pass.
    _bs_data_missing = False

    # 1. FCF (TTM) > 0 — mirrors strength._check_gates FCF logic
    _fcf = fund_dict.get("fcf")
    _fcf_hist = fund_dict.get("fcf_history") or []
    _fcf_ttm = _fcf_hist[0] if _fcf_hist else _fcf
    if _fcf_ttm is None:
        _bs_data_missing = True
    elif not gates.get("fcf_positive", False):
        bs_failures.append("FCF ≤ 0")

    # 2. EBITDA margin > 0%
    if fund_dict.get("ebitda_margin") is None:
        _bs_data_missing = True
    elif not gates.get("ebitda_margin_positive", False):
        bs_failures.append("EBITDA margin ≤ 0%")

    # Leverage gate: quality mode uses Net Debt/EBITDA ≤ 3.5 (net cash = auto-pass)
    # deep_contrarian keeps the original D/E < 0.6 gate unchanged
    if config.mode == "quality":
        nd_e = fund_snap.get("net_debt_ebitda")
        if nd_e is None:
            # Snapshot lacked Net Debt/EBITDA (KPI 42) — fetch latest history value
            # before deciding (batch KPI sometimes empty / not in licence).
            nd_hist = _fetch_kpi_history(ins_id, KPI["net_debt_ebitda"], api)
            if nd_hist:
                nd_e = nd_hist[0]
        if nd_e is None:
            _bs_data_missing = True
        elif float(nd_e) > 3.5:
            bs_failures.append(f"Net Debt/EBITDA {nd_e:.1f} > 3.5")
        # nd_e <= 3.5 (including negative = net cash) → pass
    else:
        if fund_dict.get("debt_to_equity") is None:
            _bs_data_missing = True
        elif not gates.get("debt_equity_low", False):
            bs_failures.append("D/E ≥ 0.6")

    # 5. Positive equity
    if fund_dict.get("equity") is None:
        _bs_data_missing = True
    elif not gates.get("equity_positive", False):
        bs_failures.append("Equity ≤ 0")

    if _bs_data_missing and "BS_DATA_SAKNAS" not in result.all_flags:
        result.all_flags.append("BS_DATA_SAKNAS")

    # Resource universe: relax mature-company gates in a stage-aware way before
    # deciding elimination (guardrail only — see _apply_resource_stage_guardrails).
    if _is_resource:
        kept, _, gate_mode, res_flags = _apply_resource_stage_guardrails(
            stage=result.stage, fund_dict=fund_dict, gates=gates,
            roic=None, bs_failures=bs_failures,
        )
        result.resource_gate_mode = gate_mode
        for f in res_flags:
            if f not in result.resource_flags:
                result.resource_flags.append(f)
        # Surface guardrail notes in the general flag list (survivors + rejects).
        result.all_flags.extend(
            f for f in result.resource_flags if f not in result.all_flags
        )
        bs_failures = kept

    if bs_failures:
        result.eliminated       = True
        result.elimination_stage  = "BALANCE_SHEET"
        result.elimination_reason = "Failed: " + ", ".join(bs_failures)
        return result

    # Append strength flags (non-gate)
    result.all_flags.extend(strength_result.flags)

    # ── 3.5. QUALITY GATE ────────────────────────────────────────────────────

    quality_data   = _build_quality_data(fund_snap, ins_id, api, mode=config.mode)
    value_data     = _build_value_data(fund_snap, ins_id, api)

    quality_result = calculate_quality_score(
        quality_data, include_growth=(config.mode == "quality")
    )
    value_result   = calculate_value_score(value_data)

    result.quality_score  = quality_result.score
    result.value_score    = value_result.score
    result.quality_result = quality_result
    result.value_result   = value_result
    result.roic           = quality_result.roic
    result.p_fcf          = value_result.p_fcf
    result.all_flags.extend(quality_result.flags)
    result.all_flags.extend([f for f in value_result.flags if f not in result.all_flags])

    # Store KAP fundamental fields (populated only in quality mode via quality_data)
    result.net_debt_ebitda   = fund_snap.get("net_debt_ebitda")
    result.dividend_yield_pct = quality_data.get("dividend_yield_pct")
    result.pe_ratio          = fund_snap.get("pe")
    result.ev_ebit_ratio     = fund_snap.get("ev_ebit")
    result.revenue_cagr_5y   = quality_data.get("revenue_cagr_5y")
    result.revenue_cagr_10y  = quality_data.get("revenue_cagr_10y")
    result.eps_cagr_10y      = quality_data.get("eps_cagr_10y")

    # Valuation sanity bands (quality mode only)
    if config.mode == "quality":
        vb = check_valuation_bands({"pe": result.pe_ratio, "ev_ebit": result.ev_ebit_ratio})
        result.valuation_bands = vb
        result.all_flags.extend([f for f in vb.flags if f not in result.all_flags])

    # Resource guardrail (PR2): for pre-revenue miners ROIC is not applicable, so
    # never let the ROIC gate eliminate them even if a stray value is present.
    # Mature resource rows keep the normal ROIC gate.
    _resource_skip_roic = (
        _is_resource and result.stage in _RESOURCE_PRE_REVENUE_STAGES
    )
    if _resource_skip_roic and "ROIC_NOT_APPLICABLE" not in result.all_flags:
        result.all_flags.append("ROIC_NOT_APPLICABLE")

    # Mode-dependent ROIC gate.
    #   quality         -> HARD gate: ROIC data REQUIRED and must clear 15%.
    #                      Missing ROIC = reject (a proven compounder must prove it).
    #   deep_contrarian -> soft gate: only reject when ROIC data exists and fails 10%
    #                      (cyclical troughs legitimately lack/àdepress ROIC).
    if _resource_skip_roic:
        pass  # pre-revenue miner: ROIC gate intentionally bypassed
    elif config.mode == "quality":
        if quality_result.roic is None:
            # ROIC data missing -> do NOT reject (would empty the list when
            # Borsdata snapshot lacks ROIC). Keep but flag for transparency.
            if "ROIC_SAKNAS" not in result.all_flags:
                result.all_flags.append("ROIC_SAKNAS")
        elif not quality_result.passes_gate_quality:
            result.eliminated        = True
            result.elimination_stage = "QUALITY_GATE"
            result.elimination_reason = (
                f"ROIC {quality_result.roic:.1f}% < 15% gate [quality mode]"
            )
            return result
    else:
        if quality_result.roic is not None and not quality_result.passes_gate_deep:
            result.eliminated        = True
            result.elimination_stage = "QUALITY_GATE"
            result.elimination_reason = (
                f"ROIC {quality_result.roic:.1f}% < 10% gate [deep_contrarian mode]"
            )
            return result

    # ── 3.6. KAP BADGE (quality mode only) ───────────────────────────────────

    if config.mode == "quality":
        rev5  = result.revenue_cagr_5y
        rev10 = result.revenue_cagr_10y
        nd_e  = result.net_debt_ebitda
        div   = result.dividend_yield_pct

        # 1. Growth: revenue CAGR 5y >= 5% AND (10y >= 5% OR DATA_GAP)
        _growth_ok = (
            rev5 is not None and rev5 >= 5.0
            and (rev10 is None or rev10 >= 5.0)
        )
        # 2. Leverage: Net Debt/EBITDA <= 3.5 or net cash (nd_e <= 0) or no data
        _leverage_ok = nd_e is None or float(nd_e) <= 3.5
        # 3. Valuation bands: both metrics within range
        _val_ok = result.valuation_bands is None or result.valuation_bands.passed
        # 4. Dividend: yield >= 1%
        _div_ok = div is not None and float(div) >= 1.0

        result.kap_badge = _growth_ok and _leverage_ok and _val_ok and _div_ok

    # ── 4. COMPOSITE SCORING ─────────────────────────────────────────────────

    # Insider data for catalyst enrichment (fetched per-survivor, cached 1 h)
    insider_dict: dict | None = None
    if ins_id is not None:
        try:
            insider_dict = fetch_insider_data(ins_id, api)
        except Exception as e:
            logger.debug("insider_data failed for %s: %s", ticker, e)

    # Catalyst (also computes Viking Regime)
    catalyst_result = calculate_catalyst_score(
        price_data    = price_dict,
        ticker        = ticker,
        df            = price_df if (price_df is not None and not price_df.empty) else None,
        insider_data  = insider_dict,
    )
    result.catalyst_score  = catalyst_result.score
    result.catalyst_result = catalyst_result
    result.all_flags.extend(catalyst_result.flags)

    viking_raw = 100.0 if catalyst_result.viking_regime_green else 0.0
    result.viking_bonus_raw = viking_raw
    result.viking_bonus_pts = viking_raw * W_VIKING

    # Value Trap check (requires both hat and strength)
    hate_result_enriched = calculate_hate_score(
        price_data     = price_dict,
        analyst_data   = analyst_dict,
        short_data     = short_dict,
        strength_score = result.strength_score,
    )
    result.hate_result = hate_result_enriched
    result.all_flags.extend([f for f in hate_result_enriched.flags if f not in result.all_flags])

    # 5-pillar composite (mode-based weights) + Viking flat bonus
    w = _WEIGHTS.get(config.mode, _WEIGHTS["quality"])
    result.composite_score = round(
        result.necessity_score * w["necessity"]
        + result.hat_score     * w["hate"]
        + result.quality_score * w["quality"]
        + result.value_score   * w["value"]
        + result.catalyst_score * w["catalyst"]
        + viking_raw            * W_VIKING,   # flat +5p when regime green
        2,
    )

    # Data confidence (mean of available sources)
    confs = [
        price_dict.get("confidence", 0.5) if price_dict else 0.0,
        hate_result_enriched.confidence,
        1.0 if fund_snap else 0.0,
    ]
    result.data_confidence = round(sum(confs) / len(confs), 2)

    # ── 4.5. RESOURCE COMPOSITE (PR3; us_ca_resource + stage present only) ────
    # Additive, deterministic resource scoring for survivors. Nordic rows never
    # enter this branch (no stage), so their composite/output is unchanged.
    if _is_resource and result.stage:
        _apply_resource_composite(result, inst_info, fund_snap)
        # PR5/PR6 existing-source overlay (context/watchlist only, separate
        # score). Commodity RS uses the candidate + proxy close series; macro
        # uses the once-per-run FRED reading passed down from run_pipeline.
        _apply_existing_source_overlay(
            result, inst_info, fund_snap, analyst_dict, short_dict,
            price_df=price_df, api=api, macro_context=resource_macro_context,
        )

    return result


# ─── Universe helpers ─────────────────────────────────────────────────────────

def _build_universe(config: PipelineConfig, api) -> list[dict]:
    """
    Return list of instrument dicts to scan.
    Each dict: {ticker, ins_id, inst_info, yf_ticker}
    """
    universe: list[dict] = []
    seen_ids: set[int] = set()

    # ── Static US/CA resource universe (PR1 foundation) ───────────────────────
    # Does NOT touch Börsdata Nordic scoring. yfinance ticker symbols only;
    # ins_id stays None so fundamentals fall back gracefully. Stage/commodity
    # metadata is attached to inst_info for future stage-aware scoring (PR2).
    if config.universe == "us_ca_resource":
        from contrarian_alpha.universe_static import load_resource_universe
        # Country → Börsdata-compatible marketId for _MARKET_NAME display only.
        _country_market = {"US": 11, "CA": 12}
        records = load_resource_universe()   # may raise; caller surfaces the error
        for rec in records:
            yf_ticker = rec.yf_ticker or rec.ticker
            if yf_ticker in {u["ticker"] for u in universe}:
                continue
            inst_info = {
                "name": rec.name,
                "marketId": _country_market.get(rec.country, 11),
                "instrumentType": 1,
                "resource_meta": rec.to_metadata(),
                "stage": rec.stage,
                "primary_commodity": rec.primary_commodity,
                "secondary_commodity": rec.secondary_commodity,
                "exchange": rec.exchange,
                "country": rec.country,
                "yf_ticker": yf_ticker,
            }
            universe.append({
                "ticker":      yf_ticker,
                "ins_id":      None,
                "inst_info":   inst_info,
                "branch_name": "",
                "sector_name": "",
            })
        # Still honour any manual tickers as a supplement.
        for t in config.manual_tickers:
            if t not in {u["ticker"] for u in universe}:
                universe.append({
                    "ticker":      t,
                    "ins_id":      None,
                    "inst_info":   {"name": t, "marketId": 11, "instrumentType": 1},
                    "branch_name": "",
                    "sector_name": "",
                })
        return universe

    if api is not None and api.is_configured:
        try:
            instruments = api.get_instruments()
            # Load branch and sector metadata for name lookup
            branch_meta: dict[int, str] = {}
            sector_meta: dict[int, str] = {}
            try:
                for b in api.get_branches():
                    branch_meta[b["id"]] = b.get("name", "")
            except Exception:
                pass
            try:
                for s in api.get_sectors():
                    sector_meta[s["id"]] = s.get("name", "")
            except Exception:
                pass

            for inst in instruments:
                mid = inst.get("marketId")
                if mid not in config.market_ids:
                    continue
                # Skip non-equity instrument types
                if inst.get("instrumentType", 1) not in (1, None):
                    continue
                ins_id = inst.get("insId")
                if ins_id is None or ins_id in seen_ids:
                    continue
                seen_ids.add(ins_id)
                ticker_raw = inst.get("ticker", "")
                suffix = _MARKET_SUFFIX.get(mid, ".ST")
                yf_ticker = f"{ticker_raw}{suffix}" if ticker_raw else ""

                universe.append({
                    "ticker":      yf_ticker or ticker_raw,
                    "ins_id":      ins_id,
                    "inst_info":   inst,
                    "branch_name": branch_meta.get(inst.get("branchId", -1), ""),
                    "sector_name": sector_meta.get(inst.get("sectorId", -1), ""),
                })
        except Exception as e:
            logger.warning("Failed to build Börsdata universe: %s", e)

    # Add / supplement with manual tickers — försök slå upp mot Börsdata
    # först, så "BOL" blir Boliden med ins_id (fundamenta, rätt yf-suffix,
    # sektormetadata) i stället för en suffixlös yfinance-sträng utan data.
    _bd_by_ticker: dict[str, dict] = {}
    _branch_meta2: dict[int, str] = {}
    _sector_meta2: dict[int, str] = {}
    if api is not None and api.is_configured:
        try:
            for inst in api.get_instruments():
                if inst.get("instrumentType", 1) not in (1, None):
                    continue
                tk = str(inst.get("ticker", "")).upper()
                if tk:
                    _bd_by_ticker.setdefault(tk, inst)
            for b in api.get_branches():
                _branch_meta2[b["id"]] = b.get("name", "")
            for s in api.get_sectors():
                _sector_meta2[s["id"]] = s.get("name", "")
        except Exception as e:
            logger.debug("manual-ticker Börsdata lookup unavailable: %s", e)

    for t in config.manual_tickers:
        if t in {u["ticker"] for u in universe}:
            continue
        # Kandidatformer: som skrivet, utan suffix, bindestreck→mellanslag
        # ("ERIC-B" och "ERIC B.ST" ska båda träffa Börsdatas "ERIC B").
        base = t.split(".")[0]
        forms = [t.upper(), base.upper(), base.replace("-", " ").upper()]
        inst = next((_bd_by_ticker[f] for f in forms if f in _bd_by_ticker), None)
        if inst is not None:
            ins_id = inst.get("insId")
            mid = inst.get("marketId")
            suffix = _MARKET_SUFFIX.get(mid, "")
            # yfinance-form: "ERIC B" + ".ST" → "ERIC-B.ST" (mellanslag → bindestreck)
            raw_tk = str(inst.get("ticker", base)).replace(" ", "-")
            universe.append({
                "ticker":      f"{raw_tk}{suffix}" if suffix else raw_tk,
                "ins_id":      ins_id,
                "inst_info":   inst,
                "branch_name": _branch_meta2.get(inst.get("branchId", -1), ""),
                "sector_name": _sector_meta2.get(inst.get("sectorId", -1), ""),
            })
        else:
            universe.append({
                "ticker":      t,
                "ins_id":      None,
                "inst_info":   {"name": t, "marketId": 0, "instrumentType": 1},
                "branch_name": "",
                "sector_name": "",
            })

    return universe


def _batch_fetch_fundamentals(ins_ids: list[int], api) -> dict[int, dict]:
    """
    Batch-fetch KPI screener snapshots from Börsdata.
    Returns {ins_id: snapshot_dict}.
    Uses fundamentals TTLCache (24 h) keyed per ins_id.
    Falls back to empty dicts on failure.
    """
    if not ins_ids:
        return {}

    try:
        from contrarian_alpha.cache import get_fundamentals, set_fundamentals
        _cache_ok = True
    except Exception:
        _cache_ok = False
        get_fundamentals = set_fundamentals = None

    result: dict[int, dict] = {}
    uncached: list[int] = []

    for iid in ins_ids:
        if _cache_ok:
            hit = get_fundamentals(f"fund:{iid}")
            if hit is not None:
                result[iid] = hit
                continue
        uncached.append(iid)

    if not uncached or api is None or not api.is_configured:
        for iid in uncached:
            result[iid] = {}
        return result

    try:
        fresh = api.get_fundamentals_snapshot_fast(uncached)
        for iid, snap in fresh.items():
            result[iid] = snap
            if _cache_ok and snap:
                try:
                    set_fundamentals(f"fund:{iid}", snap)
                except Exception:
                    pass
        for iid in uncached:
            result.setdefault(iid, {})
    except Exception as e:
        logger.warning("Batch fundamentals fetch failed: %s", e)
        for iid in uncached:
            result[iid] = {}

    return result


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    """
    Run the full Contrarian Alpha Screener pipeline.

    Returns PipelineResult with:
      .results     — top_n passing instruments, ranked by composite_score desc
      .eliminated  — all instruments that were eliminated (with reason)

    Example:
        cfg = PipelineConfig(market_ids=[1, 18, 19], top_n=20)
        result = run_pipeline(cfg)
        for r in result.results:
            print(r.rank, r.ticker, r.composite_score, r.all_flags)
    """
    if config is None:
        config = PipelineConfig()

    t_start = time.monotonic()
    logger.info("Contrarian Alpha pipeline start | markets=%s top_n=%d",
                config.market_ids, config.top_n)

    # ── Initialise Börsdata API ───────────────────────────────────────────────
    api = None
    if _BORSDATA_AVAILABLE:
        try:
            api = BorsdataAPI()
            if not api.is_configured:
                logger.warning("Börsdata API key missing — universe limited to manual tickers")
                api = None
        except Exception as e:
            logger.warning("BorsdataAPI init failed: %s", e)

    # ── Build universe ────────────────────────────────────────────────────────
    universe = _build_universe(config, api)
    universe_count = len(universe)
    logger.info("Universe: %d instruments", universe_count)

    if universe_count == 0:
        logger.error("Empty universe — check Börsdata API key or manual_tickers")
        return PipelineResult(
            results=[], eliminated=[], timestamp=datetime.now(tz=timezone.utc).isoformat(),
            universe_count=0, necessity_passed=0, hate_passed=0,
            bs_passed=0, composite_ranked=0, run_duration_s=0.0, config=config,
        )

    # ── NECESSITY-förgrind på metadata — FÖRE all datahämtning ───────────────
    # Necessity behöver bara sektor/bransch, som redan följde med instrument-
    # listan. Tidigare hämtades priser för HELA universumet (~2000 anrop mot
    # Börsdatas 9 anrop/s-tak = minuter) innan grinden fällde majoriteten.
    # Rader utan metadata (manuella tickers) avgörs i fullflödet som förut
    # (yfinance-sektoruppslag); resurs-universumet grindas aldrig hårt.
    eliminated: list[ContrairianAlphaResult] = []
    scan_list: list[dict] = []
    if config.universe == "us_ca_resource":
        scan_list = list(universe)
    else:
        for u in universe:
            info = u["inst_info"]
            _sect = u["branch_name"] or u["sector_name"]
            if not (info.get("sectorId") or info.get("branchId") or _sect):
                scan_list.append(u)     # metadata saknas — avgör i fullflödet
                continue
            entry = get_necessity_score(
                gics_industry=info.get("branchId"),
                gics_sector=info.get("sectorId"),
                sector_name=_sect,
            )
            if float(entry.score) < config.necessity_threshold:
                eliminated.append(ContrairianAlphaResult(
                    ticker=u["ticker"], ins_id=u["ins_id"],
                    name=info.get("name", u["ticker"]),
                    market=_MARKET_NAME.get(info.get("marketId", 0), "?"),
                    sector=u["sector_name"], branch=u["branch_name"],
                    composite_score=0.0, necessity_score=float(entry.score),
                    necessity_entry=entry, eliminated=True,
                    elimination_stage="NECESSITY",
                    elimination_reason=(
                        f"Necessity {entry.score:.0f} < "
                        f"{config.necessity_threshold:.0f} ({entry.label})"),
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                ))
            else:
                scan_list.append(u)
    logger.info("Necessity pre-gate: %d/%d kvar — data hämtas bara för dem",
                len(scan_list), universe_count)

    # ── Batch-fetch fundamentals — bara för förgrind-överlevarna ─────────────
    ins_ids = [u["ins_id"] for u in scan_list if u["ins_id"] is not None]
    fund_snapshots = _batch_fetch_fundamentals(ins_ids, api)
    for u in scan_list:
        if u["ins_id"] is None:
            fund_snapshots[None] = {}

    # ── Per-ticker: fetch price data in parallel (bara förgrind-överlevare) ───
    try:
        from contrarian_alpha.cache import is_delisted as _is_delisted, delisted_count as _delisted_count
        _universe_tickers = frozenset(u["ticker"] for u in scan_list)
        _pre_run_in_universe = sum(1 for t in _universe_tickers if _is_delisted(t))
        _pre_run_total = _delisted_count()
    except Exception:
        _is_delisted = lambda t: False
        _delisted_count = lambda: 0
        _universe_tickers = frozenset()
        _pre_run_in_universe = 0
        _pre_run_total = 0

    def _fetch_price_task(u: dict):
        return u["ticker"], u["ins_id"], _fetch_price_df(u["ticker"], u["ins_id"], api)

    price_data: dict[str, object] = {}
    _price_done = 0
    with ThreadPoolExecutor(max_workers=config.max_price_workers) as pool:
        futures: dict[Future, dict] = {pool.submit(_fetch_price_task, u): u for u in scan_list}
        for fut in as_completed(futures):
            try:
                ticker, ins_id, df = fut.result()
                price_data[ticker] = df
            except Exception as e:
                u = futures[fut]
                logger.debug("Price fetch failed for %s: %s", u["ticker"], e)
                price_data[u["ticker"]] = None
            _price_done += 1
            if config.progress_cb:
                try:
                    config.progress_cb(_price_done, len(scan_list),
                                       f"priser {_price_done}/{len(scan_list)}")
                except Exception:
                    pass

    # ── Blankning (FI-registret) — HELA registret i ett anrop ────────────────
    shorts_map: dict[int, float] = {}
    if api is not None:
        try:
            shorts_map = api.get_short_positions() or {}
            logger.info("FI-registret: %d instrument med blankning >= 0.5%%",
                        len(shorts_map))
        except Exception as e:
            logger.warning("short positions fetch failed: %s", e)

    # ── Värderingshistorik — batch (50 instrument/anrop) ─────────────────────
    valuation_map = _batch_valuation_data(scan_list, fund_snapshots, api)

    # ── Sektorrelativ 3m-avkastning ur de redan hämtade priserna ─────────────
    def _ret_3m(df) -> float | None:
        try:
            close = df["Close"].dropna()
            if len(close) < 64:
                return None
            return float(close.iloc[-1] / close.iloc[-64] - 1.0) * 100.0
        except Exception:
            return None

    _returns: dict[str, float] = {}
    _sector_groups: dict[str, list[float]] = {}
    for u in scan_list:
        df = price_data.get(u["ticker"])
        if df is None or getattr(df, "empty", True):
            continue
        r = _ret_3m(df)
        if r is None:
            continue
        _returns[u["ticker"]] = r
        if u["sector_name"]:
            _sector_groups.setdefault(u["sector_name"], []).append(r)
    _sector_median = {s: float(np.median(v))
                      for s, v in _sector_groups.items() if len(v) >= 3}

    delisted_skipped = _pre_run_in_universe + (_delisted_count() - _pre_run_total)

    # ── Resource macro context (PR6): fetch the FRED yield-curve reading ONCE
    # per run (only for the resource universe) via the existing cached helper.
    # Shared by every candidate's existing-source overlay; None → NOT_AVAILABLE.
    resource_macro_context: dict | None = None
    if config.universe == "us_ca_resource":
        resource_macro_context = _fetch_resource_macro_context(api)

    # ── Pipeline: per ticker, parallellt ─────────────────────────────────────
    # max_fund_workers fanns i konfigurationen men användes aldrig — steget
    # körde sekventiellt. Trådpool är samma riskprofil som prishämtningen
    # ovan (delad session + trådsäker rate limiter i borsdata_api).
    passing: list[ContrairianAlphaResult] = []
    necessity_passed = hate_passed = bs_passed = 0

    def _ticker_task(u: dict) -> ContrairianAlphaResult:
        ticker   = u["ticker"]
        ins_id   = u["ins_id"]
        _rel = None
        _r = _returns.get(ticker)
        _med = _sector_median.get(u["sector_name"]) if u["sector_name"] else None
        if _r is not None and _med is not None:
            _rel = round(_r - _med, 2)
        try:
            return _run_single_ticker(
                ticker      = ticker,
                ins_id      = ins_id,
                inst_info   = u["inst_info"],
                fund_snap   = fund_snapshots.get(ins_id) or {},
                price_df    = price_data.get(ticker),
                branch_name = u["branch_name"],
                sector_name = u["sector_name"],
                config      = config,
                api         = api,
                resource_macro_context = resource_macro_context,
                short_pct   = (shorts_map.get(ins_id, 0.0)
                               if ins_id is not None else None),
                sector_rel_3m = _rel,
                valuation_data = valuation_map.get(ins_id)
                                 if ins_id is not None else None,
            )
        except Exception as e:
            logger.warning("Pipeline error for %s: %s", ticker, e)
            return ContrairianAlphaResult(
                ticker=ticker, ins_id=ins_id, name=u["inst_info"].get("name", ticker),
                market=_MARKET_NAME.get(u["inst_info"].get("marketId", 0), "?"),
                sector=u["sector_name"], branch=u["branch_name"],
                composite_score=0.0, eliminated=True,
                elimination_stage="ERROR", elimination_reason=str(e),
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
            )

    _scan_done = 0
    with ThreadPoolExecutor(max_workers=max(1, config.max_fund_workers)) as pool:
        futures = {pool.submit(_ticker_task, u): u for u in scan_list}
        results_by_ticker: dict[str, ContrairianAlphaResult] = {}
        for fut in as_completed(futures):
            u = futures[fut]
            res = fut.result()   # _ticker_task fångar själv sina fel
            results_by_ticker[u["ticker"]] = res
            _scan_done += 1
            if config.progress_cb:
                try:
                    config.progress_cb(
                        _scan_done, len(scan_list),
                        f"analys {_scan_done}/{len(scan_list)} · {u['ticker']}")
                except Exception:
                    pass

    # Stabil ordning (universumordningen) så körningar är jämförbara.
    for u in scan_list:
        res = results_by_ticker.get(u["ticker"])
        if res is None:
            continue
        if res.eliminated:
            eliminated.append(res)
            stage = res.elimination_stage
            if stage == "HATE":
                necessity_passed += 1
            elif stage == "BALANCE_SHEET":
                necessity_passed += 1
                hate_passed += 1
            elif stage == "QUALITY_GATE":
                necessity_passed += 1
                hate_passed += 1
                bs_passed += 1
        else:
            necessity_passed += 1
            hate_passed += 1
            bs_passed += 1
            passing.append(res)

    # ── Rank by composite score ────────────────────────────────────────────────
    passing.sort(key=lambda r: r.composite_score, reverse=True)
    for rank, res in enumerate(passing, start=1):
        res.rank = rank

    top_results = passing[: config.top_n]

    t_end = time.monotonic()
    duration = round(t_end - t_start, 2)

    logger.info(
        "Pipeline done in %.1fs | universe=%d necessity=%d hate=%d bs=%d ranked=%d",
        duration, universe_count, necessity_passed, hate_passed, bs_passed, len(passing),
    )

    return PipelineResult(
        results          = top_results,
        eliminated       = eliminated,
        timestamp        = datetime.now(tz=timezone.utc).isoformat(),
        universe_count   = universe_count,
        necessity_passed = necessity_passed,
        hate_passed      = hate_passed,
        bs_passed        = bs_passed,
        composite_ranked = len(passing),
        run_duration_s   = duration,
        config           = config,
        delisted_count   = delisted_skipped,
    )


# ─── CLI / smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    )

    # Offline smoke test with synthetic data (no API key required)
    from contrarian_alpha.necessity  import NECESSITY_MAP
    from contrarian_alpha.strength   import STRENGTH_COMPOSITE_WEIGHT
    from contrarian_alpha.catalyst   import CATALYST_COMPOSITE_WEIGHT

    print(f"\n{'='*70}")
    print("  CONTRARIAN ALPHA SCREENER — ENGINE SMOKE TEST (5-pillar model)")
    print(f"{'='*70}")
    for mode_name, ww in _WEIGHTS.items():
        wsum = sum(ww.values())
        print(f"  [{mode_name}] N={ww['necessity']} H={ww['hate']} "
              f"Q={ww['quality']} V={ww['value']} C={ww['catalyst']}  "
              f"(sum={wsum:.2f})  + Viking flat {W_VIKING}")
    print(f"  Gates: Necessity>={NECESSITY_THRESHOLD}  Hat>={HAT_THRESHOLD}  "
          f"BS=[FCF>0, D/E<0.6, EBITDA>0, Equity>0]  "
          f"Quality=[ROIC>15% quality / >10% deep_contrarian]")
    print(f"{'─'*70}")

    # Run with manual tickers only (no Börsdata needed)
    cfg = PipelineConfig(
        manual_tickers = ["UUUU", "FCX", "NEM", "XOM", "CVX", "AAPL", "MSFT", "NVDA"],
        market_ids     = [],   # skip Börsdata universe
        top_n          = 10,
        fetch_analyst_data = False,
        fetch_short_data   = False,
    )

    pr = run_pipeline(cfg)

    print(f"\n  Universe: {pr.universe_count} tickers | "
          f"Duration: {pr.run_duration_s:.1f}s")
    print(f"  Pass rates: {pr.pass_rates}")
    print(f"\n  TOP {len(pr.results)} RESULTS:")
    print(f"  {'#':>3}  {'Ticker':<8}  {'Composite':>9}  "
          f"{'N':>5}  {'H':>5}  {'Qual':>5}  {'Val':>5}  {'Cat':>5}  {'Vik':>5}  Flags")
    print(f"  {'─'*3}  {'─'*8}  {'─'*9}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")
    for r in pr.results:
        v_str = "GREEN" if r.viking_bonus_raw > 0 else "    -"
        flag_str = ",".join(r.all_flags[:2]) if r.all_flags else ""
        print(f"  {r.rank:>3}  {r.ticker:<8}  {r.composite_score:>9.2f}  "
              f"{r.necessity_score:>5.0f}  {r.hat_score:>5.1f}  "
              f"{r.quality_score:>5.1f}  {r.value_score:>5.1f}  "
              f"{r.catalyst_score:>5.1f}  {v_str:>5}  {flag_str}")

    if pr.eliminated:
        print(f"\n  ELIMINATED ({len(pr.eliminated)}):")
        for r in sorted(pr.eliminated, key=lambda x: x.elimination_stage):
            print(f"    [{r.elimination_stage:<14}]  {r.ticker:<8}  {r.elimination_reason}")

    print(f"\n{'═'*70}\n")
