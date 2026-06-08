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

# ─── Composite weights (must sum to 1.0) ─────────────────────────────────────

W_NECESSITY = 0.25
W_HAT       = 0.25
W_STRENGTH  = 0.30
W_CATALYST  = 0.15
W_VIKING    = 0.05   # VikingBonus: 100 if green else 0  →  max 5p contribution

# ─── Gate thresholds ─────────────────────────────────────────────────────────

from contrarian_alpha.necessity import NECESSITY_THRESHOLD, get_necessity_score, NecessityEntry
from contrarian_alpha.hate      import HAT_THRESHOLD, calculate_hate_score, HateResult
from contrarian_alpha.hate      import fetch_analyst_data, fetch_short_data
from contrarian_alpha.strength  import calculate_strength_score, StrengthResult
from contrarian_alpha.catalyst  import (
    calculate_catalyst_score, CatalystResult, compute_regime_color
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
    from borsdata_api import BorsdataAPI, KPI
    _BORSDATA_AVAILABLE = True
except ImportError:
    _BORSDATA_AVAILABLE = False
    logger.warning("borsdata_api not found — universe will be limited to manual tickers")


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Runtime configuration for run_pipeline()."""

    # Universe
    market_ids: list[int] = field(default_factory=lambda: [1, 4, 5, 6, 11, 12, 18, 19])
    # 1=SE Large, 18=SE Mid, 19=SE Small, 4=NO, 5=FI, 6=DK, 11=US, 12=CA
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
    viking_bonus_raw: float = 0.0  # 100 or 0
    viking_bonus_pts: float = 0.0  # contribution = raw * W_VIKING

    # Rich sub-results (for UI drilldown)
    necessity_entry:  NecessityEntry | None    = None
    strength_result:  StrengthResult | None    = None
    hate_result:      HateResult | None        = None
    catalyst_result:  CatalystResult | None    = None

    # Pipeline flags
    eliminated:           bool       = False
    elimination_stage:    str        = ""   # "NECESSITY" | "HATE" | "BALANCE_SHEET"
    elimination_reason:   str        = ""
    all_flags:            list[str]  = field(default_factory=list)

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
    delisted_skipped: int = 0

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
    1:  ".ST",   # Sweden Large Cap
    18: ".ST",   # Sweden Mid Cap
    19: ".ST",   # Sweden Small Cap
    4:  ".OL",   # Norway
    5:  ".HE",   # Finland
    6:  ".CO",   # Denmark
    11: "",      # US (no suffix)
    12: ".TO",   # Canada TSX
}

_MARKET_NAME: dict[int, str] = {
    1: "SE Large", 18: "SE Mid", 19: "SE Small",
    4: "NO", 5: "FI", 6: "DK", 11: "US", 12: "CA",
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
            "sma50_slope":    round(sma50_slope, 6),
            "close_history":  close_history,
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

    # Börsdata path
    if api is not None and ins_id is not None:
        try:
            df = api.get_stockprices_df(ins_id, max_count=300)
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
                    raw = yf.Ticker(ticker).history(period="1y", auto_adjust=True, progress=False)
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

    # Revenue (MSEK)
    rev = snapshot.get("revenue_m")
    if rev is not None:
        fund["revenue"] = float(rev)

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


# ─── Single-ticker pipeline ───────────────────────────────────────────────────

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

    # ── 1. NECESSITY GATE ────────────────────────────────────────────────────

    gics_sector = inst_info.get("sectorId")
    gics_branch = inst_info.get("branchId")
    _sect_name  = branch_name or sector_name

    # Fallback: no GICS codes and no name → try yfinance sector + industry
    if not gics_sector and not gics_branch and not _sect_name:
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

    hate_result = calculate_hate_score(
        price_data    = price_dict,
        analyst_data  = analyst_dict,
        short_data    = short_dict,
    )
    result.hat_score    = hate_result.score
    result.hate_result  = hate_result

    if result.hat_score < config.hate_threshold:
        result.eliminated       = True
        result.elimination_stage  = "HATE"
        result.elimination_reason = (
            f"Hat {result.hat_score:.1f} < {config.hate_threshold:.0f} "
            "(stock not sufficiently hated/neglected)"
        )
        return result

    # ── 3. BALANCE SHEET GATE ────────────────────────────────────────────────

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
    if not gates.get("fcf_positive",          False): bs_failures.append("FCF ≤ 0")
    if not gates.get("ebitda_margin_positive", False): bs_failures.append("EBITDA margin ≤ 0%")
    if not gates.get("debt_equity_low",        False): bs_failures.append("D/E ≥ 0.6")
    if not gates.get("equity_positive",        False): bs_failures.append("Equity ≤ 0")

    if bs_failures:
        result.eliminated       = True
        result.elimination_stage  = "BALANCE_SHEET"
        result.elimination_reason = "Failed: " + ", ".join(bs_failures)
        return result

    # Append strength flags (non-gate)
    result.all_flags.extend(strength_result.flags)

    # ── 4. COMPOSITE SCORING ─────────────────────────────────────────────────

    # Catalyst (also computes Viking Regime)
    catalyst_result = calculate_catalyst_score(
        price_data    = price_dict,
        ticker        = ticker,
        df            = price_df if (price_df is not None and not price_df.empty) else None,
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

    # Composite formula
    result.composite_score = round(
        result.necessity_score * W_NECESSITY
        + result.hat_score     * W_HAT
        + result.strength_score * W_STRENGTH
        + result.catalyst_score * W_CATALYST
        + viking_raw            * W_VIKING,
        2,
    )

    # Data confidence (mean of available sources)
    confs = [
        price_dict.get("confidence", 0.5) if price_dict else 0.0,
        hate_result_enriched.confidence,
        1.0 if fund_snap else 0.0,
    ]
    result.data_confidence = round(sum(confs) / len(confs), 2)

    return result


# ─── Universe helpers ─────────────────────────────────────────────────────────

def _build_universe(config: PipelineConfig, api) -> list[dict]:
    """
    Return list of instrument dicts to scan.
    Each dict: {ticker, ins_id, inst_info, yf_ticker}
    """
    universe: list[dict] = []
    seen_ids: set[int] = set()

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

    # Add / supplement with manual tickers
    for t in config.manual_tickers:
        if t not in {u["ticker"] for u in universe}:
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

    # ── Batch-fetch fundamentals (1 API call per KPI → all instruments) ───────
    ins_ids = [u["ins_id"] for u in universe if u["ins_id"] is not None]
    fund_snapshots = _batch_fetch_fundamentals(ins_ids, api)
    # Instruments without ins_id get empty snapshot
    for u in universe:
        if u["ins_id"] is None:
            fund_snapshots[None] = {}

    # ── Per-ticker: fetch price data in parallel ───────────────────────────────
    try:
        from contrarian_alpha.cache import is_delisted as _is_delisted, delisted_count as _delisted_count
        _universe_tickers = frozenset(u["ticker"] for u in universe)
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
    with ThreadPoolExecutor(max_workers=config.max_price_workers) as pool:
        futures: dict[Future, dict] = {pool.submit(_fetch_price_task, u): u for u in universe}
        for fut in as_completed(futures):
            try:
                ticker, ins_id, df = fut.result()
                price_data[ticker] = df
            except Exception as e:
                u = futures[fut]
                logger.debug("Price fetch failed for %s: %s", u["ticker"], e)
                price_data[u["ticker"]] = None

    delisted_skipped = _pre_run_in_universe + (_delisted_count() - _pre_run_total)

    # ── Pipeline: run per ticker ───────────────────────────────────────────────
    passing:   list[ContrairianAlphaResult] = []
    eliminated: list[ContrairianAlphaResult] = []
    necessity_passed = hate_passed = bs_passed = 0

    for i, u in enumerate(universe):
        ticker   = u["ticker"]
        ins_id   = u["ins_id"]
        fund_snap = fund_snapshots.get(ins_id) or {}
        df        = price_data.get(ticker)

        if config.progress_cb:
            try:
                config.progress_cb(i + 1, universe_count, ticker)
            except Exception:
                pass

        try:
            res = _run_single_ticker(
                ticker      = ticker,
                ins_id      = ins_id,
                inst_info   = u["inst_info"],
                fund_snap   = fund_snap,
                price_df    = df,
                branch_name = u["branch_name"],
                sector_name = u["sector_name"],
                config      = config,
                api         = api,
            )
        except Exception as e:
            logger.warning("Pipeline error for %s: %s", ticker, e)
            # Build minimal elimination record
            res = ContrairianAlphaResult(
                ticker=ticker, ins_id=ins_id, name=u["inst_info"].get("name", ticker),
                market=_MARKET_NAME.get(u["inst_info"].get("marketId", 0), "?"),
                sector=u["sector_name"], branch=u["branch_name"],
                composite_score=0.0, eliminated=True,
                elimination_stage="ERROR", elimination_reason=str(e),
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
            )

        if res.eliminated:
            eliminated.append(res)
            stage = res.elimination_stage
            if stage == "HATE":
                necessity_passed += 1
            elif stage == "BALANCE_SHEET":
                necessity_passed += 1
                hate_passed += 1
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
        delisted_skipped = delisted_skipped,
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

    print(f"\n{'═'*70}")
    print("  CONTRARIAN ALPHA SCREENER — ENGINE SMOKE TEST")
    print(f"{'═'*70}")
    print(f"  Composite weights: N={W_NECESSITY} H={W_HAT} S={W_STRENGTH} "
          f"C={W_CATALYST} V={W_VIKING}  (sum={W_NECESSITY+W_HAT+W_STRENGTH+W_CATALYST+W_VIKING})")
    print(f"  Gates: Necessity>={NECESSITY_THRESHOLD}  Hat>={HAT_THRESHOLD}  "
          f"BS=[FCF>0, D/E<0.6, EBITDA>0, Equity>0]")
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
          f"{'N':>5}  {'H':>5}  {'S':>5}  {'C':>5}  {'V':>5}  Flags")
    print(f"  {'─'*3}  {'─'*8}  {'─'*9}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")
    for r in pr.results:
        v_str = "GREEN" if r.viking_bonus_raw > 0 else "    -"
        flag_str = ",".join(r.all_flags[:2]) if r.all_flags else ""
        print(f"  {r.rank:>3}  {r.ticker:<8}  {r.composite_score:>9.2f}  "
              f"{r.necessity_score:>5.0f}  {r.hat_score:>5.1f}  "
              f"{r.strength_score:>5.1f}  {r.catalyst_score:>5.1f}  "
              f"{v_str:>5}  {flag_str}")

    if pr.eliminated:
        print(f"\n  ELIMINATED ({len(pr.eliminated)}):")
        for r in sorted(pr.eliminated, key=lambda x: x.elimination_stage):
            print(f"    [{r.elimination_stage:<14}]  {r.ticker:<8}  {r.elimination_reason}")

    print(f"\n{'═'*70}\n")
