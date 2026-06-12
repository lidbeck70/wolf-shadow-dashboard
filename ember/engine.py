"""
ember/engine.py
EMBER strategy — scan orchestration and result data model.

run_ember_scan() fetches shared macro context once, then runs each ticker
in a ThreadPoolExecutor (parallel yfinance downloads) and returns a ranked
EmberScanResult.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from ember.config import (
    RISK_PCT, ATR_STOP_MULT,
    EMBER_ETF_UNIVERSE, EMBER_STOCK_UNIVERSE,
    DEFAULT_SECTOR_ETF, EMBER_SECTOR_ETF, TICKER_THEME_MAP, _THEME_LABEL,
)
from ember.gates import (
    GateResult,
    _download_robust, _ema, _atr_series,
    compute_trend_gates, compute_entry_gates, compute_notrade_flags,
)
from ember.scoring import (
    MacroScore, SentimentScore,
    compute_macro_score, compute_sentiment_score, cycle_asymmetry_bonus,
)

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class EmberSetupResult:
    ticker: str
    typ: str          # "ETF" | "Aktie"
    sektor: str       # human-readable theme label

    trend_gates:   list[GateResult] = field(default_factory=list)
    entry_gates:   list[GateResult] = field(default_factory=list)
    notrade_flags: list[GateResult] = field(default_factory=list)

    trend_pass:    bool = False
    entry_pass:    bool = False
    notrade_clear: bool = True
    eligible:      bool = False

    macro:     Optional[MacroScore]     = None
    sentiment: Optional[SentimentScore] = None

    price:  Optional[float] = None
    ema20:  Optional[float] = None
    atr14:  Optional[float] = None
    entry:  Optional[float] = None
    stop:   Optional[float] = None
    t1:     Optional[float] = None
    t2:     Optional[float] = None
    rr:     Optional[float] = None
    shares: Optional[int]   = None

    cykel_label:       str            = "DATA_GAP"
    percentile_10y:    Optional[float] = None
    hat_score:         Optional[float] = None
    necessity:         Optional[int]   = None

    candle_pattern: str = "NONE"

    asymmetry_score: float = 0.0
    setup_quality:   int   = 0
    cycle_bonus:     float = 0.0

    error: Optional[str] = None


@dataclass
class EmberScanResult:
    eligible:    list[EmberSetupResult]
    near_misses: list[EmberSetupResult]
    all_results: list[EmberSetupResult]
    timestamp:   datetime = field(default_factory=datetime.now)


# ── Per-ticker pipeline ───────────────────────────────────────────────────────

def _scan_ticker(
    ticker: str,
    typ: str,
    sektor: str,
    sector_etf: str,
    theme_map: dict,
    ratios_dict: Optional[dict],
    account_size: float,
) -> EmberSetupResult:
    r = EmberSetupResult(ticker=ticker, typ=typ, sektor=sektor)

    # Pull theme board data (cycle position, HAT, necessity)
    theme_key = TICKER_THEME_MAP.get(ticker.upper())
    if theme_key and theme_key in theme_map:
        td = theme_map[theme_key]
        r.cykel_label   = td.get("cykel_label", "DATA_GAP")
        r.percentile_10y = td.get("percentile_10y")
        r.hat_score     = td.get("hat_score")
        r.necessity     = td.get("necessity")

    # Daily data (2y)
    df_daily = _download_robust(ticker, "2y")
    if df_daily.empty or "Close" not in df_daily.columns:
        r.error = f"Daglig kursdata ej tillgänglig för {ticker}"
        return r

    close_d = df_daily["Close"].squeeze()
    if isinstance(close_d, pd.DataFrame):
        close_d = close_d.iloc[:, 0]
    close_d = close_d.dropna()

    if len(close_d) < 60:
        r.error = f"För lite historik ({len(close_d)} dagar)"
        return r

    r.price = round(float(close_d.iloc[-1]), 2)

    # Weekly data (5y) for 50W EMA
    df_weekly = _download_robust(ticker, "5y")
    close_w   = pd.Series(dtype=float)
    if not df_weekly.empty and "Close" in df_weekly.columns:
        cw = df_weekly["Close"].squeeze()
        if isinstance(cw, pd.DataFrame):
            cw = cw.iloc[:, 0]
        close_w = cw.dropna().resample("W").last().dropna()

    # Trend gates
    r.trend_gates = compute_trend_gates(close_d, close_w, sector_etf)
    r.trend_pass  = all(g.passed for g in r.trend_gates if g.is_blocker)

    # Entry gates + levels
    (r.entry_gates, r.ema20, r.atr14,
     r.entry, r.stop, r.rr, r.candle_pattern) = compute_entry_gates(close_d, df_daily)

    # entry_pass: both hard-gate conditions (pullback + RSI) must pass
    r.entry_pass = all(g.passed for g in r.entry_gates if g.is_blocker)
    r.setup_quality = sum(1 for g in r.entry_gates if g.passed)

    # Recompute levels with ATR stop model
    if r.entry is not None and r.atr14 is not None and not np.isnan(r.atr14) and r.atr14 > 0:
        risk_per_unit = ATR_STOP_MULT * r.atr14
        r.stop = round(r.entry - risk_per_unit, 2)
        risk   = r.entry - r.stop
        if risk > 0:
            r.t1     = round(r.entry + 2 * risk, 2)
            r.t2     = round(r.entry + 3 * risk, 2)
            r.rr     = 2.0
            if account_size > 0:
                r.shares = max(1, int(account_size * RISK_PCT / risk))

    # No-trade flags
    r.notrade_flags = compute_notrade_flags(close_d, df_daily, r.percentile_10y)
    r.notrade_clear = not any(f.passed and f.is_blocker for f in r.notrade_flags)

    # Overall eligibility
    r.eligible = r.trend_pass and r.entry_pass and r.notrade_clear

    # Macro scoring (shared context — passed in from caller)
    r.macro = compute_macro_score(ratios_dict, r.cykel_label)

    # Sentiment scoring (per-ticker yfinance call)
    r.sentiment = compute_sentiment_score(ticker)

    # Asymmetry ranking score: RR × macro quality + cycle phase bonus
    macro_pts = r.macro.total if r.macro else 50.0
    r.cycle_bonus    = cycle_asymmetry_bonus(r.cykel_label)
    r.asymmetry_score = round(
        (r.rr or 0.0) * macro_pts / 100.0 + r.cycle_bonus,
        2,
    )

    return r


# ── Scan entry point ──────────────────────────────────────────────────────────

def run_ember_scan(
    tickers: Optional[list[str]] = None,
    account_size: float = 100_000.0,
    max_workers: int = 6,
) -> EmberScanResult:
    """
    Scan tickers (default = full universe) and return ranked EmberScanResult.
    Macro context (ratios, theme board) is fetched once and shared across all workers.
    """
    if tickers is None:
        tickers = list(dict.fromkeys(EMBER_ETF_UNIVERSE + EMBER_STOCK_UNIVERSE))

    # Shared macro context fetched once
    ratios_dict: Optional[dict] = None
    try:
        from alpha_regime.commodity_ratios import fetch_all_ratios
        ratios_dict = fetch_all_ratios()
    except Exception as exc:
        logger.debug("fetch_all_ratios: %s", exc)

    theme_map: dict = {}
    try:
        from blindspot.theme_board import build_theme_board
        for tr in build_theme_board():
            theme_map[tr.key] = {
                "cykel_label":   tr.cykel_label,
                "percentile_10y": tr.percentile_10y,
                "hat_score":      tr.hat_score,
                "necessity":      tr.necessity,
            }
    except Exception as exc:
        logger.debug("build_theme_board: %s", exc)

    def _sector_etf(t: str) -> str:
        key = TICKER_THEME_MAP.get(t.upper())
        return EMBER_SECTOR_ETF.get(key, DEFAULT_SECTOR_ETF) if key else DEFAULT_SECTOR_ETF

    def _typ(t: str) -> str:
        return "ETF" if t.upper() in {x.upper() for x in EMBER_ETF_UNIVERSE} else "Aktie"

    def _sektor(t: str) -> str:
        key = TICKER_THEME_MAP.get(t.upper())
        return _THEME_LABEL.get(key, "Råvara") if key else "Råvara"

    # Parallel per-ticker scans
    all_results: list[EmberSetupResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                _scan_ticker,
                t, _typ(t), _sektor(t), _sector_etf(t),
                theme_map, ratios_dict, account_size,
            ): t
            for t in tickers
        }
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                all_results.append(fut.result())
            except Exception as exc:
                logger.warning("_scan_ticker(%s) raised: %s", t, exc)
                err = EmberSetupResult(ticker=t, typ="?", sektor="?", error=str(exc))
                all_results.append(err)

    # Partition
    eligible    = [r for r in all_results if r.eligible and not r.error]
    near_misses = [r for r in all_results
                   if not r.eligible and not r.error
                   and (r.trend_pass or r.entry_pass)]

    eligible.sort(key=lambda r: r.asymmetry_score, reverse=True)
    near_misses.sort(key=lambda r: r.setup_quality, reverse=True)

    return EmberScanResult(
        eligible=eligible,
        near_misses=near_misses,
        all_results=all_results,
    )
