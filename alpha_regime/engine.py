"""
alpha_regime/engine.py
Orchestrate data fetching and signal evaluation for both regime modes.
Returns a RegimeResult dataclass consumed by alpha_regime/ui.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Optional imports (graceful degradation) ──────────────────────────────────
try:
    from market_cycle.engine import detect_market_cycle
    _MARKET_CYCLE_OK = True
except ImportError:
    _MARKET_CYCLE_OK = False
    logger.warning("market_cycle not available")

try:
    from long_trend.long_trend_loader import fetch_long_history
    _LONG_TREND_OK = True
except ImportError:
    _LONG_TREND_OK = False
    logger.warning("long_trend not available")

_BORSDATA_OK = False
_get_api = None
try:
    from borsdata_api import get_api as _get_api
    _BORSDATA_OK = True
except ImportError:
    try:
        from dashboard.borsdata_api import get_api as _get_api
        _BORSDATA_OK = True
    except ImportError:
        pass

from alpha_regime.indicators import compute_market_indicators
from alpha_regime.quality_signals import (
    SignalResult,
    eval_trend,
    eval_discount,
    eval_cycle,
    eval_quality,
    score_quality_signals,
)
from alpha_regime.contrarian_signals import (
    get_contrarian_stage,
    ContrairianStageResult,
)

# ── Commodity ratios (optional) ───────────────────────────────────────────────
_COMMODITY_OK = False
try:
    from alpha_regime.commodity_ratios import (
        fetch_all_ratios    as _fetch_ratios,
        detect_exposure     as _detect_exposure,
        EXPOSURE_TO_RATIO   as _EXPOSURE_TO_RATIO,
        fetch_context_gauges as _fetch_context_gauges,
    )
    _COMMODITY_OK = True
except ImportError:
    pass


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class RegimeResult:
    mode: str
    ticker: str
    market_ticker: str

    # Shared
    market_phase: str = "UNKNOWN"
    market_confidence: float = 0.0
    price: float = 0.0
    ema50: float = 0.0
    ema200: float = 0.0
    price_vs_ma200: float = 0.0
    trend_phase: str = "Neutral"
    sentiment_score: Optional[float] = None
    market_indicators: dict = field(default_factory=dict)

    # Quality mode
    signals: list = field(default_factory=list)
    signals_passed: int = 0
    quality_verdict: str = "WAIT"

    # Contrarian mode
    contrarian: Optional[ContrairianStageResult] = None

    # Contrarian: commodity ratios + price context
    commodity_ratios: dict = field(default_factory=dict)
    detected_exposure: Optional[list] = None  # list of ratio keys, e.g. ["metal_miners", "gdxj_gdx"]
    branch_name: Optional[str] = None         # Börsdata branch/sector name
    price_3m_low: Optional[float] = None      # ticker's 3-month low (for next-trigger box)
    price_6m_low: Optional[float] = None      # ticker's 3–6 month low

    # CONTEXT ONLY — never feeds the ACCUMULATE/DISTRIBUTE confirmation count.
    # Currency (SEK/NOK) trends lack mean reversion and must never act as contrarian buy signals.
    context_gauges: dict = field(default_factory=dict)

    # Metadata
    pe: Optional[float] = None
    ev_ebit: Optional[float] = None
    quality_score: Optional[float] = None
    kap_badge: bool = False

    error: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> float:
    s = series.dropna()
    if len(s) < span:
        return float("nan")
    return float(s.ewm(span=span, adjust=False).mean().iloc[-1])


def _fetch_price(ticker: str, period: str = "2y") -> pd.DataFrame:
    if _LONG_TREND_OK:
        try:
            df = fetch_long_history(ticker, period=period)
            if not df.empty:
                return df
        except Exception as exc:
            logger.debug("fetch_long_history(%s): %s", ticker, exc)
    # Direct yfinance fallback
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, auto_adjust=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.dropna(how="all")
    except Exception as exc:
        logger.warning("yfinance fallback for %s: %s", ticker, exc)
        return pd.DataFrame()


def _kpi_latest_and_history(api, ins_id: int, kpi_id: int, max_points: int = 6):
    """Return (latest_value, newest_first_history) from annual KPI history."""
    try:
        rows = api.get_kpi_history(ins_id, kpi_id, "year", "mean")
    except Exception:
        return None, []
    vals: list[float] = []
    for r in sorted(rows, key=lambda x: (x.get("y") or 0), reverse=True):
        v = r.get("v")
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if len(vals) >= max_points:
            break
    return (vals[0] if vals else None), vals


# Börsdata KPI IDs (see borsdata_api.KPI for the full table)
_KPI_PE, _KPI_EV_EBIT = 2, 10
_KPI_ROIC, _KPI_ROCE = 37, 36
_KPI_GROSS_MARGIN, _KPI_OP_MARGIN = 28, 29
_KPI_DIV_YIELD = 1


def _fetch_fundamentals(ticker: str) -> dict:
    """
    Resolve ticker → Börsdata instrument ID, then fetch:
      pe, ev_ebit          — latest annual values
      quality_data         — full input dict for contrarian_alpha.quality
    Returns {} when Börsdata is unavailable or the ticker can't be resolved.
    """
    if not _BORSDATA_OK or _get_api is None:
        return {}
    try:
        api = _get_api()
        if not api.is_configured:
            return {}

        ins_id = api.resolve_instrument_id(ticker)
        if ins_id is None:
            logger.warning("Börsdata: could not resolve ticker %s", ticker)
            return {}

        result: dict = {"ins_id": ins_id}

        pe, _ = _kpi_latest_and_history(api, ins_id, _KPI_PE)
        ev_ebit, _ = _kpi_latest_and_history(api, ins_id, _KPI_EV_EBIT)
        if pe is not None and pe > 0:
            result["pe"] = pe
        if ev_ebit is not None and ev_ebit > 0:
            result["ev_ebit"] = ev_ebit

        # Quality pillar inputs (values are in percent units, as quality.py expects)
        roic, roic_hist = _kpi_latest_and_history(api, ins_id, _KPI_ROIC)
        roce, _ = _kpi_latest_and_history(api, ins_id, _KPI_ROCE)
        gm, gm_hist = _kpi_latest_and_history(api, ins_id, _KPI_GROSS_MARGIN)
        om, om_hist = _kpi_latest_and_history(api, ins_id, _KPI_OP_MARGIN)
        div_yield, _ = _kpi_latest_and_history(api, ins_id, _KPI_DIV_YIELD)

        quality_data: dict = {}
        if roic is not None:
            quality_data["roic"] = roic
            quality_data["roic_history"] = roic_hist[:4]
        if roce is not None:
            quality_data["roce"] = roce
        if gm is not None:
            quality_data["gross_margin"] = gm
            quality_data["gross_margin_history"] = gm_hist[1:6]
        if om is not None:
            quality_data["operating_margin"] = om
            quality_data["op_margin_history"] = om_hist[1:6]
        if div_yield is not None:
            quality_data["dividend_yield_pct"] = div_yield

        # Growth consistency (KAP) — CAGR fractions → percent
        try:
            growth = api.get_growth_history(ins_id)
            for src, dst in (
                ("revenue_cagr_5y", "revenue_cagr_5y"),
                ("revenue_cagr_10y", "revenue_cagr_10y"),
                ("earnings_cagr_10y", "eps_cagr_10y"),
            ):
                v = growth.get(src)
                if v is not None:
                    quality_data[dst] = round(float(v) * 100, 2)
        except Exception as exc:
            logger.debug("get_growth_history(%s): %s", ticker, exc)

        if quality_data:
            result["quality_data"] = quality_data

        # Branch/sector name for commodity exposure auto-detection
        try:
            instruments = api.get_instruments()
            for inst in instruments:
                if inst.get("insId") == ins_id:
                    bn = inst.get("branchName") or inst.get("sectorName") or ""
                    if bn:
                        result["branch_name"] = str(bn)
                    break
        except Exception:
            pass

        return result
    except Exception as exc:
        logger.warning("_fetch_fundamentals(%s) failed: %s", ticker, exc)
        return {}


def _fetch_sentiment(ticker: str) -> Optional[float]:
    """Return composite sentiment score 0-100, or None on failure."""
    try:
        import concurrent.futures
        from retail_sentiment.engine import _fetch_all_sources, build_ticker_report
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_fetch_all_sources, [ticker])
            sources = fut.result(timeout=10)
        report = build_ticker_report(ticker, sources)
        raw = report.scores.get("composite", None)
        if raw is None:
            return None
        # Normalise: assume composite is 0-100; if it's -1 to 1, rescale
        if isinstance(raw, float) and abs(raw) <= 1.0:
            return round((raw + 1) / 2 * 100, 1)
        return round(float(raw), 1)
    except Exception as exc:
        logger.debug("_fetch_sentiment(%s): %s", ticker, exc)
        return None


def _classify_trend_phase(price: float, ema50: float, ema200: float) -> str:
    if any(v != v for v in (price, ema50, ema200)):
        return "Neutral"
    if price > ema200 and ema50 > ema200:
        return "Bullish"
    if price < ema200 and ema50 < ema200:
        return "Bearish"
    return "Neutral"


# ── Main orchestration ────────────────────────────────────────────────────────

def run_regime_analysis(
    ticker: str,
    mode: str = "quality",
    market_ticker: str = "SPY",
    quality_score: Optional[float] = None,
    kap_badge: bool = False,
) -> RegimeResult:
    """
    Full regime analysis for a single ticker.

    Parameters
    ----------
    ticker         : stock ticker (e.g. "ATCO-A.ST")
    mode           : "quality" or "contrarian"
    market_ticker  : market benchmark for cycle detection (e.g. "SPY", "^OMX")
    quality_score  : pre-computed quality score (0-100); if None, skipped
    kap_badge      : whether ticker holds KAP badge

    Returns
    -------
    RegimeResult
    """
    result = RegimeResult(mode=mode, ticker=ticker, market_ticker=market_ticker)

    # 1. Fetch ticker price history ──────────────────────────────────────────
    ticker_df = _fetch_price(ticker, period="2y")
    if ticker_df.empty:
        result.error = f"No price data for {ticker}"
        return result

    close = ticker_df["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()

    result.price = float(close.iloc[-1])
    result.ema50 = _ema(close, 50)
    result.ema200 = _ema(close, 200)
    result.trend_phase = _classify_trend_phase(result.price, result.ema50, result.ema200)
    if result.ema200 != 0 and result.ema200 == result.ema200:
        result.price_vs_ma200 = round((result.price / result.ema200 - 1) * 100, 2)

    # 2. Fetch market price + compute market cycle ────────────────────────────
    market_df = _fetch_price(market_ticker, period="2y")
    if not market_df.empty:
        indicators = compute_market_indicators(market_df)
        result.market_indicators = indicators
        if _MARKET_CYCLE_OK and indicators:
            try:
                cycle_res = detect_market_cycle(indicators)
                result.market_phase = cycle_res.get("phase", "UNKNOWN")
                result.market_confidence = cycle_res.get("confidence", 0.0)
            except Exception as exc:
                logger.warning("detect_market_cycle failed: %s", exc)
    else:
        logger.warning("No market data for %s", market_ticker)

    # 3. Fetch fundamentals (P/E, EV/EBIT) ───────────────────────────────────
    fund = _fetch_fundamentals(ticker)
    result.pe = fund.get("pe")
    result.ev_ebit = fund.get("ev_ebit")

    # Compute quality score from Börsdata data when not supplied by caller
    if quality_score is None and fund.get("quality_data"):
        try:
            from contrarian_alpha.quality import calculate_quality_score
            qres = calculate_quality_score(
                fund["quality_data"],
                include_growth=(mode == "quality"),
            )
            if "NO_QUALITY_DATA" not in qres.flags:
                quality_score = qres.score
        except Exception as exc:
            logger.debug("quality score compute failed for %s: %s", ticker, exc)
    result.quality_score = quality_score
    result.kap_badge = kap_badge

    # 4. Fetch sentiment (best-effort, non-blocking) ──────────────────────────
    result.sentiment_score = _fetch_sentiment(ticker)

    # 5. Evaluate signals ─────────────────────────────────────────────────────
    if mode == "quality":
        sig_trend = eval_trend(result.price, result.ema50, result.ema200)
        sig_discount = eval_discount(result.pe, result.ev_ebit)
        sig_cycle = eval_cycle(result.market_phase, result.market_confidence)
        sig_quality = eval_quality(result.quality_score, result.kap_badge)

        result.signals = [sig_trend, sig_discount, sig_cycle, sig_quality]
        result.signals_passed, result.quality_verdict = score_quality_signals(result.signals)

    else:  # contrarian
        result.contrarian = get_contrarian_stage(
            phase=result.market_phase,
            price_vs_ma200_pct=result.price_vs_ma200,
            sentiment_score=result.sentiment_score,
            cycle_confidence=result.market_confidence,
        )

        # 3m / 6m price lows for next-trigger box
        try:
            _cv = close.values
            _n63  = min(63,  len(_cv))
            _n126 = min(126, len(_cv))
            result.price_3m_low = float(np.min(_cv[-_n63:]))
            if _n126 > _n63:
                result.price_6m_low = float(np.min(_cv[-_n126:-_n63]))
        except Exception:
            pass

        # Commodity ratio rubber band
        result.branch_name = fund.get("branch_name")
        if _COMMODITY_OK and result.contrarian is not None:
            try:
                result.commodity_ratios = _fetch_ratios()
                result.context_gauges   = _fetch_context_gauges()
                raw_exp = _detect_exposure(result.branch_name)
                # detected_exposure is now a list of ratio keys (or None)
                result.detected_exposure = _EXPOSURE_TO_RATIO.get(raw_exp) if raw_exp else None
                if result.detected_exposure:
                    # Append rationale for the first STRETCHED ratio found (max +1 confirmation)
                    for _rk in result.detected_exposure:
                        _ratio = result.commodity_ratios.get(_rk)
                        # Only append if the cheap asset is genuinely cheap (not just expensive numerator).
                        # UNKNOWN is NOT a valid confirmation: only an explicit cheap-leg driver may confirm,
                        # otherwise an unclassified/missing driver would produce a false contrarian buy claim.
                        if (_ratio and _ratio.status == "RUBBER_BAND_STRETCHED"
                                and getattr(_ratio, "driver", "UNKNOWN")
                                in ("DENOMINATOR_CHEAP", "BOTH")):
                            result.contrarian.rationale.append(
                                f"Rubber Band: {_ratio.label} at {_ratio.percentile:.0f}th percentile"
                                f" — {_ratio.denominator_label} historically stretched cheap"
                            )
                            break  # max +1 ACCUMULATE confirmation
            except Exception as exc:
                logger.debug("commodity_ratios: %s", exc)

    return result
