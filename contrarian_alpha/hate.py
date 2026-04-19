"""
hate.py — Hat Score (0-100) for Contrarian Alpha Screener.

Measures how hated / neglected / sold-off a stock is.
Higher score = more contrarian opportunity. Threshold: HAT_THRESHOLD = 45.

7 components (weights sum to 100):
  1. Price vs SMA200          max 20p  (below SMA200 = institution-abandoned)
  2. 52-week low proximity    max 15p  (near 52w low = max pain)
  3. Retail sentiment silence max  5p  (StockTwits volume drought = forgotten)
  4. Analyst downgrades 90d  max 15p  (recent cuts = Wall St. hate)
  5. Short interest (EODHD)  max 15p  (high float short % = active hate)
  6. Sector rotation outflow max 10p  (sector lagging market = nobody wants it)
  7. StockTwits bear ratio   max 20p  (retail bears dominant = sentiment floor)

Adapted from blindspot/scoring/hat.py + retail_sentiment/sources/twitter.py.

Value Trap flag:
  Hat Score > 85 AND Strength Score < 50 → "POTENTIAL_VALUE_TRAP"
  (hated + financially weak = falling knife, not contrarian gem)

Input dicts (all optional, use None/empty dict when unavailable):

  price_data:
    close          float  Current price
    sma200         float  200-day SMA
    high_52w       float  52-week high
    low_52w        float  52-week low

  sentiment_data:  (from retail_sentiment/sources/twitter.fetch_ticker_sentiment)
    message_count  int    Number of StockTwits messages in feed
    bear_ratio     float  0-1 fraction of tagged-bearish messages
    bull_ratio     float  0-1 fraction of tagged-bullish messages
    watchlist_count int   Symbol watchlist count
    confidence     float  0-1 data quality

  analyst_data:    (from EODHD upgrades-downgrades endpoint or yfinance)
    downgrades_90d int    Downgrade actions in last 90 days
    upgrades_90d   int    Upgrade actions in last 90 days
    consensus      str    'Strong Buy'|'Buy'|'Hold'|'Underperform'|'Sell'

  short_data:      (from EODHD fundamentals SharesStats or shorts endpoint)
    short_float_pct float  % of float sold short  (e.g. 8.5 = 8.5%)
    days_to_cover   float  Short interest / avg daily volume

  sector_data:
    sector_vs_market_3m  float  Sector ETF perf vs SPY over 3 months (pp, e.g. -12.5)
    sector_vs_market_6m  float  Sector ETF perf vs SPY over 6 months (pp)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# ─── Pipeline constants ───────────────────────────────────────────────────────

HAT_THRESHOLD = 45              # Hat Score >= 45 required to proceed
HAT_COMPOSITE_WEIGHT = 0.40     # 40% of Composite Score (placeholder; set in composite.py)

VALUE_TRAP_HAT_MIN      = 85    # Hat score above this...
VALUE_TRAP_STRENGTH_MAX = 50    # ...combined with strength below this → Value Trap

# ─── Component max points (must sum to 100) ───────────────────────────────────

_MAX_SMA200     = 20
_MAX_52W_LOW    = 15
_MAX_RETAIL_SIL =  5
_MAX_ANALYST    = 15
_MAX_SHORT      = 15
_MAX_SECTOR     = 10
_MAX_BEAR_RATIO = 20

# ─── Result model ────────────────────────────────────────────────────────────

@dataclass
class HateResult:
    score: float
    breakdown: dict[str, float]     = field(default_factory=dict)
    flags: list[str]                = field(default_factory=list)
    confidence: float               = 1.0   # 0-1, fraction of components with real data

    @property
    def passes_threshold(self) -> bool:
        return self.score >= HAT_THRESHOLD

    @property
    def is_value_trap(self) -> bool:
        return "POTENTIAL_VALUE_TRAP" in self.flags


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ─── Component scorers ────────────────────────────────────────────────────────

def _score_sma200_gap(price_data: dict) -> tuple[float, bool]:
    """
    How far is the price below SMA200? (max 20p)
    Returns (points, has_real_data).
    0% below → 0p | 10% below → 10p | 20%+ below → 20p
    If price is above SMA200 → 0p (stock is in uptrend, not hated enough).
    """
    close  = price_data.get("close",  0.0)
    sma200 = price_data.get("sma200", 0.0)
    if sma200 <= 0 or close <= 0:
        return 8.0, False  # neutral default — moderate assume some discount
    pct_below = (sma200 - close) / sma200 * 100
    pts = _clamp(pct_below / 20.0 * _MAX_SMA200, 0.0, float(_MAX_SMA200))
    return pts, True


def _score_52w_low_proximity(price_data: dict) -> tuple[float, bool]:
    """
    How close is the price to the 52-week low? (max 15p)
    At 52w low → 15p | At mid-range → 0p | Above mid-range → 0p
    """
    close   = price_data.get("close",   0.0)
    high_52 = price_data.get("high_52w", 0.0)
    low_52  = price_data.get("low_52w",  0.0)
    if high_52 <= low_52 or close <= 0:
        return 5.0, False  # neutral default
    position = (close - low_52) / (high_52 - low_52)   # 0=at low, 1=at high
    pts = _clamp((0.5 - position) / 0.5 * _MAX_52W_LOW, 0.0, float(_MAX_52W_LOW))
    return pts, True


def _score_retail_silence(sentiment_data: dict | None) -> tuple[float, bool]:
    """
    How quiet is retail about this stock on StockTwits? (max 5p)
    Silence = forgotten = hated / irrelevant.
    0 messages → 5p | <5 → 4p | <20 → 2p | >=20 → 0p
    Weighted by confidence.
    """
    if not sentiment_data:
        return 3.0, False  # moderate default: assume some neglect
    count = sentiment_data.get("message_count", 0) or 0
    conf  = sentiment_data.get("confidence", 0.0)
    if conf == 0.0:
        return 3.0, False
    if count == 0:
        raw = 5.0
    elif count < 5:
        raw = 4.0
    elif count < 20:
        raw = 2.0
    elif count < 50:
        raw = 1.0
    else:
        raw = 0.0
    return raw * min(1.0, conf + 0.1), True   # slight confidence boost for any data


def _score_analyst_downgrades(analyst_data: dict | None) -> tuple[float, bool]:
    """
    Analyst downgrade pressure over last 90 days. (max 15p)

    Logic:
    - Net downgrades (downgrades − upgrades) drives the base score
    - Consensus modifier adds/subtracts up to 3p
    - Cap at 15p

    Scoring:
      net=0  → 0p
      net=1  → 5p
      net=2  → 8p
      net=3  → 11p
      net=4+ → 14p  (before consensus modifier)

    Consensus modifier:
      'Strong Buy' → -3p  (contradicts hate signal)
      'Buy'        → -1p
      'Hold'       →  0p
      'Underperform' → +1p
      'Sell'       → +2p
    """
    if not analyst_data:
        return 4.0, False  # moderate default
    downs = analyst_data.get("downgrades_90d", 0) or 0
    ups   = analyst_data.get("upgrades_90d",   0) or 0
    net   = max(0, downs - ups)

    _tiers = {0: 0.0, 1: 5.0, 2: 8.0, 3: 11.0}
    base = _tiers.get(net, 14.0)

    consensus_modifiers = {
        "strong buy":    -3.0,
        "buy":           -1.0,
        "hold":           0.0,
        "underperform":   1.0,
        "sell":           2.0,
    }
    consensus = str(analyst_data.get("consensus", "hold")).lower()
    modifier = consensus_modifiers.get(consensus, 0.0)

    pts = _clamp(base + modifier, 0.0, float(_MAX_ANALYST))
    return pts, True


def _score_short_interest(short_data: dict | None) -> tuple[float, bool]:
    """
    Short interest as % of float (EODHD). (max 15p)
    High short % = active bet against stock = institutionally hated.

    <2%   → 0p  (negligible)
    2-5%  → 4p
    5-10% → 8p
    10-15%→ 11p
    15-20%→ 13p
    >20%  → 15p
    """
    if not short_data:
        return 4.0, False  # moderate default
    pct = short_data.get("short_float_pct")
    if pct is None:
        return 4.0, False
    pct = float(pct)
    if pct < 2:    return 0.0,  True
    if pct < 5:    return 4.0,  True
    if pct < 10:   return 8.0,  True
    if pct < 15:   return 11.0, True
    if pct < 20:   return 13.0, True
    return float(_MAX_SHORT),    True


def _score_sector_outflow(sector_data: dict | None) -> tuple[float, bool]:
    """
    Sector ETF relative performance vs market (3m primary, 6m secondary). (max 10p)
    Underperforming sector = capital rotating away = sentiment headwind.

    3m relative perf (pp vs SPY):
      > 0%   → 0p
      -5% to 0  → 2p
      -10% to -5% → 5p
      -15% to -10% → 8p
      < -15% → 10p

    6m secondary: adds up to 2p extra if also lagging.
    """
    if not sector_data:
        return 3.0, False  # moderate default
    rel_3m = sector_data.get("sector_vs_market_3m")
    if rel_3m is None:
        return 3.0, False
    rel_3m = float(rel_3m)
    if rel_3m > 0:      base = 0.0
    elif rel_3m > -5:   base = 2.0
    elif rel_3m > -10:  base = 5.0
    elif rel_3m > -15:  base = 8.0
    else:               base = 10.0

    # 6m bonus: up to 2p if 6m also weak
    bonus = 0.0
    rel_6m = sector_data.get("sector_vs_market_6m")
    if rel_6m is not None and float(rel_6m) < -10:
        bonus = 2.0
    elif rel_6m is not None and float(rel_6m) < -5:
        bonus = 1.0

    pts = _clamp(base + bonus, 0.0, float(_MAX_SECTOR))
    return pts, True


def _score_stocktwits_bear(sentiment_data: dict | None) -> tuple[float, bool]:
    """
    StockTwits bear ratio. (max 20p)
    High bear ratio = retail capitulation / active bearishness = contrarian signal.

    bear_ratio 0.0  → 0p
    bear_ratio 0.3  → 9p   (30% bear)
    bear_ratio 0.5  → 15p  (majority bear)
    bear_ratio 0.7+ → 20p  (strong bear consensus)

    Scaled linearly 0-0.7+ capped at 20p, weighted by confidence.
    """
    if not sentiment_data:
        return 7.0, False  # moderate default
    conf       = sentiment_data.get("confidence", 0.0)
    bear_ratio = sentiment_data.get("bear_ratio", 0.0) or 0.0
    if conf == 0.0:
        return 7.0, False
    raw = _clamp(bear_ratio / 0.70 * _MAX_BEAR_RATIO, 0.0, float(_MAX_BEAR_RATIO))
    # Scale down by confidence — partial data should be treated with caution
    pts = raw * _clamp(conf, 0.5, 1.0)
    return round(pts, 1), True


# ─── Value Trap flag ─────────────────────────────────────────────────────────

def _check_value_trap(hat_score: float, strength_score: float | None) -> bool:
    """True if Hat is very high but Strength is very low — falling knife, not gem."""
    if strength_score is None:
        return False
    return hat_score > VALUE_TRAP_HAT_MIN and strength_score < VALUE_TRAP_STRENGTH_MAX


# ─── Main scoring function ───────────────────────────────────────────────────

def calculate_hate_score(
    price_data: dict,
    sentiment_data: dict | None = None,
    analyst_data: dict | None = None,
    short_data: dict | None = None,
    sector_data: dict | None = None,
    strength_score: float | None = None,
) -> HateResult:
    """
    Calculate Hat Score (0-100) for a single instrument.

    Args:
        price_data:     Required. close, sma200, high_52w, low_52w.
        sentiment_data: Optional. StockTwits bear_ratio, message_count, confidence.
        analyst_data:   Optional. downgrades_90d, upgrades_90d, consensus.
        short_data:     Optional. short_float_pct from EODHD.
        sector_data:    Optional. sector_vs_market_3m, sector_vs_market_6m.
        strength_score: Optional float (0-100) from strength.py — used for
                        Value Trap detection only.

    Returns:
        HateResult with score, breakdown, flags, confidence.
    """
    if not price_data:
        return HateResult(
            score=0.0,
            flags=["NO_PRICE_DATA"],
            confidence=0.0,
        )

    # Score each component
    sma_pts,      sma_real      = _score_sma200_gap(price_data)
    low_pts,      low_real      = _score_52w_low_proximity(price_data)
    sil_pts,      sil_real      = _score_retail_silence(sentiment_data)
    analyst_pts,  analyst_real  = _score_analyst_downgrades(analyst_data)
    short_pts,    short_real    = _score_short_interest(short_data)
    sector_pts,   sector_real   = _score_sector_outflow(sector_data)
    bear_pts,     bear_real     = _score_stocktwits_bear(sentiment_data)

    total = _clamp(
        sma_pts + low_pts + sil_pts + analyst_pts + short_pts + sector_pts + bear_pts,
        0.0, 100.0,
    )
    total = round(total, 1)

    # Confidence: fraction of components backed by real data
    real_count  = sum([sma_real, low_real, sil_real, analyst_real, short_real, sector_real, bear_real])
    confidence  = round(real_count / 7, 2)

    breakdown = {
        "sma200_gap":        round(sma_pts, 1),
        "low_52w_proximity": round(low_pts, 1),
        "retail_silence":    round(sil_pts, 1),
        "analyst_downgrades":round(analyst_pts, 1),
        "short_interest":    round(short_pts, 1),
        "sector_outflow":    round(sector_pts, 1),
        "stocktwits_bear":   round(bear_pts, 1),
    }

    flags: list[str] = []
    if not sma_real:    flags.append("PRICE_DATA_PARTIAL")
    if not analyst_real: flags.append("ANALYST_DATA_MISSING")
    if not short_real:  flags.append("SHORT_DATA_MISSING")
    if not sector_real: flags.append("SECTOR_DATA_MISSING")

    if _check_value_trap(total, strength_score):
        flags.append("POTENTIAL_VALUE_TRAP")

    return HateResult(
        score=total,
        breakdown=breakdown,
        flags=flags,
        confidence=confidence,
    )


# ─── EODHD data fetchers ─────────────────────────────────────────────────────

def _get_eodhd_key() -> str:
    try:
        import streamlit as st
        key = st.secrets.get("EODHD_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("EODHD_API_KEY", "")


def fetch_analyst_data(ticker: str, api_key: str | None = None) -> dict:
    """
    Fetch analyst upgrades/downgrades from EODHD (last 90 days).
    Falls back to yfinance analyst info if no API key.

    Returns dict with downgrades_90d, upgrades_90d, consensus.
    """
    import requests

    api_key = api_key or _get_eodhd_key()
    cutoff  = datetime.now(tz=timezone.utc) - timedelta(days=90)

    if api_key:
        # Normalise ticker for EODHD
        eodhd_ticker = (
            ticker.replace(".ST", ".STO")
                  .replace(".OL", ".OSL")
                  .replace(".CO", ".CPH")
                  .replace(".HE", ".HEL")
        )
        try:
            url = f"https://eodhd.com/api/upgrades-downgrades/{eodhd_ticker}"
            r = requests.get(url, params={"api_token": api_key, "fmt": "json"}, timeout=15)
            if r.status_code == 200:
                rows = r.json()
                downs = ups = 0
                for row in rows:
                    try:
                        date = datetime.fromisoformat(row.get("date", "2000-01-01")).replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
                    if date < cutoff:
                        continue
                    action = str(row.get("action", "")).lower()
                    if "down" in action:
                        downs += 1
                    elif "up" in action:
                        ups += 1
                return {"downgrades_90d": downs, "upgrades_90d": ups, "consensus": "Hold"}
        except Exception as e:
            logger.debug("EODHD analyst fetch failed for %s: %s", ticker, e)

    # yfinance fallback
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        consensus_map = {
            1: "Strong Buy", 2: "Buy", 3: "Hold", 4: "Underperform", 5: "Sell"
        }
        rec = info.get("recommendationMean")
        consensus = "Hold"
        if rec is not None:
            consensus = consensus_map.get(round(float(rec)), "Hold")
        return {
            "downgrades_90d": 0,
            "upgrades_90d":   0,
            "consensus":      consensus,
        }
    except Exception as e:
        logger.debug("yfinance analyst fetch failed for %s: %s", ticker, e)

    return {}


def fetch_short_data(ticker: str, api_key: str | None = None) -> dict:
    """
    Fetch short interest from EODHD fundamentals (SharesStats).
    Falls back to yfinance shortPercentOfFloat.

    Returns dict with short_float_pct, days_to_cover.
    """
    import requests

    api_key = api_key or _get_eodhd_key()

    if api_key:
        eodhd_ticker = (
            ticker.replace(".ST", ".STO")
                  .replace(".OL", ".OSL")
                  .replace(".CO", ".CPH")
                  .replace(".HE", ".HEL")
        )
        try:
            url = f"https://eodhd.com/api/fundamentals/{eodhd_ticker}"
            r = requests.get(url, params={"api_token": api_key, "fmt": "json"}, timeout=15)
            if r.status_code == 200:
                data = r.json()
                stats = data.get("SharesStats", {}) or {}
                pct_raw = stats.get("PercentFloat") or stats.get("ShortPercentFloat")
                dtc_raw = stats.get("ShortRatio") or stats.get("DaystoCover")
                result: dict = {}
                if pct_raw is not None:
                    try:
                        pct = float(pct_raw)
                        # EODHD returns 0-1 or 0-100 depending on field
                        result["short_float_pct"] = pct * 100 if pct < 1.0 else pct
                    except (ValueError, TypeError):
                        pass
                if dtc_raw is not None:
                    try:
                        result["days_to_cover"] = float(dtc_raw)
                    except (ValueError, TypeError):
                        pass
                if result:
                    return result
        except Exception as e:
            logger.debug("EODHD short fetch failed for %s: %s", ticker, e)

    # yfinance fallback
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        result = {}
        pct = info.get("shortPercentOfFloat")
        if pct is not None:
            try:
                pct = float(pct)
                result["short_float_pct"] = pct * 100 if pct < 1.0 else pct
            except (ValueError, TypeError):
                pass
        dtc = info.get("shortRatio")
        if dtc is not None:
            try:
                result["days_to_cover"] = float(dtc)
            except (ValueError, TypeError):
                pass
        return result
    except Exception as e:
        logger.debug("yfinance short fetch failed for %s: %s", ticker, e)

    return {}


def fetch_sector_data(sector_etf: str, market_etf: str = "SPY") -> dict:
    """
    Compute sector ETF performance relative to market ETF over 3m and 6m.
    sector_etf: e.g. 'XLE' (energy), 'XME' (metals), 'GDX' (gold miners)
    """
    try:
        import yfinance as yf
        import numpy as np

        tickers_data = yf.download(
            [sector_etf, market_etf], period="1y",
            auto_adjust=True, progress=False, threads=True
        )
        closes = tickers_data.get("Close", tickers_data)
        if closes is None or closes.empty:
            return {}

        result: dict = {}
        for label, n_days in (("3m", 63), ("6m", 126)):
            key = f"sector_vs_market_{label}"
            try:
                sec = closes[sector_etf].dropna()
                mkt = closes[market_etf].dropna()
                if len(sec) < n_days or len(mkt) < n_days:
                    continue
                sec_perf = (sec.iloc[-1] / sec.iloc[-n_days] - 1) * 100
                mkt_perf = (mkt.iloc[-1] / mkt.iloc[-n_days] - 1) * 100
                result[key] = round(float(sec_perf - mkt_perf), 2)
            except Exception:
                pass
        return result
    except Exception as e:
        logger.debug("Sector relative perf failed for %s: %s", sector_etf, e)
        return {}


# ─── CLI diagnostics ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "_label": "Uranium miner (hated, strong fundamentals)",
            "_strength": 78.0,
            "price":    {"close": 12.5,  "sma200": 18.0,  "high_52w": 22.0, "low_52w": 10.5},
            "sentiment":{"message_count": 3, "bear_ratio": 0.62, "bull_ratio": 0.15, "confidence": 0.8},
            "analyst":  {"downgrades_90d": 3, "upgrades_90d": 0, "consensus": "Underperform"},
            "short":    {"short_float_pct": 12.5, "days_to_cover": 8.2},
            "sector":   {"sector_vs_market_3m": -14.0, "sector_vs_market_6m": -22.0},
        },
        {
            "_label": "Gold miner (mildly hated)",
            "_strength": 65.0,
            "price":    {"close": 35.0,  "sma200": 38.5,  "high_52w": 48.0, "low_52w": 32.0},
            "sentiment":{"message_count": 18, "bear_ratio": 0.38, "bull_ratio": 0.30, "confidence": 1.0},
            "analyst":  {"downgrades_90d": 1, "upgrades_90d": 1, "consensus": "Hold"},
            "short":    {"short_float_pct": 6.2,  "days_to_cover": 3.5},
            "sector":   {"sector_vs_market_3m": -5.5, "sector_vs_market_6m": -8.0},
        },
        {
            "_label": "Tech darling (loved, fails threshold)",
            "_strength": 72.0,
            "price":    {"close": 185.0, "sma200": 155.0, "high_52w": 195.0, "low_52w": 120.0},
            "sentiment":{"message_count": 210, "bear_ratio": 0.08, "bull_ratio": 0.75, "confidence": 1.0},
            "analyst":  {"downgrades_90d": 0, "upgrades_90d": 4, "consensus": "Strong Buy"},
            "short":    {"short_float_pct": 1.2,  "days_to_cover": 1.1},
            "sector":   {"sector_vs_market_3m": 12.0, "sector_vs_market_6m": 18.0},
        },
        {
            "_label": "Value Trap (hated + weak fundamentals)",
            "_strength": 28.0,
            "price":    {"close": 2.10,  "sma200": 6.50,  "high_52w": 9.80, "low_52w": 1.95},
            "sentiment":{"message_count": 1, "bear_ratio": 0.80, "bull_ratio": 0.05, "confidence": 0.6},
            "analyst":  {"downgrades_90d": 5, "upgrades_90d": 0, "consensus": "Sell"},
            "short":    {"short_float_pct": 28.0, "days_to_cover": 14.0},
            "sector":   {"sector_vs_market_3m": -18.0, "sector_vs_market_6m": -30.0},
        },
        {
            "_label": "Copper miner (no sentiment/short data)",
            "_strength": 61.0,
            "price":    {"close": 8.20, "sma200": 11.0, "high_52w": 14.5, "low_52w": 7.8},
            "sentiment": None,
            "analyst":  {"downgrades_90d": 2, "upgrades_90d": 0, "consensus": "Hold"},
            "short":    None,
            "sector":   {"sector_vs_market_3m": -9.0, "sector_vs_market_6m": -12.0},
        },
    ]

    print(f"\n{'─'*82}")
    print(f"  HAT SCORE  |  Threshold: {HAT_THRESHOLD}  |  Value Trap: Hat>{VALUE_TRAP_HAT_MIN} & Strength<{VALUE_TRAP_STRENGTH_MAX}")
    print(f"{'─'*82}")

    for case in test_cases:
        label    = case["_label"]
        strength = case["_strength"]
        result   = calculate_hate_score(
            price_data    = case["price"],
            sentiment_data= case["sentiment"],
            analyst_data  = case["analyst"],
            short_data    = case["short"],
            sector_data   = case["sector"],
            strength_score= strength,
        )
        gate = "PASS" if result.passes_threshold else "FAIL"
        trap = " *** VALUE TRAP ***" if result.is_value_trap else ""
        print(f"\n  {label}")
        print(f"    Score : {result.score:>5.1f}/100   [{gate}]   Conf: {result.confidence:.0%}   Strength: {strength}{trap}")
        print(f"    Break : {result.breakdown}")
        non_data_flags = [f for f in result.flags if "MISSING" not in f and "PARTIAL" not in f]
        if non_data_flags:
            print(f"    Flags : {non_data_flags}")

    print(f"\n{'─'*82}\n")
