"""
screener_ovtlyr.py
OVTLYR-style quantitative screener with z-score normalization.

Scoring model (weighted composite):
  Trend      30%  — EMA alignment (10/20/50/200) + MACD signal
  Momentum   25%  — RSI zone + 5/20-day price change
  Volatility 15%  — ATR relative to price + hist vol percentile
  Volume     15%  — Volume vs 20-day SMA ratio
  ADX        15%  — Trend strength (14-period)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ── Memory safety constants ──────────────────────────────────────────
MAX_SCREENER_TICKERS = 150   # Streamlit Cloud memory safety
BATCH_SIZE = 50              # tickers per yf.download() batch

# ── Retail sentiment integration ─────────────────────────────────────
try:
    from retail_sentiment.engine import build_single_report
    _HAS_RETAIL = True
except ImportError:
    _HAS_RETAIL = False

# Try Börsdata first
try:
    from borsdata_api import get_api, BorsdataAPI, _get_nordic_tickers
    _HAS_BORSDATA = True
except ImportError:
    try:
        from dashboard.borsdata_api import get_api, BorsdataAPI, _get_nordic_tickers
        _HAS_BORSDATA = True
    except ImportError:
        _HAS_BORSDATA = False
        _get_nordic_tickers = None

# ── Ticker universes ──────────────────────────────────────────────────
# Import from existing modules (used as fallback)
try:
    from cagr.cagr_loader import NORDIC_TICKERS as _FALLBACK_NORDIC
except ImportError:
    _FALLBACK_NORDIC = {}

try:
    from heatmap.heatmap_streamlit import US_TICKERS, CANADA_TICKERS
except ImportError:
    US_TICKERS = {}
    CANADA_TICKERS = {}


def _build_nordic_universe() -> Dict[str, dict]:
    """Build Nordic universe from Börsdata API with fallback to hardcoded list."""
    try:
        if _get_nordic_tickers is not None:
            dynamic_tickers = _get_nordic_tickers()
            if dynamic_tickers:
                # Build a dict compatible with the existing universe format
                return {t: {"name": t.split(".")[0], "sector": "Unknown", "country": "Nordic"} for t in dynamic_tickers}
    except Exception:
        pass
    return _FALLBACK_NORDIC


# Backwards-compatible alias
NORDIC_TICKERS = _FALLBACK_NORDIC

UNIVERSES = {
    "Nordic": None,  # Resolved dynamically in run_ovtlyr_screener
    "US": US_TICKERS,
    "Canada": CANADA_TICKERS,
}


# ── Indicator computation ─────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    fast = _ema(close, 12)
    slow = _ema(close, 26)
    macd_line = fast - slow
    signal = _ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return (100 - (100 / (1 + rs))).fillna(50)

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=close.index)

    plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, float("nan"))
    minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, float("nan"))

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, float("nan"))
    adx = dx.rolling(period).mean()
    return adx.fillna(0)


# ── Scoring functions ─────────────────────────────────────────────────

def _score_trend(df: pd.DataFrame) -> float:
    """Score 0-100 based on EMA alignment + MACD."""
    close = df["Close"]
    score = 0

    ema10 = _ema(close, 10).iloc[-1]
    ema20 = _ema(close, 20).iloc[-1]
    ema50 = _ema(close, 50).iloc[-1]
    ema200 = _ema(close, 200).iloc[-1] if len(close) >= 200 else _ema(close, len(close)).iloc[-1]
    price = close.iloc[-1]

    # EMA stack alignment (0-60)
    if price > ema10 > ema20 > ema50 > ema200:
        score += 60  # Perfect bullish stack
    elif price > ema50 > ema200:
        score += 40  # Strong bullish
    elif price > ema200:
        score += 20  # Above 200

    # MACD (0-40)
    _, _, macd_hist = _macd(close)
    if len(macd_hist) > 1:
        hist_now = macd_hist.iloc[-1]
        hist_prev = macd_hist.iloc[-2]
        if hist_now > 0 and hist_now > hist_prev:
            score += 40  # Bullish and accelerating
        elif hist_now > 0:
            score += 25  # Bullish but decelerating
        elif hist_now > hist_prev:
            score += 15  # Bearish but improving

    return min(100, max(0, score))

def _score_momentum(df: pd.DataFrame) -> float:
    """Score 0-100 based on RSI + price change."""
    close = df["Close"]
    rsi_val = _rsi(close, 14).iloc[-1]

    score = 0

    # RSI zone (0-50)
    if 50 < rsi_val < 70:
        score += 50  # Sweet spot
    elif 40 < rsi_val <= 50:
        score += 35  # Neutral-bullish
    elif rsi_val >= 70:
        score += 20  # Overbought — risky
    elif 30 < rsi_val <= 40:
        score += 25  # Neutral-bearish
    else:
        score += 10  # Oversold

    # Price change (0-50)
    if len(close) >= 20:
        chg_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100
        chg_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100

        if chg_5d > 0 and chg_20d > 0:
            score += 50  # Both positive
        elif chg_20d > 0:
            score += 30  # Longer trend positive
        elif chg_5d > 0:
            score += 20  # Short-term bounce

    return min(100, max(0, score))

def _score_volatility(df: pd.DataFrame) -> float:
    """Score 0-100: lower volatility = higher score (safer for entries)."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # ATR relative to price
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1)),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]
    atr_pct = atr14 / close.iloc[-1] * 100 if close.iloc[-1] > 0 else 5

    # Lower ATR% = higher score
    if atr_pct < 1.5:
        score = 90
    elif atr_pct < 2.5:
        score = 70
    elif atr_pct < 3.5:
        score = 50
    elif atr_pct < 5:
        score = 30
    else:
        score = 10

    return score

def _score_volume(df: pd.DataFrame) -> float:
    """Score 0-100 based on volume vs 20-day average."""
    if "Volume" not in df.columns:
        return 50
    vol = df["Volume"]
    vol_sma = vol.rolling(20).mean()
    if vol_sma.iloc[-1] == 0:
        return 50
    ratio = vol.iloc[-1] / vol_sma.iloc[-1]

    # Higher ratio on up days = bullish volume
    is_up = df["Close"].iloc[-1] > df["Open"].iloc[-1]

    if ratio > 1.5 and is_up:
        return 95  # Strong volume on up day
    elif ratio > 1.2 and is_up:
        return 80
    elif ratio > 1.0 and is_up:
        return 65
    elif ratio > 1.0:
        return 45  # Volume but down day
    elif ratio > 0.7:
        return 35  # Below average
    else:
        return 15  # Very low volume

def _score_adx(df: pd.DataFrame) -> float:
    """Score 0-100 based on ADX trend strength."""
    adx = _adx(df["High"], df["Low"], df["Close"], 14).iloc[-1]

    if adx > 40:
        return 95  # Very strong trend
    elif adx > 30:
        return 80  # Strong trend
    elif adx > 25:
        return 65  # Moderate trend
    elif adx > 20:
        return 50  # Weak trend
    elif adx > 15:
        return 30  # Very weak
    else:
        return 10  # No trend / consolidation


# ── Per-Ticker Fear & Greed (0-100) ──────────────────────────────────

def _score_fear_greed(df: pd.DataFrame) -> float:
    """Per-ticker Fear & Greed index (0=extreme fear, 100=extreme greed)."""
    try:
        close = df["Close"]
        volume = df["Volume"]
        high = df["High"]
        low = df["Low"]

        score = 0

        # 1. Volume Z-score (20p) — high volume = greed
        vol_sma = volume.rolling(20).mean().iloc[-1]
        vol_std = volume.rolling(20).std().iloc[-1]
        vol_z = (volume.iloc[-1] - vol_sma) / max(1, vol_std)
        vol_z_clamped = max(-3, min(3, vol_z))
        score += (vol_z_clamped + 3) / 6 * 20

        # 2. RSI deviation from 50 (20p) — RSI > 50 = greed
        rsi_val = _rsi(close, 14).iloc[-1]
        score += (rsi_val / 100) * 20

        # 3. Price vs MA20 (20p) — above MA20 = greed
        ma20 = close.rolling(20).mean().iloc[-1]
        price_dev = (close.iloc[-1] - ma20) / ma20
        dev_score = max(0, min(20, (price_dev + 0.05) / 0.10 * 20))
        score += dev_score

        # 4. Volatility shift (20p) — expanding vol = fear, contracting = greed
        atr_fast = ((high - low).rolling(5).mean()).iloc[-1]
        atr_slow = ((high - low).rolling(20).mean()).iloc[-1]
        vol_ratio = atr_fast / max(0.001, atr_slow)
        vol_shift = max(0, min(20, (1.5 - vol_ratio) / 1.0 * 20))
        score += vol_shift

        # 5. Breadth proxy: % of last 20 days that closed up (20p)
        up_days = (close.diff().tail(20) > 0).sum()
        score += (up_days / 20) * 20

        return round(max(0, min(100, score)), 1)
    except Exception as e:
        logger.warning("_score_fear_greed error: %s", e)
        return 50.0


# ── Viking's Nine (0-9) ─────────────────────────────────────────────

def _vikings_nine(df: pd.DataFrame, retail_score: float = 0, fg_score: float = 50) -> Tuple[int, List[str]]:
    """Returns (score 0-9, list of passed factor names)."""
    try:
        close = df["Close"]
        volume = df["Volume"]

        factors = []

        # Trend (2)
        ema10 = _ema(close, 10).iloc[-1]
        ema20 = _ema(close, 20).iloc[-1]
        price = close.iloc[-1]

        if price > ema10 > ema20:
            factors.append("EMA Stack")

        _, _, macd_hist = _macd(close)
        if macd_hist.iloc[-1] > 0:
            factors.append("MACD Bull")

        # Momentum (2)
        rsi_val = _rsi(close, 14).iloc[-1]
        if 45 < rsi_val < 70:
            factors.append("RSI Sweet")

        if len(close) >= 20:
            chg_5d = close.iloc[-1] / close.iloc[-5] - 1
            chg_20d = close.iloc[-1] / close.iloc[-20] - 1
            if chg_5d > 0 and chg_20d > 0:
                factors.append("Price Up")

        # Volume (2)
        vol_sma = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / max(1, vol_sma)
        if vol_ratio > 1.2:
            factors.append("Vol > 1.2x")

        is_up_day = close.iloc[-1] > df["Open"].iloc[-1]
        if vol_ratio > 1.0 and is_up_day:
            factors.append("Vol Up-Day")

        # Volatility (1)
        atr = ((df["High"] - df["Low"]).rolling(14).mean()).iloc[-1]
        atr_pct = atr / price * 100
        if atr_pct < 3.5:
            factors.append("Low ATR")

        # Sentiment (2)
        if retail_score > 60:
            factors.append("Retail Hot")
        if fg_score > 50:
            factors.append("F&G Greed")

        return len(factors), factors
    except Exception as e:
        logger.warning("_vikings_nine error: %s", e)
        return 0, []


# ── Signal Start Date + Return ───────────────────────────────────────

def _signal_tracking(ticker: str, composite_score: float, close_price: float) -> dict:
    """Track signal start and compute return since signal."""
    try:
        key = f"signal_start_{ticker}"

        if composite_score >= 60:
            if key not in st.session_state:
                st.session_state[key] = {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "price": close_price,
                }

            start = st.session_state[key]
            signal_return = (close_price / start["price"] - 1) * 100
            return {
                "signal_date": start["date"],
                "signal_return": round(signal_return, 1),
            }
        else:
            # Score dropped below threshold — reset
            if key in st.session_state:
                del st.session_state[key]
            return {"signal_date": "-", "signal_return": None}
    except Exception as e:
        logger.warning("_signal_tracking error: %s", e)
        return {"signal_date": "-", "signal_return": None}


# ── OC Model (Overhead Clusters) ────────────────────────────────────

def _overhead_clusters(df: pd.DataFrame, lookback: int = 60, num_bins: int = 20) -> dict:
    """
    Build volume profile and detect overhead supply.
    Returns OC status (M1=under resistance, M2=broken through) and key levels.
    """
    try:
        recent = df.tail(lookback)
        close = recent["Close"]
        volume = recent["Volume"]
        price_now = close.iloc[-1]

        # Build volume profile: bin prices and sum volume per bin
        price_min = close.min()
        price_max = close.max()
        bins = np.linspace(price_min, price_max, num_bins + 1)

        vol_profile = []
        for i in range(len(bins) - 1):
            mask = (close >= bins[i]) & (close < bins[i + 1])
            bin_vol = volume[mask].sum()
            bin_center = (bins[i] + bins[i + 1]) / 2
            vol_profile.append({"price": bin_center, "volume": bin_vol})

        # Find High Volume Nodes (HVN) above current price
        total_vol = sum(v["volume"] for v in vol_profile)
        overhead_nodes = [
            v for v in vol_profile
            if v["price"] > price_now and v["volume"] > total_vol / num_bins * 1.5
        ]

        # OC Status
        if not overhead_nodes:
            oc_status = "M2"   # No significant overhead supply — breakout mode
            oc_score = 100
        elif overhead_nodes[0]["price"] < price_now * 1.03:
            oc_status = "M1"   # Heavy resistance just above
            oc_score = 30
        else:
            oc_status = "M1+"  # Resistance exists but not immediate
            oc_score = 60

        nearest_resistance = overhead_nodes[0]["price"] if overhead_nodes else None

        return {
            "oc_status": oc_status,
            "oc_score": oc_score,
            "overhead_nodes": len(overhead_nodes),
            "nearest_resistance": nearest_resistance,
        }
    except Exception as e:
        logger.warning("_overhead_clusters error: %s", e)
        return {"oc_status": "-", "oc_score": 50, "overhead_nodes": 0, "nearest_resistance": None}


# ── Retail score helper ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False, max_entries=200)
def _get_retail_score(ticker: str) -> float:
    """Fetch retail sentiment score for a single ticker (cached)."""
    try:
        if not _HAS_RETAIL:
            return 0.0
        report = build_single_report(ticker)
        return float(report.get("composite_score", 0))
    except Exception:
        return 0.0


# ── Composite scoring with z-score normalization ──────────────────────

WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.25,
    "volatility": 0.15,
    "volume": 0.15,
    "adx": 0.15,
}

def score_single_ticker(df: pd.DataFrame) -> dict:
    """Score a single ticker's DataFrame. Returns dict of raw sub-scores."""
    if df is None or df.empty or len(df) < 50:
        return None
    try:
        return {
            "trend": _score_trend(df),
            "momentum": _score_momentum(df),
            "volatility": _score_volatility(df),
            "volume": _score_volume(df),
            "adx": _score_adx(df),
        }
    except Exception as e:
        logger.warning("score_single_ticker error: %s", e)
        return None


def _zscore_normalize(values: pd.Series) -> pd.Series:
    """Z-score normalization: (x - mean) / std, then scale to 0-100."""
    mean = values.mean()
    std = values.std()
    if std == 0:
        return pd.Series(50.0, index=values.index)
    z = (values - mean) / std
    # Map z-scores to 0-100 (z of -2 → 0, z of +2 → 100)
    normalized = (z + 2) / 4 * 100
    return normalized.clip(0, 100)


def _batch_download(tickers: list, period: str = "1y") -> dict:
    """Download ticker data in batches to limit peak memory usage."""
    all_data = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        try:
            raw = yf.download(batch, period=period, auto_adjust=True, threads=True, progress=False)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
                        df = raw.xs(ticker, level=1, axis=1).dropna()
                        if not df.empty and len(df) >= 20:
                            all_data[ticker] = df
                    except (KeyError, ValueError):
                        pass
            elif len(batch) == 1:
                if not raw.empty and len(raw) >= 20:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = [c[0] for c in raw.columns]
                    all_data[batch[0]] = raw
        except Exception:
            pass
    return all_data


@st.cache_data(ttl=1800, show_spinner=False, max_entries=5)
def run_ovtlyr_screener(
    universe: str = "Nordic",
    min_volume: int = 100_000,
    period: str = "1y",
    ticker_list: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Run the full OVTLYR screener on a universe of tickers.

    Args:
        universe: Legacy universe string ("Nordic", "US", "Canada", "All")
                  or "custom" when ticker_list is provided.
        min_volume: Minimum average 20-day volume filter.
        period: yfinance download period (default "1y").
        ticker_list: Explicit list of yfinance tickers. When provided,
                     overrides the universe parameter.

    Returns DataFrame with columns:
      Ticker, Name, Sector, Country,
      Trend, Momentum, Volatility, Volume, ADX,
      Composite (z-score weighted), Signal, Rank
    """
    try:
        if ticker_list is not None and len(ticker_list) > 0:
            # New path: explicit ticker list from ticker_universe
            tickers_meta = {
                t: {"name": t.split(".")[0], "sector": "Unknown", "country": "Intl"}
                for t in ticker_list
            }
        elif universe == "Nordic":
            tickers_meta = _build_nordic_universe()
        else:
            tickers_meta = UNIVERSES.get(universe, _FALLBACK_NORDIC)

        if not tickers_meta:
            return pd.DataFrame()

        tickers = list(tickers_meta.keys())

        # Cap tickers for memory safety
        if len(tickers) > MAX_SCREENER_TICKERS:
            tickers = tickers[:MAX_SCREENER_TICKERS]
    except Exception as e:
        logger.warning("run_ovtlyr_screener universe build error: %s", e)
        return pd.DataFrame()

    # Batched download (memory-safe)
    try:
        all_data = _batch_download(tickers, period)
    except Exception:
        all_data = {}

    results = []
    for ticker in tickers:
        meta = tickers_meta.get(ticker, {})
        try:
            if ticker in all_data:
                df = all_data[ticker]
            else:
                try:
                    tk = yf.Ticker(ticker)
                    df = tk.history(period=period, auto_adjust=True)
                except Exception:
                    continue

            if df is None or df.empty or len(df) < 50:
                continue

            # Check minimum volume
            avg_vol = df["Volume"].tail(20).mean() if "Volume" in df.columns else 0
            if avg_vol < min_volume:
                continue

            scores = score_single_ticker(df)
            if scores is None:
                continue

            close_price = float(df["Close"].iloc[-1])

            # ── New OVTLYR columns ──────────────────────────
            # Retail score
            try:
                retail_score = _get_retail_score(ticker) if _HAS_RETAIL else 0.0
            except Exception:
                retail_score = 0.0

            # Fear & Greed
            try:
                fg_score = _score_fear_greed(df)
            except Exception:
                fg_score = 50.0

            # Viking's Nine
            try:
                v9_count, v9_factors = _vikings_nine(df, retail_score, fg_score)
            except Exception:
                v9_count, v9_factors = 0, []

            # Overhead Clusters
            try:
                oc_result = _overhead_clusters(df)
            except Exception:
                oc_result = {"oc_status": "-", "oc_score": 50, "overhead_nodes": 0, "nearest_resistance": None}

            results.append({
                "Ticker": ticker,
                "Name": meta.get("name", ticker),
                "Sector": meta.get("sector", "Unknown"),
                "Country": meta.get("country", "Unknown"),
                **scores,
                "_close": close_price,
                "_avg_vol": float(avg_vol),
                "F&G": fg_score,
                "V9": f"{v9_count}/9",
                "OC": oc_result["oc_status"],
                "Retail": retail_score if _HAS_RETAIL else "-",
            })
        except Exception as e:
            logger.warning("Screener error for %s: %s", ticker, e)
            continue

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)

    # Z-score normalize each component across the universe
    for col in ["trend", "momentum", "volatility", "volume", "adx"]:
        df_results[f"{col}_z"] = _zscore_normalize(df_results[col])

    # Weighted composite
    df_results["Composite"] = (
        df_results["trend_z"] * WEIGHTS["trend"] +
        df_results["momentum_z"] * WEIGHTS["momentum"] +
        df_results["volatility_z"] * WEIGHTS["volatility"] +
        df_results["volume_z"] * WEIGHTS["volume"] +
        df_results["adx_z"] * WEIGHTS["adx"]
    ).round(1)

    # Rank
    df_results["Rank"] = df_results["Composite"].rank(ascending=False, method="min").astype(int)

    # Signal
    df_results["Signal"] = df_results["Composite"].apply(
        lambda x: "STRONG BUY" if x >= 75 else ("BUY" if x >= 60 else ("HOLD" if x >= 40 else "SELL"))
    )

    # ── Signal tracking (depends on Composite) ─────────────────────
    signal_dates = []
    signal_rets = []
    for _, row in df_results.iterrows():
        try:
            sig = _signal_tracking(row["Ticker"], row["Composite"], row["_close"])
            signal_dates.append(sig["signal_date"])
            signal_rets.append(sig["signal_return"])
        except Exception:
            signal_dates.append("-")
            signal_rets.append(None)
    df_results["Signal Date"] = signal_dates
    df_results["Signal Ret %"] = signal_rets

    # Round display columns
    for col in ["trend", "momentum", "volatility", "volume", "adx", "Composite"]:
        df_results[col] = df_results[col].round(1)

    # Sort by composite descending
    df_results = df_results.sort_values("Composite", ascending=False).reset_index(drop=True)

    # Select display columns
    display_cols = [
        "Rank", "Ticker", "Name", "Sector", "Country",
        "trend", "momentum", "volatility", "volume", "adx",
        "Composite", "Signal",
        "F&G", "V9", "OC", "Signal Date", "Signal Ret %", "Retail",
    ]
    return df_results[display_cols]
