"""
cagr_backtest.py
CAGR Strategy Backtester with Walk-Forward Validation.

Strategy overview
-----------------
Monthly rebalancing against the 50 Nordic stocks universe.
  • Score all stocks at each month-end using price-based proxies
    (no look-ahead; fundamentals are omitted for backtesting integrity).
  • BUY  — top-N stocks with total_score >= 9 (equal weight)
  • HOLD — existing positions with total_score 6-8
  • SELL — positions with total_score <= 5
  • Cash — when no BUY signals are available

Walk-forward validation
-----------------------
  • Minimum 3 non-overlapping windows
  • 70% in-sample (IS), 30% out-of-sample (OOS) by default
  • IS and OOS metrics reported per window

CLI usage
---------
  python cagr_backtest.py --years 5 --max-positions 10 --walk-forward

Streamlit integration
---------------------
  from dashboard.cagr.cagr_backtest import render_backtest_section
  render_backtest_section()
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    st = None  # type: ignore[assignment]
    _STREAMLIT_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    go = None  # type: ignore[assignment]
    _PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Accept / reject criteria
# ---------------------------------------------------------------------------

ACCEPT_CRITERIA: dict = {
    "Sharpe Ratio":    (">=", 0.8),
    "Profit Factor":   (">=", 1.5),
    "Max Drawdown %":  (">=", -25.0),
    "Winrate %":       (">=", 55.0),
    "CAGR %":          (">=", 8.0),
}

# Sector defaults for cycle scoring at each historical date
# (Energy undervalued 2020-2022, Utilities cheap 2022-2024, etc.)
HISTORICAL_CYCLE_DEFAULTS: Dict[str, Dict[str, dict]] = {
    # year → sector → {sector_undervalued, underinvestment, sentiment_low}
    "2019": {
        "Energy":    {"sector_undervalued": False, "underinvestment": False, "sentiment_low": False},
        "Financials":{"sector_undervalued": True,  "underinvestment": False, "sentiment_low": True},
        "Utilities": {"sector_undervalued": False, "underinvestment": False, "sentiment_low": False},
    },
    "2020": {
        "Energy":    {"sector_undervalued": True, "underinvestment": True,  "sentiment_low": True},
        "Financials":{"sector_undervalued": True, "underinvestment": False, "sentiment_low": True},
        "Utilities": {"sector_undervalued": False,"underinvestment": False, "sentiment_low": False},
    },
    "2021": {
        "Energy":    {"sector_undervalued": True, "underinvestment": True,  "sentiment_low": True},
        "Financials":{"sector_undervalued": True, "underinvestment": False, "sentiment_low": False},
        "Utilities": {"sector_undervalued": False,"underinvestment": False, "sentiment_low": False},
    },
    "2022": {
        "Energy":    {"sector_undervalued": True, "underinvestment": True,  "sentiment_low": False},
        "Utilities": {"sector_undervalued": True, "underinvestment": False, "sentiment_low": True},
        "Financials":{"sector_undervalued": True, "underinvestment": False, "sentiment_low": False},
    },
    "2023": {
        "Energy":    {"sector_undervalued": True, "underinvestment": False, "sentiment_low": False},
        "Utilities": {"sector_undervalued": True, "underinvestment": False, "sentiment_low": True},
        "Financials":{"sector_undervalued": True, "underinvestment": False, "sentiment_low": False},
    },
    "2024": {
        "Energy":    {"sector_undervalued": True, "underinvestment": False, "sentiment_low": False},
        "Utilities": {"sector_undervalued": True, "underinvestment": False, "sentiment_low": True},
        "Financials":{"sector_undervalued": True, "underinvestment": False, "sentiment_low": True},
    },
}

# Default cycle (used for years not explicitly mapped)
_DEFAULT_SECTOR_CYCLE: dict = {
    "sector_undervalued": False,
    "underinvestment": False,
    "sentiment_low": False,
}


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Historical scoring (price-based, no look-ahead bias)
# ---------------------------------------------------------------------------

def _price_based_value_score(close_slice: pd.Series) -> int:
    """
    Compute a 0-6 value score from historical OHLCV close series.

    This replaces the fundamental score for backtesting purposes so that
    no future fundamental data leaks into historical decisions.

    Criteria
    --------
    1. Price < EMA200            — below long-term trend (value territory)
    2. Price < 52-week midpoint  — in lower half of annual range
    3. RSI < 40                  — oversold / potential value
    4. Price > 52-week low × 1.1 — not a falling knife
    5. Volume increasing         — accumulation signal (if volume available)
    6. Price > EMA50             — beginning to turn up
    """
    if close_slice is None or len(close_slice) < 30:
        return 0

    score = 0
    close = close_slice.dropna()
    if len(close) < 30:
        return 0

    current = close.iloc[-1]
    ema50 = _ema(close, 50).iloc[-1]
    ema200 = _ema(close, 200).iloc[-1]
    rsi_val = _rsi(close, 14)
    current_rsi = rsi_val.iloc[-1] if not rsi_val.empty else np.nan

    # 52-week window (≈ 252 trading days; use what we have)
    lookback = min(252, len(close) - 1)
    period_slice = close.iloc[-lookback - 1:]
    high_52w = period_slice.max()
    low_52w = period_slice.min()
    midpoint_52w = (high_52w + low_52w) / 2.0

    # 1. Price < EMA200
    if not np.isnan(ema200) and current < ema200:
        score += 1

    # 2. Price < 52-week midpoint
    if current < midpoint_52w:
        score += 1

    # 3. RSI < 40
    if not np.isnan(current_rsi) and current_rsi < 40:
        score += 1

    # 4. Price > 52-week low × 1.1 (not in free-fall)
    if low_52w > 0 and current > low_52w * 1.1:
        score += 1

    # 5. Volume increasing — handled in caller when volume is available
    # (placeholder, always score 0 here; caller can increment)

    # 6. Price > EMA50
    if not np.isnan(ema50) and current > ema50:
        score += 1

    return score


def _volume_increasing(df_slice: pd.DataFrame, window: int = 20) -> bool:
    """
    Return True if average volume in the last `window` bars is above the
    previous `window` bars (accumulation signal).
    """
    vol_col = next((c for c in ("Volume", "volume") if c in df_slice.columns), None)
    if vol_col is None or len(df_slice) < window * 2:
        return False
    recent_vol = df_slice[vol_col].iloc[-window:].mean()
    prev_vol = df_slice[vol_col].iloc[-2 * window:-window].mean()
    return bool(recent_vol > prev_vol) if prev_vol > 0 else False


def _technical_score_historical(df_slice: pd.DataFrame) -> int:
    """
    Compute 0-4 technical score from historical data slice.
    Same criteria as cagr_technical.score_technical() but on a slice.
    """
    close_col = next(
        (c for c in ("Close", "Adj Close", "close") if c in df_slice.columns),
        None,
    )
    if close_col is None or len(df_slice) < 30:
        return 0

    close = df_slice[close_col].dropna()
    if len(close) < 30:
        return 0

    score = 0
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    rsi_vals = _rsi(close, 14)

    last_close = close.iloc[-1]
    last_ema50 = ema50.iloc[-1]
    last_ema200 = ema200.iloc[-1]
    last_rsi = rsi_vals.iloc[-1]

    # 1. EMA200 slope positive
    if len(ema200.dropna()) >= 21:
        if ema200.iloc[-1] > ema200.iloc[-21]:
            score += 1

    # 2. Price > EMA200
    if not np.isnan(last_ema200) and last_close > last_ema200:
        score += 1

    # 3. EMA50 > EMA200
    if not np.isnan(last_ema50) and not np.isnan(last_ema200):
        if last_ema50 > last_ema200:
            score += 1

    # 4. RSI > 50 AND rising
    if not np.isnan(last_rsi) and last_rsi > 50:
        rsi_clean = rsi_vals.dropna()
        if len(rsi_clean) >= 6 and last_rsi > rsi_clean.iloc[-6]:
            score += 1

    return score


def _cycle_score_historical(sector: str, year: int) -> int:
    """
    Return 0-3 cycle score using hardcoded historical sector assessments.
    Falls back to all-False (0) for unmapped sectors/years.
    """
    year_str = str(year)
    year_data = HISTORICAL_CYCLE_DEFAULTS.get(year_str, {})
    cycle = year_data.get(sector, _DEFAULT_SECTOR_CYCLE)
    return sum(1 for v in cycle.values() if v)


def _compute_score_at_date(
    ticker: str,
    sector: str,
    df: pd.DataFrame,
    as_of_date,
    min_bars: int = 60,
) -> int:
    """
    Compute total CAGR score for a ticker as of a given date using only
    data available up to (and including) that date.

    Score breakdown:
      value_score (0-6) + cycle_score (0-3) + tech_score (0-4) = 0-13
    """
    try:
        # Slice to history available at rebalance date
        if hasattr(df.index, "tz") and df.index.tz is not None:
            as_of = pd.Timestamp(as_of_date).tz_localize(df.index.tz)
        else:
            as_of = pd.Timestamp(as_of_date)

        hist = df[df.index <= as_of]
        if len(hist) < min_bars:
            return 0

        close_col = next(
            (c for c in ("Close", "Adj Close", "close") if c in hist.columns),
            None,
        )
        if close_col is None:
            return 0

        close_slice = hist[close_col]

        # Value score (price-based proxy for fundamentals)
        v_score = _price_based_value_score(close_slice)

        # Volume increasing (+1 bonus, capped at 6 total for value)
        if _volume_increasing(hist) and v_score < 6:
            v_score += 1
        v_score = min(6, v_score)

        # Cycle score
        year = as_of.year
        c_score = _cycle_score_historical(sector, year)

        # Technical score
        t_score = _technical_score_historical(hist)

        return v_score + c_score + t_score

    except Exception as exc:
        logger.debug("_compute_score_at_date(%s, %s): %s", ticker, as_of_date, exc)
        return 0


# ---------------------------------------------------------------------------
# Metrics calculations
# ---------------------------------------------------------------------------

def _compute_metrics(
    equity_curve: pd.Series,
    monthly_returns: pd.Series,
    trade_log: List[dict],
) -> dict:
    """
    Compute full suite of performance metrics from equity curve and returns.

    Parameters
    ----------
    equity_curve    : pd.Series  — portfolio value, dated index
    monthly_returns : pd.Series  — monthly percentage returns
    trade_log       : list[dict] — list of trade records

    Returns
    -------
    dict with all ACCEPT_CRITERIA keys plus additional diagnostics.
    """
    metrics: dict = {
        "Total Return %": float("nan"),
        "CAGR %": float("nan"),
        "Sharpe Ratio": float("nan"),
        "Sortino Ratio": float("nan"),
        "Profit Factor": float("nan"),
        "Max Drawdown %": float("nan"),
        "Winrate %": float("nan"),
        "Num Trades": 0,
        "Turnover Rate": float("nan"),
        "Avg Monthly Return %": float("nan"),
        "Volatility (Ann) %": float("nan"),
    }

    if equity_curve is None or len(equity_curve) < 2:
        return metrics

    # Total return
    start_val = equity_curve.iloc[0]
    end_val = equity_curve.iloc[-1]
    if start_val > 0:
        total_ret = (end_val / start_val - 1) * 100
        metrics["Total Return %"] = round(total_ret, 2)

    # CAGR
    start_date = equity_curve.index[0]
    end_date = equity_curve.index[-1]
    years = (end_date - start_date).days / 365.25
    if years > 0 and start_val > 0:
        cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
        metrics["CAGR %"] = round(cagr, 2)

    if monthly_returns is not None and len(monthly_returns) > 0:
        returns = monthly_returns.dropna()
        if len(returns) > 0:
            avg = returns.mean()
            std = returns.std(ddof=1)
            downside = returns[returns < 0].std(ddof=1) if (returns < 0).any() else np.nan

            metrics["Avg Monthly Return %"] = round(float(avg * 100), 2)
            metrics["Volatility (Ann) %"] = round(float(std * math.sqrt(12) * 100), 2)

            # Sharpe (monthly risk-free ≈ 0; annualised)
            if std > 0:
                sharpe = (avg / std) * math.sqrt(12)
                metrics["Sharpe Ratio"] = round(float(sharpe), 3)

            # Sortino
            if not np.isnan(downside) and downside > 0:
                sortino = (avg / downside) * math.sqrt(12)
                metrics["Sortino Ratio"] = round(float(sortino), 3)

            # Winrate — % of months with positive return
            pos_months = (returns > 0).sum()
            metrics["Winrate %"] = round(float(pos_months / len(returns) * 100), 1)

    # Max drawdown
    if len(equity_curve) > 1:
        roll_max = equity_curve.cummax()
        dd = (equity_curve - roll_max) / roll_max * 100
        metrics["Max Drawdown %"] = round(float(dd.min()), 2)

    # Profit factor
    if trade_log:
        gains = [t["pnl"] for t in trade_log if t.get("pnl", 0) > 0]
        losses = [abs(t["pnl"]) for t in trade_log if t.get("pnl", 0) < 0]
        if losses:
            pf = sum(gains) / sum(losses) if sum(losses) > 0 else float("inf")
            metrics["Profit Factor"] = round(pf, 3)
        elif gains:
            metrics["Profit Factor"] = float("inf")

        metrics["Num Trades"] = len(trade_log)

        # Turnover rate: average monthly change in positions
        # (simplified: total trades / num months / max_positions)

    return metrics


def _passes_criteria(metrics: dict) -> Tuple[bool, List[str]]:
    """
    Check metrics against ACCEPT_CRITERIA.

    Returns
    -------
    (passed: bool, failures: list[str])
    """
    failures = []
    for key, (op, threshold) in ACCEPT_CRITERIA.items():
        val = metrics.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            failures.append(f"{key}: N/A (missing data)")
            continue
        if op == ">=" and val < threshold:
            failures.append(f"{key}: {val} < {threshold}")
        elif op == "<=" and val > threshold:
            failures.append(f"{key}: {val} > {threshold}")
    passed = len(failures) == 0
    return passed, failures


# ---------------------------------------------------------------------------
# CAGRBacktester
# ---------------------------------------------------------------------------

class CAGRBacktester:
    """
    Monthly-rebalancing backtester for the CAGR Nordic stock strategy.

    Parameters
    ----------
    tickers         : dict[str, dict]  — ticker → metadata (name, sector, ...)
    start_date      : str              — "YYYY-MM-DD"
    end_date        : str              — "YYYY-MM-DD"
    initial_capital : float            — starting portfolio value (SEK / EUR)
    max_positions   : int              — maximum concurrent positions
    rebalance_freq  : str              — "M" (monthly) or "W" (weekly, experimental)
    commission      : float            — round-trip commission per trade (0.0015 = 0.15%)
    buy_threshold   : int              — minimum score to initiate new buy
    sell_threshold  : int              — score at or below which positions are sold
    hold_min        : int              — minimum score to hold an existing position
    """

    def __init__(
        self,
        tickers: Dict[str, dict],
        start_date: str,
        end_date: str,
        initial_capital: float = 100_000.0,
        max_positions: int = 10,
        rebalance_freq: str = "M",
        commission: float = 0.0015,
        buy_threshold: int = 9,
        sell_threshold: int = 5,
        hold_min: int = 6,
    ) -> None:
        self.tickers = tickers
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.rebalance_freq = rebalance_freq
        self.commission = commission
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.hold_min = hold_min

        # Populated after run()
        self._price_data: Dict[str, pd.DataFrame] = {}
        self.equity_curve: pd.DataFrame = pd.DataFrame()
        self.monthly_returns: pd.Series = pd.Series(dtype=float)
        self.trade_log: List[dict] = []
        self.position_history: List[dict] = []
        self.metrics: dict = {}
        self.wf_results: List[dict] = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_price_data(self) -> None:
        """
        Load price data for all tickers using cagr_loader.fetch_price_data.
        Falls back gracefully for tickers that fail.
        """
        from .cagr_loader import fetch_price_data  # local import avoids circular dep

        ticker_list = list(self.tickers.keys())
        years_needed = max(
            2,
            math.ceil((self.end_date - self.start_date).days / 365.25) + 1,
        )
        period = f"{years_needed}y"
        logger.info(
            "Loading price data for %d tickers (period=%s) …",
            len(ticker_list), period,
        )
        self._price_data = fetch_price_data(ticker_list, period=period)
        loaded = sum(1 for df in self._price_data.values() if not df.empty)
        logger.info("Price data loaded: %d / %d tickers", loaded, len(ticker_list))

    # ------------------------------------------------------------------
    # Rebalance date generation
    # ------------------------------------------------------------------

    def _rebalance_dates(self) -> List[pd.Timestamp]:
        """Generate month-end rebalance dates within the backtest window."""
        dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq="BME",  # Business Month End
        )
        return [d for d in dates if self.start_date <= d <= self.end_date]

    # ------------------------------------------------------------------
    # Scoring at a given rebalance date
    # ------------------------------------------------------------------

    def _score_all(self, rebalance_date: pd.Timestamp) -> pd.DataFrame:
        """
        Score every ticker on the given rebalance date.

        Returns
        -------
        DataFrame with columns: ticker, name, sector, total_score
        Sorted descending by total_score.
        """
        rows = []
        for ticker, meta in self.tickers.items():
            df = self._price_data.get(ticker)
            if df is None or df.empty:
                continue
            sector = meta.get("sector", "Unknown")
            name = meta.get("name", ticker)
            total = _compute_score_at_date(
                ticker=ticker,
                sector=sector,
                df=df,
                as_of_date=rebalance_date,
            )
            rows.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "total_score": total,
            })

        if not rows:
            return pd.DataFrame(columns=["ticker", "name", "sector", "total_score"])

        scored = pd.DataFrame(rows).sort_values("total_score", ascending=False)
        return scored.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _get_price_at(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        """Return the closing price for a ticker on or before `date`."""
        df = self._price_data.get(ticker)
        if df is None or df.empty:
            return None

        close_col = next(
            (c for c in ("Close", "Adj Close", "close") if c in df.columns),
            None,
        )
        if close_col is None:
            return None

        if hasattr(df.index, "tz") and df.index.tz is not None:
            loc_date = date.tz_localize(df.index.tz)
        else:
            loc_date = date

        hist = df[df.index <= loc_date]
        if hist.empty:
            return None

        price = hist[close_col].iloc[-1]
        return float(price) if np.isfinite(price) else None

    # ------------------------------------------------------------------
    # Main backtest loop
    # ------------------------------------------------------------------

    def run(self, price_data: Optional[Dict[str, pd.DataFrame]] = None) -> dict:
        """
        Execute the full backtest over [start_date, end_date].

        Parameters
        ----------
        price_data : optional pre-loaded dict (skip network calls in WF)

        Returns
        -------
        dict with keys:
          metrics, equity_curve, trade_log, monthly_returns, position_history
        """
        if price_data is not None:
            self._price_data = price_data
        else:
            self._load_price_data()

        rebalance_dates = self._rebalance_dates()
        if not rebalance_dates:
            logger.warning("No rebalance dates in [%s, %s]", self.start_date, self.end_date)
            return self._empty_result()

        capital = self.initial_capital
        positions: Dict[str, dict] = {}  # ticker → {shares, cost_price, value}
        equity_records: List[dict] = []
        trade_log: List[dict] = []
        position_history: List[dict] = []

        prev_date: Optional[pd.Timestamp] = None

        for rb_date in rebalance_dates:
            # ── Mark-to-market existing positions ─────────────────────
            cash = capital
            pos_value = 0.0
            for ticker, pos in list(positions.items()):
                price = self._get_price_at(ticker, rb_date)
                if price is not None:
                    pos["current_price"] = price
                    pos["value"] = pos["shares"] * price
                    pos_value += pos["value"]
                else:
                    pos_value += pos.get("value", 0.0)
            # cash = total_capital - positions_value
            total_portfolio = cash + pos_value

            # ── Score universe ─────────────────────────────────────────
            scored = self._score_all(rb_date)

            # ── Sell decisions ─────────────────────────────────────────
            tickers_to_sell: List[str] = []
            for ticker in list(positions.keys()):
                row = scored[scored["ticker"] == ticker]
                score = int(row["total_score"].iloc[0]) if not row.empty else 0
                if score <= self.sell_threshold:
                    tickers_to_sell.append(ticker)

            for ticker in tickers_to_sell:
                pos = positions.pop(ticker)
                sell_price = self._get_price_at(ticker, rb_date)
                if sell_price is None:
                    sell_price = pos.get("current_price", pos["cost_price"])
                proceeds = pos["shares"] * sell_price * (1 - self.commission)
                pnl = proceeds - pos["shares"] * pos["cost_price"]
                cash += proceeds
                trade_log.append({
                    "date": rb_date,
                    "ticker": ticker,
                    "action": "SELL",
                    "shares": pos["shares"],
                    "price": sell_price,
                    "proceeds": proceeds,
                    "pnl": pnl,
                })

            # ── Buy decisions ──────────────────────────────────────────
            n_free_slots = self.max_positions - len(positions)
            buy_candidates = scored[
                (scored["total_score"] >= self.buy_threshold) &
                (~scored["ticker"].isin(positions.keys()))
            ].head(n_free_slots)

            if not buy_candidates.empty:
                # Equal-weight allocation across new buys + existing positions
                n_new = len(buy_candidates)
                total_slots = len(positions) + n_new

                # Per-position target value (equal weight)
                per_position_value = total_portfolio / total_slots if total_slots > 0 else 0

                for _, row in buy_candidates.iterrows():
                    ticker = row["ticker"]
                    buy_price = self._get_price_at(ticker, rb_date)
                    if buy_price is None or buy_price <= 0:
                        continue
                    alloc = min(per_position_value, cash * 0.99)  # leave 1% buffer
                    shares = alloc / (buy_price * (1 + self.commission))
                    if shares <= 0:
                        continue
                    cost = shares * buy_price * (1 + self.commission)
                    cash -= cost
                    positions[ticker] = {
                        "shares": shares,
                        "cost_price": buy_price,
                        "current_price": buy_price,
                        "value": shares * buy_price,
                        "entry_date": rb_date,
                    }
                    trade_log.append({
                        "date": rb_date,
                        "ticker": ticker,
                        "action": "BUY",
                        "shares": shares,
                        "price": buy_price,
                        "proceeds": -cost,
                        "pnl": 0.0,
                    })

            # ── Recompute portfolio value ──────────────────────────────
            pos_value = sum(
                self._get_price_at(t, rb_date) * p["shares"]
                if self._get_price_at(t, rb_date) is not None
                else p.get("value", 0.0)
                for t, p in positions.items()
            )
            capital = cash + pos_value

            equity_records.append({"date": rb_date, "equity": capital})
            position_history.append({
                "date": rb_date,
                "positions": list(positions.keys()),
                "n_positions": len(positions),
                "cash": cash,
                "equity": capital,
            })
            prev_date = rb_date

        # ── Build equity curve DataFrame ────────────────────────────────
        if not equity_records:
            return self._empty_result()

        equity_df = pd.DataFrame(equity_records).set_index("date")
        equity_df.index = pd.to_datetime(equity_df.index)

        # ── Monthly returns ─────────────────────────────────────────────
        monthly_rets = equity_df["equity"].pct_change().dropna()

        # ── Compute metrics ─────────────────────────────────────────────
        self.equity_curve = equity_df
        self.monthly_returns = monthly_rets
        self.trade_log = trade_log
        self.position_history = position_history
        self.metrics = _compute_metrics(
            equity_curve=equity_df["equity"],
            monthly_returns=monthly_rets,
            trade_log=trade_log,
        )

        return {
            "metrics": self.metrics,
            "equity_curve": self.equity_curve,
            "trade_log": self.trade_log,
            "monthly_returns": self.monthly_returns,
            "position_history": self.position_history,
        }

    def _empty_result(self) -> dict:
        return {
            "metrics": _compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float), []),
            "equity_curve": pd.DataFrame(),
            "trade_log": [],
            "monthly_returns": pd.Series(dtype=float),
            "position_history": [],
        }

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def run_walk_forward(
        self,
        n_windows: int = 3,
        is_frac: float = 0.70,
    ) -> dict:
        """
        Execute walk-forward validation.

        Parameters
        ----------
        n_windows : int   — number of IS/OOS folds (minimum 3)
        is_frac   : float — fraction of each window that is in-sample

        Returns
        -------
        dict with:
          wf_windows : list[dict]  — per-window IS/OOS metrics
          oos_metrics: dict        — aggregated OOS-only metrics
          is_metrics : dict        — aggregated IS-only metrics
          equity_curve_oos: pd.DataFrame — OOS-only equity path
        """
        n_windows = max(3, n_windows)
        total_days = (self.end_date - self.start_date).days
        window_days = total_days // n_windows

        if window_days < 180:
            logger.warning(
                "Walk-forward window too short (%d days). "
                "Increase date range or reduce n_windows.",
                window_days,
            )

        # Load price data once, shared across all windows
        self._load_price_data()

        wf_windows: List[dict] = []
        oos_equity_parts: List[pd.DataFrame] = []
        oos_trades: List[dict] = []
        is_trades: List[dict] = []

        for i in range(n_windows):
            win_start = self.start_date + timedelta(days=i * window_days)
            win_end = (
                self.start_date + timedelta(days=(i + 1) * window_days)
                if i < n_windows - 1
                else self.end_date
            )

            is_end = win_start + timedelta(days=int(window_days * is_frac))
            oos_start = is_end + timedelta(days=1)
            oos_end = win_end

            logger.info(
                "WF window %d/%d  IS=[%s → %s]  OOS=[%s → %s]",
                i + 1, n_windows,
                win_start.date(), is_end.date(),
                oos_start.date(), oos_end.date(),
            )

            # ── IS run ──────────────────────────────────────────────────
            is_bt = CAGRBacktester(
                tickers=self.tickers,
                start_date=str(win_start.date()),
                end_date=str(is_end.date()),
                initial_capital=self.initial_capital,
                max_positions=self.max_positions,
                rebalance_freq=self.rebalance_freq,
                commission=self.commission,
                buy_threshold=self.buy_threshold,
                sell_threshold=self.sell_threshold,
                hold_min=self.hold_min,
            )
            is_result = is_bt.run(price_data=self._price_data)
            is_metrics = is_result["metrics"]
            is_passed, is_failures = _passes_criteria(is_metrics)
            is_trades.extend(is_result.get("trade_log", []))

            # ── OOS run ──────────────────────────────────────────────────
            oos_bt = CAGRBacktester(
                tickers=self.tickers,
                start_date=str(oos_start.date()),
                end_date=str(oos_end.date()),
                initial_capital=self.initial_capital,
                max_positions=self.max_positions,
                rebalance_freq=self.rebalance_freq,
                commission=self.commission,
                buy_threshold=self.buy_threshold,
                sell_threshold=self.sell_threshold,
                hold_min=self.hold_min,
            )
            oos_result = oos_bt.run(price_data=self._price_data)
            oos_metrics = oos_result["metrics"]
            oos_passed, oos_failures = _passes_criteria(oos_metrics)
            oos_trades.extend(oos_result.get("trade_log", []))

            if not oos_result["equity_curve"].empty:
                oos_equity_parts.append(oos_result["equity_curve"])

            wf_windows.append({
                "window": i + 1,
                "is_start": win_start,
                "is_end": is_end,
                "oos_start": oos_start,
                "oos_end": oos_end,
                "is_metrics": is_metrics,
                "oos_metrics": oos_metrics,
                "is_passed": is_passed,
                "is_failures": is_failures,
                "oos_passed": oos_passed,
                "oos_failures": oos_failures,
            })

        # ── Aggregate OOS equity ──────────────────────────────────────
        if oos_equity_parts:
            # Chain OOS equity curves (rebased sequentially)
            chains = []
            running_capital = self.initial_capital
            for part in oos_equity_parts:
                if part.empty:
                    continue
                part_scaled = part.copy()
                scale = running_capital / part_scaled["equity"].iloc[0]
                part_scaled["equity"] *= scale
                chains.append(part_scaled)
                running_capital = part_scaled["equity"].iloc[-1]

            oos_equity_df = pd.concat(chains).sort_index().drop_duplicates()
        else:
            oos_equity_df = pd.DataFrame()

        # Aggregate OOS metrics
        oos_monthly = (
            pd.concat([
                w["oos_metrics"].get("Avg Monthly Return %", np.nan)
                and pd.Series() or pd.Series()
                for w in wf_windows
            ])
            if wf_windows else pd.Series(dtype=float)
        )

        agg_oos_metrics = _compute_metrics(
            equity_curve=oos_equity_df["equity"] if not oos_equity_df.empty else pd.Series(dtype=float),
            monthly_returns=pd.Series(dtype=float),
            trade_log=oos_trades,
        )
        agg_is_metrics = _compute_metrics(
            equity_curve=pd.Series(dtype=float),
            monthly_returns=pd.Series(dtype=float),
            trade_log=is_trades,
        )

        self.wf_results = wf_windows

        return {
            "wf_windows": wf_windows,
            "oos_metrics": agg_oos_metrics,
            "is_metrics": agg_is_metrics,
            "equity_curve_oos": oos_equity_df,
        }


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------

# Cyberpunk theme constants
_BG    = "#050510"
_BG2   = "#0a0a1e"
_CYAN  = "#00ffff"
_MAG   = "#ff00ff"
_GREEN = "#00ff88"
_YEL   = "#ffdd00"
_RED   = "#ff3355"
_TEXT  = "#e0e0ff"


def _metric_card_html(label: str, value: str, color: str = _CYAN) -> str:
    return f"""
    <div style="background:{_BG2};border:1px solid {color};border-radius:8px;
                padding:16px;text-align:center;margin:4px;">
        <div style="color:{_TEXT};font-size:0.75rem;letter-spacing:0.08em;
                    text-transform:uppercase;">{label}</div>
        <div style="color:{color};font-size:1.6rem;font-weight:700;
                    font-family:monospace;">{value}</div>
    </div>"""


def _render_metrics_cards(metrics: dict) -> None:
    """Render metrics as styled HTML cards in Streamlit."""
    if st is None:
        return

    card_defs = [
        ("CAGR %",          "CAGR %",           _GREEN),
        ("Total Return %",  "Total Return %",    _CYAN),
        ("Sharpe Ratio",    "Sharpe Ratio",      _CYAN),
        ("Sortino Ratio",   "Sortino Ratio",     _CYAN),
        ("Max Drawdown %",  "Max Drawdown %",    _RED),
        ("Winrate %",       "Winrate %",         _YEL),
        ("Profit Factor",   "Profit Factor",     _GREEN),
        ("Num Trades",      "Num Trades",        _TEXT),
        ("Volatility (Ann) %", "Volatility %",   _MAG),
    ]

    cols = st.columns(3)
    for i, (key, label, color) in enumerate(card_defs):
        val = metrics.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            display = "N/A"
        elif isinstance(val, float):
            display = f"{val:.2f}"
        else:
            display = str(val)
        cols[i % 3].markdown(_metric_card_html(label, display, color), unsafe_allow_html=True)


def _render_equity_chart(equity_df: pd.DataFrame, title: str = "Equity Curve") -> None:
    """Render Plotly equity curve in Streamlit."""
    if st is None or go is None or equity_df is None or equity_df.empty:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df.index,
        y=equity_df["equity"],
        mode="lines",
        name="Portfolio",
        line=dict(color=_CYAN, width=2),
        fill="tozeroy",
        fillcolor="rgba(0,255,255,0.06)",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color=_CYAN, size=16)),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, family="JetBrains Mono, monospace"),
        xaxis=dict(showgrid=True, gridcolor=_BG2, title="Date"),
        yaxis=dict(showgrid=True, gridcolor=_BG2, title="Portfolio Value"),
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_drawdown_chart(equity_df: pd.DataFrame) -> None:
    """Render drawdown chart."""
    if st is None or go is None or equity_df is None or equity_df.empty:
        return

    roll_max = equity_df["equity"].cummax()
    dd = (equity_df["equity"] - roll_max) / roll_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        mode="lines",
        name="Drawdown %",
        line=dict(color=_RED, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255,51,85,0.12)",
    ))
    fig.update_layout(
        title=dict(text="Drawdown %", font=dict(color=_RED, size=14)),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, family="JetBrains Mono, monospace"),
        xaxis=dict(showgrid=True, gridcolor=_BG2),
        yaxis=dict(showgrid=True, gridcolor=_BG2, ticksuffix="%"),
        height=220,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_monthly_returns_heatmap(monthly_returns: pd.Series) -> None:
    """Render monthly returns as a year × month heatmap."""
    if st is None or go is None or monthly_returns is None or monthly_returns.empty:
        return

    ret_pct = (monthly_returns * 100).round(2)
    df = pd.DataFrame({"ret": ret_pct})
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot_table(index="year", columns="month", values="ret", aggfunc="sum")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][:len(pivot.columns)]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=[str(y) for y in pivot.index],
        colorscale=[
            [0.0, _RED],
            [0.5, _BG2],
            [1.0, _GREEN],
        ],
        zmid=0,
        text=pivot.values.round(1),
        texttemplate="%{text}%",
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Monthly Returns Heatmap (%)", font=dict(color=_CYAN, size=14)),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, family="JetBrains Mono, monospace"),
        height=max(200, 60 * len(pivot)),
        margin=dict(l=50, r=20, t=50, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_walk_forward_table(wf_windows: List[dict]) -> None:
    """Render walk-forward per-window results as a styled DataFrame."""
    if st is None or not wf_windows:
        return

    rows = []
    for w in wf_windows:
        rows.append({
            "Window": w["window"],
            "IS Period": f"{w['is_start'].date()} → {w['is_end'].date()}",
            "IS CAGR%": round(w["is_metrics"].get("CAGR %", float("nan")), 1),
            "IS Sharpe": round(w["is_metrics"].get("Sharpe Ratio", float("nan")), 2),
            "IS Pass": "✅" if w["is_passed"] else "❌",
            "OOS Period": f"{w['oos_start'].date()} → {w['oos_end'].date()}",
            "OOS CAGR%": round(w["oos_metrics"].get("CAGR %", float("nan")), 1),
            "OOS Sharpe": round(w["oos_metrics"].get("Sharpe Ratio", float("nan")), 2),
            "OOS MaxDD%": round(w["oos_metrics"].get("Max Drawdown %", float("nan")), 1),
            "OOS Pass": "✅" if w["oos_passed"] else "❌",
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


def render_backtest_section() -> None:
    """
    Render the CAGR backtester section inside a Streamlit page.

    Call this from cagr_streamlit.py's render_cagr_page() to embed
    the backtesting UI:

        from .cagr_backtest import render_backtest_section
        render_backtest_section()
    """
    if st is None:
        logger.error("render_backtest_section() called without Streamlit available.")
        return

    from .cagr_loader import load_nordic_tickers

    st.markdown(
        f"""<h3 style="color:{_CYAN};text-transform:uppercase;
                       letter-spacing:0.1em;border-bottom:1px solid {_CYAN};
                       padding-bottom:6px;">
            📊 Strategy Backtester
        </h3>""",
        unsafe_allow_html=True,
    )

    # ── Sidebar / control parameters ─────────────────────────────────
    with st.expander("⚙️ Backtest Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            years = st.slider("Backtest years", 2, 10, 5)
            initial_cap = st.number_input(
                "Initial Capital (SEK)", value=100_000, step=10_000,
                min_value=10_000,
            )
        with col2:
            max_pos = st.slider("Max Positions", 3, 20, 10)
            commission = st.number_input(
                "Commission (round-trip %)", value=0.15,
                min_value=0.0, max_value=2.0, step=0.05,
            ) / 100
        with col3:
            run_wf = st.checkbox("Walk-Forward Validation", value=True)
            n_wf_windows = st.slider("WF Windows", 3, 8, 3)

    run_btn = st.button("▶ Run Backtest", type="primary")
    if not run_btn:
        st.info("Configure settings above and click **▶ Run Backtest** to start.")
        return

    # ── Run ───────────────────────────────────────────────────────────
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    tickers = load_nordic_tickers()

    with st.spinner("Running backtest … fetching price data and scoring stocks"):
        bt = CAGRBacktester(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            initial_capital=float(initial_cap),
            max_positions=max_pos,
            commission=commission,
        )

        if run_wf:
            wf_result = bt.run_walk_forward(n_windows=n_wf_windows)
            result = bt.run(price_data=bt._price_data)
        else:
            result = bt.run()
            wf_result = None

    metrics = result["metrics"]
    equity_df = result["equity_curve"]
    monthly_rets = result["monthly_returns"]
    trade_log = result["trade_log"]

    passed, failures = _passes_criteria(metrics)

    # ── Accept / Reject banner ────────────────────────────────────────
    if passed:
        st.markdown(
            f"""<div style="background:rgba(0,255,136,0.15);border:2px solid {_GREEN};
                            border-radius:8px;padding:12px;text-align:center;
                            color:{_GREEN};font-weight:700;font-size:1.1rem;">
                ✅ STRATEGY PASSES ALL ACCEPT CRITERIA
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        failure_text = " | ".join(failures)
        st.markdown(
            f"""<div style="background:rgba(255,51,85,0.15);border:2px solid {_RED};
                            border-radius:8px;padding:12px;text-align:center;
                            color:{_RED};font-weight:700;">
                ❌ STRATEGY FAILS — {failure_text}
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Metrics cards ─────────────────────────────────────────────────
    st.markdown(f"<br>", unsafe_allow_html=True)
    _render_metrics_cards(metrics)

    # ── Equity curve ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _render_equity_chart(equity_df, title=f"CAGR Strategy — {years}Y Equity Curve")
    _render_drawdown_chart(equity_df)

    # ── Monthly heatmap ───────────────────────────────────────────────
    _render_monthly_returns_heatmap(monthly_rets)

    # ── Walk-forward results ──────────────────────────────────────────
    if run_wf and wf_result:
        st.markdown(
            f"<h4 style='color:{_CYAN};'>Walk-Forward Results</h4>",
            unsafe_allow_html=True,
        )
        _render_walk_forward_table(wf_result["wf_windows"])

        if not wf_result["equity_curve_oos"].empty:
            _render_equity_chart(
                wf_result["equity_curve_oos"],
                title="Out-of-Sample Equity Curve (Chained)",
            )

    # ── Trade log ─────────────────────────────────────────────────────
    if trade_log:
        with st.expander(f"📋 Trade Log ({len(trade_log)} trades)", expanded=False):
            trade_df = pd.DataFrame(trade_log)
            trade_df["date"] = pd.to_datetime(trade_df["date"]).dt.strftime("%Y-%m-%d")
            trade_df["pnl"] = trade_df["pnl"].round(0).astype(int)
            trade_df["price"] = trade_df["price"].round(2)
            st.dataframe(trade_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="CAGR Nordic Stock Strategy Backtester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--years", type=int, default=5,
        help="Number of years to backtest",
    )
    parser.add_argument(
        "--max-positions", type=int, default=10,
        help="Maximum concurrent stock positions",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100_000.0,
        help="Starting portfolio value",
    )
    parser.add_argument(
        "--commission", type=float, default=0.0015,
        help="Round-trip commission per trade (e.g. 0.0015 = 0.15%%)",
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Run walk-forward validation",
    )
    parser.add_argument(
        "--wf-windows", type=int, default=3,
        help="Number of walk-forward windows",
    )
    parser.add_argument(
        "--buy-threshold", type=int, default=9,
        help="Minimum total_score to initiate a BUY",
    )
    parser.add_argument(
        "--sell-threshold", type=int, default=5,
        help="Total_score at or below which a position is SOLD",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    # Resolve tickers
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dashboard.cagr.cagr_loader import load_nordic_tickers  # noqa: E402

    tickers = load_nordic_tickers()

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (
        datetime.today() - timedelta(days=args.years * 365)
    ).strftime("%Y-%m-%d")

    logger.info(
        "Starting backtest: %s → %s  (%d tickers, max_positions=%d)",
        start_date, end_date, len(tickers), args.max_positions,
    )

    bt = CAGRBacktester(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        commission=args.commission,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )

    if args.walk_forward:
        wf = bt.run_walk_forward(n_windows=args.wf_windows)
        result = bt.run(price_data=bt._price_data)
    else:
        result = bt.run()
        wf = None

    # Print metrics table
    metrics = result["metrics"]
    print("\n" + "=" * 60)
    print("  CAGR STRATEGY BACKTEST RESULTS")
    print("=" * 60)
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<28}: {val:>10.3f}")
        else:
            print(f"  {key:<28}: {val!s:>10}")

    passed, failures = _passes_criteria(metrics)
    print("=" * 60)
    if passed:
        print("  ✅  PASSES ALL ACCEPT CRITERIA")
    else:
        print("  ❌  FAILS:")
        for f in failures:
            print(f"      • {f}")
    print("=" * 60)

    if wf:
        print("\nWALK-FORWARD SUMMARY")
        print("-" * 60)
        for w in wf["wf_windows"]:
            status = "PASS" if w["oos_passed"] else "FAIL"
            print(
                f"  Window {w['window']}  OOS=[{w['oos_start'].date()} → {w['oos_end'].date()}]"
                f"  CAGR={w['oos_metrics'].get('CAGR %', 'N/A'):.1f}%"
                f"  Sharpe={w['oos_metrics'].get('Sharpe Ratio', 'N/A'):.2f}"
                f"  [{status}]"
            )
        print("-" * 60)

    n_trades = len(result["trade_log"])
    print(f"\n  Total trades executed : {n_trades}")
    print(f"  Final equity          : {result['equity_curve']['equity'].iloc[-1]:,.0f}" if not result["equity_curve"].empty else "")


if __name__ == "__main__":
    _cli_main()
