"""
backtest_engine.py
Unified EMA crossover backtest engine for all screener modes.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=10)
def fetch_backtest_data(ticker: str, years: int = 3) -> pd.DataFrame:
    """Fetch OHLCV data for backtesting."""
    try:
        period = f"{years}y"
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, auto_adjust=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.dropna()
    except Exception as e:
        logger.warning("fetch_backtest_data(%s): %s", ticker, e)
        return pd.DataFrame()


def run_backtest(
    ticker: str,
    years: int = 3,
    mode: str = "swing",
) -> dict:
    """
    Run EMA crossover backtest.

    Modes:
      "swing"  — EMA 10/20 cross, exit on EMA 10 break, ½ ATR stop
      "long"   — EMA 50/200 cross, exit on EMA 200 break
      "ovtlyr" — EMA 10/20 cross with ADX filter + volume confirm

    Returns dict with:
      trades: list of dicts
      metrics: dict (win_rate, profit_factor, max_dd, total_return, sharpe, etc)
      equity_curve: list of dicts
    """
    df = fetch_backtest_data(ticker, years)
    if df.empty or len(df) < 200:
        return {"trades": [], "metrics": {}, "equity_curve": [], "error": "Insufficient data"}

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0, index=df.index)

    # Compute indicators based on mode
    if mode == "swing":
        fast_ema = close.ewm(span=10).mean()
        slow_ema = close.ewm(span=20).mean()
        trend_ema = close.ewm(span=50).mean()
        exit_ema = close.ewm(span=10).mean()
    elif mode == "long":
        fast_ema = close.ewm(span=50).mean()
        slow_ema = close.ewm(span=200).mean()
        trend_ema = close.ewm(span=200).mean()
        exit_ema = close.ewm(span=200).mean()
    else:  # ovtlyr
        fast_ema = close.ewm(span=10).mean()
        slow_ema = close.ewm(span=20).mean()
        trend_ema = close.ewm(span=50).mean()
        exit_ema = close.ewm(span=10).mean()

    # ATR for stop loss
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1)),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # ADX for OVTLYR mode
    adx = pd.Series(25.0, index=df.index)
    if mode == "ovtlyr":
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0),
            index=df.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0),
            index=df.index,
        )
        plus_di = 100 * plus_dm.rolling(14).mean() / atr.replace(0, float("nan"))
        minus_di = 100 * minus_dm.rolling(14).mean() / atr.replace(0, float("nan"))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, float("nan"))
        adx = dx.rolling(14).mean().fillna(0)

    vol_sma = volume.rolling(20).mean()

    # Simulate trades
    trades = []
    equity = [100.0]
    in_trade = False
    entry_price = 0.0
    entry_date = None
    stop_loss = 0.0

    for i in range(201, len(df)):
        date = df.index[i]
        price = close.iloc[i]

        if not in_trade:
            # Entry signal
            cross_up = (
                fast_ema.iloc[i] > slow_ema.iloc[i]
                and fast_ema.iloc[i - 1] <= slow_ema.iloc[i - 1]
            )
            above_trend = price > trend_ema.iloc[i]

            # Mode-specific filters
            if mode == "ovtlyr":
                adx_ok = adx.iloc[i] > 20
                vol_ok = (
                    volume.iloc[i] > vol_sma.iloc[i] * 1.0
                    if vol_sma.iloc[i] > 0
                    else True
                )
                entry_ok = cross_up and above_trend and adx_ok and vol_ok
            else:
                entry_ok = cross_up and above_trend

            if entry_ok:
                in_trade = True
                entry_price = price
                entry_date = date
                stop_loss = price - atr.iloc[i] * 0.5

        else:
            # Exit signals
            exit_signal = False
            exit_reason = ""

            # Stop loss
            if price <= stop_loss:
                exit_signal = True
                exit_reason = "Stop Loss"

            # EMA exit
            elif price < exit_ema.iloc[i]:
                exit_signal = True
                exit_reason = "EMA Trail"

            # Cross down
            elif (
                fast_ema.iloc[i] < slow_ema.iloc[i]
                and fast_ema.iloc[i - 1] >= slow_ema.iloc[i - 1]
            ):
                exit_signal = True
                exit_reason = "Cross Down"

            if exit_signal:
                ret_pct = (price / entry_price - 1) * 100
                trades.append({
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(date.date()),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(price, 2),
                    "return_pct": round(ret_pct, 2),
                    "exit_reason": exit_reason,
                    "duration": (date - entry_date).days,
                })
                equity.append(equity[-1] * (1 + ret_pct / 100))
                in_trade = False

        if not in_trade:
            equity.append(equity[-1])

    # Compute metrics
    if not trades:
        return {"trades": [], "metrics": {"total_return": 0, "num_trades": 0}, "equity_curve": []}

    returns = [t["return_pct"] for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    total_return = (equity[-1] / equity[0] - 1) * 100
    years_actual = len(df) / 252
    cagr = ((equity[-1] / equity[0]) ** (1 / max(years_actual, 0.1)) - 1) * 100

    peak = pd.Series(equity).cummax()
    dd = (pd.Series(equity) - peak) / peak * 100
    max_dd = dd.min()

    monthly_eq = pd.Series(equity[::21]) if len(equity) > 21 else pd.Series(equity)
    monthly_ret = monthly_eq.pct_change().dropna()
    sharpe = (
        (monthly_ret.mean() / monthly_ret.std()) * np.sqrt(12)
        if monthly_ret.std() > 0
        else 0
    )

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.01

    metrics = {
        "total_return": round(total_return, 1),
        "cagr": round(cagr, 1),
        "max_dd": round(max_dd, 1),
        "sharpe": round(sharpe, 2),
        "win_rate": round(len(wins) / len(returns) * 100, 1),
        "profit_factor": round(gross_profit / gross_loss, 2),
        "num_trades": len(trades),
        "avg_return": round(np.mean(returns), 1),
        "avg_win": round(np.mean(wins), 1) if wins else 0,
        "avg_loss": round(np.mean(losses), 1) if losses else 0,
        "avg_duration": round(np.mean([t["duration"] for t in trades]), 0),
        "best_trade": round(max(returns), 1),
        "worst_trade": round(min(returns), 1),
    }

    return {"trades": trades, "metrics": metrics, "equity_curve": equity}


def run_batch_backtest(
    tickers: list,
    years: int = 3,
    mode: str = "swing",
) -> pd.DataFrame:
    """
    Run backtest on multiple tickers and return summary DataFrame.
    """
    results = []
    for ticker in tickers:
        bt = run_backtest(ticker, years, mode)
        m = bt.get("metrics", {})
        if m and m.get("num_trades", 0) > 0:
            results.append({
                "Ticker": ticker,
                "Total Return %": m.get("total_return", 0),
                "CAGR %": m.get("cagr", 0),
                "Max DD %": m.get("max_dd", 0),
                "Sharpe": m.get("sharpe", 0),
                "Win Rate %": m.get("win_rate", 0),
                "Profit Factor": m.get("profit_factor", 0),
                "Trades": m.get("num_trades", 0),
                "Avg Return %": m.get("avg_return", 0),
            })

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values("Total Return %", ascending=False)
        .reset_index(drop=True)
    )
