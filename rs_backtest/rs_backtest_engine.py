"""
RS Momentum Backtest Engine
EMA(20)/EMA(50) crossover strategy with RS pre-filtering.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Sector → Tickers mapping
# ---------------------------------------------------------------------------

SECTOR_MAP: dict[str, dict] = {
    "Energy": {
        "etf": "XLE",
        "tickers": ["XOM", "CVX", "COP", "DVN", "OXY", "EOG", "FANG", "SLB", "HAL", "MPC", "VLO", "PSX", "HES", "CTRA", "APA", "EQT"],
    },
    "Materials": {
        "etf": "XLB",
        "tickers": ["NEM", "GOLD", "FNV", "WPM", "KGC", "FCX", "NUE", "STLD", "CLF", "CF", "MOS", "ALB", "SCCO", "MP"],
    },
    "Financials": {
        "etf": "XLF",
        "tickers": ["JPM", "BAC", "GS", "MS", "WFC", "BLK", "SCHW", "CB", "MMC", "SPGI", "CME", "BRK-B"],
    },
    "Technology": {
        "etf": "XLK",
        "tickers": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "QCOM", "TXN", "AMAT", "NOW", "INTU", "IBM"],
    },
    "Healthcare": {
        "etf": "XLV",
        "tickers": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "AMGN", "ISRG", "SYK", "MDT", "GILD", "ZTS"],
    },
    "Industrials": {
        "etf": "XLI",
        "tickers": ["CAT", "DE", "HON", "GE", "RTX", "UNP", "BA", "LOW", "LMT", "MMM", "ITW", "ETN", "PH", "ROK"],
    },
    "Consumer Disc.": {
        "etf": "XLY",
        "tickers": ["TSLA", "AMZN", "HD", "COST", "BKNG", "NFLX", "MCD", "NKE", "TJX", "SBUX", "GM", "F"],
    },
    "Consumer Staples": {
        "etf": "XLP",
        "tickers": ["PG", "KO", "PEP", "WMT", "CL", "PM", "MO", "MDLZ", "GIS", "EL", "STZ", "KHC"],
    },
    "Utilities": {
        "etf": "XLU",
        "tickers": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PEG", "XEL", "WEC", "ED", "ETR"],
    },
    "Real Estate": {
        "etf": "XLRE",
        "tickers": ["AMT", "PLD", "CCI", "EQIX", "PSA", "O", "WELL", "DLR", "SPG", "AVB", "EQR", "VTR"],
    },
    "Comm. Services": {
        "etf": "XLC",
        "tickers": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "VZ", "T", "ATVI", "EA", "TTWO", "MTCH"],
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_download(ticker: str, start: str, end: str, progress: bool = False) -> pd.DataFrame:
    """Download OHLCV data via yfinance, returning empty DataFrame on failure."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=progress, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def _compute_emas(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return EMA20, EMA50, EMA200."""
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    return ema20, ema50, ema200


def _compute_sharpe(returns: pd.Series, trading_days: int = 252) -> float:
    """Annualised Sharpe ratio from a series of daily % returns."""
    if returns.empty or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(trading_days))


def _max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown as a negative percentage."""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min() * 100)


# ---------------------------------------------------------------------------
# RS Rankings
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def compute_rs_rankings(tickers: tuple, period: str = "7mo") -> dict:
    """Compute 6-month relative strength for all tickers.

    Returns dict: ticker -> {return_6m, rs_rank, rs_percentile}
    """
    end = datetime.today()
    # 7mo window to ensure ~126 trading days
    start = end - timedelta(days=215)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    returns: dict[str, float] = {}
    for ticker in tickers:
        df = _safe_download(ticker, start_str, end_str)
        if df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].dropna()
        if len(close) < 20:
            continue
        # Use last 126 trading days (≈6 months)
        lookback = close.iloc[-126:] if len(close) >= 126 else close
        ret = (lookback.iloc[-1] / lookback.iloc[0] - 1) * 100
        returns[ticker] = float(ret)

    if not returns:
        return {}

    series = pd.Series(returns).sort_values(ascending=False)
    n = len(series)
    result: dict[str, dict] = {}
    for rank, (ticker, ret) in enumerate(series.items(), start=1):
        result[ticker] = {
            "return_6m": round(ret, 2),
            "rs_rank": rank,
            "rs_percentile": round((1 - rank / n) * 100, 1),
        }
    return result


# ---------------------------------------------------------------------------
# Single-ticker EMA backtest
# ---------------------------------------------------------------------------


def run_ema_backtest(ticker: str, years: int = 3, sector_etf: str = "XLE") -> dict:
    """Run EMA crossover backtest for a single ticker.

    Strategy:
      BUY  when EMA20 crosses above EMA50 AND Close > EMA200
      SELL when EMA20 crosses below EMA50 OR  Close < EMA200

    Returns dict with keys: trades, metrics, equity_curve, ticker, sector_etf
    """
    end = datetime.today()
    # Extra 300 days for EMA200 warm-up
    start = end - timedelta(days=int(years * 365) + 300)
    df = _safe_download(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    empty_result = {
        "ticker": ticker,
        "sector_etf": sector_etf,
        "trades": [],
        "metrics": {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "num_trades": 0,
            "avg_duration": 0.0,
            "sharpe": 0.0,
        },
        "equity_curve": [],
    }

    if df.empty or "Close" not in df.columns:
        return empty_result

    close = df["Close"].dropna()
    if len(close) < 210:
        return empty_result

    ema20, ema50, ema200 = _compute_emas(close)

    # Trim to the requested backtest window only (after warm-up)
    cutoff = end - timedelta(days=int(years * 365))
    close = close[close.index >= pd.Timestamp(cutoff)]
    ema20 = ema20[ema20.index >= pd.Timestamp(cutoff)]
    ema50 = ema50[ema50.index >= pd.Timestamp(cutoff)]
    ema200 = ema200[ema200.index >= pd.Timestamp(cutoff)]

    if len(close) < 2:
        return empty_result

    dates = close.index
    trades: list[dict] = []
    in_trade = False
    entry_date = None
    entry_price = None

    equity = 100.0
    equity_curve: list[dict] = [{"date": str(dates[0].date()), "equity": equity}]
    daily_returns: list[float] = []

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]
        c = float(close.iloc[i])
        c_prev = float(close.iloc[i - 1])
        e20 = float(ema20.iloc[i])
        e50 = float(ema50.iloc[i])
        e200 = float(ema200.iloc[i])
        e20_prev = float(ema20.iloc[i - 1])
        e50_prev = float(ema50.iloc[i - 1])

        if in_trade:
            # Daily P&L for equity curve
            day_ret = (c - c_prev) / c_prev
            equity = equity * (1 + day_ret)
            daily_returns.append(day_ret)

            # SELL conditions
            crossed_below = (e20_prev >= e50_prev) and (e20 < e50)
            price_below_ema200 = c < e200
            if crossed_below or price_below_ema200:
                trade_ret = (c - entry_price) / entry_price * 100
                duration = (curr_date - entry_date).days
                trades.append({
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(curr_date.date()),
                    "entry_price": round(float(entry_price), 2),
                    "exit_price": round(c, 2),
                    "return_pct": round(trade_ret, 2),
                    "duration_days": duration,
                })
                in_trade = False
                entry_date = None
                entry_price = None
        else:
            daily_returns.append(0.0)
            # BUY conditions
            crossed_above = (e20_prev < e50_prev) and (e20 >= e50)
            price_above_ema200 = c > e200
            if crossed_above and price_above_ema200:
                in_trade = True
                entry_date = curr_date
                entry_price = c

        equity_curve.append({"date": str(curr_date.date()), "equity": round(equity, 4)})

    # Close any open trade at last price
    if in_trade and entry_price is not None:
        last_c = float(close.iloc[-1])
        trade_ret = (last_c - entry_price) / entry_price * 100
        duration = (dates[-1] - entry_date).days
        trades.append({
            "entry_date": str(entry_date.date()),
            "exit_date": str(dates[-1].date()),
            "entry_price": round(float(entry_price), 2),
            "exit_price": round(last_c, 2),
            "return_pct": round(trade_ret, 2),
            "duration_days": duration,
        })

    # Metrics
    if not trades:
        return {**empty_result, "equity_curve": equity_curve}

    returns_arr = [t["return_pct"] for t in trades]
    wins = [r for r in returns_arr if r > 0]
    losses = [r for r in returns_arr if r <= 0]

    win_rate = len(wins) / len(trades) * 100 if trades else 0.0
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
    total_return = equity - 100.0
    max_dd = _max_drawdown(pd.Series([p["equity"] for p in equity_curve]))
    avg_duration = float(np.mean([t["duration_days"] for t in trades]))
    sharpe = _compute_sharpe(pd.Series(daily_returns))

    metrics = {
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": round(max_dd, 2),
        "total_return": round(total_return, 2),
        "num_trades": len(trades),
        "avg_duration": round(avg_duration, 1),
        "sharpe": round(sharpe, 2),
    }

    return {
        "ticker": ticker,
        "sector_etf": sector_etf,
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
    }


# ---------------------------------------------------------------------------
# Sector-level RS backtest
# ---------------------------------------------------------------------------


def run_rs_sector_backtest(years: int = 3, top_pct: float = 20.0) -> pd.DataFrame:
    """Run backtest for top RS% tickers across all 11 sectors.

    Steps per sector:
      1. Compute RS rankings for all tickers
      2. Keep top `top_pct`% by 6M momentum
      3. Run EMA backtest on each
      4. Aggregate metrics

    Returns DataFrame with one row per sector.
    """
    rows = []
    all_results: dict[str, list[dict]] = {}

    for sector, info in SECTOR_MAP.items():
        etf = info["etf"]
        tickers = info["tickers"]

        # RS rankings
        rs = compute_rs_rankings(tuple(tickers), period="7mo")
        if not rs:
            continue

        # Filter top N%
        n_keep = max(1, int(np.ceil(len(rs) * top_pct / 100)))
        sorted_tickers = sorted(rs.keys(), key=lambda t: rs[t]["return_6m"], reverse=True)
        top_tickers = sorted_tickers[:n_keep]

        # Run backtests
        results = []
        for t in top_tickers:
            r = run_ema_backtest(t, years=years, sector_etf=etf)
            if r["metrics"]["num_trades"] > 0:
                results.append(r)

        all_results[sector] = results

        if not results:
            continue

        metrics_list = [r["metrics"] for r in results]
        win_rates = [m["win_rate"] for m in metrics_list]
        pfs = [m["profit_factor"] for m in metrics_list]
        dds = [m["max_drawdown"] for m in metrics_list]
        total_rets = [m["total_return"] for m in metrics_list]

        best_idx = int(np.argmax(total_rets))
        worst_idx = int(np.argmin(total_rets))

        rows.append({
            "Sector": sector,
            "ETF": etf,
            "Tickers_Tested": len(top_tickers),
            "Avg_WinRate": round(float(np.mean(win_rates)), 1),
            "Avg_ProfitFactor": round(float(np.mean(pfs)), 2),
            "Avg_MaxDD": round(float(np.mean(dds)), 2),
            "Avg_TotalReturn": round(float(np.mean(total_rets)), 2),
            "Best_Ticker": results[best_idx]["ticker"],
            "Best_Return": round(total_rets[best_idx], 2),
            "Worst_Ticker": results[worst_idx]["ticker"],
            "Worst_Return": round(total_rets[worst_idx], 2),
            "_results": results,  # for trade drill-down — stripped before display
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df, all_results
