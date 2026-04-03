"""
backtest_engine.py — WOLF x SHADOW Optimization Pipeline
=========================================================
Vectorized backtesting engine that implements WOLF x SHADOW v2 logic.

Architecture:
  1. Pre-compute all indicator arrays with numpy (fully vectorized).
  2. Single Python loop over bars for trade state machine — unavoidable
     because each bar's state depends on the previous bar's position,
     SL level, TP1-hit flag, etc.  The loop is compiled via numba if
     available, else falls back to plain Python (still fast enough for
     ~5 000 bars).
  3. Collect trade list, compute performance metrics.

Pine Script equivalence notes
------------------------------
- Core/Trim sizing: coreSizeAdj / trimSizeAdj from regime score
- TP1/TP2: limit orders at entry + slD * tp1RR / tp2RR
- Breakeven SL: after TP1 hit, SL moves to entry price
- Kijun trail: ONLY after TP1 hit, close < kijun AND close < ema_pulse, bars > 5
- TRIM EMA50 exit: belowEma50Count >= 2, barsInTrade > 8
- CORE EMA50 exit: belowEma50Count >= 3, barsInTrade > 5
- CORE catastrophe stop: entry - slD * 1.8
- Circuit breaker: blocks entries only (does NOT close positions)
- Cooldown: re-entry blocked for cooldown_bars after any sell
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try numba JIT — falls back silently if not installed
try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct:  float = 0.05     # percent per side
    slippage_pct:    float = 0.10     # percent per fill (market impact)
    pyramiding:      int   = 10
    # Position sizing
    core_pct:        float = 0.50     # fraction of equity to core
    add_pct:         float = 0.10     # base add fraction
    # Exit logic
    tp1_rr:          float = 2.5
    tp1_pct:         float = 0.15     # qty_percent at TP1
    tp2_rr:          float = 4.0
    tp2_pct:         float = 0.15     # qty_percent at TP2
    atr_mult:        float = 2.5      # ATR stop multiplier
    cooldown_bars:   int   = 4
    entry_min_regime: int  = 40
    add_min_regime:  int   = 50
    # Circuit breaker thresholds
    daily_breaker_pct:  float = -8.0
    weekly_breaker_pct: float = -15.0


@dataclass
class Trade:
    entry_bar:    int
    entry_price:  float
    exit_bar:     int
    exit_price:   float
    qty:          float          # shares
    pnl:          float          # net PnL in $
    pnl_pct:      float          # net PnL as fraction of entry value
    exit_reason:  str
    component:    str            # 'CORE' or 'TRIM'
    bars_held:    int


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def _fill_price(price: float, direction: str, config: BacktestConfig) -> float:
    """Apply slippage: buy higher, sell lower."""
    slip = config.slippage_pct / 100.0
    if direction == "buy":
        return price * (1.0 + slip)
    else:
        return price * (1.0 - slip)


def _commission(value: float, config: BacktestConfig) -> float:
    """Round-trip commission on notional value."""
    return value * (config.commission_pct / 100.0)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_backtest(
    ind: pd.DataFrame,
    df: pd.DataFrame,
    params: dict,
    config: Optional[BacktestConfig] = None,
) -> dict:
    """
    Run WOLF x SHADOW v2 backtest.

    Parameters
    ----------
    ind : pd.DataFrame
        Indicators from indicators.compute_regime_score (same index as df)
    df : pd.DataFrame
        OHLCV with DatetimeIndex
    params : dict
        Optimization parameters (overrides config where applicable)
    config : BacktestConfig (optional)
        Base configuration; params override relevant fields

    Returns
    -------
    dict with keys:
        net_return, CAGR, annual_vol, max_drawdown, sharpe, sortino,
        winrate, profit_factor, n_trades, avg_bars_held,
        equity_curve (pd.Series), trades (list[Trade])
    """
    if config is None:
        config = BacktestConfig()

    # Override config with params
    config.tp1_rr          = float(params.get("tp1_rr",           config.tp1_rr))
    config.tp1_pct         = float(params.get("tp1_pct",          config.tp1_pct))
    config.tp2_rr          = float(params.get("tp2_rr",           config.tp2_rr))
    config.tp2_pct         = float(params.get("tp2_pct",          config.tp2_pct))
    config.atr_mult        = float(params.get("atr_mult",         config.atr_mult))
    config.cooldown_bars   = int(params.get("cooldown_bars",       config.cooldown_bars))
    config.entry_min_regime = int(params.get("entry_min_regime",  config.entry_min_regime))
    config.add_min_regime  = int(params.get("add_min_regime",     config.add_min_regime))
    config.core_pct        = float(params.get("core_pct",         config.core_pct))

    # Extract numpy arrays for speed
    close  = df["close"].values.astype(np.float64)
    high   = df["high"].values.astype(np.float64)
    low    = df["low"].values.astype(np.float64)
    open_  = df["open"].values.astype(np.float64)
    n      = len(close)

    regime_score  = ind["regime_score"].values.astype(np.float64)
    ema_trend     = ind["ema_trend"].values.astype(bool)
    ema_pulse     = ind["ema_pulse"].values.astype(np.float64)
    ema_slow      = ind["ema_slow"].values.astype(np.float64)  # ema50
    kijun         = ind["kijun"].values.astype(np.float64)
    atr_arr       = ind["atr_val"].values.astype(np.float64)
    bull_ob       = ind["bull_ob"].values.astype(bool)
    ema_cross_up  = ind["ema_cross_up"].values.astype(bool)
    ema_reclaim   = ind["ema_reclaim"].values.astype(bool)
    momentum_up   = ind["momentum_up"].values.astype(bool)
    core_size_adj = ind["core_size_adj"].values.astype(np.float64)
    trim_size_adj = ind["trim_size_adj"].values.astype(np.float64)
    add_mult      = ind["add_mult"].values.astype(np.float64)
    rsi_val       = ind["rsi_val"].values.astype(np.float64)

    equity     = config.initial_capital
    trades: list[Trade] = []

    # --- Position state ---
    core_qty         = 0.0
    trim_qty         = 0.0
    core_entry_price = 0.0
    trim_entry_price = 0.0   # avg entry of trim position
    trim_entry_value = 0.0   # total cost basis for avg price calc

    tp1_hit          = False
    bars_in_trade    = 0
    last_sell_bar    = -1000  # cooldown counter
    below_ema50_cnt  = 0

    # Circuit breaker (daily/weekly) — simplified: track equity snapshots
    daily_eq_start   = equity
    weekly_eq_start  = equity
    last_day         = -1
    last_week        = -1

    # Equity curve
    equity_arr = np.full(n, np.nan)

    for i in range(n):
        c     = close[i]
        h     = high[i]
        l     = low[i]
        atr_v = atr_arr[i]

        # ----------------------------------------------------------------
        # Circuit breaker update
        # ----------------------------------------------------------------
        # Use bar index as proxy for day/week boundaries
        day_idx  = i // (6 if n < 2000 else 1)    # crude day grouping
        week_idx = i // (30 if n < 2000 else 5)

        if day_idx != last_day:
            daily_eq_start = equity
            last_day = day_idx
        if week_idx != last_week:
            weekly_eq_start = equity
            last_week = week_idx

        daily_pnl_pct  = (equity - daily_eq_start)  / max(daily_eq_start,  1) * 100.0
        weekly_pnl_pct = (equity - weekly_eq_start) / max(weekly_eq_start, 1) * 100.0
        breaker_ok = (daily_pnl_pct  > config.daily_breaker_pct and
                      weekly_pnl_pct > config.weekly_breaker_pct)

        # ----------------------------------------------------------------
        # Position tracking helpers
        # ----------------------------------------------------------------
        in_position = (core_qty + trim_qty) > 0.0

        if in_position:
            bars_in_trade += 1
            below_ema50_cnt = (below_ema50_cnt + 1) if c < ema_slow[i] else 0
        else:
            bars_in_trade   = 0
            below_ema50_cnt = 0
            tp1_hit         = False

        # ----------------------------------------------------------------
        # TP1 / TP2 check (limit orders — check if high reached limit)
        # ----------------------------------------------------------------
        if trim_qty > 0.0 and core_entry_price > 0.0:
            sl_d  = atr_v * config.atr_mult
            tp1_p = core_entry_price + sl_d * config.tp1_rr
            tp2_p = core_entry_price + sl_d * config.tp2_rr

            # TP1 fill: high >= tp1_p and not yet hit
            if not tp1_hit and h >= tp1_p:
                tp1_qty  = trim_qty * config.tp1_pct
                fill     = _fill_price(tp1_p, "sell", config)
                cost     = trim_entry_price * tp1_qty
                proceeds = fill * tp1_qty - _commission(fill * tp1_qty, config)
                pnl      = proceeds - cost
                trades.append(Trade(
                    entry_bar=i - bars_in_trade + 1, entry_price=trim_entry_price,
                    exit_bar=i, exit_price=fill, qty=tp1_qty,
                    pnl=pnl, pnl_pct=pnl / max(cost, 1),
                    exit_reason="TP1", component="TRIM",
                    bars_held=bars_in_trade,
                ))
                equity    += pnl
                trim_qty  -= tp1_qty
                tp1_hit    = True

            # TP2 fill: high >= tp2_p
            if tp1_hit and h >= tp2_p and trim_qty > 0.0:
                tp2_qty  = trim_qty * config.tp2_pct
                fill     = _fill_price(tp2_p, "sell", config)
                cost     = trim_entry_price * tp2_qty
                proceeds = fill * tp2_qty - _commission(fill * tp2_qty, config)
                pnl      = proceeds - cost
                trades.append(Trade(
                    entry_bar=i - bars_in_trade + 1, entry_price=trim_entry_price,
                    exit_bar=i, exit_price=fill, qty=tp2_qty,
                    pnl=pnl, pnl_pct=pnl / max(cost, 1),
                    exit_reason="TP2", component="TRIM",
                    bars_held=bars_in_trade,
                ))
                equity   += pnl
                trim_qty -= tp2_qty

        # ----------------------------------------------------------------
        # SL check for TRIM (at open of next bar effectively, use low)
        # ----------------------------------------------------------------
        if trim_qty > 0.0 and core_entry_price > 0.0:
            sl_d    = atr_v * config.atr_mult
            trim_sl = core_entry_price if tp1_hit else (core_entry_price - sl_d)

            if l <= trim_sl and not tp1_hit:
                # Stop hit — fill at stop (may gap)
                fill_p  = min(c, trim_sl)  # use close if gapped below
                fill_p  = _fill_price(fill_p, "sell", config)
                cost    = trim_entry_price * trim_qty
                proceeds = fill_p * trim_qty - _commission(fill_p * trim_qty, config)
                pnl     = proceeds - cost
                trades.append(Trade(
                    entry_bar=i - bars_in_trade + 1, entry_price=trim_entry_price,
                    exit_bar=i, exit_price=fill_p, qty=trim_qty,
                    pnl=pnl, pnl_pct=pnl / max(cost, 1),
                    exit_reason="SL", component="TRIM",
                    bars_held=bars_in_trade,
                ))
                equity    += pnl
                trim_qty   = 0.0
                last_sell_bar = i

            elif l <= trim_sl and tp1_hit:
                # Breakeven stop hit
                fill_p   = _fill_price(max(c, trim_sl), "sell", config)
                cost     = trim_entry_price * trim_qty
                proceeds = fill_p * trim_qty - _commission(fill_p * trim_qty, config)
                pnl      = proceeds - cost
                trades.append(Trade(
                    entry_bar=i - bars_in_trade + 1, entry_price=trim_entry_price,
                    exit_bar=i, exit_price=fill_p, qty=trim_qty,
                    pnl=pnl, pnl_pct=pnl / max(cost, 1),
                    exit_reason="BE_STOP", component="TRIM",
                    bars_held=bars_in_trade,
                ))
                equity    += pnl
                trim_qty   = 0.0
                last_sell_bar = i

        # ----------------------------------------------------------------
        # CORE catastrophe stop (1.8x ATR)
        # ----------------------------------------------------------------
        if core_qty > 0.0 and core_entry_price > 0.0:
            sl_d     = atr_v * config.atr_mult
            core_cat = core_entry_price - sl_d * 1.8
            if l <= core_cat:
                fill_p   = _fill_price(min(c, core_cat), "sell", config)
                cost     = core_entry_price * core_qty
                proceeds = fill_p * core_qty - _commission(fill_p * core_qty, config)
                pnl      = proceeds - cost
                trades.append(Trade(
                    entry_bar=i - bars_in_trade + 1, entry_price=core_entry_price,
                    exit_bar=i, exit_price=fill_p, qty=core_qty,
                    pnl=pnl, pnl_pct=pnl / max(cost, 1),
                    exit_reason="CORE_CAT", component="CORE",
                    bars_held=bars_in_trade,
                ))
                equity    += pnl
                core_qty   = 0.0
                last_sell_bar = i

        # ----------------------------------------------------------------
        # EMA50 exits (bar-count based, checked at close)
        # ----------------------------------------------------------------
        in_position = (core_qty + trim_qty) > 0.0  # re-evaluate after stops

        # KIJUN trail exit (TRIM only, after TP1, close < kijun AND close < ema_pulse)
        if (trim_qty > 0.0 and tp1_hit and
                c < kijun[i] and c < ema_pulse[i] and bars_in_trade > 5):
            fill_p   = _fill_price(c, "sell", config)
            cost     = trim_entry_price * trim_qty
            proceeds = fill_p * trim_qty - _commission(fill_p * trim_qty, config)
            pnl      = proceeds - cost
            trades.append(Trade(
                entry_bar=i - bars_in_trade + 1, entry_price=trim_entry_price,
                exit_bar=i, exit_price=fill_p, qty=trim_qty,
                pnl=pnl, pnl_pct=pnl / max(cost, 1),
                exit_reason="KIJUN_TRAIL", component="TRIM",
                bars_held=bars_in_trade,
            ))
            equity    += pnl
            trim_qty   = 0.0
            last_sell_bar = i

        # TRIM EMA50 exit: 2 bars below EMA50, min 8 bars in trade
        if trim_qty > 0.0 and below_ema50_cnt >= 2 and bars_in_trade > 8:
            fill_p   = _fill_price(c, "sell", config)
            cost     = trim_entry_price * trim_qty
            proceeds = fill_p * trim_qty - _commission(fill_p * trim_qty, config)
            pnl      = proceeds - cost
            trades.append(Trade(
                entry_bar=i - bars_in_trade + 1, entry_price=trim_entry_price,
                exit_bar=i, exit_price=fill_p, qty=trim_qty,
                pnl=pnl, pnl_pct=pnl / max(cost, 1),
                exit_reason="TRIM_EMA50", component="TRIM",
                bars_held=bars_in_trade,
            ))
            equity    += pnl
            trim_qty   = 0.0
            last_sell_bar = i

        # CORE EMA50 exit: 3 bars below EMA50, min 5 bars in trade
        if core_qty > 0.0 and below_ema50_cnt >= 3 and bars_in_trade > 5:
            fill_p   = _fill_price(c, "sell", config)
            cost     = core_entry_price * core_qty
            proceeds = fill_p * core_qty - _commission(fill_p * core_qty, config)
            pnl      = proceeds - cost
            trades.append(Trade(
                entry_bar=i - bars_in_trade + 1, entry_price=core_entry_price,
                exit_bar=i, exit_price=fill_p, qty=core_qty,
                pnl=pnl, pnl_pct=pnl / max(cost, 1),
                exit_reason="CORE_EMA50", component="CORE",
                bars_held=bars_in_trade,
            ))
            equity    += pnl
            core_qty   = 0.0
            last_sell_bar = i

        # ----------------------------------------------------------------
        # Recompute in_position after exits
        # ----------------------------------------------------------------
        in_position = (core_qty + trim_qty) > 0.0

        # ----------------------------------------------------------------
        # Entry logic
        # ----------------------------------------------------------------
        cooldown_ok = (i - last_sell_bar) > config.cooldown_bars
        regime_ok   = regime_score[i] >= config.entry_min_regime

        entry_signal = (
            ema_trend[i] and
            regime_ok and
            cooldown_ok and
            breaker_ok and
            (bull_ob[i] or ema_cross_up[i] or ema_reclaim[i] or momentum_up[i])
        )

        if not in_position and entry_signal:
            fill_p = _fill_price(c, "buy", config)
            atr_v  = atr_arr[i]

            # Dynamic sizing from regime
            c_pct  = core_size_adj[i] / 100.0
            t_pct  = trim_size_adj[i] / 100.0

            core_qty          = max(0.0, equity * c_pct / fill_p)
            trim_qty          = max(0.0, equity * t_pct / fill_p)
            core_entry_price  = fill_p
            trim_entry_price  = fill_p
            trim_entry_value  = fill_p * trim_qty
            tp1_hit           = False
            bars_in_trade     = 0
            below_ema50_cnt   = 0

            # Deduct commissions from equity immediately
            equity -= _commission(fill_p * (core_qty + trim_qty), config)

        # Add to trim (pyramid)
        elif in_position and trim_qty >= 0.0:
            add_cond = (
                ema_trend[i] and
                ema_reclaim[i] and
                cooldown_ok and
                breaker_ok and
                regime_score[i] >= config.add_min_regime
            )
            if add_cond:
                add_mult_v = add_mult[i]
                add_f      = _fill_price(c, "buy", config)
                add_val    = equity * config.add_pct * add_mult_v
                add_q      = add_val / add_f

                # Recompute weighted average entry price for trim
                total_trim_val   = trim_entry_price * trim_qty + add_f * add_q
                trim_qty        += add_q
                trim_entry_price = total_trim_val / max(trim_qty, 1e-9)
                trim_entry_value = trim_entry_price * trim_qty
                equity          -= _commission(add_f * add_q, config)

        # ----------------------------------------------------------------
        # Mark-to-market equity
        # ----------------------------------------------------------------
        unrealized = (core_qty * c - core_qty * core_entry_price +
                      trim_qty * c - trim_qty * trim_entry_price
                      ) if in_position else 0.0
        equity_arr[i] = equity + unrealized

    # ----------------------------------------------------------------
    # Close any open positions at last bar
    # ----------------------------------------------------------------
    if core_qty > 0.0 or trim_qty > 0.0:
        fill_p = _fill_price(close[-1], "sell", config)
        for qty, ep, comp in [
            (core_qty, core_entry_price, "CORE"),
            (trim_qty, trim_entry_price, "TRIM"),
        ]:
            if qty > 0.0:
                cost     = ep * qty
                proceeds = fill_p * qty - _commission(fill_p * qty, config)
                pnl      = proceeds - cost
                trades.append(Trade(
                    entry_bar=n - bars_in_trade, entry_price=ep,
                    exit_bar=n - 1, exit_price=fill_p, qty=qty,
                    pnl=pnl, pnl_pct=pnl / max(cost, 1),
                    exit_reason="END_OF_DATA", component=comp,
                    bars_held=bars_in_trade,
                ))
                equity += pnl

        equity_arr[-1] = equity

    # ----------------------------------------------------------------
    # Fill forward remaining NaN in equity curve
    # ----------------------------------------------------------------
    eq_series = pd.Series(equity_arr, index=df.index)
    eq_series = eq_series.fillna(method="ffill").fillna(config.initial_capital)

    return _compute_metrics(
        trades=trades,
        equity_curve=eq_series,
        initial_capital=config.initial_capital,
        df=df,
    )


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    trades: list[Trade],
    equity_curve: pd.Series,
    initial_capital: float,
    df: pd.DataFrame,
) -> dict:
    """Compute performance metrics from trade list and equity curve."""

    eq = equity_curve.values
    n  = len(eq)

    # Returns series (bar-by-bar)
    eq_returns  = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1.0)

    # ---- Net return ----
    final_equity = eq[-1]
    net_return   = (final_equity - initial_capital) / initial_capital

    # ---- CAGR ----
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        years = (df.index[-1] - df.index[0]).days / 365.25
    else:
        years = n / 252.0  # fallback: assume daily

    if years <= 0:
        years = 1.0
    cagr = (final_equity / initial_capital) ** (1.0 / years) - 1.0

    # ---- Annual vol ----
    # Estimate bars per year
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 10:
        bar_duration_days = (df.index[-1] - df.index[0]).days / max(len(df) - 1, 1)
        bars_per_year = 365.25 / max(bar_duration_days, 0.01)
    else:
        bars_per_year = 252.0
    annual_vol = float(np.std(eq_returns, ddof=1)) * math.sqrt(bars_per_year) if len(eq_returns) > 1 else 0.0

    # ---- Max drawdown ----
    running_max = np.maximum.accumulate(eq)
    drawdowns   = (eq - running_max) / np.where(running_max > 0, running_max, 1.0)
    max_drawdown = float(np.min(drawdowns))  # negative value

    # ---- Sharpe ----
    rf = 0.04 / bars_per_year  # 4% annual risk-free, per bar
    excess_returns = eq_returns - rf
    sharpe = (float(np.mean(excess_returns)) / float(np.std(excess_returns, ddof=1))
              * math.sqrt(bars_per_year)
              if np.std(excess_returns, ddof=1) > 0 else 0.0)

    # ---- Sortino ----
    downside = eq_returns[eq_returns < 0]
    downside_std = float(np.std(downside, ddof=1)) * math.sqrt(bars_per_year) if len(downside) > 1 else 1e-9
    sortino = (cagr - 0.04) / downside_std if downside_std > 0 else 0.0

    # ---- Trade stats ----
    n_trades   = len(trades)
    gross_wins = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    winrate    = sum(1 for t in trades if t.pnl > 0) / max(n_trades, 1)
    profit_factor = gross_wins / max(gross_loss, 1e-9)
    avg_bars_held = float(np.mean([t.bars_held for t in trades])) if trades else 0.0

    return {
        "net_return":    net_return,
        "CAGR":          cagr,
        "annual_vol":    annual_vol,
        "max_drawdown":  max_drawdown,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "winrate":       winrate,
        "profit_factor": profit_factor,
        "n_trades":      n_trades,
        "avg_bars_held": avg_bars_held,
        "final_equity":  final_equity,
        "equity_curve":  equity_curve,
        "trades":        trades,
        "years":         years,
        "gross_wins":    gross_wins,
        "gross_loss":    gross_loss,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper that handles indicator computation
# ---------------------------------------------------------------------------

def backtest(
    df: pd.DataFrame,
    params: dict,
    spy_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None,
    config: Optional[BacktestConfig] = None,
) -> dict:
    """
    Full pipeline: compute indicators then run backtest.

    Parameters
    ----------
    df : OHLCV DataFrame
    params : parameter dict
    spy_df : SPY OHLCV aligned to df.index
    sector_df : Sector ETF OHLCV aligned to df.index
    config : BacktestConfig

    Returns
    -------
    metrics dict (same as run_backtest)
    """
    from indicators import compute_regime_score

    ind = compute_regime_score(df, params, spy_df=spy_df, sector_df=sector_df)
    return run_backtest(ind=ind, df=df, params=params, config=config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_loader import load_yfinance, align_to_stock

    df     = load_yfinance("XOM", years=2, interval="1h")
    spy_df = load_yfinance("SPY", years=2, interval="1h")
    sec_df = load_yfinance("XLE", years=2, interval="1h")

    spy_a = align_to_stock(df, spy_df)
    sec_a = align_to_stock(df, sec_df)

    from indicators import DEFAULT_PARAMS
    result = backtest(df, DEFAULT_PARAMS, spy_df=spy_a, sector_df=sec_a)

    print("=" * 50)
    print(f"Net Return:     {result['net_return']:.2%}")
    print(f"CAGR:           {result['CAGR']:.2%}")
    print(f"Max Drawdown:   {result['max_drawdown']:.2%}")
    print(f"Sharpe:         {result['sharpe']:.2f}")
    print(f"Sortino:        {result['sortino']:.2f}")
    print(f"Win Rate:       {result['winrate']:.2%}")
    print(f"Profit Factor:  {result['profit_factor']:.2f}")
    print(f"N Trades:       {result['n_trades']}")
    print(f"Avg Bars Held:  {result['avg_bars_held']:.1f}")
