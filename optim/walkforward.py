"""
walkforward.py — WOLF x SHADOW Optimization Pipeline
=====================================================
Rolling (and anchored) walk-forward validation.

Minimum 3 windows, 70% IS / 30% OOS per window.
Reports per-window IS and OOS metrics.
Computes OOS consistency and degradation ratio.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Window generators
# ---------------------------------------------------------------------------

def walk_forward_windows(
    df: pd.DataFrame,
    n_windows: int = 3,
    is_fraction: float = 0.70,
    anchored: bool = False,
    min_is_bars: int = 100,
    min_oos_bars: int = 30,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate IS/OOS window pairs.

    Parameters
    ----------
    df : OHLCV DataFrame with DatetimeIndex
    n_windows : number of windows (minimum 3)
    is_fraction : fraction of each window used for IS (e.g. 0.70)
    anchored : if True, IS always starts at the beginning of df
    min_is_bars : minimum bars required for IS window
    min_oos_bars : minimum bars required for OOS window

    Returns
    -------
    List of (is_df, oos_df) tuples
    """
    n_windows = max(n_windows, 3)
    n = len(df)

    if anchored:
        windows = _anchored_windows(df, n_windows, is_fraction,
                                    min_is_bars, min_oos_bars)
    else:
        windows = _rolling_windows(df, n_windows, is_fraction,
                                   min_is_bars, min_oos_bars)

    if len(windows) < 1:
        raise ValueError(
            f"Not enough data for walk-forward validation. "
            f"Have {n} bars, need at least "
            f"{n_windows * (min_is_bars + min_oos_bars)} bars."
        )

    if len(windows) < n_windows:
        logger.warning(
            "Requested %d windows but only %d fit in %d bars "
            "(min_is=%d, min_oos=%d). Proceeding with %d windows.",
            n_windows, len(windows), n, min_is_bars, min_oos_bars, len(windows),
        )

    return windows


def _rolling_windows(
    df: pd.DataFrame,
    n_windows: int,
    is_fraction: float,
    min_is_bars: int,
    min_oos_bars: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Classic rolling walk-forward:
    Window 1: bars [0 : W]          IS=[0:IS_end],  OOS=[IS_end:W]
    Window 2: bars [step : W+step]  IS=[step:...],  OOS=[...]
    ...
    where W = total window size and step = OOS size.
    """
    n          = len(df)
    oos_size   = max(min_oos_bars, int(n / (n_windows + (1 - is_fraction) * n_windows)))
    window_size = int(oos_size / (1.0 - is_fraction))
    is_size     = window_size - oos_size

    if is_size < min_is_bars:
        is_size   = min_is_bars
        oos_size  = max(min_oos_bars, int(is_size * (1.0 - is_fraction) / is_fraction))
        window_size = is_size + oos_size

    step    = oos_size  # slide by OOS size each window
    windows = []

    for w in range(n_windows):
        start   = w * step
        is_end  = start + is_size
        oos_end = is_end + oos_size

        if oos_end > n:
            oos_end = n
        if is_end >= n:
            break
        if (oos_end - is_end) < min_oos_bars:
            break

        is_df  = df.iloc[start:is_end].copy()
        oos_df = df.iloc[is_end:oos_end].copy()
        windows.append((is_df, oos_df))

    return windows


def _anchored_windows(
    df: pd.DataFrame,
    n_windows: int,
    is_fraction: float,
    min_is_bars: int,
    min_oos_bars: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Anchored walk-forward: IS always starts at bar 0, expands by OOS size each window.
    """
    n        = len(df)
    oos_size = max(min_oos_bars, n // (n_windows * 3))
    windows  = []

    for w in range(1, n_windows + 1):
        is_end  = int(n * is_fraction * w / n_windows)
        oos_end = is_end + oos_size

        if oos_end > n:
            oos_end = n
        if is_end < min_is_bars:
            continue
        if (oos_end - is_end) < min_oos_bars:
            break

        is_df  = df.iloc[0:is_end].copy()
        oos_df = df.iloc[is_end:oos_end].copy()
        windows.append((is_df, oos_df))

    return windows


# ---------------------------------------------------------------------------
# Full walk-forward run
# ---------------------------------------------------------------------------

def run_walk_forward(
    df: pd.DataFrame,
    params: dict,
    spy_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None,
    n_windows: int = 3,
    is_fraction: float = 0.70,
    anchored: bool = False,
    config=None,
) -> dict:
    """
    Run walk-forward validation for a single parameter set.

    Returns
    -------
    dict with keys:
      windows : list of per-window result dicts
      is_metrics_avg : averaged IS metrics
      oos_metrics_avg : averaged OOS metrics
      oos_positive_pct : fraction of OOS windows with positive return
      degradation_ratio : OOS_score / IS_score
      is_score : mean IS objective score
      oos_score : mean OOS objective score
      overfit_warning : bool
    """
    from backtest_engine import backtest, BacktestConfig
    from objective import score_metrics

    if config is None:
        config = BacktestConfig()

    windows = walk_forward_windows(df, n_windows=n_windows,
                                   is_fraction=is_fraction, anchored=anchored)

    results = []

    for w_idx, (is_df, oos_df) in enumerate(windows):
        # Align reference data to IS window
        spy_is = spy_oos = None
        sec_is = sec_oos = None

        if spy_df is not None:
            spy_is  = spy_df.reindex(is_df.index,  method="ffill")
            spy_oos = spy_df.reindex(oos_df.index, method="ffill")
        if sector_df is not None:
            sec_is  = sector_df.reindex(is_df.index,  method="ffill")
            sec_oos = sector_df.reindex(oos_df.index, method="ffill")

        is_metrics  = backtest(is_df,  params, spy_is,  sec_is,  config)
        oos_metrics = backtest(oos_df, params, spy_oos, sec_oos, config)

        is_score  = score_metrics(is_metrics)
        oos_score = score_metrics(oos_metrics)

        results.append({
            "window":         w_idx + 1,
            "is_start":       is_df.index[0],
            "is_end":         is_df.index[-1],
            "oos_start":      oos_df.index[0],
            "oos_end":        oos_df.index[-1],
            "is_n_bars":      len(is_df),
            "oos_n_bars":     len(oos_df),
            # IS metrics
            "is_cagr":        is_metrics["CAGR"],
            "is_mdd":         is_metrics["max_drawdown"],
            "is_sharpe":      is_metrics["sharpe"],
            "is_n_trades":    is_metrics["n_trades"],
            "is_winrate":     is_metrics["winrate"],
            "is_score":       is_score,
            # OOS metrics
            "oos_cagr":       oos_metrics["CAGR"],
            "oos_mdd":        oos_metrics["max_drawdown"],
            "oos_sharpe":     oos_metrics["sharpe"],
            "oos_n_trades":   oos_metrics["n_trades"],
            "oos_winrate":    oos_metrics["winrate"],
            "oos_score":      oos_score,
            # Equity curves
            "is_equity":      is_metrics["equity_curve"],
            "oos_equity":     oos_metrics["equity_curve"],
        })

    # Aggregate
    is_scores  = [r["is_score"]  for r in results]
    oos_scores = [r["oos_score"] for r in results]

    mean_is  = float(np.mean(is_scores))
    mean_oos = float(np.mean(oos_scores))

    oos_positive_pct  = sum(1 for r in results if r["oos_cagr"] > 0) / max(len(results), 1)
    degradation_ratio = mean_oos / mean_is if abs(mean_is) > 1e-6 else 0.0

    def _avg_metric(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    is_metrics_avg = {
        "CAGR":          _avg_metric("is_cagr"),
        "max_drawdown":  _avg_metric("is_mdd"),
        "sharpe":        _avg_metric("is_sharpe"),
        "n_trades":      _avg_metric("is_n_trades"),
        "winrate":       _avg_metric("is_winrate"),
    }
    oos_metrics_avg = {
        "CAGR":          _avg_metric("oos_cagr"),
        "max_drawdown":  _avg_metric("oos_mdd"),
        "sharpe":        _avg_metric("oos_sharpe"),
        "n_trades":      _avg_metric("oos_n_trades"),
        "winrate":       _avg_metric("oos_winrate"),
    }

    overfit_warning = (
        degradation_ratio < 0.5 or
        is_metrics_avg["CAGR"] > 2.0 or
        (is_metrics_avg["CAGR"] > 0 and oos_metrics_avg["CAGR"] < 0)
    )

    return {
        "windows":          results,
        "is_metrics_avg":   is_metrics_avg,
        "oos_metrics_avg":  oos_metrics_avg,
        "oos_positive_pct": oos_positive_pct,
        "degradation_ratio": degradation_ratio,
        "is_score":         mean_is,
        "oos_score":        mean_oos,
        "overfit_warning":  overfit_warning,
        "n_windows":        len(results),
        "params":           params,
    }


# ---------------------------------------------------------------------------
# Multi-param walk-forward (top-N param sets)
# ---------------------------------------------------------------------------

def run_walk_forward_batch(
    df: pd.DataFrame,
    params_list: list[dict],
    spy_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None,
    n_windows: int = 3,
    is_fraction: float = 0.70,
    anchored: bool = False,
    config=None,
) -> list[dict]:
    """
    Run walk-forward for each parameter set in params_list.
    Returns list of result dicts sorted by OOS score descending.
    """
    results = []
    for i, params in enumerate(params_list):
        logger.info("Walk-forward [%d/%d] ...", i + 1, len(params_list))
        try:
            r = run_walk_forward(
                df, params, spy_df, sector_df,
                n_windows=n_windows, is_fraction=is_fraction,
                anchored=anchored, config=config,
            )
            r["param_idx"] = i
            results.append(r)
        except Exception as exc:
            logger.warning("WF failed for param set %d: %s", i, exc)

    results.sort(key=lambda x: x["oos_score"], reverse=True)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_loader import load_yfinance, align_to_stock
    from indicators import DEFAULT_PARAMS

    df  = load_yfinance("XOM", years=2, interval="1h")
    spy = load_yfinance("SPY", years=2, interval="1h")
    sec = load_yfinance("XLE", years=2, interval="1h")

    spy_a = align_to_stock(df, spy)
    sec_a = align_to_stock(df, sec)

    result = run_walk_forward(df, DEFAULT_PARAMS, spy_a, sec_a, n_windows=3)

    print("\n=== Walk-Forward Summary ===")
    for w in result["windows"]:
        print(f"Window {w['window']}: IS CAGR={w['is_cagr']:.1%}  OOS CAGR={w['oos_cagr']:.1%}  "
              f"OOS MDD={w['oos_mdd']:.1%}  OOS Trades={w['oos_n_trades']}")

    print(f"\nIS avg:  CAGR={result['is_metrics_avg']['CAGR']:.1%}  "
          f"MDD={result['is_metrics_avg']['max_drawdown']:.1%}")
    print(f"OOS avg: CAGR={result['oos_metrics_avg']['CAGR']:.1%}  "
          f"MDD={result['oos_metrics_avg']['max_drawdown']:.1%}")
    print(f"OOS positive windows: {result['oos_positive_pct']:.0%}")
    print(f"Degradation ratio:    {result['degradation_ratio']:.2f}")
    print(f"Overfit warning:      {result['overfit_warning']}")
