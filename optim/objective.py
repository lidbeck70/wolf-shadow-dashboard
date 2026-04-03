"""
objective.py — WOLF x SHADOW Optimization Pipeline
====================================================
Optuna objective function.

Score = CAGR - 0.5 * abs(max_drawdown)

Constraints:
  - max_drawdown > -50% (reject worse)
  - IS CAGR > 200% annualized → penalize (likely overfit)

Walk-forward pruning: reports intermediate values at each WF window
so Optuna MedianPruner can kill bad trials early.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import optuna

from data_loader import align_to_stock
from backtest_engine import backtest, BacktestConfig
from walkforward import walk_forward_windows

logger = logging.getLogger(__name__)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------

def score_metrics(metrics: dict) -> float:
    """
    Primary score: CAGR - 0.5 * |max_drawdown|

    Both are fractions (e.g., CAGR=0.25 = 25%).
    Returns -999 if constraints violated.
    """
    cagr = metrics.get("CAGR", -1.0)
    mdd  = metrics.get("max_drawdown", -1.0)   # negative value

    # Hard constraint: reject if drawdown worse than -50%
    if mdd < -0.50:
        return -999.0

    # Penalise likely overfit
    overfit_penalty = 0.0
    if cagr > 2.0:  # 200% CAGR annualized
        overfit_penalty = (cagr - 2.0) * 2.0  # steep penalty above 200%

    return cagr - 0.5 * abs(mdd) - overfit_penalty


# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial) -> dict:
    """
    Define Optuna parameter space for WOLF x SHADOW v2.

    All ranges specified in the task brief.
    """
    return {
        # Ichimoku
        "tenkan_len":       trial.suggest_int("tenkan_len",        5,  15),
        "kijun_len":        trial.suggest_int("kijun_len",        20,  40),
        "spanb_len":        trial.suggest_int("spanb_len",        40,  80),
        "displacement":     26,   # fixed — changing displacement changes kumo semantics
        # EMA
        "ema_pulse":        trial.suggest_int("ema_pulse",         5,  20),
        "ema_fast":         trial.suggest_int("ema_fast",         10,  40),
        "ema_slow":         trial.suggest_int("ema_slow",         30, 120),
        "ema_macro":        200,  # fixed
        # RSI
        "rsi_len":          trial.suggest_int("rsi_len",          10,  20),
        "rsi_hot":          70,   # fixed
        # Extension
        "ext_pct":          trial.suggest_float("ext_pct",        1.5,  4.0),
        # ATR
        "atr_len":          14,   # fixed
        "atr_mult":         trial.suggest_float("atr_mult",       0.5,  3.0),
        # Order block
        "ob_lookback":      5,    # fixed
        # TP
        "tp1_rr":           trial.suggest_float("tp1_rr",         1.5,  4.0),
        "tp2_rr":           trial.suggest_float("tp2_rr",         2.5,  6.0),
        "tp1_pct":          trial.suggest_float("tp1_pct",        0.05, 0.30),
        "tp2_pct":          trial.suggest_float("tp2_pct",        0.05, 0.30),
        # Position sizing
        "core_pct":         trial.suggest_float("core_pct",       0.30, 0.70),
        "add_pct":          0.10,  # fixed
        # Entry gate
        "entry_min_regime": trial.suggest_int("entry_min_regime", 20,  60),
        "add_min_regime":   50,   # fixed
        # Cooldown
        "cooldown_bars":    trial.suggest_int("cooldown_bars",     2,   8),
    }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(
    df: pd.DataFrame,
    spy_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None,
    config: Optional[BacktestConfig] = None,
    n_wf_windows: int = 3,
    use_full_data: bool = True,
):
    """
    Factory that returns an Optuna objective callable.

    Parameters
    ----------
    df : OHLCV stock data
    spy_df : SPY data (aligned)
    sector_df : Sector ETF data (aligned)
    config : BacktestConfig defaults
    n_wf_windows : number of walk-forward windows for pruning
    use_full_data : if False, only uses 50% of data (Stage A coarse)
    """
    import pandas as pd

    # Subsample for Stage A
    if not use_full_data:
        n = len(df)
        half = n // 2
        # Use the LATER half (more recent, more relevant)
        df       = df.iloc[half:].copy()
        if spy_df is not None:
            spy_df   = spy_df.iloc[half:].copy()
        if sector_df is not None:
            sector_df = sector_df.iloc[half:].copy()

    if config is None:
        config = BacktestConfig()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)

        # Validate TP2 > TP1
        if params["tp2_rr"] <= params["tp1_rr"]:
            raise optuna.exceptions.TrialPruned()

        # Validate EMA ordering is at least somewhat sensible
        if params["ema_pulse"] >= params["ema_fast"]:
            raise optuna.exceptions.TrialPruned()
        if params["ema_fast"] >= params["ema_slow"]:
            raise optuna.exceptions.TrialPruned()

        # Validate Ichimoku: tenkan < kijun < spanB
        if params["tenkan_len"] >= params["kijun_len"]:
            raise optuna.exceptions.TrialPruned()
        if params["kijun_len"] >= params["spanb_len"]:
            raise optuna.exceptions.TrialPruned()

        try:
            windows = walk_forward_windows(df, n_windows=n_wf_windows,
                                           is_fraction=0.70, anchored=False)

            window_scores = []
            for w_idx, (is_df, oos_df) in enumerate(windows):
                # Slice reference data to match IS window
                spy_w = None
                sec_w = None
                if spy_df is not None:
                    spy_w = spy_df.reindex(is_df.index, method="ffill")
                if sector_df is not None:
                    sec_w = sector_df.reindex(is_df.index, method="ffill")

                metrics = backtest(is_df, params, spy_df=spy_w,
                                   sector_df=sec_w, config=config)

                # Skip windows with too few trades
                if metrics["n_trades"] < 3:
                    score = -5.0
                else:
                    score = score_metrics(metrics)

                window_scores.append(score)

                # Optuna pruning: report intermediate value after each window
                trial.report(float(np.mean(window_scores)), step=w_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # Final score: mean across windows
            final_score = float(np.mean(window_scores))

            # Log n_trades and max_drawdown for analysis
            trial.set_user_attr("window_scores", window_scores)
            trial.set_user_attr("n_wf_windows",  n_wf_windows)

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            logger.debug("Trial %d failed: %s", trial.number, exc)
            return -10.0

        return final_score

    return objective


# ---------------------------------------------------------------------------
# Full IS/OOS objective (Stage B)
# ---------------------------------------------------------------------------

def make_full_objective(
    df: pd.DataFrame,
    spy_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None,
    config: Optional[BacktestConfig] = None,
    n_wf_windows: int = 3,
    is_fraction: float = 0.70,
):
    """
    Stage B objective: uses full data, reports IS metrics + OOS score.
    Returns OOS score as primary metric.
    """
    import pandas as pd

    if config is None:
        config = BacktestConfig()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)

        if params["tp2_rr"] <= params["tp1_rr"]:
            raise optuna.exceptions.TrialPruned()
        if params["ema_pulse"] >= params["ema_fast"]:
            raise optuna.exceptions.TrialPruned()
        if params["ema_fast"] >= params["ema_slow"]:
            raise optuna.exceptions.TrialPruned()
        if params["tenkan_len"] >= params["kijun_len"]:
            raise optuna.exceptions.TrialPruned()
        if params["kijun_len"] >= params["spanb_len"]:
            raise optuna.exceptions.TrialPruned()

        try:
            windows = walk_forward_windows(df, n_windows=n_wf_windows,
                                           is_fraction=is_fraction, anchored=False)

            is_scores  = []
            oos_scores = []

            for w_idx, (is_df, oos_df) in enumerate(windows):
                spy_is = spy_oos = None
                sec_is = sec_oos = None

                if spy_df is not None:
                    spy_is  = spy_df.reindex(is_df.index,  method="ffill")
                    spy_oos = spy_df.reindex(oos_df.index, method="ffill")
                if sector_df is not None:
                    sec_is  = sector_df.reindex(is_df.index,  method="ffill")
                    sec_oos = sector_df.reindex(oos_df.index, method="ffill")

                # IS
                is_metrics = backtest(is_df,  params, spy_is,  sec_is,  config)
                # OOS
                oos_metrics = backtest(oos_df, params, spy_oos, sec_oos, config)

                is_score  = score_metrics(is_metrics)
                oos_score = score_metrics(oos_metrics)

                # Overfit flag: IS CAGR > 200%
                if is_metrics.get("CAGR", 0) > 2.0:
                    oos_score -= (is_metrics["CAGR"] - 2.0) * 2.0

                is_scores.append(is_score)
                oos_scores.append(oos_score)

                trial.report(float(np.mean(oos_scores)), step=w_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            mean_is  = float(np.mean(is_scores))
            mean_oos = float(np.mean(oos_scores))
            degradation = mean_oos / mean_is if abs(mean_is) > 1e-6 else 0.0

            trial.set_user_attr("is_score",    mean_is)
            trial.set_user_attr("oos_score",   mean_oos)
            trial.set_user_attr("degradation", degradation)
            trial.set_user_attr("overfit_flag", bool(mean_is > mean_oos * 2 or mean_is > 2.0))

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            logger.debug("Full trial %d failed: %s", trial.number, exc)
            return -10.0

        return mean_oos

    return objective


# ---------------------------------------------------------------------------
# Import pandas (needed inside factory functions)
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402  (must be after the factory defs to avoid circular)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import optuna
    from data_loader import load_yfinance, align_to_stock

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df     = load_yfinance("XOM", years=2, interval="1h")
    spy_df = load_yfinance("SPY", years=2, interval="1h")
    sec_df = load_yfinance("XLE", years=2, interval="1h")

    spy_a = align_to_stock(df, spy_df)
    sec_a = align_to_stock(df, sec_df)

    obj = make_objective(df, spy_a, sec_a, use_full_data=True, n_wf_windows=2)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                                    n_warmup_steps=1))
    study.optimize(obj, n_trials=5, show_progress_bar=True)
    print("Best score:", study.best_value)
    print("Best params:", study.best_params)
