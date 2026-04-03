"""
runner.py — WOLF x SHADOW Optimization Pipeline
================================================
Two-stage Optuna optimization runner.

Stage A (coarse):
  - 150 trials, TPESampler(seed=42)
  - MedianPruner(n_startup=10, n_warmup=1)
  - 50% of data (recent half) for speed
  - Purpose: identify promising parameter regions

Stage B (refinement):
  - 75 trials per top-5 region from Stage A
  - Full data
  - MedianPruner(n_startup=5)
  - Seeded with best params from Stage A (warm-start)

Multi-ticker: runs optimization per ticker (or pooled).
Parallel workers via joblib (default n_cores - 1).
Saves Optuna study to SQLite for resume capability.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
from joblib import Parallel, delayed

from data_loader import load_data, align_to_stock, load_market_data
from backtest_engine import BacktestConfig
from objective import make_objective, make_full_objective, suggest_params
from walkforward import run_walk_forward_batch

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Default market proxies per ticker
# ---------------------------------------------------------------------------
_SECTOR_MAP: dict[str, str] = {
    # Energy
    "XOM": "XLE",  "CVX": "XLE",  "COP": "XLE",  "DVN": "XLE",
    "EOG": "XLE",  "MRO": "XLE",  "HAL": "XLE",  "SLB": "XLE",
    "EQNR.OL": "XLE",
    # Materials
    "BOL.ST": "XLB",
    # Default
    "_default": "XLK",
}


def _get_sector_etf(ticker: str) -> str:
    return _SECTOR_MAP.get(ticker.upper(), _SECTOR_MAP["_default"])


# ---------------------------------------------------------------------------
# Single-ticker optimization run
# ---------------------------------------------------------------------------

def optimize_ticker(
    ticker: str,
    df,
    spy_df,
    sector_df,
    db_path: str,
    stage_a_trials: int = 150,
    stage_b_trials: int = 75,
    top_n_regions: int = 5,
    n_wf_windows: int = 3,
    n_jobs: int = 1,
    config: Optional[BacktestConfig] = None,
) -> dict:
    """
    Run Stage A + Stage B for a single ticker.

    Returns
    -------
    dict with keys: ticker, stage_a_study, stage_b_study, top_params
    """
    if config is None:
        config = BacktestConfig()

    study_name_a = f"{ticker}_stage_a"
    study_name_b = f"{ticker}_stage_b"
    storage_url  = f"sqlite:///{db_path}"

    # ----------------------------------------------------------------
    # STAGE A — coarse, 50% data
    # ----------------------------------------------------------------
    logger.info("[%s] Stage A: %d trials (50%% data)", ticker, stage_a_trials)
    t0 = time.time()

    sampler_a = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    pruner_a  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)

    study_a = optuna.create_study(
        study_name=study_name_a,
        storage=storage_url,
        direction="maximize",
        sampler=sampler_a,
        pruner=pruner_a,
        load_if_exists=True,
    )

    # How many more trials we need
    existing_a = len([t for t in study_a.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_a = max(0, stage_a_trials - existing_a)

    if remaining_a > 0:
        obj_a = make_objective(
            df=df, spy_df=spy_df, sector_df=sector_df,
            config=config,
            n_wf_windows=n_wf_windows,
            use_full_data=False,   # 50% data
        )
        study_a.optimize(
            obj_a,
            n_trials=remaining_a,
            n_jobs=n_jobs,
            show_progress_bar=(logger.level <= logging.INFO),
            callbacks=[_log_callback(ticker, "A")],
        )

    elapsed_a = time.time() - t0
    logger.info("[%s] Stage A done in %.1f s. Best score: %.4f",
                ticker, elapsed_a, study_a.best_value)

    # ----------------------------------------------------------------
    # Identify top-N regions from Stage A
    # ----------------------------------------------------------------
    completed_a = [t for t in study_a.trials
                   if t.state == optuna.trial.TrialState.COMPLETE]
    completed_a.sort(key=lambda t: t.value or -999, reverse=True)
    top_a_trials = completed_a[:top_n_regions]
    top_a_params = [_trial_to_params(t) for t in top_a_trials]

    logger.info("[%s] Top-%d Stage A params identified", ticker, len(top_a_params))

    # ----------------------------------------------------------------
    # STAGE B — refinement, full data, warm-started from top regions
    # ----------------------------------------------------------------
    logger.info("[%s] Stage B: %d trials per region x %d regions (full data)",
                ticker, stage_b_trials, len(top_a_params))
    t1 = time.time()

    sampler_b = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=5,
        consider_endpoints=True,
    )
    pruner_b = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study_b = optuna.create_study(
        study_name=study_name_b,
        storage=storage_url,
        direction="maximize",
        sampler=sampler_b,
        pruner=pruner_b,
        load_if_exists=True,
    )

    # Enqueue top params from Stage A as warm start
    for p in top_a_params:
        try:
            study_b.enqueue_trial(p)
        except Exception:
            pass  # may fail if already queued

    existing_b = len([t for t in study_b.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_b = max(0, stage_b_trials * len(top_a_params) - existing_b)

    if remaining_b > 0:
        obj_b = make_full_objective(
            df=df, spy_df=spy_df, sector_df=sector_df,
            config=config,
            n_wf_windows=n_wf_windows,
            is_fraction=0.70,
        )
        study_b.optimize(
            obj_b,
            n_trials=remaining_b,
            n_jobs=n_jobs,
            show_progress_bar=(logger.level <= logging.INFO),
            callbacks=[_log_callback(ticker, "B")],
        )

    elapsed_b = time.time() - t1
    logger.info("[%s] Stage B done in %.1f s. Best OOS score: %.4f",
                ticker, elapsed_b, study_b.best_value)

    # ----------------------------------------------------------------
    # Collect final top-5 parameter sets (by OOS score)
    # ----------------------------------------------------------------
    completed_b = [t for t in study_b.trials
                   if t.state == optuna.trial.TrialState.COMPLETE]
    completed_b.sort(key=lambda t: t.value or -999, reverse=True)
    top_b_params = [_trial_to_params(t) for t in completed_b[:5]]

    return {
        "ticker":         ticker,
        "stage_a_study":  study_a,
        "stage_b_study":  study_b,
        "top_params":     top_b_params,
        "top_a_params":   top_a_params,
        "elapsed_stage_a": elapsed_a,
        "elapsed_stage_b": elapsed_b,
    }


# ---------------------------------------------------------------------------
# Multi-ticker runner
# ---------------------------------------------------------------------------

def run_optimization(
    tickers: list[str],
    csv_dir: Optional[str] = None,
    years: int = 5,
    interval: str = "1h",
    output_dir: str = "/home/user/workspace/optim/output",
    stage_a_trials: int = 150,
    stage_b_trials: int = 75,
    top_n_regions: int = 5,
    n_wf_windows: int = 3,
    n_jobs: int = -1,
    config: Optional[BacktestConfig] = None,
    force_download: bool = False,
) -> dict:
    """
    Run full two-stage optimization for all tickers.

    Parameters
    ----------
    tickers : list of ticker symbols
    csv_dir : directory containing CSV files (optional)
    years : lookback period in years for yfinance download
    interval : data frequency
    output_dir : directory to save SQLite DB, results
    stage_a_trials : trials for Stage A per ticker
    stage_b_trials : trials for Stage B per top region per ticker
    top_n_regions : top N regions from Stage A to refine in Stage B
    n_wf_windows : walk-forward windows
    n_jobs : parallel workers (-1 = all cores - 1)
    config : BacktestConfig
    force_download : force yfinance re-download

    Returns
    -------
    dict: {ticker: optimization_result}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(output_dir / "wolf_shadow_optuna.db")

    if n_jobs == -1:
        import multiprocessing
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

    logger.info("Starting optimization for %d tickers: %s", len(tickers), tickers)
    logger.info("Stage A: %d trials | Stage B: %d trials per region",
                stage_a_trials, stage_b_trials)
    logger.info("Workers: %d | WF windows: %d", n_jobs, n_wf_windows)

    if config is None:
        config = BacktestConfig()

    results: dict = {}
    t_total = time.time()

    for ticker in tickers:
        logger.info("\n" + "=" * 60)
        logger.info("Processing ticker: %s", ticker)
        logger.info("=" * 60)

        # Load stock data
        try:
            df = load_data(ticker, csv_dir=csv_dir, years=years,
                           interval=interval, force_download=force_download)
        except Exception as exc:
            logger.error("[%s] Failed to load data: %s — skipping", ticker, exc)
            continue

        # Load SPY
        try:
            spy_raw = load_data("SPY", csv_dir=csv_dir, years=years,
                                interval=interval, force_download=force_download)
            spy_df  = align_to_stock(df, spy_raw)
        except Exception as exc:
            logger.warning("[%s] Failed to load SPY: %s — market score = 0", ticker, exc)
            spy_df = None

        # Load sector ETF
        sector_sym = _get_sector_etf(ticker)
        try:
            sec_raw    = load_data(sector_sym, csv_dir=csv_dir, years=years,
                                   interval=interval, force_download=force_download)
            sector_df  = align_to_stock(df, sec_raw)
        except Exception as exc:
            logger.warning("[%s] Failed to load %s: %s — sector score = 0",
                           ticker, sector_sym, exc)
            sector_df = None

        # Run optimization
        try:
            result = optimize_ticker(
                ticker=ticker,
                df=df,
                spy_df=spy_df,
                sector_df=sector_df,
                db_path=db_path,
                stage_a_trials=stage_a_trials,
                stage_b_trials=stage_b_trials,
                top_n_regions=top_n_regions,
                n_wf_windows=n_wf_windows,
                n_jobs=n_jobs,
                config=config,
            )
            result["df"]        = df
            result["spy_df"]    = spy_df
            result["sector_df"] = sector_df
            results[ticker]     = result

        except Exception as exc:
            logger.error("[%s] Optimization failed: %s", ticker, exc, exc_info=True)

    elapsed_total = time.time() - t_total
    logger.info("\nTotal optimization time: %.1f s (%.1f min)",
                elapsed_total, elapsed_total / 60)

    return results


# ---------------------------------------------------------------------------
# Walk-forward runner for final top params
# ---------------------------------------------------------------------------

def run_final_walkforward(
    results: dict,
    n_windows: int = 3,
    is_fraction: float = 0.70,
    config: Optional[BacktestConfig] = None,
) -> dict:
    """
    Run walk-forward on the top parameter sets identified in Stage B.
    Returns {ticker: [wf_result, ...]}
    """
    wf_results: dict = {}

    for ticker, res in results.items():
        logger.info("[%s] Running walk-forward on top %d params ...",
                    ticker, len(res.get("top_params", [])))
        df        = res.get("df")
        spy_df    = res.get("spy_df")
        sector_df = res.get("sector_df")

        if df is None:
            continue

        wf_list = run_walk_forward_batch(
            df=df,
            params_list=res.get("top_params", []),
            spy_df=spy_df,
            sector_df=sector_df,
            n_windows=n_windows,
            is_fraction=is_fraction,
            config=config,
        )
        wf_results[ticker] = wf_list

    return wf_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trial_to_params(trial: optuna.Trial) -> dict:
    """Extract params dict from a completed trial, adding fixed params."""
    p = dict(trial.params)
    # Fill in fixed params that are not in the search space
    defaults = {
        "displacement": 26, "ema_macro": 200, "rsi_hot": 70,
        "atr_len": 14, "ob_lookback": 5, "add_pct": 0.10,
        "add_min_regime": 50,
    }
    for k, v in defaults.items():
        if k not in p:
            p[k] = v
    return p


def _log_callback(ticker: str, stage: str):
    """Optuna callback for progress logging."""
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.number % 25 == 0:
            best = study.best_value if study.best_value is not None else float("nan")
            logger.info("[%s] Stage %s | Trial %d | Best: %.4f",
                        ticker, stage, trial.number, best)
    return callback


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    results = run_optimization(
        tickers=["XOM"],
        years=2,
        interval="1h",
        output_dir="/home/user/workspace/optim/output",
        stage_a_trials=10,    # quick test
        stage_b_trials=5,
        n_wf_windows=2,
        n_jobs=1,
    )
    print("Optimization complete. Top params for XOM:")
    if "XOM" in results:
        for i, p in enumerate(results["XOM"]["top_params"]):
            print(f"  #{i+1}: {p}")
