"""
run_all.py — WOLF x SHADOW Optimization Pipeline
=================================================
CLI entry point that ties the entire optimization pipeline together.

Usage examples
--------------
# Download data, run full pipeline on 5 tickers (5 years):
python run_all.py --tickers XOM DVN COP EQNR.OL BOL.ST --years 5

# Use local CSV files, specify tickers:
python run_all.py --csv-dir ./data/ --tickers XOM DVN

# Quick test run (fewer trials, shorter history):
python run_all.py --tickers XOM --years 2 --stage-a-trials 20 --stage-b-trials 10 --n-jobs 1

# Resume from existing SQLite database:
python run_all.py --tickers XOM --years 5 --resume

# Skip Stage A/B, only run walk-forward on best SQLite results:
python run_all.py --tickers XOM --wf-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

np.random.seed(42)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Silence noisy third-party loggers
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("joblib").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WOLF x SHADOW v2 — Optuna Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data
    data_grp = p.add_argument_group("Data")
    data_grp.add_argument(
        "--tickers", nargs="+", required=True,
        help="List of ticker symbols, e.g. XOM DVN COP EQNR.OL BOL.ST",
    )
    data_grp.add_argument(
        "--csv-dir", type=str, default=None,
        help="Directory containing OHLCV CSV files (auto-matched by ticker name)",
    )
    data_grp.add_argument(
        "--years", type=int, default=5,
        help="Lookback period in years for yfinance download (default: 5)",
    )
    data_grp.add_argument(
        "--interval", type=str, default="1h",
        choices=["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
        help="Data frequency (default: 1h)",
    )
    data_grp.add_argument(
        "--force-download", action="store_true",
        help="Force re-download from yfinance even if cached",
    )

    # Optimization
    opt_grp = p.add_argument_group("Optimization")
    opt_grp.add_argument(
        "--stage-a-trials", type=int, default=150,
        help="Number of trials for Stage A coarse search (default: 150)",
    )
    opt_grp.add_argument(
        "--stage-b-trials", type=int, default=75,
        help="Trials per top region in Stage B refinement (default: 75)",
    )
    opt_grp.add_argument(
        "--top-n-regions", type=int, default=5,
        help="Top-N regions from Stage A to refine in Stage B (default: 5)",
    )
    opt_grp.add_argument(
        "--wf-windows", type=int, default=3,
        help="Number of walk-forward windows (minimum 3, default: 3)",
    )
    opt_grp.add_argument(
        "--is-fraction", type=float, default=0.70,
        help="In-sample fraction per walk-forward window (default: 0.70)",
    )
    opt_grp.add_argument(
        "--anchored-wf", action="store_true",
        help="Use anchored (expanding) walk-forward instead of rolling",
    )
    opt_grp.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Parallel workers (-1 = all CPU cores - 1, default: -1)",
    )
    opt_grp.add_argument(
        "--resume", action="store_true",
        help="Resume from existing SQLite study (auto-detected)",
    )
    opt_grp.add_argument(
        "--wf-only", action="store_true",
        help="Skip Stage A/B; only run walk-forward on existing best params",
    )

    # Costs
    cost_grp = p.add_argument_group("Transaction Costs")
    cost_grp.add_argument(
        "--commission", type=float, default=0.05,
        help="Commission percent per side (default: 0.05%%)",
    )
    cost_grp.add_argument(
        "--slippage", type=float, default=0.10,
        help="Slippage percent per fill (default: 0.10%%)",
    )
    cost_grp.add_argument(
        "--initial-capital", type=float, default=100_000.0,
        help="Initial capital in base currency (default: 100000)",
    )

    # Output
    out_grp = p.add_argument_group("Output")
    out_grp.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).parent / "output"),
        help="Directory for results, plots, and SQLite DB",
    )
    out_grp.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging",
    )

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    _setup_logging(args.verbose)

    logger.info("=" * 65)
    logger.info("  WOLF x SHADOW v2 — Optuna Optimization Pipeline")
    logger.info("=" * 65)
    logger.info("Tickers : %s", args.tickers)
    logger.info("Years   : %d  |  Interval: %s", args.years, args.interval)
    logger.info("Stage A : %d trials  |  Stage B: %d trials/region",
                args.stage_a_trials, args.stage_b_trials)
    logger.info("WF      : %d windows (%.0f%% IS)", args.wf_windows, args.is_fraction * 100)
    logger.info("Output  : %s", args.output_dir)
    logger.info("=" * 65)

    t_start = time.time()

    # ----------------------------------------------------------------
    # Imports (here to avoid slowing --help)
    # ----------------------------------------------------------------
    from backtest_engine import BacktestConfig
    from runner import run_optimization, run_final_walkforward
    from report import generate_all_reports

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BacktestConfig(
        initial_capital=args.initial_capital,
        commission_pct=args.commission,
        slippage_pct=args.slippage,
    )

    # ----------------------------------------------------------------
    # Stage A + B (unless --wf-only)
    # ----------------------------------------------------------------
    if not args.wf_only:
        logger.info("\n[PIPELINE] Running Stage A + Stage B optimization ...")

        optimization_results = run_optimization(
            tickers=args.tickers,
            csv_dir=args.csv_dir,
            years=args.years,
            interval=args.interval,
            output_dir=args.output_dir,
            stage_a_trials=args.stage_a_trials,
            stage_b_trials=args.stage_b_trials,
            top_n_regions=args.top_n_regions,
            n_wf_windows=args.wf_windows,
            n_jobs=args.n_jobs,
            config=config,
            force_download=args.force_download,
        )

    else:
        # Load existing results from SQLite
        logger.info("[PIPELINE] --wf-only: loading existing params from SQLite ...")
        optimization_results = _load_existing_results(
            tickers=args.tickers,
            output_dir=args.output_dir,
            csv_dir=args.csv_dir,
            years=args.years,
            interval=args.interval,
            config=config,
        )

    # ----------------------------------------------------------------
    # Walk-forward validation
    # ----------------------------------------------------------------
    logger.info("\n[PIPELINE] Running final walk-forward validation ...")

    wf_results = run_final_walkforward(
        results=optimization_results,
        n_windows=args.wf_windows,
        is_fraction=args.is_fraction,
        config=config,
    )

    # ----------------------------------------------------------------
    # Reports
    # ----------------------------------------------------------------
    logger.info("\n[PIPELINE] Generating reports ...")

    report_paths = generate_all_reports(
        optimization_results=optimization_results,
        wf_results=wf_results,
        output_dir=args.output_dir,
    )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 65)
    logger.info("  PIPELINE COMPLETE  (%.1f min)", elapsed / 60)
    logger.info("=" * 65)
    logger.info("Results saved to: %s", args.output_dir)
    logger.info("  CSV report    : %s", report_paths.get("csv", "N/A"))
    logger.info("  MD report     : %s", report_paths.get("markdown", "N/A"))
    logger.info("  Equity curves : %s", report_paths.get("equity_curves", []))
    logger.info("=" * 65)

    # Print quick summary to stdout
    print("\n" + "=" * 55)
    print(f"{'TICKER':<12} {'OOS CAGR':>10} {'OOS MDD':>10} {'DEGRAD':>8} {'OVERFIT':>8}")
    print("-" * 55)
    for ticker, wf_list in wf_results.items():
        if not wf_list:
            continue
        best = wf_list[0]
        o    = best["oos_metrics_avg"]
        warn = "⚠" if best.get("overfit_warning") else "OK"
        print(
            f"{ticker:<12} "
            f"{o['CAGR']:>10.1%} "
            f"{o['max_drawdown']:>10.1%} "
            f"{best.get('degradation_ratio', 0):>8.2f} "
            f"{warn:>8}"
        )
    print("=" * 55)

    return 0


# ---------------------------------------------------------------------------
# --wf-only loader
# ---------------------------------------------------------------------------

def _load_existing_results(
    tickers: list[str],
    output_dir: str,
    csv_dir,
    years: int,
    interval: str,
    config,
) -> dict:
    """
    Load top params from existing SQLite studies for --wf-only mode.
    """
    import optuna
    from data_loader import load_data, align_to_stock
    from runner import _trial_to_params, _get_sector_etf

    db_path      = str(Path(output_dir) / "wolf_shadow_optuna.db")
    storage_url  = f"sqlite:///{db_path}"
    results      = {}

    for ticker in tickers:
        study_name_b = f"{ticker}_stage_b"
        study_name_a = f"{ticker}_stage_a"

        # Try Stage B first, then Stage A
        for sname in (study_name_b, study_name_a):
            try:
                study = optuna.load_study(study_name=sname, storage=storage_url)
                completed = [t for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE]
                if completed:
                    completed.sort(key=lambda t: t.value or -999, reverse=True)
                    top_params = [_trial_to_params(t) for t in completed[:5]]
                    break
            except Exception:
                top_params = []
                continue
        else:
            logger.warning("[%s] No existing study found — using defaults", ticker)
            from indicators import DEFAULT_PARAMS
            top_params = [DEFAULT_PARAMS]

        # Load data
        try:
            df = load_data(ticker, csv_dir=csv_dir, years=years, interval=interval)
        except Exception:
            continue

        spy_df = sector_df = None
        try:
            spy_raw = load_data("SPY", csv_dir=csv_dir, years=years, interval=interval)
            spy_df  = align_to_stock(df, spy_raw)
        except Exception:
            pass

        sector_sym = _get_sector_etf(ticker)
        try:
            sec_raw   = load_data(sector_sym, csv_dir=csv_dir, years=years, interval=interval)
            sector_df = align_to_stock(df, sec_raw)
        except Exception:
            pass

        results[ticker] = {
            "ticker":      ticker,
            "top_params":  top_params,
            "df":          df,
            "spy_df":      spy_df,
            "sector_df":   sector_df,
        }

    return results


if __name__ == "__main__":
    sys.exit(main())
