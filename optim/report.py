"""
report.py — WOLF x SHADOW Optimization Pipeline
================================================
Generate:
  1. CSV: all tested parameter sets with IS and OOS metrics
  2. Markdown summary: top 5 param sets, IS/OOS table, overfit warnings, risk section
  3. Equity curve plots (matplotlib, saved to output/)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # headless rendering
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _MPL = True
except ImportError:
    _MPL = False
    logger.warning("matplotlib not available — equity curve plots will be skipped")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(
    wf_results: dict,  # {ticker: [wf_result, ...]}
    output_dir: str | Path,
    filename: str = "optimization_results.csv",
) -> Path:
    """
    Export all walk-forward results to a flat CSV file.

    Columns: ticker, rank, param_*, is_cagr, is_mdd, is_sharpe, is_n_trades,
             oos_cagr, oos_mdd, oos_sharpe, oos_n_trades, oos_winrate,
             oos_positive_pct, degradation_ratio, overfit_warning, oos_score
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    rows = []
    for ticker, wf_list in wf_results.items():
        for rank, wf in enumerate(wf_list, start=1):
            params = wf.get("params", {})
            row = {
                "ticker":             ticker,
                "rank":               rank,
                "oos_score":          round(wf.get("oos_score", 0.0), 4),
                "is_score":           round(wf.get("is_score",  0.0), 4),
                "degradation_ratio":  round(wf.get("degradation_ratio", 0.0), 3),
                "oos_positive_pct":   round(wf.get("oos_positive_pct", 0.0), 3),
                "overfit_warning":    wf.get("overfit_warning", False),
                # IS avg
                "is_cagr":            round(wf["is_metrics_avg"].get("CAGR", 0), 4),
                "is_mdd":             round(wf["is_metrics_avg"].get("max_drawdown", 0), 4),
                "is_sharpe":          round(wf["is_metrics_avg"].get("sharpe", 0), 3),
                "is_n_trades":        round(wf["is_metrics_avg"].get("n_trades", 0), 1),
                "is_winrate":         round(wf["is_metrics_avg"].get("winrate", 0), 3),
                # OOS avg
                "oos_cagr":           round(wf["oos_metrics_avg"].get("CAGR", 0), 4),
                "oos_mdd":            round(wf["oos_metrics_avg"].get("max_drawdown", 0), 4),
                "oos_sharpe":         round(wf["oos_metrics_avg"].get("sharpe", 0), 3),
                "oos_n_trades":       round(wf["oos_metrics_avg"].get("n_trades", 0), 1),
                "oos_winrate":        round(wf["oos_metrics_avg"].get("winrate", 0), 3),
            }
            # Add all params
            for k, v in params.items():
                row[f"param_{k}"] = round(v, 4) if isinstance(v, float) else v
            rows.append(row)

    if not rows:
        logger.warning("No results to export to CSV")
        return out_path

    # Determine fieldnames in a deterministic order
    fixed_keys = [
        "ticker", "rank", "oos_score", "is_score", "degradation_ratio",
        "oos_positive_pct", "overfit_warning",
        "is_cagr", "is_mdd", "is_sharpe", "is_n_trades", "is_winrate",
        "oos_cagr", "oos_mdd", "oos_sharpe", "oos_n_trades", "oos_winrate",
    ]
    param_keys = sorted(k for k in rows[0] if k.startswith("param_"))
    fieldnames = fixed_keys + param_keys

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("CSV exported: %s (%d rows)", out_path, len(rows))
    return out_path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

_RISK_WARNING = """
## ⚠ Risk Warnings

1. **Past performance does not guarantee future results.** Backtest results are hypothetical.
2. **Survivorship bias**: The optimization only considers tickers you provide. Tickers that went 
   bankrupt or were delisted are not included.
3. **Look-ahead bias**: All indicator calculations use only information available at the time of 
   the bar. However, parameter selection itself creates a selection bias — always validate on 
   fresh OOS data before live trading.
4. **Overfitting**: Parameters with IS CAGR > 200% are flagged as likely overfit. Walk-forward 
   degradation ratio < 0.5 also indicates overfitting.
5. **Transaction costs**: Commission (0.05% per side) and slippage (0.1%) are included. Real 
   costs may be higher, especially for illiquid tickers.
6. **Liquidity risk**: The strategy assumes fills at model prices. Large positions may not fill 
   at these prices in practice.
7. **Regime change risk**: Optimized parameters may work poorly in market regimes not present 
   in the training data.
8. **This is not financial advice.** Always test manually on TradingView before live deployment.
"""


def _fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def _fmt_f(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}"


def generate_markdown(
    wf_results: dict,
    study_info: Optional[dict] = None,
    output_dir: str | Path = "output",
    filename: str = "optimization_report.md",
) -> Path:
    """
    Generate Markdown summary report.

    Parameters
    ----------
    wf_results : {ticker: [wf_result, ...]}
    study_info : optional dict with {ticker: {stage_a_best, stage_b_best, n_trials_a, n_trials_b}}
    output_dir : output directory
    filename : output filename

    Returns
    -------
    Path to generated file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    lines = []
    lines.append("# WOLF x SHADOW v2 — Optimization Report\n")
    lines.append(f"*Generated by the WOLF x SHADOW Optuna Pipeline*\n")
    lines.append("---\n")

    # --- Table of Contents ---
    lines.append("## Table of Contents\n")
    lines.append("1. [Executive Summary](#executive-summary)")
    lines.append("2. [Per-Ticker Results](#per-ticker-results)")
    lines.append("3. [IS vs OOS Comparison](#is-vs-oos-comparison)")
    lines.append("4. [Overfit Analysis](#overfit-analysis)")
    lines.append("5. [Risk Warnings](#risk-warnings)")
    lines.append("\n---\n")

    # --- Executive Summary ---
    lines.append("## Executive Summary\n")

    all_oos = []
    for ticker, wf_list in wf_results.items():
        for wf in wf_list[:1]:  # best per ticker
            all_oos.append(wf.get("oos_score", 0))

    if all_oos:
        lines.append(f"- **Tickers optimized**: {', '.join(wf_results.keys())}")
        lines.append(f"- **Best OOS score (across all tickers)**: {max(all_oos):.4f}")
        overfit_count = sum(
            1 for wf_list in wf_results.values()
            for wf in wf_list[:1]
            if wf.get("overfit_warning", False)
        )
        lines.append(f"- **Tickers with overfit warning**: {overfit_count}/{len(wf_results)}")
        lines.append("")

    if study_info:
        lines.append("### Optimization Run Statistics\n")
        lines.append("| Ticker | Stage A Trials | Stage B Trials | Stage A Best | Stage B Best |")
        lines.append("|--------|---------------|----------------|-------------|-------------|")
        for ticker, info in study_info.items():
            lines.append(
                f"| {ticker} | {info.get('n_trials_a', '?')} | {info.get('n_trials_b', '?')} | "
                f"{info.get('stage_a_best', 0):.4f} | {info.get('stage_b_best', 0):.4f} |"
            )
        lines.append("")

    lines.append("---\n")

    # --- Per-Ticker Results ---
    lines.append("## Per-Ticker Results\n")

    for ticker, wf_list in wf_results.items():
        lines.append(f"### {ticker}\n")

        if not wf_list:
            lines.append("*No results available.*\n")
            continue

        lines.append("**Top 5 Parameter Sets** (sorted by OOS score)\n")
        lines.append(
            "| Rank | OOS Score | OOS CAGR | OOS MDD | OOS Sharpe | "
            "IS CAGR | IS MDD | Degrad. | Overfit |"
        )
        lines.append(
            "|------|----------|----------|---------|-----------|"
            "--------|--------|---------|---------|"
        )

        for rank, wf in enumerate(wf_list[:5], start=1):
            o = wf["oos_metrics_avg"]
            s = wf["is_metrics_avg"]
            warn = "⚠️ YES" if wf.get("overfit_warning") else "OK"
            lines.append(
                f"| {rank} | {wf.get('oos_score', 0):.4f} | "
                f"{_fmt_pct(o['CAGR'])} | {_fmt_pct(o['max_drawdown'])} | "
                f"{_fmt_f(o['sharpe'])} | "
                f"{_fmt_pct(s['CAGR'])} | {_fmt_pct(s['max_drawdown'])} | "
                f"{wf.get('degradation_ratio', 0):.2f} | {warn} |"
            )

        lines.append("")

        # Best param set detail
        best_wf = wf_list[0]
        params  = best_wf.get("params", {})
        if params:
            lines.append("**Best Parameter Set (Rank 1)**\n")
            lines.append("```")
            for k, v in sorted(params.items()):
                if isinstance(v, float):
                    lines.append(f"  {k:<25} = {v:.4f}")
                else:
                    lines.append(f"  {k:<25} = {v}")
            lines.append("```\n")

        # Walk-forward window detail
        lines.append("**Walk-Forward Windows (Best Params)**\n")
        lines.append(
            "| Window | IS Period | OOS Period | IS CAGR | OOS CAGR | "
            "OOS MDD | OOS Trades |"
        )
        lines.append(
            "|--------|-----------|------------|---------|----------|"
            "---------|------------|"
        )
        for w in best_wf.get("windows", []):
            lines.append(
                f"| {w['window']} | "
                f"{str(w['is_start'])[:10]}–{str(w['is_end'])[:10]} | "
                f"{str(w['oos_start'])[:10]}–{str(w['oos_end'])[:10]} | "
                f"{_fmt_pct(w['is_cagr'])} | {_fmt_pct(w['oos_cagr'])} | "
                f"{_fmt_pct(w['oos_mdd'])} | {w['oos_n_trades']} |"
            )
        lines.append("")

    lines.append("---\n")

    # --- IS vs OOS Comparison ---
    lines.append("## IS vs OOS Comparison\n")
    lines.append("*(Best parameter set per ticker)*\n")
    lines.append(
        "| Ticker | IS CAGR | OOS CAGR | IS MDD | OOS MDD | "
        "IS Sharpe | OOS Sharpe | OOS Win% | Degrad. |"
    )
    lines.append(
        "|--------|---------|----------|--------|---------|"
        "----------|-----------|---------|---------|"
    )

    for ticker, wf_list in wf_results.items():
        if not wf_list:
            continue
        best = wf_list[0]
        o = best["oos_metrics_avg"]
        s = best["is_metrics_avg"]
        lines.append(
            f"| {ticker} | {_fmt_pct(s['CAGR'])} | {_fmt_pct(o['CAGR'])} | "
            f"{_fmt_pct(s['max_drawdown'])} | {_fmt_pct(o['max_drawdown'])} | "
            f"{_fmt_f(s['sharpe'])} | {_fmt_f(o['sharpe'])} | "
            f"{_fmt_pct(o['winrate'])} | {best.get('degradation_ratio', 0):.2f} |"
        )

    lines.append("\n---\n")

    # --- Overfit Analysis ---
    lines.append("## Overfit Analysis\n")
    lines.append(
        "**Degradation ratio** = OOS score / IS score. Values < 0.5 flag potential overfit.\n"
    )
    lines.append(
        "**IS CAGR > 200%** on in-sample data is treated as a strong overfitting signal.\n"
    )

    any_overfit = False
    for ticker, wf_list in wf_results.items():
        for rank, wf in enumerate(wf_list[:5], start=1):
            if wf.get("overfit_warning"):
                any_overfit = True
                lines.append(
                    f"- ⚠️ **{ticker} Rank {rank}**: "
                    f"IS CAGR={_fmt_pct(wf['is_metrics_avg']['CAGR'])}, "
                    f"OOS CAGR={_fmt_pct(wf['oos_metrics_avg']['CAGR'])}, "
                    f"Degradation={wf.get('degradation_ratio', 0):.2f}"
                )

    if not any_overfit:
        lines.append("✅ No overfit warnings detected in top-5 results.\n")
    else:
        lines.append(
            "\n> **Action**: Avoid deploying flagged parameter sets. "
            "Use Rank 1 only if degradation ratio ≥ 0.5 and OOS CAGR > 0.\n"
        )

    lines.append("\n---\n")

    # --- Risk Warnings ---
    lines.append(_RISK_WARNING)

    report_text = "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("Markdown report written: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Equity curve plots
# ---------------------------------------------------------------------------

def plot_equity_curves(
    wf_results: dict,
    output_dir: str | Path = "output",
    dpi: int = 120,
) -> list[Path]:
    """
    Generate equity curve plots for the best parameter set per ticker.
    One subplot per walk-forward window (IS + OOS side by side).

    Returns list of saved plot paths.
    """
    if not _MPL:
        logger.warning("matplotlib unavailable — skipping equity curve plots")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for ticker, wf_list in wf_results.items():
        if not wf_list:
            continue

        best_wf  = wf_list[0]
        windows  = best_wf.get("windows", [])
        n_win    = len(windows)

        if n_win == 0:
            continue

        fig, axes = plt.subplots(
            1, n_win,
            figsize=(6 * n_win, 4),
            facecolor="#0d0d1a",
        )
        if n_win == 1:
            axes = [axes]

        fig.suptitle(
            f"{ticker} — WOLF x SHADOW v2 Walk-Forward Equity Curves",
            color="white", fontsize=13, y=1.02,
        )

        for ax, w in zip(axes, windows):
            is_eq  = w.get("is_equity")
            oos_eq = w.get("oos_equity")

            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="gray")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

            if is_eq is not None and len(is_eq) > 0:
                norm_is = is_eq / is_eq.iloc[0] * 100
                ax.plot(norm_is.index, norm_is.values,
                        color="#00BCD4", linewidth=1.2, label="IS", alpha=0.9)

            if oos_eq is not None and len(oos_eq) > 0:
                norm_oos = oos_eq / oos_eq.iloc[0] * 100
                ax.plot(norm_oos.index, norm_oos.values,
                        color="#FF9800", linewidth=1.5, label="OOS", alpha=0.9)

            ax.axhline(100, color="gray", linewidth=0.5, linestyle="--")
            ax.set_title(f"Window {w['window']}", color="white", fontsize=10)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
            ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white",
                      framealpha=0.8, loc="upper left")
            ax.set_xlabel("")
            ax.xaxis.set_tick_params(rotation=25, labelsize=7)
            ax.yaxis.set_tick_params(labelsize=8)

            # Add CAGR annotation
            cagr_is  = w.get("is_cagr",  0)
            cagr_oos = w.get("oos_cagr", 0)
            mdd_oos  = w.get("oos_mdd",  0)
            ax.text(
                0.97, 0.05,
                f"IS CAGR: {cagr_is:.1%}\nOOS CAGR: {cagr_oos:.1%}\nOOS MDD: {mdd_oos:.1%}",
                transform=ax.transAxes, fontsize=7, color="white",
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", alpha=0.8),
            )

        plt.tight_layout()
        out_path = output_dir / f"{ticker}_equity_curves.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Equity curve saved: %s", out_path)
        saved.append(out_path)

    return saved


def plot_param_importance(
    study,
    ticker: str,
    output_dir: str | Path = "output",
    dpi: int = 120,
) -> Optional[Path]:
    """
    Plot Optuna parameter importance for a study (Stage B).
    """
    if not _MPL:
        return None

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as exc:
        logger.warning("[%s] Could not compute param importance: %s", ticker, exc)
        return None

    output_dir = Path(output_dir)
    out_path   = output_dir / f"{ticker}_param_importance.png"

    params = list(importances.keys())[:15]
    values = [importances[p] for p in params]

    fig, ax = plt.subplots(figsize=(8, max(4, len(params) * 0.4)),
                           facecolor="#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    bars = ax.barh(params[::-1], values[::-1], color="#00BCD4", alpha=0.85)
    ax.set_xlabel("Importance", color="white")
    ax.set_title(f"{ticker} — Parameter Importance (Stage B)",
                 color="white", fontsize=11)
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Param importance saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# All-in-one report generator
# ---------------------------------------------------------------------------

def generate_all_reports(
    optimization_results: dict,
    wf_results: dict,
    output_dir: str | Path = "output",
) -> dict:
    """
    Generate all report artefacts.

    Parameters
    ----------
    optimization_results : {ticker: result from runner.optimize_ticker}
    wf_results : {ticker: [wf_result, ...] from runner.run_final_walkforward}
    output_dir : output directory

    Returns
    -------
    dict with paths: csv, markdown, equity_curves, param_importance
    """
    output_dir = Path(output_dir)

    # CSV
    csv_path = export_csv(wf_results, output_dir)

    # Study info for Markdown
    study_info = {}
    for ticker, res in optimization_results.items():
        sa = res.get("stage_a_study")
        sb = res.get("stage_b_study")
        completed_a = [t for t in (sa.trials if sa else [])
                       if t.state.name == "COMPLETE"]
        completed_b = [t for t in (sb.trials if sb else [])
                       if t.state.name == "COMPLETE"]
        study_info[ticker] = {
            "n_trials_a":   len(completed_a),
            "n_trials_b":   len(completed_b),
            "stage_a_best": sa.best_value if sa and sa.best_value is not None else 0.0,
            "stage_b_best": sb.best_value if sb and sb.best_value is not None else 0.0,
        }

    md_path       = generate_markdown(wf_results, study_info, output_dir)
    eq_paths      = plot_equity_curves(wf_results, output_dir)
    imp_paths     = []

    for ticker, res in optimization_results.items():
        sb = res.get("stage_b_study")
        if sb:
            p = plot_param_importance(sb, ticker, output_dir)
            if p:
                imp_paths.append(p)

    return {
        "csv":              csv_path,
        "markdown":         md_path,
        "equity_curves":    eq_paths,
        "param_importance": imp_paths,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick smoke-test with synthetic data
    from indicators import DEFAULT_PARAMS

    dummy_wf = {
        "XOM": [{
            "oos_score":          0.32,
            "is_score":           0.48,
            "degradation_ratio":  0.67,
            "oos_positive_pct":   1.00,
            "overfit_warning":    False,
            "is_metrics_avg":  {"CAGR": 0.35, "max_drawdown": -0.18, "sharpe": 1.2, "n_trades": 22, "winrate": 0.59},
            "oos_metrics_avg": {"CAGR": 0.22, "max_drawdown": -0.14, "sharpe": 0.9, "n_trades": 8,  "winrate": 0.55},
            "windows": [],
            "params": DEFAULT_PARAMS,
        }]
    }
    csv_p = export_csv(dummy_wf, "/tmp/wolf_test")
    md_p  = generate_markdown(dummy_wf, output_dir="/tmp/wolf_test")
    print("CSV:", csv_p)
    print("MD:",  md_p)
