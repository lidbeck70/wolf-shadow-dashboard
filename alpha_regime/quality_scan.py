"""
alpha_regime/quality_scan.py — Quality-köpsignaler för den schemalagda skanningen.

Quality-strategins regel (strategy_rules.QUALITY): "Alla fyra gates gröna =
BUY. Tre gröna = WATCH. Två eller färre = WAIT." — trend, värdering,
marknadscykel och bolagskvalitet. Samma analys som REGIME → Alpha Regime →
Quality & Contrarian (run_regime_analysis i quality-läge) körs här på
bolagen i Quality-listan, och verdiktet sparas med listan så larmet kan
säga till när ett bolag NYTT står i BUY.

Marknadscykeln mäts mot bolagets hemmamarknad (^OMX för Stockholm osv.),
annars S&P 500. Sentimentet hämtas inte — det påverkar inte verdiktet.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

TOP_N = 25                  # så många ur Quality-listan analyseras per körning
SIGNAL_LABELS = {"TREND": "Trend", "DISCOUNT": "Värdering", "CYCLE": "Cykel", "QUALITY": "Kvalitet"}

# Hemmamarknadens index för cykelgaten (samma val som Alpha Regime-fliken erbjuder)
BENCHMARK_BY_SUFFIX = {".ST": "^OMX", ".OL": "OBX.OL", ".CO": "^OMXC25", ".HE": "^OMXH25"}
DEFAULT_BENCHMARK = "SPY"


def benchmark_for(ticker: str) -> str:
    t = str(ticker or "").upper()
    for sfx, bm in BENCHMARK_BY_SUFFIX.items():
        if t.endswith(sfx):
            return bm
    return DEFAULT_BENCHMARK


def _default_analyze(ticker: str, benchmark: str, kap_badge: bool):
    from alpha_regime.engine import run_regime_analysis
    return run_regime_analysis(ticker, mode="quality", market_ticker=benchmark,
                               kap_badge=kap_badge, with_sentiment=False)


def scan(results: list, top: int = TOP_N, analyze: Optional[Callable] = None) -> dict:
    """{ticker: {verdict, passed, total, benchmark, gates, phase}} för de `top`
    högst rankade. results är PipelineResult.results (eller dicts med samma
    fält). Ett bolag som inte går att analysera får verdict "ERROR" och larmar inte."""
    from alpha_regime.screener_link import to_yf_ticker
    analyze = analyze or _default_analyze
    out: dict = {}
    for r in list(results or [])[:top]:
        get = (lambda k, d=None: r.get(k, d)) if isinstance(r, dict) else (lambda k, d=None: getattr(r, k, d))
        ticker = str(get("ticker") or "").strip().upper()
        if not ticker:
            continue
        yf_t = to_yf_ticker(ticker)
        bm = benchmark_for(yf_t)
        try:
            res = analyze(yf_t, bm, bool(get("kap_badge", False)))
        except Exception as exc:
            logger.warning("Quality-signal %s: %s", ticker, exc)
            out[ticker] = {"verdict": "ERROR", "error": str(exc)[:120], "benchmark": bm}
            continue
        if getattr(res, "error", None):
            out[ticker] = {"verdict": "ERROR", "error": str(res.error)[:120], "benchmark": bm}
            continue
        gates = {SIGNAL_LABELS.get(s.name, s.name): {"passed": bool(s.passed), "label": s.label}
                 for s in (getattr(res, "signals", None) or [])}
        out[ticker] = {"verdict": res.quality_verdict, "passed": int(res.signals_passed),
                       "total": len(gates) or 4, "benchmark": bm, "phase": res.market_phase,
                       "gates": gates}
    return out
