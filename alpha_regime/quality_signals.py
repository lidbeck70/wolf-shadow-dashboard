"""
alpha_regime/quality_signals.py
Pure, I/O-free evaluators for the 4 Quality-mode confirmation signals.
All functions take already-computed values; data fetching lives in engine.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Cycle phases that indicate early-to-mid cycle (good entry environment)
QUALITY_CYCLE_PASS_PHASES = {
    "DISBELIEF",
    "DISBELIEF_NEW",
    "HOPE",
    "OPTIMISM",
    "BELIEF",
}

_QUALITY_SCORE_THRESHOLD = 55.0


@dataclass
class SignalResult:
    name: str            # "TREND" | "DISCOUNT" | "CYCLE" | "QUALITY"
    passed: bool
    label: str           # short status label shown on the card
    detail: str          # one-line explanation
    value: Optional[float] = None   # key metric (e.g. % above EMA200)
    warning: Optional[str] = None   # non-blocking caution (e.g. TOO_CHEAP)


def eval_trend(
    price: float,
    ema50: float,
    ema200: float,
) -> SignalResult:
    """
    TREND signal: bullish structure requires price > EMA200 AND EMA50 > EMA200.
    """
    if any(v != v for v in (price, ema50, ema200)):  # NaN check
        return SignalResult(
            name="TREND",
            passed=False,
            label="NO DATA",
            detail="Price / EMA data unavailable",
        )

    price_vs_200 = (price / ema200 - 1) * 100 if ema200 != 0 else 0.0
    ema50_vs_200 = (ema50 / ema200 - 1) * 100 if ema200 != 0 else 0.0
    above_ema200 = price > ema200
    ema50_above = ema50 > ema200
    passed = above_ema200 and ema50_above

    if passed:
        label = "BULLISH"
        detail = f"Price {price_vs_200:+.1f}% vs EMA200 · EMA50 {ema50_vs_200:+.1f}% above"
    elif above_ema200 and not ema50_above:
        label = "MIXED"
        detail = f"Price above EMA200 but EMA50 {ema50_vs_200:.1f}% below — crossover pending"
    elif not above_ema200 and ema50_above:
        label = "MIXED"
        detail = f"EMA50 above EMA200 but price {price_vs_200:.1f}% below — momentum lost"
    else:
        label = "BEARISH"
        detail = f"Price {price_vs_200:.1f}% below EMA200 · EMA50 {ema50_vs_200:.1f}% below"

    return SignalResult(
        name="TREND",
        passed=passed,
        label=label,
        detail=detail,
        value=round(price_vs_200, 1),
    )


def eval_discount(
    pe: Optional[float],
    ev_ebit: Optional[float],
) -> SignalResult:
    """
    DISCOUNT signal: neither P/E nor EV/EBIT should be in EXPENSIVE territory.
    Uses the same bands as check_valuation_bands() — [7,25] for P/E, [4,20] for EV/EBIT.
    TOO_CHEAP is a warning only (still passes).
    Passes if at least one metric has data and none are EXPENSIVE.
    """
    _PE_MIN, _PE_MAX = 7.0, 25.0
    _EVEBIT_MIN, _EVEBIT_MAX = 4.0, 20.0

    if pe is None and ev_ebit is None:
        return SignalResult(
            name="DISCOUNT",
            passed=False,
            label="NO DATA",
            detail="No P/E or EV/EBIT data from Börsdata",
        )

    parts = []
    warnings = []
    expensive = False

    if pe is not None and pe > 0:
        if pe > _PE_MAX:
            parts.append(f"P/E {pe:.1f} > {_PE_MAX}")
            expensive = True
        elif pe < _PE_MIN:
            parts.append(f"P/E {pe:.1f} (below {_PE_MIN} — may be value trap)")
            warnings.append("PE_TOO_CHEAP")
        else:
            parts.append(f"P/E {pe:.1f} in band")

    if ev_ebit is not None and ev_ebit > 0:
        if ev_ebit > _EVEBIT_MAX:
            parts.append(f"EV/EBIT {ev_ebit:.1f} > {_EVEBIT_MAX}")
            expensive = True
        elif ev_ebit < _EVEBIT_MIN:
            parts.append(f"EV/EBIT {ev_ebit:.1f} (below {_EVEBIT_MIN})")
            warnings.append("EVEBIT_TOO_CHEAP")
        else:
            parts.append(f"EV/EBIT {ev_ebit:.1f} in band")

    passed = not expensive
    label = "EXPENSIVE" if expensive else ("REASONABLE" if not warnings else "TOO CHEAP")
    detail = " · ".join(parts) if parts else "Metrics in acceptable range"
    warning_str = ", ".join(warnings) if warnings else None

    return SignalResult(
        name="DISCOUNT",
        passed=passed,
        label=label,
        detail=detail,
        value=pe,
        warning=warning_str,
    )


def eval_cycle(
    phase: str,
    confidence: float,
) -> SignalResult:
    """
    CYCLE signal: market is in an early-to-mid cycle phase favourable for entry.
    Passes for DISBELIEF, DISBELIEF_NEW, HOPE, OPTIMISM, BELIEF.
    """
    passed = phase in QUALITY_CYCLE_PASS_PHASES
    label = "FAVOURABLE" if passed else "UNFAVOURABLE"
    detail = (
        f"Market cycle: {phase} ({confidence:.0f}% confidence) — good entry window"
        if passed
        else f"Market cycle: {phase} ({confidence:.0f}% confidence) — late/down cycle, wait"
    )
    return SignalResult(
        name="CYCLE",
        passed=passed,
        label=label,
        detail=detail,
        value=round(confidence, 1),
    )


def eval_quality(
    quality_score: Optional[float],
    kap_badge: bool,
) -> SignalResult:
    """
    QUALITY signal: composite quality score >= threshold OR company holds KAP badge.
    If no quality data is available, this signal is skipped (treated as N/A, not FAIL).
    """
    if quality_score is None and not kap_badge:
        return SignalResult(
            name="QUALITY",
            passed=False,
            label="NO DATA",
            detail="Quality score not available (Börsdata required)",
        )

    passed = kap_badge or (quality_score is not None and quality_score >= _QUALITY_SCORE_THRESHOLD)

    if kap_badge:
        label = "KAP SCREENED"
        detail = f"★ KAP badge — all 4 KAP criteria met · Score {quality_score:.0f}/100" if quality_score else "★ KAP badge — all 4 KAP criteria met"
    elif quality_score is not None and passed:
        label = "QUALITY PASS"
        detail = f"Quality composite {quality_score:.0f}/100 ≥ {_QUALITY_SCORE_THRESHOLD:.0f} threshold"
    else:
        label = "LOW QUALITY"
        detail = f"Quality composite {quality_score:.0f}/100 < {_QUALITY_SCORE_THRESHOLD:.0f} threshold"

    return SignalResult(
        name="QUALITY",
        passed=passed,
        label=label,
        detail=detail,
        value=quality_score,
    )


def score_quality_signals(signals: list[SignalResult]) -> tuple[int, str]:
    """
    Count passed signals and return (count, verdict).
    N/A signals (label == 'NO DATA') are excluded from denominator.
    """
    available = [s for s in signals if s.label != "NO DATA"]
    passed = sum(1 for s in available if s.passed)
    total = len(available)

    if total == 0:
        return 0, "WAIT"
    if passed == total:
        return passed, "BUY"
    if passed >= total - 1:
        return passed, "WATCH"
    return passed, "WAIT"
