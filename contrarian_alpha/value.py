"""
value.py — Value Score (0-100) for Quality-Contrarian Screener.

Measures how cheap a company is RELATIVE TO ITS OWN HISTORY.
A commodity producer's "fair" EV/EBITDA is 4×; a quality compounder's
is 18× — absolute-level comparisons mislead.  The contrarian question is:
"Is this company cheap FOR ITSELF right now?"

2 components (each max 50p):
  1. P/FCF vs own 5-10y median     max 50p  (KPI 76 history)
  2. EV/EBITDA vs own 5-10y median max 50p  (KPI 11 history)

Composite weight: VALUE_COMPOSITE_WEIGHT = 0.20

Relative scoring requires >= 3 valid historical data points per metric.
When history is too short, falls back to absolute-level scoring (max 30p
instead of 50p) and sets the VALUE_HISTORY_LIMITED flag.

Extreme multiples are filtered before computing the median to prevent
crisis-year outliers from skewing the baseline:
  P/FCF    values >= 100 are discarded
  EV/EBITDA values >= 60  are discarded

Börsdata KPI IDs:
  P/FCF       KPI 76
  EV/EBITDA   KPI 11

Input dict keys (all optional, degrade gracefully):
  p_fcf               float   Current P/FCF (KPI 76)
  p_fcf_history       list    Newest-first 3-10 annual values (KPI 76 history)
  ev_ebitda           float   Current EV/EBITDA (KPI 11)
  ev_ebitda_history   list    Newest-first 3-10 annual values (KPI 11 history)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─── Composite weight ─────────────────────────────────────────────────────────

VALUE_COMPOSITE_WEIGHT = 0.20

# ─── Outlier filter thresholds ───────────────────────────────────────────────

_P_FCF_MAX_VALID     = 100.0   # discard P/FCF values above this
_EV_EBITDA_MAX_VALID  = 60.0   # discard EV/EBITDA values above this

# Minimum valid historical points needed for relative scoring
_MIN_HISTORY = 3

# ─── Component max points ─────────────────────────────────────────────────────

_MAX_P_FCF     = 50
_MAX_EV_EBITDA = 50

# ─── Result model ────────────────────────────────────────────────────────────

@dataclass
class ValueResult:
    score:               float
    breakdown:           dict[str, float]  = field(default_factory=dict)
    flags:               list[str]         = field(default_factory=list)
    p_fcf:               float | None      = None   # current P/FCF
    p_fcf_median:        float | None      = None   # own historical median
    p_fcf_discount:      float | None      = None   # % discount vs own median (positive = cheap)
    ev_ebitda:           float | None      = None   # current EV/EBITDA
    ev_ebitda_median:    float | None      = None   # own historical median
    ev_ebitda_discount:  float | None      = None   # % discount vs own median

    @property
    def using_absolute_fallback(self) -> bool:
        """True when history was too short and absolute scoring was used."""
        return "VALUE_HISTORY_LIMITED" in self.flags


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _valid_positives(values: list, max_val: float) -> list[float]:
    """Filter to positive, finite, non-outlier floats."""
    result = []
    for v in (values or []):
        if v is None:
            continue
        try:
            f = float(v)
            if 0 < f < max_val:
                result.append(f)
        except (TypeError, ValueError):
            pass
    return result


def _parse_positive(value, max_val: float) -> float | None:
    """Return float if value is a valid positive below max_val, else None."""
    if value is None:
        return None
    try:
        f = float(value)
        return f if 0 < f < max_val else None
    except (TypeError, ValueError):
        return None


def _median(values: list[float]) -> float:
    n = len(values)
    s = sorted(values)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _discount_pct(current: float, median: float) -> float:
    """
    % by which current multiple is below its historical median.
    Positive = cheap (below own history).
    Negative = expensive (above own history).
    """
    if median <= 0:
        return 0.0
    return (median - current) / median * 100.0


# ─── Relative scoring (vs own historical median) ─────────────────────────────

def _score_relative(current: float, median: float, max_pts: float) -> float:
    """
    Score the discount/premium of current multiple vs own historical median.

    discount >= 50% → 100% of max_pts
    discount >= 30% →  80%
    discount >= 15% →  60%
    discount >=  5% →  44%
    discount >=  0% →  30%  (at or barely below median)
    premium  <= 15% →  16%
    premium  <= 30% →   6%
    premium   > 30% →   0p
    """
    disc = _discount_pct(current, median)

    if disc >= 50:    frac = 1.00
    elif disc >= 30:  frac = 0.80
    elif disc >= 15:  frac = 0.60
    elif disc >=  5:  frac = 0.44
    elif disc >=  0:  frac = 0.30
    elif disc >= -15: frac = 0.16
    elif disc >= -30: frac = 0.06
    else:              frac = 0.00

    return round(frac * max_pts, 1)


# ─── Absolute fallback scoring ────────────────────────────────────────────────

def _score_p_fcf_absolute(p_fcf: float) -> float:
    """Absolute P/FCF score (capped at 30p, lower than relative mode's 50p max)."""
    if p_fcf < 10:    return 30.0
    if p_fcf < 15:    return 22.0
    if p_fcf < 20:    return 14.0
    if p_fcf < 30:    return  6.0
    return 0.0


def _score_ev_ebitda_absolute(ev: float) -> float:
    """Absolute EV/EBITDA score (capped at 30p)."""
    if ev < 6:    return 30.0
    if ev < 10:   return 22.0
    if ev < 15:   return 14.0
    if ev < 20:   return  6.0
    return 0.0


# ─── Component scorers ────────────────────────────────────────────────────────

def _score_p_fcf(data: dict) -> tuple[float, float | None, float | None, float | None, bool]:
    """
    P/FCF component: max 50p (relative) or 30p (absolute fallback).

    Returns (pts, current, median_or_None, discount_or_None, used_history).
    """
    cur = _parse_positive(data.get("p_fcf"), _P_FCF_MAX_VALID)
    raw_hist = data.get("p_fcf_history") or []
    hist = _valid_positives(raw_hist, _P_FCF_MAX_VALID)

    # If no current value, promote newest valid history item as current
    if cur is None:
        if hist:
            cur = hist[0]
            hist = hist[1:]
        else:
            return 12.0, None, None, None, False   # moderate default — no data

    if len(hist) >= _MIN_HISTORY:
        med  = _median(hist)
        disc = _discount_pct(cur, med)
        pts  = _score_relative(cur, med, float(_MAX_P_FCF))
        return pts, cur, round(med, 2), round(disc, 1), True

    # Absolute fallback
    return _score_p_fcf_absolute(cur), cur, None, None, False


def _score_ev_ebitda(data: dict) -> tuple[float, float | None, float | None, float | None, bool]:
    """
    EV/EBITDA component: max 50p (relative) or 30p (absolute fallback).

    Returns (pts, current, median_or_None, discount_or_None, used_history).
    """
    cur = _parse_positive(data.get("ev_ebitda"), _EV_EBITDA_MAX_VALID)
    raw_hist = data.get("ev_ebitda_history") or []
    hist = _valid_positives(raw_hist, _EV_EBITDA_MAX_VALID)

    if cur is None:
        if hist:
            cur = hist[0]
            hist = hist[1:]
        else:
            return 12.0, None, None, None, False

    if len(hist) >= _MIN_HISTORY:
        med  = _median(hist)
        disc = _discount_pct(cur, med)
        pts  = _score_relative(cur, med, float(_MAX_EV_EBITDA))
        return pts, cur, round(med, 2), round(disc, 1), True

    return _score_ev_ebitda_absolute(cur), cur, None, None, False


# ─── Main scoring function ───────────────────────────────────────────────────

def calculate_value_score(value_data: dict) -> ValueResult:
    """
    Calculate Value Score (0-100) for a single instrument.

    Args:
        value_data: Dict with P/FCF and EV/EBITDA current values and history.
                    See module docstring for keys.

    Returns:
        ValueResult with score, breakdown, flags, raw multiples, and discounts.

    Usage in engine:
        result = calculate_value_score(vdata)
        composite += result.score * weights["value"]
    """
    if not value_data:
        logger.debug("calculate_value_score: empty value_data dict")
        return ValueResult(score=0.0, flags=["NO_VALUE_DATA"])

    pfcf_pts, pfcf_cur, pfcf_med, pfcf_disc, pfcf_hist = _score_p_fcf(value_data)
    ev_pts,   ev_cur,   ev_med,   ev_disc,   ev_hist   = _score_ev_ebitda(value_data)

    total = _clamp(pfcf_pts + ev_pts, 0.0, 100.0)

    breakdown = {
        "p_fcf_score":     round(pfcf_pts, 1),
        "ev_ebitda_score": round(ev_pts, 1),
    }

    flags: list[str] = []
    if pfcf_cur is None:  flags.append("P_FCF_DATA_MISSING")
    if ev_cur is None:    flags.append("EV_EBITDA_DATA_MISSING")
    if (pfcf_cur is not None and not pfcf_hist) or (ev_cur is not None and not ev_hist):
        flags.append("VALUE_HISTORY_LIMITED")

    return ValueResult(
        score              = round(total, 1),
        breakdown          = breakdown,
        flags              = flags,
        p_fcf              = pfcf_cur,
        p_fcf_median       = pfcf_med,
        p_fcf_discount     = pfcf_disc,
        ev_ebitda          = ev_cur,
        ev_ebitda_median   = ev_med,
        ev_ebitda_discount = ev_disc,
    )


# ─── CLI diagnostics ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "_label": "Commodity miner at cycle trough — 50%+ discount to own history",
            "p_fcf":    8.0,  "p_fcf_history":    [12.0, 16.0, 20.0, 18.0, 15.0],
            "ev_ebitda": 4.5, "ev_ebitda_history": [6.0, 8.5, 10.0, 9.0, 7.5],
        },
        {
            "_label": "Quality compounder at fair value — at own median",
            "p_fcf":    24.0, "p_fcf_history":    [22.0, 26.0, 23.0, 25.0],
            "ev_ebitda": 16.0, "ev_ebitda_history": [14.0, 18.0, 15.0, 17.0],
        },
        {
            "_label": "Expensive growth stock — 30%+ premium to own history",
            "p_fcf":    45.0, "p_fcf_history":    [32.0, 28.0, 30.0, 35.0],
            "ev_ebitda": 28.0, "ev_ebitda_history": [18.0, 16.0, 20.0, 22.0],
        },
        {
            "_label": "Oil stock — absolute fallback (history too short)",
            "p_fcf":    9.0,  "p_fcf_history":    [14.0, 12.0],   # only 2 points
            "ev_ebitda": 5.5, "ev_ebitda_history": [7.0],
        },
        {
            "_label": "Turnaround — negative P/FCF years filtered from history",
            "p_fcf":    11.0, "p_fcf_history":    [-8.0, -12.0, 14.0, 16.0, 18.0],
            "ev_ebitda": 7.0, "ev_ebitda_history": [6.5, 9.0, 11.0, 12.0],
        },
        {
            "_label": "No data",
        },
    ]

    print(f"\n{'─'*80}")
    print(f"  VALUE SCORE  |  P/FCF (max {_MAX_P_FCF}p)  +  EV/EBITDA (max {_MAX_EV_EBITDA}p)")
    print(f"  Relative-to-own-history | min {_MIN_HISTORY} historical points | fallback = absolute (max 30p)")
    print(f"{'─'*80}")

    for case in test_cases:
        label = case.pop("_label")
        result = calculate_value_score(case)
        mode_str = "ABSOLUTE fallback" if result.using_absolute_fallback else "RELATIVE to own history"

        if result.p_fcf is not None and result.p_fcf_median is not None:
            pfcf_str = (f"P/FCF={result.p_fcf:.1f}x  median={result.p_fcf_median:.1f}x"
                        f"  disc={result.p_fcf_discount:+.0f}%")
        elif result.p_fcf is not None:
            pfcf_str = f"P/FCF={result.p_fcf:.1f}x  (no history)"
        else:
            pfcf_str = "P/FCF=n/a"

        if result.ev_ebitda is not None and result.ev_ebitda_median is not None:
            ev_str = (f"EV/EBITDA={result.ev_ebitda:.1f}x  median={result.ev_ebitda_median:.1f}x"
                      f"  disc={result.ev_ebitda_discount:+.0f}%")
        elif result.ev_ebitda is not None:
            ev_str = f"EV/EBITDA={result.ev_ebitda:.1f}x  (no history)"
        else:
            ev_str = "EV/EBITDA=n/a"

        print(f"\n  {label}")
        print(f"    Score: {result.score:>5.1f}/100   [{mode_str}]")
        print(f"    {pfcf_str}")
        print(f"    {ev_str}")
        print(f"    Break: {result.breakdown}")
        if result.flags:
            print(f"    Flags: {result.flags}")

    print(f"\n{'─'*80}\n")
