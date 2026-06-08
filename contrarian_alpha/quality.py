"""
quality.py — Quality Score (0-100) for Quality-Contrarian Screener.

Measures the durability and quality of a business's economic engine.
Used as the Quality pillar in the 5-pillar composite formula.

4 components (weights sum to 100):
  1. ROIC              max 30p  (Return on Invested Capital — moat indicator)
  2. ROCE              max 20p  (Return on Capital Employed — capital efficiency)
  3. Gross margin stability (5y) max 25p  (CoV — durable pricing power)
  4. Operating margin trend      max 25p  (OLS slope over 3-5y)

Composite weight: QUALITY_COMPOSITE_WEIGHT_QUALITY = 0.30 ("quality" mode)
                  QUALITY_COMPOSITE_WEIGHT_DEEP    = 0.20 ("deep_contrarian" mode)

Hard gate (enforced by engine.py, mode-dependent):
  quality mode:        ROIC > 15%
  deep_contrarian:     ROIC > 10%  (waived when hate_score > 70)

Börsdata KPI IDs used:
  ROIC             KPI 37
  ROCE             KPI 36
  Gross margin     KPI 28
  Operating margin KPI 29
  Revenue growth   KPI 94  (5y CAGR gate, quality mode only — used by engine)

Input dict keys (all optional, degrade gracefully):
  roic                  float   Current ROIC % (KPI 37)
  roic_history          list    Newest-first [t0, t-1, t-2, t-3] (KPI 37 history)
  roce                  float   Current ROCE % (KPI 36)
  gross_margin          float   Current gross margin % (KPI 28)
  gross_margin_history  list    Newest-first 3-5 annual values (KPI 28 history)
  operating_margin      float   Current operating margin % (KPI 29)
  op_margin_history     list    Newest-first 3-5 annual values (KPI 29 history)
  revenue_growth_5y     float   Annualised 5y revenue growth % (computed by engine from KPI 94)
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─── Composite weights ────────────────────────────────────────────────────────

QUALITY_COMPOSITE_WEIGHT_QUALITY = 0.30   # "quality" mode
QUALITY_COMPOSITE_WEIGHT_DEEP    = 0.20   # "deep_contrarian" mode

# ─── Hard-gate thresholds (enforced by engine.py) ────────────────────────────

GATE_ROIC_QUALITY   = 15.0   # ROIC % required in quality mode
GATE_ROIC_DEEP      = 10.0   # ROIC % required in deep_contrarian mode
GATE_REVENUE_CAGR   = 0.0    # 5y CAGR must be positive (quality mode only)

# ─── Component max points (must sum to 100) ───────────────────────────────────

_MAX_ROIC     = 30
_MAX_ROCE     = 20
_MAX_GM_STAB  = 25
_MAX_OM_TREND = 25

# ─── Result model ────────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    score:           float
    breakdown:       dict[str, float]  = field(default_factory=dict)
    flags:           list[str]         = field(default_factory=list)
    roic:            float | None      = None   # current ROIC %
    roce:            float | None      = None   # current ROCE %
    roic_trend:      float | None      = None   # pp improvement over history period
    gm_stability:    float | None      = None   # gross margin CoV (lower = more stable)
    op_margin_slope: float | None      = None   # pp/year slope of operating margin

    @property
    def passes_gate_quality(self) -> bool:
        """True if ROIC passes the strict quality-mode gate (> 15%)."""
        return self.roic is not None and self.roic > GATE_ROIC_QUALITY

    @property
    def passes_gate_deep(self) -> bool:
        """True if ROIC passes the relaxed deep-contrarian gate (> 10%)."""
        return self.roic is not None and self.roic > GATE_ROIC_DEEP


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _valid_floats(values: list) -> list[float]:
    """Filter to finite, non-None floats within a plausible range."""
    result = []
    for v in (values or []):
        if v is None:
            continue
        try:
            f = float(v)
            if f == f and abs(f) < 1e9:   # not NaN, not absurdly large
                result.append(f)
        except (TypeError, ValueError):
            pass
    return result


def _linear_slope(series: list[float]) -> float:
    """
    OLS slope of an oldest-first time series.
    Returns slope in units-per-step (pp/year for annual data).
    Returns 0.0 if fewer than 3 points.
    """
    n = len(series)
    if n < 3:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(series) / n
    num = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(series))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


# ─── Component scorers ────────────────────────────────────────────────────────

def _score_roic(data: dict) -> tuple[float, float | None, float | None, bool]:
    """
    ROIC: max 30p. Returns (pts, roic_current, trend_pp, has_real_data).

    Base score from current ROIC:
      >= 20% → 22p | >= 15% → 18p | >= 12% → 14p | >= 10% → 10p
      >=  7% →  6p | >=  4% →  3p |  <  4% →  0p

    Trend bonus (up to 8p, capped at _MAX_ROIC):
      improvement >= 5pp over history → +8p
      improvement >= 2pp              → +5p
      stable (within ±2pp)            → +2p
      declining                       → 0p

    trend_pp = roic_history[0] − roic_history[-1]  (positive = improving)
    """
    roic = data.get("roic")
    history = _valid_floats(data.get("roic_history") or [])

    if roic is None and not history:
        return 8.0, None, None, False   # moderate default when data missing

    roic_current: float | None = float(roic) if roic is not None else history[0]

    if roic_current is None:
        base = 0.0
    elif roic_current >= 20: base = 22.0
    elif roic_current >= 15: base = 18.0
    elif roic_current >= 12: base = 14.0
    elif roic_current >= 10: base = 10.0
    elif roic_current >=  7: base =  6.0
    elif roic_current >=  4: base =  3.0
    else:                     base =  0.0

    trend: float | None = None
    bonus = 0.0
    if len(history) >= 3:
        trend = history[0] - history[-1]
        if trend >= 5.0:    bonus = 8.0
        elif trend >= 2.0:  bonus = 5.0
        elif trend >= -2.0: bonus = 2.0   # stable
        # declining → 0 bonus

    pts = _clamp(base + bonus, 0.0, float(_MAX_ROIC))
    return pts, roic_current, trend, roic_current is not None


def _score_roce(data: dict) -> tuple[float, float | None, bool]:
    """
    ROCE: max 20p. Returns (pts, roce_current, has_real_data).

      >= 15% → 20p | >= 12% → 16p | >= 10% → 12p
      >=  7% →  8p | >=  4% →  4p |  <  4% →  0p
    """
    roce = data.get("roce")
    if roce is None:
        return 6.0, None, False   # moderate default

    v = float(roce)
    if v >= 15:   pts = 20.0
    elif v >= 12: pts = 16.0
    elif v >= 10: pts = 12.0
    elif v >=  7: pts =  8.0
    elif v >=  4: pts =  4.0
    else:          pts =  0.0

    return pts, v, True


def _score_gm_stability(data: dict) -> tuple[float, float | None, bool]:
    """
    Gross margin CoV over 5y: max 25p. Returns (pts, cov, has_real_data).

    CoV = StdDev / |mean|  (coefficient of variation)
    Lower CoV → more stable margins → more durable moat.

    CoV <  5% → 25p | < 10% → 20p | < 15% → 14p
    CoV < 25% →  8p | < 40% →  3p |  >= 40% →  0p

    If mean <= 0 (chronically loss-making at gross level) → 0p.
    """
    gm_hist = _valid_floats(data.get("gross_margin_history") or [])
    gm_cur  = data.get("gross_margin")

    # Prepend current value if provided; cap at 6 data points
    if gm_cur is not None:
        gm_hist = ([float(gm_cur)] + gm_hist)[:6]

    if len(gm_hist) < 3:
        return 8.0, None, False   # too little history — moderate default

    mean = statistics.mean(gm_hist)
    if mean <= 0:
        return 0.0, None, True   # chronically negative gross margin

    stdev = statistics.stdev(gm_hist)
    cov = stdev / abs(mean)

    if cov < 0.05:    pts = 25.0
    elif cov < 0.10:  pts = 20.0
    elif cov < 0.15:  pts = 14.0
    elif cov < 0.25:  pts =  8.0
    elif cov < 0.40:  pts =  3.0
    else:              pts =  0.0

    return pts, round(cov, 4), True


def _score_op_margin_trend(data: dict) -> tuple[float, float | None, bool]:
    """
    Operating margin OLS slope over 3-5y: max 25p.
    Returns (pts, slope_pp_per_year, has_real_data).

    Slope computed on oldest-first series. History is newest-first → reversed.

    slope >= +2.0 pp/yr → 25p  (strongly improving)
    slope >= +0.5 pp/yr → 20p  (clearly improving)
    slope >= +0.1 pp/yr → 14p  (barely improving)
    slope in ±0.1 pp/yr →  8p  (stable)
    slope < -0.1 pp/yr  →  3p  (declining — data at least available)
    slope < -1.0 pp/yr  →  0p  (sharply declining)

    Bonus +3p if currently positive AND clearly improving: confirmed compounder.
    """
    om_hist = _valid_floats(data.get("op_margin_history") or [])
    om_cur  = data.get("operating_margin")

    if om_cur is not None:
        om_hist = ([float(om_cur)] + om_hist)[:6]

    if len(om_hist) < 3:
        return 8.0, None, False   # moderate default — missing history

    oldest_first = list(reversed(om_hist))
    slope = _linear_slope(oldest_first)

    if slope >= 2.0:    base = 25.0
    elif slope >= 0.5:  base = 20.0
    elif slope >= 0.1:  base = 14.0
    elif slope >= -0.1: base =  8.0
    elif slope >= -1.0: base =  3.0
    else:               base =  0.0

    # Confirmed compounder bonus
    bonus = 3.0 if (om_cur is not None and float(om_cur) > 0 and slope >= 0.1) else 0.0
    pts = _clamp(base + bonus, 0.0, float(_MAX_OM_TREND))
    return pts, round(slope, 3), True


# ─── Main scoring function ───────────────────────────────────────────────────

def calculate_quality_score(quality_data: dict) -> QualityResult:
    """
    Calculate Quality Score (0-100) for a single instrument.

    Args:
        quality_data: Dict with ROIC, ROCE, gross margin and operating margin
                      values and history. See module docstring for keys.

    Returns:
        QualityResult with score, breakdown, flags, and raw indicator values.

    Usage in engine:
        result = calculate_quality_score(qdata)
        composite += result.score * weights["quality"]
        # Gate check handled separately in engine._run_single_ticker()
        if mode == "quality" and not result.passes_gate_quality:
            eliminate at QUALITY_GATE stage
    """
    if not quality_data:
        logger.debug("calculate_quality_score: empty quality_data dict")
        return QualityResult(score=0.0, flags=["NO_QUALITY_DATA"])

    roic_pts, roic_val, roic_trend, roic_real = _score_roic(quality_data)
    roce_pts, roce_val, roce_real             = _score_roce(quality_data)
    gm_pts,   gm_cov,   gm_real              = _score_gm_stability(quality_data)
    om_pts,   om_slope,  om_real             = _score_op_margin_trend(quality_data)

    total = _clamp(roic_pts + roce_pts + gm_pts + om_pts, 0.0, 100.0)

    breakdown = {
        "roic":            round(roic_pts, 1),
        "roce":            round(roce_pts, 1),
        "gm_stability":    round(gm_pts, 1),
        "op_margin_trend": round(om_pts, 1),
    }

    flags: list[str] = []
    if not roic_real: flags.append("ROIC_DATA_MISSING")
    if not roce_real: flags.append("ROCE_DATA_MISSING")
    if not gm_real:   flags.append("GM_HISTORY_SHORT")
    if not om_real:   flags.append("OM_HISTORY_SHORT")

    return QualityResult(
        score           = round(total, 1),
        breakdown       = breakdown,
        flags           = flags,
        roic            = roic_val,
        roce            = roce_val,
        roic_trend      = round(roic_trend, 2) if roic_trend is not None else None,
        gm_stability    = gm_cov,
        op_margin_slope = om_slope,
    )


# ─── CLI diagnostics ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "_label": "Quality compounder — high ROIC, stable margins, improving",
            "roic": 24.0, "roic_history": [24.0, 21.0, 18.0, 16.0],
            "roce": 18.0,
            "gross_margin": 58.0, "gross_margin_history": [57.0, 56.5, 57.5, 56.0, 55.5],
            "operating_margin": 22.0, "op_margin_history": [20.0, 18.5, 17.0, 15.5],
        },
        {
            "_label": "Commodity producer — moderate ROIC, volatile margins",
            "roic": 12.0, "roic_history": [12.0, 18.0, 8.0, 15.0],
            "roce": 10.0,
            "gross_margin": 35.0, "gross_margin_history": [42.0, 22.0, 38.0, 30.0],
            "operating_margin": 15.0, "op_margin_history": [20.0, 8.0, 18.0, 12.0],
        },
        {
            "_label": "Turnaround play — ROIC recovering from trough",
            "roic": 8.0, "roic_history": [8.0, 4.0, 1.0, -2.0],
            "roce": 6.0,
            "gross_margin": 28.0, "gross_margin_history": [22.0, 18.0, 15.0],
            "operating_margin": 5.0, "op_margin_history": [1.0, -3.0, -5.0],
        },
        {
            "_label": "Value trap — declining ROIC and margins",
            "roic": 3.0, "roic_history": [3.0, 6.0, 9.0, 12.0],
            "roce": 2.0,
            "gross_margin": 15.0, "gross_margin_history": [20.0, 25.0, 28.0, 30.0],
            "operating_margin": 2.0, "op_margin_history": [5.0, 8.0, 10.0, 12.0],
        },
        {
            "_label": "Sparse data — current values only, no history",
            "roic": 11.0,
            "roce": 9.0,
            "gross_margin": 40.0,
            "operating_margin": 12.0,
        },
        {
            "_label": "No data at all",
        },
    ]

    print(f"\n{'─'*80}")
    print(f"  QUALITY SCORE")
    print(f"  Gates: ROIC > {GATE_ROIC_QUALITY:.0f}% (quality mode) | > {GATE_ROIC_DEEP:.0f}% (deep_contrarian)")
    print(f"  Components: ROIC({_MAX_ROIC}p) ROCE({_MAX_ROCE}p) GM-Stability({_MAX_GM_STAB}p) OM-Trend({_MAX_OM_TREND}p)")
    print(f"{'─'*80}")

    for case in test_cases:
        label = case.pop("_label")
        result = calculate_quality_score(case)
        gate_q = "✓ QUALITY" if result.passes_gate_quality else "✗ quality"
        gate_d = "✓ DEEP"    if result.passes_gate_deep    else "✗ deep"
        roic_s = f"ROIC={result.roic:.1f}%" if result.roic is not None else "ROIC=n/a"
        trend_s = f"(trend={result.roic_trend:+.1f}pp)" if result.roic_trend is not None else ""
        print(f"\n  {label}")
        print(f"    Score: {result.score:>5.1f}/100   {gate_q}  {gate_d}   {roic_s} {trend_s}")
        print(f"    Break: {result.breakdown}")
        if result.flags:
            print(f"    Flags: {result.flags}")

    print(f"\n{'─'*80}\n")
