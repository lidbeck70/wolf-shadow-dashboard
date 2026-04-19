"""
strength.py — Fundamental Strength Score (0-100) for Contrarian Alpha Screener.

Adapted from blindspot/scoring/strength.py, extended with 5 hard-gate checks
and Altman Z-score. Instruments that miss any gate are flagged — they still
receive a score (for transparency) but are marked for downstream handling.

Hard gates (all must pass; failure → flag):
  1. FCF (TTM) positive
  2. EBITDA margin > 0%
  3. D/E < 0.6
  4. Altman Z-score > 1.8
  5. Equity > 0

Soft score components (sum → 0-100):
  1. FCF quality          max 30p  (TTM positive + 3y consistency)
  2. EBITDA margin        max 15p  (0%→0p, 15%→7.5p, 30%+→15p)
  3. FCF yield            max 15p  (0%→0p, 5%→7.5p, 10%+→15p)
  4. Debt/Equity          max 20p  (≤0.3→20p … >2.0→0p)
  5. Altman Z-score       max 15p  (>2.99→15p, >2.5→12p, >1.8→8p, else→0p)
  6. EV/EBITDA bonus      max  5p  (≤6→5p, ≤10→3p, ≤15→1p)

Composite Score weight: STRENGTH_COMPOSITE_WEIGHT = 0.30

Börsdata fundamentals dict keys (flat, all optional):
  fcf              float   Free cash flow TTM (MSEK or native currency)
  fcf_history      list    [fcf_t0, fcf_t-1, fcf_t-2]  (newest first)
  fcf_yield        float   FCF / market cap * 100 (%)
  ebitda           float   EBITDA absolute
  ebitda_margin    float   EBITDA / revenue * 100 (%)
  debt_to_equity   float   Total debt / equity (ratio, NOT percentage)
  ev_ebitda        float   EV / EBITDA multiple
  equity           float   Total stockholders' equity (positive = solvent)
  total_assets     float   For Altman Z
  total_liabilities float  For Altman Z (market_cap / total_liabilities = X4)
  working_capital  float   Current assets − current liabilities
  retained_earnings float  Cumulative retained earnings
  ebit             float   Operating profit (for Altman Z X3)
  revenue          float   Total revenue (for Altman Z X5)
  market_cap       float   Market capitalisation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─── Composite weight ─────────────────────────────────────────────────────────

STRENGTH_COMPOSITE_WEIGHT = 0.30  # 30% of total Composite Score

# ─── Hard-gate thresholds ─────────────────────────────────────────────────────

GATE_FCF_POSITIVE      = 0.0    # FCF TTM must be > 0
GATE_EBITDA_MARGIN_MIN = 0.0    # EBITDA margin must be > 0%
GATE_DE_MAX            = 0.6    # D/E must be < 0.6
GATE_ALTMAN_Z_MIN      = 1.8    # Altman Z must be > 1.8
# Gate 5: equity > 0 (implicit; checked directly)

# ─── Result model ────────────────────────────────────────────────────────────

@dataclass
class StrengthResult:
    score: float                          # 0–100
    breakdown: dict[str, float] = field(default_factory=dict)
    flags: list[str]            = field(default_factory=list)
    altman_z: float | None      = None    # Raw Z-score (for display)
    gate_results: dict[str, bool] = field(default_factory=dict)  # gate → passed

    @property
    def passes_all_gates(self) -> bool:
        return len(self.flags) == 0

    @property
    def flag_count(self) -> int:
        return len(self.flags)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _altman_z(fund: dict) -> float | None:
    """
    Altman Z-score (original 1968 model, non-financial firms).

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    X1 = working_capital / total_assets
    X2 = retained_earnings / total_assets
    X3 = ebit / total_assets
    X4 = market_cap / total_liabilities
    X5 = revenue / total_assets

    Returns None if total_assets is missing or zero (cannot compute).
    Returns partial score using available components when some inputs are missing,
    weighted by the fraction of components available.
    """
    total_assets = fund.get("total_assets")
    if not total_assets or total_assets <= 0:
        return None

    components: list[tuple[float, float]] = []  # (weight, raw_value / total_assets or ratio)

    wc = fund.get("working_capital")
    if wc is not None:
        components.append((1.2, wc / total_assets))

    re = fund.get("retained_earnings")
    if re is not None:
        components.append((1.4, re / total_assets))

    ebit = fund.get("ebit")
    if ebit is not None:
        components.append((3.3, ebit / total_assets))

    mktcap = fund.get("market_cap")
    total_liab = fund.get("total_liabilities")
    if mktcap is not None and total_liab and total_liab > 0:
        components.append((0.6, mktcap / total_liab))

    revenue = fund.get("revenue")
    if revenue is not None:
        components.append((1.0, revenue / total_assets))

    if not components:
        return None

    max_weight = 1.2 + 1.4 + 3.3 + 0.6 + 1.0  # = 8.5
    actual_weight = sum(w for w, _ in components)
    raw_z = sum(w * v for w, v in components)

    # Scale partial result to full-model equivalent
    if actual_weight < max_weight:
        raw_z = raw_z * (max_weight / actual_weight)

    return round(raw_z, 3)


# ─── Gate checks ─────────────────────────────────────────────────────────────

def _check_gates(fund: dict, z: float | None) -> tuple[list[str], dict[str, bool]]:
    """Run all 5 hard-gate checks. Returns (flags, gate_results)."""
    flags: list[str] = []
    gates: dict[str, bool] = {}

    # 1. FCF (TTM) positive
    fcf = fund.get("fcf")
    fcf_history = fund.get("fcf_history") or []
    fcf_ttm = fcf_history[0] if fcf_history else fcf
    gate_fcf = fcf_ttm is not None and fcf_ttm > GATE_FCF_POSITIVE
    gates["fcf_positive"] = gate_fcf
    if not gate_fcf:
        flags.append("FCF_NEGATIVE" if (fcf_ttm is not None and fcf_ttm <= 0) else "FCF_MISSING")

    # 2. EBITDA margin > 0%
    ebitda_margin = fund.get("ebitda_margin")
    gate_ebitda = ebitda_margin is not None and ebitda_margin > GATE_EBITDA_MARGIN_MIN
    gates["ebitda_margin_positive"] = gate_ebitda
    if not gate_ebitda:
        flags.append("EBITDA_MARGIN_NEGATIVE" if (ebitda_margin is not None and ebitda_margin <= 0)
                     else "EBITDA_MARGIN_MISSING")

    # 3. D/E < 0.6
    de = fund.get("debt_to_equity")
    gate_de = de is not None and de < GATE_DE_MAX
    gates["debt_equity_low"] = gate_de
    if not gate_de:
        flags.append(f"DE_RATIO_HIGH({de:.2f})" if de is not None else "DE_RATIO_MISSING")

    # 4. Altman Z > 1.8
    gate_z = z is not None and z > GATE_ALTMAN_Z_MIN
    gates["altman_z_ok"] = gate_z
    if not gate_z:
        flags.append(f"ALTMAN_Z_LOW({z:.2f})" if z is not None else "ALTMAN_Z_MISSING")

    # 5. Positive equity
    equity = fund.get("equity")
    gate_equity = equity is not None and equity > 0
    gates["equity_positive"] = gate_equity
    if not gate_equity:
        flags.append("EQUITY_NEGATIVE" if (equity is not None and equity <= 0) else "EQUITY_MISSING")

    return flags, gates


# ─── Soft score components ────────────────────────────────────────────────────

def _score_fcf(fund: dict) -> float:
    """FCF quality: max 30p. Reused from blindspot with TTM emphasis."""
    fcf = fund.get("fcf")
    fcf_history = fund.get("fcf_history") or []
    # Normalise: newest-first list
    history = fcf_history if fcf_history else ([fcf] if fcf is not None else [])

    if len(history) >= 3:
        positive = sum(1 for f in history[:3] if f is not None and f > 0)
        # TTM positive is mandatory gate — here reward consistency beyond that
        return positive / 3 * 30.0
    if len(history) == 2:
        positive = sum(1 for f in history if f is not None and f > 0)
        return positive / 2 * 25.0
    if len(history) == 1:
        return 20.0 if (history[0] is not None and history[0] > 0) else 0.0
    return 0.0


def _score_ebitda_margin(fund: dict) -> float:
    """EBITDA margin: max 15p. 0%→0p, 15%→7.5p, 30%+→15p."""
    margin = fund.get("ebitda_margin")
    if margin is None:
        return 0.0
    return _clamp(margin / 30.0 * 15.0, 0.0, 15.0)


def _score_fcf_yield(fund: dict) -> float:
    """FCF yield: max 15p. 0%→0p, 5%→7.5p, 10%+→15p."""
    yield_ = fund.get("fcf_yield")
    if yield_ is None or yield_ <= 0:
        return 0.0
    return _clamp(yield_ / 10.0 * 15.0, 0.0, 15.0)


def _score_debt_equity(fund: dict) -> float:
    """D/E: max 20p. Hard gate is <0.6, soft scoring rewards even lower debt."""
    de = fund.get("debt_to_equity")
    if de is None:
        return 5.0  # small benefit of doubt
    if de <= 0.1:  return 20.0
    if de <= 0.3:  return 18.0
    if de <= 0.5:  return 15.0
    if de < 0.6:   return 12.0  # Passes gate but not stellar
    if de <= 1.0:  return 6.0   # Failed gate — partial credit
    if de <= 1.5:  return 3.0
    return 0.0


def _score_altman_z(z: float | None) -> float:
    """Altman Z contribution: max 15p. Z>2.99→15p, Z>2.5→12p, Z>1.8→8p, else→0p."""
    if z is None:
        return 5.0  # data missing — partial benefit of doubt
    if z > 2.99:  return 15.0  # Safe zone
    if z > 2.50:  return 12.0
    if z > 1.80:  return 8.0   # Grey zone lower bound (gate passes)
    if z > 1.23:  return 3.0   # Deep grey zone (gate fails)
    return 0.0                  # Distress zone


def _score_ev_ebitda(fund: dict) -> float:
    """EV/EBITDA bonus: max 5p. Low multiple = undervalued."""
    ev_ebitda = fund.get("ev_ebitda")
    if ev_ebitda is None or ev_ebitda <= 0:
        return 0.0
    if ev_ebitda <= 6:   return 5.0
    if ev_ebitda <= 10:  return 3.0
    if ev_ebitda <= 15:  return 1.0
    return 0.0


# ─── Public API ───────────────────────────────────────────────────────────────

def calculate_strength_score(fundamentals: dict) -> StrengthResult:
    """
    Calculate Fundamental Strength Score (0-100) for a single instrument.

    Args:
        fundamentals: Flat dict with financial data (see module docstring for keys).

    Returns:
        StrengthResult with score, breakdown, flags, altman_z, gate_results.

    Usage in pipeline:
        result = calculate_strength_score(fund_dict)
        if not result.passes_all_gates:
            # Downstream: flag or discard — do not silently pass
            ...
        composite_contribution = result.score * STRENGTH_COMPOSITE_WEIGHT
    """
    if not fundamentals:
        logger.debug("calculate_strength_score: empty fundamentals dict")
        return StrengthResult(
            score=0.0,
            flags=["NO_FUNDAMENTAL_DATA"],
            gate_results={g: False for g in
                          ("fcf_positive", "ebitda_margin_positive",
                           "debt_equity_low", "altman_z_ok", "equity_positive")},
        )

    z = _altman_z(fundamentals)
    flags, gate_results = _check_gates(fundamentals, z)

    fcf_pts      = _score_fcf(fundamentals)
    ebitda_pts   = _score_ebitda_margin(fundamentals)
    yield_pts    = _score_fcf_yield(fundamentals)
    de_pts       = _score_debt_equity(fundamentals)
    z_pts        = _score_altman_z(z)
    ev_pts       = _score_ev_ebitda(fundamentals)

    total = _clamp(fcf_pts + ebitda_pts + yield_pts + de_pts + z_pts + ev_pts, 0.0, 100.0)

    breakdown = {
        "fcf_quality":    round(fcf_pts, 1),
        "ebitda_margin":  round(ebitda_pts, 1),
        "fcf_yield":      round(yield_pts, 1),
        "debt_equity":    round(de_pts, 1),
        "altman_z":       round(z_pts, 1),
        "ev_ebitda_bonus":round(ev_pts, 1),
    }

    return StrengthResult(
        score=round(total, 1),
        breakdown=breakdown,
        flags=flags,
        altman_z=z,
        gate_results=gate_results,
    )


# ─── CLI diagnostics ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "_label": "Stark råvaruproducent (alla gates gröna)",
            "fcf": 450e6, "fcf_history": [450e6, 380e6, 310e6], "fcf_yield": 8.5,
            "ebitda_margin": 38.0, "debt_to_equity": 0.25, "ev_ebitda": 7.2,
            "equity": 2_000e6, "total_assets": 5_000e6, "total_liabilities": 1_200e6,
            "working_capital": 600e6, "retained_earnings": 1_800e6,
            "ebit": 700e6, "revenue": 3_500e6, "market_cap": 4_200e6,
        },
        {
            "_label": "Guldgruva (D/E lite högt)",
            "fcf": 120e6, "fcf_history": [120e6, -30e6, 90e6], "fcf_yield": 4.2,
            "ebitda_margin": 22.0, "debt_to_equity": 0.72, "ev_ebitda": 9.5,
            "equity": 800e6, "total_assets": 2_500e6, "total_liabilities": 900e6,
            "working_capital": 200e6, "retained_earnings": 600e6,
            "ebit": 280e6, "revenue": 1_200e6, "market_cap": 1_800e6,
        },
        {
            "_label": "SaaS-bolag (negativ FCF, låg marginal)",
            "fcf": -80e6, "fcf_history": [-80e6, -120e6, -60e6], "fcf_yield": -2.1,
            "ebitda_margin": -5.0, "debt_to_equity": 1.8, "ev_ebitda": 45.0,
            "equity": 200e6, "total_assets": 800e6, "total_liabilities": 600e6,
            "working_capital": -50e6, "retained_earnings": -250e6,
            "ebit": -90e6, "revenue": 400e6, "market_cap": 1_500e6,
        },
        {
            "_label": "Oljebolag (högt skuldsatt men lönsamt)",
            "fcf": 900e6, "fcf_history": [900e6, 750e6, 400e6], "fcf_yield": 11.2,
            "ebitda_margin": 45.0, "debt_to_equity": 0.95, "ev_ebitda": 4.8,
            "equity": 5_000e6, "total_assets": 18_000e6, "total_liabilities": 9_500e6,
            "working_capital": 1_200e6, "retained_earnings": 8_000e6,
            "ebit": 2_500e6, "revenue": 12_000e6, "market_cap": 8_000e6,
        },
        {
            "_label": "Kärnkraftsbolag (saknar en del data)",
            "fcf": 200e6, "ebitda_margin": 30.0, "debt_to_equity": 0.45,
            "equity": 1_500e6, "market_cap": 3_000e6,
            # No Altman Z inputs — partial score expected
        },
        {
            "_label": "Insolventa bolaget (negativt eget kapital)",
            "fcf": -10e6, "fcf_history": [-10e6], "ebitda_margin": -2.0,
            "debt_to_equity": 3.5, "equity": -200e6, "total_assets": 500e6,
            "total_liabilities": 700e6, "working_capital": -100e6,
            "retained_earnings": -300e6, "ebit": -50e6,
            "revenue": 300e6, "market_cap": 50e6,
        },
    ]

    col_w = 46
    print(f"\n{'─'*80}")
    print(f"  STRENGTH SCORE  |  Composite weight: {STRENGTH_COMPOSITE_WEIGHT:.0%}  |  Gates: FCF>0, EBITDA>0%, D/E<0.6, Z>1.8, Equity>0")
    print(f"{'─'*80}")

    for case in test_cases:
        label = case.pop("_label")
        result = calculate_strength_score(case)
        gate_str = "ALL PASS" if result.passes_all_gates else f"{result.flag_count} FLAG(S)"
        z_str = f"Z={result.altman_z:.2f}" if result.altman_z is not None else "Z=n/a"
        print(f"\n  {label}")
        print(f"    Score : {result.score:>5.1f}/100   {gate_str}   {z_str}")
        print(f"    Breakdown: {result.breakdown}")
        if result.flags:
            print(f"    Flags    : {result.flags}")

    print(f"\n{'─'*80}\n")
