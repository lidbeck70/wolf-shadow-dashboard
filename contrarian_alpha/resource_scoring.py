"""
resource_scoring.py — Resource-composite v1 (PR3).

A *deterministic, transparent* scoring layer for the ``us_ca_resource`` universe
only (Rick Rule / Eric Sprott style screening). It is intentionally additive:
engine.py calls compute_resource_composite() only when
``config.universe == "us_ca_resource"`` and resource metadata exists, so Nordic
and existing Contrarian Alpha behavior are bit-for-bit unchanged.

Design constraints (see CLAUDE.md / PR3 brief):
  * No new external dependencies — works off the static CSV metadata plus the
    scores the pipeline already computes (hate, catalyst, quality, value).
  * Never fabricate precision. When cash/runway, dilution or jurisdiction data
    is missing we score conservatively toward neutral and raise a *_DATA_MISSING
    / *_UNKNOWN flag plus a low confidence, rather than inventing a number.
  * Optional CSV columns are read if present (cash_musd, quarterly_burn_musd,
    debt_musd, shares_out_m, shares_yoy_growth_pct, jurisdiction) so the seed
    list can be enriched later (PR4) without code changes.
  * Stage weights: explorers/developers weight hate/catalyst/survival/
    jurisdiction/commodity over ROIC/FCF quality-value; producers/royalty lean
    on mature quality/value where data exists.

This is "resource scoring v1 — not a buy signal; watchlist/ranking only."
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Neutral score used whenever a factor cannot be measured. Chosen as a mild
# midpoint (not 50) so genuinely-scored names still rank above pure unknowns.
_NEUTRAL = 50.0
# Low confidence attached to any neutral/missing-data factor.
_CONF_MISSING = 0.2
_CONF_FULL = 1.0


# ─── Commodity necessity map ──────────────────────────────────────────────────
# Higher = more strategic / necessary (energy transition, monetary, critical
# minerals). Transparent and deterministic — no market data involved.

_COMMODITY_SCORES: dict[str, float] = {
    # Energy / nuclear
    "uranium": 95.0,
    # Electrification / energy transition
    "copper": 92.0,
    "lithium": 90.0,
    "rare earth": 90.0,
    "rare earths": 90.0,
    "critical minerals": 90.0,
    "nickel": 85.0,
    "cobalt": 85.0,
    "graphite": 85.0,
    "tin": 80.0,
    "zinc": 75.0,
    # Precious / monetary (Sprott/Rule core)
    "gold": 85.0,
    "silver": 82.0,
    "platinum": 80.0,
    "palladium": 80.0,
    "pgm": 80.0,
    # Energy
    "oil": 80.0,
    "oil & gas": 80.0,
    "gas": 78.0,
    "natural gas": 78.0,
    "energy": 80.0,
    # Agriculture / fertilizer necessity
    "potash": 80.0,
    "phosphate": 78.0,
    # Out of favor / lower necessity
    "coal": 55.0,
    "diamonds": 50.0,
}
_COMMODITY_DEFAULT = 55.0

# Strategic set used only for a transparent STRATEGIC_COMMODITY flag.
_STRATEGIC_COMMODITIES = {
    "uranium", "copper", "lithium", "rare earth", "rare earths",
    "critical minerals", "nickel", "cobalt", "graphite",
    "gold", "silver", "oil", "oil & gas", "gas", "natural gas", "energy",
}


# ─── Jurisdiction map ─────────────────────────────────────────────────────────
# Country/exchange baseline (Fraser-Institute style intuition, kept coarse and
# transparent). Canada/US/Australia get a high baseline; unknown stays neutral.

_JURISDICTION_COUNTRY: dict[str, float] = {
    "CA": 85.0, "CANADA": 85.0,
    "US": 85.0, "USA": 85.0, "UNITED STATES": 85.0,
    "AU": 85.0, "AUS": 85.0, "AUSTRALIA": 85.0,
    "NZ": 78.0, "NEW ZEALAND": 78.0,
    "GB": 75.0, "UK": 75.0, "UNITED KINGDOM": 75.0,
    "FI": 75.0, "SE": 75.0, "NO": 75.0,
}
# Optional fine-grained jurisdiction field (province/state/region).
_JURISDICTION_REGION: dict[str, float] = {
    "quebec": 90.0, "ontario": 88.0, "saskatchewan": 88.0,
    "nevada": 90.0, "arizona": 85.0, "alaska": 82.0,
    "western australia": 90.0, "new south wales": 85.0,
    "chile": 68.0, "peru": 62.0, "mexico": 58.0, "argentina": 55.0,
    "drc": 30.0, "congo": 30.0, "mali": 35.0, "burkina faso": 35.0,
    "russia": 25.0, "venezuela": 20.0,
}
_JURISDICTION_EXCHANGE_COUNTRY: dict[str, str] = {
    ".TO": "CA", ".V": "CA", ".CN": "CA", "TSX": "CA", "TSXV": "CA", "CSE": "CA",
    ".AX": "AU", "ASX": "AU",
    "NYSE": "US", "NASDAQ": "US", "NYSE AMERICAN": "US", "OTC": "US",
}


# ─── Stage weight profiles ────────────────────────────────────────────────────
# Factors: hate, catalyst, survival, dilution, jurisdiction, commodity,
#          quality, value. Each profile sums to 1.0.
# Pre-revenue juniors weight survival / jurisdiction / catalyst / commodity;
# producers & royalty lean on mature quality / value where data exists.

_STAGE_WEIGHTS: dict[str, dict[str, float]] = {
    "explorer": {
        "hate": 0.15, "catalyst": 0.20, "survival": 0.20, "dilution": 0.10,
        "jurisdiction": 0.15, "commodity": 0.15, "quality": 0.025, "value": 0.025,
    },
    "developer": {
        "hate": 0.12, "catalyst": 0.18, "survival": 0.18, "dilution": 0.10,
        "jurisdiction": 0.14, "commodity": 0.12, "quality": 0.08, "value": 0.08,
    },
    "producer": {
        "hate": 0.10, "catalyst": 0.15, "survival": 0.05, "dilution": 0.10,
        "jurisdiction": 0.15, "commodity": 0.15, "quality": 0.15, "value": 0.15,
    },
    "energy": {
        "hate": 0.10, "catalyst": 0.15, "survival": 0.05, "dilution": 0.10,
        "jurisdiction": 0.15, "commodity": 0.15, "quality": 0.15, "value": 0.15,
    },
    "services": {
        "hate": 0.10, "catalyst": 0.12, "survival": 0.08, "dilution": 0.10,
        "jurisdiction": 0.15, "commodity": 0.05, "quality": 0.20, "value": 0.20,
    },
    "royalty": {
        "hate": 0.08, "catalyst": 0.12, "survival": 0.05, "dilution": 0.05,
        "jurisdiction": 0.20, "commodity": 0.15, "quality": 0.20, "value": 0.15,
    },
    # Unknown/empty stage — balanced fallback.
    "_default": {
        "hate": 0.12, "catalyst": 0.15, "survival": 0.13, "dilution": 0.10,
        "jurisdiction": 0.15, "commodity": 0.15, "quality": 0.10, "value": 0.10,
    },
}


# ─── Result model ─────────────────────────────────────────────────────────────

@dataclass
class ResourceScore:
    """Deterministic resource-composite v1 output for one instrument."""

    resource_composite: float = 0.0     # 0-100 blended score
    survival_score: float = _NEUTRAL
    dilution_score: float = _NEUTRAL
    jurisdiction_score: float = _NEUTRAL
    commodity_score: float = _COMMODITY_DEFAULT
    resource_confidence: float = 0.0    # 0-1 mean of factor confidences
    stage_profile: str = "_default"     # which weight table was used
    flags: list[str] = field(default_factory=list)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _opt_float(value) -> float | None:
    """Parse an optional numeric metadata value; None/blank/garbage → None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def score_commodity(primary: str, secondary: str = "") -> tuple[float, list[str]]:
    """
    Map primary (and secondary as a mild tie-breaker) commodity to a necessity
    score. Unknown commodity → neutral default + COMMODITY_UNKNOWN flag.
    """
    flags: list[str] = []
    p = (primary or "").strip().lower()
    if not p:
        return _COMMODITY_DEFAULT, ["COMMODITY_UNKNOWN"]

    base = _COMMODITY_SCORES.get(p)
    if base is None:
        return _COMMODITY_DEFAULT, ["COMMODITY_UNKNOWN"]

    if p in _STRATEGIC_COMMODITIES:
        flags.append("STRATEGIC_COMMODITY")

    # A recognised strategic secondary nudges the score up slightly (diversified
    # optionality) but never above 98 — keep it transparent, not compounding.
    s = (secondary or "").strip().lower()
    if s and s in _COMMODITY_SCORES and _COMMODITY_SCORES[s] >= 85.0:
        base = min(98.0, base + 3.0)

    return round(base, 1), flags


def score_jurisdiction(
    country: str, exchange: str = "", jurisdiction: str = "",
) -> tuple[float, float, list[str]]:
    """
    Return (score, confidence, flags). Priority: explicit fine-grained
    jurisdiction field → country code/name → exchange-implied country →
    neutral + JURISDICTION_UNKNOWN.
    """
    flags: list[str] = []

    region = (jurisdiction or "").strip().lower()
    if region and region in _JURISDICTION_REGION:
        score = _JURISDICTION_REGION[region]
        if score < 50.0:
            flags.append("HIGH_RISK_JURISDICTION")
        return round(score, 1), _CONF_FULL, flags

    c = (country or "").strip().upper()
    if c and c in _JURISDICTION_COUNTRY:
        return round(_JURISDICTION_COUNTRY[c], 1), _CONF_FULL, flags

    ex = (exchange or "").strip().upper()
    implied = _JURISDICTION_EXCHANGE_COUNTRY.get(ex)
    if implied and implied in _JURISDICTION_COUNTRY:
        flags.append("JURISDICTION_FROM_EXCHANGE")
        return round(_JURISDICTION_COUNTRY[implied], 1), 0.6, flags

    flags.append("JURISDICTION_UNKNOWN")
    return _NEUTRAL, _CONF_MISSING, flags


def score_survival(
    meta: dict, stage: str,
) -> tuple[float, float, list[str]]:
    """
    Cash-runway survival score from optional CSV columns cash_musd,
    quarterly_burn_musd, debt_musd. When those are absent we cannot reliably
    compute runway, so we return a *stage-aware neutral* + SURVIVAL_DATA_MISSING
    and low confidence rather than a fabricated number.
    """
    flags: list[str] = []
    cash = _opt_float(meta.get("cash_musd"))
    burn = _opt_float(meta.get("quarterly_burn_musd"))
    debt = _opt_float(meta.get("debt_musd"))
    stg = (stage or "").strip().lower()

    if cash is None or burn is None or burn <= 0:
        # No reliable runway input. Producers/royalty are typically self-funding
        # → slightly higher neutral; pre-revenue juniors depend on raises → lower.
        flags.append("SURVIVAL_DATA_MISSING")
        if stg in ("producer", "royalty", "energy"):
            return 60.0, _CONF_MISSING, flags
        return 45.0, _CONF_MISSING, flags

    # Net cash after subtracting debt (if provided); floored at 0.
    net_cash = cash - debt if debt is not None else cash
    if net_cash < 0:
        flags.append("NET_DEBT_POSITION")
        net_cash = 0.0

    runway_q = net_cash / burn  # quarters of runway
    if runway_q >= 8:       # >= 2 years
        score = 95.0
    elif runway_q >= 6:
        score = 85.0
    elif runway_q >= 4:     # ~1 year
        score = 70.0
    elif runway_q >= 2:
        score = 50.0
    elif runway_q >= 1:
        score = 30.0
        flags.append("SHORT_RUNWAY")
    else:
        score = 15.0
        flags.append("CRITICAL_RUNWAY")

    return round(score, 1), _CONF_FULL, flags


def score_dilution(
    meta: dict, stage: str,
) -> tuple[float, float, list[str]]:
    """
    Share-count / dilution risk from optional CSV column shares_yoy_growth_pct
    (preferred). yfinance sharesOutstanding trend is intentionally NOT used in
    v1 (unreliable point-in-time value); when no reliable input exists we return
    a stage-aware neutral + DILUTION_DATA_MISSING.
    """
    flags: list[str] = []
    yoy = _opt_float(meta.get("shares_yoy_growth_pct"))
    stg = (stage or "").strip().lower()

    if yoy is None:
        flags.append("DILUTION_DATA_MISSING")
        # Explorers/developers dilute aggressively → conservative neutral;
        # producers/royalty rarely dilute → mildly favorable neutral.
        if stg in ("explorer", "developer"):
            return 40.0, _CONF_MISSING, flags
        return 60.0, _CONF_MISSING, flags

    if yoy <= 0:            # net buyback / flat
        score = 95.0
    elif yoy <= 5:
        score = 80.0
    elif yoy <= 15:
        score = 55.0
    elif yoy <= 30:
        score = 30.0
        flags.append("HIGH_DILUTION")
    else:
        score = 12.0
        flags.append("SEVERE_DILUTION")

    return round(score, 1), _CONF_FULL, flags


def get_stage_weights(stage: str) -> tuple[str, dict[str, float]]:
    """Return (profile_key, weight_dict) for a resource stage."""
    stg = (stage or "").strip().lower()
    if stg in _STAGE_WEIGHTS:
        return stg, _STAGE_WEIGHTS[stg]
    return "_default", _STAGE_WEIGHTS["_default"]


# ─── Composite ────────────────────────────────────────────────────────────────

def compute_resource_composite(
    stage: str,
    meta: dict | None,
    country: str = "",
    exchange: str = "",
    primary_commodity: str = "",
    secondary_commodity: str = "",
    hate_score: float = 0.0,
    catalyst_score: float = 0.0,
    quality_score: float | None = None,
    value_score: float | None = None,
) -> ResourceScore:
    """
    Blend resource-specific factors (survival, dilution, jurisdiction, commodity)
    with the pipeline's existing hate/catalyst/quality/value scores using
    stage-aware weights.

    quality_score / value_score may be None when no fundamentals were available
    (common for data-sparse US/CA rows); they are then treated as neutral so a
    producer is not falsely penalised, and a QUALITY_DATA_MISSING /
    VALUE_DATA_MISSING flag is raised for transparency.

    Returns a ResourceScore. Never raises — degrades to neutral + flags.
    """
    meta = meta or {}
    jurisdiction_field = str(meta.get("jurisdiction") or "")

    survival, conf_surv, f_surv = score_survival(meta, stage)
    dilution, conf_dil, f_dil = score_dilution(meta, stage)
    juris, conf_jur, f_jur = score_jurisdiction(country, exchange, jurisdiction_field)
    commodity, f_comm = score_commodity(primary_commodity, secondary_commodity)

    flags: list[str] = [*f_surv, *f_dil, *f_jur, *f_comm]

    # Commodity confidence: full when recognised, low when unknown.
    conf_comm = _CONF_MISSING if "COMMODITY_UNKNOWN" in f_comm else _CONF_FULL

    # Quality / value: substitute neutral when data missing (do not penalise).
    q = quality_score
    if q is None:
        q = _NEUTRAL
        flags.append("QUALITY_DATA_MISSING")
    v = value_score
    if v is None:
        v = _NEUTRAL
        flags.append("VALUE_DATA_MISSING")

    profile_key, w = get_stage_weights(stage)

    factors = {
        "hate": float(hate_score),
        "catalyst": float(catalyst_score),
        "survival": survival,
        "dilution": dilution,
        "jurisdiction": juris,
        "commodity": commodity,
        "quality": float(q),
        "value": float(v),
    }
    composite = sum(factors[k] * w[k] for k in w)

    # Overall confidence: mean of the resource-specific factor confidences only
    # (hate/catalyst are always present; quality/value handled via flags above).
    resource_confidence = round(
        (conf_surv + conf_dil + conf_jur + conf_comm) / 4.0, 2
    )
    if resource_confidence < 0.5 and "LOW_DATA_CONFIDENCE" not in flags:
        flags.append("LOW_DATA_CONFIDENCE")

    # ── Commodity/regime trigger — deferred (PR4) ─────────────────────────────
    # A lightweight commodity-ratio / alpha_regime context flag (e.g. gold vs
    # gold-miners, uranium spot regime) would live here. The existing pipeline
    # exposes an OVTLYR Viking regime per-ticker but no commodity-level ratio
    # signal that can be reused without a larger refactor, so we intentionally
    # leave a placeholder rather than overbuild. See PR3 brief / CLAUDE.md.

    return ResourceScore(
        resource_composite=round(composite, 2),
        survival_score=survival,
        dilution_score=dilution,
        jurisdiction_score=juris,
        commodity_score=commodity,
        resource_confidence=resource_confidence,
        stage_profile=profile_key,
        flags=flags,
    )
