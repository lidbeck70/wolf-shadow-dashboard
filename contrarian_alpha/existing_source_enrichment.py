"""
existing_source_enrichment.py — Existing-source overlay (PR5).

A *lightweight, additive* context overlay for the ``us_ca_resource`` universe
only. It reuses data the pipeline has ALREADY fetched from the project's
existing free/optional sources — it makes no new network calls of its own and
adds no new dependencies or paid APIs:

  * yfinance price snapshot the engine already computed (close / 52w high-low /
    20-day average volume) → 52-week drawdown, a liquidity flag, and — combined
    with the static CSV's optional ``shares_out_m`` — a coarse market-cap bucket.
  * EODHD/yfinance analyst upgrades-downgrades and short-interest dicts the hate
    stage already fetched (when enabled) → analyst-revision and short-interest
    flags.
  * Sentiment (ApeWisdom/StockTwits) and FRED macro context are surfaced as
    transparent *placeholders* only. Wiring those live would require network
    calls from this layer / a larger refactor of the retail_sentiment and
    alpha_regime engines, so they are intentionally deferred (see TODOs) rather
    than faked.

Design constraints (mirror resource_scoring.py / CLAUDE.md):
  * Additive and universe-gated. engine.py calls this only when
    ``config.universe == "us_ca_resource"``; Nordic scoring is untouched.
  * No fabricated precision. Missing inputs produce *_DATA_MISSING /
    *_NOT_AVAILABLE flags and a neutral overlay contribution — never an invented
    number. The single intentional penalty is genuinely-low liquidity.
  * The overlay score is kept *separate* from resource_composite. It never
    changes the deterministic PR3 composite math; it is "context/watchlist only,
    not a buy trigger."
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Reuse the project's liquidity threshold so the overlay and the flags page
# agree on what "too thin to trade" means.
try:
    from contrarian_alpha.flags import LIQUIDITY_THRESHOLD_USD
except Exception:  # pragma: no cover - defensive import fallback
    LIQUIDITY_THRESHOLD_USD = 500_000.0

# Overlay neutral midpoint. Signals nudge above/below; pure-unknown rows stay
# here rather than being penalised for missing data.
_OVERLAY_NEUTRAL = 50.0

# Market-cap buckets (USD). Coarse and transparent — only ever a bucket label,
# never a fabricated exact figure.
_MCAP_BUCKETS: list[tuple[float, str]] = [
    (50_000_000.0, "nano"),
    (300_000_000.0, "micro"),
    (2_000_000_000.0, "small"),
    (10_000_000_000.0, "mid"),
]

# Short-interest thresholds (% of float).
_SHORT_HIGH = 15.0
_SHORT_ELEVATED = 8.0

# Drawdown depth that marks a genuinely washed-out / contrarian name.
_DEEP_DRAWDOWN_PCT = -50.0


@dataclass
class ExistingSourceOverlay:
    """Transparent existing-source overlay for one resource candidate."""

    market_cap_bucket: str = "unknown"          # nano|micro|small|mid|large|unknown
    liquidity_flag: str = "UNKNOWN"             # OK|THIN|LOW|UNKNOWN
    drawdown_52w_pct: float | None = None       # % from 52-week high (<=0)
    commodity_relative_strength: float | None = None  # placeholder (not wired)
    short_interest_flag: str = "UNKNOWN"        # HIGH|ELEVATED|NORMAL|UNKNOWN
    analyst_revision_flag: str = "UNKNOWN"      # NET_DOWNGRADES|NET_UPGRADES|NEUTRAL|UNKNOWN
    sentiment_attention_flag: str = "NOT_WIRED"     # placeholder
    macro_context_flag: str = "NOT_WIRED"           # placeholder
    resource_overlay_score: float = _OVERLAY_NEUTRAL  # 0-100, separate from composite
    existing_source_flags: list[str] = field(default_factory=list)


def _opt_float(value) -> float | None:
    """Parse an optional numeric value; None/blank/garbage → None."""
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


def _market_cap_bucket(mcap_usd: float | None) -> str:
    if mcap_usd is None or mcap_usd <= 0:
        return "unknown"
    for ceiling, label in _MCAP_BUCKETS:
        if mcap_usd < ceiling:
            return label
    return "large"


def enrich_resource_candidate(
    close: float = 0.0,
    high_52w: float = 0.0,
    low_52w: float = 0.0,
    avg_volume_20d: float | None = None,
    meta: dict | None = None,
    analyst_data: dict | None = None,
    short_data: dict | None = None,
    market_cap_usd: float | None = None,
    sentiment: dict | None = None,
    macro: dict | None = None,
) -> ExistingSourceOverlay:
    """
    Build an ExistingSourceOverlay from data the pipeline already has in hand.

    All inputs are optional; this function performs NO network I/O so it is safe
    to unit-test without mocking external services. Missing inputs raise
    transparent flags and leave the overlay score at neutral rather than
    penalising the candidate — the sole intentional penalty is low liquidity.
    """
    meta = meta or {}
    flags: list[str] = []
    ov = ExistingSourceOverlay()

    # Coerce numeric inputs defensively — callers may pass raw/garbage values.
    close_f = _opt_float(close)
    high_f = _opt_float(high_52w)

    # ── 52-week drawdown (yfinance snapshot the engine already computed) ───────
    if close_f and close_f > 0 and high_f and high_f > 0:
        ov.drawdown_52w_pct = round((close_f - high_f) / high_f * 100.0, 1)
        if ov.drawdown_52w_pct <= _DEEP_DRAWDOWN_PCT:
            flags.append("DEEP_52W_DRAWDOWN")
    else:
        flags.append("DRAWDOWN_DATA_MISSING")

    # ── Liquidity flag (avg 20d volume × close, existing threshold) ────────────
    vol = _opt_float(avg_volume_20d)
    if vol is not None and close_f and close_f > 0:
        daily_usd = close_f * vol
        if daily_usd < LIQUIDITY_THRESHOLD_USD:
            ov.liquidity_flag = "LOW"
            flags.append("LOW_LIQUIDITY")
        elif daily_usd < LIQUIDITY_THRESHOLD_USD * 4:
            ov.liquidity_flag = "THIN"
            flags.append("THIN_LIQUIDITY")
        else:
            ov.liquidity_flag = "OK"
    else:
        ov.liquidity_flag = "UNKNOWN"
        flags.append("LIQUIDITY_DATA_MISSING")

    # ── Market-cap bucket ──────────────────────────────────────────────────────
    # Prefer a real market cap if the caller supplied one; otherwise estimate
    # coarsely from the static CSV's shares_out_m × current close (flagged as an
    # estimate, never presented as precise). Unknown when neither is available.
    mcap = _opt_float(market_cap_usd)
    if mcap is None:
        shares_m = _opt_float(meta.get("shares_out_m"))
        if shares_m is not None and shares_m > 0 and close_f and close_f > 0:
            mcap = shares_m * 1_000_000.0 * close_f
            flags.append("MARKET_CAP_ESTIMATED")
    ov.market_cap_bucket = _market_cap_bucket(mcap)
    if ov.market_cap_bucket == "unknown":
        flags.append("MARKET_CAP_DATA_MISSING")
    elif ov.market_cap_bucket == "nano":
        flags.append("NANO_CAP")

    # ── Short interest (EODHD → yfinance, already fetched by hate stage) ───────
    short_pct = _opt_float((short_data or {}).get("short_float_pct")) if short_data else None
    if short_pct is None:
        ov.short_interest_flag = "UNKNOWN"
        flags.append("SHORT_INTEREST_DATA_MISSING")
    elif short_pct >= _SHORT_HIGH:
        ov.short_interest_flag = "HIGH"
        flags.append("HIGH_SHORT_INTEREST")
    elif short_pct >= _SHORT_ELEVATED:
        ov.short_interest_flag = "ELEVATED"
        flags.append("ELEVATED_SHORT_INTEREST")
    else:
        ov.short_interest_flag = "NORMAL"

    # ── Analyst revisions (EODHD upgrades-downgrades, already fetched) ─────────
    if analyst_data:
        downs = int(_opt_float(analyst_data.get("downgrades_90d")) or 0)
        ups = int(_opt_float(analyst_data.get("upgrades_90d")) or 0)
        if downs > ups:
            ov.analyst_revision_flag = "NET_DOWNGRADES"
            flags.append("ANALYST_NET_DOWNGRADES")
        elif ups > downs:
            ov.analyst_revision_flag = "NET_UPGRADES"
        else:
            ov.analyst_revision_flag = "NEUTRAL"
    else:
        ov.analyst_revision_flag = "UNKNOWN"
        flags.append("ANALYST_DATA_MISSING")

    # ── Commodity relative strength — placeholder (not wired in PR5) ───────────
    # TODO: a real RS reading needs the commodity-proxy ETF price series
    # (alpha_regime.commodity_ratios), which is a network fetch — deferred to
    # keep this layer pure/testable. commodity_proxy metadata already ships on
    # the resource composite for future use.
    ov.commodity_relative_strength = None
    flags.append("COMMODITY_RS_NOT_AVAILABLE")

    # ── Sentiment attention — placeholder (not wired in PR5) ───────────────────
    # TODO: reuse retail_sentiment (ApeWisdom/StockTwits) attention/neglect once a
    # pure, cached per-ticker accessor exists; today its engine fetches live.
    if sentiment:
        attention = str(sentiment.get("attention") or sentiment.get("label") or "").upper()
        ov.sentiment_attention_flag = attention or "UNKNOWN"
    else:
        ov.sentiment_attention_flag = "NOT_WIRED"
        flags.append("SENTIMENT_NOT_WIRED")

    # ── Macro context — placeholder (not wired in PR5) ─────────────────────────
    # TODO: reuse FRED yield-curve / real-rate context (data_health / alpha_regime)
    # once a lightweight cached accessor is exposed.
    if macro:
        ov.macro_context_flag = str(macro.get("regime") or macro.get("label") or "UNKNOWN").upper()
    else:
        ov.macro_context_flag = "NOT_WIRED"
        flags.append("MACRO_CONTEXT_NOT_WIRED")

    # ── Overlay score — cautious, from clean signals only ──────────────────────
    # Contrarian nudges: deep drawdown, high short interest and net analyst
    # downgrades all signal genuine capitulation/neglect (Rule/Sprott style).
    # Missing inputs contribute nothing (stay neutral). Low liquidity is the one
    # intentional penalty — a name too thin to trade is a worse candidate.
    score = _OVERLAY_NEUTRAL
    signals = 0
    if ov.drawdown_52w_pct is not None:
        signals += 1
        if ov.drawdown_52w_pct <= _DEEP_DRAWDOWN_PCT:
            score += 12.0
        elif ov.drawdown_52w_pct <= -30.0:
            score += 6.0
    if ov.short_interest_flag == "HIGH":
        signals += 1
        score += 8.0
    elif ov.short_interest_flag == "ELEVATED":
        signals += 1
        score += 4.0
    elif ov.short_interest_flag == "NORMAL":
        signals += 1
    if ov.analyst_revision_flag == "NET_DOWNGRADES":
        signals += 1
        score += 6.0
    elif ov.analyst_revision_flag in ("NET_UPGRADES", "NEUTRAL"):
        signals += 1
    if ov.liquidity_flag == "LOW":
        score -= 15.0
    elif ov.liquidity_flag == "THIN":
        score -= 5.0

    if signals == 0:
        flags.append("OVERLAY_NO_SIGNAL")
    elif signals < 2:
        flags.append("OVERLAY_LOW_SIGNAL")

    ov.resource_overlay_score = round(max(0.0, min(100.0, score)), 1)
    ov.existing_source_flags = flags
    return ov
