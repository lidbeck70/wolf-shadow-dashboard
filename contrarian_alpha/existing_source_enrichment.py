"""
existing_source_enrichment.py — Existing-source overlay (PR5 + PR6 wiring).

A *lightweight, additive* context overlay for the ``us_ca_resource`` universe
only. It reuses data the pipeline has ALREADY fetched from the project's
existing free/optional sources — this module itself makes no network calls and
adds no new dependencies or paid APIs. Every wired input is either passed in by
the engine (which fetches through the project's existing cached paths) or left
as a transparent *_NOT_AVAILABLE flag:

  * yfinance price snapshot the engine already computed (close / 52w high-low /
    20-day average volume) → 52-week drawdown, a liquidity flag, and — combined
    with the static CSV's optional ``shares_out_m`` — a coarse market-cap bucket.
  * EODHD/yfinance analyst upgrades-downgrades and short-interest dicts the hate
    stage already fetched (when enabled) → analyst-revision and short-interest
    flags.
  * Commodity relative strength (PR6): candidate vs. commodity-proxy ETF close
    series. The engine fetches the proxy series through its EXISTING yfinance
    price cache (``_fetch_price_df``) — shared across all rows of a run — and
    passes both close lists in here. The RS maths + classification live here as
    pure functions so they are unit-testable with no network. Context only,
    never a buy trigger → ``OUTPERFORMING_PROXY`` / ``LAGGING_PROXY`` /
    ``RS_NEUTRAL`` / ``COMMODITY_RS_NOT_AVAILABLE``.
  * Macro context (PR6): FRED T10Y2Y yield-curve snapshot. The engine fetches it
    ONCE per run via the existing disk-cached ``ember.fred_cache`` helper and
    passes the reading in. Classification is a pure function here →
    ``COMMODITY_MACRO_TAILWIND`` / ``COMMODITY_MACRO_HEADWIND`` / ``MACRO_NEUTRAL``
    / ``MACRO_CONTEXT_NOT_AVAILABLE``.
  * Sentiment (ApeWisdom/StockTwits) attention: the *adapter/interface* is wired
    and unit-tested (``classify_sentiment_attention`` → ``LOW_ATTENTION`` /
    ``HYPE_RISK`` / ``NORMAL_ATTENTION`` / ``SENTIMENT_NOT_AVAILABLE``), but the
    engine does NOT fetch it live: retail_sentiment fetches every source live via
    a thread pool with no pure per-ticker cached accessor, so wiring it into the
    scoring path would mean an uncontrolled network call / a larger refactor.
    Left as a documented placeholder that lights up the moment a caller supplies
    a sentiment dict (see TODO).

Design constraints (mirror resource_scoring.py / CLAUDE.md):
  * Additive and universe-gated. engine.py calls this only when
    ``config.universe == "us_ca_resource"``; Nordic scoring is untouched.
  * No fabricated precision. Missing inputs produce *_DATA_MISSING /
    *_NOT_AVAILABLE flags and a neutral overlay contribution — never an invented
    number. The single intentional penalty is genuinely-low liquidity.
  * The overlay score is kept *separate* from resource_composite. It never
    changes the deterministic PR3 composite math; it is "context/watchlist only,
    not a buy trigger." The PR6 RS/sentiment/macro readings are pure context —
    they are surfaced as flags and do NOT feed resource_overlay_score.
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

# Commodity relative-strength (PR6). RS = candidate %return − proxy %return over
# ``_RS_WINDOW`` trading days. Bounds are deliberately wide: RS is context, not a
# trigger, so only a clear divergence flips it off neutral.
_RS_WINDOW = 60
_RS_OUTPERFORM_PP = 5.0    # candidate ahead of proxy by ≥5pp → OUTPERFORMING_PROXY
_RS_LAGGING_PP = -5.0      # candidate behind proxy by ≥5pp → LAGGING_PROXY

# Sentiment attention (PR6). Composite/hype score thresholds (0-100) for the
# contrarian reading: quiet names are constructive, crowded/hyped names warn.
_SENTIMENT_HYPE_SCORE = 70.0
_SENTIMENT_LOW_SCORE = 25.0
_SENTIMENT_LOW_MSG_COUNT = 5.0   # very few messages → forgotten/low attention

# Macro context (PR6). Yield-curve 4-week change thresholds mirror ember.regime
# so the overlay and the Ember dashboard agree on what steepening/inverting is.
_YC_TAILWIND_PP = 0.05     # T10Y2Y steepening > +0.05pp 4W → commodity tailwind
_YC_HEADWIND_PP = -0.20    # T10Y2Y inverting < -0.20pp 4W → commodity headwind


@dataclass
class ExistingSourceOverlay:
    """Transparent existing-source overlay for one resource candidate."""

    market_cap_bucket: str = "unknown"          # nano|micro|small|mid|large|unknown
    liquidity_flag: str = "UNKNOWN"             # OK|THIN|LOW|UNKNOWN
    drawdown_52w_pct: float | None = None       # % from 52-week high (<=0)
    commodity_relative_strength: float | None = None  # candidate−proxy %return (pp)
    commodity_rs_flag: str = "COMMODITY_RS_NOT_AVAILABLE"  # OUTPERFORMING_PROXY|LAGGING_PROXY|RS_NEUTRAL|COMMODITY_RS_NOT_AVAILABLE
    short_interest_flag: str = "UNKNOWN"        # HIGH|ELEVATED|NORMAL|UNKNOWN
    analyst_revision_flag: str = "UNKNOWN"      # NET_DOWNGRADES|NET_UPGRADES|NEUTRAL|UNKNOWN
    sentiment_attention_flag: str = "SENTIMENT_NOT_AVAILABLE"  # LOW_ATTENTION|HYPE_RISK|NORMAL_ATTENTION|SENTIMENT_NOT_AVAILABLE
    macro_context_flag: str = "MACRO_CONTEXT_NOT_AVAILABLE"     # COMMODITY_MACRO_TAILWIND|COMMODITY_MACRO_HEADWIND|MACRO_NEUTRAL|MACRO_CONTEXT_NOT_AVAILABLE
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


# ── Commodity relative strength (PR6) — pure, no network ──────────────────────

def _window_return_pct(closes, window: int) -> float | None:
    """
    % price return over the last ``window`` trading days of a close series.

    ``closes`` is oldest-first. Returns None when the series is missing, too
    short, or the reference price is non-positive — never a fabricated number.
    """
    if not closes:
        return None
    seq = [c for c in (_opt_float(x) for x in closes) if c is not None and c > 0]
    if len(seq) < window + 1:
        return None
    start = seq[-(window + 1)]
    end = seq[-1]
    if start <= 0:
        return None
    return (end - start) / start * 100.0


def compute_commodity_rs(candidate_closes, proxy_closes, window: int = _RS_WINDOW) -> float | None:
    """
    Relative strength = candidate %return − commodity-proxy %return over
    ``window`` trading days, in percentage points (rounded to 1dp).

    Pure function: both close series are passed in (oldest-first). Returns None
    when either series is unavailable/too short so the caller can flag it as
    COMMODITY_RS_NOT_AVAILABLE rather than invent a reading.
    """
    cand = _window_return_pct(candidate_closes, window)
    prox = _window_return_pct(proxy_closes, window)
    if cand is None or prox is None:
        return None
    return round(cand - prox, 1)


def classify_commodity_rs(rs: float | None) -> str:
    """Categorical, context-only RS flag (never a buy trigger)."""
    if rs is None:
        return "COMMODITY_RS_NOT_AVAILABLE"
    if rs >= _RS_OUTPERFORM_PP:
        return "OUTPERFORMING_PROXY"
    if rs <= _RS_LAGGING_PP:
        return "LAGGING_PROXY"
    return "RS_NEUTRAL"


# ── Sentiment attention (PR6) — pure classifier over an injected dict ─────────

_SENTIMENT_LABELS = {
    "LOW_ATTENTION", "HYPE_RISK", "NORMAL_ATTENTION", "SENTIMENT_NOT_AVAILABLE",
}


def classify_sentiment_attention(sentiment: dict | None) -> str:
    """
    Contrarian retail-attention reading from an existing-source sentiment dict.

    Low attention (a forgotten name) is constructive context; excessive hype is
    a warning. Accepts an explicit label, a 0-100 composite/score, a hype flag,
    or a raw message_count — whatever the caller already has. Missing/empty →
    SENTIMENT_NOT_AVAILABLE. Pure: no network, no fabrication.
    """
    if not sentiment:
        return "SENTIMENT_NOT_AVAILABLE"

    label = str(sentiment.get("attention") or sentiment.get("label") or "").strip().upper()
    if label in _SENTIMENT_LABELS:
        return label

    score = _opt_float(sentiment.get("composite_score"))
    if score is None:
        score = _opt_float(sentiment.get("score"))
    if score is not None:
        if score >= _SENTIMENT_HYPE_SCORE:
            return "HYPE_RISK"
        if score <= _SENTIMENT_LOW_SCORE:
            return "LOW_ATTENTION"
        return "NORMAL_ATTENTION"

    if sentiment.get("hype") or sentiment.get("hype_alert"):
        return "HYPE_RISK"

    msg = _opt_float(sentiment.get("message_count"))
    if msg is not None:
        return "LOW_ATTENTION" if msg < _SENTIMENT_LOW_MSG_COUNT else "NORMAL_ATTENTION"

    return "SENTIMENT_NOT_AVAILABLE"


# ── Macro context (PR6) — pure classifier over an injected yield-curve dict ───

_MACRO_LABELS = {
    "COMMODITY_MACRO_TAILWIND", "COMMODITY_MACRO_HEADWIND",
    "MACRO_NEUTRAL", "MACRO_CONTEXT_NOT_AVAILABLE",
}


def classify_macro_context(macro: dict | None) -> str:
    """
    Broad commodity macro reading from an existing-source macro dict.

    Accepts an explicit label, an ember-style GREEN/AMBER/RED regime, or a raw
    T10Y2Y 4-week change in percentage points. A steepening curve is a commodity
    tailwind; a deepening inversion is a headwind. Missing/empty →
    MACRO_CONTEXT_NOT_AVAILABLE. Pure: no network, no fabrication.
    """
    if not macro:
        return "MACRO_CONTEXT_NOT_AVAILABLE"

    label = str(macro.get("regime") or macro.get("label") or "").strip().upper()
    if label in _MACRO_LABELS:
        return label
    if label == "GREEN":
        return "COMMODITY_MACRO_TAILWIND"
    if label == "RED":
        return "COMMODITY_MACRO_HEADWIND"
    if label == "AMBER":
        return "MACRO_NEUTRAL"

    change_4w = _opt_float(macro.get("t10y2y_change_4w"))
    if change_4w is not None:
        if change_4w > _YC_TAILWIND_PP:
            return "COMMODITY_MACRO_TAILWIND"
        if change_4w < _YC_HEADWIND_PP:
            return "COMMODITY_MACRO_HEADWIND"
        return "MACRO_NEUTRAL"

    return "MACRO_CONTEXT_NOT_AVAILABLE"


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
    candidate_closes: list | None = None,
    proxy_closes: list | None = None,
    rs_window: int = _RS_WINDOW,
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

    # ── Commodity relative strength (PR6) ──────────────────────────────────────
    # Candidate vs. commodity-proxy ETF return over rs_window. The engine passes
    # both close series (fetched through its existing yfinance price cache); the
    # maths/classification here are pure. Context only — never a buy trigger.
    ov.commodity_relative_strength = compute_commodity_rs(
        candidate_closes, proxy_closes, window=rs_window
    )
    ov.commodity_rs_flag = classify_commodity_rs(ov.commodity_relative_strength)
    flags.append(ov.commodity_rs_flag)

    # ── Sentiment attention (PR6) ──────────────────────────────────────────────
    # Adapter is wired + tested: a caller-supplied sentiment dict is classified
    # into a contrarian attention flag. The engine does NOT fetch it live (no
    # pure per-ticker cached retail_sentiment accessor → would be an uncontrolled
    # network call / larger refactor), so in the live pipeline this stays
    # SENTIMENT_NOT_AVAILABLE. TODO: wire once such an accessor exists.
    ov.sentiment_attention_flag = classify_sentiment_attention(sentiment)
    if ov.sentiment_attention_flag == "SENTIMENT_NOT_AVAILABLE":
        flags.append("SENTIMENT_NOT_AVAILABLE")

    # ── Macro context (PR6) ────────────────────────────────────────────────────
    # Broad commodity-macro reading from the FRED T10Y2Y yield-curve snapshot the
    # engine fetches once per run via the existing ember.fred_cache disk cache.
    ov.macro_context_flag = classify_macro_context(macro)
    if ov.macro_context_flag == "MACRO_CONTEXT_NOT_AVAILABLE":
        flags.append("MACRO_CONTEXT_NOT_AVAILABLE")

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
