"""
resource_watchlists.py — practical watchlist slices for the US/CA Resource mode.

Pure helpers (no Streamlit / pandas) so they can be unit-tested in isolation.
They read the *existing* output fields on ``ContrairianAlphaResult`` rows and
group them into watchlist views — they never change scoring math and are used
for the ``us_ca_resource`` universe only.

Every view is a *watchlist slice, not a buy signal*. Missing/None values are
handled gracefully (treated as "unknown", never as a negative signal).

Entry point:
    from contrarian_alpha.resource_watchlists import build_resource_watchlists
    views = build_resource_watchlists(results)   # dict[key -> list[dict rows]]

The companion :data:`WATCHLIST_META` maps each key to its display title and
caption, and :data:`WATCHLIST_COLUMNS` to the recommended (mobile-friendly)
column order.
"""
from __future__ import annotations

from typing import Any

# ─── Transparent defaults (no exact thresholds exist upstream) ───────────────
# Deep drawdown default per PR15 brief: <= -40% counts as a "deep" 52w drawdown.
DEEP_DRAWDOWN_PCT = -40.0
# "High" resource composite for the low-data-quality validation slice.
HIGH_COMPOSITE = 60.0

# commodity_rs_flag values that count as improving/strong relative strength.
_STRONG_RS_FLAGS = {"OUTPERFORMING_PROXY", "RS_NEUTRAL"}

# liquidity_flag values that warrant a low-liquidity warning.
_LOW_LIQUIDITY_FLAGS = {"LOW", "THIN"}

# Flags/notes that mean "verify this manually before trusting the score".
_VALIDATION_TOKENS = (
    "needs_validation",
    "DATA_AS_OF_MISSING",
    "SURVIVAL_DATA_MISSING",
    "DILUTION_DATA_MISSING",
    "LOW_DATA_CONFIDENCE",
    "QUALITY_DATA_MISSING",
    "VALUE_DATA_MISSING",
)

# Subset used to flag "high score but the data behind it is thin".
_MISSING_DATA_TOKENS = (
    "SURVIVAL_DATA_MISSING",
    "DILUTION_DATA_MISSING",
    "DATA_AS_OF_MISSING",
    "QUALITY_DATA_MISSING",
    "VALUE_DATA_MISSING",
    "LOW_DATA_CONFIDENCE",
)


# ─── Field access (works on dataclass objects and plain dicts) ───────────────

def _get(row: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from an object (getattr) or a mapping (get)."""
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _num(value: Any) -> float | None:
    """Coerce to float, or None for missing/non-numeric/NaN values."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN != NaN


def _flag_tokens(row: Any) -> list[str]:
    """Collect all flag/note strings attached to a row (for token matching)."""
    tokens: list[str] = []
    for name in ("resource_flags", "existing_source_flags", "all_flags"):
        val = _get(row, name, None)
        if isinstance(val, (list, tuple)):
            tokens.extend(str(t) for t in val)
    notes = _get(row, "resource_notes", "")
    if notes:
        tokens.append(str(notes))
    return tokens


def _has_token(row: Any, tokens: tuple[str, ...]) -> bool:
    haystack = " ".join(_flag_tokens(row)).lower()
    return any(tok.lower() in haystack for tok in tokens)


def _key_flags(row: Any, limit: int = 3) -> str:
    """Compact, de-duplicated flag string for table display."""
    seen: list[str] = []
    for name in ("resource_flags", "existing_source_flags"):
        val = _get(row, name, None)
        if isinstance(val, (list, tuple)):
            for t in val:
                s = str(t)
                if s and s not in seen:
                    seen.append(s)
    if not seen:
        return "—"
    shown = seen[:limit]
    if len(seen) > limit:
        shown.append(f"+{len(seen) - limit}")
    return ", ".join(shown)


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Normalise a result row into a display dict with graceful None handling."""
    return {
        "ticker":            _get(row, "ticker", "") or "",
        "name":              _get(row, "name", "") or "",
        "stage":             _get(row, "stage", "") or "",
        "commodity":         _get(row, "primary_commodity", "") or "",
        "resource_composite":      _num(_get(row, "resource_composite")),
        "resource_overlay_score":  _num(_get(row, "resource_overlay_score")),
        "drawdown_52w_pct":        _num(_get(row, "drawdown_52w_pct")),
        "commodity_relative_strength": _num(_get(row, "commodity_relative_strength")),
        "commodity_rs_flag":       _get(row, "commodity_rs_flag", "") or "",
        "resource_data_quality":   _get(row, "resource_data_quality", "") or "",
        "liquidity_flag":          _get(row, "liquidity_flag", "") or "",
        "flags":                   _key_flags(row),
    }


# ─── View predicates ─────────────────────────────────────────────────────────

def _is_deep_drawdown_strong_rs(row: Any) -> bool:
    dd = _num(_get(row, "drawdown_52w_pct"))
    if dd is None or dd > DEEP_DRAWDOWN_PCT:
        return False
    return (_get(row, "commodity_rs_flag", "") or "") in _STRONG_RS_FLAGS


def _is_outperforming_proxy(row: Any) -> bool:
    return (_get(row, "commodity_rs_flag", "") or "") == "OUTPERFORMING_PROXY"


def _is_high_score_low_quality(row: Any) -> bool:
    comp = _num(_get(row, "resource_composite"))
    if comp is None or comp < HIGH_COMPOSITE:
        return False
    dq = (_get(row, "resource_data_quality", "") or "").upper()
    return dq == "LOW" or _has_token(row, _MISSING_DATA_TOKENS)


def _needs_validation(row: Any) -> bool:
    return _has_token(row, _VALIDATION_TOKENS)


def _is_low_liquidity(row: Any) -> bool:
    return (_get(row, "liquidity_flag", "") or "").upper() in _LOW_LIQUIDITY_FLAGS


# ─── Public metadata (Swedish titles, consistent with existing UI) ───────────

WATCHLIST_ORDER = [
    "top_composite",
    "deep_drawdown_strong_rs",
    "outperforming_proxy",
    "high_score_low_quality",
    "needs_validation",
    "low_liquidity",
]

WATCHLIST_META: dict[str, dict[str, str]] = {
    "top_composite": {
        "title": "Topp Resource Composite",
        "caption": "Sorterad på resource_composite (fallande). Watchlist-slice, "
                   "ej köpsignal.",
    },
    "deep_drawdown_strong_rs": {
        "title": "Djup nedgång + stark/förbättrad RS",
        "caption": f"52v drawdown ≤ {DEEP_DRAWDOWN_PCT:.0f}% och commodity-RS "
                   "OUTPERFORMING_PROXY/RS_NEUTRAL. Watchlist-slice, ej köpsignal.",
    },
    "outperforming_proxy": {
        "title": "Slår commodity-proxy",
        "caption": "commodity_rs_flag = OUTPERFORMING_PROXY. Watchlist-slice, "
                   "ej köpsignal.",
    },
    "high_score_low_quality": {
        "title": "Hög score men låg datakvalitet",
        "caption": f"resource_composite ≥ {HIGH_COMPOSITE:.0f} men datakvalitet "
                   "LOW eller saknad data — kräver manuell validering.",
    },
    "needs_validation": {
        "title": "Kräver manuell validering",
        "caption": "Flaggor/noteringar som needs_validation, DATA_AS_OF_MISSING, "
                   "SURVIVAL/DILUTION_DATA_MISSING, LOW_DATA_CONFIDENCE.",
    },
    "low_liquidity": {
        "title": "Låg likviditet – varning",
        "caption": "liquidity_flag = LOW/THIN. Watchlist-slice, ej köpsignal.",
    },
}

# Recommended (mobile-friendly) column order for display.
WATCHLIST_COLUMNS = [
    "ticker",
    "name",
    "stage",
    "commodity",
    "resource_composite",
    "resource_overlay_score",
    "drawdown_52w_pct",
    "commodity_rs_flag",
    "resource_data_quality",
    "liquidity_flag",
    "flags",
]


# ─── Builder ─────────────────────────────────────────────────────────────────

def build_resource_watchlists(results: Any) -> dict[str, list[dict[str, Any]]]:
    """Group resource result rows into watchlist views.

    Args:
        results: an iterable of ``ContrairianAlphaResult``-like rows (objects or
            dicts). ``None`` / empty is tolerated and yields empty views.

    Returns:
        An ordered dict keyed by :data:`WATCHLIST_ORDER`; each value is a list of
        normalised display dicts (see :func:`_row_to_dict`). Views never raise on
        missing columns/None values — such rows simply drop out of the filtered
        views (and sort last in the composite view).
    """
    rows = list(results) if results else []

    def _sort_by_composite(items: list[Any]) -> list[dict[str, Any]]:
        def _key(r: Any) -> float:
            v = _num(_get(r, "resource_composite"))
            return v if v is not None else float("-inf")
        return [_row_to_dict(r) for r in sorted(rows, key=_key, reverse=True)]

    views: dict[str, list[dict[str, Any]]] = {
        "top_composite": _sort_by_composite(rows),
        "deep_drawdown_strong_rs": [
            _row_to_dict(r) for r in rows if _is_deep_drawdown_strong_rs(r)
        ],
        "outperforming_proxy": [
            _row_to_dict(r) for r in rows if _is_outperforming_proxy(r)
        ],
        "high_score_low_quality": [
            _row_to_dict(r) for r in rows if _is_high_score_low_quality(r)
        ],
        "needs_validation": [
            _row_to_dict(r) for r in rows if _needs_validation(r)
        ],
        "low_liquidity": [
            _row_to_dict(r) for r in rows if _is_low_liquidity(r)
        ],
    }
    return {k: views[k] for k in WATCHLIST_ORDER}
