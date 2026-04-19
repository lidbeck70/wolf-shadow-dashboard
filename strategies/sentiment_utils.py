"""
strategies/sentiment_utils.py
==============================
Helpers for running sentiment plugins and integrating results into
strategy entry/exit signals — non-breaking by design.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def compute_sentiment_bias(
    df: pd.DataFrame,
    plugin_keys: List[str],
) -> Dict[str, Any]:
    """
    Run the requested sentiment plugins on *df* and return an aggregate.

    Parameters
    ----------
    df          : OHLCV DataFrame (DatetimeIndex, Open/High/Low/Close/Volume)
    plugin_keys : list of plugin keys, e.g. ["ovtlyr_fg", "retail_flow"]

    Returns
    -------
    dict with keys:
      aggregate_score  : float 0-100
      aggregate_bias   : "bullish" | "neutral" | "bearish"
      confidence       : float 0-1
      plugins          : {key: compute_signal() result}
      available        : bool (False when registry missing or no data)
    """
    _empty = {
        "aggregate_score": 50.0,
        "aggregate_bias": "neutral",
        "confidence": 0.0,
        "plugins": {},
        "available": False,
    }

    try:
        from sentiment.registry import SENTIMENT_PLUGINS
    except ImportError:
        return _empty

    if df is None or len(df) < 15:
        return _empty

    plugin_results: Dict[str, dict] = {}
    scores: List[float] = []

    for key in plugin_keys:
        plugin = SENTIMENT_PLUGINS.get(key)
        if plugin is None:
            continue
        try:
            sig = plugin["compute_signal"](df)
            plugin_results[key] = sig
            scores.append(float(sig.get("score", 50.0)))
        except Exception:
            pass

    if not scores:
        return {**_empty, "plugins": plugin_results}

    agg = sum(scores) / len(scores)

    if agg >= 62:
        bias = "bullish"
    elif agg <= 38:
        bias = "bearish"
    else:
        bias = "neutral"

    return {
        "aggregate_score": round(agg, 2),
        "aggregate_bias": bias,
        "confidence": round(abs(agg - 50.0) / 50.0, 3),
        "plugins": plugin_results,
        "available": True,
    }


def apply_sentiment_weight(
    entry_result: dict,
    sentiment_result: dict,
) -> dict:
    """
    Non-breaking sentiment enrichment for an entry/exit result dict.

    Nudges *confidence* by up to ±0.10 based on sentiment aggregate score.
    Never changes the *signal* key. Adds *sentiment* and *sentiment_adjusted*
    keys to the returned dict.

    Parameters
    ----------
    entry_result      : dict returned by a strategy entry_fn / exit_fn
    sentiment_result  : dict returned by compute_sentiment_bias()
    """
    out = dict(entry_result)

    if not sentiment_result.get("available"):
        out["sentiment_adjusted"] = False
        return out

    agg_score = float(sentiment_result.get("aggregate_score", 50.0))
    # (score - 50) in [-50, 50]; scale to nudge of ±0.10 on 0-1 confidence
    nudge = (agg_score - 50.0) * 0.002

    orig = float(entry_result.get("confidence", 0.0))
    out["confidence"] = round(max(0.0, min(1.0, orig + nudge)), 4)
    out["sentiment"] = sentiment_result
    out["sentiment_adjusted"] = True
    return out
