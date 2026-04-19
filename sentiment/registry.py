"""
sentiment/registry.py
=====================
Collects all sentiment plugins into a single lookup dict.

Each value is a SENTIMENT_PLUGIN dict with keys:
  key, name, description, color, compute_score, compute_signal

Usage
-----
    from sentiment.registry import SENTIMENT_PLUGINS

    plugin = SENTIMENT_PLUGINS["ovtlyr_fg"]
    score  = plugin["compute_score"](ohlcv_df)   # -> float 0-100
    signal = plugin["compute_signal"](ohlcv_df)  # -> dict
"""

from .ovtlyr_clone import SENTIMENT_PLUGIN as _OVTLYR_PLUGIN
from .retail_flow  import SENTIMENT_PLUGIN as _RETAIL_PLUGIN
from .options_flow import SENTIMENT_PLUGIN as _OPTIONS_PLUGIN

SENTIMENT_PLUGINS: dict = {
    plugin["key"]: plugin
    for plugin in (
        _OVTLYR_PLUGIN,
        _RETAIL_PLUGIN,
        _OPTIONS_PLUGIN,
    )
}

__all__ = ["SENTIMENT_PLUGINS"]
