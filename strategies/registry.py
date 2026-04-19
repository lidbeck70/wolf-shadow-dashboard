"""
Strategy registry — collects all strategy descriptors into a single dict.

Each value is a STRATEGY dict with keys:
  key, name, description, color, params, entry_fn, exit_fn, risk_fn
"""

from .wolf   import STRATEGY as WOLF_STRATEGY
from .alpha  import STRATEGY as ALPHA_STRATEGY
from .viking import STRATEGY as VIKING_STRATEGY

STRATEGIES: dict = {
    "Wolf":   WOLF_STRATEGY,
    "Alpha":  ALPHA_STRATEGY,
    "Viking": VIKING_STRATEGY,
}
