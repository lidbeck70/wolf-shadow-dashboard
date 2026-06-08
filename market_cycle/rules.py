"""
market_cycle/rules.py
=====================
14-phase market psychology cycle rules.

Each phase has:
  conditions  — list of {field, op, value} dicts
  weight      — float multiplier for the score
  description — brief description of investor psychology
  emoji       — visual identifier
  color       — hex color for UI rendering

Operators: "gt", "gte", "lt", "lte", "between" (value=[min, max])
Fields must match keys returned by indicators.compute_indicators().
"""

from __future__ import annotations

MARKET_CYCLE_RULES: dict[str, dict] = {

    # ── 1 · DISBELIEF ─────────────────────────────────────────────────────────
    # Market has bottomed; most investors refuse to believe the recovery is real.
    "DISBELIEF": {
        "description": "Market has bottomed but the crowd remains deeply skeptical of any recovery",
        "emoji": "🤨",
        "color": "#4a7c59",
        "weight": 1.0,
        "conditions": [
            {"field": "rsi",           "op": "between", "value": [35, 56]},
            {"field": "momentum_30",   "op": "between", "value": [2, 22]},
            {"field": "momentum_90",   "op": "between", "value": [-35, 5]},
            {"field": "price_vs_ma200","op": "between", "value": [-18, 8]},
            {"field": "drawdown_90",   "op": "between", "value": [-30, -6]},
            {"field": "macd_diff",     "op": "gt",      "value": 0},
        ],
    },

    # ── 2 · HOPE ──────────────────────────────────────────────────────────────
    # First signs of sustained recovery; cautious optimism returns.
    "HOPE": {
        "description": "Early signs of recovery; cautious optimism begins to emerge",
        "emoji": "🌱",
        "color": "#5a9a6a",
        "weight": 1.0,
        "conditions": [
            {"field": "rsi",           "op": "between", "value": [45, 62]},
            {"field": "momentum_30",   "op": "between", "value": [5, 25]},
            {"field": "momentum_60",   "op": "between", "value": [-5, 20]},
            {"field": "price_vs_ma200","op": "between", "value": [-8, 18]},
            {"field": "price_vs_ma50", "op": "between", "value": [-5, 15]},
            {"field": "macd_diff",     "op": "gt",      "value": 0},
        ],
    },

    # ── 3 · OPTIMISM ──────────────────────────────────────────────────────────
    # Uptrend clearly established; mainstream investors start participating.
    "OPTIMISM": {
        "description": "Uptrend confirmed; mainstream investors begin to enter the market",
        "emoji": "😊",
        "color": "#78b44a",
        "weight": 1.0,
        "conditions": [
            {"field": "rsi",           "op": "between", "value": [52, 68]},
            {"field": "momentum_30",   "op": "between", "value": [8, 28]},
            {"field": "momentum_60",   "op": "between", "value": [8, 35]},
            {"field": "momentum_90",   "op": "between", "value": [5, 35]},
            {"field": "price_vs_ma200","op": "between", "value": [5, 28]},
            {"field": "price_vs_ma50", "op": "between", "value": [2, 18]},
        ],
    },

    # ── 4 · BELIEF ────────────────────────────────────────────────────────────
    # Strong conviction; volume increases as more capital flows in.
    "BELIEF": {
        "description": "Strong trend conviction; broad participation and increasing volume",
        "emoji": "💪",
        "color": "#00E5FF",
        "weight": 1.0,
        "conditions": [
            {"field": "rsi",             "op": "between", "value": [58, 74]},
            {"field": "momentum_30",     "op": "between", "value": [10, 35]},
            {"field": "momentum_60",     "op": "between", "value": [12, 45]},
            {"field": "momentum_90",     "op": "between", "value": [15, 50]},
            {"field": "price_vs_ma200",  "op": "between", "value": [12, 40]},
            {"field": "price_vs_ma50",   "op": "between", "value": [5, 22]},
            {"field": "volume_vs_avg20", "op": "gte",     "value": 1.1},
        ],
    },

    # ── 5 · THRILL ────────────────────────────────────────────────────────────
    # Excitement and strong gains; investors feel invincible.
    "THRILL": {
        "description": "Exciting gains; retail participation surges and FOMO drives momentum",
        "emoji": "🚀",
        "color": "#e8a020",
        "weight": 1.0,
        "conditions": [
            {"field": "rsi",             "op": "between", "value": [63, 82]},
            {"field": "momentum_30",     "op": "between", "value": [18, 55]},
            {"field": "momentum_60",     "op": "between", "value": [22, 65]},
            {"field": "momentum_90",     "op": "between", "value": [28, 75]},
            {"field": "price_vs_ma200",  "op": "between", "value": [28, 65]},
            {"field": "price_vs_ma50",   "op": "between", "value": [12, 38]},
        ],
    },

    # ── 6 · EUPHORIA ──────────────────────────────────────────────────────────
    # Peak — maximum financial risk; everyone is bullish.
    "EUPHORIA": {
        "description": "Market peak — maximum financial risk; extreme bullishness and overvaluation",
        "emoji": "🎆",
        "color": "#ff6030",
        "weight": 1.2,
        "conditions": [
            {"field": "rsi",             "op": "gte",     "value": 72},
            {"field": "momentum_30",     "op": "gte",     "value": 22},
            {"field": "momentum_60",     "op": "gte",     "value": 30},
            {"field": "momentum_90",     "op": "gte",     "value": 40},
            {"field": "price_vs_ma200",  "op": "gte",     "value": 45},
            {"field": "drawdown_90",     "op": "between", "value": [-8, 0]},
            {"field": "volume_vs_avg20", "op": "gte",     "value": 1.3},
        ],
    },

    # ── 7 · COMPLACENCY ───────────────────────────────────────────────────────
    # Slight pullback dismissed as a buying opportunity.
    "COMPLACENCY": {
        "description": "Minor pullback treated as a buying opportunity; bulls remain firmly in control",
        "emoji": "😌",
        "color": "#b06090",
        "weight": 1.0,
        "conditions": [
            {"field": "rsi",             "op": "between", "value": [50, 68]},
            {"field": "momentum_30",     "op": "between", "value": [-10, 12]},
            {"field": "momentum_60",     "op": "between", "value": [5, 30]},
            {"field": "price_vs_ma200",  "op": "between", "value": [18, 50]},
            {"field": "drawdown_90",     "op": "between", "value": [-18, -4]},
            {"field": "price_vs_ma50",   "op": "gte",     "value": -2},
        ],
    },

    # ── 8 · ANXIETY ───────────────────────────────────────────────────────────
    # Trend weakening; concern and uncertainty grow.
    "ANXIETY": {
        "description": "Trend falters; investors growing concerned but not yet selling aggressively",
        "emoji": "😟",
        "color": "#cc8844",
        "weight": 1.0,
        "conditions": [
            {"field": "rsi",             "op": "between", "value": [40, 58]},
            {"field": "momentum_30",     "op": "between", "value": [-20, -2]},
            {"field": "momentum_60",     "op": "between", "value": [-8, 18]},
            {"field": "price_vs_ma200",  "op": "between", "value": [3, 28]},
            {"field": "drawdown_90",     "op": "between", "value": [-28, -10]},
            {"field": "price_vs_ma50",   "op": "between", "value": [-10, 3]},
        ],
    },

    # ── 9 · DENIAL ────────────────────────────────────────────────────────────
    # Investors refuse to accept the downtrend.
    "DENIAL": {
        "description": "Downtrend denied — investors believe the dip will reverse imminently",
        "emoji": "🙈",
        "color": "#cc5533",
        "weight": 1.0,
        "conditions": [
            {"field": "rsi",             "op": "between", "value": [35, 55]},
            {"field": "momentum_30",     "op": "between", "value": [-28, -6]},
            {"field": "momentum_60",     "op": "between", "value": [-25, 2]},
            {"field": "price_vs_ma200",  "op": "between", "value": [-8, 18]},
            {"field": "drawdown_90",     "op": "between", "value": [-35, -14]},
            {"field": "macd_diff",       "op": "lt",      "value": 0},
        ],
    },

    # ── 10 · PANIC ────────────────────────────────────────────────────────────
    # Fear-driven selling accelerates; high volume selloffs.
    "PANIC": {
        "description": "Fear-driven capitulation selling; volume spikes as holders rush for the exit",
        "emoji": "😱",
        "color": "#cc3333",
        "weight": 1.1,
        "conditions": [
            {"field": "rsi",             "op": "between", "value": [20, 42]},
            {"field": "momentum_30",     "op": "between", "value": [-40, -14]},
            {"field": "momentum_60",     "op": "between", "value": [-40, -14]},
            {"field": "price_vs_ma200",  "op": "between", "value": [-40, -8]},
            {"field": "drawdown_90",     "op": "between", "value": [-50, -22]},
            {"field": "volume_vs_avg20", "op": "gte",     "value": 1.4},
        ],
    },

    # ── 11 · CAPITULATION ─────────────────────────────────────────────────────
    # Exhausted sellers; climactic volume marks the near bottom.
    "CAPITULATION": {
        "description": "Exhausted selling — climactic volume as last holders throw in the towel",
        "emoji": "🏳️",
        "color": "#aa2222",
        "weight": 1.2,
        "conditions": [
            {"field": "rsi",             "op": "between", "value": [15, 32]},
            {"field": "momentum_30",     "op": "between", "value": [-55, -22]},
            {"field": "momentum_90",     "op": "between", "value": [-60, -28]},
            {"field": "price_vs_ma200",  "op": "between", "value": [-55, -20]},
            {"field": "drawdown_90",     "op": "between", "value": [-65, -32]},
            {"field": "volume_vs_avg20", "op": "gte",     "value": 1.8},
        ],
    },

    # ── 12 · ANGER ────────────────────────────────────────────────────────────
    # Market remains depressed; investors blame others and stay away.
    "ANGER": {
        "description": "Market stays depressed — investors blame brokers, companies, and media",
        "emoji": "😤",
        "color": "#882222",
        "weight": 1.0,
        "conditions": [
            {"field": "rsi",             "op": "between", "value": [28, 48]},
            {"field": "momentum_30",     "op": "between", "value": [-18, 6]},
            {"field": "momentum_90",     "op": "between", "value": [-65, -28]},
            {"field": "price_vs_ma200",  "op": "between", "value": [-55, -20]},
            {"field": "drawdown_90",     "op": "between", "value": [-65, -28]},
            {"field": "volume_vs_avg20", "op": "lte",     "value": 1.1},
        ],
    },

    # ── 13 · DEPRESSION ───────────────────────────────────────────────────────
    # Maximum financial risk — maximum pain; true bottom territory.
    "DEPRESSION": {
        "description": "Maximum despair — most investors have given up; true bottom territory",
        "emoji": "💀",
        "color": "#5a1020",
        "weight": 1.2,
        "conditions": [
            {"field": "rsi",             "op": "between", "value": [22, 40]},
            {"field": "momentum_30",     "op": "between", "value": [-12, 6]},
            {"field": "momentum_90",     "op": "between", "value": [-60, -28]},
            {"field": "price_vs_ma200",  "op": "lte",     "value": -28},
            {"field": "drawdown_90",     "op": "between", "value": [-65, -32]},
            {"field": "volume_vs_avg20", "op": "lte",     "value": 0.75},
        ],
    },

    # ── 14 · DISBELIEF_NEW ────────────────────────────────────────────────────
    # New cycle begins; early recovery dismissed after deep bear market.
    "DISBELIEF_NEW": {
        "description": "New cycle begins — early recovery dismissed; deep bear market scars linger",
        "emoji": "🌅",
        "color": "#3a6a4a",
        "weight": 1.0,
        "conditions": [
            {"field": "rsi",             "op": "between", "value": [35, 52]},
            {"field": "momentum_30",     "op": "between", "value": [3, 24]},
            {"field": "momentum_60",     "op": "between", "value": [-12, 12]},
            {"field": "momentum_90",     "op": "between", "value": [-45, -8]},
            {"field": "price_vs_ma200",  "op": "between", "value": [-38, -4]},
            {"field": "drawdown_90",     "op": "between", "value": [-45, -12]},
            {"field": "macd_diff",       "op": "gt",      "value": 0},
        ],
    },
}

# Ordered list of phases (cycle order)
PHASE_ORDER: list[str] = list(MARKET_CYCLE_RULES.keys())
