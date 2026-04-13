"""
Nordic Gold color constants and helper functions for the OVTLYR dashboard.

All rgba() strings use the format rgba(r,g,b,a) — never 8-digit hex.
"""

# ------------------------------------------------------------------ #
#  Base palette — Nordic Gold
# ------------------------------------------------------------------ #
BG       = "#0c0c12"   # main background — warm dark
BG2      = "#14141e"   # card / panel background
CYAN     = "#c9a84c"   # PRIMARY ACCENT — gold (replaces cyan everywhere)
MAGENTA  = "#8b7340"   # SECONDARY ACCENT — bronze (replaces magenta)
GREEN    = "#2d8a4e"   # bullish / buy — deep forest green
RED      = "#c44545"   # bearish / sell — deep red
YELLOW   = "#d4943a"   # warning / hold — warm amber
TEXT     = "#e8e4dc"   # primary text — warm white
DIM      = "#8a8578"   # secondary / muted text — warm gray

# Additional Nordic Gold palette
GOLD     = "#c9a84c"   # same as CYAN — for explicit gold references
BRONZE   = "#8b7340"   # same as MAGENTA
GOLD_DIM = "#5a4f32"   # muted gold for borders
TEXT_FAINT = "#5a5750" # very muted text
SURFACE  = "#181820"   # slightly lighter surface for hover/active

# ------------------------------------------------------------------ #
#  Semantic maps
# ------------------------------------------------------------------ #
REGIME_COLORS: dict[str, str] = {
    "green":  GREEN,
    "orange": YELLOW,
    "red":    RED,
}

SIGNAL_COLORS: dict[str, str] = {
    "BUY":    GREEN,
    "HOLD":   YELLOW,
    "REDUCE": MAGENTA,
    "SELL":   RED,
    "LONG":   GREEN,
    "SHORT":  RED,
    "FLAT":   DIM,
}

# ------------------------------------------------------------------ #
#  Helper functions
# ------------------------------------------------------------------ #

def risk_color(score: int) -> str:
    """
    Return a hex color representing a risk score 0–100.

    0–33  → GREEN  (low risk)
    34–66 → YELLOW (medium risk)
    67–100→ RED    (high risk)
    """
    if score <= 33:
        return GREEN
    elif score <= 66:
        return YELLOW
    else:
        return RED


def sentiment_color(score: int) -> str:
    """
    Return a hex color for a Fear & Greed score 0–100.

    0–25  → RED    (extreme fear)
    26–45 → YELLOW (fear)
    46–55 → TEXT   (neutral)
    56–75 → YELLOW (greed)
    76–100→ RED    (extreme greed / euphoria — danger zone)
    """
    if score <= 25:
        return RED       # extreme fear — oversold opportunity but risky
    elif score <= 45:
        return YELLOW    # fear zone — cautious but possible entry
    elif score <= 55:
        return TEXT      # neutral
    elif score <= 75:
        return YELLOW    # greed building — tread carefully
    else:
        return RED       # extreme greed — Rule 5 violation zone


def ob_color(ob_type: str, alpha: float = 0.3) -> str:
    """
    Return an rgba() string for an order block zone.

    Parameters
    ----------
    ob_type : str
        "bullish" or "bearish" (case-insensitive).
    alpha : float
        Opacity 0.0–1.0 (default 0.3 for fill areas).

    Returns
    -------
    str — e.g. "rgba(45,138,78,0.3)"

    Note: Always returns rgba() — never 8-digit hex.
    """
    ob_type = ob_type.lower()
    alpha = max(0.0, min(1.0, alpha))

    if ob_type == "bullish":
        return f"rgba(45,138,78,{alpha})"      # forest green with alpha
    elif ob_type == "bearish":
        return f"rgba(196,69,69,{alpha})"      # deep red with alpha
    else:
        return f"rgba(138,133,120,{alpha})"    # warm gray for unknown/invalidated


def signal_badge_css(signal: str) -> str:
    """
    Return an inline CSS style string for a signal badge.
    Suitable for use in st.markdown() HTML blocks.
    """
    color = SIGNAL_COLORS.get(signal.upper(), DIM)
    return (
        f"background-color: {color}22; "
        f"color: {color}; "
        f"border: 1px solid {color}; "
        f"border-radius: 4px; "
        f"padding: 4px 12px; "
        f"font-weight: 700; "
        f"font-size: 1.0rem; "
        f"letter-spacing: 0.08em;"
    )


def regime_badge_css(regime: str) -> str:
    """
    Return an inline CSS style string for a regime color badge.
    """
    color = REGIME_COLORS.get(regime.lower(), DIM)
    return (
        f"background-color: {color}22; "
        f"color: {color}; "
        f"border: 1px solid {color}; "
        f"border-radius: 4px; "
        f"padding: 4px 12px; "
        f"font-weight: 700; "
        f"font-size: 1.0rem; "
        f"text-transform: uppercase; "
        f"letter-spacing: 0.1em;"
    )
