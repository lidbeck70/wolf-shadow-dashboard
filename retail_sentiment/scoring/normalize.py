"""
normalize.py — Score normalization utilities.
Maps raw values to a 0-100 scale for consistent cross-source comparison.
"""


def normalize_z_to_100(z: float) -> float:
    """Convert a z-score (-3 to +3) to a 0-100 scale.
    z=-3 -> 0, z=0 -> 50, z=+3 -> 100
    """
    z_clamped = max(-3.0, min(3.0, z))
    return round((z_clamped + 3) / 6 * 100, 1)


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to 0-100 given a min/max range."""
    if max_val == min_val:
        return 50.0
    return round(max(0.0, min(100.0, (value - min_val) / (max_val - min_val) * 100)), 1)
