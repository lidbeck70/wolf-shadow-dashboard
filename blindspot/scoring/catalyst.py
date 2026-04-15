"""
catalyst.py — Catalyst Score (0-20): Early reversal detection.
Detects if a hated stock is starting to turn around.

3 components:
- Price > SMA50 (+8)
- SMA50 slope > 0 (+6)
- Volume z-score > 1.0 (+6)
"""
import logging

logger = logging.getLogger(__name__)


def calculate_catalyst_score(price_data: dict) -> tuple:
    """Calculate catalyst score (0-20) for early reversal detection.

    Args:
        price_data: Dict with close, sma50, sma50_slope, volume stats.

    Returns:
        (catalyst_score, breakdown_dict)
    """
    if not price_data:
        return 0.0, {}

    breakdown = {}

    close = price_data.get("close", 0)
    sma50 = price_data.get("sma50", 0)
    sma50_slope = price_data.get("sma50_slope", 0)
    current_volume = price_data.get("current_volume", 0)
    avg_volume_20d = price_data.get("avg_volume_20d", 0)
    std_volume_20d = price_data.get("std_volume_20d", 0)

    # 1. Price > SMA50 (+8)
    price_above_sma50 = 0.0
    if close > 0 and sma50 > 0 and close > sma50:
        price_above_sma50 = 8.0
    breakdown["price_above_sma50"] = price_above_sma50

    # 2. SMA50 slope > 0 (+6)
    slope_positive = 0.0
    if sma50_slope > 0:
        slope_positive = 6.0
    breakdown["sma50_slope_positive"] = slope_positive

    # 3. Volume z-score > 1.0 (+6)
    vol_surge = 0.0
    if std_volume_20d > 0 and avg_volume_20d > 0:
        vol_z = (current_volume - avg_volume_20d) / std_volume_20d
        if vol_z > 1.0:
            vol_surge = 6.0
        breakdown["volume_z"] = round(vol_z, 2)
    else:
        breakdown["volume_z"] = 0.0
    breakdown["vol_surge"] = vol_surge

    total = price_above_sma50 + slope_positive + vol_surge
    return round(total, 1), breakdown
