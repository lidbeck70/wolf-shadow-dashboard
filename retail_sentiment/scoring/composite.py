"""
composite.py — Confidence-weighted composite score calculation.
Automatically reweights when sources have low confidence (e.g., Twitter=0).
"""
from retail_sentiment.config import BASE_WEIGHTS


def calculate_composite(scores: dict, confidences: dict) -> float:
    """Calculate confidence-weighted composite score.

    Sources with confidence=0 are effectively excluded, and the remaining
    weights are renormalized so the composite is always on a 0-100 scale.

    Args:
        scores: Dict of source_name -> score (0-100)
        confidences: Dict of source_name -> confidence (0-1)

    Returns:
        Composite score 0-100, or 0.0 if no sources available.
    """
    numerator = 0.0
    denominator = 0.0

    for source, base_weight in BASE_WEIGHTS.items():
        conf = confidences.get(source, 0.0)
        eff_weight = base_weight * conf
        numerator += eff_weight * scores.get(source, 0.0)
        denominator += eff_weight

    if denominator == 0:
        return 0.0

    return round(numerator / denominator, 1)
