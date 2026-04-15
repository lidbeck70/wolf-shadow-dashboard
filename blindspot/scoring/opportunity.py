"""
opportunity.py — Opportunity Score: the final contrarian signal.

Formula:
    opportunity = hat * (necessity_floored / 100) * (strength / 100) + catalyst_bonus
    where:
        necessity_floored = max(20, necessity)
        catalyst_bonus = catalyst * (hat / 100)

Also computes flags:
    - low_confidence: overall_confidence < 0.5
    - unmapped_sector: necessity_confidence < 0.5
    - value_trap_risk: hat > 60 and catalyst == 0
    - potential_reversal: hat > 60 and catalyst >= 10
"""


def calculate_opportunity(hat: float, necessity: float, strength: float,
                          catalyst: float, overall_confidence: float,
                          necessity_confidence: float) -> tuple:
    """Calculate the final opportunity score.

    Returns:
        (opportunity_score, flags_dict)
    """
    necessity_floored = max(20.0, necessity)

    base = hat * (necessity_floored / 100) * (strength / 100)
    catalyst_bonus = catalyst * (hat / 100) if hat > 0 else 0.0
    opportunity = base + catalyst_bonus

    # Cap at reasonable maximum
    opportunity = min(opportunity, 100.0)

    flags = {
        "low_confidence": overall_confidence < 0.5,
        "unmapped_sector": necessity_confidence < 0.5,
        "value_trap_risk": hat > 60 and catalyst == 0,
        "potential_reversal": hat > 60 and catalyst >= 10,
    }

    return round(opportunity, 1), flags
