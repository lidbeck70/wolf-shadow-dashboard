"""
berserk/rotation_link.py — Råvarurotationen (dina månadsbetyg) som manuell spärr i 🪓 BERSERK-skannern.

Rotationen betygsätter råvaror för hand (rotation.py: hat, case, katalysator → AGERA / Bevaka / Vila).
Den har ingen historik och kan inte backtestas — därför ingen automatisk regel, bara en markering:
KÖP i ett tema du satt på Vila flaggas (ROTATION: VILA) och kan döljas. Papperskontot påverkas inte.
"""

from __future__ import annotations

from typing import Optional

# BERSERK-tema → rotationens råvara (teman utan motsvarighet saknar betyg)
THEME_TO_ROTATION = {
    "guld": "guld", "silver": "silver", "platina": "platina", "palladium": "palladium", "uran": "uran",
    "olja": "olja", "naturgas": "gas", "naturgas_eu": "gas", "kol": "kol", "koppar": "koppar",
    "zink_nickel": "zink", "jarnmalm": "jarnmalm", "litium": "litium",
}
FLAG_VILA = "ROTATION: VILA"


def load_grades() -> dict:
    """Månadens betyg ur Råvarurotationen ({} om de inte går att läsa)."""
    try:
        import rotation
        return rotation.migrate_grades(rotation._load().get("grades") or {})
    except Exception:
        return {}


def theme_status(theme: str, grades: dict) -> Optional[str]:
    """AGERA / Bevaka / Vila för temats råvara — None när temat saknar motsvarighet eller betyg."""
    key = THEME_TO_ROTATION.get(theme or "")
    if not key or not grades.get(key):
        return None
    import rotation
    return rotation.status(grades[key])[0]


def apply(rows: list, grades: dict) -> list:
    """Sätter row['rotation'] och flaggar KÖP i teman på Vila."""
    import rotation
    for r in rows:
        r["rotation"] = theme_status(r.get("theme"), grades)
        if r.get("status") == "KÖP" and r["rotation"] == rotation.VILA:
            r["flags"] = list(r.get("flags") or []) + [FLAG_VILA]
    return rows
