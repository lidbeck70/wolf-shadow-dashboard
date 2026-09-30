"""
commodity_gate.py — råvarugrinden för Contrarian Alpha i Deep Contrarian-läget.

Strategiguiden: "Råvarurelaterat: gruvbolag, guld, silver, olja, gas". Den
gamla nödvändighetsgrinden släppte in allt "fysiskt nödvändigt" (telekom,
banker, livsmedel, läkemedel, fastigheter) — 53 % av Norden. Den här grinden
körs FÖRE nödvändighet och all datahämtning och släpper bara in råvaror.

Klassning i ordning:
  1. Börsdatas bransch-id (löpnummer, se necessity.BORSDATA_BRANCH_MAP).
  2. Bransch-/sektornamnet (rader utan id).
  3. Temat ur ember.regime.detect_theme: arket, Holdings och Yahoos bransch
     (manuella tickers, US-listan, globala rader utan bransch).

Quality-läget påverkas inte. Skogsbruk, jordbruk och ädelstenar är inte med.
Rena funktioner utom steg 3, som kan fråga Yahoo en gång per ticker (cachat).
"""

from __future__ import annotations

from typing import Callable, Optional

# Börsdata-branscher som räknas som råvara (id → etikett för tratten)
COMMODITY_BRANCH_IDS: dict[int, str] = {
    1: "Olja & gas – borrning", 2: "Olja & gas – exploatering", 3: "Olja & gas – transport",
    4: "Olja & gas – försäljning", 5: "Olja & gas – service", 6: "Kol", 7: "Uran",
    16: "Gruv – prospekt & drift", 17: "Gruv – industrimetaller", 18: "Gruv – guld & silver",
    20: "Gruv – service",
}
# Uttryckligen inte råvara här (VAL): ädelstenar 19, skog 21, jordbruk 60, fiskodling 61
EXCLUDED_BRANCH_IDS: frozenset = frozenset({19, 21, 60, 61})

# Namn (svenska Börsdata-namn och engelska branschord) → råvara
NAME_KEYWORDS: tuple = (
    "olja", "oil", "petroleum", "offshore", "drilling", "borrning",
    "naturgas", "natural gas", "lng", "exploatering", "exploration & production",
    "kol", "coal", "uran", "uranium",
    "gruv", "mining", "mineral", "metals", "metall", "guld", "gold", "silver",
    "koppar", "copper", "nickel", "zink", "zinc", "litium", "lithium", "sällsynta", "rare earth",
)
NAME_EXCLUDE: tuple = ("gasförsörjning", "skog", "forest", "jordbruk", "agri", "ädelsten", "diamond",
                       "gaming", "kolumn", "fintech")

# EMBER-teman som räknas (agri är inte råvara i den här fliken)
COMMODITY_THEMES: frozenset = frozenset({"uran", "silver", "guld", "koppar", "olja", "naturgas", "kol",
                                         "sallsynta", "platina", "palladium"})


def by_branch(branch_id) -> Optional[bool]:
    """True/False ur Börsdatas bransch-id, None när id saknas eller är okänt."""
    try:
        bid = int(branch_id)
    except (TypeError, ValueError):
        return None
    if bid in COMMODITY_BRANCH_IDS:
        return True
    if bid in EXCLUDED_BRANCH_IDS:
        return False
    return False if bid > 0 else None


def by_name(*names: str) -> Optional[bool]:
    """True när ett namn innehåller ett råvaruord, False när det finns ett namn
    utan råvaruord, None när inget namn finns."""
    texts = [str(n or "").strip().lower() for n in names if str(n or "").strip()]
    if not texts:
        return None
    for t in texts:
        if any(x in t for x in NAME_EXCLUDE):
            continue
        if any(k in t for k in NAME_KEYWORDS):
            return True
    return False


def classify(ticker: str, branch_id=None, branch_name: str = "", sector_name: str = "",
             theme_fn: Optional[Callable[[str], Optional[str]]] = None) -> tuple:
    """(är_råvara, etikett). Etiketten förklarar beslutet i tratten."""
    b = by_branch(branch_id)
    if b is not None:
        if b:
            return True, COMMODITY_BRANCH_IDS[int(branch_id)]
        return False, f"bransch {branch_name or branch_id}"
    n = by_name(branch_name, sector_name)
    if n:
        return True, branch_name or sector_name
    if theme_fn is None:
        def theme_fn(t):
            try:
                from ember.regime import detect_theme
                return detect_theme(t)
            except Exception:
                return None
    theme = theme_fn(ticker)
    if theme in COMMODITY_THEMES:
        return True, f"tema {theme}"
    if n is False:
        return False, f"bransch {branch_name or sector_name}"
    return False, "okänd bransch" + (f" (tema {theme})" if theme else "")
